"""学习率 finder（LR range test，Leslie Smith 1990 年代的思路）——
用一次短扫描代替「靠猜 + 反复重训」。

怎么做的
-------
1. 从一个极小 lr（1e-7）出发，**几何**地（等比）升到 1.0，每一步用一个 batch 更新一次；
2. 记录每一步的 loss，得到「loss vs log(lr)」这条曲线 —— 典型形状是
   先慢慢下降、然后在某个 lr 附近陡然上升（发散）；
3. 取**最低点之前 loss 下降最陡**的那个 lr 作为建议值。

   ⚠️ 不是取 loss 最低的那个点 —— 最低点往往已经贴着了发散边缘，用它训练迟早炸。
   要的是「性价比最高」的点：斜率最陡 = 每升高一个 lr 数量级换来的 loss 下降最多。

4. 扫完整条区间后把权重**还原**，模型跟没扫过一样。

为什么每一步的 lr 要**等比**而不是等差
------------------------------------
loss 对 lr 的敏感度大致是「每个数量级一档」：1e-4 → 1e-3 的变化远大于
1e-3 → 1.1e-3。等差扫的话，前面那段（1e-7 附近）几步就跨完了，什么都没看清；
等比扫让每个数量级分到相同的步数，曲线在 log 轴上才是均匀的。

三个必须做对、做错就会得到错误建议的细节
------------------------------------
1. **平滑要用带偏差修正的 EMA**。原始 loss 每步都抖，直接找最陡点在噪声上找。
   用普通 EMA 时初值是 0，前几步的平滑值被人为压低 → 「最陡下降」落在第一步，
   建议值直接退化成 min_lr。除以 ``1 - beta**(t+1)`` 才是无偏的。
2. **扫描时必须关掉梯度裁剪**（`max_grad_norm=0`）。裁剪的作用正是把梯度爆炸压住 ——
   开着它，lr 到 1.0 也不炸，曲线上看不到该有的那个陡升拐点，建议值会偏高。
3. **扫描时必须关掉 AMP**。GradScaler 会动态改 scale、必要时跳过 step，
   等于往曲线里混入一个与 lr 无关的噪声源。扫 lr 用 fp32。

另外：扫描**不做梯度累积**（`grad_accum_steps=1`），保证「一个点 = 一次更新 = 一个 lr」；
优化器类型 / weight_decay 保持和真实训练一致，因为最优点依赖于它们。

局限（别把建议值当真理）
----------------------
- 它给出的是**起点附近**的最优 lr。训练中途的 lr 该由 scheduler 决定，不是一条恒定值。
- 建议值依赖初始权重、batch size、数据。换了这些就要重扫。
- 数据量不够时会循环复用 batch，曲线会被轻微压低（结果里会明确提示）。
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.train import build_optimizer

#: 起点：比任何有意义的学习率都小两个数量级以上
DEFAULT_MIN_LR = 1e-7

#: 终点：大到足以让这个模型发散，否则看不到拐点
DEFAULT_MAX_LR = 1.0

#: 扫多少步。步数太少曲线全是噪声，太多则慢
DEFAULT_NUM_STEPS = 100

#: EMA 的动量。0.98 是常见的取值（越接近 1 越平滑、越滞后）
DEFAULT_BETA = 0.98

#: 平滑 loss 超过「历史最低点」这么多倍，就认为发散了（fastai 的判据）
DEFAULT_DIVERGENCE_FACTOR = 4.0


# ============================================================
# 数据结构
# ============================================================
@dataclass
class SweepPoint:
    """扫描中的一个点：第 step 步、用了 lr、得到 loss（原始 + 平滑）。"""

    step: int
    lr: float
    loss: float
    smoothed: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "lr": self.lr,
            "loss": self.loss,
            "smoothed": self.smoothed,
        }


@dataclass
class LRSweepResult:
    """一次扫描的全部产出。"""

    points: list[SweepPoint]
    suggested_lr: float
    suggested_step: int
    min_loss: float          # 平滑曲线的最低点
    min_loss_lr: float
    diverged_at_lr: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggested_lr": self.suggested_lr,
            "suggested_step": self.suggested_step,
            "min_loss": self.min_loss,
            "min_loss_lr": self.min_loss_lr,
            "diverged_at_lr": self.diverged_at_lr,
            "notes": list(self.notes),
            "points": [p.to_dict() for p in self.points],
        }

    # ---------- 给人看的输出 ----------

    def compare_to(self, current_lr: float) -> str:
        """把建议值和当前配置的 lr 比一比 —— 这才是跑 finder 的目的。"""
        ratio = current_lr / self.suggested_lr if self.suggested_lr else float("inf")
        if 0.3 <= ratio <= 3.0:
            return f"当前配置 lr={current_lr:.2e} 与建议值同量级（{ratio:.2f}×），可以先不动。"
        if ratio > 1:
            return (
                f"当前配置 lr={current_lr:.2e} 比建议值高 {ratio:.1f} 倍"
                f"（建议 {self.suggested_lr:.2e} 及以下）—— 偏大，训练早期容易震荡或发散。"
            )
        return (
            f"当前配置 lr={current_lr:.2e} 只有建议值的 {ratio:.2f} 倍"
            f"（建议 {self.suggested_lr:.2e} 附近）—— 偏小，收敛会慢。"
        )

    def describe(self, current_lr: float | None = None) -> str:
        lines = ["学习率扫描（LR range test）"]
        if self.points:
            first, last = self.points[0], self.points[-1]
            ratio = (last.lr / first.lr) ** (1 / max(1, len(self.points) - 1))
            lines.append(
                f"  · 扫了 {len(self.points)} 步：lr {first.lr:.1e} → {last.lr:.1e}"
                f"（每步 ×{ratio:.2f}）"
            )
        lines.append(
            f"  · 建议 lr = {self.suggested_lr:.3e}"
            f"（step {self.suggested_step}，loss 下降最陡处）"
        )
        lines.append(
            f"  · 扫描内平滑 loss 最低 {self.min_loss:.4f}，出现在 lr={self.min_loss_lr:.2e}"
            " —— 别取这里，它贴着发散边缘"
        )
        if self.diverged_at_lr is not None:
            lines.append(f"  · 发散点：lr≈{self.diverged_at_lr:.2e}（曲线在这里开始陡升）")
        else:
            lines.append("  · 扫描区间内没有发散：把 --max_lr 调大才能看到拐点")

        lines.append(ascii_curve([p.lr for p in self.points], [p.smoothed for p in self.points],
                                 mark_index=self.suggested_step))

        for note in self.notes:
            lines.append(f"  ⚠️  {note}")
        if current_lr is not None:
            lines.append(f"  → {self.compare_to(current_lr)}")
        return "\n".join(lines)


# ============================================================
# 纯函数部分（不需要训练，全部可离线测）
# ============================================================
def geometric_lrs(min_lr: float, max_lr: float, num_steps: int) -> list[float]:
    """在 [min_lr, max_lr] 上按**等比**取 num_steps 个点（含两端）。

    首项恰好是 min_lr、末项恰好是 max_lr，相邻比值恒定 —— 这样在 log 轴上才是等距的。
    """
    if min_lr <= 0:
        raise ValueError("min_lr 必须 > 0（log 轴上的起点不能是 0）")
    if max_lr <= min_lr:
        raise ValueError(f"max_lr ({max_lr}) 必须大于 min_lr ({min_lr})")
    if num_steps < 2:
        raise ValueError("num_steps 至少为 2")

    ratio = (max_lr / min_lr) ** (1.0 / (num_steps - 1))
    return [min_lr * ratio**i for i in range(num_steps)]


def smooth_losses(losses: list[float], beta: float = DEFAULT_BETA) -> list[float]:
    """带**偏差修正**的指数滑动平均（EMA）。

        avg_t = beta * avg_{t-1} + (1 - beta) * loss_t
        out_t = avg_t / (1 - beta^(t+1))          # ← 这一步是关键

    没有最后那个除法，`avg_0` 就等于 `(1-beta)*loss_0`（一个接近 0 的数）——
    前几步的平滑值被人为压低，「最陡下降」会落在曲线开头，建议值退化成 min_lr。
    偏差修正让 `out_0 == losses_0`，曲线从头就是无偏的。
    """
    out: list[float] = []
    avg = 0.0
    for t, value in enumerate(losses):
        avg = beta * avg + (1.0 - beta) * value
        out.append(avg / (1.0 - beta ** (t + 1)))
    return out


def suggest_lr(lrs: list[float], smoothed: list[float]) -> tuple[int, float]:
    """返回 `(索引, 建议 lr)`：最低点**之前**、log 轴上下滑最陡的那一点。

    为什么约束在「最低点之前」：最低点之后 loss 已经开始回升，
    那一段的"下降"没有意义，纯粹是噪声。
    """
    if not lrs:
        return 0, 0.0
    if len(lrs) == 1:
        return 0, lrs[0]

    i_min = min(range(len(smoothed)), key=smoothed.__getitem__)
    if i_min == 0:
        # 整条曲线单调上升：没有可用的下降段
        return 0, lrs[0]

    best_i, best_slope = 1, None
    for i in range(1, i_min + 1):
        # 横轴是 log(lr)，所以斜率是「每个数量级换多少 loss」
        d_log_lr = math.log(lrs[i]) - math.log(lrs[i - 1])
        slope = (smoothed[i] - smoothed[i - 1]) / d_log_lr if d_log_lr > 0 else 0.0
        if best_slope is None or slope < best_slope:
            best_slope, best_i = slope, i
    return best_i, lrs[best_i]


def ascii_curve(
    xs: list[float],
    ys: list[float],
    mark_index: int | None = None,
    width: int = 52,
    height: int = 11,
) -> str:
    """把曲线画成纯 ASCII 的图（本仓库不依赖 matplotlib）。

    横轴取 log10（因为扫描本身是等比的，log 轴上才均匀），纵轴是 loss。
    `mark_index` 处画一个 `X`，就是建议的学习率。
    """
    if not xs or not ys:
        return "  （没有数据可画）"

    lo, hi = min(ys), max(ys)
    span = hi - lo
    log_x = [math.log10(x) for x in xs]
    x_lo, x_hi = log_x[0], log_x[-1]
    x_span = max(x_hi - x_lo, 1e-12)

    def row_of(value: float) -> int:
        if span <= 0:
            return height // 2
        frac = (value - lo) / span          # 0 = loss 最低，画在最下面
        return height - 1 - round(frac * (height - 1))

    def col_of(i: int) -> int:
        frac = (log_x[i] - x_lo) / x_span
        return min(width - 1, max(0, round(frac * (width - 1))))

    grid = [[" "] * width for _ in range(height)]
    for i, value in enumerate(ys):
        grid[row_of(value)][col_of(i)] = "."
    if mark_index is not None and 0 <= mark_index < len(ys):
        grid[row_of(ys[mark_index])][col_of(mark_index)] = "X"

    lines = ["", "  （横轴对数刻度，纵轴 loss；X = 建议的 lr）"]
    for r in range(height):
        if span <= 0:
            label = f"{lo:>7.3f}"
        else:
            label = f"{hi - span * r / (height - 1):>7.3f}"
        lines.append(f"  {label} |{''.join(grid[r])}|")
    lines.append(f"  {'':>7} +{'-' * width}+")
    left, right = f"{xs[0]:.1e}", f"{xs[-1]:.1e}"
    pad = width - len(left) - len(right)
    lines.append(f"  {'':>7}  {left}{' ' * max(1, pad)}{right}")
    return "\n".join(lines)


# ============================================================
# 扫描配置
# ============================================================
def resolve_sweep_config(cfg: TrainConfig) -> TrainConfig:
    """把配置改成「适合扫描」的样子，每一项都有理由（见模块 docstring）。"""
    return cfg.merge(
        amp=False,              # GradScaler 会往曲线里混入与 lr 无关的噪声
        grad_accum_steps=1,     # 一个点 = 一次更新 = 一个 lr
        max_grad_norm=0.0,      # 裁剪会掩盖发散，而发散正是要看的拐点
        lr_scheduler="none",    # lr 由扫描自己一步步写
        compile=False,          # 编译开销和形状特化都不属于这条曲线
    )


# ============================================================
# 扫描主流程
# ============================================================
def run_lr_sweep(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    cfg: TrainConfig,
    device: str = "cpu",
    *,
    min_lr: float = DEFAULT_MIN_LR,
    max_lr: float = DEFAULT_MAX_LR,
    num_steps: int = DEFAULT_NUM_STEPS,
    beta: float = DEFAULT_BETA,
    divergence_factor: float = DEFAULT_DIVERGENCE_FACTOR,
) -> LRSweepResult:
    """跑一次 LR 扫描，返回结果（**不动模型**，结束后权重原样还原）。

    `model` 必须是**没有包装过**的原始模型：这里既不包 DDP 也不开梯度检查点 /
    compile —— 它们都不改变数值，只会让扫描变慢、把变量搞多。
    """
    lrs = geometric_lrs(min_lr, max_lr, num_steps)
    sweep_cfg = resolve_sweep_config(cfg)

    notes: list[str] = []
    n_batches = len(loader)
    if n_batches == 0:
        raise ValueError("loader 是空的，没法扫描学习率")
    if n_batches < num_steps:
        notes.append(
            f"数据只有 {n_batches} 个 batch 而扫描 {num_steps} 步，已循环复用同一批样本 —— "
            "loss 曲线会被轻微压低，建议换更大的数据集或调小 --steps。"
        )

    # ⚠️ 权重快照：扫描会（故意）把 lr 推到发散，模型权重会被毁掉。
    #    而发散意味着 NaN —— 不还原的话，接下来的训练从 NaN 权重开始，
    #    你会以为是「选中的 lr 有问题」，其实是 finder 没收拾干净。
    snapshot = copy.deepcopy(model.state_dict())
    was_training = model.training

    optimizer = build_optimizer(model, sweep_cfg)
    raw: list[float] = []
    diverged_at_lr: float | None = None

    model.train()                       # 和真实训练一样：BN 用批统计、Dropout 打开
    stream = itertools.cycle(loader)
    try:
        for lr in lrs:
            x, y = next(stream)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            value = loss.item()
            if not math.isfinite(value):
                diverged_at_lr = lr
                notes.append(f"lr={lr:.3e} 时 loss 变成了 {value}，扫描提前中止。")
                break
            raw.append(value)
    finally:
        model.load_state_dict(snapshot)
        model.train(was_training)

    points, smoothed, truncated_at = _finalize(lrs, raw, beta, divergence_factor)
    if diverged_at_lr is None:
        diverged_at_lr = truncated_at

    if not points:
        return LRSweepResult(
            points=[],
            suggested_lr=lrs[0],
            suggested_step=0,
            min_loss=float("inf"),
            min_loss_lr=lrs[0],
            diverged_at_lr=diverged_at_lr if diverged_at_lr is not None else lrs[0],
            notes=notes or ["第一步就发散了 —— min_lr 就已经太大。"],
        )

    used_lrs = [p.lr for p in points]
    idx, suggested = suggest_lr(used_lrs, smoothed)
    min_idx = min(range(len(smoothed)), key=smoothed.__getitem__)

    if idx <= 1:
        notes.append(
            "整个区间内没看到 loss 下降（建议值落在最左端）—— "
            "max_lr 可能还不够大，或这份数据太容易；把 --max_lr 调到 1.0 再试。"
        )
    elif idx == len(points) - 1:
        notes.append("建议值已经贴到扫描上界，最优 lr 可能在上界之外，把 --max_lr 调大再扫一次。")

    return LRSweepResult(
        points=points,
        suggested_lr=suggested,
        suggested_step=idx,
        min_loss=smoothed[min_idx],
        min_loss_lr=used_lrs[min_idx],
        diverged_at_lr=diverged_at_lr,
        notes=notes,
    )


def _finalize(
    lrs: list[float],
    raw: list[float],
    beta: float,
    divergence_factor: float,
) -> tuple[list[SweepPoint], list[float], float | None]:
    """平滑 + 在发散点截断。

    返回 `(点列表, 平滑后的 loss, 截断处的 lr)`；没有截断时第三个值是 None。

    截断的理由：发散之后的曲线一路往上，既没有信息量，又会把 ascii 图的
    纵轴拉垮（4 倍于最低点的一小段能占掉半个图）。
    """
    if not raw:
        return [], [], None
    smoothed = smooth_losses(raw, beta)

    running_min = math.inf
    cut, truncated_at = len(raw), None
    for i, value in enumerate(smoothed):
        running_min = min(running_min, value)
        if i > 0 and value > divergence_factor * running_min:
            cut, truncated_at = i + 1, lrs[i]
            break

    raw, smoothed = raw[:cut], smoothed[:cut]
    points = [SweepPoint(i, lrs[i], raw[i], smoothed[i]) for i in range(cut)]
    return points, smoothed, truncated_at
