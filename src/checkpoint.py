"""Checkpoint 的保存与恢复 —— 「断点续训」要恢复的**全部**状态。

为什么单独一个模块
----------------
原来的 `save_checkpoint` 只是「把几个对象 torch.save 进去」，看起来没问题，
但它**要恢复什么**这件事从来没被系统地想过：存了 optimizer 状态却没人读回来，
存了 epoch 却没有 `--resume` 入口 —— 等于一份写完就没打开过的保险单。

而「续训要恢复哪些状态」恰恰是面试高频追问题，所以这里把每一项都显式列出，
并且**每一条都能用测试证明它必要**。

必须恢复的六类状态（按"漏了之后多难查"排序）
------------------------------------------
| 状态 | 漏了会怎样 |
|------|-----------|
| `optimizer.state_dict()` | Adam 的一阶/二阶矩和 step 计数被重置 → **续训后 loss 先尖峰再回落**。最容易被漏，因为它不报错、也不改变"能不能跑" |
| `global_step` | lr 调度错位：cosine 从头开始 → 等效于偷偷改了学习率计划 |
| `epoch` | ① 数据顺序回到第 1 轮的 shuffle；② 早停/最优判断的基准丢失 |
| `best_val_acc` + `patience_counter` | **静默的逻辑错误**：最优判断的基准清零 → `best.pt` 可能被一个更差的 epoch 覆盖；早停计数清零 → 早停永远触发不了 |
| `scaler.state_dict()`（AMP） | GradScaler 的 scale 回到初值 → 前几步可能溢出、被跳过 |
| RNG（torch / cuda / numpy / python / **DataLoader 的 generator**） | Dropout mask 和每轮 shuffle 顺序不再一致 → 严格复现失败 |

RNG 那一条最容易被忽略，而且这里其实有**两股独立的随机流**：

1. **全局 torch RNG** —— `small_cnn` 里有 Dropout，每次前向都在消耗它。
2. **`train_loader.generator`** —— 单进程 shuffle 用的那个 `torch.Generator`，
   它**每轮迭代都会前进**（`torch.randperm(n, generator=...)` 直接推进它）。

`src/data.py` 里给训练集 DataLoader 显式传了一个固定 seed 的 generator，
所以这两股流是分开的：Dropout 抽多少不影响数据顺序，数据顺序也只由 generator
决定。两股都要存，否则续训后的 mask 或数据顺序会和连续训练不一样。

（顺带一个坑：`DataLoader(shuffle=True)` 不带 generator 时，每轮 `__iter__`
会从**全局 RNG** 取一个种子来建临时 generator。那样两股流就耦合了 ——
Dropout 多抽一次都会改变下一轮的数据顺序，说不清也测不准。）

向后兼容
--------
老 checkpoint 里没有上面这些键。`load_checkpoint` 用 `.get()` 取值、把缺失的键
收集到 `ResumeState.missing` 里交给调用方打印 —— **能加载就别报错**，
但也不能装作没这回事（静默用默认值 = 静默改变训练语义）。

本模块的对外接口
----------------
- `save_checkpoint` / `load_checkpoint`：存与取
- `ResumeState`：训练循环需要的那份"进度"
- `capture_rng` / `restore_rng`：两股随机流
- `resolve_resume_path`：`--resume last|best|<路径>` 的解析
- `diff_configs`：揪出"续训时偷偷改了超参"

谁在用它：`src/main.py`（`last.pt` 每个 epoch 都落盘，是续训的默认目标）
与 `tests/test_checkpoint.py`。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.distributed import unwrap_model


@dataclass
class ResumeState:
    """训练循环需要从 checkpoint 取回的全部"进度"信息。"""

    epoch: int = 0                  # 已经**训完**的最后一个 epoch
    global_step: int = 0
    best_val_acc: float = 0.0
    patience_counter: int = 0
    history: list[dict] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)   # 老 checkpoint 缺哪些键

    @property
    def next_epoch(self) -> int:
        return self.epoch + 1

    def describe(self) -> str:
        text = (
            f"已训完 {self.epoch} 个 epoch（global_step={self.global_step}），"
            f"将从第 {self.next_epoch} 个 epoch 继续"
        )
        if self.missing:
            text += f"；⚠️ 该 checkpoint 缺少 {self.missing}，已用默认值补齐"
        return text


# ============================================================
# 随机数状态
# ============================================================
def capture_rng(loader: Any = None) -> dict[str, Any]:
    """抓取所有会影响训练的随机源状态。

    为什么单独抓 `loader.generator`：单进程训练的数据顺序由它决定
    （见 `src/data.py` 里 `build_dataloaders` 的注释），它每轮都会前进 ——
    不存它，续训后的第一个 epoch 会从第 1 轮的那个排列重新开始。

    ⚠️ 存的是**对象本身的状态**而不是它的种子。种子里没有"迭代了多少次"
       这个信息，用它还原只会得到第 1 轮的顺序。
    """
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:      # 容器里驱动可见但没设备
            pass
    generator = getattr(loader, "generator", None) if loader is not None else None
    if generator is not None:
        state["loader"] = generator.get_state()
    return state


def restore_rng(state: dict[str, Any], loader: Any = None) -> None:
    """`capture_rng` 的逆操作。缺哪一项就跳过哪一项（老 checkpoint 兼容）。"""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    generator = getattr(loader, "generator", None) if loader is not None else None
    if "loader" in state and generator is not None:
        generator.set_state(state["loader"])


# ============================================================
# 保存 / 加载
# ============================================================
def save_checkpoint(
    path: Path | str,
    *,
    state: ResumeState,
    model: nn.Module,
    optimizer: Any,
    scaler: Any = None,
    loader: Any = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """保存一个**可续训**的 checkpoint。

    为什么必须先 `unwrap_model`：
      DDP 包装后 `model.state_dict()` 的 key 会变成 `module.blocks.0.0.weight`，
      存成那样单进程代码就加载不了了 —— 而且保存时不会有任何报错。

    为什么 `config` 也要存：
      半年后回看这个实验，只有权重没有超参 = 无法复现。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": 2,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_val_acc": state.best_val_acc,
        "patience_counter": state.patience_counter,
        "history": state.history,
        "config": state.config,
        "metrics": metrics if metrics is not None else state.metrics,
        "rng": capture_rng(loader),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    *,
    model: nn.Module | None = None,
    optimizer: Any = None,
    scaler: Any = None,
    loader: Any = None,
    device: str = "cpu",
    map_location: Any = None,
) -> ResumeState:
    """把 checkpoint 恢复进传进来的对象，返回进度信息。

    ⚠️ `model` 传**未包装**的模型（DDP 的 `module`）—— 存的时候剥掉了前缀，
       读的时候也得是对应的那个。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{path}")

    ckpt = torch.load(
        path, map_location=map_location if map_location is not None else device,
        weights_only=False,
    )

    if model is not None:
        # strict=True：key 对不上就报错。静默跳过不匹配的键才是真正危险的
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if "rng" in ckpt:
        restore_rng(ckpt["rng"], loader)

    state = ResumeState(
        epoch=int(ckpt.get("epoch", 0)),
        global_step=int(ckpt.get("global_step", 0)),
        best_val_acc=float(ckpt.get("best_val_acc", 0.0)),
        patience_counter=int(ckpt.get("patience_counter", 0)),
        history=list(ckpt.get("history", [])),
        config=dict(ckpt.get("config", {})),
        metrics=dict(ckpt.get("metrics", {}) or {}),
    )

    # 老 checkpoint（format_version < 2）缺的键：能加载，但要说清楚
    expected = ("best_val_acc", "patience_counter", "history", "rng")
    state.missing = [key for key in expected if key not in ckpt]
    if "scaler" not in ckpt and scaler is not None:
        state.missing.append("scaler")
    return state


# ============================================================
# CLI 辅助：把 `--resume` 的值解析成路径
# ============================================================
# 两个"关键字"，等价于"这个 run 目录下的 last.pt / best.pt"
RESUME_KEYWORDS = ("last", "best")


def resolve_resume_path(value: str, run_dir: Path | str, cwd: Path | str | None = None) -> Path:
    """把 `--resume` 的取值解析成一个绝对路径。

    支持三种写法（日常用的是前两种）：

        --resume last                     → <run_dir>/last.pt
        --resume best                     → <run_dir>/best.pt
        --resume outputs/exp/epoch3.pt    → 相对当前工作目录解析
        --resume /abs/path/to/ckpt.pt     → 原样使用

    为什么要有关键字：训练中途崩了，用户手里的信息是"这个实验"，不是
    "某个绝对路径"。`--resume last` 跟 `git checkout HEAD` 是一个路子 ——
    常用场景给个短写法，特殊场景再写全路径。
    """
    text = str(value).strip()
    if text in RESUME_KEYWORDS:
        return Path(run_dir) / f"{text}.pt"
    path = Path(text)
    if path.is_absolute():
        return path
    return Path(cwd) / path if cwd is not None else path.resolve()


# 这些键在续训时本来就会/允许变，差异不算异常
RESUME_IGNORED_CONFIG_KEYS = (
    "exp_name",        # 可能故意建成一个新实验
    "output_dir",
    "epochs",          # 最常见的用法就是"接着多训几轮"
    "device",
    "data_dir",
    "num_workers",
    "save_every_epoch",
    "log_interval",
    "ddp_timeout_minutes",
)


def diff_configs(
    saved: dict[str, Any],
    current: dict[str, Any],
    ignore: tuple[str, ...] = RESUME_IGNORED_CONFIG_KEYS,
) -> list[tuple[str, Any, Any]]:
    """比出「续训时的配置」与「原实验配置」的关键差异。

    为什么需要这个提醒：**续训很容易变成偷偷改了训练语义。**
    最典型的是 `--lr` —— 恢复 optimizer 状态之后，lr 调度器（它在这份代码里
    是闭包住 base_lr 的）会用新配置的 lr 覆盖回去，于是"接着训"实际变成了
    "换学习率重新调度"，而 loss 曲线上看不出任何断点。

    忽略掉的那几项是"变了也不奇怪"的：`epochs` 天天都在变，
    `exp_name` / `output_dir` 是输出位置。返回空列表表示关键超参一致。
    """
    ignored = set(ignore)
    return [
        (key, saved.get(key, "<未记录>"), current.get(key))
        for key in sorted(set(saved) | set(current))
        if key not in ignored and saved.get(key, "<未记录>") != current.get(key)
    ]
