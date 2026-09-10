"""学习率 finder 的命令行入口 —— 训练之前先问一句「lr 该给多少」。

为什么单独一个工具，而不是塞进 `src/main.py` 加个 `--lr_find` 开关
----------------------------------------------------------------
「扫一遍 lr」和「训练」是两条控制流：扫描只跑 ~100 个 step、每个 step 换一个 lr、
不存 checkpoint、不用 scheduler，最后还要把权重还原。硬塞进 `main.py` 会让那个
已经 500 行的入口再多出一堆 if 分支，而它本身只该负责「跑一次训练」。

这个工具**复用** `src/` 里全部构件（`build_dataloaders` / `build_model` /
`build_optimizer`），所以扫的就是真实训练的那套东西，不存在「finder 用的模型
和实际训练不一样」这种经典错位。

用法
----
    python tools/lr_finder.py --dataset synthetic --steps 60
    python tools/lr_finder.py --dataset mnist --steps 100 --max_lr 1.0
    python tools/lr_finder.py --config configs/mnist.yaml --lr 1e-3   # 顺便和当前 lr 比一比

产物
----
    outputs/lr_finder/lr_sweep.json   完整曲线（每一步的 lr / loss / 平滑 loss）+ 结论
    outputs/lr_finder/lr_sweep.md     给人看的报告

退出码：0 = 扫完了（建议值在 stdout 与产物里）；1 = 参数/环境有问题（异常会往上抛）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# 这个工具要真跑前向反向，整条训练栈都躲不掉（不像 ddp_probe / compile_probe 那种
# 纯环境自检可以只挂 src/ 目录省掉 torch）—— 所以这里直接把仓库根挂上，走正常包导入。
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch.nn as nn  # noqa: E402

from src.config import TrainConfig  # noqa: E402
from src.console import force_utf8_stdout  # noqa: E402
from src.data import build_dataloaders  # noqa: E402
from src.lr_finder import (  # noqa: E402
    DEFAULT_BETA,
    DEFAULT_DIVERGENCE_FACTOR,
    DEFAULT_MAX_LR,
    DEFAULT_MIN_LR,
    DEFAULT_NUM_STEPS,
    LRSweepResult,
    run_lr_sweep,
)
from src.model import build_model  # noqa: E402
from src.train import set_seed  # noqa: E402

OUT_DIR = HERE / "outputs" / "lr_finder"

#: 这几个是**工具自己的**参数，不是 TrainConfig 的字段 ——
#: 忘了剔除就会撞上 merge() 的「未知配置项」校验（main.py 的 --config 踩过同一个坑）。
TOOL_ONLY_ARGS = {"config", "min_lr", "max_lr", "steps", "beta", "divergence_factor"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="学习率 finder（LR range test）：扫一遍 lr，报告该用多少",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="YAML 配置（与 src/main.py 同一套）")

    # 和 main.py 一样：默认 None，交给 merge() 忽略，实现「命令行 > YAML > 默认值」
    p.add_argument("--dataset", type=str, default=None, choices=["mnist", "cifar10", "synthetic"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--model", type=str, default=None, choices=["small_cnn", "mlp"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="当前在用的 lr；扫描结果会拿它和建议值比一比",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])

    # 扫描参数（不属于 TrainConfig）
    p.add_argument("--min_lr", type=float, default=DEFAULT_MIN_LR, help="扫描起点")
    p.add_argument("--max_lr", type=float, default=DEFAULT_MAX_LR, help="扫描终点")
    p.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS, help="扫多少步")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA, help="loss 平滑用的 EMA 动量")
    p.add_argument(
        "--divergence_factor",
        type=float,
        default=DEFAULT_DIVERGENCE_FACTOR,
        help="平滑 loss 超过历史最低点的多少倍就判为发散",
    )
    return p.parse_args()


def build_report(
    result: LRSweepResult,
    cfg: TrainConfig,
    image_size: int,
    n_batches: int,
) -> str:
    """生成 markdown 报告，结构对齐 experiments/ 下那几份。"""
    lines = [
        "# 学习率扫描（LR range test）",
        "",
        f"- 数据：`{cfg.dataset}`（train {n_batches} 个 batch，batch_size={cfg.batch_size}）",
        f"- 模型：`{cfg.model}`（输入 {image_size}×{image_size}，seed={cfg.seed}）",
        f"- 优化器：`{cfg.optimizer}`（weight_decay={cfg.weight_decay}）"
        f" —— 最优点依赖优化器，所以扫描用的就是它",
        "",
        "## 结论",
        "",
        f"**建议学习率：{result.suggested_lr:.3e}**"
        f"（第 {result.suggested_step} 步，loss 下降最陡处）",
        "",
        f"- 当前配置 `lr={cfg.lr:.2e}` → {result.compare_to(cfg.lr)}",
        f"- 扫描内平滑 loss 最低 {result.min_loss:.4f}（lr={result.min_loss_lr:.2e}）"
        " —— 这是发散边缘，不要取",
    ]
    if result.diverged_at_lr is not None:
        lines.append(f"- 发散点：lr≈{result.diverged_at_lr:.2e}")
    lines += ["", "## 曲线（横轴对数刻度，X = 建议值）", "```", result.describe().split("\n", 1)[1].rstrip(), "```"]

    if result.notes:
        lines += ["", "## 注意事项", ""]
        lines += [f"- ⚠️ {n}" for n in result.notes]

    lines += [
        "",
        "## 怎么用这个数",
        "",
        "- 直接把它当 lr 起点是可以的，但更稳的做法是再**除以 2~3** 起步，"
        "配合 warmup 一起用（扫描给的是「起点附近」的最优，不是全程最优）。",
        "- 换了 batch size / 优化器 / 初始权重就要重扫 —— 这个数不是模型的性质，"
        "是「这份配置 + 这份数据」的性质。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    # 必须早于任何 print：Windows 上管道输出走系统 locale（英文机器 = cp1252），
    # 这份日志全是中文，不钉住 UTF-8 会直接 UnicodeEncodeError。
    force_utf8_stdout()

    args = parse_args()

    base = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    overrides = {k: v for k, v in vars(args).items() if k not in TOOL_ONLY_ARGS}
    cfg = base.merge(**overrides)
    device = cfg.resolve_device()

    print(f"\n{'=' * 62}")
    print("  学习率扫描（LR range test）")
    print(f"{'=' * 62}")
    print(
        f"  数据 {cfg.dataset} | 模型 {cfg.model} | 优化器 {cfg.optimizer}"
        f"(wd={cfg.weight_decay}) | batch_size {cfg.batch_size}"
    )
    print(
        f"  lr 范围 {args.min_lr:.1e} → {args.max_lr:.1e}（等比，{args.steps} 步）"
        f" | 设备 {device} | seed {cfg.seed}\n"
    )

    loaders = build_dataloaders(
        dataset_name=cfg.dataset,
        data_dir=str(HERE / cfg.data_dir),
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    meta = loaders["meta"]
    cfg = cfg.merge(num_classes=meta["num_classes"], in_channels=meta["in_channels"])
    image_size = meta.get("image_size", 28)

    # 和真实训练用同一套种子 —— 扫描必须在「训练会从这个权重出发」的那个点上做，
    # 否则扫出来的建议值和实际训练不是同一回事。
    set_seed(cfg.seed)
    model = build_model(
        cfg.model,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        image_size=image_size,
    ).to(device)

    print(f"  开始扫描（{args.steps} 步，每步一次参数更新）...")
    result = run_lr_sweep(
        model,
        loaders["train"],
        nn.CrossEntropyLoss(),
        cfg,
        device=device,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.steps,
        beta=args.beta,
        divergence_factor=args.divergence_factor,
    )

    print("\n" + result.describe(current_lr=cfg.lr))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "lr_sweep.json"
    json_path.write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "device": device,
                "image_size": image_size,
                "train_batches": len(loaders["train"]),
                "min_lr": args.min_lr,
                "max_lr": args.max_lr,
                "steps": args.steps,
                "beta": args.beta,
                "divergence_factor": args.divergence_factor,
                "result": result.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path = OUT_DIR / "lr_sweep.md"
    md_path.write_text(
        build_report(result, cfg, image_size, len(loaders["train"])), encoding="utf-8"
    )

    print(f"\n  曲线与结论已保存：{md_path}")
    print(f"  原始数据          ：{json_path}")
    print(f"{'=' * 62}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
