"""
训练主入口：argparse + 配置覆盖 + checkpoint + 日志 + 显存统计。

命令行参数优先级：**命令行 > YAML 配置文件 > dataclass 默认值**

用法
----
    # 用配置文件
    python -m src.main --config configs/mnist.yaml

    # 覆盖单个参数
    python -m src.main --config configs/mnist.yaml --epochs 1 --batch_size 64

    # 不开配置文件，纯命令行
    python -m src.main --exp_name quicktest --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 让 `python src/main.py` 和 `python -m src.main` 都能跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.model import build_model, count_parameters, enable_gradient_checkpointing
from src.train import build_lr_scheduler, evaluate, train_one_epoch

HERE = Path(__file__).resolve().parent.parent


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int) -> None:
    """固定所有随机源，保证实验可复现。

    ⚠️ 面试点：完全可复现还需要 cudnn.deterministic=True，但会损失性能。
       通常做法是固定 seed + 不开 deterministic，允许浮点层面的微小差异。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_mem_str() -> str:
    """当前 + 峰值显存（GB）。"""
    if not torch.cuda.is_available():
        return "N/A"
    cur = torch.cuda.memory_allocated() / 1024 ** 3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return f"cur {cur:.2f}GB / peak {peak:.2f}GB"


def save_checkpoint(path: Path, model, optimizer, epoch, global_step, cfg, metrics) -> None:
    """保存 checkpoint。

    为什么要把 cfg 也存进去？
      半年后回看这个实验，只有权重没有超参 = 无法复现。配置跟着 checkpoint 走。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    """按配置建优化器。

    AdamW vs SGD（面试常问）
    ----------------------
    - AdamW：自适应学习率（每参数单独缩放），收敛快、对 lr 不敏感，是默认选择。
              W 表示 decoupled weight decay —— 权重衰减不参与梯度动量计算，比 L2 正则更干净。
    - SGD+momentum：泛化有时更好（尤其 CV），但需要精调 lr + warmup + 长训练。
    经验：拿不准就用 AdamW。
    """
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    raise KeyError(f"未知优化器 '{cfg.optimizer}'，可选：adamw / sgd")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch 训练模板")
    p.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")

    # 全部默认 None —— merge() 会忽略 None，实现「命令行 > YAML」的优先级
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, choices=["mnist", "cifar10"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--model", type=str, default=None, choices=["small_cnn", "mlp"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    p.add_argument("--lr_scheduler", type=str, default=None, choices=["none", "cosine", "step"])
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="混合精度")
    p.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="梯度检查点：用多算一次前向换取激活显存",
    )
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    return p.parse_args()


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    args = parse_args()

    # ---- 组装配置：YAML -> 命令行覆盖 ----
    # 注意：`--config` 是运行参数，不是配置项，必须从覆盖集合里剔除
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    cfg = cfg.merge(**overrides)
    device = cfg.resolve_device()

    if cfg.amp and device == "cpu":
        print("⚠️  指定了 --amp 但设备是 CPU，已自动关闭混合精度")
        cfg = cfg.merge(amp=False)

    set_seed(cfg.seed)

    # ---- 输出目录 ----
    run_dir = HERE / cfg.output_dir / cfg.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 双路日志：控制台 + 文件 ----
    log_file = (run_dir / "train.log").open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n{'=' * 60}\n  PyTorch 训练模板 · {cfg.exp_name}\n{'=' * 60}")
    log(cfg.summary())
    log(f"  设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    log(f"  输出目录: {run_dir}\n")

    # ---- 数据 ----
    log("[1/5] 加载数据...")
    t0 = time.time()
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
    log(
        f"      train {len(loaders['train'].dataset)} 张 | "
        f"val {len(loaders['val'].dataset)} 张 | test {len(loaders['test'].dataset)} 张 "
        f"({time.time() - t0:.1f}s)"
    )

    # ---- 模型 ----
    log("[2/5] 建模型...")
    model = build_model(
        cfg.model, in_channels=cfg.in_channels, num_classes=cfg.num_classes
    ).to(device)

    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # ---- 梯度检查点（可选项，模型不支持则明确告知而不是静默忽略）----
    ckpt_active = False
    if cfg.gradient_checkpointing:
        ckpt_active = enable_gradient_checkpointing(model)
        if ckpt_active:
            log("      梯度检查点：开（反向重算激活，用计算换显存）")
        else:
            log(f"      ⚠️  {cfg.model} 未实现梯度检查点，该配置已忽略")

    total, trainable = count_parameters(model)
    log(f"      {cfg.model}: 总参数 {total:,} / 可训练 {trainable:,}")

    # ---- 优化器 / 调度器 / 损失 ----
    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(loaders["train"]) // cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.epochs
    lr_schedule = build_lr_scheduler(optimizer, cfg.lr_scheduler, total_steps, cfg.warmup_steps)

    # AMP 的 GradScaler 只在 CUDA + amp 开启时使用
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device == "cuda") else None

    log(
        f"[3/5] 优化器 {cfg.optimizer}(lr={cfg.lr}) | 调度器 {cfg.lr_scheduler} | "
        f"warmup {cfg.warmup_steps} | 总更新步数 {total_steps}"
    )
    log(
        f"      AMP {'开' if cfg.amp else '关'} | 梯度裁剪 {cfg.max_grad_norm or '关'} | "
        f"梯度检查点 {'开' if ckpt_active else '关'}"
        + ("（配置要求开但模型不支持，已忽略）" if cfg.gradient_checkpointing and not ckpt_active else "")
    )

    # ---- 训练 ----
    log(f"\n[4/5] 开始训练（{cfg.epochs} epochs）...\n")
    history: list[dict] = []
    best_val_acc = 0.0
    best_path = run_dir / "best.pt"
    patience_counter = 0
    global_step = 0
    t_train = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_stats, global_step = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            cfg=cfg,
            global_step=global_step,
            lr_schedule=lr_schedule,
            total_steps=total_steps,
            scaler=scaler,
            logger=log,
        )
        val_stats = evaluate(model, loaders["val"], criterion, device)

        log(
            f"  ▶ epoch {epoch}/{cfg.epochs}  "
            f"train_loss {train_stats['loss']:.4f} train_acc {train_stats['acc']:.4f} | "
            f"val_loss {val_stats['loss']:.4f} val_acc {val_stats['acc']:.4f} | "
            f"{train_stats['time']:.1f}s | {gpu_mem_str()}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "lr": train_stats["lr"],
                "time": train_stats["time"],
            }
        )

        # ---- 保存最优 & 早停 ----
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            save_checkpoint(best_path, model, optimizer, epoch, global_step, cfg, val_stats)
            patience_counter = 0
        else:
            patience_counter += 1
            if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
                log(f"  ⏹ 早停触发（val_acc 连续 {patience_counter} 个 epoch 未提升）")
                break

        if cfg.save_every_epoch:
            save_checkpoint(run_dir / f"epoch{epoch}.pt", model, optimizer, epoch, global_step, cfg, val_stats)

    train_time = time.time() - t_train

    # ---- 测试集最终评测（只跑一次）----
    log(f"\n[5/5] 用最优权重在测试集上评测...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_stats = evaluate(model, loaders["test"], criterion, device)
    log(f"      测试集：loss {test_stats['loss']:.4f} | acc {test_stats['acc']:.4f}")

    # ---- 汇总落盘 ----
    summary = {
        "exp_name": cfg.exp_name,
        "config": cfg.to_dict(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "total_params": total,
        "trainable_params": trainable,
        "gradient_checkpointing_active": ckpt_active,
        "best_val_acc": best_val_acc,
        "test_acc": test_stats["acc"],
        "test_loss": test_stats["loss"],
        "total_train_time_sec": round(train_time, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        if device == "cuda"
        else None,
        "history": history,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    cfg.to_yaml(run_dir / "config_resolved.yaml")

    log(f"\n{'=' * 60}")
    log(f"  训练完成")
    log(f"  最优验证准确率 : {best_val_acc:.4f}")
    log(f"  测试集准确率   : {test_stats['acc']:.4f}")
    log(f"  总耗时         : {train_time:.1f}s")
    log(f"  峰值显存       : {summary['peak_vram_gb']} GB")
    log(f"  产物目录       : {run_dir}")
    log(f"{'=' * 60}\n")

    log_file.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
