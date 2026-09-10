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
import torch.distributed as dist
import torch.nn as nn

# 让 `python src/main.py` 和 `python -m src.main` 都能跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import force_utf8_stdout
from src.config import TrainConfig
from src.data import build_dataloaders
from src.distributed import (
    barrier,
    maybe_convert_sync_bn,
    set_epoch,
    setup_distributed,
    unwrap_model,
    wrap_model,
)
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


def gpu_name() -> str | None:
    """当前 GPU 名字；取不到就返回 None（容器里可能驱动可见但无设备）。"""
    if not torch.cuda.is_available():
        return None
    try:
        if torch.cuda.device_count() > 0:
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def save_checkpoint(path: Path, model, optimizer, epoch, global_step, cfg, metrics) -> None:
    """保存 checkpoint。

    为什么要把 cfg 也存进去？
      半年后回看这个实验，只有权重没有超参 = 无法复现。配置跟着 checkpoint 走。

    为什么必须先 unwrap？
      DDP 包装后 `model.state_dict()` 的 key 会变成 `module.blocks.0.0.weight`。
      存成那样，单进程代码就加载不了了 —— 而且保存时不会有任何报错。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
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
    p.add_argument(
        "--dataset", type=str, default=None, choices=["mnist", "cifar10", "synthetic"]
    )
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
    p.add_argument(
        "--sync_bn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="把 BatchNorm 换成 SyncBatchNorm（DDP 下 BN 统计量默认不跨卡同步）",
    )
    p.add_argument(
        "--ddp_timeout_minutes",
        type=int,
        default=None,
        help="分布式集合通信超时；某个 rank 崩了时，这是唯一能把「卡住」变成「报错」的机制",
    )
    return p.parse_args()


# ============================================================
# 主流程
# ============================================================
def run_training() -> None:
    # 第一件事就把输出编码钉死成 UTF-8。
    # 这条日志全是中文，而 Windows 上输出到管道时 Python 用系统 locale 编码
    # （英文系统 = cp1252），不设就会 UnicodeEncodeError 把整个训练带崩。
    # 终端 / 本机是中文 locale 时永远复现不了 —— 典型 CI-only 故障。
    force_utf8_stdout()

    args = parse_args()

    # ---- 组装配置：YAML -> 命令行覆盖 ----
    # 注意：`--config` 是运行参数，不是配置项，必须从覆盖集合里剔除
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()

    # AMP + CPU 的冲突要在这里先化解：TrainConfig 的校验会直接抛 ValueError，
    # 但用户只是想要个提醒，不该看到一坨 traceback。
    amp_downgraded = False
    if overrides.get("amp") and (overrides.get("device") or cfg.device) == "cpu":
        overrides["amp"] = False
        amp_downgraded = True

    cfg = cfg.merge(**overrides)
    device = cfg.resolve_device()

    if cfg.amp and device == "cpu":
        cfg = cfg.merge(amp=False)
        amp_downgraded = True

    # ---- 分布式：torchrun 会把 RANK/LOCAL_RANK/WORLD_SIZE 写进环境变量 ----
    # 单进程（python src/main.py）时 world_size=1，ctx.enabled=False，下面所有
    # 分布式分支都短路，行为与之前完全一致。
    ctx = setup_distributed(device, timeout_minutes=cfg.ddp_timeout_minutes)
    if ctx.enabled:
        device = ctx.device          # nccl 下每个 rank 绑定自己的 local_rank

    # ⚠️ 所有 rank 用**同一个** seed，这一点很反直觉但必须如此：
    #    切分训练/验证集用的就是这个 seed。如果按 rank 偏移，
    #    每个进程切出的验证集都不一样，各 rank 的指标根本对不上。
    #    各 rank 的数据差异由 DistributedSampler 的 rank 决定，不靠 seed。
    #    （模型初始权重也不需要对齐 —— DDP 构造时会广播 rank 0 的参数。）
    set_seed(cfg.seed)

    # ---- 输出目录 ----
    run_dir = HERE / cfg.output_dir / cfg.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 双路日志：控制台 + 文件（**只有 rank 0 写**）----
    # N 个进程同时往同一个 train.log 里写会互相覆盖，写出损坏的日志。
    log_file = (run_dir / "train.log").open("w", encoding="utf-8") if ctx.is_main else None

    def log(msg: str) -> None:
        if log_file is None:         # 非主进程：静默，但**必须继续往下跑**
            return
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n{'=' * 60}\n  PyTorch 训练模板 · {cfg.exp_name}\n{'=' * 60}")
    log(cfg.summary())
    log(f"  设备: {device}" + (f" ({gpu_name()})" if device == "cuda" and gpu_name() else ""))
    log(f"  输出目录: {run_dir}\n")
    if ctx.enabled:
        log(f"  {ctx.describe()}\n")
    if amp_downgraded:
        log("⚠️  指定了 AMP 但设备是 CPU，已自动关闭混合精度\n")

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
        ctx=ctx,
    )
    meta = loaders["meta"]
    cfg = cfg.merge(num_classes=meta["num_classes"], in_channels=meta["in_channels"])
    # image_size 是数据集属性而非超参，不进配置，直接从 meta 取
    image_size = meta.get("image_size", 28)
    train_sampler = loaders["samplers"]["train"]
    log(
        f"      train {len(loaders['train'].dataset)} 张 | "
        f"val {len(loaders['val'].dataset)} 张 | test {len(loaders['test'].dataset)} 张 "
        f"({time.time() - t0:.1f}s)"
    )
    if ctx.enabled:
        log(
            f"      本 rank 分到 {len(loaders['train'])} 个 batch"
            f"（全局等效 batch = {cfg.effective_batch_size} × {ctx.world_size} = "
            f"{cfg.effective_batch_size * ctx.world_size}）"
        )

    # ---- 模型 ----
    log("[2/5] 建模型...")
    model = build_model(
        cfg.model,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        image_size=image_size,
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
    log(f"      {cfg.model}: 总参数 {total:,} / 可训练 {trainable:,}（输入 {image_size}×{image_size}）")

    # ---- SyncBatchNorm（必须在包 DDP 之前换）----
    # 注意：SyncBatchNorm 只在 CUDA 上可用（CPU 前向直接抛 ValueError），
    # 所以 maybe_convert_sync_bn 在 gloo 后端下会拒绝转换。
    sync_bn_active = maybe_convert_sync_bn(model, cfg.sync_bn, ctx)
    if cfg.sync_bn:
        if sync_bn_active:
            log("      SyncBatchNorm：已启用（BN 统计量跨卡同步）")
        elif not ctx.enabled:
            log("      SyncBatchNorm：未启用（单进程不需要）")
        elif ctx.backend != "nccl":
            log("      ⚠️  SyncBatchNorm：已忽略（只在 CUDA 上可用，当前 backend=gloo）")
        else:
            log("      SyncBatchNorm：未启用（模型里没有 BatchNorm）")

    # ---- 优化器 / 调度器 / 损失（要在包 DDP 之前建好，参数引用才对得上）----
    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    # ---- 包成 DDP（单进程时原样返回）----
    model = wrap_model(model, ctx)

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
        # ⚠️ 每个 epoch 必须调一次，否则每轮的 shuffle 顺序完全相同（等于没打乱）
        set_epoch(train_sampler, epoch)

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
            logger=log,          # 非主进程的 log 是空操作，不会重复刷屏
            ctx=ctx,
        )
        val_stats = evaluate(model, loaders["val"], criterion, device, ctx=ctx)

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

        # ---- 保存最优 & 早停（只有 rank 0 落盘；早停条件各 rank 一致，不会分叉）----
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            if ctx.is_main:
                save_checkpoint(best_path, model, optimizer, epoch, global_step, cfg, val_stats)
            patience_counter = 0
        else:
            patience_counter += 1
            if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
                log(f"  ⏹ 早停触发（val_acc 连续 {patience_counter} 个 epoch 未提升）")
                break

        if cfg.save_every_epoch and ctx.is_main:
            save_checkpoint(run_dir / f"epoch{epoch}.pt", model, optimizer, epoch, global_step, cfg, val_stats)

    train_time = time.time() - t_train

    # ---- 测试集最终评测（只跑一次）----
    # 非主进程也要读同一个 best.pt —— 所以必须 barrier 等 rank 0 写完再读，
    # 否则会读到一个写了一半的文件。
    barrier(ctx)
    log("\n[5/5] 用最优权重在测试集上评测...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model"])
    test_stats = evaluate(model, loaders["test"], criterion, device, ctx=ctx)
    log(f"      测试集：loss {test_stats['loss']:.4f} | acc {test_stats['acc']:.4f}")

    # ---- 汇总落盘（只有 rank 0 写）----
    # 注意 loss/acc 这类指标是 all-reduce 后的全局值，只有 rank 0 的数是准的，
    # 所以 summary 由主进程独占写入是顺理成章的。
    summary = {
        "exp_name": cfg.exp_name,
        "config": cfg.to_dict(),
        "device": device,
        "gpu": gpu_name(),
        "total_params": total,
        "trainable_params": trainable,
        "gradient_checkpointing_active": ckpt_active,
        "distributed": {
            "enabled": ctx.enabled,
            "backend": ctx.backend,
            "world_size": ctx.world_size,
            "sync_bn": sync_bn_active,
            # 全局等效 batch = 本地 micro batch × 梯度累积 × 进程数
            "global_effective_batch_size": cfg.effective_batch_size * ctx.world_size,
        },
        "best_val_acc": best_val_acc,
        "test_acc": test_stats["acc"],
        "test_loss": test_stats["loss"],
        "total_train_time_sec": round(train_time, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        if device.startswith("cuda")
        else None,
        "history": history,
    }
    if ctx.is_main:
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        cfg.to_yaml(run_dir / "config_resolved.yaml")

    log(f"\n{'=' * 60}")
    log("  训练完成")
    log(f"  最优验证准确率 : {best_val_acc:.4f}")
    log(f"  测试集准确率   : {test_stats['acc']:.4f}")
    log(f"  总耗时         : {train_time:.1f}s")
    log(f"  峰值显存       : {summary['peak_vram_gb']} GB")
    log(f"  产物目录       : {run_dir}")
    log(f"{'=' * 60}\n")

    if log_file is not None:
        log_file.close()


def main() -> None:
    """真正的入口：`force_utf8_stdout` + 分布式进程组的兜底收尾。

    为什么收尾要放在最外层
    --------------------
    训练中途抛异常（OOM、loss 变 NaN、用户 Ctrl-C）时，如果某个 rank 直接死掉
    而其他 rank 还阻塞在集合通信上，整个作业会一直挂着 —— **不会自己退出**，
    直到 `ddp_timeout_minutes` 超时。`finally` 里销毁进程组，能让其余进程尽快
    收到「对端已退出」并报错退出，而不是白等 30 分钟。
    """
    force_utf8_stdout()
    try:
        run_training()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
