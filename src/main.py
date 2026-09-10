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

    # 断点续训：从这个 run 上次停下的地方接着训（last.pt 每个 epoch 都会落盘）
    python -m src.main --config configs/mnist.yaml --resume last
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

# 让 `python src/main.py` 和 `python -m src.main` 都能跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import force_utf8_stdout
from src.checkpoint import (
    ResumeState,
    diff_configs,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
)
from src.compile_support import maybe_compile
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
from src.train import (
    build_lr_scheduler,
    build_optimizer,
    evaluate,
    set_seed,
    train_one_epoch,
)

HERE = Path(__file__).resolve().parent.parent

# ⚠️ 这些是**运行控制**参数，不是配置项，必须从"覆盖 TrainConfig"的集合里剔除。
#    `TrainConfig.merge()` 对未知键是直接抛 ValueError 的 —— 这是好事（配置项
#    拼错了会立刻报错），但也意味着任何新增的运行参数都得同时加到这里，
#    否则 `--resume xxx` 会以一个"未知配置项：['resume']"的报错收场。
RUN_ONLY_ARGS = {"config", "resume"}


# ============================================================
# 工具函数
# ============================================================
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch 训练模板")
    p.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 继续训练。可写 'last' / 'best'（本 run 目录下的同名文件）"
        "或任意 .pt 路径",
    )

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
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.compile：把训练 step 编译成优化后的图（Windows 上默认后端通常不可用，见 README）",
    )
    p.add_argument(
        "--compile_backend",
        type=str,
        default=None,
        help="编译后端，默认 inductor；它不可用时可以试 aot_eager / cudagraphs",
    )
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument(
        "--save_every_epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="每个 epoch 都存一份 epoch{N}.pt（默认由配置决定）",
    )
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
    # 注意：`--config` / `--resume` 是运行参数，不是配置项，必须从覆盖集合里剔除
    overrides = {k: v for k, v in vars(args).items() if k not in RUN_ONLY_ARGS}
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

    # ---- torch.compile（可选项）----
    # 放在 DDP **外层**：PyTorch 推荐把 compile 包在最外面 —— 这样它能看到并优化
    # 通信相关的算子；顺序反过来也能跑，但优化视野更窄。
    #
    # 为什么可能"开了却没生效"：Windows 上 inductor 后端的两条路都是断的
    # （CUDA 缺 Triton、CPU 缺 MSVC 的 cl.exe），而且报错发生在**训练第 1 步**、
    # 信息里还不提"缺什么、怎么修"。所以这里主动探测 + 冒烟前向，开不了就带着
    # 原因回退，训练照常继续。
    compile_outcome = maybe_compile(
        model,
        enabled=cfg.compile,
        backend=cfg.compile_backend,
        device=device,
        example_input=(
            torch.randn(cfg.batch_size, cfg.in_channels, image_size, image_size, device=device)
            if cfg.compile
            else None
        ),
    )
    model = compile_outcome.model
    compile_active = compile_outcome.active
    if cfg.compile:
        log(f"      torch.compile：{compile_outcome.message}")

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

    # ---- 断点续训：把进度和随机流恢复到刚才建好的这些对象里 ----
    # 位置很关键，必须在「model / optimizer / scaler / loader 都已建好」之后、
    # 「训练循环开始」之前：
    #   - `set_seed(cfg.seed)` 已经在上面跑过了，这里再叠一层 RNG 还原，顺序不能反
    #   - model 已经包了 DDP / compile，所以传进去的必须是 `unwrap_model(model)`
    #     （存的时候剥掉了前缀，读的时候也得是剥掉的那个）
    #   - 此时**还没有开始迭代**，loader 的 shuffle generator 状态是干净的初值，
    #     覆盖它不会丢掉任何东西
    state = ResumeState()
    resume_path: Path | None = None
    resumed_from_epoch: int | None = None
    if args.resume is not None:
        resume_path = resolve_resume_path(args.resume, run_dir)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"--resume {args.resume!r} 解析到 {resume_path}，但该文件不存在"
            )
        state = load_checkpoint(
            resume_path,
            model=unwrap_model(model),
            optimizer=optimizer,
            scaler=scaler,
            loader=loaders["train"],
            device=device,
        )
        resumed_from_epoch = state.epoch
        log(f"\n  ↩ 续训：从 {resume_path} 恢复")
        log(f"       {state.describe()}")
        if state.missing:
            log(
                "       ⚠️ 这些状态没被恢复，只能当默认值用 —— 续训结果会和"
                "连续训练**不完全一致**（老格式 checkpoint 的正常现象）"
            )
        # 关键超参变了要说出来：恢复 optimizer 状态之后，lr 调度器还会用新配置的
        # base_lr 覆盖回去，等于"换学习率重新调度"，而 loss 曲线上看不出断点。
        for key, saved_value, current_value in diff_configs(state.config, cfg.to_dict()):
            log(f"       ⚠️ 配置与原实验不同：{key}: {saved_value} → {current_value}")

        # `epochs` 是唯一一个"变了也正常、但会改变训练语义"的配置，所以单独说。
        # 原因：cosine / step 调度的 `total_steps = steps_per_epoch × epochs`，
        # 把它调大 → 整条 lr 曲线被重新规划 → 连**已经训过的那几轮**都对不上了。
        saved_epochs = state.config.get("epochs")
        if saved_epochs is not None and saved_epochs != cfg.epochs:
            log(
                f"       ℹ️ --epochs 由 {saved_epochs} 改成 {cfg.epochs}："
                f"「{cfg.lr_scheduler}」调度会按新的总步数（{total_steps}）重新规划，"
                f"所以这条学习率曲线与「一次训到 {cfg.epochs} 轮」不可逐位对齐。"
                "想要严格等价的断点恢复（比如崩了重来），--epochs 要和原来一样。"
            )

    # 这份 config 是**本次运行实际生效**的配置（含 num_classes/in_channels 等
    # 由数据集推导出来的字段），存进 checkpoint 才算完整可复现。
    state.config = cfg.to_dict()

    # ---- 训练 ----
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    resumed_note = f"，从第 {state.next_epoch} 个 epoch 继续" if resume_path is not None else ""
    log(f"\n[4/5] 开始训练（{cfg.epochs} epochs{resumed_note}）...\n")
    if state.next_epoch > cfg.epochs:
        log(
            f"  ⚠️ checkpoint 里已经训完 {state.epoch} 个 epoch，而 --epochs={cfg.epochs} ——"
            f" 没有要训的轮次了。要接着训就把 --epochs 调大。"
        )
    t_train = time.time()

    # ⚠️ 从这里开始，`state` 就是进度的**唯一真相**（epoch / global_step /
    #    best_val_acc / patience_counter / history）。不再另起一组同名局部变量 ——
    #    两份状态并存，迟早会出现"存下去的那份和用来判断的那份不一样"。
    for epoch in range(state.next_epoch, cfg.epochs + 1):
        # ⚠️ 每个 epoch 必须调一次，否则每轮的 shuffle 顺序完全相同（等于没打乱）
        set_epoch(train_sampler, epoch)

        train_stats, state.global_step = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            cfg=cfg,
            global_step=state.global_step,
            lr_schedule=lr_schedule,
            total_steps=total_steps,
            scaler=scaler,
            logger=log,          # 非主进程的 log 是空操作，不会重复刷屏
            ctx=ctx,
        )
        val_stats = evaluate(model, loaders["val"], criterion, device, ctx=ctx)
        state.epoch = epoch

        improved = val_stats["acc"] > state.best_val_acc
        if improved:
            state.best_val_acc = val_stats["acc"]
            state.patience_counter = 0
        else:
            state.patience_counter += 1

        log(
            f"  ▶ epoch {epoch}/{cfg.epochs}  "
            f"train_loss {train_stats['loss']:.4f} train_acc {train_stats['acc']:.4f} | "
            f"val_loss {val_stats['loss']:.4f} val_acc {val_stats['acc']:.4f} | "
            f"{train_stats['time']:.1f}s | {gpu_mem_str()}"
            + ("  ★新最优" if improved else "")
        )

        state.history.append(
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

        # ---- 落盘（只有 rank 0 写；早停条件各 rank 一致，不会分叉）----
        # 三个文件的职责不同，别混：
        #   best.pt      val 有提升时才写 —— 最后评测用
        #   last.pt      **每个 epoch 都写** —— 续训的默认目标（见 --resume last）
        #   epoch{N}.pt  save_every_epoch 时才写 —— 事后挑某一轮做对比
        if ctx.is_main:
            for path, enabled in (
                (best_path, improved),
                (last_path, True),
                (run_dir / f"epoch{epoch}.pt", cfg.save_every_epoch),
            ):
                if enabled:
                    save_checkpoint(
                        path,
                        state=state,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        loader=loaders["train"],
                        metrics=val_stats,
                    )

        # 早停判定放在**落盘之后**：否则最后一轮的 last.pt 缺一个 epoch，
        # 续训时会从更早的位置重来一遍（白算一轮）。
        if cfg.early_stop_patience > 0 and state.patience_counter >= cfg.early_stop_patience:
            log(f"  ⏹ 早停触发（val_acc 连续 {state.patience_counter} 个 epoch 未提升）")
            break

    train_time = time.time() - t_train
    history = state.history
    best_val_acc = state.best_val_acc

    # ---- 测试集最终评测（只跑一次）----
    # 非主进程也要读同一个文件 —— 所以必须 barrier 等 rank 0 写完再读，
    # 否则会读到一个写了一半的文件。
    barrier(ctx)
    # 正常情况用 best.pt；但如果本次是从别的 run 续训、而这个 run 目录里
    # 从没轮到过 val 提升（best_val_acc 是从 checkpoint 带过来的），best.pt
    # 可能压根不存在 —— 那就退回 last.pt，并说清楚用的是哪个，别静默换掉。
    eval_path = best_path if best_path.exists() else last_path
    if eval_path != best_path:
        log(f"\n[5/5] {best_path.name} 不存在（本次续训没有刷新最优），改用 {eval_path.name}")
    else:
        log("\n[5/5] 用最优权重在测试集上评测...")
    ckpt = torch.load(eval_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model"])
    del ckpt                       # 权重已拷进模型，别让它白占一份显存
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
        "torch_compile": {
            "enabled": cfg.compile,
            "active": compile_active,
            "backend": cfg.compile_backend,
            "first_call_seconds": round(compile_outcome.first_call_seconds, 3),
        },
        "distributed": {
            "enabled": ctx.enabled,
            "backend": ctx.backend,
            "world_size": ctx.world_size,
            "sync_bn": sync_bn_active,
            # 全局等效 batch = 本地 micro batch × 梯度累积 × 进程数
            "global_effective_batch_size": cfg.effective_batch_size * ctx.world_size,
        },
        # 续训是一等公民，就得和 compile / distributed 一样在 summary 里留痕 ——
        # 否则事后看一个 0.98 的实验，分不清它是"一次训了 10 轮"还是"续了 5 次"。
        "resume": {
            "enabled": resume_path is not None,
            "path": str(resume_path) if resume_path is not None else None,
            "from_epoch": resumed_from_epoch,
            "restored_everything": not state.missing,
            "missing_keys": list(state.missing),
            "eval_weights": eval_path.name,
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
