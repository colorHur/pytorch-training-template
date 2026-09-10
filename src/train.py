"""
训练循环 —— 这个文件是整个模板的心脏。

⭐ 全部手写，不依赖任何高层封装（不用 Lightning / Trainer）。
   因为面试问的就是这些细节，包起来就学不到东西了。

一个标准训练 step 的顺序（顺序错了会出 bug）
------------------------------------------
    1. optimizer.zero_grad(set_to_none=True)   清空上一步的梯度
    2. logits = model(x)                        前向
    3. loss = criterion(logits, y)              算损失
    4. scaler.scale(loss).backward()            反向（自动求导累计梯度）
    5. scaler.unscale_(optimizer)               把梯度还原成真实尺度（AMP 时）
    6. clip_grad_norm_(...)                     梯度裁剪（防梯度爆炸）
    7. scaler.step(optimizer) / optimizer.step() 更新参数
    8. scaler.update()                          调整 scale（AMP 时）

为什么 zero_grad 必须在 backward 之前？
--------------------------------------
PyTorch 的 .backward() 是 **累加** 梯度到 .grad，而不是覆盖。
如果不清零，第 2 个 batch 的梯度会叠在第 1 个上 —— 等效于用了错误的梯度更新。
（而梯度累积正是故意利用这个累加特性，见下文）

梯度累积为什么能省显存？
----------------------
显存大头是**激活值**（activations），正比于参与前向的样本数。
梯度累积每次只前向 micro_batch（如 32），激活值只占 32 份；
攒 N 次梯度后再更新，数学上等效于 batch=32*N，但显存只需 32 份。
注意：**BN 层的统计量是按 micro_batch 算的**，等效 batch 时需要小心。
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# 学习率调度器
# ============================================================
def build_lr_scheduler(optimizer, name: str, total_steps: int, warmup_steps: int = 0):
    """手写调度器，理解 lr 是怎么随 step 变化的。

    返回一个函数 f(step) -> lr，训练循环里自己调用 optimizer.param_groups 赋值。
    这样写的好处：看得见每一步 lr 怎么变，而不是黑盒 scheduler.step()。
    """
    if name == "none":
        base_lrs = [g["lr"] for g in optimizer.param_groups]

        def const(step: int) -> list[float]:
            return base_lrs

        return const

    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def schedule(step: int) -> list[float]:
        # ---- warmup 阶段：lr 从 0 线性升到 base ----
        if warmup_steps > 0 and step < warmup_steps:
            scale = (step + 1) / warmup_steps
            return [lr * scale for lr in base_lrs]

        # ---- 主体阶段 ----
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)

        if name == "cosine":
            # 余弦退火：从 base 平滑降到 0
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif name == "step":
            # 阶梯下降：每 1/3 训练进度砍一半
            scale = 0.5 ** int(progress * 3)
        else:
            raise KeyError(f"未知 lr_scheduler '{name}'，可选：none / cosine / step")

        return [lr * scale for lr in base_lrs]

    return schedule


# ============================================================
# 评测
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str) -> dict[str, float]:
    """在验证/测试集上跑一遍，返回 loss 和 accuracy。

    @torch.no_grad() 的作用：不构建计算图，省显存且更快。
    ⚠️ 忘了加会导致评估阶段显存暴涨（甚至 OOM）。
    """
    model.eval()                      # 切换 BN / Dropout 到推理模式
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


# ============================================================
# 训练主循环
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    cfg: Any,
    global_step: int,
    lr_schedule=None,
    total_steps: int = 0,
    scaler=None,
    logger=None,
) -> tuple[dict[str, float], int]:
    """跑一个 epoch，返回 (统计信息, 更新后的 global_step)。"""
    model.train()                     # ⚠️ 必须在每个 epoch 开头调用（evaluate 会切到 eval）
    running_loss, seen, correct = 0.0, 0, 0
    t0 = time.time()
    accum = cfg.grad_accum_steps

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ---- 1) 清空梯度 ----
        # set_to_none=True 比填 0 更省内存（直接把 .grad 置 None，而非分配全零张量）
        if i % accum == 0:
            optimizer.zero_grad(set_to_none=True)

        # ---- 2) 前向 ----
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.amp):
            logits = model(x)
            loss = criterion(logits, y)

        # ---- 3) 反向 ----
        # 梯度累积：损失要除以累积步数，否则梯度会放大 accum 倍
        loss_scaled = loss / accum
        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # ---- 4) 参数更新（攒够 accum 步才更新）----
        is_update_step = ((i + 1) % accum == 0) or ((i + 1) == len(loader))
        if is_update_step:
            if cfg.max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)   # 裁剪前必须先把梯度还原成真实尺度
                # 梯度裁剪：把梯度的全局 L2 范数限制在 max_grad_norm 内
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if lr_schedule is not None:
                lrs = lr_schedule(global_step)
                for group, lr in zip(optimizer.param_groups, lrs):
                    group["lr"] = lr

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            global_step += 1

        # ---- 5) 统计 ----
        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += y.size(0)

        if logger and (i + 1) % cfg.log_interval == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger(
                f"    epoch {epoch} | step {i + 1:>4}/{len(loader)} | "
                f"loss {running_loss / seen:.4f} | acc {correct / seen:.4f} | lr {cur_lr:.2e}"
            )

    stats = {
        "loss": running_loss / max(1, seen),
        "acc": correct / max(1, seen),
        "time": time.time() - t0,
        "lr": optimizer.param_groups[0]["lr"],
    }
    return stats, global_step
