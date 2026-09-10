"""
分布式训练（DDP）支持 —— 让同一份 `main.py` 既能单卡跑，也能 `torchrun` 起多进程。

设计原则
--------
**不另起一个 `train_ddp.py`。** `torchrun` 已经通过环境变量
（`RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT`）
把「我是谁、总共几个进程」告诉了每个进程，所以入口脚本自己探测一下就行：

    单进程：  python src/main.py --epochs 1
    多进程：  torchrun --nproc_per_node=2 src/main.py --epochs 1

这样做的好处是**训练逻辑只有一份**，不会出现「单卡能跑、多卡挂掉」的分叉。

DDP 到底做了什么（面试要能说清）
-------------------------------
1. **构造时广播参数**：`DistributedDataParallel(model)` 会把 rank 0 的参数
   广播给所有 rank —— 所以各 rank 的模型初始权重天然一致，**不需要手动对齐 seed**。
2. **反向时平均梯度**：每个 rank 在自己那份数据上算梯度，all-reduce 求平均。
   于是 `W 个进程 × 每进程 batch B` 的梯度 ≈ `单进程 batch W·B` 的梯度
   （等号成立的条件见 `experiments/exp_ddp_equivalence.py`）。
3. **不自动切数据**：数据划分得你自己做（`DistributedSampler`），
   否则每个 rank 都在学同一批样本 —— 这是最常见的"跑了但没提速"的原因。

⚠️ 三个必踩的坑
---------------
1. **`DistributedSampler.set_epoch(epoch)` 必须每个 epoch 调一次**。
   不调的话每个 epoch 的 shuffle 顺序完全一样，等于没有 shuffle。
2. **各 rank 的 step 数必须相等**，否则快的那个在 all-reduce 处等慢的，
   慢的又不等 —— 直接**死锁**（表现为训练卡住不动，没有报错）。
   所以训练集用 `drop_last=True` 的 `DistributedSampler` 保证切分等长。
3. **只有 rank 0 能写文件**。N 个进程同时写同一个 `train.log` / `best.pt`
   会互相覆盖甚至写出损坏的文件。

⚠️ DDP **不会**同步 BatchNorm 的 running stats
--------------------------------------------
每个 rank 用自己的那批数据算 BN 统计量，`broadcast_buffers=True`（默认）只是
在每次前向开始时把 rank 0 的 buffer 广播出去，**并没有让统计量变准**。
想让 BN 统计量真正全局化，得换成 `SyncBatchNorm`（见 `maybe_convert_sync_bn`）。
Transformer 用 LayerNorm 没有 running stats，所以 LLM 训练基本不受影响 ——
这也是为什么大家对这个问题不敏感。
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DistributedSampler, Sampler


# ============================================================
# 进程组上下文
# ============================================================
@dataclass
class DistContext:
    """描述"当前进程在分布式里的位置"。单进程时 `enabled=False`，所有分支都短路。"""

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    device: str = "cpu"

    @property
    def is_main(self) -> bool:
        """是不是主进程（只有它该写日志 / 存 checkpoint）。"""
        return self.rank == 0

    def describe(self) -> str:
        if not self.enabled:
            return "单进程（未启用分布式）"
        return (
            f"DDP 已启用 | backend={self.backend} | rank {self.rank}/{self.world_size}"
            f" | local_rank {self.local_rank} | 设备 {self.device}"
        )


def env_world_size() -> int:
    """从环境变量读进程总数。没设 / 设成 1 都视为单进程。"""
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def is_enabled(ctx: "DistContext | None") -> bool:
    """`ctx` 是否表示"真的在跑分布式"。

    为什么要单独抽一个函数：`train.py` / `data.py` 这些公开 API 里，
    `ctx=None` 的约定就是**单进程**（调用方不用为了单卡去构造一个 DistContext）。
    但 `None.enabled` 会直接抛 AttributeError —— 而且是在**训练循环中间**抛，
    排查成本很高。把判定收敛到一个地方，这类"忘记判空"的 bug 就只可能
    出现在这一行里。

    （这不是假设的问题：加 DDP 时正是因为 `ddp_no_sync` 里写了裸的 `ctx.enabled`，
      一次性打挂 12 个已有测试。）
    """
    return ctx is not None and ctx.enabled


def rendezvous_file() -> str | None:
    """读环境变量 `DDP_RENDEZVOUS_FILE`（由 `tools/ddp_launch.py` 设置）。

    有值 → 用 `FileStore` 做 rendezvous（不碰 TCP）。
    没值 → 走默认的 `env://` 机制，即 `torchrun` 设置的 `MASTER_ADDR/MASTER_PORT`。
    """
    value = os.environ.get("DDP_RENDEZVOUS_FILE", "").strip()
    return value or None


def setup_distributed(
    requested_device: str = "auto",
    timeout_minutes: int = 30,
) -> DistContext:
    """初始化进程组；单进程场景直接返回 `enabled=False` 的上下文。

    backend 选择
    ------------
    - 有 CUDA → `nccl`（NVIDIA 的集合通信库，GPU 间走 NVLink/PCIe，最快）
    - 否则    → `gloo`（CPU 上的集合通信，CI 和本机调试都用它）

    rendezvous（各进程怎么找到彼此）有两条路
    ---------------------------------------
    1. **`torchrun` 的默认方式**：创建一个 `TCPStore`，所有进程连同一个
       `MASTER_ADDR:MASTER_PORT`。这是生产环境的标准做法。
    2. **`FileStore`**：用一个文件当共享存储。`tools/ddp_launch.py` 会设置
       `DDP_RENDEZVOUS_FILE` 走这条路。

       为什么需要备选：Windows 上 TCPStore 依赖 libuv，而某些 torch 构建
       **没有编进 libuv**（报 `use_libuv was requested but PyTorch was built
       without libuv support`），而且 `USE_LIBUV=0` 也救不回来 —— 这些构建里
       Windows 的 TCPStore 只剩 libuv 一条实现。于是 TCPStore 直接不可用，
       但集合通信（gloo 的 socket）本身是好的，换个 rendezvous 方式就能跑。

    `timeout_minutes` 不是随便设的：某个 rank 崩了之后，其余 rank 会在集合通信
    上一直等。超时是唯一能把"卡住"变成"报错"的机制 —— 默认 30 分钟太久，
    调试时建议调小。
    """
    world_size = env_world_size()
    if world_size <= 1:
        return DistContext(enabled=False, device=requested_device)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    has_cuda = requested_device.startswith("cuda") or (
        requested_device == "auto" and torch.cuda.is_available()
    )
    backend = "nccl" if has_cuda else "gloo"

    if not dist.is_initialized():
        rv_file = rendezvous_file()
        if rv_file is not None:
            store = dist.FileStore(rv_file, world_size)
            dist.init_process_group(
                backend=backend,
                store=store,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=timeout_minutes),
            )
        else:
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(minutes=timeout_minutes),
            )

    if backend == "nccl":
        # ⚠️ 必须在建任何 CUDA 张量之前绑定设备，否则每个进程都会默认用 cuda:0
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return DistContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
    )


def cleanup_distributed(ctx: DistContext) -> None:
    """销毁进程组。放在 `finally` 里，否则某个 rank 异常退出会让其他 rank 死等。"""
    if is_enabled(ctx) and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    """同步点：等所有 rank 到齐再继续。

    典型用途：rank 0 写文件之后、其他 rank 读文件之前 —— 不 barrier 就会
    读到一个还没写完的文件。
    """
    if is_enabled(ctx) and dist.is_initialized():
        dist.barrier()


# ============================================================
# 指标归约
# ============================================================
def reduce_sums(values: Sequence[float], ctx: DistContext | None) -> list[float]:
    """把所有 rank 上的一组数值**求和**后返回。

    为什么传"和"而不是传"均值"？
      因为各 rank 的样本数可能不同（尤其是验证集按 stride 切分时），
      「均值的均值 ≠ 全局均值」。正确做法是各 rank 传 `(总和, 样本数)`，
      求和之后再相除 —— 即 `Σloss / Σn` 而不是 `mean(loss_r)`。

    单进程时原样返回（不引入任何通信开销）。
    """
    if not is_enabled(ctx) or not dist.is_initialized():
        return [float(v) for v in values]
    t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(v) for v in t.tolist()]


# ============================================================
# 数据切分
# ============================================================
class StridedSampler(Sampler[int]):
    """把 `[0, N)` 按 rank 交错切分：rank r 取 `r, r+W, r+2W, ...`。

    和 `DistributedSampler` 的区别：**不填充、不重复**。
    各 rank 的长度最多差 1，并起来正好是整个数据集。

    ⚠️ 验证/测试集必须用它而不是 `DistributedSampler`：
    `DistributedSampler(drop_last=False)` 会**重复样本**把长度凑成 world_size 的倍数，
    重复的样本会被算两次 → 准确率虚高。这个 bug 很隐蔽，因为数字看起来"很正常"。
    """

    def __init__(self, num_samples: int, rank: int = 0, world_size: int = 1) -> None:
        self.num_samples = int(num_samples)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank 必须落在 [0, {self.world_size})，收到 {self.rank}")
        self._indices = list(range(self.rank, self.num_samples, self.world_size))

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def build_train_sampler(
    dataset,
    ctx: DistContext | None,
    seed: int = 42,
    drop_last: bool = True,
):
    """训练集采样器：多进程时返回 `DistributedSampler`，单进程返回 `None`。

    `drop_last=True` 的作用不是"丢几个样本"，而是**保证各 rank 的 batch 数完全相等** ——
    少几个样本无所谓，deadlock 是致命的。

    返回的 sampler 必须在每个 epoch 调 `set_epoch(epoch)`，否则每轮 shuffle 顺序相同。
    """
    if not is_enabled(ctx):
        return None
    return DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
    )


def build_eval_sampler(dataset, ctx: DistContext | None):
    """验证/测试集采样器：用 `StridedSampler`（不重复），单进程返回 `None`。"""
    if not is_enabled(ctx):
        return None
    return StridedSampler(len(dataset), rank=ctx.rank, world_size=ctx.world_size)


def set_epoch(sampler, epoch: int) -> None:
    """给 sampler 设置 epoch（单进程 sampler=None 时是空操作）。

    单独抽成函数是为了让调用点显式 —— 这是最容易被忘掉的一步。
    """
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


# ============================================================
# 模型包装
# ============================================================
def wrap_model(model: nn.Module, ctx: DistContext | None) -> nn.Module:
    """把模型包成 DDP；单进程原样返回。

    几个参数为什么这么设
    --------------------
    - `device_ids=[local_rank]`：nccl 下必须显式指定，否则所有进程抢 `cuda:0`。
    - `broadcast_buffers=True`（默认）：每次前向开始把 rank 0 的 buffer
      （BN running stats 等）广播出去，保证推理行为一致。
    - `find_unused_parameters=False`（默认，**故意不打开**）：
      打开会让每次反向都遍历一遍计算图找未使用参数，明显变慢。
      只有当模型有分支在某些 batch 上不参与计算时（如多任务、MoE）才需要开。
      单任务 CNN/Transformer 保持 False。
    """
    if not is_enabled(ctx):
        return model

    if ctx.backend == "nccl":
        return nn.parallel.DistributedDataParallel(
            model, device_ids=[ctx.local_rank], output_device=ctx.local_rank
        )
    # gloo / CPU：没有 device_ids 可传
    return nn.parallel.DistributedDataParallel(model)


def unwrap_model(model: nn.Module) -> nn.Module:
    """剥掉 DDP（`.module`）与 `torch.compile`（`._orig_mod`）的包装，拿到原始模型。

    ⚠️ 存 checkpoint 前**必须**先 unwrap。否则 `state_dict()` 的 key 会带
       `module.`（或 `_orig_mod.`）前缀，单进程加载时全部对不上 ——
       存出来一个别人用不了的权重文件，而且当时看不出任何异常。

    为什么要**循环**剥：两种包装可以嵌套（`compile(DDP(model))` 或反过来），
    只剥一层就会漏；`getattr(model, "module", model)` 这种写法也兼容
    DataParallel 和其它包装器。

    这里已经为此栽过一次（加 DDP 时忘了 unwrap），所以补了嵌套用例
    `test_unwrap_model_handles_nested_wrappers` 把两层的组合钉死。
    """
    while True:
        if hasattr(model, "_orig_mod"):     # torch.compile 的 OptimizedModule
            model = model._orig_mod
        elif hasattr(model, "module"):      # DDP / DataParallel
            model = model.module
        else:
            return model


def maybe_convert_sync_bn(model: nn.Module, enabled: bool, ctx: DistContext | None) -> bool:
    """把 BatchNorm 换成 SyncBatchNorm（原地替换子模块）。返回是否真的换了。

    为什么需要：DDP **不会**同步 BN 的 running stats，每个 rank 只在自己那批数据上
    统计。batch 小的时候（比如每卡 32）统计量噪声很大，会拖累精度。
    `SyncBatchNorm` 在反向时额外做一次 all-reduce，用全局 batch 的统计量。

    ⚠️ **SyncBatchNorm 只在 CUDA 上可用**（实测踩到的坑）
    --------------------------------------------------
    `convert_sync_batchnorm()` 只是换掉模块类型，**不会报错**；
    但换成之后只要一前向，CPU 上就会抛：

        ValueError: SyncBatchNorm expected input tensor to be on GPU or XPU or privateuseone

    所以「在 CPU + gloo 的 DDP 上开 --sync_bn」会得到一个"配置看起来生效了、
    训练第一步才崩"的结果 —— 非常难查。
    这里直接按 backend 判定：**只有 nccl（即真多卡 GPU）才允许转换**，
    CPU 上返回 False 让调用方给出明确提示。

    代价：多一次通信。所以即便在 GPU 上默认也是关闭的。
    """
    if not (is_enabled(ctx) and enabled):
        return False
    if ctx.backend != "nccl":
        return False
    if not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        return False
    nn.SyncBatchNorm.convert_sync_batchnorm(model)   # 原地替换子模块并返回同一对象
    return True


def ddp_no_sync(model: nn.Module, ctx: DistContext | None, skip: bool):
    """在"不该同步梯度"的 micro-step 上跳过 all-reduce。

    梯度累积时只有最后一个 micro-step 需要 all-reduce。前面几步用 `no_sync()`
    跳过通信 —— 通信量直接从 `accum` 倍降到 1 倍。

    ⚠️ 注意：**不跳也不会算错**，只是白通信。
       因为 DDP 平均是线性的：`Σ mean_r(g_i) == mean_r(Σ g_i)`。
       所以这是一个纯性能优化，正确的说法是"能省不少通信"，而不是"不加会错"。

    返回上下文管理器（单进程时是空操作）。
    """
    if not (is_enabled(ctx) and skip) or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


__all__ = [
    "DistContext",
    "StridedSampler",
    "barrier",
    "build_eval_sampler",
    "build_train_sampler",
    "cleanup_distributed",
    "ddp_no_sync",
    "env_world_size",
    "is_enabled",
    "maybe_convert_sync_bn",
    "reduce_sums",
    "rendezvous_file",
    "set_epoch",
    "setup_distributed",
    "unwrap_model",
    "wrap_model",
]
