"""分布式（DDP）测试。

这些用例**全部在本进程内跑，不启动多进程、不需要 GPU**，所以 CI 上秒过。
真正的多进程链路由 `test_smoke.py::test_ddp_two_processes_end_to_end` 覆盖。

测试策略：DDP 的核心是一句数学命题，而不是"跑起来没报错"。所以这里最重要的是
`test_ddp_gradient_identity_*` 那几条 —— 它们直接验证
「DDP 平均梯度 == 单进程大 batch 的梯度」这个不变量。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.distributed import (
    DistContext,
    StridedSampler,
    build_eval_sampler,
    build_train_sampler,
    env_world_size,
    maybe_convert_sync_bn,
    reduce_sums,
    rendezvous_file,
    set_epoch,
    setup_distributed,
    unwrap_model,
    wrap_model,
)
from src.model import build_model


# ============================================================
# 测试辅助
# ============================================================
def disable_dropout(model: nn.Module) -> None:
    """把 Dropout 关掉，让前向完全确定。

    ⚠️ 做梯度等价性验证时**必须**关掉 Dropout：
       单进程跑一次 batch 和分两次跑，消耗的随机数个数不同，
       mask 自然不同 —— 那是随机性造成的差异，不是 DDP 的数学问题。
    """
    for m in model.modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.p = 0.0


def grads_of(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """在给定 batch 上反传一次，返回梯度列表（先清零，保证是"这一次"的梯度）。"""
    model.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def flat_grads(model, x, y) -> torch.Tensor:
    return torch.cat([g.flatten() for g in grads_of(model, x, y)])


@pytest.fixture
def batch_128():
    """固定 seed 的 (x, y)，切成两半 A / B，各 64 条。"""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(128, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (128,), generator=g)
    return x, y, x[:64], y[:64], x[64:], y[64:]


@pytest.fixture
def plain_ctx():
    return DistContext(enabled=False)


@pytest.fixture
def ddp_ctx():
    """一个"声称自己是 world_size=2 的 rank 0"的上下文。

    注意：这里**没有真的初始化 gloo 进程组**。下面所有用到它的测试走的都是
    "不依赖进程组"的分支（sampler / 开关判断 / 模型包装的短路）。
    真正需要通信的部分由多进程冒烟测试覆盖 —— 不假装本进程内能测出集合通信。
    """
    return DistContext(enabled=True, rank=0, local_rank=0, world_size=2, backend="gloo", device="cpu")


# ============================================================
# 环境探测
# ============================================================
def test_env_world_size_defaults_to_one(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert env_world_size() == 1


def test_env_world_size_reads_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "6")
    assert env_world_size() == 6


def test_env_world_size_survives_garbage(monkeypatch):
    """环境变量被人写坏时不该把训练带崩，退回单进程即可。"""
    monkeypatch.setenv("WORLD_SIZE", "not-a-number")
    assert env_world_size() == 1


def test_setup_distributed_is_disabled_without_torchrun(monkeypatch):
    """没经 torchrun 启动时，必须完全短路 —— 这是"同一份代码单卡也能跑"的关键。"""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ctx = setup_distributed("cpu")
    assert ctx.enabled is False
    assert ctx.world_size == 1
    assert ctx.is_main is True
    assert "单进程" in ctx.describe()


def test_rendezvous_file_reads_env(monkeypatch):
    monkeypatch.delenv("DDP_RENDEZVOUS_FILE", raising=False)
    assert rendezvous_file() is None
    monkeypatch.setenv("DDP_RENDEZVOUS_FILE", "   ")
    assert rendezvous_file() is None, "空字符串要当作没设置"
    monkeypatch.setenv("DDP_RENDEZVOUS_FILE", "/tmp/rv")
    assert rendezvous_file() == "/tmp/rv"


def test_is_main_only_for_rank_zero():
    assert DistContext(enabled=True, rank=0, world_size=2).is_main is True
    assert DistContext(enabled=True, rank=1, world_size=2).is_main is False


# ============================================================
# `ctx=None` 的约定：公开 API 用 None 表示"单进程"
# ============================================================
def test_is_enabled_treats_none_as_disabled():
    from src.distributed import is_enabled

    assert is_enabled(None) is False
    assert is_enabled(DistContext(enabled=False)) is False
    assert is_enabled(DistContext(enabled=True, rank=0, world_size=2)) is True


def test_all_helpers_accept_none_ctx():
    """所有接受 `ctx` 的工具函数都必须能吃下 `None`。

    这是加 DDP 时真实踩到的坑：`train.py` / `data.py` 的公开 API 约定
    `ctx=None` = 单进程（调用方不该为了单卡去构造 DistContext），
    但 `ddp_no_sync` 里写了裸的 `ctx.enabled` → **一次性打挂 12 个已有测试**，
    而且报错发生在训练循环中间。

    这条测试把这个约定固定下来：以后新加的分布式工具函数，
    只要忘了判空就会在这里红。
    """
    from src.distributed import (
        barrier,
        cleanup_distributed,
        ddp_no_sync,
        is_enabled,
        reduce_sums,
    )

    model = nn.Linear(4, 2)
    dataset = torch.utils.data.TensorDataset(torch.zeros(8, 3), torch.zeros(8, dtype=torch.long))

    # 这些都不该抛异常
    assert is_enabled(None) is False
    assert reduce_sums([1.0, 2.0], None) == [1.0, 2.0]
    assert build_train_sampler(dataset, None) is None
    assert build_eval_sampler(dataset, None) is None
    assert wrap_model(model, None) is model
    assert maybe_convert_sync_bn(model, enabled=True, ctx=None) is False
    barrier(None)
    cleanup_distributed(None)

    with ddp_no_sync(model, None, skip=True):     # 空操作上下文
        pass
    with ddp_no_sync(None, None, skip=False):     # model 是 None 也不该炸
        pass


# ============================================================
# 数据切分（DDP 最容易出错的地方）
# ============================================================
@pytest.mark.parametrize("n,world_size", [(100, 2), (100, 3), (900, 4), (7, 4), (1, 3)])
def test_strided_sampler_partitions_without_overlap(n, world_size):
    """各 rank 的索引：并起来 == 全集，两两不相交，长度最多差 1。

    这是验证集切分的正确性底线 —— 有重复 → 指标虚高；漏样本 → 指标不准。
    """
    parts = [list(StridedSampler(n, rank=r, world_size=world_size)) for r in range(world_size)]
    merged = [i for part in parts for i in part]

    assert sorted(merged) == list(range(n)), "并集必须正好是整个数据集"
    assert len(set(merged)) == n, "不能有任何重复样本"
    lengths = [len(p) for p in parts]
    assert max(lengths) - min(lengths) <= 1, "各 rank 长度最多差 1"


def test_strided_sampler_len_matches_iter():
    s = StridedSampler(10, rank=1, world_size=3)
    assert len(s) == len(list(s)) == len(range(1, 10, 3))


def test_strided_sampler_rejects_bad_rank():
    with pytest.raises(ValueError, match="rank"):
        StridedSampler(10, rank=5, world_size=2)


def test_distributed_sampler_gives_each_rank_equal_length():
    """训练集各 rank 的 batch 数**必须相等**，否则会在 all-reduce 处死锁。"""
    dataset = torch.utils.data.TensorDataset(torch.zeros(900, 3), torch.zeros(900, dtype=torch.long))
    lens = [
        len(build_train_sampler(dataset, DistContext(enabled=True, rank=r, world_size=3), seed=0))
        for r in range(3)
    ]
    assert len(set(lens)) == 1, f"各 rank 长度应相同，实际 {lens}"


def test_set_epoch_changes_shuffle_order():
    """不调 set_epoch 的话，每个 epoch 的 shuffle 顺序完全一样 —— 等于没打乱。

    这条测试就是把这个 bug 钉住：同一个 epoch 的两次迭代顺序必然相同，
    不同 epoch 之间必然不同。
    """
    dataset = list(range(64))
    sampler = build_train_sampler(
        dataset, DistContext(enabled=True, rank=0, world_size=2), seed=0
    )

    set_epoch(sampler, 0)
    order_e0_first = list(sampler)
    order_e0_second = list(sampler)
    assert order_e0_first == order_e0_second, "同一 epoch 内必须可复现"

    set_epoch(sampler, 1)
    assert list(sampler) != order_e0_first, "换个 epoch 顺序必须变（否则 shuffle 失效）"


def test_set_epoch_is_noop_on_none():
    """单进程时 sampler 是 None，set_epoch 必须是空操作而不是崩掉。"""
    set_epoch(None, 3)          # 不抛异常即通过


def test_build_samplers_return_none_when_not_distributed():
    dataset = torch.utils.data.TensorDataset(torch.zeros(10, 3), torch.zeros(10, dtype=torch.long))
    ctx = DistContext(enabled=False)
    assert build_train_sampler(dataset, ctx) is None
    assert build_eval_sampler(dataset, ctx) is None


def test_build_eval_sampler_does_not_duplicate():
    """验证集采样器不能用 DistributedSampler(drop_last=False) —— 那会重复样本。"""
    dataset = torch.utils.data.TensorDataset(torch.zeros(101, 3), torch.zeros(101, dtype=torch.long))
    indices = [
        i
        for r in range(4)
        for i in build_eval_sampler(dataset, DistContext(enabled=True, rank=r, world_size=4))
    ]
    assert len(indices) == 101
    assert len(set(indices)) == 101


# ============================================================
# 指标归约
# ============================================================
def test_reduce_sums_is_identity_when_disabled():
    """单进程时归约必须完全短路（不引入任何通信开销）。"""
    assert reduce_sums([1.5, 2, 3], DistContext(enabled=False)) == [1.5, 2.0, 3.0]


def test_sample_weighted_mean_is_not_mean_of_means():
    """把「Σloss / Σn」和「mean(各 rank 的 loss)」的差异写死成测试。

    各 rank 样本数不同时，均值的均值 ≠ 全局均值 —— 这正是 `evaluate()` 里
    必须先 all-reduce「和」再相除、而不能各自算均值再平均的原因。
    """
    # rank0：10 个样本，平均 loss 0.1；rank1：90 个样本，平均 loss 1.0
    sums, counts = [1.0, 90.0], [10.0, 90.0]
    global_mean = sum(sums) / sum(counts)                 # 正确：0.91
    mean_of_means = sum(s / c for s, c in zip(sums, counts)) / 2   # 错误：0.55
    assert global_mean == pytest.approx(0.91)
    assert mean_of_means == pytest.approx(0.55)
    assert abs(global_mean - mean_of_means) > 0.3, "差异必须显著到值得写进文档"


# ============================================================
# ⭐ 核心：DDP 的数学不变量
# ============================================================
def test_ddp_gradient_identity_without_bn(batch_128):
    """**DDP 全部的数学内容就是这一条**：平均各 rank 的梯度 == 用大 batch 算的梯度。

        grad(batch 128)  ==  ( grad(batch A 64) + grad(batch B 64) ) / 2

    等号成立需要两个条件，本用例刻意把两个都满足（mlp + 关 Dropout）：
      1. 损失是**逐样本可加的均值**（CrossEntropyLoss(reduction='mean') 满足）
      2. 前向**不存在跨样本耦合**（没有 BatchNorm 这种"看整批数据"的层）

    这也是为什么 DDP 无需改动模型就能用 —— 它只是把"一次大 batch 的梯度"
    拆成"多次小 batch 的平均"。
    """
    torch.manual_seed(0)
    model = build_model("mlp", in_channels=1, num_classes=10, image_size=28)
    disable_dropout(model)

    x, y, xa, ya, xb, yb = batch_128

    torch.manual_seed(0)
    model_a = build_model("mlp", in_channels=1, num_classes=10, image_size=28)
    disable_dropout(model_a)
    model_a.load_state_dict(model.state_dict())

    torch.manual_seed(0)
    model_b = build_model("mlp", in_channels=1, num_classes=10, image_size=28)
    disable_dropout(model_b)
    model_b.load_state_dict(model.state_dict())

    full = flat_grads(model, x, y)
    half_a = flat_grads(model_a, xa, ya)
    half_b = flat_grads(model_b, xb, yb)
    ddp_equivalent = (half_a + half_b) / 2

    max_diff = (full - ddp_equivalent).abs().max().item()
    assert torch.allclose(full, ddp_equivalent, atol=1e-6, rtol=1e-5), (
        f"DDP 梯度不变量被破坏，最大偏差 {max_diff:.3e}"
    )


def test_batchnorm_breaks_the_ddp_identity(batch_128):
    """**反例**：模型里有 BatchNorm 时，上面那条等式不再成立。

    BN 在训练态用**当前 batch 的统计量**做归一化，所以它"看得见整批数据"，
    前向不再逐样本独立：
      - 单进程 batch 128 → 统计量在 128 条上算
      - DDP 各 rank 64 条   → 统计量各自在 64 条上算
    两者算出来的归一化结果不同，梯度自然不同。

    这条测试是**故意断言"不成立"**的：
    它把"DDP 不跨卡同步 BN 统计量"这个坑固定成一个可执行的文档，
    而不是靠注释里的一句话说给人听。想要真正等价就得换 SyncBatchNorm。
    """
    torch.manual_seed(0)
    base = build_model("small_cnn", in_channels=1, num_classes=10, image_size=28)

    def clone():
        m = build_model("small_cnn", in_channels=1, num_classes=10, image_size=28)
        m.load_state_dict(base.state_dict())
        m.train()
        return m

    x, y, xa, ya, xb, yb = batch_128
    full = flat_grads(clone(), x, y)
    combined = (flat_grads(clone(), xa, ya) + flat_grads(clone(), xb, yb)) / 2

    max_diff = (full - combined).abs().max().item()
    assert max_diff > 1e-4, (
        "预期 BatchNorm 会破坏梯度等价性，但差异只有 "
        f"{max_diff:.3e} —— 如果这里失败，说明上游对 BN 的行为变了，请重新审视 README 的说明"
    )


# ============================================================
# 模型包装
# ============================================================
def test_wrap_model_is_identity_when_disabled(plain_ctx):
    model = nn.Linear(4, 2)
    assert wrap_model(model, plain_ctx) is model


def test_unwrap_model_returns_inner_module():
    """存 checkpoint 前必须 unwrap，否则 key 会带 `module.` 前缀、别人加载不了。"""
    inner = nn.Linear(4, 2)
    wrapped = nn.Module()
    wrapped.module = inner
    assert unwrap_model(wrapped) is inner
    assert unwrap_model(inner) is inner


def test_sync_bn_requires_distributed_and_batchnorm(ddp_ctx, plain_ctx):
    cnn = build_model("small_cnn", in_channels=1, num_classes=10, image_size=28)
    mlp = build_model("mlp", in_channels=1, num_classes=10, image_size=28)

    assert maybe_convert_sync_bn(cnn, enabled=False, ctx=ddp_ctx) is False, "开关关着不该换"
    assert maybe_convert_sync_bn(cnn, enabled=True, ctx=plain_ctx) is False, "单进程不该换"
    assert maybe_convert_sync_bn(mlp, enabled=True, ctx=ddp_ctx) is False, "没有 BN 不该换"


def test_sync_bn_is_refused_on_cpu_backend():
    """⭐ SyncBatchNorm 只在 CUDA 上可用，CPU（gloo）必须拒绝转换。

    这是写测试时抓到的真 bug：`convert_sync_batchnorm()` 只换模块类型、
    **不会报错**，但换成之后 CPU 上第一次前向就抛
    `ValueError: SyncBatchNorm expected input tensor to be on GPU or XPU`。
    于是「CPU 多进程 + --sync_bn」会表现为"配置生效了、训练第一步才崩"。

    所以这里断言：gloo 后端下必须返回 False（拒绝），而不是假装换成功。
    """
    cnn = build_model("small_cnn", in_channels=1, num_classes=10, image_size=28)
    gloo_ctx = DistContext(enabled=True, rank=0, local_rank=0, world_size=2, backend="gloo")
    assert maybe_convert_sync_bn(cnn, enabled=True, ctx=gloo_ctx) is False
    assert not any(isinstance(m, nn.SyncBatchNorm) for m in cnn.modules()), (
        "被拒绝时不能留下任何已转换的痕迹"
    )


def test_syncbatchnorm_forward_is_cuda_only_canary():
    """canary：把「SyncBatchNorm 在 CPU 上前向会抛 ValueError」这个事实钉住。

    这是**上游的现状**而不是我们想要的行为。哪天 PyTorch 支持 CPU 了，
    这条测试会失败，提醒我们回来放开 `maybe_convert_sync_bn` 的后端限制。
    """
    import torch.distributed as dist

    m = nn.SyncBatchNorm(4)
    x = torch.randn(2, 4, 3, 3)

    # 构造一个最小的单进程组，让 SyncBatchNorm 通过后端检查
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        rv = Path(d) / "rv"
        rv.write_bytes(b"")
        store = dist.FileStore(str(rv), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
        try:
            with pytest.raises(ValueError, match="GPU|XPU"):
                m(x)
        except AssertionError:
            pytest.fail(
                "SyncBatchNorm 现在能在 CPU 上前向了 —— 上游已变，"
                "请放开 maybe_convert_sync_bn 的 nccl 后端限制并更新 README"
            )
        finally:
            dist.destroy_process_group()
