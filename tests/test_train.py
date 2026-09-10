"""训练循环测试 —— 这是整个模板的心脏，也是面试最爱追问的地方。

测的不是"跑完不报错"，而是三条容易搞错的不变量：
  1. LR 调度在 warmup / 主体阶段的取值符合预期
  2. 梯度累积的**更新次数** = ceil(批次数 / 累积步数)（这是"等效 batch"的前提）
  3. zero_grad 的次数与时机；@torch.no_grad() 真的没建图
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.data import SyntheticDataset
from src.model import build_model
from src.train import build_lr_scheduler, evaluate, train_one_epoch

BASE_LR = 1e-3


# ============================================================
# 工具：能数调用次数的优化器
# ============================================================
class CountingAdamW(torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls += 1
        return super().zero_grad(*args, **kwargs)


def _setup(n_samples: int, batch_size: int, accum: int, **cfg_kwargs):
    torch.manual_seed(0)
    cfg = TrainConfig(
        grad_accum_steps=accum, amp=False, log_interval=10_000, **cfg_kwargs
    )
    model = build_model("small_cnn", image_size=28)
    optimizer = CountingAdamW(model.parameters(), lr=BASE_LR)
    loader = DataLoader(
        SyntheticDataset(n=n_samples, train=True), batch_size=batch_size, shuffle=False
    )
    criterion = nn.CrossEntropyLoss()
    return cfg, model, optimizer, loader, criterion


# ============================================================
# LR 调度器
# ============================================================
def _scheduler(name: str, total_steps: int, warmup_steps: int = 0):
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    return build_lr_scheduler(opt, name, total_steps, warmup_steps)


def test_constant_scheduler():
    sched = _scheduler("none", total_steps=100)
    assert sched(0) == [BASE_LR]
    assert sched(99) == [BASE_LR]


def test_warmup_ramps_linearly_from_zero():
    sched = _scheduler("cosine", total_steps=100, warmup_steps=10)
    assert sched(0) == pytest.approx([BASE_LR * 0.1])
    assert sched(4) == pytest.approx([BASE_LR * 0.5])
    assert sched(9) == pytest.approx([BASE_LR * 1.0])   # warmup 结束正好到基准值


def test_warmup_is_strictly_increasing():
    sched = _scheduler("cosine", total_steps=200, warmup_steps=20)
    lrs = [sched(s)[0] for s in range(20)]
    assert all(b > a for a, b in zip(lrs, lrs[1:]))


def test_cosine_endpoints_and_midpoint():
    sched = _scheduler("cosine", total_steps=100)
    assert sched(0) == pytest.approx([BASE_LR])
    assert sched(50) == pytest.approx([BASE_LR * 0.5])   # 1/4 周期处
    assert sched(100) == pytest.approx([0.0], abs=1e-12)


def test_cosine_is_monotonic_after_warmup():
    sched = _scheduler("cosine", total_steps=100, warmup_steps=10)
    lrs = [sched(s)[0] for s in range(10, 101)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))


def test_step_scheduler_halves_three_times():
    sched = _scheduler("step", total_steps=100)
    assert sched(0) == pytest.approx([BASE_LR])          # progress 0   -> 0.5^0
    assert sched(40) == pytest.approx([BASE_LR * 0.5])   # progress 0.4 -> 0.5^1
    assert sched(70) == pytest.approx([BASE_LR * 0.25])  # progress 0.7 -> 0.5^2
    assert sched(100) == pytest.approx([BASE_LR * 0.125])  # progress 1.0 -> 0.5^3


def test_progress_never_exceeds_one():
    """步数超出 total_steps 时不能反弹（否则训练末尾 lr 会回升）。"""
    sched = _scheduler("cosine", total_steps=100)
    assert sched(999) == pytest.approx([0.0], abs=1e-12)
    assert sched(1000) == pytest.approx([0.0], abs=1e-12)


def test_unknown_scheduler_raises():
    with pytest.raises(KeyError, match="未知 lr_scheduler"):
        _scheduler("magic", total_steps=10)(5)


def test_scheduler_handles_multiple_param_groups():
    opt = torch.optim.AdamW(
        [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3},
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-2},
        ]
    )
    sched = build_lr_scheduler(opt, "cosine", total_steps=100)
    group_lrs = sched(50)
    assert len(group_lrs) == 2
    assert group_lrs[0] == pytest.approx(5e-4)
    assert group_lrs[1] == pytest.approx(5e-3)          # 各组按自己的基准等比缩放


# ============================================================
# evaluate
# ============================================================
def test_evaluate_reports_loss_and_acc():
    model = build_model("small_cnn", image_size=28)
    loader = DataLoader(SyntheticDataset(n=64, train=False), batch_size=32)
    stats = evaluate(model, loader, nn.CrossEntropyLoss(), "cpu")
    assert set(stats) == {"loss", "acc"}
    assert 0.0 <= stats["acc"] <= 1.0
    assert stats["loss"] > 0.0


def test_evaluate_switches_to_eval_mode():
    model = build_model("small_cnn", image_size=28)
    model.train()
    loader = DataLoader(SyntheticDataset(n=32, train=False), batch_size=32)
    evaluate(model, loader, nn.CrossEntropyLoss(), "cpu")
    assert model.training is False


def test_evaluate_does_not_build_graph():
    """忘了 @torch.no_grad() 会让评估阶段显存暴涨 —— 用"梯度没被改动"来验证。"""
    model = build_model("small_cnn", image_size=28)
    with torch.no_grad():
        model(torch.randn(2, 1, 28, 28))       # 先造出一些叶子张量的状态
    param = next(model.parameters())
    param.grad = torch.full_like(param, 123.0)
    loader = DataLoader(SyntheticDataset(n=32, train=False), batch_size=32)
    evaluate(model, loader, nn.CrossEntropyLoss(), "cpu")
    assert torch.all(param.grad == 123.0), "@torch.no_grad() 失效，评估时算了梯度"


def test_evaluate_on_empty_loader_raises_rather_than_silently_returning_garbage():
    """空 loader 在真实项目里会出现（小数据集 + val_ratio 过小）。

    要求：明确报错，而不是返回 nan / 0 让下游悄悄用错值。
    """
    model = build_model("small_cnn", image_size=28)
    loader = DataLoader([], batch_size=1)
    with pytest.raises(ZeroDivisionError):
        evaluate(model, loader, nn.CrossEntropyLoss(), "cpu")


# ============================================================
# train_one_epoch：更新次数（梯度累积的核心不变量）
# ============================================================
@pytest.mark.parametrize(
    "n_samples, batch_size, accum, expected_updates",
    [
        (64, 16, 1, 4),    # 4 批，不累积 -> 4 次更新
        (64, 16, 4, 1),    # 4 批，累积 4 -> 1 次更新
        (100, 16, 3, 3),   # 7 批，累积 3 -> ceil(7/3) = 3
        (100, 16, 7, 1),   # 7 批，累积 7 -> 1
        (32, 16, 5, 1),    # 2 批，累积 5（超过批次数）-> 末尾兜底更新 1 次
    ],
)
def test_gradient_accumulation_update_count(n_samples, batch_size, accum, expected_updates):
    """更新次数必须等于 ceil(批次数 / 累积步数)。

    这条错了，"等效 batch size" 就是错的，所有梯度累积的实验结论都作废。
    """
    cfg, model, optimizer, loader, criterion = _setup(n_samples, batch_size, accum)
    _, global_step = train_one_epoch(
        model, loader, optimizer, criterion, "cpu", epoch=1, cfg=cfg, global_step=0
    )
    assert optimizer.step_calls == expected_updates
    assert global_step == expected_updates


def test_zero_grad_called_once_per_accumulation_window():
    cfg, model, optimizer, loader, criterion = _setup(100, 16, 3)
    train_one_epoch(
        model, loader, optimizer, criterion, "cpu", epoch=1, cfg=cfg, global_step=0
    )
    # 7 批、每 3 批清零一次 -> i=0,3,6 共 3 次
    assert optimizer.zero_grad_calls == 3


def test_train_one_epoch_returns_stats_and_keeps_train_mode():
    cfg, model, optimizer, loader, criterion = _setup(64, 32, 1)
    model.eval()                       # 故意先切成 eval，验证函数会切回 train
    stats, _ = train_one_epoch(
        model, loader, optimizer, criterion, "cpu", epoch=1, cfg=cfg, global_step=0
    )
    assert set(stats) == {"loss", "acc", "time", "lr"}
    assert stats["time"] >= 0.0
    assert model.training is True


def test_lr_schedule_is_applied_on_update_steps():
    """lr 在**更新步**才被写入 param_group，用的 step 是"更新前"的全局步数。

    写早或写晚都会和 optimizer 状态错位 —— 所以这里对齐到调度函数在 step=3 的取值，
    而不是想当然地以为"跑完就该是终点值"。
    """
    cfg, model, optimizer, loader, criterion = _setup(64, 16, 1)   # 4 批 -> 4 次更新
    sched = build_lr_scheduler(optimizer, "cosine", total_steps=4)
    _, global_step = train_one_epoch(
        model, loader, optimizer, criterion, "cpu", epoch=1, cfg=cfg,
        global_step=0, lr_schedule=sched, total_steps=4,
    )
    assert global_step == 4
    assert optimizer.param_groups[0]["lr"] == pytest.approx(sched(3)[0])
    assert optimizer.param_groups[0]["lr"] < BASE_LR    # 余弦确实在退火


def test_grad_clipping_does_not_break_training():
    cfg, model, optimizer, loader, criterion = _setup(64, 16, 1, max_grad_norm=1.0)
    stats, _ = train_one_epoch(
        model, loader, optimizer, criterion, "cpu", epoch=1, cfg=cfg, global_step=0
    )
    assert math.isfinite(stats["loss"])


def test_longer_training_reduces_loss_on_learnable_data():
    """端到端的行为验证：合成数据是可学习的，训练应该真的让 loss 降下来。

    只跑 3 个 epoch，不需要 CUDA，几秒钟就完。
    """
    cfg = TrainConfig(epochs=3, grad_accum_steps=1, amp=False, lr=5e-3)
    torch.manual_seed(0)
    model = build_model("small_cnn", image_size=28)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(SyntheticDataset(n=256, train=True), batch_size=64, shuffle=True)
    test_loader = DataLoader(SyntheticDataset(n=128, train=False), batch_size=64)
    criterion = nn.CrossEntropyLoss()

    before = evaluate(model, test_loader, criterion, "cpu")["loss"]
    for epoch in range(cfg.epochs):
        train_one_epoch(
            model, train_loader, optimizer, criterion, "cpu",
            epoch=epoch + 1, cfg=cfg, global_step=epoch,
        )
    after = evaluate(model, test_loader, criterion, "cpu")["loss"]
    assert after < before * 0.5, f"loss 没降下来：{before:.3f} -> {after:.3f}"
