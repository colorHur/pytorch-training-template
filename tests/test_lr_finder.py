"""学习率 finder 测试。

这个模块最容易「静默给出错误答案」—— 曲线画得漂漂亮亮，建议值却是错的。
所以测的不是「跑完不报错」，而是三条**错了就一定会得出错误建议**的不变量：

  1. **等比取点**：等差扫的话，1e-7 附近几步就跨完了，log 轴上根本不均匀
  2. **平滑必须带偏差修正**：朴素 EMA 的初值是 0，前几步被人为压低 →
     「最陡下降」落在第一步 → 建议值退化成 min_lr（而且看起来完全正常）
  3. **扫描结束后权重必须逐位还原**：扫描会故意把 lr 推到发散，
     不还原就是把 NaN 权重交给接下来的训练

另外守住几个「必须关掉」的开关（AMP / 梯度裁剪 / 梯度累积），
它们任意一个开着，曲线就不是「loss vs lr」了。
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.data import SyntheticDataset
from src.lr_finder import (
    DEFAULT_BETA,
    LRSweepResult,
    SweepPoint,
    ascii_curve,
    geometric_lrs,
    resolve_sweep_config,
    run_lr_sweep,
    smooth_losses,
    suggest_lr,
)
from src.model import build_model


def _toy_model() -> nn.Module:
    torch.manual_seed(0)
    return build_model("small_cnn", image_size=28)


def _loader(n: int = 512, batch_size: int = 32) -> DataLoader:
    return DataLoader(SyntheticDataset(n=n, train=True), batch_size=batch_size, shuffle=False)


# ============================================================
# 1) 等比取点
# ============================================================
def test_geometric_lrs_hits_both_endpoints():
    lrs = geometric_lrs(1e-7, 1.0, 100)
    assert len(lrs) == 100
    assert lrs[0] == pytest.approx(1e-7)
    assert lrs[-1] == pytest.approx(1.0)


def test_geometric_lrs_has_a_constant_ratio():
    lrs = geometric_lrs(1e-6, 1e-1, 11)
    ratios = [b / a for a, b in zip(lrs, lrs[1:])]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)
    assert ratios[0] == pytest.approx(10 ** (5 / 10))     # 5 个数量级 / 10 个间隔


def test_geometric_lrs_is_uniform_on_a_log_axis():
    """log 轴上相邻两点的间距恒定 —— 这才是「每个数量级分到同样步数」。"""
    lrs = geometric_lrs(1e-7, 1.0, 50)
    gaps = [math.log10(b) - math.log10(a) for a, b in zip(lrs, lrs[1:])]
    assert all(g == pytest.approx(gaps[0], rel=1e-9) for g in gaps)


@pytest.mark.parametrize(
    "min_lr, max_lr, num_steps",
    [
        (0.0, 1.0, 10),        # log 轴上不能从 0 出发
        (-1e-7, 1.0, 10),
        (1e-3, 1e-3, 10),      # 上下界必须有间隔，否则曲线没有任何动态范围
        (1e-2, 1e-3, 10),      # 上下界写反
        (1e-7, 1.0, 1),        # 一个点构不成扫描
    ],
)
def test_geometric_lrs_rejects_invalid_range(min_lr, max_lr, num_steps):
    with pytest.raises(ValueError):
        geometric_lrs(min_lr, max_lr, num_steps)


# ============================================================
# 2) 平滑：偏差修正是必须的
# ============================================================
def test_smoothing_is_unbiased_at_the_start():
    """平滑后的第一个点必须**等于**原始 loss。

    朴素 EMA 的初值是 0，第一步会得到 `(1-beta) * loss_0` ≈ 0.02 * loss_0 ——
    一个被人为压低的数。于是「最陡下降」必然落在曲线开头，建议值退化成 min_lr。
    这条断言守住的就是最后那个 `1 - beta**(t+1)`。
    """
    losses = [10.0, 9.0, 8.0]
    smoothed = smooth_losses(losses, beta=0.98)

    assert smoothed[0] == pytest.approx(losses[0])

    # 对照：朴素 EMA 的第一步有多离谱
    naive_first = (1 - 0.98) * losses[0]
    assert naive_first == pytest.approx(0.2)
    assert smoothed[0] > 20 * naive_first


def test_smoothing_eventually_tracks_the_series():
    """偏差修正只在开头起作用，后面应该收敛到普通 EMA 的行为。"""
    losses = [5.0] * 200
    smoothed = smooth_losses(losses, beta=DEFAULT_BETA)
    assert smoothed[-1] == pytest.approx(5.0, abs=1e-6)


def test_smoothing_damps_a_single_spike():
    losses = [3.0] * 10 + [30.0] + [3.0] * 10
    smoothed = smooth_losses(losses, beta=0.9)
    assert smoothed[10] < losses[10] / 2          # 尖峰被明显压住
    assert max(smoothed) < 10.0                   # 10 倍尖峰不该原样透传


def test_smoothing_of_a_constant_series_is_that_constant():
    losses = [2.5] * 50
    smoothed = smooth_losses(losses, beta=0.98)
    assert all(v == pytest.approx(2.5) for v in smoothed)


# ============================================================
# 3) 选点：最低点之前、最陡下降处
# ============================================================
def test_suggest_picks_the_steepest_descent():
    lrs = [1e-4, 1e-3, 1e-2, 1e-1]
    smoothed = [5.0, 3.0, 2.9, 6.0]      # 最低点在第 2 个，最陡下降在第 1 个
    index, lr = suggest_lr(lrs, smoothed)
    assert index == 1
    assert lr == pytest.approx(1e-3)


def test_suggest_ignores_the_rising_part():
    """最低点之后即使「升得比降得还猛」，也不能被选中 —— 那不是下降。"""
    lrs = [1e-4, 1e-3, 1e-2, 1e-1]
    smoothed = [5.0, 4.0, 1.0, 99.0]     # 最后一段的"斜率"绝对值最大，但方向朝上
    index, lr = suggest_lr(lrs, smoothed)
    assert index == 2
    assert lr == pytest.approx(1e-2)


def test_suggest_on_a_monotone_increasing_curve_falls_back_to_the_start():
    """整条曲线单调上升 → 没有可用的下降段，退回 min_lr 由调用方给出提示。"""
    index, lr = suggest_lr([1e-7, 1e-3, 1.0], [9.0, 10.0, 11.0])
    assert (index, lr) == (0, 1e-7)


def test_suggest_handles_degenerate_inputs():
    assert suggest_lr([], []) == (0, 0.0)
    assert suggest_lr([1e-3], [1.0]) == (0, 1e-3)


def test_suggest_never_returns_a_point_after_the_minimum():
    """建议点必须落在「起点之后、最低点之前」这个区间里。"""
    lrs = geometric_lrs(1e-7, 1.0, 20)
    # 最低点在第 5 个；最陡下降也在第 5 个（从 2.0 掉到 0.5 那一段）
    smoothed = [4.0] * 3 + [3.5, 2.0, 0.5, 1.5, 4.0] + [9.0] * 12
    index, _ = suggest_lr(lrs, smoothed)
    i_min = smoothed.index(min(smoothed))
    assert 1 <= index <= i_min


# ============================================================
# 4) 扫描必须「不动模型」（最容易被忽略、后果最严重的一条）
# ============================================================
def test_sweep_restores_parameters_bit_for_bit():
    model = _toy_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    run_lr_sweep(model, _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=8)

    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), f"参数 {key} 没被还原"


def test_sweep_restores_batch_norm_buffers():
    """BN 的 running stats 会被扫描污染（扫描是 train 模式），必须一起还原。"""
    model = _toy_model()
    before = {k: v.clone() for k, v in model.state_dict().items() if "running" in k or "batches" in k}
    assert before, "模型里应该有 BN 的统计缓冲"

    run_lr_sweep(model, _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=8)

    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), f"缓冲 {key} 没被还原"


def test_sweep_restores_training_mode():
    model = _toy_model()
    model.eval()
    run_lr_sweep(model, _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=6)
    assert model.training is False


def test_sweep_survives_divergence_without_poisoning_the_model():
    """把 max_lr 推到必然发散的量级。

    扫描中途权重会变成 NaN —— 如果没还原，接下来的训练就从 NaN 出发，
    你会以为是「选中的 lr 有问题」，其实是 finder 没收拾干净。
    """
    model = _toy_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    result = run_lr_sweep(
        model, _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=12, max_lr=1e3
    )

    assert result.diverged_at_lr is not None, "这么大的 lr 应该被测出发散"
    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(after[key], value), f"参数 {key} 没被还原"
        assert torch.isfinite(value).all(), f"参数 {key} 是 NaN/Inf"


# ============================================================
# 5) 扫描要关掉的开关（开着一个，曲线就不是 loss-vs-lr 了）
# ============================================================
def test_sweep_config_turns_off_everything_that_would_corrupt_the_curve():
    cfg = TrainConfig(
        amp=True,
        grad_accum_steps=4,
        max_grad_norm=1.0,
        lr_scheduler="cosine",
        compile=True,
    )
    sweep = resolve_sweep_config(cfg)

    assert sweep.amp is False             # GradScaler 会引入与 lr 无关的噪声
    assert sweep.grad_accum_steps == 1    # 一个点 = 一次更新 = 一个 lr
    assert sweep.max_grad_norm == 0.0     # 裁剪会掩盖发散，而发散正是要看的拐点
    assert sweep.lr_scheduler == "none"   # lr 由扫描自己写
    assert sweep.compile is False         # 编译开销不属于这条曲线


def test_sweep_config_keeps_what_the_optimum_actually_depends_on():
    """优化器 / weight_decay / batch_size 必须原样保留 —— 最优点依赖它们。"""
    cfg = TrainConfig(
        optimizer="sgd", momentum=0.8, weight_decay=1e-2, batch_size=64, lr=1e-2
    )
    sweep = resolve_sweep_config(cfg)

    assert sweep.optimizer == "sgd"
    assert sweep.momentum == pytest.approx(0.8)
    assert sweep.weight_decay == pytest.approx(1e-2)
    assert sweep.batch_size == 64
    assert sweep.lr == pytest.approx(1e-2)   # 扫描会逐 step 覆盖它，但配置值不该被改


# ============================================================
# 6) 扫描本身
# ============================================================
def test_sweep_records_one_point_per_step():
    result = run_lr_sweep(
        _toy_model(), _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=10
    )
    assert len(result.points) <= 10
    assert [p.step for p in result.points] == list(range(len(result.points)))
    assert all(p.lr > 0 for p in result.points)


def test_sweep_suggestion_lies_inside_the_swept_range():
    result = run_lr_sweep(
        _toy_model(), _loader(), nn.CrossEntropyLoss(), TrainConfig(),
        num_steps=20, min_lr=1e-6, max_lr=1.0,
    )
    assert 1e-6 <= result.suggested_lr <= 1.0
    assert result.suggested_lr == pytest.approx(result.points[result.suggested_step].lr)


def test_sweep_never_recommends_a_point_past_the_minimum():
    """保证成立的版本：建议点不晚于最低点（构造上就不允许）。"""
    result = run_lr_sweep(
        _toy_model(), _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=30
    )
    min_idx = min(range(len(result.points)), key=lambda i: result.points[i].smoothed)
    assert result.suggested_step <= min_idx
    assert result.suggested_lr <= result.min_loss_lr


def test_sweep_recommends_before_the_minimum_on_a_realistic_curve():
    """实测观察：真实曲线上「最陡下降」确实严格早于最低点 —— 这正是 finder 的价值。

    最低点贴着发散边缘，用它训练迟早炸；要的是斜率最陡的那个点。
    """
    result = run_lr_sweep(
        _toy_model(), _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=30
    )
    min_idx = min(range(len(result.points)), key=lambda i: result.points[i].smoothed)
    assert result.suggested_step < min_idx, (
        f"建议点 {result.suggested_step} 没有早于最低点 {min_idx}"
    )


def test_sweep_is_deterministic_given_the_seed():
    """同样的 seed → 同样的曲线和建议值。finder 的结论必须是可复现的。"""

    def once() -> LRSweepResult:
        torch.manual_seed(7)
        model = build_model("small_cnn", image_size=28)
        return run_lr_sweep(
            model, _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=8
        )

    a, b = once(), once()
    assert a.suggested_lr == b.suggested_lr
    assert [p.loss for p in a.points] == [p.loss for p in b.points]


def test_sweep_warns_when_batches_are_reused():
    """数据不够时会循环复用 batch，曲线会被压低 —— 必须明说，不能悄悄糊过去。"""
    loader = _loader(n=64, batch_size=32)      # 只有 2 个 batch
    result = run_lr_sweep(
        _toy_model(), loader, nn.CrossEntropyLoss(), TrainConfig(), num_steps=6
    )
    assert any("循环复用" in note for note in result.notes)


def test_sweep_does_not_warn_when_there_is_enough_data():
    loader = _loader(n=512, batch_size=32)     # 16 个 batch
    result = run_lr_sweep(
        _toy_model(), loader, nn.CrossEntropyLoss(), TrainConfig(), num_steps=6
    )
    assert not any("循环复用" in note for note in result.notes)


def test_sweep_on_an_empty_loader_raises():
    with pytest.raises(ValueError, match="空的"):
        run_lr_sweep(
            _toy_model(), DataLoader([], batch_size=1), nn.CrossEntropyLoss(),
            TrainConfig(), num_steps=4,
        )


# ============================================================
# 7) 输出：JSON 可序列化 + 人话
# ============================================================
def test_result_is_json_serializable():
    result = run_lr_sweep(
        _toy_model(), _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=6
    )
    text = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "suggested_lr" in text
    assert len(json.loads(text)["points"]) == len(result.points)


def test_describe_reports_the_suggestion_and_the_curve():
    result = run_lr_sweep(
        _toy_model(), _loader(), nn.CrossEntropyLoss(), TrainConfig(), num_steps=12
    )
    text = result.describe(current_lr=1e-3)
    assert "建议 lr" in text
    assert "发散" in text
    grid = [line for line in text.splitlines() if "|" in line]
    assert any("X" in line for line in grid), "图上应该标出建议点"


def _fake_result(suggested_lr: float) -> LRSweepResult:
    return LRSweepResult(
        points=[SweepPoint(0, suggested_lr, 1.0, 1.0)],
        suggested_lr=suggested_lr,
        suggested_step=0,
        min_loss=1.0,
        min_loss_lr=suggested_lr,
    )


@pytest.mark.parametrize(
    "current_lr, expected",
    [(1e-3, "同量级"), (1e-1, "偏大"), (1e-5, "偏小")],
)
def test_compare_to_classifies_the_gap(current_lr, expected):
    assert expected in _fake_result(1e-3).compare_to(current_lr)


# ============================================================
# 8) ASCII 图（本仓库不依赖 matplotlib）
# ============================================================
def test_ascii_curve_marks_the_suggested_point():
    text = ascii_curve([1e-7, 1e-3, 1e-1], [5.0, 3.0, 2.0], mark_index=1)
    grid = [line for line in text.splitlines() if "|" in line]
    # 只在网格里数 —— 表头那句「X = 建议的 lr」说明里也有一个 X
    assert sum(line.count("X") for line in grid) == 1


def test_ascii_curve_has_the_requested_number_of_rows():
    text = ascii_curve([1e-7, 1.0], [3.0, 2.0], height=7)
    # 两行表头 + 主体 + 坐标轴 + 刻度标签
    assert len(text.splitlines()) == 4 + 7


def test_ascii_curve_survives_a_constant_series():
    text = ascii_curve([1e-7, 1e-3, 1.0], [2.0, 2.0, 2.0])
    assert "2.000" in text
    assert "." in text


def test_ascii_curve_handles_empty_input():
    assert "没有数据" in ascii_curve([], [])


def test_ascii_curve_grid_is_ascii_only():
    """曲线本体只允许 ASCII —— 这份输出会经过 cp1252 管道，不能依赖任何盒线字符。

    （表头那句中文提示不在此列：它由 `force_utf8_stdout()` 负责。）
    """
    text = ascii_curve([1e-7, 1e-3, 1.0], [5.0, 3.0, 2.0], mark_index=1)
    grid = [line for line in text.splitlines() if "|" in line or "+" in line]
    assert grid, "应该画出了网格"
    assert all(line.isascii() for line in grid)
