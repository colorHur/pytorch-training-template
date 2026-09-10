"""数据层测试。

重点不是"能不能读 MNIST"（那依赖下载），而是：
验证集切分对不对、meta 对不对、合成的离线数据是不是真的可学习。
"""

from __future__ import annotations

import pytest
import torch

from src.data import (
    DATASET_STATS,
    SyntheticDataset,
    build_dataloaders,
    build_transforms,
    split_train_val,
)

CPU = "cpu"


# ============================================================
# 合成数据集（测试与 CI 的基础设施）
# ============================================================
def test_synthetic_dataset_shapes():
    ds = SyntheticDataset(n=64, train=True)
    assert len(ds) == 64
    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert 0 <= int(y) < 10


def test_synthetic_dataset_is_deterministic():
    """同样参数两次构造必须完全一致，否则测试不可复现。"""
    a = SyntheticDataset(n=32, train=True)
    b = SyntheticDataset(n=32, train=True)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)


def test_synthetic_train_and_test_are_disjoint():
    """训练集和测试集必须不同，否则"测试准确率"是个假指标。"""
    train = SyntheticDataset(n=32, train=True)
    test = SyntheticDataset(n=32, train=False)
    assert not torch.equal(train.x, test.x)


def test_synthetic_dataset_is_learnable():
    """关键：合成数据的类别必须**可分**。

    如果造出来的假数据连 1-NN 都分不开，那么"训练 loss 下降"这类断言就失去意义。
    这里用最近邻做一次独立于模型的合理性检查。
    """
    ds = SyntheticDataset(n=400, train=True)
    flat = ds.x.flatten(1)
    # 随机取一批，用整个数据集当"模板库"做最近邻
    sim = torch.cdist(flat[:50], flat)               # (50, 400)
    sim[torch.arange(50), torch.arange(50)] = float("inf")   # 排除自己
    nn_idx = sim.argmin(dim=1)
    acc = (ds.y[nn_idx] == ds.y[:50]).float().mean().item()
    assert acc > 0.9, f"合成数据类间重叠太多，最近邻准确率仅 {acc:.2f}"


# ============================================================
# 验证集切分
# ============================================================
def test_split_train_val_sizes_and_disjoint():
    ds = SyntheticDataset(n=100, train=True)
    train, val = split_train_val(ds, val_ratio=0.2, seed=0)
    assert len(train) == 80 and len(val) == 20
    # 两个子集下标不能重叠（否则等于用验证集训练）
    assert set(train.indices).isdisjoint(set(val.indices))


def test_split_train_val_is_reproducible():
    ds = SyntheticDataset(n=100, train=True)
    a = split_train_val(ds, 0.2, seed=42)[0].indices
    b = split_train_val(ds, 0.2, seed=42)[0].indices
    assert list(a) == list(b)


def test_split_train_val_different_seed_differs():
    ds = SyntheticDataset(n=100, train=True)
    a = split_train_val(ds, 0.2, seed=1)[0].indices
    b = split_train_val(ds, 0.2, seed=2)[0].indices
    assert list(a) != list(b)


# ============================================================
# DataLoader 组装
# ============================================================
def test_build_dataloaders_offline():
    loaders = build_dataloaders(
        "synthetic", batch_size=32, val_ratio=0.2, num_workers=0, pin_memory=False
    )
    assert set(loaders) == {"train", "val", "test", "meta"}
    assert loaders["meta"]["num_classes"] == 10
    assert loaders["meta"]["in_channels"] == 1
    assert len(loaders["train"].dataset) == 800
    assert len(loaders["val"].dataset) == 200

    x, y = next(iter(loaders["train"]))
    assert x.shape[1:] == (1, 28, 28)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_build_dataloaders_unknown_dataset():
    with pytest.raises(KeyError):
        build_dataloaders("not_a_dataset")


def test_meta_declares_correct_channels():
    """meta 里的 in_channels 会直接喂给模型构造函数，错了会 shape mismatch。"""
    for name, stats in DATASET_STATS.items():
        assert stats["in_channels"] in (1, 3), name
        assert stats["num_classes"] > 0, name
        assert len(stats["mean"]) == len(stats["std"]) == stats["in_channels"], name


# ============================================================
# transform
# ============================================================
def test_train_transform_for_mnist_has_no_augmentation():
    """MNIST 上手写数字形态固定，加了增强反而有害 —— 这条策略要锁住。"""
    train_tf = build_transforms("mnist", train=True)
    val_tf = build_transforms("mnist", train=False)
    assert len(train_tf.transforms) == len(val_tf.transforms) == 2


def test_cifar_train_transform_augments_but_val_does_not():
    assert len(build_transforms("cifar10", train=True).transforms) == 4
    assert len(build_transforms("cifar10", train=False).transforms) == 2
