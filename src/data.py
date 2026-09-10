"""
数据加载：Dataset / DataLoader / transform 的标准写法。

三个要点（面试常问）
------------------
1. **为什么需要 Dataset 抽象？**
   把「怎么读一条数据」和「怎么攒 batch / 打乱 / 多进程加载」解耦。
   训练循环只依赖 DataLoader 的迭代协议，换数据集不用改训练代码。

2. **num_workers 怎么选？**
   Windows 下默认 0（主进程加载）最稳；Linux 下 4-8 通常最快。
   注意：num_workers>0 时每个 worker 会复制一份数据集对象。

3. **Normalize 的 mean/std 从哪来？**
   经验值：MNIST ≈ (0.1307, 0.3081)；CIFAR-10 ≈ (0.4914/0.4822/0.4465, 0.2470/0.2435/0.2616)。
   严格做法是在训练集上统计一遍。归一化让输入落在 0 附近、方差为 1，
   使各层梯度尺度一致，训练更稳。
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# ---- 各数据集的归一化统计量 ----
DATASET_STATS = {
    "mnist": {"mean": (0.1307,), "std": (0.3081,), "in_channels": 1, "num_classes": 10},
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "in_channels": 3,
        "num_classes": 10,
    },
}


class IndexedDataset(Dataset):
    """给任意 Dataset 套一层，让 __getitem__ 同时返回样本和下标。

    为什么要这个？
      演示「自定义 Dataset 写法的两种方式」：
        - 继承 torch.utils.data.Dataset，实现 __len__ / __getitem__
        - 或者像这样套 wrapper（组合优于继承）
    顺便还能用来复现「按 index 取样」的调试场景。
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        return x, y, index


def build_transforms(dataset_name: str, train: bool) -> Callable:
    """训练集加数据增强，验证/测试集只做 ToTensor + Normalize。

    ⚠️ 常见错误：验证集也加 RandomCrop / Flip 会导致验证指标抖动增大。
    """
    stats = DATASET_STATS[dataset_name]
    ops: list[Callable] = []
    if train and dataset_name == "cifar10":
        # CIFAR-10 上增强收益明显；MNIST 上增强反而可能有害（手写数字形态固定）
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])]
    return transforms.Compose(ops)


def build_datasets(dataset_name: str, data_dir: str = "data"):
    """返回 (train_set, val_set, test_set)。"""
    if dataset_name not in DATASET_STATS:
        raise KeyError(f"未知数据集 '{dataset_name}'，可选：{sorted(DATASET_STATS)}")

    builder = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[dataset_name]
    train_full = builder(
        root=data_dir, train=True, download=True, transform=build_transforms(dataset_name, True)
    )
    test_set = builder(
        root=data_dir, train=False, download=True, transform=build_transforms(dataset_name, False)
    )
    return train_full, test_set


def split_train_val(train_full, val_ratio: float, seed: int = 42):
    """从训练集里切出验证集。

    ⚠️ 为什么必须切验证集？
      用测试集调超参 = 数据泄漏，测试指标会虚高。
      验证集用来选超参/早停，测试集只在最后用一次。
    """
    n_val = int(len(train_full) * val_ratio)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(train_full, [n_train, n_val], generator=generator)


def build_dataloaders(
    dataset_name: str,
    data_dir: str = "data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
):
    """一步到位建好 train / val / test 三个 DataLoader。

    pin_memory=True 会把 batch 放进页锁定内存，H2D 拷贝更快（仅 CUDA 有意义）。
    """
    train_full, test_set = build_datasets(dataset_name, data_dir)
    train_set, val_set = split_train_val(train_full, val_ratio, seed)

    common = dict(num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, **common)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "meta": DATASET_STATS[dataset_name],
    }
