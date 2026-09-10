"""
模型定义：一个干净的小 CNN + 一个 MLP 对照。

为什么选 small_cnn 而不是 resnet？
  - 模板的目的是「展示训练循环的骨架」，模型越简单越能看清骨架
  - 简单模型跑得快，方便反复调试训练逻辑
  - 真实项目换模型只需要改这一个文件

`small_cnn` 结构
----------------
    Conv(1→32, 3x3) → BN → ReLU → MaxPool
    Conv(32→64, 3x3) → BN → ReLU → MaxPool
    Flatten → Dropout → Linear(64*7*7 → 128) → ReLU → Linear(128 → num_classes)

参数量约 420K，MNIST 上 3 个 epoch 就能到 98%+。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """适合 MNIST / CIFAR-10 的小型卷积网络。"""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        # 输入假设为 28x28（MNIST）；CIFAR-10 是 32x32，池化两次后是 8x8
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MLP(nn.Module):
    """纯全连接对照模型：用来演示「没有卷积也能训」，以及参数量的差异。"""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 28 * 28, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---- 模型注册表：新增模型在这里登记，配置里写字符串就能用 ----
MODEL_REGISTRY = {
    "small_cnn": SmallCNN,
    "mlp": MLP,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """按名字建模型。新增模型只需往 MODEL_REGISTRY 加一行。"""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 '{name}'，可选：{sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
