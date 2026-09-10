"""
模型定义：一个干净的小 CNN + 一个 MLP 对照。

为什么选 small_cnn 而不是 resnet？
  - 模板的目的是「展示训练循环的骨架」，模型越简单越能看清骨架
  - 简单模型跑得快，方便反复调试训练逻辑
  - 真实项目换模型只需要改这一个文件

`small_cnn` 结构
----------------
    block1: Conv(1→32, 3x3) → BN → ReLU → MaxPool     28 -> 14
    block2: Conv(32→64, 3x3) → BN → ReLU → MaxPool    14 -> 7
    Flatten → Dropout → Linear(64*7*7 → 128) → ReLU → Linear(128 → num_classes)

参数量约 420K，MNIST 上 3 个 epoch 就能到 98%+。

关于「块」的划分（重要）
------------------------
卷积层被刻意打包成 block1 / block2 两个 `nn.Sequential`，而不是平铺成 8 层。
原因：**梯度检查点的粒度就是「块」**。检查点会丢块内所有中间激活、只留块的输入，
反向时把块重算一遍。如果按单层做检查点，每层都得存自己的输入 —— 等于什么都没省。
真实项目里这个「块」对应 Transformer 的一层，道理完全一样。

粒度不能乱选，这有实测（`experiments/exp_checkpoint_granularity.py`，batch=256）：
    不开检查点          → 全程峰值 137.7 MB
    按 block（本仓库做法）→ 全程峰值 119.3 MB   ← 有效
    整条主干当成一个段    → 全程峰值 140.6 MB   ← 反而比不开还高

把整条主干当一个段时，反向重算会把所有激活**一次性全量物化**，
中间没有任何边界可以逐段释放 —— 于是「分批驻留」退化成「全量驻留」，还要白付重算的账。
**粒度必须落在「块」上，且块内激活要显著大于块边界的张量。**
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """一个卷积单元：Conv → BN → ReLU → MaxPool。也是梯度检查点的最小粒度。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """适合 MNIST / CIFAR-10 的小型卷积网络（支持梯度检查点）。

    ⚠️ `image_size` 必须和数据集匹配：卷积主干池化两次 → 特征图是 `image_size // 4`，
    全连接层的输入维度由它算出来。写死 7*7 的话 CIFAR-10（32×32 → 8×8）会 shape mismatch。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.25,
        gradient_checkpointing: bool = False,
        image_size: int = 28,
    ):
        super().__init__()
        # 输入假设为 28x28（MNIST）；CIFAR-10 是 32x32，池化两次后是 8x8
        self.blocks = nn.Sequential(
            _conv_block(in_channels, 32),          # 28 -> 14
            _conv_block(32, 64),                   # 14 -> 7
        )
        feat = image_size // 4                     # 两次 MaxPool2d(2) 后的空间尺寸
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * feat * feat, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self.gradient_checkpointing = gradient_checkpointing

    def set_gradient_checkpointing(self, enabled: bool = True) -> "SmallCNN":
        """开关梯度检查点。返回 self，方便链式调用。"""
        self.gradient_checkpointing = enabled
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        梯度检查点（gradient checkpointing）
        -----------------------------------
        开启后每个 block 都走 `checkpoint()`：
          - 前向：不保存块内中间激活，只保存块的**输入**
          - 反向：把块再前向一遍，用重算出的激活求梯度

        省的是**激活值**，代价是**基本等于多跑一次前向**。本仓库实测
        （batch=256，见 `experiments/exp_checkpoint_granularity.py`）：

            前向保活的激活   ≈111 MB → ≈19 MB    （约 -83%）
            前向峰值         ≈117 MB → ≈51 MB    （约 -56%）
            全程峰值         ≈138 MB → ≈119 MB   （约 -13%）  ← 收益大幅缩水
            完整 step 耗时   ≈5.5 ms → ≈8.3 ms   （约 +40~50%）

        ⚠️ 注意「激活保活量」和「全程峰值」不是一回事：峰值由反向阶段的**瞬时**规模决定，
        重算会把激活重新物化出来。所以小模型上检查点的实际收益远小于直觉预期。

        ⚠️ 已被实测验证的两个坑（不要凭直觉想当然）
        --------------------------------------------
        【坑 1】`use_reentrant=False` 挡不住 BN 统计量被更新两次。

            实测（`experiments/exp_memory_accounting.py` 的 A vs D 变体，同 seed）：
              · loss 与全部梯度：逐元素**完全一致**（maxdiff = 0.0）→ 重算是精确的
              · 但 `num_batches_tracked` 变成 1 vs 2，`running_mean` 偏离 ~0.024

            原因：重算时块被重新前向一遍，而训练态 BN 的 forward **带副作用**
            （用当前 batch 统计量更新 running stats）。梯度不受影响是因为训练态 BN
            归一化用的是 batch 统计量，与 running stats 无关；但 running stats 本身错了，
            **eval 阶段会用到它**。

            这也解释了为什么 LLM 训练开检查点从无顾虑：Transformer 用 LayerNorm，
            没有 running stats，重算是纯粹的纯函数。**有状态层才是问题所在。**

            缓解手段（按推荐度排序）：
              1. 块内改用无状态归一化（LayerNorm / GroupNorm）→ 从根上消除
              2. 把 BN 排除在检查点块之外（只 checkpoint 无状态的子段）
              3. 接受偏差并实测它对最终指标的影响（本例影响见实验报告）

        【坑 2】只在 `self.training` 时启用。

            eval / 推理阶段没有反向，开检查点纯粹是无意义地多算一遍。

        `use_reentrant=False` 仍然该用：旧版可重入实现连 Dropout 的 RNG 状态都会打乱，
        非可重入实现会正确保存/恢复 RNG。它只是解决不了 BN 的副作用，那是另一回事。
        """
        if self.gradient_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.blocks(x)
        return self.classifier(x)


class MLP(nn.Module):
    """纯全连接对照模型：用来演示「没有卷积也能训」，以及参数量的差异。

    同样按 `image_size` 推算输入维度，理由见 SmallCNN 的说明。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden: int = 256,
        image_size: int = 28,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * image_size * image_size, hidden),
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


def enable_gradient_checkpointing(model: nn.Module) -> bool:
    """给支持该特性的模型开启梯度检查点，返回是否开启成功。

    为什么要返回 bool？
      并不是所有模型都值得/能够做检查点。MLP 只有一个隐层，
      包起来等于没省（每层都要存输入），所以它不实现 `set_gradient_checkpointing`。
      调用方拿到 False 时应该明确告诉用户「这个配置被忽略了」，而不是静默失效。
      面试里这也是个考点：**检查点的收益正比于「块内激活占比」**，块太浅就没意义。
    """
    setter = getattr(model, "set_gradient_checkpointing", None)
    if setter is None:
        return False
    setter(True)
    return True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
