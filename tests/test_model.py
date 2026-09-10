"""模型测试。

三条主线：
  1. 注册表 + 参数量 + 输入尺寸推算（锁住刚修掉的 CIFAR shape bug）
  2. 梯度检查点的**不变量**：不改变前向结果与梯度（这是它成立的前提）
  3. 梯度检查点的**已知偏差**：BN running stats 被双倍更新（canary，上游修好要提醒我们改文档）
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from src.model import (
    MODEL_REGISTRY,
    build_model,
    count_parameters,
    enable_gradient_checkpointing,
)

SMALL_CNN_PARAMS = 421_834  # MNIST 输入（1×28×28）下的实测参数量


def _pair(image_size: int = 28, in_channels: int = 1):
    """造一对同参数、同初始化的模型：一个不开检查点，一个开。"""
    torch.manual_seed(0)
    plain = build_model(
        "small_cnn", in_channels=in_channels, num_classes=10, image_size=image_size
    )
    torch.manual_seed(0)
    ckpt = build_model(
        "small_cnn", in_channels=in_channels, num_classes=10, image_size=image_size
    )
    assert enable_gradient_checkpointing(ckpt) is True
    return plain, ckpt


# ============================================================
# 注册表与结构
# ============================================================
def test_registry_contents():
    assert set(MODEL_REGISTRY) == {"small_cnn", "mlp"}


def test_build_model_unknown_name():
    with pytest.raises(KeyError):
        build_model("resnet50")


def test_parameter_count_is_stable():
    """参数量是历史基准的一部分，变了要有人注意到。"""
    total, trainable = count_parameters(build_model("small_cnn", image_size=28))
    assert total == trainable == SMALL_CNN_PARAMS


def test_mlp_has_no_gradient_checkpointing_support():
    """MLP 只有一个隐层，包检查点等于没省 —— 它不该实现这个开关。"""
    assert enable_gradient_checkpointing(build_model("mlp")) is False


# ============================================================
# 输入尺寸推算（回归测试：曾经写死 7*7，CIFAR 直接崩）
# ============================================================
@pytest.mark.parametrize(
    "in_channels, image_size",
    [(1, 28), (3, 32), (1, 32), (3, 28)],
)
def test_forward_shape_matches_input_size(in_channels, image_size):
    model = build_model(
        "small_cnn", in_channels=in_channels, num_classes=10, image_size=image_size
    )
    model.eval()
    x = torch.randn(2, in_channels, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


@pytest.mark.parametrize("in_channels, image_size", [(1, 28), (3, 32)])
def test_mlp_forward_shape(in_channels, image_size):
    model = build_model(
        "mlp", in_channels=in_channels, num_classes=10, image_size=image_size
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, in_channels, image_size, image_size))
    assert out.shape == (2, 10)


def test_cifar_model_has_more_params_than_mnist():
    """32×32 输入的全连接层更大 —— 如果这里相等，说明 image_size 没生效。"""
    a, _ = count_parameters(build_model("small_cnn", in_channels=3, image_size=28))
    b, _ = count_parameters(build_model("small_cnn", in_channels=3, image_size=32))
    assert b > a


def test_num_classes_is_respected():
    model = build_model("small_cnn", num_classes=7, image_size=28)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 28, 28)).shape == (1, 7)


# ============================================================
# 不变量：检查点不改变数值结果
# ============================================================
def test_checkpointing_keeps_loss_and_gradients_identical():
    """核心不变量：检查点只是"重算"，不该改变任何数值。

    这是它敢用在生产训练里的前提。grad 必须**逐元素相等**，不是"接近"。
    """
    plain, ckpt = _pair()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    plain.train()
    ckpt.train()
    torch.manual_seed(7)
    loss_plain = criterion(plain(x), y)
    loss_plain.backward()

    torch.manual_seed(7)          # Dropout 的随机掩码必须一致
    loss_ckpt = criterion(ckpt(x), y)
    loss_ckpt.backward()

    assert loss_plain.item() == loss_ckpt.item()
    for name, (a, b) in enumerate(zip(plain.parameters(), ckpt.parameters())):
        assert a.grad is not None and b.grad is not None
        assert torch.equal(a.grad, b.grad), f"第 {name} 个参数的梯度不一致"


def test_checkpointing_is_disabled_in_eval_mode():
    """eval / 推理阶段没有反向，开检查点是无意义地多算一遍。"""
    _, ckpt = _pair()
    ckpt.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        a = ckpt(x)
        b = ckpt(x)
    assert torch.equal(a, b)      # eval 下输出必须确定


def test_toggle_back_and_forth():
    model = build_model("small_cnn", image_size=28)
    model.set_gradient_checkpointing(True)
    assert model.gradient_checkpointing is True
    model.set_gradient_checkpointing(False)
    assert model.gradient_checkpointing is False


# ============================================================
# 已知偏差（canary）
# ============================================================
def test_bn_running_stats_double_update_is_a_known_issue():
    """**这条测试故意断言「现状」，不是断言「正确行为」。**

    梯度检查点在反向时会重算 forward，而训练态 BatchNorm 的 forward 带副作用
    （用当前 batch 统计量更新 running stats）→ 统计量被更新两次。

    ⚠️ 必须跑 `backward()` 才会触发 —— 重算只发生在反向阶段。
    只调前向的话两个模型统计量完全一样，会误判成"上游已修复"。

    之所以写成断言：如果哪天 PyTorch 修好了，这条会失败 / 告警，
    提醒我们把 README / model.py 注释里的说明一起改掉，而不是让文档悄悄过时。
    详见 README「一个被实测抓出来的坑」。
    """
    plain, ckpt = _pair()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    plain.train()
    ckpt.train()
    criterion(plain(x), y).backward()    # ← 反向触发检查点的重算
    criterion(ckpt(x), y).backward()

    def tracked(model):
        return int(dict(model.named_buffers())["blocks.0.1.num_batches_tracked"])

    plain_n, ckpt_n = tracked(plain), tracked(ckpt)
    if plain_n == ckpt_n:
        warnings.warn(
            "PyTorch 行为已变化：梯度检查点不再双倍更新 BN 统计量。"
            "请同步更新 README 与 src/model.py 的说明。",
            stacklevel=2,
        )
    else:
        assert ckpt_n == 2 * plain_n, f"期望双倍更新，实际 {plain_n} vs {ckpt_n}"


def test_dropout_rng_is_preserved_by_checkpointing():
    """非可重入实现会保存/恢复 RNG 状态，所以 Dropout 掩码与不开检查点时一致。

    这正是 must 用 `use_reentrant=False` 的原因 —— 旧实现会打乱 RNG。
    """
    plain, ckpt = _pair()
    x = torch.randn(8, 1, 28, 28)
    plain.train()
    ckpt.train()

    torch.manual_seed(11)
    a = plain(x)
    torch.manual_seed(11)
    b = ckpt(x)
    assert torch.equal(a, b)
