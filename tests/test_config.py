"""配置系统的测试。

配置系统是"实验可复现"的地基 —— 它错了，所有实验结论都不可信。
重点测三件事：三级优先级的合并语义、校验、YAML 往返。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config import TrainConfig
from src.data import DATASET_STATS
from src.main import RUN_ONLY_ARGS, build_parser

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# ============================================================
# 默认值与派生属性
# ============================================================
def test_defaults_are_sane():
    cfg = TrainConfig()
    assert cfg.dataset in DATASET_STATS
    assert cfg.model in ("small_cnn", "mlp")
    # 新加的开关默认必须关闭，否则会悄悄改变所有人的基准
    assert cfg.gradient_checkpointing is False


def test_effective_batch_size():
    cfg = TrainConfig(batch_size=64, grad_accum_steps=4)
    assert cfg.effective_batch_size == 256


# ============================================================
# 校验（__post_init__）
# ============================================================
@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"grad_accum_steps": 0},
        {"val_ratio": 1.0},
        {"val_ratio": -0.1},
        {"amp": True, "device": "cpu"},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


# ============================================================
# merge 的优先级语义（命令行 > YAML > 默认值 靠它实现）
# ============================================================
def test_merge_overrides_and_leaves_original_untouched():
    cfg = TrainConfig()
    merged = cfg.merge(epochs=5, lr=1e-4)
    assert (merged.epochs, merged.lr) == (5, 1e-4)
    assert (cfg.epochs, cfg.lr) == (3, 1e-3)      # 原对象不被动


def test_merge_ignores_none():
    """argparse 没传的参数都是 None，必须被忽略，否则会覆盖掉 YAML 的值。"""
    cfg = TrainConfig(epochs=9)
    assert cfg.merge(epochs=None).epochs == 9


def test_merge_can_turn_a_flag_off():
    """容易踩的坑：用 `if v` 过滤会把 False / 0 一起吞掉，命令行就关不掉开关。
    这里锁死"只用 `is not None` 过滤"的语义。"""
    cfg = TrainConfig(amp=True, device="cuda")
    assert cfg.merge(amp=False).amp is False
    assert cfg.merge(weight_decay=0.0).weight_decay == 0.0


def test_merge_rejects_unknown_key():
    with pytest.raises(ValueError, match="未知配置项"):
        TrainConfig().merge(not_a_field=1)


# ============================================================
# 序列化
# ============================================================
def test_yaml_round_trip(tmp_path):
    cfg = TrainConfig(epochs=7, lr=3e-4, grad_accum_steps=2, gradient_checkpointing=True)
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert TrainConfig.from_yaml(path) == cfg


def test_from_yaml_missing_file():
    with pytest.raises(FileNotFoundError):
        TrainConfig.from_yaml("definitely/not/here.yaml")


def test_shipped_mnist_config_is_loadable():
    """仓库里带出去的配置必须能加载 —— 否则用户第一次运行就报错。"""
    cfg = TrainConfig.from_yaml(CONFIGS / "mnist.yaml")
    assert cfg.dataset in DATASET_STATS
    assert cfg.model in ("small_cnn", "mlp")


def test_summary_mentions_effective_batch():
    text = TrainConfig(batch_size=32, grad_accum_steps=2).summary()
    assert "等效 batch size" in text
    assert "64" in text


# ============================================================
# 命令行开关的完整性（「配置里有、命令行够不着」这件事要防住）
# ============================================================
# 这两个字段**故意**不暴露在命令行：它们会被 `--dataset` 对应的元信息自动覆盖
# （main.py / tools/lr_finder.py 里都有一行 `cfg.merge(num_classes=..., in_channels=...)`）。
# 放出来只会更糟 —— 用户写 `--num_classes 5` 会被静默盖掉，还以为生效了。
AUTO_DERIVED = {"num_classes", "in_channels"}


def test_every_config_field_is_reachable_from_the_cli():
    """配置里的每个字段都必须有一个对应的命令行开关。

    这条不变式守的是一个**真实发生过两次**的坑：`save_every_epoch` 和 `momentum`
    都在 `TrainConfig` 里有、命令行上却没有 —— 结果「能选 SGD 但不能调它的动量」。
    人眼对账一定会在某个版本漂掉，所以交给测试。
    """
    options = {opt for action in build_parser()._actions for opt in action.option_strings}
    missing = [
        field.name
        for field in dataclasses.fields(TrainConfig)
        if field.name not in AUTO_DERIVED and f"--{field.name}" not in options
    ]
    assert not missing, f"这些配置项没有命令行开关：{missing}"


def test_no_cli_flag_is_unknown_to_the_config():
    """反向：命令行上的每个开关都必须是合法配置项。

    否则 `cfg.merge(**overrides)` 会抛 `ValueError: 未知配置项` —— 用户看到的是
    配置系统的报错，而真正的原因是他敲的那个开关根本没人在管。
    """
    fields = {field.name for field in dataclasses.fields(TrainConfig)}
    unknown = [
        action.dest
        for action in build_parser()._actions
        if action.dest != "help" and action.dest not in fields | RUN_ONLY_ARGS
    ]
    assert not unknown, f"这些命令行开关不在 TrainConfig 里：{unknown}"


def test_auto_derived_fields_are_not_meant_to_be_set_by_hand():
    """`AUTO_DERIVED` 不是随手加的免责名单 —— 它必须真的被覆盖掉。

    如果哪天有人把 `cfg.merge(num_classes=...)` 删了，这两个字段就会退回
    YAML 里的值，而 `--dataset cifar10`（3 通道）配 `in_channels: 1`（YAML）
    会在建模型时炸掉，或者更糟：不炸但通道数错了。
    """
    for name in AUTO_DERIVED:
        assert name in {field.name for field in dataclasses.fields(TrainConfig)}
