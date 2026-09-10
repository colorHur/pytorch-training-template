"""
配置系统：用 dataclass + YAML 管理训练超参。

为什么要做这个？
  - 手写训练脚本最容易犯的错是「超参散落各处」，改一个数要翻三个文件
  - 配置集中化 + 可覆盖（命令行 > YAML > 默认值）是工程化的第一步
  - 面试时能讲：「我的训练配置是可复现的，config 存进 checkpoint，改天也能复现今天的实验」

用法
----
    from src.config import TrainConfig
    cfg = TrainConfig.from_yaml("configs/mnist.yaml")
    cfg = cfg.merge(epochs=5, lr=1e-3)   # 局部覆盖
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """一个训练任务的全部超参。"""

    # ---- 实验标识 ----
    exp_name: str = "mnist"
    seed: int = 42
    output_dir: str = "outputs"

    # ---- 数据 ----
    dataset: str = "mnist"           # mnist / cifar10 / synthetic
    data_dir: str = "data"
    num_workers: int = 0             # Windows 下建议 0，避免多进程开销
    val_ratio: float = 0.1           # 从训练集切出的验证集比例

    # ---- 模型 ----
    model: str = "small_cnn"         # small_cnn / mlp
    num_classes: int = 10
    in_channels: int = 1

    # ---- 优化 ----
    epochs: int = 3
    batch_size: int = 128
    grad_accum_steps: int = 1        # 梯度累积，等效 batch = batch_size * grad_accum_steps
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"         # adamw / sgd
    momentum: float = 0.9            # 仅 sgd 用
    lr_scheduler: str = "cosine"     # none / cosine / step
    warmup_steps: int = 0
    max_grad_norm: float = 0.0       # 0 = 不裁剪

    # ---- 精度与显存 ----
    amp: bool = False                # 自动混合精度（fp16 + GradScaler）
    channels_last: bool = False      # NHWC 内存布局，卷积上更快
    gradient_checkpointing: bool = False  # 用计算换显存：反向时重算激活值
    compile: bool = False            # torch.compile：把训练 step 编译成优化后的图
    compile_backend: str = "inductor"  # 编译后端；Windows 上 inductor 常不可用，见 compile_support.py

    # ---- 日志与保存 ----
    log_interval: int = 50           # 每多少 step 打一次日志
    save_every_epoch: bool = True
    early_stop_patience: int = 0     # 0 = 关闭早停

    # ---- 运行设备 ----
    device: str = "auto"             # auto / cuda / cpu

    # ---- 分布式（仅 torchrun 多进程时生效）----
    sync_bn: bool = False            # DDP 下把 BN 换成 SyncBatchNorm（跨卡统计）
    ddp_timeout_minutes: int = 30    # 集合通信超时；调小能让 deadlock 快速暴露

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps 必须 > 0")
        if not 0 <= self.val_ratio < 1:
            raise ValueError("val_ratio 必须在 [0, 1) 区间")
        if self.amp and self.device == "cpu":
            raise ValueError("amp 需要 CUDA，CPU 上请关闭")

    # ---------- 派生属性（不落盘） ----------

    @property
    def effective_batch_size(self) -> int:
        """等效 batch size = micro batch × 梯度累积步数。"""
        return self.batch_size * self.grad_accum_steps

    def resolve_device(self) -> str:
        """把 'auto' 解析成实际设备。

        ⚠️ 不能只看 `torch.cuda.is_available()`：
        在容器 / CI 里（例如 `CUDA_VISIBLE_DEVICES=""`）驱动可见但**没有暴露任何设备**，
        此时 `is_available()` 仍返回 True 而 `device_count()` 是 0，
        后面一旦调用 `get_device_name(0)` 就会炸 "Invalid device id"。
        所以两个条件都要查。
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        try:
            if not torch.cuda.is_available():
                return "cpu"
            return "cuda" if torch.cuda.device_count() > 0 else "cpu"
        except Exception:
            # 驱动层面的任何异常都退回 CPU，不要因为探测失败就崩掉整个训练
            return "cpu"

    # ---------- 序列化 ----------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merge(self, **overrides: Any) -> "TrainConfig":
        """返回覆盖后的新配置（原配置不动），None 值会被忽略。

        命令行覆盖 YAML 的机制就靠它：argparse 没传的参数都是 None。
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"未知配置项：{sorted(unknown)}")
        return dataclasses.replace(self, **clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到配置文件：{path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    def summary(self) -> str:
        """一行行打印配置，训练开始时调用。"""
        lines = ["配置摘要 " + "-" * 40]
        for k, v in self.to_dict().items():
            lines.append(f"  {k:22s} = {v}")
        lines.append(f"  {'等效 batch size':22s} = {self.effective_batch_size}")
        lines.append("-" * 48)
        return "\n".join(lines)
