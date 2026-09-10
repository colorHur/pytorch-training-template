"""PyTorch 训练模板 —— 可复用的训练脚手架。"""

from __future__ import annotations

from src.compile_support import CompileOutcome, maybe_compile, probe_compile_support
from src.config import TrainConfig
from src.console import force_utf8_stdout
from src.data import build_dataloaders
from src.distributed import DistContext, reduce_sums, setup_distributed, wrap_model
from src.model import build_model, count_parameters
from src.train import build_lr_scheduler, evaluate, train_one_epoch

__all__ = [
    "TrainConfig",
    "build_dataloaders",
    "build_model",
    "count_parameters",
    "build_lr_scheduler",
    "evaluate",
    "train_one_epoch",
    "force_utf8_stdout",
    "DistContext",
    "reduce_sums",
    "setup_distributed",
    "wrap_model",
    "CompileOutcome",
    "maybe_compile",
    "probe_compile_support",
]

__version__ = "1.0.0"
