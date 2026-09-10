"""PyTorch 训练模板 —— 可复用的训练脚手架。"""

from __future__ import annotations

from src.checkpoint import (
    ResumeState,
    capture_rng,
    diff_configs,
    load_checkpoint,
    resolve_resume_path,
    restore_rng,
    save_checkpoint,
)
from src.compile_support import CompileOutcome, maybe_compile, probe_compile_support
from src.config import TrainConfig
from src.console import force_utf8_stdout
from src.data import build_dataloaders
from src.distributed import DistContext, reduce_sums, setup_distributed, wrap_model
from src.model import build_model, count_parameters
from src.train import (
    build_lr_scheduler,
    build_optimizer,
    evaluate,
    set_seed,
    train_one_epoch,
)

__all__ = [
    "TrainConfig",
    "build_dataloaders",
    "build_model",
    "count_parameters",
    "build_lr_scheduler",
    "build_optimizer",
    "set_seed",
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
    "ResumeState",
    "save_checkpoint",
    "load_checkpoint",
    "capture_rng",
    "restore_rng",
    "resolve_resume_path",
    "diff_configs",
]

__version__ = "1.0.0"
