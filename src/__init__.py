"""PyTorch 训练模板 —— 可复用的训练脚手架。"""

from src.config import TrainConfig
from src.data import build_dataloaders
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
]

__version__ = "1.0.0"
