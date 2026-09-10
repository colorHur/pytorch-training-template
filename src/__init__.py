"""PyTorch 训练模板 —— 可复用的训练脚手架。"""

from __future__ import annotations

import sys

from src.config import TrainConfig
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
]

__version__ = "1.0.0"


def force_utf8_stdout() -> None:
    """把标准输出/错误强制切到 UTF-8，避免中文日志在非 UTF-8 环境下崩掉。

    为什么需要（这是真踩过的坑）
    --------------------------
    Python 在 **Windows 上输出到管道**时用的是系统 locale 编码 ——
    英文系统是 cp1252，而 `print("⚠️ ...")` 会直接抛 `UnicodeEncodeError`
    并把整个训练脚本带崩。GitHub Actions 的 windows runner 恰好就是这个环境，
    于是"本地跑得好好的，CI 上 windows 挂掉"。

    输出到终端时通常没事（终端自己的编码能显示中文），所以这个坑只在
    **管道 / 重定向 / CI** 场景暴露 —— 最典型的"本地永远复现不了"。

    `errors="replace"`：万一遇到确实无法编码的字符，降级成替代符，
    也不要让日志把训练打断。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
