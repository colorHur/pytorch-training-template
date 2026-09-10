"""pytest 共享配置：把项目根目录挂进 sys.path，让 `from src.xxx import` 可用。

不装成包、不改 PYTHONPATH 也能直接 `pytest` —— 少一个"跑不起来"的借口。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
