"""pytest 共享配置。

1. 把项目根目录挂进 sys.path，让 `from src.xxx import` 可用 ——
   不装成包、不改 PYTHONPATH 也能直接 `pytest`，少一个"跑不起来"的借口。
2. 在 GitHub Actions 上把失败断言提升为 **check annotation**（见下）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# 让 CI 的失败信息直接出现在页面上，不用翻日志
# ============================================================
def _annotation_payload(report) -> str:
    """从失败报告里挑出最有信息量的几行。

    优先取 pytest 的断言行（以 `E ` 开头），再补上长输出的尾部 ——
    子进程测试（DDP、CLI 冒烟）会把被调进程的 stdout/stderr 打进断言消息里，
    这些内容通常才是真正的根因。
    """
    text = getattr(report, "longreprtext", "") or ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    assert_lines = [ln for ln in lines if ln.lstrip().startswith("E ")]
    picked = assert_lines[-15:] if assert_lines else []
    picked += [ln for ln in lines[-15:] if ln not in picked]
    return " | ".join(ln.strip() for ln in picked if ln.strip())[:60000]


def pytest_runtest_logreport(report) -> None:
    """失败时打印 GitHub Actions 的 `::error::` 工作流命令。

    为什么需要：Actions 的**日志接口要鉴权**，而 check-run 的
    annotations 接口对公开仓库是匿名可读的。把关键错误行提升成 annotation，
    等于给自己留了一个不用登录就能拿到失败原因的通道 ——
    不然就只能拿到一句毫无信息量的 "Process completed with exit code 1"。

    只在 CI 上生效，本地跑测试不会多出这些噪声行。
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    if report.when != "call" or not report.failed:
        return

    payload = _annotation_payload(report)
    if not payload:
        return
    nodeid = report.nodeid.replace("\n", " ")
    # 开头的换行是必需的：pytest 的进度字符（-q 下的 F）会顶在这一行前面，
    # 而 GitHub 只解析**行首**的工作流命令 —— 不换行就等于没发。
    # 换行会让 GitHub 只取第一行，所以 payload 里也要压成单行。
    print(f"\n::error title=pytest 失败 {nodeid}::{payload}", flush=True)
