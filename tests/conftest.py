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
# GitHub 对单条 workflow command 有长度限制，超了会被**静默丢弃**
# （页面上一点痕迹都没有），所以宁可截断也要保证发得出去。
MAX_MESSAGE_CHARS = 4000


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
    return " | ".join(ln.strip() for ln in picked if ln.strip())[:MAX_MESSAGE_CHARS]


def _escape(text: str) -> str:
    """转义 workflow command **消息**里的保留字符。

    规范要求 `%` `\\r` `\\n` 分别写成 `%25` `%0D` `%0A`。不转义的话，一个
    裸的 `%` 会让 GitHub 把它后面的字符当成转义序列解析 —— 而 payload 里
    `%` 是常客（百分比、`%s`、进度），轻则显示错乱，重则整条命令被丢弃。
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    """转义 property 值（`file=` / `title=`）里的保留字符。

    `key=value` 语法里 `,` 和 `:` 有结构含义，必须写成 `%2C` / `%3A`。
    最容易踩的是 `title`：nodeid 长这样 `tests/test_x.py::test_y`，里面的
    `::` 会把属性段提前截断。注意必须**先 `_escape` 再替换**，否则新插入的
    `%2C` 会被 `_escape` 二次转义成 `%252C`。
    """
    return _escape(text).replace(",", "%2C").replace(":", "%3A")


def _emit(text: str, stream=None) -> None:
    """把一行写给终端，**绕开**流的编码设置。

    ⚠️ 这是本文件最容易忽略的坑，而且我确实踩了：
    最早用的是 `print(...)`，于是这条"诊断通道"自己在 CI 的 windows runner
    上（cp1252 管道）抛 `UnicodeEncodeError`，pytest 直接 INTERNALERROR、
    退出码从 1 变成 3 —— 页面上的现象依然是"什么都没有"。
    **诊断工具成了新的故障源，比不做诊断更误导。**

    修法是把 UTF-8 字节直接写进流的 buffer：GitHub 始终按 UTF-8 解析进程
    输出，这条路与 Python 的 locale 完全无关。

    `stream` 默认取 `sys.stdout`；留出参数是为了让测试能塞一个 cp1252 的流
    进来（在 fixture 里替换 `sys.stdout` 靠不住 —— pytest 在 setup→call
    过渡时会把它换掉，我们塞进去的流会被 GC 关闭）。
    """
    if stream is None:
        stream = sys.stdout
    data = text.encode("utf-8", "replace")
    buf = getattr(stream, "buffer", None)
    if buf is not None:
        buf.write(data)
        buf.flush()
        return
    # 兜底：拿不到 buffer（流被换成纯文本对象）时退化成 ASCII 安全输出。
    # 优先级是"别把测试进程带崩"，信息量其次。
    # 注意这里对 `text` 重新编码（而不是复用 data）—— 对 bytes 做
    # backslashreplace 会得到 `\xe4\xb8\xad` 这种按字节的转义，难读；
    # 对 str 做则得到按码点的 `\u4e2d`，一眼能看出原文。
    stream.write(text.encode("ascii", "backslashreplace").decode("ascii"))
    stream.flush()


def pytest_runtest_logreport(report) -> None:
    """失败时打印 GitHub Actions 的 `::error::` 工作流命令。

    为什么需要：Actions 的**日志接口要鉴权**（实测 `/actions/jobs/{id}/logs`
    返回 403），而 check-run 的 annotations 接口对公开仓库匿名可读。把关键
    错误行提升成 annotation，等于留了一个不用登录就能拿到失败原因的通道 ——
    否则从外部只能看到一句毫无信息量的 "Process completed with exit code 1"。

    ---
    ⚠️ 四个实测踩出来的细节，少一个就等于没发：

    1. **必须覆盖 setup / call / teardown 三个阶段**。最初只处理了
       `when == "call"`，于是 fixture 里抛错（比如 module 级 fixture 启动
       子进程失败）完全收不到 —— 而退出码依然是 1，现象就是"CI 红了，
       但 annotation 里什么都没有"。
    2. **必须补一个前导换行**：pytest 的进度字符（`-q` 下的 `F`）会顶在
       同一行前面，而 GitHub 只解析**行首**的工作流命令。
    3. **必须带 `file=`**：不带文件归属的 `::error::` 不会出现在
       annotations 接口里。
    4. **不能用 `print`**：见 `_emit` —— cp1252 下它把 hook 自己搞崩，
       退出码从 1 变成 3，比不做诊断更难查。

    只在 CI 上生效（`GITHUB_ACTIONS=true`），本地跑测试不会多出噪声行。
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    if not report.failed:
        return

    payload = _annotation_payload(report)
    if not payload:
        return

    path, _, _ = report.nodeid.partition("::")
    nodeid = report.nodeid.replace("\n", " ")
    title = _escape_property(f"pytest 失败[{report.when}] {nodeid}")
    _emit(f"\n::error file={_escape_property(path)},line=1,title={title}::{_escape(payload)}\n")
