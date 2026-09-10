"""测试「测试基础设施」自己 —— 诊断通道不能成为新的故障源。

背景：GitHub Actions 的日志接口需要鉴权，所以 `conftest.py` 里做了个 hook，
把 pytest 失败提升成 check annotation（匿名可读）。这个 hook 本身翻过两次车，
而且两次的现象都是「什么都没发生」：

1. 只注册了 `when == "call"`，于是 setup / teardown 阶段的失败收不到；
2. 用 `print()` 输出含中文的 title，在 cp1252 的 windows runner 上抛
   `UnicodeEncodeError`，pytest INTERNALERROR —— 退出码从 1 变成 3，
   从外面看依然是「CI 红了但没有任何注解」。

这类「元测试」平时没人写，但诊断通道一旦失灵，排查成本高到离谱
（本次就隔着 CI 黑盒猜了三轮）。

验证分三层，各管一段：
- `_emit` 单元：编码无关性（cp1252 流下不能崩）
- hook 单元：什么时候发、发成什么格式
- 端到端子进程：真在 cp1252 下跑 pytest，退出码必须还是 1
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import conftest  # tests/ 在 sys.path 上（pytest 的 prepend 导入模式）
import pytest

TESTS_DIR = Path(__file__).resolve().parent
CP1252 = "cp1252"


def _fake_report(when: str = "call", nodeid: str = "tests/test_x.py::test_y", text: str = ""):
    """造一个最小的失败 report，只带 hook 会用到的字段。"""

    class FakeReport:
        failed = True

    report = FakeReport()
    report.when = when
    report.nodeid = nodeid
    report.longreprtext = text or "E   AssertionError: 中文断言消息\nE   assert 1 == 2"
    return report


class _PlainStream:
    """没有 `buffer` 属性的纯文本流，用来测兜底分支。"""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> None:
        self.chunks.append(text)

    def flush(self) -> None:
        pass

    def value(self) -> str:
        return "".join(self.chunks)


# ============================================================
# 第一层：_emit 必须与流的编码无关
# ============================================================
def test_emit_survives_cp1252_stream():
    """cp1252 的流下写中文不能抛 —— 这就是把 hook 搞崩的那一条。

    修复前这里是 `print(...)`：`UnicodeEncodeError` 被 pluggy 包成
    INTERNALERROR，诊断通道彻底失效。
    """
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=CP1252)

    conftest._emit("\n::error file=x.py,line=1,title=中文标题::中文消息\n", stream=stream)

    assert buffer.getvalue().decode("utf-8").startswith("\n::error file=x.py")


def test_emit_falls_back_when_stream_has_no_buffer():
    """拿不到 `buffer` 时退化成 ASCII 安全输出，而不是抛异常。"""
    stream = _PlainStream()

    conftest._emit("中文 ::error\n", stream=stream)

    out = stream.value()
    assert "::error" in out
    assert "\\u4e2d" in out        # 中文被转义，但信息没丢
    assert "中文" not in out


# ============================================================
# 第二层：hook 什么时候发、发成什么格式
# ============================================================
def test_hook_emits_annotation_anchored_at_line_start(capsys, monkeypatch):
    """注解必须顶在行首（前导换行不能省），否则 GitHub 不解析。"""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    conftest.pytest_runtest_logreport(_fake_report())

    out = capsys.readouterr().out
    assert out.startswith("\n::error file=tests/test_x.py,line=1,title=")
    assert "中文断言消息" in out


@pytest.mark.parametrize("when", ["setup", "call", "teardown"])
def test_hook_covers_every_phase(capsys, monkeypatch, when):
    """setup / teardown 阶段失败同样要报 —— fixture 抛错是最早漏掉的那类。"""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    conftest.pytest_runtest_logreport(_fake_report(when=when))

    assert f"[{when}]" in capsys.readouterr().out


def test_hook_is_silent_without_github_env(capsys, monkeypatch):
    """本地跑测试不该多出噪声行。"""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    conftest.pytest_runtest_logreport(_fake_report())

    assert capsys.readouterr().out == ""


def test_hook_is_silent_for_passing_report(capsys, monkeypatch):
    """通过的测试不该产生注解。"""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    report = _fake_report()
    report.failed = False

    conftest.pytest_runtest_logreport(report)

    assert capsys.readouterr().out == ""


# ============================================================
# 格式：GitHub 的解析规则很容易踩空
# ============================================================
def test_escapes_reserved_characters():
    """`%` `,` `:` 在 workflow command 里有语法含义，不转义注解会被丢。"""
    assert conftest._escape("准确率 92% 达成") == "准确率 92%25 达成"
    # nodeid 里的 `::` 不转义会把属性段提前截断
    assert conftest._escape_property("tests/t.py::test_x") == "tests/t.py%3A%3Atest_x"
    assert conftest._escape_property("a,b") == "a%2Cb"


def test_escape_order_does_not_double_encode():
    """必须先 `_escape` 再替换 `,`/`:`，否则 `%2C` 会被二次转义成 `%252C`。"""
    assert conftest._escape_property("50%,x") == "50%25%2Cx"


def test_payload_is_truncated_to_stay_under_the_limit():
    """超长 payload 会被 GitHub 静默丢弃，宁可截断也要发得出去。"""
    report = _fake_report(text="E   " + "x" * 50000)

    payload = conftest._annotation_payload(report)

    assert len(payload) == conftest.MAX_MESSAGE_CHARS


# ============================================================
# 第三层：端到端 —— 真在 cp1252 下跑一次 pytest
# ============================================================
def test_exit_code_stays_one_on_cp1252(tmp_path):
    """最贴近 CI 的那条：cp1252 + GITHUB_ACTIONS 下跑一个必然失败的测试。

    断言两件事：
      1. 退出码必须是 **1**（测试失败），不能是 **3**（pytest INTERNALERROR）
         —— hook 自己崩掉就会把 1 变成 3，而外部只看到"红"，看不出区别
      2. stdout 里必须真的出现 `::error file=`，且带上失败断言的内容

    在 tmp_path 里造一个转发用的 conftest：用 importlib 按文件路径加载真正的
    conftest，避免 `from conftest import *` 撞上 pytest 自己那个同名模块
    （那会变成循环导入，拿不到 hook）。
    """
    real_conftest = TESTS_DIR / "conftest.py"
    (tmp_path / "conftest.py").write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('_real_conftest', r'{real_conftest}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "pytest_runtest_logreport = mod.pytest_runtest_logreport\n",
        encoding="utf-8",
    )
    (tmp_path / "test_boom.py").write_text(
        'def test_boom():\n    assert 1 == 2, "中文断言消息"\n',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONIOENCODING": CP1252, "GITHUB_ACTIONS": "true"},
        timeout=180,
    )

    assert proc.returncode == 1, (
        f"期望退出码 1（测试失败），实际 {proc.returncode}"
        f"（3 = pytest 内部错误，说明注解 hook 自己崩了）\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert "::error file=" in proc.stdout, f"没有产生注解\n{proc.stdout[-2000:]}"
    assert "中文断言消息" in proc.stdout
