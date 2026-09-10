"""`torch.compile` 的可用性探测、降级路径与包装解构。

这个文件里最值钱的两条不是"编译能跑通"，而是：

- **探测逻辑可离线测**：`platform_name` 参数让 CI（Linux）也能覆盖 Windows 上
  "缺 Triton / 缺 MSVC"这两条分支 —— 否则这段代码只能靠"在 Windows 上手跑一遍"
  来保证，而它恰恰是 Windows 专用的。
- **冒烟前向不能污染 BN 的 running stats**：首次前向是拿来触发编译的，
  如果它顺手把统计量更新了，训练还没开始就已经偏了。
"""

from __future__ import annotations

import sys
import warnings

import pytest
import torch
import torch.nn as nn

from src.compile_support import (
    DEFAULT_BACKEND,
    FALLBACK_BACKENDS,
    CompileSupport,
    maybe_compile,
    probe_compile_support,
)
from src.distributed import unwrap_model
from src.model import build_model


# ============================================================
# 探测逻辑：Windows 的两条失败路径必须能在任何平台上被测到
# ============================================================
def test_probe_reports_triton_missing_on_windows_cuda(monkeypatch):
    """Windows + CUDA：inductor 缺 Triton，要给出原因 + 退路。"""
    monkeypatch.setattr("src.compile_support._has_module", lambda name: False)

    support = probe_compile_support("inductor", "cuda", platform_name="win32")

    assert support.ok is False
    assert "Triton" in support.reason
    assert support.fallback == "cudagraphs"
    assert support.hint                      # 必须告诉人怎么修，不能只说不行


def test_probe_reports_msvc_missing_on_windows_cpu(monkeypatch):
    """Windows + CPU：inductor 要 MSVC 的 cl.exe。"""
    monkeypatch.setattr("src.compile_support._has_msvc", lambda: False)

    support = probe_compile_support("inductor", "cpu", platform_name="win32")

    assert support.ok is False
    assert "cl.exe" in support.reason
    assert support.fallback == "aot_eager"


def test_probe_is_ok_on_linux():
    """Linux 上不设任何额外限制 —— 那是 inductor 的正常工作环境。"""
    assert probe_compile_support("inductor", "cpu", platform_name="linux").ok is True
    assert probe_compile_support("inductor", "cuda", platform_name="linux").ok is True


@pytest.mark.parametrize("backend", FALLBACK_BACKENDS)
def test_probe_passes_for_backends_that_need_no_kernel_generation(backend):
    """退路后端不做 kernel 生成，因此不受 Triton / MSVC 缺失影响。"""
    assert probe_compile_support(backend, "cuda", platform_name="win32").ok is True
    assert probe_compile_support(backend, "cpu", platform_name="win32").ok is True


def test_recommended_fallback_is_actually_available(monkeypatch):
    """给出的退路必须自己能用 —— 否则建议等于没给。"""
    monkeypatch.setattr("src.compile_support._has_module", lambda name: False)
    monkeypatch.setattr("src.compile_support._has_msvc", lambda: False)

    for device in ("cpu", "cuda"):
        support = probe_compile_support("inductor", device, platform_name="win32")
        assert support.ok is False
        assert probe_compile_support(support.fallback, device, platform_name="win32").ok is True


def test_every_backend_has_a_fallback_for_every_device(monkeypatch):
    """把"缺什么"全拉满，确认每个组合都能给出退路 —— 不留"无路可走"的坑。"""
    monkeypatch.setattr("src.compile_support._has_module", lambda name: False)
    monkeypatch.setattr("src.compile_support._has_msvc", lambda: False)

    for device in ("cpu", "cuda"):
        support = probe_compile_support(DEFAULT_BACKEND, device, platform_name="win32")
        assert support.ok is False
        assert support.fallback, f"{device} 上探测失败却没有给退路"
        assert support.fallback in FALLBACK_BACKENDS


# ============================================================
# maybe_compile：开不了也要能继续训练
# ============================================================
def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))


def test_maybe_compile_disabled_returns_the_same_object():
    model = _toy_model()

    outcome = maybe_compile(model, enabled=False)

    assert outcome.active is False
    assert outcome.model is model          # 原样返回，不许偷偷包装
    assert outcome.message == ""


def test_maybe_compile_returns_original_model_when_probe_fails(monkeypatch):
    """探测不通过 → 返回**未包装的原模型** + 人话说明，而不是抛异常。

    这里直接替换探测结果（而不是伪造平台），因为要测的是 `maybe_compile`
    对"探测失败"这个输入的**反应**，与具体缺什么无关 —— 平台分支已经由上面
    那组探测测试覆盖了。
    """
    monkeypatch.setattr(
        "src.compile_support.probe_compile_support",
        lambda backend, device, platform_name=None: CompileSupport(
            backend, False, reason="缺个组件", hint="装上它", fallback="aot_eager"
        ),
    )
    model = _toy_model()

    outcome = maybe_compile(model, enabled=True, backend="inductor", device="cpu")

    assert outcome.active is False
    assert outcome.model is model
    assert "不可用" in outcome.message and "缺个组件" in outcome.message


def test_maybe_compile_works_with_a_fallback_backend():
    """aot_eager 不依赖 Triton / MSVC，任何平台上都应当编译成功。"""
    model = _toy_model()

    outcome = maybe_compile(
        model, enabled=True, backend="aot_eager", device="cpu",
        example_input=torch.randn(4, 8),
    )

    assert outcome.active is True
    assert outcome.first_call_seconds > 0
    assert outcome.model is not model      # 这次是真的包装了
    assert "已编译" in outcome.message


def test_smoke_does_not_pollute_bn_running_stats():
    """冒烟前向必须"只看不改"：BN 的 running stats 不能被它带偏。

    首次前向是为了触发编译才跑的，如果顺手更新了统计量，训练还没开始
    统计量就已经被一批随机数据污染 —— 而且这种事不会报错，只会让指标变差。
    """
    model = build_model("small_cnn", in_channels=1, num_classes=10, image_size=28)
    model.train()
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    before = bn.running_mean.clone()

    outcome = maybe_compile(
        model, enabled=True, backend="aot_eager", device="cpu",
        example_input=torch.randn(8, 1, 28, 28),
    )

    assert outcome.active is True
    assert torch.equal(before, bn.running_mean), "冒烟前向污染了 BN 统计量"
    assert model.training is True, "冒烟后没有把模型恢复成训练模式"


# ============================================================
# unwrap_model：DDP 与 compile 的包装可以嵌套
# ============================================================
class _FakeWrapper(nn.Module):
    """模拟 DDP（`.module`）/ torch.compile（`._orig_mod`）的包装器。

    用假的而不是真的 DDP / compile：这两个东西建起来要起进程组 / 真的编译，
    而这里要测的只是"解构逻辑对不对"，与它们内部怎么实现无关。
    """

    def __init__(self, inner: nn.Module, attr: str):
        super().__init__()
        setattr(self, attr, inner)


def test_unwrap_model_handles_nested_wrappers():
    """`compile(DDP(model))` 这种嵌套也要能剥干净。

    只剥一层就会漏，而漏了的后果是 checkpoint 的 key 带前缀、单进程加载不了
    —— 保存时不报错，属于埋雷型 bug（加 DDP 那次就栽过一次）。
    """
    inner = _toy_model()
    ddp_like = _FakeWrapper(inner, "module")
    compiled_like = _FakeWrapper(ddp_like, "_orig_mod")

    assert unwrap_model(compiled_like) is inner
    assert unwrap_model(ddp_like) is inner
    assert unwrap_model(inner) is inner        # 没有包装时原样返回


# ============================================================
# canary：Windows 上 inductor 走不通是"当前的现实"
# ============================================================
@pytest.mark.skipif(sys.platform != "win32", reason="这是 Windows 特有的现状")
def test_windows_inductor_limitation_is_a_known_fact():
    """断言现状而非"正确行为"：Windows 上 inductor 需要 Triton（无 wheel）或 MSVC。

    上游哪天补上了 Windows 版 triton，或者 runner 里有了 cl.exe，这条会告警 ——
    提醒把 `compile_support.py` 的判断和 README 的说明一起更新，
    而不是让文档悄悄过时。（canary 测试的常规用法。）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    support = probe_compile_support(DEFAULT_BACKEND, device)

    if support.ok:
        warnings.warn(
            "Windows 上 inductor 变得可用了，请更新 compile_support.py 与 README",
            stacklevel=2,
        )
    else:
        assert support.fallback in FALLBACK_BACKENDS
