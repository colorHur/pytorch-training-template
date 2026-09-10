"""`torch.compile` 的可用性探测、后端选择与降级说明。

为什么需要单独一层
----------------
在 Linux 上 `torch.compile(model)` 基本是一行搞定的加速手段，但在别的平台上它会
抛出**和加速毫无关系的**异常 —— 而且是在**第一次前向**时才抛（`torch.compile()`
本身只做标记，真正的编译是懒的）：

- **Windows + CUDA**：
  `TritonMissing: Cannot find a working triton installation.`
  → Inductor 的 GPU 后端靠 Triton 生成 kernel，而 **PyTorch 不提供 Windows 版
    triton wheel**（`pip install triton` 在 Windows 上直接找不到包）。
- **Windows + CPU**：
  `InductorError: RuntimeError: Compiler: cl is not found.`
  → Inductor 的 CPU 后端要把生成的 C++ 编译成动态库，需要 MSVC 的 `cl.exe`。

两种报错里没有一个字提到"你缺什么、怎么修"，而且训练要跑到第 1 步才炸。
所以这一层做三件事：

1. `probe_compile_support()` 主动探测，给出**结构化**的原因与修法；
2. `maybe_compile()` 按探测结果编译或降级，风格与
   `enable_gradient_checkpointing()` / `maybe_convert_sync_bn()` 保持一致 ——
   **能开就开，开不了要说清楚为什么**；
3. `smoke_compile()` 拿真实形状的输入**先跑一次前向**，把"第一次前向才炸"
   提前到启动阶段，顺便量出**首次编译的开销**（一个常被忽略的成本）。

`aot_eager` / `cudagraphs` 这类**不经过 kernel 生成**的后端实测在 Windows 上可用，
所以探测失败时它们是有意义的退路 —— 只是加速幅度远不如 inductor。
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from dataclasses import dataclass

import torch

#: Inductor 是 PyTorch 的默认后端，也是唯一需要 Triton / MSVC 的那个
DEFAULT_BACKEND = "inductor"

#: 不依赖 Triton（GPU）与 MSVC（CPU）的退路，实测在 Windows 上可跑
FALLBACK_BACKENDS = ("aot_eager", "cudagraphs")


@dataclass
class CompileSupport:
    """某个后端在「当前平台 + 当前设备」上是否可用。"""

    backend: str
    ok: bool
    reason: str = ""      # 不可用的人话原因
    hint: str = ""        # 怎么修 / 有哪些退路
    fallback: str = ""    # 探测失败时建议的可用后端

    def describe(self) -> str:
        """不带「torch.compile」前缀的描述 —— 前缀由调用方按场景补。"""
        if self.ok:
            return f"可用（backend={self.backend}）"
        text = f"不可用（backend={self.backend}）：{self.reason}"
        if self.hint:
            text += f"；{self.hint}"
        return text


@dataclass
class CompileOutcome:
    """`maybe_compile()` 的结果：模型 + 是否真的编译上了 + 给人看的说明。"""

    model: torch.nn.Module
    active: bool
    message: str
    first_call_seconds: float = 0.0


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _has_msvc() -> bool:
    """Inductor 的 CPU 后端要调 MSVC 的 cl.exe 来编译生成的 C++。"""
    return shutil.which("cl") is not None


def probe_compile_support(
    backend: str = DEFAULT_BACKEND,
    device: str = "cpu",
    platform_name: str | None = None,
) -> CompileSupport:
    """判断 `backend` 在当前平台 / 设备上能否工作，并给出人话原因。

    `platform_name` 存在的唯一目的是**让测试能模拟别的平台**（默认 `sys.platform`）：
    Windows 上的两类缺失是本模块存在的全部理由，所以它必须是可测的 ——
    否则这段逻辑只能靠"在 Windows 上手跑一遍"来保证，CI（Linux）永远覆盖不到。
    """
    name = platform_name or sys.platform

    if backend != DEFAULT_BACKEND:
        # aot_eager / cudagraphs 等不做 kernel 生成，既不依赖 Triton 也不依赖 MSVC
        return CompileSupport(backend, True)

    if name == "win32":
        if device.startswith("cuda") and not _has_module("triton"):
            return CompileSupport(
                backend,
                False,
                reason="Inductor 的 GPU 后端依赖 Triton 生成 kernel，"
                "而 PyTorch 在 Windows 上没有 triton wheel",
                hint="换 --compile_backend cudagraphs（免 Triton）、改用 Linux/WSL，"
                "或自行安装非官方 triton-windows",
                fallback="cudagraphs",
            )
        if not device.startswith("cuda") and not _has_msvc():
            return CompileSupport(
                backend,
                False,
                reason="Inductor 的 CPU 后端要把生成的 C++ 编译成动态库，需要 MSVC 的 cl.exe",
                hint="装 Visual Studio Build Tools（含 C++ 工具链）、改用 Linux，"
                "或换 --compile_backend aot_eager（免编译器）",
                fallback="aot_eager",
            )

    return CompileSupport(backend, True)


def smoke_compile(
    compiled: torch.nn.Module,
    example_input: torch.Tensor,
) -> tuple[bool, float, str]:
    """跑一次前向把真实编译触发掉，返回 `(是否成功, 首次耗时秒, 错误说明)`。

    两个作用：
    - `torch.compile` 是**懒**的，`torch.compile(model)` 只做标记；真正的问题
      在第一次前向才暴露。这里把它提前到启动阶段，失败也能干净地回退。
    - **首次调用包含编译开销**，这个数字常被忽略（小模型上它可能比"省下的时间"
      还大），所以量出来告诉使用者。

    前向包在 `eval()` + `no_grad()` 里，避免这次冒烟污染 BatchNorm 的 running
    stats —— 否则还没开始训练，统计量就已经被一批随机数据带偏了。
    """
    was_training = compiled.training
    compiled.eval()
    try:
        with torch.no_grad():
            started = time.perf_counter()
            compiled(example_input)
            elapsed = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - 编译失败的原因五花八门，统一降级
        return False, 0.0, f"{type(exc).__name__}: {exc}"
    finally:
        compiled.train(was_training)
    return True, elapsed, ""


def maybe_compile(
    model: torch.nn.Module,
    enabled: bool,
    backend: str = DEFAULT_BACKEND,
    device: str = "cpu",
    example_input: torch.Tensor | None = None,
) -> CompileOutcome:
    """按需编译模型；开不了就带着原因回退，交给调用方决定怎么提示。

    - 探测不通过 → 直接不编译，返回人话说明（**不抛异常**，训练照常继续）
    - 给了 `example_input` → 先做一次冒烟前向，把编译失败拦在启动阶段
    """
    if not enabled:
        return CompileOutcome(model, False, "")

    support = probe_compile_support(backend, device)
    if not support.ok:
        return CompileOutcome(model, False, support.describe())

    try:
        compiled = torch.compile(model, backend=backend)
    except Exception as exc:  # noqa: BLE001
        return CompileOutcome(model, False, f"torch.compile 调用失败：{type(exc).__name__}: {exc}")

    message = f"已编译（backend={backend}，首次前向才真正编译）"
    first_call = 0.0

    if example_input is not None:
        ok, first_call, error = smoke_compile(compiled, example_input)
        if not ok:
            return CompileOutcome(
                model,
                False,
                f"首次前向编译失败，已回退到未编译的模型：{error}",
            )
        message = f"已编译（backend={backend}，首次前向含编译共 {first_call:.2f}s）"

    return CompileOutcome(compiled, True, message, first_call)
