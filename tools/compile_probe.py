"""`torch.compile` 环境自检：这台机器到底能不能用上编译加速、缺什么。

用法
----
    python tools/compile_probe.py                # 检查当前设备
    python tools/compile_probe.py --device cuda
    python tools/compile_probe.py --all          # 把退路后端也各真编译一次

为什么需要
--------
`torch.compile` 的报错发生在**第一次前向**，而且只给一句
`TritonMissing` / `Compiler: cl is not found` —— 不说缺的是哪个组件、不说怎么修，
更不说不开编译器还有别的路。Windows 上这几乎必然发生（PyTorch 没有 Windows 版
triton wheel；CPU 后端要 MSVC 的 `cl.exe`）。

这个自检做两件事，缺一不可：

1. **探测**（`probe_compile_support`）—— 给出结构化的原因与退路；
2. **真编译一次** —— 探测是"根据已知条件推断"，真跑一次才是 ground truth。
   两者不一致时，以真跑为准（比如某天平台行为变了，探测的判断就过时了）。

退出码：**0** = 默认后端可用；**1** = 默认不可用但退路可用；**2** = 全都不可用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 挂 `src/` 目录（不是包），与 ddp_launch.py / ddp_probe.py 同样的理由
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from compile_support import DEFAULT_BACKEND, FALLBACK_BACKENDS, probe_compile_support  # noqa: E402
from console import force_utf8_stdout  # noqa: E402


def real_compile_check(backend: str, device: str) -> tuple[bool, str]:
    """真的编译一个小函数并跑一次前向 + 反向，返回 `(是否成功, 错误)`。

    只用 sin/cos 这类算子：不依赖任何本仓库的模型，所以它能干净地区分
    「环境不行」和「我们的封装写错了」。
    """
    try:
        compiled = torch.compile(lambda x: (x.sin() + x.cos()) * 2.0, backend=backend)
        x = torch.randn(64, device=device, requires_grad=True)
        compiled(x).sum().backward()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - 自检工具，任何异常都只报告不抛
        first_line = str(exc).splitlines()[0] if str(exc) else ""
        return False, f"{type(exc).__name__}: {first_line[:160]}"


def check_one(backend: str, device: str, label: str) -> bool:
    """探测 + 真编译，打印结果。返回"这个后端是否真的能用"。"""
    support = probe_compile_support(backend, device)
    ok, error = real_compile_check(backend, device)

    print(f"\n[{label} {backend}]")
    print(f"  探测：{support.describe()}")
    if ok:
        print("  实编：✅ 通过（能编译、能前向、能反向）")
    else:
        print(f"  实编：❌ 失败 —— {error}")

    if support.ok and not ok:
        print("  ⚠️  探测说可用但实编失败 —— 以实编为准（探测的已知条件可能过时了）")
    if not support.fallback and not ok:
        print("  退路：无（换设备或换平台）")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="torch.compile 环境自检")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="要检查的后端")
    parser.add_argument("--all", action="store_true", help="顺带把退路后端也各查一遍")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    print(f"torch {torch.__version__} | 设备 {device} | 平台 {sys.platform}")

    default_ok = check_one(args.backend, device, "默认后端")

    fallback_ok = False
    if args.all or not default_ok:
        print("\n" + "-" * 64)
        print("退路后端（不经过 Inductor 的 kernel 生成，因此不需要 Triton / MSVC）")
        for backend in FALLBACK_BACKENDS:
            if backend == args.backend:
                continue
            if check_one(backend, device, "退路"):
                fallback_ok = True

    print("\n" + "=" * 64)
    if default_ok:
        print("✅ 默认后端可用 —— 直接用 `--compile` 即可")
        return 0
    if fallback_ok:
        print(
            "⚠️  默认后端（inductor）不可用，但**有退路**。\n"
            "   用 `--compile --compile_backend <上面通过的那个>` 也能编译，\n"
            "   只是加速幅度不如 inductor（它不做 kernel 生成，省的是 launch 开销）。"
        )
        return 1
    print("❌ 没有任何可用后端 —— 本机不上编译这条路，先用别的优化手段。")
    return 2


if __name__ == "__main__":
    force_utf8_stdout()
    sys.exit(main())
