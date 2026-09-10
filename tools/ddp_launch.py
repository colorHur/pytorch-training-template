"""不依赖 TCPStore 的 DDP 启动器 —— 用 `FileStore` 做 rendezvous。

为什么要写这个
-------------
标准做法是 `torchrun --nproc_per_node=N`，它需要创建一个 **TCPStore**
（所有进程通过一个 TCP 端点交换地址）。但在某些环境下这条路是断的：

- **Windows + 没编进 libuv 的 torch 构建**：
  `DistStoreError: use_libuv was requested but PyTorch was built without libuv support,
  run with USE_LIBUV=0 to disable it`。
  更坑的是 `USE_LIBUV=0` **也不管用** —— 因为这些构建里 Windows 的 TCPStore
  只有 libuv 一条实现路径，旧的非 libuv 分支已经被删了。
  （用 `python -c "import torch.distributed as d; d.TCPStore(...)"` 一试就知道。）
- 某些容器 / 受限沙箱里 loopback 上的监听被禁掉。

`FileStore` 用一个**文件**当共享存储，进程之间靠文件读写交换地址，
完全不碰 TCP。集合通信本身（gloo 的 socket）在 Windows 上是正常的，
所以整条 DDP 链路就跑通了。

用法
----
    # 2 进程，CPU（会自动把 LOCAL_RANK 映射到设备）
    python tools/ddp_launch.py --nproc_per_node 2 -- src/main.py --dataset synthetic --epochs 2

    # 参数会原样透传给被启动的脚本
    python tools/ddp_launch.py --nproc_per_node 2 -- src/main.py --epochs 3 --lr 5e-4

⚠️ 这只是**开发和调试的替代路径**。真实训练环境（Linux + 多卡）请用 `torchrun`：
    torchrun --nproc_per_node=4 src/main.py ...
它带容错重启（elastic）、更完善的日志和进程管理，本脚本故意做得极简。

输出处理
--------
每个子进程的输出**各自重定向到独立临时文件**，全部结束后按 rank 顺序打印。
为什么不直接继承父进程的 stdout？两个进程同时往同一个管道写，中文行会在
字节层面交错，日志直接变乱码。用临时文件既不会死锁（管道写满会死锁），
顺序也可控。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 挂的是 `src/` **目录**而不是包，于是 `import console` 直接命中
# `src/console.py`，完全跳过 `src/__init__.py` —— 后者会连锁导入 torch，
# 实测 `import src` 要 6.4 秒，对一个启动器来说太贵了。
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from console import force_utf8_stdout  # noqa: E402  （必须在 sys.path 之后）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 FileStore 启动 DDP（Windows / 无 TCPStore 环境）")
    p.add_argument("--nproc_per_node", "-n", type=int, default=2, help="进程数（= world_size）")
    p.add_argument(
        "--rendezvous_dir",
        type=str,
        default=None,
        help="放置 rendezvous 文件的目录，默认系统临时目录",
    )
    p.add_argument("--timeout", type=float, default=600.0, help="单个进程的超时（秒）")
    p.add_argument("script", nargs=argparse.REMAINDER, help="`--` 之后是要运行的脚本及其参数")
    return p.parse_args()


def main() -> int:
    # 必须早于任何 print：CI 的 windows runner 用 cp1252 编码管道输出，
    # 中文日志会直接抛 UnicodeEncodeError 把启动器带崩（踩过一次，见 src/console.py）。
    force_utf8_stdout()
    args = parse_args()

    script_args = [a for a in args.script if a != "--"]
    if not script_args:
        print("用法：python tools/ddp_launch.py --nproc_per_node 2 -- src/main.py [参数...]")
        return 2

    world_size = max(1, args.nproc_per_node)
    rendezvous_dir = Path(args.rendezvous_dir) if args.rendezvous_dir else Path(tempfile.gettempdir())
    rendezvous_dir.mkdir(parents=True, exist_ok=True)
    rendezvous_file = rendezvous_dir / f"ddp_rendezvous_{uuid.uuid4().hex[:8]}"
    # FileStore 要求文件预先存在（相当于一块空的共享内存）
    rendezvous_file.write_bytes(b"")
    rendezvous_file.unlink()  # 让 FileStore 自己创建，避免残留内容干扰

    print(f"[ddp_launch] world_size={world_size} | rendezvous={rendezvous_file}")
    print(f"[ddp_launch] 运行：{' '.join(script_args)}\n")

    log_paths = []
    procs = []
    for rank in range(world_size):
        env = {
            **os.environ,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "DDP_RENDEZVOUS_FILE": str(rendezvous_file),
            # 让 `python src/main.py` 能找到 src 包
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), os.environ.get("PYTHONPATH", "")]
            ).strip(os.pathsep),
        }
        log_path = rendezvous_dir / f"{rendezvous_file.name}_rank{rank}.log"
        log_paths.append(log_path)
        handle = log_path.open("w", encoding="utf-8", errors="replace")
        procs.append(
            subprocess.Popen(
                [sys.executable, *script_args],
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        )
        handle.close()   # 子进程持有自己的 fd，父进程不需要

    codes = []
    for rank, proc in enumerate(procs):
        try:
            codes.append(proc.wait(timeout=args.timeout))
        except subprocess.TimeoutExpired:
            print(f"[ddp_launch] ⚠️ rank {rank} 超时，强制结束全部进程")
            for p in procs:
                p.kill()
            codes.append(-1)

    # 按 rank 顺序回放日志，保证中文行不会交错成乱码
    for rank, log_path in enumerate(log_paths):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        print(f"{'=' * 60}\n[rank {rank}] 退出码 {codes[rank]}\n{'=' * 60}")
        print(text.rstrip())
        log_path.unlink(missing_ok=True)

    rendezvous_file.unlink(missing_ok=True)

    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        print(f"\n[ddp_launch] ❌ 失败的 rank：{failed}")
        return 1
    print(f"\n[ddp_launch] ✅ 全部 {world_size} 个 rank 正常退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
