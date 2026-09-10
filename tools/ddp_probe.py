"""DDP 环境自检：确认这台机器到底能不能跑多进程集合通信。

为什么需要单独一个自检
--------------------
DDP 依赖「多个进程能通过网络互相找到并通信」。这条路在某些环境里是断的，
而且断的方式五花八门：

- Windows 上 `torchrun` 需要 TCPStore，而部分 torch 构建没编进 libuv
  （报 `use_libuv was requested but PyTorch was built without libuv support`）
- 容器 / CI runner 里 loopback 监听被防火墙或网络策略拦掉
- 主机名解析不了，gloo 拿不到可用的设备地址
- 多网卡机器上 gloo 选错了网卡

这些**都不是使用者代码的问题**，但症状往往是"训练卡住不动"或
"报一个和 DDP 无关的连接错误"，非常难定位。所以换一台新机器（或 CI 上换 runner）
时，先跑这个自检：

    python tools/ddp_probe.py

它只做一件事：起 2 个进程，各算一个数，all_reduce 求和，校验结果对不对。
**完全不碰本仓库的任何业务代码**，所以它能干净地区分
「环境不行」和「我们的封装写错了」。

退出码：0 = 环境可用；1 = 环境不可用（stdout/stderr 里带原始报错）。
CI 里用它作为分布式测试的**能力门禁** —— 环境不支持就明确 skip，
而不是让一整套测试长期红着，让人分不清是真 bug 还是环境问题。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 同 ddp_launch.py：挂目录而不是包，跳过 src/__init__.py 的 6.4 秒 torch 导入
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from console import force_utf8_stdout  # noqa: E402  （必须在 sys.path 之后）


def _run_worker(rank: int, world_size: int, rendezvous: str, timeout_sec: int) -> int:
    """子进程：初始化进程组，做一次 all_reduce，校验结果。"""
    import torch
    import torch.distributed as dist

    store = dist.FileStore(rendezvous, world_size)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_sec),
    )
    try:
        # rank r 报 r+1，全加起来应该是 1+2+...+W
        t = torch.tensor([float(rank + 1)])
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2
        if abs(t.item() - expected) > 1e-6:
            print(f"[rank {rank}] ❌ all_reduce 结果错误：得到 {t.item()}，期望 {expected}")
            return 1
        print(f"[rank {rank}] ✅ all_reduce 正确（{t.item()}）")
        return 0
    finally:
        dist.destroy_process_group()


def probe(world_size: int = 2, timeout_sec: int = 60) -> int:
    """起 world_size 个进程做一次集合通信。返回 0 表示环境可用。"""
    if world_size < 2:
        raise ValueError("world_size 至少要 2，否则测不出通信")

    rv_dir = Path(tempfile.mkdtemp(prefix="ddp_probe_"))
    rendezvous = rv_dir / f"rv_{uuid.uuid4().hex[:8]}"
    rendezvous.write_bytes(b"")
    rendezvous.unlink()          # 让 FileStore 自己创建，避免残留内容干扰

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
    }
    procs = [
        subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--worker", str(rank),
                "--world_size", str(world_size),
                "--rendezvous", str(rendezvous),
                "--timeout", str(timeout_sec),
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for rank in range(world_size)
    ]

    codes = []
    for rank, proc in enumerate(procs):
        try:
            out, _ = proc.communicate(timeout=timeout_sec + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            out = (out or "") + f"\n[rank {rank}] ⏱ 超时 {timeout_sec + 30}s 未退出"
            codes.append(-1)
        else:
            codes.append(proc.returncode)
        print(f"--- rank {rank} (exit {codes[-1]}) ---")
        print((out or "").rstrip())

    rendezvous.unlink(missing_ok=True)
    try:
        rv_dir.rmdir()
    except OSError:
        pass

    if any(c != 0 for c in codes):
        print(
            f"\n❌ 本环境**无法**跑多进程 gloo 集合通信（rank 退出码 {codes}）。\n"
            "   这是环境限制，不是训练代码的问题。"
        )
        return 1

    print(f"\n✅ 本环境可以跑 DDP（world_size={world_size} 的 gloo 集合通信正常）")
    return 0


def main() -> int:
    # 必须早于任何 print（cp1252 管道下中文会崩，见 src/console.py）
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="DDP 环境自检")
    p.add_argument("--world_size", type=int, default=2, help="进程数，默认 2")
    p.add_argument("--timeout", type=int, default=60, help="单个进程的集合通信超时（秒）")
    # 内部使用：子进程 worker 模式
    p.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--rendezvous", type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker is not None:
        if args.rendezvous is None:
            print("worker 模式必须提供 --rendezvous")
            return 2
        return _run_worker(args.worker, args.world_size, args.rendezvous, args.timeout)

    return probe(args.world_size, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
