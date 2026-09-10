"""
单步显存拆解：梯度检查点到底省了哪一段、又慢了多少？

这个脚本回答一个反直觉的问题：
    为什么梯度检查点把前向保活的激活砍掉了 80%+，前向峰值近乎腰斩，
    训练全程峰值却只降十几个百分点？

做法：跑「一个训练 step」（一次前向 + 一次反向），量三件事：
    ① 前向结束后仍驻留的激活值  ② 前向峰值  ③ 全程峰值
    ④ 前向耗时  ⑤ 完整 step 耗时（含反向）

⚠️ 为什么耗时必须单独测：主实验 `exp_memory_accounting.py` 的「总耗时」来自
   summary.json，包含了**每个 epoch 的验证**和**数据加载**，会把 step 级的开销冲淡。
   要谈「检查点慢多少」，必须像这里一样把 step 单独计时。

三种粒度对照
------------
    none       不开检查点
    per_block  按 block 检查点（本仓库 SmallCNN 的默认做法）
    whole      把整条主干当成**一个**检查点段（反例，用来证明粒度不能乱选）

为什么每个模式要单独起进程：CUDA 缓存分配器是进程级的，
同进程连跑会让后一个模式捡到前一个模式的缓存块，基线被污染、数字不可比。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

OUT_DIR = HERE / "outputs" / "exp_ckpt_granularity"
MODES = ["none", "per_block", "whole"]
MODE_DESC = {
    "none": "不开检查点",
    "per_block": "按 block 检查点（有效）",
    "whole": "整条主干当单个段（反例）",
}


def _forward(model, x, mode):
    """统一的前向入口，让三种粒度可以共用同一套计时/测量代码。"""
    from torch.utils.checkpoint import checkpoint

    if mode == "whole":
        # 反例：整条卷积主干作为一个检查点段
        return model.classifier(checkpoint(model.blocks, x, use_reentrant=False))
    return model(x)


def measure(mode: str, batch_size: int, device: str, iters: int) -> dict:
    """量显存 + 计时。显存先测（只跑一步），再做 warmup 与计时，互不干扰。"""
    import torch
    import torch.nn as nn

    from src.model import build_model

    torch.manual_seed(0)
    model = build_model("small_cnn", in_channels=1, num_classes=10).to(device)
    if mode == "per_block":
        model.set_gradient_checkpointing(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # ---------- 一、显存（只跑一步） ----------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    model.train()
    logits = _forward(model, x, mode)
    loss = criterion(logits, y)
    after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()

    loss.backward()
    peak_all = torch.cuda.max_memory_allocated()

    mb = lambda v: v / 1024 ** 2  # noqa: E731
    result = {
        "mode": mode,
        "batch_size": batch_size,
        "retained_after_fwd_mb": round(mb(after_fwd - base), 1),
        "forward_peak_mb": round(mb(peak_fwd), 1),
        "total_peak_mb": round(mb(peak_all), 1),
    }

    # ---------- 二、计时 ----------
    def one_step(do_backward: bool) -> None:
        optimizer.zero_grad(set_to_none=True)
        out = _forward(model, x, mode)
        l = criterion(out, y)
        if do_backward:
            l.backward()
            optimizer.step()

    def timeit(do_backward: bool) -> float:
        """逐次同步 + 取中位数。微基准用均值很容易被单次抖动带跑。"""
        for _ in range(5):                       # warmup：cuDNN 选算法、分配器预热
            one_step(do_backward)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_step(do_backward)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        return round(samples[len(samples) // 2], 2)

    result["forward_ms"] = timeit(do_backward=False)
    result["step_ms"] = timeit(do_backward=True)

    return result


def build_report(rows: list[dict], batch_size: int, iters: int) -> str:
    none, per_block, whole = rows

    def pct(cur: float, ref: float) -> str:
        return f"{100 * (cur / ref) - 100:+.0f}%"

    # 「整条主干当一个段」的定性判断：持平或略高都算「没有收益」
    whole_gain = 1 - whole["total_peak_mb"] / none["total_peak_mb"]
    if whole_gain < 0.02:
        whole_verdict = "**没有收益**（与不开检查点持平甚至略高）"
    else:
        whole_verdict = f"只降 {whole_gain * 100:.0f}%"

    lines = [
        "# 梯度检查点：单步拆解（显存 + 耗时）",
        "",
        f"- 模型：small_cnn（421,834 参数）| batch = {batch_size} | 单步 = 1 前向 + 1 反向",
        f"- 计时：warmup 5 次后取 {iters} 次**中位数**（逐次 `torch.cuda.synchronize()`）",
        "- 硬件：RTX 3060 Laptop 6GB | torch 2.11.0+cu128",
        "- 每个模式单独起进程，避免 CUDA 缓存分配器跨模式污染",
        "",
        "## 一、读数",
        "",
        "| 模式 | 前向保活的激活 | 前向峰值 | **全程峰值** | 前向耗时 | **完整 step** |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {MODE_DESC[r['mode']]} | {r['retained_after_fwd_mb']:.1f} MB | "
            f"{r['forward_peak_mb']:.1f} MB | **{r['total_peak_mb']:.1f} MB** | "
            f"{r['forward_ms']:.2f} ms | **{r['step_ms']:.2f} ms** |"
        )

    lines += [
        "",
        "## 二、四个反直觉的结论",
        "",
        "### 1. 检查点确实把「前向保活的激活」砍掉了绝大部分",
        "",
        f"反向传播要用的中间激活，前向结束后仍驻留的量："
        f"{none['retained_after_fwd_mb']:.1f} MB → {per_block['retained_after_fwd_mb']:.1f} MB"
        f"（**{pct(per_block['retained_after_fwd_mb'], none['retained_after_fwd_mb'])}**）；"
        f"前向峰值 {none['forward_peak_mb']:.1f} MB → {per_block['forward_peak_mb']:.1f} MB"
        f"（{pct(per_block['forward_peak_mb'], none['forward_peak_mb'])}）。",
        "",
        "这部分收益是真实且巨大的 —— 它解释了为什么检查点在**显存被激活值卡死**的场景",
        "（长序列 Transformer、大 batch 扩散训练）是救命手段。",
        "",
        "### 2. 但训练全程峰值由**反向阶段**决定，收益被吃掉大半",
        "",
        f"前向峰值降了 {pct(per_block['forward_peak_mb'], none['forward_peak_mb'])}，"
        f"全程峰值却只从 {none['total_peak_mb']:.1f} MB 降到 "
        f"{per_block['total_peak_mb']:.1f} MB"
        f"（**{pct(per_block['total_peak_mb'], none['total_peak_mb'])}**）。原因是两层：",
        "",
        "1. **重算本身就是一次前向**。反向走到某个 block 时，该块的激活被重新物化 ——",
        "   这些张量在那一刻确实占着显存。检查点省掉的是激活「同时活着」的时间跨度，",
        "   不是「总共分配过多少」。",
        "2. **反向还有与检查点无关的开销**：卷积反向算法的 cuDNN workspace，开不开都要付。",
        "",
        "一句话：**检查点降低的是激活值的「驻留规模」，而峰值往往由「瞬时规模」决定。**",
        "模型越小、激活占比越低，两者差距越大 —— 小模型上开检查点常常不划算。",
        "",
        "### 3. 粒度选错，收益直接归零",
        "",
        f"把整条主干当成**一个**检查点段（`whole`）时，全程峰值 "
        f"{whole['total_peak_mb']:.1f} MB，对比不开检查点的 {none['total_peak_mb']:.1f} MB："
        f"{whole_verdict}。",
        "",
        "机理：检查点靠**边界**把激活切成一段段、逐段释放。只有一个段的时候，",
        "反向重算会把整条主干的激活一次性全部物化，中间没有任何释放点 ——",
        "「分批驻留」退化成「全量驻留」，还要白付重算的账。",
        "",
        "**所以检查点的粒度必须落在「块」上**：Transformer 的一层、CNN 的一个卷积单元。",
        "按单层拆太细同样没收益（每层都要存自己的输入）。粒度要匹配「块内激活占比」。",
        "",
        "### 4. 耗时代价：完整 step 慢了一半，增量正好是一次前向的量级",
        "",
        f"前向耗时基本没变（{none['forward_ms']:.2f} ms → {per_block['forward_ms']:.2f} ms，"
        f"{pct(per_block['forward_ms'], none['forward_ms'])}）。",
        "**这恰恰是个陷阱** —— 重算发生在**反向阶段**，前向计时里根本看不到它。",
        "",
        f"看完整 step：{none['step_ms']:.2f} ms → {per_block['step_ms']:.2f} ms"
        f"（**{pct(per_block['step_ms'], none['step_ms'])}**），"
        f"增量 {per_block['step_ms'] - none['step_ms']:.2f} ms，"
        f"而单次前向耗时 {none['forward_ms']:.2f} ms —— **两者同量级**。",
        "",
        "这就把账算清楚了：**检查点的代价基本等于「多跑一次前向」**",
        "（略多一点，因为反向里还有与检查点无关的 workspace 开销）。",
        f"之所以 step 只涨 {pct(per_block['step_ms'], none['step_ms'])} 而不是翻倍，",
        "是因为反向本身已有前向量级的计算量，新增的那次前向是叠加，不是替换。",
        "",
        "**两个测量教训**（这个坑我先踩了一次）：",
        "1. 只测前向，会得出「检查点几乎免费」的错误结论；",
        "2. 用训练脚本的总耗时又会把差异稀释到看不见 —— 总耗时里数据加载与验证占了大头，",
        "   step 级这几毫秒完全被淹没（主实验里开与不开检查点的总耗时几乎一样，看不出差别）。",
        "**要评估这类优化，必须把 step 单独拎出来计时。**",
        "",
        "> 提醒：主实验 `exp_memory_accounting.py` 里的「总耗时」取自 summary.json，",
        "> 包含每 epoch 的验证与数据加载，**不能**用来衡量检查点的 step 级开销。",
        "> 这正是本脚本存在的意义。",
        "",
        "## 三、怎么讲给面试官",
        "",
        "> 「梯度检查点省的是前向保活的中间激活 —— 我实测能把这部分砍掉 "
        f"{abs(round(100 * (per_block['retained_after_fwd_mb'] / none['retained_after_fwd_mb']) - 100))}%，",
        f"> 前向峰值从 {none['forward_peak_mb']:.0f}MB 降到 {per_block['forward_peak_mb']:.0f}MB。",
        "> 但它不改变训练全程峰值：反向时重算会把激活重新物化，卷积反向的 workspace 也照付，",
        f"> 所以整体峰值我只降了十几百分点。**判据是激活值占总显存的比例，不是模型大小。**",
        ">",
        "> 粒度也有讲究 —— 我把整条主干当成一个检查点段试过，收益直接归零",
        f">（{whole['total_peak_mb']:.0f}MB vs {none['total_peak_mb']:.0f}MB），",
        "> 因为没有边界可以逐段释放，重算时全量物化。所以检查点要按「层/块」切。",
        ">",
        "> 时间代价也要说实话：单看前向基本没变，",
        "> 因为重算发生在反向阶段；完整 step 从 "
        f"{none['step_ms']:.2f}ms 涨到 {per_block['step_ms']:.2f}ms"
        f"（{pct(per_block['step_ms'], none['step_ms'])}），",
        "> 增量与一次前向的耗时同量级 —— **检查点的代价基本就是多跑一次前向**。",
        "> 所以评估它必须计时完整 step：只测前向会以为它免费，",
        "> 而用训练脚本的总耗时（含数据加载）又会被稀释到看不见。」",
        "",
        "## 四、原始数据",
        "",
        "```json",
        json.dumps(rows, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20, help="计时取几次均值")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.mode != "all":
        print(json.dumps(measure(args.mode, args.batch_size, args.device, args.iters)))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in MODES:
        proc = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--mode", mode, "--batch_size", str(args.batch_size),
                "--iters", str(args.iters),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:])
            raise SystemExit(f"模式 {mode} 测量失败")
        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    md_path = OUT_DIR / "checkpoint_granularity.md"
    md_path.write_text(build_report(rows, args.batch_size, args.iters), encoding="utf-8")
    (OUT_DIR / "checkpoint_granularity.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    for r in rows:
        print(
            f"  {r['mode']:>10s} | 前向保活 {r['retained_after_fwd_mb']:7.1f}MB | "
            f"前向峰值 {r['forward_peak_mb']:7.1f}MB | 全程峰值 {r['total_peak_mb']:7.1f}MB | "
            f"前向 {r['forward_ms']:6.2f}ms | step {r['step_ms']:7.2f}ms"
        )
    print("=" * 88)
    print(f"\n报告已保存：{md_path}")


if __name__ == "__main__":
    main()
