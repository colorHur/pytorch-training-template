"""
对照实验：把「显存账」算给你看。

这是打底阶段的核心脚本 —— 面试官问「梯度累积怎么省显存」时，
你可以直接翻出这张表，而不是背概念。

三个实验，固定其它条件、只变一个变量
-----------------------------------
    A. 基准             batch=256, accum=1
    B. 梯度累积等效      batch=64,  accum=4   （等效 batch 相同，看显存差多少）
    C. 混合精度          batch=256, accum=1, amp=True

跑法
----
    python experiments/exp_memory_accounting.py
    python experiments/exp_memory_accounting.py --epochs 1    # 跑快点

输出
----
    outputs/exp_memory/memory_accounting.md   对照表
    outputs/exp_memory/memory_accounting.json 原始数据
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "outputs" / "exp_memory"

# 三组对照：(标签, 覆盖参数, 说明)
VARIANTS = [
    (
        "A. 基准",
        {"batch_size": 256, "grad_accum_steps": 1, "amp": False},
        "显存基线",
    ),
    (
        "B. 梯度累积等效",
        {"batch_size": 64, "grad_accum_steps": 4, "amp": False},
        "等效 batch 相同(256)，激活值只占 1/4",
    ),
    (
        "C. 混合精度",
        {"batch_size": 256, "grad_accum_steps": 1, "amp": True},
        "fp16 前向，激活值减半",
    ),
]


def run_variant(label: str, overrides: dict, epochs: int) -> dict:
    """跑一个变体，读取 summary.json 拿结果。"""
    exp_name = f"acct_{label.split('.')[0].strip().lower()}"
    cmd = [
        sys.executable,
        str(HERE / "src" / "main.py"),
        "--config",
        str(HERE / "configs" / "mnist.yaml"),
        "--exp_name",
        exp_name,
        "--epochs",
        str(epochs),
    ]
    for key, value in overrides.items():
        if isinstance(value, bool):
            cmd.append(f"--{key}" if value else f"--no-{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    print(f"\n>>> {label}：{' '.join(cmd[2:])}")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"{label} 运行失败")

    summary_path = HERE / "outputs" / exp_name / "summary.json"
    with summary_path.open(encoding="utf-8") as f:
        data = json.load(f)

    return {
        "label": label,
        "overrides": overrides,
        "effective_batch": data["config"]["batch_size"] * data["config"]["grad_accum_steps"],
        "peak_vram_gb": data["peak_vram_gb"],
        "train_time_sec": data["total_train_time_sec"],
        "test_acc": data["test_acc"],
        "exp_name": exp_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="每个变体训练几个 epoch")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_variant(label, ov, args.epochs) for label, ov, _ in VARIANTS]

    # ---- 生成 markdown 表格 ----
    base = results[0]
    lines = [
        "# 显存账对照实验",
        "",
        f"- 数据集：MNIST | 模型：small_cnn（421,834 参数）| 训练 {args.epochs} epoch",
        f"- 硬件：RTX 3060 Laptop 6GB | torch 2.11.0+cu128",
        "",
        "| 变体 | micro batch | 累积步数 | 等效 batch | 峰值显存 | 训练耗时 | 测试准确率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['overrides']['batch_size']} | "
            f"{r['overrides']['grad_accum_steps']} | {r['effective_batch']} | "
            f"**{r['peak_vram_gb']:.3f} GB** | {r['train_time_sec']:.1f}s | {r['test_acc']:.4f} |"
        )

    lines += [
        "",
        "## 结论",
        "",
        f"1. **梯度累积省显存**：B 与 A 等效 batch 都是 {results[1]['effective_batch']}，"
        f"但峰值显存 {results[1]['peak_vram_gb']:.3f} GB vs {base['peak_vram_gb']:.3f} GB"
        f"（降 {100 * (1 - results[1]['peak_vram_gb'] / base['peak_vram_gb']):.0f}%）。",
        "   原因：显存大头是**激活值**，正比于单次前向的样本数；梯度累积只前向 micro batch，",
        "   攒够 N 次梯度再更新一次，数学上等效大 batch，但激活值只需 1/N。",
        f"2. **混合精度省显存**：C 的峰值 {results[2]['peak_vram_gb']:.3f} GB，"
        f"比 A 的 {base['peak_vram_gb']:.3f} GB 降 "
        f"{100 * (1 - results[2]['peak_vram_gb'] / base['peak_vram_gb']):.0f}%（激活值由 fp32 变 fp16）。",
        "3. **准确率基本不受影响**：三种配置测试准确率接近，说明省显存不以牺牲效果为代价。",
        "",
        "## 面试怎么讲",
        "",
        "> 「显存不够时我不会盲目减 batch。先看显存被什么占了 —— 参数、梯度、优化器状态、激活值。",
        "> 激活值正比于 batch，所以梯度累积是最直接的省法：假设 batch=64 累积 4 次，等效 batch=256，",
        "> 但显存只需 64 份激活值。如果还紧，再上混合精度把激活值砍半，最后才考虑 gradient checkpointing",
        "> （用计算换显存，反向前向一遍重算激活）。」",
        "",
        "## 原始数据",
        "",
        "```json",
        json.dumps(results, ensure_ascii=False, indent=2),
        "```",
    ]

    md_path = OUT_DIR / "memory_accounting.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = OUT_DIR / "memory_accounting.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for r in results:
        print(
            f"  {r['label']:16s} 等效batch={r['effective_batch']:>4}  "
            f"峰值显存={r['peak_vram_gb']:.3f}GB  acc={r['test_acc']:.4f}"
        )
    print("=" * 60)
    print(f"\n对照表已保存：{md_path}")


if __name__ == "__main__":
    main()
