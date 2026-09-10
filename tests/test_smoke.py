"""端到端冒烟测试：走真实的命令行入口，跑完整流水线。

为什么必须有一条这种测试？
  单元测试全绿但 `python src/main.py` 一跑就崩，是开源项目最常见的尴尬。
  这条测试用 `--dataset synthetic`（不下载数据），几秒钟跑完一遍完整流程：
  配置 -> 数据 -> 模型 -> 训练 -> 评估 -> 落盘。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "src" / "main.py"
CONFIG = ROOT / "configs" / "mnist.yaml"


def run_main(tmp_path: Path, *extra: str, exp_name: str = "e2e"):
    """跑一次 main.py，返回 (完成后的进程, 输出目录)。"""
    out_dir = tmp_path / "outputs"
    cmd = [
        sys.executable, str(MAIN),
        "--config", str(CONFIG),
        "--output_dir", str(out_dir),
        "--exp_name", exp_name,
        "--dataset", "synthetic",         # 离线，不下载
        "--num_workers", "0",
        *extra,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    return proc, out_dir / exp_name


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    """默认配置跑一次，供多条断言复用（避免每个用例都重跑一遍）。"""
    tmp = tmp_path_factory.mktemp("e2e")
    proc, run_dir = run_main(tmp, "--epochs", "1", "--batch_size", "128")
    assert proc.returncode == 0, f"main.py 退出码非 0\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc, run_dir


# ============================================================
# 基本流水线
# ============================================================
def test_run_completes_and_reports(default_run):
    proc, _ = default_run
    assert "训练完成" in proc.stdout
    assert "测试集准确率" in proc.stdout


def test_all_artifacts_are_written(default_run):
    _, run_dir = default_run
    for name in ["best.pt", "summary.json", "config_resolved.yaml", "train.log"]:
        assert (run_dir / name).exists(), f"缺少产物 {name}"


def test_summary_contents(default_run):
    _, run_dir = default_run
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_params"] == 421_834
    assert 0.0 <= summary["test_acc"] <= 1.0
    assert summary["config"]["epochs"] == 1
    assert summary["config"]["dataset"] == "synthetic"
    assert len(summary["history"]) == 1            # 没有早停时，history 长度 = epochs
    assert summary["history"][0]["epoch"] == 1


def test_checkpoint_is_loadable_and_carries_config(default_run):
    """权重 + 配置必须一起能读回来 —— 半年后复现实验全靠它。"""
    _, run_dir = default_run
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert set(ckpt) >= {"model", "optimizer", "epoch", "global_step", "config", "metrics"}
    assert ckpt["config"]["epochs"] == 1
    assert "blocks.0.0.weight" in ckpt["model"]   # 主干参数在里面


def test_config_resolved_matches_yaml_loader(default_run):
    """落盘的 config_resolved.yaml 必须能被 TrainConfig 读回来（自洽性）。"""
    from src.config import TrainConfig

    _, run_dir = default_run
    cfg = TrainConfig.from_yaml(run_dir / "config_resolved.yaml")
    assert cfg.epochs == 1
    assert cfg.dataset == "synthetic"


# ============================================================
# 梯度检查点的两条路径
# ============================================================
def test_gradient_checkpointing_path_is_active(tmp_path):
    proc, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128", "--gradient_checkpointing",
        exp_name="e2e_ckpt",
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["gradient_checkpointing"] is True
    assert summary["gradient_checkpointing_active"] is True
    assert "梯度检查点：开" in proc.stdout


def test_unsupported_model_warns_instead_of_failing(tmp_path):
    """mlp 不支持检查点：应当明确告警并忽略，而不是静默失效或直接报错。"""
    proc, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128",
        "--model", "mlp", "--gradient_checkpointing", exp_name="e2e_mlp",
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["gradient_checkpointing_active"] is False
    assert "未实现梯度检查点" in proc.stdout


# ============================================================
# 其它配置路径
# ============================================================
def test_gradient_accumulation_path(tmp_path):
    proc, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "32", "--grad_accum_steps", "4",
        exp_name="e2e_accum",
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "等效 batch size" in proc.stdout
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["grad_accum_steps"] == 4


def test_amp_on_cpu_is_disabled_with_warning(tmp_path):
    """CPU 上开 AMP 应当被自动关掉并提示，而不是崩在 autocast 里。"""
    proc, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128",
        "--amp", "--device", "cpu", exp_name="e2e_amp",
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "已自动关闭混合精度" in proc.stdout


def test_overrides_beat_config_file(tmp_path):
    """命令行 > YAML 的优先级必须成立，否则实验参数会被配置文件偷偷改掉。"""
    proc, run_dir = run_main(
        tmp_path, "--epochs", "2", "--batch_size", "64", "--lr", "5e-4",
        exp_name="e2e_override",
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    cfg = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["config"]
    assert (cfg["epochs"], cfg["batch_size"], cfg["lr"]) == (2, 64, 5e-4)
