"""端到端冒烟测试：走真实的命令行入口，跑完整流水线。

为什么必须有一条这种测试？
  单元测试全绿但 `python src/main.py` 一跑就崩，是开源项目最常见的尴尬。
  这条测试用 `--dataset synthetic`（不下载数据），几秒钟跑完一遍完整流程：
  配置 -> 数据 -> 模型 -> 训练 -> 评估 -> 落盘。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "src" / "main.py"
CONFIG = ROOT / "configs" / "mnist.yaml"


def run_main(tmp_path: Path, *extra: str, exp_name: str = "e2e", env: dict | None = None):
    """跑一次 main.py，返回 (完成后的进程, 输出目录)。

    `env` 会叠加在当前环境上（而不是替换），方便构造"换个 locale"的场景。
    """
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
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=full_env,
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
    for name in ["best.pt", "last.pt", "summary.json", "config_resolved.yaml", "train.log"]:
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


def test_checkpoint_carries_everything_a_resume_needs(default_run):
    """`best.pt` 不只是"权重 + 配置"：它（和 `last.pt`）必须能**把训练接下去**。

    续训要的六类状态见 `src/checkpoint.py` 的表。这里逐项点名，
    将来谁把某一项从 payload 里删掉，这条测试会直接指出来是哪一个。
    """
    _, run_dir = default_run
    ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    for key in (
        "model", "optimizer", "epoch", "global_step",
        "best_val_acc", "patience_counter", "history", "config", "metrics", "rng",
    ):
        assert key in ckpt, f"last.pt 缺了续训必需的 {key}"
    assert {"torch", "python", "numpy", "loader"} <= set(ckpt["rng"])
    assert ckpt["format_version"] >= 2


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
# 断点续训：真的从存档处接下去
# ============================================================
def test_resume_continues_from_the_saved_epoch(tmp_path):
    """跑 1 轮 → `--resume last` 接着跑到 2 轮。"""
    first, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128", exp_name="e2e_resume"
    )
    assert first.returncode == 0, first.stdout[-2500:]
    assert (run_dir / "last.pt").exists(), "没有 last.pt 就没法 --resume last"

    second, _ = run_main(
        tmp_path, "--epochs", "2", "--batch_size", "128", "--resume", "last",
        exp_name="e2e_resume",
    )
    assert second.returncode == 0, second.stdout[-2500:]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["resume"]["enabled"] is True
    assert summary["resume"]["from_epoch"] == 1
    assert summary["resume"]["restored_everything"] is True
    assert summary["resume"]["missing_keys"] == []
    # history 是**接着写**的：长度 2 且编号连续，说明第 1 轮的记录被恢复了
    assert [row["epoch"] for row in summary["history"]] == [1, 2]
    assert "已训完 1 个 epoch" in second.stdout

    # 改大 --epochs 会让 cosine 调度按新的总步数重新规划 —— 这件事必须被说出来，
    # 否则用户会以为"续训 = 接着那条曲线走"
    assert "--epochs 由 1 改成 2" in second.stdout


def test_resume_reproduces_uninterrupted_training(tmp_path):
    """本里程碑的核心断言：**1 轮 + 断点 + 2 轮** ≡ **连续 3 轮**（逐位一致）。

    为什么敢断言"逐位"：这个模板把随机性的四个来源都关在明处 ——
    固定 seed、显式 shuffle generator、`num_workers=0`、CPU 上不开 AMP/compile。
    所以只要**存全了**（权重 / 优化器矩 / global_step / 早停基准 / RNG / 数据顺序），
    续训就必须严丝合缝地等于连续训练。差一点，说明漏了某样东西。

    注意这里两次都传 `--epochs 3`。这不是巧合：
    `total_steps = steps_per_epoch × epochs`，把 epochs 调大的话 cosine 曲线会被
    整体重新规划，**连已经训过的那几轮都对不上**（"接着训到更多轮"和"崩了重来"
    本来就是两件事，前者不该要求逐位等价）。上面那条测试专门盯着这个提示。
    """
    full, full_dir = run_main(
        tmp_path, "--epochs", "3", "--batch_size", "128", "--save_every_epoch",
        exp_name="e2e_full",
    )
    assert full.returncode == 0, full.stdout[-2500:]
    epoch1 = full_dir / "epoch1.pt"
    assert epoch1.exists(), "没有 --save_every_epoch 就拿不到「跑到第 1 轮就崩」的现场"

    resumed, resumed_dir = run_main(
        tmp_path, "--epochs", "3", "--batch_size", "128",
        "--resume", str(epoch1), exp_name="e2e_resumed",
    )
    assert resumed.returncode == 0, resumed.stdout[-2500:]
    assert "将从第 2 个 epoch 继续" in resumed.stdout

    uninterrupted = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    restored = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)

    assert (uninterrupted["epoch"], restored["epoch"]) == (3, 3)
    assert uninterrupted["global_step"] == restored["global_step"] > 0, (
        "global_step 没被恢复的话，lr 调度会错位 —— 权重也就跟着对不上了"
    )
    assert [h["train_loss"] for h in uninterrupted["history"]] == [
        h["train_loss"] for h in restored["history"]
    ], "loss 序列对不上，说明某一轮的随机状态没有被完整恢复"

    differing = [
        key for key in uninterrupted["model"]
        if not torch.equal(uninterrupted["model"][key], restored["model"][key])
    ]
    assert not differing, (
        f"续训与连续训练的权重不再逐位一致，涉及 {len(differing)} 个张量：{differing[:5]}"
    )


def test_resume_with_a_missing_checkpoint_fails_loudly(tmp_path):
    """路径写错必须**当场失败**，绝不能悄悄从头开始训。

    这是续训最危险的失败模式：用户以为接上了，实际白训一遍，而 summary 里的
    loss 曲线看起来一切正常 —— 等发现时已经烧掉了几个小时的 GPU 时间。
    """
    proc, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128", "--resume", "no_such_ckpt.pt",
        exp_name="e2e_resume_missing",
    )
    assert proc.returncode != 0, "续训路径不存在，进程却正常退出了"
    assert "no_such_ckpt.pt" in proc.stdout + proc.stderr
    assert not (run_dir / "summary.json").exists(), "没恢复成功却写了 summary —— 等于假装训练完成"


# ============================================================
# 可复现性：同一条命令跑两遍，参数逐位相同
# ============================================================
def parameter_digest(ckpt_path: Path) -> str:
    """把 `state_dict` 压成一个稳定摘要：key 排序 + 形状 + 原始字节。

    为什么不用 `pickle` 的哈希 / 文件哈希：**文件里还有别的字段**（时间戳之类没有，
    但 optimizer 状态、RNG 状态都在），任何一处无关变化都会让"两次不同"，
    于是这条测试就变成在测"文件是否逐字节相同"，而不是"参数是否相同"。
    只摘要参数，指向才清楚。
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        try:
            payload = tensor.numpy().tobytes()
        except TypeError:              # bfloat16 之类没有 numpy dtype
            payload = tensor.to(torch.float32).numpy().tobytes()
        digest.update(payload)
    return digest.hexdigest()


REPRO_ARGS = ("--epochs", "1", "--batch_size", "128", "--device", "cpu")


def test_same_command_produces_bit_identical_parameters(tmp_path):
    """把"靠 seed 应该能复现"升级成"被验证过能复现"。

    训练跑不出相同结果，原因通常不是"忘了 set_seed"，而是某条支路**偷偷用了另一股
    随机源**：DataLoader 每轮从全局 RNG 现取 shuffle 种子、数据增强用了 `random`
    而只 seed 了 torch、多进程 worker 各自播种……这类错误**跑得通、loss 也正常**，
    只有"同一命令跑两遍"才会暴露。

    所以判据不该是"我看代码里设了 seed"，而是一条断言。
    """
    first, dir_a = run_main(tmp_path, *REPRO_ARGS, exp_name="repro_a")
    second, dir_b = run_main(tmp_path, *REPRO_ARGS, exp_name="repro_b")
    assert first.returncode == 0, first.stdout[-2500:]
    assert second.returncode == 0, second.stdout[-2500:]

    # 先证明"确实是两次独立的运行" —— 拿同一个文件和自己比是恒真的
    assert dir_a != dir_b
    assert (dir_a / "last.pt").resolve() != (dir_b / "last.pt").resolve()

    assert parameter_digest(dir_a / "last.pt") == parameter_digest(dir_b / "last.pt"), (
        "同一条命令跑两遍，参数居然不同 —— 有一条随机支路没被 seed 管住"
    )

    # 指标层面也要一致：同一串浮点运算，应该逐位相同
    summary_a = json.loads((dir_a / "summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((dir_b / "summary.json").read_text(encoding="utf-8"))
    assert [row["train_loss"] for row in summary_a["history"]] == [
        row["train_loss"] for row in summary_b["history"]
    ]
    assert summary_a["test_acc"] == summary_b["test_acc"]


def test_changing_the_seed_changes_the_parameters(tmp_path):
    """反向断言：换个 seed，参数必须变。

    没有这一条，上面那条测试有可能是**恒真**的 —— 比如摘要函数把空字典算了进去、
    两次实际跑的是同一个目录、或者参数根本没被训练改动过。
    **先证明这个判据有分辨力，再用它下结论。**
    """
    _, base = run_main(tmp_path, *REPRO_ARGS, exp_name="seed_base")
    _, other = run_main(tmp_path, *REPRO_ARGS, "--seed", "1", exp_name="seed_other")

    assert parameter_digest(base / "last.pt") != parameter_digest(other / "last.pt"), (
        "换了 seed 参数却完全一样 —— 要么 seed 没被用上，要么这个摘要没有分辨力"
    )


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


# ============================================================
# 输出编码：一个"本地永远复现不了"的 CI-only 故障
# ============================================================
def test_chinese_log_survives_non_utf8_stdout(tmp_path):
    """中文日志在非 UTF-8 的输出编码下也必须能打印出来。

    真实故障复盘
    ------------
    Python 在 Windows 上**输出到管道**时用系统 locale 编码，英文系统 = cp1252。
    `print("训练完成")` 于是抛 `UnicodeEncodeError: 'charmap' codec ...`，
    整个训练脚本带着 exit code 1 崩掉 —— 而这种崩溃只发生在
    「管道 / 重定向 / CI」场景。本地终端是 cp936（中文 locale），
    能正常编码中文，所以**本地怎么跑都是绿的**，只有 GitHub Actions 的
    windows runner 会红。

    这里用 `PYTHONIOENCODING` 把子进程的输出编码强制成 cp1252，等价复现
    CI 环境。只要 `main.py` 开头的 `force_utf8_stdout()` 被删掉，这条测试
    立刻变红 —— 这就是它的存在意义。
    """
    proc, _ = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128",
        exp_name="e2e_encoding",
        env={"PYTHONIOENCODING": "cp1252"},
    )
    assert proc.returncode == 0, (
        "非 UTF-8 输出编码下训练崩溃了 —— 大概率是 force_utf8_stdout() 没被调用\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert "UnicodeEncodeError" not in proc.stderr
    assert "训练完成" in proc.stdout          # 中文确实完整写出来了


# ============================================================
# 分布式：真的起 2 个进程跑一遍
# ============================================================
def run_ddp(tmp_path: Path, exp_name: str, *extra: str):
    """用 `tools/ddp_launch.py` 起 2 个 CPU 进程跑完整的 DDP 训练。

    这是唯一能覆盖「进程组初始化 + 梯度 all-reduce + 指标归约 + rank0 独占落盘」
    整条链路的测试 —— 进程内的单元测试测不到真正的集合通信。

    为什么不用 `torchrun`：它需要一个 TCPStore，而某些 torch 构建
    （比如本机的 Windows 版）根本没编进 libuv，TCPStore 直接不可用，
    且 `USE_LIBUV=0` 也救不回来。本项目的启动器改用 `FileStore` 做 rendezvous，
    绕开 TCP，所以 ubuntu / windows 的 CI 都能跑。

    `*extra` 追加到 `src/main.py` 的参数后面（用于 `--epochs` / `--resume` 等）。
    """
    launcher = ROOT / "tools" / "ddp_launch.py"
    out_dir = tmp_path / "outputs"
    proc = subprocess.run(
        [
            sys.executable, str(launcher), "--nproc_per_node", "2", "--",
            str(MAIN),
            "--config", str(CONFIG),
            "--output_dir", str(out_dir),
            "--exp_name", exp_name,
            "--dataset", "synthetic",
            "--num_workers", "0",
            "--device", "cpu",
            "--epochs", "1",
            "--batch_size", "32",
            *extra,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=600,
    )
    return proc, out_dir / exp_name


def test_ddp_two_processes_end_to_end(tmp_path):
    proc, run_dir = run_ddp(tmp_path, "e2e_ddp")
    assert proc.returncode == 0, (
        f"DDP 多进程训练失败\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # 1) 确实走了分布式路径
    assert summary["distributed"]["enabled"] is True
    assert summary["distributed"]["world_size"] == 2
    assert summary["distributed"]["backend"] == "gloo"
    # 2) 全局等效 batch 必须把进程数算进去（32 × 累积1 × 2进程 = 64）
    assert summary["distributed"]["global_effective_batch_size"] == 64
    # 3) 指标是 all-reduce 后的全局值，必须是合法概率
    assert 0.0 <= summary["test_acc"] <= 1.0
    # 4) 产物齐全（只有 rank 0 写，所以日志没被两个进程写坏）
    for name in ["best.pt", "summary.json", "config_resolved.yaml", "train.log"]:
        assert (run_dir / name).exists(), f"缺少产物 {name}"
    # 5) 两个 rank 都正常退出
    assert "DDP 已启用" in proc.stdout
    assert "rank 0/2" in proc.stdout
    assert "[rank 0] 退出码 0" in proc.stdout
    assert "[rank 1] 退出码 0" in proc.stdout


def test_ddp_checkpoint_has_no_module_prefix(tmp_path):
    """DDP 存出来的 checkpoint 必须能被**单进程**代码加载（key 不能带 `module.`）。

    不 unwrap 就 `state_dict()`，key 会变成 `module.blocks.0.0.weight`，
    单进程加载时全部对不上 —— 而且保存当时毫无提示，属于"埋雷"型 bug。
    """
    proc, run_dir = run_ddp(tmp_path, "e2e_ddp_ckpt")
    assert proc.returncode == 0, proc.stdout[-3000:]

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    keys = list(ckpt["model"])
    assert keys, "checkpoint 里没有参数"
    assert not any(k.startswith("module.") for k in keys), (
        f"checkpoint 的 key 带了 DDP 前缀，单进程加载不了：{keys[:3]}"
    )

    # 真加载进一个单进程模型，确认 key 完全对得上（strict=True 由 load_state_dict 默认保证）
    from src.model import build_model

    model = build_model("small_cnn", in_channels=1, num_classes=10, image_size=28)
    model.load_state_dict(ckpt["model"])


def test_ddp_resume_reproduces_uninterrupted_training(tmp_path):
    """DDP 下「1 轮 + 断点 + 2 轮」必须和「连续 3 轮」逐位一致。

    单进程的等价性测试（`tests/test_checkpoint.py`）**覆盖不到这里**，因为多进程下
    训练集用的是 `DistributedSampler`：它的 shuffle 顺序由 `seed + epoch` 决定，
    `loader.generator` 是 `None` —— 也就是说续训要恢复的那六类状态里，
    "两股随机流"这条路在 DDP 下根本不经过，数据顺序的正确性**完全**落在
    "每个 epoch 有没有 `set_epoch(真实 epoch)`"这一件事上。

    漏掉它的症状很阴险：把续训后的 epoch 从 1 重新数，数据顺序整体错位，
    但 loss 曲线依然"很正常"，且当前进程内的单测全绿 —— 只有把两段
    loss 序列摆在一起才看得出来。

    两次都传 `--epochs 3` 是**必须的**（理由同 `test_resume_reproduces_uninterrupted_training`）：
    `total_steps = steps_per_epoch × epochs`，把 epochs 调大，cosine 会按新的总步数
    重新规划整条曲线，连已经训过的那轮都对不上。那本身是**正确语义**
    （"崩了重来"要求严格等价，"接着训更多轮"重新规划才是期望行为），
    但会让这个测试退化成"两条不同的 lr 曲线对拍" —— 实测过一次：
    那样 16/18 个张量不同，看着像 bug，其实不是。
    """
    # 参照组：一次训到 3 轮，并留下第 1 轮的存档（模拟"跑到第 1 轮就崩了"）
    proc_full, run_full = run_ddp(
        tmp_path, "ddp_full", "--epochs", "3", "--save_every_epoch"
    )
    assert proc_full.returncode == 0, proc_full.stdout[-3000:]
    epoch1 = run_full / "epoch1.pt"
    assert epoch1.exists(), "没有 --save_every_epoch 就没有可续训的现场"
    full = torch.load(run_full / "last.pt", map_location="cpu", weights_only=False)

    # 对照组：从这个存档续训到 3 轮（epochs 不变 → lr 曲线不变）
    proc_rest, run_resume = run_ddp(
        tmp_path, "ddp_resume", "--epochs", "3", "--resume", str(epoch1)
    )
    assert proc_rest.returncode == 0, proc_rest.stdout[-3000:]
    assert "从第 2 个 epoch 继续" in proc_rest.stdout, (
        "续训没被真正触发（打印里没有'从第 2 个 epoch 继续'）—— "
        "那样这个测试就变成'连续 3 轮 vs 连续 3 轮'，恒真了"
    )
    resumed = torch.load(run_resume / "last.pt", map_location="cpu", weights_only=False)

    # 1) 进度一致
    assert resumed["epoch"] == 3
    assert resumed["global_step"] == full["global_step"], (
        f"global_step 对不上：{resumed['global_step']} vs {full['global_step']}"
    )
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3]

    # 2) 权重逐位一致 —— 这才是"续训真的接上了"的证据
    differing = [
        k for k in full["model"] if not torch.equal(full["model"][k], resumed["model"][k])
    ]
    assert not differing, (
        f"{len(differing)}/{len(full['model'])} 个张量不一致：{differing[:5]}\n"
        "DDP 下最可能的原因是续训时 set_epoch() 收到的不是真实 epoch"
    )

    # 3) loss 序列逐轮一致
    full_losses = [round(h["train_loss"], 8) for h in full["history"]]
    resumed_losses = [round(h["train_loss"], 8) for h in resumed["history"]]
    assert full_losses == resumed_losses, f"{full_losses} != {resumed_losses}"
    assert [h["val_acc"] for h in full["history"]] == [h["val_acc"] for h in resumed["history"]]


# ============================================================
# 入口不变式：每个可执行入口都必须钉住 UTF-8 输出
# ============================================================
ENTRYPOINTS = [
    "src/main.py",
    "tools/ddp_launch.py",
    "tools/ddp_probe.py",
    "tools/compile_probe.py",
    "tools/lr_finder.py",
    "experiments/exp_memory_accounting.py",
    "experiments/exp_checkpoint_granularity.py",
    "experiments/exp_ddp_equivalence.py",
]


def test_every_executable_entrypoint_is_covered():
    """清单必须**完整** —— 仓库里每个带 `__main__` 块的文件都得在里面。

    为什么要有这条：只把清单写死还不够，**清单自己会漂**。
    加 `tools/compile_probe.py` 时忘了往上面加一行，于是那个入口
    「看起来被不变式守着，实际上没被扫」—— 比完全没有这条测试更危险，
    因为它会让人以为已经覆盖了。

    这条测试把「忘了加一行」变成一次明确的失败，而不是悄悄少覆盖一个入口。
    """
    declared = set(ENTRYPOINTS)
    found: set[str] = set()
    for folder in ("src", "tools", "experiments"):
        for path in sorted((ROOT / folder).glob("*.py")):
            if '__name__ == "__main__"' in path.read_text(encoding="utf-8"):
                found.add(f"{folder}/{path.name}")

    assert found == declared, (
        f"入口清单与实际不一致 —— 漏了 {sorted(found - declared)}，"
        f"或列了不存在的 {sorted(declared - found)}"
    )


@pytest.mark.parametrize("rel", ENTRYPOINTS)
def test_entrypoint_pins_utf8_stdout(rel):
    """每个可执行入口都必须在最早期调 `force_utf8_stdout()`。

    这个坑在本仓库**踩过两次**：先是 `src/main.py`（CI 的 windows runner 上
    整个训练带 exit code 1 崩掉），修完没多久，新写的 `tools/ddp_launch.py`
    又犯了同一个错 —— 换个文件、换个 print 语句而已。那次两条 DDP 冒烟测试
    一起红，还连累 CI 的排错通道（同样的 cp1252 环境把注解 hook 也搞崩了，
    现象退化成一句 "Process completed with exit code 1"）。

    两次的症状完全一样：**本地怎么跑都是绿的，只有 CI 的 windows runner 红**，
    因为本地中文 locale（cp936）能正常编码中文，而 CI 是 cp1252。
    结论是「下次记得改」靠不住，所以把不变式钉成测试，漏一个入口就红。

    这里只做源码级检查（函数名不会变，变了这条测试也会跟着红）；行为级验证
    由 `test_ddp_launcher_survives_non_utf8_stdout` 和
    `test_chinese_log_survives_non_utf8_stdout` 负责。
    """
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in text, f"{rel} 看起来不是可执行入口"
    # 断言带括号的**调用**而不是函数名 —— 只写 `from console import
    # force_utf8_stdout` 是没用的（第一版断言就被这个骗过了，摘掉调用仍然绿）
    assert "force_utf8_stdout()" in text, (
        f"{rel} 没有把 stdout 钉成 UTF-8 —— 在英文 Windows / CI 管道里，"
        f"任何一句中文 print 都会抛 UnicodeEncodeError 把进程带崩"
    )


def test_ddp_launcher_survives_non_utf8_stdout():
    """`tools/ddp_launch.py` 自己在 cp1252 管道下也不能崩。

    这是上面那条不变式的**行为级**版本，来自一次真实回归：启动器里的
    `print(f"[ddp_launch] 运行：...")` 在 CI 的 windows runner 上抛
    UnicodeEncodeError 并 exit 1，两条 DDP 测试因此一起变红。

    不给参数调用会走"用法提示"分支：打印一段中文并返回 2。
    返回码 2 是正常的参数提示，**不是**崩溃。
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "ddp_launch.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        timeout=120,
    )
    assert proc.returncode == 2, (
        f"启动器在非 UTF-8 编码下崩了\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )
    assert "UnicodeEncodeError" not in proc.stderr
    assert "用法" in proc.stdout            # 中文确实完整写出来了


# ============================================================
# torch.compile：要么真编译上，要么说清楚为什么没编译上
# ============================================================
def test_compile_with_fallback_backend_runs_training(tmp_path):
    """`--compile --compile_backend aot_eager` 必须真的编译，并且跑完整条流程。

    aot_eager 不经过 Inductor 的 kernel 生成，因此既不依赖 Triton 也不依赖
    MSVC —— 这是能**跨平台稳定断言"编译路径真的被走到了"**的那个选项。
    """
    proc, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128",
        "--compile", "--compile_backend", "aot_eager", exp_name="e2e_compile",
    )
    assert proc.returncode == 0, proc.stdout[-2500:]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    meta = summary["torch_compile"]
    assert meta["enabled"] is True
    assert meta["active"] is True
    assert meta["backend"] == "aot_eager"
    assert meta["first_call_seconds"] > 0      # 首次前向含编译，必须量出来
    assert "已编译" in proc.stdout


def test_compile_never_breaks_training(tmp_path):
    """默认后端（inductor）无论环境支不支持，训练都必须能跑完。

    这条测的是**降级**：Linux 上 inductor 通常能编译成功，Windows 上会因缺
    Triton / MSVC 而不可用 —— 但两条路都不允许把训练带崩，而且必须在日志里
    说清发生了什么（静默忽略比报错更难查，这个项目已经栽过几次）。
    """
    proc, run_dir = run_main(
        tmp_path, "--epochs", "1", "--batch_size", "128",
        "--compile", exp_name="e2e_compile_inductor",
    )
    assert proc.returncode == 0, proc.stdout[-2500:]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["torch_compile"]["enabled"] is True
    assert "torch.compile：" in proc.stdout
    if not summary["torch_compile"]["active"]:
        assert "不可用" in proc.stdout or "失败" in proc.stdout


# ============================================================
# 学习率 finder：CLI 真的能端到端跑通
# ============================================================
def run_lr_finder(tmp_path: Path, *extra: str, env: dict | None = None):
    """跑一次 `tools/lr_finder.py`，产物丢进 tmp_path。"""
    out_dir = tmp_path / "lr"
    cmd = [
        sys.executable, str(ROOT / "tools" / "lr_finder.py"),
        "--dataset", "synthetic",        # 离线，不下载
        "--steps", "6",
        "--output_dir", str(out_dir),
        *extra,
    ]
    return (
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env or {})},
        ),
        out_dir,
    )


def test_lr_finder_cli_runs_end_to_end(tmp_path):
    """真跑一次 CLI，工件必须落盘、结论必须在日志里。

    为什么值得专门跑：这是用户**第一个会用到的入口**（"lr 该给多少"），
    也是重构时最容易悄悄弄坏的那个 —— 它一条链跨了数据、模型、优化器三块。
    合成数据 + 6 步，几秒钟，不下载、不进 GPU。
    """
    proc, out_dir = run_lr_finder(tmp_path, "--batch_size", "64")
    assert proc.returncode == 0, proc.stdout[-2500:] + proc.stderr[-2500:]

    payload = json.loads((out_dir / "lr_sweep.json").read_text(encoding="utf-8"))
    assert payload["steps"] == 6
    assert 0 < len(payload["result"]["points"]) <= 6
    assert payload["result"]["suggested_lr"] > 0
    assert "建议 lr" in proc.stdout
    assert (out_dir / "lr_sweep.md").exists()


def test_lr_finder_cli_survives_non_utf8_stdout(tmp_path):
    """和其它入口一样：cp1252 管道下中文日志不能把工具带崩。

    （这个项目在这个坑上栽过两次，所以每个新入口都补一条。）
    """
    proc, _ = run_lr_finder(
        tmp_path, "--steps", "4", env={"PYTHONIOENCODING": "cp1252"}
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "UnicodeEncodeError" not in proc.stderr
    assert "建议 lr" in proc.stdout
