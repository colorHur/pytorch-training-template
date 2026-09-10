"""断点续训的测试。

这个文件的结构和别的测试不太一样：除了"存进去能读回来"这类**往返**测试，
每组关键状态都配了一条**反证** —— 故意不恢复它，断言结果真的变了。
理由是"存了但没人读"恰恰是这个模块出现之前的真实状态：一眼看不出任何问题。

一个真实的起点
------------
改造之前，`save_checkpoint` 存了 `model / optimizer / epoch / global_step /
config / metrics` 六样，README 写着"保存完整训练状态" —— 但全仓库唯一的
`torch.load` 只是训练结束后把 `best.pt` 读回来在测试集上评一次分，
**没有任何一处能把训练接下去**。既然没有加载方，多存的那几样就永远不会
被检查是否存对了。所以这里既验证实现，也把需求钉下来。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.checkpoint import (
    ResumeState,
    capture_rng,
    diff_configs,
    load_checkpoint,
    resolve_resume_path,
    restore_rng,
    save_checkpoint,
)
from src.data import IndexedDataset, SyntheticDataset, build_dataloaders
from src.model import build_model

IMAGE_SIZE = 28


# ============================================================
# 脚手架
# ============================================================
def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return build_model("small_cnn", in_channels=1, num_classes=10, image_size=IMAGE_SIZE)


def make_optimizer(model: nn.Module):
    # 优化器类型/weight_decay 必须和 `src/train.build_optimizer` 保持一致 ——
    # 被测的"矩"是 Adam 的一阶/二阶矩，换成 SGD 就测不到要测的东西了
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


def make_batch(n: int = 8, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g),
        torch.randint(0, 10, (n,), generator=g),
    )


def make_loader(n: int = 64, batch_size: int = 8, seed: int = 0):
    """和 `src/data.build_dataloaders` 里训练集 DataLoader 的构造方式**一致**：
    显式 generator + shuffle=True。脚手架不一致会给出假的结论。
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        IndexedDataset(SyntheticDataset(n=n, train=True)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def epoch_order(loader) -> list[int]:
    """走一遍 loader，取出这一轮**实际喂给模型的样本下标顺序**。

    会推进 loader 的 shuffle generator —— 这既是观察手段，也是被测行为本身。
    """
    return [int(i) for _, _, idx in loader for i in idx]


def one_step(model, optimizer, x, y) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def save_at(tmp_path: Path, name: str, model, optimizer, **kwargs) -> Path:
    """存一份 checkpoint 并返回路径（省掉每个用例里重复的三行）。"""
    path = tmp_path / name
    save_checkpoint(
        path, state=kwargs.pop("state", ResumeState(epoch=1)), model=model, optimizer=optimizer, **kwargs
    )
    return path


def read(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


class FakeScaler:
    """只要实现了 `state_dict` / `load_state_dict`，对 checkpoint 来说就和
    `torch.amp.GradScaler` 没有区别。

    这里刻意不用真的 GradScaler：CI 上没有 GPU，而它是按设备构造的 ——
    测 checkpoint 的搬运逻辑没必要把它拖进来。
    """

    def __init__(self, scale: float = 65536.0) -> None:
        self.scale = scale

    def state_dict(self) -> dict:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict) -> None:
        self.scale = state["scale"]


# ============================================================
# ResumeState
# ============================================================
def test_next_epoch_is_one_past_the_last_finished_epoch():
    """`epoch` 记的是**已训完**的轮数，不是"下一轮"。

    边界搞错会重训一轮或者跳一轮 —— 而跳一轮是静默的，
    唯一的症状是 loss 曲线上少了一段。
    """
    assert ResumeState().next_epoch == 1                 # 全新训练从第 1 轮开始
    assert ResumeState(epoch=3).next_epoch == 4


def test_describe_says_where_it_will_continue():
    text = ResumeState(epoch=3, global_step=120).describe()
    assert "3" in text and "120" in text and "第 4 个 epoch" in text


def test_describe_flags_missing_state():
    """缺状态必须出现在人话里，不能只是数据结构里的一个字段。"""
    assert "⚠️" not in ResumeState(epoch=1).describe()
    text = ResumeState(epoch=1, missing=["best_val_acc"]).describe()
    assert "best_val_acc" in text and "⚠️" in text


# ============================================================
# 两股随机流
# ============================================================
def test_capture_rng_covers_the_global_streams():
    state = capture_rng()
    assert {"torch", "python", "numpy"} <= set(state)
    assert "loader" not in state          # 没传 loader 就不该有这一项


def test_capture_rng_includes_the_loader_generator_only_when_present():
    """loader 那一项是"有才存"：多进程用 DistributedSampler 时 DataLoader
    根本没有 generator，硬塞一个默认值反而会让 restore 去动不该动的东西。
    """
    assert "loader" in capture_rng(make_loader())
    plain = DataLoader(SyntheticDataset(n=16, train=True), batch_size=4, shuffle=False)
    assert "loader" not in capture_rng(plain)


def test_shuffle_order_advances_each_epoch_and_can_be_rewound():
    """一条测试证两件事：

    ① generator 每轮都在前进 → **只存种子是不够的**（种子还原出来的永远是
       第 1 轮的顺序）；这正是它必须随 checkpoint 一起存的原因。
    ② 存状态 + 还原，能把"下一轮"精确复现 —— 这是续训和连续训练逐位等价的
       前提之一。
    """
    loader = make_loader()
    first = epoch_order(loader)
    snapshot = capture_rng(loader)
    second = epoch_order(loader)

    assert first != second, "第二轮的顺序和第一轮一样，说明 generator 压根没被推进"

    restore_rng(snapshot, loader)
    assert epoch_order(loader) == second

    # 反证：不还原的话拿到的是第三轮，和第二次不一样
    third = epoch_order(loader)
    assert third != second and third != first


def test_restore_rng_tolerates_a_legacy_state_dict():
    """老 checkpoint 只有 torch 一项（甚至一项都没有）。缺哪项跳过哪项，
    不能 KeyError —— 那会让"读一个旧权重看看"变成不可能。
    """
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    snapshot = {"torch": torch.get_rng_state()}
    torch.randn(10)                       # 把状态推走
    restore_rng(snapshot)
    assert torch.equal(torch.randn(4), expected)

    torch.randn(10)
    restore_rng({})                       # 空 dict 也不能抛


def test_train_dataloader_has_an_explicit_shuffle_generator():
    """单进程训练的 shuffle 必须由 loader 自己的 generator 决定。

    这是"能续训"的前提：随机流如果是从**全局** RNG 现取种子建出来的，
    那么 Dropout 多抽一次都会改变下一轮的数据顺序，两件事就没法分开存、
    分开还原。见 `src/data.py` 里 `build_dataloaders` 的注释。
    """
    loaders = build_dataloaders(
        "synthetic", batch_size=32, val_ratio=0.2, num_workers=0, pin_memory=False
    )
    assert loaders["train"].generator is not None
    assert loaders["val"].generator is None      # 验证集不打乱，不该有


# ============================================================
# 往返：存进去的每一样都得原样回来
# ============================================================
def test_roundtrip_restores_the_model_bit_for_bit(tmp_path):
    model = make_model()
    optimizer = make_optimizer(model)
    x, y = make_batch()
    torch.manual_seed(0)
    one_step(model, optimizer, x, y)

    path = save_at(
        tmp_path, "ckpt.pt", model, optimizer,
        state=ResumeState(epoch=1, global_step=1),
    )

    fresh = make_model(seed=999)          # 另一个 seed → 权重肯定不同
    state = load_checkpoint(path, model=fresh)
    for key, value in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), f"{key} 没被逐位恢复"
    assert (state.epoch, state.global_step) == (1, 1)


def test_roundtrip_restores_optimizer_state(tmp_path):
    """Adam 的一阶/二阶矩和 step 计数必须一起回来。

    `step` 尤其关键：Adam 的偏差修正用的就是它，丢了等于换了一套更新规则。
    """
    model = make_model()
    optimizer = make_optimizer(model)
    x, y = make_batch()
    torch.manual_seed(0)
    for _ in range(3):
        one_step(model, optimizer, x, y)

    path = save_at(tmp_path, "ckpt.pt", model, optimizer, state=ResumeState(epoch=1, global_step=3))

    reloaded = make_optimizer(make_model(seed=1))
    load_checkpoint(path, optimizer=reloaded)

    before, after = optimizer.state_dict(), reloaded.state_dict()
    assert before["param_groups"] == after["param_groups"]
    assert len(before["state"]) == len(after["state"]) > 0
    for key, saved in before["state"].items():
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(saved[field], after["state"][key][field]), f"{key}.{field}"


def test_roundtrip_restores_the_scaler(tmp_path):
    model = make_model()
    path = save_at(
        tmp_path, "ckpt.pt", model, make_optimizer(model), scaler=FakeScaler(scale=1024.0)
    )

    restored = FakeScaler(scale=65536.0)     # 初值
    load_checkpoint(path, scaler=restored)
    assert restored.scale == 1024.0


def test_save_without_a_scaler_does_not_write_the_key(tmp_path):
    """不开 AMP 就不该凭空多出一个 scaler 键 —— 否则加载时会去覆盖一个
    根本不存在的对象。"""
    model = make_model()
    path = save_at(tmp_path, "ckpt.pt", model, make_optimizer(model))
    assert "scaler" not in read(path)


def test_checkpoint_records_the_format_version(tmp_path):
    """有版本号才谈得上"向前兼容"：将来加字段时能一眼看出这份是老格式。"""
    model = make_model()
    path = save_at(tmp_path, "ckpt.pt", model, make_optimizer(model))
    assert read(path)["format_version"] == 2


def test_save_checkpoint_creates_missing_parent_directories(tmp_path):
    """`outputs/exp/` 可能还不存在；这里抛异常等于在训练快结束时才崩。"""
    model = make_model()
    path = tmp_path / "deep" / "nested" / "ckpt.pt"
    save_checkpoint(path, state=ResumeState(), model=model, optimizer=make_optimizer(model))
    assert path.exists()


def test_load_checkpoint_reports_the_path_it_could_not_find(tmp_path):
    """路径写错要报清楚是哪个路径 —— 只有文件名的话得猜半天。"""
    with pytest.raises(FileNotFoundError, match="找不到"):
        load_checkpoint(tmp_path / "nope.pt")


# ============================================================
# 包装器：存之前必须剥掉前缀
# ============================================================
class _AttrWrapper(nn.Module):
    """最小化的 DDP / DataParallel / compile 替身：同样的 `state_dict()` 前缀问题。"""

    def __init__(self, inner: nn.Module, attr: str) -> None:
        super().__init__()
        setattr(self, attr, inner)


def test_save_checkpoint_strips_wrapper_prefixes(tmp_path):
    """DDP（`.module`）与 compile（`._orig_mod`）都必须被剥掉，且要支持嵌套剥离。

    不剥的后果不是"存不下来"，而是**存下来一个别人加载不了的权重** ——
    key 变成 `module.blocks.0.0.weight`，单进程 `load_state_dict` 全对不上，
    而保存那一刻毫无提示。本仓库为此真的栽过一次。
    """
    inner = make_model()
    nested = _AttrWrapper(_AttrWrapper(inner, "module"), "_orig_mod")

    path = save_at(tmp_path, "ckpt.pt", nested, make_optimizer(inner))

    keys = list(read(path)["model"])
    assert keys, "checkpoint 里没有参数"
    assert not any(k.startswith(("module.", "_orig_mod.")) for k in keys), keys[:3]
    # 真加载进一个裸模型：strict=True 由 load_state_dict 默认保证，key 对不上会直接抛
    make_model(seed=5).load_state_dict(read(path)["model"])


# ============================================================
# 反证：不恢复这些状态，结果真的会变
# ============================================================
def test_optimizer_state_is_necessary_for_exact_resume(tmp_path):
    """反证：**不**恢复 optimizer 状态，续训后的第一步参数就不一样。

    这就是 Adam 矩被重置的代价 —— 它不报错、也不影响"能不能跑"，
    只让 loss 先尖峰再回落（很多人会把它归因于"断点处抖一下很正常"）。
    """
    model = make_model()
    optimizer = make_optimizer(model)
    x, y = make_batch()
    torch.manual_seed(0)
    for _ in range(3):
        one_step(model, optimizer, x, y)

    path = save_at(tmp_path, "ckpt.pt", model, optimizer, state=ResumeState(epoch=1, global_step=3))
    weights = {k: v.clone() for k, v in model.state_dict().items()}

    def step_after_resume(with_optimizer_state: bool) -> dict:
        resumed = make_model(seed=999)
        resumed.load_state_dict(weights)
        opt = make_optimizer(resumed)
        if with_optimizer_state:
            load_checkpoint(path, optimizer=opt)
        torch.manual_seed(123)       # 让 Dropout 的 mask 两边一致，只留优化器这一个变量
        one_step(resumed, opt, x, y)
        return {k: v.clone() for k, v in resumed.state_dict().items()}

    with_state = step_after_resume(True)
    without_state = step_after_resume(False)
    changed = [k for k in with_state if not torch.equal(with_state[k], without_state[k])]
    assert changed, "恢复 optimizer 状态与否，下一步的参数居然完全一样 —— 那存它就没有意义"


def test_load_checkpoint_restores_the_global_random_stream(tmp_path):
    """`load_checkpoint` 必须真的把全局 RNG 还原进当前进程。

    ⚠️ 这条测试的**强度**是刻意设计过的，第一版写砸了，记录下来：
    最初写的是「不恢复 RNG → 下一步 loss 和恢复时不同」，结果把
    `load_checkpoint` 里的 `restore_rng(...)` 整行摘掉，测试**依然是绿的** ——
    因为那版是**恒真**的：两次不同的随机状态当然给出不同的结果。它证明的是
    "随机性有影响"，而不是"`load_checkpoint` 还原了它"。

    修法是直接对着机制断言：存档那一刻之后"下一次抽样应该得到什么"是确定的，
    还原后抽出来的必须**逐值相等**。
    """
    model, optimizer = make_model(), make_optimizer(make_model(seed=1))

    torch.manual_seed(0)
    expected = torch.randn(4)
    torch.manual_seed(0)                      # 回到存档时的那一刻
    path = save_at(tmp_path, "ckpt.pt", model, optimizer)

    torch.manual_seed(1234)                   # 把状态推走
    assert not torch.equal(torch.randn(4), expected), "推没推动？那这条测试的下半段就没意义了"

    load_checkpoint(path)
    assert torch.equal(torch.randn(4), expected), "全局 RNG 没有被还原成存档时的状态"


def test_load_checkpoint_restores_the_shuffle_stream(tmp_path):
    """同上，但针对 `train_loader.generator` 那一股流。

    它决定了"下一个 epoch 的数据以什么顺序喂进来"。注意这里断言的是
    **存档之后第一次迭代**的排列，不是"两次不同"—— 后者随便换个 loader 就能满足。

    ⚠️ 顺序很重要，第一版就是在这里写错的：存档必须发生在"记下预期"**之前**。
    存档那一刻的状态对应的是"第 2 轮"，所以 `expected` 得取第 2 轮；
    先算 `expected` 再存档的话，还原出来的是第 3 轮，永远对不上。
    """
    loader = make_loader()
    epoch_order(loader)                       # 第 1 轮：让 generator 离开初值

    path = save_at(                            # 存档 —— 此刻"下一轮"就是第 2 轮
        tmp_path, "ckpt.pt",
        make_model(), make_optimizer(make_model(seed=1)), loader=loader,
    )
    expected = epoch_order(loader)             # 第 2 轮 = 存档后应该复现的顺序

    assert epoch_order(loader) != expected, "第 3 轮和第 2 轮一样？那 generator 没在走"

    load_checkpoint(path, loader=loader)
    assert epoch_order(loader) == expected, "shuffle 流没有被还原 —— 续训的数据顺序会跑偏"


# ============================================================
# 向后兼容：老 checkpoint 能读，但缺什么要说出来
# ============================================================
def test_legacy_checkpoint_loads_and_reports_what_is_missing(tmp_path):
    """老格式（改造前那份）只有六个键。**能加载就别报错** ——
    但也不能装作没这回事：`best_val_acc` 被默认成 0.0 之后，`best.pt` 会被
    任何一个 epoch 覆盖；`patience_counter` 归零之后早停永不触发。
    这两条都是静默的逻辑错误，只能靠 `missing` 提示出来。
    """
    model = make_model()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": make_optimizer(model).state_dict(),
            "epoch": 3,
            "global_step": 12,
            "config": {"lr": 1e-3},
            "metrics": {"acc": 0.9},
        },
        path,
    )

    state = load_checkpoint(path)
    assert (state.epoch, state.global_step) == (3, 12)
    assert state.next_epoch == 4
    assert state.config == {"lr": 1e-3}                # 有的就正常用
    assert set(state.missing) == {"best_val_acc", "patience_counter", "history", "rng"}


def test_legacy_checkpoint_does_not_invent_a_scaler_requirement(tmp_path):
    """只在调用方**真的传了** scaler 时，才把缺失的 scaler 算进 missing。"""
    model = make_model()
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "epoch": 1}, path)
    assert "scaler" not in load_checkpoint(path).missing
    assert "scaler" in load_checkpoint(path, scaler=FakeScaler()).missing


# ============================================================
# --resume 的值怎么解析
# ============================================================
def test_resolve_resume_path_keywords_point_into_the_run_dir():
    assert resolve_resume_path("last", "outputs/mnist_cnn") == Path("outputs/mnist_cnn/last.pt")
    assert resolve_resume_path("best", "outputs/mnist_cnn") == Path("outputs/mnist_cnn/best.pt")


def test_resolve_resume_path_keeps_absolute_paths_and_resolves_relative_ones(tmp_path):
    # ⚠️ 别写 "/tmp/..." 这种硬编码绝对路径：Windows 上它**不是**绝对路径
    #    （没有盘符），`is_absolute()` 返回 False —— 这条测试就变成平台相关的了。
    #    `tmp_path` 是 pytest 给的、两个平台上都绝对是绝对路径的目录。
    absolute = tmp_path / "exp" / "epoch3.pt"
    assert resolve_resume_path(str(absolute), "outputs/x") == absolute

    # 相对路径按调用方给的 cwd 解析（main.py 传项目根），
    # 不依赖测试进程的 cwd —— 否则换个目录跑测试结论就飘了
    assert resolve_resume_path("outputs/x/epoch3.pt", "outputs/x", cwd=tmp_path) == (
        tmp_path / "outputs/x/epoch3.pt"
    )

    # 不给 cwd 时退化成"相对当前工作目录"，结果必须是绝对路径 —— 否则日志里
    # 打出来的路径取决于当时在哪个目录，事后根本没法定位
    assert resolve_resume_path("epoch3.pt", "outputs/x").is_absolute()


def test_resolve_resume_path_strips_whitespace():
    assert resolve_resume_path("  last  ", "outputs/x") == Path("outputs/x/last.pt")


# ============================================================
# 续训时偷偷改了超参，要说出来
# ============================================================
def test_diff_configs_is_empty_when_nothing_important_changed():
    cfg = {"lr": 1e-3, "batch_size": 128, "epochs": 3, "exp_name": "a"}
    # epochs / exp_name / output_dir 本来就会变，不算异常
    assert diff_configs(cfg, {**cfg, "epochs": 9, "exp_name": "b"}) == []


def test_diff_configs_reports_the_learning_rate_switch():
    """最危险的一种：恢复 optimizer 状态之后，lr 调度器（它闭包住了 base_lr）
    会用新配置的 lr 覆盖回去 —— "接着训"实际变成了"换学习率重新调度"，
    而 loss 曲线上看不出任何断点。
    """
    differences = diff_configs({"lr": 1e-3, "batch_size": 128}, {"lr": 1e-4, "batch_size": 128})
    assert differences == [("lr", 1e-3, 1e-4)]


def test_diff_configs_handles_keys_missing_from_the_old_checkpoint():
    differences = {key: (old, new) for key, old, new in diff_configs({}, {"lr": 1e-3})}
    assert differences["lr"] == ("<未记录>", 1e-3)
