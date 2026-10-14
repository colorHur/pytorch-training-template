# 第三节：算对一次更新，再谈训练效果

[上一节](02-configuration-and-cli.md) · [目录](README.md) · [下一节](04-precision-memory-and-performance.md)

## 0. 先修、阅读路径与闭卷检查

先修：链式法则、均值与加权均值、第一节训练循环。第一遍专注窗口分母和数值参照；第二遍加 optimizer/裁剪/scheduler；进阶再审 LR finder 状态与 token loss。若梯度推导不清，先做第 10 段交叉熵实验。

迁移任务：把 criterion 从 mean 改为 sum，说明哪些地方不能再乘 batch_samples；进一步设计 padding token 的有效分母，但不要求立刻训练 LLM。闭卷答 [Q09–Q12](INTERVIEW.md#lesson-3)。

第一晚推导累积梯度并做数值对照，第二晚读优化器、调度器与 LR finder。目标是能解释一次更新的分子、分母和时钟，判断“等效 batch”的前提，读懂为什么一次参数更新并不等于一次前向。

## 1. 三种 step 必须分清

micro-step 是一个 DataLoader batch 的前向/反向；update window 是累计若干 micro-step 后尝试更新参数；epoch 是遍历一次训练 loader。当前 `global_step` 每个更新窗口加一。AMP 若因非有限梯度跳过 optimizer step，这个计数仍增加，所以它表示尝试窗口数，不保证参数真的变化。

完整窗口的单进程名义 batch 是 `micro_batch×accum`，多进程名义全局 batch 再乘 world_size。末批不足、尾窗口不足、训练采样器丢样本时必须按实际样本计数。不要从日志显示的 effective_batch_size 自动推断所有更新都使用相同样本数。

## 2. 正确梯度的分母是什么

令样本 i 的损失是 l_i，窗口含 N 个样本，目标平均损失为 `L=(Σ_i l_i)/N`，梯度是 `Σ_i ∇l_i/N`。如果第 j 个 microbatch 含 n_j 个样本、平均损失为 L_j，则窗口目标是 `Σ_j n_j L_j / Σ_j n_j`。

三个等大批次、accum=4 时，固定 `loss/4` 得到 `(g1+g2+g3)/4`，正确值应除以 3，更新幅度只有目标的 75%。三批大小 2、2、1 时，正确值为 `(2g1+2g2+g3)/5`，简单除以 3 也错。数 optimizer.step 调用次数看不出这种偏差。

当前 [train.py](../../src/train.py) 在每个 micro-step 把均值 loss 乘样本数，反向累计样本梯度和；窗口末尾再统一归一化。它假设 criterion 返回按样本平均的标量。若引入 token loss、ignore_index、类别权重或多任务 loss，正确分母可能是有效 token 数、权重和等，不能照搬 `y.size(0)`。

## 3. 当前源码的执行顺序

| 阶段 | 代码入口 | 状态变化及原因 |
|---|---|---|
| 窗口开始 | `zero_grad(set_to_none=True)` | 清上个窗口梯度 |
| 每批 | `window_samples += batch_samples` | 记录实际分母 |
| 每批 | no_sync 内前向、loss、反向 | 累积样本梯度和，中间 DDP 步暂不通信 |
| 窗口末尾 | AMP 的 `unscale_` | 恢复梯度尺度并做非有限值检查 |
| 窗口末尾 | 参数 grad 乘 grad_scale | 归一化为目标平均梯度 |
| 窗口末尾 | 可选裁剪、设 lr、step | 用本次最终梯度更新 |
| 更新后 | scaler.update/global_step | 调整缩放状态，计数进入下一窗口 |

`model.train()` 在每个 epoch 开头执行，因为之前的 evaluate 会切 eval。`is_update_step` 包含“累积满”与“最后一批”，避免尾梯度丢失。平均 loss 统计也按样本加权；训练平均 loss 用的是模型沿训练轨迹产生的各批 loss，不是训练结束后对整个训练集重新测的 loss。

## 4. 完整可运行的数值对照

保存为 `outputs/course/accum_probe.py`，从根目录用 `python -m outputs.course.accum_probe` 运行。不要用带 BN/Dropout 的网络开始等价实验，否则难以区分梯度归一化错误与前向差异。

```python
# 可运行实验：7 个样本、4 个不等长 microbatch 与真实大 batch
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import TrainConfig
from src.train import train_one_epoch

torch.manual_seed(7)
x = torch.randn(7, 3)
y = torch.arange(7) % 2
base = nn.Linear(3, 2)
accumulated = copy.deepcopy(base)
reference = copy.deepcopy(base)
criterion = nn.CrossEntropyLoss()
actual_opt = torch.optim.SGD(accumulated.parameters(), lr=0.1)
reference_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
_, updates = train_one_epoch(
    accumulated, loader, actual_opt, criterion, 'cpu',
    epoch=1, cfg=TrainConfig(grad_accum_steps=4, amp=False), global_step=0,
)
reference_opt.zero_grad(set_to_none=True)
criterion(reference(x), y).backward()
reference_opt.step()
assert updates == 1
for a, b in zip(accumulated.parameters(), reference.parameters()):
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
print('参数更新一致；批次大小 2、2、2、1，窗口分母为 7')
```

同初始化、相同样本顺序、相同优化器、一次更新，无状态归一化、无随机前向，是这个判据的关键。有限精度求和顺序不同，所以使用严格但非零的浮点容差；这里没有宣称所有梯度逐位相同。

运行 `python -m pytest tests/test_train.py -q`，读 `test_accumulation_matches_real_batch_even_with_short_last_batch`。它还覆盖多个窗口。真实大 batch 参照需要按相同窗口逐次更新，不能用整个数据集一次更新去对比两次优化器更新。

## 5. 梯度裁剪与优化器

全局 L2 裁剪先计算所有梯度的联合范数 G，若 G>阈值 T，近似统一乘 T/G；它控制梯度幅度，不直接约束 loss、权重范数或最终参数步长。必须在 unscale 和样本归一化后执行，否则阈值随 AMP scale 或窗口样本数改变。

SGD 使用梯度与可选动量；Adam 类维护一阶/二阶统计，更新不等于 `lr×原始梯度`。AdamW 的 weight decay 从自适应梯度项解耦，但不能说它让学习率不重要。若比较 optimizer 类型，改变 optimizer 本身就改变了算法，应固定其他变量并重新调合理超参。

“累积等于大 batch”也有边界：BN 用各 microbatch 统计，Dropout 的采样顺序可能不同，数据增广与求和精度也会影响结果。当前名义 batch 只是数量关系；精确等价需要单独实验。SGD+momentum、AdamW 的状态都在一次 step 后更新，不能每个 microbatch 提前 step 再声称是同一算法。

## 6. 调度器的时钟与端点

[optimizer_steps_per_epoch](../../src/train.py) 对 batch 数和 accum 向上取整。7 批、accum=3，实际更新 3 次；整除只报 2。少算会让 cosine 总预算与实际执行不同。`build_lr_scheduler()` 捕获构造时各参数组 base_lrs，返回 `f(step)→各组学习率`，循环在更新前写回 optimizer。

warmup 的第一个 step 用 `base_lr/warmup_steps`，到 warmup 的最后一步达到 base。cosine 使用当前更新索引：总共 T 次更新时实际索引为 0 到 T-1，理论函数的 f(T)=0 不表示最后一次实际更新恰好使用 0。读测试 `test_lr_schedule_is_applied_on_update_steps`，不要猜端点。

```python
# 可运行实验：短调度序列与尾窗口更新次数
import torch
from src.train import build_lr_scheduler, optimizer_steps_per_epoch

opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
schedule = build_lr_scheduler(opt, 'cosine', total_steps=6, warmup_steps=2)
for step in range(7):
    print(step, schedule(step)[0], '实际更新' if step < 6 else '理论端点')
assert schedule(0)[0] == 0.05
assert schedule(1)[0] == 0.1
assert optimizer_steps_per_epoch(7, 3) == 3
```

## 7. LR finder 的算法与真实边界

[lr_finder.py](../../src/lr_finder.py) 用几何序列从 min_lr 扫到 max_lr，横轴是 log(lr)。每个点做一次优化器更新，记录 loss；EMA 平滑用 `avg/(1-beta^(t+1))` 修正零初值造成的起点偏低，再在最低点之前寻找较陡下降段。建议值是初始化附近的搜索提示，不是精确最优参数，也不是训练中途的调度替代品。

```bash
python tools/lr_finder.py --dataset synthetic --steps 20
python -m pytest tests/test_lr_finder.py -q
```

读 `resolve_sweep_config()`：当前扫描关闭 AMP、累积、裁剪、scheduler 和 compile，是为了减少干扰并观察发散；这不是所有 LR 搜索方法的普遍规则。optimizer 类型、weight_decay 保持与训练一致。

扫描前 deepcopy state_dict，finally 恢复权重、BN buffer 和根模型训练模式。**当前没有承诺恢复所有 RNG、参数 grad、loader 已消费状态或训练 optimizer 状态**；它使用自己的新 optimizer。不能把“模型 state_dict 还原”扩大成“程序状态零污染”。数据不足时 `itertools.cycle(loader)` 会缓存已经产出的 batch 再循环，它不是每轮重新打乱并增广的 DataLoader，缓存也有内存成本。

## 8. 反证、自测与验收

在独立副本把 grad_scale 暂改为 `1/cfg.grad_accum_steps`，数值对照应失败。恢复后改 optimizer_steps_per_epoch 为整除，尾窗口计数测试应失败。每次只改一处、恢复哈希与相关测试，再做下一项。

自测答案：① 3 等大批、accum=4 的正确分母是实际 3 批；当前实现按样本更一般。② 2/2/1 批要按样本权重 2/2/1。③裁剪在归一化后，阈值才与最终梯度对应。④ f(T)=0 与最后一次用 f(T-1)不矛盾。⑤ LR sweep 恢复 model state_dict 不代表 RNG 和全部训练状态都恢复。⑥平均训练 loss 与最终模型训练集 loss 不同。

验收：独立画更新顺序、完成累积参数对照、列出短学习率序列，并提出一个 LR finder 状态恢复的边界。用“现象→最小输入→错误公式→修复→证据”讲 3 分钟。

参考：[PyTorch optim](https://docs.pytorch.org/docs/stable/optim.html)、[AMP 梯度累积示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)、[Cyclical Learning Rates 论文](https://arxiv.org/abs/1506.01186)。

## 9. 用数字理解裁剪与两类优化器状态

假设最终平均梯度是向量 (3,4)，L2 范数为 5，阈值为 1，裁剪结果近似为 (0.6,0.8)，方向保持、整体缩小。若先累加十个样本，再在还没除样本数时裁剪到 1，之后除 10，得到的梯度范数只剩 0.1；这说明归一化与裁剪不可以随意交换。

SGD 无动量时 `θ_(t+1)=θ_t-lr*g_t`，一阶动量引入过去梯度状态。Adam 维护指数滑动一阶、二阶统计并做偏差修正，用历史尺度影响本次更新。恢复相同权重但不同统计量，会产生不同下一步。这里不要求背每个实现细节，但要能说明优化器状态为何属于训练轨迹。

AdamW 的衰减近似是额外按当前参数做缩小；Adam 的梯度归一化与 decoupled decay 是不同机制。测试累积等价时先使用普通 SGD、无 weight_decay，减少干扰；之后可用同状态 AdamW 对照，而不能拿两份不同历史 optimizer 做参照。

再手算一个多窗口例子：9 个样本、batch2、accum3，批次为 2/2/2/2/1。第一次窗口 6 个样本，第二次 3 个，更新两次。第二窗口用第一次更新后的参数计算。若把参照写成一个 9 样本大 batch 一次更新，目标算法已经不同。

还有一个常见误读：训练 loop 中 loss 先乘 batch_samples 用于求导，但写日志仍使用未乘样本数的平均 loss，并按样本数累计统计。用于 backward 的张量与用于报告的数值可以不同；看清变量用途，不能因为有 `loss_sum` 就直接在日志里当平均 loss。

源码精读顺序：先标 `is_update_step` 与 zero_grad 的窗口边界，再追 loss→loss_sum→grad_scale，最后追 scheduler(global_step) 与 step。打印窗口样本数和更新前学习率建立短轨迹；正式训练移除临时 debug 打印，避免 CUDA 每批同步影响性能。

## 10. 把交叉熵公式接到 autograd

单样本标签 y、logits z：`L=log(Σ_j exp(z_j))-z_y`，对 z_k 求导得到 `p_k-1[k=y]`。普通 batch mean 再除 B；这是没有类别权重、ignore_index 或概率标签的基础条件。稳定实现用 log-sum-exp 技巧，不能用大数指数手算代码替代库中的稳定 loss。

```python
# 可运行实验：cross_entropy 的 logits 梯度
import torch
from torch.nn import functional as F

logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=torch.float64, requires_grad=True)
labels = torch.tensor([0, 2])
F.cross_entropy(logits, labels, reduction='mean').backward()
expected = logits.detach().softmax(dim=1)
expected[torch.arange(2), labels] -= 1
expected /= 2
torch.testing.assert_close(logits.grad, expected)
print('解析梯度与 autograd 一致:', logits.grad)
```

预测题：改成 reduction=sum，其他输入不变，梯度应是上面结果的两倍。若“模型输出先 softmax 再 CrossEntropyLoss”，库会把概率当新的 logits，计算已不是同一个目标。理解这一步后再谈损失下降、梯度尺度和 accum 分母。
