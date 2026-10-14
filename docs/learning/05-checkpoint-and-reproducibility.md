# 第五节：把训练接回原来的轨迹

[上一节](04-precision-memory-and-performance.md) · [目录](README.md) · [下一节](06-distributed-training.md)

## 0. 先修、阅读路径与闭卷检查

先修：state_dict、optimizer state、Generator 与训练进度。第一遍做 seed/state 和 MLP 配对；第二遍追创建/加载/迭代顺序；进阶讨论原子保存、worker 与 micro-step 恢复，区分已实现和设计方案。

迁移任务：漏掉一种状态后选一个最早可观察差异，而非只看最终 accuracy。闭卷答 [Q17–Q20](INTERVIEW.md#lesson-5)，明确自己验证的恢复粒度与环境。

第一晚理解存档中的状态并跑恢复实验；第二晚做连续/恢复配对和反证。能加载权重，只证明模型状态能读取。续训还要接上 optimizer、调度进度、随机流和模型选择状态。先问：Adam 第一、二阶统计为什么要存？只恢复 seed 足够吗？在第 3 个 microbatch 中断能直接继续吗？

## 1. 把训练看成状态机

一次训练状态至少包括参数、buffer、optimizer 状态、scaler、进度、数据顺序及随机源、验证最佳记录和历史。更新是把状态 S_t 变成 S_(t+1)。只恢复参数，相当于恢复了部分状态；之后的随机样本和更新方向可能已经改变。

| 类别 | 当前 checkpoint 键 | 不恢复会怎样 |
|---|---|---|
| 模型 | `model` | 参数和 BN buffer 不一致 |
| 优化器 | `optimizer` | Adam/动量丢失，更新公式不再接续 |
| 更新进度 | `epoch`、`global_step` | sampler epoch、lr 时钟错位 |
| 选模状态 | `best_val_acc`、`patience_counter` | 最佳模型选择或早停重新开始 |
| 实验记录 | `history`、`config`、`metrics` | 无法完整说明已跑部分 |
| 随机状态 | `rng` 中多种流 | Dropout、shuffle、增广可能变化 |
| AMP | 可选 `scaler` | scale 与溢出处理轨迹改变 |

“六类”是方便组织的教学分组，不是刚好只有六个字段。查 [checkpoint.py](../../src/checkpoint.py) 的 `ResumeState`、`capture_rng`、`save_checkpoint`、`load_checkpoint`。文件的 format_version 表示存档结构，不表示代码或 PyTorch 版本。

## 2. seed 与 RNG state 的区别

seed 是随机序列的起点，state 描述已经走到序列哪里。恢复同一个 seed 会从开头重新采样，不能自动跳到中断的位置。模型 Dropout 使用全局 torch 随机流；DataLoader 显式 generator 管自己的 shuffle，二者分开是为了避免某次 Dropout 抽样改变下一轮数据顺序。

```python
# 可运行实验：恢复状态接续下一次抽样，重设 seed 回到开头
import torch

g = torch.Generator().manual_seed(7)
first = torch.rand(3, generator=g)
saved = g.get_state().clone()
expected_next = torch.rand(3, generator=g)
g.manual_seed(7)
restarted = torch.rand(3, generator=g)
assert torch.equal(first, restarted)
g.set_state(saved)
restored_next = torch.rand(3, generator=g)
assert torch.equal(expected_next, restored_next)
assert not torch.equal(first, expected_next)
print('seed 重启；state 接续。所有比较为精确张量比较。')
```

还要区分 Python random、NumPy、torch CPU、CUDA 各设备随机流。当前 capture_rng 覆盖相应全局状态与 loader generator，但不自动覆盖任意自建生成器、第三方库或 worker 内部状态。多 worker、prefetch、IterableDataset 加入后必须重新定义恢复协议。

## 3. 恢复顺序为何重要

当前入口先创建模型、optimizer、loader 和 scaler，然后加载 checkpoint 到这些对象，再开始迭代。创建模型本身会消耗随机数，所以把 RNG 还原过早、再初始化对象，会把还原位置重新向前推进。构造对象→恢复状态→进入迭代是合理基线，但 helper 隐藏的随机操作仍要逐项检查。

DDP/compile 包装会改变对象结构和 state_dict 键名；当前通过 unwrap_model 读写原模型，避免 module 前缀让跨单机/分布式加载失败。optimizer 管的是参数对象引用，不能加载后随意换成一套新模型参数。

`torch.load(map_location='cuda')` 可能把存档中的 RNG 状态张量也搬去 CUDA，而 CPU RNG 与 Generator.set_state 需要 CPU 状态张量。当前 `_rng_state_to_cpu()` 做了对应还原。这种张量表示的是状态描述，不能因为其他权重在 GPU 上就认为它也应在那里。

## 4. 先做简单恢复，再做公平配对

以下三条命令都保留相同最终预算 epochs=3。先连续跑完，同时留下 epoch1.pt；再从这份存档到另一个目录恢复。这是在比较“同一预算的中断恢复”，不是比较两条不同学习率计划。

```bash
python src/main.py --dataset synthetic --device cpu --model mlp --epochs 3 --batch_size 128 --seed 42 --save_every_epoch --exp_name course05_full --output_dir outputs/course
python src/main.py --dataset synthetic --device cpu --model mlp --epochs 3 --batch_size 128 --seed 42 --resume outputs/course/course05_full/epoch1.pt --exp_name course05_resume --output_dir outputs/course
python -m pytest tests/test_checkpoint.py tests/test_smoke.py -k resume -q
```

复制以下脚本比较两份存档；只对自己生成的可信文件使用完整加载。当前格式含训练状态对象，代码使用 weights_only=False；不应拿任意来源的未知存档当课程输入。

```python
# 可运行实验：运行上方配对训练后再执行
from pathlib import Path
import torch

root = Path('outputs/course')
full = torch.load(root / 'course05_full/last.pt', map_location='cpu', weights_only=False)
resumed = torch.load(root / 'course05_resume/last.pt', map_location='cpu', weights_only=False)
assert full['epoch'] == resumed['epoch'] == 3
assert full['global_step'] == resumed['global_step']
different = []
max_difference = 0.0
for name, value in full['model'].items():
    other = resumed['model'][name]
    if not torch.equal(value, other):
        different.append(name)
    if value.is_floating_point():
        max_difference = max(max_difference, (value - other).abs().max().item())
print('不同张量:', different)
print('最大绝对差:', max_difference)
print('loss 历史:', [h['train_loss'] for h in full['history']])
assert not different, '当前配对不满足逐位一致，需要继续定位状态或执行差异'
```

先比较离散状态 epoch/global_step 和数据顺序，再比较参数、optimizer 内部状态、history。上面的脚本仅检查部分内容，不能冒充完整 optimizer/RNG 验证；仓库相关测试提供更细的守护。

## 5. 什么叫可复现

能重跑、数值接近、逐位一致是不同强度的目标。固定种子是必要工具，但不能保证跨 PyTorch 版本、设备和所有内核完全一致。[官方可复现说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)也要求限定环境。

之前原版与修复版的 small_cnn 在当前 Windows 环境均观察到跨进程续训微小漂移；精确回归改用 MLP。**这些观测没有完成根因隔离，不能写成某个 CPU 内核已被证明有问题。** 不能通过增大容差掩盖漏存状态，也不能看到一点差就认定漏存 RNG。比较同环境基线、不同模型路径与可确定性设置，按证据缩小范围。

不要认为“MLP 测试精确通过”覆盖了 CNN、CUDA+AMP、compile、多 worker 或任意 DDP 组合。项目报告要列实际验证组合。环境设置 deterministic algorithms 还可能带来额外成本或对某些操作报错，它也不是万能开关。

## 6. 恢复粒度和存档可靠性

当前存档主要在 epoch 边界生成；中途崩溃会回到最后已保存边界，不能保证从任意 micro-step 原地续上。窗口中断若要精确恢复，还需保存累积梯度、窗口样本计数、scaler 阶段、采样器位置、loader/worker 预取等。epoch 级恢复通过不能当作 batch 级恢复实现。

当前 save_checkpoint 直接 torch.save 到目标路径，没有完整实现“临时文件→落盘→原子替换→校验”的故障写入协议。写入过程中断可能留下不完整文件。工程演进时可优先增加临时文件与同文件系统原子替换，保留上一份可恢复存档，再做故障注入验证；在做完前不宣称抗写入故障。

best 与 last 目的不同：best 供选模/最终评估，last 供接近中断点的恢复。从 best 续训属于选一个更早状态开始，应检查其中 epoch 与 optimizer 是否一并匹配。评估只载权重与完整恢复是两个入口需求，不要混淆。

改变 epochs 会改变当前闭包式 cosine 总步数，旧轮次与新计划可能不能逐位对齐。改变 lr 后 optimizer 状态虽加载，scheduler 捕获的新 base_lr 还可能在后续覆盖它。看 `diff_configs()` 的提醒，但“有提醒”不等于已经实现所有语义迁移。

## 7. 反证、自测与验收

在独立副本中暂不恢复 loader.generator state，比较下一轮样本顺序；不要只看最终准确率。另一次暂不恢复 global_step，比较第一步 resumed lr。每次恢复原文件并跑相关测试后再继续。故意破坏的区别要有张量/顺序/日志证据。

答案：① seed 重启，state 接续。② Adam 动量是更新公式的一部分。③任意 micro-step 恢复尚未实现。④ map_location 可能把 RNG 状态搬错设备，需要按状态 API 要求恢复。⑤两次最终预算相同才能把中断恢复与调度变化分开分析。⑥固定 seed 不保证所有环境逐位相同。

验收：完成一份配对报告，注明恢复边界、总预算、模型与设备，比较进度/权重/history；指出原子写入和多 worker 的未覆盖项。能用自己的话解释恢复创建顺序，而不是背“六类状态”。

参考：[官方保存恢复训练状态](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[官方随机性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

## 8. 逐层读 load_checkpoint，而不是只看 torch.load

第一层是路径：resolve_resume_path 的 last/best 指向当前 run 目录；显式相对路径按 cwd 解析。改变 exp_name 后再写 `--resume last` 会去新的目录找存档，不会自动找到旧实验。配对示例使用明确 epoch1 路径正是为了避免这种歧义。

第二层是模型结构：load_state_dict 默认严格匹配键与形状。旧模型与新模型结构不同属于 warm start/迁移需求，不应该偷偷丢掉 key 后宣称精确续训。第三层恢复 optimizer/scaler；第四层恢复 RNG；最后从 payload 建 ResumeState，next_epoch 由已保存 epoch 决定。

兼容旧格式时，当前实现允许部分键缺失并记录 state.missing。能兼容读取与完整恢复是不同强度的行为。查看 missing 提醒；缺 RNG/optimizer 时，报告为从已有权重继续训练，不能套用完整轨迹恢复结论。

手写一份“恢复后的第一个 batch”检查表：正确 epoch、真实 sampler 顺序、scheduler 使用的 global_step、optimizer 动量、BN buffer、Dropout/loader 随机状态、精度模式。与连续训练的对应位置逐项比较，首次偏离点比最终准确率更能定位原因。

假设恢复后第一轮 loss 很接近，但 best 选择变了：查 best_val_acc 是否保存/恢复；假设早停提前或延后：查 patience_counter；假设两组参数同值但下一步不同：查 optimizer、随机 batch 与 scale；假设出现 key 前缀错误：查 wrap/unwrap。每种现象指向不同状态，别统一解释成“种子没固定”。

作业：为 epoch 级恢复画时序图，在每条边标注对象创建、状态加载、随机抽样、参数更新和存盘；再标出要支持 micro-step 恢复时需要新增的状态。答案关键是梯度与采样位置目前没进存档，不能把未完成的功能靠命令行名字补出来。
