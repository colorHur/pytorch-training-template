# 第六节：DDP 的梯度、数据与通信顺序

[上一节](05-checkpoint-and-reproducibility.md) · [目录](README.md) · [下一节](07-testing-debugging-and-delivery.md)

## 0. 先修、阅读路径与闭卷检查

先修：单进程累积、样本加权与进程概念。第一遍手算 rank/sample 两个分母并观察 sampler；第二遍走两个 CPU 进程及 no_sync 作用范围；进阶考虑不等长输入与每 rank RNG。先弄清计算和协议，再看吞吐。

迁移任务：构造局部样本数不同的梯度/accuracy 例子，写出全局答案与通信前提。闭卷答 [Q21–Q24](INTERVIEW.md#lesson-6)，CPU gloo 证据不扩大成多 GPU 性能结论。

第一晚用手算和采样实验理解全局语义，第二晚运行两个 CPU 进程并读分布式续训测试。先回答：DDP 会自动把数据切给各 GPU 吗？local_rank 与 rank 一样吗？每个 rank 的准确率平均一下就对了吗？no_sync 只包 backward 能达到预期吗？

## 1. 一个进程通常管理一份模型副本

DDP 数据并行让各进程拥有模型副本，处理各自数据并同步梯度；它不会自动把模型参数按卡切片，也不替你创建合适的数据 sampler。world_size 是总进程数，rank 是全局进程序号，local_rank 是本节点序号。单节点时两种 rank 常相同，多节点时不能混用。

“四张 GPU”不必等于“world_size=4”，取决于如何启动进程；本模板遵循常见每 GPU 一进程模式。设备绑定、进程组初始化、后端、超时与 rendezvous 是运行协议的一部分。FileStore/TCPStore 管进程会合信息，不等于梯度通信算法本身。

[distributed.py](../../src/distributed.py) 的 `DistContext` 集中保存这些信息。未启用分布式时 helper 能接受 None 或 disabled ctx，训练代码不需要在每处重复判断。读 `setup_distributed`、`choose_backend`、`wrap_model`、`cleanup_distributed`，先弄清谁创建和释放进程组。

## 2. 用四个样本推导全局梯度

目标是对全局样本平均。各 rank 样本数相同且各自 loss 按样本平均时，DDP 默认的 rank 平均梯度等于全局平均。样本数不同则“均值再平均”会错。例如 rank0 有 1 个样本、平均梯度 2；rank1 有 3 个样本、平均梯度 6。简单 rank 平均得到 4；正确样本平均是 `(1×2+3×6)/4=5`。

当前训练循环先得到各 rank 的样本梯度和；DDP 同步后默认除以 W，得到 `Σsum_grad_r/W`；再汇总窗口样本数 N，乘 `W/N`，最终等于 `Σsum_grad_r/N`。看第三节的 grad_scale 与 `reduce_sums([window_samples], ctx)`。它多了一次窗口样本计数归约，性能报告要包含这个成本。

这个公式假设默认 DDP 梯度平均语义，以及各 rank 对齐更新窗口。自定义 comm hook、join 的分母设置、不同训练步数会改变前提，不能只改公式就保证任意输入协议成立。当前训练 sampler 用等长策略避免各 rank 窗口数量不一致。

## 3. 训练分片与评估分片为什么不同

`DistributedSampler` 让每个 rank 取自己的样本，并用相同 seed/epoch 产生一致的全局打乱。训练采样器当前用 `drop_last=True` 对齐每 rank 样本数；DataLoader 自身保持 drop_last=False。两处 drop_last 作用于不同层，不能混为一个开关。训练为对齐可能丢少量样本，报告中应说明。

评估用 `StridedSampler`：rank r 取 r、r+W、r+2W…，不填充、不重复。若评估采样器为了对齐重复样本，就会给被重复样本额外权重；偏差可能往高或低走，并不必然使准确率虚高。

```python
# 可运行实验：训练对齐与验证不重复
from torch.utils.data import DistributedSampler
from src.distributed import StridedSampler

dataset = list(range(7))
train_parts = [list(DistributedSampler(
    dataset, num_replicas=2, rank=r, shuffle=False, drop_last=True
)) for r in range(2)]
val_parts = [list(StridedSampler(7, rank=r, world_size=2)) for r in range(2)]
print('训练:', train_parts)
print('评估:', val_parts)
assert [len(p) for p in train_parts] == [3, 3]
assert sorted(i for part in train_parts for i in part) == list(range(6))
assert sorted(i for part in val_parts for i in part) == list(range(7))
assert set(val_parts[0]).isdisjoint(val_parts[1])

sampler = DistributedSampler(list(range(20)), num_replicas=2, rank=0, seed=42)
sampler.set_epoch(0)
first = list(sampler)
sampler.set_epoch(1)
second = list(sampler)
assert first != second
sampler.set_epoch(1)
assert list(sampler) == second
print('同 seed+epoch 的顺序相同，不同 epoch 改变顺序')
```

训练/验证索引切分在各 rank 必须一致，不能用 seed+rank 切出不同集合。模型初始化也应在同步后保持一致；每 rank 的随机 Dropout 流是否偏移是另一个设计选择。数据分片、初始化、随机增广是三个独立问题。

## 4. 指标要汇总分子与分母

rank0 的 1 个样本正确，rank1 的 9 个样本全错，平均两个 accuracy 得到 0.5，真实 accuracy 是 1/10=0.1。评估应归约 `(loss_sum, correct_count, sample_count)` 后除，不是先局部除再求平均。当前 `evaluate()` 这样处理，读 `reduce_sums()` 与 `test_sample_weighted_mean_is_not_mean_of_means`。

若损失的 mean 按权重或 token 定义，评估 loss_sum 的计算也要同步修改；不能因为 accuracy 分母正确就认为所有指标都正确。空 rank 的局部计数为 0 可以参与最终归约，但其 forward 次数不同可能与通信协议冲突。

**当前边界：**不等长评估 loader 可能导致各 rank forward 次数不同；DDP 默认可能在 forward 广播 buffer。该组合存在集合通信对齐风险，本教程不把它写成已经完成故障证明或修复。可考虑显式同步 buffer 后用未包装模型评估等方案，但必须验证参数、buffer 和通信顺序；不能为消除等待随意关闭所有同步。

## 5. no_sync 的范围与通信次数

中间 micro-step 用 no_sync，最后一次正常同步。forward 和 backward 都必须在上下文内，因为 DDP 在 forward 阶段也安排相关同步逻辑。[官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)明确说明这一点。仅把 backward 缩进进去，程序仍可能算对，但通信优化没有达到预期。

注意“一窗口只做一次同步 backward”不等于“一窗口只有一次 all-reduce”：梯度有多个 bucket，还有当前样本计数归约和其他集合操作。性能报告要量化实际通信，不能用 accum 比值直接宣称同倍提速。

看 `test_no_sync_contains_forward_as_well_as_backward`：fake model 记录 forward 的上下文状态，它验证范围；这条单进程测试不能证明 NCCL 通信吞吐、实际 bucket 次数或 GPU 组合。分层测试各有证明范围。

## 6. 启动两个 CPU 进程

```bash
python tools/ddp_probe.py --help
python tools/ddp_launch.py --nproc_per_node 2 -- src/main.py --dataset synthetic --device cpu --epochs 1 --batch_size 16 --grad_accum_steps 2 --exp_name course06 --output_dir outputs/course
python -m pytest tests/test_distributed.py -q
```

CPU+gloo 用于协议与数学验证，不代表多 GPU+NCCL 已验证。Linux 合适环境也可用 torchrun 启动同一个入口。Windows 有 CUDA 并不自动拥有 NCCL，当前选择逻辑还检查后端可用性；有能力声明后，实际张量通信路径仍需探测。

CPU 两进程可能比单进程慢，因为启动、同步和线程竞争占比大；别把它当 DDP 必然无用，也别把总吞吐变化与单步速度混淆。记录全局 batch、每 rank batch、样本处理数和 wall time。两个 rank 同时打印日志时，先确认 rank 标签再判断乱序是否表示逻辑错误。

遇到 hang，先看各 rank 是否都到达相同集合通信；查数据步数、某 rank 提前异常、进程组超时与 rendezvous 路径，再看驱动后端。`barrier()` 要所有参与 rank 调用，随手加一个 barrier 不能修复错误协议，反而可能新增等待点。

## 7. DDP 续训的第二层约束

DistributedSampler 通常依据 seed+epoch 决定顺序，入口每轮必须调用 `set_epoch(真实 epoch)`。恢复第 2 轮却从 epoch0 重新打乱，会重复错误顺序，即使 loss 看起来正常。搜索 [test_smoke.py](../../tests/test_smoke.py) 中的 ddp/resume 测试，读它怎样保持最终预算并用已保存轮次恢复。

rank0 独占存档避免多进程同时写；但 rank0 一份 RNG 不自动表示所有 rank 的随机状态。若引入独立 rank RNG、不同消费次数或 worker 状态，就需要每 rank 保存/恢复，或定义可重建协议。当前简单组合的测试通过不能推出任意分布式续训逐位保证。

保存/加载模型前缀通过 unwrap_model 处理。main 在结束后读取 best 前有 barrier，目的是等待写入完成；恢复阶段各 rank 只读已有存档，不需要为了“看起来更安全”无意义增加 barrier。若未来采用异步存档，协议需要重新设计。

## 8. 反证、自测与验收

独立副本把 set_epoch 改成每轮固定值，先观察 sampler 反证，再跑相关恢复判据；不要先重训大型任务。另一次把 no_sync 改成仅 backward，forward 观测测试应变红。恢复哈希和基线后收工。

答案：①DDP 不自动分片输入。②local_rank 是节点内编号。③不等样本数不能平均局部均值。④no_sync 包住 forward/backward。⑤BN 使用各局部 batch 统计，不自动等于全局大 batch。⑥rank0 RNG 不自动恢复所有 rank。⑦CPU 两进程通过不覆盖 NCCL。

验收：解释 N=7/W=2 的训练和评估索引，推导不等 rank 样本数的梯度权重，运行一次 CPU 两进程，并指出评估不等长及每 rank RNG 的剩余边界。

参考：[DDP API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[DistributedSampler](https://docs.pytorch.org/docs/stable/data.html)、[官方 DDP 入门](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)。

## 9. 跟踪两 rank、两 micro-step 的一次窗口

在纸上画两列 rank0/rank1。第一步两边清梯度、前向、反向，各自累计样本梯度和，no_sync 暂不做常规梯度归约；第二步正常前向/反向触发梯度同步，之后两边归约窗口样本数，再按相同全局分母归一化、裁剪与更新。相同参数起点和同步后的梯度让两份参数继续保持对齐。

为什么不能只让 rank0 optimizer.step？DDP 常规协议让每 rank 都更新自己的副本；只更新一边会破坏后续一致性。为什么不能每 rank 自行决定更新窗口结束？另一边可能尚未进入集合通信，容易等待或错配。为什么不能 rank0 提前 evaluate 而 rank1 还在训练？训练/评估可能调用不同的同步操作，顺序不再一致。

再区分两个“分母”：梯度归约通常按 rank 平均，loss/accuracy 报告应按样本权重。当前训练代码在更新时修正 rank 平均到全局样本平均；统计是在另一步归约样本 loss_sum 与 correct。两段都用 all_reduce 不代表它们回答同一问题。

拓展 BN 例子：rank0 输入均值很低、rank1 很高，各自 BN 后的激活分布与一个全局 batch 的统计可能不同；同步梯度不会自动把前向 BN 变成全局 BN。SyncBatchNorm 需要支持的后端/设备路径，不能看开关显示 True 就认定真的转换了。读 maybe_convert_sync_bn 的返回值和日志。

作业记录包括：每 rank 实际样本数、第一轮索引、global batch 名义值与真实窗口值、谁存盘、谁更新、哪里有集合通信。只贴两个进程的成功日志不足以回答这些问题。完成后用一张图解释“单机训练与 DDP 的共同部分/新增协议”，面试时就能从自己代码出发回答。
