# 一次参数更新究竟在算什么

目标：从数学式子走到 `src/train.py::train_one_epoch`，理解梯度累积、尾批、AMP、裁剪和 DDP 的顺序。阅读前先会用 `tensor.backward()` 和 `optimizer.step()`。

## 1. 手算一个反例

假设三个 microbatch 各有 2 个样本，设置 `grad_accum_steps=4`。每批平均损失的梯度分别为 `g1, g2, g3`。它们组成的真实大 batch 的梯度是 `(g1+g2+g3)/3`。若代码每次固定执行 `loss/4`，最终得到 `(g1+g2+g3)/4`，更新只有应有幅度的 75%。更新**次数**仍是 1，所以只数 `optimizer.step()` 的测试无法发现错误。

再设三批大小为 `2, 2, 1`，各批平均梯度为 `g1, g2, g3`。目标是 `(2g1+2g2+g3)/5`，不是 `(g1+g2+g3)/3`。这是“均值的均值”错误。代码用 `loss * batch_samples` 得到每批样本梯度之和，窗口末尾再除以累计样本数。前提是 `criterion` 返回**按样本取均值的标量**；若改成求和、自定义权重或 token 级损失，必须重审分母。

## 2. 对照代码的状态变化

| 时机 | 操作 | 这个状态为何不能省 |
|---|---|---|
| 窗口开始 | `optimizer.zero_grad(set_to_none=True)` | 清掉上一次更新留下的梯度；同一窗口内不能反复清零 |
| 每个 microbatch | 前向、计算平均 loss、乘样本数、反向 | `.backward()` 把梯度**加**到 `.grad` |
| 窗口最后 | AMP 时 `scaler.unscale_(optimizer)` | 后面的归一化和裁剪要基于可解释的真实梯度 |
| 窗口最后 | 每个 `.grad` 除以窗口样本总数 | 从“样本梯度和”恢复“样本平均梯度” |
| 窗口最后 | 可选梯度裁剪、写入学习率、`optimizer.step()` | 裁剪属于一次更新，不属于单个 microbatch |
| 更新后 | `global_step += 1`、清空窗口样本计数 | 调度时钟是参数更新次数 |

`torch.autocast()` 只包前向与 loss；反向沿前向选定的 dtype 执行。fp16 可能使很小的梯度下溢，`GradScaler` 先放大 loss，再在更新前还原。若有非有限梯度，scaler 可能跳过 optimizer step；这里的 `global_step` 表示**尝试的更新窗口**，不是保证参数实际变化的次数。分析性能或调度时要说明这个口径。可参照 [PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。

## 3. DDP 为什么要再想一步

每个 rank 做自己的样本梯度和，DDP 在同步时默认对 rank 取平均。若有 `W` 个 rank，总样本数为 `N`，同步后的张量是 `Σ_r sum_grad_r / W`。真实全局样本平均梯度应是 `Σ_r sum_grad_r / N`，因此最终乘 `W/N`。代码在更新窗口使用 `reduce_sums([window_samples], ctx)` 汇总 `N`。

中间 microbatch 用 `no_sync()` 跳过梯度同步；最后一次正常同步。**前向也必须在该上下文内**，否则仍可能同步。参照 [PyTorch DDP 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。这一做法要求所有 rank 的更新窗口边界对齐；当前训练采样器提供等长样本数。若日后换成不等长流式数据，先设计进程结束与通信协议，不要直接复用。

## 4. 亲手验证

在仓库根目录运行：

```bash
python -m pytest tests/test_train.py -q
python src/main.py --dataset synthetic --epochs 1 --grad_accum_steps 4
```

重点读两个测试：`test_accumulation_matches_real_batch_even_with_short_last_batch` 比较**参数值**；`test_no_sync_contains_forward_as_well_as_backward` 检查作用范围。先预测 `(7 个样本，batch=2，accum=4)` 会更新几次、分母是多少，再运行测试。

**破坏实验**：在独立工作副本中暂把窗口末尾的 `grad_scale` 改回 `1 / cfg.grad_accum_steps`，运行数值等价测试，观察失败的参数差。恢复源码，再运行同一测试，确认转绿；用文件对比检查只留下预期改动。不要把破坏版提交。

## 5. 自测与答案

1. **5 个 microbatch、每 3 批更新一次，会有几次更新？** 两次。调度器的 `steps_per_epoch` 应用向上取整。
2. **最后 1 个样本的 batch 能和前面 2 个样本的 batch 平均权重相同吗？** 不能。要按样本数加权。
3. **为什么归一化必须在裁剪前？** 裁剪阈值定义在最终平均梯度上；先裁剪样本梯度和会随窗口大小改变阈值语义。
4. **这个数值测试能证明含 BatchNorm 的网络与真实大 batch 完全等价吗？** 不能。BN 使用 microbatch 统计量，前向本身就不同；测试刻意用无 BN、无 Dropout 的线性模型隔离梯度累积。
5. **“跑通一次”能证明可复现吗？** 不能。还要固定数据、权重、依赖、随机状态及指标口径，并在目标环境重新运行。

