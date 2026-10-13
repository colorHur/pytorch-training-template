# 两周学透训练脚手架

这份课程从仓库现有的 Day 0–8 学习计划整理而来。旧计划保留在个人学习空间；这里的练习直接对应当前公开代码，避免固定行号失效。建议每天 1.5–2 小时：先预测结果，亲手运行，再读解释。训练结果因设备和版本不同可能变化；不要把示例数字当作你机器上的实测值。

## 学习方法

每个主题走同一闭环：

1. **画数据流**：输入、输出、状态变化分别是什么？
2. **写出不变量**：什么必须成立？例如累积梯度应等于同一批样本的直接更新。
3. **构造反例**：选择能让错误显形的最小输入，不先跑完整 MNIST。
4. **修复并回归**：同时跑新测试和相关旧测试。
5. **口头解释**：能说清适用条件与尚未验证的情况，才算学会。

动手改代码前，先保存当前文件副本或在独立分支工作。破坏实验结束后，检查改动并确认基线测试恢复；不要把故意损坏的版本提交。每次实验记录命令、环境、预期、实际、解释，四项缺一不可。

## 每日路线

| 天 | 要回答的问题 | 阅读入口 | 当天产出 |
|---|---|---|---|
| 1 | 一次训练从命令行到参数更新经过哪些层？ | `src/main.py`、`src/config.py` | 自己画调用图，跑 `--dataset synthetic --epochs 1` |
| 2 | `.backward()` 为什么会累加？何时清零？ | `src/train.py`、[梯度累积实验](02-gradient-accumulation.md) | 手算一次两批更新，再核对张量 |
| 3 | 尾窗口和不等长批次为何会改变梯度？ | `tests/test_train.py` | 让数值等价测试先失败再通过 |
| 4 | 学习率以 micro-step 还是 optimizer step 为时钟？ | `build_lr_scheduler()`、`optimizer_steps_per_epoch()` | 列出每次更新的学习率 |
| 5 | AMP 中 scale、unscale、裁剪的顺序是什么？ | `src/train.py`、PyTorch AMP 文档 | 解释 fp16 下溢与溢出跳步 |
| 6 | checkpoint 为何不只存模型权重？ | `src/checkpoint.py`、`tests/test_checkpoint.py` | 列出恢复所需的状态与测试边界 |
| 7 | 评测集为什么不能按 rank 的均值再平均？ | `evaluate()`、`reduce_sums()` | 用 1 与 9 个样本手算反例 |
| 8 | `no_sync` 为什么要覆盖 forward？ | `src/distributed.py`、`tests/test_train.py` | 跟踪两次 forward 是否在上下文内 |
| 9 | BN、随机数与 DDP 分片怎样影响“等价”？ | `src/model.py`、`src/data.py` | 写出单机/多卡比较的前提 |
| 10 | 显存峰值为何不能只看激活？ | `experiments/exp_memory_accounting.py` | 记录测量条件、峰值与运行时间 |
| 11 | 一条测试究竟能证明什么？ | `tests/test_train.py`、[工程手册](03-engineering-practice.md) | 删除关键逻辑，确认测试会红 |
| 12 | 如何从干净环境复现？ | `requirements*.txt`、CI 配置 | 记录安装、启动、测试步骤 |
| 13 | 怎样向面试官讲一次故障定位？ | 本课程与仓库实验记录 | 用“现象→反例→根因→修复→限制”讲 3 分钟 |
| 14 | 哪些能力已被证据支持？ | README、测试结果、自己的笔记 | 写一页项目摘要，逐句指向证据 |

先读 [训练更新的逐步推导](02-gradient-accumulation.md)，再读 [工程实践手册](03-engineering-practice.md)。旧 Day 计划包含更多背景和个人环境实测，可作为补充；本课程中的验收以当前代码与亲手运行结果为准。

## 毕业检查

你应能在不看代码时解释：

- 为什么 `loss / grad_accum_steps` 在尾窗口会错；为什么不同大小的 microbatch 不能简单平均。
- `optimizer.zero_grad()`、`backward()`、`unscale_()`、梯度归一化、裁剪、`step()` 各改变了什么状态。
- DDP 为什么平均 rank 的梯度；各 rank 样本数不同时该如何加权。
- “测试通过”“GPU 实测”“Linux 可复现”“生产可用”分别需要什么证据。

答不出时回到最小反例，不靠背结论过关。

