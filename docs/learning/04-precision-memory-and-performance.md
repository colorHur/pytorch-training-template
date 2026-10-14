# 第四节：AMP、显存与有证据的性能优化

[上一节](03-training-and-optimization.md) · [目录](README.md) · [下一节](05-checkpoint-and-reproducibility.md)

## 0. 先修、阅读路径与闭卷检查

先修：dtype、前向/反向与最终梯度尺度。第一遍理解显存账与 AMP 顺序；第二遍验证峰值/计时边界与 BN 副作用；进阶看 compile/profiler/加载瓶颈。没有 GPU 时完成 CPU 与源码部分，硬件结果记未测。

迁移任务：设计一次同时记录质量、显存和时间的单变量对照，明确是否包含加载/通信/预热。闭卷答 [Q13–Q16](INTERVIEW.md#lesson-4)，不直接背历史机器数字。

第一晚理解精度与显存构成，第二晚跑对照、检查有状态层和性能计时。目标是会定位瓶颈并解释测量条件；“打开开关”只说明配置改变，收益必须测。先回答：AMP 是否把全部参数变成 fp16？缓存显存等于活跃张量吗？激活减少 80% 能推出峰值减少 80% 吗？

## 1. 写一份显存账

常见训练显存包含参数、梯度、optimizer 状态、前向保留激活，以及内核临时工作空间和分配器缓存。fp32 参数通常每元素 4 字节；若参数、梯度、Adam 两份动量都用 fp32，仅这些项粗略约 16 字节/参数。混合精度、融合 optimizer、master weight 和设备位置会改变这份账，不能把公式当所有实现的定值。

batch 增大会影响激活，但一般不按同样倍数增加参数和 Adam 状态。梯度累积减少一次前向涉及的样本数，checkpoint 减少保存的部分激活；二者都不会自动把参数与优化器状态切分。参数状态超出单卡时需要另想方案，单纯减 batch 可能不够。

| 指标 | 解释 | 容易误判的地方 |
|---|---|---|
| `memory_allocated()` | 分配器当前活跃张量占用 | 不包括所有驱动/其他进程占用 |
| `memory_reserved()` | PyTorch 分配器保留的总量 | 缓存池中的空闲块也在内 |
| `max_memory_allocated()` | 上次重置后活跃占用峰值 | 不重置会夹带以前实验峰值 |
| 设备工具显示占用 | 驱动视角的更广占用 | 不能与 allocated 直接等同 |

`empty_cache()` 释放部分未使用缓存给其他消费者；仍被张量引用的内存不会因此消失。把带计算图的 loss 或输出长期放进列表，可能保留一条又一条图；日志通常用 `loss.item()` 或适当 detach，但 `.item()` 在 CUDA 上也可能带来同步开销，不能无节制地每次打印。

## 2. AMP 的两个机制

autocast 按算子选择适合的 dtype，某些计算仍保留高精度；GradScaler 缩放 loss 的梯度，降低 fp16 小梯度下溢风险，再在 step 前还原并检查非有限值。AMP 不等于 `model.half()`，也不保证所有硬件和任务更快。bf16 与 fp16 的指数/尾数范围不同，本仓库当前前向选的是 fp16，不能套用所有 bf16 经验。

[train.py](../../src/train.py) 把 autocast 放在前向和 loss 范围，反向在外；窗口末尾只 unscale 一次，归一化、裁剪之后 scaler.step，最后 scaler.update。不要每个 microbatch unscale，后续再叠加缩放梯度会混合两个尺度。indexed CUDA device 如 `cuda:0` 的 autocast device_type 仍是 `cuda`，当前用 `torch.device(device).type` 解析。

GradScaler 的 step 可能在非有限梯度时跳过，scale 也会变化。观察 loss 与更新窗口数量不足以判断成功更新；诊断时关注非有限值、scale 趋势和参数变化。当前模板没有完整的成功更新计数日志，报告时要说明这个边界。

没有 GPU 时用 CPU 关闭 AMP；项目配置把 CPU+AMP 视作无效组合。GPU 环境可运行：

```bash
python src/main.py --dataset synthetic --device cuda --epochs 1 --amp --exp_name course04_amp --output_dir outputs/course
python src/main.py --dataset synthetic --device cuda --epochs 1 --no-amp --exp_name course04_fp32 --output_dir outputs/course
```

两次“都能跑”只是冒烟检查；要比较精度和速度还需相同预算、重复种子、测量阶段与正常指标。只比较最后一次 train_loss 不足以评估 AMP 效果。

## 3. checkpoint 为什么改变 forward 的状态

[model.py](../../src/model.py) 的 SmallCNN 可对 block 使用 activation checkpoint：前向少保存一些中间信息，反向时重算需要的信息，用计算换内存。保存范围与重算停止位置依赖具体实现；不能绝对说每个算子总被完整多算一遍。

BN 训练态 forward 会更新 running_mean、running_var 和 num_batches_tracked。重算不是只做数学运算，也会触发这些副作用。因此 loss 与梯度相同，BN buffer 仍可能不同；随后 eval 用运行统计，就可能出现差异。非可重入 checkpoint 并不自动消除 BN 副作用。

LayerNorm/GroupNorm 没有同样的运行均值机制，常作为可考虑方案；但不能推导整个 Transformer 都是无副作用纯函数。Dropout 需要正确保留 RNG，其他自定义缓存和状态也要审计。PyTorch 两种 checkpoint 实现都涉及随机状态保留，不能把 RNG 正确性简单归结为只有 `use_reentrant=False` 才能做到。

```bash
python -m pytest tests/test_model.py -q
python experiments/exp_memory_accounting.py --epochs 2
python experiments/exp_checkpoint_granularity.py
```

重点读 `test_checkpointing_keeps_loss_and_gradients_identical`、`test_bn_running_stats_double_update_is_a_known_issue`、`test_dropout_rng_is_preserved_by_checkpointing`。前者检查数学结果，第二条记录已知副作用，第三条检查随机性。测试全部通过不代表副作用已修复：有的测试正是在确认已知问题仍存在。

本模型只在 training 模式启用 checkpoint。eval/no_grad 下通常不需要保存用于反向的激活，检查点可能没有收益；没有反向时也不能声称必然产生一次额外反向重算。训练/推理的真实调用范围要靠观测，而不是猜。

## 4. 公平对照的步骤

先固定模型、数据顺序、训练预算与 seed，选择一个因素改变。完整记录 microbatch、累积、AMP、checkpoint、设备与依赖。用独立输出目录；确保原有模型和 optimizer 没被引用，避免“第二组峰值”夹带第一组常驻张量。最清楚的办法是每个变体独立进程。

例如 FP32 batch256 与 batch64×accum4 比较名义窗口，但 CNN 的 BN 前向统计不同，不能宣称严格等价。可先用无 BN/随机层模型验证数学，再用真实 CNN 评估任务指标；两种实验回答不同问题。需要正确性和资源收益一起看。

仓库 README 的历史激活、峰值与耗时数字属于原设备与配置结果，新版教程不把它们当本次新测。记录实际数字前先确认脚本是否真正跑在 CUDA、是否降级、是否跳过。在 CPU 上写“0GB CUDA”不能当 GPU 显存节省成果。

## 5. CUDA 计时需要同步

CUDA 工作通常异步提交；CPU 秒表只围住提交阶段可能低估耗时。预热后在被测段前后同步，明确是训练 step 而非数据读取或冷启动。下面只测 GPU 上固定小模型的更新，不包含 DataLoader 和主机到设备拷贝。

```python
# 可运行实验：有 GPU 才测，CPU 明确跳过
import time
import torch

if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
    print('SKIP: 没有可用 GPU，未产生 CUDA 性能结论')
else:
    torch.manual_seed(0)
    model = torch.nn.Linear(128, 64).to('cuda')
    x = torch.randn(32, 128, device='cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    def step():
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(50):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print('平均 step 秒:', elapsed / 50)
    print('峰值 allocated 字节:', torch.cuda.max_memory_allocated())
    print('reserved 字节:', torch.cuda.memory_reserved())
```

这只是平均 step 冒烟，未提供 p95 分布与重复实验方差；不能据此给服务延迟承诺。CUDA event 更适合 GPU 段计时，CPU perf_counter 同步测量方便理解；实际任务要决定是否包含加载、通信和保存。Profiler 可以定位算子与时间，但 profiler 本身会带来开销，性能基准应在关闭 profiler 时再测。

## 6. compile、数据加载与排查顺序

[compile_support.py](../../src/compile_support.py) 提供探测、冒烟与降级。Windows 是否支持某后端由当前工具链与版本决定，不能把旧机器缺 Triton/编译器写成永久事实。运行 `python tools/compile_probe.py --help` 再按当前支持参数探测，读清是否真的启用；前向通过也不能自动证明反向、其他 shape、DDP 组合都通过。

compile 首次编译、图变化重编译、steady state 三个阶段要分开报告，不能把冷启动与普通 eager step 混比。模型太小可能 Python、数据加载或同步更突出，编译不一定节省总时长。

推荐排查顺序：明确是读取、传输、计算、通信还是保存慢；看 profiler/日志；确认正确性基线；调整一个因素；重新测质量、时间、峰值。若 GPU 空闲，先看 CPU 预处理和 DataLoader，非默认 pinned memory 和 non_blocking 都不能凭开关保证拷贝与计算重叠。

## 7. 反证与自测

反证 A：在独立副本去掉模型的 training guard，用调用范围观测测试检查 eval 是否仍触发 checkpoint。若只测最终 loss 相同，可能抓不住额外开销。恢复后再跑原测试。

反证 B：不重置峰值连续跑两组，预测第二组峰值为何可能保留第一组最大值。记录过程后按独立进程重做；这次实验不要求制造 OOM。

答案：①AMP 不是全参数 fp16。②reserved 包含缓存，不等于活跃张量。③激活占总账的一部分，不能按同一百分比推峰值。④BN 重算副作用可能影响 eval。⑤平均 step 不包含未计入的数据加载，也不是 p95。⑥compile 探测通过只是被测路径支持。

验收：交一页测量条件表与至少一组对照，能指出时间边界、峰值重置时机和一条未验证 GPU 组合；没有 GPU 时完成 CPU/源码部分并明确剩余项。

参考：[AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)、[checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)、[CUDA 内存与异步执行](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

## 8. 从一次 OOM 推进到有解释的优化

假设 loss 第几轮都正常，但每轮峰值不断增长。先问是否真的测的是 allocated、是否每轮重置峰值、是否新建 optimizer 没释放，以及是否把带图 tensor 写进 history。再把 batch 固定、关闭非必要日志，观察增长是否随循环发生。如果一开始就无法容纳参数，激活优化不能解决全部问题。

假设训练前向能过、反向 OOM。检查反向保存与重算临时空间、optimizer 首次 step 才创建的动量，不能只用前向峰值估训练预算。假设加载模型时 OOM，应检查常驻参数/加载临时副本，而不是讨论梯度累积能省多少激活。

为本项目设计一张待填测量表：

| 变体 | microbatch/accum | 精度 | checkpoint | 峰值 allocated/reserved | 稳态时间与样本吞吐 | 任务指标 |
|---|---|---|---|---|---|---|
| 基线 | 自己填写 | fp32 | 关 | 实测 | 预热后重复 | 固定评估协议 |
| 单变量改进 | 写明唯一差异 | 明确 | 明确 | 实测 | 同测量边界 | 检查质量损失 |

不要用单次生成/训练耗时接近证明优化零成本；时间分布、预热、并发、常驻内存都要看。这个技巧也直接用于后续扩散模型项目：模型预加载成本、adapter 切换成本和生成成本分别测，平均值和尾延迟分别记录。

Profiler 初学只看三个问题：数据加载是否让设备等待，哪类算子占主要计算，是否存在频繁同步或小算子开销。先保存一次短 trace，不要开 profiler 贯穿全部训练。Trace 解释后，在关闭 profiler 的正常运行中再做重复计时，才把结论写入报告。
