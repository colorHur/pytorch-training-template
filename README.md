# PyTorch 训练模板

[![CI](https://github.com/colorHur/pytorch-training-template/actions/workflows/ci.yml/badge.svg)](https://github.com/colorHur/pytorch-training-template/actions/workflows/ci.yml)

一套**手写、不含高层封装**的 PyTorch 训练脚手架。目的是把训练循环的每个细节讲清楚——而不是调一个 `Trainer` 就完事。

> 为什么手写？面试问的是 `optimizer.zero_grad()` 为什么必须在 `backward()` 之前、梯度累积怎么省显存、`GradScaler` 解决什么问题、梯度检查点省的是哪部分显存。这些被封装的 API 挡住了。

这个仓库不只是「能跑」——**每个显存优化手段都配了实测数据**（7 个对照变体 + 单步拆解），
包括一个实测抓出来的反直觉结论：梯度检查点把前向保活的激活砍掉 83%，训练峰值却只降 13%。

## 特性

| 能力 | 说明 |
|------|------|
| **配置系统** | YAML + dataclass，命令行可覆盖任意参数；配置随 checkpoint 一起保存，实验可复现 |
| **手写训练循环** | 不用 Lightning/Trainer，每一步（清零→前向→反向→裁剪→更新）都显式写出 |
| **梯度累积** | 小显存跑等效大 batch，显存不增长 |
| **混合精度** | `torch.autocast` + `GradScaler`，含梯度还原时机 |
| **梯度检查点** | 不保存块内激活、反向重算，用计算换显存；含 BN 统计量陷阱的处理说明 |
| **手写 LR 调度** | warmup + cosine/step，能看到 lr 每一步怎么变（不是黑盒 `scheduler.step()`） |
| **梯度裁剪** | 按全局 L2 范数裁剪，防梯度爆炸 |
| **显存监控** | 每 epoch 打印当前/峰值显存 |
| **checkpoint** | 保存最优权重 + 完整训练状态（含配置），支持早停 |
| **测试 + CI** | 79 个 pytest 用例（CPU 可跑，多平台 CI）；离线合成数据集，秒级验证整条流水线 |

## 目录结构

```
pytorch-training-template/
├── src/
│   ├── config.py      # 配置系统：dataclass + YAML + 命令行覆盖
│   ├── data.py        # 数据加载：Dataset / DataLoader / transform
│   ├── model.py       # 模型定义 + 注册表（small_cnn / mlp）+ 梯度检查点
│   ├── train.py       # ⭐ 训练循环核心：手写 step / 评测 / LR 调度
│   └── main.py        # 入口：argparse + 日志 + checkpoint + 显存统计
├── configs/
│   └── mnist.yaml     # MNIST 标准配置
├── experiments/
│   ├── exp_memory_accounting.py       # 显存账对照实验（7 个变体）
│   └── exp_checkpoint_granularity.py  # 梯度检查点单步拆解（显存 + 耗时）
├── tests/             # pytest：配置 / 数据 / 模型 / 训练循环 / 端到端
├── .github/workflows/ci.yml           # 多平台 CI：lint + 测试（CPU）
├── outputs/           # 训练产物（git 忽略）
├── pyproject.toml     # ruff 与 pytest 配置
├── requirements.txt       # 运行时依赖
└── requirements-dev.txt   # 开发/测试依赖（CI 用）
```

## 快速开始

### 1. 环境

```bash
# 建 conda 环境
conda create -n ai-lab python=3.11 -y
conda activate ai-lab

# torch 要按 CUDA 版本从专用源装（示例：CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
pip install -r requirements.txt
```

> 国内网络可用镜像加速：PyPI 换阿里云源、torch 的 CUDA 轮子换高校镜像源，
> 能明显快于官方源。本项目在阿里云 PyPI + 镜像 torch 轮子下安装完成。

### 2. 训练

```bash
# 用配置文件
python src/main.py --config configs/mnist.yaml

# 覆盖参数（命令行优先级最高）
python src/main.py --config configs/mnist.yaml --epochs 5 --lr 5e-4

# 混合精度 + 梯度累积
python src/main.py --config configs/mnist.yaml \
    --batch_size 64 --grad_accum_steps 4 --amp

# 梯度检查点（用计算换显存，模型不支持时会明确提示而不是静默忽略）
python src/main.py --config configs/mnist.yaml --gradient_checkpointing

# 纯命令行（不用配置文件）
python src/main.py --exp_name quicktest --epochs 1
```

### 3. 显存账对照实验

```bash
# 7 个变体全跑（约 6 分钟）
python experiments/exp_memory_accounting.py --epochs 2

# 只改了报告文案、想重出报告时用（复用上次结果，不重新训练）
python experiments/exp_memory_accounting.py --report_only

# 梯度检查点的单步拆解（显存 + 耗时）
python experiments/exp_checkpoint_granularity.py
```

输出 `outputs/exp_memory/memory_accounting.md` 与 `outputs/exp_ckpt_granularity/checkpoint_granularity.md`。

### 4. 跑测试

```bash
pip install -r requirements-dev.txt
pytest                                    # 79 个用例，CPU 上约 1 分钟
ruff check src experiments tests          # lint

# 不想等下载？用内置的合成数据集跑通整条流水线
python src/main.py --dataset synthetic --epochs 1
```

测试**不需要 GPU、不需要下载任何数据**（内置 `--dataset synthetic` 离线合成数据集），
任何机器上都能跑 —— CI 就是这么跑的。

## 测试覆盖什么

| 文件 | 覆盖 |
|------|------|
| `tests/test_config.py` | 三级配置优先级、参数校验、YAML 往返 |
| `tests/test_data.py` | 验证集切分无重叠、meta 与模型匹配、合成数据真的可分 |
| `tests/test_model.py` | 输入尺寸推算、参数量、**检查点不改变梯度**、BN 双倍更新（canary） |
| `tests/test_train.py` | LR 调度取值、**更新次数 = ⌈批次数 / 累积步数⌉**、`no_grad` 生效、loss 真的会降 |
| `tests/test_smoke.py` | 真实 CLI 端到端跑通 + 产物落盘 + 命令行覆盖生效 + **非 UTF-8 输出编码下中文日志不崩** |

两条值得单独说的测试思路：

- **合成数据集必须"可学习"**。如果造出来的假数据连最近邻都分不开，"loss 下降"这类断言就没有意义 ——
  所以专门有一条测试验证它类间可分（最近邻准确率 > 0.9）。
- **BN 双倍更新写成 canary 测试**。它断言的是**现状**而非"正确行为"，这是故意的：
  上游哪天修好了，测试会告警，提醒把 README 一起改掉，而不是让文档悄悄过时。

### 写测试当场抓到的四个真 bug

这轮加测试不是走过场，跑起来立刻暴露了四个问题（第四个是 CI 抓的）：

1. **`small_cnn` 的全连接层输入维度写死 7×7** → `--dataset cifar10`（32×32 池化两次是 8×8）
   会直接 shape mismatch 崩掉。README 一直宣称支持 CIFAR-10，实际跑不通。
   已改为按 `image_size` 推算。
2. **`resolve_device()` 只看 `torch.cuda.is_available()`** → 容器 / CI 里"驱动可见但没暴露设备"
   时（`is_available()` 返回 True 而 `device_count()` 是 0）会误判成 cuda，
   随后 `get_device_name(0)` 抛 `Invalid device id`。已改为两个条件都查，且探测异常时退回 CPU。
3. **`--amp --device cpu` 会抛异常**，而不是走那段"优雅关闭混合精度"的提示 ——
   因为 config 的校验在合并配置时就先炸了，提示代码根本不可达。已改为预先化解冲突。
4. **中文日志在英文 Windows 上直接把训练带崩**。Python 输出到**管道**时用系统 locale 编码
   （英文系统是 cp1252），`print("训练完成")` 抛 `UnicodeEncodeError: 'charmap' codec ...`，
   exit code 1。这个 bug **本地永远复现不了**：终端自己会处理编码，而且本机是 cp936 能编码中文，
   所以当时 78 个测试全绿、只有 CI 的 windows job 红。
   修法是在 `main()` 最开头调 `force_utf8_stdout()`，并把管道编码钉成 UTF-8。

   顺带说，第 4 个 bug 也催生了一条特殊测试：`test_chinese_log_survives_non_utf8_stdout`
   用 `PYTHONIOENCODING=cp1252` 起子进程，**等价复现 CI 的 locale**。
   验证时我特意把修复摘掉跑了一遍，确认它确实会红 —— 测试必须能抓到 bug 才算测试。
   > 面试点：这类"本地绿、CI 红"的故障，根因通常是**环境差异**（locale / 编码 / 路径分隔符 /
   > 大小写敏感），而不是逻辑。定位手法是对着 CI 的环境变量在本地重建同等条件。

## 实测基准

**硬件**：RTX 3060 Laptop 6GB | **数据集**：MNIST | **模型**：small_cnn（421,834 参数）

### 基准性能

| 配置 | 测试准确率 | 耗时 | 峰值显存 |
|------|-----------|------|---------|
| 3 epoch / batch=128 / AdamW / cosine | **~98.8%** | ~45s | 0.08 GB |
| 1 epoch / batch=128（冒烟测试） | 98.52% | 16.0s | 0.08 GB |

### 显存账对照实验（2 epoch / 等效 batch 256）

| 变体 | micro batch | 累积步数 | 检查点 | 等效 batch | 峰值显存 | 相对基准 | 测试准确率 |
|------|------------|---------|-------|-----------|---------|---------|-----------|
| A. 基准 | 256 | 1 | — | 256 | 0.138 GB | — | 0.9867 |
| **B. 梯度累积** | **64** | **4** | — | **256** | **0.052 GB** | **-62%** | 0.9870 |
| C. 混合精度 | 256 | 1 | — | 256 | 0.129 GB | -7% | 0.9876 |
| D. 梯度检查点 | 256 | 1 | ✅ | 256 | 0.127 GB | -8% | 0.9862 |
| E. 累积 + 检查点 | 64 | 4 | ✅ | 256 | 0.053 GB | -62% | 0.9870 |

**四条结论**：

1. **梯度累积是最划算的一招**：等效 batch 不变、显存降 **62%**，几乎零代价。
   显存大头是**激活值**，正比于单次前向的样本数；累积每次只前向 64 个，激活只需 ¼。
2. **混合精度降 7%** —— 这个数字偏小是小模型的必然：显存基数只有 0.1GB 量级，
   激活值本来就不占主导。模型越大、激活占比越高，收益越接近理论值（激活直接减半）。
3. **梯度检查点只降 8%** —— 不是实现有问题，原因见下节，是个值得单独理解的反直觉点。
4. **三种手段可以叠加**（E），且准确率均不受影响（各变体差异 ≤0.001）。

### 梯度检查点为什么只省 8%？

`exp_checkpoint_granularity.py` 把单个训练 step 拆成三个时间点量（batch=256）：

| 粒度 | 前向保活的激活 | 前向峰值 | 全程峰值 | 完整 step 耗时 |
|------|--------------|---------|---------|--------------|
| 不开检查点 | 111 MB | 117 MB | 138 MB | 5.5 ms |
| **按 block 检查点** | **19 MB** | **51 MB** | 119 MB | 8.3 ms |
| 整条主干当一个段 | 13 MB | 51 MB | 138 MB | 8.0 ms |

**三个反直觉的点**：

- **省的是「前向保活的激活」，不是「峰值」**。检查点把保活量砍掉 83%、前向峰值腰斩，
  但训练全程峰值由**反向阶段**决定 —— 重算会把激活重新物化出来，卷积反向的 workspace 也照付。
  所以净收益只剩 13%。**激活「驻留规模」≠ 峰值「瞬时规模」。**
- **粒度选错，收益直接归零**。把整条主干当成一个检查点段时，没有边界可以逐段释放，
  反向重算会一次性全量物化 —— 峰值与不开检查点持平。所以粒度要落在「块」上（Transformer 的一层 / CNN 的一个卷积单元）。
- **耗时要测完整 step**。前向耗时几乎没变（重算发生在反向阶段，前向计时里看不到），
  完整 step 5.5 → 8.3 ms（+40~50%），增量正好是一次前向的量级 ——
  **检查点的代价基本就是「多跑一次前向」。**

> **判据是「激活值占总显存的比例」，不是模型大小。**
> 把 micro batch 从 256 放大到 1024（激活值 ×4），检查点收益从 8% 涨到 **14%**（见实验里的 F/G 变体）。

### ⚠️ 一个被实测抓出来的坑：检查点会让 BatchNorm 统计量更新两次

A 与 D 唯一差别是检查点开关（同 seed、同数据顺序、同超参），逐层验证：

| 检查项 | 结果 |
|--------|------|
| loss | 完全相同（小数点后 8 位一致） |
| 全部参数梯度 | **逐元素完全相同**（maxdiff = 0.0） |
| `num_batches_tracked` | 1 vs **2** ❌ |
| `running_mean` | 偏离约 0.024 ❌ |

检查点在反向时会把 block 重新前向一遍来重算激活。**重算本身是精确的**（所以梯度逐元素一致），
但训练态 BatchNorm 的 forward **带副作用** —— 它用当前 batch 的统计量更新 running stats，
重算一次就多更新一次。梯度没事是因为训练态 BN 归一化用的是 batch 统计量；
但 eval 用的是 running stats，统计量偏了指标就跟着偏（本例约 -0.05 pp）。

**为什么 LLM 训练开检查点毫无顾虑**：Transformer 用 LayerNorm，没有 running stats，
重算是纯函数。**有状态层才是问题所在。**

缓解手段：块内改用无状态归一化（GroupNorm / LayerNorm），或把 BN 排除在检查点块之外。

### 显存不够时的决策阶梯

| 顺序 | 手段 | 实测省多少 | 额外代价 |
|------|------|-----------|---------|
| 1️⃣ | **梯度累积** | **62%** | 几乎为零 —— 默认先上 |
| 2️⃣ | **混合精度** | 7%（大模型更高） | 需 GradScaler |
| 3️⃣ | **梯度检查点** | 8%（大 batch 下 14%） | step 慢 40~50% |
| 4️⃣ | 组合使用 | 62% | 真的塞不下时的兜底 |

关键认知：训练显存 = 参数 + 梯度 + 优化器状态 + **激活值**。
前三项与 batch size 无关，只有激活值随 batch 线性增长 —— 所以它通常是最大头，
也是上面三招**唯一能打**的目标。再往上还有 8-bit 优化器、ZeRO / FSDP、CPU offload，
但那些是「实在不够」的手段，前三级能解决就别上。

> MNIST 显存基数只有 0.1GB 量级，绝对差值看着不大但**比例与趋势是可信的**。
> 生产级场景（ImageNet / Transformer）显存基数在 GB 量级，同样的比例就是省几个 GB。

## 核心代码位置（想学就看这几个文件）

| 想学什么 | 看哪里 |
|---------|-------|
| 训练 step 的完整顺序 | `src/train.py` 的 `train_one_epoch()` |
| 显存优化的每一项 | `src/train.py` 模块 docstring + `src/main.py` 的 `build_optimizer()` |
| **梯度检查点怎么实现、坑在哪** | `src/model.py` 的 `SmallCNN.forward()` docstring |
| 配置怎么做到可复现 | `src/config.py` 的 `merge()` / `to_yaml()` |
| 为什么必须切 `train()`/`eval()` | `src/train.py` 的 `evaluate()` |
| 验证集为什么必须切 | `src/data.py` 的 `split_train_val()` |

## 关键概念速查（面试自测）

**Q1: `optimizer.zero_grad()` 为什么必须在 `backward()` 之前？**
PyTorch 的 `.backward()` 是**累加**梯度到 `.grad`，不是覆盖。不清零则第二个 batch 的梯度叠在第一个上，等效用了错误的梯度。

**Q2: 梯度累积 4 次，等效 batch size 怎么算？显存和 batch=4x 一样吗？**
等效 batch = `batch_size × accum_steps`。显存**不一样**——激活值只占 1/4，因为每次只前向 micro batch。注意 BN 统计量按 micro batch 算，大差异时可能需要 SyncBN。

**Q3: `loss.backward()` 后能立刻 `loss.item()` 吗？**
能，但要先 `detach()` 或用 `with torch.no_grad()`，否则保留计算图。训练循环里用 `loss.item()` 是安全的（返回 Python 标量，不持图）。

**Q4: `model.train()` / `model.eval()` 切换了什么？**
- `train()`：Dropout 生效（随机置零）、BN 用 batch 统计量并更新 running stats
- `eval()`：Dropout 关闭、BN 用 running stats
忘切 eval 会让验证结果随机抖动；忘切回 train 会让 BN 停止更新。

**Q5: 混合精度里 `GradScaler` 解决什么问题？**
fp16 动态范围小（最小正规数 ~6e-5），小梯度会下溢成 0。GradScaler 把 loss 放大若干倍再反向，梯度同步放大后不会下溢；更新前再 `unscale_` 还原。若检测到 inf/nan 则跳过该步并降低 scale。

**Q6: `clip_grad_norm_` 为什么要在 `unscale_` 之后？**
AMP 下梯度被放大过，直接裁剪相当于裁的是放大后的值，阈值失去意义。必须先还原成真实尺度再裁。

**Q7: AdamW 的 W 是什么意思？**
Decoupled weight decay——权重衰减不参与 Adam 的动量/二阶矩计算，直接作用于参数。比把 L2 正则加进 loss 更干净（后者会被自适应学习率缩放，导致大梯度参数被过度正则）。

**Q8: warmup 为什么需要？**
训练初期参数随机、梯度方向噪声大，大 lr 会破坏预训练权重或让模型发散。warmup 让 lr 从 0 线性升到目标值，平滑起步。对 Transformer 尤其重要。

**Q9: 梯度检查点省的是哪部分显存？为什么"省了 83% 激活"却只让峰值降 13%？**
省的是**前向保活的中间激活**（反向要用、必须存到反向结束的那些张量）。但因为：
① 反向重算时激活会被**重新物化**，那一刻照样占显存 —— 省掉的是激活"同时活着"的时间跨度，不是"总分配量"；
② 反向还有与检查点无关的开销（卷积反向的 cuDNN workspace）。
**峰值由反向阶段的瞬时规模决定**，所以收益会大幅缩水。判据是"激活值占总显存的比例"，不是模型大小。

**Q10: 检查点的粒度为什么不能随便选？**
检查点靠**边界**把激活切段、逐段释放。粒度太细（每层都 checkpoint）→ 每层都要存自己的输入，等于没省；
粒度太粗（整条主干当一个段）→ 反向重算时一次性全量物化，没有释放点，收益直接归零（实测峰值与不开检查点持平）。
正确粒度是"块"：Transformer 的一层、CNN 的一个卷积单元。

**Q11: 检查点对 BatchNorm 有什么影响？**
会**双倍更新 running stats**。因为重算时 block 又被前向了一遍，而训练态 BN 的 forward 带副作用（用当前 batch 统计量更新 running stats）。
注意：**梯度不受影响**（逐元素完全一致，实测 maxdiff = 0.0），因为训练态 BN 归一化用的是 batch 统计量，与 running stats 无关；
但 eval 会用 running stats，所以指标会偏（本例约 -0.05 pp）。
`use_reentrant=False` 能修好 Dropout 的 RNG 状态问题，**但挡不住这个**。缓解：块内换 GroupNorm / LayerNorm，或把 BN 排除在检查点块外。
→ 这也解释了为什么 LLM 训练开检查点毫无顾虑：Transformer 用 LayerNorm，没有 running stats，重算是纯函数。

**Q12: 评估"梯度检查点慢多少"时，测前向耗时够吗？**
不够，会严重低估。重算发生在**反向阶段**，前向计时里看不到它 —— 实测前向耗时几乎没变，但完整 step 从 5.5ms 涨到 8.3ms（+40~50%）。
反过来，用训练脚本的总耗时（含数据加载与验证）又会把差异稀释到看不见。
**必须把完整 step 单独拎出来计时。** 结论：检查点的代价基本等于「多跑一次前向」。

## 后续可扩展

- [x] `gradient checkpointing` 演示（用计算换显存的量化对比）
- [ ] 分布式训练（DDP）最小可跑示例
- [ ] `torch.compile` 加速对比
- [ ] 学习率 finder（LR range test）

## License

MIT
