# PyTorch 训练模板

一套**手写、不含高层封装**的 PyTorch 训练脚手架。目的是把训练循环的每个细节讲清楚——而不是调一个 `Trainer` 就完事。

> 为什么手写？面试问的是 `optimizer.zero_grad()` 为什么必须在 `backward()` 之前、梯度累积怎么省显存、`GradScaler` 解决什么问题。这些被封装的 API 挡住了。

## 特性

| 能力 | 说明 |
|------|------|
| **配置系统** | YAML + dataclass，命令行可覆盖任意参数；配置随 checkpoint 一起保存，实验可复现 |
| **手写训练循环** | 不用 Lightning/Trainer，每一步（清零→前向→反向→裁剪→更新）都显式写出 |
| **梯度累积** | 小显存跑等效大 batch，显存不增长 |
| **混合精度** | `torch.autocast` + `GradScaler`，含梯度还原时机 |
| **手写 LR 调度** | warmup + cosine/step，能看到 lr 每一步怎么变（不是黑盒 `scheduler.step()`） |
| **梯度裁剪** | 按全局 L2 范数裁剪，防梯度爆炸 |
| **显存监控** | 每 epoch 打印当前/峰值显存 |
| **checkpoint** | 保存最优权重 + 完整训练状态（含配置），支持早停 |

## 目录结构

```
pytorch-training-template/
├── src/
│   ├── config.py      # 配置系统：dataclass + YAML + 命令行覆盖
│   ├── data.py        # 数据加载：Dataset / DataLoader / transform
│   ├── model.py       # 模型定义 + 注册表（small_cnn / mlp）
│   ├── train.py       # ⭐ 训练循环核心：手写 step / 评测 / LR 调度
│   └── main.py        # 入口：argparse + 日志 + checkpoint + 显存统计
├── configs/
│   └── mnist.yaml     # MNIST 标准配置
├── experiments/
│   └── exp_memory_accounting.py   # 显存账对照实验
├── outputs/           # 训练产物（git 忽略）
└── requirements.txt
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

> 国内网络可用镜像加速，见 `docs/` 或 diffusers 项目的踩坑记录。

### 2. 训练

```bash
# 用配置文件
python src/main.py --config configs/mnist.yaml

# 覆盖参数（命令行优先级最高）
python src/main.py --config configs/mnist.yaml --epochs 5 --lr 5e-4

# 混合精度 + 梯度累积
python src/main.py --config configs/mnist.yaml \
    --batch_size 64 --grad_accum_steps 4 --amp

# 纯命令行（不用配置文件）
python src/main.py --exp_name quicktest --epochs 1
```

### 3. 显存账对照实验

```bash
python experiments/exp_memory_accounting.py --epochs 2
```

输出 `outputs/exp_memory/memory_accounting.md`，用实测数据说明梯度累积和混合精度各能省多少显存。

## 实测基准

**硬件**：RTX 3060 Laptop 6GB | **数据集**：MNIST | **模型**：small_cnn（421,834 参数）

### 基准性能

| 配置 | 测试准确率 | 耗时 | 峰值显存 |
|------|-----------|------|---------|
| 3 epoch / batch=128 / AdamW / cosine | **~98.8%** | ~45s | 0.08 GB |
| 1 epoch / batch=128（冒烟测试） | 98.52% | 16.0s | 0.08 GB |

### 显存账对照实验（2 epoch / batch=256 等效）

| 变体 | micro batch | 累积步数 | 等效 batch | 峰值显存 | 测试准确率 |
|------|------------|---------|-----------|---------|-----------|
| A. 基准 | 256 | 1 | 256 | 0.138 GB | 0.9867 |
| **B. 梯度累积** | **64** | **4** | **256** | **0.052 GB** | **0.9869** |
| C. 混合精度 | 256 | 1 | 256 | 0.129 GB | 0.9876 |

**结论**：
1. **梯度累积省显存 62%**（0.138 → 0.052 GB），等效 batch 和准确率都不变。
   原因：显存大头是**激活值**，正比于单次前向的样本数。梯度累积每次只前向 64 个样本，攒 4 次梯度再更新，数学上等效 batch=256，但激活值只需 1/4。
2. **混合精度省显存 7%**（0.138 → 0.129 GB）——MNIST 模型太小，效果不明显。大模型上激活值占主导时收益显著。
3. **准确率不受影响**：三种配置差异在 0.001 以内，省显存不以牺牲效果为代价。

> MNIST 太简单，显存基数本来就小（0.1GB 量级），绝对差值看起来不大但**比例很有说服力**。
> 生产级场景（ImageNet / Transformer）显存基数在 GB 量级，同样的比例就是省几个 GB。

## 核心代码位置（想学就看这三个文件）

| 想学什么 | 看哪里 |
|---------|-------|
| 训练 step 的完整顺序 | `src/train.py` 的 `train_one_epoch()` |
| 显存优化的每一项 | `src/train.py` 模块 docstring + `src/main.py` 的 `build_optimizer()` |
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

## 后续可扩展

- [ ] `gradient checkpointing` 演示（用计算换显存的量化对比）
- [ ] 分布式训练（DDP）最小可跑示例
- [ ] `torch.compile` 加速对比
- [ ] 学习率 finder（LR range test）

## License

MIT
