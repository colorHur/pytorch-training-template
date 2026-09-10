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
| **分布式训练 (DDP)** | 同一份 `main.py` 单卡/多卡通用（`torchrun` 自动识别）；数据分片、梯度平均、指标归约、rank0 独占落盘；含不依赖 TCPStore 的启动器 |
| **学习率 finder** | LR range test：训练前扫一遍 lr 报告该用多少；等比取点 + 偏差修正 EMA + 权重零污染还原，输出 ASCII 曲线 |
| **测试 + CI** | 190 个 pytest 用例（CPU 可跑，多平台 CI）；离线合成数据集，秒级验证整条流水线 |

## 目录结构

```
pytorch-training-template/
├── src/
│   ├── config.py      # 配置系统：dataclass + YAML + 命令行覆盖
│   ├── console.py     # 零依赖的控制台工具（force_utf8_stdout）
│   ├── data.py        # 数据加载：Dataset / DataLoader / transform + 分布式切分
│   ├── model.py       # 模型定义 + 注册表（small_cnn / mlp）+ 梯度检查点
│   ├── distributed.py # ⭐ DDP：进程组 / 数据切分 / 指标归约 / 模型包装
│   ├── lr_finder.py   # ⭐ LR range test：等比扫描 / 稳健选点 / 权重快照还原
│   ├── compile_support.py # torch.compile 的平台探测 + 冒烟 + 优雅降级
│   ├── train.py       # ⭐ 训练循环核心：手写 step / 评测 / LR 调度 / 优化器
│   └── main.py        # 入口：argparse + 日志 + checkpoint + 显存统计 + DDP
├── configs/
│   └── mnist.yaml     # MNIST 标准配置
├── experiments/
│   ├── exp_memory_accounting.py       # 显存账对照实验（7 个变体）
│   ├── exp_checkpoint_granularity.py  # 梯度检查点单步拆解（显存 + 耗时）
│   └── exp_ddp_equivalence.py         # DDP 等价性与吞吐实测
├── tools/
│   ├── ddp_launch.py      # 用 FileStore 启动 DDP（绕开 TCPStore，Windows 也能跑）
│   ├── ddp_probe.py       # DDP 环境自检（能力门禁：环境不支持就明确 skip）
│   ├── compile_probe.py   # torch.compile 环境自检（探测 + 真编译一次，退出码 0/1/2）
│   └── lr_finder.py       # 学习率扫描的命令行入口
├── tests/             # pytest：配置 / 数据 / 模型 / 训练循环 / 分布式 / LR finder / 端到端 / CI 元测试
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

# 分布式训练（多进程）—— 同一份代码，torchrun 会自动识别
torchrun --nproc_per_node=4 src/main.py --config configs/mnist.yaml

# 纯命令行（不用配置文件）
python src/main.py --exp_name quicktest --epochs 1
```

### 3. 学习率该给多少：先扫一遍

```bash
# 扫 100 步（MNIST 上约 16 秒），输出建议 lr + ASCII 曲线
python tools/lr_finder.py --dataset mnist --steps 100

# 顺便和当前的 lr 比一比（差多少倍会直接告诉你）
python tools/lr_finder.py --config configs/mnist.yaml --lr 1e-3

# 不下载数据也能试（合成数据只有 8 个 batch，会提示"已循环复用"）
python tools/lr_finder.py --dataset synthetic --steps 40
```

输出 `outputs/lr_finder/lr_sweep.md`（结论 + 曲线）与 `lr_sweep.json`（完整数据），
产物目录可用 `--output_dir` 改。

### 4. 显存账对照实验

```bash
# 7 个变体全跑（约 6 分钟）
python experiments/exp_memory_accounting.py --epochs 2

# 只改了报告文案、想重出报告时用（复用上次结果，不重新训练）
python experiments/exp_memory_accounting.py --report_only

# 梯度检查点的单步拆解（显存 + 耗时）
python experiments/exp_checkpoint_granularity.py

# DDP 等价性与吞吐（数学部分秒级；多进程计时约 1 分钟）
python experiments/exp_ddp_equivalence.py --math_only
python experiments/exp_ddp_equivalence.py
```

输出 `outputs/exp_memory/memory_accounting.md`、`outputs/exp_ckpt_granularity/checkpoint_granularity.md`
与 `outputs/exp_ddp/ddp_equivalence.md`。

### 5. 跑测试

```bash
pip install -r requirements-dev.txt
pytest                                    # 190 个用例，CPU 上约 2 分钟
ruff check src experiments tests tools    # lint

# 不想等下载？用内置的合成数据集跑通整条流水线
python src/main.py --dataset synthetic --epochs 1

# 手动起 2 个进程试 DDP（用 FileStore，不需要 torchrun）
python tools/ddp_launch.py --nproc_per_node 2 -- src/main.py --dataset synthetic --epochs 1 --device cpu
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
| `tests/test_distributed.py` | **DDP 梯度等价性（含 BN 反例）**、各 rank 切分不重不漏、`set_epoch` 真的换了顺序、`ctx=None` 判空约定 |
| `tests/test_compile.py` | 平台探测（用 `platform_name` 让 Windows 分支在 Linux CI 上也能测）、冒烟不污染 BN、`unwrap_model` 剥嵌套包装、平台限制（canary） |
| `tests/test_lr_finder.py` | 等比取点、**EMA 的偏差修正**、最陡下降选点、**扫描后权重逐位还原**、该关的开关都关了、结果可复现 |
| `tests/test_smoke.py` | 真实 CLI 端到端跑通（`src/main.py` 与 `tools/lr_finder.py`）+ 产物落盘 + 命令行覆盖生效 + **非 UTF-8 输出编码下中文日志不崩** + 入口清单完整性 |
| `tests/test_ci_annotations.py` | 诊断通道本身：`_emit` 编码无关性、hook 触发与转义、端到端断言退出码是 1 而不是 3 |

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

### 加 DDP 时，测试又挡下来三次

同一套测试在**写新功能时**的价值更明显 —— 它挡住的不只是别人的 bug：

1. **`SyncBatchNorm` 只在 CUDA 上可用**。`convert_sync_batchnorm()` 只换模块类型、
   **不会报错**，但换成之后 CPU 上第一次前向就抛
   `ValueError: SyncBatchNorm expected input tensor to be on GPU or XPU`。
   也就是说「CPU 多进程 + `--sync_bn`」会表现为"配置看起来生效了、训练第一步才崩"。
   已改为按 backend 判定：只有 nccl 才允许转换，gloo 下明确提示并忽略。
2. **`ctx=None` 把 12 个已有测试一次性打挂**。`train.py` 的公开 API 约定 `ctx=None` = 单进程，
   但新写的 `ddp_no_sync()` 里是裸的 `ctx.enabled` → `AttributeError`，
   而且报错发生在**训练循环中间**。修法不是就地补判空，而是抽出 `is_enabled(ctx)`
   把判定收敛到一处，再补一条 `test_all_helpers_accept_none_ctx` 把这个约定钉死。
3. **`evaluate()` 差点把空 loader 的报错吃掉**。为了给 `total` 做类型转换顺手写了
   `max(1.0, total)` —— 于是空 loader 从"抛 `ZeroDivisionError`"变成"静默返回 acc=0"。
   已有的 `test_evaluate_on_empty_loader_raises_rather_than_silently_returning_garbage`
   立刻变红。静默返回 0 会被误读成"模型全错"，比崩溃难查得多，已改回。

> 这三条的共性是：**都不会在"跑一下试试"时暴露**。第 1 条要跑到 CPU 多进程才炸，
> 第 2 条只在特定调用路径上炸，第 3 条最隐蔽 —— 它"看起来能跑"，只是结果悄悄错了。

### 同一个坑踩了两次 —— 于是我把它变成了测试

上面第 4 个 bug（中文日志在 cp1252 下崩）修完之后，我以为这事翻篇了。
结果**加 DDP 时，新写的 `tools/ddp_launch.py` 又犯了同一个错**：它的 `main()`
里直接 `print(f"[ddp_launch] 运行：...")`，没调 `force_utf8_stdout()`。
同一个坑、不同的文件，中间只隔了一次提交。

它一次带出三个症状：

1. 两条 DDP 冒烟测试变红（它们都会去调这个启动器）；
2. CI 的 windows job 连着红了 **四轮**，而我从外面只能看到一句
   "Process completed with exit code 1" —— 因为 Actions 的日志接口要鉴权；
3. **连累了我自己搭的排错通道**。为了绕过第 2 点，我加了个 pytest hook，把失败
   提升成 check annotation（这个接口匿名可读）。结果这个 hook 用 `print()` 输出
   含中文的 title，在**同一个 cp1252 环境**下抛 `UnicodeEncodeError` —— pytest
   直接 INTERNALERROR，退出码从 1 变成 3。
   **诊断工具自己成了新的故障源，比不做诊断更误导。**

最后是「在本地重建 CI 的环境」破的局：

```bash
PYTHONIOENCODING=cp1252 CUDA_VISIBLE_DEVICES="" pytest tests/ -q
```

两条 DDP 测试立刻复现，真因一行就跳出来了。顺带解释了那个一直想不通的现象：
在**纯 cp1252** 下 hook 崩掉是退出码 **3**，而 CI 上是 **1** —— 说明 CI 上
hook 其实没崩，注解是被 GitHub **丢弃**的（原因是下面第三条）。

**三条教训，都固化成了代码**：

- **「下次记得」靠不住，把不变式写成测试**。现在
  `test_entrypoint_pins_utf8_stdout` 会扫描**全部可执行入口**，少一个就红。
  写这条测试时还发现第一版断言太松 —— 只查函数名，而
  `from console import force_utf8_stdout` 这种「导入但没调用」也能骗过它，
  于是收紧成必须出现**带括号的调用**。
  顺带把实现从 `src/__init__.py` 挪进了零依赖的 `src/console.py`：`import src`
  要 **6.4 秒**（连锁导入 torch），而 `tools/` 下的启动器不该为一行工具函数付这个代价。
- **然后「清单」自己漂了**。加 `tools/compile_probe.py` 时忘了往入口清单里加一行 ——
  那个入口就变成「**看起来被不变式守着，实际上没被扫**」，比完全没有这条测试更危险。
  修法不是在清单里补一行就完事，而是加一条
  `test_every_executable_entrypoint_is_covered`：扫出仓库里所有带 `__main__` 块的文件，
  断言它们和清单**完全一致**。现在「忘了加一行」会变成一次明确的失败
  （报 `漏了 ['tools/lr_finder.py']`），而不是悄悄少覆盖一个入口。
- **诊断通道自己也要有测试**。`tests/test_ci_annotations.py` 用 12 条用例从三层
  守它：`_emit` 的编码无关性、hook 的触发条件与输出格式，以及一条**端到端**用例
  真的在 cp1252 下起一个 pytest，断言退出码必须是 **1 而不是 3**。
  （修完还在真实 CI 上做了一次端到端验证：故意让一条测试失败，确认 check annotation
  里真的出现了带中文 title、带断言详情、且归属到正确文件的那条注解 —— 而不是只让
  本地的模拟环境"看起来能工作"。验证完立刻回滚。）
- **工作流命令的格式坑**：`::error file=...,title=...::message` 里的 `,` 和 `:`
  必须转义成 `%2C` / `%3A`，`%` 要写成 `%25`，超长会被**静默丢弃**。
  最容易踩的是 `title` —— 我往里塞了 nodeid，而 nodeid 长这样
  `tests/test_x.py::test_y`，里面的 `::` 会把属性段直接截断，
  于是**注解发得出去、却一条都显示不出来**。

> 面试点：这类问题的价值不在"修了一个 bug"，而在**把一次性修复变成永久的不变式**。
> 同一个坑出现第二次时，正确的动作不是再修一次，而是问"第一次修完为什么没防住第二次"——
> 答案通常是：**修的是那个点，没修那类事**。

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

## 分布式训练（DDP）

同一份 `main.py`，单进程和多进程都能跑 —— 靠 `torchrun` 写进环境变量的
`RANK` / `LOCAL_RANK` / `WORLD_SIZE` 自动识别，**不另起 `train_ddp.py`**：

```bash
# 单卡 / CPU（什么都不用改）
python src/main.py --dataset synthetic --epochs 3

# 多卡
torchrun --nproc_per_node=4 src/main.py --epochs 30

# 本仓库自带的启动器（用 FileStore 做 rendezvous，不依赖 TCPStore）
python tools/ddp_launch.py --nproc_per_node 2 -- src/main.py --epochs 3
```

### DDP 的全部数学内容，就一条等式

```
grad(batch 2B)  ==  ( grad(batch B on rank0) + grad(batch B on rank1) ) / 2
```

实测（`experiments/exp_ddp_equivalence.py`，batch=128 切成两个 64）：

| 模型 | 有 BatchNorm | 最大绝对偏差 | 相对偏差 |
|------|:---:|---:|---:|
| `mlp` | 否 | **9.3e-09** | 5.3e-06 |
| `small_cnn` | 是 | **4.3e-03** | **1.31** |

`mlp` 的偏差落在 float32 浮点误差量级，等式成立；`small_cnn` 的**相对偏差超过 1**
（比梯度本身的尺度还大），等式彻底不成立。

**为什么 BatchNorm 破坏了它**：BN 在训练态用**当前 batch 的统计量**做归一化 ——
它在前向里"看得见整批数据"，前向不再逐样本独立。单进程 batch 128 的统计量在 128 条上算，
两个 rank 各自在 64 条上算，归一化结果不同，梯度自然不同。

**DDP 不会帮你同步这个。** `broadcast_buffers=True`（默认）只是每次前向开始把 rank 0 的
buffer 广播出去，并没有让统计量变准。要真正等价得换 `SyncBatchNorm`——
但它**只在 CUDA 上可用**（CPU 前向直接抛 `ValueError: SyncBatchNorm expected input tensor
to be on GPU or XPU`），所以本仓库按 backend 判定，gloo 下直接拒绝转换并给出提示。

> 顺带解释了为什么大家对这个坑不敏感：Transformer 用 LayerNorm，没有 running stats。
> **有状态层才是问题所在**（和梯度检查点那个坑同源）。

### CPU 上跑 DDP 会变慢 —— 而且这不是 bug

全局 batch 都对齐到 128，6 个 epoch（CPU + gloo，2 进程）：

| 配置 | 进程数 | 每进程 batch | 训练耗时 | 每 epoch | 相对单进程 |
|------|:---:|---:|---:|---:|---:|
| 单进程 | 1 | 128 | 2.85s | 0.475s | 1.0× |
| DDP 2 进程 | 2 | 64 | 6.72s | 1.12s | **0.42×** |

三个原因：① 模型只有 42 万参数，单步计算量微秒级，而每次 all-reduce 都要走 gloo 的
socket 通信，通信成本远超计算节省；② CPU 没有算力冗余，2 个进程抢同一批核；
③ gloo 是为正确性和兼容性设计的，不是为速度（真多卡用 nccl）。

**DDP 真正解决的问题是「放不下」，不是「算得慢」**：全局 batch 大到单卡显存装不下、
或模型+优化器状态超出单卡显存时，才需要它。换句话说：
**先用梯度累积 / 混合精度把单卡榨干，再考虑 DDP。**

### 三个必踩的坑（都写进了测试）

| 坑 | 症状 | 处理 |
|---|---|---|
| 忘了 `set_epoch(epoch)` | 每个 epoch 的 shuffle 顺序完全一样，等于没打乱 | 每个 epoch 开头调一次 |
| 各 rank 的 batch 数不相等 | **死锁** —— 快的等慢的，没有任何报错，训练就这么卡住 | 训练集用 `drop_last=True` 的 `DistributedSampler` |
| 存 checkpoint 没 unwrap | key 变成 `module.blocks.0.0.weight`，单进程加载不了 | 存之前 `unwrap_model()` |

另外两条只影响性能/精度、不会报错的：梯度累积时不用 `model.no_sync()`（结果对，但白通信
`accum` 倍）；以及只有 rank 0 能写日志和 checkpoint（N 个进程同时写会写坏文件）。

### 关于 `torchrun` 在 Windows 上的坑

标准做法是 `torchrun`，但它要创建一个 **TCPStore**。某些 torch 构建
（尤其是 Windows 版）**没有编进 libuv**，于是直接报：

```
DistStoreError: use_libuv was requested but PyTorch was built without libuv support,
run with USE_LIBUV=0 to disable it
```

更麻烦的是 `USE_LIBUV=0` **也救不回来** —— 这些构建里 Windows 的 TCPStore 只剩 libuv
一条实现路径。但集合通信（gloo 的 socket）本身是好的，所以本仓库的
`tools/ddp_launch.py` 换用 `FileStore`（用一个文件当共享存储）做 rendezvous，
完全绕开 TCP，Windows 和 Linux 都能跑，CI 的两个平台也覆盖了。

> 真实训练环境（Linux + 多卡）请用 `torchrun`：它带弹性容错和更完善的进程管理。

## `torch.compile`：为什么它在 Windows 上通常用不了

`torch.compile(model)` 在 Linux 上基本是一行搞定的加速手段，但在 Windows 上
**两条路都是断的**（都是实测出来的，不是推测）：

| 设备 | 报错 | 根因 |
|------|------|------|
| CUDA | `TritonMissing: Cannot find a working triton installation.` | Inductor 的 GPU 后端靠 **Triton** 生成 kernel，而 PyTorch **不提供 Windows 版 triton wheel** |
| CPU | `InductorError: RuntimeError: Compiler: cl is not found.` | Inductor 的 CPU 后端要把生成的 C++ 编译成动态库，需要 **MSVC 的 `cl.exe`** |

两个报错里没有一个字提到「你缺什么、怎么修」，而且**发生在训练第 1 步**
（`torch.compile()` 本身只做标记，真正的编译是懒的）。

### 本仓库怎么处理

`src/compile_support.py` 把这件事拆成三步，风格与「梯度检查点 / SyncBatchNorm」
一致 —— **能开就开，开不了要说清楚为什么**：

1. **主动探测**（`probe_compile_support`）：训练开始前就判断当前「平台 + 设备」
   能不能用这个后端，给出结构化的原因和退路；
2. **冒烟编译**（`smoke_compile`）：拿真实形状的输入先跑一次前向，把「第 1 步才炸」
   提前到启动阶段，顺便量出**首次编译的开销**；
3. **明确降级**：任何一步失败都回退到未编译的模型并打印原因，**训练照常继续**。

探测和冒烟两件都要做：探测负责把「注定失败」提前拦下，冒烟负责兜住「探测想不到的
失败」—— 前者是推断，后者才是 ground truth，而平台行为是会变的。

### 实测（本机 Windows + RTX 3060）

| 后端 | 是否需要 kernel 生成 | CPU | CUDA |
|------|-------------------|-----|------|
| `inductor`（默认） | 需要 Triton / MSVC | ❌ 缺 `cl.exe` | ❌ 缺 Triton |
| `aot_eager` | 不需要 | ✅ 可用 | ✅ 可用 |
| `cudagraphs` | 不需要 | ✅ 可用 | ✅ 可用 |

自检工具：

```bash
python tools/compile_probe.py --all      # 探测 + 真编译一次，报告缺什么、有什么退路
```

退出码 **0 / 1 / 2** = 「默认后端可用 / 只有退路可用 / 全不可用」，可以直接当门禁用。

> ⚠️ 说清楚，别误导：**`aot_eager` 不是加速手段** —— 它只做图捕获与算子分解、不做
> kernel 生成，通常不快甚至更慢，价值在于「验证图能不能被捕获」。`cudagraphs` 省的是
> kernel launch 开销，只在小模型 + 短 step 上可能有效。真想要 Inductor 的加速，
> 得走 Linux / WSL，或自行装非官方 triton-windows / MSVC。

```bash
# 想试就在命令行上开（本机的 CUDA 上会用 cudagraphs 跑通）
python src/main.py --compile --compile_backend cudagraphs --dataset synthetic --epochs 1
```

> 面试点：`torch.compile` **不等于「一定更快」**。它是懒加载 + **按输入形状特化**的：
> 首次调用要付编译成本（本项目会把它单独打印出来），输入形状一变还要重编译。
> 在小模型 + 短 step 上，编译开销可能比省下的时间还大 —— 所以本仓库把它做成可选项
> 而不是默认打开，这本身就是工程判断的一部分。

## 学习率 finder（LR range test）

### 它在回答什么问题

「学习率给多少」是训练里最贵的超参 —— 猜错了要么震荡发散，要么慢到你以为代码有 bug。
LR range test 的思路很朴素：**先用 100 步把这件事测出来，而不是靠反复重训去试。**

从 `1e-7` 出发**等比**升到 `1.0`，每一步换一个 lr、更新一次参数、记一次 loss，
就得到一条「loss vs log(lr)」曲线：先平、再缓缓下降、然后陡然上升。
取**最低点之前、下降最陡**的那个 lr 作为建议值。

> ⚠️ **不是取 loss 最低的那个点**。最低点往往已经贴着发散边缘，用它训练迟早炸。
> 要的是「性价比最高」的点：斜率最陡 = 每升高一个 lr 数量级，换来的 loss 下降最多。

### 实测（MNIST / small_cnn / AdamW / batch 128 / 100 步）

| 指标 | 值 |
|------|-----|
| **建议 lr** | **1.262e-3**（第 58 步，loss 下降最陡处） |
| 扫描内平滑 loss 最低 | 1.2859（出现在 lr=1.96e-1） |
| 发散点 | lr ≈ 8.50e-1 |
| 本模板默认的 lr | 1e-3 —— 与建议值差 **0.79×**（同量级） |
| 整个扫描的耗时 | **16 秒**（CPU，100 步） |

```
  （横轴对数刻度，纵轴 loss；X = 建议的 lr）
    5.494 |                                                   .|
    5.073 |                                                    |
    4.652 |                                                    |
    4.231 |                                                    |
    3.811 |                                                    |
    3.390 |                                                    |
    2.969 |                                                    |
    2.548 |                                                  . |
    2.127 |..............................                      |
    1.707 |                              X.....              . |
    1.286 |                                    ..............  |
          +----------------------------------------------------+
           1.0e-07                                      8.5e-01
```

这条曲线有两个可读的信息：**左边那段平坦**说明 lr 太小时模型几乎没动（1e-7 到 1e-5
的 loss 完全一样），**右边那个陡升**就是发散。X 落在两者之间、斜率最陡的地方。

> 顺带验证了一件事：本模板默认的 `lr=1e-3` 和扫描建议值只差 0.79 倍 ——
> 一个经验默认值和一个实测值碰上了，说明这个默认选得不离谱。
> （但注意：这个数**不是模型的性质**，是「这份配置 + 这份数据」的性质，换了就要重扫。）

### 三个做错就会「静默给出错误建议」的细节

这个模块最危险的地方在于：**它错了也照样画出一条漂亮的曲线**。所以三条都写了测试：

| 细节 | 做错的后果 | 为什么 |
|------|-----------|--------|
| 平滑要用**带偏差修正**的 EMA | 「最陡下降」落在曲线开头，**建议值退化成 min_lr** | 朴素 EMA 的初值是 0，`out[0] = (1-β)·loss₀ ≈ 0.02·loss₀`，前几步被人为压低。除以 `1-β^(t+1)` 才是无偏的（测试断言 `out[0] == losses[0]`） |
| 扫描时必须**关掉梯度裁剪** | 建议值**偏高**，选出来的 lr 一开训就炸 | 裁剪的作用正是压住梯度爆炸 —— 开着它，lr 到 1.0 也不炸，曲线上该有的陡升拐点就看不见了 |
| 扫描时必须**关掉 AMP** | 曲线混入与 lr 无关的抖动 | `GradScaler` 会动态改 scale、必要时跳过 step，等于往曲线里塞了第二个变量 |

同理还有两条：**不做梯度累积**（保证「一个点 = 一次更新 = 一个 lr」），
以及**优化器类型 / weight_decay 保持和真实训练一致**（最优点依赖它们）。
这五条都收在 `resolve_sweep_config()` 里，每一项都带理由，也都有测试守着。

### 为什么它必须「不动模型」

扫描会故意把 lr 推到发散 —— 也就是说，**跑完之后权重里很可能已经是 NaN**。
函数在开始时对权重做一次快照、在 `finally` 里原样还原，模型跟没扫过一样。

不还原的后果不是「精度差一点」，而是**接下来的训练从 NaN 权重出发**，
你看着 loss 是 NaN，会以为是「选的这个 lr 有问题」，其实是 finder 没收拾干净。
测试里把还原摘掉的失败信息很直观：权重从 `0.0025` 被推到了 `538`。

另外，还原是**逐位**的，BN 的 `running_mean` / `num_batches_tracked` 也一起还原
（扫描跑在 train 模式，统计量会被污染）。

### 局限（别把建议值当真理）

- 它给的是**起点附近**的最优 lr。训练中途由 scheduler 决定，不是恒定的一个值。
  实战用法是：拿它当上界，再**除以 2~3** 起步，配合 warmup。
- 数据量不够时会循环复用 batch（曲线被轻微压低），这种情况下结果里会有明确提示，
  不会悄悄糊过去 —— 本仓库的原则是**拿不到的数据就不报**。

> 面试点：这类工具的价值在于**把「猜」变成「测」**，但前提是你要能说清
> 它测的是什么、在什么条件下成立、以及哪几个开关必须关掉。
> 一个「跑起来了」的 lr finder 和一个「建议值可信」的 lr finder 之间，
> 差的就是上面那张表。

## 核心代码位置（想学就看这几个文件）

| 想学什么 | 看哪里 |
|---------|-------|
| 训练 step 的完整顺序 | `src/train.py` 的 `train_one_epoch()` |
| 显存优化的每一项 | `src/train.py` 模块 docstring + `src/train.py` 的 `build_optimizer()` |
| **梯度检查点怎么实现、坑在哪** | `src/model.py` 的 `SmallCNN.forward()` docstring |
| 配置怎么做到可复现 | `src/config.py` 的 `merge()` / `to_yaml()` |
| 为什么必须切 `train()`/`eval()` | `src/train.py` 的 `evaluate()` |
| 验证集为什么必须切 | `src/data.py` 的 `split_train_val()` |
| **DDP 的进程组 / 数据切分 / 指标归约** | `src/distributed.py`（模块 docstring 列了三个必踩的坑） |
| `torchrun` 不可用时怎么跑多进程 | `tools/ddp_launch.py`（FileStore rendezvous） |
| **学习率该给多少** | `src/lr_finder.py`（模块 docstring 列了三个会让建议值静默出错的地方） |
| `torch.compile` 在 Windows 上为什么用不了 | `src/compile_support.py` + `tools/compile_probe.py` |
| 入口的编码保护为什么必须存在 | `src/console.py`（同一个坑踩过两次的记录） |
| CI 的失败注解是怎么发的、坑在哪 | `tests/conftest.py` 的 `pytest_runtest_logreport()` |

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

**Q13: DDP 的数学内容是什么？为什么不用改模型就能用？**
就一条：`grad(batch N·B) == mean_r( grad(batch B on rank r) )`。
每个 rank 在自己那份数据上算梯度，反向时 all-reduce 求**平均**，等价于用了一个 `N·B` 的大 batch。
不改模型就能用的前提是：损失逐样本可加、且前向没有跨样本耦合（没有 BatchNorm 这类"看整批数据"的层）。

**Q14: 那 BatchNorm 会怎么样？**
它会打破上面那条等式。BN 训练态用**当前 batch 的统计量**归一化，所以它在前向里"看得见整批数据"：
单进程 batch 128 的统计量在 128 条上算，2 个 rank 各 64 条的统计量各自在 64 条上算，归一化结果不同，梯度自然不同。
**DDP 不会同步这个**（`broadcast_buffers` 只是把 rank0 的 buffer 广播出去，没让统计量变准）。
实测相对偏差 1.31（比梯度尺度还大）。要等价得用 `SyncBatchNorm`，代价是多一次通信，且**只在 CUDA 上可用**。
→ Transformer 用 LayerNorm 没有 running stats，所以不受影响。

**Q15: DDP 下各 rank 的 batch 数不相等会怎样？**
**死锁，而且没有任何报错** —— 快的那个在 all-reduce 处等慢的，慢的又不等它，训练就这么卡住。
所以训练集必须用 `drop_last=True` 的 `DistributedSampler` 保证切分等长。
排查手段是把 `ddp_timeout_minutes` 调小，让"卡住"尽快变成"报错"。

**Q16: DDP 和梯度累积都做，中间几步要不要用 `no_sync()`？**
用能省 `accum-1` 倍的通信量，**但不用也不会算错**。
因为 DDP 的平均是线性的：`Σ_i mean_r(g_i) == mean_r(Σ_i g_i)`。
所以这是个纯性能优化 —— 面试时如果说成"不加会错"就露馅了。

**Q17: DDP 能让训练变快吗？**
不一定，在 CPU + 小模型上实测**慢了 2.4 倍**（全局 batch 对齐到 128 时 0.42×）。
原因是通信开销超过计算节省、CPU 没有算力冗余、gloo 也不是为速度设计的。
**DDP 真正解决的是「放不下」而不是「算得慢」**：全局 batch 大到单卡显存装不下、
模型+优化器状态超出单卡显存时才需要它。顺序应该是：先用梯度累积/混合精度/检查点把单卡榨干，再考虑 DDP。

**Q18: 存 DDP 训练出来的 checkpoint 有什么坑？**
必须 `unwrap_model(model).state_dict()`。直接 `model.state_dict()` 的 key 会带 `module.` 前缀
（`module.blocks.0.0.weight`），单进程加载时全部对不上 —— 而且**保存时不会有任何提示**，属于埋雷型 bug。

**Q19: 本地测试全绿、CI 的 windows job 却红了，你怎么排查？**
先怀疑**环境差异**，不是逻辑：locale / 编码 / 路径分隔符 / 文件系统大小写敏感 / 行尾 CRLF。
本仓库踩过两次同一个：Python 在 Windows **输出到管道**时用系统 locale 编码（英文系统 = cp1252），
`print("中文")` 直接抛 `UnicodeEncodeError`，exit code 1。**终端场景不发作，中文 locale（cp936）
也不发作**，所以只有 CI 见得到。
手法是**在本地重建 CI 的环境条件**：`PYTHONIOENCODING=cp1252 pytest tests/ -q` —— 一次复现，
真因一行定位，比盯着代码猜快一个量级。

**Q20: 同一个 bug 修完之后又犯了，你会做什么？**
先问"**为什么第一次修完没防住第二次**"。答案通常是：修的是那个点，不是那类事。
本仓库的做法是把不变式写成测试：`test_entrypoint_pins_utf8_stdout` 扫描**全部可执行入口**，
漏一个就红 —— 注意断言的是带括号的**调用**而不是函数名，否则
`from console import force_utf8_stdout`（导入但没调用）就能骗过它。
另一半教训是：**诊断工具自己也要有测试**。给 CI 加的失败注解 hook 自己也会在 cp1252 下崩，
把 pytest 退出码从 1 变成 3，于是"CI 红了却什么都读不到"。
分辨方法是看退出码 —— **3 是 hook 崩了，1 是注解被 GitHub 丢弃**（格式问题）。

**Q21: LR range test 为什么不取 loss 最低的那个点？**
因为最低点通常已经贴着**发散边缘**，用它训练迟早炸。要取的是**最低点之前、loss 下降最陡**
的那个 lr —— 它的含义是"每升高一个 lr 数量级，能换多少 loss 下降"，即性价比最高的点。
实际用法是把它当**上界**，再除以 2~3 起步配合 warmup（它给的是起点附近的最优，不是全程最优）。

**Q22: 一个 lr finder 怎么做才算"可信"？**
"跑起来画出一条曲线"和"建议值可信"之间隔着三件事，而且**做错了照样画出漂亮的曲线**：
① 平滑必须用**带偏差修正**的 EMA（朴素 EMA 初值为 0，前几步被压低 → 最陡点落到曲线开头
→ 建议值退化成 min_lr）；② 必须**关掉梯度裁剪**（它正是压住发散的东西，开着就看不到拐点
→ 建议值偏高）；③ 必须**关掉 AMP**（GradScaler 会引入与 lr 无关的噪声）。
还有两条容易漏的：扫描只允许一次参数更新对应一个 lr（不做梯度累积），
以及优化器/weight_decay 要和真实训练一致（最优点依赖它们）。
最后一条同样重要：扫完**必须把权重逐位还原** —— 扫描会把 lr 推到发散，
不还原就是把 NaN 权重交给接下来的训练，而你会以为是"选的 lr 有问题"。

## 后续可扩展

- [x] `gradient checkpointing` 演示（用计算换显存的量化对比）
- [x] 分布式训练（DDP）最小可跑示例 + 等价性/吞吐实测
- [x] `torch.compile` 支持 + 平台可用性实测（Windows 上 Inductor 两条路都断、退路后端可跑通；
      真正的加速对比需要 Linux —— 本机拿不到，所以没编数据）
- [x] 学习率 finder（LR range test）：等比扫描 + 偏差修正 EMA + 权重零污染还原，
      实测 MNIST 上 100 步 16 秒、建议值 1.262e-3（与默认 lr 差 0.79×）

## License

MIT
