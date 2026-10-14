# 第一节：从五行训练代码到完整工程

[课程目录](README.md) · [下一节](02-configuration-and-cli.md)

## 0. 先修、阅读路径与闭卷检查

先修只需 Python 变量/函数/for、张量形状和最小导数；不会链式法则时先做第 9 段共享节点实验。第一遍完成环境、标量求导和最小分类循环；第二遍画模块/数据/状态关系；进阶再看验证 transform 与共享存储边界。

迁移任务：把二分类改成三类，先说明 logits/标签形状与 loss 输入，再改代码。闭卷检查 [Q01–Q04](INTERVIEW.md#lesson-1)，先不要展开答案；运行循环成功只证明跑通，能解释 grad、buffer 与模式区别才继续。

第一晚检查环境、完成自动微分实验和离线运行；第二晚理解模型、数据流、产物与自测。先问自己：梯度存在哪里？`eval()` 会关闭求导吗？`step()` 使用什么更新参数？batch 的第一维代表什么？不会的题带着问题读下去。

## 1. 确认解释器与工作目录

```bash
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('devices:', torch.cuda.device_count())"
python -m pip --version
python src/main.py --help
```

`sys.executable` 是本次进程真正用的解释器。编辑器选的环境、终端的 `python`、`pip` 的目标可能不同。`python -m pip` 把安装绑定到当前 Python，更容易追踪。缺模块时先查环境和路径，再判断依赖是否缺失。根目录应能看到 `src/`、`configs/`、`tests/`；相对文件路径一般从当前工作目录解析，不从你正在看的源文件目录解析。

按仓库 [requirements.txt](../../requirements.txt) 与 [requirements-dev.txt](../../requirements-dev.txt) 安装对应依赖。PyTorch 的 CPU/CUDA 构建按设备选择。没有 GPU 也能完成大部分课程；CPU 实验无法提供 CUDA 显存数据，应记录未测。不要为了复制旧环境截图同时安装几个互相冲突的 PyTorch。

如果 Windows 终端或重定向输出出现中文编码问题，PowerShell 可先设置 `$env:PYTHONIOENCODING='utf-8'`，Bash 可用 `export PYTHONIOENCODING=utf-8`，再运行 Python。项目主入口有输出编码兼容逻辑，但你另存的实验脚本不一定调用它。只设置编码不能解决所有终端显示问题，查清写入编码、读取编码和显示字体分别在哪一层。

## 2. 看懂 Tensor 的四个属性

`shape` 描述结构，`dtype` 描述数值表示，`device` 描述计算位置，`requires_grad` 表示是否追踪梯度。MNIST 输入为 `[B,1,28,28]`，标签为 `[B]`，模型输出为 `[B,10]`。输入通常浮点，分类标签是整数类别索引。先打印形状与类型，再判断某层为何不匹配。

模型输出的 logits 是未归一化分数。`CrossEntropyLoss` 接收 logits 与类别索引，内部执行所需的 log-softmax 与负对数似然。再手动 softmax 会改变计算语义。`x.to(device)` 返回目标设备上的张量，应重新赋值；搬模型不会自动搬每个 batch。设备传输与数据加载是独立步骤。

## 3. 用一个标量看清 backward 与 step

保存为 `outputs/course/autograd_probe.py`。运行前手算 `w=1` 时损失和梯度。

```python
# 可运行实验：梯度累加与一次 SGD 更新
import torch

w = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([w], lr=0.1)
optimizer.zero_grad(set_to_none=True)
loss = (w * 2 - 5).square()
loss.backward()
assert loss.item() == 9.0
assert w.grad.item() == -12.0
print('第一次 backward:', w.item(), w.grad.item())

# 创建新计算图；不清梯度，第二次仍累加到 .grad。
((w * 2 - 5).square()).backward()
assert w.grad.item() == -24.0
print('第二次 backward:', w.item(), w.grad.item())
optimizer.zero_grad(set_to_none=True)
assert w.grad is None
((w * 2 - 5).square()).backward()
optimizer.step()
torch.testing.assert_close(w.detach(), torch.tensor(2.2))
print('正确的一次更新:', w.item())
```

损失为 `(2w-5)^2`，导数为 `4(2w-5)`，`w=1` 时是 `-12`。SGD 执行 `w_new=w_old-lr×grad=2.2`。`backward()` 计算并累加梯度，`step()` 读取现有梯度更新参数；二者不会互相代办。

每次前向创建新计算图，普通反向后为求导保存的中间状态通常释放。对同一个 loss 再反向可能报图已释放；这里重新计算第二个 loss。不要用 `retain_graph=True` 掩盖所有报错，它增加中间状态寿命，应该先说明为何需要同一张图。

清梯度既可放在更新窗口开始，也可在上次 `step()` 后、下次 `backward()` 前。关键是**窗口内保留累积、窗口间清除残留**，不能绝对说零梯度只准出现在某一行。`set_to_none=True` 与填零张量不同：未参与计算的参数可能保持 `grad=None`，优化器对它的处理可能与零梯度不同。

## 4. 模型参数、buffer 和模式

在 `nn.Module.__init__()` 中把层赋给属性，模块系统才能发现参数；`forward()` 定义连接方式。正常调用 `model(x)`，它包含模块调用机制与 hooks；直接 `.forward(x)` 会绕过部分机制。`parameters()` 给优化器管理参数；`state_dict()` 还包含持久化 buffer，如 BN 的运行均值，两者不等价。

| 操作 | 改变什么 | 不自动做什么 |
|---|---|---|
| `model.train()` | Dropout、BN 等层的训练行为 | 不清梯度，不强制开启求导 |
| `model.eval()` | 一般关闭 Dropout、让 BN 用运行统计 | 不关闭计算图 |
| `torch.no_grad()` | 关闭范围内梯度记录 | 不切换模块模式 |
| `optimizer.zero_grad()` | 清管理参数的梯度 | 不恢复权重，不还原 buffer |

所以常规评估同时用 `eval()` 与 `no_grad()`。BN 的具体行为还依赖 `track_running_stats` 等参数，不能把常见配置推广成所有 BN 的性质。本仓库 [model.py](../../src/model.py) 包含 BN/Dropout，这些前向状态会影响后面的累积、检查点和续训实验。

## 5. Dataset 与 DataLoader

Dataset 回答怎样读取一个样本；DataLoader 管采样顺序、batch 组装、是否多进程。`len(dataset)` 数样本，`len(loader)` 通常数 batch。7 个样本、batch=2、`drop_last=False` 的批次大小是 2、2、2、1。`shuffle` 改顺序，不改变样本本身。`num_workers` 越大不保证越快，先用 0 跑通，测清加载瓶颈再增加。

[data.py](../../src/data.py) 的 `SyntheticDataset` 是可学习的离线数据，适合验证“循环真的能降低 loss”，它不能证明真实视觉任务泛化。训练集更新参数，验证集选模型与调超参，测试集做最终评估。不能把测试结果反馈成下一轮调参目标。

当前随机切分固定 seed；真实任务若有时间、用户、同视频片段关联，还要避免跨集合泄漏。CIFAR 的验证子集共享训练 Dataset 的 transform，因此会继承训练增广；需要严格评估时应另建验证 transform。已有固定索引不等于验证内容完全固定。

## 6. 离线跑通并画调用图

```bash
python src/main.py --dataset synthetic --device cpu --epochs 1 --exp_name course01 --output_dir outputs/course
python -m pytest tests/test_data.py tests/test_model.py -q
```

按照五段日志搜索 [main.py](../../src/main.py) 的 `run_training`，再沿调用进入文件：

```text
build_parser / parse_args → TrainConfig.from_yaml + merge
→ 种子、设备、输出目录 → build_dataloaders → build_model
→ build_optimizer + build_lr_scheduler → train_one_epoch
→ evaluate → save_checkpoint → 读取 best.pt → 最终测试
```

本次一般留下 `best.pt`、`last.pt`、`config_resolved.yaml`、`summary.json`、`train.log`，逐轮保存另有 epoch 文件。best 由验证指标选，last 用于恢复最后保存状态；有效配置记录 CLI 覆盖后的值；JSON 支持分析；日志便于查阶段和异常。换实验用不同 `exp_name`，防止把旧产物当成新结果。

## 7. 练习与参考答案

练习 A：标量实验保留 `-24` 梯度后执行一次更新，预测结果。答案是 `w=3.4`。梯度残留会改变更新，但不一定当场报错或让 loss 上升。

练习 B：合上资料写一个 Linear 分类器循环，用固定合成数据观察前后 loss。验收要说明模式、清零、前向、损失、反向、更新每一步，以及所有张量的形状；抄完运行不算完成。

练习 C：找到 course01 的配置 epochs、JSON history 长度和 checkpoint epoch，解释如何互相验证。如果不符，先查运行是否成功、输出目录是否混用、是否用了旧文件。

常见卡点：找不到模块先查目录与解释器；张量不匹配先查 shape/dtype/device；无梯度查 no_grad、requires_grad 和参数是否参与 loss；loss 下降但测试差查切分与过拟合，不能只加训练轮数。

自测答案：① backward 后参数通常尚未变化，要 step。② eval 与 no_grad 分别控制模块行为和求导。③模型权重不包含 Adam 动量，后者在 optimizer state。④ dataset 与 loader 分别数样本和批次。⑤可学习合成数据让“训练应有效”有可检验含义。

收工判据：独立写最小循环、画调用图、解释 logits 与标签类型、定位五种产物，完成一份预测与运行对照记录。

参考：[官方自动微分教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[官方模型定义](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)、[官方数据加载](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)。

## 8. 最小训练循环参考实现：先自己写，再核对

练习 B 卡住时，先只查看 Dataset/DataLoader 部分；写完模型与循环后再看完整参考。这里生成一个由线性边界决定标签的小任务，训练集结果只用于教学检查，没有宣称泛化效果。

```python
# 可运行实验：独立于项目模板的最小分类训练
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(11)
x = torch.randn(64, 2)
y = (x[:, 0] + x[:, 1] > 0).long()
loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
model = nn.Linear(2, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
model.eval()
with torch.no_grad():
    before = criterion(model(x), y).item()
for epoch in range(10):
    model.train()
    loss_sum = 0.0
    samples = 0
    for inputs, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        samples += labels.size(0)
    print(epoch + 1, '沿训练轨迹的平均 loss:', loss_sum / samples)
model.eval()
with torch.no_grad():
    after = criterion(model(x), y).item()
    accuracy = (model(x).argmax(1) == y).float().mean().item()
assert after < before
print('固定集合前后 loss:', before, after, 'accuracy:', accuracy)
```

逐行说明：TensorDataset 把 x/y 按索引绑在一起；Linear(2,2) 对每个二维输入输出两个 logits；标签由布尔值转 long 获得；optimizer 管 model.parameters 的引用；反向后权重尚未更新，step 才消费梯度。训练平均 loss 按实际样本加权。最终评估重新用训练结束后的模型测整个集合，不能与训练过程中逐批统计误作同一量。

尝试三种变化，每次只变一种：把 lr 改小看有限预算下下降速度；从循环中暂时移除 step 看参数为何不变；每批清零改成每两批清零，解释为何需要第三节的正确分母。先预测方向，但不要保证所有超参都产生单调 loss。

对照本项目，额外代码解决的具体问题是：配置让命令可追溯，日志让结果可查，checkpoint 让状态可恢复，测试把边界变成判据，DDP 则增加多进程数据与通信协议。先能说出这层增加的责任，再进入后续章节。

## 9. 共享节点的梯度为何相加

参考 micrograd 的小计算图思路，用自己的例子：`a=w*w`，`L=a+a`。L 对 a 的导数为 2；a 对 w 的两个输入分支各贡献 w，因此 dL/dw=4w。w=3 时梯度为 12。这是图中多路径贡献的求和，与“连续两个 backward 累到 .grad”发生在不同层次。

```python
# 可运行实验：共享节点的链式法则与有限差分
import torch

w = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
a = w * w
loss = a + a
loss.backward()
h = 1e-6
def f(value):
    return 2 * value * value
finite_difference = (f(3.0 + h) - f(3.0 - h)) / (2 * h)
torch.testing.assert_close(w.grad, torch.tensor(12.0, dtype=torch.float64))
assert abs(w.grad.item() - finite_difference) < 1e-6
print('autograd:', w.grad.item(), '有限差分:', finite_difference)
```

有限差分是局部数值检查，有步长和浮点误差；非光滑点还要注意导数定义。它不是替代高效反向传播的训练方法。先手画图和贡献，再运行，能看出梯度来自哪些路径。
