# 第七节：从“我跑过”到可验证的交付

[上一节](06-distributed-training.md) · [课程目录](README.md)

## 0. 先修、阅读路径与闭卷检查

先修：已知答案、状态观测、浮点比较与 Git diff。第一遍做 no_grad 的正反对照；第二遍写最小复现与还原记录；进阶交付小任务并检查 CI/数据/成本证据。

迁移任务：选一个现有弱判据，写出它可以漏过的错误，再设计独立观测；只在验证后升级测试。闭卷答 [Q25–Q28](INTERVIEW.md#lesson-7)，三分钟讲清问题与证据，不用代码行数代替成果。

第一晚练习写判据与缩小故障，第二晚检查 CI/Git 并整理一个真实任务交付。这节把前六节串起来：谁能复现结果，测试抓到了什么，未测路径有哪些，怎样让后续修改不破坏已有行为。

## 1. 测试层次与证明范围

| 层次 | 本仓库例子 | 能证明什么 |
|---|---|---|
| 纯逻辑 | 配置 merge、学习率取值 | 小范围输入/输出契约 |
| 数值对照 | 梯度累积与真实 batch | 给定条件下更新值正确 |
| 状态检查 | RNG/optimizer 恢复 | 指定状态已接续 |
| 流程测试 | CLI 合成数据训练 | 组件连接、日志与产物符合预期 |
| 多进程测试 | CPU gloo 采样/恢复 | 被测分布式协议可运行 |
| 硬件实测 | CUDA 峰值与计时 | 指定设备、路径和测量条件下行为 |

测试数量不是覆盖所有情况的证明。CI 全绿也不能推出 GPU/NCCL、网络下载、真实数据质量和长期服务稳定性。每个测试必须说清“失败意味着什么”和“通过仍然不知道什么”。原来的累积更新次数测试保留有价值，但需要参数数值对照补上它不能验证的部分。

## 2. 一条看着合理却抓不到错误的测试

旧 `test_evaluate_does_not_build_graph` 在参数 grad 中放一个值，evaluate 后检查没有变化。但 evaluate 即使忘记 no_grad，也没有调用 backward，参数 grad 通常仍不变。这个断言不足以证明计算图关闭。正确判据要直接观察 forward 时梯度记录是否启用，或者输出是否需要梯度。

```python
# 可运行实验：forward hook 观察 evaluate 的求导状态
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.train import evaluate

model = nn.Linear(3, 2)
loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2)
observed = []
def observe(module, inputs, output):
    observed.append((torch.is_grad_enabled(), output.requires_grad))
handle = model.register_forward_hook(observe)
try:
    evaluate(model, loader, nn.CrossEntropyLoss(), 'cpu')
finally:
    handle.remove()
assert observed and all(pair == (False, False) for pair in observed)
print('evaluate 期间没有启用求导记录:', observed)

# 反向对照：直接执行去掉装饰器的原函数，模拟忘记 no_grad。
parameter = next(model.parameters())
parameter.grad = torch.full_like(parameter, 123.0)
observed.clear()
handle = model.register_forward_hook(observe)
try:
    with torch.enable_grad():
        evaluate.__wrapped__(model, loader, nn.CrossEntropyLoss(), 'cpu')
finally:
    handle.remove()
assert all(pair == (True, True) for pair in observed)
assert torch.all(parameter.grad == 123.0)
print('去掉 no_grad 后图已构建，但旧的 grad 未变化判据仍通过:', observed)
```

将源码中的 no_grad 装饰器在独立副本暂时移除，这个实验应失败；只检查 grad 未变化的旧判据可能仍通过。新版讲义指出现有测试缺口，并提供更直接的验证；不要因为仓库里存在一个 test 名称就信它测到了标题所说的行为。

## 3. 写测试前先写已知答案

先列输入与预期，选择最小、能分辨错误的模型和数据。例如累积使用 7 个样本，batch2、accum4，让末批和窗口分母同时显形。选无 BN/Dropout 的线性模型隔离梯度计算；再在真实模型上测试状态副作用。等价参照必须独立计算，若参照也调用同一个错误 helper，两个结果可能一起错。

浮点比较的 rtol/atol 要对应目标尺度；精确离散值用 equality，权重接近用 assert_close，逐位承诺用 torch.equal。不要无限放大容差只为了绿。测试失败先检查差异量、位置与分布，再判断是数值精度还是语义错误。

mock/fake 适合隔离控制流，如 no_sync 的作用范围；它不能证明真实后端速度。真实 GPU 测试无法在 CPU 上生成目标条件时应该明确 skip，而不是用 fake 返回值宣称 GPU 路径测过。

## 4. 最小复现的工作流

1. 保存失败命令、完整 traceback、配置、版本和最近改动。
2. 把数据、模型、轮数缩到仍能复现的问题；先确认缩小后现象没有消失。
3. 写下不变量和错误的数值：哪里首次偏离？
4. 提出一个能验证的解释，每次只改一个因素。
5. 对比原版与破坏/修复版，必要时查看张量、采样索引和上下文状态。
6. 修复后先跑小判据，再跑相关回归，最后执行项目要求的检查。

不要把调小 lr、减 batch 作为所有问题的根因解释。它们可能让现象缓和，但回合边界、样本权重或状态恢复错误仍在。日志里最好保留一份失败案例；事后删掉不理想的 seed 会改变证据。

```bash
python -m pytest tests/test_train.py -q
python -m pytest tests/test_train.py -k accumulation -q
python -m pytest tests/test_smoke.py -k resume -x -q
python -m pytest tests/ -q --durations=10
python -m ruff check src experiments tests tools
```

`-k` 按名称筛选，可能命中零个测试，要看收集与执行数量；`-x` 在首个失败停止便于诊断；durations 帮助识别慢项。不要每次改一行文字就跑全部硬件实验，但改训练更新或恢复核心时应完成相关回归和要求的 CI。

## 5. 破坏实验与可靠还原

先确认本地工作状态，记录目标文件 hash 和当前差异。用独立分支/副本做实验；同一个文件的一系列修改顺序执行。下面 PowerShell 命令用于查看文件哈希：

```powershell
Get-FileHash -LiteralPath src/train.py -Algorithm SHA256
git status --short
git diff -- src/train.py
```

破坏一个明确因素，例如固定尾窗口分母、忘记 set_epoch、漏 loader RNG。看到预期判据失败后恢复原文件；哈希与基线相同、差异符合预期、测试重新通过三个证据一起看。已有未提交改动时不能直接覆盖整个文件还原；应先保存它们。当前课程协作约定不使用 git stash，避免破坏工作状态。

恢复后才进行下一实验。复制文件过程中可能改变 BOM/行尾；文本外观相同不能替代字节哈希。测试需要破坏的是逻辑，没必要删除 `.git`、数据目录或系统依赖。

## 6. CI 是自动验证契约

[ci.yml](../../.github/workflows/ci.yml) 在 main push 与 PR 上运行，矩阵包括 Ubuntu/Windows 和 Python3.11，安装 CPU 版 PyTorch，再 lint 和跑 tests。CPU gloo 多进程可以覆盖协议，却不会测试 GPU/NCCL。当前 CI 依赖安装未锁定全部精确版本，长期可能因依赖更新发生变化；复现报告需要记录实际环境，不能仅说“相同 requirements”。

遇到 CI 红，先定位失败 job/步骤，再看完整日志。比较本地与 CI 的 Python、torch、设备、shell、路径、编码和多进程行为。如果已经有正确的失败证据，补测试守护它；不要把版本升级、重试或 skip 当成通用修法。

“环境无法支持该能力”的 skip 应写清条件；“实现有错”的测试不能为了绿改成 skip。检查 compile 时确认是否降级；检查 GPU 时确认真实可用设备，不凭版本带 cu 后缀推断硬件存在。

## 7. Git 提交与公开作品

```bash
git status --short
git diff --check
git diff --stat
git diff
```

检查后按路径挑选提交文件，避免把 outputs、数据缓存、私密资料、token 和本机绝对路径带进公开仓库。提交作者按项目约定使用匿名邮箱，先查看身份配置；不要为了方便公开整个个人学习目录。更新训练代码时，测试与教程应解释同一个最终行为。

提交信息写具体变化与原因。一个可审阅改动应说明触发场景、之前错误、修复后行为和验证。例如“3 个 microbatch、accum4 时尾窗口过度缩小梯度；改按实际样本数归一化，用真实 batch 参数对照验证”。少写“全面优化”“性能提升”，没有测量就不写速度数字。

README 首屏让第一次阅读的人找到问题、关键结果、运行入口和限制；深度课程在 docs。来源只给可验证代码与实验，不拿测试条数代替真实任务能力。更强的简历故事是一次定位过程和它改变了哪些结论。

## 8. 课程结束的小任务交付

先用合成数据证明流程，再选一个公开小任务，例如 CIFAR10 分类；使用已有脚手架，不另建同类框架。预先约定数据切分、主指标、训练预算与种子，完成一个 baseline 和一次单变量改进。若涉及当前 CIFAR 验证 transform 共享问题，先修正/明确评估协议，避免“准确率提升”来自不同验证增广。

交付清单：启动命令、有效配置、数据来源与切分、依赖/代码版本、基线和改进对照、失败案例、性能条件、续训证据、README 与适用边界。至少做开发集上的错误分析；测试集冻结后不反复用它选超参。小数据也不能凭单次 seed 宣称稳定改善。

建议个人记录表：

| 日期/版本 | 假设 | 唯一改变因素 | 主指标与预算 | 实际结果 | 解释/限制 | 下一步 |
|---|---|---|---|---|---|---|
| 自己填写 | 为什么要改 | 具体参数或算法 | 预先确定 | 亲手实测 | 证据支持范围 | 能验证的问题 |

## 9. 最终自测与导师复盘

回答并提交自己的推导：①为什么参数 grad 不变不能证明 no_grad？没有 backward 也不会改 grad。②累积参照为什么不能用同一 helper？共享错误会产生假等价。③CPU CI 不能证明哪些 GPU 条件？真实硬件、NCCL、AMP/compile 组合及性能。④怎样证明还原？差异、hash、回归。⑤为什么固定预算后再比恢复？把调度计划变化与状态缺失分开。

最终验收用三分钟讲一项修复：触发输入→错误输出→最小反例→根因→修改→反证→当前边界。再给一个完整复现入口，让另一个人按说明运行。能解释、能修改、能验证，才是本课程目标。

参考：[pytest 用法](https://docs.pytest.org/en/stable/how-to/usage.html)、[Git diff](https://git-scm.com/docs/git-diff)、[GitHub Actions Python 测试](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)。

## 10. 把一条实验结果写成可信的项目材料

坏记录只有“AMP 快了、显存小了”。有效记录包含原始命令、配置、环境、预热与重复次数、数据预算和正确性对照。写“峰值减少”时附测的是哪种峰值；写“可复现”时附恢复粒度与设备；写“DDP 优化”时附每 rank 数据与实际通信证据。

假设你修了尾窗口，最终材料可以写：构造短窗口和不等长 microbatch，发现更新次数正确但参数更新幅度不等价；改用实际样本归一化；在无 BN/Dropout 参照模型下数值对照通过；真实模型的 BN/随机行为仍单独评估。它把代码能力、测试思维和结论边界放在同一段，能承接追问。

一次 PR/提交收口检查：代码是否实现文案所说行为，测试是否包含触发场景，文档是否更新分母与时钟，旧链接是否仍可导航，公开文件是否只有计划中的内容，CI 是否完成。不要把测试通过后的未测修改一起混进“已经验证”的提交。

最后做一轮导师模拟：先用一张图解释架构，再被追问“哪条测试曾经漏掉错误、为何漏掉、你如何改判据”；给一个最小输入现场手算，并指出自己的结论尚不覆盖什么。答不出时回到源码和实验，而不是新增技术关键词。
