# 项目面试：54 道分层追问与回答检查

[七节课程](README.md) · [技术栈地图](TECH-STACK.md) · [学习方法](STUDY-METHOD.md)

这些题依据现有项目、基础概念和工程边界设计，不是公司真实题库，也不保证出题频率或命中率。先闭卷回答，再展开参考。每题至少讲出定义、机制和一项验证；扩散/LLM/具身部分是下一阶段准备，不能把练习答案当作已经做过的项目成果。

评分可用 0–3：不会、提示下复述、闭卷解释、迁移并验证。每次模拟面试挑 4–6 题深入追问，比一次扫完所有答案更能发现缺口。个人分数不自动写进公开仓库。

<a id="lesson-1"></a>

## 第一节：张量、求导与训练闭环（Q01–Q04）

证据入口：[第一节](01-foundations-and-first-run.md)、[train.py](../../src/train.py)、[model.py](../../src/model.py)。

### Q01：backward 和 step 分别修改什么？为什么梯度要清零？

<details><summary>参考要点与追问</summary>

backward 沿图计算并累加参数梯度，step 根据 optimizer 状态与梯度更新参数。清零保证窗口间不混入残留；窗口内累积是有意保留。追问：zero_grad 能放在 step 后吗？可以，只要在下一窗口 backward 前。再追问 grad=None 与 grad=0：对未参与该步的参数，优化器处理可能不同。用标量实验验证，不背绝对顺序。

</details>

### Q02：交叉熵为什么接 logits？梯度与 softmax 有什么关系？

<details><summary>参考要点与追问</summary>

分类交叉熵内部包含 log-softmax；对单样本 logits 的梯度为 p-one_hot(y)，普通 batch mean 再除 B。输入应是未 softmax 分数与适合的标签格式。追问：ignore_index、类别权重是否还按 B 除？要按实际 reduction 定义重审分母。错误回答是“所有损失除 batch_size 都对”。

</details>

### Q03：eval、no_grad 和 detach 能互相替代吗？

<details><summary>参考要点与追问</summary>

eval 改模块行为；no_grad 改范围内求导记录；detach 返回与图断开关系的张量，仍可能共享底层存储。追问：评估没有 backward，grad 未变能证明没有构图吗？不能，第七节给 forward hook 正反对照。避免“eval 自动关闭求导”。

</details>

### Q04：Dataset、DataLoader、参数与 buffer 各是什么？

<details><summary>参考要点与追问</summary>

Dataset 读一条样本，loader 管采样与组批；parameters 给 optimizer，state_dict 还含 BN 运行统计等持久化 buffer。追问：7 样本/batch2 有几批？不丢末批为 4。保存 model.state_dict 是否包含 Adam 动量？不包含。用形状和存档键解释各自职责。

</details>

<a id="lesson-2"></a>

## 第二节：配置与 Python 工程（Q05–Q08）

证据入口：[第二节](02-configuration-and-cli.md)、[config.py](../../src/config.py)、[test_config.py](../../tests/test_config.py)。

### Q05：为什么 CLI 参数 default 要用 None？

<details><summary>参考要点与追问</summary>

None 表示未提供覆盖，保留 YAML；显式 False/0 是有效输入。当前 merge 只忽略 None。追问：用 if value 过滤会怎样？关闭 amp 和 warmup=0 等会丢失。展示三态覆盖实验。避免“None 只是为了语法方便”。

</details>

### Q06：dataclass 类型注解会检查所有运行时输入吗？

<details><summary>参考要点与追问</summary>

通常不会自动完整校验，需解析与 __post_init__/显式校验。当前仅检查部分字段；负 lr 在底层 optimizer 处才被拒绝。追问：未知 YAML 键与 merge 键在哪里报错？构造与显式字段检查的入口不同。不能声称所有非法超参已在配置层拦住。

</details>

### Q07：函数返回新配置比原地修改有什么好处？

<details><summary>参考要点与追问</summary>

减少调用方被隐式改状态的风险，旧配置可用于对照；dataclasses.replace 会创建新对象并执行初始化过程。追问：如果字段本身是可变列表，浅复制就一定隔离了吗？不一定，需检查引用共享。用对象身份与内容变化验证，不把“新对象”误作“全部深复制”。

</details>

### Q08：新增 label_smoothing 参数，需要改哪些位置？

<details><summary>参考要点与追问</summary>

定义默认/约束、CLI、配置覆盖、criterion 消费、有效配置记录与测试。追问：日志显示启用但 loss 没改变如何抓？设计独立数值或构造参数观测，而非只测 argparse 接受了它。错误是只加一个开关就称功能完成。

</details>

<a id="lesson-3"></a>

## 第三节：优化与梯度累积（Q09–Q12）

证据入口：[第三节](03-training-and-optimization.md)、[test_train.py](../../tests/test_train.py)。

### Q09：累积梯度何时与真实大 batch 等价？

<details><summary>参考要点与追问</summary>

相同样本、初始化、更新窗口、正确权重，且前向条件一致。BN/Dropout/增广与有限精度可能破坏严格等价。3 等大批/accum4，固定除 4 会缩小到目标的 75%。追问 2/2/1：按样本权重，不平均批均值。参照先用无随机层线性模型。

</details>

### Q10：裁剪为什么在 unscale 和样本归一化之后？

<details><summary>参考要点与追问</summary>

阈值应针对最终目标梯度；先裁放大梯度或样本和会改变含义。全局 L2 裁剪统一缩放梯度，不直接保证 loss 不爆或 Adam 参数步长固定。追问：每个 microbatch 单独裁再相加与窗口后裁一样吗？一般不一样，裁剪非线性。

</details>

### Q11：SGD、动量、Adam、AdamW 如何影响恢复？

<details><summary>参考要点与追问</summary>

普通 SGD 依赖当前梯度，动量/Adam 增加历史统计，AdamW 解耦权重衰减。相同权重但不同 optimizer state，下一步可能不同。追问学习率是否仍重要？重要。不要把 Adam 的自适应缩放写成自动找到最优 lr；用同初始化与状态做对照。

</details>

### Q12：scheduler 的 step 到底是什么？LR finder 有什么局限？

<details><summary>参考要点与追问</summary>

当前按尝试更新窗口计数，含尾窗口；总 T 次实际索引 0…T-1，函数 f(T)=0 不代表最后更新用了零。AMP 可能跳实际 step。LR finder 给初始附近建议，当前还原 model state_dict/模式，不还原全部 RNG/grad/loader 状态。追问改变 epochs 如何改变总曲线。

</details>

<a id="lesson-4"></a>

## 第四节：精度、显存与性能（Q13–Q16）

证据入口：[第四节](04-precision-memory-and-performance.md)、[model.py](../../src/model.py)、[显存实验](../../experiments/exp_memory_accounting.py)。

### Q13：AMP 是把所有模型权重变 fp16 吗？

<details><summary>参考要点与追问</summary>

autocast 按算子选择 dtype，GradScaler 调整梯度尺度；不等于 model.half。当前是 fp16 路径；bf16 范围与策略不同。追问 scale 变化与溢出跳步如何观察？看非有限梯度、scale 与参数变化，不能仅靠窗口计数。多 microbatch 中不能随意混合已还原与仍缩放的梯度。

</details>

### Q14：allocated、reserved、峰值和显卡占用有什么区别？

<details><summary>参考要点与追问</summary>

allocated 为活跃张量，reserved 含分配器缓存，峰值依重置区间，设备工具范围更广。empty_cache 不会释放仍被引用的张量。追问逐轮涨内存查什么？带图对象引用、缓存、测量区间、常驻模型。不能只写“显存泄漏，调用 empty_cache”。

</details>

### Q15：activation checkpoint 能保证前向没有副作用吗？

<details><summary>参考要点与追问</summary>

反向重算可能重复状态修改；BN running stats 即使梯度一致也可能不同。use_reentrant=False 不自动消除这些副作用。追问 Dropout RNG 和不重算全部算子的情况：按实际实现/保存策略判断。当前已知问题测试是在确认边界，不等于修复了问题。

</details>

### Q16：如何证明训练更快，而非只测到 CUDA 提交时间？

<details><summary>参考要点与追问</summary>

预热、同步或 CUDA events、固定预算、重复测量、质量对照；记录是否含加载/通信/保存。compile 分冷启动与稳态，profiler 调查后关掉再做基准。追问 p95 与平均值分别回答什么？延迟分布与均值不同；没有样本分布不能给 p95。

</details>

<a id="lesson-5"></a>

## 第五节：checkpoint 与随机状态（Q17–Q20）

证据入口：[第五节](05-checkpoint-and-reproducibility.md)、[checkpoint.py](../../src/checkpoint.py)。

### Q17：完整续训为什么不能只保存权重？

<details><summary>参考要点与追问</summary>

要恢复模型 buffer、optimizer、scaler、进度、数据/RNG 与选模状态。追问忘记 patience/best 会怎样？早停/选模轨迹变化；忘记 optimizer 会改变更新。说明哪些状态当前存档确实保存，哪些只是计划。

</details>

### Q18：seed 与 RNG state 怎样区分？恢复为什么在初始化后？

<details><summary>参考要点与追问</summary>

seed 设序列起点，state 表示当前位置；对象初始化会消耗随机数，创建对象后恢复更有利于接续。追问多 worker、自建 Generator 是否自动覆盖？不是。模型与 loader 随机流分离也要分别保存和验证。

</details>

### Q19：你声称的恢复粒度和可复现范围是什么？

<details><summary>参考要点与追问</summary>

当前主要 epoch 边界；MLP 固定环境配对通过，不推出任意 GPU/compile/worker 逐位一致。micro-step 恢复还需累计 grad、窗口计数、采样位置等。追问 CNN 微小漂移根因是否确认？未完成隔离，应报告观测和验证计划，不编原因。

</details>

### Q20：写 checkpoint 中断与加载到 GPU，分别有哪些风险？

<details><summary>参考要点与追问</summary>

当前直接 torch.save 无完整故障写入协议；可设计临时文件与原子替换并验证。map_location 会影响 RNG 状态张量，按状态 API 要求还原 CPU。追问不可信存档：完整对象加载需有可信来源；不能把下载到文件等同安全有效恢复。

</details>

<a id="lesson-6"></a>

## 第六节：DDP（Q21–Q24）

证据入口：[第六节](06-distributed-training.md)、[distributed.py](../../src/distributed.py)。

### Q21：DDP 帮你自动分数据、切模型和做全局 BN 吗？

<details><summary>参考要点与追问</summary>

常规 DDP 是模型副本与梯度同步；数据 sampler、模型切分和 BN 全局统计是不同机制。local_rank 绑定节点内设备，rank 为全局编号。追问多个 rank 都读同一 batch 会怎样？流程可能跑通但数据并行目的没实现。

</details>

### Q22：no_sync 为什么要包含 forward？一次窗口只有一次 all-reduce 吗？

<details><summary>参考要点与追问</summary>

DDP forward 参与同步安排，范围应含 forward/backward；每窗口一个同步 backward 不代表只有一个集合操作，还可能多 bucket 与计数归约。追问 fake context 测试能证明什么？范围控制，不能证明真实 NCCL 性能。需要分层验证。

</details>

### Q23：各 rank 样本数不同，梯度和 accuracy 如何汇总？

<details><summary>参考要点与追问</summary>

按实际样本权重；当前梯度和经 rank 平均后乘 W/N，accuracy 归约正确数/样本数。不平均均值。追问训练窗口不等长？还涉及通信协议，公式正确不保证不挂；评估不等 forward 次数/buffer 广播需验证。

</details>

### Q24：set_epoch 和 rank0 保存为何仍不足以保证任意 DDP 恢复？

<details><summary>参考要点与追问</summary>

set_epoch 保持 sampler 真轮次；rank0 独占写避免竞写，但一份 RNG 不自动覆盖每 rank 独立随机流与 worker。追问 barrier 能随便加吗？所有参与 rank 调用顺序须一致。按当前已测组合描述，不扩大成大规模训练经验。

</details>

<a id="lesson-7"></a>

## 第七节：测试与交付（Q25–Q28）

证据入口：[第七节](07-testing-debugging-and-delivery.md)、[CI](../../.github/workflows/ci.yml)。

### Q25：测试都绿了，为什么仍可能算错？

<details><summary>参考要点与追问</summary>

测试可能只验证次数或外围行为，参照与实现共享错误，或没覆盖硬件/边界。用 known-answer、独立参照与关键逻辑破坏确认判别力。追问这项目哪条曾弱？累积次数不足以验证幅度；grad 未变不足以验证 no_grad。

</details>

### Q26：遇到 loss 不下降，你怎样定位？

<details><summary>参考要点与追问</summary>

缩到小数据/小模型，查输入标签、loss、梯度、step、状态，再看优化超参；尝试过拟合小集合验证链路。追问减 lr 有效就证明根因吗？不一定，只可能缓和现象。记录首次偏离与反证，而非列十个随机开关。

</details>

### Q27：CPU CI 与真实硬件测试各证明什么？

<details><summary>参考要点与追问</summary>

CI 检查指定版本/平台的逻辑和流程；CPU gloo 不覆盖 NCCL、真实 GPU 性能与任意 AMP/compile 组合。追问 skip 与 bug 区别？能力不可用可明确 skip；实现失败不靠跳过掩盖。当前依赖没全锁定，需记录实际版本。

</details>

### Q28：怎样把项目讲成三分钟技术故事？

<details><summary>参考要点与追问</summary>

讲问题/触发输入、已知答案、根因、修改、对照与边界，链接到代码和结果。追问你的独立贡献、未验证项与下一步。没有实际实验就写方案，不能把将来工作说成已交付。测试条数与代码行数不是主结论。

</details>

## Python 与数据处理（Q29–Q36）

### Q29：可变对象、浅复制与深复制怎样影响实验快照？

<details><summary>参考要点与追问</summary>

共享引用会让“保存的旧状态”跟着后续变化。嵌套对象浅复制未必隔离；模型 state_dict 快照需要明确复制语义。追问 list 默认参数、tensor.detach 是否独立存储？不能一概而论，设计修改原对象的反证。关联 LR finder 的 deepcopy。

</details>

### Q30：迭代器、生成器和 itertools.cycle 的区别是什么？

<details><summary>参考要点与追问</summary>

迭代器消费状态，generator 按需 yield；cycle 缓存已产生元素再复用。追问对 DataLoader 使用 cycle 是否每轮重新增广？当前会复用 batch，不应假定重新取样。还要考虑缓存内存与恢复当前位置。

</details>

### Q31：context manager、try/finally 和 except 各解决什么？

<details><summary>参考要点与追问</summary>

with 定义资源/模式作用范围，finally 保证清理，except 决定处理失败。追问 broad except 后默认继续为何危险？可能静默改变实验语义。说明 LR finder 模型恢复、no_sync 作用范围、进程组清理的不同目的。

</details>

### Q32：独立写 Python 算法题时，怎样展示思路与复杂度？

<details><summary>参考要点与追问</summary>

先明确输入/边界，给小例子，解释不变量，再写代码和时间/空间复杂度。可练哈希统计、双指针、滑动窗口、二分、堆、树遍历与基础 DP。追问空输入、重复值、大数据，实际岗位要求再调整；本题不声明某题库足够覆盖所有笔试。

</details>

### Q33：Linux 上训练进程慢或消失，先查什么？

<details><summary>参考要点与追问</summary>

查退出码/完整日志、资源占用、进程状态、磁盘、依赖与设备，再区分代码异常和外部终止。追问后台任务、输出重定向、环境激活与路径。没有 Linux 实战就先做从零运行与日志练习，不把 Windows 经验自动改写为熟练 Linux。

</details>

### Q34：如何识别数据泄漏和过拟合？

<details><summary>参考要点与追问</summary>

核对切分、重复样本、时间/用户关联、标准化与增广来源，区分训练/验证/测试职责。过拟合表现与数据泄漏不完全一样。追问多 seed 的均值、标准差、置信区间含义不同；相关任务不能把随机切分当普遍正确。

</details>

### Q35：SQL 里 JOIN 后计数突然增大，原因可能是什么？

<details><summary>参考要点与追问</summary>

一对多/多对多连接会复制行；先明确主键、去重与计数单位。追问 NULL、left join 与 where 过滤、group by/窗口函数。用一张训练运行表与指标表做独立查询练习，SQLite 文件存在不证明已掌握分析 SQL。

</details>

### Q36：多个实验中如何判断改进可靠？

<details><summary>参考要点与追问</summary>

预先定义主指标与预算，开发集调参、测试集冻结，记录 seed 与失败，合理重复/配对比较。追问最佳 seed、改两个变量、测试集反复选模型都有哪些偏差？有限样本要说明不确定性，不能只贴最大值。

</details>

## 扩散模型与推理服务（Q37–Q44）

入口：[HF 扩散课程](https://huggingface.co/learn/diffusion-course/en/unit1/1)、[现有扩散项目](https://github.com/colorHur/diffusion-lora-lab)。这些是后续深化题，不表示所有实验已经完成。

### Q37：扩散训练中的 clean latent、noise、timestep 和目标各是什么？

<details><summary>参考要点与追问</summary>

按噪声调度把 clean latent 与噪声组合为 x_t，模型结合 timestep/条件预测指定目标。当前脚本用 epsilon MSE；epsilon、v、sample prediction 不能混用。追问训练 loss 降低是否一定图像更好？还要留出 prompt 与任务相关质量评估。

</details>

### Q38：latent diffusion 中 VAE 和文本编码器起什么作用？

<details><summary>参考要点与追问</summary>

VAE 转换图像与潜变量，文本编码器提供条件表征，去噪网络在潜空间学习。追问 scaling factor、输入范围、被冻结模块是否需要求导？逐段追目标与可训练参数，不把尺寸经验或具体 token 上限推广到所有模型。

</details>

### Q39：训练噪声 scheduler 与推理采样 scheduler 一样吗？

<details><summary>参考要点与追问</summary>

训练定义加噪/目标与 timestep 分布，推理定义去噪数值过程与步数。兼容关系依赖模型参数化和配置。追问同 seed 比采样器为何仍要多 prompt？单图不代表任务质量；步数、CFG、冷启动等要控制。

</details>

### Q40：LoRA 的 A、B、rank 和 alpha 如何影响参数量？

<details><summary>参考要点与追问</summary>

常见 ΔW=sBA，A/B 为低秩因子，参数量约 r(d_in+d_out)；scale 受具体 alpha/r 或变体规则影响。追问 target_modules、初始化、rank/步数同时改是否消融？不是单变量。不能把低秩解释成质量必然更好或零显存。

</details>

### Q41：扩散微调的 step 数和梯度累积有什么坑？

<details><summary>参考要点与追问</summary>

区分 micro-step/update/sample。当前脚本 1500 micro-step、accum4 为 375 次更新；非整除末窗口可能丢未更新梯度，需要明确策略。追问 frozen 模块、有限梯度和 scaler 的作用；保存配置与原始日志后才引用步数成果。

</details>

### Q42：怎样评估 LoRA 的泛化、风格与过拟合？

<details><summary>参考要点与追问</summary>

训练图像与测试 prompt 分开，固定多个 prompt/seed，考察文本遵从、概念/风格匹配、多样性与重复，保留人工盲评/失败。像素差只证明变化；少量数据 FID 也有统计边界。追问 rank/步数分开消融与评估泄漏。

</details>

### Q43：多请求 adapter 切换为何会串用？只锁 set_adapters 够吗？

<details><summary>参考要点与追问</summary>

共享 pipeline 是可变状态，A 切到 A 后 B 可覆盖，A 生成读 B。锁要覆盖切换和整个生成，或单 worker 有界队列。追问超时、OOM、取消、状态清理与并发验收；async 路由不自动隔离模型状态。先 fake 归属测试，再 GPU HTTP 测试。

</details>

### Q44：如何测服务延迟、显存与吞吐？

<details><summary>参考要点与追问</summary>

分冷启动、预加载、adapter 切换、排队、生成和端到端；多请求重复，记录 p50/p95、吞吐、失败率与峰值。追问低 p50 但 p95 高查什么？队列/资源争用/慢路径。预加载显存与切换增量不同，不能从几次生成相近推出零额外成本。

</details>

## Transformer、LLM 与检索延伸（Q45–Q50）

### Q45：Q/K/V 的维度、缩放和 mask 怎样工作？

<details><summary>参考要点与追问</summary>

能写 softmax(QKᵀ/√d_k + mask)V 并标维度，区分 causal mask 与 padding mask。追问 mask 加在哪里、全屏蔽行、形状广播。下一步手算小例子再实现验证，不只背 attention 公式。

</details>

### Q46：SFT 的 label masking 与普通分类 loss 有何不同？

<details><summary>参考要点与追问</summary>

按预测目标定义 token 对齐/shift 与有效标签，prompt/padding 是否计算 loss 由任务协议决定。追问长短样本累积用样本平均还是 token 平均？定义目标后选择分母，不能盲套 y.size(0)。需要另做文本任务证据。

</details>

### Q47：PEFT 库用过就等于做过 LLM 后训练吗？

<details><summary>参考要点与追问</summary>

不等于，扩散 adapter 与语言模型任务、数据、损失和评测不同。追问基座、prompt 改进、SFT 的留出对照；介绍 DPO/GRPO 时先说目标与数据需求，未实现则说明机制学习范围。不要把依赖表当项目经历。

</details>

### Q48：RAG 如何判断该改 chunk、retrieval 还是生成？

<details><summary>参考要点与追问</summary>

把检索召回/相关性、证据覆盖、生成忠实与拒答分开评测；没有证据时生成更流畅不代表有效。追问 overlap、reranker、引用验证与预算。先规则/检索基线，再根据开发集错误改，测试集不反复调参。

</details>

### Q49：KV cache、量化与批处理分别优化什么？

<details><summary>参考要点与追问</summary>

KV cache 复用自回归历史注意力投影，量化改变表示/计算成本，batch 提升资源利用但影响等待与尾延迟。追问上下文长度、cache内存、quality/perf tradeoff。未在目标硬件测前不报告速度倍数或省显存比例。

</details>

### Q50：模型输出 JSON 就意味着应用可靠了吗？

<details><summary>参考要点与追问</summary>

结构/schema、字段类型、证据正确、业务约束、错误重试分别验证。追问空输出、遗漏、矛盾、上下文预算、幂等和成本。语法合法与内容正确是不同指标；模型 judge 也要配可验证答案或人工核验。

</details>

## 具身与世界模型延伸（Q51–Q54）

### Q51：世界模型预测误差变小，控制收益必然提高吗？

<details><summary>参考要点与追问</summary>

不必然；控制关注任务相关状态/动作、闭环反馈与约束，开环误差只是部分证据。追问固定预算/数据、真实闭环指标、多个 seed 与强基线。自己的研究结论必须带场景与指标口径，不能推广成所有具身任务。

</details>

### Q52：做具身数据任务还要检查哪些变量？

<details><summary>参考要点与追问</summary>

视觉/状态/动作的时序同步、频率、坐标系、动作单位、轨迹切分、传感噪声和任务成功定义。追问同轨迹切到训练测试是否泄漏？可能。能做数据/仿真评测不等于完成真机控制，明确方向与独立贡献。

</details>

### Q53：terminated、truncated 与优势递推边界如何处理？

<details><summary>参考要点与追问</summary>

真终止与外部截断的 bootstrap 语义不同；截断用终点观测价值，reset 后观测不能替代终点，递推不能跨回合混收益。追问构造已知答案，普通转移/终止/截断/rollout末尾各测。这里是正确性准备题，不声明现有所有 agent 已修复。

</details>

### Q54：离线数据、仿真成功与真机结果能怎样表述？

<details><summary>参考要点与追问</summary>

按实际证据分别报告；分布偏移、反馈频率、延迟、接触动力学和执行约束会改变结果。追问如何验证迁移？先设计合适仿真/离线基线，再在授权目标环境测，不捏造真实设备与用户效果。岗位准备应围绕具体任务而非笼统“会具身”。

</details>

## 模拟面试怎么用

每节先取对应 4 题，每题 60–90 秒解释基础，再用 2–3 分钟接受机制与迁移追问。导师检查公式、源码证据和边界，不要求背本文。进阶模拟可混合训练、扩散和数据题，遇到尚未做过的任务先说明已有基础，再设计验证方案。

参考资料与学习路线见 [学习方法](STUDY-METHOD.md)。涉及当前库调用，以源码和项目环境为准；这些参考答案是检查点，不是唯一标准措辞。

扩散/LoRA 延伸参考 [Diffusers LoRA 训练](https://huggingface.co/docs/diffusers/training/lora) 与 [LoRA 原论文](https://arxiv.org/abs/2106.09685)；attention 延伸参考 [Transformer 原论文](https://arxiv.org/abs/1706.03762)；终止/截断语义参考 [Gymnasium 时间限制](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。个人项目验证范围仍以实际实验为准。
