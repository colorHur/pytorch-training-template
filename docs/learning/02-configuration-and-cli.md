# 第二节：命令行、配置与 Python 工程组织

[上一节](01-foundations-and-first-run.md) · [目录](README.md) · [下一节](03-training-and-optimization.md)

第一晚拆 argparse 与覆盖规则，第二晚做配置实验和源码追踪。学完应能解释 `--epochs 3` 如何变成最终配置，为什么未传值用 None、显式关闭用 False，以及新增一个参数需要维护哪些契约。

## 1. 一个参数经过哪些对象

```text
命令行字符串 → argparse 转类型 → Namespace
→ vars(args) 字典 → 去掉 RUN_ONLY_ARGS
→ TrainConfig.merge 忽略 None → 最终 cfg
→ 保存有效配置 → 训练入口使用字段
```

[main.py](../../src/main.py) 的 `build_parser()` 注册参数，`parse_args()` 解析；[config.py](../../src/config.py) 的 `merge()` 决定覆盖语义。不要把 argparse 当成整个配置系统。`--config` 与 `--resume` 是运行指令，在 `RUN_ONLY_ARGS` 中；它们不是 TrainConfig 超参，直接交给 merge 会造成未知字段错误。

优先级是 CLI > YAML > dataclass 默认值。假设默认 epochs=3、YAML epochs=5、CLI 未传 epochs，最终是 5；CLI 传 2，最终是 2。若 argparse 默认也给 3，未传参数仍会把 YAML 的 5 覆盖掉。None 在这里表示“用户没有表达这个覆盖”，而非训练参数本身的有效数值。

## 2. argparse 五个关键旋钮

| 参数 | 管什么 | 例子与边界 |
|---|---|---|
| `type=int` | 把字符串转整数 | `--epochs abc` 解析失败；不负责所有取值范围 |
| `choices` | 限定选项集合 | 未知 dataset 在进入训练前拒绝 |
| `default=None` | 区分未传与显式覆盖 | 不用一个默认数偷偷盖掉 YAML |
| `action=BooleanOptionalAction` | 支持正反布尔开关 | `--amp`/`--no-amp` 都写入 `args.amp` |
| `dest`/`help` | 属性名与帮助信息 | 连字符一般映射成下划线；别猜特殊 action 的行为 |

`Namespace` 是存属性的对象，`vars(namespace)` 提供字典。`--epochs=2` 与 `--epochs 2` 均可。解析失败一般先打印 usage 再退出，帮助信息应能让使用者知道允许的值和参数作用。异常信息是接口的一部分，字段写错时应早点拒绝。

## 3. 亲手看 None、False、0 的区别

```python
# 可运行实验：同一个配置在三种 CLI 情况下的结果
import argparse
from src.config import TrainConfig

p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=None)
p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
base = TrainConfig(epochs=5, amp=True)
expected = [(5, True), (2, True), (5, False)]
for argv, answer in zip([[], ['--epochs', '2'], ['--no-amp']], expected):
    parsed = p.parse_args(argv)
    final = base.merge(**vars(parsed))
    print(argv, vars(parsed), final.epochs, final.amp)
    assert (final.epochs, final.amp) == answer
assert base.epochs == 5 and base.amp is True
assert base.merge(warmup_steps=0).warmup_steps == 0
try:
    base.merge(bogus_field=1)
except ValueError:
    print('未知字段已拒绝')
else:
    raise AssertionError('未知字段未拒绝')
```

`if value` 会把 False、0、空字符串也过滤掉，不能用它替代 `if value is not None`。warmup_steps=0 是明确关闭 warmup，不代表没传。`dataclasses.replace()` 返回新配置并再次运行初始化校验，原配置不变；这降低了 helper 隐式修改调用方状态的风险。

课程实验脚本放在子目录时，从根目录运行 `python -m outputs.course.config_probe`，可让项目根目录处于模块搜索路径。直接 `python outputs/course/config_probe.py` 时，搜索路径的第一个位置可能是脚本子目录；这是“明明仓库里有 src 但导入失败”的常见原因。其他只有 torch 导入的实验可直接按文件运行。

## 4. dataclass 做到了什么，没做到什么

dataclass 自动生成初始化、比较和表示方法；类型注解帮助阅读与静态工具，通常不自动完成运行时类型检查。`__post_init__()` 是实例创建后做校验的地方。当前仓库明确检查 batch_size>0、grad_accum_steps>0、val_ratio 范围和 CPU+AMP 组合。

**当前配置没有完整校验所有数值字段。** 例如负学习率不是在 `TrainConfig.__post_init__()` 被拦住，后续 optimizer 构造才会拒绝。旧 argparse 加餐把所有非法数值都归因于 config 校验并不准确。学习时分别追踪：解析层能转什么类型、配置层检查什么约束、底层组件还检查什么。

`from_yaml()` 用 safe_load 读取字典后创建 TrainConfig；未知 YAML 字段通常在构造时抛 TypeError，未知 merge 字段由显式检查抛 ValueError。这两条入口的异常类型不同。不要为了“看起来统一”用宽泛 except 返回默认配置，否则打错字段可能静默跑错实验。

`num_classes`、`in_channels` 由 dataset 元信息派生，CLI 特意不开放手动设置。其他字段应能从 CLI 到达。新增字段时先问它是用户输入、运行指令还是派生值，分别决定是否进入 dataclass、RUN_ONLY_ARGS 或派生流程。

## 5. 配置文件与运行配置不是同一个快照

```bash
python src/main.py --config configs/mnist.yaml --dataset synthetic --device cpu --epochs 1 --batch_size 16 --exp_name course02 --output_dir outputs/course
python -m pytest tests/test_config.py -q
```

查看 `outputs/course/course02/config_resolved.yaml`，找到 dataset、device、epochs、batch_size。对照原 YAML，解释每个变化来自哪层。不要只把原 YAML 当成实验凭证，它没有包含这次 CLI 覆盖和数据元信息派生。

配置同值也不等于实验完全相同，还需要代码版本、依赖版本、数据版本、种子和执行环境。日志应记录“输入配置”与“实际生效配置”的关系；研究结论必须引用最终生效配置。

失败练习：`python src/main.py --epochs abc` 是解析失败；`--dataset unknown` 是 choices 失败；`--batch_size 0` 会进入配置约束检查。运行前预测各错误发生在哪层，再看 traceback。使用独立输出名称，避免把失败前留存的旧 summary 当成这次的结果。

## 6. 读懂工程里的 Python 结构

`Path` 负责表达路径，`mkdir(parents=True, exist_ok=True)` 创建需要的父目录；它不证明该路径是你想要的目录，写入前仍要看有效配置。文件读取显式给 UTF-8；写入涉及保留原字节时还要注意 LF/CRLF，不能仅用“文本看起来一样”证明还原。

闭包把外层变量带到内层函数，例如第三节 scheduler 捕获 base_lrs；这让行为不仅取决于函数参数，还取决于构造时的状态。context manager 的 `with` 管理一段作用范围，AMP/no_sync 都靠它；缩进改变范围可能改变语义。

模块划分应围绕职责：main 做组装，config 管配置契约，data 管采样加载，train 管更新，checkpoint 管恢复。函数文档最好说明输入、输出、会修改哪些状态、成立前提。不要把同一 optimizer 创建逻辑分别复制到训练与 LR finder，它们会逐渐偏离。

`python -m src.main` 与 `python src/main.py` 在本项目均有入口支持，但模块搜索机制不同。生产工程更适合稳定的包入口与安装流程；本课程先理解现有 sys.path 兼容逻辑，再设计是否需要迁移，不为整理目录随意搬文件。

## 7. 配置契约与破坏实验

找 [test_config.py](../../tests/test_config.py) 的 `test_every_config_field_is_reachable_from_the_cli` 与 `test_no_cli_flag_is_unknown_to_the_config`。它们约束配置字段和 CLI 双向对应，派生字段有例外。这种测试验证工程接口，不是验证某次训练准确率。

在独立分支把某个参数的 `default=None` 改成一个具体值，运行上面的覆盖实验，观察 YAML 是否被未传 CLI 盖掉；再把 merge 中 `is not None` 暂改成 truthiness 过滤，观察 `--no-amp` 为什么失效。恢复文件并跑 test_config。记录原版和破坏版的行为，不用“可能有坑”代替证据。

## 8. 自测、答案与验收

1. CLI 未传 amp，YAML 为 True，最终是什么？True；None 被忽略。
2. CLI 显式 `--no-amp` 为什么保留？False 是有效指令，不能按真假过滤。
3. 类型注解写 int 就能拒绝字符串吗？通常不能，需要解析和校验。
4. cfg.merge 会改 base 吗？当前实现返回新 dataclass。
5. 所有错误都该 except 后默认继续吗？不该，错误配置继续训练可能产生可信外观的错误结果。
6. 为什么保存 resolved 配置？它能展示合并与派生后的实际值，但还不是完整复现证据。

验收：能把 `--batch_size 16` 追到保存的配置，展示三种布尔状态实验，指出一个现有校验缺口，解释两条配置契约测试。输出一页“字段→来源→覆盖规则→生效位置”表。

参考：[Python argparse](https://docs.python.org/3/library/argparse.html)、[dataclasses](https://docs.python.org/3/library/dataclasses.html)、[pathlib](https://docs.python.org/3/library/pathlib.html)。

## 9. 拆解一个新增参数的工程过程

假设未来加入 label_smoothing。先定义它的范围与默认值，判断它是可序列化超参；进入 TrainConfig 后，CLI 默认仍为 None；在损失函数构造处消费它；存有效配置；补覆盖与非法取值判据。只添加命令行开关却没有交到 criterion，会产生“显示启用但训练没变”的假功能。

这项练习先写设计表，不必当场改训练代码：

| 位置 | 必须回答的问题 |
|---|---|
| 默认配置 | 默认值是否保持原训练语义？ |
| 解析层 | 类型与 help 是否清楚，未传是否 None？ |
| 校验层 | 是否有限、是否在允许区间？ |
| 消费层 | 究竟影响哪个计算对象？ |
| 存档层 | 是否出现在 resolved 配置和 checkpoint？ |
| 验证层 | 关掉它是否恢复基线，错误值是否早点拒绝？ |

异常分层再练一次：参数拼错在 argparse；未知 merge 键在配置接口；文件不存在在 from_yaml；数据标签越界可能在 criterion；保存磁盘失败在 IO。定位依赖最后一个实际抛错位置和上游值来源，不能看到 ValueError 就归因于同一层。

`try/finally` 适合清理必须释放的资源，即使异常也执行；`except` 是决定如何处理失败。LR finder 使用 finally 恢复模型，分布式入口需要释放进程组。写宽泛 except 时应能说清哪些失败允许降级、哪些必须传播；“失败后继续默认训练”会破坏配置可信性。

再看 `TrainConfig.summary()`、`to_yaml()` 与 `to_dict()`：同一个字段进入人类日志、有效配置文件和机器记录。三种表示应互相一致；拿到一个 config 字典后偷偷改值，可能让存档与本次计算不同。完成本节后，把你选择的五个参数画成“输入→合并→消费→记录”四步图。
