# IHVP 真实训练轨迹：2026-11-16 诊断交接

本目录仅供评测维护者。这里收录15条public＋30条private候选轨迹的诊断报告、固定方案源码、HVP运行时与复核入口；完整用例、参考向量和生成种子单独保管。数学任务仍为16384维、50步SPD系统，单核30秒/序列、求解进程峰值256MiB。

**当前划分（balanced-split-20261117）：** 将原45条完整轨迹按数据集×场景分层，使用两份固定CG参考的难度约束进行随机重分，保持15 public＋30 private。Cold CG两池均值比由12.82降至1.48，Warm CG由34.96降至1.42。当前实际归属以完整数据包内的`balanced-evaluator/split_manifest.json`为准，原目录名和meta中的split保留为源身份；运行必须传`--split-manifest`。详见[重划报告](reports/BALANCED_SPLIT_20261117.md)。原数据内容、原始loss及旧划分报告保留；这仍是已有数据的重划回放。

**当前模型题面（sequence-statement-20261117）：** 恢复通用应用背景、时序求解目标，以及允许保留历史解、搜索方向、低秩近似和预条件信息的说明；保留原题“不保证历史信息下一步仍有效”的边界。不声称已覆盖旧合成生成器的五类谱结构，也不公开真实数据来源、构造细节或既有成绩。此前neutral-statement-20261117精简版作为历史发布保留。当前修订不改变系统、划分清单、参考身份、资源预算、loss或得分映射，应通过新的模型解题运行比较题面影响。

**当前参考身份：Warm CG＝0分，Cold CG＝50分。** balanced-split-20261117回放的public常数为Warm 0.07454791、Cold 0.02278136，private为Warm 0.05260912、Cold 0.01541608；算法身份固定，目标机仍须分池重测。见[当前标定](reports/anchor_calibration_balanced-split-20261117-cg50-warm0.json)和[当前分数CSV](reports/scores_balanced-split-20261117-cg50-warm0.csv)。参考身份最初根据原划分的已见private结果指定，溯源见[原评分修订](reports/ANCHOR_CALIBRATION_20261117.md)；这是事后回放，不回写旧协议正式成绩，也不宣称新的独立盲测。

2250个系统的数学硬约束检查、450次预算评测均通过。原CG＝0、L-BFGS-PCG＝50的锚点仍然无效；新的Warm/Cold映射修复了计分次序，但没有改变相关性偏弱的诊断。当前正式v4、旧成绩和榜单保持原样。

## 阅读入口

| 文件 | 用途 | 能否给模型 |
|---|---|---|
| [reports/BALANCED_SPLIT_20261117.md](reports/BALANCED_SPLIT_20261117.md) | 当前重划方法、难度对照和冻结方案重计分 | 否 |
| [reports/balanced_split_summary.json](reports/balanced_split_summary.json) | 当前清单哈希、选取规则与聚合难度 | 否 |
| [reports/ANCHOR_CALIBRATION_20261117.md](reports/ANCHOR_CALIBRATION_20261117.md) | 当前Warm＝0、Cold＝50的回放标定 | 否 |
| [reports/REPORT.md](reports/REPORT.md) | 原始诊断、假设检验和方法对照 | 否 |
| [reports/baseline_summary.csv](reports/baseline_summary.csv) | 分public/private/combined的loss、时间、内存、HVP数与合规数量 | 否 |
| [reports/quality_by_profile.json](reports/quality_by_profile.json) | 固定批次与重采批次的相关性诊断 | 否 |
| [reports/DATA_PACKAGE.md](reports/DATA_PACKAGE.md) | A服务器完整数据ZIP位置、大小与SHA256 | 否 |
| [TESTER_PROMPT.md](TESTER_PROMPT.md) | 可直接交给评测维护者的复核任务 | 否 |
| [PROTOCOL.md](PROTOCOL.md)、[USAGE_CURATOR.md](USAGE_CURATOR.md) | 构造和预算说明 | 否 |
| [model_materials/](model_materials/) | 通用题面、接口、零输出模板、4维3步smoke | **仅此目录可以** |

## 原划分已有实测（归档）

下表为30条private的序列等权平均能量loss，不是HERO分类训练loss。

| 方法 | Private loss ↓ |
|---|---:|
| Cold CG | 0.00361752 |
| Warm CG | 0.00486434 |
| L-BFGS-PCG | 0.04496859 |
| Ritz-CG | 0.10497852 |
| POD-CG | 0.08574777 |
| 谱预条件PCG | 0.02704754 |
| GPT-6.1 Sol 单轮 | 0.01055912 |
| GPT-6.1 Sol 迭代 | 0.00489760 |
| Opus 5.5 单轮 | 0.06726108 |
| Opus 5.5 迭代 | 0.02029713 |

450次运行最长28.670秒，最大求解进程峰值RSS109.262MiB。参考解独立CPU证书最大9.98e-19。阶段内b相邻余弦中位数0.0327；固定批次场景为0.9949，其余四类重采批次场景接近0。该数据是实际小网络HERO变体训练轨迹，并非原大网络或原ZO配置的逐步重放；不能据此推广成“真实HERO都没有相关性”。

## 重标分数

当前重划后应使用同一清单汇总，新标定不会覆盖原划分输出：

```bash
python tools/calibrate_anchors.py --results var/balanced-public var/balanced-private \
  --split-manifest <完整数据包解压目录>/balanced-evaluator/split_manifest.json \
  --out var/balanced-calibration
```

当前封存重计分见[CSV](reports/scores_balanced-split-20261117-cg50-warm0.csv)。下文未传清单的命令仅用于原划分溯源。

现有450次运行的原始loss和资源记录保持不变，先按池平均loss，再应用一次评分映射：

```bash
python tools/calibrate_anchors.py
python tools/check_calibration.py
```

新平台复跑后，应从同平台、同一批数据的逐例结果重新标定；不跨平台沿用这里的数值：

```bash
python tools/calibrate_anchors.py --results var/replay-public var/replay-private --out var/new-calibration
```

参考必须在整个相应池上全部合法；数据缺失或未标定记未测，提交失败按0分处理。public的数值不用于private，不能逐例打分后平均。

## 完整数据

按用户2026-11-17明确授权，完整评测包以8个分卷附件放在本私有仓库的[当前Release](https://github.com/IceySakura/FrontierAM/releases/tag/ihvp-sequence-statement-20261117)，不再只有远程路径说明。维护者须登录有本仓库读取权限的GitHub账号。下载`download_data.py`并运行，它会取得`IHVP_DATA_PARTS.json`和8个分卷，逐个校验、合并为完整ZIP并校验完整SHA256：

```bash
gh release download ihvp-sequence-statement-20261117 --repo IceySakura/FrontierAM --pattern download_data.py
python3 download_data.py --out ./ihvp-data
```

预留约32GiB空间（分卷和合并后的ZIP各占一份），分卷不能分别解压。完整包名为`EVALUATOR_ONLY_ihvp_balanced_15public_30private_statement_20261117.zip`，SHA256为`dd375126fe4d6a0dec6ac569034cc9b477d376c12be67e62724e28f72d3876a3`。它与A服务器原文件完全相同，包含当前恢复版题面、同一balanced-split-20261117清单和同一45条数据。解压后从`START_BALANCED_EVALUATION.md`进入，使用`balanced-evaluator/`内工具。已有上一版完整数据的维护者可只下载本次增量ZIP。

下面是内容未改动的原始来源包，单独使用它不会应用新划分：

完整评测人ZIP为14.62GiB，在5090-A：

```text
/mnt/sdb/henryzhang/ihvp_realtrace_20261116/var/release/EVALUATOR_ONLY_ihvp_realtrace_15public_30private_20261116.zip
```

SHA256：`acfebb847feacbf0294a7df948473e07dc6c407c00eddabf7d193c66cd352a3f`。

本次私有Release交付是用户对本IHVP数据包明确授权的例外，不改变其他任务的[提交规则](../../../../docs/TASK_CHANGES.md)。完整数据、分卷和增量包仅供评测维护者，不提供给求解模型；模型不能读取Release附件或GitHub访问凭据。不要用新种子重建后声称复现同一批数据。先建立权限700的评测端目录再解压，不向模型挂载。

## 复核命令

目标是Linux amd64、单CPU线程。评测依赖见[requirements-eval.txt](requirements-eval.txt)；生成器另需原记录的PyTorch环境，重放已有数据不需要GPU或PyTorch。镜像应预装C++编译器及当前Python开发头文件。所有安装与构建在计时前完成，正式运行断网。

进入本目录或新完整包的`balanced-evaluator/`，准备已解压的数据根目录（保留源public_00…public_14和private_00…private_29子目录）及新分组清单：

```bash
python -m pip install -r requirements-eval.txt
python code/build_extension.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c <评测核> python code/validate.py --data <完整数据ZIP解压目录>/data --out var/qc
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c <评测核> python tools/replay.py --data <完整数据ZIP解压目录>/data \
  --split-manifest <完整数据ZIP解压目录>/balanced-evaluator/split_manifest.json \
  --out var/balanced-public --split public
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c <评测核> python tools/replay.py --data <完整数据ZIP解压目录>/data \
  --split-manifest <完整数据ZIP解压目录>/balanced-evaluator/split_manifest.json \
  --out var/balanced-private --split private
```

先冻结源码、构建产物与参数再运行private。`--methods`可指定固定方案，完整列表用`--help`查看。replay会记录当前源码/运行时哈希，拒绝静默降级到未编译的Sol单轮实现；原始封存回执仅供追溯，不保证与本源码交付子集的目录哈希相同。换机器后重测raw loss，不把B机数字移作新平台锚点。

Linux峰值使用进程自身的VmHWM及独立RSS监视，避免把fork前评测器的高水位算入求解器。256MiB限于求解进程，私有缓存与HVP提供器工作区在评测端。运行时提供进程协议与可选UID保护；正式模型测试必须另外实施只读挂载、网络/进程限制，不能让模型访问维护者目录。

## 生成交接ZIP与模型可见边界

```bash
python tools/package_handoff.py
```

输出`dist/MODEL_ONLY_ihvp_realtrace_20261116.zip`与`dist/EVALUATOR_ONLY_ihvp_realtrace_tools_20261116.zip`。前者严格只含model_materials下的五个白名单文件；后者是维护者源码与汇总报告，完整算子数据仍单独交接。

模型仅能读取MODEL_ONLY包和自己的代码。禁止给模型本README、生成器、runtime/backend.py、其他方案源码、汇总private结果、评测ZIP或原仓库检出目录。public池也不给模型，只反馈整池汇总；private不用于开发反馈。

Git源码和MODEL_ONLY材料不含私有用例数组、参考解或生成种子；完整评测资料仅在经用户授权的私有Release数据附件中交给维护者。`code/generate.py`只供维护者审查，不用它重新抽样本替代封存用例。下一版如调整采样窗口或训练设置，应先在public做控制实验，再用全新private种子检验；本批为已有数据回放。
