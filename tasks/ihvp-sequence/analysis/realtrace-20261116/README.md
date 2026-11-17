# IHVP 真实训练轨迹：2026-11-16 诊断交接

本目录仅供评测维护者。这里收录15条public＋30条private候选轨迹的诊断报告、固定方案源码、HVP运行时与复核入口；完整用例、参考向量和生成种子单独保管。数学任务仍为16384维、50步SPD系统，单核30秒/序列、求解进程峰值256MiB。

**结论：暂不直接替换正式题。** 2250个系统的数学硬约束检查通过，10种程序的450次评测全部合规；但Cold CG在public和private均优于全部已测改进方案，尚无有效的0/50分锚点。谱预条件PCG只是在另外五个改进候选中最好，public上也弱于Cold CG。当前只报告raw loss，不给正式分数，不接入榜单。

## 阅读入口

| 文件 | 用途 | 能否给模型 |
|---|---|---|
| [reports/REPORT.md](reports/REPORT.md) | 完整结论、假设检验和方法对照 | 否 |
| [reports/baseline_summary.csv](reports/baseline_summary.csv) | 分public/private/combined的loss、时间、内存、HVP数与合规数量 | 否 |
| [reports/quality_by_profile.json](reports/quality_by_profile.json) | 固定批次与重采批次的相关性诊断 | 否 |
| [reports/DATA_PACKAGE.md](reports/DATA_PACKAGE.md) | A服务器完整数据ZIP位置、大小与SHA256 | 否 |
| [TESTER_PROMPT.md](TESTER_PROMPT.md) | 可直接交给评测维护者的复核任务 | 否 |
| [PROTOCOL.md](PROTOCOL.md)、[USAGE_CURATOR.md](USAGE_CURATOR.md) | 构造和预算说明 | 否 |
| [model_materials/](model_materials/) | 通用题面、接口、零输出模板、4维3步smoke | **仅此目录可以** |

## 已有实测

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

## 完整数据

完整评测人ZIP为14.62GiB，在5090-A：

```text
/mnt/sdb/henryzhang/ihvp_realtrace_20261116/var/release/EVALUATOR_ONLY_ihvp_realtrace_15public_30private_20261116.zip
```

SHA256：`acfebb847feacbf0294a7df948473e07dc6c407c00eddabf7d193c66cd352a3f`。

按仓库[提交规则](../../../../docs/TASK_CHANGES.md)，GitHub不存private用例、参考答案或生成种子。应由维护者通过既有授权渠道接收这个ZIP；不要用新种子重建后声称复现同一批数据。先建立权限700的评测端目录再解压，不向模型挂载。

## 复核命令

目标是Linux amd64、单CPU线程。评测依赖见[requirements-eval.txt](requirements-eval.txt)；生成器另需原记录的PyTorch环境，重放已有数据不需要GPU或PyTorch。镜像应预装C++编译器及当前Python开发头文件。所有安装与构建在计时前完成，正式运行断网。

进入本目录，准备已解压的数据根目录（包含public_00…public_14和private_00…private_29子目录）：

```bash
python -m pip install -r requirements-eval.txt
python code/build_extension.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c <评测核> python code/validate.py --data <完整数据ZIP解压目录>/data --out var/qc
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c <评测核> python tools/replay.py --data <完整数据ZIP解压目录>/data \
  --out var/replay-public --split public
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c <评测核> python tools/replay.py --data <完整数据ZIP解压目录>/data \
  --out var/replay-private --split private
```

先冻结源码、构建产物与参数再运行private。`--methods`可指定固定方案，完整列表用`--help`查看。replay会记录当前源码/运行时哈希，拒绝静默降级到未编译的Sol单轮实现；原始封存回执仅供追溯，不保证与本源码交付子集的目录哈希相同。换机器后重测raw loss，不把B机数字移作新平台锚点。

Linux峰值使用进程自身的VmHWM及独立RSS监视，避免把fork前评测器的高水位算入求解器。256MiB限于求解进程，私有缓存与HVP提供器工作区在评测端。运行时提供进程协议与可选UID保护；正式模型测试必须另外实施只读挂载、网络/进程限制，不能让模型访问维护者目录。

## 生成交接ZIP与模型可见边界

```bash
python tools/package_handoff.py
```

输出`dist/MODEL_ONLY_ihvp_realtrace_20261116.zip`与`dist/EVALUATOR_ONLY_ihvp_realtrace_tools_20261116.zip`。前者严格只含model_materials下的五个白名单文件；后者是维护者源码与汇总报告，完整算子数据仍单独交接。

模型仅能读取MODEL_ONLY包和自己的代码。禁止给模型本README、生成器、runtime/backend.py、其他方案源码、汇总private结果、评测ZIP或原仓库检出目录。public池也不给模型，只反馈整池汇总；private不用于开发反馈。

没有任何生成种子或完整规模数组随此GitHub交付。`code/generate.py`只供维护者审查，不用它重新抽样本替代封存用例。下一版如调整采样窗口或训练设置，应先在public做控制实验，再用全新private种子检验；本批只保留为诊断记录。
