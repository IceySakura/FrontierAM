# IHVP v4 使用指南（仅维护者，禁止交给被测模型）

本版采用 **15 条全新 public＋30 条全新 private**，不再把旧 15 条容易数据混进正式计分。旧 v1–v3 数据和成绩完整保留，仅作回放诊断。它是替换本版评测池的交接包，不是追加到旧清单的增量。

## 文件与访问边界

交接有两个 ZIP：

| 材料 | 接收者 | 用法 |
| --- | --- | --- |
| `ihvp-hard-v4-with-guide.zip` | 维护者/评测宿主 | 完整教师数据、参考解、清单、源代码、验证记录和本指南；不得挂载到开发容器 |
| `ihvp-v4-model-visible.zip` | 被测模型 | 唯一允许解压到开发工作区的题目材料，恰好 6 个文件 |
| `.zip.sha256`、`.manifest.json` | 维护者 | 交接完整性核验；不是额外的模型材料 |

**模型只可以读取以下题目文件：**

1. `/workspace/problem.md`
2. `/workspace/samples/README.md`
3. `/workspace/samples/run_sample.py`
4. `/workspace/samples/smoke.json`
5. `/workspace/samples/support/api.py`
6. `/workspace/samples/support/worker.py`

模型还可以读写自己的 `/workspace/submission/`，使用已装好的通用数值库及其通常的文档。小型 `d=4,T=3` smoke 中的矩阵和答案是明确允许的协议示例，不代表完整规模数据的可见性。

**模型不可以读取或获得：**

- 完整交接 ZIP、这份指南、仓库根目录、Git 历史、PR 讨论或本次出题研究对话。
- `curator/`、生成脚本、数据构造说明、场景/变换配置、设计收据、冻结计划或随机种子。
- `data/public/`、`data/private/`、所有完整规模 NPZ、参考解、教师因子、manifest 和逐例验证记录。
- `grader/` 的宿主代码、`references/`、其他模型历史提交、回测源码或分析报告。
- 其他运行/序列保留的文件、评测日志、缓存或 private 反馈。

用文件系统隔离落实边界；不能仅在模型提示词里写“不要看”。开发容器只含六文件材料、候选源码及常规依赖，只挂载独立的样例 socket 目录。不得把 `/secure/ihvp-v4`、整个仓库、用户主目录或带私有文件的父目录挂载进去。保持断网，避免它从 GitHub 或旧对话找出本版生成器/验证记录。

## 导入与核验

在安全宿主目录解压总包。以下命令中的路径需要按机器实际情况替换；完整仓库须包含本版 PR 的代码，`source/` 只是审阅/溯源副本，不是一个独立安装完毕的仓库。

```sh
shasum -a 256 -c ihvp-hard-v4-with-guide.zip.sha256
unzip ihvp-hard-v4-with-guide.zip -d /secure/ihvp-v4
cd /path/to/FrontierAM
python scripts/seal/prepare_ihvp_hard_v4.py \
  --bundle-root /secure/ihvp-v4 \
  --out var/raw/ihvp-sequence-v4-import/data
```

导入器校验 15/30 数量、教师字节 SHA、数据 schema、维度、时间步、split、设计来源及两个池之间的重复；保持原始 NPZ 字节不变，并建立两条 public 样例引用。已存在的输出目录不覆盖。public/private 各自的源清单哈希记录在 `IMPORT.json`。

目标 Linux amd64 镜像已按本版 Dockerfile 离线构建后，可用独立 v4 封存脚本接入评测槽（本地无 Docker，因此这里不声称执行过此步）：

```sh
python scripts/seal/seal_ihvp_v4.py \
  --data-root var/raw/ihvp-sequence-v4-import/data \
  --image frontieram-ihvp-sequence:amd64-v4
```

此脚本验证数据、镜像平台和六文件白名单，生成独立 `ihvp-sequence-v4` 的封存锁；不覆盖旧锁，也不会把草稿自动改为正式。样例锁直接指向 public 宿主教师，不复制到模型工作区。public/private 各自重新标定、隔离及稳定性验收完成后，维护者再按仓库流程启用。

两个完整规模样例固定选自 public 的第一条与最后一条，不额外复制教师文件。不能把 `data/samples/manifest.json` 交给模型；它供宿主解析引用。

不要运行生成器重新生成这轮评测数据。交接数据已经冻结，应按 SHA 原样导入；若确需以后重生成，应发布新版本并保留本轮封存。不得看过 private 算法结果后重新抽种子并继续沿用 v4 名称。

## 启动样例服务

在带 NumPy/SciPy 的宿主 Python 环境中启动，只读 public 与 samples，不打开 private：

```sh
python scripts/ops/serve_ihvp_v4_samples.py \
  --data-root var/raw/ihvp-sequence-v4-import/data \
  --socket-dir /tmp/ihvp4-samples
```

socket 目录须不存在、路径足够短，且与数据根分开。在**同一个 Linux 内核宿主**的开发容器里，只将该目录绑定到 `/run/ihvp-samples`。该手工样例服务是开发诊断工具，不提供正式进程内存隔离认证；Linux amd64 的正式评测应使用受隔离的 driver。macOS 与 Docker Desktop 跨内核的 Unix socket 挂载不保证可用。

把六文件小包解压到模型的 `/workspace/`，由模型运行：

```sh
python3 /workspace/samples/run_sample.py smoke
python3 /workspace/samples/run_sample.py sample_01
python3 /workspace/samples/run_sample.py sample_02
```

候选可以通过当前 HVP 合法探测结构、学习预条件器、复用历史，这些属于允许的算法。不能把“避免 hack”理解成禁止一切从合法查询中识别结构的行为。

## 测试与反馈策略

1. **public 开发反馈**：使用完整 15 条 public，约 7.5 分钟求解时间，加上装载/验分，目标在仓库每次 15 分钟上限内。只反馈整池分数、loss/参照比值和汇总资源结果，不反馈逐例、变换配置、内部生成类别或小分组结果。`arena/public_v63.py` 对 v4 使用独立 `category=null` 配置，不改变旧版本反馈。
2. **最终冻结**：开发结束后封存唯一最终提交的源码、离线构建输入、选项、依赖、镜像与哈希。冻结后才能运行 private；不要把 private 总分、分组成绩或失败用例回传给还在开发的模型。
3. **private 最终评测**：30 条全新 private，约 15 分钟求解时间。先逐序列对 50 步等权平均，再对 30 条等权平均。旧数据的正式权重为 0，旧回放不得并入分母。
4. **版本与隔离**：v4 当前为草稿；按仓库流程在目标 Linux amd64 槽上封存新数据和镜像、重测参照并验证隔离。public/private 评分锚点独立，禁止沿用 v3 锚点或本地 Apple 芯片诊断值。正式启用后再产生 v4 的新成绩格子，保留旧版所有成绩。
5. **后续研究**：完成最终 private 评测之后，同一 holdout 不再用于新一轮调参然后仍称为“未见测试”。要继续开发，使用新版本和新 holdout；旧 private 只作已公开的回放诊断。

不要仅把 private 加到 30 条，却仍向模型反复回传 private 汇总：汇总本身也能构成调参反馈。15:30 主要改善留出规模与旧易例稀释；随机路由消除已知配对指纹，二者均不提供不可学习的证明。

## 资源、计分与本地检查

保持单 CPU 线程、`d=16384,T=50`，每条完整序列及整批平均时间均不超过 30 秒，最高 worker 峰值 RSS 不超过 256 MiB。启动、导入、通信、HVP、预条件构造、求解和输出全部计时；宿主教师装载和独立验分不计时，也不计入候选 RSS。任一提交导致的超限、非法向量、缺失输出或异常退出使整份提交 0 分；基础设施失败为“未测”。

原始 loss 是归一化平方 H 能量误差，零向量为 1、精确解为 0。固定冷启动 CG 为 0 分锚点，L-BFGS-PCG 为 50 分，零误差为 100，映射不变。目标机必须重新测得有序的 `E0>E50>0`。参照会按剩余时间用满预算；仓库的余量政策是否沿用旧版处理，由维护者在启用时明确记录，不能声称本地实验已满足 25% 余量门槛。

维护端可运行协议 smoke（不产生正式分数）：

```sh
python arena/operator_drivers/ihvp_hard_v4.py --smoke \
  --out var/outputs/ihvp-v4-smoke
python -m pytest -q arena/tests/test_ihvp_routed_v4.py \
  arena/tests/test_ihvp_routed_integration.py
```

本轮 public 资格审查使用 CG、L-BFGS-PCG、**原样冻结的 v3 迭代提交**；完整回测和数学审计记录在 `validation/`。旧程序源码与选项在 public 测试前已固定，未用 private 调参。private 仅做数学正确性审计；尚未对其做算法优劣筛选。

`source/tasks/ihvp-sequence/analysis/HARD_V4.md` 解释实测范围。已测已知程序失败不等于不存在新算法：随机路由、复合变换或全秩时序仍可能被更好的合法方法利用。

## 给测试方的交接文字

请按 IHVP v4 重测：本版采用 15 条全新 public＋30 条全新 private，旧 v1–v3 数据只回放、不参与本版计分；资源与 CG/L-BFGS-PCG 评分映射不变。模型工作区只能拿六文件 `ihvp-v4-model-visible.zip`，完整包、仓库、指南、生成器、教师数据、参考解和历史提交均留在宿主侧。请重测本机参照，只给模型 public 整池汇总反馈，不返回逐例/构造分组；最终源码、构建与参数封存后再跑 private，开发期间不反馈 private。请先回放冻结的 v3 迭代提交，再开展新一轮模型测试，并将 v4 与历史版本的结果分开记录。
