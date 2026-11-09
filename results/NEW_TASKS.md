# 新题整合状态（2026-11-05，11-09 更新）

**11-09 状态**：五道新题都已接入并有本机基线，gpt-6.1-sol（v6.1，一次 2 小时断网尝试）的成绩如下，原始指标、映射和得分见 `LEADERBOARD.md`：

| 题目 | sol 得分（满分 100，参考 50） |
| --- | --- |
| galaxy-gravity | 100.0（题目允许暴力求精确解，已反馈出题人） |
| polarization-demosaicking | 50.7（取 15:09 的重评，默认决定，待讨论） |
| trajectory-covariance | 61.5（3000 例固定子集） |
| species-tree | 暂停：出题人告知数据有误，提交已冻结待评 |
| exponential-v2 | 62.7（74/74 合格，R = 0.596，100/(1+R)；几何速度比 2.01，95% 区间 1.45–2.91；`EXPONENTIAL_V2.md` 第 4d 节） |

另：出题方的 conic 修正版题包（solver-20261105）已接入并重测 sol，见 `CONIC_REVISION_SOL.md`。

**11-10 状态（协议 v6.2，2 小时单轮）**：两道新版本题与 opus 的两道 operator 题评分结果，详见第 5 节。

| 题目 | gpt-6.1-sol | claude-opus-5-5 |
| --- | --- | --- |
| species-tree-v2（200 例，250 物种） | 0（漏掉全部分裂） | 42.3 |
| galaxy-gravity-v2（131,072 粒子，5 CPU 秒） | 99.9996 | 99.991 |
| polarization-demosaicking | —（v6.1：50.7） | 待重评（第一次评分只有第一景超时） |
| trajectory-covariance | —（v6.1：61.5） | 51.9 |

`Raw_Resource/` 中的 4 道新题已接入 FrontierAM 框架。原始包不进 git；`scripts/prepare_operator_tasks.py` 按 `tasks/operator/<题目>/task.json` 核对哈希、解压、校验包内清单、应用可移植性修改并封存到 `.runtime-ops/<题目>/`。评测时运行题目自带的评分器，通过 docker 垫片（`arena/docker_shim.py`）把它的 ARM64 平台、CPU 编号和镜像映射到本机评测 slot。

## 1. 各题概况

| 题目 | 原始包 | 包内清单校验 | 能做到的程度 |
| --- | --- | --- | --- |
| galaxy-gravity | operator 1.0.1 + public 1.0.1 | SHA256SUMS 82 个文件全部通过 | 完整评测（20 张私有快照，两条 Treecode2 参照） |
| polarization-demosaicking | maintenance v1.2 + public v1.2 | MANIFEST 199 个文件全部通过 | 完整评测（32 景，双线性和镜像 LEIC 两条参照） |
| species-tree | operator **1.0.1** + public v1 | SHA256SUMS 132 个文件全部通过（含 101 个私有数据文件） | 完整评测：50 个正式隐藏用例，经包自带的 `formal_cases` 入口加载（激活信息、标定回执、数据封签、逐例哈希全部核验通过）；1.0.1 与 1.0 的操作端代码相同，只多了私有数据 |
| trajectory-covariance | public v1 + 出题方包 `mcmc-long-run-covariance-maintainer-v1`（19.6 GB 压缩包，解压 29 GB，放在根分区 `/srv/frontieram-raw/`，封存时以软链接引用、不复制） | 公开包 2 个、出题方包 265,420 个文件全部通过 | 完整接入：3 万个私有输入与参考矩阵、OBM（弱参照）与拟合 VAR(1)（强参照）、包自带的验证器。默认评测固定的 10% 分层子集（每种设置按 ID 取前 100 条母链，含全部 5 个前缀，共 3000 个输入）；全量 3 万例需分片到多个 slot |

## 2. 默认决定（待讨论）

- **得分**：四道题自带的都是"基础参照 0 分、强参照 5 分、零误差 10 分"的分段映射。按 50/100 约定，最终分 = 原分 × 10；两个锚点在本机重跑两条参照后取实测误差，不用包内的历史值。
- **平台**：四个包都在 Apple M5 上的原生 ARM64 Docker 标定，本机是 amd64。评分器和 worker 不改逻辑，只做以下可移植性修改，每条都要求原文恰好出现一次：
  - galaxy-gravity：worker 的架构检查同时接受 `x86_64`。
  - polarization-demosaicking：worker 的 CPU 检查从"必须是 CPU 0"改为"恰好一个 CPU"。
  - species-tree：worker 的 CPU 检查从"必须是 CPU 0"改为"恰好一个 CPU"。
- **镜像**：原镜像都是 ARM64，本机只能重建 amd64 版；需要的是“镜像里装了什么”，不是镜像文件本身（`tasks/operator/<题目>/Dockerfile.amd64`）：
  - polarization：按冻结镜像的构建历史重建，官方 `python:3.11.15-slim-bookworm`，加上从冻结镜像中原样拷出的 `requirements.lock`（torch 用 CPU 版）。
  - trajectory 与 galaxy：两题共同的基础镜像 `local/trajectory-runtime:b1` 及其基础 `songberg-reproduction:mcmcse-1.5-0` 都在 mcmc 出题方包的 `images/formal-images.tar` 里。后者的 R 是在容器中安装后 commit 的，构建历史看不到，因此从镜像文件里读出 dpkg 状态（比官方 Python 基础镜像多 165 个包，其中 2 个仅 ARM64 存在）、`requirements.lock`（与 polarization 完全相同）和 R 库目录（mcmcse 1.5-0、fftwtools 0.9-11、microbenchmark 1.4.10），按原版本重建为 `frontieram-songberg-reproduction:amd64-mcmcse-1.5-0`（`tasks/operator/_images/songberg-reproduction/`），再照搬包内 `public/Dockerfile` 得到 trajectory 镜像，galaxy 的 Dockerfile 建在其上。之前用锥优化镜像（CPython 3.12.14、NumPy 2.3.5）作替代是错的：真实环境是 CPython 3.11.15、NumPy 1.26.4、SciPy 1.12.0、R 4.2.2；galaxy 的基线因此在新镜像上重测。
  - species-tree：基础镜像 `local/species-tree-cpu:b1` 不在任何包里，仍以锥优化镜像替代（Debian bookworm、gcc/g++ 12、CPython 3.12.14）；如能拿到它的 Dockerfile 或依赖清单可按原样重建。
- **宿主侧评分环境**：`.venv-ops`（Python 3.12.16，NumPy 2.3.5，与 polarization 维护包要求的宿主 NumPy 一致）。
- **评测隔离**：在实测干扰之前，四道题都先用 `grade_pool`（评测期间不与其他评测并发）。

## 3. 已发现的问题（给出题人）

- **trajectory-covariance**：
  - 每份提交的正式评测是 3 万个输入（出题方记录的 9 万次是 OBM、VAR(1) 和一份 agent 提交各 3 万次）；一个验证容器内逐例新起进程，单例上限 30 秒。原机上 OBM 每例约 0.6 秒，全量单核约 5 小时；需与出题人确认正式规模（全量分片或抽样）。
  - 包内 `run_trial.py` 的分片模式拒绝带 `build.sh` 的提交（只支持其封存的三份免构建源码），即包本身尚不支持对任意新提交做全量正式评测；我们的驱动直接启动验证容器，构建一次后跑所选用例。
- **species-tree**：
  - **出题人 11-06 告知数据有误，暂停评测。** 已接入的 1.0.1 包与本机基线保留，但在修正版数据到来之前不评任何提交；gpt-6.1-sol 的 v6.1 提交已冻结待评。
  - 运行镜像的基础镜像 `local/species-tree-cpu:b1` 不在任何包里（README：作为独立维护材料转运），目前用替代镜像；如能拿到它的 Dockerfile 或依赖清单（小文本即可），可按原样重建 amd64 版。原 ARM64 镜像本身在本机无法原生运行，不需要。
  - 正式入口 `run.py` 还要求传入出题方“独立留存”的发布锁摘要和 `SHA256SUMS` 自身摘要；我们用包内 `SHA256SUMS` 与 `formal_cases` 的封签核验代替。
  - zip 包里 `build.sh` 的可执行位在 Python 解压时丢失，导致第一次基线两条参照都在构建前被拒；已在封存脚本中按 zip 记录恢复权限位。
- **galaxy-gravity**：
  - 维护包里混入了 `scripts/weighted_tree_qmc/step5/sandbox.py`（看名字属于 species-tree），但 galaxy 的评分器确实导入了它的 docker 辅助函数，属于正常依赖。
  - README 所说的运行镜像 `runtime/galaxy-b2.docker.tar` 不在包内；它的基础镜像 `local/trajectory-runtime:b1` 在 mcmc 出题方包里，已按原样重建（见第 2 节）。
  - 原机器上，标准参照单次调用最多用 3.09 个 CPU 秒，上限是 5 秒；本机若明显更慢，参照本身可能超限。
- **galaxy-gravity（gpt-6.1-sol 首测后发现）**：sol 的提交在每次调用 5 CPU 秒的上限内（本机最多 3.84 秒）用直接 O(N²) 两两求和算出了精确加速度，相对 RMS 误差 3.5e-15，得到接近满分（包内 10 分 / 本框架 100 分）。两条参照（Treecode2 近似）的误差在 1e-4 量级。题目的粒子数与时限组合允许暴力求精确解，近似算法的优劣无法区分；建议增大 N 或收紧时限，使直接求和超时。计时（本机 EPYC 7K62，每张快照 65,536 个粒子，60 次调用）：sol 最快 3.809 / 平均 3.824 / 最慢 3.841 CPU 秒，各快照几乎相同；弱参照 0.561 / 1.205 / 1.748，强参照 0.718 / 1.290 / 1.690。sol 的直接求和用了 AVX2/FMA 向量化、单精度 rsqrt 加四阶修正回到双精度、牛顿第三定律对称减半、分块和等质量特化（`-O3 -march=native -ffast-math`）；出题人反馈自己测过的暴力法比不上基线，差别应在这些优化上，且本机（AVX2）与原标定机（Apple M5，ARM）的向量化效果不同，新的 N 或时限需在本机确认。
- **polarization-demosaicking（gpt-6.1-sol 首测后发现）**：sol 在 31 景上平均 RMSE 1.1009，略优于强参照同 31 景的 1.1090（18 景更好），但第一个运行的 ball 景超过 1 秒墙钟被终止，其余各景约 0.48 秒；按“任一景失败即整份提交失败”的规则记 0 分。两条参照的第一景只比其余慢十几到几十毫秒。诊断性重评（15:09，同一提交）32/32 景通过，ball 0.591 秒，平均 RMSE 1.09499（19 景优于强参照），得 50.7 分；第一次评分时调度核（与 grade-06 同一 L3 域）上正在跑 panel 成员的编译和验证，判定为基础设施干扰，第一次记为未测，正式得分取重评。教训：评分期间调度核上不得运行繁重的编译或测试，未登记的容器也不得占用评测 CPU（调度器会因此拒绝 grade_pool 租约）。
- **polarization-demosaicking**：每景墙钟上限 1.000 秒，包含 Python 启动和导入。维护包自己写明，本版镜像参照"尚未取得 1 秒端到端的完整 32/32 校准收据"，历史上纯双线性也只有 29/32。在本机可能同样无法通过。

## 4. 进度

| 步骤 | 状态 |
| --- | --- |
| 封存（哈希、包内清单、可移植性修改） | 四题完成 |
| amd64 任务镜像 | 四题构建完成，`uname -m` 为 x86_64。galaxy、trajectory、polarization：CPython 3.11.15 / NumPy 1.26.4（与原镜像一致）；species-tree：替代镜像 CPython 3.12.14 / NumPy 2.3.5 |
| trajectory 验证链路 | 在开发核上用 VAR(1) 参照跑 12 个正式输入（每种设置 2 个）：12/12 合格，逐例误差与出题方记录最大相差 1.2e-13，墙钟中位数为原机的 0.95 倍；验证器的 11 项沙箱自检全部通过（嵌套 PID/用户命名空间、no_new_privs、无 capabilities、uid 1000、Landlock 私有哨兵、`/proc` 三项、禁止 socket 与 mount），运行环境为 x86_64、单核配额、2 GiB、无交换、仅回环网卡（`scripts/trajectory-compare.py`） |
| v6 开发 harness 镜像 | 四题构建完成（`frontieram-<题目>-harness:v6`） |
| 镜像导入 4 个评测守护进程 | 4 个 slot 全部完成（`scripts/load-grade-images.sh`，只在 slot 空闲时导入，因为守护进程跑在计时 CPU 的 SMT 兄弟核上） |
| 本机参照基线 | 全部在 grade-06 上安静测量并激活，锚点与包内记录一致：galaxy（真实基础镜像重建后重测）弱 0.00139018 / 强 0.000596871（包内 E0 = 0.00139、E5 = 0.000597，与替代镜像上的结果完全相同）；polarization 弱 1.80825 / 强 1.10992（包内 B = 1.808、H5 = 1.110）；species-tree（50 个正式用例）弱 625 / 强 529 个漏分裂，即 0.127551 / 0.107959，与 `activation.json` 的原机锚点一个不差；trajectory（3000 个子集输入）OBM 0.35031684638830834 / VAR(1) 0.23005048805553863，与出题方在同一子集上的记录相差不到 2e-15。基线都用 `.runtime-recal` 运行（与 `.runtime-suite` 只差锥优化时限清单，不影响这些题） |
| 锚点核对 | galaxy、polarization 本机锚点与包内记录一致（见上一行） |
| CPU 上限与墙钟检查 | galaxy：单次调用最多 1.82 CPU 秒（上限 5 秒；原机记录 3.09 秒）。polarization：每景墙钟最多 0.198 秒（弱参照）、0.701 秒（强参照），上限 1.000 秒，32/32 景通过；species-tree 两条参照 50/50 例无失败；trajectory 单例墙钟最多 3.87 秒（OBM，含 R 启动），上限 30 秒 |
| 隔离检查 | docker 垫片日志（`scripts/shim-audit.py`）：galaxy 122 次、polarization 66 次、species-tree 102 次容器创建全部为 `linux/amd64`、`--cpuset-cpus 6`（grade-06 计时 CPU）、`frontieram-grade.slice`、封存的 amd64 镜像 ID、`--network none`、`--cpus 1`，内存上限与包内规定一致（2/8/4 GiB，均不带交换）。trajectory 的验证容器：CPU 6、`frontieram-grade.slice`、单核配额、2 GiB、无交换、仅回环网卡，验证器的 11 项沙箱自检在两条参照中全部通过 |

## 5. 协议 v6.2 结果（11-10）

映射与 v6.1 相同：原始指标 L ≥ 弱参照得 0；强参照 ≤ L < 弱参照得 50·(弱 − L)/(弱 − 强)；L < 强参照得 100 − 50·L/强。锚点都是本机评测核上重测的参照误差。

**species-tree-v2**（200 个正式用例，250 物种，每例 120 秒；`results/species-tree-v2/`）。参照分片实测（`suite-species-tree-v2-20261109T181626Z-e9383f`）：不加权 TREE-QMC 漏 7423 个分裂（U = 0.150263），加权 TREE-QMC 漏 6569 个（W = 0.132976），分母 200 × 247 = 49,400。

| 模型 | 原始指标 | 映射 | 得分 | 说明 |
| --- | --- | --- | --- | --- |
| gpt-6.1-sol | 49,400 / 49,400 个分裂全部漏掉，E = 1.0 | E ≥ U 记 0 | **0** | 4.4 分钟就结束回合。程序每例 0.13 秒输出一棵 250 叶的二叉树，但与真树没有一个共同分裂（输出是平衡树，最终说明写“选最大校正距离”合并，与邻接法方向相反）。它在公开用例 001 上用 `check_rf.py` 测到 `normalized_rf = 1.0`，没有处理，只核对了格式和可复现性 |
| claude-opus-5-5 | 漏 6,700 个，E = 0.135628 | 50·(U − E)/(U − W) | **42.3** | 类似加权 ASTRAL 的支持度加权四元组方法加精确动态规划，每例中位 4.2 秒、最多 11.1 秒。逐例与加权参照比：82 例更好、18 例相同、100 例更差，平均每例多漏 0.66 个分裂（95% bootstrap 区间 0.07–1.20），显著差于强参照、显著好于弱参照 |

参照耗时：加权参照每例中位 48 秒、最多 105.5 秒（上限 120 秒，余量 12%），不加权参照最多 62.5 秒；200/200 例都在时限内。余量偏紧，后续若有波动可能出现参照超时（待讨论：改为 180 秒）。

**galaxy-gravity-v2**（20 张快照、60 次调用，每次 5 CPU 秒）。参照基线（`store-galaxy-v2`，grade-06）：弱 0.00125189（单次最多 4.05 CPU 秒），强 0.000590528（最多 3.71 CPU 秒）。

| 模型 | 原始指标 | 映射 | 得分 | 说明 |
| --- | --- | --- | --- | --- |
| gpt-6.1-sol | 相对 RMS 误差 4.25e-9，单次最多 4.45 CPU 秒 | 100 − 50·L/强 | **99.9996** | 二叉空间树上的高阶调和多极展开（16 阶）加软化修正（10 阶），按当前输入标定的代价模型选精度，并有计时保护 |
| claude-opus-5-5 | 1.06e-7，单次最多 3.16 CPU 秒 | 100 − 50·L/强 | **99.991** | 笛卡尔坐标快速多极方法（FMM，双树遍历，11 阶展开，AVX2）；提交里没有 README |

结论：加倍粒子数确实使直接求和超时，但两个模型都改用高阶快速多极方法，在时限内达到 1e-7 以下的误差。映射在 L < 强参照时为 100 − 50·L/强，强参照误差约 6e-4，因此误差在 1e-6 以下的提交都挤在 99.9 分以上，题目无法再区分强解法（需反馈出题人：例如以误差的对数或以“给定误差下的耗时”计分）。另：弱参照单次最多 4.05 CPU 秒，余量 19%，低于定稿前约定的 25%；按约定后续尝试应改为 6 CPU 秒。本轮两份提交都是在题面写明的 5 秒下开发和评测的，sol 的程序会按剩余时间调整精度，事后改时限会改变它的输出，因此本轮成绩按 5 秒保留（默认决定）。

**polarization-demosaicking（opus）**：第一次评分（11-10 03:53，grade-06）31/32 景通过，第一个运行的 ball 景 1.003 秒超过 1.000 秒上限，其余 31 景 0.56–0.58 秒。opus 的程序只用 NumPy（FFT 卷积），耗时只取决于图像尺寸、与内容无关；在宿主进程内实测，每个新进程的第一次重建 0.42–0.50 秒，没有跨进程的缓存或编译。与 11-06 sol 第一次评分的情况相同（第一景超时，其余约 0.48 秒），判为评测环境的首景冷启动，第一次记为未测，在 conic 评分结束后重评一次（`scripts/v62-after.sh`），以重评为准（默认决定，与 11-06 的处理一致）。两次都发生在第一景，说明评分器需要一个不计时的预热运行（待讨论）。

**trajectory-covariance（opus）**：3000/3000 合格，L = 0.221264，单例最多 1.66 秒；L < 强参照 0.230050，得 100 − 50·L/强 = **51.9**（sol v6.1 为 0.176933，61.5 分）。

`mcmc-long-run-covariance-maintainer-v1`（19.6 GB）是 trajectory-covariance 的出题方包，正在从北大网盘下载到 `/srv/frontieram-raw/`（`Raw_Resource/` 中为软链接），下载完成后按同样流程接入。
