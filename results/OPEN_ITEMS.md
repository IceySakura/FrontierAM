# 待讨论事项与默认决定（2026-11-09，11-10 更新）

以下事项都已按默认决定推进；如需改变，告诉我即可。

| 事项 | 默认决定 | 依据 / 文档 |
| --- | --- | --- |
| polarization 正式成绩 | 取 11-06 15:09 的重评（50.7）；第一次评分因调度核上的编译负载导致第一景超时，按“平台故障记未测”处理 | `NEW_TASKS.md` |
| polarization 首景冷启动（opus） | opus 第一次评分只有第一景超时（1.003 秒，其余 0.56–0.58 秒），与 sol 11-06 的情况相同；记为未测，重评（11-10 07:14）32/32 通过、55.9 分，以重评为准。重评中第一景仍最慢（0.710 秒，其余最多 0.585 秒），首景偏慢是系统性的；建议评分器在计时前加一次不计时的预热运行（改动评分流程，待确认） | `NEW_TASKS.md` 第 5 节 |
| polarization 公开开发集 | 用 Wen 等 2019（CPDNet，CC BY-NC）105 景，裁剪为 1024×1024。公开反馈评测器已实现并验证（`arena/operator_drivers/polarization_feedback.py`，复用包内构建、沙箱运行和打分；数据封存于 `.runtime-ops/polarization-demosaicking-feedback/`），弱参照在开发核上 105/105 通过、整批 2.1937；东京工业大学数据集（1024×768、仅限研究用途）不采用，原因见请求文档；待出题人确认 | `tasks/polarization-demosaicking/analysis/POLARIZATION_PUBLIC_DEV_REQUEST.md` |
| polarization 测试集的选景方式 | 测试集删去了“双线性比 LEIC 更准”的场景，会放大两条参照的差距；保持不变，请出题人确认是否有意 | 同上 |
| galaxy-gravity 规模 | 粒子数 65,536 → 131,072，保持 5 CPU 秒。评测核复测：弱参照单次最多 4.05 CPU 秒（余量 19%，低于约定的 25%），按约定后续尝试改为 6 CPU 秒；本轮两份提交在 5 秒下开发和评测，成绩按 5 秒保留 | `tasks/galaxy-gravity/analysis/GALAXY_SCALE.md`、`NEW_TASKS.md` 第 5 节 |
| galaxy-gravity-v2 的区分度 | sol、opus 都用高阶快速多极方法达到 1e-7 以下误差（99.99 分以上），现映射在强参照以下压缩为 100 − 50·L/强，无法区分强解法；需反馈出题人（例如改用误差对数或给定误差下的耗时计分），本轮不改 | `NEW_TASKS.md` 第 5 节 |
| species-tree 新版本 | 自建 S100 式模拟（已校准），250 个物种、200 个正式用例、每例 120 秒，新的公开开发集；已评分 | `tasks/species-tree/analysis/SPECIES_TREE_SIM.md`、`tasks/species-tree/analysis/SPECIES_TREE_SCALE.md` |
| species-tree-v2 公开反馈 | 公开反馈评测器为 `FRONTIERAM_SPECIES_V2_DATA=public`（50 个公开用例）；基线已在评测核上测得 U = 0.158462、W = 0.141700（`tasks/species-tree/references/r1/anchors-public.json`）。v6.2 只对 conic 和 exponential-v2 开迭代设置，species 暂不接入 `public-eval` 命令，需要时再接 | `NEW_TASKS.md` 第 5 节 |
| species-tree-v2 时限 | 加权参照每例最多 105.5 秒（上限 120 秒，余量 12%），200/200 例通过；本轮保持 120 秒，下一轮建议 180 秒 | `NEW_TASKS.md` 第 5 节 |
| trajectory-covariance 正式规模 | 继续用固定的 10% 分层子集（3,000 个输入）作为正式成绩；全量 3 万例只在需要时分片评测 | `NEW_TASKS.md` |
| trajectory-covariance 公开反馈集 | 公开开发集只有 2 条练习轨迹，不具代表性；NUTS 采样器只有 macOS ARM64 二进制，无法在本机重生成。从每档第 991–1000 条母链（不参与正式评分）中取 300 个输入作为反馈池，agent 只看到误差、看不到数据；已实现（`FRONTIERAM_TRAJ_POOL=feedback`），核对与正式子集零重叠 | 本文件 |
| exponential-v2 的 Chebyshev 成员 | 逐例最快的成员 60/74 是独立 agent 按文献实现的 Chebyshev，不是作者代码；保持不变，下一次冻结参照组时考虑替换为作者代码 | `tasks/exponential/analysis/EXPONENTIAL_V2.md` |
| exponential-v2 的有理 Krylov 成员 | 修正后仍有一个公开用例精度略超余量，本轮不入选；下一次冻结时再评估 | `tasks/exponential/analysis/EXPONENTIAL_V2.md` |
| conic 修正版的重复尝试 | sol 在修正版上的退步来自一次尝试；默认不额外重复，v6.2 的单轮和迭代两次尝试可作为补充观察 | `tasks/conic/analysis/CONIC_REVISION_SOL.md` |
| v6.2 exponential-v2 单轮尝试 | 第一次尝试的提示有一句计时说明有误，只作记录（56.3 分）；以修正提示后的重跑为正式成绩 | `PROTOCOL_V62.md` |
| v6.2 测试评测的重复次数 | 1 次（正式评测 3 次），以控制评测时间 | `PROTOCOL_V62.md` |
| 看板的访问与内容范围 | 用户决定（11-10）：只在办公网访问（http://9.135.91.91:18765/），不做公网发布；只展示最新版本和 v6.2 成绩 | `docs/platform/DASHBOARD.md` |
| 第 5 轮 sol 尝试遇到上游限流 | 11-10 11:11 同时启动 4 个 gpt-6.1-sol 会话，其中 3 个（exponential-v2 第 3 次、polarization、trajectory）在约 25–33 分钟时因上游 429 Too Many Requests 超过重试上限、harness 退出：属基础设施故障，记为未测、不评分；按每次最多 2 个 sol 会话重跑（`scripts/rounds/round5-sol-retry.sh`）。opus 会话也出现过少量 429，均重试成功 | 本文件 |
| sol 的极短会话 | sol 的 exponential-v2 第 2 次单轮尝试 4.5 分钟结束回合（与 11-09 的 species-tree-v2 4.4 分钟、exponential-v2 重跑 23 分钟同一模式）：模型名、推理强度（max）、提示与 2 小时的会话完全相同，无限流或错误，每次请求的推理 token 明显更少；判为模型自身行为，尝试有效 | 本文件 |
| 超过参照的成绩 | 一律标“待复核”，按 `BASELINE_POLICY.md` 第 6 条复核，记录在 `BASELINE_REVIEW.md` | `BASELINE_POLICY.md` |
| polarization 评测隔离 | 11-10 预热验证：清掉镜像缓存后第一景 0.601 秒，不慢于其余各景（中位 0.618 秒），预热有效；但当时 8 个开发会话在运行，有三景跳到 0.91–1.0002 秒（一景超时）。每景上限 1 秒、正常约 0.6 秒，余量太小，评测隔离由 grade_pool 改为 quiet（评测期间开发核也保持空闲）。该次验证评分属受干扰的测量，不计成绩；待开发核空闲时在 quiet 条件下再验证一次 | `polarization-first-scene.json` |
| galaxy 给出题人的核对包 | 出题人复测两份提交为 0 分。两份提交都用 x86-64 AVX2 内联函数与编译选项，在 Apple M5（ARM64）上无法编译，判断为差异根源；核对包 `var/outputs/galaxy-v2-for-author.tar.gz`（两份提交源码、算法说明、私有集逐次调用结果、公开快照逐张结果，不含私有数据） | 本文件 |
| sol 会话普遍变短（待决定） | 11-10 第 5 轮 sol 的单轮尝试：exponential-v2 第 2 次 4.5 分钟、polarization 重跑 4.4 分钟、trajectory 重跑 3.5 分钟，均为正常结束回合、无限流。开发镜像（含 Codex，11-04/06 构建）、模型名、推理强度 max、提示与 11-09 的 2 小时会话完全相同；每次请求的推理 token 约为长会话的 1/5，且与当天上游 429 限流同时段，疑为服务端在高负载下降低了实际推理量。按协议这些是有效尝试，照常评分；是否在上游负载低时另行重跑由用户决定 | 本文件 |
| GitHub 镜像 | 用户决定（11-10 15:31）：私有评测集只放 git.woa.com（完整仓库）；GitHub（github.com/IceySakura/FrontierAM，私有）是过滤后的镜像，由 `scripts/ops/sync_github.sh` 生成：从全部历史中剔除 `*/tests/private/*`、`*/data/private/*`、`*/cases/private/*`、`*/docker_eval/private_pilot/*` 及 exponential v2 的私有种子与清单，再强制推送；`github` 远程的直接推送地址已停用。中午第一次推送曾含完整历史，强制推送后旧提交在 GitHub 上已不可达，但要彻底清除需删除并重建该仓库（或请 GitHub 支持回收），待用户决定 | 本文件 |
| 本机目录整理 | 11-10：运行数据收进 var/（var/stores、var/raw、var/outputs、var/dist），代码与文档引用全部更新，库内记录中的绝对路径同步改写；隐藏的 .runtime*/ 暂留原处（子代理在用），之后可再收进 var/runtime/ | README.md |
| 数据三层划分（用户决定 11-10） | 每题分 samples（给模型，理解题意与格式，从 public 中选取）、public（不给模型，只在服务端评测，给迭代设置提供反馈与当前分数）、private（最终评测）。待评测协议定稿后统一整理各题数据与迭代反馈 | 本文件 |
| 协议 v6.3 草案（用户决定 11-10） | 单轮 1 小时（11-10 由 2 小时改为 1 小时：单轮不跑评测）、迭代 4 小时；迭代评测异步（提交时快照，容器不暂停，时钟照走）；题面不提示用满时间，早停算能力的一部分；迭代尝试与单轮一样并行，每个尝试的 public 评测独占一个评测槽，取消全机迭代锁。反馈格式（跨题统一，按类别对比参照）待定。round 5 剩余的 4 个迭代复测暂停，等 v6.3 定稿后按新协议跑 | 本文件 |
| 迭代反馈格式（用户决定 11-10） | 跨题统一：public 总分（与最终得分同一换算，参照 = 50）+ 按类别的表：用例数、通过/答错/超时/崩溃、精度与速度只给“你 ÷ 参照”的比值、本类得分。只有当用例明显分成本质不同的几类、分开反馈确实更有指导意义时才分类（每类至少 5 个用例）；否则只报一行总计 | 本文件 |
| 机器布局改为 7 开发 + 7 评测（用户决定 11-10） | `scripts/ops/relayout.py` 已启动：先等所有槽位空闲再切换（`var/stores/store/reruns/relayout.log`）。切换后：新评测槽的镜像复制、各运行副本的评测守护进程列表更新、`./bench interference` 实测评测核两两共用 L3 域时的干扰并写入 `docs/platform/CONCURRENCY.md` | `scripts/ops/relayout.py` |
| trajectory 人类参照筛选（11-11 完成） | 为让布局切换尽快完成，11-10 停掉了筛选调度（已跑 21/30 个任务；进行中的 2 个在容器里跑完）。剩余 9 个任务在 `var/stores/store/reruns/traj-ref-screen/jobs-rest.txt`，切换后用 `scripts/ops/run_leased.py` 续跑 | 本文件 |
| conic 构建时限 1,800 秒 | 题面允许构建最多 30 分钟，会占用 private 评测的 1 小时封顶。默认不改（构建通常不到 1 分钟）；若实际出现长构建，需与出题人商定缩短 | `results/EVAL_TIME_LIMITS.md` |
| v6.3 参照基线 | 各题 v6.3 版本的 public 参照（`bench public-baseline-v63`）与 private 参照需在评测槽上测量；按用户“先不跑评测”的决定，开跑 v6.3 前再做 | `results/PROTOCOL_V63.md` |
| 计划调整（用户决定 11-10 18:00） | 改造完成后直接进入真实评测，在评测中发现并修正题目、数据、时限和设施的问题：按需重跑参照；跑 gpt-6.1-sol 在 v6.3 全部 6 道题上的单轮（1 小时）与迭代（4 小时）；核对评测效率（public ≤ 15 分钟、private ≤ 1 小时）；超过参照的格子走复核流程；最后给出结论报告 | 本文件 |
| v6.3 演练发现：conic 测试评到了私有集（11-10，已修复） | 迭代演练中，conic 的 public 测试继承了尝试的环境（私有运行副本），而 conic 的运行副本在导入时就确定，结果评测了 650 个私有用例；报告因后续检查失败只显示“未测”，没有任何私有结果进入容器。修复：评测服务启动 public 评测时直接传入 public 数据开关；评测前检查实际生效的运行副本或封存目录必须是 public 的；出报告前再检查评测器记录的数据范围（species/galaxy 的公开与私有用例编号重叠）。该次评分留在 store-public-conic，不用于任何结果 | `arena/native.py`、`arena/public_v63.py` |
| v6.3 首轮：测试被饿死约 1 小时（11-10 12:18–13:28 UTC，已修复） | galaxy-gravity-v3 的私有参照评分为 quiet 隔离，排队等待时的排空请求挡住了包括测试在内的所有新评分；迭代尝试期间开发槽不会空闲，conic 与 exponential 的迭代尝试各有一次测试超时未返回（另一次排队约 1 小时）。修复：测试不受排空请求阻挡；quiet 的参照评分改到全部尝试结束后进行。这两次迭代尝试的结果需标注，评估后决定是否重跑 | `arena/slots.py`、`scripts/rounds/v63-grade.sh` |
| v6.3 首轮：每个迭代尝试留一个空闲评测槽 | 首轮中 6 个参照分片占满评测槽，测试排队约 10 分钟；改为有 N 个迭代尝试运行时普通评分不占最后 N 个空闲评测槽 | `arena/slots.py` |
| v6.3 首轮：trajectory 迭代尝试遇上游 429（11-10） | 第 79 分钟上游返回 429 超过重试上限，harness 退出（此前 20 次测试，public 分数 51→64.5）。属基础设施故障，记为未测；日志改名为 `.429.log` 留档，第一轮结束后自动重跑 | `var/stores/store/reruns/` |
| v6.3 首轮：测试再次被饿死（11-11 09:04–10:04，已修复） | 一个重复的 species 整集评分（grade_pool 隔离，锁住全部评测槽）在只有 1 个迭代尝试运行时被放行，占用 89 分钟；trajectory 迭代尝试的第 19 次测试排队 1 小时未拿到评测槽（该尝试随后在第 127 分钟正式提交）。修复：有迭代尝试运行时，锁住评测槽的隔离（grade_pool、quiet）一律等待 | `arena/slots.py` |

| 参照原则修订（用户决定 11-11） | 参照是出题人自己做或深度参与的一版，固定锚点，不声称是 SOTA；不因被模型超过而升级，不再判断“调参还是创新”、不标“待复核”；任一格子 ≥ 80 分或高分段分不开时请出题人来改参照。galaxy-gravity-v3 参照退回出题方 Treecode2（r1），opus 解法（r2）与 agent 按论文写的 FMM（r3）只作参考材料；exponential-v3 保留原 7 方法参照组，Chebyshev 第二版不并入 | `results/BASELINE_POLICY.md`、`results/BASELINE_REVIEW.md` |
| 题面公开映射、反馈给比值（用户决定 11-11） | 保持：参照在公开集上的水平刻意可见，私有锚点、私有数据与参照代码不公开；观察“过线即停” | `results/PROTOCOL_V63.md` |
| galaxy-gravity-v3 题面需重新封签（下一轮前） | 本轮尝试看到的题面与反馈写“50 分锚点为高阶快速多极方法参照”，public 反馈按 r2 的公开锚点计分；参照已改为 r1。`scripts/seal/prepare_v63.py` 与 `arena/native.py` 的说明已改，封签与 public 锚点在下一轮前重做。本轮最终评分不受影响 | `scripts/seal/prepare_v63.py` |
| galaxy-gravity 弱参照余量不足（待反馈出题人） | 私有集上弱参照单次最多 4.05 CPU 秒（上限 5 秒），余量 19%，低于 25% | `results/BASELINE_REVIEW.md` |
| 新题 ihvp-sequence 纳入本轮（用户决定 11-11） | GitHub PR #1（出题人 Zhang Hengrui），草稿：题面、样例、生成器、评测器、两份出题人参照（CG 0 分，L-BFGS-PCG 50 分）齐全，无私有内容。缺：反馈集与私有集（在出题人处，用户暂缓决定是要原包还是用自带生成器重生成）、amd64 镜像、容器隔离评测、v6.3 接线、参照时间余量（困难例接近 30 秒）与分数饱和检查（参照之上线性映射）。本轮目标：与其他六题一样完成锚点、sol 单轮与迭代、评分、发布 | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| ihvp-sequence 开放（11-11） | 出题人题包导入（哈希一致），私有锚点 CG 0.1213 / L-BFGS-PCG 0.0543，public 锚点 0.1215 / 0.0581；状态改为 formal，演练通过，sol 单轮与迭代已开始。待出题人：时间余量门槛、50 分以上的对数映射（分数由 3 条困难序列决定）；反馈比值在简单类别上失去意义（观察项） | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| ihvp-sequence：sol 两格 100 分，题目设计缺陷（11-11，用户决定按规则发布并交出题人） | 模型用少量 HVP 查询识别生成结构（带符号 DCT + rank-12）后精确求解；触发“≥ 80 分请出题人修改”。建议方向见 INTEGRATION.md | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| ihvp-sequence-v2 实验（11-11，出题人提议“不给生成代码就猜不出来”） | 新版本 v2：工作区只有题面、smoke、协议代码与 `run_sample.py`，完整样例的 HVP 由宿主服务提供（`arena/operator_drivers/ihvp_oracle.py`），不含 `model.py`、构造数组与参考解；数据、参照、评测与 v1 相同（锚点复用）。gpt-6.1-sol 单轮与迭代 14:32 开始 | `tasks/ihvp-sequence/versions/v2/` |
| v6.3 迭代格子受测试排队故障影响，重跑（用户决定 11-11） | 逐个检查 v6.3 迭代尝试的测试：conic（第 4、5 次评测超时无报告，第 1、2 次周转 20–23 分钟）、exponential（第 5 次无报告）、trajectory-covariance（第 19 次排队 60 分钟后报错）受两次排队故障影响；其余格子正常。三格各重跑一次迭代（最多 2 个并发，11-11 14:50 起），评分后替换；原尝试日志改名 `.starved.log`，原成绩移到 `var/stores/store-v63/superseded/` 留档 | `results/OPEN_ITEMS.md` 上文两次故障条目 |
| ihvp-sequence-v2 阶段观察（11-11 16:55） | 隐藏生成代码后两次尝试都不识别 DCT 结构，改用历史解子空间投影 + CG，迭代 public 仍到 99.6–99.8：参照偏弱、数据结构过简单、映射饱和，需出题人处理。反馈的类别分数在类别锚点倒挂时失去意义（观察项，可改为只显示比值） | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| ihvp-sequence v2 结果（11-11 19:15） | 单轮私有集 99.7（E = 3.0e-4），迭代 public 99.8、私有评分待整机安静（当前 trajectory 重跑结束后）。两次都没有识别矩阵结构，靠历史解子空间投影 + CG；当前版本改为 v2，v1 成绩折叠。待出题人：生成器（解序列近似低秩）、参照（不用历史）、映射（饱和） | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| 重跑完成并替换（11-12） | conic 迭代 426/650（0 分，原 433/650），exponential 迭代 63.8（原 59.6），trajectory 迭代 62.3（原 62.3）；三次重跑共 45 次测试全部拿到报告，最长周转 9.2 分钟，无设施错误。ihvp-sequence v2 迭代私有集 99.9（E = 1.5e-4），代码不识别结构。看板、README、LEADERBOARD、V63_REPORT、结果解读已更新，“设施影响”条目已移除 | `results/V63_REPORT.md` |
| ihvp-sequence v3（出题人 PR #2，11-12） | 代码已并入：v3 评测器与困难数据导入、Galerkin-CG 审计候选；反馈只用新增 15 条困难序列，私有集 30 条（用户决定）；评分容器加固（worker 只能读协议文件）；v3 镜像已构建。待出题人总包（Release 附件本机无法下载）后导入、标定、演练并跑 sol 单轮与迭代（用户决定） | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| ihvp-sequence v3 开跑（11-12 12:09） | 出题人总包经 GitHub Release 下载（只读 token 存于 `/root/.frontieram/github.token`，不进仓库），SHA-256 校验通过；私有锚点 CG 0.1502 / L-BFGS-PCG 0.0805（原有 15 条 0.122 / 0.053，新增 15 条 0.179 / 0.108），反馈锚点（新增 15 条）0.1775 / 0.1077；单轮与迭代演练通过；gpt-6.1-sol 单轮与迭代已开始 | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| Claude Opus 5.5 补跑 v6.3（用户决定 11-12） | 分数在 50 附近的 4 题（exponential、polarization-demosaicking、species-tree、trajectory-covariance）各跑单轮与迭代，最多 2 个并发；脚本加了模型参数（`V63_MODEL`），opus 日志与成绩目录用 `opus` / `claude-opus-5-5` 前缀 | `scripts/rounds/v63-sol.sh` |
| ihvp v3 与出题人要求对照（11-12） | 服务与隔离、30+30 合并、资源限制、同机重测、六文件小包都满足；差异：反馈只用新增 15 条（用户决定保持，回传时说明：30 条会超过 15 分钟反馈上限）、题面两处改动、sol 尝试先于回放开始。回放（v2/v1 冻结程序）与 public 30 条参照重测在整机安静时进行，汇总见 `tasks/ihvp-sequence/analysis/V3_RESULTS.md` | `tasks/ihvp-sequence/analysis/INTEGRATION.md` |
| 模型调用缺 `cache_task_id`（11-12 发现并修复） | FrontierAM 用 `atr serve` 起网关，它只在设置 `ATR_CACHE_TASK_ID` 时才在 AIHub bearer 中带 `cache_task_id`；此前从未设置，v6.3 以来所有 sol / opus 尝试都没有上游账户亲和（不影响正确性与评分，影响缓存命中与路由）。`arena/native.py` 现在为 `atr serve` 设 `ATR_CACHE_TASK_ID=auto`（按网关 run ID 派生，尝试内固定、尝试间不同）。11-12 14:55 之后启动的尝试生效；当时在跑的 3 个尝试（opus exponential / polarization 迭代、sol ihvp v3 迭代）不中途重启 | `arena/native.py` |
| 评分隔离：`quiet` 改为 `dev_idle`（11-12） | 7+7 布局实测：7 路评测并行负载系数 1.039–1.046（与单路相同）；只有开发负载会拖慢计时（同 L3 域最坏 2.0–2.8 倍，跨域约 1.2 倍）。polarization-demosaicking 与 ihvp-sequence 改为 `dev_idle`（开发槽全空、评测可并行）。待做：有计时评分排队时自动开评分窗口（暂停新开发），不再手动 | `docs/platform/CONCURRENCY.md` 第 7 节 |
| 真实开发负载下的干扰（11-12 实测） | 3 个真实尝试开发时，评测负载系数 1.053（安静 1.037），只多约 1.5%；压力程序（2.0–2.8 倍）是人为最坏情况。ihvp-sequence 与 polarization-demosaicking 的私有评分改为 `shared` | `docs/platform/CONCURRENCY.md` 第 7 节 |
| opus trajectory 迭代因上游余额不足中断（11-12 15:11） | 上游返回 400 “Credit balance is too low”，harness 退出，记为未测，日志改名 `.credit.log`，待重跑。它是第一个带 `cache_task_id` 启动的 opus 尝试（前 14 次请求正常后失败）；同时在跑、没带 `cache_task_id` 的 opus species 正常。疑似账户亲和把请求固定到余额耗尽的账户上，待确认 | `arena/native.py` |
| 默认全并发 + 异常报警（用户决定 11-12） | 所有评分 `shared`，基线不强制 `quiet`，不为测试预留空闲槽；每次评分前后约 0.7 秒的负载小测（慢于该槽历史最佳 25% 报警），exponential 闸门超标只报警；报警显示在看板“实时占用”页，逐条单独处理。实测 3 路评测 + 3 个真实开发：负载系数 1.041–1.064 | `docs/platform/CONCURRENCY.md` 第 7 节 |
| opus 余额不足：按 AIHub 建议换 id 重试（11-12） | ATR 网关（本地提交 `d3432ae`，未推送团队仓库）：上游 400 “Credit balance is too low” 时缓冲该响应，轮换 `cache_task_id`（由原 id 派生，仍按尝试隔离），并改答 429 让客户端重试；审计记录保留上游原始状态与内容。之后启动的尝试生效。opus trajectory 迭代（`.credit.log`）待当前 opus 队列跑完后重跑 | ATR `src/atr/ingress/{gateway,mitm}.py` |
| ihvp-sequence v3：sol 迭代 82.3 超过 80（11-12 15:43） | 私有集 30 条：sol 单轮 E = 0.0381 → 76.3 分，迭代 E = 0.0285 → 82.3 分（映射 50 + 50·(E₅₀ − E)/E₅₀，E₅₀ = 0.0805）。按参照政策（任一格 ≥ 80 通知出题人，由出题人决定是否出新版），连同 `tasks/ihvp-sequence/analysis/V3_RESULTS.md` 一并交给出题人 | 待出题人 |
| opus 一轮并发偏低（11-12 15:43 发现） | opus 调度脚本按 `V63_CONCURRENCY=2` 启动，且 ihvp v3 回放开着“新开发暂缓”窗口直到 sol v3 评完；15:43 改为一次放出全部剩余 4 个尝试（共 6 个并行） | 已处理 |
