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
| trajectory 人类参照筛选暂停 | 为让布局切换尽快完成，11-10 停掉了筛选调度（已跑 21/30 个任务；进行中的 2 个在容器里跑完）。剩余 9 个任务在 `var/stores/store/reruns/traj-ref-screen/jobs-rest.txt`，切换后用 `scripts/ops/run_leased.py` 续跑 | 本文件 |
| conic 构建时限 1,800 秒 | 题面允许构建最多 30 分钟，会占用 private 评测的 1 小时封顶。默认不改（构建通常不到 1 分钟）；若实际出现长构建，需与出题人商定缩短 | `results/EVAL_TIME_LIMITS.md` |
| v6.3 参照基线 | 各题 v6.3 版本的 public 参照（`bench public-baseline-v63`）与 private 参照需在评测槽上测量；按用户“先不跑评测”的决定，开跑 v6.3 前再做 | `results/PROTOCOL_V63.md` |
| 计划调整（用户决定 11-10 18:00） | 改造完成后直接进入真实评测，在评测中发现并修正题目、数据、时限和设施的问题：按需重跑参照；跑 gpt-6.1-sol 在 v6.3 全部 6 道题上的单轮（1 小时）与迭代（4 小时）；核对评测效率（public ≤ 15 分钟、private ≤ 1 小时）；超过参照的格子走复核流程；最后给出结论报告 | 本文件 |
| v6.3 演练发现：conic 测试评到了私有集（11-10，已修复） | 迭代演练中，conic 的 public 测试继承了尝试的环境（私有运行副本），而 conic 的运行副本在导入时就确定，结果评测了 650 个私有用例；报告因后续检查失败只显示“未测”，没有任何私有结果进入容器。修复：评测服务启动 public 评测时直接传入 public 数据开关；评测前检查实际生效的运行副本或封存目录必须是 public 的；出报告前再检查评测器记录的数据范围（species/galaxy 的公开与私有用例编号重叠）。该次评分留在 store-public-conic，不用于任何结果 | `arena/native.py`、`arena/public_v63.py` |
| v6.3 首轮：测试被饿死约 1 小时（11-10 12:18–13:28 UTC，已修复） | galaxy-gravity-v3 的私有参照评分为 quiet 隔离，排队等待时的排空请求挡住了包括测试在内的所有新评分；迭代尝试期间开发槽不会空闲，conic 与 exponential 的迭代尝试各有一次测试超时未返回（另一次排队约 1 小时）。修复：测试不受排空请求阻挡；quiet 的参照评分改到全部尝试结束后进行。这两次迭代尝试的结果需标注，评估后决定是否重跑 | `arena/slots.py`、`scripts/rounds/v63-grade.sh` |
| v6.3 首轮：每个迭代尝试留一个空闲评测槽 | 首轮中 6 个参照分片占满评测槽，测试排队约 10 分钟；改为有 N 个迭代尝试运行时普通评分不占最后 N 个空闲评测槽 | `arena/slots.py` |

