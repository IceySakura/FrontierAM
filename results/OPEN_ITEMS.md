# 待讨论事项与默认决定（2026-11-09，11-10 更新）

以下事项都已按默认决定推进；如需改变，告诉我即可。

| 事项 | 默认决定 | 依据 / 文档 |
| --- | --- | --- |
| polarization 正式成绩 | 取 11-06 15:09 的重评（50.7）；第一次评分因调度核上的编译负载导致第一景超时，按“平台故障记未测”处理 | `NEW_TASKS.md` |
| polarization 首景冷启动（opus） | opus 第一次评分只有第一景超时（1.003 秒，其余 0.56–0.58 秒），与 sol 11-06 的情况相同；记为未测，重评（11-10 07:14）32/32 通过、55.9 分，以重评为准。重评中第一景仍最慢（0.710 秒，其余最多 0.585 秒），首景偏慢是系统性的；建议评分器在计时前加一次不计时的预热运行（改动评分流程，待确认） | `NEW_TASKS.md` 第 5 节 |
| polarization 公开开发集 | 用 Wen 等 2019（CPDNet，CC BY-NC）105 景，裁剪为 1024×1024。公开反馈评测器已实现并验证（`arena/operator_drivers/polarization_feedback.py`，复用包内构建、沙箱运行和打分；数据封存于 `.runtime-ops/polarization-demosaicking-feedback/`），弱参照在开发核上 105/105 通过、整批 2.1937；东京工业大学数据集（1024×768、仅限研究用途）不采用，原因见请求文档；待出题人确认 | `POLARIZATION_PUBLIC_DEV_REQUEST.md` |
| polarization 测试集的选景方式 | 测试集删去了“双线性比 LEIC 更准”的场景，会放大两条参照的差距；保持不变，请出题人确认是否有意 | 同上 |
| galaxy-gravity 规模 | 粒子数 65,536 → 131,072，保持 5 CPU 秒。评测核复测：弱参照单次最多 4.05 CPU 秒（余量 19%，低于约定的 25%），按约定后续尝试改为 6 CPU 秒；本轮两份提交在 5 秒下开发和评测，成绩按 5 秒保留 | `GALAXY_SCALE.md`、`NEW_TASKS.md` 第 5 节 |
| galaxy-gravity-v2 的区分度 | sol、opus 都用高阶快速多极方法达到 1e-7 以下误差（99.99 分以上），现映射在强参照以下压缩为 100 − 50·L/强，无法区分强解法；需反馈出题人（例如改用误差对数或给定误差下的耗时计分），本轮不改 | `NEW_TASKS.md` 第 5 节 |
| species-tree 新版本 | 自建 S100 式模拟（已校准），250 个物种、200 个正式用例、每例 120 秒，新的公开开发集；已评分 | `SPECIES_TREE_SIM.md`、`SPECIES_TREE_SCALE.md` |
| species-tree-v2 公开反馈 | 公开反馈评测器为 `FRONTIERAM_SPECIES_V2_DATA=public`（50 个公开用例）；基线已在评测核上测得 U = 0.158462、W = 0.141700（`results/species-tree-v2/public-anchors.json`）。v6.2 只对 conic 和 exponential-v2 开迭代设置，species 暂不接入 `public-eval` 命令，需要时再接 | `NEW_TASKS.md` 第 5 节 |
| species-tree-v2 时限 | 加权参照每例最多 105.5 秒（上限 120 秒，余量 12%），200/200 例通过；本轮保持 120 秒，下一轮建议 180 秒 | `NEW_TASKS.md` 第 5 节 |
| trajectory-covariance 正式规模 | 继续用固定的 10% 分层子集（3,000 个输入）作为正式成绩；全量 3 万例只在需要时分片评测 | `NEW_TASKS.md` |
| trajectory-covariance 公开反馈集 | 公开开发集只有 2 条练习轨迹，不具代表性；NUTS 采样器只有 macOS ARM64 二进制，无法在本机重生成。从每档第 991–1000 条母链（不参与正式评分）中取 300 个输入作为反馈池，agent 只看到误差、看不到数据；已实现（`FRONTIERAM_TRAJ_POOL=feedback`），核对与正式子集零重叠 | 本文件 |
| exponential-v2 的 Chebyshev 成员 | 逐例最快的成员 60/74 是独立 agent 按文献实现的 Chebyshev，不是作者代码；保持不变，下一次冻结参照组时考虑替换为作者代码 | `EXPONENTIAL_V2.md` |
| exponential-v2 的有理 Krylov 成员 | 修正后仍有一个公开用例精度略超余量，本轮不入选；下一次冻结时再评估 | `EXPONENTIAL_V2.md` |
| conic 修正版的重复尝试 | sol 在修正版上的退步来自一次尝试；默认不额外重复，v6.2 的单轮和迭代两次尝试可作为补充观察 | `CONIC_REVISION_SOL.md` |
| v6.2 exponential-v2 单轮尝试 | 第一次尝试的提示有一句计时说明有误，只作记录（56.3 分）；以修正提示后的重跑为正式成绩 | `PROTOCOL_V62.md` |
| v6.2 测试评测的重复次数 | 1 次（正式评测 3 次），以控制评测时间 | `PROTOCOL_V62.md` |
| 看板的访问与内容范围 | 用户决定（11-10）：只在办公网访问（http://9.135.91.91:18765/），不做公网发布；只展示最新版本和 v6.2 成绩 | `DASHBOARD.md` |
