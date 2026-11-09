# 待讨论事项与默认决定（2026-11-09）

以下事项都已按默认决定推进；如需改变，告诉我即可。

| 事项 | 默认决定 | 依据 / 文档 |
| --- | --- | --- |
| polarization 正式成绩 | 取 11-06 15:09 的重评（50.7）；第一次评分因调度核上的编译负载导致第一景超时，按“平台故障记未测”处理 | `NEW_TASKS.md` |
| polarization 公开开发集 | 用 Wen 等 2019（CPDNet，CC BY-NC）105 景，裁剪为 1024×1024，参照在本机重测；东京工业大学数据集下载后实测再定；待出题人确认 | `POLARIZATION_PUBLIC_DEV_REQUEST.md` |
| polarization 测试集的选景方式 | 测试集删去了“双线性比 LEIC 更准”的场景，会放大两条参照的差距；保持不变，请出题人确认是否有意 | 同上 |
| galaxy-gravity 规模 | 粒子数 65,536 → 131,072，保持 5 CPU 秒；在评测核上复测后定稿 | `GALAXY_SCALE.md` |
| species-tree 新版本 | 自建 S100 式模拟（已校准），约 250 个物种、≥ 200 个正式用例、120 秒（按试点实测的参照耗时最终确定），新的公开开发集 | `SPECIES_TREE_SIM.md`、`SPECIES_TREE_SCALE.md` |
| trajectory-covariance 正式规模 | 继续用固定的 10% 分层子集（3,000 个输入）作为正式成绩；全量 3 万例只在需要时分片评测 | `NEW_TASKS.md` |
| trajectory-covariance 公开反馈集 | 公开开发集只有 2 条练习轨迹，不具代表性；NUTS 采样器只有 macOS ARM64 二进制，无法在本机重生成。默认从每档第 991–1000 条母链（不参与正式评分）中取 300 个输入作为反馈池，agent 只看到误差、看不到数据 | 本文件 |
| exponential-v2 的 Chebyshev 成员 | 逐例最快的成员 60/74 是独立 agent 按文献实现的 Chebyshev，不是作者代码；保持不变，下一次冻结参照组时考虑替换为作者代码 | `EXPONENTIAL_V2.md` |
| exponential-v2 的有理 Krylov 成员 | 修正后仍有一个公开用例精度略超余量，本轮不入选；下一次冻结时再评估 | `EXPONENTIAL_V2.md` |
| conic 修正版的重复尝试 | sol 在修正版上的退步来自一次尝试；默认不额外重复，v6.2 的单轮和迭代两次尝试可作为补充观察 | `CONIC_REVISION_SOL.md` |
| v6.2 exponential-v2 单轮尝试 | 第一次尝试的提示有一句计时说明有误，只作记录（56.3 分）；以修正提示后的重跑为正式成绩 | `PROTOCOL_V62.md` |
| v6.2 测试评测的重复次数 | 1 次（正式评测 3 次），以控制评测时间 | `PROTOCOL_V62.md` |
| 看板的公网访问 | 目前只在办公网可访问（http://9.135.91.91:18765/）；公网需走正规流程申请托管，上线前再审内容 | `DASHBOARD.md` |
