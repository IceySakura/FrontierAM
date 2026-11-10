# 协议 v6.3 改造验证（2026-11-10，不跑模型）

## 单元测试

`arena/tests/`：20 个测试全部通过。覆盖反馈报告（四种状态、按类别的比值、不出现用例编号、只报总计的题）、各题原始指标与换算（含 galaxy 对数刻度）、测试优先排队、迭代尝试运行时保留一个空闲评测槽、`dev_idle` 隔离、租约与存储。

## 演练（`scripts/rounds/v63-smoke.py`：shell 脚本代替模型，在真实开发容器中运行）

| 检查 | 单轮（6 题） | 迭代（6 题） |
| --- | --- | --- |
| 工作区只有题面与 samples（无 public-dev、无 dev_full） | 6/6 | 6/6 |
| 题面写明时限 | 6/6 | 6/6 |
| 单轮容器中没有提交命令 | 6/6 | — |
| `submit --test` 立即返回；评测中再次提交被拒绝 | — | 6/6 |
| `submit --status` 显示剩余时间与测试状态 | — | 6/6 |
| 报告送达 `/workspace/feedback/test-01/`，不含用例编号 | — | 6/6 |
| `submit --final` 结束尝试并冻结当时的提交 | — | 6/6 |
| 提交次数计入结果 | — | 6/6 |

时限：单轮 60 秒设置下，模型不结束回合时在 60.5 秒冻结，后续命令未执行（`var/smoke-v63/galaxy-deadline`）。

并行：4 个迭代尝试同时提交测试，4 次评测在 19:04 同时开始，分别在 grade-06、grade-12、grade-14、grade-22 上进行（trajectory 28 秒、galaxy 9 秒、polarization 110 秒、conic 94 秒）。

## 演练中发现并修复的问题

- conic 的 public 测试继承了尝试环境中的私有运行副本，评测了私有集（报告未送出结果）。已修复，并加了两道检查（见 `results/OPEN_ITEMS.md`）。
- galaxy 参照 r2 的 `build.sh` 丢失可执行位（解法源码导出时未保留权限）。已修复导出脚本。
- exponential v3 的复用时限 60 秒低于参照组最快成员的用时（118 秒），改为 150 秒（`results/EVAL_TIME_LIMITS.md`）。
- 离线检验 exponential 反馈时，计分函数因传入了清单之外的用例而判无效；已改为只传当前版本的用例。

## public 参照基线（评测槽实测）

| task | 弱参照 | 强参照（50 分锚点） |
| --- | --- | --- |
| galaxy-gravity-v3 | 1.23e-3 | 1.07e-7（r2） |
| polarization-demosaicking-v2 | RMSE 2.194 | 1.631 |
| species-tree-v3 | nRF 0.139 | 0.122 |
| trajectory-covariance-v2 | 0.363 | 0.229 |

conic 与 exponential-v3 沿用 v2 在同一批 public 用例上的参照测量（v3 的 public 用例是其子集）。

## 机器布局

7 个开发槽 + 7 个评测槽（`paired`）。评测核之间的干扰可以忽略；开发核上的人工最坏内存带宽负载会让同一 L3 域的评测计时慢约一倍，因此计时敏感的私有评分在开发全部结束后进行（`docs/platform/CONCURRENCY.md` 第 7 节）。
