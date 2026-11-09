# 协议 v6.2：公开集评测反馈与两种开发设置（设计稿，2026-11-09）

## 动机

conic 修正版的 sol 尝试在开发中跑过全部公开用例，但它自写的测试脚本按统一的 15 秒上限而不是逐例时限判“通过”，把按真实时限只有 0.546 的结果读成了“满足短时限”，也无从知道这比参考（0.789）和自己上一次（0.718）差很多（`CONIC_REVISION_SOL.md`）。v6.2 由评测方在正式评测环境中替 agent 评测公开集，并给出与 baseline 的逐例对比，目的是：

1. 消除 agent 自行评测的误差（开发核计时噪声、判定标准写错）；
2. 让 agent 知道自己与 baseline 的差距在哪里，据此决定改进方向。

## 两种设置（都有可配置的墙钟预算，默认 7200 秒）

| 设置 | 开发中的反馈 | 结束方式 | 评分对象 |
| --- | --- | --- | --- |
| `single`（单轮） | 无 | agent 结束回合或预算耗尽 | 结束时冻结的 `/workspace/submission` |
| `iterative`（迭代） | 测试提交：公开集评测结果与 baseline 对比 | 正式提交，或预算耗尽 | 正式提交的版本；没有正式提交时，为预算耗尽时的 `/workspace/submission`（时间耗尽前的最后一版） |

两种设置都使用同一份题面加评分说明（见下）。`single` 与 v6.1 的区别只有评分说明和可配置预算，用于分离“反馈”本身的效果。

## 迭代设置的接口

容器内提供命令 `/opt/frontieram/bin/submit`：

- `submit --test [--groups G1,G2] [--cases ID1,ID2]`：测试提交。宿主机暂停开发容器，用与正式评测相同的导出器冻结当前 `/workspace/submission`，然后在评测 slot 上用正式评测器评测公开集（可只选部分组或用例），把结果写入容器的 `/workspace/feedback/<序号>/`（`report.md` 和 `report.json`），再恢复容器。命令阻塞直到结果写回，并把摘要打印到终端。
- `submit --final`：正式提交。宿主机冻结当前 `/workspace/submission` 并结束本次尝试。

返回内容：逐例结果（通过/失败、失败原因、耗时、时限、余量），同一用例上 baseline 的结果和耗时，按组汇总，以及公开集上的百分制分数（与正式分数同一映射，锚点是本机在公开集上实测的 baseline）。不返回任何私有数据信息。

## 规则（用户决定，2026-11-09）

- **计时**：测试评测的时间（包括导出、构建和评测）计入开发预算，评测期间开发容器暂停。
- **次数**：不限，只受排队约束。
- **上线范围**：先在公开集现成的题上线（conic 修正版、exponential-v2），sol 先在 v6.2 下重跑；species-tree 等修正版数据，galaxy、trajectory、polarization 补齐公开集后跟进（polarization 的公开场景需出题人提供，见 `POLARIZATION_PUBLIC_DEV_REQUEST.md`）。
- **评分说明**：题面告诉 agent 什么样的算法算好、最终的标准化分数考虑哪些方面，但不给出映射函数。

## 默认决定（待讨论）

- **测试评测的重复次数为 1**（正式评测为 3）。conic 公开集 163 例的逐例时限合计 8,154 秒，其中 25 例为 300 秒；按 3 次重复在一个 slot 上最坏要 6.8 小时，而评测时间计入 2 小时预算，一次测试提交就可能耗尽整个尝试。改为 1 次重复、4 个评测 slot 分片并行后，最坏约 35 分钟，一般远少于此；agent 还可以只评部分组或用例。反馈中注明正式评测要求三次重复都通过，并给出每例的时限余量。conic 评测器的重复次数写死为 3（`grade.py` 的循环、`scoring.py` 的 `REQUIRED_REPEATS` 和判定中的字面量），只在公开评测专用的封存副本中改为读取配置，作为有记录的修改；正式评测的封存副本不变。
- **不排队**：评测时间计入预算，排队等待也会消耗 agent 的时间，对同时开发的其他尝试不公平。因此迭代设置的尝试在本机上一次只运行一个，测试评测独占 4 个评测 slot；单轮设置的尝试不需要评测 slot，可以并行。
- **公开集基线**：在评测 slot 上用与测试评测相同的条件（1 次重复）实测一次并冻结。conic：Clarabel 0.11.1，修正版公开开发清单 163 例及其逐例时限；exponential-v2：冻结的 7 个 panel 成员在公开集（正式规模，64 个计分会话加 8 个控制例）上的 virtual best。
- **公开分数**：conic 用正式映射，p_B 换成公开集上的参考成功率；exponential-v2 用 100/(1+R)，R 相对公开集上的 virtual best。公开分数只用于反馈，不进入任何成绩。
- **预算耗尽时正在评测**：中止评测，冻结当时的提交（即这次测试提交的版本）。

## 评分说明（写入题面，不含映射函数）

- **conic**：每个实例在独立进程中运行三次，三次都必须在该实例的时限内（从进程启动到退出，含解释器启动和导入）给出通过独立检查的结果（解、原始不可行证书或对偶不可行证书）才算解出；`unresolved` 不算解出。最终分数只取决于解出的实例比例，并与一个强参考求解器在同一批实例和时限下的比例比较：达不到参考的九成为 0，与参考持平为中位分，全部解出为满分。时限内的快慢不加分。五类锥（线性、二次、二阶锥、半定、指数锥）都在计分范围内，实例规模从几十到百万变量不等，大多数实例的时限是 1 秒。
- **exponential-v2**：所有用例（计分会话和控制例）都必须在时限和精度要求内给出结果，任何一例不合格整份提交记 0 分。在此前提下，分数取决于每个会话的耗时与一组已发表方法（Expokit、KIOPS、expmv、SciPy、Chebyshev、SLEPc、ExponentialUtilities.jl）中该会话上最快者的比值，按母矩阵、结构族、场景和精度档逐层等权汇总：与这组方法的逐例最快者持平为中位分，越快分数越高。计时覆盖整个进程（含启动）。

## 实现（2026-11-09）

- **公开反馈评测器**（与正式评测器同一套代码，只换数据，正式封存不变）：
  - conic：`FRONTIERAM_RUNTIME=.runtime-conic-public scripts/prepare.py --conic-task tasks/conic/solver-20261105 --conic-profile-template tasks/conic/solver-20261105-profile_template.json --conic-public-repeats 1`。`publicise_conic_task` 删除封存副本里的私有输入，换成包内公开开发集 163 例（逐文件核对清单哈希，时限沿用公开清单），清单转为私有清单格式，重新串联评分锚点、包锁和 profile 的锁定文件哈希；重复次数的 7 处写死改为读配置，每处要求原文恰好出现一次。
  - exponential-v2：`FRONTIERAM_EXP2_SEAL=.runtime-exp2-public scripts/prepare_exponential_v2.py --graded cases/public --repeats 1`（72 例，冻结的 7 个 panel 成员）。
- **公开评测命令**（`arena/public_eval.py`）：`bench --store store-public-<task> public-baseline --task T` 在全部评测 slot 上分片测参考并登记；`bench ... public-eval --task T --source DIR --out DIR [--groups] [--cases]` 导入源码（经正式的提交校验），按全集的分片和 slot 映射只评所选用例，与公开基线逐例对比，写 `report.json` 和 `report.md`。
- **harness**（`arena/native.py`）：`./bench develop --protocol v6.2 --setting single|iterative [--development-seconds N]`。迭代设置向容器只读挂载 `/opt/frontieram/bin/submit`，并挂载评测 socket 目录；宿主侧 `EvalService` 处理测试和正式提交：测试时暂停容器、用正式导出器冻结快照（保留容器）、以子进程运行 `public-eval`、恢复容器并把报告写入 `/workspace/feedback/test-<n>/`。主循环每秒检查正式提交和截止时间；截止时立即暂停并冻结，未完成的评测继续跑完但结果不再送达，在释放迭代锁之前等它结束，因此它的评分任务不会与下一次迭代尝试重叠。迭代尝试先取得机器级锁 `/run/frontieram/v62-iterative.lock` 再开始计时。每次尝试把实际提示存为 `prompt.txt`，测试记录存入 `develop_result.json` 的 `evaluations`。
- **验证脚本**：`scripts/v62-smoke.py TASK SETTING SECONDS OUT` 用 shell 脚本代替模型，在真实开发容器中走完一次尝试并检查结果。单轮设置已在 conic 上通过：结束回合即冻结，容器内没有评测接口和 socket（`store/reruns/v62-smoke/conic-single-1`）。

## 更正（11-09 16:40）：exponential-v2 的计时说明

exponential-v2 的计时从交付矩阵第一个字节开始，READY 之前的导入不计（题面 INSTRUCTION.md 与 V2.md）。v6.2 评分说明里原写“计时覆盖整个进程，包括启动”，以及 v6.1 起附加的通用计时说明（“从进程启动到退出，包括解释器启动和导入”）都与这道题不符。已改为：v6.2 下 exponential 类题目的评分说明和计时说明都改为“按题目自己的接口规则计时”；v6.1 的提示保持原样不改（历史协议）。11-09 12:55 开始的 sol exponential-v2 单轮尝试（`exponential-v2-20261109T064531Z-785a70`）用的是有误的提示，照常评分但只作记录、不计成绩；16:44 以修正后的提示重跑（`dev-exponential-v2-20261109T084408Z-a55f39`），以重跑为正式成绩（默认决定）。v6.1 的 sol 尝试同样看到了那句通用说明，但题面原文写得很明确，成绩不改。

## 进度

1. 已完成：设计、两个公开反馈评测器、公开评测命令、harness 两种设置、单轮设置验证。
2. 进行中：gpt-6.1-sol 的单轮设置尝试（conic、exponential-v2，11-09 12:55 开始）。
3. 等 exponential-v2 私有集 panel 基线让出评测 slot 后：测两题的公开基线，验证迭代设置（测试提交、反馈写回、正式提交、截止时间），再逐个运行 sol 的迭代设置尝试，最后在私有集上正式评分。
