# 指数传播 v2：进展与待讨论事项（2026-11-05）

目标（见 `tasks/exponential/sota_v1/DESIGN.md`）：把参考换成已发表人类方法组成的参照组（panel），数据集换成更大、含标准矩阵和合成算子的 v2，参考解独立验证，分数仍按 50/100 约定。

## 1. 现状

| 部分 | 状态 |
| --- | --- |
| 参照组 | 5 个成员：Expokit（Sidje 1998，作者 Fortran）、KIOPS（Gaudreault 等 2018，作者 MATLAB/Octave）、expmv（Al-Mohy & Higham 2011，作者 MATLAB/Octave）、SciPy `expm_multiply`、Chebyshev 文献实现（Tal-Ezer & Kosloff 1984 等，由未接触任何 AI 提交的 agent 编写）。SLEPc、ExponentialUtilities.jl、phipm、rational Krylov 未接入（构建依赖或许可证问题，见 DESIGN 第 3 节） |
| 数据集 v2 生成 | 合成母矩阵（6 个结构族）与生成器完成；标准矩阵改从佛罗里达大学的 SuiteSparse 镜像下载（原两个站点不可达），9 个都能取到，road_usa（118 MB）下载中 |
| 参考解 | `dataset_v2/truth.py`：扩展精度 Chebyshev 为主、双精度截断 Taylor 审计、差异超限时用稠密特征分解仲裁（n ≤ 6000）；冒烟集 48/48 合格（`dataset_v2/truth_plan.md`） |
| 评测接入 | `arena/exponential_v2.py`（任务名 `exponential-v2`）、`scripts/prepare_exponential_v2.py`（封存到 `.runtime-exp2`）。沿用 v1 的评测器、工具、镜像和计分器，只换清单、用例和参考；在开发核上用 3 个用例、5 个成员跑通 |
| 公开开发包 | v1 公开包去掉旧数据，加 `V2.md`（覆盖 v1 关于参考、数据和计时的条款）与 v2 公开用例 |
| 本机基线（冒烟集，流程验证） | 已完成并激活（grade-06，安静条件，Chebyshev 为修正前版本）。virtual best 覆盖全部 56 个用例；各成员的合格用例数：Expokit 45、KIOPS 49、expmv 50、Chebyshev 39，SciPy 因基础设施故障只跑了 2 个（见下）。各用例的最快成员：Chebyshev 30、Expokit 16、KIOPS 5、expmv 5。不合格集中在 signed、heavytail、logspectrum、geometric 这几族，原因是 v1 标定的容差在解强衰减时不够（见第 3 节）。**这次基线不作为正式锚点**：正式基线在私有集上用重新标定、修正后的成员重测 |
| 基础设施事故（已修正） | 16:56 镜像导入脚本在 grade-06 两个评测会话之间的空档（slot 没有容器、但仍有租约）向它的 docker 守护进程导入镜像，守护进程跑在计时 CPU 的 SMT 兄弟核上：SciPy 成员的一次 `docker create` 超时（30 秒）导致该成员评测中断，那段时间其他成员的计时也受干扰。`scripts/load-grade-images.sh` 已改为按调度器租约判断空闲（安静评测期间所有 slot 都持有租约） |

## 2. 计分（50/100 约定）

沿用 v1 的 `tools/scoring.py`：逐例时间比 `r_i = T_i / T_ref_i`，按母算子、结构族、场景、精度逐层等权平均得 R，主分 `100 / (1 + R)`。v2 只改变 `T_ref_i`：取该用例上三次都合格的 panel 成员中，中位时间最短者（virtual best）。参照组自身为 50 分，平均归一化时间减半为 66.67 分，满分 100。

**默认决定**：DESIGN 第 4 节原本提议废除 `100/(1+R)`、改报几何速度比；为符合全局 50/100 约定，保留 `100/(1+R)`，只把参考换成 virtual best。几何速度比和置信区间可作为附加报告。

## 3. 发现与修正

- **v1 上的标定不能直接用到 v2**：在 v2 冒烟用例上，Expokit（容差 1000·rtol）在 1 个 reuse 用例、Chebyshev 在 2 个 heavytail 用例上精度不合格。按设计，不合格的成员不进入该例的取最小，但标定必须改在 v2 公开数据上重做。
- **Chebyshev 的缺陷与修正**：原截断规则按输入范数定精度目标，解强烈衰减时目标过松。按其 `CALIBRATION.md` 允许的“有记录的统一修正”，由独立 agent 改为指数项与 φ₁ 项分开展开、按输出范数做事后截断检验：v2 冒烟集 56/56（最大误差为容差的 0.113），v1 公开集 29/29（0.075）。
- **参考解方法**：双精度 Chebyshev 在 eta = 1e5、p10 时误差达容差的 0.108；Lanczos 审计在刚性算子上停滞。最终用扩展精度 Chebyshev，并发现双精度 Taylor 审计本身在对数谱病态例上不准（差 0.237，而主方法与稠密特征分解相差 0.000），因此加了稠密仲裁。
- **标准矩阵 PSD 检查**：bcsstk17/18 等有限元矩阵条件数约 1e10，普通 Lanczos 求最小特征值不收敛而被误判；已加移位求逆回退（bcsstk17 最小特征值为 1，正定）。thermal1 的分组应为 Schmid（原清单写成 Boeing）。

## 4. 接下来（自动排在空闲窗口，见 `scripts/exp2-public.sh`）

1. 生成正式规模的公开集（12 个母矩阵：4 个标准、8 个合成，72 个计分会话加 8 个控制例），计算参考解。
2. panel 在 v2 公开集上按各自事先冻结的规则重新标定（Expokit 容差网格、expmv 精度档），KIOPS、SciPy、Chebyshev 只做验证；冻结参数。
3. 冻结之后才生成私有集（5 个标准、6 个合成母矩阵），计算参考解。私有种子文件标注“先别生成”，就是为了保证标定时私有数据还不存在。
4. 用私有集重新封存 `exponential-v2`，跑正式 panel 基线，并用一份已有的 v1 提交做端到端验证。

## 5. 待讨论

- **数据规模**：DESIGN 要求私有 16 个母矩阵；目前标准矩阵 5 个加合成 6 个，共 11 个。可以增加合成实例或换用更多标准矩阵。
- **公开包大小**：开发容器的 `/workspace` 只有 1 GiB，正式规模的 reuse 用例每个约 30 MB，因此公开包只放冒烟集全部用例加正式规模的 one_shot 用例。
- **计时与交错**：DESIGN 要求 7 次重复并与 panel 随机交错；目前沿用 v1 评测器的 3 次重复，panel 基线与候选在同一 slot 上先后测量。
- **参考解成本**：扩展精度约为双精度估算的 5 倍，私有集参考解需要数十到上百 CPU 小时。
- **重计算的时段**：每个 L3 域都有一个评测 slot，参考解和标定会干扰计时评测，所以排在新题基线之后、恢复 minimax 锥优化评测之前的空闲窗口；minimax 的结果不进报告，晚开始不影响它。
