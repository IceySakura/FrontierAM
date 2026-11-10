# 指数传播 v2：进展与待讨论事项（2026-11-05，11-10 更新）

目标（见 `tasks/exponential/versions/v2/DESIGN.md`）：把参考换成已发表人类方法组成的参照组（panel），数据集换成更大、含标准矩阵和合成算子的 v2，参考解独立验证，分数仍按 50/100 约定。

## 1. 现状

| 部分 | 状态 |
| --- | --- |
| 参照组 | 5 个成员：Expokit（Sidje 1998，作者 Fortran）、KIOPS（Gaudreault 等 2018，作者 MATLAB/Octave）、expmv（Al-Mohy & Higham 2011，作者 MATLAB/Octave）、SciPy `expm_multiply`、Chebyshev 文献实现（Tal-Ezer & Kosloff 1984 等，由未接触任何 AI 提交的 agent 编写）。SLEPc、ExponentialUtilities.jl、phipm、rational Krylov 未接入（构建依赖或许可证问题，见 DESIGN 第 3 节） |
| 数据集 v2 生成 | 合成母矩阵（6 个结构族）与生成器完成；标准矩阵改从佛罗里达大学的 SuiteSparse 镜像下载（原两个站点不可达），9 个都能取到，road_usa（118 MB）下载中 |
| 参考解 | `dataset_v2/truth.py`：扩展精度 Chebyshev 为主、双精度截断 Taylor 审计、差异超限时用稠密特征分解仲裁（n ≤ 6000）；冒烟集 48/48 合格（`dataset_v2/truth_plan.md`） |
| 评测接入 | `arena/exponential_v2.py`（任务名 `exponential-v2`）、`scripts/seal/prepare_exponential_v2.py`（封存到 `.runtime-exp2`）。沿用 v1 的评测器、工具、镜像和计分器，只换清单、用例和参考；在开发核上用 3 个用例、5 个成员跑通 |
| 公开开发包 | v1 公开包去掉旧数据，加 `V2.md`（覆盖 v1 关于参考、数据和计时的条款）与 v2 公开用例 |
| 本机基线（冒烟集，流程验证） | 已完成并激活（grade-06，安静条件，Chebyshev 为修正前版本）。virtual best 覆盖全部 56 个用例；各成员的合格用例数：Expokit 45、KIOPS 49、expmv 50、Chebyshev 39，SciPy 因基础设施故障只跑了 2 个（见下）。各用例的最快成员：Chebyshev 30、Expokit 16、KIOPS 5、expmv 5。不合格集中在 signed、heavytail、logspectrum、geometric 这几族，原因是 v1 标定的容差在解强衰减时不够（见第 3 节）。**这次基线不作为正式锚点**：正式基线在私有集上用重新标定、修正后的成员重测 |
| 基础设施事故（已修正） | 16:56 镜像导入脚本在 grade-06 两个评测会话之间的空档（slot 没有容器、但仍有租约）向它的 docker 守护进程导入镜像，守护进程跑在计时 CPU 的 SMT 兄弟核上：SciPy 成员的一次 `docker create` 超时（30 秒）导致该成员评测中断，那段时间其他成员的计时也受干扰。`scripts/setup/load-grade-images.sh` 已改为按调度器租约判断空闲（安静评测期间所有 slot 都持有租约） |

## 2. 计分（50/100 约定）

沿用 v1 的 `tools/scoring.py`：逐例时间比 `r_i = T_i / T_ref_i`，按母算子、结构族、场景、精度逐层等权平均得 R，主分 `100 / (1 + R)`。v2 只改变 `T_ref_i`：取该用例上三次都合格的 panel 成员中，中位时间最短者（virtual best）。参照组自身为 50 分，平均归一化时间减半为 66.67 分，满分 100。

**默认决定**：DESIGN 第 4 节原本提议废除 `100/(1+R)`、改报几何速度比；为符合全局 50/100 约定，保留 `100/(1+R)`，只把参考换成 virtual best。

**补充报告（已实现，`arena/exponential_v2.py::speed_ratio`）**：几何速度比 S = exp(−平均 log(T_候选/T_virtual best))，母算子等权、母算子内用例等权；95% 区间用分层 bootstrap（重抽母算子、母算子内重抽用例、每个用例内重抽候选和每个参照成员的重复，并在每个重抽样本内重新取中位数与 virtual best，从而计入“取最快成员”本身的不确定性），2000 次。S > 1 表示比参照组快。“超越人类 SOTA”的判据按 DESIGN：S 的 95% 区间下限大于 1，且所有计分用例合格。

## 3. 发现与修正

- **v1 上的标定不能直接用到 v2**：在 v2 冒烟用例上，Expokit（容差 1000·rtol）在 1 个 reuse 用例、Chebyshev 在 2 个 heavytail 用例上精度不合格。按设计，不合格的成员不进入该例的取最小，但标定必须改在 v2 公开数据上重做。
- **Chebyshev 的缺陷与修正**：原截断规则按输入范数定精度目标，解强烈衰减时目标过松。按其 `CALIBRATION.md` 允许的“有记录的统一修正”，由独立 agent 改为指数项与 φ₁ 项分开展开、按输出范数做事后截断检验：v2 冒烟集 56/56（最大误差为容差的 0.113），v1 公开集 29/29（0.075）。
- **参考解方法**：双精度 Chebyshev 在 eta = 1e5、p10 时误差达容差的 0.108；Lanczos 审计在刚性算子上停滞。最终用扩展精度 Chebyshev，并发现双精度 Taylor 审计本身在对数谱病态例上不准（差 0.237，而主方法与稠密特征分解相差 0.000），因此加了稠密仲裁。
- **标准矩阵 PSD 检查**：bcsstk17/18 等有限元矩阵条件数约 1e10，普通 Lanczos 求最小特征值不收敛而被误判；已加移位求逆回退（bcsstk17 最小特征值为 1，正定）。thermal1 的分组应为 Schmid（原清单写成 Boeing）。

## 4. 接下来（自动排在空闲窗口，见 `scripts/references/exp2-public.sh`）

1. 生成正式规模的公开集（12 个母矩阵：4 个标准、8 个合成，72 个计分会话加 8 个控制例），计算参考解。
2. panel 在 v2 公开集上按各自事先冻结的规则重新标定（Expokit 容差网格、expmv 精度档），KIOPS、SciPy、Chebyshev 只做验证；冻结参数。
3. 冻结之后才生成私有集（5 个标准、6 个合成母矩阵），计算参考解。私有种子文件标注“先别生成”，就是为了保证标定时私有数据还不存在。
4. 用私有集重新封存 `exponential-v2`，跑正式 panel 基线，并用一份已有的 v1 提交做端到端验证。

## 4b. 2026-11-06 进展与决定

用户确定的原则：baseline 对标人类 SOTA，数据集对标公认的公正评测，超越 baseline 应能被学术界认可为超越 SOTA。据此：

- **公开集**：12 个母矩阵中 bcsstk18 未通过 PSD 检查之外的 11 个（4 个标准、7 个合成）生成了 66 个计分会话加 8 个控制例。参考解 64/66 合格；`syn_diffusion2d_p10_k16`、`syn_diffusion2d_b_p10_k16` 的主方法与审计差 0.031 和 0.029（门槛 0.02），n ≈ 67,000 无法稠密仲裁，**剔除，不计分**（`cases/public/manifest.all.json` 保留原清单）。
- **标定规则修订**（`calibration_v2/RULE.md`）：只用参考解合格的用例；超时等资源失败是成员在该用例上的失败、不使参数值不合格；网格可在公开数据上扩展；容差的“安全区”只要求到 p10 档 100 倍机器精度为止。结果：Expokit 3e-4·rtol，KIOPS 1e-3·rtol，expmv 进行中；SciPy（12 个 geometric 用例超时，精度最大 0.018）与 Chebyshev（72/72，最大 0.102）保留。
- **加强参照组**：现有 5 个成员都是经典方法；为使“超越参照组”具备学术说服力，新增三类公认的强方法，各由未接触任何模型提交的独立 agent 实现，并按同一规则在公开集上标定后冻结：有理（移位求逆）Krylov（van den Eshof & Hochbruck 2006）、ExponentialUtilities.jl（SciML）、SLEPc MFN。
- **冻结结果（11-06 17:04，`calibration_v2/FROZEN.json`）**：7 个成员入选——Expokit 3e-4·rtol、KIOPS 1e-3·rtol、expmv（p4 single，p7/p10 double）、ExponentialUtilities.jl 1e-6、SLEPc MFN 1e-3·rtol、SciPy、Chebyshev。**有理 Krylov 未入选**：修正 1（刚性会话的内层容差随倍数变化）之后，倍数 ≤ 0.03 时已无精度失败，但 `syn_heavytail_p10_k16` 的误差固定在门槛的 0.122，略高于余量 0.1，且不随倍数变化（另有一处固定的内部容差，未查实）。panel 冻结后私有集即开始生成，本轮不再改动；修好后可纳入下一个冻结版本（待讨论）。SLEPc 的网格在 3e-3 处不单调（一个精度失败），按规则选更严的 1e-3。
- **私有集**：只有在加强后的参照组冻结之后才生成（5 个标准矩阵：thermal1、email-Enron、cit-HepPh、road_usa 与 fe_ocean 的 BFS 子图；6 个合成族），参考解同样要求独立验证，不合格的剔除。
- **开发包先行**：公开开发包与参照组、私有集无关，已封存（冒烟集 56 例 + 正式规模 one_shot 40 例，172 MB），gpt-6.1-sol 的 v6.1 尝试先用它开发；评分在最终封存后进行。

## 4c. 2026-11-06 至 11-09：私有集、封存修正、正式基线

- **私有集**（panel 冻结 17:04 之后生成，18:45 封存）：11 个母矩阵（标准 5 个：Schmid/thermal1、SNAP/email-Enron、SNAP/cit-HepPh、DIMACS10/road_usa 与 DIMACS10/fe_ocean 的 BFS 子图；合成 6 族：diffusion2d、diffusion3d、geometric、heavytail、logspectrum、signed）× 3 个精度档 × 2 种场景（one_shot、reuse 16 个右端），共 66 个计分会话，另有 8 个控制例；n = 34,546–120,000，非零元最多 899,980。参考解 66/66 合格、无剔除：主方法与独立审计的最大差异为门槛的 0.0156（合格线 0.02），没有用例需要稠密仲裁，共 6.3 CPU 小时。（清单顶部的 `truth_note` 仍是生成时的“占位符”说明，逐例状态与数据已核实为真实参考解。）
- **gpt-6.1-sol 候选评测**（11-06 23:49，4 个分片在 4 个评测 slot 上并行，shared）：74/74 合格。
- **panel 基线第一次没跑成（11-06 23:49，到 11-09 才发现）**：封存时 Julia 运行时里的库符号链接被展开成实体文件（评测器出于安全拒绝链接，展开本身是必要的），ExponentialUtilities.jl 从 959 MB 变成 1.42 GB，超过单个提交 1 GiB 的上限，基线入队即被拒；后续评分在空的套件编号上报错退出。
- **修正**（不改冻结成员本身，`FROZEN.json` 不变）：
  - `scripts/seal/prepare_exponential_v2.py::seal_member`：每组链接（目标文件及解析到它的所有链接）只物化需要的名字：在成员任一普通文件中以字符串出现的名字（ELF NEEDED、JLL 库名、sysimage 中的 dlopen 字符串），加上实测加载过的名字（`calibration_v2/loader_names.json`：在宿主机上以 `LD_DEBUG=files` 跑完公开冒烟集 56 例记录，Julia 的 `dlopen("libpcre2-8")` 会自动补 `.so`，单靠字符串扫描会漏）。ExponentialUtilities.jl 封存后 1.008 GB，SLEPc 27 MB。第一版只用字符串规则，漏掉 `libpcre2-8.so`，在评测镜像里全部启动失败，因此加了实测清单。
  - `arena/exponential_v2.py`：基线引用只带成员名单，评分时逐个成员从已校验哈希的封存目录读取，并各自作为独立提交校验（每个成员都受 1 GiB 上限）；`verify()` 增加检查，封存 panel 目录中不得有 lock 之外的文件。
  - 重新封存后数据、评测器、公开包等 81 个非 panel 文件与 11-06 的封存逐字节相同，所以 sol 的候选评测仍然有效，不重评。
- **容器内功能检查**：SLEPc 和 ExponentialUtilities.jl 的标定是在宿主机上直接运行的（`run_local.py`），此前从未在评测镜像中跑过。在开发 slot 上按正式路径（离线构建、`run_docker.py`）检查：SLEPc 56 例中 55 例合格（1 例精度失败，属于正常的单例失败）；ExponentialUtilities.jl 修正后前 13 例全部合格，耗时与宿主机一致（中位比 0.97），其后为避免与计时评测同一 L3 域而停止。
- **panel 基线重跑**：11-09 11:20 启动后发现上述检查留下的孤儿评测器仍在核 4（与 grade-06 同一 L3 域）上运行了约 50 秒，该次基线作废（`var/stores/store/jobs/failed/job-20261109T032027Z-*`），11:22 在干净的机器上重新开始（`suite-exponential-v2-20261109T032227Z-c09381`），分片与 sol 的候选评测相同（同一用例划分、同一分片到 slot 的映射），评分前由 `scripts/references/exp2-panel-suite.sh` 核对。

## 4d. gpt-6.1-sol 正式成绩（11-09 15:58，`tasks/exponential/analysis/exp2-sol-20261109/score.json`）

panel 基线 `suite-exponential-v2-20261109T032227Z-c09381`（11-09 11:22–15:58，4 个分片在 4 个评测 slot 上并行，与 sol 的候选评测分片和 slot 映射逐一核对一致）覆盖全部 74 例。逐例最快的成员：Chebyshev 60 例、Expokit 9 例、SLEPc 4 例、ExponentialUtilities.jl 1 例；各成员合格例数：Chebyshev 72、Expokit 71、KIOPS 71、SLEPc 68、SciPy 63、ExponentialUtilities.jl 58、expmv 57。

| 原始指标 | 映射 | 得分（满分 100，参照组 50） |
| --- | --- | --- |
| 74/74 例合格；R = 0.596（相对 panel 逐例最快者的耗时比，按母矩阵、结构族、场景、精度逐层等权平均） | 100 / (1 + R) | **62.7** |

补充：几何速度比 S = 2.01，95% 分层 bootstrap 区间 [1.45, 2.91]（2000 次）。区间下限大于 1 且所有计分会话合格，按 DESIGN 的判据，sol 的这份提交在本私有集上显著快于参照组的逐例最快者。

按结构族的几何平均耗时比（越小越快）：几何图 0.14、有限元标准矩阵 0.29、网格扩散 0.34、带符号关联 0.75、网络算子 0.76、标准图 0.66、对角局部 0.93。66 个计分会话中 7 个比参照组慢（最慢为 1.34 倍）。重用 16 个右端的场景优势更大（几何平均 0.36–0.54），单次求解为 0.55–0.69。

需要注意：逐例最快的成员 60/74 是 Chebyshev，而它是由未接触任何模型提交的独立 agent 按文献实现的版本，不是已发表的作者代码；如果它的实现本身不够强，参照组会被低估。其余成员（Expokit、KIOPS、expmv 为作者代码，SciPy、SLEPc、ExponentialUtilities.jl 为公认的库）在多数用例上比它慢。是否需要以作者代码替换 Chebyshev 成员待讨论。

## 4e. 协议 v6.2 成绩（11-10 02:16，`results/runs/v62-*-exp2*/score.json`）

同一 panel 基线、同一分片和 slot 映射，映射 100 / (1 + R)。

| 模型与设置 | 原始指标 | S（95% 区间） | 得分 |
| --- | --- | --- | --- |
| gpt-6.1-sol，单轮（修正提示） | 58 例精度不合格 | — | **0** |
| gpt-6.1-sol，迭代 | 74/74 合格，R = 0.589 | 1.84（1.48–2.30） | **62.9** |
| claude-opus-5-5，单轮 | 74/74 合格，R = 0.412 | 2.99（2.05–4.47） | **70.8** |
| claude-opus-5-5，迭代 | 74/74 合格，R = 0.448 | 2.60（1.88–3.69） | **69.1** |

只作记录：sol 用旧题面的迭代试点 R = 0.517，65.9 分；sol 用有误计时说明的单轮尝试 R = 0.776，56.3 分。各次公开集测试分数与私有集分数的差距在 0–7 分之间（例如 opus 迭代公开集最后一次 62.8，私有集 69.1），公开集可以作为方向参考，但不能精确预测私有集排名。

## 5. 待讨论

- **数据规模**：DESIGN 要求私有 16 个母矩阵；目前标准矩阵 5 个加合成 6 个，共 11 个。可以增加合成实例或换用更多标准矩阵。
- **公开包大小**：开发容器的 `/workspace` 只有 1 GiB，正式规模的 reuse 用例每个约 30 MB，因此公开包只放冒烟集全部用例加正式规模的 one_shot 用例。
- **计时与交错**：DESIGN 要求 7 次重复并与 panel 随机交错；目前沿用 v1 评测器的 3 次重复，panel 基线与候选在同一 slot 上先后测量。
- **参考解成本**：扩展精度约为双精度估算的 5 倍，私有集参考解需要数十到上百 CPU 小时。
- **重计算的时段**：每个 L3 域都有一个评测 slot，参考解和标定会干扰计时评测，所以排在新题基线之后、恢复 minimax 锥优化评测之前的空闲窗口；minimax 的结果不进报告，晚开始不影响它。
