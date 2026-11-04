# Exponential propagation：human-SOTA benchmark v1 设计

## 1. 设计原则

目标不是找一个方便的单一 reference，而是回答“在相同协议、精度、CPU 和信息约束下，agent 是否超过已公开的人类方法组合”。因此：

1. 数据必须让 sparse matvec、Krylov dimension、stiffness 与 reuse 真正主导，不能由 process/runtime fixed overhead 主导。
2. accuracy truth 与 timing baseline 分离。truth 宁可慢且多路交叉验证；timing baseline 是 public-calibrated、测试前冻结的 human panel。
3. panel 允许按 case 形成 virtual best，但不允许看 agent、private timing 或 private error 后改算法、参数 grid 或成员。
4. operator mother 是统计独立单位；同一矩阵的 precision/scenario 变体不伪装成六份结构证据。

## 2. Dataset v2

### 2.1 规模和结构

正式 private 建议 16 个 scored mothers，每个产生 `p4/p7/p10 × one_shot/reuse` 六个 sessions，共 96 个 scored sessions；另有 8 个解析/退化 controls（不计时），合计 104。public dev 使用 8 个不同 mothers、同样六变体（48）加 8 controls。每个 mother 的所有变体必须留在同一 split；任何 standard matrix、其 induced subgraph、重标度版本或同 seed synthetic realization 都不得跨 split。

每个 structural family 在 public/private 都出现，但 mother 不重合。这既能公开算法开发，又防止记忆具体谱/ordering。建议 private 的 16 mothers 分配为：standard graph Laplacian 5、standard FEM/diffusion SPD 4、synthetic geometric/mesh 3、synthetic irregular/network 2、diagonal-plus-low/local potential 2。public 用每类 1–2 个较小但不玩具化的独立实例。

正式 scored 范围：

* `n`: 20,000–120,000；至少 75% 的 mothers `n>=50,000`。
* `nnz`: 100,000–1,000,000；至少一半 `nnz>=500,000`。
* 每行完整对称 CSR；原始 matrix 若超限，只允许在冻结前按公开 deterministic rule 取 induced subgraph。
* `eta=t*||L||inf` 使用近似对数分层：`1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 3e4, 1e5`。每个 scenario/precision 的加权统计中，`eta<=10`、`10<eta<=1e3`、`eta>1e3` 均有独立 mothers；至少 3 个 private mothers 在 `eta>=3e4`。
* 对每个 operator 先以 positive scalar 缩放，再选 `t<=1e4` 达到目标 eta；缩放不改变 sparsity/PSD。manifest 公布 eta bucket 而非 private 精确值。

### 2.2 具体 standard operators

所有下载版本、SuiteSparse collection revision、原始 SHA-256、转换脚本和 vertex ordering 在生成前冻结。推荐清单如下；括号是 collection 中常见的原始量级，最终脚本必须重新核对元数据。

**FEM / diffusion SPD（直接把 matrix 当 L，必要时只做对称一致性检查和 positive scaling）：**

* `HB/bcsstk17`（n≈10,974，nnz≈428,650）与 `HB/bcsstk18`（n≈11,948，nnz≈149,090）：经典 structural FEM SPD，较高 fill/row，适合暴露 sparse kernel 与 Lanczos 正交化成本。它们只放 public 或 calibration，因为 n 偏小。
* `thermal1`（n≈82,654，nnz≈574,458）：大 thermal FEM/diffusion，private；规模已足以消除 interpreter startup。
* `GHS_psdef/...` 中经 Cholesky/eigen spot-check 确认 SPD 且 `nnz<=10^6` 的一例作为 private。若 collection revision 中候选超限或 PSD metadata 不可靠，改用 `thermal1` 的另一独立标准 thermal operator，不进行截断 principal minor 后冒称原 matrix。

**标准 graph/mesh adjacency 转 combinatorial Laplacian `D-(A+A^T)/2`：**

* `DIMACS10/fe_4elt2`（n≈11,143，约 65k adjacency nnz）和 `DIMACS10/wing_nodal`（n≈10,937，约 75k）：public mesh topology。
* `Gleich/power-US-Grid`（n=4,941，edges≈6,594）：public control/calibration，不进入最终计时主集；它代表 highly irregular low-degree grid，但太小。
* SNAP `email-Enron`（n≈36,692，undirected edge order 1.8e5）：private social/network Laplacian。
* SNAP `cit-HepPh`（n≈34,546，edge order 4.2e5，symmetrized Laplacian 接近 nnz 上限）：private high-degree graph。
* `DIMACS10/road_usa` 或 SNAP `roadNet-CA`：原图超 protocol 上限；按公开规则“取 canonical vertex order 中首个尚未使用 vertex，做 BFS，直到 120,000 vertices，再取 induced edges”得到 private road mother。记录 parent SHA、选点 hash；这是明确命名的 derived operator，不把它称为原完整 matrix。
* `DIMACS10/fe_ocean` 同样按冻结 BFS rule 取约 100,000 vertices，作为 private unstructured mesh；public 使用完全不同的 `fe_4elt2`。

若重新核对后某 matrix 的对称化产生 `nnz>10^6`，在生成前替换，不可在生成后按 private timing 挑选。标准矩阵的意义是覆盖真实 degree distribution、separator、cache locality 和 spectral clustering，不是借名称制造多样性。

### 2.3 Synthetic families

private 另含以下 frozen-seed mothers：

* 2-D 5/9-point anisotropic diffusion，`300×300`（n=90k，nnz≈0.45–0.81m），coefficient contrast `1e4`，含 nonzero diagonal potential。
* 3-D 7-point diffusion，`46^3`（n=97,336，nnz≈0.67m），各向异性和 mixed coefficient blocks。
* random geometric graph Laplacian，n=100k，目标平均 degree 8；边 weight log-uniform。
* Barabási–Albert/configuration-model Laplacian，n=100k，约 4–8 edges/new vertex，测试 heavy-tailed degree。
* `L=B^T W B + diag(v)` 的 signed-incidence PSD operator，n=50k–100k；它不保证 off-diagonal 非正，防止 solver 偷用 pure Laplacian 假设。
* diagonal/log-spectrum + sparse orthogonal block couplings，n=120k；用于可解析 truth 与极端 spectral spread。

public 使用不同 dimension、seed、coefficient field 和 graph model realization。所有 random seeds 在评测前 sealed；generator source 与参数范围公开。

### 2.4 RHS、reuse 与权重

每个 reuse session 的 16 个 RHS 依次混合：Rademacher/dense 6、localized impulses 4、graph-smooth/low-frequency 3、high-frequency 2、zero/near-degenerate 1；下一 RHS 仍严格在上一结果返回后发送。one-shot 从相同类别分层抽取，但不复制 reuse 第一个 RHS。`b0,b1` 独立变化，并包含 `b0=0` 或 `b1=0` 的非控制 scored request，避免只优化一种项。

统计先在 mother 内平均 precision/scenario 变体，再对 family 等权；standard/synthetic、eta 三档、one-shot/reuse、precision 三档均报告 subgroup ratio。这样一个 16-RHS session 不会仅因含 16 vectors 获得 16 倍统计权。

### 2.5 Truth 计算与独立审计

primary truth 用 symmetric spectral interval `[0,lambda_upper]` 上的 high-accuracy Chebyshev/Clenshaw action，分别计算
`f0(x)=exp(-t x)` 与 `f1(x)=(1-exp(-t x))/x`（在 0 用连续延拓 t），然后组合。`lambda_upper=min(||L||inf, verified Lanczos upper bound with safety inflation)`；系数用 MPFR/Arb 256-bit 生成，degree 由 coefficient-tail bound 选到 `<=1e-3` 倍 p10 acceptance threshold，并再加 roundoff budget。计算端使用 long double accumulation 或 compensated binary64 block SpMM。

independent cross-check 至少两路：

1. restarted high-accuracy Lanczos/KIOPS/phipm，目标 `1e-3*rtol`，不使用 timing panel 的 calibrated loose tolerance；
2. 对 `eta>=1e4` 再用 rational Krylov/contour 或 scaling-and-Taylor augmented action；
3. controls/小 principal subproblems 用 dense eigendecomposition 和 arbitrary precision；
4. 要求两路 `||y_a-y_b||inf <= 0.02*(atol+rtol||y_a||inf)`；不满足则 case 是 evaluator error，不能由“多数投票”生成答案。

Chebyshev 在 eta=1e5、p10 下预计约 1k–2k sparse passes；16 RHS 用 block SpMM。以 1e6 nnz、16 RHS、单 CPU 估计每个最难 reuse truth 约 10–40 分钟，全部 104 sessions primary 约 20–60 CPU-hours；独立 Krylov/rational audit 约再 20–80 CPU-hours。正式生成可离线并行到独立 cores，但每个 timing run 仍严格单核。存储 `96*(up to 16)*120k*8` 的 raw truth 上界约 1.5 GB，连 NPZ/metadata 预计 2–4 GB。

## 3. Human-SOTA baseline panel

每个成员均保留 unmodified upstream snapshot；thin adapter 只做 FAMEXP2、CSR callback、`[[-L,b1],[0,0]]` augmentation 或作者 API 参数映射。calibration protocol 预先声明：在 public 48 sessions 上，从固定 parameter grid 选择“满足所有 request 且 max scaled error ≤0.1”的最快 setting；若 error 对参数不单调，要求从该 setting 到更严格端形成连续 safe region。选择规则、grid、原始 report、源码 hash 在任何 agent run 前冻结。private 和 agent 结果绝不用于调参。

### Expokit DSPHIV

Sidje, “Expokit: A Software Package for Computing Matrix Exponentials”, ACM TOMS 24(1), 1998。作者 Fortran distribution；custom notice 允许 non-commercial use/copy/modification with notice。GFortran 12.2 可直接静态编译，现有 frozen member 已运行。优势是 symmetric Lanczos、compiled、低 overhead；短板是固定 `m=30`、不跨 RHS 复用 basis，极 stiff 时可能 time stepping 多。adapter 保持现有 `DSPHIV(A=-L,u=b1,v=b0)`，vendor Fortran 不改。

### Al-Mohy–Higham expmv / SciPy `expm_multiply`

Al-Mohy & Higham, “Computing the Action of the Matrix Exponential, with an Application to Exponential Integrators”, SIAM J. Sci. Comput. 33(2), 2011。作者 MATLAB `expmv` 已下载，包含 2-clause BSD-style license；SciPy 的 `expm_multiply` 是该 scaling/truncated-Taylor family，SciPy BSD-3-Clause。Octave、Python/SciPy 1.18.1 已在镜像；Tencent PyPI mirror 可达，但正式冻结用镜像内版本，不在运行时 pip。优势是 nonnormal/general sparse、norm-estimation 与 large eta scaling 稳健；可能输给 symmetric Lanczos，Python fixed overhead 大。thin adapter 对每个 RHS 使用 `(n+1)` augmented sparse matrix；author MATLAB 和 SciPy implementation 作为两个 panel members，不能把 SciPy probe 冒称 author-code timing。

### KIOPS / `phiv` / `expv`（ExponentialUtilities.jl）

KIOPS：Gaudreault & Pudykiewicz, “An efficient exponential time integration method for the numerical solution of the shallow water equations on the sphere”, JCP 322, 2016（Krylov with incomplete orthogonalization）；仓库中 author commit `9414984…` 与 license 已冻结，Octave 可用。ExponentialUtilities.jl 提供 `phiv_timestep`/KIOPS/expv；MIT Expat，GitHub codeload 可达。Julia 不在 task image，且 package precompile 属大工作，本轮不安装。优势是直接计算 phi linear combination、adaptive timestep/Krylov dimension；Julia 版本可能显著降低 Octave overhead，并可复用 allocation。adapter 传 `[b0,t*b1]`、`A=-tL`；作者文件不改。正式 panel 若不能在 immutable image 中加入 Julia runtime + pinned artifacts，则只保留 author KIOPS/Octave，不能以不可复现成员参与 virtual best。

### phipm

Niesen & Wright, “A Krylov Subspace Algorithm for Evaluating the φ-Functions Appearing in Exponential Integrators”, ACM TOMS Algorithm 919, 2012（preprint 2009/2010）。官方 Leeds 页面声明 MATLAB `phipm`，正好支持 polynomial inhomogeneity、adaptive timestep 与 dimension。当前 official host DNS 失败，且检索结果没有权威 license；因此状态是“方法必须评估，代码尚不可合法冻结/再分发”。若作者页恢复且 license 允许，Octave thin adapter 直接传 `A=-L,b0,b1,t,tol`；否则记录 exclusion，不能用第三方未知改版替代并称 author baseline。

### SLEPc MFN

Hernandez et al., SLEPc 技术论文及 SLEPc MFN documentation；MFN `expokit`/`krylov` 原生实现 `f(A)v`，SLEPc 为 2-clause BSD。v3.18.2 source codeload 已下载。镜像只有 PETSc 3.18 shared runtime，未发现可链接 SLEPc dev library/header；CMake 也缺失。按约束未编译。优势是成熟 compiled sparse infrastructure、可配置 Krylov/restart；短板是本问题的 mixed exp+phi1 需 augmented `(n+1)` operator，且单 MPI rank setup overhead 可能大。thin C adapter 建一次 PETSc Mat/MFN；reuse 只更新 augmented last column并调用 `MFNSolve`，不得修改 SLEPc。

### phipm-style `phiv` 之外的 Chebyshev panel member

Tal-Ezer & Kosloff, “An Accurate and Efficient Scheme for Propagating the Time Dependent Schrödinger Equation”, J. Chem. Phys. 81, 1984；后续 symmetric matrix-function Chebyshev action。它利用 PSD 实谱区间，无 orthogonalization，degree 可预估，特别适合 block/reuse 与 medium/high accuracy。没有一个被确认可直接冻结的原作者 sparse FAMEXP2 package；建议提供独立、审计过的 C++ implementation，明确标注“literature implementation”而非 author code，BSD-3 项目许可。adapter 建 CSR 一次，按 recurrence 同时累计 f0/f1；reuse 缓存 coefficients、scaling 与 workspace。这是 human algorithmic baseline，不能因无作者 package 而从 panel 删除。

### Rational Krylov / shift-and-invert

van den Eshof & Hochbruck, “Preconditioning Lanczos Approximations to the Matrix Exponential”, SIAM J. Sci. Comput. 27(4), 2006。对 `eta` 很大的 SPD operator，`(I+gamma L)^{-1}` 上的 Lanczos 可用较低 dimension 折叠 stiff spectrum；reuse 可摊销 symbolic/numeric factorization 或 preconditioner。通用 sparse Cholesky 在 4 GiB、1 CPU 下可能因 fill 失败，iterative inner solve 也需计入误差。未找到 license 明确且精确匹配本协议的原作者代码；状态与 Chebyshev 相同：实现一个独立 documented member，或审计 PSFUN/其他明确许可实现。thin adapter 的 factor/preconditioner 在收到 L 后构造一次，后续 RHS 复用；任何 factorization 都在 READY 后计时。

### Panel 采用条件

首版 timing panel 至少应包含：Expokit DSPHIV、author KIOPS、SciPy expm_multiply、author Higham expmv/Octave、compiled Chebyshev。SLEPc、ExponentialUtilities、phipm、rational Krylov 在依赖/license/accuracy qualification 完成后加入；不是“列过论文”就算 baseline。若 SLEPc/Julia build 不能装入冻结 image，则发布 unavailable 原因，不给它虚构 timing。

## 4. Scoring 与 SOTA 声明

废除 `100/(1+R)` 饱和映射。对 case i 定义冻结 human virtual best：

`T_H,i = min_j median_r(T_j,i,r)`

其中 j 只遍历预先冻结且该 case 三次都通过 accuracy/resource 的 panel members。agent 同样至少 7 次独立 clean-process repetitions（初版可 5 次 pilot，正式 7 次），runner 在同一 pinned core、同一 CPU governor 状态下将 human members 与 agent 随机交错；报告 `speedup_i=T_H,i/T_A,i`。若某 human member 失败，不进入该 case 的 min，但公开失败；若所有 human member 失败则 evaluator error。

总体 ratio 仍按 mother→family→scenario→precision 的预注册等权层级在 log time-ratio 上聚合，主结果报告 geometric speed ratio `S=exp(-mean(log(T_A/T_H)))`，不压成分数。用 stratified hierarchical bootstrap（先 resample mothers，再在 mother 内 resample repetitions并重算 median/min）给 95% BCa CI；同时报告每个 family、eta bucket、scenario、precision 的 CI、raw median times、RSS 和 failures。virtual-best selection uncertainty必须在每个 bootstrap replicate 内重新取 panel min，不能把已经取最小的点估计当无误差常数。

“beats human SOTA”精确定义：

* 所有 mandatory sessions、requests、repetitions 均通过 correctness/resource；
* benchmark-wide speedup 的 95% CI lower bound `>1.00`；
* 预注册的 six primary subgroups（one-shot/reuse × 三 precision）均 point estimate `>=1`，且没有 subgroup 95% upper slowdown 超过 10%；
* 结果在一次独立 rerun（新 randomized order，同 frozen binaries/data）复现。

另报告更强的“casewise dominance”：至少 90% mothers 的 speedup point estimate >1，且无 mother 显著慢于 1.10×。它不是 benchmark-wide SOTA 的必要条件，二者不得混称。

## 5. 风险

* **SuiteSparse metadata/PSD 错配**：生成脚本做 symmetry residual、随机 Rayleigh、Lanczos smallest-eigenvalue 和原始 provenance 审计；不可靠 matrix 替换而非偷偷 diagonal shift。
* **数据仍太小**：以 `n>=50k/nnz>=500k` 配额和 startup 占比 `<5%` pilot gate 控制；不满足就增大，不用减时间 floor 掩盖。
* **truth 不够准**：两条算法和严格 fraction-of-tolerance gate；disagreement 直接 evaluator error。
* **virtual best 的 winner’s curse**：7 repeats、interleaved order、bootstrap 内重选 min；同时报告 panel spread。
* **public overfitting**：parameter grid/rule 预注册，mother/seed/standard operator 派生链不跨 split。
* **license**：phipm/rational code 未澄清前不分发、不进入 frozen panel；每个 archive 保留 notice。
* **dependency/image drift**：最终 image digest、compiler flags、CPU ISA 两种 binary、Julia/PETSc artifact hash 全冻结。
* **rational factor memory/time**：先在 public largest cases gate 4 GiB；失败是该 member 的正常 case failure，不放宽资源。
* **CPU noise/thermal**：exclusive pinned core、随机交错、频率/steal-time telemetry、异常 run 的预注册剔除规则。
* **reuse 泄漏未来 RHS**：stream controller 保持 request-response barrier；adapter 只能缓存 L-dependent state。

## 6. 执行计划与预算

1. **1–2 人日**：核对并下载 standard matrices，冻结 collection revision/provenance；实现 deterministic conversion、PSD/CSR validator。
2. **2–3 人日 + 2–4 CPU-hours**：生成 8 public/16 private mothers 的 metadata-only pilot，检查 n/nnz/eta/RHS 分层；不生成最终 truth。
3. **3–5 人日**：完成 Higham/Octave 与 compiled Chebyshev adapters；每个约 0.5–1.5 日。复核现有 Expokit/KIOPS。
4. **3–7 人日 + build 1–4 CPU-hours**：构建 SLEPc/PETSc frozen artifact；准备 Julia runtime/project/artifacts。若 immutable image 无法容纳，记录 exclusion。
5. **3–5 人日**：实现/审计 rational Krylov；phipm license/source 联络可并行，等待时间不计工程工时。
6. **2–4 人日 + 10–30 CPU-hours**：在 public data 执行预注册 calibration grids、accuracy margin、largest-case resource gate；冻结 panel hashes。
7. **2–3 人日 + 40–140 CPU-hours**：生成 primary truth 与独立 audit；并行机器墙钟约 6–18 小时。存储 2–4 GB，峰值单 worker <4 GiB。
8. **1–2 人日 + 30–100 CPU-hours/panel sweep**：7 repeats interleaved timing，估算以 panel 5–9 members、104 sessions 为准；先做 1-repeat dry run，任何 session 超预算立即修数据或声明 member failure。
9. **1–2 人日**：独立复核 manifest、split leakage、hash、license、score/bootstrap implementation；发布 aggregate distribution 与 calibration，不发布 private NPZ。

最小可发布路径约 3 周工程时间；包含 SLEPc/Julia/rational 与 license 清理的完整 panel 约 4–6 周。后续重计算需要的是普通单 CPU workers，不需要 GPU；truth 和 panel sweep 适合跨独立 cores 并行，但每个 session 始终单核。
