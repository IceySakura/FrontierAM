# decentralized-sum-of-nonconvex theory-v1：设计（2026-11-27）

## 目的

用户 11-27 提出：FrontierAM 的核心是评估模型在算法设计中权衡实验与理论，但至少要有一两道只看理论推进的题，看模型不做实验、只推进理论时能把人类 SOTA 刷新到什么程度。本题是其中第二道（第一道是 `blackbox-cubature-theory-v1`，见 `tasks/blackbox-cubature/analysis/DUAL_PLAN.md` 9.15），换到优化复杂度理论，问题本身是文献中的开放问题。

## 问题与来源

OptBound 基准（<https://optbound.reaslab.io/>）的 ULB-profile-63：去中心化有限和、逐分量 \(L\)-光滑（分量可非凸）、\(\phi=f+h\) 为 \(\mu\)-强凸的复合优化，静态网络 \(\mathcal W_{n,\rho}\)，每迭代至多一次分量梯度、一次 prox、一轮通信，要求给出迭代复杂度的匹配上下界。题面原文放进工作区 `PROFILE.md`。

## 人类 SOTA（11-27 读原文核对）

| 方向 | 结果 | 核对 |
| --- | --- | --- |
| 上界 | Liu–Low（AAAI 2024，arXiv:2402.02356v1）定理 5.1、推论 5.3，PMGT-KatyushaX：每节点 \(O((m+m^{3/4}\sqrt\kappa)\log)\) 次分量梯度（\(\ell_1=\ell_2=L\)），\(\tilde O((\sqrt m+m^{1/4}\sqrt\kappa)/\sqrt{1-\lambda_2}\,\log)\) 轮通信；原文 \(W\) 假设 \(0\preceq W\preceq I\)，与本题不同 | arXiv 源码与 HTML 版编号 |
| 下界（可搬用） | Woodworth–Srebro（NeurIPS 2016）定理 8：随机化算法、光滑强凸分量（预言机还含分量 prox），\(\Omega(m'+\sqrt{m'\gamma/\lambda}\log)\) | PDF 编号 |
| 下界（可搬用） | Scaman 等（ICML 2017）定理 2：谱隙 \(\gamma\) 的网络下界，证明中节点数依 \(\gamma\) 选取（\(\gamma_n=1-\cos(\pi/n)\)） | PDF 编号 |
| 下界（类更大） | Zhou–Gu（ICML 2019）定理 4.1：sum-of-nonconvex 强凸 \(\Omega(n+n^{3/4}\sqrt{L/\sigma}\log)\)，只假设平均光滑；第 4.3 节指出上界用的是逐分量光滑，结论一节把逐分量光滑列为未来工作 | PDF 编号、源码 |
| 现状 | Zhou（arXiv:2611.05807，2026-11-04）在相近问题（非凸有限和的近似局部极小）上写明 "Under individual smoothness, a factor of \(n^{1/4}\) remains" | 摘要页 |

OptBound 表第 63 行：上界记 Liu–Low，下界记未知。按以上文献，逐分量光滑下数据项的 \(N^{1/4}\) 差距是开放的；题面写明这是我们核对的结论，模型检索到更强的已有结果时应写明并据此分类。

## 形式化模型（固定）

- `Theory/Model.lean`、`Theory/Sanity.lean` 取自理论试点 ULB-63 的陈述层（`frontieram-theory-pilot/optbound/ulb-63/lean/ULB63/`），只有定义与定义自检，不含任何结论；只改了文件头注释和 import 路径。试点的 `Notation.lean`（θ_N、χ、𝓑）与 `Statements.lean` 是试点解答的结论形状，不提供。
- 评测：`scripts/grade/lean-comparator.py --fixed Theory/Model.lean=… --fixed Theory/Sanity.lean=…` 用原件替换提交中的副本，报告记录提交是否改过；工作区 `tools/lean_check.py` 在构建前比对，不一致即报错。宿主机上把模板当提交检查：143 秒通过，故意改坏的 `Sanity.lean` 被替换并记为 “changed (replaced)”。
- `MODEL_NOTES.md`：试点 `FIDELITY.md` 第 1 节（条款 ↔ 定义）原样，第 2 节（解释选择）去掉所有涉及试点解答的内容（上界协议、汇聚节点、计数论证、\(\min\{n,\gamma^{-1/2}\}\) 的形状）。

## 防泄露

- 试点的解答报告与证明只在本机 `frontieram-theory-pilot` 目录，开发容器看不到。
- 代理新增 `benchmark` 类：主机名含 `optbound` 的一律拒绝（`arena/netproxy.py`，单元测试已加）；GitHub 仍整域拒绝。题面写明不得查找 OptBound 或 FrontierAM 的相关材料。

## 计分（新题定义映射）

只有理论分，按分档评分（用户 11-27 确定，`tasks/blackbox-cubature/analysis/THEORY_REVIEW.md` 第 2 节）：结论须在 manifest 中标为 `proved`、通过 `lake comparator`、且 Lean 陈述与 `THEORY.md` 的结论匹配；然后在“假设 × 结论强度”的偏序中与预先登记的锚点比较（`THEORY_ANCHORS.md`）：严格强于某个锚点且不被已知人类结果支配为 100；不可比且满足登记的价值偏好为 50，否则 25；被已知人类结果支配或与之等同为 10；没有通过检查为 0。最终分取各条最高。题面（11-26 起草）写的是对应的四类分类（严格更强、严格更弱、相当、不可比）。评审校准：理论试点曾在同一固定模型上形式化过一份解答（只在本机试点目录，不进本仓库），可用来核对模型结论的强弱。

## 环境与运行

- 单轮设定，6 小时，验证运行（不上榜）：`scripts/rounds/theory-trial.sh decentralized-sum-of-nonconvex-theory-v1 MODEL N`。
- 开发容器：8 GiB（8.5 GiB 租约），离线 Lean v4.35 + Mathlib/CSLib（只读挂载），公网经 `arena/netproxy.py`，厂商侧联网工具关闭；无样例服务。镜像复用 `frontieram-blackbox-cubature-v3-harness:v6`（含 git、curl、socat）。
- 封装：`scripts/seal/seal_theory.py tasks/decentralized-sum-of-nonconvex/versions/theory-v1`。

## theory-v2（11-27 晚）

借鉴 DeMuon 题（`tasks/blackbox-cubature/analysis/DUAL_PLAN.md` 9.17）。题面公开登记的 5 个锚点（H-UB、H-LB-data、H-LB-pl、H-LB-net、H-LB-avg），写明两个已封死的方向（\(\kappa\le\sqrt N\) 的数据项下界、网络项下界），新增目标 D（证明某个登记的界不能再改进、或某个假设不能再弱）。计分为 \(\max(T,R)\)：\(T\) 为前沿分档；\(R\) 为六行的分行得分，每行改进或证明最优得满分、相当或公认权衡得一半、其余为 0，上界行与资源行必须来自同一个算法与调参。只有复述时 \(T=R=10\)。网络项下界按试点的上界在 \(\log n\) 以内已紧，这一点来自未发表的结果，题面不写，以免暗示上界的形状。
