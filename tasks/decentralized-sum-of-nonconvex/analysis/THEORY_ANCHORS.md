# decentralized-sum-of-nonconvex：人类 SOTA 锚点与价值偏好（11-27 预先登记）

用于 `tasks/blackbox-cubature/analysis/THEORY_REVIEW.md` 第 2 节的分档评分，在评审任何试跑之前登记。出处都读过原文（定理编号按 arXiv 版或 PDF 核对）；**换算到本题模型的部分由我完成，需要专家确认**。

本题模型：`PROFILE.md` 第 1 节与 `Theory/Model.lean`。迭代计数为每个节点每次迭代至多一次分量梯度、一次 prox、一轮通信。参数：节点数 \(n\)，每节点分量数 \(m\)，\(N=nm\)，\(\kappa=L/\mu\)，\(\rho\)，\(R=\Delta/\varepsilon\)。

**界的比较**：按阶比较，忽略常数因子；锚点中写作 \(\tilde O\) 的项忽略多对数因子。只在部分参数范围内更好、在其他范围更差，算作不可比。

## 坐标

上界（存在 \(\mathcal A\) 中的算法，使类中所有实例 \(T_\varepsilon\le B\)）：

| 坐标 | 含义 | 序（越往右，定理越强） |
| --- | --- | --- |
| U1 适用的类 | 函数、预言机、通信类 | `PROFILE.md` 第 3 节的 (+) 限制（\(W\succeq0\)、给定拓扑、凸分量、强凸 \(f\)、相同初始局部梯度）→ 标准类 → (−) 放宽（只要求局部平均光滑、受控的不精确 prox、去掉定量收缩界） |
| U2 算法 | 见证算法所在的类 | 放宽算法类（如不要求 zero-respecting）→ \(\mathcal A\) → 加限制（如分量下标按固定顺序选取） |
| U3 界 \(B\) | 迭代复杂度 | 逐点越小越强 |

下界（对 \(\mathcal A\) 中每个算法，存在类中实例使 \(T_\varepsilon\ge B\)）：

| 坐标 | 含义 | 序（越往右，定理越强） |
| --- | --- | --- |
| L1 困难实例所在的类 | 越小越强 | 比标准类大的类（如只要求分量平均光滑：困难实例可能不在本题的类中）→ 标准类 → 标准类内的 (+) 限制（凸分量、\(W\succeq0\)、给定拓扑） |
| L2 覆盖的算法类 | 越大越强 | linear-span 算法 → \(\mathcal A\)（随机化、zero-respecting）→ 一般随机算法 |
| L3 界 \(B\) | 在锚点的参数范围内 | 逐点越大越强 |

## 锚点

| 编号 | 类型 | 出处（已核对） | 换算到本题模型 | U1 / L1 | U2 / L2 |
| --- | --- | --- | --- | --- | --- |
| H-UB | 上界 | Liu–Low（AAAI 2024，arXiv:2402.02356v1）定理 5.1 与推论 5.3（PMGT-KatyushaX）。原文：\(m\) 个节点、每个节点 \(n\) 个 \((\ell_1,\ell_2)\)-光滑分量，\(F\) 为 \(\sigma\)-强凸，\(W\) 对称双随机且 \(0\preceq W\preceq I\)；SFO 调用 \(O((n+n^{3/4}(L^{1/2}+(\ell_1\ell_2)^{1/4})/\sqrt\sigma)\log)\)，通信 \(\tilde O((\sqrt n+n^{1/4}(\cdot)/\sqrt\sigma)/\sqrt{1-\lambda_2(W)}\,\log)\) | \(T=\tilde O\big((m+m^{3/4}\sqrt\kappa+(\sqrt m+m^{1/4}\sqrt\kappa)(1-\rho)^{-1/2})\log R\big)\)。换算：\(\ell_1=\ell_2=L\)；非半正定的 \(W\) 用 \((I+W)/2\)；梯度调用与通信在同一迭代内进行 | 标准类（换算后） | \(\mathcal A\) |
| H-LB-data | 下界 | Woodworth–Srebro（NeurIPS 2016，arXiv:1605.08003）定理 8：任意随机算法（每次查询得到分量的值、梯度和 prox），\(m'\) 个 \(\gamma\)-光滑、\(\lambda\)-强凸分量，\(\gamma/\lambda>161m'\)，需要 \(\Omega(m'+\sqrt{m'\gamma/\lambda}\,\log(\frac{\epsilon_0}{\epsilon}\sqrt{m'\lambda/\gamma}))\) 次查询 | 集中式地模拟去中心化算法，全部节点在 \(T\) 次迭代中至多 \(nT\) 次分量查询：\(\kappa>161N\) 时 \(T\ge c\,(m+\sqrt{m\kappa/n}\,\log(R\sqrt{N/\kappa}))\) | 凸分量（标准类内的 (+) 限制） | 一般随机算法 |
| H-LB-net | 下界 | Scaman–Bach–Bubeck–Lee–Massoulié（ICML 2017，arXiv:1702.08704）定理 1（任意直径为 \(D\) 的图、黑盒方法，误差界含指数 \(t/(1+D\tau)\)）与定理 2（gossip、谱隙 \(\gamma\)，证明中节点数依 \(\gamma\) 选取） | 固定 \(n\)、\(W\in\mathcal W_{n,\rho}\) 时：\(T\ge c\,\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\,\log R\)。按两条定理合并的标准换算：对手在 \(\mathcal W_{n,\rho}\) 中取直径约为 \(\min\{n,(1-\rho)^{-1/2}\}\) 的慢混合矩阵 | 强凸二次函数（标准类内） | 黑盒方法（含 \(\mathcal A\)） |
| H-LB-avg | 下界 | Zhou–Gu（ICML 2019，arXiv:1901.11224）定理 4.1：sum-of-nonconvex、\(F\) 为 \(\sigma\)-强凸、分量只要求平均光滑，linear-span 随机算法，\(\Omega(n'+n'^{3/4}\sqrt{L/\sigma}\log\frac{n'\sigma\Delta}{L\epsilon})\) 次查询 | 全部 \(N\) 个分量、集中式模拟：\(T\ge c\,(m+m^{3/4}n^{-1/4}\sqrt\kappa\,\log(\cdot))\) | 只平均光滑（比标准类大） | linear-span |
| H-LB-pl | 下界 | Peng–Tang–Jia（arXiv:2609.00045，2026-08-30 首发，v3 为 2026-10-10）：逐分量光滑，随机化 IFO 算法（可任意选择分量与查询点）。\(3\le\kappa\le\sqrt n\) 时 \(\Omega(n+n\log(\Delta/\varepsilon)/[1+\log(\sqrt n/\kappa)])\)，困难实例的平均目标全局强凸、分量非凸；\(\kappa\ge\sqrt n\) 时 \(\Omega(n+\kappa\sqrt n\log(\Delta/\varepsilon))\)，但困难实例只满足 PL，不在本题的类中 | 全部 \(N\) 个分量、集中式模拟：\(3\le\kappa\le\sqrt N\) 时 \(T\ge c\,(m+m\log R/[1+\log(\sqrt N/\kappa)])\) | 标准类内（平均强凸、逐分量光滑、\(h=0\)） | 随机化 IFO（含 \(\mathcal A\)） |

H-LB-data、H-LB-pl、H-LB-avg 两两互不支配（参数范围、类或算法类各不相同）。五个锚点构成目前登记的人类前沿。H-LB-net 的固定 \(n\) 形式没有逐字出现在原文中，登记为由原文标准换算得到的已知结果：模型证明这一项属于复述。H-LB-pl 是 11-27 下午核查时补登的（评审任何试跑之前），见文末“提升空间核查”。

## 分档举例（预先说明）

- 在标准类、\(\mathcal A\) 上证明 H-LB-data、H-LB-pl 或 H-LB-net 的界，包括复述其论证：**偏序不优**。例如 \(\kappa\le\sqrt N\) 区间的 \(m+m\log R/[1+\log(\sqrt N/\kappa)]\) 型数据项下界，经集中式模拟即由 H-LB-pl 得到。
- 在标准类、\(\mathcal A\) 上证明一个数据项下界，处处不弱于 H-LB-data，且在某个参数范围内严格更强（例如 \(\kappa\ge\sqrt N\) 时达到 \(m^{3/4}n^{-1/4}\sqrt\kappa\log R\)）：严格支配 H-LB-data，也支配 H-LB-avg（类更小、算法类更大、界不弱）→ **偏序优**。只在部分范围更强、别处更弱：**不可比**，按 P1 认定有价值。
- 上界在标准类上处处不差于 H-UB（多对数因子以内），且在某个参数范围严格更好：**偏序优**。只改进通信项而数据项变差：**不可比**，按 P2 认定有价值。
- 只在 (+) 限制下成立、界与 H-UB 相同的上界：被 H-UB 支配 → **偏序不优**。
- 只在 (+) 限制下成立的下界（如凸分量）：L1 更强，与锚点比较时还要求界不弱。

## 价值偏好（不可比一档的审查依据，取自人类论文原文）

- **P1 逐分量光滑下的数据项下界**：Zhou–Gu 2019 第 4.3 节指出 KatyushaX 等上界是在逐分量光滑下证明的，而其下界只假设平均光滑；结论一节把“\(F\) 为 \((\sigma,L)\)-光滑而每个 \(f_i\) 为 \((l,L)\)-光滑”的情形列为未来工作。Zhou 2026（arXiv:2611.05807）摘要在相近问题上写道 "Under individual smoothness, a factor of \(n^{1/4}\) remains"。在逐分量光滑下缩小数据项 \(N^{1/4}\) 差距的可证明进展有价值。
- **P2 通信项**：Liu–Low 2024 结论："the communication upper bound still looks unsatisfying due to the inclusion of the factor \(\sqrt n\)"（原文的 \(n\) 即本题的 \(m\)），"It is interesting to study how to design a more communication-efficient algorithm for decentralized sum-of-nonconvex optimization"。可证明地改进通信项的结果有价值。
- **P3 匹配与标准问题**：OptBound 题面（`PROFILE.md` 第 2 节）要求保留全部参数依赖的匹配上下界；第 3 节："Only the standard problem receives full credit. Results under explicit alternatives may receive partial credit"。在标准问题的某个参数范围内给出匹配上下界有价值；只在替代设定下成立、且不回应 P1、P2 的结果价值低。
- Woodworth–Srebro 2016 结论指出，要改进需要额外假设或更强的预言机；Scaman 等结论列出的扩展方向（时变网络、非强凸、时延）不在本题设定内。二者都不作为本题的价值偏好。

## 提升空间核查（11-27 下午，评审前）

问题：有没有人已经证明了匹配的界，使某个方向上根本不可能严格超过人类结果？检索到 2026-11（关键词覆盖去中心化 sum-of-nonconvex、逐分量光滑的有限和下界；读了相关论文的摘要与引言）。

| 方向 | 人类前沿 | 是否已被证明最优 | 提升空间 |
| --- | --- | --- | --- |
| 上界（数据项与通信项） | H-UB（Liu–Low 2024）。没有检索到更强的已发表上界（2025 年 IJRNC 的梯度跟踪结果依赖 \(\kappa^2\)，更弱） | 否，没有匹配下界。理论试点在同一 Lean 模型中形式化了一个处处不差、\(n\) 大时严格更好的上界（数据项 \(m^{3/4}n^{-1/4}\sqrt\kappa\)，通信项约 \(\min\{n,\log n/\sqrt{1-\rho}\}\sqrt\kappa\)，均乘对数因子；未发表，模型层的忠实性待专家确认） | **有，并已知可达** |
| 数据项下界，\(\kappa\le\sqrt N\) | H-LB-pl（2026-09，经集中式模拟） | 是（只差对数因子）：集中式由 Restarted PAGE 匹配（同一论文，均方光滑下）；去中心化由试点的上界匹配 | 没有 |
| 数据项下界，\(\kappa\ge\sqrt N\) | H-LB-data（\(\sqrt{m\kappa/n}\) 型）；H-LB-avg（只平均光滑） | 否，开放：H-LB-pl 在这一区间只覆盖 PL 类；Zhou 2026 在相近问题上写明仍差 \(n^{1/4}\) | **有**：试点的树形构造在同一模型中形式化了 \(m^{3/4}n^{-1/4}\sqrt\kappa\log R\) 型下界 |
| 网络项下界 | H-LB-net | 与已发表的上界不匹配；但按试点的上界，\(\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log R\) 在 \(\log n\) 因子以内已紧 | 只剩对数因子 |

**可达性**：最小的偏序优结论是 \(\kappa\ge\sqrt N\) 区间的数据项下界（试点中约 2,700 行 Lean，含共用的运行与二次型基础设施），其次是改进 H-UB 的上界（约 9,500 行）。试点中树形下界由一个子 agent 在给定数学证明与陈述后几小时内形式化完成，所以 6 小时内形式化这一块是现实的；真正的门槛是自己找到构造，而文献到 2026-11 仍把它列为开放。相邻问题上已经有 AI 系统给出这类构造（arXiv:2610.16991，非凸–凹有限和在逐分量光滑下的 \(n^{3/4}\) 型下界，作者声明研究流程几乎全部由基于 GPT-5.6 Sol 的系统完成）。
