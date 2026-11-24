# 从估计基因树推断物种树：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

每例输入 50 棵估计基因树（250 个叶，由各 200 bp 的序列估计，带枝长和近似贝叶斯支持度），数据由多物种溯祖模型（MSC）生成：物种树 → 各基因座的真实谱系 → 序列 → 估计基因树。输出一棵无根二叉物种树，误差为归一化 Robinson–Foulds 距离，96 例平均。

统计上，这是 MSC 加基因树估计误差（GTEE）下的物种树估计：基因数 \(m=50\)，序列长度 \(k=200\)（短，所以 GTEE 高），物种数 \(n=250\)。

## 2. 已知最好的结果（上界）

- **统计一致性**：基于四元组的方法（ASTRAL：Mirarab 等 2014；ASTRAL-III：Zhang 等 2018）在真基因树下统计一致；基于距离的 NJst / ASTRID（Liu, Yu 2011；Vachaspati, Warnow 2015）也一致。按基因树不确定性加权四元组（Zhang, Mirarab 2022；TREE-QMC 的加权版本，Han, Molloy 2023 ✓）在高 GTEE 下更准。本题强参照就是加权 TREE-QMC。
- **样本复杂度**：精确最大化四元组支持的 ASTRAL\(^*\) 在无误差基因树下，只要
  \[
  m>\tfrac92\log\!\Big(\tfrac{4\binom n4}{\epsilon}\Big)\frac{1}{(1-e^{-f})^2},
  \]
  就以至少 \(1-\epsilon\) 的概率恢复真树，即 \(m=O(f^{-2}\log n)\)，\(f\) 是最短枝长（溯祖单位）（Shekhar, Roch, Mirarab 2018 ✓）。
- **序列层面**：基于距离的方法要检测长度为 \(f\) 的枝，需要且只需要 \(m=\Theta\big(1/(f^2\sqrt k)\big)\) 个基因（Mossel, Roch 2017 ✓）。这一结果直接刻画了本题“多基因 × 短序列”的权衡：\(k\) 越大，单个基因树越准，所需基因数越少。
- **计算**：ASTRAL 在候选 clade 集构成的约束搜索空间内用动态规划精确求解；TREE-QMC 用加权 Quartet Max Cut 分治，时间复杂度 \(O(n^3m)\)（\(m\) 为基因树数，假设分解平衡）。

## 3. 下界与不可能性

- **方法特定下界**：存在物种树，使 ASTRAL\(^*\) 在 \(m\le\frac a5\log n/f^2\) 时以接近 1 的概率出错（Shekhar 等 2018 ✓），所以 ASTRAL\(^*\) 的样本复杂度阶是紧的。
- **信息论下界**（基于距离的方法）：\(m=\Omega\big(1/(f^2\sqrt k)\big)\)（Mossel, Roch 2017 ✓）；\(k=O(1)\) 时退化为 \(\Omega(f^{-2})\)。
- **不一致性**：存在基因树估计误差时，某些汇总方法和分块似然可能统计不一致（Roch, Warnow 2015；Roch, Nute, Warnow 2019 的长枝吸引）。在异常区中，最可能出现的基因树拓扑不是物种树（Degnan, Rosenberg 2006），所以“取多数基因树”不可行。
- **计算复杂性**：加权四元组共识（找一棵与输入基因树共享四元组最多的二叉树）是 NP-hard 的，已知有 1/2 近似（Lafond, Scornavacca 2019 ✓）。精确优化不可期，只能做近似、启发式，或在约束空间内精确求解。

**是否紧**：无误差基因树下 ASTRAL\(^*\) 的 \(f^{-2}\log n\) 紧，距离类方法的 \(1/(f^2\sqrt k)\) 紧；带 GTEE、并利用支持度和枝长的设定（即本题）没有匹配的上下界。

## 4. 开放问题：什么样的结果算贡献

- 带基因树估计误差时的样本复杂度，以及最优加权方案的理论：加权四元组为什么更准，最优权重是什么。
- \(m=50\)、\(k=200\)、\(n=250\) 有限样本下的准确性保证（nRF 的期望界）。
- 近似算法：加权四元组共识优于 1/2 的近似比，或在受限结构上的 PTAS。

## 5. 与本题参照、已有成绩的关系

- 弱参照：不加权 TREE-QMC（只用基因树拓扑）；强参照：加权 TREE-QMC（按支持度和枝长加权四元组）。正式集上 \(U=0.1476\)、\(W=0.1320\)。
- v6.3 成绩：GPT-6 Astra 单轮 \(E=0.1196<W\)，按 \(100-50E/W\) 得 54.7 分，是该格最好成绩；Hunyuan 4 单轮 \(E=0.1574\)，比弱参照还差，记 0；Opus 单轮用加权四元组方法，\(E=0.1389\)（28.1 分）。

## 6. 理论赛道建议

**适配度：中高。** 一致性、样本复杂度、信息论下界、NP-hardness 都有现成理论，可以清楚地区分“复现已知结果”和“新结果”。缺点是本题的 GTEE 设定恰好落在理论空白处，题面需要补模型才能让定理良定义。

**题面需补的数学前提**：GTEE 模型或支持度的语义（近似贝叶斯支持度是否校准），以及参数范围（最短枝长 \(f\)、树高）。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 所用估计器在 MSC + 真基因树下的统计一致性（四元组类或距离类） |
| 增量 | 给定 GTEE 模型（例如每条内部边以与支持度相关的概率被错误解析）下的一致性与样本复杂度；加权方案的一致性；所用搜索在约束空间内的精确性或近似比 |
| 前沿 | 带 GTEE 的匹配下界；最优权重 |

**可自动检验的部分**：nRF 已由评测器计算。可额外要求输出候选空间大小与目标值，以检验“在约束空间内精确”的声明。

**形式化可行性**：树的有限组合部分（例如“分裂集唯一决定无根树”、四元组相容性）可以在 Lean 4 中形式化；统计一致性需要先形式化 MSC，超出现状。

## 参考文献

1. J. H. Degnan, N. A. Rosenberg, Discordance of species trees with their most likely gene trees, PLoS Genet. 2 (2006) e68。
2. S. Mirarab 等, ASTRAL: genome-scale coalescent-based species tree estimation, Bioinformatics 30 (2014) i541–i548。
3. C. Zhang, M. Rabiee, E. Sayyari, S. Mirarab, ASTRAL-III: polynomial time species tree reconstruction from partially resolved gene trees, BMC Bioinformatics 19 (2018) 153。
4. L. Liu, L. Yu, Estimating species trees from unrooted gene trees, Syst. Biol. 60 (2011) 661–667。
5. P. Vachaspati, T. Warnow, ASTRID: accurate species trees from internode distances, BMC Genomics 16 (2015) S3。
6. C. Zhang, S. Mirarab, Weighting by gene tree uncertainty improves accuracy of quartet-based species trees, Mol. Biol. Evol. 39 (2022) msac215。
7. ✓ Y. Han, E. K. Molloy, Improving quartet graph construction for scalable and accurate species tree estimation from gene trees, Genome Res. 33 (2023) 1042–1052. <https://doi.org/10.1101/gr.277629.122>。TREE-QMC（本题两条参照）。
8. ✓ S. Shekhar, S. Roch, S. Mirarab, Species tree estimation using ASTRAL: how many genes are enough?, IEEE/ACM Trans. Comput. Biol. Bioinform. 15 (2018) 1738–1747. <https://arxiv.org/abs/1704.06831>。
9. ✓ E. Mossel, S. Roch, Distance-based species tree estimation under the coalescent: information-theoretic trade-off between number of loci and sequence length, Ann. Appl. Probab. 27 (2017) 2926–2955. <https://doi.org/10.1214/16-AAP1273>。
10. S. Roch, T. Warnow, On the robustness to gene tree estimation error (or lack thereof) of coalescent-based species tree methods, Syst. Biol. 64 (2015) 663–676。
11. S. Roch, M. Nute, T. Warnow, Long-branch attraction in species tree estimation: inconsistency of partitioned likelihood and topology-based summary methods, Syst. Biol. 68 (2019) 281–297。
12. ✓ M. Lafond, C. Scornavacca, On the weighted quartet consensus problem, Theoret. Comput. Sci. 769 (2019) 1–17；CPM 2017. <https://doi.org/10.1016/j.tcs.2018.10.005>。NP-hardness 与 1/2 近似。
