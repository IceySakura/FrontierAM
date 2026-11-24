# 偏振马赛克重建：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

焦平面分割（DoFP）偏振相机的 2×2 微偏振片阵列（90°、45°、135°、0°）给出单通道马赛克，要求重建四个方向的全分辨率强度图。理想线性偏振片满足 Malus 定律带来的约束 \(I_0+I_{90}=I_{45}+I_{135}=S_0\)。指标是 132 个真实场景（Qiu32、Wen100）的平均 RMSE，每景 1 秒。

## 2. 已知最好的结果（上界）

- **频域分析**：马赛克可以写成强度基带加若干偏振载波（调制到 \((\pi,0)\)、\((0,\pi)\)、\((\pi,\pi)\) 等频率）。若 Stokes 图在相应频带内带限，可以在线性系统框架下理想重建，完全消除瞬时视场（IFOV）误差（Tyo, LaCasse, Ratliff 2009 ✓；插值策略见 Ratliff, LaCasse, Tyo 2009）。这与 Bayer 去马赛克的亮度-色度复用理论同构（Alleysson, Süsstrunk, Hérault 2005；综述 Gunturk 等 2005）。
- **平稳高斯先验**下，线性最小均方误差（Wiener）去马赛克是贝叶斯最优的线性估计（标准结论）。
- **残差插值 / 边缘感知方法**：EARI（Morimatsu, Monno, Tanaka, Okutomi，ICIP 2020 ✓）用四方向边缘感知引导图在残差域插值；综述见 Mihoubi, Lapray, Bigué（2018）。学习方法是经验上的 SOTA。

## 3. 下界与不可能性

- **混叠**：非带限内容在 2× 子采样下与其他频率成分不可区分，所以不存在对所有图像都精确的重建，这是采样定理意义下的不可能性。
- 对自然图像类没有公认的极小极大理论，真实场景上的 RMSE 下界无从给出。

**是否紧**：只有在带限模型或平稳高斯模型下有最优性结论；对真实数据集没有。

## 4. 开放问题

- 自然图像统计模型（稀疏、分段光滑，加上偏振结构）下的极小极大去马赛克理论。
- 重建误差传到偏振度（DoLP）与偏振角（AoP）上的放大分析。

## 5. 与本题参照、已有成绩的关系

- 弱参照逐方向双线性（\(B=2.108\)），强参照为出题人的镜像延拓 LEIC（\(H_5=1.501\)）。
- v6.3 GPT-6.2 Sol 单轮 57.0 分；Opus 在 v6.2 用频域解调加学习滤波（55.9 分）；Sol 用约 7.7k 参数的小卷积网络，只用 2 张公开场景训练。`results/INSIGHTS.md` 判断提升来自组合还是参照偏弱，需要组件消融。

## 6. 理论赛道建议

**适配度：低。** 评价完全依赖真实图像集上的经验误差，现有理论（带限、平稳高斯）与数据分布不匹配，模型给出的“定理”难以与分数建立联系。建议保持为纯性能题，或只要求方法的频域 / 线性系统分析（教科书级）。

**可要求的定理（若保留）**：带限模型下所用线性重建算子的精确重建条件；平稳高斯先验下滤波器的 LMMSE 最优性。

**形式化可行性**：带限模型下的精确重建可以表述为有限维线性代数命题（周期网格上的 DFT），能够机器检查，但与分数无关。

## 参考文献

1. ✓ J. S. Tyo, C. F. LaCasse, B. M. Ratliff, Total elimination of sampling errors in polarization imagery obtained with integrated microgrid polarimeters, Opt. Lett. 34 (2009) 3187–3189. <https://doi.org/10.1364/OL.34.003187>。
2. B. M. Ratliff, C. F. LaCasse, J. S. Tyo, Interpolation strategies for reducing IFOV artifacts in microgrid polarimeter imagery, Opt. Express 17 (2009) 9112–9125。
3. D. Alleysson, S. Süsstrunk, J. Hérault, Linear demosaicing inspired by the human visual system, IEEE Trans. Image Process. 14 (2005) 439–449。
4. B. K. Gunturk, J. Glotzbach, Y. Altunbasak, R. W. Schafer, R. M. Mersereau, Demosaicking: color filter array interpolation, IEEE Signal Process. Mag. 22 (2005) 44–54。
5. S. Mihoubi, P.-J. Lapray, L. Bigué, Survey of demosaicking methods for polarization filter array images, Sensors 18 (2018) 3688。
6. ✓ M. Morimatsu, Y. Monno, M. Tanaka, M. Okutomi, Monochrome and color polarization demosaicking using edge-aware residual interpolation, ICIP 2020, 2571–2575. <https://arxiv.org/abs/2007.14292>。
