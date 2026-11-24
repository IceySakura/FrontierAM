# 黑盒积分改造为理论-实验双考题：调研与方案（草案，2026-11-24）

背景见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md) 第 5 节：理论赛道只收按标准遴选的双考题，统一结构为“声明的问题类、按 oracle 计价、困难区间实例加真实实例”。本文回答黑盒积分怎么改。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；试点仓库 `frontieram-theory-pilot/exploration/gaussian-kinks-survey.md` 的初步调研是本方案的出发点。

## 0. 结论

- **现状（v2）已无理论空间**：三族都是低维参数化结构，识别后积分可精确算出。顶尖模型迭代已接近饱和：GPT-6.2 Sol 迭代 \(G=1.734\)（乘积峰 +2.00，已到截断上限；折点 ridge +1.86；fourier +1.34），按 \(50+25G\) 得 93.4 分。对这些提交做理论分析，只能复述“可分离用张量 Gauss、单方向 ridge 化为一维积分”这类已知结论。
- **改造方向**：v3 改为高斯测度下的分段光滑被积函数。折点或跳跃位于方向未知、可能不单调的超曲面上，只能点查询，按查询次数计价。
- **为什么选它**：
  - 理论缺口有文献作者自己写明：单调性失效时预积分“at present no theoretical support”；预积分方向的选择只有启发式；黑盒下的折点定位没有理论；多维未知超曲面上没有锐利的复杂度结果。
  - 实验侧有真实实例（多资产期权收益，部分有闭式或半解析参考值）和成熟的强参照（主动子空间预积分、数值光滑化）。
- **双考结构**：困难区间实例（合成、专门落在理论缺口上）与真实实例（期权）分两条赛道计分；题面附理论附录（函数类、代价模型、已知结果、开放问题）；程序可选输出误差界，评测端核对覆盖率。
- **工作量与风险**：约 6 个工作日。最大风险是实验余量：要先用原型确认强参照在困难族上离可达精度还有一到两个数量级，否则实验区分不出理论改进。

## 1. 现状诊断（v2）

| 族 | 形式 | 为什么能精确识别 |
| --- | --- | --- |
| fourier ridge | \(\exp(\kappa\cos(2\pi k\cdot x+\varphi))\) | 单方向脊函数；整数 \(k\) 时在单位立方体上的积分恒为 \(I_0(\kappa)\)，与 \(k,\varphi\) 无关（`AUTHOR_NOTE.md` 第 3 节） |
| 乘积峰 | 各坐标因子之积（Genz product peak） | 可分离，按坐标做一维 Gauss 积分即精确 |
| 折点 ridge 叠加 | \(1+\sum_{j\le3}c_j\max(0,v_j\cdot(x-u_j))^{p_j}\) | 分段多项式，折点在超平面上；定位折面（梯度跳变）、拟合幂次后用截断幂公式精确积分 |

各模型迭代的分族增益（\(\log_{10}\) 相对误差对 HumanPortfolio-v1 的配对增益，截到 \(\pm2\)）：

| 模型 | fourier | 折点 ridge | 乘积峰 | \(G\) | 分数 |
| --- | --- | --- | --- | --- | --- |
| GPT-6.2 Sol | +1.34 | +1.86 | +2.00 | 1.734 | 93.4 |
| GPT-6.1 Sol | +0.88 | +1.60 | +2.00 | 1.496 | 87.4 |
| GPT-6 Astra | +1.23 | +0.84 | +2.00 | 1.357 | 83.9 |
| Claude Opus 5.5 | +1.56 | +0.51 | +2.00 | 1.356 | 83.9 |
| Hunyuan 4 | +0.81 | −0.90 | +2.00 | 0.640 | 66.0 |

分数差别来自谁更彻底地识别了参数化结构。理论上，这三族在识别之后都是已知结论，与理论空白无关。v2 保留为纯实验题，榜单不变。

## 2. 理论调研

### 2.1 光滑类：经典结果已紧

- \(C^r\) 类上确定性最优 \(n^{-r/d}\)、随机化 \(n^{-r/d-1/2}\)（Bakhvalov）；维数灾难（Hinrichs, Novak, Ullrich, Woźniakowski 2014 ✓）。
- 高斯测度：高斯核 RKHS 中的解析函数，张量 Gauss–Hermite 只需 \(O((\ln\varepsilon^{-1})^d)\) 个点，且指数 \(d\) 是紧的（Kuo, Sloan, Woźniakowski 2017 ✓）。
- 光滑函数上 Owen 扰乱网的方差为 \(O(n^{-3}(\log n)^{d-1})\)（Owen 1997）。

结论：光滑部分没有空间，理论空间全在非光滑部分。

### 2.2 非光滑：跳跃与折点

- **一维**：只有一个未知奇异点时，自适应求积的最坏误差为 \(n^{-r}\)，任何非自适应方法不快于 \(n^{-1}\)；多个奇异点只有渐近结果（Plaskota, Wasilkowski 2005 ✓；逼近问题见 Plaskota, Wasilkowski, Zhao 2008 ✓）。
- **多维、等权 QMC**：半空间偏差是紧的，\(\Theta(n^{1/2-1/(2d)})\)（计数意义；Alexander 1990 ✓ 下界，Matoušek 1995 ✓ 上界）。所以等权 QMC 对“沿某个超平面跳跃”的函数，最坏误差不快于 \(n^{-1/2-1/(2d)}\)。
- **多维、RQMC**：一类不连续函数上 RQMC 的均方根误差为 \(O(n^{-1/2-1/(4d-2)+\epsilon})\)，某些指示函数为 \(O(n^{-1/2-1/(2d)})\)（He, Wang 2015 ✓）。
- **多维、自适应对非自适应**：一个初等的观察是，任意 \(n\) 个点在 \(x_1\) 方向的投影之间必有长度 \(\ge1/(n+1)\) 的空隙，两条落在空隙两端的平行超平面所定义的跳跃函数，在所有采样点上取值相同、积分却相差 \(\ge1/(n+1)\)。因此非自适应方法对“沿未知超平面跳跃”的类最坏误差 \(\Omega(1/n)\)；而自适应方法沿直线二分，可以指数精度定位超平面。“未知（弯曲）超曲面、多个奇异面”这一般情形，检索没有找到锐利的信息复杂度结果，是空白。
- **随机化设定下的自适应**：Novak 1996 提出的问题最近被 Heinrich 解决：对某些问题，随机化自适应比非自适应快 \(n\) 的幂次，标量情形差距可达 \(n^{1/4}\)（Heinrich 2024 ✓）。这说明“自适应何时有用”在随机化设定下仍是活跃问题。

### 2.3 高斯测度下的折点：预积分（条件 QMC）

| 结果 | 内容 | 前提 |
| --- | --- | --- |
| ANOVA 光滑化（Griebel, Kuo, Sloan 2010、2013） | \(\max(\varphi,0)\) 的低阶 ANOVA 项比原函数更光滑 | 单个折点，\(\varphi\) 光滑且沿某变量单调 |
| 预积分光滑化（Griewank, Kuo, Leövey, Sloan 2018 ✓） | 沿一个变量先积掉，折点或跳跃变成光滑的 \(d-1\) 维被积函数 | 同上（严格单调） |
| 单调性基本必要（Gilbert, Kuo, Sloan 2022 ✓） | 单调性失效时，预积分后一般出现平方根等奇性；作者写明此时 QMC“at present no theoretical support” | 指出空白 |
| 条件 RQMC 速率（He 2019 ✓） | 典型期权问题达到 \(O(n^{-1+\epsilon})\)；但最高阶 ANOVA 项只有 \(O(n^{-1/2-1/(4d-2)+\epsilon})\) | 单个不连续面，单调 |
| 主动子空间预积分（Liu, Owen 2023 ✓） | 沿主动子空间首向量预积分；证明预积分不增加扰乱网方差；在篮子、彩虹期权上优于其他方法 | 方向选择是启发式，没有速率保证，需要梯度 |
| 数值光滑化（Bayer, Ben Hammouda, Tempone 2023 ✓） | 求根加一维积分，配合自适应稀疏网格或 QMC | 根不聚集，方向固定 |

### 2.4 误差估计

带保证的自适应 QMC 误差界只对“函数锥”成立，即 Walsh 系数衰减足够规则的被积函数（Hickernell, Jiménez Rugama 2016 ✓；锥的框架见 Clancy, Ding, Hamilton, Hickernell, Zhang 2014）。QMCPy 的 `CubQMCSobolG` 是其实现。对折点、跳跃函数，这类保证是否成立、在什么条件下成立，没有现成结论。

### 2.5 文献明确的空白（理论题的候选）

1. 不单调或分段光滑的折点函数（max / min 结构、多个折点面）在预积分后的正则性，以及相应的 RQMC 速率。
2. 只用点查询、只花预算中 \(o(N)\) 的部分选出预积分方向（或主动子空间），并保持接近 \(N^{-1}\) 的速率。
3. 黑盒条件下折点和跳跃的定位查询复杂度，以及定位与积分之间的最优预算分配。
4. 分段光滑类（未知超曲面）上自适应与非自适应的最优误差，以及多维的匹配下界。
5. 误差估计在这类函数上的可靠性（覆盖率保证）。

## 3. 实验侧调研

### 3.1 真实实例：多资产期权

| 族 | 结构 | 参考值 |
| --- | --- | --- |
| 篮子 call / put | 单折点，沿共同驱动方向单调 | 无闭式；几何篮子有闭式可作控制变量；高精度预积分 RQMC 交叉验证 |
| 彩虹 call on max / min | 多折点：\(\max_iS_i=K\) 处的收益折点，加上 \(S_i=S_j\) 处 max 本身的折点；后者一般不单调 | 两资产 Stulz 1982，\(n\) 资产 Johnson 1987（多元正态 CDF，6 维以内可算到约 \(10^{-10}\)，Genz 1992） |
| 价差 / 交换 | 负权重，沿共同方向不单调 | \(K=0\) 时 Margrabe 1978 闭式；\(K\ne0\) 为一维积分 |
| 数字期权 | 跳跃 | GBM 下闭式 |
| 算术亚式、带时间步的多资产 | \(d=L\times m\)，有效维数高 | 无闭式；几何亚式闭式作控制变量；两种独立高精度方法交叉验证 |

BENCHOP（von Sydow 等 2015）是期权定价领域公认的基准项目，可作可信度锚点和部分题例来源，但它偏低维、偏 PDE 方法，不能单独构成数据集。

### 3.2 合成困难族：专门落在理论空白上

所有实例都对 \(z\) 施加一次随机正交旋转（题面声明），使按坐标对齐的技巧无效，方向必须从查询中发现。

- **H1 不单调折点**：折点面在任何单一方向上都不严格单调，例如 \(\varphi(z)=(a\cdot z)^2-(b\cdot z)^2-c\) 一类鞍形，使预积分理论的单调性前提失效（对应空白 1）。
- **H2 多折点面**：\(\max(\varphi_1,\varphi_2,\dots,0)\)，\(\varphi_i=q_i(a_i\cdot z)\) 取不同方向的光滑函数，折点面彼此相交（对应空白 1、2）。
- **H3 弯曲跳跃面**：\(h(z)\,\mathbf 1\{\varphi(z)>0\}\)，\(\varphi\) 为二次或更一般的光滑函数（对应空白 3、4）。
- **H4 低有效维嵌入**：非光滑部分只依赖 \(k\le3\) 个未知方向，环境维数 \(d\) 到 64（对应空白 2、3）。
- **全维光滑背景项**：每个合成实例再叠加一个积分有闭式的全维光滑项（如 \(\exp(w\cdot z)\)、低次多项式），防止“识别出 \(k\) 维结构后直接精确积分”。

参考值：光滑背景项用闭式；非光滑部分只依赖 \(k\le3\) 维，在该子空间里用按折点分块的自适应求积算到 \(10^{-12}\) 量级。

### 3.3 参照（黑盒版本，所有查询计入预算）

| 角色 | 方法 | 说明 |
| --- | --- | --- |
| 0 分 | 普通 RQMC（Owen 扰乱 Sobol'，\(N\) 点） | 旋转后无法用 PCA 或布朗桥构造 |
| 50 分 | 冻结的强参照组合：主动子空间预积分（梯度用有限差分，查询计入预算）加数值光滑化变体 | 按事先声明的规则选择，不逐例事后取最优 |
| 诊断 | 知道结构的“oracle 求解器” | 估计可达精度，用来衡量实验余量，不参与计分 |

现成代码：QMCPy（扰乱 Sobol'、`CubQMCSobolG`）、SciPy（多元正态 CDF）。主动子空间预积分与数值光滑化没有公开代码，需要复现。

## 4. v3 题目设计（草案）

- **问题**：估计 \(\mu=\mathbb E[f(Z)]\)，\(Z\sim N(0,I_d)\)；\(f\) 只能点查询，查询点为 \(\mathbb R^d\) 中的有限向量。
- **规模**：合成族 \(d\in\{4,8,16,32,64\}\)；期权族 \(d=L\times m\)（\(L\in\{2,4,8\}\) 个资产、\(m\in\{1,4,8\}\) 个时间步）。预算 \(N=2^{14}\) 次查询（含试探与有限差分）；每例 1 CPU、60 秒、2 GiB。
- **声明的函数类**（理论附录给出正式定义）：\(f=\sum_j\mathbf 1_{\Omega_j}h_j\)。区域 \(\Omega_j\) 由至多 \(K\) 个 \(C^r\) 函数的零水平集切出（曲率有界），每片 \(h_j\in C^r\)，增长不超过 \(Ce^{c|z|}\)。允许折点或有界跳跃。参数 \(K,r,C,c\) 公开。
- **计分**：
  - 每例 \(e=\log_{10}(|\hat I-I|/|I|)\)，截到 \([-12,0]\)（上限由参考值精度决定）。
  - 与强参照同例配对的增益截到 \(\pm2\)，按族 × 维度平均得到每条赛道的 \(G\)，再按现行映射换算（\(G\ge0\) 为 \(50+25G\)；\(G<0\) 时以 0 分锚点线性插值）。
  - 困难区间赛道与真实实例赛道分开报告，总分默认为两者平均。
- **可检验声明**：结果中可选给出 `error_bound` 与 `confidence`（如 0.95 或 deterministic）。评测端统计覆盖率（\(|\hat I-I|\le\) 声明界的比例）和界的松紧（声明界与实际误差之比的中位数）。先只作诊断，供理论评审使用。
- **数据**：公开开发集（带真值）、public 反馈集（留出、独立种子，修正 v2 之后 continuum 那类“反馈集等于样例”的问题）、private 正式集。
- **理论附录**：函数类、代价模型、第 2 节的已知结果与第 2.5 节的空白，以及期望的结论类型（类上的速率、定位复杂度、下界、误差估计的可靠性）。

## 5. 理论交付的期望层级（rubric 种子）

| 级别 | 示例 |
| --- | --- |
| 教科书 | 单调单折点预积分后的光滑性与 \(O(n^{-1+\epsilon})\) RQMC 速率；重复 RQMC 置信区间的有效性 |
| 增量 | max / min 结构下预积分函数的正则性刻画；黑盒沿直线定位折点的查询复杂度；非自适应 \(\Omega(1/n)\) 下界加自适应上界，构成可证明的分离；锥条件下误差界的可靠性 |
| 前沿 | 声明类上自适应与非自适应的匹配下界；只用 \(o(N)\) 次点查询、可证明的方向选择，并保持接近 \(N^{-1}\) 的速率 |

**红线**（声称突破即有错）：不借助结构的光滑类积分打破维数灾难或 Bakhvalov 速率；对不连续函数的最高阶 ANOVA 项、在不做光滑化的情况下声称 \(O(n^{-1+\epsilon})\)；等权 QMC 对沿任意超平面的跳跃函数声称快于 \(n^{-1/2-1/(2d)}\)。

**Lean 候选**：非自适应 \(\Omega(1/n)\) 下界（抽屉原理，初等）；一维单调情形预积分的光滑性（隐函数定理）；二维平方根奇性的显式例子；二分定位的查询复杂度。

## 6. 计划与里程碑

| 步骤 | 内容 | 预计 |
| --- | --- | --- |
| M0 | 理论附录草稿（函数类、代价模型、已知结果、开放问题），即本文第 2、4 节的正式化 | 0.5 天 |
| M1 | 实例生成器与参考值：合成族 H1–H4（低维求积加闭式背景项）、期权族（闭式、多元正态 CDF、交叉验证） | 2 天 |
| M2 | 黑盒参照与原型：普通 RQMC、主动子空间预积分、数值光滑化、oracle 求解器；画各族误差随 \(N\) 的曲线，确认实验余量 | 2 天 |
| M3 | 题面 v3、评测器（新协议字段、误差界诊断）、本机标定锚点、公开 / 反馈 / 正式三套数据 | 1 天 |
| M4 | 双考试跑：GPT-6.2 Sol、Opus 两个模型，单会话（时长按 THEORY_TRACK 3.0 待定的总时长）；检查理论交付是否不平凡、有无可利用的漏洞 | 1 天加运行时间 |
| M5 | 决定是否转正；把第 5 节整理成 rubric 种子交给专家 | — |

M2 是关卡：若强参照在困难族上已接近 oracle 精度，就回到 M1 调整族的设计（更多折点面、更强的不单调性、跳跃），再测。

**进度（11-24）**

- **M0 完成**：[`versions/v3/THEORY_APPENDIX.md`](../versions/v3/THEORY_APPENDIX.md)。内容：函数类 \(\mathcal F_d(K,r,M,c)\)（按 \(k\le K\) 个光滑函数的符号模式分片，导数与增长界，非退化，折点、跳跃两型）、信息与代价（点查询，\(N=2^{14}\)，60 秒）、误差准则、已知结果表、5 个开放问题、期望结论。隐藏实例取 \(K\le6\)，因为三资产彩虹期权需要 6 个面。
- **M1 完成**：生成器在 [`versions/v3/generation/`](../versions/v3/generation/)，包括 `synthetic.py`（H1–H3）、`options.py`（10 个期权族）、`oracle.py`（随机正交旋转后的黑盒）、`generate.py`、`verify.py`。原型集在 `versions/v3/data/prototype/`，共 72 例（命名空间 3，开发用，不是评测集）：
  - 合成 30 例：H1–H3 × \(d\in\{4,8,16,32,64\}\) × 2 个种子。H4（低有效维嵌入）没有单列，因为 H1–H3 的非光滑部分都只依赖两个隐藏方向，再叠加全维光滑项。
  - 期权 42 例：21 个配置 × 2 个种子，\(d\) 从 2 到 64。
- **参考值精度**：每例至少两条独立路线。
  - 合成族用两种积分次序。方法 A 内层精确分片、外层自适应；方法 B 交换次序、交点数值定位。两者都在外层的结构变化点做 \(t=t^*\pm s^2\) 代换，相互吻合到 \(10^{-15}\) 量级，相对不确定度 \(\le1.2\times10^{-13}\)。
  - 期权族中，闭式（几何篮子、Margrabe、Stulz、两资产数字）、两种次序的条件嵌套求积、\(d-1\le3\) 时预积分加张量 Gauss–Hermite，三者两两吻合，相对不确定度 \(\le10^{-11}\)。
  - 8 资产篮子与路径族（亚式、篮子亚式、亚式价差）用沿两个不同单调方向的预积分（单个根处闭式）加扰乱 Sobol'（\(2^{20}\times16\)），剩余维度按主动子空间排序。相对不确定度（3 SE）为 \(3\times10^{-8}\) 到 \(1.4\times10^{-6}\)；唯一例外是深度虚值的 4 资产篮子亚式（\(I=0.49\)），为 \(8\times10^{-6}\)。
- **独立检查**：72/72 通过，包括路线两两吻合、经旋转 oracle 的 RQMC（\(2^{18}\times16\)）、参考值所依赖的逐点恒等式。逐例结果与每例误差下限 `e_floor`（\(=\max(-12,\log_{10}(10\times\) 相对不确定度\())\)）见 [`VERIFICATION.md`](../versions/v3/data/prototype/VERIFICATION.md)。路径族的 `e_floor` 在 −6.5 到 −4.1，比按结构积分的 oracle 求解器在 \(2^{14}\) 点下的误差低一个数量级以上；M2 测出实际可达精度后，再决定是否加点。
- **过程中的两处修正**：方法 B 最初没有在竖直切线点分段，只到 \(10^{-9}\)–\(10^{-11}\)，分段后与 A 吻合到 \(10^{-15}\)。预积分后的剩余维度最初用任意正交基，改为按主动子空间排序后，同样点数下标准误降 8–35 倍。
- **下一步 M2**：黑盒参照（普通 RQMC、主动子空间预积分、数值光滑化、oracle 求解器），画各族误差随 \(N\) 的曲线，确认实验余量。

## 7. 风险

1. **实验余量不足**：标准期权上强参照已接近 \(N^{-1}\)。缓解：分数主要由困难族决定，并以 M2 的实测为关卡。
2. **参考值精度**：没有闭式的实例只能交叉验证到约 \(10^{-7}\)–\(10^{-9}\)。缓解：真实赛道优先选闭式和半解析实例；误差截断下限按参考精度设在 \(-12\) 或更高。M1 实测：路径族与 8 资产篮子的相对不确定度为 \(3\times10^{-8}\)–\(8\times10^{-6}\)，按例设 `e_floor`（见第 6 节进度）。
3. **参数化识别的漏洞**：模型可能认出“这是 GBM 篮子”，拟合参数后用近似公式。缓解：函数类声明是一般的；合成族加全维背景项和随机旋转；真实族混用多种收益与模型。
4. **理论太难，只能交出教科书结果**：缓解：第 5 节给出可达的增量目标（二维结果、定位复杂度、非自适应下界）；M4 试跑检验。
5. **查询预算和时间**：收益函数计算很便宜，\(N=2^{14}\) 加 60 秒足够。参照里的有限差分梯度需要计入预算，这也是公平的一部分。

## 8. 决定（11-24 13:32 用户认可以下默认选项）

1. **测度**：高斯（对应活跃的研究社区与真实实例），不保留 \([0,1]^d\)。后者可以通过 \(\Phi^{-1}\) 变换容纳期权实例，但被积函数在边界无界，函数类的定义更麻烦。
2. **维数与预算**：\(d\) 最高到 64、\(N=2^{14}\)。
3. **两条赛道的权重**：各半。
4. **是否要求输出误差界**：可选，只作诊断。
5. **v2 榜单**：保留为纯实验题，v3 另起双考榜。

## 参考文献

1. ✓ A. Hinrichs, E. Novak, M. Ullrich, H. Woźniakowski, The curse of dimensionality for numerical integration of smooth functions, Math. Comp. 83 (2014) 2853–2863。
2. ✓ F. Y. Kuo, I. H. Sloan, H. Woźniakowski, Multivariate integration for analytic functions with Gaussian kernels, Math. Comp. 86 (2017) 829–853. <https://doi.org/10.1090/mcom/3144>。
3. A. B. Owen, Scrambled net variance for integrals of smooth functions, Ann. Statist. 25 (1997) 1541–1562。
4. ✓ L. Plaskota, G. W. Wasilkowski, Adaption allows efficient integration of functions with unknown singularities, Numer. Math. 102 (2005) 123–144。
5. ✓ L. Plaskota, G. W. Wasilkowski, Y. Zhao, The power of adaption for approximating functions with singularities, Math. Comp. 77 (2008). <https://www.ams.org/journals/mcom/2008-77-264/S0025-5718-08-02103-0/>。
6. ✓ R. Alexander, Geometric methods in the study of irregularities of distribution, Combinatorica 10 (1990) 115–136；✓ J. Matoušek, Tight upper bounds for the discrepancy of half-spaces, Discrete Comput. Geom. 13 (1995) 593–601. <https://doi.org/10.1007/BF02574066>。
7. ✓ Z. He, X. Wang, On the convergence rate of randomized quasi-Monte Carlo for discontinuous functions, SIAM J. Numer. Anal. 53 (2015). <https://doi.org/10.1137/15M1007963>。
8. ✓ Z. He, On the error rate of conditional quasi-Monte Carlo for discontinuous functions, SIAM J. Numer. Anal. 57 (2019) 854–874. <https://doi.org/10.1137/18M118270X>。
9. ✓ S. Heinrich, Randomized complexity of mean computation and the adaption problem, J. Complexity (2024) 101872. <https://doi.org/10.1016/j.jco.2024.101872>。
10. E. Novak, On the power of adaption, J. Complexity 12 (1996) 199–237；✓ E. Novak, Optimal algorithms for numerical integration: recent results and open problems, 2024. <https://arxiv.org/abs/2307.06787>。
11. M. Griebel, F. Y. Kuo, I. H. Sloan, The smoothing effect of integration in \(\mathbb R^d\) and the ANOVA decomposition, Math. Comp. 82 (2013) 383–400。
12. ✓ A. Griewank, F. Y. Kuo, H. Leövey, I. H. Sloan, High dimensional integration of kinks and jumps — smoothing by preintegration, J. Comput. Appl. Math. 344 (2018) 259–274. <https://arxiv.org/abs/1712.00920>。
13. ✓ A. D. Gilbert, F. Y. Kuo, I. H. Sloan, Preintegration is not smoothing when monotonicity fails, in Advances in Modeling and Simulation, Springer, 2022. <https://arxiv.org/abs/2112.11621>。
14. ✓ S. Liu, A. B. Owen, Preintegration via active subspace, SIAM J. Numer. Anal. 61 (2023) 495–514. <https://doi.org/10.1137/22M1479129>。
15. ✓ C. Bayer, C. Ben Hammouda, R. Tempone, Numerical smoothing with hierarchical adaptive sparse grids and quasi-Monte Carlo methods for efficient option pricing, Quant. Finance 23 (2023) 209–227。
16. ✓ F. J. Hickernell, Ll. A. Jiménez Rugama, Reliable adaptive cubature using digital sequences, in MCQMC 2014, Springer, 2016. <https://arxiv.org/abs/1410.8615>；N. Clancy, Y. Ding, C. Hamilton, F. J. Hickernell, Y. Zhang, The cost of deterministic, adaptive, automatic algorithms: cones, not balls, J. Complexity 30 (2014) 21–45。
17. R. M. Stulz, Options on the minimum or the maximum of two risky assets, J. Financ. Econ. 10 (1982) 161–185；H. Johnson, Options on the maximum or the minimum of several assets, J. Financ. Quant. Anal. 22 (1987) 277–283；W. Margrabe, The value of an option to exchange one asset for another, J. Finance 33 (1978) 177–186。
18. A. Genz, Numerical computation of multivariate normal probabilities, J. Comput. Graph. Statist. 1 (1992) 141–149。
19. L. von Sydow 等, BENCHOP — the BENCHmarking project in option pricing, Int. J. Comput. Math. 92 (2015) 2361–2379。
20. A. Genz, Testing multidimensional integration routines, 1984。
