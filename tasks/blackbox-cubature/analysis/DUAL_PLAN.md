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
| 50 分 | 冻结的强参照：GPCA-RQMC（Xiao–Wang 2019；有限差分梯度估计 \(C=E[\nabla f\nabla f^\top]\)，查询计入预算，再在其特征基上做一个扰乱 Sobol' 网） | 按事先声明的规则选择，不逐例事后取最优。11-24 晚按用户要求实现了文献 SOTA 管线（预积分、GPCA、ODIS、数值光滑化），按查询计价时它在原型集上输给 GPCA-RQMC，保留为对照，见第 9.1 节 |
| 0 分 | 普通蒙特卡洛（\(N\) 个独立点），同 v2 | 原定的普通 RQMC 与强参照差得太近，映射会退化 |
| 诊断 | 知道结构的“oracle 求解器” | 估计可达精度，用来衡量实验余量，不参与计分 |

现成代码：QMCPy（扰乱 Sobol'、`CubQMCSobolG`）、SciPy（多元正态 CDF）。主动子空间预积分与数值光滑化没有公开代码，已在 `versions/v3/baselines/` 复现。

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

**进度（11-24）**（本节记录的 72 例原型集、旧合成族和 M2 暂定强参照，已被第 9 节的修订取代，保留作历史记录）

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
- **M2 完成（11-24 下午）**：黑盒参照与诊断方法在 [`versions/v3/baselines/`](../versions/v3/baselines/)（`methods.py`、`run.py`），按查询计数、强制预算。在 72 例原型集上跑了 \(N=2^{10}\) 到 \(2^{16}\)，每档 8 个种子。报告与图在 [`data/prototype/m2/REPORT.md`](../versions/v3/data/prototype/m2/REPORT.md)，单次运行最长 1.6 秒。方法如下：
  - 普通 RQMC：0 分锚点。
  - 主动子空间 RQMC：有限差分梯度（步长 0.05）估计主动子空间，再在该基上做扰乱 Sobol'。
  - 主动子空间预积分加数值光滑化：沿直线用 Gauss–Hermite 探测零与非零的边界，二分定位，非零段用截断正态 Gauss 规则积分，线积分精度约 \(10^{-10}\)。
  - 低维诊断：\(d\le4\) 时外层改用张量 Gauss–Hermite。
  - oracle 求解器：沿结构方向精确做线积分，每条线算 1 次查询。
- **强参照的确定**：黑盒预积分每条线约 28 次查询，线数只有 oracle 的约 1/32，在 \(2^{14}\) 下多数族输给主动子空间 RQMC，到 \(2^{16}\) 才在路径族追平。\(d\le3\) 时，主动子空间 RQMC 主阶段只有 \(N/2\) 点，又不如普通 RQMC。所以冻结的强参照取：\(d\le3\) 普通 RQMC，\(d\ge4\) 主动子空间 RQMC（第 3.3 节）。
- **试算分数**（\(2^{14}\)，草案规则：每例增益 \(g=\mathrm{clip}(e_{\text{强}}-e,\pm2)\)，按（族，维度）格平均，再按赛道平均，\(G\) 取两赛道均值；\(G\ge0\) 时分数为 \(50+25G\)，\(G<0\) 时为 \(50(G-G_0)/(-G_0)\)，\(G_0=G_{\text{普通 RQMC}}=-0.51\)）：

  | 方法 | \(G\) 困难 | \(G\) 期权 | \(G\) | 分数 |
  | --- | --- | --- | --- | --- |
  | 普通 RQMC | −0.58 | −0.44 | −0.51 | 0 |
  | 主动子空间 RQMC | 0.00 | −0.12 | −0.06 | 44.1 |
  | 黑盒预积分 | −1.03 | −0.53 | −0.78 | −26.3（低于 0 分锚点） |
  | 强参照 | 0 | 0 | 0 | 50 |
  | oracle 求解器 | +1.48 | +0.86 | +1.17 | 79.3 |

- **关卡结论：通过**。困难族上，强参照比 oracle 差 H1 1.25、H2 1.22、H3 2.26 个数量级，在各档 \(N\) 都稳定，满足“一到两个数量级”的要求。期权路径族差 0.7–1.8 个数量级。低维期权上，这里的 oracle 外层也用 RQMC，低估了可达精度：黑盒“预积分加张量 Gauss–Hermite”用不到 \(2^{14}\) 次查询，在交换、价差、2–3 资产篮子、几何篮子上达到 \(10^{-10}\) 到 \(10^{-9}\)，比强参照好约 6 个数量级。彩虹与数字期权上它失效：外层在直线与边界相切处、内部 \(S_i=S_j\) 处不光滑，正对应开放问题 1、3。
- **M2 发现、留给 M3 决定的事项**：
  1. 合成族整体只依赖 4 个方向（2 个非光滑方向，加背景项的 2 个岭函数方向）。主动子空间方法在 \(d=64\) 与 \(d=4\) 时误差几乎一样，维数轴对它基本不起作用。若希望维数真正起作用，可把背景改为真正全维的光滑项（如沿一组正交方向的多岭函数之和，或乘积形式）。
  2. 低维、单一光滑边界的期权族（交换、价差、2–3 资产篮子、几何篮子）会被简单的确定性方法做满 +2，区分不了强模型。可以减少这些族的比重，或换成更高维、带内部折点的配置。
  3. 篮子亚式的误差下限约为 −5.2，而强参照在 −3.7，可计入的最大增益只有约 1.5；深度虚值那一例的下限是 −4.1。正式集要么提高参考精度，要么只收误差下限不高于“强参照减 2”的实例。
- **下一步 M3**：题面 v3、评测器、本机标定锚点，以及公开、反馈、正式三套数据（先定上面三点）。

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

## 9. M2 之后的修订（11-24 17:11 用户要求）

用户意见：强参照直接取人类论文的 SOTA 解法，同时看理论与实验强度；合成族按讨论过的思路，取理论分析里最困难的极端情形；期权参考相关工作的数据来源，尽量做难；真值精度看强工作怎么处理。

### 9.1 强参照：文献 SOTA 管线

文献现状（✓ 为已核对原文）：

- 预积分（条件 QMC）在“非光滑面沿所选变量严格单调”时把折点或跳跃变成光滑的 \(d-1\) 维被积函数（Griewank 等 2018 ✓），扰乱网的 RMSE 为 \(O(n^{-1+\epsilon})\)（He 2019 ✓）；对扰乱网，预积分不会增大方差（Liu–Owen 2023 ✓）。
- 方向与降维：沿主动子空间首方向预积分（Liu–Owen 2023 ✓，用 128 个梯度样本估计 \(C=E[\nabla f\nabla f^\top]\)，Newton 求根）；梯度主成分（GPCA）给剩余变量排序以降低有效维数（Xiao–Wang 2019 ✓）；“条件 QMC 加 GPCA”两步法（Xiao–Wang 2018 ✓）。
- 条件期望没有解析式时，用数值光滑化：一维求根（Newton）加一维 Gauss–Laguerre 求积，再接自适应稀疏网格或 QMC（Bayer–Ben Hammouda–Tempone 2023 ✓）。
- 尾部与增长：最优漂移重要性抽样（ODIS，Glasserman–Heidelberger–Shahabuddin 1999 ✓）与预积分合用，在临界指数增长下仍有接近 \(n^{-1}\) 的速率，实验在亚式、两资产篮子亚式、Heston 亚式上最好（Chen–Xu–Du–Wang 2025 ✓）。

实现与实测（11-24 晚至 11-25，原型集 78 例）：这条管线的黑盒版本已实现（`versions/v3/baselines/methods.py`）。它用有限差分梯度定 ODIS 漂移与主动子空间，主动子空间中再加入 Stein 均值梯度 \(E[f(Z)Z]\)；沿首方向逐线求出零值与非零值的分界（Newton 型割线法，跳跃时退回二分），分段做 Gauss 求积；剩余维度按主动子空间排序做扰乱 Sobol'。ODIS 设有稀有事件触发条件和方差验证，只在预积分模式下使用。

按查询计价时，它在 \(N=2^{14}\) 下输给其中最简单的组件 GPCA-RQMC（Xiao–Wang 2019）：预积分模式 30.2 分，管线的 RQMC 模式 45.2 分，二者都以 GPCA-RQMC 为 50 分。原因是文献把一次线积分记为一个样本，而黑盒下定位一条线上的分界要十几到二十几次查询；跳跃型收益的方向估计噪声大；未经光滑化的跳跃上做重要性抽样会放大误差。

**冻结的强参照因此取 GPCA-RQMC**，即按查询计价时实测最强的文献方法。管线的两个模式保留为对照。M2 曾暂定的“\(d\le3\) 普通 RQMC、\(d\ge4\) 主动子空间 RQMC”作废（新原型集最低 \(d=8\)）。

### 9.2 合成族：理论最难的极端情形

理论依据：单调性失效时预积分后一般出现平方根奇性，且至今没有 QMC 收敛结果（Gilbert–Kuo–Sloan 2022 ✓，例子为抛物线与双曲线）；不平行于坐标轴的不连续面使 RQMC 只有 \(O(n^{-1/2-1/(4d-2)+\epsilon})\)（He 2017 ✓）；跳跃型岭函数的平均维数随 \(\sqrt d\) 增长，折点有界（Hoyt–Owen 2020 ✓）。

新设计（\(d\in\{8,16,32,64\}\)，随机正交旋转；非光滑部分落在 \(k\) 维子空间，\(k\in\{2,8,d\}\) 构成原 H4 的维数轴；叠加真正全维的乘积型光滑背景，均值有闭式）：

| 族 | 非光滑部分 | 最难之处 |
| --- | --- | --- |
| H1 双曲面折点 | \(\alpha\,(Q_+(z)-Q_-(z)-c)^+\)，\(Q_\pm\) 为正交子空间上的二次型 | 除零锥方向外任何方向都不单调（GKS 例 2 的高维推广） |
| H2 相交折点 | \(\alpha\max(c_1-Q_1,\;c_2-Q_2,\;0)\)，两个椭球“帐篷” | 不单调，且有函数值不为零处的内部折点，零值检测无效 |
| H3 闭合跳跃 | \(\alpha e^{w\cdot z-\|w\|^2/2}\,\mathbf 1\{Q(z)<c\}\)，椭球或球 | 任何方向都不单调；跳跃；可放到尾部做成稀有事件 |

每个二次型在其子空间内是带中心偏移的各向同性形式，即缩放的非中心卡方变量，各项相互独立，所以真值都化成闭式或一维积分（非中心卡方的分布函数与恒等式），精度可到 \(10^{-12}\)。

### 9.3 期权族：取自文献测试问题并做难

去掉低维欧式族（交换、价差、2–3 资产篮子、几何篮子、2–3 资产彩虹与数字）。新族与出处：

| 族 | 出处与参数 | 维数 |
| --- | --- | --- |
| 算术亚式看涨 | He 2019、Liu–Owen 2023、Chen 等 2025：\(S_0=100\)，\(\sigma=0.4\)，\(r=0.1\)，\(T=1\)，\(K\in\{80,100,120\}\) | 50–80 |
| 数字亚式 | Xiao–Wang 2018：\(\mathbf 1\{S_A>K\}\) | 同上 |
| 两资产篮子亚式 | Liu–Owen（\(\sigma=(0.1,0.2)\)，\(w=(0.8,0.2)\)，\(\rho=0.5\)，\(K\in\{80,100,120\}\)）、Chen 等（\(\sigma=(0.3,0.1)\)，\(w=(0.7,0.3)\)，\(K\in\{90,100,110\}\)） | 60–100 |
| 彩虹亚式（max / min） | Liu–Owen：同上两资产 | 100 |
| 离散障碍（向下敲出看涨、双障碍、带障碍的亚式） | Xiao–Wang 2018：\(d\in\{16,64\}\)，\(K\in\{100,110\}\)，\(L\in\{85,\dots,105\}\)；Achtsis–Cools–Nuyens 2013 | 16–64 |
| Heston 亚式 | Chen 等 2025：\(S_0=50\)，\(V_0=\theta=\sigma_v=0.2\)，\(\kappa=1\)，\(\rho=0.5\)，\(r=0.05\)，\(K\in\{40,50,60\}\)，Euler 截断格式 | 30–60 |

做难的方向：虚值（稀有事件）、更高波动、双障碍（沿水平方向不单调）、数字型跳跃、Heston 截断带来的大量折点。函数类中非光滑面个数 \(K\) 需相应放宽（障碍期权每个监测日一个面），理论附录随之更新。

### 9.4 真值精度

真值精度的限制来自真值的计算方法，不来自数据：实例参数都是精确给定的，没有闭式的实例只能用有限样本的 RQMC 逼近。强工作的做法有三种：

1. 有半解析方法时用它：单资产离散监测的障碍、回望期权用 Hilbert 变换（Feng–Linetsky 2008 ✓，指数收敛）或 z 变换（Fusai–Germano–Marazzina 2016 ✓，约 \(10^{-12}\)）；Heston 欧式用傅里叶方法（BBT 2023 用 Premia）。
2. 没有时，用大样本 RQMC（或 MC）的结果当真值并注明统计误差：BBT 2023 的 4 资产篮子为 \(11.04\pm10^{-3}\)；Chen 等 2025 用自家方法 \(n=2^{19}\)。
3. 对无偏随机化方法只报独立重复的标准误或 RMSE，不需要真值（Liu–Owen 50 次、Chen 等 100 次）。

本题的做题方法可能有偏，必须有真值。做法：单资产 GBM 离散障碍期权用递推求积（转移核为高斯，障碍处截断）算到约 \(10^{-12}\)；其余用预积分加大样本 RQMC，按例记 `e_floor`，正式集只收 `e_floor` 不高于“强参照误差减 2”的实例。

### 9.5 实施顺序

强参照管线 → 新合成族与真值验证 → 新期权族与真值验证 → 重跑 M1 检查与 M2 余量报告。

### 9.6 进度（11-25）

- **实现**：
  - 合成族在 `versions/v3/generation/synthetic.py`：H1、H2、H3、H3t，\(k\in\{k_{\min},d/2,d\}\)，背景为全维乘积型。真值有两条路线，SciPy 非中心卡方与自写的泊松混合（并换积分变量），相互吻合到 \(10^{-15}\)。
  - 路径期权在 `versions/v3/generation/paths.py`：数字亚式、彩虹亚式 max/min、向下敲出、双敲出、Heston 亚式；带障碍的亚式代码保留，原型集暂不收。亚式与两资产篮子亚式沿用 `options.py`，参数改为文献难区间（`sample_hard`）。
  - 路径期权的真值路线有：均匀缩放方向预积分、单调方向预积分、Stulz 条件化（剩余维度按主成分排序）、梯度方向预积分、第二步资产冲击条件化，以及离散障碍的递推求积 QUAD。QUAD 与文献 Wiener–Hopf 基准吻合到给出的全部位数。
  - 原型集在 `versions/v3/data/prototype/`，共 78 例（合成 46、期权 32）。独立检查 78/78 通过，包括路线两两吻合、经旋转 oracle 的 RQMC、逐点恒等式，见 `VERIFICATION.md`。
  - 基线在 `versions/v3/baselines/`，有普通 MC、普通 RQMC、GPCA-RQMC、黑盒预积分、文献管线的两个模式、两种 oracle。报告见 `data/prototype/m2/REPORT.md`。
- **真值精度**：合成族与离散障碍 \(\le10^{-13}\)；亚式与数字亚式 \(\le1.3\times10^{-7}\)；两资产篮子亚式 \(\le7\times10^{-6}\)；彩虹亚式 \(\le3.7\times10^{-6}\)；Heston 亚式 \(\le2.3\times10^{-5}\)。带障碍的亚式只有 \(1.6\times10^{-4}\)，误差下限 −2.8 而强参照在 −2.25，所以撤出原型集，等二维递推求积（状态为对数价格与累计和）。
- **M2 结果**（\(N=2^{14}\)；普通 MC 为 0 分、GPCA-RQMC 为 50 分；每例增益截到 ±2，按格、赛道平均得 \(G\)，\(G\ge0\) 时 \(50+25G\)，\(G<0\) 时 \(50(G-G_0)/(-G_0)\)，\(G_0=-0.68\)）：

  | 方法 | \(G\) | 分数 |
  | --- | --- | --- |
  | 普通 MC | −0.68 | 0 |
  | 普通 RQMC | −0.29 | 28.7 |
  | 文献管线，预积分模式 | −0.27 | 30.2 |
  | 文献管线，RQMC 模式 | −0.06 | 45.2 |
  | GPCA-RQMC（强参照） | 0 | 50 |
  | oracle（结构方向精确线积分，每线算 1 次查询） | +0.70 | 67.5 |

- **余量**：合成族上，拆分 oracle（非光滑部分取精确值，只对光滑背景做 GPCA-RQMC）比强参照准 1.2（H1、H2）、1.8（H3）、3.5（H3t）个数量级。期权族上 oracle 比强参照准 0.4 到 2.5 个数量级；彩虹族上这个 oracle 不比强参照好，说明它低估了那里的可达精度。
- **测量能力**：Heston 亚式和彩虹 max 的误差下限离强参照不到一个数量级，两资产篮子亚式和彩虹 min 约 1.5 个数量级。按 9.4 的规则，正式集之前要么继续提高这几族的真值精度，要么只收误差下限够低的实例。
- **待办（M3 前）**：见 `results/OPEN_ITEMS.md` 的对应行。

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
21. ✓ Y. Xiao, X. Wang, Conditional quasi-Monte Carlo methods and dimension reduction for option pricing and hedging with discontinuous functions, J. Comput. Appl. Math. 343 (2018) 289–308。
22. ✓ Y. Xiao, X. Wang, Enhancing quasi-Monte Carlo simulation by minimizing effective dimension for derivative pricing, Comput. Econ. 54 (2019) 343–366。
23. ✓ J. Chen, Y. Xu, J. Du, X. Wang, Randomized quasi-Monte Carlo and importance sampling for super-fast growing functions with applications to finance, 2025. <https://arxiv.org/abs/2510.06705>。
24. ✓ P. Glasserman, P. Heidelberger, P. Shahabuddin, Asymptotically optimal importance sampling and stratification for pricing path-dependent options, Math. Finance 9 (1999) 117–152。
25. ✓ Z. He, Quasi-Monte Carlo for discontinuous integrands with singularities along the boundary of the unit cube, 2017. <https://arxiv.org/abs/1702.03361>。
26. ✓ C. R. Hoyt, A. B. Owen, Mean dimension of ridge functions, SIAM J. Numer. Anal. 58 (2020). <https://doi.org/10.1137/19M127149X>。
27. ✓ N. Achtsis, R. Cools, D. Nuyens, Conditional sampling for barrier option pricing under the LT method, SIAM J. Financial Math. 4 (2013). <https://doi.org/10.1137/110855909>。
28. ✓ L. Feng, V. Linetsky, Pricing discretely monitored barrier options and defaultable bonds in Lévy process models: a fast Hilbert transform approach, Math. Finance 18 (2008). <https://doi.org/10.1111/j.1467-9965.2008.00338.x>。
29. ✓ G. Fusai, G. Germano, D. Marazzina, Spitzer identity, Wiener–Hopf factorization and pricing of discretely monitored exotic options, Eur. J. Oper. Res. (2016). <https://doi.org/10.1016/j.ejor.2015.11.027>。
30. ✓ A. D. Gilbert, F. Y. Kuo, I. H. Sloan, Analysis of preintegration followed by quasi-Monte Carlo integration for distribution functions and densities, 2021. <https://arxiv.org/abs/2112.10308>。
