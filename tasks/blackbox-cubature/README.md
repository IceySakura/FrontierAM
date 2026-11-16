# blackbox-cubature

在 \([0,1]^d\)（\(d\) = 2、3、5、8）上估计未知函数的积分。程序看不到函数，只能通过标准输入输出向评测器查询函数值，每例最多 2048 个点、每次最多 512 个点；1 个 CPU、2 GiB、每例 30 秒。尚未上榜（`results/OPEN_ITEMS.md`）。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `versions/v1/` | `task.json`（原始包、题面改动、镜像、参照）、`Dockerfile.amd64`、`data/`（`public/` 24 例、`private/` 144 例与三份提交的逐例实测） |
| `references/r1/` | `mc/`（普通蒙特卡洛，0 分）、`portfolio/`（出题人冻结的 HumanPortfolio-v1 经查询协议实现，50 分） |

评测驱动：`arena/operator_drivers/blackbox_cubature.py`；题目类：`arena/operator_tasks.py` 的 `BlackboxCubature`；分片汇总：`scripts/grade/cubature-suite-score.py`。

## 题目背景

多维数值积分（cubature）。隐藏函数有三族：polynomial_ridge（沿一个方向变化的 4 次多项式）、fourier_ridge（整数频率的周期函数 \(\exp(\kappa\cos(2\pi k\cdot x+\varphi))\)）、product_peak（Genz 的乘积峰，积分值可达 \(10^{17}\)）。相关方法：准蒙特卡洛（Sobol' 序列、数字网、Owen 扰动；QMCPy）、自适应求积（Genz–Malik；Johnson 的 hcubature/pcubature；Cuba）、周期函数的格点规则与梯形法则（Sloan & Joe；Trefethen & Weideman），以及信息基复杂度中自适应选点的理论。

## 数据

出题人的隐藏题分 A、B、C 三组，每组 72 例（3 个套件 × 4 个维度 × 6 例：polynomial 3、fourier 2、product_peak 1），三组互不重叠。

| 层 | 内容 |
| --- | --- |
| samples（工作区） | 出题人的公开试跑包：12 例、9 个 Genz 风格族、2–3 维（`environment/public-dev`，附 `tests/grade.py`） |
| public（迭代反馈） | A 组 H1 套件，24 例 |
| private（最终评分） | B、C 两组，144 例 |

public 只取 24 例是为了满足 public 单槽 15 分钟的评测上限（24 × 30 秒 = 12 分钟）；private 在 4 个评测槽上最坏 18 分钟。

## 计分（满分 100，参照 = 50）

每例 \(e = \log_{10}\) 相对误差，截到 \([-15, 0]\)；与本机重跑的 HumanPortfolio-v1 同例配对，增益 \(g = e_{\text{参照}} - e\) 截到 \([-2, 2]\)，polynomial_ridge 截到 \([-2, 0]\)（做差扣分、做好不加分）。按（族，维度）取平均、维度平均、三族平均得 \(G\)。\(G \ge 0\)：\(50 + 50\,G/(4/3)\)；\(G < 0\)：\(\max(0,\ 50\,(G - G_0)/(-G_0))\)，\(G_0\) 为普通蒙特卡洛在同一组用例上的增益。任一例失败记 0。

出题人原指标（三族各取中位数后再取中位数的 log10 绝对误差）只由 fourier 一族决定，Sobol 与 HumanPortfolio 的顺序在三组之间反转，因此改用上述指标，待出题人确认；理由与实测见 `results/OPEN_ITEMS.md`。

## 本机锚点（private 144 例）

| 参照 | 增益 \(G\) | 得分 | 单例最长 |
| --- | --- | --- | --- |
| 普通蒙特卡洛 | \(G_0 = -0.818\) | 0 | 0.23 秒 |
| HumanPortfolio-v1 | 0 | 50 | 0.96 秒 |
