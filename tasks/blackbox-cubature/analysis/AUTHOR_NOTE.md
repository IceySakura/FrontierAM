# 给出题人：blackbox-cubature 接入时的两处改动（2026-11-17）

题目已接入 FrontierAM 并上榜（版本 v2）。有两处改动想请你确认。

## 1. 指标：族中位数的中位数 → 与参照配对的增益

原指标是三族各取中位数、再取中位数的 log10 绝对误差。我们在 216 个隐藏用例上实测（本机重跑）发现：

- product_peak 的积分值达 1e4–1e17，绝对误差恒为最大；polynomial 恒为最小。三族中位数的中位数在 A/B/C 每组切分里都等于 fourier 一族的中位数，另两族不影响分数。
- Sobol 与 HumanPortfolio 只差 0.15（按用例 bootstrap 的标准差 0.22），在 A、B 两组上 Sobol 反而更好，参照顺序不稳定。

现指标：每例 log10 相对误差截到 [−15, 0]，与本机重跑的 HumanPortfolio-v1 同例配对，增益截到 ±2（每例最多计 100 倍），按族 × 维度平均、维度平均、三族平均得 G；G ≥ 0 为 50 + 25G，G < 0 时以普通蒙特卡洛为 0 分线性插值。三组切分上结果一致：蒙特卡洛 0、Sobol 约 25–37、HumanPortfolio 50。

## 2. 多项式族 → 折点 ridge 叠加

polynomial_ridge（1 + a2 t² + a4 t⁴）是 x 的 4 次多项式：8 维只有 495 个单项式，2048 点最小二乘拟合即可精确积分；认出 ridge 方向后用精确矩也可以。你的 GPT-6.1-sol 提交在全部 108 个多项式用例上都达到机器精度，这一族只能区分“试没试多项式拟合”。

v2 用同种子替换为 2–3 条方向各异、带折点的 ridge 叠加：f = 1 + Σ c_j max(0, v_j·(x − u_j))^{p_j}（p = 2 或 3），精确积分用截断幂公式按有理数计算（已与数值积分核对）。fourier 与乘积峰原样保留。生成脚本 `versions/v2/generation/generate.py`，预言机 `versions/v2/grader/oracle.py`。

## 其他

- 每例时限 120 秒 → 30 秒（public 反馈单槽 15 分钟上限）；已知最慢的提交每例不到 8 秒。
- 本机锚点（private：B、C 两组 144 例）：蒙特卡洛 G0 = −0.713；你的 sol 提交 53.9 分。
- 详细实测：`analysis/kinked-ridge-proposal/`、`results/OPEN_ITEMS.md`。
