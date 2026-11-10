# 星系快照中的引力加速度（131,072 个粒子）

粒子数加倍到 131,072、每次调用仍是 5 CPU 秒，使精确的直接求和无法在预算内完成，比较近似算法的精度。

看板页面：http://9.135.91.91:18765/tasks/galaxy-gravity-v2.html（办公网）

## 怎么读这道题

1. **题目背景**：下面第一节。
2. **题面原文**：[`statement.md`](statement.md)，与给模型的版本逐字相同。
3. **数据**：见“数据与代码在哪里”；公开开发材料的文本文件在 [`public/`](public/)。
4. **评测与计分**：“评测依据”一节；评测器位置见下表。
5. **Baseline**：“Baseline”一节；参照源码在 [`reference/`](reference/)。
6. **模型解法**：“模型成绩与解法”一节；每份提交的完整源码在 [`solutions/`](solutions/)。

## 题目背景

N 体引力计算是天体物理模拟的核心：每一步都要算出每个粒子受到其余所有粒子的引力。直接两两求和是 \(O(N^2)\)，规模一大就算不动，因此有树算法（Barnes–Hut，按张角把远处的粒子团当作一个质点）和快速多极子方法（FMM），以精度换速度。题目的本意是比较在固定 CPU 预算下，谁的近似算法精度更高。

**为什么改规模**：原版 65,536 个粒子时，高度优化的直接两两求和能在 5 CPU 秒内算出精确解，近似算法的优劣无法区分。粒子数翻倍后直接求和的耗时乘 4（本机约 15 CPU 秒），树算法参照只乘约 2（约 3–4 CPU 秒）。

## 数据与代码在哪里

| 内容 | 位置 |
| --- | --- |
| 题包 | tasks/operator/galaxy-gravity-v2/（FrontierAM 重建说明）；封存副本 .runtime-ops/galaxy-gravity-v2/（本服务器，git 忽略） |
| 公开开发材料 | public/（本目录，文本文件）；10 张公开快照在 .runtime-ops/galaxy-gravity-v2/packages/public/public-dev/cases/ |
| 私有测试集 | 20 张私有快照（我方以私有种子用原包生成器生成，参考解用原包直接求和程序），封存在 .runtime-ops/galaxy-gravity-v2/private/ |
| 评测器 | 原包评测器 tasks/galaxy-gravity/harbor/tests/grade.py（只改输入大小检查与架构检查） |
| 参照实现 | reference/（两条 Treecode2 参照，原包 baselines/） |
| 详细文档 | `results/galaxy-gravity/GALAXY_SCALE.md`、`results/BASELINE_REVIEW.md` |

公开包中的数据文件（未复制到本目录）：public-dev/cases/case-01.in（4,194,304 字节）；public-dev/cases/case-01.ref（3,145,728 字节）；public-dev/cases/case-02.in（4,194,304 字节）；public-dev/cases/case-02.ref（3,145,728 字节）；public-dev/cases/case-03.in（4,194,304 字节）；public-dev/cases/case-03.ref（3,145,728 字节）；public-dev/cases/case-04.in（4,194,304 字节）；public-dev/cases/case-04.ref（3,145,728 字节）；public-dev/cases/case-05.in（4,194,304 字节）；public-dev/cases/case-05.ref（3,145,728 字节）；public-dev/cases/case-06.in（4,194,304 字节）；public-dev/cases/case-06.ref（3,145,728 字节）；……。

## 评测依据

- **输入**：二进制快照，每个粒子一行（质量和三维位置）；正式快照每张 131,072 个等质量粒子。
- **数据**：新快照由公开生成器以私有种子生成，参考加速度用原题包的直接求和程序计算；另附 10 张正式规模的公开快照。
- **正式集**：5 类场景（Hernquist 球、Jaffe 球、Einasto 球、扁平盘、四星团群）各 4 张，每张以不同随机种子调用 3 次，共 60 次调用。
- **资源**：每次调用最多 5 个进程 CPU 秒（含启动和读写）、2 GiB 内存、单线程。
- **有效性**：任一次调用超限、异常或输出不合法，整份提交记 0 分。
- **原始指标**：加速度的整体相对 RMS 误差（越小越好）。
- **换算**：设弱参照误差为 \(U\)、强参照为 \(W\)。\(E \ge U\) 记 0 分；\(W \le E < U\) 记 \(50(U-E)/(U-W)\)；\(E < W\) 记 \(100 - 50E/W\)。

## Baseline

两条参照与原题相同（Treecode2 的两组参数），在本机评测核 grade-06 上实测 20 张快照、60 次调用：

| 参照 | 加速度相对 RMS 误差 | 记分 | 单次最多 CPU 秒（上限 5） |
| --- | --- | --- | --- |
| 弱参照 | 0.00125189 | 0 | 4.05 |
| 强参照 | 0.000590528 | 50 | 3.71 |

同样的硬件上，直接两两求和约需 15 CPU 秒，超出上限约 3 倍。

## 模型成绩与解法（协议 v6.2，最新）

| 模型 | 设置 | 原始指标 | 换算 | 得分（满分 100，参照 50） | 复核 | 源码 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | \(L = 4.25\times10^{-9}\)，单次最多 4.45 CPU 秒 | \(100 - 50\,L/\text{强}\) | 100.0 | 待复核 | [`solutions/gpt-6.1-sol-single/`](solutions/gpt-6.1-sol-single/) |
| Claude Opus 5.5 | 单轮 | \(L = 1.06\times10^{-7}\)，单次最多 3.16 CPU 秒 | \(100 - 50\,L/\text{强}\) | 100.0 | 待复核 | [`solutions/claude-opus-5-5-single/`](solutions/claude-opus-5-5-single/) |

超过参照（50 分）的成绩在复核结论出来之前标“待复核”：复核会解释超出的原因、寻找更强的人类方法并做消融，见 `results/BASELINE_REVIEW.md` 与 `results/BASELINE_POLICY.md`。

### GPT-6.1 Sol（单轮）

二叉空间树上的高阶调和多极展开（16 阶）处理牛顿核，软化核与牛顿核之差用 10 阶笛卡尔展开，近场直接求和；按当前输入标定的代价模型选择精度，并带计时保护。误差比强参照小五个数量级。

源码：[`solutions/gpt-6.1-sol-single/`](solutions/gpt-6.1-sol-single/)（提交 `galaxy-gravity-v2-20261109T132936Z-9e4112`）；未复制的二进制或大文件：solve（1,241,096 字节）。

### Claude Opus 5.5（单轮）

笛卡尔坐标快速多极方法（FMM，双树遍历，11 阶展开，AVX2 向量化，Plummer 软化，双精度）。提交里没有 README。

源码：[`solutions/claude-opus-5-5-single/`](solutions/claude-opus-5-5-single/)（提交 `galaxy-gravity-v2-20261109T130056Z-aea0a3`）；未复制的二进制或大文件：solve（430,048 字节）。

## 观察与说明

加倍粒子数确实让直接求和超时，但两个模型都改用了高阶快速多极方法，在时限内把误差降到 \(10^{-7}\) 以下。映射在误差低于强参照后为 \(100 - 50\,L/\text{强}\)，强参照误差约 \(6\times10^{-4}\)，所以误差在 \(10^{-6}\) 以下的提交都挤在 99.9 分以上，这道题已经无法区分强解法（已记为待反馈出题人的事项）。

弱参照单次最多 4.05 CPU 秒，离上限只有 19% 的余量；后续尝试按约定改为 6 CPU 秒，本轮成绩按题面写明的 5 秒保留。

## 历史版本

原版 65,536 粒子（v6.1，sol 用直接求和得满分）见 `results/NEW_TASKS.md`。
