# 评分修订：Warm CG＝0，Cold CG＝50

2026-11-17按用户要求，将这批真实轨迹候选的零分参考改为Warm CG，50分参考改为Cold CG。按30条private的平均能量loss，Cold CG为已测最强，Warm CG为次强；这里包括四个agent的排序，Warm CG也略好于Sol迭代。方法身份固定，不随新提交或不同池的排名动态更换。

这是对已有450次合法运行的**事后重标定**，没有重新生成数据、调算法或改变原始loss；不回写成旧协议的正式得分，不宣称重新做了一次独立盲测。既有正式v4及其榜单不变。原报告和旧CG/L-BFGS标定失败记录作为历史保留。

## 锚点

| 池 | 序列数 | 0分：Warm CG，E0 | 50分：Cold CG，E50 |
|---|---:|---:|---:|
| public | 15 | 0.1700374709541636 | 0.046378494982853 |
| private | 30 | 0.004864342424220298 | 0.003617515800103131 |

两池均满足E0>E50>0。选择依据是这批private的排序；不声称Warm CG在public也排名第二。public与private各用各自数值，不能跨池或跨机器套用锚点。

## 映射

先对50个时刻平均，再对本池序列等权平均得到E，最后只换算一次：

$$S(E)=\begin{cases}
0,& E\ge E_0,\\
50\,(E_0-E)/(E_0-E_{50}),& E_{50}\le E<E_0,\\
100-50E/E_{50},&0\le E<E_{50}.
\end{cases}$$

因此Warm CG为0分，Cold CG为50分，零loss为100分。合法但误差较大的程序可以得0分；缺失数据/标定记未测，不能冒充提交失败。任一序列不合法的提交不进入有效loss排名，按协议记0分。

## 本批回放

见[完整分数CSV](scores_cg50-warm0-20261117.csv)与[机器可读标定](anchor_calibration_cg50-warm0-20261117.json)。private上除Cold CG为50分外，其余现有方案均为0分，因为loss均不低于Warm CG。Sol迭代loss为0.0048976021，只比0分参考高约0.68%；这里不把微小差距解释为统计显著性或模型能力的普遍结论。

该映射修复了计分次序；它没有修复多数场景中时序相关性偏弱的问题，也没有证明Cold CG是所有算法中的最优解。新方案优于Cold CG时仍可超过50分。

## 复核

```bash
python tools/calibrate_anchors.py
python tools/check_calibration.py
```

默认读取封存的reports/results_summary.json，并输出本版本JSON和CSV。若在目标评测机重新运行，应在相同硬件、数据和预算下重跑两份参考及候选，再使用逐例结果：

```bash
python tools/calibrate_anchors.py --results var/replay-public var/replay-private --out var/new-calibration
```

脚本核对池大小、参考全池合法性、数据版本与重复记录，先聚合原始loss后评分。禁止用有效子集均值标定参考、逐例算分后平均，或把public的常数用于private。输出不同的标定需使用新目录，避免覆盖封存结果。
