# r4：出题人 B.4 锚点（galaxy-gravity v4）

来自出题人评测包 `galaxy-gravity-b4-evaluation-0.4.0-precalibration`（`baselines/`）。与 r1 使用同一份 Treecode2 源码（`Treecode2-16051d8.tar.gz`，SHA-256 `71f92207…`）和相同的构建脚本，唯一差别是 0 分锚点的张角：

| 目录 | 锚点 | 张角 | 与 r1 的差别 |
| --- | --- | --- | --- |
| `baselines/standard` | 0 分：经典简单树 | 0.8 | r1 为 0.7（出题人独立调参后改为 0.8） |
| `baselines/modified` | 50 分：改进树 | 0.7 | 相同 |

改进树同时是逐例时限的标定程序：每张快照跑 3 次，时限 = ceil(1.25 × 最大 CPU 时间 / 0.1 秒) × 0.1 秒。
