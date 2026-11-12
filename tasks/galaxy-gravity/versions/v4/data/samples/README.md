# 星系快照引力题 · 固定公开开发数据

本目录只提供固定开发数据，不提供快照生成器、随机种子规则或正式数据清单。公开数据不计分，也与正式 80 张隐藏快照独立。输入和输出都是无文件头的小端 IEEE 754 binary64：输入每行依次为 `m,x,y,z`，32 字节；输出每行依次为 `ax,ay,az`，24 字节。所有样本使用引力常数 1 和软化长度 0.01。

## 数据范围

- `smoke.in` / `smoke.ref`：32 粒子接口自检样例。
- `data/dev-hernquist.in` / `.ref`：65,536 粒子，Hernquist 球。
- `data/dev-disc.in` / `.ref`：65,536 粒子，扁平星系盘。
- `data/dev-group.in` / `.ref`：65,536 粒子，四个星团组成的群组。
- `data/dev-jaffe.in` / `.ref`：65,536 粒子，Jaffe 球。
- `data/dev-einasto.in` / `.ref`：65,536 粒子，单个 Einasto 球。

五张固定开发快照各覆盖一种正式分布，使用独立开发数据；`manifest.json` 只记录规模、类别和文件 SHA256，不记录生成规则。正式评测的 262,144、1,048,576 和 4,194,304 粒子数据，以及所有正式参考值和逐例时限，均保留在维护侧。

## 检查程序

先用 32 粒子样例检查接口：

```sh
./solve smoke.in smoke.out
python3 check_output.py smoke.out
```

再用固定开发数据检查全量输出和 RMS 误差：

```sh
./solve data/dev-hernquist.in dev-hernquist.out
python3 check_dev.py data/dev-hernquist.in data/dev-hernquist.ref dev-hernquist.out
```

`check_dev.py` 只用于开发，不代表正式评分。它会检查输出长度、有限性，并按正式题面的全粒子相对误差公式计算 RMS。
