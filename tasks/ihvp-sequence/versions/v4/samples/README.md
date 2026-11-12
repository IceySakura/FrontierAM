# 样例

完整样例从本版全新 public 池中固定选取第一条和最后一条，由维护端服务提供。教师数组、生成器、清单、种子与参考解不进入开发工作区。

- `smoke.json`：可直接阅读的 `d=4, T=3` 协议示例（矩阵、右端项与参考解），不参与评分。
- `sample_01`、`sample_02`：两条完整规模序列（`d=16384, T=50`），取自 public 反馈池。算子不在工作区里；运行时由评测方的样例服务通过与正式评测相同的接口提供当前右端项与 HVP 回复。

用自己的提交运行一条样例（默认 `/workspace/submission`）：

```sh
python3 /workspace/samples/run_sample.py sample_01
python3 /workspace/samples/run_sample.py sample_02 /path/to/other/submission
python3 /workspace/samples/run_sample.py smoke
```

结束后打印该序列是否有效、墙钟时间、HVP 向量数、每步与平均的归一化平方能量误差（与正式原始指标相同的定义）。计时在开发核上进行，只作参考；求解器的标准错误输出直接显示在终端。`support/worker.py` 与 `support/api.py` 是评测端使用的同一份协议代码。
