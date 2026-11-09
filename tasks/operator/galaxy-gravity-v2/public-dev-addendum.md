
## 正式规模的公开快照

`cases/` 下有 10 张正式规模（131,072 个粒子）的公开快照，五类模型各 2 张，种子与正式评测不同。`case-NN.in` 为输入，`case-NN.ref` 为按同一直接求和程序算出的参考加速度，`manifest.json` 列出每张的模型和 SHA-256。检查自己的输出：

```sh
./solve cases/case-01.in out.bin
python3 check_output.py out.bin --reference cases/case-01.ref
```
