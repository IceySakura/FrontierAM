# PDHCG 参照

从出题包 `baseline/pdhcg.py` 原样保留的 CPU 参考实现：重启 PDHCG，内层固定步数 CG，并带活动集 KKT 精化及回退。它用于本题的冻结参照，不要求候选使用相同算法，也不宣称是 SOTA。

`run.sh` 接受题面规定的 `--input` 和 `--output`。封存脚本将公开的 schema 和 KKT 检查模块一并放进参照提交，使其自包含；求解过程中不读取评测侧参考答案。参照与候选使用相同 supervisor、进程次数和独立检查器。

每次 Harbor 评测在当前机器重跑参照；bench 则通过 `--calibrate` 和 `bench baseline` 在目标评测机标定。原题包的历史计时不参与本仓库评分。
