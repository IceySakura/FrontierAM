# 反馈集

本版本的反馈池由此前出题方交付的 15 条 `d=16384, T=50` 序列组成，覆盖五种谱/时间结构和三档基础谱条件数。此前交付没有对应的外部模型参赛成绩；不能据此填写模型榜单。

这里的 public 表示迭代开发的反馈集。按照 FrontierAM 协议，它不随题面完整交给模型。`../samples/recipes.json` 只公开其中 `case_04`、`case_14` 的两条生成配方和一个额外的小型协议演示。

GitHub 接入不在此目录保存 15 个教师数据文件或完整种子计划。维护者从单独接收的出题包准备它们，默认本地路径为：

```text
var/raw/ihvp-sequence-v1/data/public/manifest.json
var/raw/ihvp-sequence-v1/data/public/cases/<id>/teacher.npz
```

使用 `scripts/seal/prepare_ihvp_sequence.py --source-bundle /path/to/ihvp-sequence-v1` 可按原始 SHA-256 导入。文件只写到 git 忽略路径。每条 manifest 记录 `id`、相对 `path`、`sha256`、规模和算子谱界，评测器只在宿主侧读取教师文件。

旧机器上的校准数值是历史材料。新平台必须按当前协议重新运行参照，核验时限余量、计分和确定性后才能开放模型评测。这里没有声明目标 amd64 容器验收已完成。
