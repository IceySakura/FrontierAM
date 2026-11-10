# 最终私有集

本版本使用出题方单独保管的 15 条独立序列，覆盖与反馈池相同的五种结构和三档基础谱条件数，每条 `d=16384, T=50`。

GitHub 不保存私有用例正文、参考答案、生成种子或私有冻结计划。不要将它们写入本目录、PR、issue 或聊天内容。维护者接收私有材料后，通过数据准备脚本放入：

```text
var/raw/ihvp-sequence-v1/data/private/manifest.json
var/raw/ihvp-sequence-v1/data/private/cases/<id>/teacher.npz
```

这些路径被仓库的 `var/` 规则忽略。准备脚本校验原文件 SHA-256，并保持当前私有集身份，不重新采样或挑选用例。正式运行时，教师文件与答案只能由宿主评分端读取。

通用生成器的 `--private` 模式会创建新的本地种子计划，供新版本或独立实验使用；它不能重建或替换已封存的 v1 私有集。数据变化必须按 `docs/TASK_CHANGES.md` 建立新版本并重新校准。
