请评测“时序题面恢复版”IHVP候选（sequence-statement-20261117，balanced-split-20261117）。本提示仅给评测维护者，不给被测模型。

1. 本次恢复通用应用背景、时序求解目标及允许保存历史解、搜索方向、低秩近似和预条件信息的说明。不声称覆盖旧合成谱类别，也不保证相关强度或历史复用收益。沿用balanced-split-20261117的15:30划分；系统内容、清单哈希、接口、预算、loss和参考身份均不变。请从新版MODEL_ONLY包启动新的单轮与迭代模型解题运行；旧提交只用于平台复核，不能代替新的模型实验。记录模型版本、开发预算和public反馈次数。
2. 模型只收到MODEL_ONLY包的五个文件（README.md、statement.md、interface.md、solver.py、smoke.json）、自己的提交及public整池汇总反馈。题面内的通用背景与历史利用权限正常提供，不再人为删去；不要额外提供真实数据来源、生成方式、具体相关性或谱统计、算法推荐或先前成绩。整个仓库、EVALUATOR_ONLY包、生成器、种子、算子数组、参考解、其他方案及private报告均不向模型开放，须实际隔离目录和挂载。
3. 接收当前发布balanced_data_package.json对应的重划完整ZIP，核对SHA256和split_manifest.json哈希。解压后按START_BALANCED_EVALUATION.md进入balanced-evaluator/。源case目录名和meta的split不是当前pool；所有replay及calibrate命令必须传--split-manifest split_manifest.json。每条仍50步、16384维；模型开发只使用新public汇总反馈，代码、参数与构建冻结后再评新private，private结果不回传给模型用于修改。
4. 目标环境为Linux amd64，单进程单CPU线程，每完整序列最多30秒，求解进程峰值RSS最多256MiB。启动、导入、IPC、全部HVP、预处理与输出均计时。固定硬件、BLAS、依赖和构建，记录源码哈希、HVP数、耗时和进程自身峰值RSS。平台或数据缺失记未测。
5. 同机同池重跑固定参考：Warm CG＝0、Cold CG＝50，必须E_Warm>E_Cold>0且两份参考全池合规。public/private分别标定，先对序列等权汇总loss，再换算一次分数。若次序不成立，交付原始指标并标记未完成标定，不交换或重新挑选参考。
6. 维护者应按README.md和reports/BALANCED_SPLIT_20261117.md保留版本记录。新划分仅使用两CG参考难度，未按agent成绩选择；重划不会恢复独立盲测资格。结果单列为重划回放实验，不覆盖正式v4、原划分及旧榜单。

交付各模型在public/private的raw loss、分数、平均/最长耗时、平均/最大RSS、HVP数、合规数量、源码指纹和开发预算。由维护者独立决定是否通过平台与计分验收；上述维护者材料不进入模型工作区。
