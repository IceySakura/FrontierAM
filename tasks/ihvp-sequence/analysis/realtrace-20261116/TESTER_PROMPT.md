请复核本次IHVP真实轨迹候选及2026-11-17评分修订，入口README.md，当前规则详见reports/ANCHOR_CALIBRATION_20261117.md。本提示只给评测维护者，不给被测模型。

1. 本轮将Warm CG固定为0分参考、Cold CG固定为50分参考。它们是按这批30条private结果选出的两个最强baseline，故本次是对已有结果的事后重标定，不冒充旧协议成绩或新盲测。不要继续使用旧的CG＝0、L-BFGS-PCG＝50，也不要改正式v4或旧榜单。
2. 按reports/DATA_PACKAGE.md接收A服务器14.62GiB完整数据ZIP，核对SHA256：acfebb847feacbf0294a7df948473e07dc6c407c00eddabf7d193c66cd352a3f。数据仍是15 public＋30 private，每条50步、16384维；不重抽种子或混入旧数据。完整数据ZIP中的旧计分说明作为归档，以本修订为准。
3. 在目标Linux amd64环境，单进程单CPU线程、每完整序列最多30秒、求解进程峰值RSS最多256MiB，重跑两份参考及10种固定方案。启动、导入、IPC、HVP和预处理都计时；预先编译Sol单轮C++扩展，不静默使用fallback。记录硬件、BLAS、依赖、源码和构建哈希；用VmHWM和独立RSS排除fork前评测器高水位。缺少数据或平台失败记未测，不记选手0分。
4. public和private分别标定，必须E0=E_Warm>E50=E_Cold>0，且两份参考在各自整池全部合法。先按序列等权汇总loss再换算一次；不逐例打分后平均，不跨池或机器借用常数。可用tools/calibrate_anchors.py对本机逐例结果评分，并复核端点：Warm=0、Cold=50、零loss=100。
5. 代码、参数、构建在private前冻结；本轮只复核现有冻结方案，不根据private结果重选提交或调数据。若以后跑新模型，开发只用public汇总反馈，private不给模型。模型只收到MODEL_ONLY包及自己的提交，禁止访问整个仓库、EVALUATOR_ONLY包、生成器、种子、算子数组、参考解、其他方案源码、private报告及本提示；用目录/挂载隔离，不能只口头禁止。
6. 复核数据哈希、正定谱界、参考解证书和时序指标。新的评分锚点不等于相关性问题已修复；区别小网络HERO变体与原ZO配置逐步回放，也区别线性系统能量loss与HERO训练指标。

交付public/private各自的raw loss、分数、平均/最长耗时、平均/最大RSS、HVP数和合规数量，附环境/源码指纹及与封存结果的差异解释。明确结论是“可接入新候选协议”还是“仍需补充数据/平台验收”。
