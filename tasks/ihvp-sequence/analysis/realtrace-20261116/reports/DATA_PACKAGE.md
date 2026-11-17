评分修订：完整数据ZIP保持原字节和SHA256不变，内含旧诊断报告。当前计分请使用本交接目录的[Warm＝0、Cold＝50标定](ANCHOR_CALIBRATION_20261117.md)。

完整数据包位于5090-A（评测维护者专用）

文件：/mnt/sdb/henryzhang/ihvp_realtrace_20261116/var/release/EVALUATOR_ONLY_ihvp_realtrace_15public_30private_20261116.zip
大小：15,701,110,605字节（14.62GiB）。
SHA256：acfebb847feacbf0294a7df948473e07dc6c407c00eddabf7d193c66cd352a3f

包含15条public、30条private、2250个系统、参考解、所有构造/评测源码及最终汇总报告。数据CRC已全量验证，附加报告的CRC也已验证。

在有5090-A访问权限的机器上下载：

```bash
rsync -avP 5090-A:/mnt/sdb/henryzhang/ihvp_realtrace_20261116/var/release/EVALUATOR_ONLY_ihvp_realtrace_15public_30private_20261116.zip .
shasum -a 256 EVALUATOR_ONLY_ihvp_realtrace_15public_30private_20261116.zip
```

本地EVALUATOR_ONLY_ihvp_realtrace_tools_20261116.zip是报告、使用指南、评测工具和清单；完整算子数据在上述A服务器的ZIP里。

完整ZIP仅交给评测维护者。模型只拿通用题面、worker接口、自己的提交和公开反馈，不能读取本ZIP及其中的生成器/种子/算子/参考解/其他提交/private报告。解压目录先设为700，且不挂载给模型。
