# 接入验证与维护者交接

2026-11-11，GitHub 出题方接入，状态为 **draft**。本次包含题面、样例、生成与准备脚本、局部评测器、两份参照、任务注册和页面生成；不包含私有数据，不开放正式模型尝试。

## 已验证

- 本地使用独立环境：Python 3.13.9、NumPy 2.3.5、SciPy 1.16.3、psutil 7.1.3、threadpoolctl 3.6.0。
- IHVP 与反馈相关测试 **75 passed**；完整 `arena/tests` 结果为 **85 passed, 3 failed**。失败均在已有的 `test_slots_and_store.py`：本机没有 Docker，macOS 子进程仍调用真实 Docker。未据此修改旧题或槽位逻辑。
- 独立样例工具仅携带 `data/samples/`，从无仓库路径的工作目录运行，能够验证完整包；零解在 smoke 和两条完整规模序列上均合法，原始 loss 为 1，分数均为“未测”。小规模 HVP 重建精确解的 loss 约为 `9.8e-33`。缺失样例正文和预测溢出会被拒绝。
- 两条完整样例由原反馈池选取，生成后按源数组哈希核对并复核参考解。维护端准备脚本将原始 15 条反馈与 15 条私有数据留在忽略目录；不向 GitHub 写入正文、答案或私有种子。
- `build_task_pages.py --task ihvp-sequence` 与单题看板构建成功；其他题目页面和已有成绩保留。

本地可复跑：

```sh
python -m pytest -q arena/tests/test_ihvp_sequence.py arena/tests/test_feedback.py
python scripts/seal/prepare_ihvp_sequence.py --samples-only
python var/outputs/ihvp-sequence-v1/samples/check.py --verify-package
python scripts/site/build_task_pages.py --task ihvp-sequence
python scripts/site/build_dashboard.py --task ihvp-sequence --out var/dist/ihvp-preview
```

生成样例和构建上下文的入口见 [样例说明](../versions/v1/data/samples/README.md)。评测器 `grader/evaluate.py` 可用于本地 smoke 或显式重测，但只提供协议隔离，不能作为不可信提交的安全边界。正式框架驱动会拒绝 draft 模型尝试。

## 目标机发布前仍需完成

1. 单独接收并封存 v1 反馈池与私有集；冻结源码、数据、镜像、依赖和评测机标识。
2. 用完整样例和预置 Linux amd64 wheelhouse 离线构建目标镜像；实际验证开发与评测运行。当前主机没有 Docker，构建上下文准备成功不等于镜像构建通过。
3. 实现并验收正式隔离部署：提交只能获得当前 HVP，不能读教师文件或未来系统；实际约束单进程、单线程和 256 MiB 峰值 RSS。
4. 在相同目标评测槽重跑两份参照，分别绑定反馈与私有集的锚点，核验参照强度。CG 映射为 0，L-BFGS-PCG 映射为 50，零 loss 映射为 100。
5. 检查参照至少 25% 时间余量、两次最终评分一致和分数不饱和。原参照的困难例会接近 30 秒，时间余量尚未满足；不能直接启用历史锚点。
6. 维护者确认全部发布条件后接通正式驱动并激活。缺少平台、数据或校准证据的结果保持“未测”，不能填成 0 分。

完整原题来源与历史聚合记录见 [PROVENANCE.md](PROVENANCE.md)。这些本地检查不替代目标平台性能校准或发布验收。
