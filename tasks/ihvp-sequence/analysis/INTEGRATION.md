# 接入验证与维护者交接

2026-11-11，GitHub 出题方接入，状态为 **draft**。本次包含题面、样例、生成与准备脚本、局部评测器、两份参照、任务注册和页面生成；不包含私有数据，不开放正式模型尝试。

## 已验证

- 本地使用独立环境：Python 3.13.9、NumPy 2.3.5、SciPy 1.16.3、psutil 7.1.3、threadpoolctl 3.6.0。
- IHVP 与反馈相关测试 **75 passed**；完整 `arena/tests` 结果为 **85 passed, 3 failed**。失败均在已有的 `test_slots_and_store.py`：本机没有 Docker，macOS 子进程仍调用真实 Docker。未据此修改旧题或槽位逻辑。
- 独立样例工具仅携带 `data/samples/`，从无仓库路径的工作目录运行，能够验证完整包；零解在 smoke 和两条完整规模序列上均合法，原始 loss 为 1，分数均为“未测”。小规模 HVP 重建精确解的 loss 约为 `9.8e-33`。缺失样例正文和预测溢出会被拒绝。
- 两条完整样例由原反馈池选取，生成后按源数组哈希核对并复核参考解。维护端准备脚本将原始 15 条反馈与 15 条私有数据留在忽略目录；不向 GitHub 写入正文、答案或私有种子。
- 离线构建上下文已在忽略目录物化：19 个文件、4 个精确锁定的 Linux amd64 / CPython 3.12 兼容 wheel，共 96,849,110 字节；逐文件 SHA 校验和独立样例包校验通过。Dockerfile、题面和依赖锁与当前源码一致。这是构建材料校验，尚未实际构建或运行 Docker 镜像。
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

## FrontierAM 接入（2026-11-11，维护端）

已完成：

- 任务镜像 `frontieram-ihvp-sequence:amd64-v1`：按 PR 的 Dockerfile 离线构建（四个锁定 wheel 从镜像源下载，基础镜像自带 NumPy 2.3.5 / SciPy 1.16.3），镜像内 `check.py --verify-package` 通过；已复制到 7 个评测守护进程。开发镜像 `frontieram-ihvp-sequence-harness:v6` 在其上构建。
- 封签 `scripts/seal/seal_ihvp_sequence.py`：开发工作区（题面 + 样例）与数据根目录写入 `.runtime-ops/ihvp-sequence/lock.json`。样例在本机由生成器重建，逐位哈希与源环境不同（BLAS/FFT 舍入，生成说明已预期），参考解复核通过。
- 隔离评测（`arena/operator_drivers/ihvp_sequence.py`）：出题方评测器在任务镜像内运行，断网、绑定一个评测核；评测器以 root 读取只对 root 可读的教师数据（目录 0700），提交的 worker 以 UID 1000 启动（`runtime/api.py` 的 `IHVP_WORKER_UID`，默认关闭），只能通过管道拿到当前右端项与 HVP 回复。256 MiB 峰值 RSS 仍由评测器检查，容器内存上限 4 GiB 作兜底。
- v6.3 接线：反馈适配（按五类场景分类，`arena/public_v63.py`）、开发镜像与计分说明（`arena/native.py`）、演练脚本。
- 单元测试：完整 `arena/tests` 93 项通过。

两条样例上的参照实测（评测核 grade-06，单次）：

| 样例 | 冷启动 CG：loss / 墙钟 | L-BFGS-PCG：loss / 墙钟 | 峰值 RSS |
| --- | --- | --- | --- |
| sample_01（宽谱平滑，条件数 1e3） | 2.1e-17 / 15.7 秒 | 3.9e-17 / 23.5 秒 | 65 MiB |
| sample_02（阶段突变，条件数 1e5） | 1.7e-3 / 28.5 秒 | 8.2e-5 / 28.5 秒 | 76 MiB |

待出题人决定或提供：

1. 反馈集与私有集（各 15 条）的原始题包；或同意用自带生成器以维护端私有种子重新生成（按 PR 说明属于新版本）。
2. 时间余量：两份参照按剩余时间分配计算，困难序列用到约 28.5 秒（预留 1.2 秒清理），25% 余量的门槛对这类“用满预算”的参照不适用；建议改为“重复评测无超时”。
3. 计分饱和：简单序列两份参照都解到机器精度，分数由困难序列决定；sample_02 上两份参照差 20 倍，误差跨数量级，而 50 分以上为线性映射 \(100-50E/E_{50}\)，误差比参照小 10 倍即得 95 分。可考虑对数映射（galaxy-gravity-v3 的做法）。
4. 题面第 105 行“本次接入为草稿……”在开放模型尝试时需要改写；`task.json` 的 instruction 同理。

## 开放（2026-11-11）

- 出题人题包收到（1,026,377,847 字节，SHA-256 与 PROVENANCE 一致），用 `prepare_ihvp_sequence.py --source-bundle` 导入 15 条反馈序列与 15 条私有序列（两池无重叠，样例配方与反馈池吻合），数据目录 0700，只在维护端。
- 私有集锚点（grade-06，quiet）：冷启动 CG \(E_0 = 0.1213\)，L-BFGS-PCG \(E_{50} = 0.0543\)（出题方 M3 Max 记录 0.1071 / 0.0468）；最长序列 28.5 秒，峰值 RSS 79 MiB。public 反馈锚点：0.1215 / 0.0581。
- \(E\) 几乎全部来自 3 条条件数 1e7 的困难序列（L-BFGS-PCG 在其上 0.24–0.32，其余 12 条 ≤ 7e-5）；困难序列若降到约 0.01，\(E \approx 0.002\)，线性映射约 98 分。饱和风险已列入待出题人决定。
- 反馈的“你 ÷ 参照”比值在简单类别上失去意义（两份参照都到机器精度，比值可达 1e16）；本轮不改反馈格式，记为观察项。
- `task.json` 状态改为 formal；题面第 105 行与 instruction 中的草稿说明已改写（任务镜像内的 `/workspace/problem.md` 是旧文本，开发工作区与评测都不使用它）。
- 迭代演练通过（异步测试、状态、反馈报告、不含用例编号）；gpt-6.1-sol 单轮与迭代尝试 11-11 13:03 / 13:14 开始。
