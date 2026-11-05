# 新题整合状态（2026-11-05）

`Raw_Resource/` 中的 4 道新题已接入 FrontierAM 框架。原始包不进 git；`scripts/prepare_operator_tasks.py` 按 `tasks/operator/<题目>/task.json` 核对哈希、解压、校验包内清单、应用可移植性修改并封存到 `.runtime-ops/<题目>/`。评测时运行题目自带的评分器，通过 docker 垫片（`arena/docker_shim.py`）把它的 ARM64 平台、CPU 编号和镜像映射到本机评测 slot。

## 1. 各题概况

| 题目 | 原始包 | 包内清单校验 | 能做到的程度 |
| --- | --- | --- | --- |
| galaxy-gravity | operator 1.0.1 + public 1.0.1 | SHA256SUMS 82 个文件全部通过 | 完整评测（20 张私有快照，两条 Treecode2 参照） |
| polarization-demosaicking | maintenance v1.2 + public v1.2 | MANIFEST 199 个文件全部通过 | 完整评测（32 景，双线性和镜像 LEIC 两条参照） |
| species-tree | operator v1 + public v1 | SHA256SUMS 30 个文件全部通过 | 只能用 50 个公开开发用例检查评分链路和参照；私有正式数据未随包提供 |
| trajectory-covariance | 只有 public v1 | SHA256SUMS 2 个文件全部通过 | 只接入开发侧；没有 operator 包（无私有数据、参照、评分器） |

## 2. 默认决定（待讨论）

- **得分**：四道题自带的都是"基础参照 0 分、强参照 5 分、零误差 10 分"的分段映射。按 50/100 约定，最终分 = 原分 × 10；两个锚点在本机重跑两条参照后取实测误差，不用包内的历史值。
- **平台**：四个包都在 Apple M5 上的原生 ARM64 Docker 标定，本机是 amd64。评分器和 worker 不改逻辑，只做以下可移植性修改，每条都要求原文恰好出现一次：
  - galaxy-gravity：worker 的架构检查同时接受 `x86_64`。
  - polarization-demosaicking：worker 的 CPU 检查从"必须是 CPU 0"改为"恰好一个 CPU"。
  - species-tree：worker 的 CPU 检查从"必须是 CPU 0"改为"恰好一个 CPU"。
- **镜像**：包内只有 polarization 带了镜像，而且是 ARM64 的；galaxy、species-tree、trajectory 的运行镜像都没有随包提供。本机按以下方式重建 amd64 镜像（`tasks/operator/<题目>/Dockerfile.amd64`）：
  - polarization：按冻结镜像的构建历史重建，官方 `python:3.11.15-slim-bookworm`，加上从冻结镜像中原样拷出的 `requirements.lock`（torch 用 CPU 版）。
  - galaxy、species-tree：原基础镜像 `trajectory-runtime:b1` 和 `species-tree-cpu:b1` 未提供，以同一作者的锥优化镜像为基础（Debian bookworm、CPython 3.12.14、NumPy 2.3.5、gcc 12）。
  - trajectory：同上，另外安装题面承诺的 R。
- **宿主侧评分环境**：`.venv-ops`（Python 3.12.16，NumPy 2.3.5，与 polarization 维护包要求的宿主 NumPy 一致）。
- **评测隔离**：在实测干扰之前，四道题都先用 `grade_pool`（评测期间不与其他评测并发）。

## 3. 已发现的问题（给出题人）

- **trajectory-covariance**：只收到公开包，无法评测。
- **species-tree**：私有评分数据（`data-seal.json` 与 50 个正式用例）按 README 需另行转运，本次未收到。
- **galaxy-gravity**：
  - 维护包里混入了 `scripts/weighted_tree_qmc/step5/sandbox.py`（看名字属于 species-tree），但 galaxy 的评分器确实导入了它的 docker 辅助函数，属于正常依赖。
  - README 所说的运行镜像 `runtime/galaxy-b2.docker.tar` 不在包内。
  - 原机器上，标准参照单次调用最多用 3.09 个 CPU 秒，上限是 5 秒；本机若明显更慢，参照本身可能超限。
- **polarization-demosaicking**：每景墙钟上限 1.000 秒，包含 Python 启动和导入。维护包自己写明，本版镜像参照"尚未取得 1 秒端到端的完整 32/32 校准收据"，历史上纯双线性也只有 29/32。在本机可能同样无法通过。

## 4. 进度

（初步检查结果待补）
