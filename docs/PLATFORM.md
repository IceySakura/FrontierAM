# FrontierAM：两题离线评测迁移包

面向 Linux x86_64 服务器。包含固定版本 Nexus、本地扩展、指数传播题、更新后的优化求解器题（`conic-solver-20261102`，0.3.0 / success-v2）、公开数据和私有裁判数据。无需原来的 Mac 项目目录，也无需再次下载任务附件。

**候选程序和模型的工具执行环境均断网。** 宿主上的推理桥单独访问 AIHub。Nexus 负责运行和记录，Docker、独立工具入口及原优化器裁判的 Landlock/seccomp 负责隔离。模型只获得公开题包，不能访问本仓库的私有数据、参考实现或凭证。

## 服务器要求

- Linux x86_64、Docker Engine 及 Buildx 插件，使用本地 Unix socket；cgroup v2、systemd（cgroup driver 为 systemd）。
- Docker 可用内存至少 10 GiB；每个开发/评测 slot 需要一个带 SMT 兄弟线程的完整物理核。
- 内核须启用 Landlock ABI ≥ 3 和 seccomp（通常需要 Linux 6.2 及以上；以 `smoke` 实测为准）。缺少这些能力时拒绝运行，不降级为弱隔离。
- `uv`、Python 3.12、兼容的 Codex CLI；使用 npm 安装 CLI 时还需 Node.js/npm。部署和镜像构建阶段可以联网，正式工具/构建/求解阶段断网。

两题原版流水线已在 ARM64 和本 x86_64 服务器（AMD EPYC 7K62，32 vCPU / 62 GiB）上各完整通过一次 smoke；新的并发流水线也已在 x86_64 上通过。不将 ARM64 的耗时当作 x86_64 性能，baseline 必须在评分机器上重测。

## 首次安装

```bash
git clone git@git.woa.com:iceyghli/FrontierAM.git
cd FrontierAM

# 也可从离线 bundle 克隆：
# git clone frontieram-offline-bench.bundle FrontierAM

uv sync --project nexus --python 3.12.16 --locked --extra dev --extra runtime

# 安装服务器版 Codex；也可以使用已有的兼容版本。
npm install -g @openai/codex@0.166.0-alpha.5.1
codex --version

./scripts/build-images.sh
sudo ./bench host setup            # CPU 隔离、评测专用 Docker 守护进程、slot 布局（root，可重复执行）
./bench prepare --codex "$(command -v codex)"
./bench smoke --task both --parallel 2
```

Codex 安装采用[官方文档中的 npm 方式](https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted)，固定到本机验证版本 `0.166.0-alpha.5.1`；已确认 npm 提供该版本的 Linux x64 包。`prepare` 会检查所需启动选项、定位 npm 包里的原生可执行文件并记录版本及 SHA256；`smoke` 还会实际验证工具限制。更换 Codex 后须重新准备并验证，不能悄悄换版本继续同一轮。

非默认 Docker 地址可通过 `DOCKER_HOST=unix:///path/to/docker.sock` 设置。镜像默认 `linux/amd64`；`PLATFORM=linux/arm64 ./scripts/build-images.sh` 可构建 ARM64 版本。`prepare` 拒绝用跨架构模拟环境产生数值评测结果。

`prepare` 只生成被 Git 忽略的 `.runtime/`：本机路径、镜像 ID、Codex 哈希、复制后的执行代码与封存 profile。不会修改原题包。已有 `.runtime/` 时拒绝覆盖；更新环境前请保留已有结果，将旧 `.runtime/` 改名，再重新准备。

## 机器隔离：slot、守护进程与评测隔离策略

`./bench host setup` 把机器划成互不重叠的部分，写入 `/etc/frontieram/machine.json`（`./bench host plan` 只预览）：

- **housekeeping**（默认物理核 0–1）：systemd 下的 `system.slice`、`user.slice` 以及其余顶层 cgroup（监控 agent、编辑器、默认 dockerd/containerd）全部限制在这里，每 30 秒由 `frontieram-isolate.timer` 重新收紧新出现的顶层 cgroup。
- **`frontieram.slice`** 独占其余 CPU。每个 slot 是一个完整物理核：容器在一个硬件线程，另一个线程只运行该 slot 的宿主侧进程（开发控制器 Codex/Nexus，或评测计时进程），以 `systemd-run` scope 固定。
- **每个评测 slot 一套独立 Docker**：`frontieram-grade-containerd-<slot>` 加 `frontieram-grade-docker-<slot>`，运行在该 slot 的兄弟线程上。容器 stdio 经它们中继、位于计时路径上；共用一个守护进程时，它是 4 并发下最大的干扰源。开发的 `docker exec` 流量走默认 dockerd。
- `--grade-placement packed|spread` 决定评测核是否挤在同一 L3 域。

slot 租约是机器级的 `flock`（`/run/frontieram/slots/`），持有进程崩溃时自动释放；内存按容器上限计入机器预算，避免无 swap 的 OOM。`./bench host status` 和每个评测任务开始前都会复核隔离，失效时拒绝评测（记为未测）而非在争用下计时。

**并发不是假定安全的，而是实测的**：`./bench interference` 让参考实现（或 `--candidate` 给出的真实提交）在 1/2/4 个评测并发、可选全部开发 slot 满载内存带宽压力下重复评测并计分。指数传播的理想值恰为 50。共享硬件上的缓存和内存带宽争用无法分区，所以指数传播在每个用例上**配对测量**：同一 slot 上交替运行参考实现与候选，用同一时段的参考时间计分。每次评测还记录 **load factor**（配对参考时间 ÷ 安静 baseline），超过题目设定的闸门就自动在全机安静下重测。

| 策略 | 评测期间保持空闲 | 用于 |
| --- | --- | --- |
| `shared` | 无 | 指数传播（配对计分，load factor 闸门 1.04） |
| `grade_pool` | 所有其他评测 slot | 优化器（实测前的保守默认） |
| `quiet` | 所有其他评测 slot 和开发 slot（等待进行中的开发结束，暂停新的开发和评测） | 被闸门退回的重测 |

本机实测结论与数字见 `results/platform/CONCURRENCY.md`。

## 开发与评测是两个独立流程

```bash
export AIHUB_KEY_FILE=/secure/aihub.key     # 仓库外私有文件，权限 600

# 1. baseline：在评测 slot 上评测冻结参考实现，合格后自动激活为计分锚点
./bench baseline --task exponential
./bench baseline --task conic

# 2. 开发：每次尝试独占一个开发 slot；结束即冻结并进入评测队列，开发证据不进入评测
./bench develop --task exponential --model deepseek-v4-pro --count 2

# 3. 评测：按题目的隔离策略从队列取冻结源码评测，对照 baseline 打分
./bench grade-worker --drain
./bench sentinel --task exponential          # 参考实现复测，确认环境未漂移
./bench leaderboard --task exponential
./bench status
```

开发产物存放在 `store/dev-runs/`；只有冻结后的规范 `source.tar` 被复制到 `store/submissions/`，评测时用同一受审查的提取器重新校验后再解包。评测任务以原子重命名认领，同一提交不会被两个 worker 评测；崩溃的 worker 由 `./bench recover` 重新入队并保留残留证据。`./bench submit --source DIR` 可把任意源码目录作为提交入队；`./bench rescore` 在更换 baseline 后用已存的原始测量重算分数。

可用模型字段：`claude-opus-5-5`、`deepseek-v4-pro`、`hy4-preview`、`minimax-m3.1-flash`、`step-5-preview`。

默认网关为 `http://api.aihub.woa.com/standard/v1`，可用 `--base-url` 修改。每次是一个完整 agent 开发会话，可以多次调用工具；“一次尝试”不是只发一个 HTTP 请求。开发上限 7200 秒（与指数传播原题的两小时开发时限一致）、800 次实际 HTTP 请求（含重试和上下文压缩请求），每次响应最多 64000 输出 token，context setting 为 131072，约 110000 token 时由 Codex 自动压缩上下文。网络等待和重试消耗同一原始期限；提前结束或到时后立即冻结，不续跑、不替模型修代码。

开发资源统一为单核、4 GiB、禁用 swap、32 个进程/线程、128 个 fd、256 MiB 临时目录、1 GiB 工作目录。数值评测使用题目各自的原定限制：指数传播 4 GiB；优化器 8 GiB，包含裁判开销。

原题中的“绑定 CPU0”在本仓库表示“绑定该 slot 的唯一 CPU”：冻结代码中的 CPU 编号改为参数，评测语义不变；改动与新旧哈希记录在 `provenance/TRUSTED_CODE_RELOCKS.json`，由 `scripts/relock.py` 维护。prompt 附注告知模型 CPU 编号按次分配、不得写死。

## 新增题目

在 `arena/tasks.py` 实现一个 `Task` 并注册到 `TASKS`：开发配置（公开题包与工作区 runtime）、`grade()`（只接收重新校验后的冻结源码，必须把 `Placement` 传给它创建的每个容器）、`reference_archive()`（冻结参考实现）、`metrics()`、`score(candidate, baseline)`，以及由 `./bench interference` 实测决定的 `grade_isolation`。slot、队列、baseline 管理、隔离复核与排行榜不需要改动。

## 最终得分约定（所有题目）

所有题目的最终得分统一为**满分 100、参考实现 = 50**：

- **参考在本机重跑**：参考实现必须在本机上重新运行，作为计分锚点（baseline），不使用历史记录或其他机器上的数字。
- **映射的三个锚点**：题目原始指标经单调映射得到最终得分，参考的成绩映射为 50，题目允许的最好成绩映射为 100，无效提交为 0。
- **已有映射的题目**：如果题目自带以参考为锚点的分数（例如优化器原题 0–10 分、参考 5 分），直接线性换算到 100 分。
- **没有映射的新题**：出题时就要定好映射公式，并写进结果报告。
- **报告内容**：每份结果报告同时给出原始指标、映射公式和最终得分；基础设施失败记为未测，不记 0 分。

## 两题的结果口径

| 题目 | 评测范围 | 最终得分（满分 100，参考 = 50） |
| --- | --- | --- |
| 指数传播 | 固定 23 个会话，每个 3 次，共 69 次验解 | 100 / (1 + R)，R 为相对参考的时间比（配对测量）；同时报告正确性、相对耗时、CPU、内存和失败原因 |
| 优化求解器 | 全部 650 个留出实例，每例 3 次；时限按题目规则用本机参考运行重新标定 | 成功率 p 按原题映射：p ≤ 0.9·p_B 为 0，p_B 处为 50，全部解出为 100（原题 0–10 分 × 10）；见 `results/conic/CONIC_FULL.md` |

优化器完整 650 例及原始 `candidate` 裁判都保存在 `tasks/conic/original/`。全集评测通过原裁判的用例列表模式分片运行，裁判代码不改；本机重标定只改时限，见 `tasks/conic/local_recal/`。题目包自带的 20 例平衡子集只用于快速探针，不能当作 650 例分布的无偏总分。

基础设施失败记为未测，不能当作算法 0 分；构建/提交无效按题意另记 0 分。所有原始记录、源码快照和失败尝试均保留。

## 仓库结构与验证记录

- `nexus/`：上游固定提交及两题扩展；源提交见 `provenance/nexus.json`。
- `tasks/`：原始题包、公开数据、私有数据及固定参考。仅运行方可访问完整仓库。
- `tasks/conic/reference_submission/`：优化器私有参考提交（冻结 Clarabel 0.11.1 实现），原始文件与 `standard-calibration.json` 记录的哈希逐一相同；另补同版本 Linux x86_64 wheel，来源与校验见 `tasks/conic/REFERENCE_PROVENANCE.json`。尚未在本机重跑校准。
- `docker/`、`scripts/`、`bench`：架构对应镜像、准备、校准、验证和运行入口。
- `fixtures/`：仅用于连通性回放的冻结提交，不进入真实模型的工作区。
- `results/platform/VALIDATION.md`：本次验证结论、边界和已知限制。

已修复开发到时的双重冻结竞态：控制器与工具 watchdog 同时冻结时，只在核验同一容器已经冻结、且隔离配置仍正确后接受“已冻结”结果；替换容器或实际未冻结仍报错。测试覆盖正常结束、到时、重试、输出截断、未知工具、清理与隔离边界。

维护者可在 `nexus/` 下执行相关回归测试：

```bash
uv run --no-sync pytest tests/unit/test_frontieram*.py \
  tests/unit/extensions/tasks/test_frontieram_runtime.py -q
```

部署准备联网与候选环境断网是不同阶段。整个原题目录和 API 凭证都只保留在可信运行方，不得直接挂载到候选容器。不要把此含私有裁判数据的仓库交给做题 agent。
