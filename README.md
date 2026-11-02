# FrontierAM：两题离线评测迁移包

面向 Linux x86_64 服务器。包含固定版本 Nexus、本地扩展、指数传播题、更新后的优化求解器题（`conic-solver-20261102`，0.3.0 / success-v2）、公开数据和私有裁判数据。无需原来的 Mac 项目目录，也无需再次下载任务附件。

**候选程序和模型的工具执行环境均断网。** 宿主上的推理桥单独访问 AIHub。Nexus 负责运行和记录，Docker、独立工具入口及原优化器裁判的 Landlock/seccomp 负责隔离。模型只获得公开题包，不能访问本仓库的私有数据、参考实现或凭证。

## 服务器要求

- Linux x86_64、Docker Engine 及 Buildx 插件，使用本地 Unix socket；cgroup v2。
- Docker 可用内存至少 10 GiB；数值评测必须能绑定 CPU0。
- 内核须启用 Landlock ABI ≥ 3 和 seccomp（通常需要 Linux 6.2 及以上；以 `smoke` 实测为准）。缺少这些能力时拒绝运行，不降级为弱隔离。
- `uv`、Python 3.12、兼容的 Codex CLI；使用 npm 安装 CLI 时还需 Node.js/npm。部署和镜像构建阶段可以联网，正式工具/构建/求解阶段断网。

本机已完成 ARM64 上的两题完整验证。x86_64 镜像另行构建，目标服务器仍必须执行下面的 `smoke`；不将模拟运行或 ARM64 的耗时当作 x86_64 性能验证。

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
./bench prepare --codex "$(command -v codex)"
./bench smoke --task both --output outputs/first-smoke
```

Codex 安装采用[官方文档中的 npm 方式](https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted)，固定到本机验证版本 `0.166.0-alpha.5.1`；已确认 npm 提供该版本的 Linux x64 包。`prepare` 会检查所需启动选项、定位 npm 包里的原生可执行文件并记录版本及 SHA256；`smoke` 还会实际验证工具限制。更换 Codex 后须重新准备并验证，不能悄悄换版本继续同一轮。

非默认 Docker 地址可通过 `DOCKER_HOST=unix:///path/to/docker.sock` 设置。镜像默认 `linux/amd64`；`PLATFORM=linux/arm64 ./scripts/build-images.sh` 可构建 ARM64 版本。`prepare` 拒绝用跨架构模拟环境产生数值评测结果。

`prepare` 只生成被 Git 忽略的 `.runtime/`：本机路径、镜像 ID、Codex 哈希、复制后的执行代码与封存 profile。不会修改原题包。已有 `.runtime/` 时拒绝覆盖；更新环境前请保留已有结果，将旧 `.runtime/` 改名，再重新准备。

## 先验证，再评分

`smoke` 不调用付费模型。它使用本地协议夹具驱动**真实 Codex → Nexus → 离线工具 → 冻结提交 → Docker 裁判**，验证指数传播 69 次验解和优化器 20×3 次裁判执行。优化器夹具故意返回 `unresolved`，预期成功率为 0，同时逐次检查隐藏数据读取、裁判改写、源码改写和 TCP/UDP 均被禁止；这不是模型得分。

指数传播的时间参考必须在目标机器重新校准：

```bash
./bench calibrate --task exponential --output outputs/reference-calibration
```

校准使用已冻结的作者 KIOPS 实现及统一 `0.01` 容差倍率，不按私有实例重新选算法或调参。只有参考通过全部 69 次验解，才建立本机时间锚点。未校准时仍能验解和记录耗时，但不生成指数传播分数。更换机器、镜像、数据或校准报告会使校准失效。

## 跑一个模型的一次 30 分钟尝试

将 AIHub key 放到仓库外的私有文件，权限设为 `600`。仓库不携带任何 API key 或太湖 PAT。

```bash
export AIHUB_KEY_FILE=/secure/aihub.key

./bench run --task exponential --model claude-opus-5-5 \
  --output outputs/exponential-opus-attempt1

./bench run --task conic --model claude-opus-5-5 \
  --output outputs/conic-opus-attempt1
```

可用模型字段：`claude-opus-5-5`、`deepseek-v4-pro`、`hy4-preview`、`minimax-m3.1-flash`、`step-5-preview`。

默认网关为 `http://api.aihub.woa.com/standard/v1`，可用 `--base-url` 修改。每次是一个完整 agent 开发会话，可以多次调用工具；“一次尝试”不是只发一个 HTTP 请求。开发上限 1800 秒、200 次实际 HTTP 请求（含重试），每次响应最多 64000 输出 token，context setting 为 131072。网络等待和重试消耗同一原始期限；提前结束或到时后立即冻结，不续跑、不替模型修代码。

开发资源统一为单核、4 GiB、禁用 swap、32 个进程/线程、128 个 fd、256 MiB 临时目录、1 GiB 工作目录。数值评测使用题目各自的原定限制：指数传播 4 GiB；优化器 8 GiB，包含裁判开销。

若目标服务器安排多个开发工作槽，可用 `--develop-only` 先冻结，再串行验分：

```bash
./bench run --task exponential --model deepseek-v4-pro --develop-only \
  --output outputs/deepseek-development
./bench grade --task exponential \
  --submission outputs/deepseek-development/task/source \
  --output outputs/deepseek-grade
```

一个 Docker host 上的当前 profile 都绑定 CPU0；不要在同一 CPU0 上同时开发和计时验分。需要并行时，为工作槽使用独立 Docker host/profile，保留资源配置记录；数值验分串行执行。

## 两题的结果口径

| 题目 | 本迁移入口的评测范围 | 输出 |
| --- | --- | --- |
| 指数传播 | 固定 23 个会话，每个 3 次，共 69 次验解 | 本机参考校准后 0–100 分；正确性、相对耗时、CPU、内存和失败原因 |
| 优化求解器 | 新版固定 20 例探针，五类各 4 例，每例 3 次 | 成功数/成功率、分类结果、实际耗时；不套全量分数 |

优化器完整 650 例及原始 `candidate` 裁判都保存在 `tasks/conic/original/`。当前 Nexus profile 是先前选定的低成本 20 例探针。它保留新版每例 1–300 秒时限、1e-8 验解和三次状态一致条件；不再计算旧版综合时间 `G`。20 例为平衡子集，不能当作原 650 例分布的无偏总分。全量原题的映射仍是 0–10、成功率唯一计分；参考参数为历史记录重分类，未声称已在新机器重跑校准。

基础设施失败记为未测，不能当作算法 0 分；构建/提交无效按题意另记 0 分。所有原始记录、源码快照和失败尝试均保留。

## 仓库结构与验证记录

- `nexus/`：上游固定提交及两题扩展；源提交见 `provenance/nexus.json`。
- `tasks/`：原始题包、公开数据、私有数据及固定参考。仅运行方可访问完整仓库。
- `docker/`、`scripts/`、`bench`：架构对应镜像、准备、校准、验证和运行入口。
- `fixtures/`：仅用于连通性回放的冻结提交，不进入真实模型的工作区。
- `results/VALIDATION.md`：本次验证结论、边界和已知限制。

已修复开发到时的双重冻结竞态：控制器与工具 watchdog 同时冻结时，只在核验同一容器已经冻结、且隔离配置仍正确后接受“已冻结”结果；替换容器或实际未冻结仍报错。测试覆盖正常结束、到时、重试、输出截断、未知工具、清理与隔离边界。

维护者可在 `nexus/` 下执行相关回归测试：

```bash
uv run --no-sync pytest tests/unit/test_frontieram*.py \
  tests/unit/extensions/tasks/test_frontieram_runtime.py -q
```

部署准备联网与候选环境断网是不同阶段。整个原题目录和 API 凭证都只保留在可信运行方，不得直接挂载到候选容器。不要把此含私有裁判数据的仓库交给做题 agent。
