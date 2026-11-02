# Batch 指南

完成 [Single 快速开始](quickstart.md)后，可以用 Batch 处理一组输入。本页介绍 Batch 的提交、并发、资源和结果管理。
需要持续交互的任务见 [Live 指南](live.md)。

首次提交自己的任务、Harbor 数据集或新 scaffold，可先完成[提交前本地验证](local-validation.md)。
该页提供从任务输入、任务编译、Nexus wheel 到依赖 wheelhouse 的完整脚本；验证后回到本页配置提交、并发和资源。

## 提交 Batch

Batch 会为每条输入创建一个 Single 子任务，并负责并发调度、失败重试和结果汇总。子任务可以通过两种方式提供：

| 方式 | 适合场景 | 本页示例 |
| --- | --- | --- |
| Generator | 需要用 Python 动态生成配置 | 生成 5 个 `identity + main_agent` 子任务 |
| Dataset＋Profile（Compose） | 已有结构化数据集，需要为每条数据应用同一份运行配置 | 运行 5 个 SWE-bench Verified 子任务 |

开始前应已完成[快速开始](quickstart.md)，具有有效 PAT、目标 Dynamic Environment 配额和模型凭据。
两个示例任选一个；所有命令在同一个 Bash 终端执行。以下以正式环境为例：

```bash
set -euo pipefail

export CONTROL_BASE_URL='https://control-runner-nexus-agent.mcp.woa.com'
export TAIHU_TOKEN_FILE="$HOME/.nexus/auth/taihu-pat"
export NEXUS_ENVIRONMENT='<dynamic-environment-id>'
export MODEL_SERVICE_API_APP_ID='<app-id>'
export MODEL_SERVICE_API_APP_KEY='<app-key>'
export MODEL_SERVICE_API_BASE_URL='<openai-compatible-base-url>'
export NEXUS_MODEL='gpt-5.5'
```

为所选示例准备一次输出目录和幂等键。重试同一个提交时复用它们；切换示例、输入或配置时，这是一个新任务，应重新准备：

```bash
umask 077
OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nexus-batch.XXXXXX")"
IDEMPOTENCY_KEY="batch-$(openssl rand -hex 16)"
printf '%s\n' "$IDEMPOTENCY_KEY" > "$OUTPUT_DIR/idempotency-key"
```

### 示例一：Generator

Generator 是一个普通 Python 文件，通过 `generate_configs()` 逐条返回 `BatchItem`。示例
[batch_generator.py](../examples/configs/online-service/batch_generator.py) 复用
[main-agent-single.yaml](../examples/configs/online-service/main-agent-single.yaml)，生成 5 条具有唯一 `run_id` 和提示词的任务：

```python
from pathlib import Path

from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


BASE = RunInput.from_file(Path(__file__).with_name('main-agent-single.yaml'))


def get_total_count():
    return 5


def generate_configs():
    for index in range(5):
        run_id = f'online-main-agent-{index + 1:02d}'
        config = BASE.model_copy(
            deep=True,
            update={
                'run_id': run_id,
                'agent_config': {
                    **BASE.agent_config,
                    'user_prompt': f'Reply with exactly NEXUS_BATCH_ITEM_{index + 1:02d}.',
                },
            },
        )
        yield BatchItem(run_id=run_id, config=config)
```

<a id="提交前预览-task-list"></a>

#### 提交前本地验证

任务编译及后续依赖 wheelhouse 校验统一见[提交前本地验证](local-validation.md)，Generator 输入的切换方式见
[Generator 本地验证](local-validation.md#generator)。正式提交仍使用原始 `--generator` 或 Dataset/Profile 输入，
不直接传入本地验证生成的 Task List、Data List 或 wheelhouse。

#### 提交 Batch

`BatchItem.metadata` 会完整保留，不自动截断。本地 `nexus materialize` 和
`nexus submit batch --generation local` 默认允许每条 metadata 最大 **256 KiB**，
按紧凑 JSON 的 UTF-8 字节数计算。可以用正整数环境变量覆盖（单位为字节）：

```bash
export NEXUS_SUBMIT_MAX_BATCH_ITEM_METADATA_BYTES=262144
```

服务端独立使用分布式配置 `batch_item_metadata_max_size_bytes`，默认 **131072 字节（128 KiB）**。
任务列表读取、服务端远程生成和子任务创建使用这一配置，不继承提交端的上限。
本地生成成功不代表服务端会接受；例如默认设置下，200 KiB 的 metadata 可通过本地生成，
但会被服务端拒绝。需要更大上限时，应先升级并配置服务端，再调整提交端。
整个批次的 `get_batch_metadata()` 仍保持 64 KiB 上限。

```bash
uv run nexus submit batch \
  --generator docs/examples/configs/online-service/batch_generator.py \
  --generation local \
  --environment "$NEXUS_ENVIRONMENT" \
  --max-inflight 20 \
  --control-url "$CONTROL_BASE_URL" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --timeout-per-run 600 \
  --wait \
  --json > "$OUTPUT_DIR/submission.json"

jq '{execution_id, batch_id, terminal, batch_url}' "$OUTPUT_DIR/submission.json"
```

这里使用 `--generation local`：Generator 在本地生成 Task List，真正的 5 个子任务仍在云端执行。第一次编写 Generator 时推荐使用
该模式，可以在上传前发现导入或配置错误。

<a id="示例二dataset--swe"></a>

### 示例二：Dataset＋Profile（SWE）

已有 Dataset 时，不需要再编写 Generator。Nexus Dataset 的每个 item 都是一个可合并的 RunInput 片段；与 Profile 的组合称为 Compose。提交时，Nexus Cloud
会把 item 和公共运行配置组合成完整的 Single 子任务。

示例 Dataset 位于
[online-service-swe-verified-5](../examples/datasets/online-service-swe-verified-5/dataset.yaml)，包含 5 个
SWE-bench Verified instance。先确认数据：

```bash
uv run nexus dataset items \
  docs/examples/datasets/online-service-swe-verified-5 \
  --limit 5
```

每个 item 只声明该题目的任务字段，例如：

```yaml
id: astropy__astropy-12907
body:
  task_name: swe_task
  task_config:
    remote_dataset: swe-public/SWE-bench_Verified
    instance_id: astropy__astropy-12907
    run_oracle: false
    raise_on_runtime_error: true
```

公共的 Runtime、Agent、模型和 Processor 配置放在
[swe-batch.yaml](../examples/configs/online-service/swe-batch.yaml)。提交时先应用 Dataset item，再应用公共配置。

#### 提交前本地验证

可以先按[常规 Compose 本地验证](local-validation.md#常规-compose)，检查代表任务的任务编译结果和依赖 wheelhouse。
验证与提交使用相同 Dataset 和 Profile，任务数量可从 1 逐步扩大。

#### 提交 Batch

```bash
export GONGFENG_RUNTIME_TOKEN='<runtime-token>'

uv run nexus submit batch \
  --data docs/examples/datasets/online-service-swe-verified-5 \
  --use docs/examples/configs/online-service/swe-batch.yaml \
  --limit 5 \
  --generation local \
  --environment "$NEXUS_ENVIRONMENT" \
  --max-inflight 5 \
  --control-url "$CONTROL_BASE_URL" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --timeout-per-run 7200 \
  --wait \
  --json > "$OUTPUT_DIR/submission.json"

jq '{execution_id, batch_id, terminal, batch_url}' "$OUTPUT_DIR/submission.json"
```

`--limit 5` 表示只取前 5 个 item。外层 Nexus Dataset 决定运行哪些题目，每条 `swe_task` 再从
`swe-public/SWE-bench_Verified` 读取完整题目内容。

!!! note "Dataset 格式"
    `--data` 接受 Nexus `DatasetItem`，不是任意原始 JSON。每条 item 至少包含 `id` 和 `body`；`body` 会按照命令行顺序与
    后续 `--use` 配置合并。

### 调整并发与 Manager 资源

使用 `--max-inflight` 调整整个 Batch 的 child 并发：它限制已提交但尚未结束的 child 数量。通常只需要设置这个参数，
Dynamic Environment 会自动计算需要的 Execution Manager 数量。

需要手工控制 Execution Manager 布局或单实例资源时再设置：

- `--manager-count`：Execution Manager 的数量。
- `--manager-slots`：单个 Execution Manager 最多同时执行多少个 child。
- `--manager-cpu-cores`：每个 Execution Manager 分配的 CPU core 数。
- `--manager-memory-gib`：每个 Execution Manager 分配的内存，单位为 GiB。

`--manager-count` × `--manager-slots` 是 Manager 总容量，必须不小于 `--max-inflight`。例如使用 2 个 Manager、每个 20 slots，
每个分配 2 cores、4 GiB，并把 Batch 并发限制为 30：

以下为附加到提交命令的参数片段：

```text
--manager-count 2 \
--manager-slots 20 \
--manager-cpu-cores 2 \
--manager-memory-gib 4 \
--max-inflight 30
```

CPU 和内存参数都是单个 Execution Manager 的规格；Execution Manager 总用量等于单实例规格乘以 `--manager-count`。slots 是
逻辑并发容量，提高 slots 不会自动增加 CPU 或内存。Batch Manager 使用独立的 `--batch-manager-cpu-cores` 和
`--batch-manager-memory-gib`，通常不需要调整，也不计入 child 执行容量。

每个 Manager 的内存不能超过 CPU 核数的 4 倍（GiB），例如 `2 cores / 8 GiB` 合法，
`2 cores / 16 GiB` 在自动加配关闭时拒绝。添加 `--auto-increase-cpu` 可在超出比例时自动将 CPU 提升到
`ceil(memory_gib / 4)` 个整核：`2 cores / 16 GiB` 变成 `4 cores / 16 GiB`，
`2 cores / 17 GiB` 变成 `5 cores / 17 GiB`。内存、Manager 数量和 slots 保持不变，合法配比不调整。
该选项也适用于 `submit single` 和 `submit live`；Batch 中同时作用于 Batch Manager 和 Execution Manager，
不能与固定池 `--pool` 一起使用。配额检查和实际容器创建均使用调整后的 CPU。

使用 `--no-auto-increase-cpu` 可显式关闭自动加配。两者都不传时，由服务端配置
`submission_auto_increase_cpu` 决定（默认 `true`）；服务端可通过 Control 配置将其设为 `false`。
显式传入的提交选项优先于服务端默认值。显式关闭时，本地在打包前拒绝不合法配比；
未传选项时保留原始申请量，由服务端按当前默认值处理。

直接调用提交 API 时，在 `resources.managers` 或 `resources.batch_manager` 对象中设置
`"auto_increase_cpu": true` / `false`；省略或 `null` 表示使用服务端默认值。
服务端同样校验比例：有效选项为关闭时超出比例会拒绝。该设置仅用于新提交，不调整已有 Manager；
同一幂等键重试仍返回已接收的任务，不因服务端默认值变化而重新加配。

没有特殊资源布局要求时，只设置 `--max-inflight`，不要传其他 Manager 参数。最终申请量不能超过个人 Environment Grant；
SWE 任务使用的沙盒资源另行计算，不由这些参数控制。

### 重试与停止

- `--retry-failed <count>` 设置失败 child 的重试次数，默认 0；同一个 item 的各次尝试可在 Batch 详情中查看。
- `--fail-fast` 用于遇到失败时提前停止剩余任务，默认关闭。
- 停止整个批次时，在控制台停止 Batch parent；关闭终端或中断 `--wait` 不会停止远端批次。

这里的 child 重试与网络超时后的提交重试不同：网络重试复用提交 `IDEMPOTENCY_KEY`，child 重试由 Batch 管理。
具体状态与停止操作见[任务与结果](runs-and-results.md#停止任务)。

### 查看结果

响应保存在 `$OUTPUT_DIR/submission.json`。`--wait --json` 会返回 `execution_id`、`batch_id`、`batch_url`、`result_url`，并在 parent 终态后附加 `terminal`。以
`terminal.status` 判断最终结果。在控制台的 **Batches** 中可以查看每个 child 的独立状态和结果。详细读取与停止方式见
[任务与结果](runs-and-results.md)。

## 代码与运行环境

### 上传代码与依赖

Dynamic 提交默认会：

1. 上传项目代码、当前 Nexus wheel 和 `uv.lock`。
2. 按锁文件准备运行依赖。
3. 依赖就绪后启动 Batch Manager 和 Execution Manager。

source-package 模式不是任意 Python 项目打包器。`--project-root` 与 `--source` 都必须指向一个完整 Nexus 源码项目：

- 根目录存在 `pyproject.toml`，且 distribution 名是 `nexus-agent`；
- 根目录存在有效 `uv.lock`，`uv lock --check` 必须通过；
- source 根目录的 Nexus 版本必须与本次构建的 wheel 一致；
- `pyproject.toml` 必须定义 Cloud 固定使用的 `distributed` 和 `runtime` extras。

通常让二者指向同一个 Nexus checkout，并把自定义 task/agent/module 放在该项目树内：

```text
--project-root /data/projects/nexus \
--source /data/projects/nexus
```

`--project-root` 用于构建 Nexus wheel 并读取 `uv.lock`；`--source` 是上传的 source package 根目录。把 `--source` 指向 Nexus
checkout 的父目录会在云端报 `dependency_source_project_invalid`；两个目录版本不一致会报 `dependency_input_mismatch`。

#### 模块依赖

Cloud 会根据任务实际使用的 Nexus 模块补齐它们声明的 Host 依赖，不需要用户在运行中的 Manager 里手工安装。依赖必须与当前
`uv.lock` 兼容；版本冲突时提交会在依赖构建阶段失败。

同一 Batch 的 child 应使用相同的模块依赖。不同 item 需要不同依赖时，应拆成多个 Batch，或先把依赖并集声明进项目。
完整本地诊断流程见[提交前本地验证](local-validation.md)，参数与报告字段见
[Module CLI](../how-to/manage-modules/module-cli.md#本地构建模块-wheelhouse)。

Generator 的 `--generation remote` 会把 Generator 放到云端执行，适合数据只能从云端访问的场景。远程 Nexus Dataset 需要 WeData
认证时使用 `--wedata-key-file ~/.nexus/auth/wedata/key`。

### 使用自定义镜像

项目依赖较大或希望复用已经构建好的运行环境时，可以将代码和依赖预装进平台 workload 镜像仓库，再通过
`--workload-image-version` 提交。使用前请联系 Oncall 同学获取平台允许使用的 `repository` 和 `base_image`，并开通仓库推送
权限。`repository` 是唯一允许推送的 workload 仓库，`base_image` 是必须使用的基础镜像 digest。

用户镜像必须基于平台给出的 `base_image` 构建，保留 `/opt/nexus-venv` 和 `/workspace`，并预装任务所需的全部代码与依赖。
使用不可变 tag，不要使用 `latest`；CLI 只接收 tag，不接收完整镜像地址。平台使用固定方式启动 Manager，不执行用户镜像的
`ENTRYPOINT` 或 `CMD`。

最小 Dockerfile 可以写成：

```dockerfile
ARG NEXUS_WORKLOAD_BASE
FROM ${NEXUS_WORKLOAD_BASE}

WORKDIR /workspace
COPY . /workspace
RUN /opt/nexus-venv/bin/pip install --no-cache-dir -e .
```

构建并推送到平台分配的仓库：

```bash
NEXUS_WORKLOAD_BASE='<workload.base_image>'
NEXUS_WORKLOAD_REPOSITORY='<workload.repository>'
NEXUS_WORKLOAD_VERSION='my-project-20260804-01'

docker build \
  --build-arg NEXUS_WORKLOAD_BASE="$NEXUS_WORKLOAD_BASE" \
  -t "$NEXUS_WORKLOAD_REPOSITORY:$NEXUS_WORKLOAD_VERSION" .
docker push "$NEXUS_WORKLOAD_REPOSITORY:$NEXUS_WORKLOAD_VERSION"
```

每次提交新镜像任务前，按本页开头准备新的 `OUTPUT_DIR` 和 `IDEMPOTENCY_KEY`；网络重试复用原值。
提交时只传 tag：

```bash
uv run nexus submit batch \
  --generator docs/examples/configs/online-service/batch_generator.py \
  --generation local \
  --workload-image-version "$NEXUS_WORKLOAD_VERSION" \
  --environment "$NEXUS_ENVIRONMENT" \
  --max-inflight 20 \
  --control-url "$CONTROL_BASE_URL" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --wait \
  --json > "$OUTPUT_DIR/submission.json"
```

镜像模式不会上传 source package、Nexus wheel 或 `uv.lock`，也不会创建 dependency builder。它不能与 `--config-only`、`--source`
或 `--wheel-extra` 同时使用，也不会自动补齐模块 Host 依赖；所有代码与依赖必须已经在镜像内。若使用远程 Generator，Generator
文件也必须已经位于镜像的 `/workspace` 中。

### 使用 Predefined Environment

Predefined Environment 不接受代码上传，必须使用已有 pool，并继续用 `--max-inflight` 控制 Batch 并发：

以下为参数片段，替换占位值后附加到提交命令：

```text
--environment <environment-id> --pool <pool-id> --config-only --max-inflight <count>
```

`--config-only` 只提交任务配置，使用 Manager 镜像中已经安装的代码和依赖，也不会自动补齐模块 Host 依赖；它不适用于本页需要
上传示例代码的 Dynamic 流程。
Predefined 不申请或推导 Manager；`--max-inflight` 不能超过 Environment 和个人 Grant 的有效上限。
