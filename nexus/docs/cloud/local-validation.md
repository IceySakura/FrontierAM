# 提交前本地验证

第一次接入 Batch 任务、更换 Agent scaffold 或调整依赖时，建议先在本地完成本页流程，再提交 Cloud。
只想先跑通一个 Single，可以直接看[快速开始](quickstart.md)；本地验证不是每次提交的必需步骤。

本页把任务输入、任务编译、Nexus wheel 和依赖 wheelhouse 串成一条流程。任务编译是将 Generator 或
Dataset＋Profile 展开、合并并校验为标准 Task List，携带本地任务文件时同时生成 Data List。命令仍为 `nexus materialize`。
提供 **Generator、常规 Compose、
Harbor Compose** 三种并列的输入示例；选择其中一种后，共用后面的任务编译、wheel 构建和提交步骤：

```text
准备任务输入 → 编译任务 → 构建 Nexus wheel → 构建依赖 wheelhouse → 检查报告
                                                                  ↓
                                                    用同一份原始输入提交 Cloud
```

本地 Task List 和 wheelhouse 用于本地验证。当前 `nexus submit batch` 仍接收原始 Generator 或 Dataset/Profile 输入，
会重新准备任务和代码，不直接接收这里生成的 JSONL、Data List 或 wheelhouse。

## 流程中的输入与产物

| 文件 | 内容 | 用在哪里 |
| --- | --- | --- |
| 任务目录、Generator、Profile/实验 YAML | 数据和运行配置的来源 | 编译时读取这些输入；Cloud 提交使用对应的 Generator 或 Dataset/Profile 参数 |
| `tasks.jsonl`（Task List） | 每个任务合并后的 RunInput，含任务、Agent、LLM Client、Runtime Provider 和处理器等配置 | 本地检查配置，供 `modules wheel` 分析依赖 |
| `tasks.data-list.bin`（Data List，可选） | 打包后的本地任务文件；v3 Task List 通过 `data_ref` 引用 | 和对应 Task List 配套保留 |
| `dist/nexus_agent-*.whl`（Nexus wheel） | 当前 checkout 的 Nexus 代码和模块元数据 | 作为依赖校验的精确代码版本 |
| `wheelhouse/` | 包含 Nexus wheel 和所需依赖的离线安装包集合、最终依赖锁、报告 | 验证能否构建并离线安装 Host Python 环境 |

<a id="完整本地验证脚本"></a>

## 1. 准备项目和输出目录

在支持 Compose 的 Nexus checkout 根目录，用 **Bash** 执行。需要 `uv`、`jq`、`openssl`；命令使用 Python 3.13。
按第 1～6 步完成本地验证，第 7 步按需提交 Cloud。第 2 步的三个输入示例**只选择一个**；
其余步骤共用。所有代码块在同一个 Bash 终端执行。

本地验证会访问 Python 包索引，不启动 Sandbox 或调用模型。`${oc.env:...}` 会在本地任务编译时解析，
需要提前配置对应变量；`token__env`、`api_key__env` 等引用则保留到任务执行时读取。

```bash
set -euo pipefail
umask 077

# 1. 准备统一的验证目录。
export PROJECT_ROOT="$(pwd -P)"
export OUTPUT_DIR="$(mktemp -d /tmp/nexus-cloud-validate-XXXXXX)"
export TASK_VALIDATE_PATH="$OUTPUT_DIR/tasks.jsonl"
export WHEEL_VALIDATE_PATH="$OUTPUT_DIR/dist"
export WHEELHOUSE_VALIDATE_PATH="$OUTPUT_DIR/wheelhouse"

uv lock --check
```

## 2. 选择任务输入

下面三个示例分别定义 `TASK_ARGS` 和 `INPUT_UV_ARGS`。`TASK_ARGS` 保存任务输入，任务编译和提交时共用；
`INPUT_UV_ARGS` 指定读取该输入需要的本地依赖。更换示例时一起替换两个数组。
Compose 的 Profile 按出现顺序合并；可在 `TASK_ARGS` 末尾追加 `--set` 或 `--set-json` 覆盖个别字段。

| 输入方式 | 数据与配置来源 | 本地任务编译的预期产物 |
| --- | --- | --- |
| [Generator](#generator) | Python 动态生成任务配置 | 5 条 v2 记录 |
| [常规 Compose](#常规-compose) | SWE Dataset + 公共 Profile | 1 条 v2 记录 |
| [Harbor Compose](#harbor-compose) | Harbor 任务目录 + Agent/Runtime/LLM Profile | 1 条 v3 记录 + Data List |

### Generator

适用于通过 Python 动态生成 Batch 任务。下面使用仓库自带的
[batch_generator.py](../examples/configs/online-service/batch_generator.py)，生成 5 个 `identity + main_agent` 任务。
它复用 [Single 快速开始的模型配置](quickstart.md#4-配置模型服务)，需要本地已设置
`MODEL_SERVICE_API_APP_ID`、`MODEL_SERVICE_API_APP_KEY`、`MODEL_SERVICE_API_BASE_URL`。

```bash
# 2A. 选择 Generator 输入。
INPUT_UV_ARGS=(--extra distributed)
TASK_ARGS=(
  --generator "$PROJECT_ROOT/docs/examples/configs/online-service/batch_generator.py"
)
```

预期编译得到 5 条 v2 记录，不产生 Data List。Generator 模式不接受 `--limit`、`--split` 或 `--data-list-output`，
数量由 Generator 自身控制。选择此例后，继续[第 3 步：任务编译](#3-任务编译)。

### 常规 Compose

适用于 Dataset + 公共运行配置。下面组合仓库内的 5 题 SWE Dataset 和
[swe-batch.yaml](../examples/configs/online-service/swe-batch.yaml)，先选 1 个任务验证。
本地需要 `MODEL_SERVICE_API_APP_ID`、`MODEL_SERVICE_API_APP_KEY`、`MODEL_SERVICE_API_BASE_URL`
以及 `GONGFENG_RUNTIME_TOKEN`；配置说明见 [Dataset + SWE 示例](submit-batch.md#示例二dataset--swe)。

```bash
# 2B. 选择常规 Dataset + Profile 输入。
INPUT_UV_ARGS=(--extra distributed)
TASK_ARGS=(
  --data "$PROJECT_ROOT/docs/examples/datasets/online-service-swe-verified-5"
  --use "$PROJECT_ROOT/docs/examples/configs/online-service/swe-batch.yaml"
  --limit 1
)
```

预期编译得到 1 条 v2 记录，包含 `swe_task`、`swe_agent`、Gongfeng、OpenAI Client 和 chat 处理器。
这个示例没有需要打包的本地任务文件。选择此例后，继续[第 3 步：任务编译](#3-任务编译)。

### Harbor Compose

适用于 `t3-harbor:///` 任务目录 + Harbor Bridge Profile。把 `HARBOR_DATASET_PATH` 改为真实任务根目录的绝对路径，
任务中应包含 `task.toml`、`instruction.md` 等文件。`flat=true` 只扫描直属任务目录，多级嵌套时改为 `flat=false`。

```bash
# 2C. 选择 Harbor Dataset + Profile 输入。
export HARBOR_DATASET_PATH=/absolute/path/to/harbor/tasks
INPUT_UV_ARGS=(--extra distributed --with 'harbor==0.20.0')
TASK_ARGS=(
  --data "t3-harbor://${HARBOR_DATASET_PATH}?flat=true"
  --use harbor_bridge:mode=agent
  --use "$PROJECT_ROOT/nexus/extensions/harbor_bridge/configs/runtime/gongfeng-prod.yaml"
  --use "$PROJECT_ROOT/nexus/extensions/harbor_bridge/configs/agents/claude-code.yaml"
  --use "$PROJECT_ROOT/nexus/extensions/harbor_bridge/configs/llm/aws-opus48-litellm.yaml"
  --limit 1
)
```

预期编译得到 1 条 v3 记录和 1 个 Data List frame，包含 Harbor Bridge、Claude Code、Gongfeng、LiteLLM。
这些 Profile 使用运行时凭据引用，本地验证不要求真实 Runtime/LLM 凭据；云端执行前需要完成凭据注入。

选好输入后，三个示例都执行下面相同的第 3～6 步。

<a id="共用流程物化wheel-和报告"></a>

<a id="3-物化任务"></a>

## 3. 任务编译

将选定的输入合并为完整 RunInput，并写入 Task List。需要打包的本地任务文件进入 Data List，
对应的 v3 Task List 通过 `data_ref` 引用文件。

```bash
# 3. 编译所选任务。诊断信息在 stderr，JSON 摘要保存到文件。
uv run --python 3.13 "${INPUT_UV_ARGS[@]}" \
  nexus materialize "${TASK_ARGS[@]}" \
  --output "$TASK_VALIDATE_PATH" > "$OUTPUT_DIR/materialize.json"

jq -e '.total_items > 0' "$OUTPUT_DIR/materialize.json"
jq '{path, record_version, total_items, size_bytes, data_list}' "$OUTPUT_DIR/materialize.json"
```

确认摘要中的 `total_items` 与第 2 步所选示例一致。带本地任务文件时，摘要还包含 `data_list` 的路径和校验信息。

## 4. 构建 Nexus wheel

构建当前项目的 Nexus 代码包，供下一步读取精确的模块元数据。它与包含所有依赖的 wheelhouse 是两个不同产物。

```bash
# 4. 构建当前 Nexus 代码 wheel。
uv build --wheel --out-dir "$WHEEL_VALIDATE_PATH"
```

## 5. 构建依赖 wheelhouse

使用刚生成的 Task List、Nexus wheel 和当前 `uv.lock`，解析任务实际使用的模块依赖。
命令还会自动验证无索引离线安装及 Nexus 导入/版本；失败时不会生成成功报告。

```bash
# 5. 分析实际模块依赖，构建 wheelhouse，并自动验证离线安装。
uv run --python 3.13 --extra distributed --extra builder \
  nexus modules wheel \
  --task-list "$TASK_VALIDATE_PATH" \
  --project-root "$PROJECT_ROOT" \
  --lock "$PROJECT_ROOT/uv.lock" \
  --nexus-wheel "$WHEEL_VALIDATE_PATH"/nexus_agent-*.whl \
  --output "$WHEELHOUSE_VALIDATE_PATH"
```

<a id="怎么判断每一步通过"></a>

## 6. 检查验证结果

先核对报告中的模块是否与输入配置一致，再确认各阶段产物。

```bash
# 6. 检查模块集合和依赖要求。
jq '{
  representative_item_id,
  module_names: .wheelhouse.module_names,
  host_requirements: .wheelhouse.module_host_requirements
}' "$WHEELHOUSE_VALIDATE_PATH/module-wheel-report.json"

echo "本地验证产物：$OUTPUT_DIR"
```

| 步骤 | 应看到什么 | 失败时先查哪里 |
| --- | --- | --- |
| 任务编译 | 条数与所选输入一致：Generator 示例为 5，两个 Compose 示例为 1；有本地任务文件时产生 v3 + Data List | [任务编译失败](troubleshooting.md#任务编译失败) |
| Nexus wheel | `dist/` 下出现一个 `nexus_agent-*.whl` | [Nexus wheel 构建失败](troubleshooting.md#nexus-wheel-构建失败) |
| 依赖 wheelhouse | 命令成功，产生 `module-wheel-report.json`；分阶段离线安装、完整依赖检查、最终锁版本比对和 Nexus 导入/版本检查通过 | [依赖冲突](troubleshooting.md#依赖冲突)、[离线安装失败](troubleshooting.md#离线安装失败) |
| 模块报告 | 与所选输入中的任务、Agent scaffold、Runtime Provider、LLM Client 和处理器对应；以各输入示例的预期结果为准 | 核对原始 Profile 和首条 RunInput，不能只保留 `task_name` |

默认 `--task-list-version 2` 会在发现本地任务文件时自动升级为 v3。若输入没有本地文件，则保留 v2，
不创建空 Data List。Data List 默认是 Task List 同名的 `.data-list.bin` 文件，可用 `--data-list-output` 指定其他路径。
文件包和 Task List 必须配套保留；依赖分析只需要 JSONL 中的 RunInput，不读取 Data List 内容。

`modules wheel --task-list` **只分析第一条记录**。代表任务应覆盖实际使用的模块；如果不同任务使用不同 scaffold
或 Host 依赖，应分别验证，并按[Batch 模块依赖约束](submit-batch.md#模块依赖)组织批次。

这些检查证明配置可编译、声明的 **Host Python 依赖**可构建并离线安装。它们不代表任务已经执行成功，
也不验证沙箱内的 scaffold CLI、Runtime 额外依赖、系统软件、模型凭据或 verifier 行为。

<a id="验证通过后提交-cloud"></a>

## 7. 提交 Cloud

先完成[认证与运行环境](access-and-environments.md)，在同一终端设置 `CONTROL_BASE_URL`、`TAIHU_TOKEN_FILE`，
确认有目标 Dynamic Environment 的配额和有效个人 PAT。
使用本地插值的配置，需要在提交时提供有效的环境变量值。使用 `__env` 引用的配置（如 Harbor Profile），
则需要确保引用的 `GONGFENG_RUNTIME_TOKEN`、`OPENAI_API_KEY` 在**云端任务环境**中可用；
本地 `export` 不会自动转发这些变量。按所在环境约定配置凭据注入，未配置时先联系环境管理员。

在刚才的同一个 Bash 终端中，继续使用 `TASK_ARGS`，而不是传入本地验证产生的文件：

```bash
export NEXUS_ENVIRONMENT='<已获授权的 dynamic-environment-id>'
IDEMPOTENCY_KEY="batch-validated-$(openssl rand -hex 16)"
printf '%s\n' "$IDEMPOTENCY_KEY" > "$OUTPUT_DIR/idempotency-key"
```

上面的 key 只为本次逻辑任务生成一次。网络重试时执行下方提交命令，复用 key 和相同输入；更换输入后需要新 key。

```bash
uv run --python 3.13 "${INPUT_UV_ARGS[@]}" \
  nexus submit batch "${TASK_ARGS[@]}" \
  --generation local \
  --environment "$NEXUS_ENVIRONMENT" \
  --project-root "$PROJECT_ROOT" \
  --source "$PROJECT_ROOT" \
  --max-inflight 1 \
  --control-url "$CONTROL_BASE_URL" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --wait --json > "$OUTPUT_DIR/submission.json"

jq '{execution_id, batch_id, terminal}' "$OUTPUT_DIR/submission.json"
```

这一步才会创建云端 Batch。Generator 示例会提交 5 个任务、按 `--max-inflight 1` 串行运行；
两个 Compose 示例保留 `--limit 1`，只提交一个任务。检查返回的 Batch/Execution 状态和结果后，再调整数量及并发。
扩大批次与资源配置见 [Batch 指南](submit-batch.md#调整并发与-manager-资源)，
查询与停止见[任务与结果](runs-and-results.md)。

本地验证后应保持源码、`uv.lock`、任务数据和 Profile 一致；其中任何一项改变，之前的报告就不再验证完整的提交输入。
本地 wheelhouse 是诊断产物，提交 CLI 会重新构建、上传自己的代码与任务输入。

## 补充输入方式

### 已有完整实验 YAML

这是通过文件输入编译任务的入口。完成第 1 步后，用下面的命令替换第 3 步；成功后继续第 4～6 步。
下例读取 Harbor 实验 YAML，常规实验无需额外的 `--with 'harbor==0.20.0'`：

```bash
uv run --python 3.13 --extra distributed --with 'harbor==0.20.0' \
  nexus materialize /absolute/path/to/harbor-experiment.yaml \
  --limit 1 --output "$TASK_VALIDATE_PATH" > "$OUTPUT_DIR/materialize.json"
```

文件中的相对路径以该 YAML 所在目录为准；`--split`、`--limit` 可覆盖文件中的设置，`data.id` 可选择指定任务。
位置参数用于完整 Compose/RunInput 文件，单独的 `body` Profile 用 `--use` 引用，不能与位置参数混用。

这个文件入口用于本地验证。当前 `submit batch` 不支持位置参数实验 YAML；提交时需要把该文件对应的
`data` 和 `use` 选择写为 `TASK_ARGS`。切换输入后不要复用上一种输入的参数数组去提交。

### 只检查 Profile 组合

这是一种只检查模块组合的本地入口。完成第 1 步后，可以用仓库的[模块验证示例](../examples/configs/harbor/module-wheel-validation.yaml)：

```bash
uv run --python 3.13 nexus materialize \
  docs/examples/configs/harbor/module-wheel-validation.yaml \
  --output "$TASK_VALIDATE_PATH" > "$OUTPUT_DIR/materialize.json"
```

成功后继续第 4～6 步。它生成一条用于依赖分析的 v2 RunInput，适合检查更换 scaffold 后的模块集合；
不验证真实 Harbor 任务的数据、文件包或任务专属配置，不能替代 [Harbor Compose](#harbor-compose) 中的真实任务编译。

## 参数与回归验证

命令参数、报告字段和依赖解析规则见 [Module CLI](../how-to/manage-modules/module-cli.md#本地构建模块-wheelhouse)。
修改任务编译或模块构建代码后，可运行本地 E2E：

```bash
NEXUS_MATERIALIZE_WHEEL_E2E=1 \
uv run --python 3.13 --extra dev --extra distributed --extra builder --with 'harbor==0.20.0' \
  pytest tests/e2e/test_materialize_module_wheelhouse.py -q -s
```

E2E 使用本地 Harbor fixture，验证 v2/v3 任务编译、实际 wheelhouse 构建、离线安装及 Harbor/Claude Code 导入，
不调用 Cloud 或模型。可设置 `NEXUS_MATERIALIZE_E2E_OUTPUT` 保存产物，默认使用 pytest 临时目录。
检查结束后，可删除自己创建的验证目录；不要把任务数据、凭据或 wheelhouse 提交到仓库。
