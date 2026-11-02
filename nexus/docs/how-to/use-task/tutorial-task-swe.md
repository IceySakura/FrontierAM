# SWE 任务说明

本文说明 `swe_task` 的完整任务配置，并给出最小的 `run_oracle` 示例。`runtime_provider_config`、`openai_clients`、`processors` 已在 README 中说明，本文不再展开。

## 任务概览
- 任务名：`swe_task`，用于单条 SWE-bench 实例的自动修复与评测。
- 关键组件：使用 Gongfeng runtime 拉起实例镜像，在推理阶段收集补丁，随后在评测 runtime 内运行官方测试并生成评分报告。
- 主要文件：`workspace/swe.yaml`（任务配置）、`nexus/extensions/tasks/swe/task.py`（执行逻辑）。

## 任务配置字段（`workspace/swe.yaml`）
仅列出与任务运行直接相关的字段（其余保持默认或参考 README）。

- `task_name`: 固定为 `swe_task`，用于注册并调用任务。
- `task_config` 下必填（三选一）：
  - `remote_dataset`: 远程数据集路径，格式为 `namespace/dataset_name`（如 `swe-public/SWE-bench_Verified`）。 配合 `split` 使用可指定数据集分片。 **推荐方式**，可通过 `nexus dataset list` 查看可用数据集。
  - `dataset_path`: 本地 JSON/JSONL 数据集路径，需包含目标实例。
  - `instance`: 直接传入完整的实例数据对象。
  - `instance_id`: 目标实例 ID（如 `django__django-13023`）。
  - `eval_timeout_seconds`: 评测超时秒数，默认 `1800`。
- `task_config` 下控制行为的可选字段：
  - `split` (string，默认空): 远程数据集的分片名称，仅在使用 `remote_dataset` 时生效。
  - `run_oracle` (bool，默认 `false`): 若为 `true`，直接使用数据集中提供的黄金补丁，无需代理推理。
  - `patch_text` (string，默认空): 直接传入补丁文本内容（diff 格式）；与 `run_oracle` 互斥，优先级高于代理推理。
  - `skip_empty_patch_eval` (bool，默认 `false`): 代理未产出补丁时是否跳过评测。
  - `skip_infer_patch_eval` (bool，默认 `false`): 推理完成后是否跳过评测阶段，直接返回补丁。
  - `skip_git_truncate` (bool，默认 `false`): 是否跳过 git 历史截断。 默认情况下，`infer_runtime_init` 会删除 `base_commit` 之后的所有提交以防止模型窥探未来变更；设为 `true` 可禁用此行为。
  - `infer_runtime_name` / `eval_runtime_name`: 推理与评测使用的 runtime 名称，默认 `main` / `eval`。
  - `image_prefix`: 生成实例镜像名的前缀，默认 `mirrors.tencent.com/hunyuan_yanguan`。
  - `image_override`: 可选直接指定任务镜像（若不为空，覆盖自动生成的镜像）。
  - `pre_agent_script`: 可选的预处理命令列表（字符串数组），在代理运行前于实例工作目录执行。
- 代理配置：
  - `agent_name`: 默认 `swe_agent`。
  - `agent_config.max_iterations`: 代理最大迭代轮次，默认 `5`。

## 执行流程（摘自 `nexus/extensions/tasks/swe/task.py`）
- 加载实例数据：支持三种方式获取实例：
  - 从 `remote_dataset` 远程服务器获取指定 `instance_id` 的实例（可选 `split` 参数）。
  - 从 `dataset_path` 本地文件中读取 `instance_id` 对应的条目。
  - 直接通过 `instance` 字段传入完整实例数据。
- 镜像准备：通过 `image_prefix` + `instance_id` 生成运行镜像；若配置了 `image`，则优先生效。
- 生成 patch：
  - `run_oracle = true`：直接取数据集内的 golden patch（标准答案）。
  - 提供 `patch_text`：直接使用传入的补丁文本。
  - 否则：配置推理 runtime，执行 `infer_runtime_init`（默认截断 base_commit 之后的 git 历史，可通过 `skip_git_truncate` 禁用），再执行 `pre_agent_script`（如有），设置问题描述与工作目录，运行代理获取补丁；若 `skip_infer_patch_eval = true` 或补丁为空且 `skip_empty_patch_eval = true`，则直接返回。
- 评测 patch：在评测 runtime 内应用 patch 并运行测试，收集日志与测试输出。
- 评分与产物：
  - `SwebenchTaskOutput` 会直接返回 `patch_text` / `report_json` / `test_output` / `apply_log`（字符串）以及 `score`、`completed`、`resolved`、错误信息。

## 最小 `run_oracle` 示例

推荐使用远程数据集，无需下载本地文件。 可以通过 `nexus dataset` 命令查看可用数据集和实例（详见 [Dataset CLI 使用指南](../use-datasets/dataset-cli.md)）：

```bash
# 列出所有可用数据集
nexus dataset list

# 列出数据集中的实例 ID
nexus dataset list swe-public/SWE-bench_Verified
```

1) 编辑 `workspace/swe.yaml`，保留已有导入/运行时配置，仅调整关键字段：

```yaml
imports:
  - nexus.extensions
task_name: "swe_task"
task_config:
  remote_dataset: "swe-public/SWE-bench_Verified"  # 远程数据集路径
  instance_id: "django__django-11095"
  split: ""                   # 可选，留空使用默认 split
  run_oracle: true            # 使用 golden patch
  # patch_text: "diff --git a/..."  # 或直接传入补丁文本
agent_name: "demo_agent"
runtime_provider_name: "gongfeng"
runtime_provider_config:
  api_base_url: "http://hyrl-sandbox.prod.woa.com:8052"
  token: "<your-token>"
  runtime_image: "mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest"
```

2) 执行任务（示例命令）：

```bash
uv run nexus run-single workspace/swe.yaml
```

3) 结果会直接出现在任务输出中：`report_json` / `test_output` / `apply_log` / `patch_text`。

## 使用本地数据集示例

如果需要使用本地数据集文件（如自定义数据集或离线环境），可以使用 `dataset_path` 参数：

```bash
# 下载数据集到本地（可选）
nexus dataset download swe-public/SWE-bench_Verified -o ./workspace --format jsonl

# 或获取单个实例
nexus dataset get swe-public/SWE-bench_Verified django__django-13023 -o ./workspace/django__django-13023.json
```

配置文件示例：

```yaml
imports:
  - nexus.extensions
task_name: "swe_task"
task_config:
  dataset_path: "./workspace/django__django-13023.json"  # 本地数据集路径
  instance_id: "django__django-13023"
  run_oracle: true
agent_name: "demo_agent"
runtime_provider_name: "docker"
```

数据集可以是单个 JSON 文件或 JSONL 文件，需要在每个 instance 中包含 `dataset_type` 字段。

## 完整推理示例（与 `workspace/swe.yaml` 同步）

该示例展示不使用 `run_oracle` / `patch_text`，而由 agent 产出补丁的完整配置。

```yaml
imports:
  - nexus.extensions.tasks.swe
  - nexus.extensions.agents.swe_agent

task_name: "swe_task"
task_config:
  remote_dataset: "swe-public/SWE-bench_Verified"  # 远程数据集路径
  instance_id: "django__django-11095"
  split: "test"                    # 可选，留空使用默认 split
  eval_timeout_seconds: 1800
  # 由 agent 推理生成 patch
  run_oracle: false
  skip_empty_patch_eval: false
  infer_runtime_name: "main"
  eval_runtime_name: "eval"
  image_prefix: "mirrors.tencent.com/hunyuan_yanguan"
  # image_override: "mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-11095:latest"
  pre_agent_script:
    - echo "agent 启动前执行的初始化命令"

runtime_provider_name: "gongfeng"
runtime_provider_config:
  api_base_url: "http://hyrl-sandbox.prod.woa.com:8052"
  token: "<your-token>"
  runtime_image: "mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest"
  request_timeout: 600.0

# Agent 配置
agent_name: "swe_agent"
agent_config:
  max_iterations: 5

openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "https://api.openai.com/v1"
      request_args:
        model: "gpt-4.1"

# Optional tracing processors
processors:
  chat:
    name: "chat_tracker"
  trace:
    name: "json_line"
    args:
      file_path: "workspace/trace.jsonl"
```

如果需要使用本地数据集，将 `remote_dataset` 替换为 `dataset_path`：

```yaml
task_config:
  dataset_path: "/cephfs/django__django-13023.json"  # 本地数据集路径
  instance_id: "django__django-13023"
  # ... 其他配置相同
```

数据集样例

```json
{
    "dataset_type": "swe-bench",
    "repo": "django/django",
    "instance_id": "django__django-13023",
    "base_commit": "f83b44075dafa429d59e8755aa47e15577cc49f9",
    "patch": "diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py\n--- a/django/db/models/fields/__init__.py\n+++ b/django/db/models/fields/__init__.py\n@@ -1501,7 +1501,7 @@ def to_python(self, value):\n             return self.context.create_decimal_from_float(value)\n         try:\n             return decimal.Decimal(value)\n-        except decimal.InvalidOperation:\n+        except (decimal.InvalidOperation, TypeError, ValueError):\n             raise exceptions.ValidationError(\n                 self.error_messages['invalid'],\n                 code='invalid',\n",
    "test_patch": "diff --git a/tests/model_fields/test_decimalfield.py b/tests/model_fields/test_decimalfield.py\n--- a/tests/model_fields/test_decimalfield.py\n+++ b/tests/model_fields/test_decimalfield.py\n@@ -21,9 +21,24 @@ def test_to_python(self):\n         # Uses default rounding of ROUND_HALF_EVEN.\n         self.assertEqual(f.to_python(2.0625), Decimal('2.062'))\n         self.assertEqual(f.to_python(2.1875), Decimal('2.188'))\n-        msg = '“abc” value must be a decimal number.'\n-        with self.assertRaisesMessage(ValidationError, msg):\n-            f.to_python('abc')\n+\n+    def test_invalid_value(self):\n+        field = models.DecimalField(max_digits=4, decimal_places=2)\n+        msg = '“%s” value must be a decimal number.'\n+        tests = [\n+            (),\n+            [],\n+            {},\n+            set(),\n+            object(),\n+            complex(),\n+            'non-numeric string',\n+            b'non-numeric byte-string',\n+        ]\n+        for value in tests:\n+            with self.subTest(value):\n+                with self.assertRaisesMessage(ValidationError, msg % (value,)):\n+                    field.clean(value, None)\n \n     def test_default(self):\n         f = models.DecimalField(default=Decimal('0.00'))\n",
    "problem_statement": "DecimalField.to_python() raises TypeError on dict values.\nDescription\n\t\nA call to DecimalField.to_python() with a dictionary as the value parameter produces TypeError instead of ValidationError. This is a problem, for example, when you try to save a model object, and a decimal field got set to a dictionary by mistake. The TypeError exception that comes back makes it hard to track the problem to the field if the object has a lot of fields.\nI am proposing a patch to fix it:\n​https://github.com/django/django/pull/13023\n",
    "hints_text": "",
    "created_at": "2020-06-04T20:40:44Z",
    "version": "3.2",
    "FAIL_TO_PASS": "[\"test_invalid_value (model_fields.test_decimalfield.DecimalFieldTests)\", \"test_lookup_really_big_value (model_fields.test_decimalfield.DecimalFieldTests)\"]",
    "PASS_TO_PASS": "[\"test_default (model_fields.test_decimalfield.DecimalFieldTests)\", \"test_filter_with_strings (model_fields.test_decimalfield.DecimalFieldTests)\", \"test_get_prep_value (model_fields.test_decimalfield.DecimalFieldTests)\", \"test_max_decimal_places_validation (model_fields.test_decimalfield.DecimalFieldTests)\", \"test_max_digits_validation (model_fields.test_decimalfield.DecimalFieldTests)\", \"test_max_whole_digits_validation (model_fields.test_decimalfield.DecimalFieldTests)\", \"Trailing zeros in the fractional part aren't truncated.\", \"test_save_without_float_conversion (model_fields.test_decimalfield.DecimalFieldTests)\", \"test_to_python (model_fields.test_decimalfield.DecimalFieldTests)\"]",
    "environment_setup_commit": "65dfb06a1ab56c238cc80f5e1c31f61210c4577d",
    "difficulty": "<15 min fix"
}
```

运行命令与 oracle 示例一致：

```bash
uv run nexus run-single workspace/swe.yaml
```

## 批量运行（Batch Mode）

对于大规模评测场景，可使用 `nexus run-batch` 配合 generator 文件批量运行多个 SWE-bench 实例。 Generator 位于 `nexus/extensions/tasks/swe/generator.py`。

**推荐方式**：使用 `SWE_DATASET` 环境变量指定远程数据集（如 `swe-public/SWE-bench_Verified`），无需下载本地文件。 可通过 `nexus dataset list` 查看可用数据集。

如需使用本地数据集，可通过 `nexus dataset download` 下载，里面是数据集由流水线自动从仓库 [swe-public](https://git.woa.com/groups/hunyuan-dataset/swe-public/) 打包。

### 环境变量

Generator 通过环境变量配置运行参数：

| 环境变量                        | 说明                                              | 默认值                                     |
|-----------------------------|-------------------------------------------------|-----------------------------------------|
| `SWE_DATASET`               | 远程数据集路径（如 `swe-public/SWE-bench_Verified`），优先于本地文件 | 空（使用本地文件）                               |
| `SWE_DATASET_SPLIT`         | 远程数据集分片名称                                       | 空（使用默认分片）                               |
| `SWE_DATASET_PATH`          | 本地 JSONL 数据集路径（仅当 `SWE_DATASET` 未设置时使用）         | `./workspace/swe-bench.jsonl`           |
| `SWE_INSTANCE_IDS`          | 逗号分隔的实例 ID 列表，仅运行指定实例                           | 空（运行全部）                                 |
| `SWE_SKIP`                  | （分批跑）跳过前 N 条实例                                  | `0`                                     |
| `SWE_LIMIT`                 | （分批跑）最多运行 N 条实例                                 | 无限制                                     |
| `SWE_RUN_ORACLE`            | 使用 golden patch（`1`/`true`）                     | `false`                                 |
| `SWE_AGENT_NAME`            | Agent 名称（oracle 模式忽略）                           | `swe_agent`                             |
| `SWE_MAX_ITERATIONS`        | Agent 最大迭代次数                                    | `5`                                     |
| `SWE_RUNTIME_PROVIDER_NAME` | Runtime provider 类型                             | `gongfeng`                              |
| `SWE_RUNTIME_BASE_URL`      | Runtime API 地址                                  | `http://hyrl-sandbox.prod.woa.com:8052` |
| `SWE_RUNTIME_TAG`           | Runtime 镜像标签                                    | `latest`                                |
| `SWE_RUNTIME_TIMEOUT`       | Runtime 超时秒数                                    | `1800`                                  |
| `OPENAI_API_KEY`            | OpenAI API 密钥                                   | `sk-proj-1234567890`                    |
| `OPENAI_BASE_URL`           | OpenAI API 地址                                   | `https://api.openai.com/v1`             |
| `OPENAI_MODEL`              | 模型名称                                            | `gpt-4.1`                               |

### 示例命令

**1) 使用远程数据集批量评测（推荐）**

直接从数据集服务器获取实例列表，无需下载本地文件：

```bash
SWE_RUN_ORACLE=1 \
SWE_DATASET=swe-public/SWE-bench_Verified \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/swe/generator.py \
  -d ./workspace/batch_remote.db \
  --max-concurrent 50
```

**2) Oracle 模式批量评测本地文件**

使用 golden patch 验证数据集和评测流程是否正常：

```bash
SWE_RUN_ORACLE=1 \
SWE_DATASET_PATH=./workspace/swe-bench.jsonl \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/swe/generator.py \
  -d ./workspace/batch_bench.db \
  --max-concurrent 50
```

**3) Oracle 模式批量评测 SWE-rebench 数据集并跳过前 100 条**

```bash
SWE_RUN_ORACLE=1 \
SWE_DATASET_PATH=./workspace/swe-rebench.jsonl \
SWE_SKIP=100 \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/swe/generator.py \
  -d ./workspace/batch_rebench.db \
  --max-concurrent 250
```

**4) Oracle 模式批量评测 SWE-turing 数据集**
```bash
SWE_RUN_ORACLE=1 \
SWE_DATASET_PATH=./workspace/swe-bench-turing.jsonl \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/swe/generator.py \
  -d ./workspace/batch_turing.db \
  --max-concurrent 250
```

**5) Oracle 模式批量评测 SWE-bench_pro 数据集**
```bash
SWE_RUN_ORACLE=1 \
SWE_DATASET_PATH=./workspace/swe-bench_pro.jsonl \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/swe/generator.py \
  -d ./workspace/batch_bench_pro.db \
  --max-concurrent 250
```

**6) Agent 推理模式**

使用 Agent 生成补丁并评测：

```bash
SWE_DATASET=swe-public/SWE-bench_Verified \
SWE_AGENT_NAME=swe_agent \
SWE_MAX_ITERATIONS=10 \
OPENAI_API_KEY=sk-xxx \
OPENAI_BASE_URL=https://api.openai.com/v1 \
OPENAI_MODEL=gpt-4.1 \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/swe/generator.py \
  -d ./workspace/batch_agent.db \
  --max-concurrent 20
```

也可以使用本地数据集：

```bash
SWE_DATASET_PATH=./workspace/swe-bench.jsonl \
SWE_AGENT_NAME=swe_agent \
# ... 其他配置相同
```

**7) 部分运行（跳过/限制）**

跳过前 100 条，只运行后续 50 条：

```bash
SWE_RUN_ORACLE=1 \
SWE_DATASET=swe-public/SWE-bench_Verified \
SWE_SKIP=100 \
SWE_LIMIT=50 \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/swe/generator.py \
  -d ./workspace/batch_partial.db \
  --max-concurrent 50
```

### 结果存储、恢复运行、重试失败任务

参见 [run-batch.md](run-batch.md)。
