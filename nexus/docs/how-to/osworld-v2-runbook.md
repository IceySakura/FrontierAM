# OSWorld V2 运行指南

本指南面向 `osworld-v2.1` 的本地与平台用户。V2入口为 `osworld_v2` run-part /
`os_world_v2` task，V1运行见 [V1 runbook](osworld-runbook.md)。
实现原理、HITL、GitLab和独立评分的设计见
[V2设计说明](../developing/osworld-v2-hunyuan-design.md)。

## 1. 准备环境

在仓库根目录执行：

```bash
cd /data/workspace/nexus
test -e nexus/extensions/tasks/os_world_v2/.env || \
  (umask 077; cp nexus/extensions/tasks/os_world_v2/.env.example \
    nexus/extensions/tasks/os_world_v2/.env)
chmod 600 nexus/extensions/tasks/os_world_v2/.env
```

编辑私有 `.env`，填写：

| 用途 | 必填配置 |
|---|---|
| AGS | `AGS_SECRET_ID`、`AGS_SECRET_KEY`、`AGS_OSWORLD_TOOL_ID` |
| 输入数据 | `OSWORLD_V2_DATASET_URI`（有权限的明确版本）、完整 `WEDATA_KEY` |
| Hunyuan主模型 | `HUNYUAN_BASE_URL`、`HUNYUAN_MODEL`、服务要求的鉴权 |
| GPT/Claude主模型 | `NEXUS_BASE_URL`、`NEXUS_MODEL`、`NEXUS_API_KEY` |
| 输出产物 | `ARTIFACT_TAIHU_PAT`，或执行 `nexus artifact login` |
| 特定题目 | 第5节的GitLab、辅助模型及独立grader配置 |

模板不是可直接运行的配置。显式加载已填写且可信的文件：

```bash
set +x
set -a
source nexus/extensions/tasks/os_world_v2/.env
set +a

uv run nexus modules install os_world_v2
```

还需安装所选agent/client/runtime的扩展依赖。下文 `--no-sync` 使用已准备的环境，
不会补装缺失包。`run-single`和generator不自动加载 `.env`。
不要开启 `set -x`、打印完整环境或提交密钥。每次实验建议使用新shell。

### 运行前检查

- 使用包含所需V2功能的已发布代码提交，平台必须能够安装该提交。
- tool提供OSWorld HTTP/CDP；CC/Codex需要TCP9281，082还需要TCP3000。
- tool具备mock网站，026/041还需GitLab、Runner和Pages；变更tool后申请新实例验证。
- 运行目录可写且有足够空间，宿主没有占用转发所需端口的通配监听。
- 首次运行先选020，并发1；不要直接启动全量。

远端输入只读预检，不申请AGS或调用模型：

```bash
uv run --no-sync python - <<'PY'
from nexus.extensions.tasks.os_world_v2.data.index import dataset_index
items = dataset_index()
print("Tasks:", len(items))
print("Missing dependencies:", [
    task_id for task_id, item in items.items()
    if item["metadata"].get("missing_external_dependencies")
])
PY
```

## 2. 六种启动模式

| model_family | harness | Agent | 本地主模型client |
|---|---|---|---|
| gpt | bare | cua_gpt_agent | openai_responses |
| claude | bare | cua_anthropic_agent | anthropic_messages |
| gpt | codex | codex_native_bridge | openai_responses |
| claude | cc | claude_code_multiturn | anthropic_messages |
| hunyuan | codex | codex_native_bridge | infer_proxy |
| hunyuan | cc | claude_code_multiturn | infer_proxy |

CC和Codex使用 `execution_profile=osworld_v2`。CLI显示的 `CC_MODEL` /
`CODEX_MODEL` 不等于最终主模型路由；切换family时同时检查URL、model和鉴权。
Claude bare不支持本模块的HITL题；需要交互时选择已支持的CC、Codex或GPT bare入口。

### 2.1 直接运行一题

加载环境后，选择下面一条命令执行：

```bash
export TASK_ID=task_020

uv run --no-sync nexus run-single \
  --use "osworld_v2:model_family=gpt,harness=bare,task_id=$TASK_ID" \
  -o /tmp/osworld-v2-gpt-bare.json

uv run --no-sync nexus run-single \
  --use "osworld_v2:model_family=claude,harness=bare,task_id=$TASK_ID" \
  -o /tmp/osworld-v2-claude-bare.json

uv run --no-sync nexus run-single \
  --use "osworld_v2:model_family=gpt,harness=codex,task_id=$TASK_ID" \
  -o /tmp/osworld-v2-gpt-codex.json

uv run --no-sync nexus run-single \
  --use "osworld_v2:model_family=claude,harness=cc,task_id=$TASK_ID" \
  -o /tmp/osworld-v2-claude-cc.json

uv run --no-sync nexus run-single \
  --use "osworld_v2:model_family=hunyuan,harness=codex,task_id=$TASK_ID" \
  -o /tmp/osworld-v2-hunyuan-codex.json

uv run --no-sync nexus run-single \
  --use "osworld_v2:model_family=hunyuan,harness=cc,task_id=$TASK_ID" \
  -o /tmp/osworld-v2-hunyuan-cc.json
```

`--use`题号写 `task_020`，避免YAML把裸数字解析为整数。
手写YAML/JSON时 `task_config.task_id` 必须是字符串，例如 `"105"`、`"007"`。

### 2.2 导出YAML再运行

```bash
umask 077
uv run --no-sync python -m nexus.extensions.tasks.os_world_v2.scripts.export_single \
  --model-family hunyuan --harness codex --task-id 020 \
  -o /tmp/osworld-v2-hunyuan-codex.yaml

uv run --no-sync nexus run-single /tmp/osworld-v2-hunyuan-codex.yaml \
  -o /tmp/osworld-v2-single.json
```

导出命令的 `--task-id` 接受字符串 `020`。替换family/harness即可导出其他模式。
一次导出六份：

```bash
uv run --no-sync python -m nexus.extensions.tasks.os_world_v2.scripts.export_single \
  --all-modes --task-id 020 -o /tmp/osworld-v2-configs
```

`--all-modes`不自动切换服务，GPT/Claude共用当前shell的 `NEXUS_*`。
使用不同服务时分别加载环境并导出。环境或预算修改后重新导出；
生成文件可能包含解析后的私有配置，保持私有权限。

## 3. 批量运行

```bash
export OSWORLD_TASK_IDS=020
export OSWORLD_SKIP=0
unset OSWORLD_LIMIT

# 按需选择一个generator
uv run --no-sync nexus run-batch \
  -g nexus/extensions/tasks/os_world_v2/generators/generator_native_gpt_bare.py \
  -d /tmp/osworld-v2-gpt-bare.db --max-concurrent 1
uv run --no-sync nexus run-batch \
  -g nexus/extensions/tasks/os_world_v2/generators/generator_native_claude_bare.py \
  -d /tmp/osworld-v2-claude-bare.db --max-concurrent 1
uv run --no-sync nexus run-batch \
  -g nexus/extensions/tasks/os_world_v2/generators/generator_native_gpt_codex.py \
  -d /tmp/osworld-v2-gpt-codex.db --max-concurrent 1
uv run --no-sync nexus run-batch \
  -g nexus/extensions/tasks/os_world_v2/generators/generator_native_claude_cc.py \
  -d /tmp/osworld-v2-claude-cc.db --max-concurrent 1
uv run --no-sync nexus run-batch \
  -g nexus/extensions/tasks/os_world_v2/generators/generator_native_hunyuan_codex.py \
  -d /tmp/osworld-v2-hunyuan-codex.db --max-concurrent 1
uv run --no-sync nexus run-batch \
  -g nexus/extensions/tasks/os_world_v2/generators/generator_native_hunyuan_cc.py \
  -d /tmp/osworld-v2-hunyuan-cc.db --max-concurrent 1
```

`OSWORLD_TASK_IDS`是逗号分隔题号；为空选全量，再应用 `OSWORLD_SKIP/LIMIT`。
`OSWORLD_LIMIT=0`会得到空选集，不是无限制。显式single不受batch筛选影响。
新实验用新DB路径，避免在同一DB中混用题包、模型或环境。

## 4. 预算和常用配置

以下是generator在未设置环境变量时的默认值；私有 `.env` 可覆盖它们。

| 环境变量 | 用途 | 默认 |
|---|---|---|
| `OSWORLD_BENCHMARK_RELEASE` | benchmark | osworld-v2.1 |
| `OSWORLD_V2_DATASET_SPLIT` | 数据切分 | test |
| `OSWORLD_SANDBOX_TIMEOUT` | 实例TTL，不是agent预算 | 10800秒 |
| `OSWORLD_WEBSITE_READY_TIMEOUT` | mock网站就绪等待 | 900秒 |
| `OSWORLD_OFFICIAL_TIMEOUT` | 官方worker单次调用 | 900秒 |
| `OSWORLD_SETUP_WAIT_TIME` | setup后等待 | 5秒 |
| `OSWORLD_SETUP_ONLY` | 不运行agent/评分 | false |
| `CC_TIMEOUT` | CC执行预算 | 1800秒 |
| `CC_MAX_TURNS` | CC主agent每次CLI轮次 | 100 |
| `OSWORLD_CC_USER_TURNS` | CC外层用户轮次 | 20 |
| `TASK_TIMEOUT` | Codex执行预算 | 5400秒 |
| `OSWORLD_CODEX_SESSION_MODE` | auto / exec / app_server | auto |
| `OSWORLD_CODEX_USER_TURNS` | Codex app-server用户轮次 | 20 |
| `OSWORLD_CODEX_MAX_USER_REQUESTS` | Codex用户输入请求数 | 100 |
| `OSWORLD_CODEX_INPUT_TIMEOUT` | Codex单次交互等待 | 300秒 |
| `OSWORLD_MAX_ITERATIONS` | bare agent迭代 | 100 |
| `NEXUS_HOME` | 本地缓存/产物/鉴权根目录 | ~/.nexus |

**轮次不是整题模型调用数。** CC子agent、压缩调用会使总调用超过主agent轮次；
Codex的用户轮次不限制内部工具迭代。整体耗时还包括setup、评分和上传。

Hunyuan + CC可用以下窗口配置；它不是任意大输入都不越界的保证：

```bash
export CC_AUTO_COMPACT_WINDOW=200000
export CC_AUTOCOMPACT_PCT_OVERRIDE=80
export CC_MAX_OUTPUT_TOKENS=16384
```

完整变量清单见 `nexus/extensions/tasks/os_world_v2/.env.example`。

## 5. 辅助服务与独立评分

### 5.1 裁判模型及用户模拟器

在私有环境中定义密钥，再合并以下内容到 `OSWORLD_V21_ENV_JSON`。
平台对应 `nexus_run_input.task_config.v21.environment`：

```json
{
  "OSWORLD_EVAL_MODEL_PROVIDER": "aws_third",
  "OSWORLD_EVAL_MODEL_NAME": "anthropic.claude-sonnet-4-6",
  "OSWORLD_EVAL_MODEL_BASE_URL": "${oc.env:OSWORLD_EVAL_MODEL_BASE_URL}",
  "OSWORLD_EVAL_MODEL_API_KEY": "${oc.env:OSWORLD_EVAL_MODEL_API_KEY}",
  "OSWORLD_USER_SIM_PROVIDER": "aws_third",
  "OSWORLD_USER_SIM_MODEL": "anthropic.claude-sonnet-4-6",
  "OSWORLD_USER_SIM_BASE_URL": "${oc.env:OSWORLD_USER_SIM_BASE_URL}",
  "OSWORLD_USER_SIM_API_KEY": "${oc.env:OSWORLD_USER_SIM_API_KEY}",
  "OSWORLD_AWS_THIRD_TIMEOUT": "120"
}
```

示例用于具备相应模型权限的AIHub Bedrock Invoke服务，其他服务需填写匹配的provider。
裁判与模拟器配置独立于被测主模型。仅export顶层变量、但不加入上述对象，不会传入官方worker。
合并时保留已有GitLab等设置。

请求探针（实际调用辅助模型；不是完整题目评测）：

```bash
uv run --no-sync python -m nexus.extensions.tasks.os_world_v2.scripts.check_auxiliary_models \
  --env nexus/extensions/tasks/os_world_v2/.env \
  --repo /data/workspace/OSWorld-V2 \
  --output /tmp/osworld-v2-auxiliary-results.json
```

该开发探针需要固定版本源码目录；正常运行从COS取题包，不依赖此checkout。

### 5.2 GitLab题026/041

选择已提供GitLab、Runner和Pages的tool，在 `v21.environment` 中合并：

```json
{
  "GITLAB_URL": "http://localhost",
  "GITLAB_PAGES_URL": "http://pages.localhost",
  "GITLAB_PRIVATE_TOKEN": "${oc.env:GITLAB_PRIVATE_TOKEN}"
}
```

使用完整URL，不能填裸 `localhost`。token须与tool服务端一致；
平台需在实际worker使用的Rainbow中配置，不自动读取本机 `.env`。
本地默认先等待600秒，GitLab总就绪预算1800秒，可分别用
`OSWORLD_GITLAB_INITIAL_WAIT`、`OSWORLD_GITLAB_READY_TIMEOUT`设置。
部署与验收条件见[设计说明](../developing/osworld-v2-hunyuan-design.md#6-gitlab与题目准备)。

仅验证setup：

```bash
export OSWORLD_SETUP_ONLY=true
uv run --no-sync nexus run-single \
  --use "osworld_v2:model_family=hunyuan,harness=cc,task_id=task_041" \
  -o /tmp/osworld-v2-041-setup.json
unset OSWORLD_SETUP_ONLY
```

026同理。setup-only应为 `scored=false`，不能算有效benchmark得分。
需要复用自己的测试实例时附加
`--set "runtime_configs.main.runtime_credential.instance_id=$INSTANCE_ID"`；
复用实例不会由runner自动停止，创建者负责释放。

### 5.3 025、103、104独立grader

这三题默认额外申请独立AGS评分实例。预装依赖tool推荐设置：

```bash
export OSWORLD_CAD_GRADER_BACKEND=ags
export OSWORLD_CAD_GRADER_TOOL_ID="$GRADER_TOOL_ID"
export OSWORLD_CAD_GRADER_INSTALL_DEPENDENCIES=false
export OSWORLD_CAD_GRADER_TIMEOUT=900
export OSWORLD_PDF_GRADER_BACKEND=ags
export OSWORLD_PDF_GRADER_TOOL_ID="$GRADER_TOOL_ID"
export OSWORLD_PDF_GRADER_INSTALL_DEPENDENCIES=false
export OSWORLD_PDF_GRADER_TIMEOUT=1800
```

`GRADER_TOOL_ID`须由部署者提供，包含CAD和PDF所需依赖；未预装时设
`install_dependencies=true`并确保安装网络可用。grader的sudo密码须匹配task配置。
025仍需配置辅助裁判模型。

平台合入 `nexus_run_input.task_config.v21`：

```json
{
  "cad_grader": {
    "backend": "ags",
    "tool_id": "${oc.env:OSWORLD_CAD_GRADER_TOOL_ID}",
    "install_dependencies": false,
    "timeout": 900
  },
  "pdf_grader": {
    "backend": "ags",
    "tool_id": "${oc.env:OSWORLD_PDF_GRADER_TOOL_ID}",
    "install_dependencies": false,
    "timeout": 1800
  }
}
```

tool留空时仍创建新实例，CAD使用主tool模板，PDF优先使用CAD tool。
`backend=host`要求宿主自行具备完整系统依赖。

## 6. 在平台发起

### 6.1 数据、凭据与代码

1. 从已发布的完整V2数据集导出平台JSONL并上传：

   ```bash
   uv run --no-sync python -m nexus.extensions.tasks.os_world_v2.scripts.export_platform_dataset \
     -o /tmp/osworld-v2-platform.jsonl
   ```

2. 绑定V2评估集。每题的task_id、bundle_sha256、task_files留在数据行，
   不在公共参数中写死。
3. 配置实际worker读取的Rainbow key和第1/5节凭据。
4. `nexus_env.ref`指定支持所选harness与题包的已推送提交。
   切换Codex时加入 `codex` extension，不能只替换agent名字。
5. 先单题、并发1；检查输出及产物后扩大范围。

### 6.2 Agent配置

以下片段放入评估集 `nexus_run_input`；切换harness时完整替换agent配置。

CC：

```json
{
  "agent_name": "claude_code_multiturn",
  "agent_config__override__": {
    "execution_profile": "osworld_v2",
    "working_dir": "/home/user",
    "timeout": 9000,
    "max_turns": 20,
    "enable_user_input_hooks": true,
    "execution_options": {
      "expected_claude_code_version": "2.1.205",
      "guide_version": "v2",
      "max_turns": 500,
      "thinking_preset": "auto",
      "system_grep": true,
      "process_guard": true
    }
  }
}
```

Codex：

```json
{
  "agent_name": "codex_native_bridge",
  "agent_config__override__": {
    "execution_profile": "osworld_v2",
    "execution_options": {
      "version": "0.154.0",
      "model": "gpt-6-astra",
      "guide_version": "v1",
      "model_catalog_preset": "gpt6_catalog_v8",
      "timeout": 9000,
      "session_mode": "auto",
      "max_turns": 20,
      "max_user_requests": 100,
      "hook_input_timeout": 300
    }
  }
}
```

两个guide/catalog preset也可替换为自定义 `guide_text` / `model_catalog_json`。
不要漏掉它们。Codex模型名是catalog身份；实际推理模型在平台模型表单指定。

在 `nexus_run_input.runner.hooks` 配置输出hook，并确认平台最终请求包含此字段：

```json
{
  "runner": {
    "hooks": {
      "output_contract": {
        "name": "osworld_v2_output_contract",
        "args": {}
      }
    }
  }
}
```

### 6.3 主模型表单

- **Hunyuan**：选择有权限的一站式模型服务，平台构造主模型路由；
  不把本地infer_proxy参数直接塞入公共 `nexus_run_input`。
- **Opus + CC/Codex**：选择外部服务URL，填写获授权的AIHub URL及精确模型ID；
  Bedrock Invoke推理超参示例：

  ```json
  {
    "api_key": "<在私有表单填写>",
    "nexus_main_client_name": "anthropic_messages",
    "main_client_args": {
      "api_style": "bedrock_invoke",
      "provider": "aws_third",
      "ensure_v1_path": true,
      "max_retries": 3,
      "timeout": 120
    }
  }
  ```

Hunyuan平台bridge支持有界请求时，可在**任务推理超参**中合并：

```json
{
  "main_client_args": {
    "max_retries": 2,
    "timeout": 600,
    "model_call_total_timeout_s": 900,
    "enable_conditional_retry": true
  }
}
```

Hunyuan + CC还应在 `agent_config.execution_options.extra_env` 合并：

```json
{
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",
  "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "80",
  "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384"
}
```

这些是起始预算示例，需按服务窗口、速度及并发校准。`main_client_args`不属于
评估集的 `nexus_run_input`。使用本节配置前替换代码ref和服务占位符，
并核对所选模型服务的参数。

## 7. 数据发布、缓存与产物

### 发布题包（数据维护者）

只有打包机需要完整固定源码、资产及外部依赖：

```bash
uv run --no-sync python -m nexus.extensions.tasks.os_world_v2.scripts.build_dataset \
  --repo /data/workspace/OSWorld-V2 \
  --assets /data/workspace/OSWorld-V2/cache/osworld_v2_assets_v2.1 \
  --dependencies /path/to/dependencies \
  --dependency-profile task033-py310-20261019 \
  --output /path/to/new-build

uv run --no-sync nexus dataset publish /path/to/new-build \
  "$OSWORLD_V2_DATASET_URI" --no-parquet --visibility PRIVATE
```

使用新的不可变版本，不覆盖既有实验版本。检查构建manifest的缺失依赖；
筛除缺依赖题不代表剩余题全都可评分。发布后重新导出JSONL，
更新平台数据版本；仅修改公共环境变量不会替换每题task_files。

### 本地路径

```text
$NEXUS_HOME/
├── osworld-v2-inputs/archives/<sha256>
└── osworld-artifacts/run-<唯一标识>-<随机后缀>/
    ├── publication.json
    ├── host.tar.gz
    └── files/
        ├── input/repo/
        ├── input/assets/
        ├── official-cache/
        ├── phase-001/
        └── run-result.json
```

`NEXUS_HOME`默认 `~/.nexus`；改变它也会改变Nexus鉴权/缓存位置。
`-o`只指定最终JSON，`-d`只指定batch数据库。
真实产物目录读取 `task_output.artifacts.local_directory`。

下载产物时先加载artifact凭据，使用返回的完整URI：

```bash
uv run --no-sync nexus artifact ls "$ARTIFACT_URI"
uv run --no-sync nexus artifact download "$HOST_ARCHIVE_URI" \
  -o /tmp/osworld-v2-host.tar.gz
```

输入WEDATA_KEY不能用于输出artifact鉴权。`published`仅说明收集的归档上传成功，
不是整个guest文件系统备份。更完整用法见
[Artifact CLI](use-artifacts/artifact-cli.md)。

## 8. 结果检查与排障

```bash
uv run --no-sync python - /tmp/osworld-v2-single.json <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
task = result.get("task_output") or {}
evaluation = task.get("evaluator_result") or {}
print({
    "success": result.get("success"),
    "error": result.get("error"),
    "completed": task.get("completed"),
    "scored": evaluation.get("scored"),
    "score": task.get("score"),
    "agent": task.get("agent_output"),
    "judge": task.get("judge_output"),
    "artifacts": task.get("artifacts"),
})
PY
```

setup-only、运行/评分错误、有效零分是三种不同结果。平台有效性规则可使用：

```text
success == true and error == null and task_output != null and task_output.completed == true and task_output.evaluator_result != null and task_output.evaluator_result.scored == true and task_output.score != null
```

| 现象 | 检查/处理 |
|---|---|
| task_id需要string | YAML/JSON写 `"105"`，run-part写 `task_105` |
| Codex字段不支持/缺guide或catalog | 核对实际代码ref与完整execution_options |
| Sandbox tool localhost not found | `AGS_OSWORLD_TOOL_ID`填授权sdt ID；localhost填网站suffix |
| AGS code310508 | tool未暴露目标端口；更新tool后新建实例 |
| HTTP502 / code330502 | 服务可能尚未启动或不可达；检查服务与对应阶段产物 |
| Cannot allocate V2.1 loopback transport | 在Nexus宿主同一网络命名空间检查下列端口 |
| 026/041长时间准备 | 检查GitLab就绪、token、Runner/Pages，不只看/platform |
| 普通清理失败 | 查看清理诊断与原始CLI退出码，不把它当模型失败 |
| 模型步数超过max_turns | 区分主/子agent、压缩调用和用户轮次 |
| OCR/STEP/截图评分失败 | 查看official-cache独立grader和截图诊断，不强行记零分 |
| artifact下载401 | 检查本机PAT来源、有效期与授权；平台凭据不会自动同步到本机 |

宿主端口检查，显式使用Shell以兼容argv式执行器：

```bash
bash -lc "ss -ltnp | grep -E ':(5000|9222|8080|8000|5910|3000)([[:space:]]|$)'"
```

没有匹配时grep返回1。关注通配地址监听；先确认进程用途再改端口或网络隔离，
不要误杀业务进程。该检查不在AGS guest中执行。

桌面截图单次超时依次30/60/120秒，最多3次。持续控制错误不能仅凭502或-1推断OOM。
`artifacts.release`中的accepted表示Stop请求接受，pending/failed需要按实例ID核查；
释放结果不改变分数，TTL是最终兜底。

实际运行会创建实例、调用模型并产生费用。配置验证不等于完整题目验收，
先完成单题setup、执行、评分及产物检查，再扩大并发。
