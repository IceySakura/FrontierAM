# OSWorld 运行手册：完整 YAML、Run Part 与 Batch

本页面向运行者，说明准备环境、发起六模式任务、配置平台、下载产物和排查错误。
实现机制与设计边界见 [OSWorld 设计说明](../developing/osworld-hunyuan-design.md)。

| 入口 | 配置来源 | 适用场景 |
|---|---|---|
| **Run Single 入口一：完整 YAML** | 已保存的完整 RunInput | 审阅、固化、复用实验配置 |
| **Run Single 入口二：Run Part `--use`** | 本次调用按模型族、harness 和题号生成 RunInput | 快速切换组合、单题调试 |
| **Run Batch：同一配置构建链路** | 原生 generator 按数据集筛选生成 RunInput | 多题评测、并发、结果数据库 |

第 2 节只配置环境；第 3、4、5 节分别给出三种入口的六模式命令。
它们是可选的运行方式，**不是需要依次执行的步骤**。

## 1. 六种模型与 harness 组合

`model_family` 选择模型族和 client 配置方式，`harness` 选择执行方式：

| model_family | harness | Agent | 本地模型 Client |
|---|---|---|---|
| `gpt` | `bare` | `cua_gpt_agent` | `openai_responses` |
| `claude` | `bare` | `cua_anthropic_agent` | `anthropic_messages` |
| `gpt` | `codex` | `codex_native_bridge` | `openai_responses` |
| `claude` | `cc` | `claude_code_native_bridge` | `anthropic_messages` |
| `hunyuan` | `codex` | `codex_native_bridge` | `infer_proxy` |
| `hunyuan` | `cc` | `claude_code_native_bridge` | `infer_proxy` |

两个维度独立表达，但当前只支持表中的六种组合，不是全部九种笛卡尔组合。
`bare` 不启动 Codex/Claude Code CLI，仍由 CUA agent 执行截图和动作循环。

AGS 桌面 `/screenshot` 读取最多尝试 3 次，单次 HTTP 和宿主等待超时依次为
30、60、120 秒；重试间隔为 0.5、1 秒。仅瞬态错误或无效图片触发重试，
鉴权失败直接返回，取消操作立即传播。此策略由 V1/V2 共用的任务 runtime 提供。

V2 的六模式、独立环境模板及题目/cache 路径见 **`osworld-v2-runbook.md`**。
V2 使用 `os_world_v2` Task / `osworld_v2` run-part；本文下面的入口只用于 V1。

具体后端模型由 `NEXUS_MODEL` 或 `HUNYUAN_MODEL` 配置。
`CC_MODEL`、`CODEX_MODEL` 是 CLI/catalog 使用的标识，不等于后端路由模型。
服务端仍需支持所选 agent 所需的图像、工具和协议能力。

CC/Codex 配置使用表中的 agent 名，并保留 `execution_profile: osworld` 与
嵌套的 `execution_options`。不要换成普通 `claude_code` / `codex` 入口。

## 2. 公共准备

### 2.1 解释器和源码

使用仓库对应的虚拟环境，并确认加载的是该仓库的源码：

```bash
cd /data/workspace/nexus
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -c 'import sys,nexus; print(sys.executable); print(nexus.__file__)'
```

期望解释器位于 `/data/workspace/nexus/.venv`，源码位于
`/data/workspace/nexus`。使用其他目录时，相应调整仓库路径并检查实际导入位置。

### 2.2 加载私有配置

generator 不自动读取 `.env`。仅对自己信任、**只包含环境变量赋值**且符合 shell
语法的私有文件执行：

```bash
set +x
set -a
source nexus/extensions/tasks/os_world/.env
set +a
```

不要输出凭据、开启 xtrace 或提交私有文件。私有文件建议权限为 `0600`，
无密钥模板见同目录 `.env.example`。若 `.env` 混有运行命令或调试示例，
先把需要的赋值整理到单独文件，再加载；不要直接执行整份文件。

`WEDATA_KEY` 必须是完整 JSON，在 shell 中正确引用：

```bash
# 仅为格式示意，不是真实凭据。
export WEDATA_KEY='{"id":"YOUR_ID","subject":"YOUR_SUBJECT","key":"YOUR_KEY"}'
```

不要使用未引用的 `export WEDATA_KEY={"id":...}`。可以不输出密钥地检查：

```bash
python -c 'from nexus.dataset.auth import load_wedata_key; load_wedata_key(); print("Wedata key JSON valid")'
```

### 2.3 COS 数据与评分依赖

Run Part、batch generator 和 YAML 导出通过数据集 URI 获取线上索引：

```bash
export OSWORLD_DATASET_URI='hy_gui_eval/osworld@v1.1.1'
export OSWORLD_DATASET_SPLIT=test
unset OSWORLD_DATASET_MANIFEST
unset OSWORLD_EXAMPLES_ROOT OSWORLD_TASK_ROOT OSWORLD_TEST_META_PATH
unset OSWORLD_TASK_ID OSWORLD_TASK_IDS OSWORLD_DOMAIN OSWORLD_SKIP OSWORLD_LIMIT

export OSWORLD_VALIDATE_EVALUATOR_HOST_DEPS=1
export NEXUS_OSWORLD_CACHE_DIR="$PWD/.cache/osworld"
```

- **`OSWORLD_DATASET_MANIFEST` 不强制要求**，仅作为显式离线索引覆盖，
  不替代 COS 资源。常规运行保持未设置。
- 已含 `task_files` 的完整 YAML 执行时无需再次用 URI 枚举题目；
  修改 URI 环境变量也不会改变该 YAML 中固定的数据版本。
- `OSWORLD_VALIDATE_EVALUATOR_HOST_DEPS` 是配置生成时的宿主评分依赖预检。
  设为 `0` 只适合配置调试，不会安装依赖或关闭 evaluator。
  直接运行已有完整 YAML 不重新执行 generator 的这项预检。
- setup/evaluator 保留 best-effort 下载回退；同源 COS 不等于严格禁止公网回退。

### 2.4 AGS、代理和结果目录

仓库六份 YAML 使用 `control_plane`：

| provisioning 模式 | 必需变量 |
|---|---|
| `control_plane` | `AGS_SECRET_ID`、`AGS_SECRET_KEY`、`AGS_OSWORLD_TOOL_ID` |
| `template` | `AGS_API_KEY`、`AGS_OSWORLD_TEMPLATE` |

新生成配置可使用以下设置：

```bash
export OSWORLD_AGS_PROVISION_MODE=control_plane
export AGS_REGION=ap-guangzhou
export AGS_DOMAIN=ap-guangzhou.tencentags.com
export OSWORLD_SANDBOX_TIMEOUT=15300
export OSWORLD_SKIP_GUI_START=true
export OSWORLD_COMMAND_AS_USER=false
```

私有文件若只有 `AGS_TOOL_ID`，需显式映射到 `AGS_OSWORLD_TOOL_ID`。
选择 template 时改设 `OSWORLD_AGS_PROVISION_MODE=template` 并准备对应凭据。
**环境变量不会自动将已有 control-plane YAML 改成 template。**

两道大资源题默认走大沙盒策略。template 模式遇到它们会明确拒绝；
仅确认自选模板容量足够时才设置 `OSWORLD_LARGE_SANDBOX_POLICY=false`。

代理默认关闭；启用时需同时配置可用的私有代理文件：

```bash
# 仅在需要时启用。
# export OSWORLD_ENABLE_PROXY=true
# export OSWORLD_PROXY_CONFIG_FILE=/absolute/path/private-proxy.json
# export OSWORLD_FORCE_PROXY=false
```

结果 JSON/数据库由 `-o` / `-d` 指定。需要本地运行记录时可设置：

```bash
# 在生成配置之前设置；每次实验使用独立目录。
# export OSWORLD_SESSION_NAMESPACE="$PWD/runs/my-experiment"
```

`OSWORLD_SESSION_NAMESPACE` 不是 COS 鉴权或启动的必填条件。
修改该环境变量不会给已有 YAML 自动添加轨迹 hook，需重新导出或明确修改配置。
完整轨迹与诊断的暂存位置、COS 上传配置见第 6.1 节，不需要配置 `traj_dump_dir`。

### 2.5 模型服务配置：按模型族准备一次

本节只设置模型服务，不发起任务。选择对应模型族后，执行第 3、4 或 5 节的一条命令。
切换 GPT 与 Claude 时必须同时切换正确的 `NEXUS_*`，不能仅改变 `model_family`。

#### GPT：适用于 bare 和 Codex

沿用私有文件的 `GPT_CUA_*` 命名时显式映射：

```bash
export NEXUS_MODEL="${GPT_CUA_MODEL:?需要 GPT 路由模型名}"
export NEXUS_BASE_URL="${GPT_CUA_BASE_URL:?需要支持 CUA/Responses 的服务地址}"
export NEXUS_API_KEY="${GPT_CUA_API_KEY:?需要该服务的凭据}"
```

若已直接配置正确的 `NEXUS_MODEL/BASE_URL/API_KEY`，不用再次映射。
网关 URL 与凭据路由格式必须匹配，不要假设入口会自动补齐任意网关参数。

#### Claude：适用于 bare 和 Claude Code

以下是 **AIHub Bedrock invoke 网关**示例，不是所有 Claude endpoint
的通用配置：

```bash
export NEXUS_MODEL='anthropic.claude-opus-4-8'
export NEXUS_BASE_URL='http://api.aihub.woa.com'
export NEXUS_API_KEY="${AI_HUB_KEY:?需要该 AIHub 服务的凭据}"
export NEXUS_ANTHROPIC_API_STYLE=bedrock_invoke
export NEXUS_LLM_PROVIDER=aws_third
```

如使用 `/v1/messages` 服务，配置 `NEXUS_ANTHROPIC_API_STYLE=messages`，
并替换为该服务匹配的模型、URL、key 和 provider；不能只改变 style。

`NEXUS_ANTHROPIC_API_STYLE` 应与服务协议匹配；它不选择 harness 或思考强度。
当前 `bedrock_invoke` 配置用于网关服务，不可直接当作 AWS SDK/SigV4 凭据配置。

**完整 YAML 特别注意：** 当前 `osworld_claude_bare.yaml` 的client `api_style`
为 `messages`，`osworld_claude_cc.yaml` 为 `bedrock_invoke`。
请让服务URL、凭据和协议匹配所选文件；改变协议时按第3.7节重新导出YAML。
只 export 环境变量不会改变 YAML 字面量。Run Part 和 batch 每次生成时会读取它。

#### Hunyuan：适用于 Codex 和 Claude Code

```bash
export HUNYUAN_MODEL='YOUR_DEPLOYED_SERVICE_GROUP'
export HUNYUAN_BASE_URL='http://infer-proxy-dev.polaris:8080/v1'
export HUNYUAN_SERVING_PROFILE=claude_passthrough
# 服务需要鉴权时，从私有环境注入 HUNYUAN_API_KEY；否则 client 使用 unused 占位。
```

将服务组占位值换成实际部署名称。本地配置使用 `infer_proxy`，不需要 `NEXUS_API_KEY`。
需要固定请求参数时使用 `config.enforced_request_args`，不要将它放入 `client_args`。
`claude_passthrough` 是 InferProxy 请求配置档位，不表示调用 Claude 模型，
也不是 `NEXUS_ANTHROPIC_API_STYLE` 的取值。

## 3. Run Single 入口一：完整 YAML（六模式）

以下均为完整 RunInput，不是 `compose/use` 包装。先完成公共准备和对应模型配置，
再选择一条命令。执行的题目、agent、runtime 和数据版本以 YAML 为准，
六份示例不保证选择同一道题。

### 3.1 GPT 裸模型

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_gpt_bare.yaml -o /tmp/gpt-bare.json
```

### 3.2 Claude 裸模型

适用于与模板 `messages` style 匹配的服务；Bedrock invoke 服务先重新导出。

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_claude_bare.yaml -o /tmp/claude-bare.json
```

### 3.3 GPT + Codex

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_gpt_codex.yaml -o /tmp/gpt-codex.json
```

### 3.4 Claude + Claude Code

此模板使用 `bedrock_invoke`；其他 style 先重新导出。

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_claude_cc.yaml -o /tmp/claude-cc.json
```

### 3.5 Hunyuan + Codex

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_hunyuan_codex.yaml -o /tmp/hunyuan-codex.json
```

### 3.6 Hunyuan + Claude Code

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_hunyuan_cc.yaml -o /tmp/hunyuan-cc.json
```

### 3.7 换题或改变生成时配置：重新导出

在正确的模型服务、AGS 和数据环境下，指定组合与题号生成新配置。例如，
完成 2.5 节的 Claude Bedrock 配置后：

```bash
python -m nexus.extensions.tasks.os_world.scripts.export_single \
  --model-family claude --harness cc \
  --task-id bb5e4c0d-f964-439c-97b6-bdb9747de3f4 \
  -o /tmp/osworld-claude-cc.yaml
```

随后将第 3.4 节命令中的输入文件替换为导出的文件即可。
其他组合使用表中的 `model-family/harness`；不要只改 `task_config.task_id`
而保留旧题的 `task_files`。

完整 YAML 不会在执行时重跑 generator：

- 字面量保持不变，`${oc.env:…}` 才会在运行时解析。
- 仓库模板保留了模型路由和凭据的环境引用；当前导出器会固化读取到的模型名、
  URL 等值，同时保留凭据引用，二者不保证完全相同的引用方式。
- 修改 `OSWORLD_MODEL_FAMILY/HARNESS`、provisioning 或 CLI 预算变量，
  不会自动重建已有 YAML；请重新导出。
- `--all-modes` 可一次导出六份文件，但 GPT 与 Claude 都读取同一组 `NEXUS_*`。
  它不会自动切换服务凭据；通常应分别加载服务配置、按模式导出。

## 4. Run Single 入口二：Run Part `--use`（六模式）

本入口在每次调用时读取公共环境并生成配置，因此适合切换模型族、harness 和题目。
`--use` 是 Nexus 正式 Run Part 功能，使用两个连字符。

先设置单题 ID：

```bash
export TASK_ID=bb5e4c0d-f964-439c-97b6-bdb9747de3f4
```

### 4.1 GPT 裸模型

```bash
nexus run-single --use "osworld_native:model_family=gpt,harness=bare,task_id=$TASK_ID" \
  -o "$PWD/runs/my-experiment/single.json"
```

### 4.2 Claude 裸模型

```bash
nexus run-single --use "osworld_native:model_family=claude,harness=bare,task_id=$TASK_ID" \
  -o "$PWD/runs/my-experiment/single.json"
```

### 4.3 GPT + Codex

```bash
nexus run-single --use "osworld_native:model_family=gpt,harness=codex,task_id=$TASK_ID" \
  -o "$PWD/runs/my-experiment/single.json"
```

### 4.4 Claude + Claude Code

```bash
nexus run-single --use "osworld_native:model_family=claude,harness=cc,task_id=$TASK_ID" \
  -o "$PWD/runs/my-experiment/single.json"
```

### 4.5 Hunyuan + Codex

```bash
nexus run-single --use "osworld_native:model_family=hunyuan,harness=codex,task_id=$TASK_ID" \
  -o "$PWD/runs/my-experiment/single.json"
```

### 4.6 Hunyuan + Claude Code

```bash
nexus run-single --use "osworld_native:model_family=hunyuan,harness=cc,task_id=$TASK_ID" \
  -o "$PWD/runs/my-experiment/single.json"
```

### 4.7 参数优先级与覆盖

Run Part 只接受 `task_id`、`model_family`、`harness`：

- `task_id` 必填；服务模型、URL、key 通过 2.5 节配置，不是 Run Part 参数。
- 显式模式参数优先于 `OSWORLD_MODEL_FAMILY`、`OSWORLD_HARNESS`；
  未传的维度可读取环境变量。
- 显式单题 ID 优先于批量 ID/skip/limit，但仍尊重 `OSWORLD_DOMAIN`
  和显式 `OSWORLD_TEST_META_PATH` 的选题范围。题目不存在或被排除时会报错。
- 切换一个正交维度只需修改对应参数，但切换 GPT/Claude 时仍需更新服务凭据。
- 后置 `--set` 可覆盖生成后的字段。例如 CC 的
  `--set agent_config.max_turns=20` 和 `--set agent_config.timeout=1800`
  是支持的覆盖；其他策略字段通常位于
  `agent_config.execution_options`。它不会绕过配置生成阶段的校验。

不要通过单独覆盖 `agent_name` 或 `task_id` 留下不匹配的关联配置。
完整语法见 [Run Part 与 `--use` 使用指南](run-parts.md)。

## 5. Run Batch：同一配置构建链路（六模式）

六个固定 generator 与 Run Part/YAML 导出共用配置构建函数，
只绑定模型族和 harness，不受同名模式环境变量影响。

首次测试先限定一题、单并发：

```bash
export OSWORLD_TASK_IDS=bb5e4c0d-f964-439c-97b6-bdb9747de3f4
export OSWORLD_LIMIT=1
```

### 5.1 GPT 裸模型

```bash
nexus run-batch -g nexus/extensions/tasks/os_world/generators/generator_native_gpt_bare.py \
  -d /tmp/gpt-bare.db --max-concurrent 1
```

### 5.2 Claude 裸模型

```bash
nexus run-batch -g nexus/extensions/tasks/os_world/generators/generator_native_claude_bare.py \
  -d /tmp/claude-bare.db --max-concurrent 1
```

### 5.3 GPT + Codex

```bash
nexus run-batch -g nexus/extensions/tasks/os_world/generators/generator_native_gpt_codex.py \
  -d /tmp/gpt-codex.db --max-concurrent 1
```

### 5.4 Claude + Claude Code

```bash
nexus run-batch -g nexus/extensions/tasks/os_world/generators/generator_native_claude_cc.py \
  -d /tmp/claude-cc.db --max-concurrent 1
```

### 5.5 Hunyuan + Codex

```bash
nexus run-batch -g nexus/extensions/tasks/os_world/generators/generator_native_hunyuan_codex.py \
  -d /tmp/hunyuan-codex.db --max-concurrent 1
```

### 5.6 Hunyuan + Claude Code

```bash
nexus run-batch -g nexus/extensions/tasks/os_world/generators/generator_native_hunyuan_cc.py \
  -d /tmp/hunyuan-cc.db --max-concurrent 1
```

### 5.7 选题、结果和通用 generator

- 批量筛选使用 `OSWORLD_DOMAIN`、`OSWORLD_TASK_IDS`、`OSWORLD_SKIP`、
  `OSWORLD_LIMIT`。确认需要全量后再取消这些限制及不需要的 curated meta。
- 需要 JSONL 时，在所选命令末尾追加
  `--export-format jsonl --export-path /tmp/results.jsonl`。
- 也可将 `-g` 的值换为 `nexus/extensions/tasks/os_world/generators/generator_native.py`，
  同时设置 `OSWORLD_MODEL_FAMILY` 与 `OSWORLD_HARNESS`，使用同一个通用入口。
- `-g` generator 模式不与 `--use/--set` 混用。
- 换模型、harness 或评测口径后使用新的 DB；恢复旧 DB 不会自动改写已经保存的配置。

## 6. 配置行为与执行参数

三种入口共享执行和评分机制，但读取配置的时机不同：

| 项目 | 完整 YAML | Run Part / Batch |
|---|---|---|
| 配置来源 | 已保存的 RunInput | 本次调用生成 RunInput |
| 模式、题目、runtime | 文件中的配置 | 模式参数/固定 generator、环境和数据集 |
| 生成时环境变量 | 不重新读取以重建配置 | 生成时读取 |
| `${oc.env:…}` | 执行时解析 | 构建后由运行入口解析 |
| URI 索引枚举 | 已含 task_files 时不需要 | 需要 |
| 正式运行 | 申请 runtime、执行 agent、评分 | 相同 |

### 6.1 结果输出、产物上传与下载

`nexus run-single ... -o single.json` 写入最终 RunOutput（含任务结果、agent
输出和错误）。它不指定截图、逐步日志或清理前诊断文件的目录。
每次运行自动在
`~/.nexus/osworld-artifacts/`（随 `NEXUS_HOME`）创建唯一的暂存目录，
包含截图、CC/Codex 日志、清理前诊断和 `run-result.json`。
`trajectory_dir` 只是 task 向 agent 内部传递的临时路径，不需要用户配置。

运行产物统一使用 **artifact 服务**，自动发布，无需选择上传后端或输出 dataset。
输入数据仍使用 dataset/task_files，`WEDATA_KEY` 仍用于输入数据鉴权，
不能替代 artifact 服务的太湖 PAT。

**本地 single / Run Part / batch：**在实际启动 Nexus 的 shell 中配置
`ARTIFACT_SERVICE_URL` 和 `ARTIFACT_TAIHU_PAT`，或先执行
`nexus artifact login` 保存 PAT（`~/.nexus/auth/taihu`，也支持
`ARTIFACT_TAIHU_PAT_FILE`）。例如：

```bash
export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
read -rsp 'Taihu PAT: ' ARTIFACT_TAIHU_PAT; export ARTIFACT_TAIHU_PAT; echo
nexus artifact whoami
nexus run-single tests/e2e/config/osworld_hunyuan_cc.yaml -o single.json
```

六模式、Run Part 和 batch 的发起命令不变，均读取同一宿主凭证。
`.env` 必须先由启动进程加载，不会因为文件存在就自动生效；
含 shell 示例的 `.env` 不应直接 `source`。

**太极平台：**优先由平台给实际运行 Nexus 的 host 进程注入上述两个环境变量，
这时 task_config 无需额外字段。注意 `nexus_env_rainbow_key` 是配置插值取值表，
**不等于将其中变量导出到远端 host 的 os.environ**。
如果仅能使用七彩石插值，将以下片段合入评估集自定义参数
（不要替换整份配置，也不要放进推理超参或数据集单题行）：

```json
{
  "nexus_env_rainbow_key": "<实际七彩石key[:version]>",
  "nexus_run_input": {
    "task_config": {
      "artifact_taihu_pat": "${oc.env:ARTIFACT_TAIHU_PAT}",
      "artifact_service_url": "${oc.env:ARTIFACT_SERVICE_URL,https://artifact-nexus-agent.mcp.woa.com}"
    }
  }
}
```

七彩石对应配置：

```yaml
default:
  ARTIFACT_TAIHU_PAT: <通过平台安全配置填写真实PAT>
  ARTIFACT_SERVICE_URL: https://artifact-nexus-agent.mcp.woa.com
```

**原始平台请求/配置日志可能含解析后的凭证**，必须限制这类日志的访问权限。
若需要令牌完全不进入 RunInput，应使用第一种 host 环境变量/凭证文件注入方式。
凭证优先级为本次 task 显式字段 > host 环境变量 > host 登录文件。
模型服务 URL、模型名与推理超参不因上传配置而改变。
可合并的 JSON 示例见 `docs/examples/configs/platform_osworld_artifact.json`。

任务结束后检查：

| 字段/文件 | 查看内容 |
|---|---|
| `task_output.artifacts.uri` | 本题 task 产物目录 URI |
| `task_output.artifacts.status` | `published`、`partial` 或 `upload_failed` |
| `task_output.artifacts.upload_failed` / `attempts` | 是否存在上传失败、已尝试次数 |
| `task_output.artifacts.runner_details` | 完整 Agent 输出、processor 对话和 runner 错误的独立归档目录及上传状态 |
| 暂存目录 `publication.json` | 完整 URI 列表、沙盒快照状态和 `sandbox_direct` / `host_relay` 传输方式 |
| `host.tar.gz` | 宿主截图、轨迹、评分结果与诊断 |
| `cc-*.tar.gz` / `codex-*.tar.gz` | 沙盒日志、退出码与进程身份快照 |

每次运行有独立的 `/tasks/YYYY/MM/DD/osworld-run-<随机ID>/` 目录。
使用结果中的 URI 下载：

```bash
nexus artifact download '<task_output.artifacts.uri>/host.tar.gz' -o ./restored
nexus artifact download '<task_output.artifacts.runner_details.uri>/host.tar.gz' -o ./runner-details
```

上传失败最多追加 3 次重试，每次尝试最多 300 秒；上传失败不会改变已有 score。
`partial` 时检查 `publication.json` 和 `runner_details` 中的失败信息，再下载已成功的产物。
`host_relay` 表示已通过宿主中转，不等于产物丢失。

平台中对应路径为 `trial_details.run_output_data.task_output.artifacts`，
任务成功或执行异常都使用此字段。异常时保留 `success=false`、原 error，
部分 task_output 的 `completed=false`、`score=null`，不把失败伪装成零分。
任务未进入执行、进程被强杀或 runner 无法返回结果时，不能保证产物存在；
已结束的历史任务不会自动补写新字段。

缺少有效 PAT 或上传失败时，保留本地暂存目录，
`task_output.artifacts.local_directory` 指向该目录，`publication.json` 记录发布状态。
可手动执行 `nexus artifact publish <暂存目录>/files --task osworld-recovery` 补传宿主产物；
未上传的沙盒快照仍需在实例释放前恢复，宿主补传无法恢复已经销毁的沙盒文件。
截图和模型输出可能含敏感数据，发布前请确认 artifact 权限。

`-o` 仍保存最终 RunOutput；`OSWORLD_SESSION_NAMESPACE` 仅用于本地结果目录的源码记录，
轨迹由 task 的暂存目录管理。task 归档的 `run-result.json` 保存完整 task 详情；
失败时另存 `agent-output.json`。task 返回后，输出 hook 将完整顶层 Agent、
`processor_results` 对话和错误堆栈保存在独立 runner-details 归档的 `run-result.json` 中。
两个归档分别发布，后者不覆盖先前的 task 归档。

`-o` 只保存结果、状态、原始 CLI 码及产物目录引用，不内嵌 transcript、trajectory、
截图、setup 执行日志或完整 getter 观测。超长 final_answer/error 在摘要中截短，
完整内容查看归档。六模式默认输出覆盖超过 6 MiB 的成功/失败样本回归，
落盘小于 1 MiB。若自定义其他输出 hook 在此之后追加大字段，需要自行约束其输出。

`processor_results` 中的平台统计不是轨迹正文，输出 hook 保留：

- `token_counter`：所有客户端的原始字段、token 数、调用次数与费用，不截断统计值。
- `bridge_toolcall_anomaly`：请求数、步骤数、异常/重试数、比率和异常分类计数；
  逐请求 `requests` 明细只在 runner-details 归档中保存。
- `execution_timeout_tracker`（启用时）：执行时间、预算及终止原因等统计。

平台 bridge 在 nexus 返回后读取这些字段生成 `token_usage`、工具异常和超时统计。
因此不能将 `processor_results` 整体清空，也不能仅将统计放进 artifacts 代替。
未启用的统计不会凭空填零。

平台评测集 `custom_parameters` 应配置以下规则（不是放进 `nexus_run_input`）：

```json
{
  "nexus_run_valid_rule": "task_output.status == 'success'"
}
```

规则允许正常零分、不可完成题，以及已完成评分的预算/上下文/输出长度限制结果；
拒绝 `agent_error`、`judge_error` 和缺少标准状态的结果。
仅合并上面的 `nexus_run_valid_rule` 字段，
不要用片段替换整份评测集自定义参数。历史任务不会因规则更新自动重算。

六模式 YAML 和 generator 已配置输出 hook。平台手工配置 `nexus_run_input` 时，
将以下内容合并到现有 `runner.hooks`，保留其他run hook。
Hook在runner初始化时加载，并在任务结束后整理最终结果与产物引用。

```yaml
runner:
  hooks:
    output_contract:
      name: osworld_output_contract
      args: {}
```

`task_output.agent_output`、`task_output.judge_output` 分别提供
`exit_status`、语义 `exit_code` 和实际 `error`。Hook 将标准 Agent 输出同步至
顶层 `agent_output`，归档并移除顶层对话/轨迹。
对话查看器需要的`_default_chat_trace`及已启用的`_default_atif_trace`保留到bridge后处理，
由bridge保存轨迹文件后删除。直接本地运行时，这两个已生成字段仍保留在RunOutput中。
任务正常返回时直接提供标准阶段输出；任务抛异常时，由此 hook 在归档前补回失败摘要
（包括状态、阶段错误及 artifacts），保留 runner 的 `success=false` 和原错误。
平台必须保留此 hook 才能取得失败 task_output 并精简完整 RunOutput；
未配置时沿用通用 runner 行为，异常任务的 task_output 为 null。
语义码只取 `0/-1`，原 CLI 码查看 `process_exit_code`、
`cc_return_code` 或 `codex_return_code`，不要用它们判断平台运行有效性。
不可完成题查看 `agent_output.conclusion=infeasible`；它的标准状态为 `finished`。
`run_evaluator: false` 仅用于调试，输出 `judge_error`，不计为有效评测。
详细分类见 `docs/developing/osworld-hunyuan-design.md` 的“阶段输出契约”。

### 6.2 OCR 下载与命令超时排查

#### AGS 实例释放

OSWorld自己创建的实例在任务产物和runner详情上传后关闭本地连接，再发送一次停止请求，
最多等待5秒。`task_output.artifacts.release`记录每个实例的ID、tool、已知TTL和状态：

- `stop_accepted`：停止API已返回成功，不代表实例已经达到`STOPPED`。
- `stop_failed`：本次请求返回错误，记录异常类型、错误码及可用的request ID。
- `stop_pending`：5秒内没有结果，继续返回任务主结果。
- `detached`：外部复用实例仅解除本地连接，不发送停止请求。

停止错误不会改变已完成的分数、任务状态或原始异常。不额外睡满5秒，也不重试停止API。
本地连接清理另有最多5秒预算；停止请求由专用短超时SDK客户端执行，
不占用会在事件循环退出时被等待的默认线程池。template模式使用对应SDK的单次kill。
已配置的实例TTL是最终兜底；`stop_pending`不能当作释放成功，仍需按实例ID核查。

释放状态还写入`artifacts.local_directory/release-status.json`。该文件在释放时才产生，
不回写此前已发布的不可变归档；平台保存的最终响应包含此次释放状态。
晚到的SDK结果仅记入进程日志，不修改已返回的状态快照。
动态创建且要求删除的tool会记录`tool_cleanup_pending`，待实例生命周期确认后清理。

`compare_image_text` 在评分宿主侧使用 EasyOCR；只给 AGS 镜像预装模型不一定能覆盖
宿主评分进程。模型缓存沿用 EasyOCR 的 `EASYOCR_MODULE_PATH` / `MODULE_PATH`
或默认 `~/.EasyOCR` 下的 `model` 目录。

- 缓存命中且 MD5 正确时不下载。缺失或损坏时，仅下载默认英文 OCR 的 craft 和 english_g2 模型。
- **每个模型最多尝试 3 次（首次 + 2 次重试），每次下载及解压总时限 600 秒**，
  尝试之间等待 2 秒。单个模型连续失败约 1804 秒后退出（不含等待其他 case 的缓存锁时间）。
- 下载耗尽或缓存不可用时，检查评分宿主的缓存权限、磁盘空间和下载网络。
  `EvaluatorInfrastructureError` 不是模型正常 0 分；修复环境后重新运行。
- 只重试资源下载，不重跑 agent。这不是整个 OCR 推理的超时限制。

出现 `/execute HTTP deadline exceeded` 时：

1. 查看日志中的 deadline 和尝试次数；有明确 deadline 的请求最多尝试 3 次。
2. 检查实例是否仍在运行、控制服务是否可达，并查看已保存的 CC/Codex 诊断。
3. **不要直接手工重放原命令**：HTTP 超时不代表命令未执行，内置重试也可能产生重复副作用。
4. 不要把 HTTP 超时与 `agent_config.timeout` / 最大轮数混为一谈，按最终任务错误和评分状态判断。

### 6.3 Harness 参数

OSWorld CC 默认使用系统 GNU grep，镜像需提供 `/usr/bin/grep`。
完整 YAML / 平台配置中的位置为：

```yaml
agent_config:
  execution_options:
    system_grep: true
```

以下是当前 generator 默认值，不是自动探测的模型能力；已有 YAML 以实际字段为准。

| 范围 | 参数 | 默认值 |
|---|---|---|
| CC | `CC_VERSION` / `CC_MODEL` | `2.1.205` / `claude-opus-4-8` |
| CC | `CC_THINKING_PRESET` | `auto` |
| CC | `CC_MAX_TURNS` / `CC_TIMEOUT` | `100` / `1800` 秒 |
| Codex | `CODEX_VERSION` / `CODEX_MODEL` | `0.154.0` / `gpt-6-astra` |
| Codex | `CODEX_CONTEXT_LIMIT` / `CODEX_MAX_OUTPUT_TOKENS` | `262144` / `64000` |
| Codex | `CODEX_CONTEXT_TOKEN_MARGIN` | `4096`，默认压缩阈值为 `194048` |
| Codex | `CODEX_REASONING_EFFORT` / `CODEX_REASONING_SUMMARY` | `none` / `none` |
| Codex | `TASK_TIMEOUT` | `5400` 秒 |

CC 完整 YAML / 平台 `nexus_run_input` 使用标准顶层预算：

```yaml
agent_config:
  execution_profile: osworld
  timeout: 1800
  max_turns: 100
  execution_options:
    guide_version: v2
```

`CC_TIMEOUT`、`CC_MAX_TURNS` 分别生成上述 `timeout`、`max_turns`。
显式顶层值优先于 `execution_options` 中的值；省略顶层时仍支持已有嵌套配置。
旧 `execution_options.cc_timeout` 仅作为输入别名，内部统一为 `timeout`；
同层同时配置两个名称时 `timeout` 优先。建议不要重复配置同一预算。
这是 Agent 执行预算，不是 `runner.timeout`、实例存活时间或模型请求超时。

- 固定 CC 版本与镜像内 `claude --version` 精确匹配时跳过 npm，否则需要可用的 npm 源。
- CC 工作目录默认是沙盒用户的 `$HOME`；需要时设置
  `agent_config.execution_options.working_dir`，确保该用户有写权限。
- CC 可选防误杀 Hook：`execution_options.process_guard: true`；
  generator 使用 `OSWORLD_CC_PROCESS_GUARD=true`。启用会限制工具命令，应记录实验开关。
- CC 主会话终态后的后台等待上限：
  `execution_options.parent_terminal_grace_seconds: 60`，设为 `null` 禁用。
  它不同于整段执行的 `agent_config.timeout`；判定与停止条件见设计说明。

`execution_options` 按对应 profile 的配置模型校验。未声明的字段会被拒绝；
平台与完整 YAML 使用同一字段结构。

Codex 完整 YAML 使用 `guide_version: v1` 与
`model_catalog_preset: gpt6_catalog_v8` 引用包内资源，无需内嵌全文。
此处 v1 是 **Codex** 指南版本，与 Claude Code 的 `guide_version: v2` 独立。
显式 `guide_text` / `model_catalog_json` 仍优先于对应选择器；preset 预算默认
保持 262144 上下文、64000 输出预留、4096 margin、194048 压缩阈值。
详细覆盖规则见 `nexus/extensions/tasks/os_world/assets/codex/README.md`。

更改这些生成时参数后，Run Part/batch 在下次构建时生效；完整 YAML 需重新导出。

## 7. 太极平台运行

本地与平台均由 `task.py` 使用远程 `task_files` 从 COS 取件。
平台不执行本地 generator，而是接收逐题数据、公共 RunInput 和任务级模型配置。
Hunyuan 一站式服务模式由平台注入 `bridge_litellm`，bridge 不对它套用 Infer Proxy 策略。
该模式导出的公共配置不包含本地 client 或其 enforced 参数。
外部URL表单不能表达 `enforced_request_args`，导出器对此明确报错而不是静默丢弃；
不要把它塞入 `main_client_args`。如需平台服务模式，使用 `--model-source platform-service`。

评估集参数示例见 `docs/examples/configs/platform_osworld_v1.json`，发布时更新代码 ref。

平台的 `nexus_env.ref` 必须指向包含当前原生 profile 接口、且平台可获取的代码版本。
发布前应验证所选代码版本能够解析并执行该配置。


### 7.1 导出平台配置

同一次 generator 输出机械拆为：

1. 数据集 JSONL：逐题 `domain/task_id/task_files`；
2. 评估集 `nexus_run_input`：task/runtime/agent 公共配置；
3. 创建评测任务时的模型表单：服务 URL、模型名称和
   `parameter_configuration`。

示例（Claude + CC；其他模式只改两项模式变量和模型 client）：

```bash
export OSWORLD_DATASET_URI='hy_gui_eval/osworld@v1.1.1'
export OSWORLD_DATASET_SPLIT=test
unset OSWORLD_DATASET_MANIFEST OSWORLD_EXAMPLES_ROOT OSWORLD_TASK_ROOT

export OSWORLD_MODEL_FAMILY=claude
export OSWORLD_HARNESS=cc
export NEXUS_MODEL='anthropic.claude-opus-4-8'
export NEXUS_BASE_URL='http://api.aihub.woa.com'
export NEXUS_ANTHROPIC_API_STYLE=bedrock_invoke
export NEXUS_LLM_PROVIDER=aws_third
# 安全注入 WEDATA_KEY、AGS_SECRET_ID/KEY、NEXUS_API_KEY

python -m nexus.extensions.tasks.os_world.scripts.export_platform_dataset \
  --generator nexus.extensions.tasks.os_world.generators.generator_native \
  -o /tmp/osworld-v1-dataset.jsonl \
  --eval-set-out /tmp/osworld-v1-common.json \
  --client-form-out /tmp/osworld-v1-model.json
```

导出器会拒绝字段漏分、公共/逐题字段重叠、逐题插值、逐题 `task_files`
与本地配置不同、或平台公共值无法覆盖全部题目的情况。

### 7.2 参数填写位置与优先级

依据《Nexus Benchmark 接入平台操作指南》§2.2.5，普通字段从低到高：

1. Default：`nexus_task_rainbow_key`；
2. Customize：数据集行 `meta_data_.nexus_run_input` / `nexus_key_mapping`；
3. Dataset Version：评估集 `nexus_run_input`；
4. User Override：任务启动时 `nexus_run_input_`（末尾有下划线）。

因此：

- `domain/task_id/task_files` 只放 Customize。评估集再写一份会静默覆盖每题值。
- 公共 task/runtime/agent 配置放 Dataset Version。
- 临时用户覆盖最高，但受 `__fix__` 特殊语义影响。
- 两道大沙箱题的叶子路径使用数据行 `__fix__`；普通高优先级字段跳过它。
- `__override__` 可以整体替换，即使底层是 `__fix__`。不要滥用。

### 7.3 七彩石与 WEDATA_KEY

`WEDATA_KEY` 属于公共环境/凭据，不放数据集行。评估集公共配置中：

```json
{
  "task_config": {
    "wedata_key": "${oc.env:WEDATA_KEY}"
  }
}
```

并在 `nexus_env_rainbow_key` 对应空间提供完整 JSON 对象。平台只在
Dataset Version 层解析 `${oc.env:...}`；数据集行 Customize 和
User Override 中的同类字符串不会解析。

`WEDATA_KEY` 不明文落评估集、数据集、文档或命令日志。
`AGS_SECRET_ID/KEY`、代理路径等公共环境变量遵循相同规则。

Rainbow version 配置是浅合并；嵌套 dict 会整块替换。数值/布尔优先写字面类型，
不要对 Rainbow 已带类型的值再套 `${oc.decode:...}`。

### 7.4 模型表单

平台会从评测任务的模型来源重新构建 `openai_clients.main`。因此：

- 评估集 `nexus_run_input` 不保留 `openai_clients`；
- 模型服务 URL / 模型名 / client args 使用第三份模型表单；
- API key 与 `main_client_args` 同级；
- 本地 `NEXUS_*` 只用于构建与导出；平台使用模型表单/Rainbow。

### 7.5 核对数据集行

```json
{
  "id": "chrome__TASK_ID",
  "meta_data_": {
    "instruction": "...",
    "domain": "chrome",
    "nexus_run_input": {
      "task_config": {
        "domain": "chrome",
        "task_id": "TASK_ID"
      },
      "task_files": {
        "type": "zip",
        "path": "blobs/TASK_ID.zip",
        "sha256": "...",
        "size": 1234,
        "remote": {
          "namespace": "hy_gui_eval",
          "dataset": "osworld",
          "version": "v1.1.1",
          "split": "test"
        }
      }
    }
  }
}
```

`size` 必须填写，逐题行不能包含 `${oc.env:...}`。

### 7.6 清除本地路径

dataset-backed 配置导出时会移除：

- `task_root/task_file`；
- `cache_dir/setup_cache_dir`；
- `runner/processors/run_id`。

这些是本机执行状态，不是题目数据。平台运行需要的 task JSON 和资源已经在
逐题 `task_files` 中。

### 7.7 发布前检查

发布前应检查：

- URI-only 能枚举预期题集，远端索引与发布清单一致；
- 六模式的导出配置可解析，每行 `task_files` 与本地 RunInput 指向相同数据；
- 公共层不含逐题 `task_files`、本地路径或重复的模型 client；
- 平台 SDK 合并与 Rainbow resolver 正确解析所需字段；
- 使用授权 WEDATA_KEY 可读取 COS task.json。

配置与数据检查不能代替真实 AGS/模型冒烟及完整评测。平台正式接入还需：

- `nexus_env.ref` 使用与配置匹配的已提交 SHA，并列全 extensions；
- 使用实际存在的 `nexus_env_rainbow_key`；
- 按模型来源填写模型表单；
- 使用平台的数据集版本，而不是把本地 JSONL 路径写进 Run Input。

COS 预取当前仍有公网下载兜底；“本地/平台同源 COS”不等于严格零公网策略。

## 8. 网络与现场排障

1. 按第 2.4 节启用代理；`setup_results` 的 proxy success 只代表启动探针通过，
   不保证所有网站可用。配置只对新任务生效，不会热修改在跑实例。
2. 在独立诊断沙盒分别检查普通 curl/Python、CC WebFetch、Chrome 页面和本地 Bridge；
   不要在正在评分的沙盒中插入调试动作。
3. `cc_env.json` 不包含完整环境，不能因缺少代理变量就判断未透传。
   只检查实际进程环境中的非敏感字段，禁止输出完整私有 env。
4. 使用工具的实际 HTTP 状态和页面内容定位问题。403、验证码或 429 不等于代理未启动；
   不要绕过验证码、自动换站点或改分。
5. `/file` 404 时核对预期文件是否存在、路径是否正确，并查看完整异常链；
   timeout、HTTP 5xx、鉴权或参考文件获取失败需按基础设施错误排查。
6. CDP 失败时分别检查 Chrome 进程、端口、HTTP 发现和 WebSocket 连接。
   不要在评分结束时启动空白浏览器掩盖原现场。
7. 下载 artifact 后查看 `cc_diagnostics.json` / `codex_diagnostics.json`、
   stdout/stderr、`failure_kind` 和 `failure_stage`。诊断尾部不等于完整轨迹；
   沙盒销毁前未取得的文件不能事后恢复。
