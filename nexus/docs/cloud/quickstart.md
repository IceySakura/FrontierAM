# 快速开始

本页带你完成一个 `identity + main_agent` Single 任务：提交到云端，等待结束，读取结果并确认模型返回
`NEXUS_ONLINE_SERVICE_OK`。示例不创建 Runtime、不调用工具，需要有效的模型服务凭据。

命令在同一个 Bash 终端执行，需要 `uv`、`curl`、`jq` 和 `openssl`。首次使用按顺序完成各步；已有环境与凭据时可以复用。

## 1. 准备项目

在 Nexus 源码项目根目录安装 Cloud 提交依赖：

```bash
set -euo pipefail

uv sync --extra cloud
```

然后确认 CLI 可以正常使用：

```bash
uv run nexus --help
```

确认当前项目使用 Nexus `1.7.0` 或更高版本：

```bash
uv run python -c "from importlib.metadata import version; print(version('nexus-agent'))"
```

还没有安装 Nexus 或版本低于 `1.7.0` 时，先完成[安装](../get-started/installation.md)。Cloud 提交会使用当前 Nexus 项目和
同目录的 `uv.lock`；项目根目录的 `pyproject.toml` 必须声明 `project.name = "nexus-agent"`。普通业务项目目录不能直接作为
`--project-root`。

本页从 Nexus checkout 根目录运行，所以不需要显式传 `--project-root`。如果任务代码位于其他目录，再按
[代码与运行环境](submit-batch.md#上传代码与依赖)中的目录约束显式指定。

## 2. 选择运行环境

下面使用正式环境的个人 API 入口。其他部署应成对替换 Control/Application 地址；基址不含末尾 `/v1`。
`NEXUS_ENVIRONMENT` 是该部署中的运行 Environment ID，不是“正式/测试”等部署名称：

```bash
export CONTROL_BASE_URL='https://control-runner-nexus-agent.mcp.woa.com'
export APPLICATION_BASE_URL='https://application-runner-nexus-agent.mcp.woa.com'
export NEXUS_ENVIRONMENT='<dynamic-environment-id>'
export TAIHU_TOKEN_FILE="$HOME/.nexus/auth/taihu-pat"
```

`NEXUS_ENVIRONMENT` 填写 **My Environments** 中已授权 Environment 的 ID。配额申请方式见
[认证与运行环境](access-and-environments.md)。

## 3. 申请太湖 PAT

1. 打开[太湖个人令牌](https://tai.it.woa.com/user/app/auth)，点击 **创建个人令牌**。
2. 填写令牌名称，在 **授权应用** 中搜索并选择 **`nexus-agent`**。
3. 设置过期时间，点击 **下一步** 完成身份验证。
4. 复制生成的个人令牌，并保存到本地私有文件。

![在太湖个人令牌页面选择 nexus-agent](../assets/images/nexus-cloud/taihu-personal-token.png)

创建本地凭据文件：

```bash
mkdir -p ~/.nexus/auth
chmod 700 ~/.nexus/auth
read -rsp 'Taihu PAT: ' TAIHU_PAT && echo
printf '%s' "$TAIHU_PAT" > "$TAIHU_TOKEN_FILE"
unset TAIHU_PAT
chmod 600 "$TAIHU_TOKEN_FILE"
```

`nexus-agent` 授权包含 Nexus Cloud 服务。PAT 只保存在本地私有文件，不要写入 YAML、命令历史或 Git。

## 4. 配置模型服务

示例使用 Model Service：

```bash
export MODEL_SERVICE_API_APP_ID='<app-id>'
export MODEL_SERVICE_API_APP_KEY='<app-key>'
export MODEL_SERVICE_API_BASE_URL='<openai-compatible-base-url>'
export NEXUS_MODEL='gpt-5.5'
```

项目中的示例配置为
[main-agent-single.yaml](../examples/configs/online-service/main-agent-single.yaml)。它要求模型只返回
`NEXUS_ONLINE_SERVICE_OK`，不创建 Runtime，也不调用工具。

## 5. 提交 Single

先为这次逻辑任务准备私有输出目录和幂等键。这一段只执行一次；遇到网络错误时复用保存的 key：

```bash
umask 077
OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nexus-single.XXXXXX")"
IDEMPOTENCY_KEY="single-$(openssl rand -hex 16)"
printf '%s\n' "$IDEMPOTENCY_KEY" > "$OUTPUT_DIR/idempotency-key"
```

提交并保存响应：

```bash
uv run nexus submit single \
  docs/examples/configs/online-service/main-agent-single.yaml \
  --environment "$NEXUS_ENVIRONMENT" \
  --control-url "$CONTROL_BASE_URL" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --wait --json > "$OUTPUT_DIR/submission.json"

jq '{execution_id, terminal, result_url, task_web_ui}' "$OUTPUT_DIR/submission.json"
```

CLI 在 stderr 中输出 accepted 通知，`--wait` 会继续轮询到终态，再向 stdout 输出最终 JSON。
应看到 `terminal.status: succeeded`；`task_web_ui` 可以在浏览器中打开。顶层 `status` 是初次受理时的状态，不代表最终结果。

若提交响应丢失，保留原始配置、代码与凭据值，使用相同 key 重试提交；不要重新执行 key 生成步骤。已经取得 `execution_id`
但等待中断时，直接查询原任务，不必再次提交。完整说明见[任务与结果](runs-and-results.md#提交响应与幂等重试)。

## 6. 读取并确认结果

先检查终态，再通过当前已授权的 Application API 读取完整 `RunOutput`：

```bash
jq -e '.terminal.status == "succeeded"' "$OUTPUT_DIR/submission.json"
EXECUTION_ID="$(jq -er '.execution_id' "$OUTPUT_DIR/submission.json")"

curl --fail-with-body -sS \
  -H "Authorization: Bearer $(cat "$TAIHU_TOKEN_FILE")" \
  "$APPLICATION_BASE_URL/v1/executions/$EXECUTION_ID/result" > "$OUTPUT_DIR/result.json"

jq -e '.success == true and .agent_output.response == "NEXUS_ONLINE_SERVICE_OK"' "$OUTPUT_DIR/result.json"
jq '.agent_output.response' "$OUTPUT_DIR/result.json"
```

预期输出 `NEXUS_ONLINE_SERVICE_OK`，才算完成本例。若不是 `succeeded`，先读取 Control 状态和错误，不要继续假设结果存在。
页面操作见[从控制台读取结果](runs-and-results.md#从控制台读取结果)，排错见[常见问题](troubleshooting.md)。

## 7. 可选配置与下一步

- `--host-env KEY=VALUE` 可重复传入普通环境变量，作用于 Dynamic 任务 Pod，不能与 `--pool` 一起使用。值会进入提交配置和
  Pod 清单，不要传密码或 Token。Console 的 **New Submission → Advanced overrides** 也支持此项。
- `--run-trace-sink <backend-name>` 可重复指定已授权的 Run Trace Backend；省略时使用当前账号的默认配置。查看方式见
  [Run Trace Backend 与明细轨迹](runs-and-results.md#run-trace-backend-与明细轨迹)。
- 打开控制台的 **Runs** 可查看或停止任务。关闭终端、中断本地 `--wait` 都不会停止云端任务。

以上配置改变了提交内容；用于新任务时应生成新的幂等键，不能与刚才的请求共用 key。

完成 Single 后，根据任务类型继续：

- **批量任务**：[Batch 指南](submit-batch.md)，组织输入、调整并发并读取子任务结果。
- **交互任务**：[Live 指南](live.md)，创建任务后读取事件并提交用户输入。
- **新的 Batch 输入或依赖**：[提交前本地验证](local-validation.md)，可选地编译任务、构建 wheel 并校验依赖。
