# Frontend Task

在空目录下，通过一句话 prompt 驱动 Claude Code agent 生成完整的前端项目。输出为整个工作目录的 tar.gz 压缩包（包含图片、数据库等二进制资源），而非 git diff。

## 架构

```
Host (本地)                              Runtime Docker (沙箱)
├── generator.py                         ├── Claude Code CLI
│   读取 JSONL 数据集                      │   根据 prompt 生成前端项目
│   构建 BatchItem 配置                    │   写入 /workspace
├── task.py                              ├── _tar_working_dir()
│   加载 prompt                            │   打包 /workspace 为 tar.gz
│   启动 agent → 收集 archive              │   base64 编码返回
└── LLM Client (litellm)                 └── Anthropic Proxy → 转发 LLM 请求
```

## 快速开始

### 1. 准备数据集

JSONL 格式，每行一个实例，必须包含 `instance_id` 和 `problem_statement`（或 `prompt`）字段：

```jsonl
{"instance_id": "{prompt_md5值}", "problem_statement": "做一个极简风格的建筑事务所官网，首页全屏大图，案例区瀑布流布局"}
{"instance_id": "{prompt_md5值}", "problem_statement": "实现一个电路模拟器前端，支持拖拽电阻电容到画布上连线"}
```
or 
```jsonl
{"prompt": "做一个极简风格的建筑事务所官网，首页全屏大图，案例区瀑布流布局"}
{ "prompt": "实现一个电路模拟器前端，支持拖拽电阻电容到画布上连线"} # 自动把prompt md5作为instance_id
```

### 2. 首次运行

nexus项目为整个LLM部门的合作项目，自己的工作目录请新建一个workspace文件夹，workspace文件夹是gitignore的。首次运行需要创建uv venv环境：

```bash
cd nexus
mkdir workspace
cp nexus/extensions/tasks/frontend/generator.py workspace/.

# 创建venv环境
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
nexus modules install frontend

# 首次运行：
#   - 模型服务：以下示例调用太极一站式部署的hy3模型，请保证相应的服务有实例 (下面示例为HY3-dev0413)
#   - 沙盒资源token：个人沙盒门户账号token请在https://sbox.woa.com/?tab=resource申请
FRONTEND_DATASET_PATH=path/to/dataset.jsonl FRONTEND_RUNTIME_TOKEN={沙盒门户账号token} LLM_PROVIDER=hy3 OPENAPI_MODEL_NAME="openai/HY3-dev0413" \
  uv run nexus run-batch \
    -g workspace/frontend/generator.py \
    -d output.db
```

### 3. 导出结果

```bash
uv run nexus export output.db -o output.jsonl
```

### 4. 校验与解压

```bash
# 校验所有压缩包
python workspace/utils/verify_archives.py output.jsonl

# 校验并解压到目录
python workspace/utils/verify_archives.py output.jsonl --extract-to ./projects
```

## 环境变量

### 数据集与过滤

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `FRONTEND_DATASET_PATH` | JSONL 数据集路径 | `root/swe-bench-no-hints.jsonl` |
| `FRONTEND_INSTANCE_IDS` | 逗号分隔的 instance_id 列表，只跑指定实例 | 全部 |
| `FRONTEND_SKIP` | 跳过前 N 条 | 0 |
| `FRONTEND_LIMIT` | 最多跑 N 条 | 不限 |

### Runtime 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `FRONTEND_RUNTIME_PROVIDER_NAME` | Runtime 提供者 | `gongfeng` |
| `FRONTEND_RUNTIME_TOKEN` | 沙箱 API Token | 需提前申请 |
| `FRONTEND_RUNTIME_API_BASE_URL` | 沙箱 API 地址 | `http://hyrl-sandbox.prod.woa.com:8052` |
| `FRONTEND_RUNTIME_CONTAINER_IMAGE` | 主容器镜像 | `mirrors.tencent.com/hunyuan_yanguan/webenv.claudecode.testset:v3` |
| `FRONTEND_RUNTIME_IMAGE` | Nexus runtime 镜像 | `current`（自动解析为当前 Host 版本） |
| `FRONTEND_WORKING_DIR` | Agent 工作目录 | `/workspace` |

### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供者：`hy3` 或 `opus` | `hy3` |
| `ENABLE_THINKING` | 是否启用 thinking 模式 | `false` |
| `REASONING_EFFORT` | 推理强度 | hy3: `high`，opus: `max` |

**hy3 模式**（`LLM_PROVIDER=hy3`，默认）：
hy3模式的请求链路走太极平台部署服务，API Key为空，只需要在OPENAPI_MODEL_NAME中给出服务组名字即可。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAPI_API_KEY` | API Key | `EMPTY` |
| `OPENAPI_BASE_URL` | API 地址 | `http://infer-proxy-yb-test.production.polaris:8000/openapi`（idc环境下访问） |
| `OPENAPI_MODEL_NAME` | 模型名 | `openai/HY3-dev0413` |
| `OPENAPI_TEMP` | 温度 | `0.9` |

hy3模型完整的请求命令为：
```bash
FRONTEND_DATASET_PATH=path/to/dataset.jsonl FRONTEND_RUNTIME_TOKEN={沙盒门户账号token} LLM_PROVIDER=hy3 OPENAPI_BASE_URL="http://infer-proxy-yb-test.production.polaris:8000/openapi" OPENAPI_MODEL_NAME="openai/HY3-dev0413" ENABLE_THINKING=1 \
  uv run nexus run-batch \
    -g workspace/frontend/generator.py \
    -d output.db
```

**opus 模式**（`LLM_PROVIDER=opus`）：
opus模式的请求链路走蒸馏平台，使用aws的claude账号。账号的`app_id`和`app_key`由OPENAI_API_KEY给出:

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | API Key | `{app_id}:{app_key}?provider=aws_third&timeout=3000` |
| `OPENAI_BASE_URL` | API 地址 | `http://trpc-gpt-eval.production.polaris:8080`（idc、devcloud环境下均可） |
| `OPENAPI_MODEL` | 模型名 | `bedrock/anthropic.claude-opus-4-6-v1` |

蒸馏平台aws opus完整的请求命令为 (截止2026/04/29, 当前分支暂时只支持抓快思考)：
```bash
FRONTEND_DATASET_PATH=path/to/dataset.jsonl FRONTEND_RUNTIME_TOKEN={沙盒门户账号token} LLM_PROVIDER=opus OPENAPI_BASE_URL="http://trpc-gpt-eval.production.polaris:8080" OPENAPI_MODEL_NAME="bedrock/anthropic.claude-opus-4-6-v1" ENABLE_THINKING=0 \
  uv run nexus run-batch \
    -g workspace/frontend/generator.py \
    -d output.db
```

### Skills 配置

通过 `FRONTEND_SKILLS_DIR` 将本地 skill 目录同步到沙盒的 `~/.claude/skills/`，Claude Code 会自动按目录名注册为 slash 命令。

**目录名映射规则：**
- 目录 `frontend_codesign` → 命令 `/frontend_codesign`
- 目录 `frontend-codesign` → 命令 `/frontend-codesign`
- 目录 `frontendCodesign` → 命令 `/frontendCodesign`
- 目录名就是命令名，没有额外映射或配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `FRONTEND_SKILLS_DIR` | 逗号分隔的本地 skill 目录路径，每个目录须包含 `SKILL.md` | 空（不安装） |
| `SKILL_NAME` | 在 prompt 前自动添加 `/<SKILL_NAME> ` 前缀触发技能 | 空（不触发） |

`SKILL_NAME` 通常与 `FRONTEND_SKILLS_DIR` 中某个目录名对应。例如：

```bash
# 同步 skill 目录并自动在 prompt 前添加 /frontend_codesign 触发
FRONTEND_SKILLS_DIR="/path/to/frontend_codesign" \
SKILL_NAME="frontend_codesign" \
FRONTEND_DATASET_PATH=... uv run nexus run-batch ...

# 同步多个 skills
FRONTEND_SKILLS_DIR="/path/to/skill_a,/path/to/skill_b" \
SKILL_NAME="skill_a" \
  ...
```

### 搜索配置

通过 `FRONTEND_ENABLE_SEARCH` 开关控制是否启用外部 WebSearch MCP，同时屏蔽 Claude Code 内置的 WebSearch 工具避免冲突。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `FRONTEND_ENABLE_SEARCH` | "1"/"true" 启用 WebSearch MCP | `false` |
| `FRONTEND_SEARCH_MCP_URL` | WebSearch MCP 的 SSE 地址 | `https://mcp-api.tencent-cloud.com/sse/7bb631b3c03b41f2` |

启用后的效果：
1. 在 `~/.claude/mcp.json` 中注册 `WebSearchMCP` server（type: sse）
2. 通过 `--disallowedTools WebSearch` 屏蔽内置搜索工具
3. Agent 只能使用外部 MCP 提供的搜索能力

```bash
FRONTEND_ENABLE_SEARCH=1 FRONTEND_DATASET_PATH=... uv run nexus run-batch ...

# 自定义 MCP 地址
FRONTEND_ENABLE_SEARCH=1 FRONTEND_SEARCH_MCP_URL="https://your-mcp-server/sse/xxx" ...
```

### Claude Code 插件（截止2026/04/29仍为实验特性，未经大规模抓取验证）
Frontend Task链路初步支持nexus中的claude code使用插件，大致实现逻辑为将本地claude code安装好的插件拷贝到nexus的claude code中，并通过pre_start_scripts机制修改nexus的claude code settings启用插件。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CLAUDE_PLUGINS` | 逗号分隔的插件名，离线安装到容器内 | 空（不安装） |

使用前需在本地先安装一次插件：

```bash
claude plugin marketplace add anthropics/claude-plugins-official
claude plugin install frontend-design
```

之后通过环境变量启用：

```bash
CLAUDE_PLUGINS="frontend-design" FRONTEND_DATASET_PATH=... uv run nexus run-batch ...
```

## 输出格式

导出的 JSONL 每行包含：

| 字段 | 说明 |
|------|------|
| `task_output.completed` | 是否完成 |
| `task_output.archive_b64` | 项目 tar.gz 的 base64 编码 |
| `task_output.file_list` | 压缩包内的文件列表 |
| `task_output.error` | 错误信息（正常为空） |
| `task_output.termination_reason` | 终止原因（正常为空） |
| `agent_output.output` | Claude Code 的 stream-json 输出 |
| `processor_results.chat.trajectories` | Agent 对话轨迹 |

### 解压单个项目

```python
import json, base64, tarfile, io

with open('output.jsonl') as f:
    data = json.loads(f.readline())

tar_bytes = base64.b64decode(data['task_output']['archive_b64'])
with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
    tar.extractall('my_project/')
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `task.py` | 任务逻辑：加载 prompt → 启动 agent → 打包工作目录 |
| `generator.py` | Batch 入口：读取数据集、配置 runtime/agent/LLM、生成 BatchItem |
| `pyproject.toml` | 模块元数据，注册 `frontend_task` |
