# Harbor Trial

`harbor_trial` 是 Nexus 对 Harbor 原生任务模型的集成入口。Harbor 负责 Task、Trial、多步骤、Verifier、Reward、Artifact 和 Docker Compose 语义，Nexus 负责提供选定的 Agent 与 Runtime。

该模块与 `harbor_bridge` 相互独立；在 `harbor_trial` 持续演进期间，原有 Bridge 仍可继续使用。

## 支持范围

当前支持：

- Harbor 单步骤和多步骤任务。
- Direct 单容器模式：使用 `[environment].docker_image` 指定的预构建 Linux 镜像。
- 单服务 DinD 模式：没有任务 Compose、需要构建 Dockerfile 时，复用 Harbor 基础 Compose 模板，只运行 inner `main`。
- Compose 多容器模式：在一个启用 DinD 的 Nexus Runtime 内运行 Docker Compose。
- Compose 中的 inner `main`：复用 Harbor 原生预构建镜像选择规则，支持 `force_build` 和 Compose 自定义 `build.context` / `build.dockerfile`。
- Compose main / sidecar：支持 `image`、`build` 及 `image + build`；构建行为由最终合并后的 Compose 定义决定。
- 自动将 Compose 服务名、hostname 和网络 alias 加入 `NO_PROXY`/`no_proxy`，并保留外层 Runtime 与任务原有配置。
- 已接入 Nexus execution-target 协议的 Agent，目前包括 Claude Code 和 Codex。
- Agent 安装、HOME/config、启动前操作、LLM Proxy 和进程启动都可以在 inner `main` 内完成。
- 可配置全局 Agent 用户和分步骤 Agent 用户，但 Direct 和 Compose 均暂不承诺完整的跨用户支持。用户配置不代表 Agent 安装、HOME/config、文件权限及进程启动已实现一致的跨用户适配；Compose 支持也不等于跨用户支持。
- Verifier 共享环境或独立环境，以及 Harbor 原生 Reward。
- 多维 Reward 完整保留；Nexus `score` 读取 `reward` 维度，不存在时使用 `0.0`。
- Harbor `mcp_servers` 转换为 Nexus Agent MCP 配置；Claude Code 支持 stdio、SSE 和 streamable HTTP，Codex 当前仅支持 stdio。
- Harbor `skills_dir` 注入 Claude Code 的当前运行 HOME，Direct 和 Compose inner `main` 均支持。
- Task、Step 和 sidecar Artifact 收集。
- 本地运行使用 `task_dir`；可移植或分布式运行使用 Nexus `task_files`。

Direct 模式不会构建 `environment/Dockerfile`，必须配置 `[environment].docker_image`。没有任务 Compose 且需要构建时，自动选择单服务 DinD；任务无需补写 Compose 文件。单服务 DinD 和 Compose 多容器模式均会在外层 Runtime 的 DinD daemon 中执行 `docker compose build`，由 Compose 处理最终定义中各服务的 `build:`，不会因为同时配置了 `image:` 就跳过构建。构建出的镜像和缓存仅存在于当前 Runtime，不会推送到 Registry，Runtime 释放后也会消失。

单服务 DinD 和 Compose 多容器模式均要求 Runtime Provider 支持 `enable_dind=true`，目前可使用 Gongfeng/gVisor；本地 Docker Provider 不支持嵌套 DinD。

Direct 支持静态 `public`、`no-network` 和 `allowlist`；DinD 对齐 Harbor Daytona Compose 的静态 `public` / `no-network`，不支持白名单或运行中切换策略。后端与生命周期限制见下节。由于 Nexus 暂时无法强制限制磁盘容量，`storage_mb` 会产生警告并被忽略。Agent 原生会话恢复暂不支持。

## 静态网络策略

按最终运行模式选择隔离层级，策略在对应环境创建时生效，运行中不切换。Direct 在 Sandbox 层隔离；单服务 DinD 和 Compose 多容器模式保持外层 Sandbox 的构建网络，在 inner `main` 层隔离。`force_build` 将 Direct 转为单服务 DinD 时，会在创建 Runtime 前重新校验网络能力和外层策略。

| Harbor 网络模式 | Direct | 单服务 DinD / Compose |
| --- | --- | --- |
| `public` | 不下发任务级出网限制 | 外层不下发任务级限制，保留任务 Compose 网络定义 |
| `no-network` | Sandbox 设置 `default_action: deny`，不添加放行规则 | 外层保留构建网络；追加 Harbor 原生断网模板，inner `main` 使用 `network_mode: none` |
| `allowlist` | Sandbox 默认拒绝，逐条放行 `allowed_hosts` | 明确拒绝，不退回外层白名单或静默放行 |

DinD 的断网模板只修改 `main`，不统一修改 sidecar。`main` 的 `network_mode: none` 也会切断其与 sidecar 的网络通信，不是“仅禁止公网”。任务显式 `networks` 等配置与该模板冲突时，由合并后的 `docker compose config` 校验报错，不改写任务拓扑。Dockerfile 构建网络与运行容器网络分离；外层仍受平台网络、代理和镜像源限制。

非 `public` 策略要求 Gongfeng Runtime Provider，该 Provider 默认声明静态网络能力。未指定调度名称或显式指定 `gongfeng` 时，后端均可按网络策略路由到算力，无需为此改为 `suanli`。基于同一协议的 Taiji Runtime Provider 也接受网络参数，但仍需满足其独立的资源与调度配置要求。其他 Runtime Provider 不声明断网/白名单能力，会拒绝隔离任务，不静默降级。

`runtime_provider_name` 选择 Nexus Provider 实现，内层 `runtime_provider` 选择后端调度路径，两者不是同一个字段。保留已有 Provider 配置即可，静态网络策略不要求额外指定调度名称。

Direct 白名单支持域名、前导通配符域名、IPv4 地址及 IPv4 CIDR；不支持 IPv6 地址或 IPv6 CIDR。任务配置使用 Harbor 原生格式，不额外引入 Bridge 字段。

### Shared Verifier 全程断网

以下为 Harbor 任务的相关配置片段：

```toml
[environment]
docker_image = "example/task:v1"
network_mode = "no-network"

[verifier]
environment_mode = "shared"
```

Agent 和 Shared Verifier 都继承环境启动策略。若 `[agent]`、`[verifier]` 或任何 Step 声明不同策略，Harbor 原生校验会在启动 Runtime 前拒绝；仅改变白名单也属于策略变化。不调用运行期网络更新接口。

### Separate Verifier 独立断网

```toml
[environment]
docker_image = "example/agent:v1"
network_mode = "public"

[verifier]
environment_mode = "separate"

[verifier.environment]
docker_image = "example/verifier:v1"
network_mode = "no-network"
```

Agent 和 Separate Verifier 使用各自 Runtime 和启动策略。Separate 的网络模式应放在 `[verifier.environment]`，Step 级覆盖放在 `[steps.verifier.environment]`。仅在 `[verifier]` 设置 `no-network`、但环境启动基线仍为 `public`，原生语义是启动后切换，静态适配会在运行 Agent 前拒绝。不同 Step 新建的 Separate Runtime 可以使用不同基线。

### Direct 静态白名单

```toml
[environment]
docker_image = "example/task:v1"
network_mode = "allowlist"
allowed_hosts = ["packages.example.com", "*.example.org", "192.0.2.1", "198.51.100.0/24"]
```

`runtime_provider_config.network_policy` 或对应 Runtime 配置中的 `network_policy` 若与 Direct 启动策略或 DinD 外层联网要求冲突，会明确报错。适配器不会合并放行规则，也不会静默覆盖冲突配置；应移除这些覆盖，将任务策略配置在 Harbor 任务中。Provider 全局策略也会影响 Separate 环境预检，不应拿 Agent 专用策略作为 Provider 全局默认值。

断网从任务运行环境创建起生效，覆盖 Agent 安装和 Verifier 执行，而不仅是评分脚本。不会自动添加依赖源、镜像仓库、模型服务或其他基础设施白名单。DinD 的镜像拉取与 Dockerfile 构建使用外层网络，不受 inner `main` 的断网策略影响；在 `main` 启动后才下载依赖仍会失败，应将依赖放入镜像。Direct 仍要求预构建 `docker_image`，本功能不增加 Direct Dockerfile 构建能力。

本地单测覆盖分层策略映射、原生模板、继承/预检及 Direct/单服务 DinD/Compose 创建链路，不代替真实 Sandbox 的隔离验收。部署时仍需验证构建联网、main 断网、Runtime 控制链路和 Agent LLM 调用。

## 执行结构

Direct 单容器：

```text
Nexus Runner
  -> Harbor Trial
    -> NexusEnvironment -> task Runtime
    -> NexusAgent -> task Runtime
    -> Harbor Verifier -> task Runtime
```

单服务 DinD：

```text
Nexus Runner
  -> Harbor Trial
    -> NexusEnvironment -> 启用 DinD 的 Nexus Runtime（使用运行配置的外层镜像）
      -> Harbor 基础 Compose 模板 -> 从 environment/Dockerfile 构建 main
      -> NexusAgent / Harbor Verifier -> main
```

外层 Sandbox 沿用 Provider / Runtime 配置中的预构建 `image` / `runtime_image`，需满足现有 DinD 运行条件。任务 Dockerfile 仅用于内层 `main`，不会传给 Provider 构建或覆盖外层镜像。单服务 DinD 沿用 Compose 的 Agent execution-target 要求，需使用已适配的 Agent。

Compose 多容器：

```text
Nexus Runner
  -> Harbor Trial
    -> NexusEnvironment -> 启用 DinD 的 Nexus Runtime
      -> Harbor DinDComposeOps -> main + sidecars
      -> Nexus Agent execution transport -> 在 main 中运行 Agent
      -> Harbor Verifier -> main
      -> Harbor Artifact collection -> main/sidecars
```

多步骤任务的每一步都会创建一次新的 Nexus Agent 调用，但 Direct Runtime 或 Compose project 会持续复用。因此，文件、服务和容器状态会跨步骤保留，Agent 自身的原生对话历史不会恢复。

## Codex 日志与原生轨迹

选择 `agent_name: codex` 时，Harbor Trial 自动为每次调用启用文件日志和原生 session 保存，不需要任务增加开关或修改评分脚本：

- Runtime 执行端实时将 CLI 的 stdout/stderr 合并保存为 `/logs/agent/codex.txt`，保留真实退出码。
- 每次调用使用独立 CODEX_HOME，结束或异常收尾时仅将 `sessions/` 复制到 Agent 日志目录，不复制配置、认证文件或整个 HOME。
- 装配入口为 Codex 选择 `NexusCodexAgent`，由该专用适配注入日志参数；通用 `NexusAgent.run()` 仅执行并暂存当次 output，不提前写入 `AgentContext.metadata`，避免跳过 Harbor 原生后处理。
- Harbor Trial 负责下载日志或同步挂载目录，再调用 `populate_context_post_run`：Codex 专用适配复用 Harbor 原生 session → ATIF 转换和 token/cost 回填，然后执行通用 `nexus_agent_output` 回填。不会用 stdout 或最终答案拼造轨迹，也不使用 Bridge 的 T3 格式。
- Harbor 继续负责将生成的 `trajectory.json` 上传回 Agent 环境，再按原生 `artifacts` 配置收集、交付 Separate Verifier。Shared Verifier 可直接读取 Agent 日志目录。

Direct、单服务 DinD 和多服务 Compose 沿用相同链路；DinD 的外层日志目录与 inner main 使用已有日志挂载共享。多步骤继续不恢复会话，产物随 Harbor 的步骤清理、归档流程隔离。日志目录遵循 Harbor 的 `environment_logs_dir`，默认是 `/logs/agent`。

Separate Verifier 需要任务显式声明要交付的文件，例如：

```toml
artifacts = [
    { source = "/logs/agent/codex.txt", service = "main" },
    { source = "/logs/agent/trajectory.json", service = "main" },
]
```

普通 Nexus Codex 调用默认不变，同样会写 session（`--ephemeral` 已不再使用，否则 `multi_agent_version=v2` 派生子 agent 会失败）；只有设置可选的 `logs_dir` 才启用上述 Runtime 侧文件采集。文件内容不塞入 `CodexOutput`，无需开启 `retain_json_events_stdout`。本入口目前只支持 Linux 上的 `codex`，不扩展 `codex_native_bridge` 或其他 Agent。与原生 Harbor 一样，无 session 或转换失败时不伪造轨迹；执行前失败或外层 Runtime 被强制销毁时不保证有完整产物。

Runtime 端需包含本次更新，开发运行推荐 `worker_venv: true`。本地测试覆盖日志采集、异常收尾、原生 ATIF 转换及文件交付，不代表已跑通真实 Sandbox 或完整任务评分。

## Harbor Cookbook 支持情况

以下 recipe 可以作为当前能力的验收用例：

| Recipe | 状态 | 说明 |
| --- | --- | --- |
| `simple-task` | 支持 | 有预构建镜像可走 Direct；仅有 Dockerfile 时走单服务 DinD |
| `multi-step` | 支持 | 共享环境、逐步验证、提前停止和逐步 Artifact |
| `multi-container` | 支持 | Compose inner `main` 和 sidecar 均可使用镜像或 Dockerfile 构建 |
| `mcp-tools` | 支持 Claude Code | Harbor `streamable-http` 会转换为 Nexus MCP 配置；Codex 暂不支持 HTTP MCP |
| `computer-use-ubuntu` | 支持 Claude Code | 桌面 sidecar 通过 streamable HTTP MCP 暴露工具，需要图像能力模型做真实 E2E |
| `computer-use-windows` | 条件支持 Claude Code | Windows 沙箱由 sidecar 通过 Daytona 创建，需要对应凭据和外网访问 |
| `simulated-user` | 条件支持 Claude Code | MCP sidecar 自行访问 Anthropic，需要 sidecar 凭据和网络；该请求不走 Nexus LLM Proxy |
| `skills` | 支持 Claude Code | `skills_dir` 内容会复制到当前 Agent HOME 的 `.claude/skills` |
| `multi-reward` | 部分支持 | 完整 `rewards` 会保留；若没有 `reward` 维度，Nexus 标量 `score` 为 `0.0` |
| `dns-blacklisting` | 支持 Compose | 依赖任务镜像 ENTRYPOINT；它模拟域名访问结果，不是 Runtime 网络隔离策略 |

## 前置准备

以下教程假设：

- Nexus 仓库位于 `/data/workspace/nexus`。
- Harbor Cookbook 位于 `/data/workspace/harbor-cookbook`。
- 使用 `uv` 管理 Nexus 环境。
- Gongfeng Runtime 地址可访问，并已准备 `GONGFENG_RUNTIME_TOKEN`。
- 已准备兼容 OpenAI API 的模型服务凭据，例如 `AIHUB_API_KEY`。

先安装开发依赖并确认扩展依赖：

```bash
uv sync --extra dev
uv run nexus modules
```

设置凭据，避免将密钥写进 YAML 或提交到仓库：

```bash
export GONGFENG_RUNTIME_TOKEN='<your-gongfeng-token>'
export AIHUB_API_KEY='<your-model-api-key>'
export SWE_RUNTIME_BASE_URL='http://hyrl-sandbox.prod.woa.com:8052'
```

下面的命令均在 `/data/workspace/nexus` 中执行。

## 通用运行配置

创建 `workspace/harbor-trial.yaml`：

```yaml
imports:
  - nexus.extensions.harbor_trial.task
  - nexus.extensions.agents.claude_code
  - nexus.extensions.openai_clients.openai
  - nexus.runtime.providers.gongfeng

task_name: harbor_trial_task

task_config:
  task_dir: /absolute/path/to/harbor-task
  instance_id: example-task
  agent_timeout: 900
  verifier_timeout: 600

runtime_provider_name: gongfeng
runtime_provider_config:
  api_base_url: ${oc.env:SWE_RUNTIME_BASE_URL,"http://hyrl-sandbox.prod.woa.com:8052"}
  token__env: GONGFENG_RUNTIME_TOKEN
  runtime_image: current
  request_timeout: 24000

runtime_configs:
  main:
    image: ${oc.env:NEXUS_DIND_OUTER_IMAGE,"mirrors.tencent.com/hunyuan_yanguan/ubuntu:24.04"}
    worker_venv: true

agent_name: claude_code
agent_config: {}

openai_clients:
  main:
    name: openai
    config:
      client_args:
        api_key__env: AIHUB_API_KEY
        base_url: http://api.aihub.woa.com/standard/v1
        timeout: 900
        max_retries: 3
      request_args:
        model: hy4-preview
        max_tokens: 32768
      enforced_request_args:
        max_tokens: 32768

runner:
  timeout: 3600
```

`runtime_configs.main.image` 是承载 Nexus Runtime、Docker daemon 和 Compose 的外层基础镜像，不是 Harbor 任务的 inner `main` 镜像。保留 `runtime_provider_config.runtime_image: current`，即可复用当前 Nexus Runtime 工具和 DinD 启动逻辑，不需要额外制作 Harbor 专用 DinD 镜像。

Harbor 任务目录中只要存在 `environment/docker-compose.yaml`，适配器就会自动进入 Compose/DinD 模式，不需要再增加 task-level 开关。

## Multi-step 手把手教程

示例任务路径：

```text
/data/workspace/harbor-cookbook/harbor_cookbook/recipes/multi-step
```

### 1. 理解任务

该任务在同一个 `main` 环境中依次执行三个步骤：

1. `scaffold`：创建可执行的 `/app/greet.sh`，输出 `hello`。
2. `implement`：读取 `/app/config.txt`，把脚本改为输出 `Hello, Harbor!`。
3. `document`：创建 `/app/README.md`，说明脚本用法。

关键目录如下：

```text
multi-step/
├── task.toml
├── environment/Dockerfile
├── tests/helpers.sh
└── steps/
    ├── scaffold/
    │   ├── instruction.md
    │   └── tests/test.sh
    ├── implement/
    │   ├── instruction.md
    │   ├── workdir/config.txt
    │   ├── workdir/setup.sh
    │   └── tests/test.sh
    └── document/
        ├── instruction.md
        └── tests/test.sh
```

`task.toml` 中配置了 `[environment].docker_image`，且默认 `force_build=false`，因此该示例走 Direct 模式并直接使用预构建任务镜像；即使目录中存在 `environment/Dockerfile`，Nexus 也不会在 Direct 模式现场构建它。

每一步拥有独立的 instruction、Agent timeout 和 Verifier timeout，但复用同一个 Runtime。`scaffold` 生成的 `/app/greet.sh` 会保留给 `implement` 和 `document` 使用。

### 2. 创建运行配置

复制通用配置为 `workspace/harbor-multi-step.yaml`，将 `task_config` 修改为：

```yaml
task_config:
  task_dir: /data/workspace/harbor-cookbook/harbor_cookbook/recipes/multi-step
  instance_id: multi-step-gongfeng-hy4
  agent_timeout: 900
  verifier_timeout: 600
```

其他部分保持不变。

### 3. 启动任务

```bash
uv run nexus run single workspace/harbor-multi-step.yaml \
  -o workspace/harbor-multi-step-result.json \
  -v
```

运行时可以重点观察：

- `scaffold -> implement -> document` 按顺序执行。
- 每一步产生一次新的 Agent 调用。
- 三个步骤使用相同的 Runtime。
- `implement` 开始前，`workdir/config.txt` 被上传到 `/app/config.txt`，并执行 `workdir/setup.sh`。
- `implement` 的 healthcheck 会确认 `/app/greet.sh` 已由上一步创建。
- 每一步结束后执行对应 Verifier，并按 `min_reward` 决定是否继续。

### 4. 检查结果

```bash
uv run python -c "import json; d=json.load(open('workspace/harbor-multi-step-result.json')); t=d['task_output']; steps=t['trial_result']['step_results'] or []; print({'success': d['success'], 'score': t['score'], 'reward': t['reward'], 'steps': [{'name': s['step_name'], 'rewards': (s.get('verifier_result') or {}).get('rewards'), 'exception': s.get('exception_info')} for s in steps]})"
```

默认 `multi_step_reward_strategy = "mean"`，三个步骤都通过时最终 Reward 应为 `1.0`。如果 `scaffold` 的 Reward 低于 `1.0`，后续步骤会提前停止；`implement` 的继续阈值是 `0.5`。

## Multi-container（DinD）手把手教程

示例任务路径：

```text
/data/workspace/harbor-cookbook/harbor_cookbook/recipes/multi-container
```

### 1. 理解任务

该任务包含两个 inner 容器：

- `main`：Agent 和 Verifier 所在容器。
- `api-server`：提供 `http://api-server:5000` REST API 的 sidecar。

两个服务位于同一个 Compose 网络中，所以 `main` 可直接通过服务名 `api-server` 访问 sidecar，不需要向外映射端口。

### 2. 配置 main 本地构建

确保 `task.toml` 的 `[environment]` 没有配置 `docker_image`：

```toml
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
```

同时提供 `environment/Dockerfile`：

```dockerfile
FROM mirrors.tencent.com/hunyuan_yanguan/ubuntu:24.04

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
```

由于没有 `docker_image`，Harbor Trial 会在外层 Runtime 的 DinD daemon 中从该 Dockerfile 构建 inner `main`。

### 3. 配置 api-server 本地构建

在 `environment/docker-compose.yaml` 中为 sidecar 配置 `build`：

```yaml
services:
  main:
    depends_on:
      api-server:
        condition: service_healthy

  api-server:
    build:
      context: ./api-server
    expose:
      - "5000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import socket; s=socket.create_connection(('localhost',5000),timeout=2); s.close()"]
      interval: 2s
      timeout: 5s
      retries: 15
      start_period: 5s
```

相对构建上下文 `./api-server` 以任务的 `environment/` 目录为基准，对应：

```text
environment/api-server/Dockerfile
environment/api-server/server.py
```

api-server 的 Dockerfile 示例：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install flask

COPY server.py .

EXPOSE 5000

CMD ["python", "server.py"]
```

`python:3.12-slim` 等价于 `docker.io/library/python:3.12-slim`，默认引用 Docker Hub 官方镜像。若执行环境容易遇到 Docker Hub 限流，建议替换为组织内已同步并验证可用的 Python 基础镜像。

### 4. 创建运行配置

复制通用配置为 `workspace/harbor-multi-container.yaml`，将 `task_config` 修改为：

```yaml
task_config:
  task_dir: /data/workspace/harbor-cookbook/harbor_cookbook/recipes/multi-container
  instance_id: multi-container-gongfeng-hy4
  agent_timeout: 900
  verifier_timeout: 600
```

必须保留以下外层 Runtime 配置：

```yaml
runtime_configs:
  main:
    image: ${oc.env:NEXUS_DIND_OUTER_IMAGE,"mirrors.tencent.com/hunyuan_yanguan/ubuntu:24.04"}
    worker_venv: true
```

这里的 `image` 只负责启动外层 Sandbox。inner `main` 和 `api-server` 仍按照各自 Dockerfile 在 DinD 内构建，两者不要混淆。

### 5. 启动任务

```bash
uv run nexus run single workspace/harbor-multi-container.yaml \
  -o workspace/harbor-multi-container-result.json \
  -v
```

日志中应出现类似命令：

```text
docker compose ... build
docker compose ... up -d
```

第一行由 Compose 根据合并后的定义构建服务；第二行启动服务，不再禁止启动阶段的隐式构建。

随后应看到 Agent 在 `main` 内运行，并访问：

```text
http://api-server:5000
```

### 6. 检查结果

```bash
uv run python -c "import json; d=json.load(open('workspace/harbor-multi-container-result.json')); t=d['task_output']; tr=t['trial_result']; print({'success': d['success'], 'score': t['score'], 'reward': t['reward'], 'agent_success': tr['agent_result']['metadata']['nexus_agent_output']['success'], 'exception_info': tr['exception_info']})"
```

成功结果应类似：

```text
{'success': True, 'score': 1.0, 'reward': 1.0, 'agent_success': True, 'exception_info': None}
```

## 镜像选择规则

无任务 Compose 时，优先按 Harbor 原生规则选择预构建镜像；需要构建则走单服务 DinD。有任务 Compose 或附加 Compose 时，继续走 Compose DinD：

| 场景 | 配置 | 行为 |
| --- | --- | --- |
| 无 Compose | 配置 `docker_image`，`force_build=false` | Direct 使用预构建镜像，即使存在 Dockerfile |
| 无 Compose | 配置 `docker_image`，`force_build=true`，无 Dockerfile | 仍走 Direct，使用预构建镜像 |
| 无 Compose | 配置 `docker_image`，`force_build=true`，有 Dockerfile | 单服务 DinD，从 Dockerfile 构建 main |
| 无 Compose | 无 `docker_image`，有 Dockerfile | 单服务 DinD，从 Dockerfile 构建 main，与 `force_build` 无关 |
| 无 Compose | 无 `docker_image`，也无 Dockerfile | 创建 Runtime 前报错，缺少环境定义 |
| Compose inner main | 配置 `docker_image`，`force_build=false` | 选择 Harbor 预构建模板 |
| Compose inner main | 配置 `docker_image`，`force_build=true`，存在 `environment/Dockerfile` | 选择 Harbor 构建模板 |
| Compose inner main | 配置 `docker_image`，`force_build=true`，不存在 `environment/Dockerfile` | 仍选择预构建模板 |
| Compose inner main | 未配置 `docker_image` | 选择构建模板，默认 context 为 `environment`；任务 Compose 可覆盖构建路径 |
| Compose 任一服务 | 最终定义只有 `image`，没有 `build` | 直接使用或拉取镜像，不构建 |
| Compose 任一服务 | 最终定义有 `build`，可同时有 `image` | 由 `docker compose build` 构建；有 `image` 时为构建结果打该标签 |

模板选择复用 Harbor 的 `should_use_prebuilt_docker_image`。任务 Compose 和附加 Compose 文件随后按顺序合并，最终定义由 Compose 解析校验；不额外要求根目录存在 Dockerfile。例如任务可以仅提供以下 Compose 和 `environment/app/Dockerfile.runtime`：

```yaml
services:
  main:
    image: task-main:local
    build:
      context: ./app
      dockerfile: Dockerfile.runtime
```

`build.context` 相对于 `environment` 解析，`dockerfile` 相对于构建 context 解析。自定义构建路径无需再配置 `[environment].docker_image`。

在现有 Nexus 运行配置的 `task_config` 中添加 `force_build`，即可启用原生构建选择：

```yaml
task_name: harbor_trial_task
task_config:
  force_build: true
```

`force_build` 默认为 `false`，不等于 `--no-cache`，也不是镜像拉取失败后的自动回退。无任务 Compose 时，若同时存在 Dockerfile 且设置 `force_build=true`，会在创建 Runtime 前选择单服务 DinD；没有 Dockerfile 时仍使用 Direct 预构建镜像。外层镜像沿用现有运行配置，不新增任务级镜像配置。

## 常见问题

### DinD 为什么不能使用本地 Docker Provider

单服务 DinD 和 Compose 多容器模式都需要在 Nexus Runtime 内启动 Docker daemon，也就是 DinD。本地 Docker Provider 会拒绝这种嵌套模式，请使用支持 `enable_dind=true` 的 Gongfeng/gVisor Runtime。

### 为什么配置了 Dockerfile 却没有构建

先检查任务模式和镜像配置：

- 无任务 Compose 且配置了 `docker_image` 时默认走 Direct；有 Dockerfile 并设置 `force_build=true` 才转为单服务 DinD 构建。
- 无任务 Compose、无 `docker_image`、只有 Dockerfile 时，直接走单服务 DinD。
- Compose inner `main` 的默认模板由 `docker_image`、`force_build` 和根 Dockerfile 是否存在共同决定。
- 检查合并后的服务定义是否有 `build:`；main 和 sidecar 同时配置 `image:` 与 `build:` 也会构建。

### 如何确认两个服务确实都在本地构建

使用 `-v` 运行并搜索：

```text
docker compose ... build
```

构建命令不指定服务列表，由 Compose 根据最终定义选择构建服务。检查构建输出中对应服务的构建步骤；只有 `image:`、没有 `build:` 的服务不会构建。

### 为什么基础镜像拉取失败或出现 429

`ubuntu:24.04`、`python:3.12-slim` 等短名称默认来自 Docker Hub。首次现场构建需要拉取基础镜像，可能受网络、鉴权或 Docker Hub 限流影响。生产或稳定测试环境建议显式使用组织内 Registry 中已同步的基础镜像。

### 为什么 `storage_mb` 没有生效

当前 Nexus Runtime 无法强制实施 Harbor 的 `storage_mb` 请求，因此会记录警告并继续执行，不会因此中断任务。


## Agent 执行用户

`claude_code` 和 `codex` 支持 Harbor 的 `[agent].user`，步骤覆盖由 Harbor 解析后传入。
不增加安装用户配置，安装、配置和 CLI 使用同一个有效用户：

```toml
[agent]
user = "alice"
```

- Direct 模式的 Worker 必须以 root 启动；镜像应预先创建 alice，并提供有效、可写的 HOME。
  指定用户名或 UID 均可。普通用户 Worker 不会尝试自行提权，而是明确报错。
- Worker、Proxy、RPC 和通信桥保持管理身份；系统级准备可使用公共执行层的 `exec_as_root()`。
- CC/Codex 的版本检查、npm/URL 安装、配置、MCP、Skills、pre-start 和 CLI 在目标用户下执行。
  Direct 的安装目录保持为该账户 HOME 下的 `.local/share/nexus/agents/<agent>`。
- Compose 模式在内层 `main` 中完成安装与执行，使用 `docker compose exec --user`，不在外层安装后仅将 CLI 放入内层。
  安装目录使用 Nexus 可写传输挂载下的 `agent-<uid>/install/<agent>`；每次调用的 HOME 使用同一用户目录下独立的 `home-*`。
  默认挂载点为 `/opt/nexus-agent`，因此无需账户登记的 HOME 存在或可写，可保留任务的只读根文件系统。
  安装缓存按 UID 隔离，HOME 在安装、配置与运行期间保持一致，随当前 Compose Runtime 的传输目录一起回收。
- 用户目录优先由用户创建；Host 上传的 strict MCP/hooks 文件以私有文件交付到用户目录，
  不靠 `chmod 777` 或将配置改为全员可读来绕过权限。任务工作目录本身须允许该用户访问。
- Node/npm 与用于文件操作的 Python 是基础设施依赖，由镜像或 Nexus Runtime 提供。
  本功能不在 Agent 安装阶段以 root 运行 npm，也不自动安装系统软件包。
- Codex 的 `CODEX_HOME` 继续按调用隔离；原生 sessions 从实际目标环境回收，配置和凭据不导出。
- 没有指定用户的 Direct 模式保留原行为；Compose 没有指定用户时使用服务默认账户。
  未声明完整用户执行能力的其他 Agent 会拒绝带用户的任务，不视为部分支持。

修改 Python 后优先启用 `runtime_configs.<name>.worker_venv: true`，无需仅为此重建 Runtime 基础镜像。
此身份分工不是针对恶意 Agent 的安全沙箱；仍需使用 Runtime 的隔离机制保护管理服务。

离线验证：

```bash
uv run pytest tests/unit/extensions/agents/test_execution_identity.py -q
uv run pytest tests/integration/runtime/test_installed_agent_identity.py -q
```

集成测试要求 Linux root、Docker Compose 和本地已有 `python:3.12-slim-bookworm` 镜像。
仅在一次性容器内创建 alice，不修改宿主账户，不拉取镜像，不请求 LLM；测试后清理容器。
