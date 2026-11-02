# Installed Agents

`claude_code`、`gemini_cli`、`codex`、`openclaw`、`handy`、`pi`、`workbuddy` 都属于 runtime 内动态安装的 agent。它们共享同一套 Nexus runtime 隔离能力，但各自保留了上游 CLI 的交互方式和专有参数。

从 `0.13.0` 开始，仓库内这些 installed agent 做了两件统一化处理：

- 公共输入字段统一为 `user_prompt`、`working_dir`、`timeout`，公共输出字段统一为 `success`、`output`、`stderr`、`exit_code`、`execution_time`
- 统一 runtime 默认 npm registry 为 `http://mirrors.tencent.com/npm/`

建议：

- `codex`、`gemini_cli`、`openclaw` 虽然现在默认就会走腾讯 npm mirror，但示例里仍显式写出 `npm_registry: "http://mirrors.tencent.com/npm/"`，方便排查网络问题

## 实现详情

| agent_name | 上游程序 | 运行时代理 | 常用额外参数 |
| --- | --- | --- | --- |
| `claude_code` | Claude Code 二进制 | `AnthropicProxy` | `version`、`binary_url` |
| `gemini_cli` | `@google/gemini-cli` | `GeminiProxy` | `version`、`model`、`approval_mode`、`output_format`、`npm_registry` |
| `codex` | `@openai/codex` 或原生二进制 | `ResponsesProxy` | `version`、`model`、`npm_registry`、`binary_url`、`mcp_servers` |
| `openclaw` | `openclaw` | `OpenAIProxy` | `version`、`model`、`npm_registry` |
| `handy` | `@tencent/handy-agent` | `AnthropicProxy` | `version`、`model`、`max_output_tokens`、`npm_registry`、`extra_env`、`collect_artifacts` |
| `pi` | `@earendil-works/pi-coding-agent` | `OpenAIProxy` | `version`、`tools`、`output_format`、`load_context_files`、`trust_project_files`、`npm_registry` |
| `workbuddy` | `@tencent/nexus-wb-cli` | `OpenAIProxy` | `version`、`npm_registry`、`preset`、`mode`、`effort` |

## 共享模板

```yaml
imports:
  - nexus.extensions

task_name: "echo"
task_config:
  message: "Hello, Nexus!"
  run_oracle: false

runtime_provider_name: "docker"
runtime_provider_config:
  runtime_image: "current"

agent_name: "<agent_name>"
agent_config:
  working_dir: "/workspace"
  timeout: 3600

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: ${oc.env:NEXUS_AGENT_API_KEY,"YOUR_API_KEY_HERE"}
        base_url: ${oc.env:NEXUS_AGENT_BASE_URL,"http://trpc-gpt-eval.production.polaris:8080"}
      request_args:
        model: ${oc.env:NEXUS_AGENT_MODEL,"anthropic.claude-sonnet-4-6"}

processors:
  chat:
    name: "chat_tracker"
```

如果你走 Gongfeng provider，把 `runtime_provider_name` 改成 `gongfeng`，并把 `runtime_image` 改成 `mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:0.13.0` 或你自己分支构建出来的 tag。

## 可直接运行的示例

下面这些配置都已经对齐到统一 runtime：

- `tests/e2e/config/claude_code_local_hello.yaml`
- `tests/e2e/config/gemini_cli_local_hello.yaml`
- `tests/e2e/config/codex_local_hello.yaml`
- `tests/e2e/config/openclaw_local_hello.yaml`
- `tests/e2e/config/handy_local_hello_example.yaml`
- `tests/e2e/config/pi_local_hello_example.yaml`
- `tests/e2e/config/pi_gongfeng_hello_example.yaml`（gongfeng provider，已实测跑通）

启动方式一致：

```bash
uv run nexus run-single tests/e2e/config/<config_name>.yaml
```

### Claude Code

完整示例：`tests/e2e/config/claude_code_local_hello.yaml`

```yaml
agent_name: "claude_code"
agent_config:
  working_dir: "/workspace"
  timeout: 3600
  version: ${oc.env:CLAUDE_CODE_VERSION,"latest"}

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: ${oc.env:CLAUDE_CODE_API_KEY,"YOUR_API_KEY_HERE"}
        base_url: ${oc.env:CLAUDE_CODE_BASE_URL,"http://trpc-gpt-eval.production.polaris:8080"}
      request_args:
        model: ${oc.env:CLAUDE_CODE_MODEL,"anthropic.claude-sonnet-4-6"}
```

说明：

- `version` 控制下载的 Claude Code 版本
- 如果你有内部分发地址，也可以改用 `binary_url` 和 `binary_checksum`

### Gemini CLI

完整示例：`tests/e2e/config/gemini_cli_local_hello.yaml`

```yaml
agent_name: "gemini_cli"
agent_config:
  working_dir: "/workspace"
  timeout: 3600
  version: "0.33.1"
  model: ${oc.env:GEMINI_CLI_MODEL,"anthropic.claude-sonnet-4-6"}
  npm_registry: ${oc.env:GEMINI_CLI_NPM_REGISTRY,"http://mirrors.tencent.com/npm/"}
  approval_mode: "yolo"
  output_format: "json"

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: ${oc.env:GEMINI_CLI_API_KEY,"YOUR_API_KEY_HERE"}
        base_url: ${oc.env:GEMINI_CLI_BASE_URL,"http://trpc-gpt-eval.production.polaris:8080"}
      request_args:
        model: ${oc.env:GEMINI_CLI_MODEL,"anthropic.claude-sonnet-4-6"}
```

说明：

- `Gemini CLI` 会把 `model` 直接透传到代理层，最稳妥的方式是让 `agent_config.model` 和 `openai_clients.main.config.request_args.model` 保持一致
- `approval_mode: "yolo"` 适合本地验证；如果你想要更保守的执行流程，可以改成 `plan`

### Codex

完整示例：`tests/e2e/config/codex_local_hello.yaml`

```yaml
agent_name: "codex"
agent_config:
  working_dir: "/workspace"
  timeout: 3600
  version: "0.114.0"
  model: ${oc.env:CODEX_MODEL,"gpt-5"}
  npm_registry: ${oc.env:CODEX_NPM_REGISTRY,"http://mirrors.tencent.com/npm/"}

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: ${oc.env:CODEX_E2E_API_KEY,"YOUR_API_KEY_HERE"}
        base_url: ${oc.env:CODEX_E2E_BASE_URL,"http://trpc-gpt-eval.production.polaris:8080/openai/v1"}
      request_args:
        model: ${oc.env:CODEX_MODEL,"gpt-5"}
```

说明：

- `codex` 通过 `ResponsesProxy` 把 Responses API 请求转回 host 侧 `openai_client`
- 默认版本保持为 `0.114.0`；也可以显式设置 `0.146.x`、`0.147.x`、`0.154.0` 等版本，legacy proxy 会按请求 payload 做兼容转换
- npm 安装路径只复用与目标版本一致的缓存或预装 Codex，不匹配时安装目标版本；显式设置 `latest` 时仍执行 npm 安装，不直接复用预装版本。这一规则同样适用于 Harbor Trial
- 如果你切换模型，建议同时改 `agent_config.model` 和 `request_args.model`
- 隔离的确定性验证可设置 `codex_home_name: training-parity`，在 `$HOME/.nexus-codex/` 下使用稳定目录名；默认仍使用每次运行独立的 UUID。 同一 runtime 重复使用目录名会覆盖运行文件
- `normalize_exec_command_metadata: true` 仅归一化 `exec_command` 输出的 chunk ID 和 wall time，默认关闭。 V1/V2 对比时，在两侧 host 都设置 `NEXUS_DETERMINISTIC_TOOL_CALL_IDS=1`，可使用每条轨迹独立递增的工具调用 ID
- 成功的 native compaction 会将同一 Codex 线程的下一次普通请求标记为 `compaction_resume`，失败的 summary 不会触发该标记

#### Windows、COS 二进制与 MCP

原生 `agent_name: codex` 可连接 Windows Nexus Runtime Server，仍使用 `ResponsesProxy`。
保留已有的 Windows provider 配置，将 agent 部分改为：

```yaml
agent_name: codex
agent_config:
  working_dir: 'C:\Users\Administrator\nexus-workspace\codex'
  version: '0.146.1'
  binary_url: ${oc.env:CODEX_BINARY_URL}
  binary_checksum: ${oc.env:CODEX_BINARY_SHA256}
  mcp_servers:
    - name: everything
      command: cmd.exe
      args: [/d, /c, npx, -y, '@modelcontextprotocol/server-everything']
      env:
        NPM_CONFIG_REGISTRY: https://mirrors.tencent.com/npm/
      startup_timeout_sec: 300
      tool_timeout_sec: 60
      required: true
      enabled_tools: [echo]
  user_prompt: '调用 everything MCP 的 echo 工具返回「Windows MCP 验证成功」，并报告实际结果。'
```

- `binary_url` 接受原始可执行文件的 HTTP(S) 地址，包括 COS 签名 URL；Windows 应指向对应架构的 `codex.exe`，不接受压缩包。`binary_checksum` 是可选的 SHA256；校验失败保留旧文件。将签名 URL 放入本地 secret.env，不提交到仓库。
- URL 安装直接运行二进制，不依赖 Node/npm。`version` 仍需与二进制一致，供代理选择协议兼容行为；默认版本不变。
- 不设置 `binary_url` 时继续通过 npm 安装，Windows 上需要 Node.js/npm。上述 MCP 的 npx 启动方式也需要 Node.js/npm。
- `mcp_servers` 沿用 native bridge 的 stdio 配置，支持 `name`、`command`、`args`、`env`、`cwd`、超时、`enabled`/`required` 和工具过滤。MCP 程序在 runtime 内运行；Windows npm shim 应使用 `cmd.exe /c npx` 启动。
- `extra_env` 作用于 Codex 进程；MCP 所需的自定义环境变量请显式填写在对应 server 的 `env` 中。`extra_config_toml` 仍可用于其他 Codex 原生配置。
- 每次运行使用独立的 `CODEX_HOME`，位于 Windows 用户目录下的 `.nexus-codex`；CLI 安装和会话文件保留以便复用、排查，runtime 生命周期仍由 provider 管理。

### OpenClaw

完整示例：`tests/e2e/config/openclaw_local_hello.yaml`

```yaml
agent_name: "openclaw"
agent_config:
  working_dir: "/workspace"
  timeout: 3600
  onboard_timeout: 900
  version: ${oc.env:OPENCLAW_VERSION,"2026.3.13"}
  model: ${oc.env:OPENCLAW_MODEL,"anthropic.claude-sonnet-4-6"}
  npm_registry: ${oc.env:OPENCLAW_NPM_REGISTRY,"http://mirrors.tencent.com/npm/"}

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: ${oc.env:OPENCLAW_LITELLM_API_KEY,"YOUR_API_KEY_HERE"}
        base_url: ${oc.env:OPENCLAW_LITELLM_BASE_URL,"http://trpc-gpt-eval.production.polaris:8080"}
      request_args:
        model: ${oc.env:OPENCLAW_MODEL,"anthropic.claude-sonnet-4-6"}
```

说明：

- `openclaw` 默认会把 `agent_config.model` 当成代理层暴露的模型 ID
- 如果你省略 `agent_config.model`，它会回退到 `openai_clients.main` 的默认模型
- `onboard_timeout` 控制 `openclaw onboard` 初始化命令的超时，默认 `900` 秒，且不会超过 `timeout`
- 长任务建议把 `runtime_provider_config.request_timeout` 设置为大于 `timeout + onboard_timeout + 安装缓冲` 的值，避免外层 `invoke` 先超时

### Handy

完整示例：`tests/e2e/config/handy_local_hello_example.yaml`

```yaml
agent_name: "handy"
agent_config:
  working_dir: "/workspace"
  timeout: 3600
  version: ${oc.env:HANDY_VERSION,"0.1.14"}
  npm_registry: ${oc.env:HANDY_NPM_REGISTRY,"http://mirrors.tencent.com/npm/"}
  model: ${oc.env:HANDY_MODEL,"claude-opus-5"}
  max_output_tokens: 64000
  # 可选：把 Handy 的文件交付物取回宿主
  collect_artifacts: true
  artifact_output_dir: "outputs/handy-artifacts"
```

说明：

- `handy` 使用包内置的 app composition、主 Agent、后台 worker 和插件，不传 `--bare` 或额外的 `--app-config`
- Nexus 会把 Handy 的模型请求路由到 runtime 内的 `AnthropicProxy`，调用方不能通过 `extra_env` 覆盖代理地址或认证变量
- Web、媒体配置可以通过 `extra_env` 注入；不要把密钥直接写入配置文件
- `model` / `max_output_tokens` 只决定 **Handy 自己看到的**模型身份与输出预算（它据此推断上下文预算等行为）。真正路由到哪个后端模型由 `openai_clients.main` 决定，CLI 请求里的模型名会被丢弃。这两项对应的环境变量（`ANTHROPIC_MODEL`、`ANTHROPIC_DEFAULT_OPUS_MODEL`、`CLAUDE_CODE_MAX_OUTPUT_TOKENS`）属于受保护变量，通过 `extra_env` 传入会被忽略

#### 文件交付物（artifact）

Handy 的 `submit_file_artifact` 工具用于把它产出的文件（报告、图表、PDF、压缩包等）交付给用户。默认走对象存储上传；置 `collect_artifacts: true` 后，Nexus 改让它落在沙箱本地磁盘，并在任务结束前把文件取回宿主。

- `collect_artifacts`：默认 `false`，即完全保持原行为
- `artifact_dir`：沙箱内落盘目录，默认 `/tmp/nexus/handy-artifacts`。**必须位于每次运行的临时 session 目录之外** —— 那个目录会在运行结束时被删除，Handy 自己的默认值 `~/.hyness/artifacts` 正好落在里面（`HOME` 被固定到该处），产物会在取回前消失
- `artifact_output_dir`：宿主侧写入目录，按 `<artifact_id>/<filename>` 分层存放（Handy 只保证文件名安全，不保证唯一）。不设置时只回元数据、不取文件
- `HandyOutput.artifacts` 始终携带每个交付物的 `filename` / `size` / `sha256` / `title` / `sandbox_path` / `local_path`；取回的字节会与 Handy 记录的 `sha256` 比对，不一致则不落盘
- 单个文件上限约 64 MB：`download_file` 会把整个文件以 base64 放进一次 RPC 响应，而 Socket.IO 消息上限为 100 MB。超限的文件会留在沙箱并在对应 artifact 的 `fetch_error` 中说明，不会导致整次运行失败。需要交付更大的文件时，改用 Handy 原生的对象存储上传（`HANDY_ARTIFACT_BASE_URL`），那条链路的字节不经过 Nexus
- 取回失败不会丢弃 agent 已完成的输出：错误记录在 `fetch_error` 与日志中，`success` / `output` 仍按实际执行结果返回

### pi

完整示例：`tests/e2e/config/pi_local_hello_example.yaml`

`pi` 是 [earendil-works/pi](http://github.com/earendil-works/pi) 的 coding agent CLI（npm 包 `@earendil-works/pi-coding-agent`），以 headless 单轮模式（`pi -p`）运行。

#### 跑分档（默认）

```yaml
agent_name: "pi"
agent_config:
  working_dir: "/workspace"
  timeout: 3600
  version: ${oc.env:PI_VERSION,"0.84.1"}
  # 以下均为默认值，写出来只是为了显式说明
  load_context_files: false
  trust_project_files: false
  output_format: "json"
```

#### 日常开发档

```yaml
agent_name: "pi"
agent_config:
  working_dir: "/workspace"
  timeout: 3600
  load_context_files: true   # 读取仓库的 AGENTS.md / CLAUDE.md
  trust_project_files: true  # 信任 .pi/ 下的 settings、extensions、skills
```

说明：

- **模型路由靠 `models.json`，不是环境变量**。pi 没有 `ANTHROPIC_BASE_URL` 那样的全局 base URL 覆盖，所以 Nexus 在每次运行的私有配置目录里写一份 `models.json`，注册一个指向 runtime 内 `OpenAIProxy` 的 provider。`model_id` 只是 pi 看到的模型名，真正路由到哪个后端由 `openai_clients.main` 决定
- **`compat` 配置由 Nexus 固定写入，调用方无需也无法配置**。pi 的 `detectCompat` 按 provider 名和 baseUrl 推断兼容性，`127.0.0.1` 这种未知 host 会被当成原生 OpenAI，于是发送 `developer` role、`max_completion_tokens`、`store`、`reasoning_effort`。而 `OpenAIProxy` 会把请求体原样转给宿主侧客户端（`_prepare_request_body` 默认不改字段），不会替你纠正。因此 `build_models_config` 固定写入 `supportsDeveloperRole: false`、`maxTokensField: max_tokens`、`supportsStore: false`、`supportsReasoningEffort: false`；这几项经 pi 0.84.1 实际请求体验证。若将来接的后端模型确实支持这些字段，需要改 `build_models_config` 而不是配置文件
- **`load_context_files` 与 `trust_project_files` 是两个独立开关**。在 pi 里 `--no-approve` 并不会阻止 `AGENTS.md` 的加载，`--no-context-files` 也不会阻止项目 settings 生效，两者需要分别控制
- **默认不读 context 文件的原因**：pi 的 `loadProjectContextFiles` 会从 `working_dir` 一路向上遍历到文件系统根目录，逐层查找 `AGENTS.override.md`、`AGENTS.md`、`AGENTS.MD`、`CLAUDE.md`、`CLAUDE.MD`。沙箱里 `/workspace` 上层若存在任何同名文件都会被吸进 system prompt，使结果取决于挂载布局而非任务本身。跑评测时应保持关闭；真实开发任务里这些文件是正当上下文，建议打开
- **配置与会话完全隔离**：`PI_CODING_AGENT_DIR` 和 `PI_CODING_AGENT_SESSION_DIR` 被重定向到每次运行的临时目录，真实的 `~/.pi` 不会被创建或读取。这两个变量以及 `HOME`、`PATH`、`PI_OFFLINE`、`PI_SKIP_VERSION_CHECK`、`PWD` 属于受保护变量，通过 `extra_env` 传入会被忽略
- **工具默认全开**：`read`、`bash`、`edit`、`write`，外加 pi 默认关闭的只读工具 `grep`、`find`、`ls`。传 `tools: []` 会显式下发 `--no-tools` 全部禁用 —— pi 在既无 `--tools` 也无 `--no-tools` 时会回退到 `read`/`bash`/`edit`/`write`，所以这里必须显式关闭，不能只是省略参数。传 `tools: ["read", "grep", "find", "ls"]` 得到只读模式。pi 自身没有工具审批机制，无需 `--dangerously-skip-permissions` 之类的开关
- **`output_format: json` 会额外产出 `trace`**：包含 `event_counts`、`turns`、`tool_calls`、`malformed_lines`、`final_text`，以及最终的 `stop_reason` / `error_message`。注意事件名取自实际运行时输出——工具活动是 `tool_execution_start` / `tool_execution_end`（不是导出类型里写的 `tool_call` / `tool_result`），且单个 turn 可能跨多组 `message_start` / `message_end`，用 message 计数当轮数会算错
- **JSON 模式下 `success` 不能只看退出码**：pi 只在 text 模式把失败的一轮转成 exit 1（`print-mode.ts` 里那段逻辑被 `mode === 'text'` 包住），JSON 模式下即使最终 assistant message 的 `stopReason` 是 `error` / `aborted`，进程仍然退出 0。实测同一个上游 500：`--mode json` 退出 0，text 模式退出 1。因此 Nexus 会从 `agent_end` 的末条 assistant message 读取 `stopReason`，命中 `error` / `aborted` 时把 `success` 置为 False 并把 `errorMessage` 追加到 `stderr`，避免把模型或代理故障报成成功
- **`thinking` 需要配合 `reasoning: true`**：模型未被声明为 reasoning-capable 时，pi 会静默忽略 `--thinking`，此时 Nexus 会打一条 warning
- **Node 版本要求 `>= 22.19.0`**（与 `handy`、`openclaw` 一致）。pi 发布的 `dist/` 是预编译但未打包的（`dist/cli.js` 仍 import `./config.js`，并依赖约 140 个 runtime 包），因此安装必须解析依赖，不能只取 tarball

#### 在 gongfeng 上跑（实测记录）

`tests/e2e/config/pi_gongfeng_hello_example.yaml` 是实测跑通的配置（`echo` 任务，score 1.0，pi 真实调用 `write` 工具落盘）。两个容易踩的坑：

- **`worker_venv` 目前是必需的**。预构建的 `nexus-runtime` 镜像里还没有 `pi_agent` 模块，不加会报 `No module named 'nexus.extensions.agents.pi_agent'`。加上后 Nexus 会把本地工作树打成 wheel 送进沙箱（约 2 分钟）：

  ```yaml
  runtime_configs:
    main:
      worker_venv:
        modules: []
  ```

  pi 合入主干、runtime 镜像重建后这段就可以删掉。

- **`purpose` 要与 token 的配额桶匹配**。`test` / `eval` 桶可能报「当前Token在container+<bucket>桶下没有可用配额」，此时改用 `purpose: prod`。

模型侧要注意 `api_key` 里的 `provider=` 会决定 litellm 走哪个 provider：`provider=aws_third` 会被映射成 litellm 的 `bedrock`（见 `nexus/extensions/openai_clients/litellm.py` 的 `_API_KEY_PROVIDER_TO_LITELLM_PROVIDER`），此时 `base_url` 不带 `/v1`、`model` 必须写成 litellm 的 bedrock 形式（`bedrock/anthropic.<id>`），而不是网关侧的 `api_aws_third_*` 名字。

#### SWE-bench 冒烟实测

用官方 generator（`nexus/extensions/tasks/swe/generator.py`）跑 2 个真实实例，`SWE_AGENT_NAME=pi` 即可，无需为 pi 定制 generator：

| 实例 | 语言 | 轮数 | resolved | score |
| --- | --- | --- | --- | --- |
| `apache__datafusion-14468` | Rust | 41 | 是 | 1.0 |
| `stern__stern-300` | Go | 18 | 否 | 0.0 |

两次 `stop_reason` 均为 `stop`，合计 415 次 `bash`、36 次 `edit`、12 次 `read` 调用。`working_dir` 不需要在 `agent_config` 里写，`swe_task` 会按实例注入（SWE-bench 镜像是 `/testbed`）。

跑之前有两件事要准备：

- **宿主要装 `swe` 模块的依赖**（`dataclasses-json`、`unidiff`），否则 `swe_task` 起步就报 `ModuleNotFoundError`
- **`main` runtime 也要 `worker_venv`**。`SWE_WORKER_VENV=1` 只会给 `eval` 加（`main` 仅在 `claude_code_interactive` 时才加），而 pi 的 `runtime_setup` 跑在 `main` 里。镜像尚未包含 `pi_agent` 时，需要在 generator 产出的配置上补一层：

  ```python
  from nexus.extensions.tasks.swe.generator import generate_configs as _official

  def generate_configs():
      for item in _official():
          runtime_configs = dict(item.config.runtime_configs or {})
          runtime_configs['main'] = {**(runtime_configs.get('main') or {}), 'worker_venv': True}
          item.config.runtime_configs = runtime_configs
          yield item
  ```

  runtime 镜像重建并包含 `pi_agent` 之后，这层封装和 `SWE_WORKER_VENV` 都不再需要。

### WorkBuddy（wb-cli）

上游程序是私有 npm 包 `@tencent/nexus-wb-cli`，默认安装 dist-tag `latest`，实际版本见日志。`version` 写精确版本号可以固定版本，也可以写别的 dist-tag。环境变量、连通检查和 SWE 冒烟见 [`docs/examples/configs/wb-cli/README.md`](../../examples/configs/wb-cli/README.md)。
