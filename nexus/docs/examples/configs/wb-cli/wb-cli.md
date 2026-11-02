# wb

本目录的 Nexus 冒烟见 [README.md](./README.md)。示例安装 `@tencent/nexus-wb-cli@latest`，走 Chat Completions。不要给那次冒烟加 `--features '{"rollout":true}'`，也不要传 `--permission-mode`。`rollout` 只作用于 `models.json` 里 `"api": "openai-responses"` 的模型。

`wb` 在当前目录跑一次 WorkBuddy 的 agent，然后退出。它和桌面端里的 agent 是同一个：同一套 session、工具、技能和 MCP。用在脚本、CI、RL rollout，凡是不需要桌面端界面的地方。

跑一次请求，回复到 stdout：

```
wb "把 README 里的安装步骤改成三步"
```

不给 prompt 就从 stdin 读，一行一个 prompt，都在同一个 session 里，EOF 退出：

```
printf '%s\n' "先看 src 的结构" "把 utils 里重复的函数合并" | wb
```

`--json` 把过程和结果按 JSON Lines 输出，给程序用：

```
wb --json "把 README 里的安装步骤改成三步"
```

## 安装

`wb` 随桌面端安装包一起装好，不需要系统 Node，把启动脚本加进 PATH 即可：

| 平台 | 位置 |
| --- | --- |
| macOS | `/Applications/WorkBuddy.app/Contents/Resources/bin/wb` |
| Windows | `<安装目录>\resources\bin\wb.cmd` |
| Linux | `<安装目录>/resources/bin/wb` |

没有桌面端的机器（Linux 沙箱、容器）装 npm 包：

```
npm install -g ./wb-5.7.0-linux-x64.tgz
wb --help
```

包由 `pnpm pack:wb` 在目标平台上打出（`out/wb-<版本>-<平台>-<架构>.tgz`），只装得上同平台，Node 22 以上都行：绑定 Node ABI 的 `better-sqlite3` 不在包里，安装时 npm 按当前 Node 装它（先下预编译，没有就本机编译，所以要么能访问它的下载源，要么机器上有编译工具链）。Bun 跑不了（它不支持 better-sqlite3）。精简的 Linux 镜像装上 `ca-certificates`。文档面板要另装 Chromium，见「文档面板」；加 `--with-docs-engine` 才带文档引擎（约 220 MB）。

## 接入模型

模型用 `--model` 选。地址和 key 可走环境变量，也可按模型写入 `models.json`。`CODEBUDDY_API_KEY` 始终必填，作为这次运行的身份；自定义模型请求使用该模型配置的 `apiKey`。

通过环境变量接入自己的 OpenAI Chat Completions 兼容服务：请求发到 `<base>/chat/completions`，key 作为 `Authorization: Bearer`。Responses API 需在模型配置里显式选择，见下文新增章节。

```
export CODEBUDDY_BASE_URL=http://127.0.0.1:18466/v1
export CODEBUDDY_API_KEY=sk-local
wb --model my-model "把 README 里的安装步骤改成三步"
```

多个服务，或每个模型不同的 key：写进 `<config-dir>/models.json`（缺省 `~/.workbuddy/models.json`），用 `--model custom-local:<id>` 引用。

```json
{
  "models": [
    {
      "id": "my-model",
      "name": "my-model",
      "vendor": "local",
      "url": "http://127.0.0.1:18466/v1/chat/completions",
      "apiKey": "sk-local",
      "supportsToolCall": true,
      "maxInputTokens": 32000,
      "maxOutputTokens": 4096
    }
  ]
}
```

WorkBuddy 网关：给 WorkBuddy 的 key 和产品环境（`internal`、`ioa`、`external`、`cloudhosted`、`selfhosted`），`--model` 从模型目录里选。

```
export CODEBUDDY_API_KEY=ck_xxx CODEBUDDY_INTERNET_ENVIRONMENT=ioa
wb --model gpt-5.6-luna "把 README 里的安装步骤改成三步"
```

## aihub：Responses API


在 `<config-dir>/models.json` 中显式写入 `"api": "openai-responses"`，agent 就改走 Responses API；仅在模型目录里标注不会切换协议。`<config-dir>` 由 `WORKBUDDY_CONFIG_DIR` 指定，缺省 `~/.workbuddy`。已有其他模型时，将下面这一项合并到 `models` 数组，不要覆盖整个文件。

### 配置模型

```json
{
  "models": [
    {
      "id": "gpt-6-sol",
      "name": "gpt-6-sol",
      "vendor": "OpenAI",
      "api": "openai-responses",
      "url": "http://api.aihub.woa.com/openai/v1",
      "apiKey": "${AIHUB_KEY}?provider=azure&cache_task_id=${CODEBUDDY_SESSION_ID}&timeout=3600",
      "supportsReasoning": true,
      "supportsToolCall": true,
      "supportsImages": true,
      "maxInputTokens": 400000,
      "maxOutputTokens": 128000,
      "reasoning": {
        "defaultEffort": "max",
        "supportedEfforts": ["low", "medium", "high", "xhigh", "max"]
      }
    }
  ]
}
```

`url` 是 base URL，末尾带不带 `/responses` 都可以。`apiKey` 中两种占位符的来源不同：

- `${AIHUB_KEY}` 从环境变量展开。变量名可换，但不要用 `CODEBUDDY_` 前缀：`wb` 不会把这类环境变量透传给 agent。
- `${CODEBUDDY_SESSION_ID}` 在每次请求时替换为来源会话 id，作为 aihub 的 `cache_task_id`。普通新会话中，它就是结果行里的 `codebuddy.ai/conversationId`；subagent、辅助调用、`/btw`、fork 和 `--resume` 沿用来源会话的值。fork 后不要用新会话 id 代替来源 id。
- `providerData.upstream.rootSessionId` 记录实际使用的来源会话 id。保留 `cache_task_id`，避免多轮请求换账号后无法回放加密推理。

### 运行与 rollout

```bash
export CODEBUDDY_API_KEY='wb-local'  # wb 运行身份，任意非空值；不发给 aihub
export AIHUB_KEY='替换为你的 aihub key'
wb --json --model custom-local:gpt-6-sol --features '{"rollout":true}' \
  "Prove that there are infinitely many primes."
```

Responses API 请求使用 `models.json` 的 `apiKey`，不是 `CODEBUDDY_API_KEY`。不要把真实 key 写进共享文档、脚本仓库或排障日志。

| 行为 | 开启 rollout | 不开启 rollout |
| --- | --- | --- |
| Responses 请求的 `store` | `true` | 显式 `false` |
| AI 会话标题 | 不生成 | 新会话额外用同一模型调用一次；独立会话，不进任务 transcript，沿用来源 `cache_task_id` |
| 加密推理 `include` | 开启思考且 `supportsReasoning: true` 时发送 | 同样条件，与 `store` 无关 |
| 原始请求落盘 | 仍需单独配置 `OTEL_LOG_RAW_API_BODIES` | 同样需要单独配置 |

按以上配置，实际发送到 `/responses` 的请求体形如（`instructions`、`tools`、`input` 省略）：

```json
{
  "model": "gpt-6-sol",
  "store": true,
  "include": ["reasoning.encrypted_content"],
  "max_output_tokens": 128000,
  "reasoning": { "effort": "max", "summary": "auto" },
  "stream": true,
  "instructions": "…",
  "tools": [],
  "input": []
}
```

### 配置字段与回放

| 字段 / 行为 | 说明 |
| --- | --- |
| `supportsReasoning` | 必须为 `true` 且开启思考才发送 `include: ["reasoning.encrypted_content"]`；非推理模型会拒绝此参数 |
| `supportsImages` | `true` 时用户消息和工具结果中的图片发送为 `input_image`，`detail: "auto"`；设为 `false` 则替换为省略提示；恢复会话时从本地 blob 还原 |
| `maxInputTokens` | 按模型实际上下文窗口填写；结合响应 usage 决定自动压缩时机 |
| `maxOutputTokens` | 示例为 `128000`，发送为 `max_output_tokens` |
| `reasoning.defaultEffort` | 示例为 `max`，发送为 `reasoning.effort` |
| `reasoning.supportedEfforts` | 要用 `max` 就必须包含它，否则降为 `high`，目前只在日志提示 |
| 推理回放 | 只回放同一模型产出的加密推理，密文逐字节保留；其他模型的条目及无法回放推理的那次调用，不带上游 `fc_` / `msg_` id，避免配对校验错误 |
| 上下文超长 | 按窗口和 usage 自动压缩；流内 `context_length_exceeded` 按 400 处理并触发压缩 |
| 中止请求 | 等待首个输出事件期间取消会立即中止并断开连接；CLI 的 SIGINT / Ctrl-C 用法仍见「多回合」 |

## 一次请求

当前目录就是 agent 的 workspace。回复在 stdout，stderr 没有输出。退出码 0 完成、1 失败、2 用法错误、130 被 Ctrl-C 打断，全表见「参考」。

```
cd ~/work/my-project
wb --model my-model "把 README 里的安装步骤改成三步"
```

prompt 也可以从 stdin 来：

```
echo "总结一下这个仓库" | wb --model my-model
```

## 多回合

不给 prompt，`wb` 从 stdin 读：一行一个 prompt，来一行跑一个回合，全部在同一个 session 里；stdin 结束后跑完排队的回合再退出。语义和 `claude -p --input-format stream-json` 一致：回合进行中来的行排队，按序执行。终端里直接敲 `wb` 就是一行一回合的 REPL，Ctrl-D 结束。

```
printf '%s\n' "先看 src 的结构" "把 utils 里重复的函数合并" | wb --model my-model
```

一行能解析成 JSON 就按 JSON，否则整行是文本。JSON 的三种写法：

```
"分析一下这份文档"
{"type":"text","text":"分析一下这份文档"}
[{"type":"text","text":"分析一下"},{"type":"resource_link","uri":"file:///work/report.md","name":"report.md"}]
```

块是 [ACP user message](https://agentclientprotocol.com/protocol/v1/prompt-turn#1-user-message) 里的 content block：`text`、`resource_link`（引用文件）、`resource`（内嵌内容）、`image`、`audio`。多行文本用 JSON 字符串里的 `\n`；空行跳过。

打断正在跑的回合，给 `wb` 发 SIGINT（终端 Ctrl-C，程序里 `kill -INT`）：这一回合的结果是 `cancelled`，后面的行照常跑，退出码不受影响。空闲时的 SIGINT 结束输入，退出码 130；SIGTERM 打断当前回合并结束输入，退出码 143。

## 模型、模式、预设、推理强度

这些选项只作用于这次运行，不写回任何配置。

```
wb --mode ask "这个仓库的测试是怎么组织的"
wb --mode plan "把认证模块拆成独立包，给个方案"
wb --preset work "把这份会议纪要整理成待办"
wb --model gpt-5.6-luna --effort high "把 README 里的安装步骤改成三步"
```

| 选项 | 取值 |
| --- | --- |
| `--mode <mode>` | `craft`（缺省：读写文件、执行命令）、`ask`（只用只读工具）、`plan`（只读，交出方案，方案就是回复）、`quick`（纯对话，无工具） |
| `--preset <name>` | 桌面端首页的预设：`code`、`work`、`design` |
| `--effort <level>` | 推理强度：`minimal`、`low`、`medium`、`high`、`xhigh`、`max`，只对支持它的模型生效 |

使用 Responses API 时，要使用 `max`，模型配置的 `reasoning.supportedEfforts` 必须包含 `max`，否则会降为 `high`。

## 继续 session

```
wb --resume last "接着把测试也补上"
wb --resume 51422d18-… --mode ask "刚才改了哪些文件"
```

`last` 是当前目录里最近的 session；id 在 `--json` 结果行的 `codebuddy.ai/conversationId`。续上的 session 沿用原来的 preset 和 effort，`--model` 和 `--mode` 可以换。多回合模式同样可以 `--resume`。

使用 Responses API 时，`--resume` 保持来源 `cache_task_id` 和加密推理回放不变；切换模型只会回放同一模型的推理。fork 用法见 [rollout 轨迹与原始请求](#rollout-traces)。

## 无人值守

`wb` 不会停下来等人。需要人决定的请求（敏感数据外发、连接外部服务、向你提问）一律拒绝，回合继续，agent 会在回复里说明。plan 模式里 agent 交出方案时，`wb` 拒绝执行，把方案当作回复输出。工具调用直接在本机执行。

## 在脚本里用

stdout 只有回复正文，诊断都在 stderr（缺省不输出，`WORKBUDDY_LOG_LEVEL=info` 才有）。要过程和结构化的结果，用 `--json`：stdout 变成 JSON Lines，每行四种之一，看字段就能分辨。

```
echo "reply with the single word pong" | wb --json
```

```
{"sessionId":"51422d18-…","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"reply with the single word pong"}}}
{"sessionId":"51422d18-…","method":"_codebuddy.ai/command","params":{"action":"workspace_info","params":{"isGitWorkspace":false}}}
{"sessionId":"51422d18-…","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"pong"}}}
{"stopReason":"end_turn","_meta":{"taskOutcome":"SUCCESS","codebuddy.ai/conversationId":"51422d18-…","codebuddy.ai/transcriptPath":"/home/me/.workbuddy/projects/…/51422d18-….jsonl"}}
```

| 行 | 含义 |
| --- | --- |
| `{sessionId, update}` | agent 的一条 ACP `session/update`：`agent_message_chunk`、`agent_thought_chunk`、`tool_call`、`tool_call_update`、`usage_update`… |
| `{sessionId, method, params}` | session 事件，如 `_codebuddy.ai/command`（agent 打开了文档、网页） |
| `{sessionId, channel, data}` | 文档面板事件：`docs:selectionChanged`、`docs:selectionSend` |
| `{stopReason, _meta}` | 一个回合的结果，每个输入行一条，按顺序。`_meta` 里有 `taskOutcome`、session id、完整 transcript 的路径 `codebuddy.ai/transcriptPath`、plan 模式的方案 `codebuddy.ai/plan`、异常终态 `codebuddy.ai/terminalError` |

这些 `_meta` 字段按场景出现，不保证每次都有；Responses API 的限流失败可能缺少 `codebuddy.ai/terminalError`，也可能显示为 `stopReason: "refusal"`。请结合 `taskOutcome` 和日志判断，见 [Responses API 排障](#responses-troubleshooting)。

## rollout 轨迹与原始请求

`--json` 每回合最后一行给出结果，`_meta` 里的 `codebuddy.ai/conversationId` 用于续跑，`codebuddy.ai/transcriptPath` 指向主 transcript。

```json
{"stopReason":"end_turn","_meta":{"taskOutcome":"SUCCESS","codebuddy.ai/conversationId":"a33e522e-…","codebuddy.ai/transcriptPath":"/…/projects/<workspace-key>/a33e522e-….jsonl"}}
```

### transcript 里有什么

| 需要的数据 | 保存位置 / 说明 |
| --- | --- |
| 加密推理 | `type: "reasoning"` 条目的 `providerData.encrypted_content`，与响应密文逐字节一致 |
| 类型 | 条目的 `type` 为 `reasoning`；agent 压缩是本地摘要，不产生 `compaction` 类型密文块 |
| 解密需要的模型和区域 | 输出条目的 `providerData.upstream.model`、`providerData.upstream.region` |
| 账号（可选） | 响应头 `x-account-id` 对应 `providerData.upstream.accountId` |
| 来源会话 | `providerData.upstream.rootSessionId`，即请求使用的 `cache_task_id` |
| 顺序与角色 | 行序及 `parentId` 链；`message`（user / assistant）、`reasoning`、`function_call`、`function_call_result` |
| subagent 轨迹 | 主 transcript 去掉 `.jsonl` 后的目录下，`subagents/agent-*.jsonl` |
| `instructions`、`tools`、完整 `input` | 需要额外记录原始请求；`tools` 列表不在 transcript 中 |

放量前检查已有 reasoning 块的密文是否为 `gAAAAAB…` 前缀，并核对 model、region；响应提供账号时同时核对 account。没有 reasoning 条目不一定是错误：简单问题或较低 effort 可能不产出推理。

下面检查主 transcript；subagent 文件也应逐个做相同检查。`T` 替换为结果行给出的路径：

```bash
T='/path/to/transcript.jsonl'
jq -c 'select(.type == "reasoning") | {enc: (.providerData.encrypted_content // "" | .[0:7]), len: (.providerData.encrypted_content // "" | length), model: .providerData.upstream.model, region: .providerData.upstream.region, account: .providerData.upstream.accountId}' "$T"
```

### 记录每次调用的原始请求

```bash
export OTEL_LOG_RAW_API_BODIES='file:/data/rollout/raw'
wb --json --model custom-local:gpt-6-sol --features '{"rollout":true}' \
  "Prove that there are infinitely many primes."
```

把示例目录改成当前用户可写的受控目录。目录权限为 `0700`，文件为 `0600`；内容包含完整 prompt 和工具输出，不要公开上传。

| 文件 | 内容 |
| --- | --- |
| `<message_id>.request.json` | 实际发出的请求体，包括 `instructions`、`tools`、完整 `input` |
| `<message_id>.response.json` | `id`、`request_id`、`output`、`usage`；与请求文件同名，一次调用一对 |
| `index.jsonl` | 每次调用一行：`timestamp`、`session_id`、`root_session_id`、`query_source`、`model`、`request_id`、`message_id`、`request_file`、`response_file` |

`query_source` 区分主对话 `conversation`、标题生成 `conversation_topic` 等。`request_id` 目前是 aihub 回显的拼接值 `<本地 id>, <aihub id>`，不要假定它是单个 id。将变量设为其他真值（如 `1`）时，会改在 trace 的 span event 中记录，单条截断到 60KB。

### 多轮、续跑与 fork

同一进程不给 prompt，从 stdin 一行输入一轮；回合进行中输入的行只会排队，不会插进当前回合。跨进程继续任务时使用结果行里的会话 id，并继续带上需要的运行选项：

```bash
wb --json --model custom-local:gpt-6-sol --features '{"rollout":true}' \
  --resume '<conversationId>' "继续这个任务"
```

`--resume` 不改变来源 `cache_task_id`，历史加密推理原样回放。`--resume <id> --fork-session` 创建新的会话，但继续沿用来源会话的 `cache_task_id`，不会因为 fork 换到别的账号。

## 文档面板

`--remote-debugging-port <port>` 给这次运行一个文档面板：一个 headless 浏览器，和桌面端右侧面板一样，DevTools 协议开在 `127.0.0.1:<port>`（和 Chrome 的旗标同名）。你的程序用 Playwright 的 `connectOverCDP`、puppeteer 或裸 CDP 接上去，就能像人一样在文档里打字、选中文字、点"AI 编辑"、截图。

```
wb --json --remote-debugging-port 9222 --open report.docx --model my-model
```

- `--open <file>`：开局就把文档开在面板里，像用户先打开了它再提问。可重复。
- 不给 `--open` 也行：agent 用 `present_files` 呈现的 Word / Excel / PowerPoint 会自动开进面板，流里同时出现 `_codebuddy.ai/command`。

`http://127.0.0.1:<port>/json` 列出所有页面，文档页面的地址里带 `localFilePath=`。每回合结果行的 `_meta` 里有 `codebuddy.ai/remoteDebuggingPort` 和开着的 `codebuddy.ai/documents`。

人在页面上选中文字、点"AI 编辑"、输入指令回车，流里出现一条：

```
{"sessionId":"…","channel":"docs:selectionSend","data":{"documentResourceUri":"file:///work/report.docx","filePath":"/work/report.docx","fileType":"word","selection":{"fileId":"…","rangeId":"…","ranges":[{"begin":149,"end":158,"length":9}],"aiPrompt":"改成更正式的说法",…},"sequence":1,"sequenceEpoch":"…"}}
```

只选中不发送时是 `docs:selectionChanged`，载荷相同。要让 agent 处理这次选区，把它写成下面这行发回 stdin（选区块加指令文本，和桌面端发出的一样）：

```json
[
  {
    "type": "resource_link",
    "name": "选区",
    "uri": "tdoc-selection://<selection.rangeId>",
    "_meta": {
      "mentionType": "tencentDocsSelection",
      "type": "document-selection",
      "documentResourceUri": "<data.documentResourceUri>",
      "filePath": "<data.filePath>",
      "fileType": "<data.fileType>",
      "fileId": "<selection.fileId>",
      "rangeId": "<selection.rangeId>",
      "selectionPayload": { "…": "data.selection 原样" },
      "sequence": "<data.sequence>",
      "sequenceEpoch": "<data.sequenceEpoch>"
    }
  },
  { "type": "text", "text": "<selection.aiPrompt>" }
]
```

必填的是 `type`、`_meta.mentionType`、`_meta.filePath`（或 `documentResourceUri`）、`_meta.rangeId`、`_meta.selectionPayload`，其余给界面用。WorkBuddy CLI 仓库里的 `docs/examples/wb-docs-coedit.mjs` 是完整的一遍：起 `wb`、接 CDP、agent 改文档、人点 AI 编辑、把选区发回去。本 Nexus 仓库没有这份脚本。

面板里正文是 canvas 渲染，DOM 里没有正文文本；状态栏的字数是最便宜的"变了"信号，要核对内容就看文件或截图。`wb` 退出时面板随之关闭，没保存的改动丢弃。

面板用 Playwright 的 chromium-headless-shell，`wb` 会在 Playwright 的缓存目录里找到它（`PLAYWRIGHT_BROWSERS_PATH` 或系统缓存目录），也可以用 `WORKBUDDY_CHROMIUM` 指定任意 Chromium。Linux 加 `--with-deps`，再装 `fonts-noto-cjk`。

```
npx playwright-core install chromium-headless-shell
```

文档引擎随桌面端安装包分发；npm 包默认不带，打包时加 `--with-docs-engine` 才带，或者事后把 `linux-x64` 的 `editor_sdk` 放到包里的 `lib/node_modules/@tencent/tencent-docs-ai-engine/bin/linux-x64/`（下载地址见该包的 `scripts/download-editor-sdk.js`）。引擎第一次启动约 12 秒，一个 `wb` 进程只付一次。

## 数据目录

`WORKBUDDY_CONFIG_DIR` 是数据目录，缺省 `~/.workbuddy`。多个任务互不干扰，就一个任务一个目录；空目录第一次用要下载插件市场，多花十几秒，先跑一次再复制给其它任务。agent 会在 workspace 里写 `.workbuddy/`（工作记忆等），做 workspace 对比时排除它。

| 需要 | 位置 |
| --- | --- |
| 自定义模型服务 | `<config-dir>/models.json` |
| 回合记录（完整 transcript） | `<config-dir>/projects/<workspace-key>/<conversationId>.jsonl` |
| 日志 | `<config-dir>/logs/daemon.log`；`WORKBUDDY_LOG_LEVEL=info` 同时镜像到 stderr |

## 功能开关

`--features` 传 JSON 对象或文件路径，缺省全关，未知名字是用法错误。

```
wb --features '{"tencentDocsEngine":true}' "把依赖升到最新并跑一遍测试"
```

| 功能 | 作用 |
| --- | --- |
| `tencentDocsEngine` | 启动时就拉起文档引擎并交给 agent，让它在后台改文档；`--open` 蕴含它 |
| `rollout` | Responses API 请求 `store: true`，且不生成 AI 会话标题；未开启时显式 `store: false`。不自动开启原始请求落盘 |

## 参考

| 选项 | 作用 |
| --- | --- |
| `--model <id>` | 服务上的模型名、目录里的模型，或 `custom-local:<id>` |
| `--effort <level>` | `minimal`、`low`、`medium`、`high`、`xhigh`、`max` |
| `--mode <mode>` | `craft`、`ask`、`plan`、`quick` |
| `--preset <name>` | `code`、`work`、`design` |
| `--resume <last\|id>` | 继续一个 session |
| `--fork-session` | 与 `--resume <id>` 配合使用：fork 后沿用来源 `cache_task_id` |
| `--json` | JSON Lines 输出 |
| `--features <json\|file>` | 功能开关 |
| `--remote-debugging-port <port>` | 文档面板，DevTools 开在 `127.0.0.1:<port>` |
| `--open <file>` | 开局就开在面板里的文档，可重复 |
| `-v, --version`、`-h, --help` | 版本、用法 |

| 环境变量 | 作用 |
| --- | --- |
| `CODEBUDDY_BASE_URL` | 自己的 OpenAI 兼容服务地址 |
| `CODEBUDDY_API_KEY` | 必填：运行身份；环境变量接入模型时也作 Bearer token。自定义模型使用 `models.json` 的 `apiKey`，直连 aihub 时运行身份不发给 aihub |
| `CODEBUDDY_INTERNET_ENVIRONMENT` | WorkBuddy 网关的产品环境 |
| `ACC_USER_ID`、`ACC_USER_NICKNAME` | key 下的身份，缺省按 key 派生 |
| `WORKBUDDY_CONFIG_DIR` | 数据目录，缺省 `~/.workbuddy` |
| `WORKBUDDY_LOG_LEVEL` | 日志镜像到 stderr：`debug`、`info`、`warn`、`error` |
| `WORKBUDDY_CHROMIUM`、`PLAYWRIGHT_BROWSERS_PATH` | 文档面板用的浏览器 |
| `CODEBUDDY_SMALL_FAST_MODEL`、`CODEBUDDY_BIG_SLOW_MODEL`、`CODEBUDDY_CODE_SUBAGENT_MODEL`、`CODEBUDDY_CUSTOM_HEADERS` | `wb` 不透传 `CODEBUDDY_*` 环境变量，不要依赖这些变量配置 agent |
| `AIHUB_KEY` | 本文 aihub 示例的模型凭证变量，被 `${AIHUB_KEY}` 引用；名字可换，但不能以 `CODEBUDDY_` 开头 |
| `OTEL_LOG_RAW_API_BODIES` | 可选；使用 `file:<dir>` 落盘原始请求、响应和索引，详见轨迹章节 |

| 退出码 | 含义 |
| --- | --- |
| 0 | 回合完成；多回合时全部完成，中途被 Ctrl-C 打断过也算 |
| 1 | 有回合失败，或 `wb` 起不来 |
| 2 | 用法错误 |
| 130 | 被 SIGINT 结束：一次请求被 Ctrl-C 打断，或多回合空闲时 Ctrl-C |
| 143 | 被 SIGTERM 结束 |

## Responses API 排障


### 先区分限流与拒答

aihub 可能返回 HTTP 200，但在流中发送 `error` 和 `response.failed`。agent 在输出开始前识别这类错误，按普通 OpenAI 接口的状态处理；失败响应不会写入历史。

| 流内错误 | 当前处理 |
| --- | --- |
| 限流 429 | 退避重试，默认只重试 1 次 |
| 额度耗尽 429 | 不重试；先处理额度问题 |
| 请求或图片无效 400 | 不重试；修正请求或图片 |
| `server_error` 500 | 不重试 |
| `context_length_exceeded` | 按 400 处理并触发压缩 |

**不要仅凭 `stopReason: "refusal"` 判断模型拒答。** 当前限流失败也可能得到 `taskOutcome: "FAILED_MODEL_REQUEST"`、`stopReason: "refusal"`，而 `_meta` 没有 `codebuddy.ai/terminalError`。到 `<config-dir>/logs/<日期>/sdk/conversations/<conversationId>.log` 查找 `exceeded rate limit` 等错误信息确认原因。

### 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 400：`The requested item was created under a different Azure OpenAI resource` | 多轮落到不同账号；检查 `apiKey` 是否带 `cache_task_id=${CODEBUDDY_SESSION_ID}` |
| 限流重试一次后失败 | harness 可用 `--resume <conversationId>` 重跑失败回合；保留同一来源会话。当前不要依赖设置 `CODEBUDDY_RETRY_WATCHDOG`：`wb` 不透传 `CODEBUDDY_*` 环境变量 |
| 401，Authorization 中仍是字面量 `${AIHUB_KEY}` | 环境变量未正确展开或未传入 agent；确认变量已导出，变量名不能以 `CODEBUDDY_` 开头 |
| 请求中 `reasoning.effort` 变成 `high` | `reasoning.supportedEfforts` 未包含 `max`；补上后再运行 |
| 某轮没有 reasoning 条目 | 简单问题或较低 effort 下可能正常，不必据此判失败 |
| 找不到 `tools` 或 subagent 轨迹 | `tools` 在原始请求里，需要 `OTEL_LOG_RAW_API_BODIES`；subagent 记录在独立 JSONL 文件中 |
| 意外多一次模型调用 | 未开 rollout 时会生成 AI 会话标题；开启 `--features '{"rollout":true}'` 后不生成 |

### 注意事项

- 缺少 `cache_task_id` 不会在启动时告警，可能直到账号切换才出现 400 / 429；运行前自行检查配置。
- `max` 降为 `high` 目前只记日志，没有 stderr 可见告警。
- 自定义模型请求目前还可能带 `x-user-id`、`x-product`、`x-ide-*` 等内部头和重复的 `x-api-key`；接入自定义服务时，请确认服务端对这些请求头的处理方式。
