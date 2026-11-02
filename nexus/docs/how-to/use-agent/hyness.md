# Hyness App / Harness 接入

临时替换 System Prompt、复用已发布双包的步骤见
[用独立 App 配置做 System Prompt 实验](hyness-prompt-experiments.md)。

## 独立 App 发布包（2026-10-21）

三个预设：`hyness_mini_swe_agent`、`hyness_swe_agent`、
`hyness_workbuddy`。它们各自从 npm registry 安装固定的 engine 和独立 App，
mini/SWE 使用 `0.0.1-scaffold-apps.20261021.7` 引擎与 `0.1.0-scaffold-apps.20261021.7` App；
WorkBuddy 使用 `0.0.1-scaffold-apps.20261021.8` 引擎与 `0.1.0-scaffold-apps.20261021.12` App。
这些包已发布到腾讯 npm 镜像的 `scaffold-apps` tag；预设固定完整版本，
不再需要 `HYNESS_PACKAGE_DIR`。本地候选测试可通过 `HYNESS_PACKAGES` 显式传入两个绝对 `.tgz` 路径。
三个预设统一使用不带 `_app` 后缀的名称。旧 `hyness_mini_swe_agent_app`、
`hyness_swe_agent_app`、`hyness_workbuddy_app` 文件已删除；调用脚本需去掉 `_app`。
短名称原来的 alpha 内置 Harness 配置已替换为对应独立 App 的发布版。
已有 `_app` 评测迁移时只调整入口名称，沿用相同包版本、App、模型、任务和评分配置。
更早的 alpha 内置 Harness 与独立 App 的 Prompt、工具和提交协议不同，不能混合比较分数。

```bash
export AGENT_CONFIG=hyness_swe_agent
export RUNTIME_CONFIG=swe_runtime
export LLM_CONFIG=/path/to/model.yaml
export DATASET_PATH=/path/to/swe-multi.jsonl
export SWE_LIMIT=2
export AGENT_TIMEOUT=1800
export AGENT_MAX_TURNS=100
uv run nexus run-batch -g nexus/extensions/configs/generator_swe_template.py \
  -d /tmp/hyness-app-smoke.db --max-concurrent 2
```

mini App 使用 SWE-bench 原生模板、bash 与 XML 工具反馈；SWE App 通过 MCP alias
提供 `bash`、`str_replace_editor`、`submit`；WorkBuddy App 从 `prompts/system.md`
读取完整 SP、从 `prompts/task.md` 渲染首条用户消息，固定原生 CLI 执行 20 个工具。
WorkBuddy App 不再选择 `runtime.context.source`；Markdown 保留捕获的固定正文，
环境路径改为可移植写法，旧时间和目录树快照不注入。各 App 的 Prompt、工具与策略归 App/插件所有。
本次 Nexus 只增加显式 schema 保留配置与安装后的 App 路径环境变量，未改训练 loop。

`preserve_tool_schema: true` 让 Anthropic 代理保留 `default`、`additionalProperties`、
空 `properties`、省略/显式空 `required` 等 JSON Schema 差异。默认 false 保留原有模型
兼容清理。只有模型支持这些 schema 时才启用；该开关不改变消息或路由。
`NEXUS_HYNESS_APP_CONFIG` 是运行时解析后的 App 文件路径，供已授权的准备脚本定位资源。
WorkBuddy 预设从 App 的依赖中定位 `prepare.mjs`，显式准备 better-sqlite3 addon；
不再要求手工计算安装 prefix 或传原生二进制路径。

已安装包的离线检查：

```bash
export HYNESS_REFERENCE_WORKSPACE=/path/to/checkouts  # 包含 nexus/ 与 hyness/
export HYNESS_EVIDENCE_DIR=/tmp/hyness-app-evidence
export HYNESS_TEST_PREFIX=/path/to/npm-install-prefix
export HYNESS_TEST_NODE=/path/to/supported/node
uv run python tests/integration/anthropic_proxy/check_scaffold_apps.py
```

这项检查保留真实 App-engine 校验、CLI、MCP/原生工具、代理与退出协议，仅替换安装步骤和
模型响应。Hyness 的 `docs/EVALUATION_SOP.md` 记录源码打包、SDK 检查及真实 SWE 流程。

本次命名合并不改变独立 App 的执行代码或重新发布 npm；历史 trace、数据库和已导出的
Parquet 内容不改写。若数据行已经固化 `agent_config`，更换共享 YAML 不会自动更新该行。
以下章节描述通用入口及各 App 的配置；OpenCC 仍使用其配套 alpha 双包。

本模块用同一个 `agent_name: hyness` 把不同 Hyness App/Harness 接到现有 Nexus 模型代理、
任务评分、chat_tracker 和 Gateway。prompt、工具、插件和 hooks 由选中的 App/Harness 提供。
无需安装额外的 Python adapter。

## 通过共享 scaffold 配置选择

使用 [共享配置层](../../../nexus/extensions/configs/README.md) 的三轴接口，不需要新的 generator：

```bash
export RUNTIME_CONFIG=swe_runtime
export AGENT_CONFIG=hyness_mini_swe_agent
export LLM_CONFIG=hy3.openai
export DATASET_PATH=/path/to/swe.jsonl
export SWE_LIMIT=2
# 在环境里设置模型和 runtime 所需凭据后执行。
uv run nexus run-batch -g nexus/extensions/configs/generator_swe_template.py -d hyness-smoke.db --max-concurrent 2
```

| `AGENT_CONFIG` | Hyness Harness | 依赖 |
| --- | --- | --- |
| `hyness` | 调用方选择，缺省使用 App 默认值 | 必填 `HYNESS_PACKAGES`、`HYNESS_APP_CONFIG`，无内置业务包版本 |
| `hyness_opencc` | `opencc` | 引擎和配套 OpenCC App 包；精确版本见下文 |
| `hyness_mini_swe_agent` | `mini-swe-agent` | `.7` 引擎 + mini `.7` App，`bash` 和提交 hooks |
| `hyness_swe_agent` | `swe-agent` | `.7` 引擎 + SWE `.7` App，`bash / str_replace_editor / submit` |
| `hyness_workbuddy` | `workbuddy` | 精确版本的引擎 + WorkBuddy App，包含完整 Markdown SP 和原生工具依赖 |

原生 `AGENT_CONFIG=mini_swe_agent`、`swe_agent`、`workbuddy` 保持原含义。
Hyness 的 YAML 都在 `nexus/extensions/configs/agents/`；模型族差异由各自的
`agent_config_by_llm` 引用所选 llm 的事实，任务和 verifier 仍由 runtime/task 轴决定。

通用入口用 `HYNESS_PACKAGES`（JSON 数组）、`HYNESS_APP_CONFIG`、`HYNESS_HARNESS_ID`、
`HYNESS_APP_CONFIG_OVERLAY` 覆盖包和资源选择；`AGENT_MAX_TURNS`、`AGENT_TIMEOUT`、
`AGENT_WORKING_DIR` 控制执行预算和目录。任意新的 App 可以复用此接口，无需增加 agent 模块。

### 新 App 优先使用通用入口

四个命名预设保存明确的工具边界和运行设置；新的 App 不应从 OpenCC 预设随意替换一个路径。
使用 `AGENT_CONFIG=hyness`，明确提供安装包和 App，Harness、overlay 缺省均为 null：

```bash
export AGENT_CONFIG=hyness
export HYNESS_PACKAGES='["@tencent/hyness@0.0.1-scaffold-apps.20261021.7","@tencent/hyness-mini-swe-agent@0.1.0-scaffold-apps.20261021.7"]'
export HYNESS_APP_CONFIG='@tencent/hyness-mini-swe-agent/apps/mini-swe-agent/app.yaml'
export HYNESS_HARNESS_ID=null
export HYNESS_APP_CONFIG_OVERLAY=null
# RUNTIME_CONFIG、LLM_CONFIG、DATASET_PATH 仍沿用上面的三轴用法。
```

这里用已安装验证的 mini-SWE 双包和 App 默认 Harness 演示。其他独立 App 包可追加到 `HYNESS_PACKAGES`，再把
`HYNESS_APP_CONFIG` 指向包内文件；`harness_id` 不限于 Nexus 预先知道的几个名称，最终由 Hyness
验证是否存在。使用 App 自己的默认 Harness 时显式设置 `HYNESS_HARNESS_ID=null`，避免 shell 中
前一个实验的值残留。自定义 App 的 prompt、工具、hooks、后台与压缩策略由该 App 及其 overlay
决定；通用 YAML 不额外套用 OpenCC 的 deny 列表或关闭压缩。

固定实验建议将这些值写入独立的 `body/merge/source` YAML，用 `AGENT_CONFIG=/path/to/my_hyness.yaml`
选文件，连同包版本、App、Harness 和策略一起版本化。无需修改共享 loader，也不需要新增 Python agent。

包列表会去除重复项和首尾空格，同一个包名同时指定两个版本会在 agent 输入校验时失败。
不同包允许不同版本号：引擎和业务 App 的兼容性由包依赖及现有的引擎解析一致性检查验证，
不能只比较版本字符串相等。`.tgz` 的包身份在安装时才可知；此处冲突检查针对显式 npm 包名。

## 显式 App 选择与旧配置兼容

直接配置 mini-SWE：

```yaml
imports:
  - nexus.extensions.agents.hyness
agent_name: hyness
agent_config:
  packages:
    - "@tencent/hyness@0.0.1-scaffold-apps.20261021.7"
    - "@tencent/hyness-mini-swe-agent@0.1.0-scaffold-apps.20261021.7"
  app_config: "@tencent/hyness-mini-swe-agent/apps/mini-swe-agent/app.yaml"
  harness_id: mini-swe-agent
  preserve_tool_schema: true
  max_turns: 100
```

显式指定其他 `app_config` 时，省略的 `app_config_overlay` 为 `None`，省略的 `harness_id`
由 App 自己决定；不再带入 OpenCC overlay/Harness。旧配置省略这三个字段时仍选择原来的
OpenCC App、`opencc` Harness 和 training overlay。显式 null 保持 null，跨序列化也不变。

共享 run-part loader 把 YAML null 解释为删除。要明确关闭 OpenCC overlay，使用
`HYNESS_APP_CONFIG_OVERLAY=null`，或者在 run-part 中写 `${oc.decode:null}`，让 null 在合并后生效。
共享配置已经按这个方式处理。选择自定义 App 的共享 profile 时，App、Harness 和 overlay
应一起明确选择，不能把 OpenCC profile 的工具限制视为所有 App 的默认策略。

WorkBuddy 使用最终发布包：

```yaml
agent_config:
  packages:
    - "@tencent/hyness@0.0.1-scaffold-apps.20261021.8"
    - "@tencent/hyness-workbuddy@0.1.0-scaffold-apps.20261021.12"
  app_config: "@tencent/hyness-workbuddy/apps/workbuddy/app.yaml"
  harness_id: workbuddy
```

共享 `hyness_workbuddy` 预设包含这些值及安装后的 addon 准备脚本。
`.8` 引擎修复默认 Anthropic 路径没有向工具提供 `modelCall` 的问题；`.12` App 精确依赖它。
前台 `Agent` 的 `general-purpose` 和 `Explore` 已验证经 Nexus 模型代理调用子模型、执行原生 Read，
再将文件内容返回父任务，无需额外设置 `ANTHROPIC_CONTEXT_WINDOW`。旧 `.7`＋`.11` 默认配置会使
Agent 返回 `NATIVE_OUTCOME_UNKNOWN`，不能把外层正常退出当作工具成功。
`.12` 保留 `.11` 的权限拒绝修复和 660 秒原生工具预算，覆盖 Bash 的 600 秒上限及回执时间。
新建 WorkBuddy 会话已带 `--features '{"rollout":true}'`；rollout 模式仍可能返回原生权限拒绝。
明确拒绝会作为 `WORKBUDDY_PERMISSION_DENIED` 工具错误返回，随后独立的工具调用可以继续。
真正的中断、回执丢失和持久化失败仍需要显式恢复，不会自动重放写入。
原生 CLI 和依赖由 App 携带，无需单独设置 `HYNESS_WORKBUDDY_PATH/PACKAGE_ROOT/SHA256`。
Agent 安装包回归使用固定模型响应、真实 CLI 和子工具；后台 Agent、resume、完整 RL 子轨迹、
预算和 token/logprob 归属仍需独立验收。现有 Parquet 若固化旧包版本，需要显式更新或重新生成。

WorkBuddy profile 设置 `session_persistence: true`，以确认原生工具的持久化回执。
其他 profile 保持默认关闭；自定义 App 需要会话工具时可显式打开。该状态保存在本次调用的
临时目录，执行结束仍会清理，不能与 `--no-session-persistence` 同时配置。

## OpenCC 配置与版本

把原配置的 `agent_name` 和 agent import 改为 `hyness`，保留任务、模型、评分、
超时、轮数、MCP 与启动脚本；删除 HyCode 的 `version`，明确填写配套包：

```yaml
imports:
  - nexus.extensions.harbor_bridge.task
  - nexus.extensions.agents.hyness
agent_name: hyness
agent_config:
  packages:
    - "@tencent/hyness@0.0.1-alpha.20261020123306"
    - "@tencent/hyness-opencc@0.2.1-alpha.20261020.1"
  app_config: "@tencent/hyness-opencc/apps/opencc/app.yaml"
  app_config_overlay: "@tencent/hyness-opencc/apps/opencc/overlays/training.yaml"
  harness_id: opencc
  max_turns: 600
  timeout: 43200
runtime_configs:
  main:
    worker_venv:
      modules: []
```

共享 `hyness_opencc` 默认使用上面这组包。OpenCC 的 `package.json` 精确依赖
`@tencent/hyness@0.0.1-alpha.20261020123306`；安装后 App 启动器和宿主 SDK 解析到同一引擎。
2026-10-20 的安装包测试保留共享 YAML 的全部六项 deny，验证启动、Bash 文件修改、
模型请求中排除禁用工具、execution-result v1 和 usageBreakdown 转换通过。

开发验证可把 `packages` 换成两个宿主机绝对 `.tgz` 路径；Nexus 按内容哈希上传到普通 sandbox。
不要覆盖已发布版本。历史实验包的结果只对应其原始版本，不能代替当前包的兼容验证。

使用 execution target（切换用户或 Compose）时请使用 registry 包；本模块会拒绝
未交接到目标文件系统的本地 tarball。该路径复用公共 execution transport。

包必须提供 native Anthropic 模型入口、execution-result v1 和 usageBreakdown。
包名或 main 分支本身不能证明这些协议已具备；版本或协议不匹配会报初始化/协议错误。

### 自定义 App 交付

App 包包含 YAML、Prompt、Harness、所需插件和精确引擎依赖。用 Hyness 现有的
Data Production 导出流程生成，打包步骤见 Hyness 仓库的 `docs/EVALUATION_SOP.md`。
安装后选择 `AGENT_CONFIG=hyness`，设置 `HYNESS_PACKAGES`、`HYNESS_APP_CONFIG`，
按 App 需要选择 Harness 和 overlay；不需要新增 Python Agent。

## 首轮直接输入图片（stream-json）

`agent_config.input_format` 默认是 `text`，保留已有纯文本 stdin 行为。
需要模型在首轮直接看到图片、无需先调用 `Read` 时，显式设置为 `stream-json`：

```yaml
agent_name: hyness
agent_config:
  packages:
    - "@tencent/hyness@0.0.1-scaffold-apps.20261021.8"
  app_config: "@tencent/hyness/app-definitions/hyness/app.yaml"
  harness_id: null
  app_config_overlay: null
  input_format: stream-json
  user_prompt:
    - type: text
      text: "描述这张图片，不要调用工具。"
    - type: image_url
      image_url:
        url: "data:image/png;base64,<图片的 base64>"
```

Nexus 将标准图文 `user_prompt` 编码为一条 SDK `user` 消息，以一行 JSON 加换行写入 stdin；
CLI 的输出仍使用 `stream-json` 和原有 `execution-result` 协议。普通字符串（包括看起来像
JSON 的文本）始终作为用户正文，不会作为控制消息执行。此开关不提供原始 SDK 控制流或交互式续写。

支持文本以及 JPEG、PNG、GIF、WebP 的 base64 `image_url` data URI，保留图文块顺序。
调用方应预先读取或下载图片；当前不解析远程 URL 或文件路径，不支持音频、视频和文件块。
OpenAI 的 `image_url.detail` 没有对应的 Anthropic 图片字段，不会转发；图片仍可能经过
Hyness 自身的缩放处理。选择的 App/Harness 和模型也必须支持图文输入，纯文本任务模板可能拒绝图片。

`extra_cli_args` 中若已有 `--input-format`，其值必须与 `input_format` 相同；不要用它单独切换协议。
本地验证使用上面固定版本的默认 Hyness App、真实 CLI 和 Nexus Anthropic 代理，模型后端为替身：
检查首个模型请求包含实际图片、未发生工具调用，以及原有 text 模式与 stream-json 文本输入均正常。

```bash
HYNESS_TEST_PREFIX=/path/to/npm-prefix \
HYNESS_TEST_NODE=/path/to/node \
uv run pytest tests/integration/anthropic_proxy/test_hyness_apps.py \
  -k input_reaches_first_model_request_without_tools -q
```

## 接入边界

- 复用 `call_runtime_agent`、`AnthropicProxy` 和 `AgentExecutionTransport`。
  真实模型仍由 `openai_clients.main` 或 HyTuner 提供。
- 原生 Anthropic 传输；不接受通过 `ANTHROPIC_API_PROTOCOL` 切到 Pi。
- 任务通过 UTF-8 stdin 交给 CLI，支持 text 或显式 stream-json 图文输入及超过操作系统单参数长度限制的输入；模型请求仍走同一代理。
- Hyness execution-result 映射为 HyCode 已有的 64/65/66 控制终态；
  保留原始 stdout、stderr、实际进程退出码和 execution termination。协议错误也保留 CLI 原始报错。
  错误码、subtype 和结果分类直接复用 HyCode；Runtime 不健康时由 Hyness 返回兼容的 `ENV_ERROR`。
  Span 中 `hycode.*` 记录兼容结果协议的退出来源、轮数和可评分标记，
  `hyness.process_exit_code` 保留实际进程退出码；诊断记录失败不改变任务结果。
- Gateway 默认使用已有 `hycode` scaffold policy，沿用 Hyness 的兼容工具调用及子轨迹关联。
  可通过现有 `openai_clients.main.config.client_args.scaffold_policy` 显式选择已经验收的策略；
  不按 Harness 名称自动切换。Hyness 承载 WorkBuddy 工具与直接运行原生 WorkBuddy 的身份协议不同。
- 使用现有 active timeout；训练暂停期间暂停执行时间预算，
  `disable_invoke_wait_timeout` 显式配置优先于训练默认值。
  Hyness 在自己的调用边界管理超时状态，正常结束、异常和取消都会释放本次状态。
- `max_turns`、`stop_on_tool_error`、MCP、JSON 配置补丁和 pre-start scripts
  沿用 HyCode 输入。tool-error stop 通过临时训练 overlay 生效，不修改安装的 App。
  Hyness 使用现有 MCP specs 转换器，在自身边界合并并检查冲突；HyCode 保持原有 MCP 处理。
  没有注入任务 MCP 时保留调用方的显式 MCP 参数。HyCode 侧仅保留此前的结果分类函数提取
  和启动脚本 transport 类型放宽，不为 Hyness 引入新的公共启动层或改变 HyCode 调用流程。
- HyTuner 只需增加 Hyness 的 SWE/防作弊名称识别并更新 Nexus gitlink；
  Parquet 中的 Nexus agent 配置也要按上面替换，gitlink 更新不会改写已有 Parquet。

## 验证与限制

回归检查实际安装的 CLI、代理、工具、完整 Prompt、提交协议、退出码和会话清理；
模型响应采用本地替身，不能据此推断随机采样得分或 GPU RL 训练行为相同。
OpenCC 和其他 App 使用各自配套的安装目录，不得混装不同引擎并把跳过的测试当作通过。

main 的 HyCode 工具 span 导入保留在 HyCode 启动流程中。Hyness 继续复用退出分类，
但不支持 HyCode 专用的本地 span 文件协议；`emit_spans` 默认 false、不会传入 runtime，
显式启用会在输入校验时被拒绝。模型请求轨迹仍由 Nexus 原有客户端和 tracker 记录。
这不是关闭现有 Langfuse 记录，也不声明新增 Hyness 工具 span 导入能力。

自定义 App 的动态能力、长时间 partial rollout、完整自动压缩和训练 token/logprob
链路仍需按实际场景验收。历史数据库、Parquet 和轨迹不随代码更新而改写。

运行回归：

```bash
uv run pytest tests/unit/extensions/agents/test_hyness.py \
  tests/unit/extensions/agents/test_hyness_protocol.py \
  tests/unit/extensions/agents/test_hycode.py \
  tests/unit/extensions/configs/test_hyness_configs.py -q

uv run python -m nexus.extensions.configs.validate --strict \
  agents/hyness.yaml \
  agents/hyness_opencc.yaml agents/hyness_mini_swe_agent.yaml \
  agents/hyness_swe_agent.yaml agents/hyness_workbuddy.yaml

# 安装包在仓库外；模型使用本地替身，不读取真实 LLM key。
HYNESS_TEST_PREFIX=/path/to/npm-prefix \
HYNESS_TEST_NODE=/path/to/node \
uv run pytest tests/integration/anthropic_proxy/test_hyness_apps.py -q

# 指向真实安装的两个命令；Hyness 配套 overlay 也从安装包取。
HYCODE_TEST_BINARY=/path/to/hycode/opencc-bin \
HYNESS_TEST_BINARY=/path/to/node_modules/.bin/opencc \
HYNESS_TEST_OVERLAY=/path/to/opencc/apps/opencc/overlays/training.yaml \
uv run pytest tests/integration/anthropic_proxy/test_hycode_gateway_e2e.py -q
```
