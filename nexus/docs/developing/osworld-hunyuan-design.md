# OSWorld：模型协议、执行与评分设计

本文描述 OSWorld 的模型 client、harness、runtime、平台配置与评分边界。
六模式命令和环境变量见 [OSWorld 运行指南](../how-to/osworld-runbook.md)。

## 模型与 harness

| 模式 | Agent | 本地 client |
|---|---|---|
| GPT bare | `cua_gpt_agent` | `openai_responses` |
| Claude bare | `cua_anthropic_agent` | `anthropic_messages` |
| GPT + Codex | `codex_native_bridge` | `openai_responses` |
| Claude + CC | `claude_code_native_bridge` | `anthropic_messages` |
| Hunyuan + Codex | `codex_native_bridge` | `infer_proxy` |
| Hunyuan + CC | `claude_code_native_bridge` | `infer_proxy` |

本地与平台使用相同的 task、harness 和评分流程。平台一站式模型服务可注入
`bridge_litellm`；协议选择依据实际 client 能力，而不是模型品牌或 CLI 模型别名。

裸 CUA 直接调用模型 client。CC/Codex 使用 `execution_profile: osworld`：

```text
普通 agent → 通用 native_bridge
显式 profile → profile.run(client) → OSWorld bridge adapter → profile.execute(...)
```

两个 agent 的 profile 工厂分别使用 registry。OSWorld task 模块负责注册
`osworld` policy，通用工厂不导入具体 task。

### 配置构建与执行入口

```text
YAML 导出 → single_config → build_item → 保存 YAML → run-single
Run Part  → single_config → build_item             → run-single
Generator → generate_configs → build_item          → run-batch
```

三者在相同有效 RunInput 下使用同一执行与评分机制，但环境取值时机不同。
完整 YAML 的字面量已经固化；Run Part 与 generator 在调用时构建配置。
共享构建函数不意味着不同环境下生成的 RunInput 天然逐字段一致。

### Anthropic 与 InferProxy 协议策略

`anthropic_messages` 的 `api_style` 决定上游 HTTP 编码，不选择 harness 或模型权重：

| 默认行为 | `messages` | `bedrock_invoke` |
|---|---|---|
| 路径 | `<base>/v1/messages` | `<base>/model/<model>/invoke` |
| 模型名 | 请求体 `model` | URL 路径 |
| 协议版本 | 请求头 `anthropic-version` | 请求体 `anthropic_version`，默认 `bedrock-2023-05-31` |
| 自动路由 provider | `anthropic` | `aws_third` |

显式 endpoint/provider 和已有 key 路由后缀优先；Bedrock 模式通过网关访问，
不是 AWS SDK/SigV4 或 Converse。

Hunyuan 本地使用标准注册名 `infer_proxy`，OSWorld bridge adapter 在宿主 handler
内部构造适配 client，集中处理 request ID、reasoning 拆分、工具结束原因与上下文预算。
原 client 对象、SDK 重试配置和注册项不被修改；普通调用保持通用行为。
适配后的 `max_retries` 表示总尝试次数，其 SDK 副本关闭内部重试。
`config.enforced_request_args` 高于运行时参数及 `extra_body`，
其中固定的输出预算不被自适应重试改写。
`claude_passthrough` 是 InferProxy 服务配置档位，不表示使用 Claude 模型。

### Codex 协议与预算

- client 提供 `create_responses()`：`api_protocol=osworld-responses`，原生请求直接传递。
- Chat client：`api_protocol=responses-chat`，由 `ResponsesProxy` 转换为 Chat Completion。
- 两条路径共用安装、指南、catalog、动作执行和结果分类。

Codex 默认版本为 `0.154.0`，CLI 模型标识为 `gpt-6-astra`。实际服务由模型 client
配置决定。默认 context window 为262144、输出预留64000、margin为4096，
压缩阈值为194048；这些值可通过 execution options 配置，并须匹配服务容量。

默认 reasoning effort/summary 为 `none`。Responses→Chat 转换时，effort为
`none` 表示不发送 Chat 的 `reasoning_effort` 字段。InferProxy 自身的推理参数
由 client 配置控制，不从 CLI 模型标识推导。
0.154.0 catalog 使用 `tool_mode=direct`、`use_responses_lite=false`，采用本地压缩；
显式 0.153.4 仍支持。

### Claude Code thinking 与压缩

Claude 与 Hunyuan 共用 CC 执行 profile。`thinking_preset=auto` 根据实际 client
选择 thinking 编码。压缩发生在 CC 会话内部；压缩次数、轨迹条目数和模型轮数
是不同概念。

```yaml
agent_name: claude_code_native_bridge
agent_config:
  execution_profile: osworld
  timeout: 5400
  max_turns: 200
  execution_options:
    guide_version: v2
    extra_env:
      CLAUDE_AUTOCOMPACT_PCT_OVERRIDE: "94"
```

本地 generator 将 `CC_AUTOCOMPACT_PCT_OVERRIDE` 写入
`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`；显式设置 `CC_AUTO_COMPACT_WINDOW` 时，将其写入
`CLAUDE_CODE_AUTO_COMPACT_WINDOW`。平台配置直接填写对应 `execution_options.extra_env`。
完整 YAML 不会根据 generator 环境变量重新构建配置。

窗口配置不会扩大模型服务容量。验证压缩需区分：参数传入、CLI触发压缩、
压缩后继续执行。`cc_env.json` 只记录白名单字段，不能仅凭其中缺少某字段判断未透传。

## 资源与安装

| 位置 | 职责 |
|---|---|
| `tasks/os_world/prompts/claude_code/` | CC 指南 |
| `tasks/os_world/assets/codex/` | Codex 指南、catalog、覆盖规则 |
| `tasks/os_world/execution/claude_code.py` | CC options、profile、执行保护和诊断 |
| `tasks/os_world/execution/codex_profile.py` | Codex options、资源解析、profile和结果分类 |
| `tasks/os_world/execution/codex.py` | Codex进程执行与证据采集 |
| `agents/codex/native_bridge/adapters/osworld.py` | Codex安装、Node bootstrap和环境校验 |

`guide_text` 和 `model_catalog_json` 的显式内容优先于对应版本/preset选择器；
空字符串也属于显式指南。无效显式内容会报错，不回退到内置资源。
CC与Codex的指南版本各自独立。

Codex安装使用可用的托管或系统Node；需要bootstrap时使用adapter持有的目录和下载URL。
`large_sandbox.py` 引用同一组Node常量，并就地归一化命令结果。

固定CC版本先检查PATH及专用安装目录中的CLI，版本精确匹配时复用，否则执行npm安装。
`latest` 使用通用查找与安装流程。MCP通过 `claude mcp add` 注册。

## 数据与平台配置

本地和平台均从 COS release 获取题目及 `task_files`。数据集索引用于选题，
题目包提供 `task.json`、setup资源和evaluator参考文件；`WEDATA_KEY` 用于认证。
单题导出应同时构建 `task_id` 与 `task_files`，避免二者不匹配。

平台一站式服务模式将模型配置与评估集配置分开：

- 模型来源为 `ONE_STOP_SERVICE`；SDK根据服务/模型参数构建 `openai_clients.main`。
- 评估集公共 `nexus_run_input` 不重复指定 `openai_clients`。
- 本地 `infer_proxy` 专属参数不能直接作为 `bridge_litellm` 参数使用。
- 平台与宿主、沙箱运行环境须使用相互匹配的包版本，`nexus_env.ref` 指定可构建的提交。

合并顺序为 Default → Customize → Dataset Version → User Override。
逐题 `domain/task_id/task_files` 位于 Customize，`wedata_key` 在公共配置中引用Rainbow。
平台服务模式的导出命令不支持同时指定 `--client-form-out`；外部URL表单属于另一种模型配置方式。

### COS 与平台字段边界

本地 URI/split 经 DatasetClient 枚举索引、构建每题 `task_files`；平台使用数据集行
`meta_data_.nexus_run_input.task_files`，不运行本地 generator。
两条路径最终由同一 task 读取 COS ZIP 中的 `task.json/setup/evaluator`。
`OSWORLD_DATASET_MANIFEST` 仅提供本地离线索引覆盖，不替代 COS 资源；
setup/evaluator 仍允许 best-effort 公网回退，因此这不是严格零公网策略。

远程 ZIP 的 `path/sha256/size/remote` 描述资源版本和范围读取所需元数据；
`size` 是读取 central directory 的必要条件。导出器拆分逐题配置、公共 RunInput
和模型表单，并检查字段是否漏分、重叠及逐题文件引用是否保持一致。
本地路径、cache、runner/processors/run_id 不属于题目数据，不随平台公共配置导出。

Default 来自 `nexus_task_rainbow_key`，Customize 来自数据集行或 `nexus_key_mapping`，
Dataset Version 来自评估集 `nexus_run_input`，User Override 来自 `nexus_run_input_`。
数据行的 `__fix__` 保护特定叶子路径，普通高优先级值不覆盖它；
`__override__` 可以整体替换。大沙盒题通过数据行的叶子覆盖保留资源要求。
Rainbow 的 `${oc.env:...}` 只在 Dataset Version 层解析，版本配置是浅合并。
它是插值取值表，不保证远端 Nexus host 的环境变量同时被设置。

模型表单重新构建 `openai_clients.main`；`bridge_litellm` 不套用 InferProxy 策略。
外部 URL 表单不能表达 `enforced_request_args`，导出器拒绝静默丢弃这些字段，
它们不能作为普通 `main_client_args` 搬运。

`OSWORLD_SESSION_NAMESPACE` 是可选的本地结果记录与缓存分组参数，不是运行前提。
最终 RunOutput 由 `-o` 或 `-d` 保存；任务运行产物统一通过 artifact 服务发布，
与输入 dataset 分离。

## 运行产物、凭证与传输

实现集中在 `tasks/os_world/data/artifacts.py`，通用 artifact SDK 不加入 OSWorld 分支。
每次运行创建独立暂存目录和 `/tasks/YYYY/MM/DD/osworld-run-<随机ID>/`，
避免重跑、并发任务覆盖同名产物。task 的 `finally` 在成功或异常路径上尝试发布；
已有评分和主异常优先于产物上传结果。

### 收集与数据路径

- 宿主截图、轨迹、评分结果和诊断打为 `host.tar.gz`，保留空日志与目录结构，
  跳过符号链接、设备、socket 等特殊文件。
- CC/Codex 删除会话文件前，系统 `/usr/bin/python3` 仅归档明确列出的日志、
  退出码与进程身份文件，不递归打包工作目录、配置或凭证。
  活进程的快照仅表示当时状态，不保证包含后续写入。
- 快照产生大小、MD5、SHA256 和缺失文件清单；宿主只接收这些元数据。
  `sandbox-artifacts.json` 保存快照时的声明，最终发布状态另存 `publication.json`。
- 评分后宿主通过 artifact SDK 创建上传会话；沙盒向预签名 URL 直接 PUT，
  支持单对象与服务端完成的分块上传，宿主收集 ETag 并确认 `ready`。
  沙盒不需要安装 Nexus worker。
- 直传失败时，经 `/file` 将同一快照流式拉回宿主，每次读取 1 MiB，
  严格验证大小和 SHA256 后原子落盘，再用同一 SDK 和幂等键上传。
  不关闭 TLS 校验、不修改签名域名、不重放模型或 GUI 动作。
  此兜底适用于跨地域内网路由或证书域名不匹配等网络差异。
- CLI stdout/stderr 仍按执行通路读取用于终态判断，不能以 artifact 上传结果替代；
  setup、输入 task_files 和 evaluator 文件读取也不依赖 artifact 服务。

最终只保留 `task_output.artifacts`，其中 `uris` 列出确认 `ready` 的 URI，
并保存完整状态及 `sandbox_direct/host_relay` 路径。
task 异常路径的诊断摘要由 OSWorld 输出 hook 补回（见“阶段输出契约”），
core context 与 runner 不新增通用槽位，其他任务行为不变。
部分结果的 score 为 null，不能参与普通得分；取消仍传播，
进程强杀、task 尚未进入或返回结果前的外层故障不在此保证范围内。
平台现有 trial_details.run_output_data 透传即可保留产物，无需修改平台解析器。
上传的 `run-result.json` 是 task 结果，
不是包含 runner processor 输出的完整 `-o` 文件。

### 凭证边界

PAT/COS 密钥不交给 guest helper。预签名上传信息通过私有目录内临时文件传入，
不进入常驻 argv 或最终输出；helper 退出即删除该配置。
凭证优先级为 task 显式参数、host 环境、host 登录文件。
`artifact_taihu_pat` 使用 `SecretStr`，排除 task 输入序列化，只传给本次 host client，
不通过全局环境传播，以免并发运行串用凭证。

这一保护不覆盖平台在 task 构造之前保存的原始 RunInput。
Rainbow 插值结果可能出现在平台请求/配置日志中；若要求 PAT 不进入 RunInput，
须由部署环境向实际 host 注入凭证。运行操作与安全注意事项见 runbook。

### 失败与时间预算

guest helper 有 45 秒进程级期限，快照阶段 host 总等待最多 60 秒；
中转下载最多 120 秒。发布首次失败后最多重试 3 次，间隔 2、4、8 秒，
每次发布尝试最多 300 秒，SDK 内部重试也受该外层预算约束。
重试复用目录与上传幂等键，宿主产物优先于沙盒产物发布。
超时不能停止打包线程，因此重试等待同一个打包任务，不并发写同一临时文件。

上传失败只设置 `upload_failed`，不替换已有分数或主异常；取消继续传播。
部分上传成功保留可下载 URI，本地暂存用于补传。实例释放前未取到的文件不能事后恢复。
诊断与快照均增加有界墙钟耗时，不是零成本操作，也不是评测成功的依据。

## Runtime 与 CDP

六模式使用 `runtime_provider_name: ags`。provider创建标准 `AGSOSWorldRuntime`；
OSWorld task在setup和agent执行前调用 `get_osworld_runtime` 装配 `OSWorldTaskRuntime`。

装配保留实例handle、端口映射、超时、共享状态和worker venv。初始化成功后同步更新
Context与provider缓存，使agent、tools和provider获取同一个adapter。
实例释放由原provider负责。初始化失败或取消时通过Context清理，保留主异常。
非OSWorld任务使用标准runtime。

任务runtime负责文件上传/读取、观测接口、截图校验和只读请求重试。
命令运行使用 `/execute`，控制器Python前缀以换行连接，支持函数定义等复合语句。

### 共享内存与桌面缓存

OSWorld task装配runtime时检查 `/dev/shm`：

- 容量不足512 MiB时执行 `mount -o remount,size=512M /dev/shm`，并验证容量。
- 容量足够则跳过，不缩小更大挂载，不改变Chrome参数或profile。
- guest sudo密码默认 `password`，可通过runtime配置 `sudo_password` 覆盖，仅通过stdin传入。
- 初始化失败明确报错并清理；操作仅针对本次实例，不修改共享Tool。

实例内存大小与 `/dev/shm` 挂载容量相互独立。Chrome页面崩溃或重新附加超时
可能与共享内存有关，但需结合进程、容量和CDP实际返回诊断。

setup的精确无参数 `update-desktop-database` 调用限定到当前HOME下的
`.local/share/applications`；显式路径和复合shell命令按声明执行。
控制面网络权限与guest文件权限不是同一层权限。

### CDP连接

镜像负责9222→1337转发，setup按题目声明启动Chrome。runtime支持
`image`、`auto`、`off`，均不注入guest代理。getter不隐式创建浏览器或替换profile。
题目声明的精确socat动作仅在确认对应镜像服务active时复用，避免重复监听。

直接可达时，Playwright使用provider endpoint与认证头。需要宿主代理时，
同步Playwright通过生命周期包装器使用通用 `CDPRelay`；连接超时为15秒。
特殊任务可显式使用独占9222的上下文，普通连接不启用该上下文。

setup成功、HTTP发现成功、WebSocket握手和CDP初始化必须分别检查。
连接失败应记录实际阶段与返回，不能通过启动另一份空白Chrome掩盖现场。
显式开启 CDP 的 Chrome 启动命令使用固定 loopback Origin 白名单，
不启用 `--remote-allow-origins=*`；此设置不改变外站访问权限。

### 宿主与沙盒网络

Runner 到模型、AGS、资源服务使用宿主网络，保留 host NO_PROXY；
不能仅按 CPU/CVM 标签推断需要哪个代理。
任务启用 proxy 后，Chrome 与 CC/Bash/WebFetch 使用 guest tinyproxy，
大小写 HTTP_PROXY、HTTPS_PROXY、ALL_PROXY 指向 `http://127.0.0.1:18888`。
上游代理凭据仅留在 tinyproxy 私有配置，不传给 CC。
NO_PROXY/no_proxy 保留已有排除项并包含 localhost、127.0.0.1、::1、[::1]，
确保本地 Bridge/CDP 不经过外部代理；未启用的任务不注入这些变量。

服务启动探针、工具继承环境与网站实际可用性是独立检查项。
非交互式 shell 的继承不能替代实际 CC WebFetch 或 Chrome 验证。

## 执行、诊断与评分

OSWorld CC adapter使用通用CC启动、轮询、输出解析及清理流程，
不额外施加内存或进程数限额。
正常退出后保留任务应用供evaluator检查。
prompt以私有文件保存并从stdin输入，不进入CC或父shell的argv，降低按命令文本匹配
进程时误杀harness的风险。

Bash PreToolUse为每次命令注入系统GNU grep函数，避免调用CC内嵌ugrep；
只更新工具输入，不覆盖权限判断。此Hook与可选防误杀Hook共用本次运行的settings。
它通过 `updatedInput` 定义并导出 `grep() { /usr/bin/grep "$@"; }`，
覆盖 shell snapshot 同名函数，保留其他工具输入；子 agent Bash 也使用同一设置。
配置字段见运行指南的“Harness 参数”。

CC 默认工作目录从沙盒用户 `$HOME` 获取，显式 working_dir 不自动回退或提权。
OSWorld adapter 不主动设置 `CLAUDE_ENABLE_STREAM_WATCHDOG`，调用方的 extra_env 优先；
该 CLI 变量与宿主的预算判断不是同一机制。

旁路诊断只观察，不参与退出分类：

- 记录轮询返回值/异常并写入 `cc_probes.jsonl`。
- CLI启动前记录进程、资源和 `osworld.service` 状态。
- 异常清理前记录日志尾部、进程树、内存/OOM计数、磁盘、服务状态和有界journal。
- 权限不足、工具缺失或采集超时记录为诊断错误，不覆盖任务主异常。

诊断有有界墙钟开销；资源快照最长等待6秒。诊断不能保证识别全部OOM victim或信号发送者，
也不构成进程隔离。
异常尾部每项最多 64 KiB、最多等待 3 秒，保存 `cc_diagnostics.json`；
它不是完整轨迹，也不能据此将未知退出码 `-1` 判为预算耗尽。

Codex prompt/配置以文件上传并校验哈希，`exec ... -` 从 stdin 读取 prompt。
轮询记录原始响应、wrapper/Codex PID、starttime 与 PGID；连续三次有效 GONE
才分类为 `process_missing`，读失败、无效身份/退出码和 PID 复用归为 probe。
清理前保存有界日志、最近事件和进程树至 `codex_diagnostics.json`。
无法确认身份时不按旧 PID 杀进程；错误保留 timed_out、failure_kind、failure_stage，
由 provider 的实例释放回收剩余资源。

### CC预算与失败分类

- `max_turns`：明确的CLI轮数上限终态（退出码0/1）映射为
  `success=true, exit_status=max_iterations`，进入评分，不设置虚假DONE marker。
- `agent_config.timeout`：host轮询使用对应墙钟预算。到期后确认沙箱可访问且CC仍在运行，
  采集证据，使用pidfd/starttime核验并停止本次执行树，读取输出后评分。
  记录 `cc_return_code=-1`、`budget_reason=cc_timeout` 及终止结果。
- 终止、取证和读取有额外有界耗时，Agent `timeout` 不是整题总耗时。
- GONE、API终态错误、无法确认终止或读取失败是执行错误，不能按接近截止时间推断为预算耗尽。
- 外部取消传播为取消。最终得分由现场evaluator计算，可为0或部分分。

预算参数在 Agent 入口和 OSWorld profile 内统一使用 `timeout`、`max_turns`。
显式顶层参数优先于 `execution_options`，未显式提供时不注入通用 Agent 的默认 timeout。
OSWorld 只在输入解析边界接受 `execution_options.cc_timeout`：同层 `timeout` 优先，
解析后的 options、bridge 编排和序列化只使用 `timeout`。
`budget_reason=cc_timeout` 是诊断标记，保持不变。参数统一不增加计时器或改变清理流程。

### 阶段输出契约

V1 在 `tasks/os_world/outputs.py` 定义和分类阶段输出，不改变通用 Agent、
runner、bridge 或其他任务的输出约定。CC/Codex profile 在返回或抛出前记录原始
CLI 证据；任务将其投影为标准输出。裸 CUA 使用本次 ExecutionContext 的委托
client 观察实际请求错误/finish_reason，仍由原 client 执行请求、重试、缓存与 tracing，
运行后恢复原 client；普通模型文本不作为错误证据。

| Agent 标准状态 | 语义码 | 是否进入 Judge |
|---|---|---|
| `finished` | 0 | 是 |
| `agent_timeout`、`max_iterations` | 0 | 是 |
| `context_length_exceeded`、`output_length_exceeded` | 0 | 是 |
| `runtime_error_by_agent` | 0 | 是，必须有明确 runtime reason |
| `inference_error`、`runtime_error_by_env`、`environment_error`、`unknown_error` | -1 | 否 |

`exit_code` 是语义码，不是 OS 进程码。原码保存在 `process_exit_code` 及 harness
专用字段；轨迹和 stderr 保存在独立 artifact，终止原因保留于摘要。`TASK_INFEASIBLE` 仍通过独立
`conclusion=infeasible` 映射到 evaluator 的 FAIL，不是新退出状态，也不改变评分算法。
上下文超长和输出截断不推断为不可完成；请求超时不等于 Agent 总预算耗尽。
只允许主 session 的非子 agent result 决定 CC 终态。
CC 在多次 `max_tokens` 响应后可能以
`API Error: Claude's response exceeded the <N> output token maximum. ...` 退出，
而非输出标准错误码。仅当该文本属于主会话的 error result 时，归为
`output_length_exceeded/0`，保留原 CLI 码和错误原文并继续评分。
普通 assistant 文本、子会话报错及已恢复的旧错误不触发该分类；清理失败仍优先判为环境故障。

Codex 的墙钟预算必须已有严格清理成功的证据才允许 `agent_timeout/0`；
未确认终止、探针失败和进程失踪仍属于环境故障。新增终止确认记录不增加终止动作，
不改变轮询、清理时机或实例释放职责。安装/版本检查失败归 `environment_error`。
非零 CLI 码（包括 137）本身不证明是 Agent 造成的 OOM；只有明确
`Process.OOMKilled`、`Process.NonZeroExit`、`Process.CrashLoopBackOff` reason
可归为 `runtime_error_by_agent`。缺少可判定证据时不猜测。

Judge 正常返回为 `finished/0`，评分异常为 `judge_error/-1`，保留实际异常；
禁用评分不生成有效评测。任务状态按 Agent 优先：
Agent -1 → `agent_error`；Agent 0 且 Judge -1 → `judge_error`；均为 0 → `success`。
失败保持原异常传播，并在 `ctx.tool_state['osworld_partial_task_output']` 暂存标准阶段输出和 error。
OSWorld 输出 hook 仅在 runner 失败且没有 task_output 时补回摘要，随后才归档 runner 详情；
不修改 runner 的 success/error，也不会把基础设施失败包装成模型零分。
任务开始清除旧摘要，hook 消费或关闭时清理暂存状态；core context 和 runner 不承担该协议。
Artifact 上传失败仍非致命，不覆盖阶段结果和分数。

六模式入口在`runner.hooks`显式配置 `osworld_output_contract` run hook；
该配置在runner初始化时解析，传入任务业务模型前从参数副本剥离，将任务的标准 Agent
摘要投影到顶层 `RunOutput.agent_output`，没有修改通用 runner 的 capture 机制。
task 先归档完整详情再返回摘要；runner 的 processor 结果产生更晚，由 hook
以同一次运行的 artifact 凭据另存独立归档。已发布的 task archive 不修改、不重放上传。
完整 transcript 和错误堆栈保存在归档，任务返回状态摘要。
精简保留 bridge 消费的 `token_counter` 和 `execution_timeout_tracker` 原始结果，
以及 `bridge_toolcall_anomaly` 除 `requests` 之外的全部聚合字段。
聚合值不经过文本摘要/截断，不重算、不填充缺失计数；完整 processor 结果仍先归档。
bridge 在 hook 之后提取 token_usage，清空统计会导致平台总体指标错误地显示为零。
`_default_chat_trace`和已启用的`_default_atif_trace`由hook原样保留，
交给bridge保存对话/ATIF文件后再删除，平台最终响应无需携带这些内部大字段。
直接本地运行、不经过bridge后处理时，RunOutput保留这两个已生成的轨迹字段。
结果采用字段白名单和有界文本；归档失败保留本地目录和失败标志，不改变评分状态。
任务本身始终生成标准嵌套输出和 `task_output.status`。

轮数预算优先于普通成功与结束标记：退出码 0/1，且 subtype 为
`error_max_turns/max_turns`，或 subtype 缺失/为 success 且 stderr 明确包含
`reached max turns`，才按最大轮数分类。正常退出不强制要求文本结束标记，
但进程消失、API 错误和其他非零退出不能据此改判成功。

主/子会话文本完整保留，仅主 session 的非子 agent 事件决定 final_answer 和 subtype。
主 CLI result 优先于后续 assistant 文本。OSWorld 可在主会话明确结束后启动不可重置的
后台等待期限；新主会话工作、矛盾终态或观测失败会撤销提前结束。
到期后需重新核验终态和进程清理结果，不能伪造 rc=0 或 CLI result。
详见 [主会话终态与后台任务](claude-code-parent-terminal.md)。

### OCR 评分资源

`compare_image_text` 在评分宿主使用 EasyOCR。缓存按 `EASYOCR_MODULE_PATH` /
`MODULE_PATH` 或 `~/.EasyOCR/model` 定位，校验 MD5 后复用；缺失或损坏时下载
默认英文 craft/english_g2 模型，每模型最多三次，每次下载与解压预算 600 秒。
下载在独立子进程完成，超时终止并回收进程；并发使用有界文件锁，校验后原子替换。
构造 Reader 时禁用隐式下载，保持识别模型、GPU 默认选择和匹配规则。
耗尽重试或缓存不可用抛 `EvaluatorInfrastructureError`，不静默转换为模型 0 分。
这是资源下载期限，不是 OCR 推理期限，也不重跑 agent。

### `/execute` HTTP deadline

`run_command(timeout=T)` 同时约束guest命令和host HTTP等待。host deadline为T与
正数provider request timeout的较小值，包含连接和响应体读取。

有明确deadline的 `/execute` 请求超时后最多重试两次，总计三次；每次重发相同payload，
保持相同deadline。每次60秒时总等待约180秒，外部取消可提前结束。
其他接口、无显式deadline的调用、连接重置、已返回HTTP/命令失败不进入这层重试。
退出码124按 `TimeLimitExceeded` 解析。

**HTTP超时不能证明guest命令没有执行。重发可能重复GUI/文件副作用，也不能保证前次命令
已经结束。** 三次耗尽后抛出明确超时，不能视为正常预算完成。

### 可选进程保护Hook

`execution_options.process_guard: true` 或generator变量
`OSWORLD_CC_PROCESS_GUARD=true` 启用Bash `PreToolUse` Hook，默认关闭。

Hook通过单次运行的 `--settings` 注入，不覆盖用户配置。它拒绝pkill/killall、
pgrep+kill及包含杀进程操作的 `/proc` 扫描，提示使用应用退出或身份核验后的子进程操作。
普通命令不会通过返回allow覆盖其他权限策略，ExitPlanMode不会自动批准。

安装、解析或审计失败明确阻止工具调用。`cc_process_guard.jsonl` 仅记录时间、
tool_use_id、命令哈希、判定和原因。该Hook是命令文本防护，不能阻止任意脚本内部
或其他工具执行kill，也不能修复OOM、磁盘不足和网站错误。启用会改变模型可执行动作，
需记录开关并验证评测口径。

## Bridge装配与依赖边界

统一协议入口为 `nexus/runtime/bridge/adapters/ags_osworld.py`。
profile通过 `claude_bridge_options` / `codex_bridge_options` 取参数，
由 `run(client)` 调用该adapter编排；setup、评分和桌面runtime属于task层。

| 通用能力 | Adapter职责 |
|---|---|
| Anthropic schema、工具/system转换、回复content/thinking转换 | PDF渲染、严格参数、tool-result顺序、usage和日志策略 |
| SSE生成、原生响应提取 | 路由装配、context错误映射和stream清理 |
| LLM handler参数工具、心跳、reasoning注入 | 原生Responses、compaction、缓存、诊断、取消和投递 |
| RSIO驱动与关闭 | 独立pending、回复分派、final通知、取消与emit预算 |
| Replay guard接口 | 取消tombstone、TTL/容量/字节预算、异常轻量化 |
| HostClient回调注册与keepalive调度 | 连接预算、代理选择、串行重连和session清理 |
| 标准bundle builder | 补充依赖、adapter启动器、缓存键、原子发布与并发构建合并 |

原生Responses请求不经过Chat转换，保留computer工具、reasoning、output及namespace语义。
流式接口在完整响应返回后生成SSE，等待期间保活，不是逐token透传。
Chat转换边界拒绝无法无损转换的原生computer工具，不在共享schema中全局禁止。
reasoning缓存深拷贝 `responses_output_items`，保护未确认的响应。

同ID请求去重、取消确认、迟到报文处理和有限投递属于模型请求协议；bridge层不重放GUI命令。
provider的 `/execute` 重发具有上一节所述的独立风险。

OSWorld编排在AGS上将bridge保留到实例释放；显式
`stop_bridge_after_run=True` 可强制停止。普通通用调用在执行后停止bridge。
`osworld-*` 或AGS上的 `responses-chat` 使用90秒就绪等待及一次初始连接；
`anthropic` 使用300秒就绪等待。宿主网络策略按协议显式选择。

沙箱启动器为 `python -m nexus.runtime.bridge.adapters.ags_osworld`，POSIX使用独立session。
普通启动器为 `python -m nexus.runtime.bridge`。bundle包含必要的replay、shell、PDF和psutil依赖。

**模块加载不等于策略启用。** 通用helper顶部导入 `AGSOSWorldBridgeAdapter`，
因此普通native调用也会加载统一adapter模块，但不调用OSWorld协议编排。
宿主handler与provider访问延迟加载，精简guest bundle不加载模型client、trace系统或task注册。
标准builder复制整个 `runtime/bridge`，普通bundle也包含adapter源码；
普通入口不因此启用OSWorld协议或安装额外PDF依赖。

## 验证范围

单元测试位于 `tests/unit/extensions/tasks/osworld/`，按bridge、task/setup、
dataset/prefetch、evaluator/getters、runtime/harness分组，不打入wheel。
`tests/unit/gui_agent/` 用于通用GUI demo测试。

回归包括配置解析、profile调用、真实FastAPI协议往返、取消/replay、reasoning缓存、
原生Responses payload、namespace及六模式冷启动。wheel验证从仓库加载测试，
从实际wheel加载模块和资源，再验证精简sandbox bundle。

配置传递、协议兼容、真实AGS启动、全量评测及分数可复现是不同验收层次。
无模型测试不能证明真实沙箱、网站、模型服务或全量评测的可用性。
