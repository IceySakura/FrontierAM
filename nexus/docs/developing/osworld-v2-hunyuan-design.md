# OSWorld V2：Hunyuan 与多 harness 适配设计

本文说明 `nexus/extensions/tasks/os_world_v2/` 的当前实现：输入资产、运行隔离、
模型协议、HITL、GitLab、独立评分及结果契约。运行命令与用户配置见
[OSWorld V2 runbook](../how-to/osworld-v2-runbook.md)。

## 1. 边界与组件

V2 独立注册 `os_world_v2` task 和 `osworld_v2` run-part；V1 的 task、生成器和题包
不是 V2 入口。V2 复用通用 agent/client/AGS 能力以及 task-owned OSWorld runtime，
特化逻辑放在 V2 adapter，而不修改官方评分函数或通用 provider 的任务语义。

支持的 benchmark 为 `osworld-v2.1`，官方源码固定到
`6f50176ede5a3009ea92849699524ca4903e7a5e`。源码版本与题包 manifest 共同校验；
模型选择不会改变 evaluator、reference 或题目版本。

| 组件 | 位置 / 职责 |
|---|---|
| 任务入口 | `task.py`：输入输出、阶段错误、产物及实例释放 |
| 六模式配置 | `generators/config.py`、`policy.py`：模型、agent、runtime 和任务配置 |
| 远端输入 | `data/index.py`、`data/bundle.py`：索引、校验、独立解包 |
| 官方运行 | `adapter/adapter.py`、`worker.py`：setup、agent、evaluate、多阶段/gate |
| host 到 guest | `adapter/transport.py`、`uploads.py`、`preparation.py` |
| CC / Codex | `adapter/claude_multiturn.py`、`codex.py`、`codex_hitl.py` |
| 评分隔离 | `adapter/cad_grader.py`、`pdf_grader.py` |
| 结果归档 | `adapter/outputs.py`、`artifacts.py`，共享 OSWorld artifact 服务 |

```text
本地 generator / 平台合并配置
  → RunInput（os_world_v2 + 远端 task_files）
  → 校验题包并解包到本次运行目录
  → AGS 桌面 + OSWorldTaskRuntime
  → host loopback transport + 独立官方 worker
  → 官方 setup → 动态指令 → agent → 官方 evaluate
                                ├─ 主模型 client
                                └─ 官方用户模拟器（有交互题）
  → 结果/轨迹/产物发布 → 本地连接清理 → 有界实例释放
```

## 2. 输入资产、依赖与可复现性

### 2.1 发布与运行分离

打包机读取固定源码、资产和外部依赖；运行机器通过 `nexus dataset` 读取私有COS题包。
`OSWORLD_V2_DATASET_URI` 必须含明确版本，不接受 `latest` 或 `dev`。
每行 `meta_data_.nexus_run_input` 携带：

- `task_config.task_id`：三位字符串；
- `task_config.domain`、`bundle_sha256`；
- `task_files`：remote引用、ZIP路径、size、SHA256。

公共评估集配置不覆盖每题身份。下载/解包检查大小、hash和manifest；
每次attempt独立解包，官方代码不能借共享目录修改另一题输入。
`$NEXUS_HOME/osworld-v2-inputs/archives/<sha256>` 是可复用压缩资产缓存，
本次官方缓存和工作副本位于产物目录的 `files/`。

007依赖资产包含 `task_007_hitl/`。050大归档通过 `archive_parts` 分片，
分别校验每片与重建后的完整SHA256。033的 `task033-py310-20261019`
dependency profile 固定LIBERO提交 `36cca4b242a957079753977935fccddd3cc185e0`，
使用锁定版本及文件hash的wheelhouse，通过 `pip --no-index --require-hashes`
安装。该profile明确标识依赖归档身份，不冒充其他归档，也不改官方task/评分源码hash。
发布时必须提供完整依赖，不能仅删除缺失标记。

### 2.2 宿主与guest依赖

官方worker使用宿主Python解释器。其依赖声明在
`nexus/extensions/tasks/os_world_v2/pyproject.toml` 的 `host` extra，
包括文档/表格、日历、headless OpenCV、音视频数值指标、辅助模型和OCR客户端。
AGS里安装一个Python包不能补齐宿主worker的import。

不把官方standalone runner的全部VM provider、GUI库和开发工具复制到host依赖。
`opencv-python-headless`避免引入无必要的host图形库。
安装easyocr不等于准备了模型权重；Python包与系统可执行文件分别预检。

FFmpeg/ffprobe需要独立提供。官方 `packages/osworld-ffmpeg` 是本地构建项目，
module安装器不会解析另一仓库的 `tool.uv.sources`。部署应安装审核后的固定wheel
或系统二进制，保留对应许可证。`imageio-ffmpeg`不提供ffprobe。
无法改变平台host系统包时，025/103/104使用下文的独立AGS grader；
这不消除其他题目的host依赖。

开发预检：

```bash
uv run --no-sync python -m nexus.extensions.tasks.os_world_v2.scripts.check_host_dependencies \
  --repo /path/to/verified-source --report /tmp/osworld-v2-import-audit.json
```

该检查逐题导入并初始化Controller，阻断外部连接，不运行setup/evaluate或模型；
通过不代表网站、OCR权重或媒体二进制可用。

## 3. 配置与平台边界

### 3.1 配置来源

本地必须显式加载可信 `.env`。generator生成RunInput时将多数值固化，只有保留的
`${oc.env:...}` 在运行时解析；修改环境后需要重新导出配置。
官方worker使用独立HOME且不自动加载dotenv，不继承任意宿主环境。
辅助模型/GitLab通过 `task_config.v21.environment` 显式传入，本地对应
`OSWORLD_V21_ENV_JSON`，键限制为 `OSWORLD_`、`GITLAB_`、`WEBSITE_` 前缀。

平台配置按职责分开：

| 位置 | 内容 |
|---|---|
| 数据集JSONL | 每题身份、bundle hash、remote task_files |
| 评估集 `nexus_run_input` | task、agent、runtime、hooks、官方worker配置 |
| Rainbow / worker环境 | AGS、输入数据、artifact、GitLab及辅助模型凭据 |
| 创建任务时的模型表单 | 主模型路由、鉴权、采样与请求预算 |

平台四层合并优先级为 `user > dataset_version > customize > default`；
评估集公共参数高于数据行。切换harness时 `agent_config__override__` 用于完整替换，
防止残留另一个agent的字段。本地 `openai_clients.main` 不直接复制到公共评估集，
平台根据任务的模型来源构造主client。

### 3.2 Hunyuan与Anthropic主模型

本地Hunyuan使用 `infer_proxy`；平台一站式服务通常构造 `bridge_litellm`，
包含Polaris路由和SessionID。两者不是同一部署，修改Nexus代码ref不会更新平台
独立部署的nexus-bridge。模型client的实际model与CC/Codex显示的CLI/catalog身份分开。

Anthropic主模型使用 `anthropic_messages`。AIHub Bedrock passthrough配置配对为
`api_style=bedrock_invoke`、`provider=aws_third`；Messages网关使用与之匹配的
`api_style=messages`。模型ID、URL、provider必须由同一服务契约确定。
平台 `api_key` 位于 `main_client_args` 同级。

`thinking_preset=auto` 是协议保留方式：原生Anthropic选择 `bedrock-claude`，
Chat转换选择 `hunyuan`；它不是high/max思考强度。实际强度由请求参数与client覆盖决定。

### 3.3 文件、压缩与请求预算

非原生Anthropic的V2 CC路径选择 `osworld-anthropic`，
将支持范围内的PDF转换成有界图片，再以 `image_url` 发给Chat后端；
原生Anthropic保留原生文档。任意格式或不支持视觉的模型不因此获得文件能力。

Hunyuan + CC默认注入窗口200000、压缩百分比80、最大输出16384；
`CC_EXTRA_ENV_JSON` 的同名guest变量优先。压缩不是硬截断，大工具结果仍可能越界。
验收检查 `compact_boundary`、压缩前后token和后续真实工具执行。

平台bridge可用 `main_client_args.model_call_total_timeout_s` 约束单次模型调用的
请求、响应消费及重试退避总时长；`timeout`是单次请求时限，
`max_retries`在该bridge是外层总尝试次数。确定性上下文/不支持的file等错误应快速返回，
其他瞬态错误有界重试。使用这些配置前确认部署的bridge支持它们。
它们既不是整个agent预算，也不限制新发起的后续调用。

## 4. AGS runtime与传输

### 4.1 资源及服务契约

task调用共享 `get_osworld_runtime()` 装配 `OSWorldTaskRuntime`，provider负责实例释放。
预检将 `/dev/shm` 至少扩到512MiB；048需要至少1GiB，已有更大挂载不缩小。
容量、磁盘空间和进程内存是不同资源。

当前桌面资源提示为总内存64GB、初始空闲磁盘约37GB，写入bare system prompt、
CC append-system-prompt和Codex developer_instructions。这是镜像资源规划，
不是实时余量；长视频、模型下载、DinD数据和CLI日志会消耗同一可写盘。
总盘容量应通过新实例返回值及guest `df`/statvfs验证，不能把CPU/内存扩容当作磁盘扩容。
在已配置的OSWorld custom tool中，省略 `CustomConfiguration.Resources.Storage`
的新实例返回51200Mi；这不是跨地域/tool类型的容量保证。更新tool时提交完整可写配置，
保留CPU/Memory、镜像、启动命令、环境、端口和探针，去掉只读ImageDigest。
不要将实例返回的Storage值直接当作UpdateTool可接受的枚举。新配置不对已有实例在线扩容。

AGS tool需要OSWorld HTTP、CDP及对应harness端口。`cdp_proxy.service`由镜像提供，
task验证其契约；V2不在每次运行另行安装guest CDP代理。
CC/Codex按普通用户启动，保留PID/进程身份与退出码，不新增任务agent的memory/pids cgroup限额。
`nohup`/`setsid`不保证抵御 `osworld.service` 的整组清理。

### 4.2 host loopback与网站访问

`Transport`为每题选择 `127.x.x.x`，监听5000、9222、8080、8000、5910、3000，
将官方worker使用的固定端口映射到同一AGS认证网关。
20次地址分配失败会抛 `Cannot allocate V2.1 loopback transport`；
宿主 `0.0.0.0` 通配监听可能让所有候选地址冲突，这与AGS是否暴露端口不同。

网站请求仅允许题目声明的域名及配置的GitLab精确origin。
小请求走stdin，超过32KiB改为上传私有guest暂存文件，避免shell参数长度限制。
GET/HEAD可有界重试；状态写入不在结果未知时盲目重放。
082的host loopback:3000映射到guest:3000，tool必须暴露该端口且应用已启动。

大于16MiB的multipart upload使用8MiB块，guest组装并校验大小/SHA256，
再提交到原始本地 `/setup/upload`。总量上限1GiB；保留官方目的权限和HTTP状态。
最终提交先分离启动再轮询，避免嵌套请求被guest控制服务阻塞；
仅不可变块允许安全重试，最终上传不按未知结果重复执行。

### 4.3 CDP、截图与控制读取

V1/V2共享 `CDPRelay`。每次握手先发现上游endpoint并连接WebSocket，再完成下游握手；
最多3次、总预算30秒、单次最多10秒、退避1/2秒。瞬态状态与旧WebSocket端点404
可重新发现，401/403及无效协议配置快速失败。连接建立后不透明重连、不重放CDP命令。
V2启用3秒ping/pong，避免镜像TCP代理的空闲超时中断连接。

桌面 `/screenshot` 最多3次，每次HTTP和host等待分别为30/60/120秒，
间隔0.5/1秒；鉴权失败不重试、取消传播。该预算不代表完整工具链所有后续请求的总时限。

V2 CC的PID/退出码及HITL目录读取显式接入共享只读请求策略：
瞬态HTTP/连接错误最多5次，间隔0.5/1/1.5/2秒，读取总deadline包含退避。
启动、停止、回答写入及模型命令不进入这条幂等重试路径。
Codex的进程身份探针由其adapter管理，不能假定与CC读取策略完全相同。

## 5. Harness与HITL

### 5.1 CC

V2使用 `claude_code_multiturn` + `execution_profile=osworld_v2`。
独立guest stdlib supervisor完成stream-json初始化并管理stdin/stdout；
host通过既有文件hook将AskUserQuestion等交互交给官方simulator。
缺失答案、simulator错误或未处理的权限请求不转换成自动批准。
无simulator题禁用用户问答/计划工具，并以拒绝hook兜底，继续自主执行。

外层 `agent_config.max_turns` 是用户轮次；
`execution_options.max_turns` 是每次CC CLI的 `--max-turns`。
父会话预算不是父子agent和压缩请求合计预算。轨迹事件、模型调用、工具数与用户轮次
必须分别统计。父会话结果决定终态；TASK_DONE后的后台等待有独立宽限和进程确认流程。
V2的完成标记为TASK_DONE。

保留GNU grep、可选Bash防误杀、私有prompt、诊断、轮询与清理。
PTY版 `claude_code_interactive` 不在本模块的AGS支持入口内。

### 5.2 Codex

V2使用 `codex_native_bridge` + `execution_profile=osworld_v2`：

| 参数 | 语义 |
|---|---|
| `session_mode=auto` | 有simulator走app-server，其余走exec |
| `session_mode=app_server` | 显式结构化会话 |
| `session_mode=exec` | 无HITL的执行 |
| `timeout` | 执行预算，app-server包含交互等待 |
| `max_turns` | app-server用户轮次，包含初始prompt，不是模型/tool累计次数 |
| `max_user_requests` | `requestUserInput` RPC数；一次可有多问题 |
| `hook_input_timeout` | 单次交互等待，受剩余执行预算约束 |

exec不使用用户轮次/请求数限制。guide和model catalog必须显式提供，
或分别由 `guide_version`、`model_catalog_preset` 生成。

app-server复用通用JSON-RPC、thread/turn和event组件，不建立另一条PTY执行链。
答案按question ID返回同一thread/turn；任务回调结束已完成轮次，不虚构新prompt。
权限/MCP elicitation不是用户模拟问题，不自动批准。
guest需要pydantic2时安装到独立 `/tmp/nexus-codex-hitl-pydeps`，不替换系统包。
`codex_interactive` 的通用PTY入口与该V2 profile不是同一执行路径。

### 5.3 生命周期及结果独立性

普通session目录清理失败不能改写已得到的退出码或阻断评分；
Codex记录清理诊断，保留成功或原始执行错误。通用 `codex_interactive` 关闭失败
同样保留执行结果及stderr诊断。取消、预算耗尽、进程终止未确认单独处理，
不能把未知执行状态当作成功。完成CLI执行并不等于官方评分满分。

## 6. GitLab与题目准备

### 6.1 GitLab部署和访问

026/041需要包含GitLab、Runner和Pages的AGS镜像。服务由
`osworld-gitlab.service`初始化；管理员token运行时注入，不写入镜像层。
Python的gitlab客户端包不能代替服务部署。

三个配置必须一致：
tool注入的token、私有 `.env`、平台Rainbow/worker的 `v21.environment`。
guest浏览器通过mock反向代理访问 `http://localhost`；
host worker的相同URL通过V2精确origin转发到guest。
Pages使用 `http://pages.localhost`。远端GitLab保持直接访问，
未声明的localhost端口不自动开放，也无需为该路径暴露GitLab的80/443到AGS公网。

仅运行setup的本地GitLab题先等待 `gitlab_initial_wait`（默认600秒），
再每5秒检查service active/success、`/run/osworld-gitlab-ready`、
登录页HTTP200及GitLab标识。`gitlab_ready_timeout`默认1800秒，包含初始等待；
服务失败或缺失不跳过。远端GitLab及其他题不执行这一等待。
这与官方setup完成后的 `setup_wait_time` 不同。

041的动态指令以官方task字典中的运行时值为准，setup后同步到agent，
保证生成账号凭据进入实际指令。管理员token不是agent的登录密码。
验收应分别检查API鉴权、Runner注册、CI job、Pages发布页及浏览器访问；
只完成setup不代表CI/Pages或内容评分成功。

### 6.2 专用准备适配

| 题目 | 当前处理 | 不改变的边界 |
|---|---|---|
| 048 | 等待graphical.target、至少1GiB shm；精确匹配的防火墙REJECT规则转为DROP | 保留游戏UID allow规则、sudo限制、Unix socket及完整性验证；DROP改变拒绝时延，不开放访问 |
| 061 | 长Flatpak准备命令分离启动，以短只读请求轮询 | 不重复启动安装；原stdin、权限、deadline及退出码保留 |
| 033 | 锁定依赖包、分片校验、长安装分离轮询 | 官方代码、离线hash安装与评分保持固定 |
| 074 | CDP keepalive；评分截图独立诊断/有限恢复 | 不重启agent浏览器、不以agent截图替代裁判截图 |
| 082 | 已有3000端口转发 | 官方应用启动及状态初始化仍须成功 |

## 7. 独立评分

### 7.1 CAD：103/104

每次评分新申请独立AGS，不复用agent实例。host冻结提交STEP、记录SHA256，
与验证过的官方代码及reference一起传入grader。只读输入由root拥有，
评分以非特权用户和无网络namespace执行；grader凭据不注入agent。
返回标记 `evaluation_execution=ags_isolated_frozen_step`。

默认 `backend=ags`、timeout900秒，范围300–3600秒。未指定专用tool时复用主tool模板，
仍创建新实例。`install_dependencies=false`要求tool预装FreeCAD/numpy/scipy并支持
OSWorld `/execute`、`/setup/upload`、sudo和网络namespace。
独立实例的申请、准备、上传和评分均计入外层预算。

STEP加载优先原官方提取器，包括正常判零。兼容路径仅在独立受限探针确认
FreeCAD0.19/OCCT7.5.1的候选 `Import.insert` 原生SIGSEGV、
且可信reference仍能正常导入时启用。候选改用 `Part.Shape.read` 承载完整shape，
不融合、不修复、不丢弃实体、不改变位置；reference仍使用官方加载。
单实体门槛、最大实体测量、曲面分类、权重和容差保持官方规则。

超时、SIGKILL、一般Python错误或缺依赖不能授权回退或变成0分。
所有尝试共享240秒墙钟预算；每进程CPU180秒、地址空间4GiB、文件256MiB、FD128。
新增探针的逐进程CPU额度不等于一次调用的累计CPU额度。
结果记录import_path、版本、loader hash及每次stdout/stderr、退出码、signal。
一致性测试同时对比实体数、体积、位置、曲面及关键评分指标；
原官方崩溃输入没有可宣称一致的官方成功分数。

### 7.2 PDF/OCR：025

默认 `pdf_grader.backend=ags`、timeout1800秒；tool依次选择PDF专用、CAD专用、
主task模板。每次独立创建实例。官方完整性检查仍作用于原任务实例；
提交PDF冻结后，可信GT和原评分代码在独立grader运行。

PDF环境与CAD Python隔离，依赖Tesseract及eng数据、PyMuPDF1.26.7、
NumPy2.2.6、Pillow11.3.0、pypdf6.1.3。部署文件位于模块内
`deployment/pdf_grader/`；`Dockerfile.combined`叠加CAD与PDF，
PDF-only镜像不能直接承接103/104。

评分进程非特权、无网络；CPU600秒、地址空间4GiB、文件128MiB、FD128，
单PDF最大100MiB。官方generate_text经受限RPC返回host worker，
辅助模型endpoint与密钥不传入grader。请求、响应校验及评分公式不变。
缺失提交按官方规则处理；`ocr_no_text`、低置信度和调用失败不能伪装正常0分。

### 7.3 074独立截图与辅助评分

评分截图使用临时Chrome profile、原headless参数和1920×1080校验。
仅拦截官方evaluator内精确匹配的三条截图命令。明确失败且进程清理确认后，
最多3次、总150秒，每次新profile；权限、磁盘、sandbox配置错误不反复重试。
HTTP响应丢失或清理不确定时不重放，避免重复启动未知状态的Chrome。
成功要求退出0及正确PNG；DBus stderr本身不是失败标准。

`aws_third`是V2 worker注册的AIHub Anthropic Bedrock Invoke backend，
复用 `AnthropicMessagesClient`、Bearer认证，与AWS SDK凭据方式分开。
辅助裁判和simulator独立于主模型。每次HTTP默认120秒、最大300秒，
client单次尝试，外层最多3次；503等瞬态错误指数退避加抖动，上限10秒，
400/401/403快速返回。空/截断响应不作为有效评分。
078记录模型调用和图片hash诊断，不修改视觉裁判的分数规则。

## 8. 输出、轨迹和释放

`runner.hooks.output_contract`选择 `osworld_v2_output_contract`。
失败摘要先放task状态，再由hook补入结果和归档；没有hook时异常可能返回空task_output。
hook保留token_counter及bridge尚需的 `_default_chat_trace`、
启用时的 `_default_atif_trace`，由bridge保存后移除；本地直接运行保留这些内部字段。

| 字段 | 含义 |
|---|---|
| `task_output.status` | success / agent_error / judge_error |
| `agent_output.exit_code` | 语义码；原CLI码见process_exit_code、cc_return_code、codex_return_code |
| `judge_output` | 独立评分状态，不能覆盖已成功agent输出 |
| `evaluator_result.scored` | 区分setup-only与真实评分 |
| `artifacts.runner_details` | 完整runner/processor归档 |
| `artifacts.release` | 实例释放状态，不改变分数 |

预算耗尽及上下文/输出长度限制可能进入评分；控制链路故障不能伪装正常零分。
每阶段保存独立agent输出与评分，仅最终阶段摘要进入主响应。
归档只包含显式收集的文件，不是guest全盘备份。

输入数据用WEDATA_KEY，输出artifact用ARTIFACT_TAIHU_PAT，两者不能互换。
CC/Codex清理前快照诊断文件，优先guest直传COS，失败可由host转发；
发布失败保留本地目录和原始任务结果。

完成评分、发布和本地连接清理后，主实例发起一次Stop请求，最多等待5秒：
`stop_accepted` / `stop_failed` / `stop_pending`。外部复用实例记录detached。
accepted只表示停止请求接受，不等于STOPPED；实例ID与TTL保留以便核查。
主实例释放记录在归档发布之后生成，写入最终响应和本地release-status.json，
不回写已发布归档。独立grader先取回诊断再释放，其状态随主host产物归档。

## 9. 诊断与验收

| 文件 | 用途 |
|---|---|
| `worker-phases.jsonl`、`official-cache/worker-stderr.log` | 官方操作阶段与脱敏stderr |
| `official-cache/cdp-relay.jsonl` | discovery/upstream_ws/forwarding、状态及endpoint hash |
| `phase-*/control-reads.jsonl` | CC只读类别、命令hash、最终状态和耗时 |
| `phase-*/cc_stream.jsonl`、`cc_diagnostics.json` | CC父子消息、压缩、进程与控制故障 |
| `phase-*/codex_stream.jsonl`、`codex_diagnostics.json` | Codex执行事件、原退出码与清理诊断 |
| `codex_appserver_events.jsonl`、`session-result.json` | Codex结构化会话、thread/turn和终态 |
| `codex_interactions.jsonl`、`codex_protocol_manifest.json` | 模拟器问答与协议源hash |
| `official-cache/cad-*/` | STEP、manifest、import路径、完整指标及进程诊断 |
| `official-cache/pdf-025-*/` | 冻结PDF、字段/OCR、模型RPC和评分诊断 |
| `official-cache/task074-capture.jsonl` | PNG、Chrome进程、资源及清理状态 |
| `official-cache/auxiliary-model-calls.jsonl`、`evaluation-diagnostics.jsonl` | 辅助请求尝试、状态、图片hash及078评分 |

轨迹和PDF字段可能含任务内容、账号或cookie，按私有数据处理。原生退出码缺失、
HTTP502和OOM不是同一诊断；没有内核/资源证据时保留归因边界。

```bash
uv run --no-sync pytest tests/unit/extensions/tasks/osworld \
  tests/unit/extensions/tasks/osworld_v2 \
  tests/unit/extensions/agents/test_codex* -q
```

验收分层：配置/schema → 输入与依赖 → setup-only → 真实主模型/HITL →
评分及产物 → 多题并发。静态测试、模拟器smoke、完整benchmark评分各自独立，
不能以一次链路通过宣称108题或六种模型服务全部通过。

单独回放CAD评分时使用真实提交STEP，输出到新的目录：

```bash
uv run --no-sync python -m nexus.extensions.tasks.os_world_v2.scripts.grade_cad_submission \
  --env nexus/extensions/tasks/os_world_v2/.env \
  --repo /path/to/verified-source --assets /path/to/verified-assets \
  --task-id 104 --submission /path/to/submission.step \
  --tool-id "$GRADER_TOOL_ID" --preinstalled --output /tmp/cad-replay-104
```

该开发命令需要本地可信源码/资产用于校验，会申请真实grader实例。
不得用reference代替缺失的agent提交；`reference_leak`是必须保留的负例。
