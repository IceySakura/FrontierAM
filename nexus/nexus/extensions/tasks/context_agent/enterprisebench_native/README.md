# EnterpriseBench Native

`enterprisebench_native` 是 SyntaraOffice EnterpriseBench 的 Nexus 原生编排实现。
它直接使用 Nexus Runtime API 完成沙盒初始化、Agent 执行、Verifier 调度、结果解析和
artifact 持久化。运行该任务的 Host 环境不需要安装或导入 `harbor`，实现也不依赖
`harbor_bridge`。

为尽量保持与原始 EnterpriseBench 评分结果可比，Runtime 镜像需预装固定
版本的 `harbor-rewardkit==0.1.7` 和 `mcp<2`，再由 RewardKit 调度 OpenCode
Agentic Judge。这里保留的是 RewardKit 的评分协议，不是 Harbor 的 Environment、
Task 或 Verifier 实现。因此“Native”不表示 Runtime 内完全没有 Harbor 命名的包。

## HyTuner RL 分支兼容

本实现使用 `context_agent_general`，保留通用 `rm_system` / `rubric_accuracy`
后端、`max_iterations` 兼容字段和 20 轮默认预算。 最后一轮强制回答仍使用原工具
schema，以保持 HyTuner 的 append-only token 前缀；不会切换为 `tools=None`。

EnterpriseBench 的 `reward_mode` 仍支持 `sample_acc` 和 `rubric_acc`。输入超限为
`TRUNCATION`，单次输出达到上限的 Agent `exit_status` 为 `OUTPUT_LENGTH_EXCEEDED`；
两者在 task output 和 HyTuner 的 `termination_reason` 中统一为 `TRUNCATION`，跳过
Judge、给零分，并标记为 overlong。这里未增加大工具返回的自动分页或截断。

Context Agent 遇到 `ToolNameResolutionError` 后仍向模型返回错误并继续执行。
Task 配置 `zero_score_on_tool_name_resolution_error`（默认 `false`）决定是否将
含此错误的已评分轨迹最终 `score` 归零。EnterpriseBench 的 `reward` 和
`rewards["reward"]` 同步更新，`raw_reward`、`sample_accuracy`、`rubric_accuracy`
及 criteria 保留原始 Judge 结果。不修改结束原因，也不额外 mask。计数覆盖直接调用及
`tool_call` / `tool_describe` / `tool_search select:` 内部捕获的工具名解析错误，
不包含普通参数校验失败。`reward_extra_info` 中保存
`tool_name_resolution_error_count`、`tool_name_resolution_zeroed` 和
`tool_name_resolution_original_score`。Task 通过 `extra_fields` 传递监控信息；runner
只透传，HyTuner 只消费 Task 分数和汇总指标，不再实现这项扣分。
Judge/环境失败仍走原有异常与 mask 策略。关闭开关也照常统计错误次数。
错误计数仅存于 `extra_fields.reward_extra_info`，不再提供顶层同名副本。
HyTuner 对没有统计结果的轨迹保留缺失标记，不按零错误处理；`None`、NaN、Inf、
负数及非整数计数均不参与错误次数均值和出错轨迹比例的分母。
`traj/{train,val-aux}/tool_name_resolution_error/valid_count`、`missing_count`、
`missing_ratio` 分别记录有效统计轨迹数、缺失轨迹数和缺失比例。全缺失时仅输出
计数和缺失指标，不输出未定义的 `count/mean` 与 `trajectory_ratio`。

可直接设置布尔值，也可在数据的 Nexus 配置中使用已有环境变量插值：

```json
{
  "task_config": {
    "zero_score_on_tool_name_resolution_error": "${oc.env:ZERO_SCORE_ON_TOOL_NAME_RESOLUTION_ERROR,false}"
  }
}
```

运行环境将 `ZERO_SCORE_ON_TOOL_NAME_RESOLUTION_ERROR` 设为 `true` 开启，`false` 或
未设置时关闭；变量名由配置选择，Task 不额外读取固定环境变量。通用
`context_agent_general_task` 支持同一开关。已有数据不会自动加入该配置。

日志指标 `traj/train/tool_name_resolution_error/count/sum` 为当前训练 batch 的错误总数，
`count/mean` 为每轨迹平均次数，`trajectory_count` / `trajectory_ratio` 为出现错误的
轨迹数和比例。验证对应前缀为 `traj/val-aux`，统计整轮验证；随现有 logger 写入 SwanLab。

ContextAgent 任务通过 task span 的 `agent.termination_reason` 上报分类，普通和 Gateway
两条路径由原有 HyTuner runner 读取。 Judge 异常映射为
`REWARD_ERROR`、Agent 阶段超时映射为 `TASK_TIMEOUT`、环境初始化错误映射为
`INITIALIZATION_ERROR`、Enterprise 环境启动超时和 Runtime RPC 超时映射为 `ENV_TIMEOUT`。 这些原因使用
训练配置原有的 mask 策略，不把基础设施失败作为正常任务零分。

默认 `judge_execution_mode: host`；合入代码不会自动启用 runtime-direct，也不会
修改训练数据或网络策略。部署前须检查镜像中的 `setpriv`、sandbox 用户、服务身份
和 Judge 预装依赖；启用 runtime-direct 时另需检查出站白名单及 `cryptography`。

## 配置来源

Native Dataset builder 在构建数据集时读取原始任务文件，并将其中与 Case 相关的
信息转换成 Nexus Dataset item：

- `task.toml` 提供任务名称/描述、Verifier 环境变量、资源提示和 MCP 服务声明；
- Dockerfile 和 docker-compose 提供基础镜像、World 启动参数和基准时间；
- `instruction.md` 的内容写入 `task_config.instruction`；
- `tests/judge.toml` 中的 rubric、harness、prompt 文件名、模式和评分规则写入
  `task_config.judge_config`，但其中的 model 和 timeout 不进入 Dataset item。

模型选择和所有运行超时属于一次评测的运行策略，统一由运行 YAML 提供。

运行时不会解析 `task.toml` 或原始 `tests/judge.toml`。生成的 Native Taskfiles 会
删除以下已转换或由 Nexus 接管的文件，避免形成两套运行配置：

```text
task.toml
instruction.md
tests/judge.toml
tests/opencode.json
tests/test.sh
tests/rewardkit_opencode_agent.py
```

Taskfiles 仍保留 Runtime 需要的 payload，例如：

```text
environment/initial_external_services/
environment/initial_workspace/
tests/prompt_template.md
solution/                         # 若原任务提供
```

当前生成器还会保留 `environment/Dockerfile`、`environment/docker-compose.yaml`、
`environment/agent-requirements.txt` 和原始 `fetch_grading_context.py` 等源文件，但
Native rollout 不把前三者作为配置来源。`tests/fetch_grading_context.py` 则属于 Case
自己的 Verifier payload：它随 Taskfiles 上传到 `/tests` 并直接执行，不由 Native
模块提供或覆盖。当前 300 条任务中的该脚本虽然内容相同，但它依赖具体 World 镜像中
`/app/packages/core` 和 `/data/<service>` 的布局，因此仍由每条任务携带更符合其语义
归属，也允许后续数据版本按 Case 调整实现。

## Judge

每个 Case 的 Dataset item 只保存评测语义，不保存实际 Judge 模型：

```yaml
task_config:
  judge_config:
    judge:
      judge: opencode
      prompt_template: prompt_template.md
      mode: batched
    criteria:
      - name: c001
        description: ...
        type: binary
    scoring:
      aggregation: weighted_mean
      threshold: 0.5
```

真正使用的 Judge 模型由运行 YAML 中的 `openai_clients.judge` 唯一决定：

```yaml
openai_clients:
  judge:
    name: context_agent_openai_responses
    config:
      request_args:
        model: gpt-5.6-sol
```

Native Task 会在 Runtime 中动态写入：

```text
/tests/judge.toml
/tests/opencode.json
/tests/test.sh
/tests/rewardkit_opencode_agent.py
```

其中 `prompt_template.md` 和 `fetch_grading_context.py` 来自该 Case 的 Taskfiles。
Native Task 会在上传 `prompt_template.md` 时加入 Nexus Agent 最终回答和轨迹文件位置
的说明，但会原样保留并执行 Case 自带的 `fetch_grading_context.py`。上面列出的 Judge
控制文件则由 `enterprisebench_native` 模块生成或提供。

`rewardkit_opencode_agent.py` 会为每次 OpenCode 调用创建独立的临时
`XDG_DATA_HOME`。这样同一 Runtime 内并发 Judge 或 RewardKit 重试不会共享 OpenCode
的 SQLite 会话数据库，从而避免 `database is locked` 冲突。

对于在 Responses `content[].reasoning_text` 中返回可见推理、但不填充 `summary` 的
Judge（例如 DSV4 Flash），Native Responses 转换器会把该文本镜像到 OpenCode 的
reasoning summary。这样完整 reasoning 会出现在 OpenCode JSONL 以及最终保存的
`judge_trajectories.json` 中，同时原始 reasoning item 和 encrypted content 仍用于后续
轮次回放。

工具返回中的 `input_image` / `input_file` 保持结构化内容块，包括图片 detail、
文件名及 Base64/URL/文件 ID；Host Responses client 直接将它们作为多模态工具结果发送，
不会把媒体编码序列化成普通文本。普通消息里的图片和 PDF 也遵循同样规则。
工具调用轮次仍从缓存回放原始 provider output；没有工具调用的 assistant 文本
（包括 OpenCode 压缩摘要）保留在原位置，避免压缩后丢失任务上下文。

OpenCode 中使用 `nexus-responses/judge` 作为逻辑路由名。Host 侧的
`openai_clients.judge` 会覆盖该逻辑名称并决定实际请求的模型、接口、推理强度和重试
策略。

Judge 请求参数以 `openai_clients.judge.config.request_args` 为最终权威来源。Native
会把其中的 `max_output_tokens`（兼容 `max_tokens`）同步为 OpenCode provider 的输出
上限，把 `reasoning.effort`（兼容 `reasoning_effort`）同步为 OpenCode variant；Host
侧转发时会移除 OpenCode 对这些同名参数的默认值，避免覆盖 YAML。对话消息和动态
`tools` 仍由 OpenCode 提供，因为它们属于本轮 Agentic Judge 状态而不是模型策略。

RewardKit 自身的并发数默认由 Native Task 提供（分别为 4 和 2）。如需调整，可以在
具体运行 YAML 中覆盖：

```yaml
task_config:
  rewardkit_max_concurrent_llm: 4
  rewardkit_max_concurrent_agent: 2
```

Native 将这些值转换成 `REWARDKIT_*` 环境变量，通过
`runtime.run_command(extra_env=...)` 临时注入 `/tests/test.sh` 及其子进程。它们不是
用户运行 `run-single` 前需要设置的 Host 环境变量，也不是 Runtime 容器的全局环境。
Dataset item 也不保存 `REWARDKIT_*` 或其他 verifier 环境变量；这些运行时参数全部由
Native Task 根据 Nexus YAML 生成。
逻辑模型路由和轨迹目录由 Nexus 固定；并发数来自上述 YAML；OpenCode variant 来自
`openai_clients.judge.config.request_args.reasoning.effort`。即使 Host shell 中存在同名
变量，也不会作为这些值的配置来源。

一条 Case 的全部 criteria 会按照 RewardKit 的 `batched` 模式交给同一个 OpenCode
Judge 会话，不会为每条 criterion 单独启动一个 Judge。

## Timeout 所属层级

运行策略统一写在运行 YAML，而不是写入每条 Dataset item：

```yaml
runtime_provider_config:
  request_timeout: 7200

task_config:
  boot_timeout: 7200
  healthcheck_retries: 240
  healthcheck_interval_sec: 3
  healthcheck_timeout_sec: 5
  healthcheck_start_period_sec: 5
  agent_timeout: 7200
  judge_timeout: 2400
  test_timeout: 7200

openai_clients:
  main:
    config:
      client_args:
        timeout: 7200
  judge:
    config:
      client_args:
        timeout: 7200

runner:
  timeout: 21600
```

其中：

- `runtime_provider_config.request_timeout`：一次 Host 到 Runtime RPC 的超时；
- `boot_timeout`：Runtime 获取和 World 服务启动各自的阶段时限；World 阶段包含健康检查；
- `healthcheck_*`：World 服务健康检查策略；
- `agent_timeout`：完整主 Agent rollout 的超时；
- `judge_timeout`：整个评测阶段的总时限，包括写入 Agent 交付物、代理启动、文件上传、
  依赖准备、所有 RewardKit/OpenCode 尝试和结果读取；重试不会重置总时限；
- `test_timeout`：Runtime 中 `/tests/test.sh` 整体命令的超时，覆盖依赖安装、外部状态
  快照、RewardKit 和 OpenCode Judge；保留兼容旧配置，但不能延长 `judge_timeout`；
- `openai_clients.*.client_args.timeout`：单次模型 HTTP 请求超时；
- `runner.timeout`：整条 Case 的总超时。

评测失败（请求失败、超时、非正常退出、缺失或无效 criteria）由 Task 抛出
`EnterpriseJudgeError` 或具体阶段异常，现有 runner 返回 `success=false`、`task_output=null`，
已完成的 `agent_output` 仍保留。正常评测的零分和按既有规则跳过评测不属于评测失败。
部分评测输出记录到 Task span 的 `partial_task_output`，需要 trace processor 持久化；
启用 `artifact_root` 时也会保存评测 records。超时后的代理清理和部分结果读取分别最多 10 秒，
这段有限的收尾时间可能使总耗时略超过 `judge_timeout`。Runtime 中 verifier 命令另有
进程组超时保护，避免 Host 请求取消后评测无限继续。

阶段异常记录 `stage`、`elapsed`、`limit` 和原始异常链；内部 RPC 超时不会再被
runner 误归因为整条任务超时。外层 runner 的取消仍正常向上传播。
Batch 按原有配置重试失败任务；重试会重新执行整条任务，不是仅补评。

## Runtime 数据流

```text
Taskfiles/environment/initial_external_services
  -> /initial_data
  -> /data
  -> World 服务状态

Taskfiles/environment/initial_workspace
  -> /initial_workspace
  -> /workdir
  -> Agent 工作目录

Taskfiles/tests/fetch_grading_context.py
  -> /tests/fetch_grading_context.py
  -> 评分前生成外部服务状态差异

Dataset item.judge_config
  -> Nexus 动态生成 /tests/judge.toml

Nexus Native verifier 文件
  -> /tests/test.sh
  -> /tests/opencode.json
  -> /tests/rewardkit_opencode_agent.py

AgentOutput
  -> /tmp/agent_deliverable/answer.txt
  -> /tmp/agent_deliverable/trajectory.txt

RewardKit/OpenCode 输出
  -> /logs/verifier/
  -> Runtime 内解析为结构化 task output

artifact_root 非空时
  -> 仅下载 /workdir 中 created/modified 文件
  -> Host artifact_root/<run_id>/
  -> 在 Host 写入 Agent/Judge 结构化记录
```

RewardKit 的 `reward.json`、`reward-details.json`、OpenCode JSONL 和 verifier stdout
均在 Runtime 内解析，再通过 `runtime.invoke` 作为结构化结果返回。`artifact_root` 为空
字符串时，不调用 `runtime.download_file`，不会把 Runtime 文件下载到 Host；Task output
中的 score、criteria、Judge trajectory 和 external-services diff 仍会正常返回。

## 构建 Native Dataset

```bash
python -m nexus.extensions.tasks.context_agent.enterprisebench_native.dataset \
  --source /path/to/tencent_aug24_delivery \
  --out /path/to/enterprisebench_native_eval_75 \
  --tasks /path/to/eval75_tasks.txt \
  --name enterprisebench_native_eval_75 \
  --split enterprisebench_native_eval_75 \
  --image-tag v2
```

`--out` 会创建一个完整的数据集目录；配合 `--force` 时会清空该目录。因此不要把它
直接指向已经包含其他 split 的统一数据集根目录。如需合并到统一数据集，应先在独立
目录生成，再复制 `items/` 并更新目标 `dataset.yaml`。

当前已发布的远端数据集是：

```text
yufeihuang/enterprisebench_native_eval_75@v1
```

对应 split：

```text
enterprisebench_native_eval_75
```

该远端数据集是私有数据集，读取时需要有效的 WeData 凭据。

## 运行单条 Case

远端数据示例：

```bash
nexus run-single \
  nexus/extensions/tasks/context_agent/enterprisebench_native/single_case.yaml \
  -o workspace/enterprisebench_single_native/result.json
```

本地数据示例：

```bash
nexus run-single \
  nexus/extensions/tasks/context_agent/enterprisebench_native/single_case_local.yaml \
  -o workspace/enterprisebench_single_native_local/result.json
```

使用 DSV4 Flash 作为 Agentic Judge：

```bash
nexus run-single \
  nexus/extensions/tasks/context_agent/enterprisebench_native/single_case_ds_v4_judge.yaml \
  -o workspace/enterprisebench_single_native_ds_v4_judge/result.json
```

该示例沿用 `base.yaml` 中的主 Agent 配置，只覆盖 Judge 为
`deepseek-v4.1-flash-expires-on-0910`，并使用 AIHub 的 `/v1/responses` 接口。

`base.yaml` 是完整、独立的 Runtime 和模型配置，不会加载
`base.yaml`。如需使用 Infer Proxy HY4 模型，可参考工作区中的：

```text
workspace/enterprise_native/enterprisebench_single_infer_proxy_hy4.yaml
```

### Context Agent 工具状态大小

`context_agent_general` 在构造每次工具调用和清理调用的 `ToolContext` 时，排除
`__mcp_call_records__` 累计审计记录，避免完整历史被反复复制到 trace 输入或传入 Runtime。
该记录仍完整保留在 Host 的 `ExecutionContext.tool_state` 中，供 Task 读取；模型对话、
工具结果、工具配置、deferred tool 列表和执行所需的其他状态均保留。
此优化仅应用于 Context Agent 的工具执行适配器，不修改 Nexus core，也不清空并发共享状态。

### MCP 下载目录权限

Task 在环境上传、World 启动及 workspace 文件链接准备完成后，将工作目录及其现有
真实子目录的属主对齐到 Syntara sandbox 用户，并确保属主具有 rwx 权限；同时预建
`workspace/` 和 Google Drive 默认使用的 `downloads/`。优先读取实际 MCP 服务环境的
`SETUID` / `SETGID`，未提供时使用镜像中的 `model` 用户。普通文件权限不变，遍历不跟随
符号链接，Drive 的业务访问控制不变。这样 MCP 的降权写入进程可以正常下载到这些目录，
而不依赖 Agent 自行修改权限。本初始化不覆盖 Agent 后续新建目录或文件的自定义权限。

### Agent 执行身份

EnterpriseBench Task 将确认的 MCP sandbox UID/GID 写入工具状态的 `execution_identity`。
`context_agent_general` 的 bash、curl 和持久 Python REPL 子进程以同一 UID/GID 执行，
清空补充组，使用 umask 022，并设置对应用户的 HOME/USER/LOGNAME；Runtime 管理、
环境准备和服务启动仍以原管理身份执行。身份通过每次工具请求传递，不修改进程级环境
或全局 UID，避免并行任务互相影响。未设置该状态的其他 Task 继续使用原执行身份。
本行为针对 Context Agent 子进程，不改变其他 harness 或 judge 的执行用户，也不是
对全部文件检索工具的 OS 级权限隔离。模型不能再通过这些命令进行 root 专属系统安装；
Python 包需要预装或安装到用户可写位置。

### 服务原始数据隔离

Task 将 `/initial_data`、`/data`、`/tests`、`/logs`、`/judge_workspace` 的入口与
Agent UID 隔离。支持 ACL 时显式拒绝 Agent 用户并保留其他服务组的穿越权限，
内部文件属主和权限不变；无 ACL 时仅允许安全的 root-owned 布局，否则启动失败。
服务数据若与 Agent 共用 UID，需要先分离服务身份，不能直接放宽权限继续运行。

Context Agent 的 read_lines、grep、msearch/mclick 在独立 UID sandbox 检索进程中
读取和建立索引，保留跨请求索引／指针；不借用用户可编程 Python REPL。
命令和检索进程使用 `setpriv --no-new-privs` 防止通过 sudo/setuid 程序提权，
需要镜像提供 setpriv。其他未设置 execution_identity 的任务保留原路径。
这是服务文件的访问控制，不是完整容器或网络隔离；本地管理 API、同 UID 服务、
自定义上传目录和额外状态存储仍需单独审计。

### 工作副本、交付物与进程生命周期

Agent 可原地读写 workdir 下的普通文件，初始化会对齐文件属主并增加 owner read/write；
文件现有执行位保留，setuid/setgid 位移除。指向 workdir 内文件的 manifest 链接仍可正常使用，
不跟随链接修改外部目标；硬链接工作副本会先分离，防止影响 pristine 数据。
`/initial_workspace` 仅供管理进程/judge 使用，Agent 不可读写。

`/tmp/agent_deliverable` 在 Agent 启动前建立为 root:0700，拒绝非 root 属主或符号链接目录；
answer/trajectory 使用目录文件描述符、O_NOFOLLOW 与新建0600 inode写入，避免跟随预置链接。
最终回答优先采用非None的 final_answer（显式空串不回退），没有该字段或为None时
才从无tool_calls的assistant消息中回退提取。

bash/curl 和 Python REPL 启动独立进程组；命令完成/超时/取消、REPL超时/取消或会话清理时
终止对应进程组。持久REPL在正常轮次间仍保留变量。该机制处理普通子进程，主动脱离会话
的进程还需要cgroup等更强运行时机制约束，不属于完整容器级进程隔离。

### Judge 启动、进度与失败保护

Judge 不再运行时 `pip install`：镜像须预装 `harbor-rewardkit==0.1.7`、
`mcp<2`、LiteLLM 与 OpenCode（原有 v2/v3/v3.1 构建链已安装）。缺包/版本不符
会在依赖预检失败，不联网自动修复。价格表读取使用 LiteLLM 本地备份。

Task 配置（均不改变 rubric 和主 Agent）：

```yaml
task_config:
  judge_timeout: 2400                 # 原有总预算，包括准备和重试
  judge_max_consecutive_failures: 3   # 完整逻辑请求连续失败上限；0关闭
  judge_max_consecutive_rate_limits: 3 # 连续最终HTTP 429上限；0关闭
  judge_no_progress_timeout: 600      # 秒；0关闭无进展保护，但保留总预算
```

Host和runtime_direct均只使用Judge client的`client_args.max_retries`控制单个逻辑
请求的总HTTP尝试次数（含首次请求，不是额外重试数），不再有Task层重复上限。
一次成功模型响应清零连续失败计数；连续失败保护统计的是逻辑
请求失败，不是客户端内部的每个HTTP attempt，内部attempt仍见provider_call轨迹。

无进展指没有准备阶段完成、成功模型响应或工具完成。模型请求开始、失败或重试
本身不算进展；单次长思考/长工具超过该时间也会被终止，可按实际任务提高或关闭。
轮询期间只回传计数和工具时间元数据，不上传完整事件正文。首次请求尚未出现时
归因 `judge.first_request`，之后为 `judge.no_progress`；连续错误归因
`judge.model_failures` / `judge.rate_limit`。这些均为Judge不可用，不能作为模型零分。

诊断保存在 `task_output.judge_details.host_progress` 和 `.runtime_progress`，包含
准备阶段、OpenCode invocation、首请求等待、模型请求耗时与工具起止时间。失败时
同样进入Task的 `partial_task_output` span；runtime另有
`/logs/verifier/judge-timing.jsonl`。HTTP重试明细仍使用原有provider_call记录。
Task超时/保护触发会尝试终止本次Judge独立进程组并取消host待处理请求。

本轮未修改RewardKit的schema重试、OpenCode/AI SDK重试机制、协议转换和串行工具设置。

### 实验性 runtime-direct Judge

`judge_execution_mode: host` 是默认值。可用
`single_case_runtime_direct.yaml` 显式切换为 `runtime_direct`：

```yaml
task_config:
  judge_execution_mode: runtime_direct
runtime_provider_config:
  network_policy:
    default_action: deny
    egress:
      - action: allow
        target: api.aihub.woa.com
```

网络策略作用于整个runtime，不是只在Judge阶段启用。严格要求Agent全程禁止API
出站的实验不应使用本模式，除非平台支持阶段性策略或使用独立Judge runtime。
Task不会自动放行网络；须按judge endpoint配置最小目标域名白名单。

链路：OpenCode → runtime内原生Responses记录代理 → 配置的API endpoint。
没有host LLMRequestHandler、没有Responses/Chat转换，也没有跨轮历史补全。
Host只配置、监控、回收结构化reward和完整dump。仍复用现有
`openai_clients.judge` 配置（当前只接受openai_responses及其子类）；请求字段以
YAML为准，input/tools/stream来自OpenCode，parallel_tool_calls仍固定false。
禁止在request_args覆盖input/messages/tools/stream或注入额外HTTP头。

凭据在Agent结束后以runtime临时公钥加密下发，不以明文出现在RPC输入、taskfiles
或命令参数；仅在relay内存保留。OpenCode拿到的是局部会话令牌，不是真实API key。
代理仅监听127.0.0.1，固定上游、不跟随重定向；结束时关闭连接、取消请求并清理
内存配置。此方式不是针对root对手的安全隔离，也不支持Agent仍有后台恶意进程
的强隔离承诺。模型/工具能看到的输出仍应按敏感任务数据管理。

runtime worker需有aiohttp和cryptography（不运行时安装）。原生SSE边收边转发，
不重建事件；已向OpenCode发送流内容后不做代理层自动重试。仅在收到HTTP429/5xx
或流开始前连接失败时，按Judge client的max_retries限制总尝试次数。连续失败阈值在runtime
计算，host轮询发现后停止Judge；原有总超时/无进展超时仍有效。

结果：
- `judge_trajectories`：由原有OpenCode JSONL重建，含prompt/schema和工具返回。
- `judge_details.direct_responses`：本轮完整请求及原生SSE事件/JSON响应、错误尝试、
  请求时长；保留provider reasoning/usage，凭据字段脱敏。
- runtime路径：`/logs/verifier/direct_responses/`，请求头Authorization不记录。
- 普通host token_counter/LLM spans不再观察这些模型调用；usage在direct_responses
  原始响应中，不应把host计数为0解释成没有消耗token。

分数依旧以RewardKit结构化结果为准，不从聊天文本猜分。OpenCode/SDK如果自身没有
回传某个原始item，本模式不会从旧响应缓存补回；应通过多轮请求dump验证具体模型
兼容性。此模式为显式实验开关，未改变已有host模式行为或Judge权限/工作目录。

### Host Judge：原始 output 缓存回放

host模式由runtime代理缓存模型原始Responses output，OpenCode负责工具循环。
下一轮根据工具结果call_id恢复该轮完整原始output（包含可见reasoning.content、
encrypted_content、原始ID及顺序），使用现有responses_output_items字段交给host
Responses client。Chat字段用于展示，原始output用于实际回放，不依赖OpenCode
重建的assistant历史。

同一provider响应再次到达时幂等复用原缓存位置；相同call_id但output内容不同则拒绝。
单个响应内多段reasoning/text被拆分时，按原始item ID识别已缓存响应的片段，
校验内容和归属后整体回放一次；真正独立的assistant消息保留，不按文本全局去重。

缓存按proxy实例隔离。未知call_id、调用组不完整或顺序不匹配明确失败，不猜测修复。
当前面向线性Judge流程；不保证任意compaction或multi-agent重写兼容。已有独立
assistant文本保留，但不主动恢复本轮已删除的工具历史。
Host handler的preserve_thinking关闭，避免第二套缓存重复补全；这不关闭模型思考，
完整reasoning由runtime原始output回放。

传给host client的输入仍为Chat-shaped messages，Langfuse使用原有tracker，
不需要原生Responses messages适配。Judge与主Agent独立分区的问题并未在此解决。

返回OpenCode使用原始output/state重建缓冲SSE，保留item ID、顺序、真实completed/
incomplete状态；不缓存incomplete输出中的部分工具调用。不是实时原生流式。
PDF/图片工具结果继续使用已有结构化转换。parallel_tool_calls=false策略不变。
公共Responses client的支持范围仍有效，不承诺任意原生工具类型均支持。

### Host Judge 协议校验与参数优先级

- YAML显式配置优先；未配置的支持参数（temperature、top_p、text等）保留OpenCode值。
  store、metadata、service_tier通过task内部字段穿过公共handler过滤，在task-owned client
  中恢复并移除内部字段，不泄漏给模型API。不修改公共runner/handler/client。
- function工具保留strict（包括false）、description、parameters；tools=[]保留。
  native/custom工具明确拒绝，不再降级或丢弃。
- 工具关联历史仍恢复缓存的原始output。独立assistant/reasoning通过已有的
  responses_output_items保留顺序、ID、phase和内容，包括reasoning-only消息。
- previous_response_id/conversation、item_reference、compaction、未知历史条目、
  未知请求字段和不支持的内容块明确报错；自动truncation不允许。
- 检查重复call_id、结果顺序、未完成调用以及调用和结果之间插入的其他消息，
  不再自动重排后继续评测。普通线性工具循环保持原逻辑。
- stream=true由代理返回缓冲SSE，host仍为非流式请求；stream=false和background=true
  不支持。parallel_tool_calls=false不变。以上不改变runtime_direct。

### Host Judge Responses轨迹流

host结果在 `task_output.judge_details.host_responses` 保存 `format=responses_stream_v1`
和单一 `stream`，不再逐轮重复requests快照。只改变记录方式，不改变模型请求或重试。

- request事件：request_id、started_at，参数只记录parameters_set/parameters_remove。
  tools不写入流，统一放在judge_details.tools。输入用retain_items保留上一状态前N个items，再追加
  input_append；前缀变化可以截断旧状态，绝不误当作严格append-only。非列表输入
  用input_replace明确替换。
- response事件：原始response（包括完整output/usage）、headers；
  append_output_to_history=true时将output追加到记录器历史，下一请求通常只需记录工具结果。
- error事件记录失败；request_end记录结束时间和耗时。HTTP重试仍保留各自的事件，
  相同输入和tools无需再次复制。request_id仅为记录序号，不是provider response id。
- 重建请求：应用参数更新，history=history[:retain_items]+input_append，作为input；
  response事件按标志追加output。出现并发请求时仅当前活动请求响应追加公共历史，
  其余response仍完整记录并通过request_id关联，后续request以实际输入重建。

最后一轮output、reasoning、PDF/base64和失败记录均保留，凭据脱敏。原生响应流在这里
指有序请求/响应记录，不是实际HTTP SSE逐token流。大的新工具输出仍占用存储。

host结果judge_trajectories为空，不再重复messages/events/request或host_progress。
内部OpenCode事件仍用于计算judge_rounds和监控。结构化reward、runtime_progress和
CLI stderr保留（无API请求的启动失败不能只靠API轨迹诊断）。direct模式结果不变。
历史result.json不自动迁移。失败记录随partial_task_output保留。

### Judge 长度错误与单代理限制

Judge的上下文超限（context_length_exceeded等已识别的服务端错误）和
`status=incomplete, incomplete_details.reason=max_output_tokens`立即终止本次评测。
不会等待连续失败阈值，也不自动扩容上下文/重启Judge修复。host在HTTP响应记录后、
返回OpenCode前拦截，抛`EnterpriseJudgeLengthError(stage=judge.length)`；后续模型
请求被同一monitor拒绝。Task报告success=false，部分结果的termination_reason为
REWARD_ERROR（不是主模型TRUNCATION），原始失败请求/部分响应保留。

runtime_direct也将长度问题设为不可恢复abort_reason=judge.length，不重试该请求；
已开始转发的SSE遇到长度失败时关闭流并等待Host终止进程。此前已转发的数据无法
撤回，所以不能把它视为严格保证OpenCode从未看见任何部分工具事件；最终结果仍
被Task拒绝作为有效评测。其他incomplete原因保留原有失败处理，不伪装为长度错误。

OpenCode 1.18.14已验证的配置：
- compaction.auto=false，compaction.prune=false：关闭自动上下文压缩和旧工具输出裁剪。
- tools.task=false：配置禁用Task子代理委派工具。
- agent.general.disable=true、agent.explore.disable=true；CLI明确--agent build。
- 按运行要求保留--dangerously-skip-permissions，避免权限审批等待；不再生成额外的
  permission规则。保留工具禁用、Build agent选择和general/explore禁用配置，
  实际工具暴露仍需结合请求验证。

这些是Judge harness行为限制，不是操作系统安全沙箱；root bash仍有创建子进程的
能力。不支持任意外部插件提供的独立委派工具。当前模型请求依旧串行，评分rubric
和主Agent不变。文件路径权限策略没有扩大。


Host工具定义集中存储于`judge_details.tools.definitions`，按完整定义去重。
`judge_details.tools.bindings`记录request_id、side(request/response)、definition_index，
同一side沿用上一绑定直到变化；index=null表示该侧tools字段缺失。首次为空列表或null
仍作为定义保留。请求和响应可能规范化为不同schema，分别存储而不假定相同。
`host_responses.stream`的请求参数及原始响应对象不再携带顶层tools；实际API请求
不受影响。复原完整请求/响应时结合tools绑定；模型正文中的同名业务字段不删除。


### 交付物写入预算

`judge.deliverable`继续将Host的完整final_answer和transcript写入runtime，不移除
trajectory.txt。该步骤不再单独硬编码60秒上限，与后续评测共用从Judge阶段开始
计算的同一个`judge_timeout`截止时间；写入耗时会扣减后续评测预算，不重新计时。
写入超时保留`EnterpriseJudgeTimeoutError(stage=judge.deliverable)`及剩余预算，
不再二次包装成limit=None的泛化错误。底层RPC或运行平台自身的限制仍可能先触发。


### Host Judge 传输重发去重

EnterpriseBench host模式使用Task专用JudgeLLMRequestHandler：同一bridge request_id的
执行结果保留时间为handler创建时剩余Judge预算+60秒清理宽限，不再使用公共handler
固定300秒TTL；本次Judge期间不因容量上限淘汰已完成记录。缓存随该次handler生命周期
释放，不跨任务共享。运行中重发等待同一个结果，完成后重发返回已缓存的成功或错误。

这不改变runtime重发间隔、不改变client max_retries，也不按input内容去重。新的UUID
仍是新请求；host进程重启后不保证去重。该修复防止延迟重发因记录过期而重复调用模型，
不保证消除最初的丢回包或stall。公共handler、direct模式、reasoning回放策略均不变。
实现仅在注册前替换子类的_guard，依赖当前公共handler的内部接口；升级Nexus时需核对。
