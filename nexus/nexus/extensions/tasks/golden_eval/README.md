# golden_eval

多轮人工交互评测任务：发一句开场 prompt，agent 在隔离沙箱里产出可渲染的项目；
人看着产物继续发引导 prompt，反复几轮；结束时保存最终产物与完整轨迹，释放沙箱。

一份 YAML = 一个 run = 一个模型。要同时评多个模型，就把 YAML 复制几份改掉模型名，
分别起多个 run，它们完全独立、互不干扰。

和 `frontend_task` 的分工：那个是批量单轮、跑完即打包；这个是多轮交互、轮次由外部驱动，
只在会话结束时归档一次。

---

## 准备

```bash
cd nexus
uv sync --extra dev
uv run nexus modules install litellm     # bedrock 系模型要 boto3，缺了会在调模型时才炸
```

沙箱 token 在 https://sbox.woa.com/?tab=resource 申请。注意容量按用途分别申请，
默认用途是 `test`，没额度会报 `260400 当前 Token 没有用途 test 的容器配额`。

---

## 配置

存成 `golden_eval.yaml`。这是一份普通的 `RunInput`，没有任何"live runner 专属"字段。

```yaml
# __init__.py 是空的，所以 imports 要写到模块而不是包
imports:
  - nexus.extensions.tasks.golden_eval.task
  - nexus.extensions.agents.claude_code
  - nexus.extensions.openai_clients.litellm
  - nexus.trace.processors.chat_tracker

runtime_provider_name: gongfeng
runtime_provider_config:
  token: ${oc.env:SANDBOX_TOKEN}
  api_base_url: http://hyrl-sandbox.prod.woa.com:8052
  # 通用镜像：TencentOS + Node 24 + Python 3.11，可联外网，npm 装包数秒。
  # 镜像里没有浏览器，沙箱无法自行渲染或截图。
  image: mirrors.tencent.com/hunyuan_yanguan/webenv.claudecode.testset:v3
  runtime_image: current
  runtime_provider: gongfeng
  request_timeout: 10000.0
  resource_spec: {cpu: 16, memory: 32}

runtime_configs:
  main:
    # 把当前代码打成 wheel 传进沙箱，golden_eval 的改动才会在沙箱侧生效
    worker_venv: true

openai_clients:
  main:
    name: litellm
    config:
      client_args:
        base_url: http://api.aihub.woa.com/standard/v1
        api_key: ${oc.env:AIHUB_API_KEY}
        cache_control_enabled: true
      request_args:
        model: openai/claude-opus-5

task_name: golden_eval
task_config:
  working_dir: /workspace
  # 开场素材，在 agent 启动前写入沙箱，第一轮 prompt 即可引用。
  # 相对路径按 working_dir 解析；二进制先 base64。这些文件会一并进归档。
  # input_files:
  #   - path: city.csv
  #     content: "城市,人口\n深圳,1766\n"
  #     encoding: utf8
  #
  # 无人值守时用它代替真人；live runner 接管时会被覆盖。
  # user_inputs:
  #   - 把配色换成暖色调

# 必须是支持多轮的 scaffold。填成单轮的 claude_code 不会报错，
# 但它的输入模型没有 get_user_input 字段，后续轮次会被静默丢弃。
agent_name: claude_code_multiturn
agent_config:
  user_prompt: >-
    做一个待办清单页面，支持新增、勾选完成、删除和按状态筛选，数据存 localStorage，深色主题。

    约束：
    1. 所有文件放在 /workspace，入口必须是 index.html。
    2. 前端所有接口请求必须使用相对路径（如 fetch("api/items")），禁止以 / 开头的绝对路径，
       页面会被挂载在子路径下。
    3. 若需要后端，前后端必须同一进程同一端口，监听 0.0.0.0:8080，静态文件也由它托管；
       写完后自己用 nohup 常驻启动，并确认 8080 已在监听。
    4. 纯静态页面不需要起服务，平台会自动托管。
  working_dir: /workspace
  version: '2.1.117'
  max_turns: 5
  timeout: 86400
  # 模型中途提问时转给人。无人值守跑批必须关，否则会一直挂到超时。
  enable_user_input_hooks: true
  include_hook_events: true
  # 只建目录。不要在这里起静态服务 —— 会占掉 8080，让 agent 自己的后端无处可绑。
  pre_start_scripts:
    - code: mkdir -p /workspace

# 不配就没有轨迹，只有产物
processors:
  chat:
    name: chat_tracker
    args:
      tools_mode: branch_last
```

---

## 跑法一：无人值守（标准单次运行）

轮次全部预设在 `task_config.user_inputs` 里，不需要人。记得把
`enable_user_input_hooks` 改成 `false`，否则模型一提问就没人应答，会挂到超时。

```bash
export SANDBOX_TOKEN=<沙箱 token>
export AIHUB_API_KEY=<模型 key>

uv run nexus run single golden_eval.yaml
```

---

## 跑法二：人工多轮（live runner）

### 1. 起服务

`-d` **必须带**，否则 run 结束后什么都不落盘。`${oc.env:X}` 由 **live runner 进程**
解析，所以环境变量要在这个 shell 里导出。

```bash
export SANDBOX_TOKEN=<沙箱 token>
export AIHUB_API_KEY=<模型 key>

uv run nexus run live --port 8765 -d ./golden_eval_history.db
```

### 2. 建会话

```bash
RID=$(curl -s -X POST http://127.0.0.1:8765/v1/runs/yaml \
  -H 'Content-Type: application/json' \
  -d "$(jq -Rs '{run_input_yaml: ., live: {working_dir: "/workspace"}}' golden_eval.yaml)" \
  | jq -r .run_id)
echo $RID
```

要评多个模型就改掉 YAML 里的 `request_args.model` 再执行一次，各自拿到独立的 `RID`。

### 3. 等到可交互

状态机是 `starting → running ⇄ idle → closed`。**只有 `idle` 能发轮次。**

```bash
watch -n5 "curl -s http://127.0.0.1:8765/v1/runs/$RID | jq -r .status"
```

沙箱创建约 1-2 分钟，首轮生成再 1-3 分钟。模型服务繁忙时可能显著更久，
表现为长时间停在 `running`，不是卡死。

### 4. 起预览服务并查看产物

沙箱里要有东西监听 8080 才能预览。agent 自己写了后端就用它的，纯静态则兜底起一个：

```bash
curl -s -X POST "http://127.0.0.1:8765/v1/runs/$RID/runtimes/main/proxy/api/v1/call_executor" \
  -H 'Content-Type: application/json' \
  -d '{"executor_name":"run_command","arg":{"command":"cd /workspace && { (ss -ltn 2>/dev/null||netstat -ltn) | grep -q \":8080 \" && echo already; } || { [ -f server.js ] && (nohup node server.js >/tmp/app.log 2>&1 &) || (nohup python3 -m http.server 8080 >/tmp/serve.log 2>&1 &); sleep 2; echo started; }","timeout":30}}'
```

然后浏览器打开：

```
http://<主机>:8765/v1/runs/<RID>/runtimes/main/proxy/debug/ports/8080/
```

代理支持 GET/POST/PUT/PATCH/DELETE 和 WebSocket，全栈产物可以真实点击交互。
每轮结束后重跑上面那条命令即可（已经在跑就直接返回 `already`）。

### 5. 发后续轮次

```bash
curl -s -X POST "http://127.0.0.1:8765/v1/runs/$RID/turns" \
  -H 'Content-Type: application/json' \
  -d '{"text":"把配色换成暖色调，再加一个联系表单"}'
```

模型中途反问时也用这个接口回答，看 SSE 里 `input.required` 的 `source` 字段区分：
`turn` 是轮次结束在问"下一步做什么"，`hook` 是模型在问你具体问题。

```bash
curl -sN "http://127.0.0.1:8765/v1/runs/$RID/events?cursor=0"
```

### 6. 收尾

发空 turn 让任务正常结束。**不能只发一次**——模型处在 `hook` 提问状态时，
空 turn 会被当成对问题的空回答，会话继续。循环到真的 `closed`：

```bash
while true; do
  S=$(curl -s "http://127.0.0.1:8765/v1/runs/$RID" | jq -r .status)
  echo "$S"
  case "$S" in
    closed|failed|killed) break ;;
    idle) curl -s -X POST "http://127.0.0.1:8765/v1/runs/$RID/turns" \
            -H 'Content-Type: application/json' -d '{"text":""}' >/dev/null; sleep 8 ;;
    *) sleep 10 ;;
  esac
done
```

### 7. 取产物与轨迹

```bash
curl -s "http://127.0.0.1:8765/v1/runs/$RID" > result.json

python3 - <<'PY'
import base64, io, json, tarfile
d = json.load(open('result.json'))
out = d['output']
task_output = out['task_output']
print('轮次', task_output['turns'], '文件', task_output['file_list'])

with tarfile.open(fileobj=io.BytesIO(base64.b64decode(task_output['archive_b64'])), mode='r:gz') as tar:
    tar.extractall('./artifact')

trajectories = out['processor_results']['chat']['trajectories']
json.dump(trajectories, open('trace.json', 'w'), ensure_ascii=False, indent=2)
PY
```

沙箱在任务返回时自动释放，预览地址随即失效。回看产物用解压出来的 `./artifact`
自行静态托管；全栈产物要恢复交互，得自己跑后端或拿产物重开一个 run。

---

## 换 harness

同一个题目要横向对比不同 harness 时，**只有两处需要改**：`imports` 里的 agent 模块，
和 `agent_name` + `agent_config` 整块。其余部分（`runtime_*`、`openai_clients`、
`task_config`、`processors`）四个 harness 完全一致，包括模型——模型配置在 `openai_clients` 里，
不在 `agent_config` 里，所以 harness 和 model 两个维度可以自由组合。

各 harness 的 `agent_config` 字段并不通用，不认识的字段会被 `_build_input` 静默丢掉，
所以**按 harness 各存一份模板**，不要写并集配置。

全仓库 55 个 agent 里只有下面 4 个支持多轮（输入模型带 `get_user_input`）。
单轮 agent 配上来不会报错，但注入的回调会被过滤，跑完第一轮就结束。

### claude_code_multiturn（已验证）

上面「配置」章节那份就是。管道 + stream-json，输出结构化，解析确定，是默认选择。

### claude_code_interactive

PTY 驱动真正的 Claude Code 终端 UI，能触发交互专属分支（不同系统提示、不同工具集、
plan mode、output-style 等）。代价是靠虚拟终端截屏取文本，对时序敏感。

```yaml
imports:
  - nexus.extensions.tasks.golden_eval.task
  - nexus.extensions.agents.claude_code          # 与 multiturn 同一个模块
  - nexus.extensions.openai_clients.litellm
  - nexus.trace.processors.chat_tracker

agent_name: claude_code_interactive
agent_config:
  user_prompt: >-
    ……同上……
  working_dir: /workspace
  version: '2.1.117'
  max_turns: 5
  timeout: 86400
  # PTY 尺寸与时序，截屏解析不稳时先调这几个
  cols: 200
  rows: 50
  prompt_settle_seconds: 1.0
  snapshot_seconds: 0.5
  pre_start_scripts:
    - code: mkdir -p /workspace
```

**没有 `enable_user_input_hooks`**：它用原生的 `respond_and_wait` 处理中途提问，
不走 hook 那条路，配了也会被丢掉。

> ⚠️ 未实测，且有一个已知风险：`claude_code` 模块声明了 runtime 依赖 `pyte>=0.8.2`
> （PTY 虚拟终端就靠它），而 `nexus modules` 显示该模块 Runtime 侧是 `0/1 ✗`，
> 说明运行时镜像里可能没有。切过去之前先跑一次确认能起来。

### dsh

走 ACP 协议，字段体系和 cc 差别最大。

```yaml
imports:
  - nexus.extensions.tasks.golden_eval.task
  - nexus.extensions.agents.dsh
  - nexus.extensions.openai_clients.litellm
  - nexus.trace.processors.chat_tracker

agent_name: dsh
agent_config:
  user_prompt: >-
    ……同上……
  working_dir: /workspace
  version: '0.1.0-rc.6'
  acp_version: '0.1.0-rc.6'
  max_turns: 5
  timeout: 86400
  permission_mode: workspace-write
  permission_policy: allow
```

差异最需要注意的两点：**没有 `pre_start_scripts`**，所以 `/workspace` 只能靠镜像自带或
任务侧准备（`golden_eval` 的 `input_files` 是任务通过 `upload_file` 写的，不受影响）；
**没有 `enable_user_input_hooks` 和 `extra_cli_args`**，中途交互由 `permission_mode` /
`permission_policy` 控制。

可参考仓库里的 `docs/examples/configs/dsh_acp_multiturn.yaml`。模块状态 `✓ ✓`。

> ⚠️ 未在 golden_eval 下实测。

### hycode_dynamicagent_multiturn

字段和 `claude_code_multiturn` 几乎一致，去掉了 MCP 和自定义 binary 那几项。

```yaml
imports:
  - nexus.extensions.tasks.golden_eval.task
  - nexus.extensions.agents.hycode_dynamicagent
  - nexus.extensions.openai_clients.litellm
  - nexus.trace.processors.chat_tracker

agent_name: hycode_dynamicagent_multiturn
agent_config:
  user_prompt: >-
    ……同上……
  working_dir: /workspace
  version: latest
  max_turns: 5
  timeout: 86400
  enable_user_input_hooks: true
  pre_start_scripts:
    - code: mkdir -p /workspace
```

模块状态 `✓ ✓`。

> ⚠️ 未在 golden_eval 下实测。

### 任务侧不用改

`golden_eval` 读的 `turns`、`ended_by_user`、`reached_max_turns`、`session_id`
都在各 harness 输出模型的共有集合里，而且用的是 `getattr(..., default)` 的防御式写法，
换 harness 任务一行不动。

接一个新 harness 时建议先手工跑通一次三轮会话，确认三件事：多轮上下文真的延续、
产物落在 `working_dir`、空 turn 能正常收尾。这三条是 `golden_eval` 的隐含假设，
不同 harness 未必都满足。

## 坑

下面每一条都是实际踩出来的。

### 收尾必须用空 turn，不能用 kill

空 turn 让 `get_user_input` 返回 `None`，多轮循环正常退出，任务返回，产物和轨迹落库，
沙箱在上下文退出时释放。

`POST /kill` 走的是 `CancelledError`，任务不产生 `RunOutput`，库里 `output_json` 是 NULL，
**产物和轨迹全部丢失**。实测四个 kill 的 run 无一例外。

### 空 turn 只在轮次真正结束时才是结束信号

模型中途调 `AskUserQuestion` 时（事件里 `source=hook`），空 turn 会被当成对问题的空回答，
会话继续而不是结束。所以收尾要写成循环。

### 改完任务代码必须重启 live runner

`imports` 只触发一次导入，模块已在 `sys.modules` 里时不会重新加载。旧类定义上没有的新字段会被
`_build_input` 静默过滤掉，表现为**参数传了但不生效**，不报错不警告，极难排查。

### agent 必须选支持多轮的 scaffold

`claude_code_multiturn` 或 `claude_code_interactive`。填成单轮的 `claude_code` 不会报错，
但它的输入模型没有 `get_user_input` 字段，注入的回调同样被静默过滤，跑完一轮就结束。

### 不要在 pre_start_scripts 里起静态服务

它会占掉 8080，让 agent 自己写的后端绑不上端口。页面能加载但所有接口 404。
预览服务要在 agent 产出之后按需拉起。

### 不要用 openai_config_path

`/v1/runs/yaml` 走 `RunInput.from_dict` 解析，而 `openai_config_path` 不是 `RunInput` 的字段
（只存在于 `CreateRunRequest`），会被静默丢掉，转换时还被硬编码成 `None`。
结果是 `openai_clients` 为空，任务报 `OpenAI client 'main' not found`。
直接在 YAML 里内联 `openai_clients` 即可。

### 产物页面必须用相对路径

页面被挂在 `/v1/runs/{id}/runtimes/main/proxy/debug/ports/8080/` 这样的深层前缀下，
`fetch('/api/x')` 会打到 live runner 根目录。必须写 `fetch('api/x')`。
前后端也要同端口同源——分端口的话页面里的 `localhost` 指的是使用者自己的机器。
这两条要写进开场 prompt 的约束里，上面的配置已经带了。

---

## 参考

### 任务输入

| 字段 | 说明 | 默认 |
|---|---|---|
| `prompt` | 开场 prompt；留空则用 `agent_config.user_prompt` | `''` |
| `working_dir` | agent 工作目录 | `/workspace` |
| `runtime_name` | 运行时名 | `main` |
| `input_files` | 开场素材，`{path, content, encoding}` 列表 | `[]` |
| `user_inputs` | 脚本化后续轮次；被外部注入覆盖 | `[]` |
| `archive_exclude` | 归档排除的目录名 | 见 `DEFAULT_ARCHIVE_EXCLUDE` |
| `max_archive_mb` | 超过此大小只返回文件清单 | `64` |
| `collect_timeout` | 归档超时（秒） | `300` |
| `release_runtime` | 归档后是否立即释放 | `False` |

`release_runtime` 默认关闭：`ExecutionContext` 退出时会先 `drain_trace()` 再释放，
任务提前释放会让这次 flush 打在已销毁的沙箱上拿到 502，进而把 live runner 拖死。

### 任务输出

| 字段 | 说明 |
|---|---|
| `completed` | 是否成功 |
| `archive_b64` | 工作目录 tar.gz 的 base64 |
| `file_list` | 包内文件清单 |
| `archive_bytes` | 压缩包原始字节数 |
| `archive_skipped` | 超过 `max_archive_mb` 时为真，此时 `archive_b64` 为空 |
| `turns` | 实际完成轮次 |
| `ended_by_user` | 是否由空 turn 正常结束 |
| `reached_max_turns` | 是否撞上 `max_turns` |
| `session_id` | Claude Code 会话 id |
| `error` / `termination_reason` | 失败信息 |

轨迹不在这里，在 `processor_results.chat.trajectories`，由 `chat_tracker` 采集。
多轮会合并成**一条递增的长轨迹**，不是每轮一条。

### 用到的 live runner 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/v1/runs/yaml` | 用 RunInput YAML 建会话 |
| GET | `/v1/runs/{id}` | 状态；`closed` 后 `output` 里有产物与轨迹 |
| GET | `/v1/runs/{id}/events?cursor=N` | SSE：进度、轮次边界、模型在做什么 |
| POST | `/v1/runs/{id}/turns` | 发轮次；空串 = 正常收尾 |
| ANY | `.../runtimes/main/proxy/debug/ports/8080/` | 产物渲染 |
| ANY | `.../runtimes/main/proxy/api/v1/call_executor` | 在沙箱里执行命令、上传下载文件 |

### 轮次为什么不由任务驱动

任务给 `run_agent` 传了一个基于 `user_inputs` 的脚本化输入源，但 runner 合并参数时
`agent_input_overrides` 排在最后：

```python
merged_kwargs = {**run_input.agent_config, **task_kwargs, **(agent_input_overrides or {})}
```

live runner 正是通过它注入自己的回调，所以有人接管时人优先、没人接管时走脚本，
同一份 YAML 两种跑法通吃，任务里不需要分支。

代价是任务观察不到单轮边界，**每轮快照只能由调用方负责**——run 处于 `idle` 时
经运行时代理读沙箱即可。任务只负责最后那一次归档。
