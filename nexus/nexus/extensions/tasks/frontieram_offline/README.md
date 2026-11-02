# FrontierAM 指数传播题：Nexus 本地离线接入

本目录把既有指数传播题接入 Nexus 的本地 Task/Agent 生命周期。当前范围是使用真实 Codex 程序、真实本机 Docker 和本地 HTTP 模拟服务检查开发、冻结、构建、评分与错误归因链路。此接入阶段没有调用真实模型，也没有上传或部署到平台；模拟服务重放已有提交所得的分数只是集成诊断结果，不能作为一次新的模型试验。

## 本地架构与职责

```text
nexus run-single
  └─ Task: frontieram_exponential_offline
      ├─ RuntimeProvider: frontieram_offline → 一个 network=none 的开发容器
      ├─ Agent: frontieram_codex_offline → 宿主上运行冻结版本的 Codex
      │   ├─ 本地 ResponsesProxy → 原生转换器 → BoundedOpenAIClient → 本地 HTTP fixture
      │   └─ stdio MCP workspace.exec → 已分配的离线开发容器
      └─ 停止开发 → pause → 冻结源码 → 独立离线构建 → 私有评测 → 清理
```

“离线”约束的是候选程序的运行环境。Codex 控制器、模型协议转换与 Docker 管理在可信宿主执行；本地验证把推理 endpoint 指向 loopback。候选程序没有网络、宿主目录挂载、Docker socket、凭证或私有测试文件，唯一开发工具是 `workspace.exec`。模型推理服务日后若启用，仍在宿主侧访问，不会因此给候选程序联网权限。

Nexus 本身不负责上述断网保证；边界由执行工具落到专用 Docker 环境时建立。参考 `HY_eval_flywheel/repoprobe_with_noise` 的 `c39c41ed38bc4cf9d40ae8f4b2afdf010ed19d8a` 版本：其 Codex 路径保持推理连接，只对派生 shell 使用原生隔离，并另禁用网页搜索；Claude 路径的 PreToolUse 命令检测则是另一种机制。此题需要编译并运行任意本地算法，不能把命令字符串过滤当作断网保证，也不能直接照搬只读代码理解任务的写入限制。当前保留可写工作区，把脚本及其子进程一并约束在无网络环境。Codex 的宿主 shell 已禁用，外部 MCP 工具的隔离由工具本身负责，不能推定它会自动继承 Codex 原生 shell 的隔离。

| 部分 | 来源及本接入的职责 |
| --- | --- |
| 运行生命周期 | 复用 Nexus `runner.single`、Task/Agent 注册与 RuntimeProvider 接口，以及 token 统计。 |
| 模型通信 | 复用 Nexus `OpenAIClient`、ResponsesProxy 的路由、SSE 和 Chat/Responses 转换；缺失调用 ID、对象形式参数的兼容由原生转换器完成。 |
| 开发控制器 | 本目录注册专用的宿主 Codex Agent，限制工具和配置，并记录停止、冻结、日志与推理证据。不是 Nexus 内置的 runtime-server Codex Agent。 |
| 离线容器 | 专用 RuntimeProvider 使用已有冻结的创建、配置校验、pause 和导出原语。没有直接使用需要容器网络服务的通用 runtime-server Docker 路径。 |
| 数值裁判 | 复用旧项目中经过哈希校验的构建器、评测器、数据、参考标定和评分代码；不启动旧 campaign 主循环或旧模型桥。 |
| 新增边界 | 不可延长的开发截止、真实 HTTP 次数预算、有限重试、响应完整性和工具身份检查、先冻结后归因、错误流有界读取，以及运行资源归属与清理。 |

原生 Nexus 提供了可复用的协议和生命周期组件，但其默认重试、超时及转换规则不等同于本题合同。尤其是原生转换器并不会自行拒绝 `finish_reason=length` 的工具调用；这项拒绝由本目录在派发工具之前实施。上述分工不意味着所有服务端、操作系统或 Docker 故障均已解决。

## 冻结资产与评价合同

`source_root` 指向既有项目的 `task_v2/docker_eval` 目录。所需资产不随本 Nexus 仓库分发；[assets.py](assets.py) 记录了要求的文件哈希、镜像 ID 和 Codex 二进制哈希。该目录必须保留公开包、私有数据与冻结清单、参考标定、原评测代码、协议元数据、`runtime/bin/docker` 和 `runtime/bin/codex-frozen`；相邻的 `../public/tools/scoring.py` 也是锁定资产。完整数值验证另需保留既有 campaign 的冻结示例源码。

启动前检查资产，正式评分前再次校验私有数据。发生 `AssetDriftError` 应核对资产来源，不能通过修改锁定哈希来掩盖差异。开发容器仅接收公开 tar；私有数据和参考信息留在可信裁判侧。

继承的主要约束如下；数学任务、精度、字节接口及各会话限制以冻结公开包中的 `PILOT.md`、`INSTRUCTION.md`、`API.md` 和规则文件为准。

- 开发和构建使用单 CPU、固定 CPU 0、4 GiB 内存且无额外 swap；限制进程/线程数、文件描述符以及临时存储，系统镜像只读。
- 完整模型试验的开发上限是 1800 秒墙钟时间，推理、重试等待和工具操作都占用同一窗口。每次响应最多 64000 个输出 token，最多 200 次实际 HTTP 请求。较短的验证用例是诊断，不是等预算的模型比较。
- 最终回复或正常开发截止后先暂停整个工作区，再导出 `/workspace/submission`。没有后续修补、继续开发或隐藏测试反馈。提交只接受规定的普通文件，保留源码与摘要清单。
- 可选的 `build.sh` 在另一离线容器中执行，沿用 600 秒墙钟、600 秒容器总 CPU 和 4 GiB 内存的构建限制。构建时不提供公开或私有测试数据。
- 数值评分沿用冻结协议：`READY` 后从交付输入开始，到全部输出收齐和正常退出中较晚者结束；输入相关准备属于耗时。时间、容器级资源、数值正确性和 0–100 分映射由裁判计算，不采信候选程序自报值。

推理重试存在一项需要单独标注的差异：旧公开 `PILOT.md` 的文字只允许满足条件的 429 重试；本接入的诊断控制器按冻结 AIHub 策略同时检查 HTTP 状态和网关错误码，允许其中标记可重试的 429/500/502/503/504，单次逻辑请求最多四次实际尝试。SDK 和原生外层重试不再叠加，重试不延长原截止、不切换供应商。因而当前接入保留数值与资源合同，但不能宣称所有推理行为与旧轮完全相同。正式重跑前应统一公开协议；旧轮记录保持原样，不与新诊断或新协议结果混成排行。

## 本地准备与环境变量

从 Nexus 仓库根目录执行命令。使用 Python 3.12–3.14，先准备已锁定的依赖环境，例如：

```sh
uv sync --locked --extra dev --extra runtime
```

依赖安装可以联网，后续命令使用 `--no-sync`，不在运行过程中安装依赖。还需预先启动本机 Docker daemon 并备好锁定的 arm64 Linux 镜像；当前 Codex 二进制也依赖原本机平台。适配层不会自动下载镜像、模型或额外候选库。

| 环境变量 | 用途 |
| --- | --- |
| `FRONTIERAM_SOURCE_ROOT` | 既有 `task_v2/docker_eval` 的绝对路径；不是 Nexus 仓库根目录。 |
| `FRONTIERAM_DOCKER_HOST` | 本机 daemon 的 `unix://` socket 地址；不接受远程 TCP daemon。 |
| `FRONTIERAM_OUTPUT_DIR` | 本次证据目录的绝对路径，必须尚不存在，且不能位于冻结资产树内。每次换新目录。 |
| `FRONTIERAM_BASE_URL` | 仅 YAML 运行需要：已启动的本地模拟服务地址，例如 `http://127.0.0.1:8765/v1`。必须包含 `/v1`，不包含 `/chat/completions`。 |

例如，先把下面两处占位路径替换为本机位置：

```sh
export FRONTIERAM_SOURCE_ROOT="/absolute/path/to/task_v2/docker_eval"
export FRONTIERAM_DOCKER_HOST="unix:///absolute/path/to/docker.sock"
export FRONTIERAM_OUTPUT_DIR="$PWD/outputs/frontieram-local-$(date +%Y%m%d-%H%M%S)"
```

不要预先创建 `FRONTIERAM_OUTPUT_DIR`。输出目录需要可写，且父目录应仅对本机受信任用户开放。运行材料包括模型/工具历史，不能把它当作可直接公开的数据包。

## 验证入口

优先使用自带验证器。它自动启动本地 HTTP fixture，不读取真实 API key，也不需要设置 `FRONTIERAM_BASE_URL`：

```sh
uv run --no-sync python -m nexus.extensions.tasks.frontieram_offline.verification \
  --source-root "$FRONTIERAM_SOURCE_ROOT" \
  --docker-host "$FRONTIERAM_DOCKER_HOST" \
  --output "$FRONTIERAM_OUTPUT_DIR" \
  --cases runtime_parallel parameter_retry missing_id object_arguments length unknown_tool retry_cap deadline network_boundary full_numerical
```

`--cases` 可以只选其中一部分；参数省略时运行验证器定义的默认用例。除 `runtime_parallel` 只检查 RuntimeProvider 外，其余用例均使用真实 Codex 和离线容器，区别如下：

| 用例 | 检查目标 |
| --- | --- |
| `runtime_parallel` | 两个并行工作区的身份、写入与释放互不干扰，不进行并行性能测量。 |
| `parameter_retry` | 模拟可重试 500 后原请求重试；普通工具参数错误回到历史，后续有效调用成功。 |
| `missing_id`、`object_arguments` | 原生转换兼容缺失 ID / 对象参数，且下一轮消息中的调用和工具结果正确配对。 |
| `length`、`unknown_tool` | 截断响应或未知工具在执行前被拒绝，禁止的文件不会被写入。 |
| `retry_cap` | 持续可重试错误达到上限即停止，记录推理失败而非算法零分。 |
| `deadline` | 延迟响应不能延长开发截止或使过期工具调用生效。 |
| `network_boundary` | 经真实 MCP 执行脚本文件和编译程序，检查外部 TCP/UDP 与宿主模型代理不可达；必须从成功工具结果验收，不能只搜索调用参数中的标记。 |
| `full_numerical` | 经 CLI 运行 Nexus 任务，将已有冻结源码重放到工作区，再完成构建和全部私有数值评测；不是让真实模型重新解题。 |

如已有自己的本地 HTTP fixture，可设置 `FRONTIERAM_BASE_URL`，再用[示例配置](../../../../docs/examples/configs/frontieram_offline.yaml)检查标准 CLI 路径：

```sh
export FRONTIERAM_BASE_URL="http://127.0.0.1:8765/v1"
uv run --no-sync nexus run-single docs/examples/configs/frontieram_offline.yaml \
  -o "${FRONTIERAM_OUTPUT_DIR}.nexus_result.json"
```

该 YAML 不会启动 HTTP fixture；上面的端口必须已有符合 Chat Completions 协议、支持工具回合的本地服务。默认 `allow_live_endpoint=false`、`key_file=null`、`verification_only=true`、`evaluate=false`，开发预算也缩短为 60 秒。`verification_only` 是结果用途标记，不是断网开关；`allow_live_endpoint` 和 endpoint 校验才限制真实服务接入。`evaluate=false` 表示只验收开发与源码冻结，结果没有数值分数。

独立的行为检查不需要 Docker 或真实模型：

```sh
uv run --no-sync pytest -q \
  tests/unit/test_frontieram_nexus_native_boundaries.py \
  tests/unit/test_frontieram_offline_client.py \
  tests/unit/test_frontieram_offline_agent.py \
  tests/unit/test_frontieram_offline_task.py \
  tests/unit/extensions/tasks/test_frontieram_runtime.py
uv run --no-sync ruff check nexus/extensions/tasks/frontieram_offline
uv run --no-sync ty check nexus/extensions/tasks/frontieram_offline
```

其中原生边界测试会明确断言原生组件的现状，包括其未提供的保护。该组测试通过不等于原生默认配置已经满足本题的所有约束。

## 分数、失败与证据

| 情况 | 结果含义 |
| --- | --- |
| 正常 final 或经过验证的开发截止，提交有效并完成评测 | 使用冻结评分器产生 0–100 分；源码必须已冻结。 |
| 正常结束，但提交缺失/不合法，或规定的构建失败 | `score=0`，属于提交质量问题；不由维护者补做。 |
| 上游推理失败、截断、非法工具身份、控制器/基础设施异常、未验证的中断 | `status=unmeasured`、`score=null`；不能当作算法零分。推理中断优先于随后的空提交错误。 |
| `evaluate=false` 且开发和冻结正常 | `verification_completed`、`score=null`；明确没有运行数值评价。 |
| 取消或清理失败 | 保留中断/清理证据，不把未完整结束的运行作为有效模型成绩。 |

验证器输出根目录包含 `verification_result.json`；每个用例有独立子目录。直接用 YAML 时，任务材料位于 `FRONTIERAM_OUTPUT_DIR`，CLI 总结果位于 `-o` 指定的文件。重点查看：

- `nexus_result.json` / CLI 总结果：Nexus 外层执行结果、Task 和 Agent 输出。外层 `success` 不等于候选算法正确，必须看 `task_output.status`、`score` 和 `verification_only`。
- `task_result.json`、`agent/agent_result.json`：资格、停止原因、错误类别、开发耗时和实际 HTTP 次数。
- `agent/http_audit.jsonl`：请求计数、请求体摘要、重试、协议兼容与错误类别；与本地 fixture 的请求摘要核对是否确实重试了同一请求。
- `agent/codex_events.jsonl`、`agent/codex_stderr.log`、`agent/boundary.json`：控制器事件、错误流、共享截止、观测到停止与工作区冻结的时间。清理耗时单独记录。
- `source_manifest.json`、`source/`、构建和评测报告、`score.json`：实际提交及数值评分依据。
- 出现时的 `task_interrupted.json`、`cleanup_error.json`、`grading_cleanup_error.json` 和资源归属记录：取消、资源清理或裁判失败证据。

失败运行保留在原目录；修复后使用新目录重试，不覆盖失败证据，也不把多次运行择优算作一次模型尝试。2026-11-02 的本地验证完成了 10 个集成用例、69/69 次数值运行和 130 项单元/边界检查，真实模型调用为零。[本机完整记录](../../../../outputs/frontieram_nexus/LOCAL_VALIDATION.md)区分了框架能力、专用适配、断网证据和剩余限制；该目录为本地证据，不随上游代码分发。

## 真实服务与剩余限制

当前不启动 live 模型任务。以后如需真实 AIHub 试验，应复制为独立、未入库的 profile，显式启用 `allow_live_endpoint`，使用代码允许的 AIHub base URL，并通过 `key_file` 指向权限为 `0600` 的本机密钥文件；不能把密钥正文写入 YAML 或公开日志。loopback fixture 配置拒绝读取真实密钥。仅切换 `verification_only=false` 不会开启真实服务，也不会自动恢复正式预算或统一公开协议。

本实现仍依赖外部冻结资产、本机 Docker 行为和既定硬件环境；没有验证跨平台迁移、平台分布式执行、真实供应商限流和长时间稳定性。清理会核对运行归属，清理失败会影响结果资格；这不是对宿主崩溃、daemon 丢失或所有取消时序的无泄漏保证。真实模型的完成率、任务难度和分数区分度需要在统一的新试验合同下另行测量，不能从本地故障注入或既有提交重放中推导。
