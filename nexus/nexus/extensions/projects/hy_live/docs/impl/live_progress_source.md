# 后脑实时进展的数据来源：ACP 事件推送

> **代码位置**：推送的接收、路由与重放在 nexus（`coordination/acp_updates.py`、`coordination/backend.py`），update 写成 Event 在 `hylive` 包（`hylive.coordinator.Coordinator.on_update`、`hylive.execution_events`）。

> **边界**：本文回答「后脑的执行进展从哪里取、怎么到 Host、会丢什么」。 harness 直接把原始推送写成 Event，规则见 [`harness_events.md`](harness_events.md)。

## 1. 结论

进展直接取自 CodeBuddy 的 ACP `session/update`。 runtime 在一轮 prompt 运行期间把每条事件实时推给 Host，Host 以 `AcpUpdate` 交给 core，由 `Coordinator.on_update` 写成 harness Event。

- **归属**：Host 每轮生成一个 `turn_id` 随 prompt 传进沙箱，沙箱只给这一次 prompt 运行期间的事件打上它。 归属在 CLI 所在的沙箱里判断，Host 不靠时间窗口推断。
- **可靠性**：推送不做重发。 一轮返回时带回沙箱为这一轮推了多少帧（`relay_frames`，emit 失败的也算）。 Host 一帧都没收到时，用 invoke 返回的完整事件整轮重放；收到一部分时只打 warning、不补（原因见第 2 节）。

**一轮** 指一次 ACP `session/prompt`：Host 发一条消息，CodeBuddy 跑完整个 agent 循环才返回。 对委派的 project Session，一轮通常就是整个任务，所以没有推送时，任务进行中看不到任何进展。

## 2. 链路

```text
CodeBuddy CLI ──stdio──► AcpConnection._dispatch
                           ├─ update listener ─► AcpUpdateRelay ─sio acp_session_update {channel_id, turn_id, seq, update}─►
                           │                                         Host AcpUpdateRouter ─► AcpUpdate ─► Coordinator.on_update（实时）
                           └─ updates 队列 ─► drain_updates() ─► invoke 返回值 events + relay_frames ─► 一帧都没收到时重放 ─► Coordinator.on_update
```

- **runtime 侧**（`installed_common/acp.py`、`installed_common/host_rpc.py`、`codebuddy/host_relay.py`）：`AcpConnection.add_update_listener` 在入队前同步回调；`AcpUpdateRelay` 用单个发送协程按顺序推送。 session 在一次 prompt 开始前清空队列后调用 `begin_turn(relay_turn_id)`，收尾 `drain_updates` 后调用 `end_turn()`，只有这中间的事件会被推送并打上 `turn_id`，`seq` 每轮从 1 开始，emit 失败也计数；返回结果前先 `flush()`，让帧先于结果离开沙箱。
- **裁剪**：推送前 `trim_acp_update` 去掉推理文字、非工具 update 的 `_meta` 和 `available_commands_update`；工具的 `tool_call` / `tool_call_update` 原样推送，因为 harness 要用其中的 `rawOutput` / `content` 写输出摘要、用 `_meta` 里的工具名写标题。 正文 `agent_message_chunk` 不截，因为这一轮最后的正文要完整交给中脑。 工具 update 里的字符串最多被 Host 读 400 个单位（中日韩字符或英文词，长串每 32 字符算一词），即至多 12800 个非空白字符，所以长字符串保留 25600 个非空白字符、总长不超过 51200 字符。 结构和文件路径不截，Host 写出的 Event 与不截时逐字相同。 裁剪只在沙箱里做，invoke 返回的 `events` 不裁剪。
- **开关**：`work_agent_configs.codebuddy.relay_updates`（`CodeBuddyMultiturnInput.relay_updates`，必须同时给 `relay_turn_id`）。 镜像的 host relay 不认识这个 key 时 session 直接启动失败，所以开关要和支持它的镜像一起打开（见 [`runtime_image.md`](runtime_image.md)）；开关关闭时这两个都不会发给沙箱。
- **Host 侧**（`coordination/acp_updates.py`、`coordination/backend.py::_run_with_acp_updates`）：每一轮开始时生成 `turn_id`，把 channel 和它绑定到这一轮服务的 task；`turn_id` 不匹配的帧直接丢弃。 Host 的预算超时包含 session 启动时间，而沙箱的 prompt 计时从拿到 session 锁才开始，所以 Host 放弃后 CLI 可能还要跑几十秒；这期间同一 Session 上下一个 task 已经绑定，旧 prompt 的帧靠 `turn_id` 区分，不会算到它头上。 帧和 invoke 结果走不同的路径，最后几帧可能比结果晚到，所以收到的帧少于 `relay_frames` 时先最多等 2 秒（`_TRAILING_FRAME_TIMEOUT_S`）；然后解绑，之后到达的帧一律丢弃。
- **重放**：解绑后一帧都没收到（`relay_updates` 关闭、镜像不支持或全部丢失）时，用 invoke 的 `events` 按顺序整轮重放。 收到一部分时只打 warning：`seq` 只给裁剪后保留的 update 编号，对不上未裁剪的 `events`，补发会把已经写过的帧再写一遍；而且同一 task 的另一条流可能仍在推送，重放会打乱顺序。 开关关闭时没有推送，重放就是唯一来源，进展只在每轮结束时更新。

## 3. 验证

`tests/unit/extensions/projects/hy_live/test_hy_live_acp_updates.py` 覆盖按 channel 和 `turn_id` 路由、解绑后丢帧；`test_hy_live_executor.py` 覆盖晚到的帧等待而不重放、推送不全只打 warning、没有推送时按顺序重放；runtime 侧见 `tests/unit/extensions/agents/test_installed_common_host_rpc.py` 与 `test_codebuddy_session_runtime.py`。

## 4. 已知限制

- **改裁剪规则要换沙箱代码**：`trim_acp_update` 在沙箱里运行，要重新打 runtime 镜像，或者用 worker venv 才能生效。 推送与重放都经 `Coordinator.on_update` 写 Event，写法相同。
- **丢帧不补**：推送丢了一部分帧时，这些帧对应的 Event 就缺了，只能从 warning 看到。
- **工具执行期间没有心跳**：一条挂死的命令和一条很慢的命令看起来一样，都是一条一直 `pending` 的记录。
- **CodeBuddy 私有字段**：会话标题由 CodeBuddy 在每轮最后一个事件里重发；非工具 update 上的 `agentPhase` 等 `_meta` 被裁掉。 CLI 升级后需要重跑第 3 节的单测并核对推送内容。
