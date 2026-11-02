# CodeBuddy 向用户提问：AskUserQuestion 与 ACP elicitation

> **代码位置**：打开、通知和收掉问题在 `hylive` 包（`hylive.coordinator.Coordinator`、`hylive.harness`、`hylive.acp_requests`）；nexus 这边 `hylive_shell.AcpRequestRelay` 把沙箱的请求交给 core、把回答送回沙箱，`coordination/backend.py::AcpSessionExecutor` 执行 ACP 原语。

> **边界**：CodeBuddy 主动向用户提问的链路。 沙箱侧能力要求 runtime 镜像支持 `client_questions`（见 [`runtime_image.md`](runtime_image.md)）；真链路验收见 example [`ask_user_question`](../../example/ask_user_question.md)。

## 1. 为什么要声明能力

CodeBuddy 只有在 ACP `initialize` 时看到下面两项，才会暴露自带的 `AskUserQuestion`、才会发 `elicitation/create`：

```json
{"clientCapabilities": {"elicitation": {"form": {}}, "_meta": {"codebuddy.ai": {"question": true}}}}
```

`AcpConnection.initialize` 默认仍发 `clientCapabilities: {}`，其它 agent 的握手不变。 只有 CodeBuddy host relay 的 `client_questions` 打开时才声明上面这两项（`host_relay.CLIENT_QUESTION_CAPABILITIES`）。 没声明时 E2E 里能看到 CodeBuddy 的 `ToolSearch` 报「No matching tools found for: AskUserQuestion」。

开关：

| 层 | 配置 | 默认 |
|---|---|---|
| 公共 `CodeBuddyMultiturnInput` | `client_questions` | `false`；打开时要求 `permission_policy='defer'`，向 host relay 下发 `extension_config.client_questions=true` |
| hy-live `AcpSessionExecutor` | `work_agent_configs.codebuddy.client_questions` | 有 Host 路由且 `permission_policy` 为 `defer` 时默认 `true`；显式写 `false` 关闭 |
| example CLI | `--client-questions` / `--no-client-questions` | 打开；镜像不支持时须显式 `--no-client-questions` |

`extension_config` 属于会话启动配置，同一会话每一轮都必须一样（`session_runtime` 会拒绝变化的启动配置）。 镜像的 host relay 不认识 `client_questions` 时会话直接起不来，报 `Unknown host_relay_extension config keys: ['client_questions']`，该 Run 记为失败并以 `failed` 通知小脑（不会静默降级），所以这类镜像上必须显式写 `client_questions: false`。

## 2. AskUserQuestion

CodeBuddy 把它当作一条 `session/request_permission` 发来，`toolCall._meta["codebuddy.ai/toolName"] == "AskUserQuestion"`，问题在 `toolCall.rawInput.questions[]`。 这不是权限。

```text
CodeBuddy ──session/request_permission──► AcpConnection (defer)
            AcpPermissionRelay ──acp_permission_request──► Host AcpRequestRelay ──AcpPermissionRequested──► core
                                                             Coordinator.on_permission_requested
                                                             └─ is_ask_user_question → Coordinator.publish_input_request
                                                                  harness.open_input（Run → needs_input）
                                                                  [通知] 执行中需要用户补充。工作：… 问题：…
用户回答 → 小脑 answer_task_input → 中脑 answer_task_input(accept|decline|cancel)
            Coordinator.answer_task_input → answer_question → AcpRespondPermission
Host AcpRequestRelay ──task_files:session_response:acp_permission {option_id, ext_requests}──► AcpPermissionRelay
            1. connection.extension_request('_codebuddy.ai/resolveInterruption', …)，等 CodeBuddy 回复
            2. 再回 session/request_permission
CodeBuddy 在同一回合里接着做，不另发 prompt。
```

- **问题文字**：各问题一行，有选项时写成「问题（可选：A、B）」（`hylive.acp_requests.question_text`），再按策略可见文本去掉凭据。
- **回答**：`_codebuddy.ai/resolveInterruption` 的参数是 `{sessionId, toolCallId, decision, answers?}`。 `accept` 时 `decision=allow`，`answers` 里每个 `q_i` 都填用户那句话；`decline` 和 `cancel` 时 `decision=deny`，CodeBuddy 当作跳过。 随后权限请求回 `allow_once` 选项（接受）或 `reject_once` 选项（其余），选项按 `optionId` 或 `kind` 匹配，找不到时回字面值 `allow-once` / `reject-once`。
- **只认 `client_questions` 会话里的 `_codebuddy.ai/resolveInterruption`**：沙箱只在 host relay 配了 `client_questions=true` 时处理 `ext_requests`；没配时整个键忽略（记 warning），只按 `option_id` 回。 配了时也只发 `_codebuddy.ai/resolveInterruption`，别的方法丢掉并记 warning，其余照常。 `ext_requests` 格式不对时直接按 `cancelled` 回。
- **扩展请求失败**：调用报错、超时（40 秒）或回复里没有为真的 `resolved` 时只记 warning，权限请求仍按 Host 给的选项答复。 当前 CodeBuddy 接受答案后返回 `{}`，没有 `resolved`，所以这条 warning 每次提问都会出现。 warning 里只有方法名、参数的键和错误类型，不含用户回答。

## 3. elicitation/create（form）

```text
CodeBuddy ──elicitation/create──► AcpConnection.elicitation_handler = AcpElicitationRelay.request
            ──acp_elicitation_request {channel_id, elicitation_token, params}──► Host AcpRequestRelay
                 ──AcpElicitationRequested──► core Coordinator.on_elicitation_requested
                 mode != form → 立刻回 {action: decline}
                 form → Coordinator.publish_input_request(message) → 同上的补充问题与通知
用户回答后 answer_elicitation → AcpRespondElicitation → Host ──task_files:session_response:acp_elicitation {channel_id, elicitation_token, result}──► 沙箱
            accept → {action: accept, content: 按表单回填；题面带选项，见下}
            decline / cancel → {action}
```

回填规则（`hylive.acp_requests.elicitation_content`）：题面把选项列出来（`问题（可选：A、B）`，多选 `（可多选：A、B）`）；带 `oneOf` / `enum` 的字段按选项精确或唯一子串匹配回填，数组取所有命中的选项；无枚举约束的 string 字段填用户原话。 任一必答题匹配不上时回 `{action: decline}`（日志只记题目 key，不记用户原话），不替用户编答案。

沙箱没有 elicitation handler、提问超过 `deferred_permission_timeout`、prompt 已经结束或连接关闭时，`AcpConnection` 回 `{action: cancel}`。 elicitation 与延后的权限请求用同一个 `deferred_permission_timeout`；hy-live 把它设成 `None`（`coordination/backend.py::_codebuddy_input`），沙箱不计时，只有 Host 回答、prompt 结束或 `session/cancel` 才会收掉。 不带 `sessionId` 的 elicitation 算在正在跑的 prompt 上：没有 prompt 时直接回 `cancel`，任一 prompt 结束时一并收掉。

**谁负责撤回**：沙箱放弃等待（计时到期、prompt 结束、连接关闭）时不会通知 Host，权限请求和 elicitation 都一样，没有沙箱到 Host 的撤回事件。 所以 Host 是开着的问题的唯一撤回方（见第 4 节），hy-live 必须保持 `None`；配了有限超时的调用方要接受 Host 那边的问题在沙箱已回 `cancel` 后仍开着，Host 迟到的回复会因 token 已用或未知而被忽略。

## 4. 回合结束、取消与预算

- **等用户的时间不计入 2000 秒工作预算**：问题开着时 `Harness.waiting_on_user` 为真，`TurnScheduler.ts_supervise` 暂停计时；墙钟上限 `timeout_s` 仍然计（见 [`timeout_and_cancel.md`](timeout_and_cancel.md) 第 0 节）。
- **取消 Run**（`Coordinator.stop_task`）、**工作预算到期**（`Coordinator.release_turn`）、**回合结束**（`Coordinator.on_turn_ended`）都会把这个 Run 上开着的问题按 `cancel` 收掉；Host 随即回 `deny` + `reject`（AskUserQuestion）或 `{action: cancel}`（elicitation）。 沙箱这边，prompt 结束或 `session/cancel` 时 `AcpConnection` 本来就会把同一会话还挂着的权限和 elicitation 回成 `cancelled` / `cancel`，Host 晚到的回复因 token 已用而被忽略。
- **问题来自已结束的 Run**：不通知用户，直接按 `cancel` 回。
- **等待方先登记**：`publish_input_request` 在通知小脑之前登记等待方，回答来得再快也不会丢。

## 5. 兼容性

权限回复里多出来的 `ext_requests` 键，不支持 `client_questions` 的镜像会忽略，支持的镜像在 `client_questions` 关闭时同样忽略，只按 `option_id` 回答；只有 Host 打开 `client_questions` 而镜像不支持时会话才会起不来（见第 1 节）。

## 6. 代码索引

| 主题 | 路径 |
|---|---|
| 声明能力、`elicitation/create`、`extension_request` | `nexus/extensions/agents/installed_common/acp.py` |
| `ext_requests`、`AcpElicitationRelay` | `nexus/extensions/agents/installed_common/host_rpc.py` |
| `client_questions`、`CLIENT_QUESTION_CAPABILITIES` | `nexus/extensions/agents/codebuddy/host_relay.py` |
| 扩展把能力与 handler 交给连接 | `nexus/extensions/agents/codebuddy/session_runtime.py` |
| 公共开关 `client_questions` | `nexus/extensions/agents/codebuddy/agent.py` |
| Host 转发请求、送回回答 | `hylive_shell.py`（`AcpRequestRelay`） |
| 识别提问、题面、回填 | `hylive.acp_requests`（`is_ask_user_question`、`question_text`、`elicitation_content`） |
| 打开问题、通知、收问题 | `hylive.coordinator`（`on_permission_requested`、`on_elicitation_requested`、`publish_input_request`、`answer_task_input`、`answer_question`、`answer_elicitation`、`notify_settled`） |
| hy-live 默认打开 | `coordination/backend.py`（`AcpSessionExecutor._run_turn`） |
| 单测 | HyLive `tests/test_acp_requests.py`、`tests/unit/extensions/projects/hy_live/test_hy_live_shell.py`、`tests/unit/extensions/agents/test_installed_common_{acp,host_rpc}.py`、`test_codebuddy_{host_relay,session_runtime,multiturn_agent}.py` |
| 真链路 case | [`../../example/ask_user_question.md`](../../example/ask_user_question.md) |
