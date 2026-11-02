# Gateway Direct-Trajectory H4：从一次请求到训练 Segment 的完整执行流程

> 固定实现：Nexus H4 code `1ae27d8b09d67c8d4b46e21e273b672f804ed852`；HyTuner H4 validation
> commit `d5633441787e78ec853421eefb394a8e62bb2a2c`，其 `third_party/nexus` 精确 pin 到上述 code；functional
> baseline `patrickkklu/gateway-core@6312f04f62a3a45440aa975ca9f53594cd93f63f`；public schema
> `gateway-trajectory-v1`。 本文按 H4 的真实调用顺序解释代码，不混入 H4 之后的 policy-extension 重构。 Production
> policy 只有 `generic` 与 `nexus_main_agent`。

## 阅读方式

本文只沿一条时间线前进。 每一步都同时回答四个问题：

1. 谁调用谁；
2. 输入和输出是什么；
3. live `Trajectory` 与 flat `RuntimeState` 此时怎样变化；
4. 示例中的 message、token、mask 最终落在哪里。

先记住唯一总原则：

```text
wire event
  → ScaffoldPolicy.inspect_event()
  → EventFacts
  → TrajectoryAssembler direct commit
  → live Trajectory + flat RuntimeState
  → isolated freeze
  → frozen Trajectory
  → Segment / legacy projection
```

没有 `AssemblyTree`，没有 `Materializer`，没有 policy node，也没有第二套 roots/parent/children。

## 贯穿全文的 executable example

下面不是随意编造的 token 表，而是 `tests/unit/gateway/test_direct_trajectory_oracle.py` 固定的最终行为 oracle。
它用小整数锁定 physical topology、logical links 与 Segment mask；H4 Nexus main-agent policy 不产生其中的 implicit
spawn/join facts。 Steps 10–12 会明确区分 production 的 resent-history 路径与显式 observation hook。 Oracle 中的
multi-agent facts 由测试 policy 显式产生，再由同一套 assembler/freezer 归约；H4 production branch 没有提交或注册
CC、Codex、Search、Hermes 或 OpenClaw policy，因此本文不把这些 scaffold 写成已经接入。

### Main lane

```text
n0 user "main prompt"              token account [10]
  ↓ physical parent
n1 assistant "spawn child"         token account [11,12]
                                    gen_start=1, gen_len=1
                                    sampled token = [12]
  ↓ physical parent
n2 tool "child returned"           token account [13,14,15]
                                    observation/context，非 sampled
  ↓ physical parent
n3 assistant "main final"          token account [16]
                                    gen_start=0, gen_len=1
                                    sampled token = [16]
```

### Child lane

```text
n4 user "child prompt"              token account [20]
  ↓ physical parent
n5 assistant "child final"          token account [21,22,23]
                                     gen_start=1, gen_len=2
                                     generation glue = [21]
                                     sampled tokens = [22,23]
```

### Logical relation，不是 physical merge

```text
child Chain:
  parent_chain_id = "main"
  roots           = [n4]
  spawn           = SpawnLink(node=n1, tool_use_id="spawn-1")
  join            = JoinLink(node=n2, kind="notify")
```

Physical forest 只有：

```text
roots = [n0, n4]
n0 → n1 → n2 → n3
n4 → n5
```

绝对没有 `n5 → n2`、`n5 → n3` 或 child leaf → main continuation 的 parent edge。

最终训练 payload：

| Segment | prompt_ids | response_ids | response_mask |
| --- | --- | --- | --- |
| child | `[20,21]` | `[22,23]` | `[1,1]` |
| main | `[10,11]` | `[12,13,14,15,16]` | `[1,0,0,0,1]` |

因此 child 返回 main 的 `[13,14,15]` 对 main Segment 是 **mask=0**。 它是下一次 main inference 看见的 observation，不是 main model 采样的 token。

## Step 0：HyTuner 选择 Gateway runner path

入口在 `nexus/runner/hytuner.py`：

```python
async def run(..., **kwargs):
    use_gateway = kwargs.pop("use_gateway", False)
    if use_gateway:
        return await _run_gateway(...)
```

`_run_gateway()` 先要求显式 `trajectory_id`，然后给 Nexus run 配置一个 per-run client：

```text
openai_clients["main"].name = "gateway"
client_args:
  tokenizer / processor / server_manager
  tool_parser / template_parser
  trajectory_id
  scaffold_policy = "nexus_main_agent"
  metadata = {trajectory_id, validate}
  max_model_len / token_budget
request_args:
  sampling_params
```

这里有三个重要边界：

- `scaffold_policy="nexus_main_agent"` 是显式注入，不存在 `None → nexus_main_agent`；
- `apply_chat_template_kwargs` 默认注入 `is_training=True`，与 legacy HyTunerClient render 对齐；调用方显式传入
  `is_training=False` 时后者优先；
- partial rollout 仍由 HyTuner `GeneratorWrapper` 所有，Gateway 不建立第二个 abort/resume 状态机。

同一函数还在 runner 层构造 `CompactionManager`，并挂载 `NexusRunTracker` span hooks。 Tracker 负责 task/runtime/tool/timing facts，不参与 message topology。

此时尚未创建 `GatewaySession`。 Session 是第一次 completion 到达时按精确 trajectory ID 懒创建的。

### 本步涉及的模块

| 模块 | 本步职责 |
| --- | --- |
| `nexus/runner/hytuner.py` | gateway/legacy path 选择、client spec、compaction、run lifecycle |
| `nexus/gateway/nexus_run_tracker.py` | 准备接收 Nexus spans 和 runtime facts |
| `nexus/gateway/__init__.py` | 对外只暴露 client、Trajectory、singleton manager |

## Step 1：构造 `GatewayOpenAIClient`、`Processor` 与 backend

`GatewayOpenAIClient.__init__()` 做两件事。

第一，选择 backend：

| client arg | backend | representation |
| --- | --- | --- |
| `server_manager` | `HyTunerServerBackend` | token-level |
| `openai_client` | `OpenAIAPIBackend` | message-level |
| 两者都没有 | 显式失败 | — |

旧 `llm_generator` 参数会被拒绝，因为 RPC/HTTP 都应通过 HyTuner-side generation endpoint。

第二，构造 `Processor`：

```text
Processor(
  backend,
  tokenizer,
  hf_processor,
  tool_parser,
  template_parser,
  apply_chat_template_kwargs,
  process_vision_info_fn,
  priority,
)
```

`tokenizer is not None` 决定 token lane；否则走 message lane。 这不是运行时每轮猜测。

`Processor` 同时读取 `GATEWAY_LEGACY_SEAM_COMPAT`：

- 未设置：优先 direct extension；必要时定位 closure seam；
- `=1`：每次 reused prefix 都采用 legacy canonical-prefix subtraction，用于 Qwen bitwise 接缝兼容。

这个环境变量只控制 prompt splice，不写进 `Trajectory`。

### `backend.py` 在这里统一了什么

`BackendResult` 是 Gateway 内部统一返回：

- token lane：`token_ids`、`log_probs`、`routed_experts`、`stop_reason`、top-k extra；
- message lane：`message`、usage/provider fields；
- shared：inference time、extra fields。

`HyTunerServerBackend` 只把 resolved request 投影为 vLLM sampling params，再调用一次：

```text
server_manager.generate_tokens(
  request_id,
  prompt_ids,
  sampling_params,
  multimodal data,
  priority,
  request_meta,
)
```

`VllmCompletionsBackend` 从 `token_id:<int>` wire tokens 恢复 IDs，并对 logprob/top-k 长度 fail loudly。 `OpenAIAPIBackend` 则保留 provider message/usage。

### `types.py` 在这里守住什么

Pydantic internal types 提前验证：

- `BackendResult` 的 token/logprob/top-k 行数；
- `PromptRender` 的 text partition 与 token partition；
- `ProcessorResult` 与 `GatewayCompletionResult` 的 lane-specific facts。

这些类型是内部 transport，不是另一份 trajectory IR。

## Step 2：Scaffold 发起第一次 completion

假设 main scaffold 提交：

```python
messages = [{"role": "user", "content": "main prompt"}]
request_meta = {"request_id": "r-main-1", "iteration": 0}
```

它调用：

```text
GatewayOpenAIClient.create_completion(messages, request_meta=..., ...)
  → GatewayOpenAIClient._create_completion(...)
```

Public `create_completion()` 刻意不使用基础 client 的 cache replay。 原因很直接：cached response 若在新进程返回，而 trajectory side effect 没有恢复，会产生“有模型输出、无轨迹记录”的训练损坏。

`_create_completion()` 按顺序处理：

1. 校验 call-site trajectory ID 与 client 绑定 ID 相同；
2. 合并 construction metadata 与 per-call request meta，per-call key 优先；
3. 把 configured request args、override、sampling params 和 call kwargs 解析成一份 `request_kwargs`；
4. 明确拒绝 streaming 与 `n != 1`；
5. 在唯一 chokepoint 对历史 tool-call arguments 做 non-mutating `str → dict` normalization；
6. 调用 `_session()`。

`wire.py` 的 copy helpers在这里及后续保留 wire 语义：

- `copy_wire_value(..., exclude_none=False)` 不把“字段缺失”和“字段显式为 null”合并；
- `copy_value(..., exclude_none=True)` 只用于不需要保留 null/absent 差别的 provider/plain facts；
- 所有 copy 都递归且不修改 scaffold 原对象。

## Step 3：SessionManager 懒创建 per-trajectory Session

Client 调用 process-wide singleton：

```text
session_manager.get_or_create(
  trajectory_id,
  processor=this_client.processor,
  metadata=...,
  scaffold_policy="nexus_main_agent",
)
```

`GatewaySessionManager` 的规则：

- 第一次 lookup 必须提供 Processor；
- 后续 lookup 可复用；
- 同一 trajectory ID 已属于另一 Processor 时显式报错；
- active map 只保存 `trajectory_id → GatewaySession`，不保存模型 I/O 状态。

`GatewaySession.__init__()` 创建：

```text
GatewaySession
├── TrajectoryAssembler(scaffold_policy="nexus_main_agent")
│   ├── live Trajectory(state="active")
│   └── RuntimeState
├── TokenBudgetState
├── RLock
├── in_flight counter
├── latest_messages
└── session timeline/metadata
```

`TrajectoryAssembler` 通过 `build_scaffold_policy("nexus_main_agent")` 解析显式 policy。 `None` 或未知字符串立即失败。
字符串解析会惰性 import `policies/__init__.py`；这个模块只注册精确的 `NEXUS_MAIN_AGENT_POLICY` 与 `GENERIC_POLICY`，
重复 ID 同样显式失败。直接传入满足 runtime Protocol 的自定义 `ScaffoldPolicy` 对象时不需要进入 built-in registry。

这一步之后的稳态内存是：

```text
one live Trajectory + one flat RuntimeState + scalar Session state
```

其中 `RuntimeState` 除 ledger 和只追加的 `llm_by_message_id` 索引外，只含 next node/message/request/fact sequence
等整数，以及至多一个等待下一次 output 消费的 `PendingBoundary`。 Pending claim 只带 rule 与 NodeId cross-reference，不形成 adjacency。

没有 frozen Trajectory，也没有 AssemblyTree。

## Step 4：Session 在锁内建立 recording/execution 隔离

`GatewaySession.create_completion()` 在拿到 session lock 后：

```python
recording_messages = deepcopy(messages)
execution_messages = deepcopy(messages)
request_snapshot = deepcopy(request_kwargs)
execution_request = deepcopy(request_kwargs)
recording_meta = deepcopy(request_meta)
execution_meta = deepcopy(request_meta)
```

为什么是两组：

- recording copy 是将来写入 Trajectory 的事实；
- execution copy 可以被 template/backend integration 消费或修改；
- backend mutation 不能倒灌到 audit facts。

Session 接着调用：

```text
TrajectoryAssembler.extract_context(recording_messages, tools, recording_meta)
TrajectoryAssembler.frozen_prefix(recording_messages, tools, context=request_facts)
```

最后把 `_in_flight += 1`，释放 lock，才进入耗时的模型 I/O。

这意味着 active snapshot 只能看到已经 commit 的 turn，不会看到“只有 request、没有 response”的半个节点。

## Step 5：H4 ScaffoldPolicy 直接解释 event

`TrajectoryAssembler.extract_context()` 先构造 immutable `RequestEvent`，然后直接调用：

```text
self._policy.inspect_event(event)
```

H4 没有独立的 `policy_inspector.py` 或 `policy_compiler.py`。 `ScaffoldPolicy` 是一个
`@runtime_checkable Protocol`，定义 policy 必须提供的属性和方法：

```python
class ScaffoldPolicy(Protocol):
    policy_id: str
    response_policy: ResponsePolicy
    block_terminator_ids: tuple[int, ...]
    message_key: str
    tools_key: str

    def inspect_event(
        self,
        event: RequestEvent | ResponseEvent | ObservationEvent,
    ) -> EventFacts: ...
```

`build_scaffold_policy()` 要求显式 policy：`None` 和未知名字立即失败；字符串从 built-in registry 解析，对象则必须满足
上述 runtime Protocol。 因此 H4 允许自定义 policy 对象或 `GenericScaffoldPolicy` subclass，它不像后续实现那样从类型
系统上禁止 override `inspect_event()`。

### 5.1 三种 event 与临时 anchor

Assembler 传给 policy 的 event 已经是 wire copy：

| Event | policy 看到的 message | 可用 anchor |
| --- | --- | --- |
| `RequestEvent` | 全部 input messages | `InputMessage(index)` |
| `ResponseEvent` | 原 request + 当前 output message | `OutputMessage()` |
| `ObservationEvent` | 当前 observation，可带 parent messages | `ObservationMessage()` |

这些 anchor 解决“policy inspection 时真实 NodeId 还不存在”的问题。 Commit 后，Assembler 才把 unbound anchor 绑定到
真实 NodeId，并把 typed facts 写入 flat ledger。

### 5.2 Generic message 与 tools identity

`GenericScaffoldPolicy.inspect_event()` 先按 event 类型生成 identity。 `protocol_message_identity()` 固定投影：

```text
role / content / tool_calls / tool_call_id / name
```

`tool_calls[].function.arguments` 若为合法 JSON，会递归 canonicalize object key order；provider-only decoration默认不进入
identity。 原始 wire message仍由 Assembler 单独 lossless copy，identity normalization 不覆盖 `original_message`，也不
修改将来发回 scaffold 的 message。

Tools 使用完整 wire value 的 deterministic hashable projection。 Prefix reuse 必须 message semantic identity 与 tools
identity 同时匹配。

### 5.3 Generic metadata 与 timeline

Generic 固定读取：

```text
request_id / agent_id / iteration / retry / sequence|seq
```

它为 Generic main lane 生成 context key：

```text
(policy_id, "main", False)
```

Response/observation timeline统一为：

```text
event_type, request_id, agent_id, parent_agent_id,
is_sidechain, start_time, end_time, duration, status,
span_id, parent_span_id, iteration, retry, sequence, attributes
```

没有 timing/facts 时 timeline为空，不制造空事件。

### 5.4 Nexus 在 Generic facts 上增加 declared-chain 与 compaction

H4 `NexusMainAgentScaffoldPolicy` 继承 Generic，并 override `inspect_event()`：

```python
class NexusMainAgentScaffoldPolicy(GenericScaffoldPolicy):
    def inspect_event(self, event):
        facts = super().inspect_event(event)
        # 读取 Nexus wire metadata/messages
        return replace(
            facts,
            context_key=...,
            membership=...,
            declared_claim=...,
            boundary_facts=...,
            timeline=...,
        )
```

它只增加当前 Core 已启用的 Nexus main-agent 语义：

- `request_meta.declared_chain` → `DeclaredChainClaim`，非 main kind 同时产生 `SidechainMembership`；
- `request_meta.invocation_kind == "summary_inference"`；
- `request_meta.invocation_kind == "compaction_resume"`；
- 最右侧带 `_is_compacted_summary` 的 input message。

显式 invocation boundary 锚定 request edge；空 request 的 response boundary 锚定 `OutputMessage()`；compatibility
marker锚定真实 input index。 Nexus 还把 context key改为：

```text
("nexus_main_agent", chain_id, membership_exists)
```

并在已有 timeline 中更新 `is_sidechain`。 H4 Nexus main-agent policy 不产生 implicit spawn、retrieval、delivery 或 notification facts。

#### 请求级 invocation 与预算分类

`EventFacts` 和 `ScaffoldContext` 中的 `invocation_kind` 表示 **本次 LLM 调用的目的**，不是历史 Segment
的类型。 公共 policy 将显式 `request_meta.invocation_kind="summary_inference"` 规范化为
`InvocationKind.SUMMARY_INFERENCE`；其他非空字符串标签（包括 `completion`、`worker`、`compaction_resume`）
保留原有普通推理语义，规范化为 `InvocationKind.AGENT_INFERENCE`。 缺省值同样是普通推理，非法类型和空字符串显式报错。

HyCode 在缺少显式 invocation metadata 时复用已有 compaction boundary detector：只有 summary trigger 位于
**当前请求的最后一条消息** 时，才把本次调用归为 summary。 历史 trigger、继承的父 agent summary，以及
compact 后恢复工作时的 summary 内容，都不能单独把普通调用变成 summary。 显式普通调用标签与当前 summary
trigger 冲突时直接报错，避免轨迹与预算采用不同解释。 `invocation_source` 保留 `request_meta`、
`hycode_summary_inference` 或 `default` 等来源，原始 metadata 不被改写。

`TrajectoryAssembler.extract_context()` 保存分类快照；同一请求的后续 event facts 复用这个结果。
切段同样使用该分类：当前 summary 调用必须得到 `summary_inference` boundary，即使 policy 只提供了历史
resume boundary 或没有额外 marker；已有 summary boundary 的规则和证据保持不变。
`GatewaySession` 直接用该快照选择 `COMPACTION_OUTPUT` 或 `AGENT_OUTPUT`，不再重新读取原始 metadata 猜测。
排除 summary 输出消费不影响独立的 observation 计账，也不免除上下文占用或 segment token 统计。
本次公共接口与 HyCode 适配不包含 Codex detector 的移植。

### 5.5 Generic 与 Nexus 对象生命周期

Production 创建两个进程级无状态对象：

```python
GENERIC_POLICY = GenericScaffoldPolicy(policy_id="generic")
NEXUS_MAIN_AGENT_POLICY = NexusMainAgentScaffoldPolicy()
```

`policies/__init__.py` 显式注册二者；重复 ID 直接失败。 Policy 对象共享，但 `GatewaySession`、`TrajectoryAssembler`、live
`Trajectory`、`RuntimeState` 和 freeze snapshot 都按 trajectory/session 隔离。 Policy ID 不进入 public dump。

### 5.6 EventFacts 与 anchors validation

Policy 返回 `EventFacts` 后，Assembler 再检查：

- Request 恰好为每个 input index 产生一个 identity；
- Response 只能使用 `OutputMessage`；
- Observation 只能使用 `ObservationMessage`；
- 所有 spawn/retrieval/delivery/notification/child-prompt/boundary anchors 都属于本 event。

错误 anchor 立即失败，不允许一个 policy fact 悄悄指向另一条历史。 H4 的保护边界在这里：policy 可以自定义
`inspect_event()`，但它不能用非法 anchor 越过当前 event。

## Step 6：第一次请求没有 frozen prefix

树上还没有这个 context 的 LLM Node，所以：

```text
frozen_prefix = None
```

Context key 是：

```text
(policy_id, lane_key, membership_exists)
```

第一次 main Nexus request 通常是：

```text
("nexus_main_agent", "main", False)
```

Declared child 则例如：

```text
("nexus_main_agent", "child", True)
```

Frozen prefix 只复用本 context 已经用同一份 tools 发出过、并得到 completion 的路径（见 12.1），因此即便 messages
字面相同，也不会续用另一条 lane 采样出的 token。 没有“最后一个节点”之类的 cursor：选哪段历史只取决于树和
当前请求。

## Step 7：Processor render、tokenize 与 budget

Session 离开 lock 后调用：

```text
Processor.create_completion(
  request_id,
  execution_messages,
  execution_request,
  frozen_prefix,
  prepare_generation=TokenBudgetState.prepare_generation,
  request_meta,
)
```

### 7.1 Token lane render

`Processor.render_and_tokenize()`：

1. 调用 tokenizer/template parser 取得完整 canonical text；
2. `_build_prompt_render()` 为每个 message 分配 text/token span；
3. 对整个 tail canonical encode，再把 IDs bookkeeping 回 message spans；
4. 产出 `(raw_prompt_ids, prompt_ids, RequestMultiModal, PromptRender)`；带图时再由
   `_build_expanded_prompt()` 展开训练视图。

结构化 template parser可直接提供 role spans；否则 Processor 用 monotone prefix rendering发现 spans。 若模板不暴露稳定 per-message prefix，代码保留 exact whole prompt，并保守地把 fused input tail归到最后一个 message。 账本边界可以保守，模型实际看到的 token stream 不能变化。

图片的 expanded-token 逐 message ownership 由 `partition_expanded_ids()` 按 placeholder run 划分；
非图片模态（video/audio）目前显式 `NotImplementedError`。

### 7.2 第一次 main prompt

Oracle 把完整 prompt缩成：

```text
input message n0 owns [10]
generation prompt/glue owned by n1 = [11]
backend raw_prompt_ids = [10,11]
```

### 7.3 TokenBudgetState

Budget state按 context记录：

- initial prompt length；
- trajectory-wide generated consumption；
- context-local segment consumption；
- last accounted prompt IDs。

它以真实“frozen sampled stream + rendered tail”为尺，不对 messages 重新 canonical render 后猜长度。 Prefix不再 extension 时会重置该 context 的 initial prompt baseline；remaining ≤ 0 抛 `ContextLengthExceededError`。

Budget只影响本次执行的 effective `max_tokens`，不写入 `TokenAccount` 或 Trajectory schema。

## Step 8：Backend 返回真实 sampled facts

对 HyTuner path：

```text
Processor
  → HyTunerServerBackend.create_completion(raw_prompt_ids=[10,11], ...)
  → GeneratorWrapper.generate_tokens(...)
  → TokenOutput(token_ids=[12], ...)
  → BackendResult
```

Backend返回的 `[12]` 是唯一 sampled truth。 Processor：

1. 用 tokenizer decode token IDs；
2. 用 tool parser解释 assistant content/tool calls；
3. 保留 `raw_response`/reasoning/tool facts；
4. 返回 `ProcessorResult`。

它不会从 response text再 encode得到 `[12]`。 Logprobs/top-k 必须与 sampled span严格对齐。

若生成的 assistant message 发起 child：

```python
{
  "role": "assistant",
  "content": "spawn child",
  "tool_calls": [{
    "id": "spawn-1",
    "function": {"name": "...", "arguments": {...}},
  }],
}
```

Production Nexus main-agent policy 只会把它作为普通 assistant wire message；后续 scaffold 若要支持 implicit multi-agent，
需要在自己的 `inspect_event()` 中识别 spawn tool 与 prompt argument，并返回带 `OutputMessage()` anchor 的
`SpawnIntent`。跨事件匹配、parent scope、delivery precedence 与最终 link 仍由通用 freeze reducer 负责。

## Step 9：Session 回锁并 direct commit 第一次 turn

模型 I/O 返回后，Session重新拿 lock：

1. `TokenBudgetState.record_generation(context, raw_prompt_ids, [12])`；
2. `TrajectoryAssembler.record_completion(...)`；
3. 更新 latest messages/session timeline。

Assembler 的 commit 是整个 Direct-Trajectory 核心。

### 9.1 Request sequence

`next_request_sequence` 先分配本次 occurrence序号。 它只用于 variants/audit，不等于 NodeId。

### 9.2 `_find_or_create_path()`

对 request input逐个处理：

1. 以 parent + semantic identity找候选 children；
2. message lane优先同 context候选；
3. token lane额外要求 `templated_message` 与 `token_account.token_ids` 完全一致；
4. 未命中时直接创建 public `TrajectoryNode`；
5. 命中但 wire message不同则写 `as_sent_variants`；
6. 每个 occurrence的 typed facts绑定到这个真实 NodeId。

第一次请求创建：

```text
Trajectory.roots = [0]
Trajectory.nodes[0] = n0(user)
n0.parent = None
n0.children = []
RuntimeFactsLedger.by_node[0] = NodeRuntimeFacts(...)
```

`_new_node()` 是唯一 physical topology写入口：

```text
parent is None → append Trajectory.roots
otherwise      → append parent.children
```

不存在 parallel runtime adjacency。

### 9.3 Prompt reconstruction assertion

Assembler计算：

```text
input path token IDs [10]
+ PromptRender.generation_prompt_ids [11]
= token_training.prompt_ids [10,11]
```

不相等立即失败。

### 9.4 Response event inspection

Assembler为 output构造新的 `ResponseEvent`。Policy 重新计算 output message identity和 timeline；Session提供的
`ScaffoldContext` 只复用 lane/request ordinals，不复用 message identity或 anchored facts。

### 9.5 Boundary merge

输出 node的 boundary按以下来源合并：

```text
request policy boundary
+ response policy boundary
+ pending explicit finalize_segment boundary
+ physical continuity boundary
```

Semantic kind优先，physical causes/evidence保留。 Incompatible semantic kinds无法消歧时显式失败。

### 9.6 创建真实 output node

创建 n1：

```text
n1.parent = n0
n0.children = [n1]
n1.generated_by = "llm"
n1.original_message = assistant wire message
n1.templated_message = generation_prompt_text + completion_text
n1.token_account = TokenAccount(
    token_ids=[11,12],
    gen_start=1,
    gen_len=1,
)
```

并写入：

- resolved request kwargs snapshot；
- per-call input message overrides；
- provider model/response/usage/finish/timing/preemption；
- invocation iteration/retry；
- node timeline；
- boundary（若有）。

`_new_node(n1)` 同时把 n1 登记进 `llm_by_message_id`：回传的 assistant 消息继承 n1 的 message_id，物理连续性检查
靠它从回传节点找回 n1。

现在 live state已经是 public physical path：

```text
roots=[0]
n0.children=[1]
n1.parent=0
```

## Step 10：按显式 policy facts记录 child execution并准备 logical correlation

多 agent scaffold 可以单独提交 child request。 H4 Nexus main-agent policy 可从显式 `request_meta.declared_chain` 得到：

```text
SidechainMembership(
  chain_id="child",
  kind="worker",
  parent_chain_id="main",
  detector="declared_chain",
)
```

Nexus production 到这里不会产生 implicit spawn/join。 H4 的测试或 downstream custom policy 可以在自己的
`inspect_event()` 中把 scaffold wire protocol显式翻译为：

```text
SpawnIntent(anchor=OutputMessage(), tool_use_id="spawn-1", child_prompt="child prompt")
ChildPromptCandidate(anchor=InputMessage(...), value="child prompt", order=0)
NotificationDelivery(anchor=ObservationMessage(), agent_id="child")
```

这里的 H4 边界是：policy 只产生 event-local typed facts，不决定最终 Chain、spawn/join node 或 physical parent。
Assembler 在 commit 后把 anchor 绑定到真实 NodeId；exact-before-prefix、multi-root intersection、parent scope、delivery
precedence 与 ambiguity preservation 仍全部由通用 freeze reducer 完成。 Custom policy 必须自己保证只在真正的 child
wire event 上产生 `ChildPromptCandidate`；H4 尚未从结构上强制这一点。

如果 child history没有共享 physical prefix，Assembler创建第二个 root n4，再创建 child output n5：

```text
Trajectory.roots = [n0, n4]
n4 → n5
```

Oracle token ownership：

```text
n4 token_ids = [20]
n5 token_ids = [21,22,23]
n5.gen_start = 1
n5.gen_len = 2
```

所以：

```text
Trajectory.prompt_ids(n5) = n4.tokens + n5.glue = [20,21]
n5.generated_ids = [22,23]
```

此时 live `Trajectory.chains` 仍可为空或未最终解析；membership/spawn/delivery facts只以 ordered occurrences绑定在 ledger节点上。 跨 branch correlation留到 freeze，因为只有看到完整 forest后才能正确处理 parallel/nested/late delivery。

### Typed facts为何不是第二套 IR

Ledger occurrence形如：

```text
node_id → [
  (sequence, SidechainMembership(...)),
  (sequence, SpawnIntent(BoundAnchor(node_id), ...)),
  ...
]
```

它没有：

- runtime node object；
- message副本；
- parent/children；
- roots；
- path/segment列表。

Fact的 `BoundAnchor(NodeId)` 只是指向 public node的 cross-reference。

## Step 11：Tool observation 回到 main——production resent history 与可选显式 hook

当前 Nexus production 唯一的 `record_observation()` 调用者是 `nexus_run_tracker`，只用来补记 run 结束后最后一次
completion 之后的 terminal observation。 平常的 tool/env output 随下一次 completion 的完整 request history 一起进入 Gateway；`record_completion()` 的 request-path commit 会直接创建/复用 observation node，
并用同一次真实 `PromptRender` 赋予 token ownership。

Gateway 仍暴露显式 observation hook，供测试、control-plane integration 或 detached/downstream adapter 在下一次
completion 前预先登记 wire observation：

```python
client.record_observation(
    message={"role": "tool", "content": "child returned"},
    parent_messages=[... main history through n1 ...],
    request_meta=...,
)
```

调用链：

```text
GatewayOpenAIClient.record_observation
  → GatewaySession.record_observation
  → TrajectoryAssembler.record_observation
```

Assembler首先拒绝显式 `token_ids`：

```text
observation token IDs are derived from the next rendered prompt
```

这是关键事实：tool content什么时候、以哪些 tokens进入模型，只能由下一次真实 chat-template render证明。

### Parent选择

`parent_messages` 必填：按 identity + context 精确匹配 path，0 个或多个都失败。 不提供 `parent_messages` 直接报错，
不会隐式接到某个“最后节点”后面。

显式 hook 路径中，oracle 的 n2先被接到 n1：

```text
n1.children = [n2]
n2.parent = n1
n2.generated_by = "observation"
n2.token_account = None          # 此刻仍没有下一次真实 render
```

Observation event可产生：

- delivery receipt；
- notification delivery；
- timeline。

若同 identity child已经存在则 merge facts，不复制 topology。

Production resent-history 路径不会经历这个 `token_account=None` 中间状态：第二次 request commit n2时，真实 render
已经可用，因此创建 n2的同时写入 `[13,14,15]`。 两条路径得到相同 parent、message、token account与最终 mask，
但 provenance不同：resent-history新建 input node为 `generated_by="client"`，显式 hook保留
`generated_by="observation"`。

### Terminal observation

显式 hook预登记后，若 episode在 observation后直接结束，n2保留 wire message，但 `token_account=None`。 Segment
不虚构 token。 只有下一次 request render覆盖它时，它才获得 templated/token span。

## Step 12：第二次 main completion复用 shared prefix并提交 observation tail

Main scaffold下一次提交：

```text
[n0 user, n1 assistant, n2 tool]
```

此时 production 路径尚未在 live topology中创建 n2；显式 hook 路径已经有一个无 token account 的 n2。 两者的
n2 都还没有 token account，所以 request 前的 frozen prefix 只覆盖 n0–n1，n2属于本次 canonical tail。

### 12.1 Prefix authorization

`TrajectoryAssembler.frozen_prefix()`：

1. 重新 inspect当前 request；
2. 收集本 context、同一 tools identity 的 LLM Node（这里是 n1）路径上的节点：`{n0, n1}`；
3. 从根开始按当前 identities 逐层找子节点，只保留上一步集合里、已有 templated span 和 token account 的节点；
4. 取能走到的最深路径 `[n0, n1]`；长度相同时先比请求原文逐字相等的条数，再取 node_id 最大的；
5. `FrozenPrefix.node_ids` 记下这条路径，记录时前 `count` 条消息直接挂到这些节点上，不再重新匹配。

命中：

```text
FrozenPrefix.count = 2
FrozenPrefix.node_ids = (0, 1)
FrozenPrefix.message_token_ids = ([10], [11,12])
FrozenPrefix.text = recorded n0 span + recorded n1 span
```

注意：reuse的是 `[10,11,12]` 这条真实 sampled physical stream，不是重新 render n1 text后猜出的 IDs。

### 12.2 Chat-template/external prefix被改写时

假设第一次 sampled assistant bytes含：

```text
{"a": 1}TAIL
```

而下一次 canonical re-render把它格式化成：

```text
{"a":1}TAIL
```

Generic semantic identity仍相同。 Processor发现 canonical text不 byte-prefix frozen text时：

1. 记录 splice divergence warning；
2. 保留 recorded sampled bytes；
3. 从 canonical rendered prefix提取新 tail；能定位 closure seam时把它补到 tail头部；
4. backend收到 `frozen sampled bytes + tail`。

因此无损外部改写不会错误分叉，也不会把模型历史悄悄换成 retokenized bytes。

若 frozen tail无法在 isolated rendered prefix中定位，例如 sampled terminator与重渲染 terminator不同，代码使用
空 seam并继续复用 frozen sampled stream；只要完整 request仍以 canonical rendered prefix开头，tail依然可确定，
不会因此 reset budget或制造 physical restart。 若完整 request连 canonical prefix都不扩展，则显式报错。

若 tools、context或实际 realization确实不同，Assembler不会强行共用 node：

- 相同 semantic message可共享 `message_id`；
- 不同 templated bytes/token IDs拥有不同 `node_id` realization；
- output边界标记 `render_restart` / physical cause；
- canonical Segment安全切分。

H4 Generic identity 还会忽略 provider-only decoration，并 canonicalize `tool_calls[].function.arguments` 中合法 JSON 的
object key order。例如 `{"b": 2, "a": 1}` 与 `{"a": 1, "b": 2}` 可以授权复用同一个 semantic message；第一次与
后续真实 wire 仍分别保存在 `original_message` / `as_sent_variants`，不会覆盖审计事实。

需要精确区分两类“前缀变化”：

- wire message 语义相同、但 chat template 重渲染 bytes 改变：Processor 保留 sampled stream并安全拼接 tail；若 token
  realization 确实不兼容，则创建同一 `message_id` 的新 physical realization并标记 `render_restart`；
- wire `role/content/tool_calls/tool_call_id/name` 本身改变：Generic/Nexus 视为新的 semantic identity。若某个后续
  scaffold 有可忽略的私有 content annotation，H4 需要该 scaffold 在自定义 policy 中显式提供等价 identity；H4
  built-in Nexus 没有一套声明式 annotation matcher。

所以 H4 不会把 chat-template 或外部 render 差异错误归为新的对话语义，也不会强行共用不兼容的 token realization；
但它同样不会擅自忽略真实 content 差异。`wire.py` 只负责 lossless copy，identity canonicalization 在
`scaffold_policy.py`，splice/restart 决策在 `processor.py` 与 Assembler。

### 12.3 第二次 prompt与 observation token ownership

Oracle第二次真实 render把 n2归为：

```text
n2.templated_message → token_ids [13,14,15]
```

generation glue为空，backend prompt：

```text
[10,11,12,13,14,15]
```

Backend采样 main final：

```text
[16]
```

显式 hook 路径会把 previously unowned leaf observation n2补全：

```text
n2.token_account = TokenAccount([13,14,15], gen_len=0)
```

Production resent-history 路径则在同一 request commit中直接创建带上述 account 的 n2。

再创建 n3：

```text
n3.parent = n2
n3.token_account = TokenAccount([16], gen_start=0, gen_len=1)
```

这就是 main Segment最终 mask：

```text
12      → main sampled       → 1
13,14,15→ observation/glue   → 0,0,0
16      → main sampled       → 1
```

## Step 13：Shared physical prefix跨 Chain时不重复、不误分叉

另一个固定 oracle覆盖：

```text
n0 shared system → n1 shared user → n2 shared assistant
                                      ↓
                                    n3 child continuation user → n4 child assistant
```

第二次 request通过 Nexus `declared_chain` 声明 child，但 history复用了 main的 n0–n2。

Freeze后：

```text
main.roots  = [n0]
child.roots = [n3]
child.evidence.anchor_node = n2
```

Node membership：

```text
n0,n1,n2 → main
n3,n4    → child
```

没有复制 n0–n2，也没有从 root重新造一条 child tree。 Child Segment的 prompt可以包含共享 prefix，但 shared LLM n2只由 main Segment拥有一次 sampled target。

Token fixture：

| Segment | prompt | response | mask |
| --- | --- | --- | --- |
| child@n4 | `[10,11,12,13,14,15]` | `[16]` | `[1]` |
| main@n2 | `[10,11,12]` | `[13]` | `[1]` |

这里 n2的 training ownership首先由 Chain partition决定：n0–n2属于 main，child ownership从 n3开始，所以
child selected path只把 n4当 target；`prompt_ids(n4)` 仍沿 physical ancestors读取 n0–n3，因而共享 prefix完整保留。
Global first-visited dedup是更一般的重叠路径保护，但不是这个 fixture中 n2归 main的原因。

如果一段最初被声明为 synthetic child，随后 main沿同一 shared path继续，resolver会把共同 branch归回 main，而不是保留一个没有独立 sidechain ownership的假 child。

## Step 14：Compaction只开新 Segment，不改变 Chain身份

Nexus main-agent policy 直接识别三种 marker并产生 `BoundaryFact`：

1. `request_meta.invocation_kind="summary_inference"`；
2. `request_meta.invocation_kind="compaction_resume"`；
3. rightmost input message含 `_is_compacted_summary=True`。

前两种 invocation marker 的 `BoundaryFact` anchor 指向 request edge；兼容 marker 则指向最右侧带标记的真实 input，
并单独记录它是否恰好位于 request edge。 Assembler先把临时 anchor绑定到真实 input NodeId，用 claim防止同一历史
重复触发，再在真正 output LLM node创建后把合并后的 boundary附着到 output：

```text
SegmentBoundary(
  starts_new_segment=True,
  kind="summary_inference" | "compaction_resume",
  causes=("compaction",),
  evidence={"boundary_rules": [...]},
)
```

Boundary不会：

- 切断 physical parent；
- 改变 logical chain ID；
- 保存 Segment node列表；
- 创建 summary → resume content lineage edge。

### Full rewrite

若 compacted request完全重写 history，它可以形成新的 physical root。 同一 main Chain拥有多个 roots。

### Prefix preserving

若 compacted request保留旧 prefix，新 path可从旧 forest中间分叉。 仍只有一份共享 nodes。

### Oracle crossing

把 main final n3标记 `compaction_resume` 后：

```text
main conversation segment response = [12]
main compaction_resume response     = [16]
```

Observation `[13,14,15]` 仍是新 segment prompt context，不会把旧 response `[12]` 再训练一次。

`finalize_segment()` 提供显式 pending boundary；它只在下一次 output commit时消费。 没有 output就不会凭空产生 Segment。

## Step 15：请求 snapshot 或 finish

### Active snapshot

`GatewaySession.get_trajectory()` 在同一 `RLock` 内联合复制：

```python
deepcopy((
    segment_assembler.trajectory,
    segment_assembler._runtime_state,
    session_timeline,
    latest_messages,
    metadata,
    ended,
    termination_reason,
))
```

然后对 copied pair调用：

```text
freeze_trajectory(..., copy_inputs=False)
```

所以 snapshot只包含已经 commit的状态，并与后续 live mutation深隔离。

### Finish

`GatewaySessionManager.end_session()`：

1. 精确查找 trajectory ID；
2. 调用 `session.end()`；
3. 仅在 end成功后从 active map移除。

`session.end()` 若 `_in_flight > 0` 显式失败；否则在 lock 内调用
`get_trajectory(state="completed", copy_inputs=False)`，对 live `Trajectory` 和 `RuntimeState` 原地 freeze，写入
terminal facts，再返回这份 live 中间对象，不复制整条 trajectory。

这个流程保留非事务式错误语义。 它没有 rollback，也没有 silent fallback。

## Step 16：为什么 active snapshot 需要 copied Trajectory + copied ledger

### 稳态

```text
live Trajectory + flat ledger/cache
```

### Snapshot瞬间

```text
live Trajectory + live flat state
          +
copied Trajectory + copied flat state
```

### Snapshot返回后

```text
live Trajectory（session继续时）
          +
frozen Trajectory（调用者持有）
```

Copied runtime state只服务 active snapshot 的 freeze，随后释放。Finish 路径直接消费 live pair，不创建这份 O(T) copy。

Freeze不再执行 plain-data dump round-trip，因此 active snapshot 的峰值是 live pair 加 copied/frozen pair。
若调用者长期持有多个 active snapshots，也会累积返回的 frozen副本。 但它不是旧 AssemblyTree：

| 旧双结构 | 当前 snapshot copy |
| --- | --- |
| steady state长期维护不同 schema | 只在 active snapshot 暂时复制 |
| 必须 tree → Trajectory逐节点转换 | copy本身已经是 public Trajectory |
| 两边都含 topology/message事实 | sidecar不含 topology/message |
| 可能出现两套 parent/children漂移 | copied pair来自同一 lock时刻，freeze不建 shadow adjacency |

Freeze copy的价值是：

- 后续 completion不能修改调用者snapshot；
- structural validation/selection/chain resolution不会污染 active session 的 live state；
- active snapshot可重复、可通过 pickle 序列化；
- finish原地得到自包含对象，不带 runtime back-reference。

## Step 17：Freeze phase 1——验证真实 physical forest

`trajectory_freeze.freeze_trajectory()` 首先从 public：

```text
Trajectory.roots
TrajectoryNode.children
TrajectoryNode.parent
```

按 roots/children insertion order DFS 做验证，但不据此生成新 ID。

它验证：

- 所有 referenced node存在；
- 每个 node只被到达一次；
- `parent` 与 traversal expected parent一致；
- 没有 unreachable nodes；
- ledger node set与 Trajectory node set完全相同。

`NodeId` 在 runtime `_new_node()` 时按创建时序分配，后续 active snapshot、重复 export 和 finish 都保持不变。
验证阶段还会检查 runtime fact ledger 的 node set，并要求 bound fact anchors 与 boundary claims 均指向真实
node。

`TokenAccount` 的 `gen_start`、`gen_len` 和 token-aligned numeric shapes 在对象构造时由
`TokenAccount.__post_init__()` 验证；freeze 不通过 dump 重复验证这些字段。

`message_id` 不改。 它继续表示在线 occurrence创建顺序，后面 delivery precedence依赖它。

## Step 18：Freeze phase 2——LLM sibling selection

Freezer遍历：

```text
Trajectory.roots
每个 node.children
```

只对同一个 physical sibling group中的多个 LLM nodes处理 selection，再按 context key分 stream。

Policy：

- `KEEP_ALL`：保留每个 candidate；
- `PREFER_EXTENDED_ELSE_LAST`：优先已有 children的 extended response，否则最后一个。

只有 multi-LLM sibling group写 `selected/rejected_sibling`；单 LLM child保持 `None`。 Rejected nodes仍留在 topology和 chain partition，只是不从 rejected leaf发起当前 training correlation/export。

## Step 19：Freeze phase 3——Chain membership与roots

Freezer先建立普通 main Chain：

```text
Chain(chain_id="main", kind="main")
```

对每个 physical branch，它检查完整 subtree中的 LLM membership facts：

- 含 main LLM的 branch不能整体冒充 sidechain；
- sidechain branch的每个 LLM都必须有一致 membership identity；
- declared agent ID成为 chain ID；
- anonymous sidechain使用 collision-safe `chain@<root>`；
- 相同 agent ID但 kind/parent/detector冲突时加 `@<root>` 避免碰撞。

`Chain.roots` 是 ownership entry，不必是 physical root。 Shared-prefix child示例中 child root就是 n3；它的 physical parent n2仍属于 main。

Nested child沿 physical traversal继承当前 ownership；新的 membership在中途开新的 Chain root。

## Step 20：Freeze phase 4——spawn/join correlation

这一步只读取 NodeId-bound scalar facts和 public forest。

### 20.1 Candidate范围

- 从 non-rejected child chain leaf path发起 resolution；
- source spawn/retrieval/delivery candidate可来自 branch外完整 forest；
- candidate按 node `message_id` 保持真实 occurrence时序。

Policy 可以从完整 event 收集 spawn/retrieval/delivery evidence，但应只为真实可 spawn 的 child lane 产生
`ChildPromptCandidate`。因此这里消费的是 policy 已经产生并由 Assembler 绑定的 flat facts，不会重新解释 scaffold
wire marker。H4 依靠 policy 实现与 tests 维护这个约束，没有 class-level membership rule compiler。

### 20.2 Spawn prompt matching

对 child prompt：

1. exact prompt match；
2. exact为空才用 prefix relationship；
3. 多 root/leaf各自得到候选；
4. 有公共交集就只保留交集；
5. 交集为空才保留 union ambiguity。

不会因为“列表第一个”碰巧出现就绑定。

### 20.3 Parent precedence

依次取第一个非空 rung：

```text
declared parent
→ local parent-agent claim
→ spawn node所属 chain
→ join node所属 chain
→ structural parent
→ external parent-agent
```

若该 rung有多个候选，`parent_chain_id=None`，候选进入 evidence。

### 20.4 Parent scope后重算

得到 local parent后，spawn/retrieval/receipt/notification重新按 ownership scope。 若原 spawn存在但不在 chosen parent：

```text
typed spawn link清除
evidence.spawn_parent_mismatch = true
```

### 20.5 Delivery reducer

支持：

- inline receipt；
- poll retrieval + receipt；
- notification。

同 tool call的 receipt先按最新 `message_id`，同 message/node再按 fact sequence。 同一最新 message内 delivered true/false冲突时保留 `delivery_ambiguous`，不选一个。

最终 Join从所有成功 candidate中选择最新 `message_id`。 若同一最新 message仍有多个 node/kind/tool ID，typed fields只填写可唯一确定的部分。

Oracle的逻辑结果：

```text
spawn.node = n1
join.node  = n2
join.kind  = "notify"
```

这些都是 cross-reference，不改变 n1/n2/n5的 parent。

## Step 21：Freeze phase 5——timeline 与 terminal facts

`_attach_direct_session_events()` 用 `(start_time,end_time)` 把 Session-level inference timing合并到唯一 matching LLM node。 无唯一 candidate的事件保留在 trajectory-level timeline。

然后设置：

```text
session_id
state
latest_messages
termination_reason
metrics
extra
```

设置完成后直接返回 typed `Trajectory`，不调用 `dump()` / `from_dump()`。 Physical forest 和 runtime reference
约束已经在 phase 1 验证；JSON-safe encoding、dump schema 和 dump-only cross-reference 约束仅在调用者显式执行
`Trajectory.dump()` / `Trajectory.from_dump()` 时验证。 这使 typed pickle/Ray 路径可以携带 plain-data schema
不支持的 processor-native 对象，同时保持 dump-compatible consumer 的严格校验。

Freeze返回的对象不包含 policy、RuntimeState或 cache。

## Step 22：`Trajectory.to_segments()` 派生训练样本

Frozen Trajectory自己负责 projection，不重新调用 policy或 tokenizer。

### 22.1 Path order

`_selected_paths()`：

- rejected sibling跳过；
- foreign child切换到其 Chain；
- root order按 chain depth安排 deepest child first、main last；
- 每个 chain只发出含 LLM的 leaf path。

Oracle顺序：

```text
child@5
main@1
```

### 22.2 Semantic grouping

`_semantic_groups()` 只读取 output LLM node上的 frozen `SegmentBoundary`。 Boundary owner就是新 segment第一个 LLM target。

### 22.3 Global target ownership

`exported: set[NodeId]` 实现 first-visited dedup：

- LLM node只能在一个 Segment成为 `node_ids` target；
- 同一个 node仍可出现在其他 Segment prompt/messages；
- shared prefix不丢失，也不重复训练。

### 22.4 Physical grouping

Canonical projection再按 physical-only boundary安全切开。 Legacy projection不把 physical-only boundary改成新的 semantic type，而用 `split_cause` + warning timeline传递诊断。

### 22.5 Payload construction

第一个 LLM target：

```text
prompt_ids = 祖先 canonical tokens + first_target 的 glue
（HyTuner `_materialize_record()` 从 payload 的 token_ids 重建）
```

每个 target按顺序追加：

```text
turn_prompt中新增加的 observation/glue → mask 0
generated_ids                         → mask 1
trailing_ids                          → mask 0
```

同时：

- logprobs只覆盖 generated tokens；mask-0位置projection补 0；
- top-k所有 generated turn必须一致存在或全无；
- routed experts当前 segment内 last-wins；
- reward只汇总本 Segment node rewards；
- `final_reward` 不广播；
- active Trajectory的 Segment `is_closed=False`；
- completed但末尾仍有未回答 tool calls时也 `is_closed=False`。

### 22.6 Oracle逐 token

Main：

```text
first target n1:
  prompt       [10,11]
  generated    [12]          mask [1]

next target n3:
  turn prompt  [10,11,12,13,14,15]
  previous expected stream [10,11,12]
  added input             [13,14,15] mask [0,0,0]
  generated               [16]       mask [1]

final:
  response_ids  [12,13,14,15,16]
  response_mask [ 1, 0, 0, 0, 1]
```

Child：

```text
prompt          [20,21]
generated       [22,23]
response_mask   [1,1]
```

## Step 23：Legacy projection与HyTuner main-only filter

`Trajectory.to_legacy_hytuner()` 输出：

```text
segments
routed_experts
reward_score
termination_reason
metrics
extra_fields
request_id
```

它要求所有含 LLM的 HyTuner trajectory都有 token payload。 Top-k被投影为 legacy field names，mask-0 rows补零。

`nexus.runner.hytuner._run_gateway()` 接着：

1. `sidechain_info is None` 过滤 main segments；
2. sidechain被丢弃时 warning；
3. 无 main时构造一个空兼容 segment；
4. 汇总 `num_preempted`；
5. 构造 `HyTunerRunOutput`。

所以当前训练只收到 main：

```text
prompt_ids     [10,11]
response_ids   [12,13,14,15,16]
response_mask  [1,0,0,0,1]
```

Child Segment仍存在于 frozen Trajectory/legacy pre-filter projection，但本次不启用 child training。

`to_legacy_chat_completion()` 走另一条 human-readable artifact path。 Numpy/R3 structural arrays属于 token lane，会从 JSON chat artifact移除，避免 `json.dumps`失败。

## Step 24：NexusRunTracker补充run facts

`NexusRunTracker` 从 span processors收集：

- task name；
- termination reason；
- observation/reward time；
- runtime IDs/eval runtime ID；
- task/agent/observation/compaction timeline；
- tool arguments。

`finalize_trajectory()`：

1. 对已有 frozen Trajectory写 final facts；
2. 或在第一次 completion前结束时按 `allow_empty=True` 创建空 Trajectory；
3. 汇总每个 LLM `ProviderFacts` 的 inference/abort wait；
4. 从 latest messages构建 tool-call stats；
5. 把 runtime/tool/timing写到 metrics/extra。

它不修改 parent/children/roots/chains。

## Error与异常语义

当前实现故意 fail loudly：

| 场景 | 行为 |
| --- | --- |
| 未显式提供 policy | `ValueError` |
| 未知 policy 名称 | `ValueError` |
| 自定义对象不满足 `ScaffoldPolicy` runtime Protocol | `TypeError` |
| 重复注册同一个 `policy_id` | `ValueError` |
| 自定义 policy 抛错，或返回的 facts/anchor 无法 commit | 原异常或 assembler validation error；不 fallback |
| fact anchor不属于 event | `ValueError` |
| render/token partition不一致 | validation error |
| physical path无法重建 backend prompt | `ValueError` |
| observation显式声明 token IDs | `ValueError` |
| parent_messages 0 或多 path match | `ValueError` |
| message/token representation混入同 Segment | `ValueError` |
| finish时有 in-flight request | `RuntimeError` |
| unsupported streaming / `n>1` | `NotImplementedError` |
| rejected/custom reward export | `NotImplementedError` |

没有 rollback，也没有“出错就当 main”“未知 policy就用 nexus_main_agent”等 silent fallback。

## H4 Policy extension的最小契约

H4 的扩展点是一个满足 `ScaffoldPolicy` Protocol 的无状态对象。最常见写法是继承
`GenericScaffoldPolicy`，复用 Generic 的 anchors、message/tools identity、request metadata 与 timeline，再只补本
scaffold 的 wire semantics：

```python
from dataclasses import replace

from nexus.gateway.scaffold_policy import (
    GenericScaffoldPolicy,
    RequestEvent,
    ResponseEvent,
    SidechainMembership,
)


class ExamplePolicy(GenericScaffoldPolicy):
    def __init__(self):
        super().__init__(policy_id="example")

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if not isinstance(request, RequestEvent):
            return facts

        meta = request.request_meta or {}
        child_id = meta.get("child_id")
        if child_id is None:
            return facts

        return replace(
            facts,
            context_key=("example", str(child_id), True),
            membership=SidechainMembership(
                chain_id=str(child_id),
                kind="subagent",
                detector="child_id",
            ),
        )
```

真实 multi-agent scaffold 还可以在同一次 `inspect_event()` 中产生 `SpawnIntent`、`RetrievalIntent`、
`DeliveryReceipt`、`NotificationDelivery` 与 ordered `ChildPromptCandidate`。这些都是 event-local、带 anchor 的 typed
facts；policy 不读取或修改 `Trajectory`，不保存 parent/children/roots，也不实现跨事件关联。

Scaffold 不实现：

- prefix cache；
- Node创建；
- chain partition；
- exact/prefix prompt reducer；
- nested/parallel parent resolution；
- delivery precedence；
- Segment/mask。

这些都在 Generic assembler/freezer/Trajectory 中。 `wire.py` 只负责 lossless copy；H4 的 protocol identity、tool-call
arguments canonicalization 与 Generic timeline 都集中在 `scaffold_policy.py`。

H4 的优点是扩展能力直接、没有 recognizer/compiler DSL，也没有第二套 IR。它的限制同样明确：subclass 可以覆盖整个
`inspect_event()`；框架不会从结构上强制调用 `super()`，也不会阻止 scaffold 重复三类 event/anchor plumbing。因此，H4
要求 review 约束新增 policy：调用 `super()`，只增补 scaffold facts，不改变 Generic identity/timeline invariant。H4
production tree 只注册 Generic 与 Nexus main-agent；本提交不声称已经用 CC、Codex、Search、Hermes 或 OpenClaw adapter 验证这一
扩展面。

## 全部 Gateway production模块覆盖表

| 模块 | 何时执行 | 输入 | 输出/副作用 |
| --- | --- | --- | --- |
| `nexus/gateway/__init__.py` | import | — | public exports |
| `client.py` | client construction与每次 call | OpenAI messages/kwargs/meta | Session call、ExtendedChatCompletion、count/observation hooks |
| `session_manager.py` | first call/end | trajectory ID | active Session ownership |
| `session.py` | every completion/snapshot/end | recording/execution copies | orchestration、budget、latest/timeline |
| `scaffold_policy.py` | construction、每个 event、registry | wire event + policy object | flat fact dataclasses、Generic inspection、protocol identity/timeline与显式 registry |
| `policies/__init__.py` | 首次按名称 build policy | built-in policy objects | 精确注册 Generic/nexus_main_agent，拒绝重复 ID |
| `policies/generic.py` | built-in registration | — | 无 scaffold-specific semantics 的 Generic policy 对象 |
| `policies/nexus_main_agent.py` | every Nexus main-agent event | Generic `EventFacts` + Nexus wire metadata/messages | declared-chain membership/context与 compaction boundaries |
| `wire.py` | all wire seams | provider/model values | 保留 null/absent 差异的 lossless copy |
| `types.py` | Processor/backend seams | render/backend data | validated internal transport models |
| `processor.py` | every model call | messages/frozen prefix/request | exact prompt、backend result、assistant message |
| `backend.py` | model I/O | prompt IDs or messages | unified `BackendResult` |
| `token_budget.py` | token lane before/after generate | context + real prompt IDs | effective max tokens/consumption |
| `trajectory_assembler.py` | pre-call/commit/observation | events/render/token facts | direct live topology + bound facts |
| `runtime_state.py` | every commit/cache rebuild | NodeId-keyed facts | flat ledger/rebuildable lookup |
| `trajectory_freeze.py` | snapshot/finish | consistent pair（active copied，finish live） | resolved Trajectory |
| `trajectory.py` | live storage/frozen query/export | public facts | schema、segments、legacy、dump/pickle |
| `nexus_run_tracker.py` | Nexus spans/finalize | spans/task output | run metrics/extra enrichment |
| `nexus/runner/hytuner.py` | outer lifecycle | HyTuner run input | Gateway client、freeze、main-only output |

## Public schema为何仍是 scaffold-agnostic

以下字段都不存在：

```text
Trajectory.scaffold
Trajectory.policy
Trajectory.profile
TrajectoryNode.scaffold
TrajectoryNode.framework
Chain.policy
TokenAccount.scaffold
```

Public object只保存：

- wire message事实；
- physical forest；
- resolved logical Chain/links/evidence；
- token/provider/timeline facts；
- Segment boundary；
- terminal/reward/metrics。

`detector/evidence` 是已解析普通数据，不是 policy实例或重新解释所需 DSL。Consumer 无需知道它来自哪一种 scaffold，
即可读 dump、查询 path、投影 Segment。

## H4 验证证据

完整证据与边界见同目录的
[`gateway-direct-trajectory-h4-validation-report.md`](gateway-direct-trajectory-h4-validation-report.md)。核心结果是：

| Gate | 状态 |
| --- | --- |
| Exact H4 Gateway suite（fresh checkout） | **PASS：355 passed** |
| Original H4 local run | **PASS：351 passed，4 skipped**；skip 差异来自当时缺少 optional dependencies |
| Baseline differential characterization | **PASS：96 + 493 cases，0 divergence** |
| Ruff、format、compileall、diff/static architecture checks | **PASS** |
| H4 compaction bitwise（含 Qwen legacy seam） | **PASS**：Taiji task `basic_train_petersheng_20260802122637_e4657716`，3 train + 2 val committed golden exact |

三个 2026-08-02 non-bitwise Ray experiments使用后续 Nexus/HyTuner revision，不作为 H4 revision 的 PASS 证据。
