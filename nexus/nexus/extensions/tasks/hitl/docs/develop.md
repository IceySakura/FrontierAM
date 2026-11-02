# 二次开发

本文介绍如何基于 HITL 提供的多 Agent 组件开发自定义的 Agent。三层组件各自的设计见 [design_actor.md](design_actor.md)、[design_usersim.md](design_usersim.md) 与 [design_label.md](design_label.md)。

---

## 1. 已有能力

- `actor/` 把 harness 包成命名 session，逐轮投递消息、逐轮取回结果并渲染成文本。
- `trace/` 的 label 与三个 processor 按来源分离轨迹、用量与 langfuse 记录。
- `user_agent/` 提供需求队列、persona、prompt 装配、工具集、周期提醒与历史压缩。

HITL 的基础设施支持大多数多 agent 场景的开发，不局限于 user simulator 相关任务。

---

## 2. 驱动已有 harness

worker manager 的基本使用方法如下。如果希望任务能够灵活切换不同的后端，应当限制只使用 manager 的公开接口并满足其语义，或对特定manager 做特殊处理。细节见[design_actor.md](design_actor.md) 。

```python
manager = make_worker_manager(
    agent_input.worker_backend,          # 'print_claude_code' 或 'pty_claude_code'
    default_working_dir=agent_input.working_dir,
    default_config=agent_input.worker_config,
    openai_client_name=agent_input.worker_client_name,
)
await manager.setup()                    # 后端共享资源，一次 run 一次
session = await manager.start(name='worker', workspace=agent_input.working_dir)

while not done:
    result = await manager.send(session, SessionMessage(content=instruction))
    report = manager.report(result)
    ...                                  # 按 result.kind 决定下一轮

traces = manager.traces(session)         # 要在 shutdown 之前取
await manager.shutdown()
```

`result.session_lost` 为真时结束整个 run。session 被 runtime 关掉并注销之后，下一次对该 sessionn 对象`send` 会重启sessionn它不带这段对话的历史，proxy 也不再带 label。

---

## 3. label 机制使用

label 用于隔离不同的 agent 会话，其具体使用方法见 [design_label.md](design_label.md)。

---

## 4. 接入一个新的 harness

在 `actor/` 下新建一个子包，实现 `SessionManager` 的抽象方法与类属性并注册，然后在 `actor/factory.py` 里导入该模块。骨架与类属性见 [design_actor.md](design_actor.md) §7。

---

## 5. 用 hook 注入自定义逻辑

通用 agent fn 提供 hook 机制，从而支持不改动 agent 就能加入自定义逻辑。事件类型由具体的 agent fn 定义，例如 agent_with_user 包含以下三类：
* `RunStart` 在 agent 循环正式开始前调用一次，用于插入额外初始化逻辑。
* `WorkerDone` 在每次 worker 调用完成后触发，额外附带 worker 名称 与 `TurnResult`。
* `RunFinish` 在 agent 循环结束时调用一次，用于插入额外收尾逻辑。

```python
from nexus.extensions.tasks.hitl.agents.agent_with_user import RunFinish, WorkerDone
from nexus.extensions.tasks.hitl.user_agent.hooks import HookEvent, HookOutcome, run_hook_registry


@run_hook_registry.register('mytopic')
class MyHook:
    events = (WorkerDone, RunFinish)

    async def on_event(self, event: HookEvent) -> HookOutcome | None:
        if isinstance(event, WorkerDone):
            logger.info('round %d: %s', event.state.worker_rounds, event.result.kind)
        elif isinstance(event, RunFinish):
            runtime = await require_context().get_runtime()
            got = await runtime.download_file(files=[FileItem(path='/testbed/dist.tar')])
            return HookOutcome(results={'mytopic': {
                'accepted': len(event.state.accepted),
                'artifact_bytes': len(got.files[0].content or ''),
            }})
        return None
```

每个 event 都带这次 run 的 `RunState`，即 `event.state`，队列、计数与验收记录都在上面，字段说明见该类本身。`isinstance` 收窄之后才能读到它与该事件自己的字段。hook 可以直接修改它，也可以返回 `HookOutcome`，其 `results` 会写入 `output.hook_results`。

`agent_config['hooks']` 是一个列表，每项 `{'name': ..., 'args': {...}}` 配置一个 hook，按列表顺序调用，各自构造实例。`name` 含 `::` 时按 `pkg.mod::Class` import，否则从注册表查找。

配置在 run 开始前校验，解析不出、没有 `on_event`、`events` 为空或含该 agent 不触发的事件类都直接报错。`on_event` 抛出的异常记进 `output.hook_failures`，不中断 run。

hook 另外定义 `on_failure(event, error) -> bool` 则由它自己决定：返回 True 时后续 hook 不再执行，run 以失败告终。判断发生在出错当下，拿得到是哪个事件、抛的什么，所以同一个 hook 可以在上传产物失败时中断、在记日志失败时放过。`RunStart` 与 `WorkerDone` 上直接中断循环，`RunFinish` 上循环已经结束，异常改为写进 `output.stderr`，轨迹与用量照常收集。

---

## 6. 复用 user simulator 的组件

自行实现一个 user 循环时，`user_agent/` 下这些模块可以直接取用：

- `queue` 需求队列
- `personas` persona 的加载与校验
- `tools.toolset` 工具集装配
- `prompts` system prompt 与运行期消息
- `dispatch` 工具调用的分发
- `reminders` 周期提醒
- `compact` 历史压缩
- `skills` skill 的加载与上传
- `hooks` hook 的契约与解析

`agents/agent_with_user.py` 是它们拼成的一个实现，可以作为参考的骨架。


### 工具 handler与接入

由于 neuxs 工具设计的限制，一些复杂的 host 端逻辑需要通过回调实现，即 handler。可参考 HITL 的做法创建自己的handler，从而表达多样化的逻辑。

新增一件工具时，把它加入 `toolset.CATALOGUE`，并在 `meta` 里声明它属于哪一类用法，从而共用该类工具在 user system prompt里的使用说明。
