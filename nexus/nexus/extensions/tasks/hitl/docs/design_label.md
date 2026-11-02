# label 与按 label 分组的 processor

为了在 HITL 涉及的多 agent 场景下准确的区分轨迹归属，上传正确的 langfuse 记录和用量记录，我们开发了 label 机制以及对应的 nexus span processor。

本文说明 label 的取值、实现机制与两条传递路径，自行装配时的用法，以及按 label 分组的三个 processor。

---

## 1. label 的取值

一个 label 是一个用 `/` 分段的字符串，用于唯一标识一个 LLM 请求的来源。在 HITL，主循环的角色主要为 user（user simulator 模拟出来的那一方）和 worker，他们的 label 如下：

| 形式　 | 取值　　　　　　　 | 出处　　　　　　　　　　　　　　　　 |
| --------| --------------------| --------------------------------------|
| user　 | `user/<session>`　 | `user_label(session='orchestrator')` |
| worker | `worker/<session>` | `worker_label(session)`　　　　　　　|
| 无归属 | `_unattributed`　　| `UNATTRIBUTED`　　　　　　　　　　　 |

label 的取值只要求是非空字符串，上表是内置组件采用的约定。`user_label` 与 `worker_label` 是生成这套约定的 helper，自行装配时可以传任意取值，`filter_by_label` 与 `group_by_label` 对它们一样工作。`is_wellformed` 判断的是一个 label 是否符合上表的形式，自定义取值下它返回 False。

worker 的 label 由 manager 从 session 名生成，取值不经过调用方；能自由取值的是 host 上进入 `labelled()` 的那一侧。

段名不允许包含 `/`，传入 `user_label` / `worker_label` 时抛 `LabelError`。

---

## 2. label 的实现机制

label 通过绑定为 client 类的只读属性实现，基于任务上下文变量来确定取值。

### span 属性的写入

上游 `OpenAICompatibleClient.create_completion` 带 `@observe('openai_completion', self_attrs=['_client_key', '_chat_phase'])`。装饰器在进入调用之前从 client 实例上 getattr 这两个名字，取到的非 None 值写进本次调用的 span 属性。原生 client 没有 `_chat_phase` 这个属性，它的 span 不携带 label。

### label 的作用域

`labelled(client, label)` 是一个同步上下文管理器，进入时把 label 写入上下文变量，退出时 reset。取值是任务局部的，同一个 client 上并发的两次调用各自读到自己的那一个。

进入 `labelled` 与发起调用之间不能 await：`@observe` 在 await 被包装的调用之前读取 `_chat_phase`，label 要在调用发起时已经就位。

label 传入空值时取 `_unattributed`。不带 label 的调用在导出时继承 chat 树上最近的带 label 的祖先，而祖先在 session 之间是共享的。

### client 的属性绑定

`enable_labels(client)` 为传入的实例造一个同名子类，把读上下文变量的 property 放在子类上，再重绑实例的 `__class__`：

```python
labelled_class = type(
    f'Labelled{base.__name__}',
    (base,),
    {'_chat_phase': property(lambda self: current_label())},
)
client.__class__ = labelled_class
```

重绑作用于传入的实例，原类与它的其他实例保持不变。子类上留有标记，对同一个实例重复调用 `enable_labels` 返回该实例本身。`_chat_phase` 的 setter 抛 `AttributeError`。

`__class__` 重绑要求新旧类的实例布局一致，带 `__slots__` 的类、`int`、`object()` 的实例都不满足，`enable_labels` 对它们抛 `LabelError`。

---

## 3. label 的传递路径

nexus 的 LLM 调用主要分为 host 直接调用和沙箱内通过代理转发到 host 调用，我们在两条调用链路上分别支持了 label 机制。

### 沙箱内

label 在沙箱内的传递经过以下步骤：

1. **沙箱写入。** 每个 session 的 proxy 把自己的 label 写进请求参数里的一个私有字段（`actor/labelling/proxy.py`）。这个 label 在 proxy 构造时就已固定，因此并发的多轮不会互相覆盖归属。
2. **host 剥离。** host 上的 llm_handler 取出该字段，随即进入 `labelled()` 上下文（`actor/labelling/handler.py`）。这个 handler 包在重放保护内部那一层之外，继承了去重、ack/心跳协议与失败时的错误码路径。
3. **写到 span。** 客户端的 `@observe` 装饰器把当前 label 复制到这次调用的 span 上。
4. **读回。** chat tracker 从 span 上读到它，导出为轨迹的 `role` 字段。
5. **筛选。** 读取方按 label 取出其中一份。

第 2 步剥离之后，handler 断言请求里不再有以 `_hitl` 开头的字段。上游的参数清洗按固定名单过滤，名单外的字段会作为一个 completion 参数发给 provider。

没有带 label 的沙箱请求计入 `unattributed_requests` 并告警，该轮继续执行。

### host

host 端通过 `labelled()` 的调用上下文，host 上的模型调用可直接被标记。其使用方式参考下一节。

---

## 4. 使用方式

为了为 host 端模型调用启用 label，需要按以下方式配置：

```python
from nexus.extensions.tasks.hitl.trace import labels

# run 级 tracker 取 branch_last，配置见 §5
labels.require_labelling_tracker()

# 装上 _chat_phase，一个 client 一次，先于 labelled
client = labels.enable_labels(context.get_openai_client('main'))

# 块内放这一次调用
with labels.labelled(client, labels.user_label()):
    response = await client.create_completion(messages=history, tools=tools)
```

---

## 5. HITL span processor

HITL 提供三个基于 label 机制设计的 span processorr，用于提供多 agent 场景下的可观测性。为了使用这三个 processor，需要参考如下方法配置 generator：

```python
processors = {
    # branch_last 见 §4，label 由 role 字段承载
    'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}},
    'usage': {
        'name': 'hitl_usage',
        # profile 按 client 名指定，未列出的 client 取 default_profile
        'args': {'profiles': {'worker': 'bedrock_claude'}, 'default_profile': 'default'},
    },
    'langfuse': {
        'name': 'multiagent_langfuse',
        'args': {'session_id': instance_id, 'trace_name': instance_id},
    },
}
```

### 5.1 chat_tracker：轨迹导出

chat tracker 为 nexus 原生的 tracker 形式，但会被添加额外 role 的字段放置 label。其输出形式如下，其中的 `role` 就是 §1 的 label：。由于要求输出为字典，我们要求 chat tracker 锁定为 branch_last 的模式

一条轨迹是一个字典，其中的 `role` 就是 §1 的 label：

```
{'messages': [...], 'tools': [...], 'role': '<label>', 'extra': {...}}
```

### 5.2 hitl_usage：用量统计

HITL 提供独立的 usage 统计，用于追踪每轮的 token 用量与缓存命中率。其追踪以 client 的 label 分桶，以 turn 为粒度统计输入、输出、缓存读写与命中 token数，输出 schema 如下：

```python
{
    'by_label': {
        '<label>': {
            # 跨模型总计
            'calls': int, 'invalid_calls': int,
            'input_tokens': int, 'output_tokens': int, 'total_tokens': int,
            'cache_read_tokens': int, 'cache_creation_tokens': int, 'reasoning_tokens': int,
            # 按 model 串再分一层
            'by_model': {'<model>': {...同上的 token 计数...}},
            # 每轮一个数，逗号分隔
            'series': {'input': '1200,3400,...', 'output': ..., 'cache_read': ...,
                       'cache_creation': ..., 'reasoning': ..., 'messages': ...},
            # 单轮 input 与单轮消息数的最大值
            'peak_input_tokens': int, 'peak_messages': int,
        },
    },
    'totals': {...每个 label 的总计再相加...},
    'profiles': {'<client 名>': '<profile 名>'},
}
```

HITL 以 usage profile 区分不同模型的字段位置，绝大多数 openai 请求都可以使用 `default`，而对于 claude 模型，需要特别使用 `bedrock_claude` 的配置。


### 5.3 multiagent_langfuse：langfuse 上传

HITL 提供增强后的 langfuse tracker，支持独立追踪多个 agent 轨迹，并统一挂在对应 langfuse 记录的 span内。对于未标记的轨迹会统一记为 unattributed。

HITL 的 langfuse tracker 优化了内存使用，不再在每次上传完整 message 作为指纹，而是使用消息哈希值，相比原生的 langfuse tracker 在 long running 任务下有更好的性能，不容易超出 langfuse 的存储上上限。
---
