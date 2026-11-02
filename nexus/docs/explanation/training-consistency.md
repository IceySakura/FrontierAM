# Training Consistency

本文档介绍 Agent 训练场景中的 **Token 一致性问题** ，以及 Nexus 的 HyTunerClient 如何解决这一问题。

## 问题背景

在 Agent 的多轮对话中，消息历史会在模型和用户之间往返传递。 传统的 OpenAI 兼容 API 实现中，这个往返过程会引入 **Token 不一致性** ，对训练场景造成严重影响。

不一致性主要来自三个环节。 首先，模型输出的 token 序列在转换为字符串后，经 tokenizer 重新编码可能产生不同的 token 序列——这是因为同一个字符串可能对应多种 token 切分方式，而模型生成时的切分与 tokenizer 的默认编码策略不一定一致。

其次，当模型输出 tool call 时，vLLM 等推理框架会进行 JSON 解析和重新序列化：

```python
# vLLM Hermes Tool Parser (hermes_tool_parser.py:145-157)
raw_function_calls = [json.loads(match) for match in function_call_tuples]
tool_calls = [
    ToolCall(
        function=FunctionCall(
            name=function_call["name"],
            arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
        ),
    )
    for function_call in raw_function_calls
]
```

这个 `json.loads()` + `json.dumps()` 的往返会导致空白符被规范化（原始的 `{"a":  1}` 变成 `{"a": 1}`）、键的顺序可能改变、数字格式可能变化（`1.0` 变成 `1`）等问题。

第三，当用户将 assistant message 传回服务端时，vLLM 会进行 JSON 解析（`_postprocess_messages` 中的 `json.loads()`），然后通过 Chat Template 的 `tojson` filter 重新渲染，引入额外的格式变化。

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/training-consistency/problem-flow.svg" style="width: 100%"></object><figcaption>传统方案中的 Token 不一致问题</figcaption></figure>

在训练场景中，Token 不一致会带来以下影响：

1. **Logprobs 错位** ：推理时记录的 logprobs 对应的是模型实际输出的 token 序列，但训练数据中的 token 序列已经被修改，导致 logprobs 与 token 无法对齐，影响 PPO/DPO 等依赖 logprobs 的训练算法

2. **Prefix Cache 失效** ：vLLM 等推理框架的 Automatic Prefix Caching (APC) 基于 token 序列精确匹配，历史 token 序列变化会导致 cache miss，需要对整个序列重新 prefill

## 解决方案：HyTunerClient

Nexus 的 `HyTunerClient` 通过 **Token 级别的状态保持** 解决了这一问题。

### 核心思路

**保留历史 token 序列不动，在文本层面提取新增内容并单独 tokenize，再与历史 tokens 拼接。**

### 实现机制

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/training-consistency/solution-flow.svg" style="width: 100%"></object><figcaption>HyTunerClient 的 Token 一致性保证</figcaption></figure>

`HyTunerClient` 维护两个关键状态：

```python
class HyTunerClient(OpenAICompatibleClient):
    def __init__(self, config: OpenAIConfig):
        # ...
        self.last_messages: list[dict] = []  # 上一次的消息历史
        self.last_tokens: list[int] = []     # 上一次的完整 token 序列
```

### 第一轮调用

首次调用时 `last_messages` 为空，`apply_chat_template` 渲染完整 prompt 文本后一次性 tokenize：

```python
full_text = self.apply_chat_template(messages, tools, is_first_msg=True)
prompt_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
```

生成完成后，保存完整的 token 链：

```python
self.last_messages = messages + [response_message]
self.last_tokens = prompt_ids + response_token_ids
```

### 后续调用

后续调用时，采用 **text 级增量 tokenization** ：

```python
# 1. 通过消息前缀匹配找到公共前缀长度
prefix_len = self._find_common_prefix(messages, self.last_messages)
last_message_count = len(self.last_messages)
can_reuse_prefix = (
    last_message_count > 0
    and prefix_len == last_message_count
    and len(messages) > prefix_len
)

# 2. 根据是否可复用缓存，决定前缀文本与缓存 tokens
if can_reuse_prefix:
    # 正常 multi-turn：旧消息原封不动，仅在末尾追加
    first_part_text = self.apply_chat_template(
        messages[:prefix_len], tools, add_generation_prompt=False, is_first_msg=False
    )
    cached_tokens = self.last_tokens
else:
    # 首轮、消息被替换、或发生 compaction — 全量 tokenize
    first_part_text = ''
    cached_tokens = []

# 3. 渲染完整 prompt 文本，并校验 first_part_text 是其前缀
full_text = self.apply_chat_template(messages, tools, is_first_msg=not can_reuse_prefix)
if not full_text.startswith(first_part_text):
    raise RuntimeError('Failed to parse new part of messages for consistent text.')

# 4. 仅对新增 text 做 encode
new_text = full_text[len(first_part_text):]
new_tokens = self.tokenizer.encode(new_text, add_special_tokens=False)

# 5. 拼接：使用缓存的 token 序列 + 新增 tokens
prompt_ids = cached_tokens + new_tokens
```

### 关键保证

1. **历史 tokens 不变** ：`last_tokens` 保存的是模型实际生成时的 token 序列，不会被重新 tokenize

2. **增量追加** ：新消息只在末尾追加，通过 `full_text[len(first_part_text):]` 在文本层面提取增量部分，仅对增量文本做一次 tokenize

3. **一致性验证** ：通过 `full_text.startswith(first_part_text)` 确保 `apply_chat_template` 对历史部分的渲染保持稳定

4. **训练数据完整** ：在 span 中记录完整的 token 级别信息

```python
update_span(span_id, attributes={
    '_hytuner_training_data': {
        'prompt_ids': prompt_ids,
        'response_ids': token_output.token_ids,
        'response_logprobs': token_output.log_probs,
        'messages': self.last_messages,
    }
})
```

### 严格前缀检查

默认情况下，`HyTunerClient` 支持 compaction 或其他 history rewrite：一旦 `_find_common_prefix` 发现
前缀变短，就放弃历史 token 缓存并全量 tokenize，新的 trajectory segment 会从当前 prompt 重新开始。

如果希望防止这类 history rewrite 被静默接受，可以在 hytuner client 的 `client_args` 中打开严格模式：

```yaml
openai_clients:
  main:
    name: hytuner
    config:
      client_args:
        strict_prefix_check: true
```

打开后，后续 LLM 请求必须保留前一次 `last_messages` 作为完整前缀；如果 agent 修改、删除、压缩历史
消息，`HyTunerClient` 会在生成前报错。 同时，Nexus 的 `CompactionManager` 在该模式下会拒绝执行
compaction。

## 限制与注意事项

1. **默认支持修改历史但会切 segment** ：未开启 `strict_prefix_check` 时，history rewrite 会退化为全量 tokenize

2. **不支持修改 tools** ：目前不支持在对话过程中修改 tool schemas

3. **依赖 template 文本确定性** ：要求 `apply_chat_template(..., tokenize=False)` 对相同的消息输入始终产生相同的文本

## 总结

Token 不一致性是 Agent 训练中一个容易被忽视但影响深远的问题。 `HyTunerClient` 通过 Token 级别的状态保持和增量 tokenization，从根本上解决了这一问题，确保训练数据与模型实际行为完全一致。
