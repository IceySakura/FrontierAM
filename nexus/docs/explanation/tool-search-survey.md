# Tool Search 调研

本文调研业界主流的 **工具检索（Tool Search / Tool Discovery）** 方案， 关于 Nexus Tool Search 的使用方法，见 [Tool Search](../how-to/tool-search.md)。

## 背景：为什么需要工具检索

当 Agent 可用工具增长到几十上百个（尤其接入多个 MCP server 后），把全部工具 schema 一次性放进模型上下文会引发两个公认的问题：

1. **上下文膨胀** ：工具定义本身消耗大量 token 。 Anthropic 给出的数据是，一个典型的多 server 组合（GitHub、Slack、Sentry、Grafana、Splunk）在 Agent 干活之前就要吃掉约 55k token 的工具定义。
2. **选择精度下降** ：Anthropic 观察到，可用工具超过 30–50 个后，模型选对工具的能力显著下降。

业界的共识方案是 **按需暴露（just-in-time / progressive disclosure）** ：初始只给模型少量工具，其余「藏起来」，让模型在需要时检索发现、再加载调用。

## 业界方案横向对比

### Anthropic Tool Search Tool

Anthropic 默认提供 **服务端** 工具检索，同时 **官方支持自定义的客户端检索**。 

服务端用法是在 `tools` 里放一个特殊的检索工具（`tool_search_tool_regex_20251119` 或 `tool_search_tool_bm25_20251119`），并给需要延迟的工具加 `defer_loading: true` 。 模型初始只看到检索工具 + 未延迟工具；当它调用检索工具时，API 返回 3–5 个最相关的 `tool_reference` 块，并 **自动展开** 成完整工具定义注入对话。 

自定义的客户端检索需要提供一个 **普通工具**，在它的 `tool_result` 里返回 `tool_reference` 块（检索算法任选，例如 embeddings / 语义检索），被引用的工具仍需在顶层 `tools` 里以 `defer_loading: true` 定义。

- **检索算法** ：服务端两个变体——Regex（模型构造 Python `re.search` 正则，上限 200 字符）或 BM25（自然语言查询）。 两者都检索工具名、描述、参数名、参数描述。
- **保留缓存** ：延迟工具不进系统提示前缀，`tool_reference` 内联追加到对话中，因此 **prompt caching 不被破坏** 。
- **工具规模** ：最多 10000 个工具，宣称典型场景能把工具定义 token 降低 85% 以上。

请求里声明延迟工具（`defer_loading: true`）：

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": { "type": "string" },
      "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
    },
    "required": ["location"]
  },
  "defer_loading": true
}
```

模型调用服务端检索工具后，响应里出现 `server_tool_use` 与 `tool_search_tool_result` 两类块；`tool_reference` 会被 API 自动展开为完整定义，因此模型随后能像普通工具一样直接调用：

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll search for tools to help with the weather information."
    },
    {
      "type": "server_tool_use",
      "id": "srvtoolu_01ABC123",
      "name": "tool_search_tool_regex",
      "input": {
        "query": "weather"
      }
    },
    {
      "type": "tool_search_tool_result",
      "tool_use_id": "srvtoolu_01ABC123",
      "content": {
        "type": "tool_search_tool_search_result",
        "tool_references": [{ "type": "tool_reference", "tool_name": "get_weather" }]
      }
    },
    {
      "type": "text",
      "text": "I found a weather tool. Let me get the weather for San Francisco."
    },
    {
      "type": "tool_use",
      "id": "toolu_01XYZ789",
      "name": "get_weather",
      "input": { "location": "San Francisco", "unit": "fahrenheit" }
    }
  ],
  "stop_reason": "tool_use"
}
```

自定义客户端检索则相反：模型调用 **你自己的** 检索工具，你返回一个标准 `tool_result`，其 `content` 里放 `tool_reference` 块（被引用工具需在顶层 `tools` 中以 `defer_loading: true` 定义）：

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_your_tool_id",
  "content": [{ "type": "tool_reference", "tool_name": "discovered_tool_name" }]
}
```

### OpenAI Tool Search

> **前提** ：OpenAI tool search **仅支持 Responses API**（`client.responses.create()`），Chat Completions 不支持；且要求 `gpt-5.4` 及以上模型。 下文的 `tool_search_call` / `tool_search_output` / `function_call` 都是 Responses API 的输出条目格式。

OpenAI 的工具检索通过在 `tools` 数组里加一个 `{"type": "tool_search"}` 条目来激活，并用 `defer_loading: true` 标记延迟工具（推荐按 namespace / MCP server 分组——模型初始只看到容器的名字和描述，函数 schema 延迟加载）。 它支持两种模式，**区别只在于「谁来执行检索」**：

- **Hosted（服务端）** ：OpenAI 自己检索并直接回填命中的工具，应用无需介入。
- **Client-executed（客户端）** ：模型发出检索请求，**应用自己执行检索逻辑** 并回填结果。

**请求声明** ：`namespace` 容器（含延迟函数）与 `tool_search` 激活条目并列放在 `tools` 里：

```json
{
  "tools": [
    {
      "type": "namespace",
      "name": "crm",
      "description": "CRM tools for customer lookup and order management.",
      "tools": [
        {
          "type": "function",
          "name": "list_open_orders",
          "description": "List open orders for a customer ID.",
          "defer_loading": true,
          "parameters": {
            "type": "object",
            "properties": {
              "customer_id": { "type": "string" }
            },
            "required": ["customer_id"],
            "additionalProperties": false
          }
        }
      ]
    },
    {
      "type": "tool_search"
    }
  ]
}
```

**Hosted 模式流程** ：模型发起检索（`tool_search_call`，`execution: "server"`、`call_id: null`）→ OpenAI 在同一响应里回填命中工具（`tool_search_output`）→ 模型直接发起对真实函数的 `function_call`。 三个条目依次出现在输出里：

```json
[
  {
    "type": "tool_search_call",
    "execution": "server",
    "call_id": null,
    "status": "completed",
    "arguments": {
      "paths": ["crm"]
    }
  },
  {
    "type": "tool_search_output",
    "execution": "server",
    "call_id": null,
    "status": "completed",
    "tools": [
      {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools for customer lookup and order management.",
        "tools": [
          {
            "type": "function",
            "name": "list_open_orders",
            "description": "List open orders for a customer ID.",
            "defer_loading": true,
            "parameters": {
              "type": "object",
              "properties": {
                "customer_id": { "type": "string" }
              },
              "required": ["customer_id"],
              "additionalProperties": false
            }
          }
        ]
      }
    ]
  },
  {
    "type": "function_call",
    "name": "list_open_orders",
    "namespace": "crm",
    "call_id": "call_abc123",
    "arguments": "{\"customer_id\":\"CUST-12345\"}"
  }
]
```

**Client-executed 模式流程** ：与 hosted 的区别是 `execution: "client"`、`call_id` 为真实标识符，且 **检索由应用执行**。 模型先发出带检索意图的 `tool_search_call`（`arguments` 匹配你自定义的检索参数 schema，例如这里的 `goal`）：

```json
[
  {
    "type": "tool_search_call",
    "execution": "client",
    "call_id": "call_abc123",
    "status": "completed",
    "arguments": {
      "goal": "Find the shipping ETA tool for order_42."
    }
  }
]
```

应用自己检索后，回填 **同一个 `call_id`** 的 `tool_search_output`，带上命中的工具定义：

```json
[
  {
    "type": "tool_search_output",
    "execution": "client",
    "call_id": "call_abc123",
    "status": "completed",
    "tools": [
      {
        "type": "function",
        "name": "get_shipping_eta",
        "description": "Look up shipping ETA details for an order.",
        "defer_loading": true,
        "parameters": {
          "type": "object",
          "properties": {
            "order_id": { "type": "string" }
          },
          "required": ["order_id"],
          "additionalProperties": false
        }
      }
    ]
  }
]
```

### OpenClaw

OpenClaw 是 **客户端** 方案。 它默认提供一个 `tool_search_code` 工具（JS 桥接），也可在 `mode: "tools"` 下退化为三个结构化工具 `tool_search` / `tool_describe` / `tool_call` 。 检索是 **关键字匹配** （查询如 `"create a GitHub issue"`）。

默认的 code 模式下，模型写一小段 JS，通过 `openclaw.tools` 桥接 search → describe → call（以下为官方文档示例）：

```js
const hits = await openclaw.tools.search("create a GitHub issue");
const tool = await openclaw.tools.describe(hits[0].id);
return await openclaw.tools.call(tool.id, {
  title: "Crash on startup",
  body: "Steps to reproduce...",
});
```

`search` 支持选项（如 `limit`），`describe` / `call` 用工具 id（形如 `mcp:calendar:create_event`）：

```js
const hits = await openclaw.tools.search("calendar event", { limit: 5 });
const calendarCreate = await openclaw.tools.describe("mcp:calendar:create_event");
await openclaw.tools.call(calendarCreate.id, {
  summary: "Planning",
  start: "2026-05-09T14:00:00Z",
});
```

配置（结构化）：

```json5
{
  tools: {
    toolSearch: {
      mode: "tools",          // 或省略走默认 code 模式
      codeTimeoutMs: 10000,
      searchDefaultLimit: 8,
      maxSearchLimit: 20,
    },
  },
}
```

### Hermes（Nous Research）

Hermes 是 **客户端 + 三桥接工具** 方案。 启用后，它用 `tool_search` / `tool_describe` / `tool_call` 三个工具替换掉 MCP 与插件工具的 schema；核心工具（`terminal` / `read_file` 等）始终直接暴露，只延迟外部工具。 检索用 **BM25** （对工具名、描述、参数名分词），并在 **零 IDF 退化** 时回退到工具名子串匹配——即查询词恰好出现在所有工具中（IDF 为 0、BM25 给不出区分度）时，改用最朴素的「工具名是否包含该词」来兜底，避免排序失效。

三个桥接工具的签名（官方文档）：

```text
tool_search(query, limit?)     — search the deferred-tool catalog
tool_describe(name)            — load the full schema for one tool
tool_call(name, arguments)     — invoke a deferred tool
```

一次完整交互（官方文档示例，箭头右侧为返回值）：

```text
tool_search("create a github issue")
  → { matches: [{ name: "mcp_github_create_issue", ... }, ...] }

tool_describe("mcp_github_create_issue")
  → { parameters: { type: "object", properties: { ... } } }

tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

当模型调用 `tool_call` 时，Hermes **拆掉桥接外壳，像模型直接调用一样分发底层工具** ；pre/post hook 和活动流里显示的是真实底层工具名，而非桥接包装。 激活分三档：`auto`（延迟 schema 超过上下文 `threshold_pct`，默认 10%，时自动启用）/ `on` / `off`，每次组装上下文时重新评估：

```yaml
tools:
  tool_search:
    enabled: auto          # auto（默认）/ on / off
    threshold_pct: 10
    search_default_limit: 5
    max_search_limit: 20
```

### LiteLLM

LiteLLM 不自己实现检索逻辑，而是 **透传 Anthropic 的服务端工具检索** ：原样转发 `defer_loading` 标记和 beta header（`anthropic-beta: advanced-tool-use-2025-11-20`）给后端。 因此它 **仅支持承载 Anthropic 模型的少数 provider**，具体为：

| Provider | Chat Completions API | Messages API | 备注 |
|----------|:---:|:---:|------|
| Anthropic API | ✅ | ✅ | |
| Azure Anthropic（Microsoft Foundry） | ✅ | ✅ | |
| Google Cloud Vertex AI | ✅ | ✅ | |
| Amazon Bedrock | ✅ | ✅ | 仅 Invoke API、仅 Opus 4.5 |

此外 BM25 变体在 Bedrock 上不可用（只有 regex 变体四个 provider 通用）。 换言之，LiteLLM 并未把工具检索抽象成跨厂商的通用能力，而是把 Anthropic 的原生特性透传给恰好兼容的几个后端。

### 对比总表

| 方案 | 检索位置 | 模型接口 | 检索算法 | 工具上限 |
|------|----------|----------|----------|----------|
| Anthropic | 服务端（也支持自定义客户端） | 原生 `tool_reference` 协议 | Regex / BM25 | 10000 |
| OpenAI | 服务端 / 客户端 | 原生 `tool_search_call` | 未公开（按 path/goal） | — |
| OpenClaw | 客户端 | 三桥接工具 或 code 工具 | 关键字 | — |
| Hermes | 客户端 | 三桥接工具 | BM25 + 子串回退 | — |
| LiteLLM | 服务端（透传 Anthropic） | 原生 `tool_reference` | Regex / BM25 | 10000 |

## Nexus 的设计决策

Nexus 的核心诉求是做一套 **通用、不绑定厂商** 的工具检索方案：在所有 OpenAI 兼容模型上都能用同一套机制，而不是依赖某家供应商的服务端能力或私有协议。 这一诉求决定了下面两个关键选择。

### 决策一：客户端检索，不绑定厂商

**选择** ：Nexus 在 `ToolExecutor` 里自己实现检索，全程走 **普通 function calling**，不使用任何供应商的服务端 tool search 或私有协议块。

**理由** ：

- **通用、不绑定厂商** ：各家的服务端方案都把检索绑定到自己的接口，Anthropic 用 `tool_reference` / `server_tool_use` 协议块，OpenAI 的 tool search 仅限 Responses API，LiteLLM 也只覆盖承载 Anthropic 模型的少数 provider。 Nexus 面向所有 OpenAI 兼容模型（走 Chat Completions 兼容接口），唯有客户端方案能在它们之上提供 **一致** 的工具检索。

### 决策二：关键字加权打分，而非 BM25 或语义向量

**选择** ：用简单的分层加权——工具名完全相等（+100）> 命中名字分词（+60）> 名字子串（+30）> 命中描述词（+10），并支持 `+` 前缀的必需词。

**理由** ：

- **零依赖** ：不需要嵌入模型、不需要预建索引，纯本地实现。
- **可解释** ：打分规则透明，排序结果可预测，便于调试。

## 未来可演进方向

1. **可配置的检索算法** ：把检索算法做成可插拔——除当前的关键字加权打分外，按需切换到 BM25（参考 Hermes，工具规模增长后提升召回）甚至语义向量检索，而桥接工具的对外接口保持不变。
2. **自动激活档位** ：参考 Hermes 的 `auto` 模式，当延迟工具的 schema 体量超过上下文一定比例时才启用 tool search，避免小工具集下徒增桥接工具开销。
3. **厂商原生能力集成** ：在用户确实使用支持原生 tool search 的厂商（如 Anthropic 服务端、OpenAI Responses API）时，作为可选后端接入其原生能力，以换取 prompt caching 等优化；客户端方案则作为通用兜底，对上层保持同一套接口。

## 参考资料

- [Anthropic — Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [OpenAI — Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
- [OpenClaw — Tool search](https://docs.openclaw.ai/tools/tool-search)
- [Hermes (Nous Research) — Tool search](https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/features/tool-search)
- [LiteLLM — Anthropic tool search](https://docs.litellm.ai/docs/providers/anthropic_tool_search)
