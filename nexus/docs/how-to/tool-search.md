# Tool Search

本文档介绍 Nexus 的 **客户端工具检索（Tool Search）** 机制：设计动机、工作原理与用法。

## 概述

当一个 Agent 可用工具很多时（尤其加载了大量 MCP 工具），把全部工具 schema 一次性塞进模型的 `tools` 列表会带来两个问题：

1. **上下文膨胀** ：每个工具 schema 都占 token ，几十上百个工具会挤占有效上下文。
2. **选择困难** ：工具越多，模型选错或漏选的概率越高。

Tool Search 的思路是 **把一部分工具「延迟（defer）」暴露** ：这些工具不出现在模型初始的 `tools` 列表里，而是通过三个内置的 **桥接工具（bridge tools）** 让模型按需检索、查看、调用。 模型看到的初始工具集因此大幅收缩，需要时再「现查现用」。

这套机制完全运行在 **客户端（Nexus 侧）** ，不依赖任何模型供应商的原生工具检索能力，因此对所有 OpenAI 兼容模型通用。

## 核心概念

### 工具暴露方式（ToolExposure）

每个工具有两种暴露方式：

| 暴露方式 | 含义 |
|----------|------|
| `direct` | 直接出现在模型的 `tools` schema 中，模型可直接调用（默认） |
| `deferred` | 从模型 schema 中隐藏，但仍可通过 `tool_call` 间接调用 |

### 桥接工具（Bridge Tools）

启用 Tool Search 后，三个 LOCAL 工具被自动注入并暴露给模型：

| 工具 | 作用 |
|------|------|
| `tool_search` | 按关键字检索 deferred 工具 |
| `tool_describe` | 按名字获取某个 deferred 工具的完整 `input_schema` |
| `tool_call` | 按名字调用某个 deferred 工具，传入与其 `input_schema` 匹配的参数 |

模型的典型工作流是 **检索 → （查看 schema）→ 调用** ：

```
tool_search(query="...")  ->  tool_describe(tool_name="...")  ->  tool_call(tool_name="...", params={...})
```

`tool_search` 在命中结果唯一时会直接附带该工具的 `input_schema` ，模型可省去 `tool_describe` 一步。

## 配置

Tool Search 通过 `ToolSearchConfig` 配置。 Main Agent 已内置 `WithToolSearch` mixin ，可直接在 `agent_config.tool_search` 下配置。

### ToolSearchConfig 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `False` | 是否启用 Tool Search |
| `exposure_overrides` | `dict[str, ToolExposure]` | `{}` | 按 **canonical（注册）名** 指定单个工具的暴露方式 |
| `defer_all_mcp` | `bool` | `False` | 是否把所有 MCP 工具自动设为 `deferred` |
| `inject_reminder` | `bool` | `True` | 是否在首个 user prompt 末尾追加隐藏工具目录提示（本地工具列名 + MCP server 名与工具数量） |

### 延迟规则（优先级）

某个工具是否被延迟，按如下规则判定：

1. 若 `exposure_overrides` 中显式指定了该工具，则以此为准（`deferred` 延迟、`direct` 不延迟）。
2. 否则，若 `defer_all_mcp` 为真且该工具是 MCP 工具，则延迟。
3. 否则，直接暴露。

也就是说 `exposure_overrides` 优先级高于 `defer_all_mcp` ：可以「延迟全部 MCP ，但把其中某个保留为 direct 」。

### 约束

- **不能与 `tool_schema_overrides` 同时使用** 。 Main Agent 在启用 Tool Search 时若检测到 `tool_schema_overrides` ，会直接抛错（`tool_schema_overrides cannot be used when tool_search.enabled is true`）。两者都改写模型可见的工具 schema ，语义会冲突。
- **桥接工具名是保留名** （`tool_search` / `tool_describe` / `tool_call`）。 不能把同名工具作为真实工具提供，也不能在 `allowed_tool_names` 中显式列出它们。
- 指向不存在工具的 `exposure_overrides` 条目会被 **静默忽略** ，不报错。
- 当 `enabled` 为真且 `inject_reminder` 为真（默认）时，框架会在首个 user prompt 末尾追加一段 `<system-reminder>`：列出本地 deferred 工具名，以及可用 MCP server 的名字与工具数量（**不列 MCP 内部具体工具名**），帮助模型更有方向地检索。无任何 deferred 工具时不追加。把 `inject_reminder` 设为 `false` 可关闭该提示。

## 用法

### 在 YAML 配置中启用

下面是一个完整示例（即仓库内的 `workspace/test_tool_search.yaml`）：加载三个本地 stdio MCP server（通用 demo / 浏览器 / 文件系统，共 **50 个工具**），用 `defer_all_mcp` 把它们全部藏到桥接工具背后，让模型必须先检索再调用。 注意 `tools` 是白名单——每个 MCP 工具都要显式列出才放行（MCP 工具名为 `<服务名>_<工具名>`，需连上 server 后才能确定，可先跑一次看日志里的 `Context MCP tools` 列表）。

```yaml
task_name: "identity"
task_config: {}

runtime_provider_name: "gongfeng"
runtime_provider_config:
  runtime_image: "current"
  api_base_url: http://hyrl-sandbox.prod.woa.com:8052
  token: "<RUNTIME_TOKEN>"
runtime_configs:
  main:
    image: mirrors.tencent.com/hunyuan_yanguan/ubuntu:24.04
    worker_venv: false

agent_name: "main_agent"
agent_config:
  max_iterations: 15
  system_prompt: |
    You are a helpful assistant. Most tools are NOT listed directly — there are
    50+ tools available behind tool search. Use tool_search to find a tool by
    keyword, read its schema with tool_describe if needed, then invoke it with
    tool_call. Put your final answer in the last turn.
  user_prompt: |
    Create a directory at /private/tmp/ts_demo, write a file notes.txt inside it
    containing the single line "tool search works", then read the file back and
    report its exact contents. You must discover the right tools via tool_search.
  # tools 是白名单：50 个 MCP 工具都要列出才放行
  tools:
    # everything (13) — 通用 demo server
    - "everything_echo"
    - "everything_get-annotated-message"
    - "everything_get-env"
    - "everything_get-resource-links"
    - "everything_get-resource-reference"
    - "everything_get-structured-content"
    - "everything_get-sum"
    - "everything_get-tiny-image"
    - "everything_gzip-file-as-resource"
    - "everything_simulate-research-query"
    - "everything_toggle-simulated-logging"
    - "everything_toggle-subscriber-updates"
    - "everything_trigger-long-running-operation"
    # playwright (23) — 浏览器自动化
    - "playwright_browser_click"
    - "playwright_browser_close"
    - "playwright_browser_console_messages"
    - "playwright_browser_drag"
    - "playwright_browser_drop"
    - "playwright_browser_evaluate"
    - "playwright_browser_file_upload"
    - "playwright_browser_fill_form"
    - "playwright_browser_handle_dialog"
    - "playwright_browser_hover"
    - "playwright_browser_navigate"
    - "playwright_browser_navigate_back"
    - "playwright_browser_network_request"
    - "playwright_browser_network_requests"
    - "playwright_browser_press_key"
    - "playwright_browser_resize"
    - "playwright_browser_run_code_unsafe"
    - "playwright_browser_select_option"
    - "playwright_browser_snapshot"
    - "playwright_browser_tabs"
    - "playwright_browser_take_screenshot"
    - "playwright_browser_type"
    - "playwright_browser_wait_for"
    # filesystem (14)
    - "fs_create_directory"
    - "fs_directory_tree"
    - "fs_edit_file"
    - "fs_get_file_info"
    - "fs_list_allowed_directories"
    - "fs_list_directory"
    - "fs_list_directory_with_sizes"
    - "fs_move_file"
    - "fs_read_file"
    - "fs_read_media_file"
    - "fs_read_multiple_files"
    - "fs_read_text_file"
    - "fs_search_files"
    - "fs_write_file"
  # 本地 stdio MCP server（无鉴权、无外网）。工具名为 "<服务名>_<工具名>"
  mcp_servers:
    - name: "everything"
      transport: "stdio"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-everything"]
    - name: "playwright"
      transport: "stdio"
      command: "npx"
      args: ["-y", "@playwright/mcp@latest"]
    - name: "fs"
      transport: "stdio"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"]
  # 客户端工具检索：把全部 50 个 MCP 工具藏到 tool_search/tool_describe/tool_call 背后
  tool_search:
    enabled: true
    defer_all_mcp: true
    inject_reminder: true

openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "<OPENAI_API_KEY>"
        base_url: "http://trpc-gpt-eval.production.polaris:8080/v1"
      request_args:
        model: "deepseek-v4-pro"

processors:
  chat:
    name: "chat_tracker"
```

如果只想延迟特定工具，用 `exposure_overrides` 替代 `defer_all_mcp`（键为 canonical 名）：

```yaml
  tool_search:
    enabled: true
    exposure_overrides:
      "fs_write_file": "deferred"
```

### 运行结果

注意首条 `user` 消息末尾被自动追加了一段 `<system-reminder>`（由 `inject_reminder` 控制，默认关闭）：它只列出可用的 MCP server 名字与工具数量（`everything (13 tools)` / `fs (14 tools)` / `playwright (23 tools)`），不展开具体工具名，给模型一个检索方向。

```jsonc
[
  {
    "role": "system",
    "content": "You are a helpful assistant. Most tools are NOT listed directly — there are\n50+ tools available behind tool search. Use tool_search to find a tool by\nkeyword, read its schema with tool_describe if needed, then invoke it with\ntool_call. Put your final answer in the last turn.\n"
  },
  {
    "role": "user",
    "content": "Create a directory at /private/tmp/ts_demo, write a file notes.txt inside it\ncontaining the single line \"tool search works\", then read the file back and\nreport its exact contents. You must discover the right tools via tool_search.\n\n\n<system-reminder>\nAdditional tools are available via tool_search but are not listed above.\nAvailable MCP servers: everything (13 tools), fs (14 tools), playwright (23 tools)\nUse tool_search to find a tool by keyword, then tool_call to invoke it.\n</system-reminder>"
  },
  {
    "content": "",
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_00_gFBWFctMOUTuDUn08yKU0878",
        "function": {
          "arguments": "{\"query\": \"create directory\"}",
          "name": "tool_search"
        },
        "type": "function",
        "index": 0
      },
      {
        "id": "call_01_aixa4qjeDdlbUhIUs8cY9035",
        "function": {
          "arguments": "{\"query\": \"write file\"}",
          "name": "tool_search"
        },
        "type": "function",
        "index": 1
      },
      {
        "id": "call_02_UyAs9vzDAMwXoz5Q3oPP6867",
        "function": {
          "arguments": "{\"query\": \"read file\"}",
          "name": "tool_search"
        },
        "type": "function",
        "index": 2
      }
    ],
    "reasoning_content": "The user wants me to:\n1. Create a directory at /private/tmp/ts_demo\n2. Write a file notes.txt inside it containing \"tool search works\"\n3. Read the file back and report its contents\n\nI need to discover the right tools via tool_search. Let me search for directory creation, file writing, and file reading tools."
  },
  {
    "role": "tool",
    "tool_call_id": "call_00_gFBWFctMOUTuDUn08yKU0878",
    "content": "{\"matches\": [{\"canonical_name\": \"fs_create_directory\", \"description\": \"Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Perfect for setting up directory structures for projects or ensuring required paths exist. Only works within allowed directories.\", \"name\": \"fs_create_directory\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_directory_tree\", \"description\": \"Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.\", \"name\": \"fs_directory_tree\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_list_directory\", \"description\": \"Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is essential for understanding directory structure and finding specific files within a directory. Only works within allowed directories.\", \"name\": \"fs_list_directory\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_list_directory_with_sizes\", \"description\": \"Get a detailed listing of all files and directories in a specified path, including sizes. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is useful for understanding directory structure and finding specific files within a directory. Only works within allowed directories.\", \"name\": \"fs_list_directory_with_sizes\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_get_file_info\", \"description\": \"Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.\", \"name\": \"fs_get_file_info\", \"source\": \"mcp\"}], \"total\": 9}"
  },
  {
    "role": "tool",
    "tool_call_id": "call_01_aixa4qjeDdlbUhIUs8cY9035",
    "content": "{\"matches\": [{\"canonical_name\": \"fs_write_file\", \"description\": \"Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.\", \"name\": \"fs_write_file\", \"source\": \"mcp\"}, {\"canonical_name\": \"everything_gzip-file-as-resource\", \"description\": \"Compresses a single file using gzip compression. Depending upon the selected output type, returns either the compressed data as a gzipped resource or a resource link, allowing it to be downloaded in a subsequent request during the current session.\", \"name\": \"everything_gzip-file-as-resource\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_edit_file\", \"description\": \"Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style diff showing the changes made. Only works within allowed directories.\", \"name\": \"fs_edit_file\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_get_file_info\", \"description\": \"Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.\", \"name\": \"fs_get_file_info\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_read_file\", \"description\": \"Read the complete contents of a file as text. DEPRECATED: Use read_text_file instead.\", \"name\": \"fs_read_file\", \"source\": \"mcp\"}], \"total\": 14}"
  },
  {
    "role": "tool",
    "tool_call_id": "call_02_UyAs9vzDAMwXoz5Q3oPP6867",
    "content": "{\"matches\": [{\"canonical_name\": \"fs_read_file\", \"description\": \"Read the complete contents of a file as text. DEPRECATED: Use read_text_file instead.\", \"name\": \"fs_read_file\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_read_media_file\", \"description\": \"Read an image or audio file. Returns the base64 encoded data and MIME type. Only works within allowed directories.\", \"name\": \"fs_read_media_file\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_read_text_file\", \"description\": \"Read the complete contents of a file from the file system as text. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Use the 'head' parameter to read only the first N lines of a file, or the 'tail' parameter to read only the last N lines of a file. Operates on the file as text regardless of extension. Only works within allowed directories.\", \"name\": \"fs_read_text_file\", \"source\": \"mcp\"}, {\"canonical_name\": \"fs_read_multiple_files\", \"description\": \"Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.\", \"name\": \"fs_read_multiple_files\", \"source\": \"mcp\"}, {\"canonical_name\": \"everything_gzip-file-as-resource\", \"description\": \"Compresses a single file using gzip compression. Depending upon the selected output type, returns either the compressed data as a gzipped resource or a resource link, allowing it to be downloaded in a subsequent request during the current session.\", \"name\": \"everything_gzip-file-as-resource\", \"source\": \"mcp\"}], \"total\": 14}"
  },
  {
    "content": "Good, I found the tools. Let me get their full schemas, then proceed.",
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_00_CdoCJCUNER7piXYFodMt8598",
        "function": {
          "arguments": "{\"tool_name\": \"fs_create_directory\"}",
          "name": "tool_describe"
        },
        "type": "function",
        "index": 0
      },
      {
        "id": "call_01_gSRqRibsfe7PXSMVQBWG9095",
        "function": {
          "arguments": "{\"tool_name\": \"fs_write_file\"}",
          "name": "tool_describe"
        },
        "type": "function",
        "index": 1
      },
      {
        "id": "call_02_GS3Fv1SpoRmCWHLc5bA74343",
        "function": {
          "arguments": "{\"tool_name\": \"fs_read_text_file\"}",
          "name": "tool_describe"
        },
        "type": "function",
        "index": 2
      }
    ],
    "reasoning_content": "I found the tools I need:\n1. `fs_create_directory` - to create the directory\n2. `fs_write_file` - to write the file\n3. `fs_read_text_file` - to read the file back (the newer, preferred one over `fs_read_file`)\n\nLet me first get the full schemas for these tools, then invoke them."
  },
  {
    "role": "tool",
    "tool_call_id": "call_00_CdoCJCUNER7piXYFodMt8598",
    "content": "{\"canonical_name\": \"fs_create_directory\", \"description\": \"Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Perfect for setting up directory structures for projects or ensuring required paths exist. Only works within allowed directories.\", \"input_schema\": {\"$schema\": \"http://json-schema.org/draft-07/schema#\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"], \"type\": \"object\"}, \"name\": \"fs_create_directory\", \"source\": \"mcp\"}"
  },
  {
    "role": "tool",
    "tool_call_id": "call_01_gSRqRibsfe7PXSMVQBWG9095",
    "content": "{\"canonical_name\": \"fs_write_file\", \"description\": \"Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.\", \"input_schema\": {\"$schema\": \"http://json-schema.org/draft-07/schema#\", \"properties\": {\"content\": {\"type\": \"string\"}, \"path\": {\"type\": \"string\"}}, \"required\": [\"path\", \"content\"], \"type\": \"object\"}, \"name\": \"fs_write_file\", \"source\": \"mcp\"}"
  },
  {
    "role": "tool",
    "tool_call_id": "call_02_GS3Fv1SpoRmCWHLc5bA74343",
    "content": "{\"canonical_name\": \"fs_read_text_file\", \"description\": \"Read the complete contents of a file from the file system as text. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Use the 'head' parameter to read only the first N lines of a file, or the 'tail' parameter to read only the last N lines of a file. Operates on the file as text regardless of extension. Only works within allowed directories.\", \"input_schema\": {\"$schema\": \"http://json-schema.org/draft-07/schema#\", \"properties\": {\"head\": {\"description\": \"If provided, returns only the first N lines of the file\", \"type\": \"number\"}, \"path\": {\"type\": \"string\"}, \"tail\": {\"description\": \"If provided, returns only the last N lines of the file\", \"type\": \"number\"}}, \"required\": [\"path\"], \"type\": \"object\"}, \"name\": \"fs_read_text_file\", \"source\": \"mcp\"}"
  },
  {
    "content": "Now let me create the directory and write the file.",
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_00_Qz0iHcxo8jrP5p3Aap6h0248",
        "function": {
          "arguments": "{\"tool_name\": \"fs_create_directory\", \"params\": {\"path\": \"/private/tmp/ts_demo\"}}",
          "name": "tool_call"
        },
        "type": "function",
        "index": 0
      }
    ],
    "reasoning_content": "Now I have the schemas. Let me:\n1. Create the directory /private/tmp/ts_demo\n2. Write the file notes.txt with content \"tool search works\"\n3. Read it back\n\nLet me do step 1 first, then 2 and 3."
  },
  {
    "role": "tool",
    "tool_call_id": "call_00_Qz0iHcxo8jrP5p3Aap6h0248",
    "content": "Successfully created directory /private/tmp/ts_demo"
  },
  {
    "content": "Directory created. Now let me write the file and read it back.",
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_00_LYCcQsqBqdR48vhiUNua4151",
        "function": {
          "arguments": "{\"tool_name\": \"fs_write_file\", \"params\": {\"path\":\"/private/tmp/ts_demo/notes.txt\",\"content\":\"tool search works\"}}",
          "name": "tool_call"
        },
        "type": "function",
        "index": 0
      }
    ],
    "reasoning_content": "Directory created. Now write the file and then read it back."
  },
  {
    "role": "tool",
    "tool_call_id": "call_00_LYCcQsqBqdR48vhiUNua4151",
    "content": "Successfully wrote to /private/tmp/ts_demo/notes.txt"
  },
  {
    "content": "File written. Now let me read it back.",
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_00_U6Pl0gyFmCJo7eTdd6v58845",
        "function": {
          "arguments": "{\"tool_name\": \"fs_read_text_file\", \"params\": {\"path\":\"/private/tmp/ts_demo/notes.txt\"}}",
          "name": "tool_call"
        },
        "type": "function",
        "index": 0
      }
    ],
    "reasoning_content": "File written. Now read it back."
  },
  {
    "role": "tool",
    "tool_call_id": "call_00_U6Pl0gyFmCJo7eTdd6v58845",
    "content": "tool search works"
  },
  {
    "content": "All done. Here's a summary:\n\n1. **Created directory** `/private/tmp/ts_demo` using `fs_create_directory`\n2. **Wrote file** `notes.txt` with the line `\"tool search works\"` using `fs_write_file`\n3. **Read it back** using `fs_read_text_file` — the exact contents are:\n\n```\ntool search works\n```\n\nThe file contains exactly that single line, confirming everything worked as expected.",
    "role": "assistant",
    "reasoning_content": "All three steps completed successfully. Let me summarize the results."
  }
]
```

这条轨迹完整走了 **检索 → 查看 schema → 调用** 三段（共 **6 轮**）：50 个 MCP 工具全程隐藏，模型先从 reminder 得知有 `everything` / `fs` / `playwright` 三个 server，于是并行发起三个 `tool_search`，再并行用 `tool_describe` 取回三个工具的完整 `input_schema`，最后顺序 `tool_call` 执行。 并行检索/查看说明桥接工具在 `parallel_tool_calls` 下也正常工作。 一个细节：检索命中里 `fs_read_file` 标了 `DEPRECATED`，模型据此主动改用了 `fs_read_text_file`——印证了清晰的工具描述对检索质量的价值。

### 在代码中使用 ToolExecutor

```python
from nexus.core.tool import ToolExecutor, ToolSearchConfig, ToolExposure

executor = ToolExecutor(
    tool_funcs,
    allowed_tool_names=['alpha', 'beta'],
    tool_search_config=ToolSearchConfig(
        enabled=True,
        exposure_overrides={'beta': ToolExposure.DEFERRED},
    ),
)

# 模型可见 schema：只含 alpha + 三个桥接工具，不含 beta
schemas = executor.get_tool_schemas()
```

### tool_search 查询语法

`tool_search` 的 `query` 支持以下写法：

| 写法 | 含义 |
|------|------|
| `"read file"` | 空格分词，逐词参与排序（匹配工具名与描述） |
| `"+browser click"` | `+` 前缀表示 **必需词** ：工具必须匹配 `browser` 才返回 |
| `"select:<name>"` | 直接按名字返回该工具的完整 schema（等价于 `tool_describe`） |

排序按相关度从高到低：工具名完全相等 > 命中工具名分词 > 工具名子串 > 命中描述中的词。 `max_results` 由模型在调用 `tool_search` 时自行指定（默认 5 ，上限 10）。

## 调试建议

- 若模型用具体关键字搜不到工具，多半是 **查询词与工具名/描述用词不一致** （Tool Search 是关键字匹配，不做语义检索）。 隐藏工具的目录会通过 user prompt 末尾的 reminder 提示给模型（本地工具名 + 可用 MCP server），可据此引导搜索方向。 若想提高首次命中率，可在工具描述中补充同义词。
- 运行时若报 `Tool '<name>' not found ... Context MCP tools: [...]` ，说明 `tools` 中写的 MCP 工具名与实际加载到的名字不一致；以日志里的 `Context MCP tools` 列表为准修正即可（MCP 工具名由服务端决定，需连上后才能确定）。

## 参见

- [Main Agent](../reference/agent/main-agent.md) ：`tool_search` 等参数所在的 Agent 入口。
- [Tool Meta Override 使用指南](./tool_override_usage.md) ：另一种改写工具 schema 的机制（与 Tool Search 互斥）。
