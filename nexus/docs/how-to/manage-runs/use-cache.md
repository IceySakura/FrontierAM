# Use Cache

Nexus 提供了缓存机制，用于录制和回放函数调用。 这对于以下场景非常有用：

- **确定性回放**：重现 Agent 的执行轨迹
- **调试和测试**：使用录制的交互进行调试
- **节省成本**：缓存 LLM 响应，减少 API 调用
- **故障恢复**：当外层函数失败但内层函数成功时，保留内层结果，重试时直接从缓存读取

## 核心概念

### 缓存阶段（Cache Stage）

缓存系统有两个阶段：

1. **REPLAY（回放）**：从缓存读取已记录的结果
2. **CACHE（录制）**：执行函数并将结果写入缓存

当提供的缓存数据包含已有条目时，系统从 REPLAY 阶段开始。 当所有缓存条目耗尽后，自动切换到 CACHE 阶段继续录制。

### 层级缓存结构

缓存使用树状结构存储函数调用：

```
outer_func() [INCOMPLETE]
├── inner_func_1() [SUCCESS] <- 已缓存
└── inner_func_2() [SUCCESS] <- 已缓存
```

当 `outer_func` 执行失败时，其子调用 `inner_func_1` 和 `inner_func_2` 的结果仍然被保留。 重试时，内层函数直接从缓存返回，外层函数重新执行。

### 缓存条目状态

每个缓存条目有三种状态：

- **SUCCESS**：函数成功完成
- **EXCEPTION**：函数抛出异常（当 `cache_exceptions=True` 时）
- **INCOMPLETE**：函数开始执行但未完成（可用于恢复场景）

## 配置文件使用

在配置文件中启用缓存：

```yaml
# 基本配置
imports:
  - nexus.extensions
  - nexus.extensions.tasks.identity  # identity task 需要单独导入

task_name: "identity"
task_config:
  prompt: "What is 2 + 2?"

agent_name: "demo_agent"
agent_config:
  max_iterations: 10

# 启用缓存
cache:
  enabled: true
  file_path: /tmp/cache/trajectory.pkl
  ignore_args_hash: false  # 是否忽略参数哈希进行匹配
  overwrite_tags: []       # 需要重新执行的标签列表

# OpenAI 客户端配置
openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: "your-api-key"
        base_url: "http://your-api-endpoint"
      request_args:
        model: "gpt-5"
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用缓存 |
| `file_path` | str | null | 缓存文件路径（pickle 格式） |
| `ignore_args_hash` | bool | false | 若为 true，仅按函数名匹配，忽略参数差异 |
| `overwrite_tags` | list[str] | null | 带有这些标签的条目将被重新执行 |

## 命令行使用

使用 `nexus run` 命令执行带缓存的任务：

```bash
# 首次运行，录制缓存
nexus run config.yaml

# 如果中途失败，再次运行会从缓存恢复
nexus run config.yaml

# 使用 --cache-file 覆盖配置文件中的缓存路径
nexus run config.yaml --cache-file /path/to/cache.pkl
```

## 编程使用

### 使用 Runner API

```python
from nexus.runner.single import RunInput, CacheConfig, run

# 创建运行输入
run_input = RunInput(
    imports=['nexus.extensions', 'nexus.extensions.tasks.identity'],
    task_name='identity',
    task_config={'prompt': 'What is 2 + 2?'},
    agent_name='demo_agent',
    agent_config={'max_iterations': 10},
    openai_clients={
        'main': {
            'name': 'litellm',
            'config': {
                'client_args': {
                    'api_key': 'your-api-key',
                    'base_url': 'http://your-api-endpoint',
                },
                'request_args': {'model': 'gpt-5'},
            },
        }
    },
    cache=CacheConfig(
        enabled=True,
        file_path='/tmp/cache/trajectory.pkl',
    ),
)

# 执行任务
output = await run(run_input)

# 查看缓存统计
print(f"Cache stats: {output.cache_stats}")
```

### 直接使用 Cache API

对于更细粒度的控制，可以直接使用缓存 API：

```python
from nexus.utils.cache import (
    cache_context,
    cacheable,
    load_cache,
    save_cache,
    get_cache_stats,
)

# 定义可缓存的函数
@cacheable(tags=['llm'])
async def llm_call(messages: list[dict]) -> str:
    # 实际的 LLM 调用
    return "response"

@cacheable(tags=['tool'])
def execute_tool(name: str, args: dict) -> dict:
    # 实际的工具执行
    return {"result": "done"}

# 录制模式
cache_data = {}
with cache_context(cache_data):
    response = await llm_call([{"role": "user", "content": "Hello"}])
    result = execute_tool("search", {"query": "test"})

# 保存缓存
save_cache(cache_data, '/tmp/cache.pkl')

# 回放模式
cache_data = load_cache('/tmp/cache.pkl')
with cache_context(cache_data):
    # 这些调用将从缓存返回，不会实际执行
    response = await llm_call([{"role": "user", "content": "Hello"}])
    result = execute_tool("search", {"query": "test"})

# 查看统计
print(get_cache_stats(cache_data))
```

### @cacheable 装饰器参数

```python
@cacheable(
    ignore_args=['timestamp'],  # 忽略这些参数进行哈希计算
    cache_exceptions=False,     # 是否缓存异常
    tags=['llm', 'completion'], # 用于分类和选择性失效的标签
)
async def my_function(...):
    ...
```

## 故障恢复示例

缓存的一个重要应用场景是故障恢复。 当 Agent 执行到一半报错时，可以利用缓存从断点继续：

### 场景说明

假设 Agent 执行过程如下：

1. LLM 调用成功 -> 缓存 ✓
2. 工具调用成功 -> 缓存 ✓
3. LLM 调用成功 -> 缓存 ✓
4. 工具调用**失败** -> 未缓存

当我们修复问题后重新运行：

1. LLM 调用 -> **从缓存读取**
2. 工具调用 -> **从缓存读取**
3. LLM 调用 -> **从缓存读取**
4. 工具调用 -> **重新执行**（这次成功）
5. 继续执行...

### 代码示例

```python
from nexus.utils.cache import cache_context, cacheable, load_cache, save_cache

# 模拟可能失败的工具
should_fail = True

@cacheable(tags=['llm'])
async def llm_call(prompt: str) -> str:
    print(f"LLM called with: {prompt}")
    return f"Response to: {prompt}"

@cacheable(tags=['tool'])
def risky_tool(action: str) -> str:
    global should_fail
    print(f"Tool called with: {action}")
    if should_fail:
        raise RuntimeError("Tool failed!")
    return f"Tool result: {action}"

async def agent_loop():
    # Step 1: LLM 调用
    response1 = await llm_call("What should I do?")
    print(f"Got: {response1}")

    # Step 2: 工具调用（可能失败）
    result = risky_tool("search for info")
    print(f"Tool result: {result}")

    # Step 3: 继续
    response2 = await llm_call(f"Got result: {result}")
    return response2

# 第一次运行：会在 risky_tool 处失败
cache_data = {}
try:
    with cache_context(cache_data):
        await agent_loop()
except RuntimeError as e:
    print(f"Failed: {e}")
    save_cache(cache_data, '/tmp/recovery.pkl')

# 修复问题
should_fail = False

# 第二次运行：从缓存恢复，跳过已成功的步骤
cache_data = load_cache('/tmp/recovery.pkl')
with cache_context(cache_data):
    result = await agent_loop()  # LLM 调用从缓存读取，工具重新执行
    print(f"Final result: {result}")
```

## 下一步

- **[Use Cache Demo](use-cache-demo.ipynb)**：完整的交互式示例，包含故障恢复场景演示
- **[Run Batch](../run-batch.md)**：了解如何批量运行任务
