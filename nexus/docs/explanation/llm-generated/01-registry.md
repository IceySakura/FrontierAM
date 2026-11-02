# Registry 注册表系统

Registry 是 Nexus 框架的核心扩展机制，通过装饰器模式实现 Task、Agent、Tool 等组件的统一注册与管理。

## 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [核心 API](#核心-api)
4. [使用示例](#使用示例)
5. [内置注册表](#内置注册表)
6. [高级用法](#高级用法)

## 概述

Registry 系统提供：

- **装饰器注册**：通过 `@registry.register('name')` 注册组件
- **懒加载**：支持延迟导入，减少启动时间
- **类型安全**：泛型支持，保证类型安全
- **验证处理**：可自定义 processor 进行注册时验证
- **自动安装**：导入失败时可自动安装依赖

```
┌─────────────────────────────────────────────────────────────┐
│                     Registry 系统架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │task_registry│    │agent_registry│   │tool_registry│     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                            ▼                                │
│                  ┌─────────────────┐                        │
│                  │  GLOBAL_STATE   │                        │
│                  │  ['registries'] │                        │
│                  └─────────────────┘                        │
│                                                             │
│  Registry[T] 泛型类：                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • type_name: str          # 注册表类型名              │   │
│  │ • processor: Callable     # 注册时验证/处理函数       │   │
│  │ • _registry: dict[str, RegistryEntry]               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  RegistryEntry 数据类：                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • name: str               # 条目名称                  │   │
│  │ • obj: Any | None         # 实际对象（懒加载时为None）│   │
│  │ • import_path: str | None # 导入路径                  │   │
│  │ • install_commands: list  # 自动安装命令              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 架构设计

### Registry 类

```python
class Registry(Generic[T]):
    """类型安全的泛型注册表"""

    def __init__(
        self,
        type_name: str,
        processor: Callable[[T], dict | None] | Callable[[T, dict], dict | None] | None = None,
    ):
        self.type_name = type_name
        self.processor = processor  # 注册时的验证/处理函数
        self._registry: dict[str, RegistryEntry] = {}
```

### RegistryEntry 结构

```python
@dataclass
class RegistryEntry:
    name: str
    obj: Any = None              # None 表示尚未加载
    import_path: str | None = None  # 用于懒加载
    install_commands: list[str] = field(default_factory=list)

    @property
    def is_loaded(self) -> bool:
        return self.obj is not None
```

## 核心 API

### register() - 装饰器注册

```python
@registry.register('my_component', meta={'key': 'value'})
def my_component():
    pass
```

**工作流程**：

```
┌────────────────┐
│ @register()    │
│ 装饰器调用      │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ processor()    │
│ 验证/处理对象   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 获取 import_path│
│ (模块路径)      │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 设置 nexus_meta │
│ 到对象上        │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 存入 _registry │
└────────────────┘
```

### add() - 懒加载注册

```python
registry.add(
    'lazy_component',
    import_path='my_module::MyComponent',
    install_commands=['pip install my-package']
)
```

### get() - 获取组件

```python
component = registry.get('my_component')  # 如果是懒加载，会自动导入
```

### import_module() - 批量导入

```python
registrations = Registry.import_module('nexus.extensions')
# 返回: [{'type': 'task', 'name': 'echo'}, {'type': 'agent', 'name': 'demo_agent'}, ...]
```

## 使用示例

### 创建自定义注册表

```python
from nexus.core.registry import Registry

def validate_my_type(obj, meta):
    """注册时的验证函数"""
    if not callable(obj):
        raise TypeError('Must be callable')
    return meta  # 返回处理后的 meta

# 创建泛型注册表
my_registry = Registry[MyType]('my_type', validate_my_type)

# 使用装饰器注册
@my_registry.register('component_a', meta={'version': '1.0'})
def component_a():
    pass

# 获取组件
fn = my_registry.get('component_a')
print(fn.nexus_meta)  # {'name': 'component_a', 'type': 'my_type', 'meta': {'version': '1.0'}}
```

### 懒加载模式

```python
# 添加懒加载条目
my_registry.add(
    'heavy_component',
    import_path='heavy_module::HeavyComponent',
    install_commands=['pip install heavy-package']
)

# 首次调用时才会导入
component = my_registry.get('heavy_component')  # 触发 importlib.import_module()
```

### 批量导入

```python
# 导入模块并追踪注册
registrations = Registry.import_module('nexus.extensions')
# 输出:
# Imported 'nexus.extensions' with 5 registrations:
#   - [task] echo
#   - [agent] demo_agent
#   - [tool] demo/think, demo/bash
```

## 内置注册表

Nexus 提供以下内置注册表：

| 注册表 | 类型 | 说明 | 位置 |
|--------|------|------|------|
| `task_registry` | `TaskFunc` | 任务注册表 | `nexus.core.task` |
| `agent_registry` | `AgentFunc` | Agent 注册表 | `nexus.core.agent` |
| `tool_registry` | `ToolFunc` | 工具注册表 | `nexus.core.tool` |
| `openai_client_registry` | `OpenAICompatibleClient` | LLM 客户端注册表 | `nexus.core.openai_client` |
| `runtime_provider_registry` | `RuntimeProvider` | Runtime Provider 注册表 | `nexus.runtime.provider` |
| `executor_registry` | `Executor` | 执行器注册表 | `nexus.core.executor` |
| `span_processor_registry` | `SpanProcessor` | Span 处理器注册表 | `nexus.trace.processors` |

### 注册表关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          内置注册表关系                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │  task_registry   │──────────────┐                                    │
│  │   • echo         │              │                                    │
│  │   • swe_bench    │              │                                    │
│  └──────────────────┘              │                                    │
│           │                        │                                    │
│           │ 调用                    │                                    │
│           ▼                        │                                    │
│  ┌──────────────────┐              │                                    │
│  │  agent_registry  │              │                                    │
│  │   • demo_agent   │◄─────────────┘                                    │
│  │   • swe_agent    │                                                   │
│  └──────────────────┘                                                   │
│           │                                                             │
│           │ 使用                                                         │
│           ▼                                                             │
│  ┌──────────────────┐     ┌──────────────────────┐                      │
│  │  tool_registry   │     │ openai_client_registry│                     │
│  │   • demo/think   │     │   • openai            │                     │
│  │   • demo/bash    │     │   • azure_openai      │                     │
│  └──────────────────┘     └──────────────────────┘                      │
│           │                                                             │
│           │ RUNTIME 工具使用                                             │
│           ▼                                                             │
│  ┌────────────────────────┐     ┌──────────────────────┐                │
│  │runtime_provider_registry│    │ executor_registry    │                │
│  │   • docker             │     │   • file             │                │
│  │   • kubernetes         │     │   • jupyter          │                │
│  └────────────────────────┘     │   • command          │                │
│                                 └──────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

## 高级用法

### 自定义 Processor

Processor 在注册时被调用，用于验证和转换对象：

```python
def task_processor(func, meta):
    """Task 注册时的验证"""
    # 检查是否为协程函数
    if not inspect.iscoroutinefunction(func):
        raise TypeError('Task must be async function')

    # 检查参数签名
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if len(params) != 1:
        raise TypeError('Task must have exactly one parameter')

    # 可以修改 meta
    meta['validated'] = True
    return meta

task_registry = Registry[TaskFunc]('task', task_processor)
```

### 获取导入路径

```python
from nexus.core.registry import get_import_path, import_from_path

# 获取对象的导入路径
path = get_import_path(my_function)
# 'my_module::my_function'

# 从路径导入对象
obj = import_from_path('my_module::my_function')
```

### 列出所有条目

```python
# 获取所有条目（包括未加载的）
entries = task_registry.list_entries()
for entry in entries:
    print(f'{entry.name}: loaded={entry.is_loaded}')

# 获取所有已加载的对象
all_tasks = task_registry.get_all()
```

## 设计原则

1. **解耦**：组件通过名称引用，不需要直接导入
2. **懒加载**：减少启动时间，按需加载
3. **类型安全**：泛型支持，IDE 可以提供类型提示
4. **可扩展**：用户可以创建自己的注册表
5. **可追踪**：导入时记录所有注册，便于调试

## 相关文档

- [Task 系统详解](./03-task.md)
- [Agent 系统详解](./04-agent.md)
- [Tool 系统详解](./05-tool.md)
