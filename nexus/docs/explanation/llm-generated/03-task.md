# Task 任务系统

Task 是 Nexus 框架中的顶层执行单元，负责编排 Agent 执行并处理具体的业务场景。

## 目录

1. [概述](#概述)
2. [Task 结构](#task-结构)
3. [注册与使用](#注册与使用)
4. [输入输出 Mixin](#输入输出-mixin)
5. [实现模式](#实现模式)
6. [验证规则](#验证规则)

## 概述

Task 在 Nexus 架构中的位置：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Task 在架构中的位置                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    Runner                                                               │
│      │                                                                  │
│      │ 配置并执行                                                        │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         Task                                     │   │
│  │  • 定义业务场景（如 SWE-Bench、代码生成）                          │   │
│  │  • 编排 Agent 执行                                               │   │
│  │  • 管理 Runtime 生命周期                                          │   │
│  │  • 执行评估与验证                                                 │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│                                 │ run_agent()                           │
│                                 ▼                                       │
│                         ┌──────────────┐                                │
│                         │    Agent     │                                │
│                         │  • LLM 交互  │                                │
│                         │  • 工具调用  │                                │
│                         └──────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Task 的职责**：
- 定义业务输入参数
- 配置执行环境（Runtime）
- 调用 Agent 完成任务
- 收集结果并计算评分

## Task 结构

### 基础类型

```python
# 输入基类
class TaskRunInput(BaseModel):
    """Task 输入基类

    只包含业务参数，运行时依赖通过 ExecutionContext 获取
    """
    model_config = {'arbitrary_types_allowed': True}

# 输出基类
class TaskRunOutput(BaseModel):
    """Task 输出基类

    包含业务结果
    """
    model_config = {'arbitrary_types_allowed': True}

# Task 函数协议
class TaskFunc(Protocol):
    async def __call__(self, task_input: TaskRunInput) -> TaskRunOutput: ...
```

### 典型 Task 结构

```python
# 1. 定义输入
class MyTaskInput(TaskRunInput, WithAgentRunner):
    problem_id: str
    problem_statement: str
    timeout: int = 300

# 2. 定义输出
class MyTaskOutput(TaskRunOutput, WithScore):
    solution: str
    passed_tests: int
    total_tests: int

# 3. 实现 Task 函数
@task_registry.register('my_task')
async def my_task(task_input: MyTaskInput) -> MyTaskOutput:
    ctx = require_context()

    # 配置 Runtime
    runtime = await ctx.get_runtime()

    # 调用 Agent
    agent_output = await task_input.run_agent(
        user_prompt=task_input.problem_statement
    )

    # 评估结果
    result = await evaluate(runtime, agent_output)

    return MyTaskOutput(
        score=result.score,
        solution=result.solution,
        passed_tests=result.passed,
        total_tests=result.total,
    )
```

## 注册与使用

### 注册 Task

使用 `@task_registry.register()` 装饰器：

```python
from nexus.core import task_registry, TaskRunInput, TaskRunOutput

@task_registry.register('echo')
async def echo_task(task_input: EchoTaskInput) -> EchoTaskOutput:
    """Echo 示例任务"""
    ...
```

**注册时会自动验证**：
- 必须是 async 函数
- 必须恰好有一个参数
- 参数类型必须是 TaskRunInput 子类
- 返回类型必须是 TaskRunOutput 子类

### 配置运行

在 YAML 配置文件中引用：

```yaml
imports:
  - nexus.extensions.tasks  # 导入包含 Task 的模块

task_name: "echo"  # 注册名称
task_config:       # 传递给 TaskInput 的参数
  message: "Hello, World!"
  run_oracle: false
```

### 获取 Task

```python
from nexus.core import task_registry

# 获取 Task 函数
task_fn = task_registry.get('echo')

# 查看 Task 元数据
print(task_fn.nexus_meta)
# {'name': 'echo', 'type': 'task', 'meta': {}}
```

## 输入输出 Mixin

Nexus 提供了常用的 Mixin 类来扩展 Task 输入输出：

### 输入 Mixin

#### WithAgentRunner

提供 `run_agent` 函数用于调用 Agent：

```python
class WithAgentRunner(BaseModel):
    """需要调用 Agent 的 Task 使用"""
    run_agent: Callable[..., Awaitable[AgentRunOutput]]
```

**使用示例**：

```python
class MyTaskInput(TaskRunInput, WithAgentRunner):
    user_query: str

@task_registry.register('my_task')
async def my_task(task_input: MyTaskInput) -> MyTaskOutput:
    # run_agent 由 Runner 自动注入
    agent_output = await task_input.run_agent(
        user_prompt=task_input.user_query,
        max_iterations=10,
    )
```

### 输出 Mixin

#### WithScore

添加评分字段：

```python
class WithScore(BaseModel):
    """带评分的 Task 输出"""
    score: float
```

**使用示例**：

```python
class MyTaskOutput(TaskRunOutput, WithScore):
    details: str

# 返回时设置分数
return MyTaskOutput(score=0.95, details='...')
```

### Mixin 组合

可以组合多个 Mixin：

```python
class ComplexTaskInput(TaskRunInput, WithAgentRunner):
    """复杂任务输入"""
    instance_id: str
    problem_statement: str
    repo_url: str

class ComplexTaskOutput(TaskRunOutput, WithScore):
    """复杂任务输出"""
    patch: str
    test_results: dict
```

## 实现模式

### 模式 1：简单 Agent 任务

```python
@task_registry.register('simple_agent_task')
async def simple_agent_task(task_input: SimpleTaskInput) -> SimpleTaskOutput:
    """直接调用 Agent 完成任务"""

    agent_output = await task_input.run_agent(
        user_prompt=task_input.query
    )

    return SimpleTaskOutput(
        response=agent_output.main_trajectory[-1]['content']
    )
```

### 模式 2：多阶段任务

```python
@task_registry.register('multi_stage_task')
async def multi_stage_task(task_input: MultiStageInput) -> MultiStageOutput:
    """多阶段任务：Agent 执行 + 评估"""
    ctx = require_context()

    # 阶段 1：配置 Agent Runtime
    ctx.runtime_provider.configure('main', image='python:3.11')

    # 阶段 2：调用 Agent
    agent_output = await task_input.run_agent(
        user_prompt=task_input.problem
    )

    # 阶段 3：在 Runtime 中评估
    runtime = await ctx.get_runtime()
    score = await runtime.call_function(evaluate, agent_output)

    return MultiStageOutput(score=score)
```

### 模式 3：多 Runtime 任务

```python
@task_registry.register('multi_runtime_task')
async def multi_runtime_task(task_input: MultiRuntimeInput) -> MultiRuntimeOutput:
    """使用多个隔离的 Runtime"""
    ctx = require_context()

    # Agent Runtime：Python 环境
    ctx.runtime_provider.configure('main', image='python:3.11')

    # 评估 Runtime：独立沙箱
    ctx.runtime_provider.configure('eval', image='eval-sandbox:latest')

    # Agent 阶段
    agent_output = await task_input.run_agent(user_prompt='...')

    # 提取产物
    main_runtime = await ctx.get_runtime('main')
    patch = await main_runtime.call_function(get_patch)

    # 在独立环境评估
    eval_runtime = await ctx.get_runtime('eval')
    result = await eval_runtime.call_function(run_tests, patch)

    return MultiRuntimeOutput(score=result.score, patch=patch)
```

### 模式 4：Oracle 任务（跳过 Agent）

```python
@task_registry.register('oracle_task')
async def oracle_task(task_input: OracleTaskInput) -> OracleTaskOutput:
    """支持 Oracle 模式的任务，用于测试评估逻辑"""
    ctx = require_context()
    runtime = await ctx.get_runtime()

    if task_input.run_oracle:
        # 直接执行正确答案，跳过 Agent
        await runtime.call_function(
            write_solution,
            task_input.expected_solution
        )
    else:
        # 正常调用 Agent
        await task_input.run_agent(user_prompt=task_input.problem)

    # 评估（对 Agent 和 Oracle 使用相同的评估逻辑）
    score = await runtime.call_function(evaluate)

    return OracleTaskOutput(score=score)
```

## 验证规则

Task 注册时会进行以下验证：

```python
def validate_task(obj: Any) -> None:
    # 1. 必须是可调用对象
    if not callable(obj):
        raise TypeError(f'Task must be callable')

    # 2. 必须是 async 函数
    if not inspect.iscoroutinefunction(obj):
        raise TypeError(f'Task must be an async function')

    # 3. 必须恰好有一个参数
    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    if len(params) != 1:
        raise TypeError(f'Task must have exactly one input parameter')

    # 4. 参数必须有类型注解
    param = params[0]
    if param.annotation is inspect.Parameter.empty:
        raise TypeError('Task input parameter must have a type annotation')

    # 5. 参数类型必须是 TaskRunInput 子类
    param_type = param.annotation
    if not issubclass(param_type, TaskRunInput):
        raise TypeError(f'Task input must be a subclass of TaskRunInput')

    # 6. 必须有返回类型注解
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError('Task must have a return type annotation')

    # 7. 返回类型必须是 TaskRunOutput 子类
    return_type = sig.return_annotation
    if not issubclass(return_type, TaskRunOutput):
        raise TypeError(f'Task return type must be a subclass of TaskRunOutput')
```

## 完整示例

```python
"""完整的 Task 实现示例"""

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)


class CodeGenTaskInput(TaskRunInput, WithAgentRunner):
    """代码生成任务输入"""
    problem_id: str
    problem_description: str
    language: str = 'python'
    test_cases: list[dict] = []


class CodeGenTaskOutput(TaskRunOutput, WithScore):
    """代码生成任务输出"""
    generated_code: str
    passed_tests: int
    total_tests: int
    error_message: str | None = None


@task_registry.register('code_gen')
async def code_gen_task(task_input: CodeGenTaskInput) -> CodeGenTaskOutput:
    """代码生成任务

    1. 调用 Agent 生成代码
    2. 在 Runtime 中执行测试
    3. 计算通过率
    """
    ctx = require_context()

    # 配置 Runtime 环境
    ctx.runtime_provider.configure('main', image=f'{task_input.language}:latest')
    runtime = await ctx.get_runtime()

    # 构造 Agent prompt
    prompt = f"""
    Please solve the following problem in {task_input.language}:

    {task_input.problem_description}

    Write the solution to /workspace/solution.{task_input.language}
    """

    # 调用 Agent
    try:
        await task_input.run_agent(user_prompt=prompt)
    except Exception as e:
        return CodeGenTaskOutput(
            score=0.0,
            generated_code='',
            passed_tests=0,
            total_tests=len(task_input.test_cases),
            error_message=str(e),
        )

    # 读取生成的代码
    code = await runtime.call_function(
        lambda: open(f'/workspace/solution.{task_input.language}').read()
    )

    # 运行测试
    passed = 0
    for test_case in task_input.test_cases:
        result = await runtime.call_function(run_test, code, test_case)
        if result.passed:
            passed += 1

    total = len(task_input.test_cases)
    score = passed / total if total > 0 else 0.0

    return CodeGenTaskOutput(
        score=score,
        generated_code=code,
        passed_tests=passed,
        total_tests=total,
    )
```

## 相关文档

- [Agent 系统详解](./04-agent.md)
- [ExecutionContext 详解](./02-execution-context.md)
- [执行流程教程](../getting-started/02-execution-flow.md)
