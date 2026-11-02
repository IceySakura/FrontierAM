# Tutorial

Nexus 是一套面向研究场景的 Agent 框架。 它在架构设计上对 Agent、Tool、Task 等关键概念进行了解耦，使它们能够灵活组合，广泛适用于强化学习、RFT/SFT 数据生成、评测等研究场景。

本教程通过一个内置的示例任务，介绍如何使用 Nexus 运行一个完整的 Agent 任务流程。

## 前置准备

### 安装 Nexus

如果你已经有 Python 环境(3.12-3.14)，可以直接使用 pip 安装:

```bash
pip install tencent-nexus --extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
```

如果你还没有 Python 环境，可以先使用 uv 创建虚拟环境:

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建并激活虚拟环境
uv venv --python 3.13
uv pip install pip
source .venv/bin/activate

# 然后使用 pip 安装 Nexus
pip install tencent-nexus --extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
```

!!! tip "更多安装方式"

    如果你需要修改 Nexus 源码或在项目中引入 Nexus 作为依赖，请参考 [安装文档](installation.md) 了解更多安装方式。

## 快速开始

### 1. 创建配置文件

新建文件 `test.yaml`，内容如下:

```yaml title="test.yaml"
task_name: "echo"
task_config:
  message: "Hello, Nexus!"
  run_oracle: false

runtime_provider_name: "docker"
runtime_configs:
  main:
    worker_venv: false

agent_name: "demo_agent"
agent_config:
  max_iterations: 10

openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "https://api.openai.com/v1"
      request_args:
        model: "gpt-4o"

processors:
  chat:
    name: "chat_tracker"
```

配置文件说明：

- `task_name` 和 `task_config`：指定要执行的任务及其参数
- `runtime_provider_name` 和 `runtime_configs`：配置隔离的运行时环境(这里使用 Docker)。 `worker_venv` 控制是否在沙盒中创建隔离的 Worker 虚拟环境
- `agent_name` 和 `agent_config`：指定使用的 Agent 及其参数
- `openai_clients`：配置 LLM 客户端(需要替换 `<your-api-key>` 为你的真实 API 密钥)
- `processors`：配置数据处理器，用于记录和处理执行过程中的数据

!!! note "模块发现"

    Nexus 1.0 使用元数据驱动的模块发现机制，不再需要手动 `imports` 声明。 框架会自动从 `pyproject.toml` 中发现并懒加载所有扩展模块。

!!! note "配置详解"

    - **API 配置**：如需使用其他 LLM 服务，请参考 [如何配置 LLM 客户端](../how-to/configure-llm.md)
    - **Docker 配置**：首次运行会自动构建运行时镜像，需要几分钟。 后续运行会复用已构建的镜像。 如无法使用 Docker，可参考 [运行时环境配置](../how-to/use-runtime/configure-runtime.md)

### 2. 运行任务

执行以下命令运行任务：

```bash
nexus run-single test.yaml -o result.json
```

参数说明：

- `test.yaml`：配置文件路径
- `-o result.json`：将任务输出保存到 `result.json` 文件

### 3. 查看结果

任务执行过程中会实时输出日志，最终结果会保存到 `result.json` 文件中。

??? note "示例输出"

    ```
    2026-01-12 19:49:03 | INFO | nexus.core.registry | Imported 'nexus.extensions' with 73 registrations:
    2026-01-12 19:49:03 | INFO | nexus.core.registry |   - [agent] browser_gym_agent, demo_agent, toolcall_agent, swe_agent, rondo_agent, terminus2, calculator_agent
    2026-01-12 19:49:03 | INFO | nexus.core.registry |   - [tool] browser_gym/submit_result, base/write_todos, base/finish, demo/think, demo/bash, swe_agent/bash, swe_agent/str_replace_editor, swe_agent/review_submit, calculator/calculate, browser/goto, browser/click, browser/fill, browser/scroll, browser/press_key, browser/hover, browser/select_option, browser/new_tab, browser/close_tab, browser/focus_tab, browser/go_back, browser/go_forward, browser/get_observation, python_code_interpreter, google_search, hunyuan_search, visit
    2026-01-12 19:49:03 | INFO | nexus.core.registry |   - [executor] create_browser_session, destroy_browser_session, browser_goto, browser_go_back, browser_go_forward, browser_click, browser_fill, browser_scroll, browser_press_key, browser_hover, browser_select_option, create_terminal_session, destroy_terminal_session, run_command_in_terminal_session, start_command_in_terminal_session, query_terminal_command_status, send_keys_to_terminal_session, capture_terminal_pane, terminate_terminal_session_processes
    2026-01-12 19:49:03 | INFO | nexus.core.registry |   - [openai_client] azure_openai, custom_function, hytuner, litellm, openai
    2026-01-12 19:49:03 | INFO | nexus.core.registry |   - [task] calculator_task, echo, math, search, terminal_task, swe_task
    2026-01-12 19:49:03 | INFO | nexus.runner.single | Auto-generated run_id: 6b666999670a
    2026-01-12 19:49:03 | INFO | nexus.runner.single | Starting runner for task: echo (run_id: 6b666999670a)
    2026-01-12 19:49:03 | INFO | nexus.runner.single | Executing task: echo
    2026-01-12 19:49:03 | INFO | nexus.extensions.tasks.echo | Running echo task with message Hello, Nexus!
    2026-01-12 19:49:03 | INFO | nexus.core.context | Acquiring runtime 'main'...
    2026-01-12 19:49:03 | INFO | nexus.runtime.docker | Image nexus-runtime:0.8.6 already exists, skipping build
    2026-01-12 19:49:03 | INFO | nexus.runtime.docker | Runtime container nexus-runtime-0-8-6 already exists
    2026-01-12 19:49:03 | INFO | nexus.runtime.docker | Starting container nexus-6b666999670a-main...
    2026-01-12 19:49:07 | INFO | nexus.runtime.docker | Runtime main ready at http://172.17.0.1:33153
    2026-01-12 19:49:07 | INFO | nexus.core.context | Runtime 'main' acquired successfully
    2026-01-12 19:49:09 | INFO | nexus.extensions.agents.demo.agent | Tool call: Bash({"command": "echo \"Hello, Nexus!\" > /tmp/echo.txt"})
    2026-01-12 19:49:09 | INFO | nexus.extensions.agents.demo.agent | Tool output: [{'text': '(no output)', 'type': 'text'}]
    2026-01-12 19:49:12 | INFO | nexus.extensions.agents.demo.agent | Model output: I've written "Hello, Nexus!" to the file /tmp/echo.txt. Let me verify the content was written correctly:
    2026-01-12 19:49:12 | INFO | nexus.extensions.agents.demo.agent | Tool call: Bash({"command": "cat /tmp/echo.txt"})
    2026-01-12 19:49:12 | INFO | nexus.extensions.agents.demo.agent | Tool output: [{'text': 'Hello, Nexus!\n', 'type': 'text'}]
    2026-01-12 19:49:13 | INFO | nexus.extensions.agents.demo.agent | Model output: Done! I've written "Hello, Nexus!" to the file `/tmp/echo.txt` and verified the content.
    2026-01-12 19:49:13 | INFO | nexus.runtime.client | call_function: sending request to http://172.17.0.1:33153/api/v1/call_function, func=get_echoed_message, args_count=0, kwargs_keys=[]
    2026-01-12 19:49:13 | INFO | nexus.runtime.client | call_function: received response from http://172.17.0.1:33153/api/v1/call_function, status=200
    2026-01-12 19:49:13 | INFO | nexus.runner.single | Task completed: echo
    2026-01-12 19:49:13 | INFO | nexus.runtime.docker | Stopping container nexus-6b666999670a-main...
    2026-01-12 19:49:13 | INFO | nexus.runtime.docker | Runtime main released
    2026-01-12 19:49:13 | INFO | nexus.cli.commands.run | Task output:
    {"score":1.0,"echoed_message":"Hello, Nexus!"}
    2026-01-12 19:49:13 | INFO | nexus.cli.commands.run | Output saved to: result.json
    ```

日志显示了完整的执行过程：加载扩展模块、启动 Docker 运行时环境、Agent 调用工具完成任务、返回结果并清理环境。

`result.json` 文件包含以下关键信息：

- **task_output**：任务的最终输出结果（通常包含score以及其它任务特定信息）
- **agent_output**：Agent 特定的输出信息
- **processor_results**：事件处理器记录的数据，本例中 `chat_tracker` 记录了所有的对话轨迹
- **success**：任务是否成功完成
- **error**：如果失败，记录错误信息

### 4. 将任务提交到云端

完成本地运行后，可以将 Nexus 项目提交到 Nexus Cloud 执行。Nexus Cloud 支持 Single 和 Batch，任务提交后可以在控制台查看状态、读取结果或停止运行中的任务。

- 服务入口：[cloud.nexus-agent.woa.com](https://cloud.nexus-agent.woa.com)
- 使用指南：[快速开始](../cloud/quickstart.md)

## 下一步

现在你已经成功运行了第一个 Nexus 任务，可以继续探索：

- [运行真实任务](../how-to/use-task/tutorial-task-swe.md)：尝试 SWE-Bench、TerminalBench 等真实研究场景的任务
- [配置文件详解](../reference/config-yaml.md)：深入了解配置文件的各个选项
- [创建新模块](../how-to/manage-modules/create-module.md)：创建自定义的 Agent、Task 或 Tool 模块
- [Module CLI](../how-to/manage-modules/module-cli.md)：管理模块依赖
- [提交到 Nexus Cloud](../cloud/overview.md)：生产任务、长时间运行任务和批量评测建议在云端执行，便于使用稳定的资源调度、
  持久化任务状态并通过控制台统一管理
