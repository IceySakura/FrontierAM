# Shell Tool Internals

本文档对比分析 SWE-ReX、OpenHands、Codex CLI、Gemini CLI 和 Nexus 五个 AI Agent 框架中 Shell/Bash 工具的实现差异。

## 概述

AI Agent 执行 bash 命令是一个看似简单但实际复杂的问题。 核心挑战在于：

1. **状态追踪** ：如何可靠地获取命令执行状态（运行中/完成/退出码）
2. **命令边界检测** ：如何准确判断一条命令何时开始、何时结束
3. **交互式支持** ：如何处理长时间运行、需要输入的命令

五个框架采用了不同的架构：

| 框架 | 执行模式 | 底层机制 | 状态获取方式 | 命令边界检测 |
|------|----------|----------|--------------|--------------|
| SWE-ReX | 持久会话 | pexpect + PTY | `echo $?` 获取退出码 | 自定义 PS1 + expect |
| OpenHands | 持久会话 | tmux（libtmux） | PS1 元数据解析 | 屏幕轮询 + PS1 匹配 |
| Codex CLI | 子进程 | portable_pty | child.wait() | 进程退出事件 |
| Gemini CLI | 子进程 | node-pty | onExit 回调 | 进程退出事件 |
| Nexus | 持久会话 | 定制 Bash + tmux | 状态文件实时同步 | Bash Hook |

## 两种执行模式

根据是否维护持久的 shell 会话，可以将这些实现分为两大类：

### 子进程模式（Codex CLI、Gemini CLI）

每条命令启动一个新的 shell 子进程执行，命令完成后进程退出。

**特点** ：

- 命令边界检测简单：进程退出即命令完成
- 退出码获取直接：从进程退出状态获取
- 无状态：每条命令独立，不保留环境变量、工作目录等
- 不支持后台进程：`&` 启动的后台任务会随主进程退出

执行子进程最简单的方式是使用管道（pipe），例如 Python 的 `subprocess.Popen(stdout=PIPE, stderr=PIPE)`。 但 Codex CLI 和 Gemini CLI 都选择了更复杂的 [PTY（伪终端）](https://en.wikipedia.org/wiki/Pseudoterminal)，原因包括：

1. **程序行为一致性** ：许多程序会检测 `isatty()` 来决定输出格式（如是否带颜色、是否缓冲）
2. **交互式程序支持** ：某些命令可能需要 TTY 才能正常运行（如 `less`、`vim`）
3. **信号处理** ：PTY 提供正确的信号传递（Ctrl+C → SIGINT）
4. **行缓冲** ：PTY 默认行缓冲，输出更及时；pipe 可能全缓冲导致输出延迟

### 持久会话模式（SWE-ReX、OpenHands、Nexus）

维护一个长期运行的 shell 会话，所有命令在同一会话中执行。

**特点** ：

- 保持状态：环境变量、工作目录、shell 函数等在命令间保持
- 支持后台进程：可以启动后台任务并在后续命令中交互
- 命令边界检测复杂：需要区分命令输出和 shell 提示符
- 需要额外机制获取退出码：命令完成后 shell 仍在运行

## 架构对比

### Codex CLI 架构

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/shell-tool/codex-timeline.svg" style="width: 100%"></object><figcaption>Codex CLI 进程时间线</figcaption></figure>

```rust
// Codex 执行链
process_exec_tool_call()
  → execute_exec_env()
      → consume_truncated_output()
          → tokio::select! {
              child.wait()           // 进程正常退出
              expiration.wait()      // 超时
              ctrl_c()               // 用户中断
          }
```

Codex CLI 使用 Rust 的 portable_pty 库为每条命令创建独立的 PTY 子进程。 通过 `tokio::select!` 同时监听进程退出、超时、用户中断三种事件，进程退出即表示命令完成，退出码直接从 `ExitStatus` 获取。

### Gemini CLI 架构

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/shell-tool/gemini-timeline.svg" style="width: 100%"></object><figcaption>Gemini CLI 进程时间线</figcaption></figure>

```typescript
// Gemini 执行链
ShellTool
  → ShellToolInvocation
      → ShellExecutionService.execute()
          → node-pty.spawn()
          → ptyProcess.onExit()
```

Gemini CLI 使用 Node.js 的 node-pty 库，采用事件驱动模式。 通过 `onExit()` 回调检测命令完成，退出码从回调参数获取。 使用 xterm headless 解析 ANSI 输出。

### SWE-ReX 架构

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/shell-tool/swerex-timeline.svg" style="width: 100%"></object><figcaption>SWE-ReX 进程时间线</figcaption></figure>

```python
# SWE-ReX 执行链
BashAction
  → LocalRuntime.run_in_session()
      → BashSession.run()
          → shell.sendline(command)        # 通过 PTY 发送命令
          → shell.expect(PS1, timeout)     # 等待提示符出现
          → shell.sendline("echo $?")      # 获取退出码
```

SWE-ReX 使用 Python 的 pexpect 库与持久的 Bash 进程交互。 发送命令后，通过 `expect()` 阻塞等待自定义 PS1 提示符出现来判断命令完成，然后额外执行 `echo $?` 获取退出码。

### OpenHands 架构

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/shell-tool/openhands-timeline.svg" style="width: 100%"></object><figcaption>OpenHands 进程时间线</figcaption></figure>

```python
# OpenHands 执行链
Agent._execute_action_event()
  → TerminalExecutor.__call__(action)
      → session.execute(action)
          → terminal.send_keys(command)    # 发送命令
          → terminal.read_screen()          # 轮询读取屏幕
          → CmdOutputMetadata.from_ps1_match()  # 解析 PS1 元数据
```

OpenHands 使用 libtmux 管理持久的 tmux 会话。 发送命令后，每 100ms 轮询调用 `capture_pane()` 读取终端屏幕，通过正则匹配 PS1 中嵌入的 JSON 元数据（包含退出码、工作目录等）来判断命令完成。

### Nexus 架构

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/shell-tool/nexus-timeline.svg" style="width: 100%"></object><figcaption>Nexus 进程时间线</figcaption></figure>

```python
# Nexus 执行链
TerminalSession.run_command()
  → start_command()
      → tmux send-keys command Enter    # 发送命令
  → get_command_info()                   # 轮询状态文件
      → 读取 status.json                 # nexus_bash 内部更新
```

Nexus 使用定制的 GNU Bash（nexus_bash），在源码中插入 preexec/precmd 钩子。 命令执行前后，钩子自动更新状态文件，Python 层通过读取文件获取命令状态和退出码，无需依赖输出匹配。

### 两种模式对比

<div class="center-table" markdown>

| 特性 | 子进程模式 | 持久会话模式 |
|------|------------|--------------|
| 代表框架 | Codex CLI、Gemini CLI | SWE-ReX、OpenHands、Nexus |
| 命令边界检测 | 进程退出事件（简单可靠） | PS1 匹配 / Bash Hook（较复杂） |
| 退出码获取 | 进程退出状态（直接） | echo $? / PS1 嵌入 / 状态文件 |
| 状态保持 | 无（每次重置） | 有（环境变量、工作目录等） |
| 后台进程 | 不支持 | 支持 |
| 实现复杂度 | 低 | 中到高 |

</div>

## 持久会话模式的命令边界检测

持久会话模式的核心挑战是：shell 进程持续运行，如何判断一条命令何时结束？

### PS1 提示符匹配（SWE-ReX、OpenHands）

通过检测 shell 提示符的出现来判断命令完成：

**SWE-ReX** 使用 pexpect.expect() 阻塞等待：

```python
# SWE-ReX 的阻塞等待
self.shell.sendline(command)
self.shell.expect(self._ps1, timeout=action.timeout)  # 阻塞直到匹配
# 额外命令获取退出码
self.shell.sendline("echo $?")
```

**OpenHands** 使用轮询：

```python
# OpenHands 的轮询逻辑
while True:
    terminal_content = self.terminal.read_screen()
    ps1_matches = CmdOutputMetadata.matches_ps1_metadata(terminal_content)
    if time_since_last_change > no_change_timeout_seconds:
        return TerminalObservation(exit_code=-1)
    await asyncio.sleep(0.1)
```

**局限性** ：命令输出中如果包含与 PS1 相似的字符串，可能导致误判。

### Bash Hook （Nexus）

Nexus 在 Bash 源码的 **eval.c** 和 **shell.c** 中插入钩子：

```c
// shell.c - 主循环中的钩子调用
reader_loop() {
    while (1) {
        command = read_command();
        capture_preexec_hook();      // 执行前钩子
        execute_command(command);
        capture_precmd_hook(last_command_exit_value);  // 执行后钩子
    }
}
```

Python 层通过读取状态文件获取命令完成信息：

```python
# Nexus 的状态检查
async def run_command(self, command: str, timeout: float | None = None) -> CommandInfo:
    command_id = await self.start_command(command)
    while True:
        cmd_info = await self.get_command_info(command_id)
        if cmd_info and cmd_info.end_time:  # 钩子已更新状态文件
            return cmd_info
        await asyncio.sleep(1)  # 低频检查
```

**特点** ：钩子在 Bash 内部触发，不依赖输出内容匹配，但需要定制 Bash。

## Nexus 实现细节

### 文件接口

Nexus 的状态和输出通过文件系统暴露：

```
/tmp/nexus/tools/terminal/<session_id>/
├── status.json          # 实时状态
├── cmd_1.json           # 命令 1 元数据
├── cmd_content_1        # 命令 1 内容
├── output_1             # 命令 1 输出
├── cmd_2.json           # 命令 2 元数据
└── ...
```

**status.json 格式** ：

```json
{
    "status": "command_running",
    "timestamp": 1737456789.123456,
    "command_id": 1,
    "detail_file": "/tmp/nexus/tools/terminal/session/cmd_1.json"
}
```

**cmd_N.json 格式** ：

```json
{
    "command_id": 1,
    "start_time": 1737456789.123456,
    "end_time": 1737456790.234567,
    "exit_code": 0
}
```

### 与 tmux 的集成

Nexus 的 `terminal.py` 充分利用 tmux 的能力：

```python
class TerminalSession:
    async def start_command(self, command: str) -> int:
        # 检查是否有命令正在运行
        status = await self._read_status_file()
        if status.get('status') == 'command_running':
            raise RuntimeError('A command is already running')

        # 通过 tmux send-keys 发送命令
        await asyncio.create_subprocess_exec(
            self.tmux_cmd, 'send-keys', '-t', self.session_id, command, 'Enter'
        )

        # 等待 nexus_bash 更新状态文件
        return new_command_id
```

tmux 提供会话持久化、窗口管理和信号传递等能力。

### 输出分离（可选）

对于需要区分 stdout 和 stderr 的场景，Nexus 通过管道实现精确分离：

```c
// 管道捕获架构
static void setup_output_capture(...) {
    original_stdout = dup(STDOUT_FILENO);
    original_stderr = dup(STDERR_FILENO);

    pipe(stdout_pipe);
    pipe(stderr_pipe);

    capture_pid = fork();
    if (capture_pid == 0) {
        // 子进程：异步捕获输出到文件
        capture_process(stdout_pipe[0], stderr_pipe[0], ...);
    } else {
        // 父进程：重定向输出到管道
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
    }
}
```

输出文件结构：

```
├── stdout_1             # 命令 1 标准输出
├── stderr_1             # 命令 1 标准错误
├── output_1             # 命令 1 混合输出（保持时序）
```

虽然大部分场景混合输出（output_N）已经足够，但在某些场景下分离的输出仍然有价值，例如编译错误分析、日志级别区分等。

### 捕获流程详解

1. **会话初始化** ：`create_terminal_session` 创建 tmux 会话并运行 `nexus_bash`
2. **启用捕获** ：`nexus_capture <session_id>` 命令启用内置捕获钩子
3. **命令发送** ：通过 `tmux send-keys` 发送命令到会话
4. **preexec 钩子触发** ：
   - 从 history 获取命令内容
   - 分配命令 ID
   - 创建管道和捕获子进程
   - 更新 status.json 为 "command_running"
5. **命令执行** ：输出通过管道同时写入终端和文件
6. **precmd 钩子触发** ：
   - 恢复原始文件描述符
   - 等待捕获进程结束
   - 记录退出码和结束时间
   - 更新 status.json 为 "command_finished"

### 关键代码位置

- 捕获钩子：`deployment/runtime/bash/src/capture_hooks.c`
- Bash 修改：`deployment/runtime/bash/src/eval.c`, `shell.c`
- Python 接口：`nexus/runtime/executors/terminal.py`

## 适用场景

**子进程模式（Codex CLI、Gemini CLI）** ：从 Coding CLI 产品角度，子进程模式有明显优势：

- 实现简单且稳定，进程退出即命令完成，无需复杂的边界检测
- 无需修改用户的 shell 环境（如 PS1），避免造成用户困惑
- 无需 ship 定制的 bash 或依赖特定 shell 版本
- 适合面向终端用户的交互式产品

**持久会话 + PS1 匹配（SWE-ReX、OpenHands）** ：平衡了实现复杂度和功能，适合大多数 Agent 场景，尤其是需要在命令间保持状态的场景。

**持久会话 + Bash Hook（Nexus）** ：Nexus 选择这种方案是为了构建尽可能可靠且全能的 shell tool：

- 边界检测在 Bash 内部完成，不受命令输出内容干扰，可靠性最高
- 支持完整的会话状态保持和后台进程
- 适合需要支持各类 Agent 运行的通用框架，以及对数据可靠性要求高的场景（如训练数据收集）

## 总结

五种方案代表了 shell 工具实现的不同思路：

**子进程模式** ：

- **Codex CLI** ：Rust 实现，使用 portable_pty，通过 tokio::select! 处理退出/超时/中断
- **Gemini CLI** ：TypeScript 实现，使用 node-pty，通过 onExit 回调检测完成

**持久会话模式** ：

- **SWE-ReX** ：Python 实现，使用 pexpect，通过 expect() 阻塞等待 PS1
- **OpenHands** ：Python 实现，使用 libtmux，通过轮询匹配 PS1 中的 JSON 元数据
- **Nexus** ：定制 Bash + Python，通过 Bash Hook 更新状态文件

选择哪种方案取决于具体需求：是否需要会话状态保持、对边界检测可靠性的要求、以及可接受的实现复杂度。
