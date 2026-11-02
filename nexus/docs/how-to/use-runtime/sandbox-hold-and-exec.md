# Sandbox Hold & Exec / Upload / Download

在 task 执行后保持沙箱存活，通过 CLI 执行命令、上传文件、下载文件。适用于排查 task 失败原因、验证沙箱内环境状态、注入调试脚本并取回结果。

## 概述

```
nexus run-single config.yaml --hold 3600              # task 结束后沙箱保活 3600s
nexus sandbox exec <url> "cmd"                        # 在保活的沙箱中执行命令
nexus sandbox upload <url> <local_path> <remote_path> # 上传本地文件到沙箱
nexus sandbox download <url> <remote_path> <local>    # 从沙箱下载文件（local 可为 -）
kill <pid> / Ctrl+C                                   # 释放沙箱
```

## --hold 参数

在 `nexus run-single` 中添加 `--hold <seconds>`：

```bash
# task 结束后保活 1 小时
nexus run-single config.yaml --hold 3600
```

> `--hold` 仅支持 `nexus run-single`（单 case 模式），不支持 `nexus run-batch`。

### 行为说明

1. Task 正常执行（包括 agent 内部的 release/release_all 调用全部被拦截）
2. Task 结束后（无论成功/失败），打印 hold banner：

```
══════════════════════════════════════════════════════
  HOLD MODE — sandbox(es) kept alive for diagnosis
══════════════════════════════════════════════════════
  Runtime: infer
    base_url:  http://10.129.3.45:32000
  Runtime: eval
    base_url:  http://10.129.3.46:32000

  PID: 12345
  Timeout: 3600s (auto-release)
  Exit: kill 12345  OR  Ctrl+C

  Diagnose:
    nexus sandbox exec http://10.129.3.45:32000 "ls /workspace"
══════════════════════════════════════════════════════
```

3. 等待退出信号（SIGINT/SIGTERM）或超时自动释放

### 释放机制

- **主动释放**：对 nexus 进程发 `kill <pid>` 或 `Ctrl+C`
- **超时释放**：达到 `--hold` 指定秒数后自动释放
- 释放后取消 suppress，正常执行 `release_all()` 清理所有沙箱

## nexus sandbox exec

在保活的沙箱中执行任意 shell 命令：

```bash
nexus sandbox exec <base_url> <command> [--cwd DIR] [--timeout SECONDS]
```

### 参数

| 参数 | 说明 |
|------|------|
| `base_url` | 沙箱 runtime 地址（从 hold banner 复制） |
| `command` | 要执行的 shell 命令 |
| `--cwd` | 命令的工作目录（默认沙箱根目录） |
| `--timeout` | 命令超时秒数（默认 180s） |

### 示例

```bash
# 查看工作目录内容
nexus sandbox exec http://10.129.3.45:32000 "ls -la /workspace"

# 检查 Python 包版本
nexus sandbox exec http://10.129.3.45:32000 "pip show requests"

# 检查网络连通性
nexus sandbox exec http://10.129.3.45:32000 "curl -s http://example.com -o /dev/null -w '%{http_code}'"
```

## nexus sandbox upload

将本地文件上传到保活的沙箱中：

```bash
nexus sandbox upload <base_url> <local_path> <remote_path> [--timeout SECONDS]
```

### 参数

| 参数 | 说明 |
|------|------|
| `base_url` | 沙箱 runtime 地址（从 hold banner 复制） |
| `local_path` | 本地文件路径 |
| `remote_path` | 沙箱内的目标文件路径（必须是绝对路径） |
| `--timeout` | 上传超时秒数（默认 180s） |

### 支持的文件类型

- **文本文件**：`.py`、`.sh`、`.json`、`.yaml`、`.txt` 等，以 UTF-8 编码传输
- **二进制文件**：`.png`、`.zip`、`.tar.gz`、`.bin` 等，自动检测并以 base64 编码传输
- 大于 10MB 的文件会打印 warning（不硬拦，但传输较慢）
- **不支持**：目录上传（后续版本支持），每次只能传一个文件

### 路径说明

- `local_path`：本地主机上的文件路径，相对路径或绝对路径均可
- `remote_path`：沙箱内的绝对路径。如果目标路径的父目录不存在，沙箱侧会自动创建

### 示例

```bash
# 上传调试脚本
nexus sandbox upload http://10.129.3.45:32000 ./diagnose.sh /workspace/diagnose.sh

# 上传 Python 脚本
nexus sandbox upload http://10.129.3.45:32000 ./fix.py /workspace/fix.py

# 上传二进制文件（自动 base64）
nexus sandbox upload http://10.129.3.45:32000 ./model.bin /workspace/model.bin
```

## nexus sandbox download

从保活的沙箱中下载文件到本地：

```bash
nexus sandbox download <base_url> <remote_path> <local_path> [--timeout SECONDS]
```

### 参数

| 参数 | 说明 |
|------|------|
| `base_url` | 沙箱 runtime 地址（从 hold banner 复制） |
| `remote_path` | 沙箱内的源文件路径（绝对路径） |
| `local_path` | 本地写入路径，或 `-` 输出到 stdout |
| `--timeout` | 下载超时秒数（默认 180s） |

### 支持的文件类型

- 文本和二进制文件均支持，统一以 base64 从沙箱传回后在本地 decode 写入
- 输出到 stdout（`-`）时：文本内容直接打印，二进制内容写入 `sys.stdout.buffer`
- **不支持**：目录下载、通配符匹配

### 路径说明

- `remote_path`：沙箱内的绝对路径，文件必须存在
- `local_path`：本地写入路径。如果父目录不存在会自动创建。传 `-` 则输出到 stdout

### 示例

```bash
# 下载测试报告
nexus sandbox download http://10.129.3.45:32000 /workspace/report.txt ./report.txt

# 下载日志到 stdout（直接查看）
nexus sandbox download http://10.129.3.45:32000 /workspace/agent.log -

# 下载二进制产物
nexus sandbox download http://10.129.3.45:32000 /workspace/screenshot.png ./screenshot.png

# 下载到指定目录（自动创建不存在的父目录）
nexus sandbox download http://10.129.3.45:32000 /workspace/out/data.json ./results/data.json
```

## 典型工作流

### 单 case 诊断（exec）

```bash
# 终端 1：跑 task 并 hold
nexus run-single config.yaml --hold 3600

# 终端 2：等 hold banner 出现后，执行诊断
nexus sandbox exec http://10.129.3.45:32000 "git diff"
nexus sandbox exec http://10.129.3.45:32000 "pip list | grep django"
nexus sandbox exec http://10.129.3.45:32000 "cat /workspace/test_output.log"

# 诊断完毕，回到终端 1 按 Ctrl+C 释放
```

### Agent 调试（upload + exec + download）

```bash
# 终端 1：跑 task 并 hold
nexus run-single config.yaml --hold 3600

# 终端 2：上传调试脚本 → 执行 → 取回结果
nexus sandbox upload http://10.129.3.45:32000 ./diagnose.sh /workspace/diagnose.sh
nexus sandbox exec http://10.129.3.45:32000 "bash /workspace/diagnose.sh"
nexus sandbox download http://10.129.3.45:32000 /workspace/report.txt ./report.txt

# 诊断完毕，回到终端 1 按 Ctrl+C 释放
```

## 注意事项

- `--hold` 的超时值是兜底防泄漏（agent 忘了 kill），正常使用通过信号主动退出
- Hold 模式下所有 runtime release 调用被 suppress，沙箱不会被提前回收
- 如果沙箱被平台侧强制回收（超出平台 TTL），exec/upload/download 会连接失败
- 多 runtime 场景下所有 runtime 都会被保留（banner 中列出所有可用地址）
- upload/download 当前仅支持单文件操作，目录传输需先打包（如 `tar czf`）
