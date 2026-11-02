# Sandbox Benchmark 压测工具

本目录包含用于沙盒平台性能压测的脚本和工具，用于测试 Nexus 沙盒运行时（Runtime）的 Pod 创建、启动和执行性能。

## 基本原理
基于swe-bench-verified数据集(理论上任意数据集都可以)中的几个instance来构造压测数据.

环境变量`export SWE_BENCH_TEST_SLEEP=360`可使得在eval阶段, test_cmd执行前sleep 360s, 用来模拟多个沙盒同时在线.

## 目录结构

```
tests/stress/swe_sandbox/
├── README.md                 # 本文档
├── run_benchmark.py          # 核心压测脚本（CLI 工具 & Generator）
├── benchmark_base.sh         # 基础启动脚本（通用配置）
├── benchmark_test.sh         # 测试环境压测脚本
├── benchmark_prod-x3.sh      # 生产环境 x3 集群压测脚本
└── benchmark_prod-x8.sh      # 生产环境 x8 集群压测脚本
```

## 前置条件

### 1. 必需文件

| 文件 | 路径 | 说明 |
|------|------|------|
| SWE-bench 数据集 | `./workspace/swe-bench.jsonl` | JSONL 格式的 SWE-bench 测试数据集 |
| Python 虚拟环境 | `.venv/` | 包含 nexus 及其依赖的虚拟环境 |

### 2. 环境依赖

- **Python 3.10+**
- **uv** - Python 包管理器
- **nexus** - 已安装在虚拟环境中

### 3. 网络访问

确保可以访问对应环境的沙盒服务：

| 环境 | URL |
|------|-----|
| test | `http://hyrl-sandbox.test.woa.com:8052` |
| prod-x3 | `http://hyrl-sandbox.prod.woa.com:8052` |
| prod-x8 | `http://hunyuan.sandbox.production.polaris:8052` |

### 4. 环境变量（可选）

```bash
# 可选：指定 Runtime 镜像标签
export SWE_RUNTIME_TAG=latest

# 可选：指定数据集路径（默认: ./workspace/swe-bench.jsonl）
export SWE_DATASET_PATH=./workspace/swe-bench.jsonl
```

## 快速开始

直接基于 `benchmark_test.sh` 修改运行就行, 有更多需求再看下文.

### 1. 准备数据集

确保 SWE-bench 数据集文件存在：

```bash
ls -la ./workspace/swe-bench.jsonl
```

### 2. 运行压测

**测试环境（500 sandboxes）：**
```bash
bash tests/stress/swe_sandbox/benchmark_test.sh
```

**生产环境 x3（5000 sandboxes）：**
```bash
bash tests/stress/swe_sandbox/benchmark_prod-x3.sh
```

**生产环境 x8（5000 sandboxes）：**
```bash
bash tests/stress/swe_sandbox/benchmark_prod-x8.sh
```

## 详细使用方法

### 方式一：使用 Shell 脚本（推荐）

Shell 脚本会自动：
- 激活虚拟环境
- 创建工作目录和日志文件
- 设置环境变量
- 执行压测

```bash
# 测试环境
bash tests/stress/swe_sandbox/benchmark_test.sh

# 生产环境
bash tests/stress/swe_sandbox/benchmark_prod-x3.sh
bash tests/stress/swe_sandbox/benchmark_prod-x8.sh
```

### 方式二：直接使用 Python 脚本

`run_benchmark.py` 支持丰富的命令行参数：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 基本用法
python tests/stress/swe_sandbox/run_benchmark.py \
  --instance-ids "sympy__sympy-13031,django__django-12125" \
  --env test \
  --total-count-per-round 500

# 多轮压测
python tests/stress/swe_sandbox/run_benchmark.py \
  --instance-ids "sympy__sympy-13031" \
  --env prod-x3 \
  --total-count-per-round 5000 \
  --rounds 5 \
  --round-duration 1800

# 多进程模式(推荐)
python tests/stress/swe_sandbox/run_benchmark.py \
  --instance-ids "sympy__sympy-13031,django__django-12125" \
  --env prod-x3 \
  --total-count-per-round 5000 \
  --multiprocess \
  --tasks-per-process 500
```

### 方式三：作为 Generator 使用

`run_benchmark.py` 也可以作为 `nexus run-batch` 的 Generator：

```bash
STRESS_INSTANCE_IDS="sympy__sympy-13031,django__django-12125" \
STRESS_REPEAT_COUNT=2000 \
SWE_RUNTIME_BASE_URL='http://hyrl-sandbox.test.woa.com:8052' \
uv run nexus run-batch \
    -g ./tests/stress/swe_sandbox/run_benchmark.py \
    -d ./workspace/benchmark.db \
    --max-concurrent 10000 \
    --no-resume
```

## 命令行参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `--instance-ids` | 逗号分隔的 instance ID 列表 |

### 环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--env` | `prod-x3` | 环境选择：`test`, `prod-x3`, `prod-x8` |
| `--dataset-path` | `./workspace/swe-bench.jsonl` | SWE-bench 数据集路径 |
| `--db-path` | 自动生成 | 数据库文件路径 |
| `--log-path` | 自动生成 | 日志文件路径 |

### 压测配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-count-per-round` | `10000` | 每轮目标 sandbox 总数 |
| `--rounds` | `1` | 压测轮数（0 表示无限） |
| `--round-duration` | `1800` | 每轮最大持续时间（秒） |
| `--round-cooldown` | `5` | 轮次间冷却时间（秒） |
| `--separate-db` | `false` | 是否为每轮使用独立数据库 |

### 多进程模式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--multiprocess` | `false` | 启用多进程模式 |
| `--process-count` | `0` | 进程数（0 表示自动计算） |
| `--tasks-per-process` | `500` | 每进程任务数 |

### 报告生成

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--report-from` | - | 从已有目录生成报告 |

### 其他选项

| 参数 | 说明 |
|------|------|
| `-y, --yes` | 跳过确认提示 |
| `--dry-run` | 仅打印命令，不执行 |

## 环境变量（Generator 模式）

当作为 Generator 使用时，支持以下环境变量：

| 环境变量 | 必需 | 默认值 | 说明 |
|----------|------|--------|------|
| `STRESS_INSTANCE_IDS` | ✅ | - | 逗号分隔的 instance ID 列表 |
| `STRESS_REPEAT_COUNT` | ❌ | `2000` | 每个 instance 重复次数 |
| `STRESS_ROUND_PREFIX` | ❌ | - | run_id 前缀（如 `round001`） |
| `SWE_DATASET_PATH` | ❌ | `./workspace/swe-bench.jsonl` | 数据集路径 |
| `SWE_RUN_ORACLE` | ❌ | `true` | 使用 Oracle 模式（无 Agent） |
| `SWE_RUNTIME_BASE_URL` | ❌ | `http://hyrl-sandbox.prod.woa.com:8052` | Runtime 服务 URL |
| `SWE_RUNTIME_PROVIDER_NAME` | ❌ | `gongfeng` | Runtime Provider 实现类名称，可选值：`gongfeng`（默认）、`docker` |
| `SWE_RUNTIME_PROVIDER_CLUSTER` | ❌ | （自动选择） | 资源调度集群（仅 gongfeng provider），可选值：`gongfeng`、`suanli` |
| `SWE_RUNTIME_TAG` | ❌ | `latest` | Runtime 镜像标签 |
| `SWE_RUNTIME_TIMEOUT` | ❌ | `1800` | Runtime 超时时间（秒） |
| `SWE_BENCH_TEST_SLEEP` | ❌ | `0` | 测试命令执行前的睡眠时间（秒），用于模拟沙盒同时在线 |

## 输出说明

### 工作目录结构

每次压测会在 `./workspace/` 下创建一个带时间戳的目录：

```
workspace/
└── benchmark_test_20260203_101243/
    ├── benchmark_20260203_101243.log    # 主日志文件
    ├── benchmark.db                      # SQLite 数据库
    ├── benchmark_proc000.log             # 进程 0 日志（多进程模式）
    ├── benchmark_proc001.log             # 进程 1 日志（多进程模式）
    ├── ...
    ├── report_20260203_103000.json       # JSON 格式报告
    └── report_20260203_103000.txt        # 文本格式报告
```

### 统计报告内容

报告包含以下统计信息：

- **Summary**：总任务数、成功/失败/超时数、成功率
- **Timing**：平均/中位数/最小/最大启动时间、P50/P90/P95/P99
- **Time Distribution**：时间分布直方图
- **Instance Performance**：按 instance 分组的性能数据

### 从已有数据生成报告

```bash
python tests/stress/swe_sandbox/run_benchmark.py \
  --report-from ./workspace/benchmark_test_20260203_101243
```

## 示例

### 1. 快速测试（测试环境，500 sandboxes）

```bash
bash tests/stress/swe_sandbox/benchmark_test.sh
```

### 2. 生产环境压测（5000 sandboxes，多进程）

```bash
python tests/stress/swe_sandbox/run_benchmark.py \
  --instance-ids "sympy__sympy-13031,sphinx-doc__sphinx-8265,django__django-12125" \
  --env prod-x3 \
  --total-count-per-round 5000 \
  --multiprocess \
  --tasks-per-process 500 \
  --yes
```

### 3. 多轮持续压测（无限轮次）

```bash
python tests/stress/swe_sandbox/run_benchmark.py \
  --instance-ids "sympy__sympy-13031" \
  --env prod-x3 \
  --total-count-per-round 5000 \
  --rounds 0 \
  --round-duration 1800 \
  --round-cooldown 60 \
  --multiprocess
```

### 4. 查看已有压测结果

```bash
python tests/stress/swe_sandbox/run_benchmark.py \
  --report-from ./workspace/benchmark_prod-x3_20260203_120000
```

## 注意事项

1. **资源消耗**：大规模压测会消耗大量 CPU、内存和网络资源，请确保有足够的资源。

2. **网络稳定性**：确保与沙盒服务的网络连接稳定，网络问题可能导致大量超时。

3. **数据集完整性**：确保指定的 instance ID 在数据集文件中存在。

4. **中断处理**：使用 `Ctrl+C` 可以优雅地停止压测，当前轮次完成后退出。

5. **日志查看**：实时日志会同时输出到控制台和日志文件，方便监控和回溯。
