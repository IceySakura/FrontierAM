# 本地批处理：`nexus run batch`

`nexus run batch` 将任务配置写入 SQLite，再按指定并发执行，支持结果导出、失败重试和
从数据库恢复。`nexus run-batch` 是相同功能的别名；下文统一使用 `nexus run batch`。

任务可以来自 Python generator、Dataset、普通 RunInput 文件或 Compose 文件。配置
片段和合并规则见 [Run Part 指南](run-parts.md)。本文介绍本地批处理；云端调度与资源
申请使用 [Cloud 批处理](../cloud/submit-batch.md)，两者的提交参数不能直接互换。

## 快速开始

准备好 generator 后，用新数据库先运行少量任务：

```bash
nexus run batch --generator ./generator.py \
  --limit 2 --max-concurrent 2 \
  --database ./outputs/smoke.db \
  --export-format jsonl --export-path ./outputs/smoke.jsonl
```

结果保存在数据库和 JSONL 中。任务是否达到业务目标，应查看 `task_output` 和评估分数，
不能只依据命令结束或进程退出码。

### 一个完整的 generator

将下面内容保存为 `generator.py`。示例使用 Echo 的 oracle 模式，需要可用的 Docker，
由 Runtime 拉起测试环境，不需要模型凭据。

```python
from nexus import BatchItem, RunInput


def generate_configs():
    for index, message in enumerate(['hello', 'world', 'nexus']):
        yield BatchItem(
            config=RunInput(
                task_name='echo',
                task_config={'message': message, 'run_oracle': True},
                runtime_provider_name='docker',
            ),
            run_id=f'echo-{index}',
            metadata={'index': index, 'suite': 'smoke'},
        )


def get_batch_metadata():
    return {'experiment': 'echo-smoke'}
```

`generate_configs()` 必须返回 BatchItem 的迭代器或列表；建议使用 `yield`，避免先构造
整个任务列表。函数不接收 CLI 参数。`get_batch_metadata()` 可选，返回保存到 `batch_info`
的批次元信息。实际批次任务数以写入数据库的条目为准。

| BatchItem 字段 | 含义 |
|---|---|
| `config` | RunInput，包含 task、agent、runtime、模型、processor 等配置 |
| `run_id` | 可选；未指定时按生成顺序使用 `00000`、`00001` 等 ID |
| `metadata` | 可选；标签、分类等任务元信息 |

自定义 run_id 会规范化为字母、数字、下划线和连字符；生成器应保证规范化后仍唯一。
任务实现必须可导入：内置模块按名称发现；自定义 task 应通过 `RunInput.imports` 引用
可导入的模块，以便在独立 worker 或新进程恢复时加载。

## 选择任务来源

### Python generator

```bash
nexus run batch --generator ./generator.py --database ./outputs/generated.db
```

适合复杂筛选、组合参数、按业务规则动态构造不同 RunInput 的场景。

### Dataset

```bash
nexus run batch --data ./dataset --split train --limit 10 \
  --use ./common.yaml --database ./outputs/dataset.db
```

Dataset item 提供配置与文件引用，Run Part 补齐公共配置。`--split` 只用于 Dataset。
一个组合只支持一个数据源。数据源中没有匹配样本会报告错误；显式 `--limit 0` 用于
创建空批次。Dataset 格式见 [Dataset CLI 指南](use-datasets/dataset-cli.md)，配置合并见 [Run Part 指南](run-parts.md)。

### RunInput 或 Compose 文件

```bash
nexus run batch ./batch.yaml --database ./outputs/from-file.db
```

普通 RunInput 文件生成一条 BatchItem；Compose 文件可以展开 Dataset。需要在完整
RunInput 文件上叠加命令行配置时，使用 `--use`：

```bash
nexus run batch --use ./run.yaml --set runner.timeout=600 \
  --database ./outputs/overridden.db
```

位置参数配置文件不能和 `--generator` 或命令行 Compose 参数同时使用。
完整 Compose 文件直接作为位置参数，不作为 `--use` 的配置片段。

## 给 generator 叠加配置

Generator 每生成一条 BatchItem，就以它的 RunInput 为底值，按命令行出现顺序应用
`--use`、`--set`、`--set-json`，然后校验并写入数据库。

先保存一个部分配置文件 `common.yaml`：

```yaml
body:
  runner:
    timeout: 300
  runtime_configs:
    main:
      worker_venv: true
```

```bash
nexus run batch --generator ./generator.py \
  --use ./common.yaml \
  --set runner.timeout=600 \
  --set-json 'extra={"source":"manual-smoke"}' \
  --limit 2 --max-concurrent 2 --database ./outputs/with-parts.db
```

这里的最终 `runner.timeout` 是 600。`--use` 可以引用已有的命名 Run Part，也可以引用
符合 [Run Part 文件格式](run-parts.md#配置文件的两种形状) 的 YAML/JSON。

| 覆盖方式 | 规则 |
|---|---|
| `--set path=value` | 点路径；值按 YAML 解析 |
| `--set-json 'path={...}'` | 点路径；值按 JSON 解析 |
| 多个覆盖参数 | 按它们的相对顺序应用，后面的同名值优先 |
| 字典 | 默认递归合并 |
| 列表 | 默认整体替换；`path+=value` 追加 |
| `null` | 删除配置键；最终仍需通过 RunInput 校验 |

无论 `--generator` 写在覆盖参数前还是后，覆盖都应用在生成结果上。它们不传给
`generate_configs()`，也不修改 BatchItem 的 run_id、metadata 或批次 metadata。
被修改的顶层配置段在生成阶段按 Compose 规则解析环境变量并校验类型，例如
`--set 'runner.timeout=${oc.env:TASK_TIMEOUT}'`。未修改配置段中的引用保持原样，
由正常执行阶段解析；`extra` 和受保护的自由文本字段保持字面值。

Generator 不能与 `--data` 或 `--use data:...` 混用。命名 Run Part 的注册、参数及列表
合并策略见 [Run Part 指南](run-parts.md)。

## 限制任务数量

`--limit N` 选择 generator 或 Dataset 的前 N 条任务，用于冒烟或小规模运行：

| 值 | 首次生成时的行为 |
|---|---|
| 未指定 | 消费全部条目 |
| 正整数 N | 最多取前 N 条；不足 N 条时使用全部已有条目 |
| 0 | 空批次；generator 不调用 `generate_configs()`，但仍可导入模块、读取批次 metadata |
| 负数 | 报错，不开始生成 |

Generator 先限量，再对选中条目应用覆盖。采用 `yield` 的生成器不会多取第 N+1 条，
截断后会关闭其迭代器；如果函数先构造完整 list 再返回，limit 只能限制取用数量，
无法省掉构造 list 的工作。

limit 不改变单任务超时、并发数或重试次数；也不会裁剪已经生成并锁定的数据库。
普通单条 RunInput 文件不是样本集合，limit 不用于裁剪这种输入。

## 并发与多进程

默认在一个进程内按 `--max-concurrent` 并发执行：

```bash
nexus run batch --generator ./generator.py \
  --max-concurrent 8 --database ./outputs/batch.db
```

使用多个本地 worker 时，必须提供文件数据库：

```bash
nexus run batch --generator ./generator.py \
  --limit 100 --max-concurrent 8 --worker-processes 2 \
  --database ./outputs/parallel.db
```

`--max-concurrent` 是整个批次的总并发预算，会分给 worker，并不是每个 worker 的并发。
主数据库记录分片信息，分片数据库保存在 `parallel.db.shards/`。限量和配置覆盖在分片前
完成，worker 使用已保存的配置；实际 worker 数不超过任务数和总并发数。

有共享资源冲突时，在生成器中填写 `metadata.conflict_groups`，并开启
`--enable-conflict-groups`。具有相同组名的任务按组串行；该选项仅支持单 worker。

## 保存配置与恢复执行

批处理分为两个阶段：

1. **生成配置**：选择条目、应用覆盖，将最终 RunInput、run_id 和 metadata 写入 SQLite。
2. **执行任务**：从数据库读取固定配置，更新状态、输出、日志和轨迹。

生成完成后配置锁定。恢复运行时不调用 generator 重新生成条目，也不应用新的
`--limit`、`--use`、`--set`、`--set-json`；提供这些 generator 选项时会提示忽略。
此时不会读取覆盖配置文件或调用 Run Part 注册项，原覆盖文件删除或修改也不影响恢复。
来源互斥及 `--limit` 非负等基本参数校验仍然生效。
`--no-resume` 只决定哪些已有任务重跑，不会解除配置锁定。要应用新的生成参数，使用
新的数据库路径即可。

```bash
# 跳过成功任务，继续执行其余已有任务
nexus run batch --database ./outputs/batch.db --max-concurrent 8

# 只重新执行 failed 状态的任务
nexus run batch --database ./outputs/batch.db --retry-only

# 重跑所有已有任务
nexus run batch --database ./outputs/batch.db --no-resume

# 多进程批次继续使用主数据库，并保留多进程模式
nexus run batch --database ./outputs/parallel.db --worker-processes 2 --max-concurrent 8
```

首次生成尚未完成时，需要继续提供来源；已写入的同 ID 配置不会被覆盖，恢复时应保持
原来源和生成参数一致。Data/Compose 参数在 CLI 阶段可能先访问数据源；已有完整数据库
的恢复命令通常只需指定数据库。分片 worker 为加载注册项仍可能导入 generator 模块。

## 超时、失败与重试

```bash
nexus run batch --generator ./generator.py --database ./outputs/retries.db \
  --timeout 300 --retry-failed 2 --max-concurrent 4
```

`--timeout` 是每条任务的 Batch 执行超时，单位秒；`--retry-failed 2` 表示初次尝试之外，
最多再尝试两次。默认重试从已保存的原始 RunInput 开始。

支持 workflow 恢复的任务可加 `--resume-workflow-on-retry`，让当前执行中的立即重试
保留已提交节点；这不表示任意 task 都具备断点恢复能力。`--fail-fast` 用于失败时尽早
停止批次。状态与分数仍需分别判断：运行成功不必然代表业务评估通过。

## 结果导出

执行时自动导出：

```bash
nexus run batch --generator ./generator.py --database ./outputs/batch.db \
  --export-format jsonl --export-path ./outputs/results.jsonl
```

单进程执行可在任务完成后增量写出；多进程在 worker 完成后汇总导出。增量导出对应
本次实际执行的任务，恢复时跳过的条目不应依赖它来补齐；如需完整快照，从数据库导出：

```bash
nexus export ./outputs/batch.db --format jsonl --output ./outputs/all.jsonl
nexus export ./outputs/batch.db --format csv --output ./outputs/all.csv
nexus export ./outputs/batch.db --format full --output ./outputs/full
```

| 格式 | 内容 |
|---|---|
| `jsonl` | 每行一个 run，包含状态、错误、task/agent/processor 输出及 metadata；非空 `config.extra` 也会导出 |
| `csv` | run_id、状态、时间、重试次数和错误等摘要 |
| `full` | 配置 YAML，以及每个 run 的 metadata、状态、业务输出和 Processor 文件 |

后处理 `nexus export` 未指定输出路径时，默认生成数据库同目录的 `results.jsonl`、
`results.csv` 或 `export_full/`。full 导出是结构化结果目录，不应当作原始 SQLite 日志和
Trace 表的完整备份。

不传 `--database` 时使用内存 SQLite：配置与结果仍经过数据库，但不生成持久化数据库文件，
进程结束后不能续跑。如需留下结果，明确指定自动导出：

```bash
nexus run batch --generator ./generator.py --limit 2 \
  --export-format jsonl --export-path ./outputs/once.jsonl
```

内存模式启用自动导出时必须提供 `--export-path`。没有条目需要执行时，单进程可能不创建
增量导出文件；有持久化数据库时仍可显式执行 `nexus export`。

## 日志、Trace 与 Gateway

默认保存本地日志和原始 span 事件。按需关闭某类写入：

```bash
nexus run batch --generator ./generator.py --database ./outputs/light.db \
  --no-persist trace --no-persist log
```

`--no-persist` 仅控制本地 Batch 的相应记录，不代替业务 Processor 或其他遥测模块的开关。
查看任务状态、结果、日志和 span 树可使用：

```bash
nexus inspect-batch ./outputs/batch.db
sqlite3 ./outputs/batch.db 'SELECT run_id, status, retry_count FROM runs;'
```

本地 SQLite 的主要内容：

| 表 | 内容 |
|---|---|
| `batch_info` | 批次 metadata、BatchConfig、生成完成标记和分片信息 |
| `runs` | 每条 BatchItem 的 config、metadata、状态、输出、时间和重试计数 |
| `run_trace_events` | 原始 `start/update/end` span 事件，供 Inspector 重建 |
| `run_logs` | 按 run、chunk 保存的日志 |

`RunInput.extra` 属于 `runs.config`，不是 `runs.metadata`。本地 Batch 的 SQLite 轨迹与
Cloud 的 RunTraceSession/Trace Backend 是不同的存储路径。

需要 Gateway trajectory 时，给这次运行加上 `--use-gateway=true`。Runner 会把
`openai_clients.main` 包成 gateway client（如果它还不是），注入 `gateway_trajectory`，
并把缺省的 `trajectory_id` 填成这一条的 `run_id`。显式写过的 id 保持不变；client 和
processor 的 id 不一致会在启动时报错。训练任务不要加这个参数，HyTuner 用 rollout key
自己收尾。

可直接参考配套的 [generator](../examples/configs/online-service/batch_generator_gateway.py)
与 [Single 配置](../examples/configs/online-service/main-agent-single-gateway.yaml)：

```bash
nexus run single --use-gateway=true \
  docs/examples/configs/online-service/main-agent-single-gateway.yaml
nexus run batch --use-gateway=true \
  --generator docs/examples/configs/online-service/batch_generator_gateway.py \
  --limit 5 --max-concurrent 5 --database ./outputs/gateway.db
nexus export ./outputs/gateway.db --format full --output ./outputs/gateway
```

运行前按示例准备模型凭据。Batch 的 `run_id` 仍由 BatchItem 或序号决定，trajectory id
跟这一条的 `run_id` 对齐，不会把整个批次写成同一个 id。业务上要沿用 `instance_id`
时，把 `run_id` 设成它即可，不必再单独写 `client_args.trajectory_id`。

该示例的轨迹在 `processor_results.gateway`，full 导出写到
`runs/<run-id>/processors/gateway.json`。带 payload 的结果可用
`nexus.gateway.trajectory.load_trajectory_with_payloads()` 读取。Dataset 场景见
[对应 Compose 配置](../examples/configs/online-service/swe-batch-gateway.yaml)。

## CLI 参数速查

| 参数 | 含义 / 默认值 |
|---|---|
| `[config]` | 普通 RunInput 或 Compose YAML 文件 |
| `-g, --generator` | Python generator 文件 |
| `--data` | Dataset 引用或本地路径 |
| `--split` | Dataset split |
| `--use` | 可重复的 Run Part 名称或 YAML/JSON 文件 |
| `--set` / `--set-json` | 按顺序覆盖 RunInput 字段 |
| `--limit` | generator / Dataset 最多取多少条；默认全部，0 为空批次 |
| `-d, --database` | SQLite 路径；默认 `:memory:` |
| `--max-concurrent` | 总并发数；默认 1 |
| `--worker-processes` | 本地 worker 数；默认 1，多进程需要文件数据库 |
| `--enable-conflict-groups` | 按 metadata.conflict_groups 互斥，仅单 worker |
| `--timeout` | 单任务超时秒数；默认不设 Batch 超时 |
| `--retry-failed` | 初次运行之外的自动重试次数；默认 0 |
| `--resume-workflow-on-retry` | 立即重试时保留 workflow 已提交节点 |
| `--no-resume` | 重跑所有已保存任务；默认跳过成功任务 |
| `--retry-only` | 只选 failed 任务 |
| `--fail-fast` | 失败时尽早停止批次 |
| `--use-gateway` | 包成 gateway client 并注入 gateway_trajectory；可用 `--use-gateway=true` |
| `--no-persist trace` / `--no-persist log` | 可重复，关闭相应的本地记录 |
| `--export-format` | `jsonl`、`csv`、`full` |
| `--export-path` | 自动导出的文件或目录；内存模式启用导出时必填 |
| `-v, --verbose` | DEBUG 日志 |
| `--log-trim` | 日志前后各保留的字符数；默认 200，0 不截断 |
| `--log-trim-head` / `--log-trim-tail` | 单独覆盖日志前段、后段长度 |

完整帮助：`nexus run batch --help`。
