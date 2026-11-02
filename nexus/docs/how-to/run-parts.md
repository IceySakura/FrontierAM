# Run Part 与配置组合

Run Part 是一段可复用的 `RunInput` 配置，例如 Runtime、Agent、模型连接或 Processor。
使用 `--use` 加载 Run Part，用 `--set` 和 `--set-json` 调整其中的字段。

`nexus run single` 和 `nexus run batch` 使用相同的合并规则。批处理中既可以从 Dataset
生成输入，也可以给 Python generator 生成的输入叠加 Run Part。批次的并发、限量、恢复和
导出见 [批处理指南](run-batch.md)。

## 选择配置来源

| 场景 | 写法 | 配置起点 |
|---|---|---|
| 直接组合一条输入 | `nexus run single --use ./run.yaml --set ...` | 按命令行顺序合并 |
| Dataset 单条任务 | `nexus run single --data <dataset> --id <item> --use ...` | Dataset item 在 `--data` 的位置参与合并 |
| Dataset 批处理 | `nexus run batch --data <dataset> --split <split> --limit 10 --use ...` | 每个选中的 Dataset item 分别合并 |
| Generator 批处理 | `nexus run batch --generator generator.py --limit 10 --use ...` | 每个生成的 `BatchItem.config` 是底值，再应用命令行覆盖 |
| 保存为配置文件 | `nexus run single run.yaml` / `nexus run batch batch.yaml` | 普通 RunInput 或有序 Compose YAML |

位置参数配置文件不能再与 `--data`、`--use`、`--set`、`--set-json` 混用。需要在文件上
继续覆盖时，写成 `--use ./run.yaml`；完整 Compose 文件则直接作为位置参数使用。

一个不需要模型凭据的组合示例是 Echo 的 oracle 模式；执行前需要准备可用的 Docker：

```bash
nexus run single \
  --set task_name=echo \
  --set task_config.message=hello \
  --set task_config.run_oracle=true \
  --use docker \
  --use worker_venv
```

常用的内置 Run Part：

| 名称 | 作用 |
|---|---|
| `docker` | 本地 Docker Runtime provider |
| `gongfeng` | Gongfeng Runtime provider |
| `worker_venv` | 设置 `runtime_configs.main.worker_venv=true` |
| `openai` / `litellm` | 对应的模型客户端配置 |
| `claude_code` | Claude Code Agent 配置 |
| `chat_tracker` | ChatTracker Processor 配置 |

各模块所需的依赖、服务与凭据仍需准备。`--use` 负责组合配置，不表示外部服务已经可用。

## 合并顺序与字段规则

`--data`、`--use`、`--set`、`--set-json` 按出现顺序处理：字典递归合并，后面的同名
标量覆盖前面的值，列表默认整体替换，`null` 删除对应配置键。删除后仍须满足 RunInput
的必填字段和类型要求；有默认值的字段会按模型规则恢复默认值。

```bash
nexus run single \
  --use ./echo.yaml \
  --set task_config.message=first \
  --set-json 'task_config={"message":"second","run_oracle":true}' \
  --set task_config.message=final
```

最终 `task_config.message` 是 `final`。`--set` 的值按 YAML 解析，`true`、`10`、`0.5`
会成为布尔、整数和浮点数；`--set-json` 只接受 JSON。复杂内容应加 shell 引号。

追加列表使用 `+=`：

```bash
nexus run single --use ./echo.yaml \
  --set imports+=nexus.extensions.tasks.echo
```

`--set` / `--set-json` 的路径相对于 RunInput，例如 `task_config.message`、
`agent_config.working_dir`、`runtime_configs.main.worker_venv`、`extra.source`。
它们不是 BatchConfig 参数，也不修改 `BatchItem.metadata`。配置不会自动成为 task 入参：
业务 task 接收 `task_config` 中匹配其输入模型的字段。

### Generator 的合并顺序

Generator 先生成完整的 BatchItem，再按命令行中 `--use`、`--set`、`--set-json` 的相对
顺序覆盖其 RunInput。`--generator` 自身写在前面或后面，都不改变这个顺序：

```bash
nexus run batch --generator ./generator.py \
  --use ./common.yaml \
  --set runner.timeout=600 \
  --set-json 'extra={"source":"smoke"}' \
  --limit 10 --database ./smoke.db
```

`--use` 可以引用已注册的 Run Part 或配置文件。生成器的 `run_id`、metadata 和批次
metadata 保持不变；`generate_configs()` 不接收这些参数。被修改的顶层配置段按 Compose
规则解析环境变量并校验类型，例如 `--set 'runner.timeout=${oc.env:TASK_TIMEOUT}'`。
未修改配置段中的环境变量引用留到正常执行阶段解析；`extra` 和受保护的自由文本字段保持字面值。

Generator 不能与 `--data` 或 `--use data:...` 混用。`--limit` 只在首次生成任务集合时
生效，已锁定的数据库恢复不会解析或重新应用 Run Part 和覆盖参数，也不依赖原覆盖文件。

本页中的 generator 组合用于本地 `nexus run batch`（及别名 `nexus run-batch`）。
Cloud `nexus submit batch` 的生成入口有独立约束，见 [Cloud 批处理](../cloud/submit-batch.md)；
不要把本地 generator 的组合参数直接照搬到 Cloud 提交。

需要在生成后逐条校验、修改 item（读写 metadata、整批去重等）时，用单独的 `--hook` 参数，见
[Compose Hook](compose-hooks.md)。

## 配置文件的两种形状

`--use` 支持 YAML 和 JSON，格式如下。

### 部分配置：RunPart 文件

将下面内容保存为 `common.yaml`：

```yaml
body:
  runner:
    timeout: 300
  runtime_configs:
    main:
      worker_venv: true
source: common-runtime
```

| 字段 | 含义 |
|---|---|
| `body` | 必填；部分 RunInput 配置，不需要包含 `task_name` |
| `merge` | 可选；字段路径到合并策略的映射 |
| `source` | 可选；用于展示来源的名称 |

用 `merge` 指定 `append`、`append_unique` 或 `replace`。例如：

```yaml
body:
  imports:
    - nexus.extensions.tasks.echo
merge:
  imports: append_unique
source: echo-imports
```

`append_unique` 会去掉追加中的重复项；`replace` 也可用于完整替换某个字典，而不递归合并。

### 完整配置：RunInput 文件

将下面内容保存为 `echo.yaml`：

```yaml
task_name: echo
task_config:
  message: hello
  run_oracle: true
runtime_provider_name: docker
```

不采用 `body/merge/source` 形状的文件，必须能通过 RunInput 校验：

```bash
nexus run single --use ./echo.yaml --use ./common.yaml
```

文件路径相对于当前工作目录；文件内的本地 `task_files` 路径相对于该文件所在目录。
文件 `--use` 不接受 `:param=value` 参数，参数化配置请使用已注册的 Python Run Part。

## Compose YAML

Compose 文件把来源和覆盖顺序写在同一个文件中：

```yaml
compose:
  - data:
      uri: ./dataset
      split: train
      limit: 10
  - use: ./common.yaml
  - input:
      agent_config:
        working_dir: /app
```

```bash
nexus run batch ./batch.yaml --database ./results.db
```

`data` 贡献 Dataset item 的配置和文件引用，`use` 引用 Run Part，`input` 提供内联的部分
RunInput。一个 Compose 只支持一个数据源。本地 dataset 和 `use` 文件路径相对于 Compose
文件所在目录；完整 Compose 文件不是 `--use` 的配置片段，也不支持内嵌 generator。

`data` 可填写 `uri`、`split`、`id`、`limit`。单条运行从多样本数据集中选择时，需要指定
`id`；`limit` 只用于批处理。CLI 的 `--split`、`--limit` 可覆盖文件中的选择设置。

内联配置也可指定合并策略：

```yaml
compose:
  - use: ./echo.yaml
  - input:
      task_config:
        message: replacement
        run_oracle: true
    merge:
      task_config: replace
```

普通 RunInput 和 Compose 的内联 `input` 均可使用本地文件简写，例如
`task_files: ./files`；文件交付与 Cloud 的限制见对应运行指南。

## 注册参数化 Run Part

Run Part 函数接收一个参数字典，返回部分 RunInput 字典或 `RunPart`：

```python
from typing import Any
from nexus.runner.compose import run_part

@run_part('run_timeout')
def run_timeout(params: dict[str, Any]) -> dict[str, Any]:
    unknown = set(params) - {'seconds'}
    if unknown:
        raise ValueError(f'Unknown parameters: {sorted(unknown)}')
    return {'runner': {'timeout': int(params.get('seconds', 300))}}
```

在可发现的 Nexus 模块 metadata 中声明入口，并安装该模块：

```toml
[tool.nexus]
run_parts = { run_timeout = "my_package.run_parts" }
```

```bash
nexus run batch --generator ./generator.py \
  --use run_timeout:seconds=600 --limit 10 --database ./timed.db
```

`name:key=value,flag=true` 的参数按 YAML 值解析；参数传给 Run Part 函数，
不是传给 `generate_configs()`。

## 程序内复用 Compose

连续解析多个配置时，可以复用 `RunComposer` 的 Reader 连接及已加载样本缓存：

```python
import asyncio
from nexus.runner import run
from nexus.runner.compose import RunComposer

async def main():
    async with RunComposer() as composer:
        first = await composer.compile_single_file('first.yaml')
        second = await composer.compile_single_file('second.yaml')
    return await run(first), await run(second)

first_output, second_output = asyncio.run(main())
```

Compose 完成后得到的是普通 RunInput / BatchItem，执行器不会再次解析 Dataset。
