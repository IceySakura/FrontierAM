# AIHub ID Compose Hook

`aihub_ids.py` 是一个 compose hook：批量任务提交前，逐条检查或分配 API key 问号后面的 ID（如 `acc_global_cache_id`、
`cache_task_id`），并记进 `metadata.__extra__`。compose hook 的机制和写法见 [Compose Hook](../../../how-to/compose-hooks.md)。

## 用法

```bash
# 只检查：记录每条任务 key 里的 cache_task_id，缺失或整批重复只告警
--hook docs/examples/configs/compose-hooks/aihub_ids.py

# 给每条任务分配本批唯一的 acc_global_cache_id，原来有就替换
--hook docs/examples/configs/compose-hooks/aihub_ids.py:fields=acc_global_cache_id,mode=unique
```

可用于 `run batch`、`materialize`、`submit batch --generation local`，任务来源可以是 generator 或 Dataset（`--data`）。
例如 Harbor 任务：

```bash
uv run nexus materialize \
  --data 't3-harbor:///path/to/tasks?flat=true' \
  --use harbor_bridge:mode=agent \
  --use nexus/extensions/harbor_bridge/configs/runtime/gongfeng-prod.yaml \
  --use nexus/extensions/harbor_bridge/configs/agents/cc-search-whitebox.yaml \
  --use nexus/extensions/harbor_bridge/configs/llm/aihub-hy3-litellm.yaml \
  --hook docs/examples/configs/compose-hooks/aihub_ids.py:fields=acc_global_cache_id,mode=unique \
  --output /tmp/harbor-tasks.jsonl
```

提交时把 `materialize` 换成 `submit batch`，hook 参数不变。

## 参数

参数写在文件路径后面，用冒号隔开，多个参数用逗号，多个值用 `+`。

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `key` | 要处理的 key 的配置路径，多个用 `+` | `openai_clients.main.config.client_args.api_key` |
| `fields` | 要处理的 key 参数名 | `cache_task_id` |
| `mode` | `check` 或 `unique` | `check` |
| `source` | `unique` 按 metadata 中的哪个字段生成 ID | 任务的 `run_id` |
| `seed` | 生成 ID 时混入的种子；想让同一批任务换一组 ID 时修改它 | 空 |

## key 从哪里读

`key` 是 key 所在字段的配置路径，默认 `openai_clients.main.config.client_args.api_key`。字段是字面量时直接读写
（generator 写好的 key）；路径以 `<字段>__env` 结尾，或字段不存在、但同级配了 `<字段>__env`（Harbor profile 的
`api_key__env: OPENAI_API_KEY`）时，读那个环境变量。只处理 `key` 列出的位置，其他 client 的 key 不动、也不读。

变量的值在 `submit` 时先读 `--host-env`，再读执行命令的 shell 环境；`run batch` / `materialize` 只读后者。多项用 `+` 连接，
同一条任务的所有 key 得到同一个 ID；处理了多个 key 时 metadata 按路径分开记录。

## 两种模式

| 模式 | 做什么 | 改配置 | 会不会让整批失败 |
| --- | --- | --- | --- |
| `check` | 读出每个字段的值记进 metadata；缺失、整批重复、写成 `${...}` 的值逐条告警，结束时汇总 | 不改 | 不会 |
| `unique` | 给每个字段生成 UUID 加进 key 的参数里（规则见下） | 改 | key 读不到、变量已带该参数或 ID 重复时报错 |

- 生成的 UUID 由字段名、`seed` 和 `run_id`（或 `source` 字段）算出：同一条任务每次都一样，整批互不相同。
- **字面量 key**（每条任务自己的配置，如 generator 写的）：ID 写进问号后面，原来有就替换，其他参数保留。
- **来自 `<字段>__env` 的 key**：环境变量是整批共用的，hook 不改它。变量的值里已经带了这个参数时直接报错
  （`<变量> already sets <字段> for the whole batch`），去掉后再用；没有时把 key 写回成对变量的引用再追加 ID：
  `api_key__env: OPENAI_API_KEY` → `api_key: ${oc.env:OPENAI_API_KEY}&cache_task_id=<id>`（变量值没有 `?` 时用 `?`），
  并删掉 `__env`。凭据不进任务清单，子任务运行时展开引用，仍需照常用 `--host-env` 传入这个变量。
- 多个 hook 串联（如一个处理 `cache_task_id`、一个处理 `acc_global_cache_id`）时，后一个 hook 认得前一个写回的
  `${oc.env:NAME}&...`：按「变量值的参数 + 已追加的参数」读取，只改追加部分，结果和写在同一个 hook 里一样。

记录示例：

```json
{"__extra__": {"acc_global_cache_id": "04428265-1176-5ab2-a351-66ff31727bee"}}
```

导出结果里每行的 `metadata.__extra__` 就是这里记录的值。

## 限制

- 只看 key 问号后面的参数。client 在运行时另外改写这个参数的情况不在 hook 的视野里：`client_args.cache_task_id`、
  `cache_task_id_from_run_id`、`gateway_routing`，以及 `openai_responses` / `anthropic_messages` 对 `cache_task_id` 做的
  MD5。用到这些时，实际发出的值以导出结果里运行时记录的 `metadata._nexus.cache_task_id` 为准。
- `--generation remote`、`run single`、`submit single/live` 不能使用 compose hook。
- 不做跨批次去重。

## 报错与告警

| 信息 | 原因与改法 |
| --- | --- |
| `<key> is not configured` | 路径上没有这个 key。检查 `key` 参数 |
| `<key> reads OPENAI_API_KEY, which is set neither in --host-env nor in this environment` | 读不到变量。`submit` 时用 `--host-env` 传入，本地运行时先 `export` |
| `<变量> already sets <字段> for the whole batch` | `unique` 时环境变量的值里已经带了这个参数，整批共用、不能按条替换。从变量里去掉它 |
| `<key> has no <字段>` | `check` 时 key 里没有这个参数。用 `mode=unique` 分配 |
| `<字段> is also used by item #N` | 两条任务的 ID 相同。`check` 下通常是整批共用一个 key，改用 `mode=unique`；`unique` 下是 `run_id` 或 `source` 字段重复 |
| `<key> <字段> is resolved only at run time` | 值写成了 `${...}` 但 hook 解析不了：`${oc.env:NAME}` 的变量在 `--host-env` 和本机环境里都没有，或用了带默认值、嵌套等其他写法。`${oc.env:NAME}` 能读到时按读到的值处理 |
