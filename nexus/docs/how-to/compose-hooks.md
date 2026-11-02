# Compose Hook：生成后逐条校验与补全

Compose hook 是用 `--hook` 传入的 Python 后处理（`--use` 只用于配置片段）：每条批处理 item 在合并完 Run Part 之后、写入
Task List 或本地数据库之前，都会交给它校验、修改或拒绝。它可以读写 `config`、`run_id` 和
`metadata`，并在整批 item 之间保持状态，例如检查 ID 不重复。写进 item metadata 的字段会原样进入结果导出。

## 适用范围

| 命令 | generator | Dataset（`--data`） |
| --- | --- | --- |
| `nexus run batch` | 支持 | 支持 |
| `nexus materialize` | 支持 | 支持 |
| `nexus submit batch --generation local` | 支持 | 支持 |

### 当前不支持

| 场景 | 行为 | 原因 / 替代 |
| --- | --- | --- |
| `submit batch --generation remote` 使用 `--hook` | 报错 `--hook requires --generation local` | remote 由云端 Batch Manager 用已部署代码生成任务，提交规格也没有 hook 字段；需要发布 Manager 与 control plane 后才能支持。改用 `--generation local` |
| `run single`、`submit single`、`submit live` 使用 `--hook` | 参数不存在，直接报错 | 单次运行没有 batch item 与 metadata |
| 跨批次去重 | 不检查 | hook 状态只在一次生成内共享 |

## 执行顺序

```text
generator 或 Dataset ──逐条──▶ 合并 --use / --set ──▶ --hook（按书写顺序）──▶ 写入或上传
```

- `--hook` 总在所有 `--use` / `--set` 合并之后执行，与它在命令行中的位置无关；多个 `--hook` 按书写顺序依次处理。
- `--limit` 先截断，只有被选中的 item 会经过 hook。
- 任一 hook 抛出异常时整次生成失败，错误信息包含 hook、item 序号和 `run_id`，不会上传 Task List。
- 本地 `run batch` 恢复已锁定的 generator 数据库时不会重新执行 hook。

## 编写 hook 文件

文件中定义 `create_compose_hook(params)`。它在每次生成时调用一次，返回逐条处理函数，或带
`process()` 与可选 `finish()` 的对象。`finish()` 在最后一条之后调用，用于整批检查。

下面的 hook 要求每条 item 带指定的 metadata 字段、整批不重复，并复制到 `RunInput.extra`：

```python
def create_compose_hook(params):
    field = params.get('field', 'case')
    owners = {}

    class RequireMetadata:
        def process(self, item):
            value = str(item.metadata[field])
            if value in owners:
                raise ValueError(f'metadata.{field}={value!r} is already used by {owners[value]}')
            owners[value] = item.run_id
            item.config.extra.setdefault('labels', {})[field] = value
            return item

        def finish(self):
            if not owners:
                raise ValueError('the batch produced no items')

    return RequireMetadata()
```

保存为 `require_metadata.py` 后：

```bash
uv run nexus materialize \
  --generator ./generator.py \
  --hook ./require_metadata.py:field=case \
  --output /tmp/compose-hook-tasks.jsonl
```

`:key=value` 参数按 YAML 解析后传给 `create_compose_hook`。`process()` 必须返回 `BatchItem`，返回的
配置会重新经过 `RunInput` 校验。

可复用的 hook 可以注册，并在模块 metadata 中声明：

```python
from nexus.runner.compose import compose_hook

@compose_hook('require_metadata')
def create_require_metadata(params): ...
```

```toml
[tool.nexus]
compose_hooks = { require_metadata = "my_package.hooks" }
```

注册后用 `--hook require_metadata:field=case` 引用。

## 示例

[docs/examples/configs/compose-hooks/aihub_ids.py](../examples/configs/compose-hooks/README.md)：检查（`mode=check`）
或按条分配本批唯一的（`mode=unique`）key 中的 `cache_task_id`、`acc_global_cache_id` 等参数，把每个请求实际会带的值记进
`metadata['__extra__']`。用法、参数，以及哪些配置下它会报错、怎么改，见该目录的 README。
