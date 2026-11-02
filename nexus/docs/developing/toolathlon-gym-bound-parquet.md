# Toolathlon-Gym bound parquet

Self-contained Toolathlon synthesis rows may embed a `data_binding` object in
`nexus.task_config`. A bound row selects its own isolation topology and binds
the security-relevant task spec to the embedded archives.

## Compatibility default

Rows without `nexus.task_config.data_binding` always use the historical
single-sandbox, non-isolated behavior. The old
`TOOLATHLON_GYM_ISOLATE_MCP_SERVER` environment variable does not opt an
unbound row into isolation. This makes old parquet/config files forward
compatible and requires every new isolation decision to be explicit in data.

## Backend-mock contract

`isolation=backend_mock` means:

- all MCP servers, mock databases, state dumps, server code and evaluator data
  run in `mcp_backend`;
- the main sandbox contains only an empty `agent_workspace`;
- the agent may return only the relative files listed in `output_artifacts`;
- only those files are copied from main to the backend evaluator;
- prompt/server/evaluator changes and archive changes fail hash verification;
- MCP specs referencing `agent_workspace` and MCP tool implementations doing
  direct host-file I/O are rejected when the parquet is bound.

Mock MCPs may be read-write. The binding records `mock_state_access` as
`read_only` or `read_write`; writes stay in the backend and are persisted by
the case's own database or `--state-dump` implementation.

## Bind a parquet

The conversion command needs `pyarrow`, but normal task execution does not:

```bash
python -m nexus.extensions.tasks.toolathlon_gym.bind_parquet \
  input.parquet output.parquet \
  --dataset toolathlon_synthesis
```

For continuous mock-state data:

```bash
python -m nexus.extensions.tasks.toolathlon_gym.bind_parquet \
  input.parquet output.parquet \
  --dataset mcp_synthesis_continuous
```

The command preserves all parquet columns and only adds
`nexus.task_config.data_binding`.

Traditional Toolathlon Gym data uses a split placement:

```bash
python -m nexus.extensions.tasks.toolathlon_gym.bind_parquet \
  toolathlon_gym_union_hy3rollout.parquet \
  toolathlon_gym_union_hy3rollout.bound.parquet \
  --dataset toolathlon_gym
```

- file/workspace MCPs run in main;
- protected PostgreSQL/mock-state MCPs run in backend;
- Email, Canvas and arxiv-local use a serialized hybrid mirror around each
  call because they combine protected state with local file upload/download;
- main is the authoritative workspace, and the final evaluator receives a
  clean mirror so modifications and deletions are preserved.

## Run bound rows

Convert selected parquet rows to JSONL without changing the embedded `nexus`
object, then use:

```bash
export SYNTHESIS_MANIFEST_JSONL=/path/to/manifest.jsonl
nexus run-batch \
  -g nexus/extensions/tasks/toolathlon_gym/synthesis_generator.py \
  -d /path/to/batch.db
```

The runtime image embedded in the parquet is used by default. In particular,
encrypted synthesis rows must retain their matching secure image; do not set
`MCP_BENCHMARK_MAIN_IMAGE` unless intentionally replacing it with a compatible
image.

Operational overrides such as model client, runtime token/resources,
`max_iterations`, and run timeout remain configurable. The bound task prompt,
MCP specs, evaluator and embedded data cannot be independently changed.
