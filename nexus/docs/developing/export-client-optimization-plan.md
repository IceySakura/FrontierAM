# `nexus export` 客户端侧优化方案（v4）

- 状态：已实施（v4：可配置 DB 预算 + 默认关闭硬 deadline）
- 日期：2026-10-27
- 范围：**仅 `nexus export` 客户端**
- 基线：`origin/main`（`a3bc789367`）
- 约束：不改 executor、上传侧、sink、control plane、DB 参数、部署配置

本版只保留 4 项最值得做的改造，其他内容全部移除。

---

## 1. 问题摘要

线上 postprocess 失败：

```text
executor stage postprocess failed
└─ nexus export exit 1
   └─ Result export failed for item ..., child exe_...
      └─ Trace Database query failed for exe_... after 3 attempts: TimeoutError
```

已确认的客户端侧根因：

1. `has_partition_column()` 每条 record 都可能 probe 一次；probe 失败被静默吞掉，分区裁剪可能失效。
2. 固定 90 秒 `wait_for` 包住了整个 `_read_chunk()`：DB 传输 + spill 落盘 + JSON 扫描都算在同一个预算里；健康的大 chunk 也会被误杀。
3. `_read_root_output()` 为了找最后一个 `run.output`，会读取该 run 的所有 root update record 正文。
4. DB 读取报错（含超时）直接中止整批；只有“DB 读取成功但没有输出”才回退 API。

下面 4 项分别对应这 4 个问题。

---

## 2. 改造一：自动分区 + probe 缓存

### 解决什么

- probe 失败静默关闭分区裁剪，导致全分区扫描；
- 每条 record 重复 probe；
- classic 表 + `--parent-created-at` 变成 1054 重试。

### 客户端改动

复用 `nexus/distributed/export/follow.py` 现有能力，但先做两处客户端侧重构：

- 把 `run_trace_export()` 内部的 `prepare_scope()`（`cli.py:365-410`）抽成独立 helper，例如 `resolve_trace_read_scope()`，traces/results 共用；它现在是闭包，不能直接复用。
- `resolve_partition_key()` / `resolve_trace_backend()` / `_sink_entries()` / `_retryable()` / `_TRANSIENT_DB_CODES` 直接复用。

分区键解析规则（本版只依赖 sink envelope；source binding 作为后续增强）：

```text
1. 显式 --parent-created-at：ops override；仍做 dialect 安全校验
2. sink envelope 有 key，且 table/backend 匹配 → 直接用 key，不 probe
3. 否则 probe 一次：
   - partitioned=True + 能从 Control API 拿到 key → 用 key
   - partitioned=True + 拿不到 key → fail-closed
   - partitioned=False → 不加谓词
   - probe 失败 → fail-open + warning（显式 override 时 fail-closed）
4. 都没有 → 不加谓词
```

实现要点：

- `has_partition_column()` 用独立短超时（5–10s），不要复用数据查询预算；结果按 `(trace_backend, table)` 缓存在原始 reader 上，`asyncio.Lock`/共享 `Task` 去重；
- 显式 `--parent-created-at` 不能触发 `_execution_trace_reader()` 的 early-return 而跳过 dialect 校验；先 probe/确认，classic 表明确报错；
- Batch parent 没有 sink envelope：probe 一次，用 Batch 父执行 `created_at`；
- Batch child 合法只能是 `nexus_single`（`repositories/batch.py:1623` 强制）；直接使用 Batch parent `created_at`，**不逐 item 查 child execution**；保留一个防御性校验：若发现非 Single child，明确报错；
- `--trace-backend` 从两个 CLI 入口透传到 `resolve_trace_backend()`；多 backend 歧义时要求显式指定；
- 不新增用户参数；`--parent-created-at` 仍是可选 override。

### 验收

- 10-item Boundless Batch 只 probe 一次；
- `EXPLAIN` 只命中一个分区；
- classic 表 + 显式 key 明确报错，不是 1054；
- probe 失败有 warning。

---

## 3. 改造二：超时拆分 + 连接安全的重试

### 解决什么

- 固定 90 秒把 DB 传输、spill 落盘、JSON 扫描包在一起，无法区分“卡死”和“大 chunk 正常读取”；
- 超时后 aiomysql 已关闭连接，原连接上重试必然失败；
- 一个 chunk 最坏可能 N×90 秒；
- 用总 wall-clock deadline 兜底会把正常的长导出误判成故障。

### 客户端改动

- `wait_for` 只包 `cursor.execute()`；`cursor.fetchall()` 和本地 spill/scan 移出 DB 超时；
- `_fetch_rows()` 只做 per-call timeout，不内部重试；
- `_read_chunk_with_retry()` 保留为唯一重试层：
  - 最多 3 次；
  - 只对 timeout / connection / transient 错误重连并重跑 chunk；
  - 避免在已关闭连接上重试；
- DB 单次 `cursor.execute()` 预算可配置：`NEXUS_EXPORT_TDSQL_TIMEOUT_SECONDS`，默认 900s；这是语句总时长，不是 socket idle；
  - 健康的长读由该预算决定，超时后才进入重连/重试；
  - aiomysql 0.3.2 没有 `read_timeout`，因此不在 DB 侧实现 idle timeout；
- item / export 硬 deadline 默认关闭：`NEXUS_EXPORT_ITEM_DEADLINE_SECONDS`、`NEXUS_EXPORT_DEADLINE_SECONDS`；
  - 默认不设置，或显式设为 `0` / `off` / `none` / `disabled` 均为关闭；
  - 需要防止无限重试时才显式开启（例如 item=600s、export=3600s）；
  - 关闭时 `_single_record()` 使用 `contextlib.nullcontext()`，不创建任何总时长计时器；
- export deadline 仅在开启时生效，超时行为：
  - 抛 `ExportDeadlineExceeded`（不可重试）；
  - `writer.abort()`，保留 `.partial` 供诊断；
  - 不发布正式文件；
  - 错误信息带当前 item/phase；
- API 流式下载保持 `total=None` + `sock_read=900s`：这是真正的 read-idle 超时，只要持续有数据就不会触发；
- 本地 spill/scan 不额外套 wall-clock watchdog；避免把正常的大文件处理误判为卡死；
- 本地处理错误统一包成 `TraceProcessingError`，不进入 DB 重试；`_read_chunk_with_retry()` 只重试 `_retryable()` 判定的 DB 错误；
- 错误信息带 `phase` / `sql_kind` / `attempts` / `db_code`；
- 错误分类复用 `follow._retryable()`；补充 3024、1317 等。

### 验收

- 本地 scan 慢不再报 `Trace Database query failed ... TimeoutError`；
- 健康的长 DB 读不会因为固定的 90s 总预算被误杀；超过可配置预算才超时；
- execute 超时后重试能重连并成功/明确失败；
- item/export deadline 默认关闭；显式开启后按时生效，未开启时不会因总时长中止；
- export deadline 超时不发布正式文件，保留 `.partial`；
- 本地处理错误不会被当成 DB 错误重试；
- 现有输出与改造前字节级一致。

### 不做

- DB 侧 idle timeout（aiomysql 0.3.2 无 `read_timeout`，客户端只能做语句总时长）；
- API 下载进度 watchdog / 本地 scan `tick()`；
- SSCursor / 非缓冲游标改造；
- `MAX_EXECUTION_TIME`；
- `CAST AS BINARY`。

---

## 4. 改造三：瞬时 DB 错误按 item 回退 API

### 解决什么

- 一次 TDSQL 超时导致整个 Batch 失败；
- executor 整批重试仍命中同一个慢 item。

### 客户端改动

- results 导出中，捕获 `TraceDatabaseTimeout` / `Unavailable` / `Throttled` / `Interrupted`；
- 按 item 回退 Application API，复用现有 `get_result_to_file` 流式路径（`total=None`、`sock_read=900s`）；
- 回退前重置：
  - `scan_out`；
  - 旧 spill 文件（或交给 `ResultSpillPool.sweep`）；
- 回退后重新设置 `scan_out['prepared']` / `path`；
- 避免重复调用 `get_control_events()`（复用已取到的 start/end）；
- API 也失败时聚合两个错误；
- 日志/字段记录 `source=api-fallback`；
- auth / schema / data-invalid / processing 错误不回退；
- traces 导出不回退。

默认值：

- **results 客户端默认 `api`**（仅瞬时错误），在 `nexus/cli/commands/export.py` 和 `nexus/distributed/export/cli.py` 两个入口都生效；这样不依赖 executor/部署配置；
- `--trace-db-fallback=fail` 或 env 可显式关闭；
- traces 入口拒绝/忽略 `--trace-db-fallback` 并 warning；
- API/artifact 作为结果权威源；正常场景 DB/API 输出 byte-identical，回退场景日志记录 `source=api-fallback`。

### 验收

- 模拟 TDSQL 超时 → 该 item 走 API，导出成功；
- auth/schema/data-invalid 不回退；
- API 也失败时错误包含两个原因；
- 正常场景 DB/API 输出 byte-identical；
- 两个 CLI 入口默认一致；traces 入口拒绝该选项；
- 不传新参数也能生效。

---

## 5. 改造四：倒序读取 root record

### 解决什么

- 为了找 `run.output`，读取该 run 所有 root update record 正文；大 trace 时非常慢。

### 客户端改动

- 每个 `run_span_id` 先查最近 N 条：
  ```sql
  SELECT record_id, time_unix_nano
  FROM <table> FORCE INDEX (idx_traj_span_events_span_id)
  WHERE run_id=%s [AND parent_created_at=%s]
    AND run_span_id=%s AND span_id=%s AND event_type='update'
  ORDER BY record_id DESC
  LIMIT N
  ```
  N 默认 **128**（可配置，建议上限 256/512）；N 越大越不容易漏掉，但 body 读取成本越高；
- 先取 N 条元数据，再读取这 N 条的 `event_data`；按 key 比较 N 条内**所有** `run.output`，不因第一个命中就停止；
- 仅当 N 条中**最大 key** 对应的记录包含 `run.output` 时，才把它作为候选并停止该 span；否则保留窗口内状态，从窗口最小 `record_id` 之前继续升序读取更旧记录，避免把窗口正文读两遍；只有检测到非单调时才回退全量扫描；
- 如果最新记录没有 `run.output`，继续向更旧记录找第一条 `run.output`；找到后查一次更旧记录的 `MAX(time_unix_nano)`，只有存在时间更晚的旧记录时才继续扫描旧 body；
- 如果检测到 `record_id` 与 `time_unix_nano` 明显不单调（例如 N 条内最大 record_id 与最大 time 不是同一条），直接回退全量扫描，避免选错；
- 复用现有 `latest_key` / `candidate_none_output` / `candidate_path` / `candidate_prepared` 状态机；
- 同时支持 streaming（`scan_out`/`spill_pool`）和 legacy decode；
- 用 feature flag 控制：`NEXUS_EXPORT_ROOT_SCAN=reverse|full`，验证后默认 `reverse`。

实现约束：

- 反向扫描只作为“候选发现”：读取 N 条 body 后比较所有 `run.output` 的 key，不立即停止；
- 仅当 N 条内最大 key 命中 `run.output` 时，才认为该 span 的候选确定；否则回退全量；
- 严格意义上的“全局最大”仍需继续扫描窗口之外的旧记录确认；N 窗口是性能与正确性的折中，N 越大越安全，检测到非单调即回退全量；
- 反向扫描的 candidate 与其它 span 仍按 `(time_unix_nano, span_id, record_id)` 比较；
- `output=None` 也是终态；
- 若检测到 record_id/time 明显不单调，直接回退全量；
- streaming（`scan_out`/`spill_pool`）和 legacy decode 两条路径都保留；
- 用 `EXPLAIN` 验证 `FORCE INDEX (idx_traj_span_events_span_id)` + `ORDER BY record_id DESC` 是否 filesort；若代价过高，记录并接受/调整索引策略（不新增 DB 索引）。

### 验收

- 最新记录有 `run.output` 时只读取少量 body；
- N 条内比较所有 `run.output` 的 key；最大 key 命中才停，否则回退；
- `output=None` 终态、多 span/多 attempt、非单调 record 测试通过；
- `EXPLAIN` 已确认索引/filesort 代价可接受；
- 正常输出与改造前字节级一致。

### 不做

- `OCTET_LENGTH` 热路径；
- 按大小选择 API。

---

## 6. 明确不做

- resume / manifest / checkpoint / sidecar / 状态目录；
- executor 重试编排、上传白名单、stage timeout；
- sink 截断、artifact 指针、control plane / DB 参数；
- `CAST AS BINARY`、idle timeout、`MAX_EXECUTION_TIME`、按大小选择 API、`OCTET_LENGTH`；
- `--restart`、`--no-resume`、`--item-ids` 编排；
- 独立的 diagnostics 工程（只在前 3 项里加最小耗时/phase 日志）。

---

## 7. 落地顺序

1. **改造一 + 改造二**：低风险，先解决分区裁剪、超时归因和重连。
2. **改造三**：可以直接让本次事故的 item 绕过 TDSQL，作为应急开关。
3. **改造四**：性能优化，放最后，feature flag 灰度。

---

## 8. 测试与验收

- 单元测试：
  - 分区解析：Batch Single、Batch child 非 Single（防御性报错）、classic、override、probe 失败、多 backend 歧义；正常 Batch 不额外查 child；
  - 共享 helper：`prepare_scope()` 抽取后 traces/results 行为一致；
  - probe 缓存：10 item 只 probe 一次；短超时不占用数据查询预算；
  - 超时：本地 scan 不触发 DB timeout；execute 超时后重连；DB 语句预算可通过 `NEXUS_EXPORT_TDSQL_TIMEOUT_SECONDS` 调整；item/export deadline 默认关闭且显式开启后生效；本地处理错误不被当成 DB 错误重试；
  - 回退：瞬时错误回退 API；auth/schema/data-invalid 不回退；API 也失败聚合；两个 CLI 入口默认一致；traces 拒绝该选项；
  - 倒序扫描：最新命中、N 条内多个 `run.output` 取最大 key、最大 key 命中才停、未命中回退、`output=None`、多 span、非单调 record、`EXPLAIN` 索引/filesort；
- 集成/E2E：
  - 合成 ≥1GB `event_data`，验证不因本地 scan 触发 DB timeout；
  - 模拟 TDSQL 超时，验证 API 回退成功；
  - Boundless `EXPLAIN` 单分区；classic 无谓词；
  - 现有命令不传新参数。
- 回归：
  ```bash
  uv run --no-sync pytest -q tests/unit/distributed -n auto
  ```

---

## 9. 附录：代码映射

| 位置 | 作用 |
|---|---|
| `results.py:320` | `has_partition_column()` |
| `results.py` probe | `wait_for` 使用独立 5s 短超时，不复用数据查询预算；数据查询预算由 `NEXUS_EXPORT_TDSQL_TIMEOUT_SECONDS` 控制（默认 900s） |
| `results.py:383` | `_read_chunk_with_retry()` |
| `results.py:417` | `_read_chunk()` |
| `results.py:479` | `_read_root_output()` |
| `results.py:787` | `_fetch_rows()` |
| `results.py:917` | `_batch_parent_partition_key()` |
| `results.py:1164` | `_execution_trace_reader()` |
| `results.py:1192` | `_single_record()` |
| `results.py:1634` | `_read_result()` |
| `follow.py:207` | `resolve_partition_key()` |
| `follow.py:304` | `resolve_trace_backend()` |
| `follow.py:92-116` | `_retryable()` / `_TRANSIENT_DB_CODES` |
| `cli.py:365-410` | `prepare_scope()`（Batch probe + classic override 报错） |
| `client.py:43-52` | API 结果流式 timeout（`total=None`，`sock_read=900`） |
