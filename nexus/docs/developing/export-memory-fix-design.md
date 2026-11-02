# 导出内存与完整性设计

- 更新日期：2026-10-14。
- 基线：`main` @ `9c23580b74`（1.8.18）。
- 状态：基础方案已实现并合入（`e9a147bf`、`6b29d96`）；Cloud 读入优化已实现，
  生产端到端验收已执行（2026-10-14，见 F.5.3）；分支已 rebase 到
  `origin/main` @ `b86e6cc719`（1.8.19），准备合入。
- 范围：`nexus export <batch.db>`（SQLite 导出，jsonl/csv/full）、`nexus run batch --export-format`
  （执行期增量导出与分片汇总导出）、`nexus export --content results`（Cloud 终态结果导出）。
- 关联：[Nexus Cloud 结果与实时事件导出设计](cloud-export-events-design.md)。本文替换其中 results 的
  “写后整行校验 + 逐行解析过滤”实现；选择、过滤规则、快照复核和发布语义保持不变。
  远端 `/v1/executions/{id}/result` 的流式下发改造位于
  `codex/application-gateway-result-pipeline` 分支（设计文档随该分支维护，未随本分支）；
  本文附录 I 描述的下载超时与旧服务端兼容设计不依赖该改造。

## 设计摘要

### 设计目标

本设计解决两类独立问题：

1. 本地 SQLite 导出不再把整个 batch 的 JSON 大字段一次性加载到内存。
2. Cloud results 导出不再在关闭和过滤阶段重复解析整个 JSONL 文件。

同时保证：

- 合法输入的 JSONL 字段、顺序和字节内容保持兼容；
- 导出失败不发布不完整的正式文件；
- 默认路径仍保留完整的 Batch/Execution 快照复核；
- 孤立 UTF-16 代理不会再导致 UTF-8 写出失败。

### 核心方案

| 路径 | 方案 | 内存边界 |
| --- | --- | --- |
| 本地 SQLite | `iter_runs(fields, prefetch)` 按行读取；CSV 只查询摘要列；JSONL 分块序列化 | 约为单条记录对象图 |
| 本地增量/分片导出 | 写入 `.partial`，成功后原子替换；中断保留诊断文件 | 不保留整个 batch |
| Cloud results 写出 | 写时记录语义、偏移、hash 和产物摘要 | 不重复解析已写文件 |
| Cloud results 校验/过滤 | 默认使用索引和字节校验；strict 或索引超限时回退逐行解析 | 默认不随最大行解析对象增长 |
| Cloud results 读入 | API 响应按块落盘，消费时内存映射；`orjson` 读入；TDSQL 缺失结果按需走同一 fallback | 同时解析一条结果；仍保留单条对象树 |

Cloud 读入优化只解决客户端的“多条结果重叠”和 Python bytes 副本，
不提供字段级增量 JSON 解析。远端 Application Gateway 自身的完整解码由结果流式管线解决（见附录 I）；
客户端对该管线及旧版（缓冲式）服务端均兼容，下载超时形态见附录 I。

### 文件和发布协议

- 普通文件导出：`<target>.partial` → flush/fsync → 原子替换 `<target>`。
- Cloud JSONL：`.partial`、lock、预取文件均与目标同目录；预取文件使用 `0600`。
- Cloud results 只有校验、快照和 metadata 复核全部成功后才发布正式文件。
- `full` 目录树采用带锁的 staging/swap 和崩溃恢复；普通 POSIX rename 不能保证无名称间隙的单操作原子替换。

### 明确不做

- 不改变现有 JSONL 结果格式；
- 不引入 ijson，不实现字段级增量转换；
- 不把超大字段改成 sidecar；
- 远端结果接口的流式化由 Application Gateway 结果管线承接（已合入，见附录 I）；客户端不依赖服务端版本，两种服务端形态均可工作；
- 不改变增量导出只包含本次实际执行 run 的既有语义。

### 当前状态

- 基础方案已实现；
- Cloud 读入优化已实现，专项测试通过；
- 已合入 Application Gateway 结果流式管线（服务端）与客户端下载超时适配（附录 I）；
- 结果管线 transcoder 的分块状态机 bug 已发现并修复（F.5.3）：块边界落在字符串
  内部时会在字符串内容中插入空格，静默损坏 data URL、路径与嵌入代码；
- 生产端到端验收已执行（2026-10-14，F.5.3）：生产 batch 按 item 逐字节对比，
  可比的 9 条全部与 main 基准一致；main 对 4.0 GB item OOM 无产物；
- 单个超大字段和远端 Gateway 的内存下限仍未解决。

后续章节是实现细节、行为变化、取舍和验证证据，作为设计决策的依据，不是实施步骤。

## 附录 A：问题背景

导出链路有两条独立路径，内存问题的根因不同。

| | 本地 SQLite 导出 / 执行期导出 | Cloud results 导出 |
| --- | --- | --- |
| 根因 | `get_all_runs()` 用 `SELECT *` 把整个 batch 连同全部 JSON 大字段反序列化进内存；`json.dumps` 再为每条记录拼一个完整字符串 | 写入本身是流式的，但写完后 `_validate_jsonl_file` 以文本模式逐行 `json.loads` 整个文件，`--result-status failed/all` 的过滤又逐行解析一遍 |
| 峰值量级 | O(整个 batch 的对象图) | O(最大单行 × 3~4)；行内只要出现一个非 BMP 字符（如 emoji），CPython 就按 4 字节/字符存放整行 |
| 格式影响 | csv 与 jsonl 峰值相同，换格式不省内存 | — |

同时存在三个正确性问题，与内存修复在同一批代码上处理：

1. **孤立代理导致整个导出失败。** LLM 输出按 token 截断时，emoji 的代理对可能只剩一半（如 `'\ud83d'`）。
   它以 `"\ud83d"` 的合法转义进入数据库，读回后成为孤立代理；本地 `json.dumps` + 文本写出、
   `iterencode`、Cloud 的 `iter_json_bytes` 在编码为 UTF-8 时都会抛 `UnicodeEncodeError`。一条坏轨迹即可让整个导出失败。
2. **本地导出没有原子性。** 直接写目标路径，中途失败会在目标路径留下截断但看似完整的文件。
3. **执行期增量导出中断时的结果有歧义。** 如果只改成"完成才发布"、不处理旧文件，中断后目标路径会保留上一次 batch 的完整文件，容易被当作本次结果。

### A.1 目标与非目标

目标：

1. 本地导出峰值从 O(整个 batch) 降到 O(单条记录)。
2. Cloud results 默认的校验和过滤阶段不再解析 JSON。
3. 对改动前能成功导出的数据，输出逐字节不变（本地 jsonl/csv、Cloud results 的三条校验路径）。
4. 孤立代理不再导致导出失败，写出的仍是合法 JSON，且能往返还原。
5. 目标路径上不出现截断文件，也不保留上一次运行的旧文件。

非目标：

- `full` 格式不承诺无名称间隙的单操作原子替换。
- 单个超大字符串值的分块（本地 `iterencode` 会把单个 `str` 一次产出），需要大字段外置。
- 引入新依赖（已放宽为：仅 `orjson` 进入 `distributed` extra，且只用于 Cloud 读入解析，见 C.6；
  ijson 仍否决，写出路径维持 stdlib）。
- 改变增量导出"只包含本次实际执行的 run"的既有语义。

## 附录 B：本地 SQLite 导出

### B.1 流式读取：`BatchDatabase.iter_runs`

```python
async def iter_runs(
    self,
    *,
    status: str | None = None,
    fields: Sequence[str] | None = None,
    prefetch: int | None = None,
) -> AsyncGenerator[dict[str, Any], None]: ...
```

- 只能选择 `runs` 表的 13 个已知列，未知列抛 `ValueError`，因此列名直接拼进 SQL 也没有注入风险。
  数据库的 `SCHEMA_VERSION` 是锁定的，显式列出列名不受 schema 漂移影响。
- 排序沿用 `ORDER BY created_at`，JSON 列解析失败时置 `None`，与原 `get_all_runs` 一致。
- `get_all_runs` 改为 `iter_runs` 的薄封装：完全消费后才返回，既有调用方的行为和单连接约束不变。

**预取策略。** aiosqlite 的 `async for` 每次线程往返取 `iter_chunk_size` 行，默认 64 行。
不调整的话，"流式"读取实际会同时持有 64 行原始文本：

| 选择的列 | 默认 `prefetch` | 理由 |
| --- | --- | --- |
| 含 `config`/`metadata`/`task_output`/`agent_output`/`processor_results` 任一 | 1 | 峰值为 1 行原始文本加其反序列化结果 |
| 仅摘要列（如 csv 的 6 列） | 64 | 行很小时逐行往返慢约 17 倍（5 万行：0.14s 对 2.36s） |
| `get_all_runs` | 显式 64 | 本来就全量物化 |

实测 100 行 × 2MB：`iter_runs(prefetch=64)` 的 tracemalloc 峰值为 202MB，默认（1 行）为 6MB。

**单连接约束。** `BatchDatabase` 只有一条 aiosqlite 连接。迭代期间不能在同一连接上发起其他查询；
现有调用点的循环体内只写文件。所有调用点都用 `contextlib.aclosing` 包裹，保证异常时游标先于连接关闭。

| 调用点 | 列 |
| --- | --- |
| `export_jsonl`、`export_full` 及其分片版本、`_export_sharded_results` 的 jsonl/full | 全部列（jsonl 需要 `config.extra`，full 需要 `config`） |
| `export_csv` 及其分片版本、`_export_sharded_results` 的 csv | `_CSV_FIELDS`：`run_id, status, start_time, end_time, retry_count, error` |

csv 的内存收益主要来自只选这 6 个摘要列，而不是格式本身。

### B.2 分块序列化：`write_run_to_jsonl`

用 `json.JSONEncoder(ensure_ascii=False).iterencode(record)` 逐块写出，取代
`f.write(json.dumps(record, ensure_ascii=False) + '\n')`：

- 输出与 `json.dumps(record, ensure_ascii=False)` 逐字节相同。
- 不会再为整条记录物化一个完整字符串。
- 代价：`json.dumps` 只有一次性编码时才走 C 编码器，`iterencode()` 始终走纯 Python 编码器，
  序列化约慢 2~3 倍。把小块攒批再写入，实测没有改善；曾试过按记录尺寸在两条路径间
  自适应切换，已否决（见附录 E）。
- 局限：单个 `str` 值仍会一次产出，峰值下限是"最大单个字符串值"
  （真实轨迹实测最大块 ~2 MB，见 F.5.1）。

接口保持"写入文本文件对象"，因为执行期增量导出和各导出函数都传入文本流；
编码错误处理交给文件对象（见 3.5）。

### B.3 原子写：`AtomicTextOutput`

`nexus/cli/commands/export.py` 提供 `AtomicTextOutput`，以及在其之上的上下文管理器 `_atomic_text_output`：

| 行为 | 说明 |
| --- | --- |
| 写入位置 | `<真实目标>.partial`，与目标同目录，保证 `os.replace` 在同一文件系统内原子完成 |
| 软链接 | 先 `resolve()`，替换的是链接指向的真实文件，链接本身保留 |
| 权限 | 目标已存在时，发布前把原权限位 `chmod` 到新文件 |
| 目录目标 | 直接抛 `IsADirectoryError` |
| `commit()` | flush → fsync → close → chmod → `os.replace` → fsync 目录 |
| `abort()` | 只关闭文件，保留 `.partial` 供排查，目标不变 |
| `discard_existing=True` | 构造时就删除已有目标，等价于旧代码 `open('w')` 的截断语义 |
| 遗留的 `.partial` | 以 `'w'` 打开，会被直接覆盖 |

使用位置：`export_jsonl`、`export_csv`、`export_sharded` 的 jsonl/csv、`_export_sharded_results` 的 jsonl/csv，
以及 `run_batch` 的增量导出。

### B.4 执行期增量导出（`nexus run batch --export-format jsonl|csv`）

非分片 batch 边执行边写导出文件，文件在 `run_batch` 开头打开、在 `finally` 中关闭，因此不能套用 `with`：

1. 开始时以 `AtomicTextOutput(target, discard_existing=True)` 打开：删除旧目标，结果写入 `<target>.partial`。
2. 每个 run 结束后追加一行并 flush，与原行为一致。
3. 批次正常返回时 `commit()`，目标文件出现。
4. 其他任何情况（包括 `--fail-fast` 抛出的异常）都 `abort()`：目标路径不存在，`.partial` 保留，并输出 warning。

增量导出只包含本次实际执行的 run，resume 跳过的已成功 run 不会写入（既有语义，用户文档已说明）。
需要完整快照时，在批次结束后执行 `nexus export <db>`。

### B.5 孤立代理

- 本地所有文本写出统一使用 `encoding='utf-8', errors='backslashreplace'`：
  jsonl/csv 由 `AtomicTextOutput` 负责，`full` 格式的每个文件通过 `_write_text` 写出。
  `export_full` 改为复用 `export_run_to_full`，原来两份完全相同的逐 run 写出逻辑合并为一份。
- Cloud 的 `iter_json_bytes` 在字符串分支改为 `.encode('utf-8', 'backslashreplace')`。
  该函数也被 `distributed/artifacts/run_result.py` 使用；合法输入的字节不变，只影响原本会抛异常的输入。
- `backslashreplace` 把孤立代理写成 `\udXXX`，在 JSON 字符串中恰好是合法转义，`json.loads` 可还原出同一个字符串。
  例外：相邻的一对高、低代理（两个码点）会被 `json.loads` 合并成一个码点；这类输入本身已经损坏。
- 显式指定 UTF-8 后，在非 UTF-8 locale 下的输出会与旧代码不同（旧代码使用 locale 默认编码）。
  Linux 默认为 UTF-8 locale，此时没有差异。
- CSV 中的孤立代理会以字面量 `\ud83d` 出现，CSV 读取器不会还原。不再崩溃，但内容会有偏差。

## 附录 C：Cloud results 导出

### C.1 数据流

```text
参数校验 → _expand_executions / _ExportPlan
  → _single_record → _write_records
       AtomicJsonlOutput.write_records：写二进制 .partial，同时更新 sha256、索引并做语义检查
  → close_and_validate
       默认：_verify_integrity（不解析 JSON）
       strict 或索引已释放：_validate_jsonl_file（逐行解析，原实现）
  → partial failed/all 过滤
       默认：filter_records_by_index（按索引拷贝原始字节）
       strict 或索引已释放：filter_records（逐行解析，原实现）
  → 快照与 metadata 复核 → publish
```

选择规则、产物判定（`_has_artifacts`）、快照复核、`--force` 与发布语义与关联文档一致。
`_has_artifacts` 移到 `writer.py`，`results.py` 保留再导出。

### C.2 写入时记账

`.partial` 改为二进制流（`open('xb')`），直接写 `iter_json_bytes` 产出的字节，省掉每块的 decode/encode，
并保证 sha256 计算的就是落盘字节。每条记录写入时：

- 对写出的每个块和末尾的 `\n` 更新 sha256。
- 追加 `_IndexEntry(run_id, status, retry_count, offset, length, has_artifacts)`，其中 `length` 包含 `\n`。
- 用 `_check_record_semantics` 做与 `_validate_jsonl_file` 相同的逐行检查：字段集合与顺序、`extra` 类型、
  `run_id` 为非空字符串、`status` 为合法终态、`retry_count` 为非负整数。
  首个错误先记下，到 `close_and_validate` 时才抛出，错误文案保持不变（如 `unexpected JSONL fields at line N`），
  从而沿用"close 阶段失败"的既有契约。

行头比对依赖 `run_id` 是第一个序列化的键。`make_jsonl_record` 按 `_JSONL_FIELDS` 的顺序构造记录，
字段集合检查已经保证了顺序，`writer.py` 另有模块级断言 `_JSONL_FIELDS[0] == 'run_id'`。

### C.3 默认校验：`_verify_integrity`

对完成的 `.partial` 做一次 seek 检查加一次顺序读，不解析 JSON：

1. 索引条数等于 `expected_count`。
2. 每个条目的 offset 与上一条首尾相接（索引连续）。
3. 每行开头恰好是 `{"run_id": <id>`（按 `ensure_ascii=False` + `backslashreplace` 生成，与 `iter_json_bytes`
   逐字节一致，按精确长度读取，不受 run_id 长度影响），每行结尾恰好是 `}\n`。
4. `require_sorted` 时，用索引中的 `run_id` 检查严格递增。
5. 一次顺序读同时得到 sha256、文件大小和换行数，三者都必须与写入时一致。

**它能证明的：** 落盘字节与序列化器写出的字节完全一致；每个已索引的行都在预期位置，属于预期的 run；
文件中每条记录恰好一个换行。截断、字节翻转、插入、两行拼接、行互换、行重复、尾部追加或截断都会被检出。

**它不能证明的：** 序列化器的输出能被解析为 JSON。这一保证改为信任 `iter_json_bytes`。
需要独立重新解析每一行时，使用 strict（见 4.5）。因此，默认路径的语义检查与原实现等价，
防篡改能力更强，但不再独立校验序列化器本身。

### C.4 按索引过滤：`filter_records_by_index`

`keep` 接收 `_IndexEntry`，每条记录只求值一次。保留的行按 offset/length 以 1MB 块拷贝原始字节到同目录临时文件，
同时计算新的 sha256 并按紧凑布局重写 offset。完成后对临时文件跑 `_verify_integrity`，
再 `os.replace` 替换 `.partial` 并 fsync 目录，最后替换索引和摘要，因此可以连续多次过滤。
任务被取消时会等工作线程结束，失败时清理临时文件；`.partial` 与已发布文件保持不变。

### C.5 降级路径

| 触发条件 | 校验 | 过滤 |
| --- | --- | --- |
| 默认 | `_verify_integrity` | `filter_records_by_index` |
| `--strict-validate` 或 `NEXUS_EXPORT_STRICT_VALIDATE=1` | `_validate_jsonl_file` | `filter_records` |
| 写入时记录数超过 `_MAX_INDEX_ENTRIES`（100 万） | `_validate_jsonl_file` | `filter_records` |

- `--strict-validate` 在 `nexus export` 和 `python -m nexus.distributed.export` 两个入口都可用；
  环境变量在两个入口都生效。用于 SQLite 导出时报参数错误。
- 索引每条实测约 280 字节（`run_id`/`status` 来自 JSON 解析，字符串不共享），达到上限时约 280MB。
  一旦超过上限就在**写入期间**释放索引并输出 warning，不会把已经占用的内存留到 close 时才放弃。
  `AtomicJsonlOutput.index_available` 告诉调用方该走哪条过滤路径。
- `filter_records`（兼容路径）在改写文件**之前**就让索引失效；之后再调用 `filter_records_by_index` 会报错。
- 默认、strict、索引已释放三条路径发布的文件逐字节一致（有端到端测试）。

### C.6 读入与有界预取

Cloud results 的读入采用有限深度的磁盘预取：

1. `CloudApi.get_result_to_file()` 按块下载响应体到与输出文件同目录的
   `.prefetch-N` 文件，避免等待期间把下一条完整 body 放在 Python 堆中。
2. 消费记录时使用内存映射（`mmap` + `memoryview`），并优先用
   `orjson.loads(memoryview)` 解析；stdlib 兜底只用于 orjson 不接受的输入。
   用 mmap 而非 `read()`：解析器直接在文件后备页缓存上工作，body 不会作为
   独立堆拷贝再物化一次；这些页与磁盘内容一致、内核在内存紧张时可回收再按需
   重读，不构成真实的堆压力（对比：`read()` 会在堆里再复制一份完整 body）。
3. 同一时刻只消费一条结果；下一条可以在当前记录写盘时预取。
4. 配置 TDSQL 时优先使用 DB 结果；只有 DB 缺少 RunOutput 时，才按需使用同一
   磁盘 fallback，不为已有 DB 结果额外请求 Application API。
5. 取消、失败、磁盘写入错误和进程重启都会清理预取文件；预取文件使用 `0600`，
   并在输出锁持有后清理遗留文件。

该方案降低的是多条结果重叠和 Python bytes 副本；解析对象树和最大单条记录仍需驻留内存。
它不改变 JSONL 格式，也不改变 Cloud results 的快照、过滤和发布语义。

## 附录 D：行为变化

| 项 | 改动前 | 改动后 |
| --- | --- | --- |
| 本地 jsonl/csv 输出内容 | — | 逐字节不变 |
| 本地文本编码 | locale 默认编码 | 显式 UTF-8 + `backslashreplace` |
| 本地导出中途失败 | 目标路径留下截断文件 | 目标不变，保留 `<target>.partial` |
| 执行期增量导出运行中 | 目标路径实时增长 | 结果在 `<target>.partial` 中增长，旧目标在开始时被删除 |
| 执行期增量导出中断 / `--fail-fast` | 目标路径为本次的部分结果 | 目标路径不存在，部分结果在 `.partial`，并有 warning |
| 输出路径为软链接 | — | 更新链接指向的文件，链接与权限保留 |
| 含孤立代理的数据 | 导出失败 | 导出成功，写为 `\udXXX` 转义 |
| 本地 jsonl 导出耗时 | — | 约增加 60%（纯 Python `iterencode` 分块写出） |
| Cloud results 默认校验/过滤 | 逐行解析 | 不解析；可用 `--strict-validate` 恢复 |
| Cloud results 输出内容 | 文本模式写 `.partial` | 二进制写 `.partial`；Linux 上逐字节不变 |
| Cloud results 结果体解析 | stdlib `response.json()`，整条宽 str 进内存 | orjson 直接解析 UTF-8 字节；孤立代理/非 UTF-8 回退 stdlib，行为对齐 |
| Cloud results 取数并发行为 | 深度 2：两条记录全程在内存 | 深度 2 不变，但响应体流式落盘；任意时刻至多一条记录被消费 |
| Cloud results 峰值内存（中间验证） | 16.5 GB | 11.4 GB（−31%，orjson + 串行基准）；磁盘多占 ≤ 1 × 最大单条 |
| Cloud results 导出失败时的磁盘残留 | `.partial` | `.partial` + 至多一个 `.prefetch-N`（abort/open 时清理） |

依赖"边跑边读导出文件"的脚本需要改读 `<export-path>.partial`，应在 release notes 中说明。

### D.1 发布说明（可直接粘贴）

```markdown
### nexus export / run batch 导出行为变化

1. `nexus run batch --export-format jsonl|csv` 运行期间，结果写入
   `<export-path>.partial`，batch 正常结束后才替换为目标文件。
   依赖"边跑边读导出文件"的脚本请改读 `<export-path>.partial`。
2. 上述导出中断（失败、Ctrl-C）时目标路径不再保留本次的部分结果或
   上一次 batch 的旧文件；部分结果在 `<export-path>.partial` 中。
3. `nexus export <db> --format full` 改为带锁的目录 staging/swap：目录树在唯一
   staging 目录中构建，完成后切换到目标；中途失败时旧目标保持不变。
   普通 POSIX 文件系统切换期间目标路径可能短暂不存在。
4. 包含截断 token（孤立代理 `\udXXX`）的轨迹不再导致导出失败，
   会被写为合法 JSON 转义，可往返还原。
5. Cloud results 导出（`--content results`）内存峰值大幅下降；
   新增可选依赖 orjson（`--extra distributed`），未安装时自动回退。
```

## 附录 E：取舍记录

| 方案 | 否决原因 |
| --- | --- |
| 用"首字节、尾字节和行数"做结构校验替代解析 | 中间截断、字节翻转、插入垃圾、两行拼接、行互换、行重复 6 类损坏全部漏检；顺序检查读的是内存索引而不是文件 |
| 写入时立即抛出语义错误 | 破坏既有"close 阶段抛错、保留 `.partial`"的契约及其测试 |
| 删除 `filter_records` | 既有测试和潜在外部调用依赖它；保留为 strict 与超限时的兼容路径 |
| 到 close 时记录数超限再切 strict | 索引内存已经占用且后续过滤仍会使用，不省内存 |
| 引入 orjson（写出/本地路径） | 默认路径已不解析 JSON；strict 路径上反序列化最多只快 1.7 倍，且 tracemalloc 峰值高于 stdlib；`orjson.dumps` 无流式接口（整块 bytes），本地写出实测峰值反而更高（126 MB 对 44 MB）；`orjson.dumps` 会把 inf 写成 `null`。orjson 只用于 Cloud **读入**解析（见 C.6）：`loads` 直读 UTF-8 字节可跳过全量宽 str |
| 本地改用 `iter_json_bytes` | 需要改为二进制文件接口，而 `write_run_to_jsonl` 的调用方传入的都是文本流；`iterencode` 在保持接口的同时输出与 `json.dumps` 逐字节一致 |
| 本地把 `iterencode` 的小块攒批再写 | 实测没有提速 |
| 按记录尺寸自适应切换编码器（`sys.getsizeof` 树遍历估算 ≤ 16 MiB 走 C 编码器 `json.dumps`，否则 `iterencode`） | 曾实现后回退。估算阈值不保证临时字符串小于阈值（`getsizeof` 是 Python 对象足迹，不是序列化字节数）；共享引用会重复计入（方向保守但阈值语义模糊）；每条记录多一次整树递归遍历；同一写出器出现两条内存特征不同的路径，与"流式写出、峰值可控"的统一策略冲突。而收益（小记录恢复 C 编码器速度）从未实测。本地导出约慢 60% 是为峰值内存付出的已知代价，若未来需要提速应先实测再设计 |
| 周期对象导致估算递归溢出 | 不成立：`write_run_to_jsonl` 的输入必须是 JSON 可表示的（同一记录要以 JSON 形式入库、也要被 `json.loads` 读回），JSON 树天然无环；若真有环，`json.dumps`/入库本身早已失败 |

## 附录 F：验证结果

### F.1 单元测试

| 文件 | 覆盖 |
| --- | --- |
| `tests/unit/distributed/test_export_memory_fixes.py` | 与 `json.dumps` 逐字节一致；本地、`full` 格式、Cloud writer 的孤立代理；原子写（失败保留 `.partial`、软链接与权限、目录目标、`discard_existing`）；增量导出成功发布与 `fail_fast` 中断不残留旧文件；`iter_runs` 与 `get_all_runs` 等价、列子集、未知列、状态过滤、预取峰值（1 行对 64 行）、非法 `prefetch`；8 类损坏检出；中文/emoji/引号/超长 run_id 不误报；语义错误延迟到 close 且与 strict 文案一致；索引超限回退及超限后仍报语义错误；连续两次按索引过滤；兼容过滤使索引失效 |
| `tests/unit/distributed/test_cloud_multi_execution_export.py` | 默认、strict、索引已释放三条路径端到端发布结果逐字节一致；阶段钩子与失败注入改到 `filter_records_by_index` |
| `test_cloud_execution_jsonl_export.py`、`test_cloud_result_file_filter.py`、`drivers/test_drivers.py` | 未修改，作为回归锚点：close 阶段抛 `unexpected JSONL fields` 的契约、兼容过滤、`_validate_jsonl_file` |

对 5 项关键修复做了变异验证：分别回滚预取策略、软链接跟随、启动时删除旧目标、行头精确长度、`full` 格式的
`backslashreplace`，对应测试全部失败（5/5）。

```bash
uv run pytest tests/unit/distributed/test_export_memory_fixes.py \
  tests/unit/distributed/test_cloud_multi_execution_export.py -q
uv run pytest tests/unit/distributed tests/unit/runner/batch tests/unit/cli/test_export_cloud.py -q
```

### F.2 本地导出实测

数据：200 条 run，每条约 2.5MB 中文轨迹，SQLite 521MB。各版本在独立进程中运行，取 `ru_maxrss`；
基线版本用 `git worktree` 检出。

| 格式 | 改动前 | 改动后 | 输出 |
| --- | --- | --- | --- |
| jsonl | 984MB / 4.9s | 149MB / 7.7s | 逐字节一致 |
| csv | 983MB / 2.1s | 133MB / <0.1s | 逐字节一致 |

如果不调整预取（`iter_chunk_size` 保持 64），jsonl 峰值为 515MB。

### F.3 Cloud 校验阶段实测

3 条记录、每条约 48MB 且含一个 emoji 的 144MB 文件，测 `close_and_validate` 期间的 tracemalloc 峰值：

| 路径 | 峰值 | 耗时 |
| --- | --- | --- |
| strict（原实现） | 448.5MB | 1.2s |
| 默认 | 2.1MB | 0.3s |

### F.4 大单行场景：外部分析的复现

外部分析（CodeBuddy 评审记录）给出 976 MB 单行的内存叠加过程，并给出量化判断：
"行内出现一个 emoji，976 MB 变 3.85 GB 的 str"；"单行约 6 倍"；"两条 1.2~1.3 GB 的大行叠一起到 10 GB"；
"orjson.loads(bytes) 同一行峰值从 5.9 GB 降到 3.0 GB"；"不改代码按最大行 11~18 倍给 pod 配内存"。

按字节拼出真实的 976 MB 单行（`make_jsonl_record` 字段形状、合法可解析；构造方式为 1 MB 块循环写盘，
不在内存中构造大对象），在独立子进程中测 `ru_maxrss`：

| 场景 | 峰值 RSS | 说明 |
| --- | --- | --- |
| 旧实现 strict（`_validate_jsonl_file`），**纯 ASCII** 976 MB 行 | 2090 MB | ASCII str 紧凑 1 字节/字符，无宽度展开 |
| 旧实现 strict，**含 1 个 emoji** 的 976 MB 行 | **7959 MB** | 分析场景精确复现 |
| 默认（`_verify_integrity`），同一 emoji 行 | **132 MB**（等于基线，零增量） | 零解析：sha256 分块流式 + 行头 seek 比对 |
| `orjson.loads(bytes)`，同一 emoji 行 | 5906 MB | 与分析的"5.9 GB"一致 |
| `stdlib json.loads(bytes)`，同一 emoji 行 | 8829 MB | bytes、解码出的 str、解析对象三份叠加，比 strict 更差 |
| 旧实现 strict，两条 500 MB 行 | 2134 MB | 相邻行解析对象共存可见 |

逐条核对：

- **"一个 emoji 使整行按 4 字节/字符存放"：成立。** CPython `str` 按整串最宽字符定宽，实测
  纯 ASCII 1 字节/字符、BMP 中文 2 字节/字符、astral emoji 4 字节/字符。同一 976 MB 文件，
  行内有一个 emoji 时文本模式读入即为 3.9 GB 的 str；`json.loads` 解析出的 blob 值字符串同样
  含该 emoji，也是 3.9 GB——两者相加即 strict 峰值 ~8 GB（约为文件大小的 8 倍，分析的"约 6 倍"同量级，
  差异来自对象图构成）。
- **"上一行解析出的对象要等这一行解析完才释放"：成立。** 双行文件实测 2134 MB，
  为单行（500 MB）的约 2 倍，两条大行的对象在相邻两次 `json.loads` 之间共存。
- **"orjson.loads(bytes) 同一行峰值 5.9 GB"：精确复现（5906 MB）。** orjson 直接解析 UTF-8 字节，
  避开了 3.9 GB 的整行 str，比 stdlib 的两种方式都省；但解析出的对象里 blob 值字符串本身
  仍按 4 字节/字符存放（约 3.9 GB），加上输入 bytes（约 1 GB），**5 GB 是对象图的硬下限**，
  分析中"降到 3.0 GB"未复现（推测其测量行的字符构成不同）。
- **已落地的默认路径不解析 JSON**：校验期内存为进程基线（132 MB），优于包括 orjson 在内的
  所有解析式方案，且校验耗时 1.1 s 对 2.6 s（255 MB 文件实测，见 7.3）。
- **"不改代码：按最大行 11~18 倍配 pod"**：仅适用于未合入本设计的版本；本设计合入后，
  默认路径不再需要按行大小放大 pod。

复现方法：按字节拼出目标行写入文件（不在内存中构造大对象），各校验方式在独立子进程中运行，
取 `ru_maxrss`；行内容为 975 MB ASCII 字符加 1 个 emoji（4 字节展开的触发条件）。

### F.5 生产数据端到端实测：写入阶段并发（诊断记录）

对一条经授权的**生产样本 batch**（10 条子 run，产物约 **15 GB** JSONL）
完成端到端实测；不在文档中记录真实 Execution 标识。

> 第一轮（MR-1/2/3）在此负载的测量值（16.5 GB / 19:58，对 main 13.4 GB / 14:11）
> **已过时、不代表当前分支**，明细不再保留；当前分支各阶段实测见附录 D 与 F.5.1。

保留价值的诊断结论：**峰值的主导项是写入阶段并发**——深度 2 流水线使两条 GB 级
记录的「HTTP 响应字节 + 解析对象图 + 编码分块」同时驻留内存，而校验阶段已零解析。
该结论是第二轮读入侧优化（orjson + 磁盘预取，见 C.6）的立项依据，本节仅保留结论，
测量明细已被后续实测取代。

量化依据：峰值 ≈ 并发深度 × 最大记录的对象图体积。记录为 KB 级时深度 2 无害；
GB 级记录时它是线性放大器。实际采用的修法是**磁盘预取双槽**（读入侧 orjson +
下一条响应体落盘，见 C.6）；备选方案（batch item `size_bytes` 预估超阈值自动降
并发、大字段外置 sidecar、全局在飞字节预算）保留为磁盘不可用或需进一步压峰时的
候选项。

同批其余 3 条 execution 实测中断时 partial 已达 7.4 GB，规模同级。

### F.5.1 Cloud 读入优化的单条生产 A/B（2026-10-13）

经授权的匿名生产样本：batch 内 1 条 succeeded run（产物 478 MB），
两个版本各自独立进程、`/usr/bin/time -v`、同命令同参数：

| 版本 | 峰值 RSS | 耗时 | 输出 |
| --- | --- | --- | --- |
| main @ `9c23580b74` | 3.62 GB | 5:37 | 基准 |
| 当前工作区（基础导出优化 + orjson 读入 + 磁盘预取） | **1.68 GB（−54%）** | **0:28（12×）** | 与基准 `cmp` 逐字节一致 |

该结果只能说明当前工作区整体相对 main 的差异，不能把 54% 或 12× 单独归因于
磁盘预取；两边同时包含了基础导出、校验和序列化路径差异。

执行备注：该匿名样本的其余 item 结果接口在测试时段服务端不可服务
（curl 直连同样 60 秒零字节、间歇 502，与客户端实现无关），另一个约 2.3 GB
记录的宽 str 主导场景因此未在本次 A/B 中复测。

### F.5.2 Cloud 读入优化的全量 batch A/B（2026-10-13）

另一个经授权的匿名生产 batch：10 条 item（1 succeeded / 5 failed / 4 lost），
产物 **21,687,295,642 B（约 21.7 GB）**，内含约 8 GB 的单条记录——比 F.5 的
样本更重。两个版本各自独立进程、`/usr/bin/time -v`、同命令同参数：

| 版本 | 结果 | 耗时 | 峰值 RSS |
| --- | --- | --- | --- |
| 当前工作区（基础导出优化 + orjson 读入 + 磁盘预取） | ✅ 完成，10 条，产物 21.7 GB，写时校验通过并发布 | 19:46 | 18.55 GB（含 ~8 GB 可回收的 mmap 文件页缓存，真实约 10~11 GB） |
| main @ `9c23580b74` | ❌ **被内核 OOM 杀死（signal 9）**，未完成：partial 停在 20.7 GB，校验/发布均未执行 | 15:05（被杀时） | 28.24 GB（≈机器物理内存） |

**结论：这个量级的真实生产 batch，main 在 28 GB 内存的机器上无法完成导出；
当前分支可完整完成。** 该 batch 无法与 main 做产物字节对比（main 无可用完整产物）；
字节一致性由 F.5 的 16 GB 链条、F.5.1 的单条 A/B 与单元测试共同覆盖。

峰值 RSS 的口径说明：18.55 GB 是 `maxrss` 高水位（宽口径），**其中约 8 GB 是
可回收的**——预取 body 的 mmap 文件后备页，内容与磁盘一致、无独立副本，内存
紧张时内核可直接丢弃再按需重读。真实不可回收的匿名内存（解析对象树 + 瞬态）
约 **10~11 GB**：即"实际吃掉、挤不掉"的内存底线是 10~11 GB，而不是表里的
18.55 GB。main 被杀的直接原因：深度 2 需要同时驻留两条该量级记录，仅匿名部分
就越过了物理内存；而单条驻留的本分支，即便算上宽口径也仍在物理内存之内。

### F.5.3 生产全量验收与 transcoder 分块 bug（2026-10-14）

结果管线合入后按生产真实 runid 重新验收，发现并修复一个内容损坏 bug。

**Bug 与修复。** `_StreamingRow.iter_chunks` 的 member/extra 拷贝对每个
`_RESULT_SCAN_CHUNK_BYTES`（1 MB）块重新创建 `iter_spaced_json`，丢弃了
转写器按设计跨块存续的状态机：块边界落在字符串值内部时，下一块从
"字符串外"状态恢复，字符串内容中的每个 `:` / `,` 被当作结构分隔符补一个
空格——data URL、Windows 路径和嵌入代码被静默损坏。产出仍是合法 JSON，
写时校验（sha256/字节记账）与 `--strict-validate`（语法解析）全部通过：
**"合法 JSON、内容错误"的损坏语法校验不可能发现，只能依赖与基准的逐字节
对比**。修复为整个成员 span 只经过一个转写器，并新增缩小块尺寸、强制边界
命中字符串的端到端回归测试（`test_transcode_member_chunks_keep_transcoder_
state_across_boundaries`，旧代码 FAIL / 修复后 PASS）。

**生产验收。** 生产 batch（10 条 item，产物 21.7 GB，含约 8 GB 单条记录），
main 基线与修复后本分支按 item 单独导出并 `cmp`：

- 可比的 9 条（673 MB ~ 3.0 GB）**逐字节一致**；
- 1 条（约 4.0 GB）main 被内核 OOM 杀死（signal 9，RSS 26.4 GB）无产物可比；
  本分支同 item 峰值 RSS **302 MB**、2:55 完成。

内存规律：main 的 RSS ≈ item 体积 × 5（1.8 GB item → 9.0 GB），
本分支全程 ≤ 310 MB，与 F.5.2 的结论一致。修复前代码的 21.7 GB 全量 run
（14:18、峰值 RSS 475 MB）验证了吞吐与内存形态，但其输出含本 bug 损坏，
不作为内容基准；修复不影响内存形态（仅合并转写器实例），全量文件的
逐字节对比由上述按 item 对比覆盖。

### F.6 待验证

- 非 UTF-8 locale 环境下的本地导出差异。
- `--allow-partial` 过滤与 `--strict-validate` 在生产数据上的组合行为。
- 磁盘预取终态的逐字节对比：因 main 无法完成 21.7 GB batch 导出，
  由 F.5 的 16 GB 链条、F.5.1 的单条 A/B 与 F.5.3 的按 item 全量对比
  共同覆盖（原"ARENA 样本依赖结果服务恢复"一项已随服务恢复完成，
  见 F.5.3）。

## 附录 G：风险与遗留

| 项 | 说明 |
| --- | --- |
| 默认校验不再独立解析 JSON | 依赖 `iter_json_bytes` 的正确性；保留 `--strict-validate` 和环境变量作为逃生舱 |
| 增量导出语义变化 | 运行中的结果在 `.partial`，中断时目标路径不存在；需要 release notes |
| 本地 jsonl 导出变慢 | 约增加 60%，以峰值内存换序列化速度 |
| `full` 目录发布 | 采用唯一 staging 目录、输出锁和崩溃恢复；目录 swap 期间可能有短暂名称间隙，不承诺单操作原子替换 |
| 单个超大字符串值 | 本地 `iterencode` 会一次性物化，需要大字段外置 |
| ~~`.partial` 残留惩罚重试（设计缺口）~~ 已解决 | 已按改进方向落地（`6b29d96`）：`.partial` 加 `flock`，writer 存活=持锁（拒绝并提示持有者 pid），进程死亡=内核自动释放、重试自动接管；`--force` 为显式覆盖 |
| ~~`run_batch` 的调度与汇总~~ 已解决 | 三处全列读取改为 `iter_runs(fields=...)` 流式：分片汇总与结束汇总只读 `run_id`，resume/retry 调度只读 `run_id/status/config/metadata`；大输出 batch 启动不再把输出大字段拉进内存 |
| 检查器 TUI | `inspector` 一次加载全部 run，属于交互需要，暂不调整 |

## 附录 H：代码位置

| 文件 | 职责 |
| --- | --- |
| `nexus/runner/batch/database.py` | `iter_runs`（列白名单、预取策略）；`get_all_runs` 薄封装 |
| `nexus/cli/commands/export.py` | `AtomicTextOutput`、`_atomic_text_output`、`_write_text`；`write_run_to_jsonl` 分块写出；jsonl/csv/full 与分片导出；`--strict-validate` 参数及本地模式拒绝 |
| `nexus/runner/batch/runner.py` | `_export_sharded_results` 与 `run_batch` 增量导出的原子发布 |
| `nexus/distributed/export/writer.py` | `_IndexEntry`、`_check_record_semantics`、`_verify_integrity`、`_has_artifacts`；`AtomicJsonlOutput` 的二进制 partial、写时记账、索引上限、`filter_records_by_index`、`index_available`；Cloud 读入的 `ResultSpillFile`/`ResultSpillPool` |
| `nexus/distributed/export/results.py` | `export_execution(strict_validate=...)` 与过滤路径选择；Cloud 读入的 `_RecordPlan`、`_write_records` 双槽调度、`_prefetched_result`、`_single_record/_read_result` 的 `prefetched` 参数 |
| `nexus/distributed/export/cli.py` | `--strict-validate` 与 `NEXUS_EXPORT_STRICT_VALIDATE` |
| `nexus/distributed/export/client.py` | `_loads_json_bytes`（orjson 读入 + stdlib 兜底）、`get_result_to_file`（流式下载落盘） |
| `nexus/utils/json_stream.py` | `iter_json_bytes` 字符串编码改为 `backslashreplace`（合入结果管线后叠加其 `encode_basestring`/标量快路径） |
| `nexus/utils/json_scan.py` | `JsonFieldScanner`（结果管线，服务端流式扫描）；`object_members`（按成员字节区间解析的流式读取工具，超限回退整体解析） |
| `tests/unit/distributed/test_export_spill.py` | Cloud 读入专项：双槽调度、RAM 互斥、清理、失败路径、流式下载重试语义、空闲读超时（含旧式缓冲服务端慢响应） |
| `nexus/utils/json_transcode.py` | `iter_spaced_json`：compact → spaced 的无解析字节转写，状态机跨块存续（F.5.3 bug 与修复所在路径） |


## 附录 I：结果下载超时与旧服务端兼容（2026-10-14）

`codex/application-gateway-result-pipeline` 分支将远端 `/v1/executions/{id}/result`
由“整体解码后一次性响应”改为流式下发（设计文档随该分支维护）。本分支的客户端适配
只涉及超时形态，且不依赖该改造——以下设计对两种服务端形态均成立：

### 问题

`nexus export` 的 HTTP session 使用 `ClientTimeout(total=300, connect=10)`，而 aiohttp 的
`total` 覆盖“发起请求到读完整个响应体”全程。生产实测（1.8.18，缓冲式服务端）：
单条 4.06GB 结果服务端组装静默 109s~10min，传输 156s，三次尝试全部越过 300s 死线后失败；
同类 2.6GB 结果总计 197s 勉强通过。总时限对大结果就是不成立的约束。

### 方案

`get_result_to_file` 与 `get_result` 改为按请求覆盖超时：
`ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=900)`。

- `total=None`：不再限制整次下载的墙钟时间；
- `sock_read=900`：限制**每次空闲读间隔**（实测确认它同时覆盖等待响应头的阶段）；
  默认 900s 高于观察到的旧服务端最长组装静默（约 10 分钟），又能兜住真正的挂死连接；
- 可用 `NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS` 覆盖（非法值回退默认，`0` 收敛为 1s）；
- 其余端点（批次摘要、items 分页、control-events）payload 小，维持 session 默认 `total=300`。

### 与旧服务端的兼容

| 维度 | 旧（缓冲式）服务端 | 新（流式）服务端 | 客户端 |
| --- | --- | --- | --- |
| 传输 | 整体组装后带 `Content-Length` 发送 | chunked 分块 | 两种都按块读取落盘，无差别 |
| 响应字节 | `json.dumps` 默认格式（ensure_ascii、`, `/`: `） | artifact 中 `output` 字段的紧凑原始字节 | 解析结果语义一致（测试锁定两种格式） |
| 首字节延迟 | 等于组装耗时（可达 10 分钟） | 约为 artifact 下载耗时 | `sock_read` 需覆盖旧服务端静默期（默认 900s） |
| 中途失败 | 不存在（全有或全无） | 可能 200 后中断 | 连接错误按可重试处理，重试前清空 spill 文件（既有语义） |

`get_result`（Trace-Database / embedder 回退路径）与 `get_result_to_file` 使用同一超时；
`results.py` 的 `supports_result_stream` 探测保持不变，未实现 `get_result_to_file` 的
API 封装继续走 `get_result`。

### 明确取舍

- 未把 `JsonFieldScanner` 用于客户端解析：落盘 + mmap + orjson 读入的峰值已接近
  “单条记录对象图”下限；字段级增量解析在“单个成员占绝对大头”的轨迹数据上收益为零
  （成员解析的瞬时峰值与整体解析相同），反而丢失 orjson 的解码速度。
  `object_members` 作为无 orjson 环境的兜底工具保留在 `nexus/utils/json_scan.py` 并有单测。
- `total=None` 意味着“每 899s 滴一个字节”的病端不会被掐断；由 `sock_read` 之外的手段
  （CLI 层 Ctrl-C、运维超时）兜底，与结果管线自身 `STREAM_CLIENT_TIMEOUT` 的形态一致。
