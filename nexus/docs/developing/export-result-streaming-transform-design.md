# Cloud results 导出流式转换设计（方案 D）

- 更新日期：2026-10-14。
- 前置：[导出内存与完整性设计](export-memory-fix-design.md)（落盘预取、免解析校验）、
  远端 `/v1/executions/{id}/result` 的流式下发改造位于
  `codex/application-gateway-result-pipeline` 分支（未随本分支，设计文档随该分支维护）。
- 状态：已实施（`perf/export-memory` 分支；**默认开启**，`NEXUS_EXPORT_RESULT_TRANSCODE=0` 关闭；两条路径输出逐字节一致）。

## 1. 问题与目标

Cloud results 导出链路在远端流式化与客户端落盘预取之后，仍剩最后一个大内存点：
**消费阶段**把整条结果 `orjson.loads` 成完整 Python 对象图再重新序列化。实测（生产批次
`exe_2211b079…`，输出 21.7GB、10 条，最大单条约 15-20GB）：进程峰值 RSS 26.6GB，
其中约 15-20GB 是可回收的 mmap 文件页，**真实堆压力 ≈ 最大单条的对象图（~1×）**。
单条结果超过机器内存余量即 OOM，且整个导出失败。

目标：

1. 消费阶段峰值内存降到 MB 级（转码缓冲），与单条结果大小彻底无关；
2. 输出与现有实现**逐字节一致**（含 `compare_cloud_results` 的字节对账）；
3. 云端零改动：不改 run_result 编码格式、不需要任何组件发版；
4. 校验语义与现状对齐，NaN/Infinity 显式报错。

非目标：本地 SQLite 导出（已是流式）；Trace-Database/TDSQL 路径与 embedder 回退路径
（结果来自 DB 驱动的内存对象，无 HTTP body，维持现有 dict 组装）；`--content traces`。

## 2. 为什么可行：body 与输出行只差分隔符

输出行的成员值和 body 的成员值由**同一个编码器家族**（`nexus/utils/json_stream.iter_json_bytes`）
产生，逐维度对比：

| 维度 | body（run_result artifact） | 输出行 | 差异 |
| --- | --- | --- | --- |
| 序列化器 | `iter_json_bytes`，separators `(',', ':')` | `iter_json_bytes`，separators `(', ', ': ')` | **仅分隔符** |
| 字符串转义 | `encode_basestring`（+孤立代理 `backslashreplace`） | 同一函数作用于解码后的值，产出相同转义 | 无 |
| 数字格式 | `int.__repr__` / `json.dumps(float)` | 同 | 无 |
| 键顺序 | `sort_keys=False` 保序 | 同 | 无 |
| 非 ASCII | 原始 UTF-8（`ensure_ascii=False`） | 同 | 无 |

（历史佐证：`RUN_RESULT_VERSION` 版本门禁保证不存在更老的编码形态；
`ensure_ascii` 从未传过 True；分隔符从未变过。）

因此：**对 body 成员字节做“紧凑 → 宽松”的分隔符转码，结果与
“orjson 解析 → 重新序列化”逐字节相同**，而前者从不构建 Python 对象。
这是方案 D 的正确性核心，由等价性模糊测试锚定（第 7 节）。

## 3. 架构

替换 `_write_records` 双槽流水线的**消费端**；下载侧（`get_result_to_file` → spill 文件、
流式超时）与预取调度完全不变。

```text
现状：spill 文件 → mmap → orjson.loads 整体解析 → output dict
      → make_jsonl_record 组装 dict → iter_json_bytes 整体重序列化 → 写行
方案 D：spill 文件 → 第一遍：扫描（校验 + span + 判空 + NaN 检测）
                     → 第二遍：按行字段顺序 seek 各成员 span，转码拷贝进输出行
内存：扫描缓冲 + 拷贝缓冲 ≈ 数 MB，与单条大小无关
```

选择两遍而非单遍的原因：输出行字段顺序（`_JSONL_FIELDS`）与 body 成员顺序（RunOutput
的 pydantic 序列化序）解耦，不假设顺序；spill 是顺序读文件，多一遍扫描在页缓存下代价
是秒级，换确定性。

### 3.1 第一遍：扫描与校验

复用并扩展 `JsonFieldScanner`（`nexus/utils/json_scan.py`），对 spill 文件顺序扫描一遍：

- 严格 JSON 语法校验（现有能力）；
- 记录根层成员的字节区间（现有能力）；
- 校验 `success` 为 bool（现有能力，根层小标量捕获）；
- **新增**：字符串成员只记长度不捕获内容（现实现捕获根层字符串、超 4KiB 报错——
  改为长度计数无上限，判空/判大只需长度）；
- **新增**：容器判空（立即闭合 = 空），供 `_has_artifacts` 等价判定；
- **新增**：`NaN`/`Infinity`/`-Infinity` token 显式拒绝（现在允许通过）；
- **新增**：成对代理转义、嵌套超限、数字/键超限、重复根层成员 → 标记该条**回退**
  （扫描器与 `json.loads` 语义分歧的输入，见下方分类与第 4 节）。

第一遍的错误分两类，必须在扫描期就分流——**第二遍是流式写行，行字节一旦开始写出就无法
回退，任何“该条需要换路径”的判定都必须在第二遍开始前完成**：

- **硬错误（整个导出中止）**：JSON 语法错误；`NaN`/`Infinity` token；`success` 非 bool；
  succeeded 但 success=false。`ExportError` 携带 execution id 与成员路径——比现状
  （写行阶段才失败）更早、更可定位。
- **记录级回退（该条改走 orjson 整体解析旧路径，输出仍逐字节一致）**：扫描器与
  `json.loads` 语义存在分歧的输入——成对代理转义（stdlib 会合并码点后重编码）、
  嵌套超限、数字/键超过捕获上限（`json.loads` 可处理超大 int/长键）、重复根层成员
  （`json.loads` 为后值覆盖）。判定即停止扫描并打标记，第二遍对该条走旧 dict 组装路径；
  这类输入极端罕见，回退的内存代价可忽略。

### 3.2 第二遍：按行序转码组装

按 `_JSONL_FIELDS` 顺序发出行（`make_jsonl_record` 语义的流式等价）：

| 行字段 | 来源 |
| --- | --- |
| `run_id`/`status`/`start_time`/`end_time`/`retry_count` | plan 元数据，经 `iter_json_bytes` 序列化（构造期校验 run_id 非空、retry_count 非负 int、status 合法终态——等价于 `make_jsonl_record` 的前置检查） |
| `error` | 取值决策（扫描期判定）：body `error` 成员缺失或为 `null` → 非 succeeded 时写 fallback_error（succeeded 写 `null`）；其余（字符串/对象/标量）→ **span 转码拷贝**，不解析、无大小限制——字符串内部没有结构性分隔符，对象由转码器统一补空格 |
| `task_output`/`agent_output`/`processor_results` | body 成员 span **转码拷贝**（紧凑 → 宽松，1MB 块）；成员缺失 → 写 `null` |
| `metadata` | plan 元数据经 `_project_cache_task_ids` 投影后序列化。投影需要 body 中 `input_cache_task_id(s)`/`cache_task_id(s)` 四个成员的**解析值**（类型校验要求真实 dict/str）——这些是小成员，span ≤1MiB 上限内用 `json.loads` 解析；超限 → `ExportError` |
| `extra`（可选） | body `extra` 成员真值判定后，span 转码拷贝，追加在 `metadata` 之后（现有行为） |

转码拷贝器（新模块 `nexus/utils/json_transcode.py`）：字节级状态机，跟踪
字符串/转义/`\uXXXX` 跨块边界；结构性位置（字符串外）的 `,` 和 `:` 补空格，其余字节
原样透传。**与第一遍独立**：第二遍只依赖第一遍产出的 span 与回退标记，自身仅跟踪
字符串状态（最简状态机）——两个状态机越简单越好，各自的正确性都由模糊测试锚定。

### 3.3 写时记账适配

`AtomicJsonlOutput` 新增流式行写入 API（如 `write_row_segments(segments, *, run_id,
status, retry_count, has_artifacts)`）：segments 为“字面量字节”与“（文件， start， end）
转码拷贝”的序列；writer 内部统一 write/digest/offset 计数、行尾 `\n`、索引追加——与
现有 `write_records` 完全同一套记账。写时语义检查（`_check_record_semantics`）的字段
集合与首键断言在流式路径由**构造保证**（按 `_JSONL_FIELDS` 顺序发射）；dict 路径
（Trace-Database/embedder 回退）保持不变。

`has_artifacts`、`strict` 回退、`filter_records_by_index`、`--strict-validate` 全部
不受影响（都作用在写出的字节上）。

## 4. 语义对齐清单（旧 → 新逐项）

| 语义点 | 现状 | 方案 D |
| --- | --- | --- |
| `success` 非 bool | `ExportError('invalid RunOutput success field: X')` | 扫描期同文案报错 |
| succeeded 且 success=false | `ExportError` | 同 |
| output 缺失（404）且 succeeded | `ExportError('successful execution has no RunOutput')` | 同（无 body，走既有 None 路径） |
| `error` 取值 | succeeded → body error；否则 body error 非 None → body error，否则 fallback | 同（决策依据为扫描期的空/None 判定） |
| 缺失大成员 | `output.get()` → None → 行内 `null` | 写 `null` |
| `_has_artifacts` | None/`''`/`[]`/`{}` 为无产物；`0`/`False`/非空容器/非空串为产物 | 扫描判空等价（结构判空 + 标量类型），单测逐例对齐 |
| 缓存 ID 投影 | 四成员解析值 + 类型校验后并入 `metadata._nexus` | 同（小成员解析，1MiB 上限） |
| `extra` 追加 | `output.get('extra')` 真值时追加 | 真值判定 + span 拷贝 |
| NaN/Infinity | 写行阶段 `ExportError('JSONL record is not serializable')`，整个导出中止 | 扫描期 `ExportError`（带 execution id 与成员路径），整个导出中止——**刻意不对齐“报错时机”，比现状更早** |
| 相邻代理对转义（`\udXXX\udYYY` 成对出现） | stdlib 解析合并为一个码点后原样 UTF-8 重编码 | 转码按字节透传，字节不同 → **转码器检测到成对代理转义即回退 orjson 路径**（极罕见：要求源字符串含两个相邻孤立代理） |
| >4KiB 的数字/键、>1000 层嵌套 | orjson/stdlib 按能力解析 | 扫描期标记回退 → 该条走 orjson 路径（对齐旧行为） |
| body 根层重复成员 | `json.loads` 后值覆盖 | 扫描器报 duplicate → **该条回退 orjson 路径**（保持后值覆盖语义） |
| 缓存投影类型校验失败（metadata/`_nexus` 非 Mapping、cache 成员类型不符） | 告警 + 原样 metadata，不投影 | 同（解析小成员后逐条等价校验） |

## 5. 兼容性

- 输出行与现状**逐字节一致**：小字段同一序列化器；大成员转码后与“解析重序列化”逐字节
  相同（第 2 节 + 模糊测试锚定）。`compare_cloud_results` 两种模式均通过。
- 不改 run_result 编码（方案 A 弃用）、不需要格式探测与回退路径分支——所有 artifact
  永远是紧凑格式，一律转码。
- 不改云端任何组件；新旧服务端（缓冲式/流式）均可工作。
- 结果文件格式、`.partial`/lock/预取文件协议、发布语义不变。

## 6. 性能

| 项 | 现状（spill + mmap + orjson） | 方案 D |
| --- | --- | --- |
| 下载 | chunk → spill，≈0 | 不变 |
| 解析 | orjson 整体（最快，~1× 峰值） | 扫描一遍 + 转码一遍（纯 Python + 正则快路径，吞吐待实测） |
| 峰值内存 | 最大单条对象图（~1×） | 数 MB |

预期代价：每条多一遍顺序读 + 转码 CPU，21.7GB 批次的导出总时长预计增加数分钟。
用昨天实测的巨无霸行（~15-20GB）测转码吞吐（MB/s），若低于或json 路径 5 倍以上
再评估优化（字符串段已是 C 正则跳过，主要开销在结构 token 的 Python 循环）。

## 7. 测试计划

1. **等价性模糊测试（正确性锚点）**：随机嵌套结构（深度/宽度/unicode/转义/代理/
   数字形态含指数与 -0.0/空容器/巨 int），断言
   `转码(body) == iter_json_bytes(_loads_json_bytes(body), (', ', ': '))` 逐字节相等；
   NaN 样本断言报错。种子固定可复现，样本量 ≥1e5。
2. **黄金实测**：昨天 21.7GB 输出中的巨无霸行 → 构造紧凑 body → 转码 → 与原行逐字节
   对比（标记 slow，单独触发）。
3. **端到端**：spill 管线下新旧消费路径输出逐字节一致；NaN body → ExportError；
   缺成员 → null；404 → 占位行；100MB body 的 tracemalloc 峰值 < 50MB。
4. **语义对齐清单**：第 4 节表格逐项单测。
5. **回归**：现有 distributed/utils/runner/cli 全量；ruff --preview 与 ty 全绿。

## 8. 灰度与验收

- 环境开关 `NEXUS_EXPORT_RESULT_TRANSCODE`：**默认开启**（两条路径输出逐字节一致，
  回退仅是机制回退）；生产重导 `exe_2211b079…` 与 21.7GB 基线做 md5 对账；
  异常时置 `NEXUS_EXPORT_RESULT_TRANSCODE=0` 一键回退旧解析路径。
- 验收标准：同批次重导 md5 一致；导出期间峰值 RSS < 200MB（对照 26.6GB）；
  巨无霸行不再出现 GB 级堆分配。

## 9. 风险

| 风险 | 缓解 |
| --- | --- |
| 转码状态机错误产出损坏 JSON | 模糊等价测试 + 黄金实测 + 灰度开关；发现不一致立即回退环境变量 |
| 转码吞吐不足 | 先实测再实施；吞吐不足时的降级方案是保留现状并维持 ~1× 边界（不会更差） |
| scanner 扩展引入回归 | 扩展点独立成参数/标志，现有服务端路径（结果管线）行为由其既有测试锁定 |
| 语义对齐遗漏 | 第 4 节清单驱动开发，逐项写测试后才算完成 |
| 与磁盘预取的交互 | 仅替换消费端；双槽调度、spill 生命周期、flock、重试语义零改动 |
| 第一遍/第二遍状态机分歧导致转码字节与判定不一致 | 两遍只通过 span 与回退标记耦合；第二遍状态机仅跟踪字符串状态；模糊测试同时覆盖两个状态机的等价性 |

---

## 附录 II：① tee 与 ② flat 快路径恢复（2026-10-14 实施）

第 3 节的两遍设计在实测（3.72GiB 真实行：扫描 56.9 MiB/s）后做了两项优化：

### ① 扫描并入下载（tee）

`_ScanSlot`：下载 chunk 经 8×1MB 有界队列喂给后台扫描线程（反压式，下载持续快过
扫描时被限速到扫描速率，内存恒定 ~9MB）。下载完成时校验与 span 即绪，**消费阶段
从两遍变一遍**（省掉一遍全量重读）。语义：

- `JsonFallbackNeeded`（嵌套/大小超限、重复键、代理对）→ 标记该条 fallback → 旧
  orjson 路径，输出不变；
- NaN/Infinity、语法错误 → 硬错误，在下一个 `feed` 调用点中止下载（确定性错误，
  重试无意义）；
- 下载重试时 `reset()` 以 sentinel+join 干净重建扫描器；404 走 `abort()`。

扫描线程与转码同受 GIL 约束：聚合扫描吞吐 ≈ 单核（实测 ~169 MiB/s，见 ②），
N 路并行下载聚合超过它时消费端成为限速方——总 CPU 量不变（校验本就要做），
墙钟 = max(网络, CPU)。

### ② flat 快路径恢复 + 精确后验

原先客户端 flag 会禁用 flat 正则快路径（怕绕过 NaN/代理对检查）。改为：

- `_FLAT_OBJECT_NO_NAN`：严格模式正则，把"值不含 NaN/Infinity"编码进匹配本身；
- 代理对无需检查：平坦对象的字符串不含反斜杠（`_SIMPLE_STRING` 排除），逃逸类
  转义不可能出现；含转义的对象匹配失败自然落入慢路径的逐转义检测。

实测 203MiB 合成数据：**169.4 MiB/s**（客户端全 flag），较恢复前真实行上测得的
56.9 MiB/s 提升约 3×，已高于单流下载的 ~25 MiB/s。

### ③ 并发深度可配置

`NEXUS_EXPORT_RESULT_CONCURRENCY`（默认 **4**，范围 1~8，非法回退）。依据：
实测下载单流 ~25 MiB/s、单槽消费 ~69 MiB/s，稳态要求 `(N-1)×25 ≥ 69` → N=4；
服务端准入 `artifact_run_result_max_inflight=4` 正好容纳；磁盘峰值 ≈ N × 最大
单条。**范围仅限 spill 管线**——in-RAM 回退管线（Trace-Database/embedder，内存
线性于 N）固定保持旧深度 2。

---

## 附录 III：TDSQL-fallback 管线接入（2026-10-15，修复生产 OOM 遗漏）

生产导出流程配置 TDSQL 时 `trace_reader` 非 None，原实现把流式转换 gate 在
`plan.trace_reader is None` 上——导致 TDSQL-fallback 管线整段走旧内存模型：
TDSQL 丢行的记录经 API fallback 下载后 **orjson 整体解析**，多 GB 记录的对象图
（实测 5~7× 膨胀）撞上 18GB cgroup 限额，5 个分片 4 个 exit 137。

### 修复

- `_read_result` 增加 `scan_factory`/`scan_out`：TDSQL 先读，**缺失时 API body
  下载期间完成扫描（tee），跳过 orjson 解析**，以 `_STREAMING_PREPARED` 标记 +
  prepared/path 交接给 `_single_record` → `_streaming_record`（复用其 error
  决策、缓存投影、enrichment；control-events 时段复用 `_read_result` 已取的
  值，避免二次请求）。
- `_ScanSlot.finish()` 补 `scanner.finish()`：截断 body 在扫描层显式报错
  （与旧 `json.loads` 截断失败对齐；aiohttp 的 ClientPayloadError 仍是传输层
  第一道防线）。
- `writer.download`：404（length None）时 `scan.abort()` 并保持
  missing-row 旧语义。
- `_write_records` in-RAM 管线消费循环支持 `_StreamingRow`（write_row_chunks +
  discard）；in-RAM 并发保持 2（TDSQL 直接命中的记录来自 DB dict，对象图不可避免）。
- 超限缓存 ID 成员：`_NeedsLegacyPath` → 记录级回退（对齐旧行为的任意大小投影）。

### 验证

新增 `test_tdsql_fallback_streaming_matches_dict_path_byte_for_byte`（TDSQL 缺失
→ 流式 vs dict 路径逐字节一致）、`test_tdsql_hit_streams_from_dict_path_unchanged`
（TDSQL 命中 → dict 路径不受影响）；相关 5 套件 382 passed；ruff（--preview）与
ty 全仓 0 诊断。

---

## 附录 IV：#5 TDSQL event_data 字节级提取（实施细目，待实施）

目标：TDSQL-hit 记录内存归零。现状 `_read_root_output` 对命中的 event_data 文本
做 `json.loads(整个文档)`（`_decode_event_data`），event_data 含整条轨迹，
对象图常大于 run_result body 本身。

### 已确认事实（2026-10-15）

- 写入端：`json.dumps(record.event_data, default=str)` —— **宽松分隔符**、
  **ensure_ascii=True**（非 ASCII 全为 `\uXXXX` 转义）、`allow_nan=True`；
- 选中语义：span 记录行按 `key=(time_unix_nano, span_id, record_id)` 取
  **最新**的一条含 `run.output` 的 event_data；文档内部取**最后一个**
  `name=run.event && attributes.event_type=run.output && attributes.event_data
  是对象` 的条目的 `output`（非 dict → None，与旧一致）；
- `output` 的 JSON 结构与 API body 的 `output` 字段完全一致（同构 RunOutput）。

### 实施清单

1. **转码器规范化模式**（`iter_spaced_json` 增加参数或独立函数）：
   宽松分隔符 → spaced（`,\s*`/`:\s*` 字符串外规范化为恰好一个空格）；
   `\uXXXX` → UTF-8 解码（通用 BMP + 代理对合并，后者已实现）；
   fuzz 语料加宽松/ensure_ascii 形态，参照恒为 `json.loads` 往返。
2. **数组元素 walker**（新 helper，~60 行，字符串/转义感知，流式、不物化）：
   对 spill 文件中 `events` 数组的 span 逐元素给出 `[start, end)`。
3. **span 窗口扫描 reader**：对文件区间 `[start,end)` 以 1MB 块喂
   `JsonFieldScanner`（seek+read，不切片物化）。
4. **`_read_root_output` 改造**：每个候选 event_data 文本分块
   `str.encode('utf-8')` 流式写入 pool spill 文件（不整块驻留）→ 根扫描取
   `events` span → walker 逐元素 → 元素根扫描取 `name`/`attributes` span →
   attributes 扫描取 `event_type`/`event_data` span → event_data 扫描取
   `output` span；按 key 取最新命中者；命中者的 output span 分块拷入第二个
   spill 文件并同时喂 `_ScanSlot`（tee，单遍）→ 返回 `scan_out` 标记
   （与 API 路径同一 `scan_out` 协议）→ `_single_record` 流式组装；
   时间戳仍走 DB `MIN/MAX(time_unix_nano)`（`_read_result` 已复用）。
5. **输出一致性**：event_data 中 output 为宽松格式 → 组装前对 output span
   跑规范化转码（步骤 1），产物与 dict 路径逐字节一致；
   `metadata` 缓存投影（#6 的 span 解析）同样适用。
6. **错误路径**：event_data 非法 JSON → ExportError（对齐 `_decode_event_data`
   的 'malformed TDSQL event_data'）；`output` 非 dict → None → missing/占位
   语义与旧一致。
7. **测试矩阵**：fake TDSQL 行（多 span、多事件、last-match、attempt 乱序、
   非 dict output、丢行 fallback、宽松/ensure_ascii、跨块切割）
   × 与 dict 路径逐字节对照；内存断言（大 event_data 下峰值有界）。

### 风险

- 规范化模式动转码器核心状态机：所有导出路径共用，fuzz 必须先行；
- walker 的 last-match 与 `extract_run_output` 语义需逐项对齐
  （name/type/形状四重条件、浅拷贝差异对输出无影响——输出行由序列化决定）。
