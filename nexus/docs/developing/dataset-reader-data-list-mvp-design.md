# Dataset Reader 与 Cloud `data_list.bin` MVP 设计

## 文档状态

- 状态：基础 MVP 已通过真实 TDSQL DevCloud E2E；当前 stacked Harbor 适配已通过 focused/unit
  验证，本 SHA 尚未重新执行同一 DevCloud E2E
- 基线：MR !1111 source `jaysonhtli/t3-harbor-bridge@741773d5754d1aa1ea4c27ce86b783968ade4066`
- 基础 MVP 运行时验证版本：`19b9c5a12fc99e087d3ee174c7b120c7cc55104c`；Harbor stacked
  适配需在合入前用当前 SHA 重新验证
- 日期：2026-08-23
- 范围：统一 Dataset 读取侧，并让本地 Dataset 的 task files 可通过
  `nexus submit batch --generation local` 进入 Cloud Batch

### DevCloud 验收记录

- 报告：`/data/tmp/nexus-dataset-mvp-19b9c5a12-final-report.json`；进程 exit status 为 0。
- 远程 Dataset：23/23 succeeded，task-list v2，不含 data-list；23 次签名、69 次
  Range 请求，未发生整 ZIP 下载。
- 本地 Dataset：directory/ZIP 两个 Item 均 succeeded，task-list v3，NXDL v1 包含
  2 个 frame；两个 child 均从 `ctx.task_files` 读到预期文件。
- 真实 TDSQL：Batch active 时存在 2 条 v3 record，parent terminal 后自动清零；
  child 只保存 `run_input_ref`。
- 路径与清理：parent、task-list、child 和 TDSQL identity 均不含提交机绝对路径；
  assignment 结束后实际 Manager Pod 中的临时 `files.zip` 列表为空。
- 环境清理：测试 release、动态资源、Secrets 和 fake Dataset 已删除；集群仅暂留带
  `deletionTimestamp` 与 `batch.kubernetes.io/job-tracking` finalizer 的已完成 Pod
  tombstone。临时 DCDB 账号与本地凭据文件已删除。

本文基于 MR !1111 的 Dataset Reader 架构结论；本 stacked 分支只加入 Harbor
Reader/task-files 的最小适配，不引入 `feat/habor_quick_support` 的整套实现。本文同时定义一个最小的 Cloud
本地文件传输协议；它只解决已解析 `RunInput.task_files` 指向提交机本地目录或
ZIP 时，远端 Batch 无法访问该路径的问题。

## 1. 一页结论

MVP 由两个彼此独立、通过 `DatasetItem` / `RunInput` 连接的部分组成：

1. 读取侧统一为 `DatasetReader`：
   - 核心能力只有 `describe()`、`iter_item_ids()`、`iter_items()`；
   - Local 与 Taiji 是内建 Reader，stacked 适配提供 `t3-harbor` Reader；
   - `open_dataset()` 是所有消费者的唯一入口；
   - 按 URI scheme 注册可选 Reader，为后续 `t3-harbor://` 留出扩展点；
   - `RunComposer`、Dataset inspection CLI 与 `materialize_compose()` 共用该入口。
2. Cloud Batch 本地 task files 使用内部 `data_list.bin`：
   - 只在 `submit batch --generation local` 的 Compose/Dataset 路径启用；
   - 根据最终解析后的 `RunInput.task_files` 分类，不根据 Dataset URI 分类；
   - 没有本地 task files 时继续生成 task-list v2，不上传 data-list；
   - 任意 Item 含本地目录或 ZIP 时，整批生成 task-list v3，并上传一个
     `data_list` Artifact；
   - Batch Manager 整体下载 data-list，在本地文件中 seek，不做 HTTP Range；
   - 每个 Item 的二进制 record 写入现有 TDSQL `batch_task_records.record_data`
     `LONGBLOB`；
   - Instance Manager 为 assignment 写一个临时 `files.zip`，注入标准
     `TaskFilesSpec`，继续复用 `LocalZipTaskFiles`。

三种容易混淆的对象边界如下：

| 对象 | 用途 | 是否是用户 Dataset 格式 | 生命周期 |
|---|---|---:|---|
| `dataset.yaml` / `item.yaml` / `files.zip` | 本地 Dataset authoring 与 `dataset fetch` 输出 | 是 | 用户管理 |
| task-list JSONL | Cloud Batch 的 Item/RunInput 清单 | 否 | Batch Artifact |
| `data_list.bin` | task-list v3 引用的本地 task-files 传输容器 | 否 | Batch Artifact |

`data_list.bin` 不替代 `dataset fetch`，也不是新的 bundle、publish release 或
trace event 格式。

## 2. 目标

### 2.1 Reader 目标

- Local、Taiji 以及未来 Harbor 对 Runner 暴露同一种只读接口。
- Item ID 可以轻量、稳定地枚举，不要求先物化完整 `DatasetItem.body`。
- Item 可以流式读取，避免大 Dataset 被一次性装入内存。
- `latest` / `dev` 在 Reader 生命周期内解析成具体 Taiji 版本。
- Reader 的网络 session、缓存和关闭行为由 `open_dataset()` 统一管理。
- Harbor 只需提供 Reader factory 和标准 `DatasetItem`，无需在
  `RunComposer`、CLI、Batch materializer 中各加一条 Harbor 分支。

### 2.2 Cloud 本地文件目标

- 本地 Dataset Item 的 `files.type=directory|zip` 可以通过
  `submit batch --generation local` 执行。
- 提交机的 task-files 绝对路径不进入最终 task-list、Control DB、TDSQL identity 或
  child spec。
- 远程 Taiji task files 保持现有 signed URL + ZIP Range 读取，不被下载后重新打包。
- 复用现有 Artifact 上传、鉴权、scope、digest 和 TTL 机制。
- 复用现有 Batch Task Store 表，不新建一套 Blob 服务或数据库表。
- 复用 `TaskFilesSpec`、`task_files_from_spec()` 与 `LocalZipTaskFiles`，不增加
  `BlobTaskFiles`。
- 所有格式、大小限制、错误和清理行为 fail closed。

## 3. 非目标

本 MVP 明确不包含：

- Harbor 的通用 compiler/task 重构；本 stacked 适配只复用现有 discovery/compiler；
- 把 Publisher 泛化为任意 Reader；`dataset fetch` 仅为注册 Reader source 输出标准本地布局；
- Dataset publish、COS Dataset Store、immutable release 或 `DatasetSink`；
- 让现有 Publisher 接受任意 Reader；
- 一次 Compose 选择多个 Dataset，或多个 Dataset 各自的 ID 列表；
- `run batch` 的 data-list；本地 `run batch` 继续直接使用本地路径；
- `submit single`、`submit live` 或本地 generator 的 task-files 打包；
- remote generation 生成 data-list；
- data-list 的 HTTP Range、随机远程读取或分片下载；
- 把整个 data-list 写进一条数据库 record；
- 展开整个 ZIP 到 assignment 目录；
- 为只有 Taiji 一个真实写入目标而抽象 `DatasetSink`。

如果 local generator 产出本地 `task_files`，本 MVP 不声明其 Cloud 可执行性，
也不会借 generator 旁路启用 data-list。实现应在上传前拒绝这种输入，而不是把
提交机路径发到 Cloud。

## 4. 当前实现与需要收敛的分叉

### 4.1 Dataset 读取分叉

当前 `RunComposer` 在 `nexus/runner/compose/compiler.py` 中自行判断：

```text
data value
  ├─ 存在 dataset.yaml 的本地目录 → LocalDataset.load_items()
  └─ 其他引用                    → DatasetClient.load_items()
```

当前 `materialize_compose()` 在 `nexus/distributed/batch/materialize.py` 中重复同一判断，
但为了 Cloud Batch 的规模需要，又单独使用了 Taiji 的分页 iterator。

当前 Dataset CLI 也在每个命令内分别判断 Local/Taiji。三处虽然共享
`DatasetItem`，却没有共享读取契约、resolver 或生命周期。

### 4.2 Batch task-list 与 Task Store

当前本地 generation 只上传 task-list：

```text
CLI materialize JSONL
  → 上传 kind=task_list Artifact
  → Batch Manager 整体下载 task-list
  → 逐块解析并创建 BatchItemRow
  → 有 TDSQL：每个 identity 写 batch_task_records
  → 无 TDSQL：创建 child 时从 task-list 读取并内联 run_input 到 Control DB
```

task-list v1/v2 中的 `run_input.task_files` 是普通 JSON。如果它指向提交机绝对
路径，Cloud 的 Instance Manager 无法访问该路径。source package 只按项目根和
ignore 规则打包，也不会把该提交机路径改写为 Cloud workspace 路径；项目外的
Dataset task-files 更不会随 source package 传输。因此不能依赖 source package
解决这一问题。

现有 `batch_task_records.record_data` 是 `LONGBLOB`，适合承载单 Item record；
Control DB 只需继续保存 child 的 `run_input_ref` 和 Batch ledger，不应保存 ZIP bytes。

## 5. 统一 Dataset Reader

### 5.1 类型化描述模型

Reader 的描述信息使用明确模型，而不是 CLI 专用的任意字典：

```python
class DatasetSplitDescriptor(BaseModel):
    name: str
    item_count: int | None = None


class DatasetDescriptor(BaseModel):
    reference: str
    resolved_reference: str
    provider: str
    name: str
    default_split: str | None = None
    splits: list[DatasetSplitDescriptor] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

字段语义：

- `provider` 是 `local`、`taiji` 或扩展 scheme，例如 `t3-harbor`；
- `reference` 保留用户输入，用于诊断；
- `resolved_reference` 是当前 Reader 实际读取的身份：Local 使用规范化绝对根目录，
  Taiji 必须把 `latest/dev` 替换为具体版本；
- `splits` 按 Provider 的稳定顺序返回，`item_count` 不可廉价获得时为 `None`；
- `metadata` 只保存非敏感的 Provider 信息，不得包含 token、key、signed URL 或凭据。

### 5.2 最小 Reader Protocol

```python
from collections.abc import AsyncIterator
from typing import Protocol


class DatasetReader(Protocol):
    async def describe(self) -> DatasetDescriptor: ...

    def iter_item_ids(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[str]: ...

    def iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[DatasetItem]: ...
```

便利函数建立在 Protocol 之上，不增加 Reader 实现负担：

```python
async def get_item(reader, *, split=None, item_id: str) -> DatasetItem: ...
async def load_items(reader, *, split=None, item_id=None, limit=None) -> list[DatasetItem]: ...
```

`versions()` 不属于已打开 Reader 的公共能力。MVP 中 `dataset versions` 保持
Taiji catalog 专用；Local 和扩展 Reader 明确报告不支持，而不是伪造
`["local"]`。资产列表和 namespace 管理同样继续使用具体的 Taiji client。

### 5.3 统一契约语义

所有 Reader contract tests 使用下列语义：

- `split=None`：优先使用 descriptor 的 `default_split`；没有默认值且恰好一个
  split 时选择该 split；否则报 `dataset_split_required`。
- 显式 split 不存在时报 `dataset_split_not_found`。
- `limit` 只能是 `None` 或非负整数；`0` 产生空结果，负数和 bool 报
  `dataset_limit_invalid`。CLI 在 `limit=0` 时不调用 Provider 的正数分页接口。
- limit 在 split、Provider 过滤和无效 Item 处理之后计数，表示成功 yield 的数量。
- `item_id` 在选定 split 中精确匹配；不存在时报 `dataset_item_not_found`，不返回空列表。
- `item_id` 与正 limit 同时出现时最多 yield 一个 Item。
- 顺序必须稳定。Local 使用 `dataset.yaml` 中的 items 顺序；Taiji 使用远端索引
  顺序；未来 Harbor 使用 discovery 的规范排序。
- 同一 split 中遇到重复 ID 立即报 `dataset_duplicate_item_id`，不得静默去重。
- 对同一已解析引用和参数，`iter_item_ids()` 与 `iter_items()` 的 ID 顺序一致。
- Item 校验错误默认终止 iterator。Provider 若显式支持 `skip_invalid`，limit 只计算
  成功 yield 的 Item，且跳过原因必须可诊断。
- 一个 Reader 支持顺序调用，不承诺多个 iterator 并发消费；需要并发消费时分别
  `open_dataset()`。
- Reader 关闭后调用任何方法或继续消费 iterator 均报 `dataset_reader_closed`。
- Local/未来 Harbor 的源目录不是 immutable snapshot；读取期间发生变化可以使本次
  读取失败。Taiji 的 symbolic version 在打开时只解析一次并固定。

### 5.4 Resolver、Registry 与生命周期

统一入口是 async context manager：

```python
async with open_dataset(reference, authenticator=authenticator) as reader:
    async for item in reader.iter_items(split=split, limit=limit):
        ...
```

Resolver 按以下顺序分类，禁止失败后从一种来源静默回退到另一种来源：

1. `Path`，或以 `/`、`./`、`../`、`~` 开头的显式文件系统引用：解析为 Local；
   目录或 `dataset.yaml` 不存在时直接报本地路径错误。
2. 含 `scheme://` 的字符串：scheme 转小写后查 `dataset_source_registry`；未知
   scheme 报 `dataset_reader_scheme_not_found`。
3. 其余字符串若按当前工作目录解析后是含 `dataset.yaml` 的目录，解析为 Local；
   这保留 `--data harbor-local` 一类现有裸相对路径用法。
4. 其他字符串：按现有 Taiji Dataset URI 解析。

Local 与 Taiji 内建在 resolver 中，不为了形式一致注册到模块系统。扩展 scheme
使用 MR !1111 提供的 lazy registry 类型 `dataset_source`，对应 module metadata key
`dataset_sources`：

```toml
[tool.nexus]
dataset_sources = { "t3-harbor" = "nexus.extensions.harbor_bridge.dataset_source" }
```

Registry entry 是 Reader factory。Factory 接收 `DatasetSourceURI`，返回由
`open_dataset()` 管理的 Reader。Provider 自己解析和校验
scheme query/options；通用层不理解 Harbor 参数。

`open_dataset()` 负责：

- 创建并关闭 Taiji `DatasetClient` / `aiohttp.ClientSession`；
- 在单个 Reader 内缓存 descriptor、已解析版本、默认 split 和可复用的 Provider
  discovery/index 元数据；
- 分页 cursor 属于各自 iterator；新的顺序调用必须从头开始，不能复用上一次未消费完
  或已完成 iterator 的 cursor；
- 在异常、取消和正常退出时调用关闭逻辑；
- 保证未消费完的 iterator 不逃逸出 context；
- 不建立跨命令、跨 event loop 的全局 Reader 实例缓存。

### 5.5 三个消费者的收敛方式

```text
                            ┌─ LocalDatasetReader
reference ─ open_dataset ───├─ TaijiDatasetReader
                            └─ registry extension reader
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
             RunComposer      Dataset inspection CLI   materialize_compose
             list helper       describe / ids / get     streaming items
```

- `RunComposer` 使用 `load_items()` 便利函数，保留 single 选择和本地 run-batch
  当前需要的 list 结果。
- `materialize_compose()` 直接消费 `iter_items()`，保持 Cloud Batch 分页、流式和
  bounded memory。
- `dataset show`、`dataset items/list`、`dataset get` 使用 Reader；CLI 只负责格式化。
- `dataset versions`、无 dataset 参数的资产列表、namespace、fetch、publish 和 init
  保持各自现有具体路径。
- `PreparedRunParts`、merge 顺序、metadata 和 `DatasetItem.resolved_task_files()` 保持
  一个实现，两个消费者不得再次自行分派 Local/Taiji。
- Compose 仍只允许一个 data part；多 Dataset compose 不在本 MVP 中。

## 6. 何时生成 `data_list.bin`

判断对象是 Compose 完成 merge 后的最终 `RunInput.task_files`，而不是原始
Dataset URI。这使下列情况语义一致：

- 本地标准 Dataset 的 directory/ZIP；
- `dataset fetch` 后得到的本地 `files.zip`；
- 未来 Harbor Reader 返回的本地 directory；
- Compose 的后续 part 覆盖了 Dataset Item 中的 `task_files`；
- Taiji Reader 返回带 `remote` 坐标的 ZIP。

### 6.1 分类表

| 最终 `TaskFilesSpec` | 分类 | task-list 行 | data-list frame | Cloud 行为 |
|---|---|---|---|---|
| `None` | 无文件 | v2；混合批次中为 v3 + `data_ref:null` | 无 | 不注入 task files |
| `remote != None, type=zip` | 远程 ZIP | 原样保留在 `run_input.task_files` | 无 | 保持 signed URL + Range |
| `remote != None, type!=zip` | 非法远程类型 | 不生成 | 无 | 提交失败 |
| `remote == None, type=directory` | 本地目录 | 从 `run_input` 移除；v3 引用 frame | 生成 Nexus ZIP 后写入 | assignment 临时 ZIP |
| `remote == None, type=zip` | 本地 ZIP | 从 `run_input` 移除；v3 引用 frame | 校验后原字节写入 | assignment 临时 ZIP |
| 本地路径不存在或类型不匹配 | 非法本地输入 | 不生成 | 无 | 提交失败 |

整批版本选择规则是确定的：

- 全部 Item 都没有本地目录/ZIP：最终 task-list 为 v2，不产生或上传 data-list。
- 任意 Item 有本地目录/ZIP：最终所有行均为 v3，data-list 至少有一个 frame。
- 一个 task-list Artifact 中不混用 v2/v3 行。
- v3 不是用户可强制选择的 CLI 格式；它由 materializer 在发现本地 task files 时
  自动选择。
- `--task-list-version 1` 不适用于该 Compose/data-list 路径；generator 与 remote
  generation 的现有 v1/v2 兼容选项不因此扩展到 v3。

### 6.2 一次读取、原子输出

一次 materialization 只消费一个 items iterator，不能为了判断整批版本重新读取远端
Dataset。materializer 使用私有临时 record spool：

1. 单次消费 Reader，完成 merge、Batch Item ID 规范化和 TaskFilesSpec 分类。
2. 普通/远程 Item 把规范化 record 写入 spool。
3. 本地 Item 生成或校验 ZIP，先向临时 data-list 写 frame，再把 sanitized record
   与 `data_ref` 写入 spool。
4. 完成迭代后，根据是否出现本地 frame，把 spool 渲染成最终 v2 或 v3 JSONL。
5. 对最终文件计算 size/digest。task-list 与 data-list 输出路径必须不同；发布前先把
   两个既有目标移动到私有 backup，再依次 replace 两个 staged 文件。第二次 replace
   失败时删除已发布的新文件并恢复两个 backup，成功后才删除 backup。
6. 删除 spool 和所有 per-item 临时 ZIP。

任何异常或取消都删除临时 task-list、data-list、record spool 和临时 ZIP，不返回
部分 `MaterializedTaskList`。临时 spool 即使包含提交机路径也只存在于提交机私有
临时目录，永不上传。

`materialize_compose()` 的返回值扩展为同时描述可选 data-list 的结果，例如
`MaterializedBatchArtifacts(task_list=..., data_list=None|...)`。现有无 data-list
调用方只读取 task-list 字段。

### 6.3 本地目录和 ZIP 的规范化

- directory 使用现有 Nexus ZIP writer：稳定排序文件、启用 Zip64、保存 entry
  mode/hash，以便 `LocalZipTaskFiles` 按 entry 读取。
- directory 打包拒绝 symlink、socket、device 和任何解析后逃出根目录的路径。
- ZIP 输入先用现有 ZIP index parser 校验：禁止绝对 entry、`..` 逃逸、不支持的
  compression method、损坏的 central directory 和 multi-disk ZIP。
- ZIP 输入若声明 `sha256` / `size`，必须与实际文件一致；未声明时由 materializer
  计算。
- 已有 ZIP 原字节进入 frame，不进行二次压缩或解压重打包。
- 每个 Item 的临时 ZIP 写完并 fsync 后才进入 frame；frame 写完即可删除该临时 ZIP。

## 7. task-list v3

### 7.1 行格式

task-list 仍是 UTF-8 JSONL，每行以 `\n` 结束。v3 沿用现有 task-list writer 的
deterministic JSON 编码，并固定字段如下：

```json
{
  "version": 3,
  "item_id": "task-a",
  "run_input": {
    "task_name": "terminal_task",
    "task_config": {"message": "..."}
  },
  "metadata": {"dataset_item_id": "task-a"},
  "data_ref": {
    "version": 1,
    "offset": 32,
    "length": 1049024,
    "sha256": "64-lowercase-hex"
  }
}
```

没有本地 frame 的 v3 行必须显式写：

```json
"data_ref": null
```

规则：

- `item_id` 使用 task-list writer 最终规范化后的 Batch Item ID；原始非法 run ID
  继续记录在现有 `_nexus.original_run_id` metadata 中。
- `run_input` 继续排除 `run_id`；child execution ID 在 Instance Manager 解码后注入。
- 本地 task files 的 `run_input` 中不得再有 `task_files` 字段。
- 远程 task files 保留在 `run_input.task_files`，且 `data_ref` 为 `null`。
- `metadata` 使用现有 canonical item metadata 规则和 64 KiB 上限。
- `data_ref.offset` 从 data-list 文件起点计算，指向 frame header 的第一个字节。
- `data_ref.length` 是完整 frame 长度，包含 frame header、metadata 和 ZIP bytes。
- `data_ref.sha256` 是该完整 frame 的 SHA-256。
- offset、length 必须是 `0..2^63-1` 的非 bool 整数；MVP 的 4 GiB 限制远低于该上限。

v3 record identity 是以下 canonical JSON 对象，不含行尾换行：

```python
{
    "version": 3,
    "item_id": item_id,
    "run_input": normalized_run_input,
    "metadata": canonical_metadata,
    "data_ref": data_ref_or_none,
}
```

identity 始终通过现有 `canonical_sha256()` 计算；不能直接散列原始 task-list 行。
其 SHA-256 继续作为 `BatchItemRow.spec_sha256` 和 TDSQL
`BatchTaskRecord.record_sha256`，因此 child `run_input_ref` 的 fencing 机制不变。

### 7.2 Submission schema

新增 Artifact kind：

```python
ArtifactKind = Literal[..., "task_list", "data_list", "run_result"]
```

本地 generation 增加可选 sibling `data_list`：

```json
{
  "mode": "local_task_list",
  "task_list": {
    "ref": {"kind": "task_list", "artifact_id": "...", "sha256": "...", "size_bytes": 1234},
    "format": "jsonl",
    "manifest": {
      "version": 1,
      "record_version": 3,
      "item_count": 2,
      "size_bytes": 1234,
      "sha256": "...",
      "source": {"mode": "local_compose"},
      "datasets": []
    }
  },
  "data_list": {
    "ref": {"kind": "data_list", "artifact_id": "...", "sha256": "...", "size_bytes": 1049056},
    "format": "nxdl",
    "manifest": {
      "version": 1,
      "frame_count": 1,
      "size_bytes": 1049056,
      "sha256": "..."
    }
  }
}
```

Pydantic cross-validation 强制：

- `record_version == 3` 当且仅当 `data_list` 存在；
- data-list ref kind 必须是 `data_list`；
- ref 与 manifest 的 size/digest 必须一致；
- `frame_count >= 1`；
- v1/v2 task-list 不允许携带 data-list；
- remote materialize source 的 `task_list_version` 仍只接受 1/2。

`artifact_refs(submission)` 同时返回 task-list 和 data-list，使 Control 在同一 submission
事务中完成 owner、Environment、ready 状态和 scope 校验。data-list spec 已持久化在
parent `Execution.spec_json.generation`；MVP 不为 `distributed_batch_runs` 增加冗余
data-list ref/manifest 列。

`TaskListManifest.datasets` 继续只记录 Taiji 这类远端 Dataset 的 input/resolved
version；Local 或未来 Harbor Descriptor 中用于本机诊断的绝对 `resolved_reference`
不得复制到 Cloud submission。

## 8. `NXDL` v1 二进制格式

### 8.1 通用规则

- 文件名：`data_list.bin`；Artifact kind：`data_list`。
- 所有整数是 little-endian unsigned integer。
- 文件本身不压缩、不加密、不填充对齐字节。
- ZIP entry 可继续使用 ZIP 自身的 stored/deflate compression。
- 字符串和 JSON 使用 UTF-8。
- 文件必须精确结束在最后一个 frame 尾部，不允许 trailing bytes。

### 8.2 File header

固定 32 bytes，等价于 `struct.Struct('<4sHHQQQ')`：

| Offset | 长度 | 字段 | v1 要求 |
|---:|---:|---|---|
| 0 | 4 | `magic` | ASCII `NXDL` |
| 4 | 2 | `version` | `1` |
| 6 | 2 | `flags` | `0` |
| 8 | 8 | `header_size` | `32` |
| 16 | 8 | `frame_count` | 本文件 frame 数，至少 1 |
| 24 | 8 | `reserved` | `0` |

writer 在完成后回填 `frame_count`，再对最终完整文件计算 Artifact SHA-256。

### 8.3 Frame

每个 frame 由 24-byte header、canonical metadata JSON 和 ZIP bytes 连续组成。
header 等价于 `struct.Struct('<QQQ')`：

| Frame 相对 offset | 长度 | 字段 | 含义 |
|---:|---:|---|---|
| 0 | 8 | `frame_size` | 完整 frame bytes，包含本 24-byte header |
| 8 | 8 | `metadata_size` | metadata JSON bytes |
| 16 | 8 | `zip_size` | ZIP bytes |
| 24 | `metadata_size` | `metadata` | canonical UTF-8 JSON |
| `24 + metadata_size` | `zip_size` | `zip` | 完整 ZIP 文件 |

必须满足：

```text
frame_size == 24 + metadata_size + zip_size
```

metadata 固定 schema：

```json
{
  "item_id": "task-a",
  "task_files": {
    "type": "zip",
    "sha256": "64-lowercase-hex",
    "size_bytes": 1048576
  },
  "version": 1
}
```

canonical JSON 使用 `ensure_ascii=False`、`allow_nan=False`、`sort_keys=True`、
`separators=(',', ':')`，无换行。parser 解码后重新 canonical encode，必须与原 bytes
完全相等。

### 8.4 data-list 与 task-list 的一一校验

Batch Manager 整体下载并先校验 Artifact size/digest，再对 data-list 做唯一一次顺序
扫描，建立最多 `max_items` 条的内存 frame index，并记录本地文件 fingerprint。初始扫描
强制校验 frame 边界、完整 frame SHA-256、metadata canonical encoding、ZIP size/digest、
ZIP index、header `frame_count` 以及文件末尾边界；扫描前后的 fingerprint 必须一致。

task-list 分块物化时按 `data_ref.offset` 查询 index 并从本地文件 seek。当前读取的完整
frame SHA-256 必须与 index 中已验证的 frame identity 一致，但不再重复计算 ZIP digest
或解析 ZIP index。在 generation 被标记 complete 前，只重新从头扫描 task-list：

- task-list 完整 SHA-256 必须与 Artifact ref 一致；
- 每个非空 `data_ref` 必须等于 index 中按 task-list 顺序出现的下一个 frame ref；
- frame index 中的 `item_id` 必须等于 task-list `item_id`；
- 不允许两个 Item 引用同一 frame，也不允许 orphan 或未引用 frame；
- 非空 `data_ref` 数必须等于已验证 index 的 frame 数；
- task-list/data-list fingerprint 在交叉校验前后都必须保持不变。

finalize 不再第二次顺序扫描 data-list，也不再第二次校验 ZIP 内容。

任何一项不满足都在创建 child 前终止整个 Batch materialization。

## 9. Cloud 数据流

```text
Local/Taiji/extension DatasetReader
            │ iter_items
            ▼
      PreparedRunParts
            │ resolved RunInput.task_files
            ▼
 submit --generation local
   ├─ tasks.jsonl (v2 or v3) ───────┐
   └─ data_list.bin (v3 only) ──────┤ existing Artifact upload
                                    ▼
                              Control submission
                                    │ refs + manifests
                                    ▼
                              Batch Manager
                         ┌──────────┴──────────┐
                         │ whole-file download │
                         │ local seek/validate │
                         └──────────┬──────────┘
                                    ▼
                         TDSQL BatchTaskRecord v3
                                    │ child run_input_ref
                                    ▼
                              Instance Manager
                         ┌──────────┴──────────┐
                         │ assignment files.zip│
                         │ inject TaskFilesSpec│
                         └──────────┬──────────┘
                                    ▼
                       LocalZipTaskFiles → runtime bridge
```

### 9.1 Submitter 与 Artifact

- CLI 在所有本地 materialization 成功后上传 data-list 和 task-list；两个 Artifact
  都使用内容 digest 和 bounded idempotency key。
- 上传顺序不提供事务语义；最终 submission 事务同时校验并引用两个 ready Artifact。
  上传成功但 submission 失败的未链接 Artifact 由现有 TTL/maintenance 清理。
- Artifact ref 不暴露 backend object key、signed URL 或凭据。
- Application Gateway 复用现有文件流上传/下载和本地 spool，不为 data-list 增加
  专用存储后端。

### 9.2 Batch Manager

- 先检查 task-list manifest；v3 且没有 Batch Task Store 时立即报
  `batch_task_store_required_for_data_list`。
- 将整个 data-list 下载到 parent Batch workspace，例如
  `<batch-root>/<parent_execution_id>/data_list.bin`。Artifact client 校验完整 size/digest。
- 不把整个文件读入内存；初始顺序扫描只保留 bounded frame index，随后按 task-list
  data_ref 对本地文件执行 seek/read，一次最多持有一个 frame 和当前 TDSQL write
  chunk。
- task-list v3 解析、NXDL 校验与 TDSQL 写入都发生在现有 generation deadline 内。
- 每个物化 chunk 使用“短事务读取 fenced snapshot → 事务外磁盘读取、record 编码和
  TDSQL I/O → 短事务重新 fence 并 CAS 提交 cursor/count”的结构。snapshot 过期时从
  新 cursor 重试，TDSQL 幂等重放；远程 I/O 不持有 `BatchRunRow FOR UPDATE` 锁。
- 最后的短 Control transaction 只有在完整 task-list 与初始 data-list index 交叉校验
  通过后才把 generation 标记 complete；因此 restart 只需重建本地 frame index，
  不需要新增或持久化 data-list cursor。
- materialization 重启时重新校验本地 file fingerprint；文件缺失或改变则通过
  Artifact 重新下载，不信任旧路径。
- TDSQL insert 与 Control DB ledger 继续使用现有幂等模式：TDSQL 先
  `INSERT IGNORE` 并核对 identity，Control transaction 后提交。中途 crash 后重放
  必须得到相同 record，否则报 identity mismatch。

### 9.3 TDSQL `BatchTaskRecord` v3

现有表和主键不变：

```sql
PRIMARY KEY (parent_execution_id, item_id)
record_version INT NOT NULL
record_sha256 CHAR(64) NOT NULL
record_data LONGBLOB NOT NULL
```

v1/v2 `record_data` 的 canonical JSON 格式保持不变。v3 使用一个避免 base64 膨胀的
binary envelope。

固定 24-byte header，等价于 `struct.Struct('<4sHHQQ')`：

| Offset | 长度 | 字段 | v3 要求 |
|---:|---:|---|---|
| 0 | 4 | `magic` | ASCII `NXBR` |
| 4 | 2 | `record_version` | `3` |
| 6 | 2 | `flags` | `0` |
| 8 | 8 | `identity_size` | canonical task-list identity bytes |
| 16 | 8 | `frame_size` | exact NXDL frame bytes；无本地 frame 时为 0 |

随后依次存：

```text
canonical task-list v3 identity JSON
exact NXDL frame bytes, or empty when data_ref is null
```

完整 record 必须满足：

- `record_version == 3`；
- `record_sha256 == SHA256(identity JSON)`，并等于 `BatchItemRow.spec_sha256`；
- identity 中的 version/item_id/data_ref 与 TDSQL row、task-list 完全一致；
- `data_ref is null` 当且仅当 `frame_size == 0`；
- 非空 frame 的 length/digest、metadata 和 ZIP identity 再次通过 NXDL 校验；
- `len(record_data) == 24 + identity_size + frame_size`。

TDSQL writer 按 `record_data` 完整 bytes 的聚合大小分 chunk，而不只按 record 个数。
单条 record 不拆分；超过 write chunk target 但未超过 record hard limit 的 record 单独
写入。Batch Manager 一次只编码一条 NXBR，当前 record 加入 write chunk 会超出预算时
先 flush 旧 chunk，因此内存上界是一个 record 加一个配置的 write chunk，而不是当前
task-list chunk 的 record 数。每个 write chunk 写后仍查询并核对
`(item_id, record_version, record_sha256)`。

### 9.4 Parent 终态后的 Task Store 清理

`batch_task_records` 是 parent-owned temporary data。生命周期顺序固定为：

1. 先在 Control transaction 中持久化 parent/assignment/Batch 的 terminal state 并提交；
2. 只对 task-list v3 parent，按 `parent_execution_id` 删除全部 TDSQL rows；
3. 删除成功后正常释放 lifecycle claim 并写入 `last_reconciled_at`。

终态提交时，v3 parent 使用现有 `distributed_batch_runs.last_reconciled_at = NULL` 表示
清理尚未确认，不新增 DB 列。lifecycle claim 只重新选择
`control_state=finalized`、`record_version=3` 且该字段为 NULL 的 parent。删除失败，或
删除已经生效但连接在确认前报错时，只清除 claim 的 owner/operation/lease 字段，保留
NULL；下一轮 reconcile 会再次执行同一个 parent-key DELETE。DELETE 本身幂等，所以
部分删除、重复删除和 crash replay 都安全。清理成功后 claim 的正常 release 写入时间，
不再选择该 parent。

Task Store 错误不得回滚或改写已经提交的 Control terminal state，也不得改变业务结果；
它只留下可重试的 cleanup claim 并记录 reconcile 错误。这一状态机复用现有 parent
字段和 lifecycle lease，不要求 Control DB migration。

### 9.5 没有 TDSQL 时

- task-list v1/v2 保留当前 fallback：创建 child 时从本地 task-list 读取 record，
  把 `run_input` 内联到 Control DB child spec。
- task-list v3 不允许该 fallback，因为它会要求把 ZIP 放入 Control DB JSON，或让
  child 依赖 Batch Manager 本地文件。
- v3 在 Batch Manager 准备阶段、任何 child 创建之前报
  `batch_task_store_required_for_data_list`；已经创建的 parent 按 generation failure
  结束。
- 不静默丢弃 files、不降级成路径、不自动截断 record。

### 9.6 Instance Manager 的 assignment 临时 ZIP

Instance Manager 读取 v3 record 后：

1. 校验 record identity、frame 和 ZIP digest。
2. 在 manager-scoped 临时根下以 `0700` 创建 assignment 私有目录，并以 `0600`
   原子写入 `files.zip`。
3. 从 `run_input_ref` 还原 `run_input`，注入：

   ```python
   {
       "type": "zip",
       "path": "/tmp/.../<assignment_id>/files.zip",
       "sha256": "...",
       "size": 123456,
   }
   ```

4. 注入 child execution ID 作为 `run_id`，再启动现有 `NexusSingleDriver`。
5. `RunContext` 继续调用 `task_files_from_spec()`，得到 `LocalZipTaskFiles`。

ZIP 不整体解压。`LocalZipTaskFiles` 读取 central directory，按 task 请求读取单个
entry，再通过已有 runtime Socket.IO bridge 传输需要的文件。

资源所有权使用一个明确对象，例如
`ResolvedAssignmentSpec(spec, temporary_task_files)`。它在 executor start 前转移给
`RunningAssignment`。所有当前直接 `_running.pop()` 的终止路径统一经过一个幂等
release helper：

- assignment success；
- task failure；
- user cancel/kill；
- start 确认失败；
- local reconcile 发现 assignment 已失效；
- manager self-fence 或正常 stop。

如果 executor start timeout 且 assignment 仍存在或状态未知，沿用当前保留
reservation 的行为，同时保留临时 ZIP；只有确认 executor 不存在后才清理。写 ZIP
或启动失败时尚未转移的临时资源由局部 `finally` 清理。cleanup 失败只记录日志，
不覆盖原任务结果；manager-scoped 临时根在进程正常退出时再次整体清理。DevCloud
Pod 的临时盘在异常进程/Pod 回收后清除 crash orphan。

## 10. 配置与有效限制

所有限制约束完整结构，不只约束 ZIP payload：

| 限制 | 默认值 | 约束对象 |
|---|---:|---|
| data frame | 256 MiB | 24-byte frame header + metadata + ZIP |
| DB task record | 256 MiB | 24-byte NXBR header + identity + optional 完整 frame |
| whole data-list | 4 GiB | 32-byte NXDL header + 所有 frames |

因此 ZIP 的实际最大值略小于 256 MiB；允许一个 256 MiB ZIP 再额外追加 header 的
解释是错误的。

### 10.1 配置面

提交机预检：

- `NEXUS_SUBMIT_MAX_DATA_FRAME_BYTES`，默认 `268435456`；
- `NEXUS_SUBMIT_MAX_TASK_RECORD_BYTES`，默认 `268435456`；
- `NEXUS_SUBMIT_MAX_DATA_LIST_BYTES`，默认 `4294967296`。

Cloud 全局 hard cap 使用 `DistributedConfig`：

- `batch_data_frame_max_size_bytes`，默认 256 MiB；
- `batch_task_record_max_size_bytes`，默认 256 MiB；
- `batch_data_list_max_size_bytes`，默认 4 GiB；
- `batch_task_store_write_chunk_bytes`，默认 64 MiB。

Environment/Grant 可以用现有 resource-limit 合并机制进一步降低：

- `max_data_frame_bytes`；
- `max_batch_task_record_bytes`；
- `max_data_list_bytes`。

Control 把合并后的三个限制写入 parent `generation_limits`。CLI 限制只是早失败；
Control admission、Batch Manager parser 和 Instance Manager decoder 都必须重新执行
各自可见的 hard cap，不能信任客户端。

### 10.2 有效值

实际可接受的 data-list 还同时受以下现有限制约束：

```text
effective_data_list = min(
  configured data-list limit,
  artifact_upload_max_size_bytes,
  Environment/Grant max_artifact_size_bytes,
  Batch workspace remaining bytes
)
```

Batch workspace admission 按 `task_list.size_bytes + data_list.size_bytes` 计算，不只检查
两个文件各自大小。下载临时文件和 Artifact spool 必须是 bounded disk stream，不得
再复制一份 4 GiB 内容到内存。

TDSQL 的 `max_allowed_packet` 必须高于配置的完整 record hard limit 并预留协议开销；
256 MiB record cap 的部署建议至少配置 320 MiB。`LONGBLOB` 的理论容量不等于
驱动、proxy 和 TDSQL packet 已满足该限制。

### 10.3 DevCloud 部署要求

要真正允许 4 GiB data-list，DevCloud 至少需要：

- `NEXUS_DIST_ARTIFACT_UPLOAD_MAX_SIZE_BYTES >= 4294967296`；当前默认 1 GiB，必须上调；
- `NEXUS_DIST_ARTIFACT_TRANSFER_SPOOL_CAPACITY_BYTES >= 4294967296`，并为并发传输
  保留容量；
- Environment/Grant 的 `max_artifact_size_bytes`、`max_data_list_bytes` 不低于目标值；
- `max_workspace_bytes >= max_data_list_bytes + max_task_list_bytes`；在 4 GiB + 1 GiB
  上限下至少 5 GiB，实际部署应留文件系统余量；
- Batch Manager 临时卷和 Instance Manager 临时卷有足够空间；
- TDSQL packet/proxy 限制支持 256 MiB record；
- 4 GiB 上传/下载能在现有 Artifact deadline 内完成，否则同步上调 deadline。

这些是配置变更，不要求把 `DistributedConfig.artifact_upload_max_size_bytes` 的全局
代码默认值从 1 GiB 改为 4 GiB。

## 11. 错误与 fail-closed 行为

主要稳定错误码：

| 阶段 | 错误码 | 条件 |
|---|---|---|
| Reader resolve | `dataset_reader_scheme_not_found` | 未注册 scheme |
| Reader query | `dataset_split_required` / `dataset_split_not_found` | split 选择失败 |
| Reader query | `dataset_item_not_found` / `dataset_duplicate_item_id` | Item identity 失败 |
| Local materialize | `local_task_files_not_found` | 路径不存在 |
| Local materialize | `local_task_files_type_mismatch` | directory/ZIP 声明与实际不符 |
| Local materialize | `local_task_files_invalid_zip` | ZIP index/entry 非法 |
| Local materialize | `local_task_files_identity_mismatch` | 声明 size/hash 不匹配 |
| Limits | `batch_data_frame_exceeds_byte_limit` | 完整 frame 超限 |
| Limits | `batch_task_record_exceeds_byte_limit` | 完整 DB record 超限 |
| Limits | `batch_data_list_exceeds_byte_limit` | 完整 data-list 超限 |
| Admission | `batch_data_list_manifest_identity_mismatch` | ref/manifest 不一致 |
| Batch parse | `invalid_batch_data_list` | NXDL header/frame/JSON/ZIP 非法 |
| Batch parse | `batch_task_list_data_ref_invalid` | ref 越界、乱序、重叠或 digest 不符 |
| Batch prepare | `batch_task_store_required_for_data_list` | v3 没有 TDSQL Task Store |
| Assignment | `batch_task_record_invalid` | NXBR/identity/frame/ZIP 非法 |

失败原则：

- 本地 materialization 失败发生在 Artifact 上传和 submission 之前。
- submission schema 或 admission 失败不创建 execution。
- Batch prepare 失败可以已有 parent，但绝不创建 child。
- 已上传但未链接的 Artifact 由现有 TTL 清理。
- TDSQL identity 冲突视为数据损坏，不覆盖旧 row。
- Assignment record 损坏按 start failure 报告，不尝试跳过 task files 执行。
- 日志只记录 Artifact ID、item ID、size、digest 和错误码，不记录凭据或 signed URL。

## 12. 兼容与迁移

### 12.1 保持兼容

- task-list v1/v2 parser、writer 和无 TDSQL fallback 保持有效。
- 没有本地 task files 的 local Compose 继续得到 v2，行为和远程 Taiji Dataset 一致。
- 远程 `TaskFilesSpec.remote` 保持现有签名与 Range 行为。
- `run batch`、single、live、generator、fetch 和 publish 不自动启用 data-list。
- 本地 Dataset 的 `dataset.yaml` / `item.yaml` layout 不变。
- child `run_input_ref` schema 保持 version 1；它引用的 TDSQL record_version 可为 1/2/3。

### 12.2 需要扩展但不需要 DB migration

- `TaskListVersion`、manifest 和 parser 增加 3。
- `ArtifactKind` 增加 `data_list`。
- `LocalTaskListGeneration` 增加可选 `data_list` spec。
- `EnvironmentResourceLimits` 和 `generation_limits` 增加三个大小字段。
- `batch_task_records` 已有 `record_version INT` 与 `record_data LONGBLOB`，无需改表。
- data-list ref 保存在 parent execution spec，`distributed_batch_runs` 无需新增列。
- Artifact 表的 kind 是字符串，新增 kind 不要求 DDL。

### 12.3 发布顺序

新 CLI 不能向不认识 `data_list` / task-list v3 的旧 Cloud 安全降级。部署顺序是：

1. 先升级 Control/Application Gateway、Batch Manager 与 Instance Manager，使它们能
   接受但不必生成 v3。
2. 配置 Artifact、workspace 和 TDSQL limits。
3. 完成 v1/v2 regression 和 DevCloud v3 E2E。
4. 再发布会自动生成 v3 的 CLI/wheel。

滚动期间旧 task-list 和正在运行的 v1/v2 Batch 不需迁移或回填。

## 13. 测试矩阵

### 13.1 Reader contract

Local Reader 与 mock Taiji Reader 共用 contract tests：

- descriptor、resolved reference、默认 split、单 split fallback；
- ID iterator 不加载 Item body；
- items iterator 的 split/item_id/limit/稳定顺序；
- limit 0/负数/bool；
- missing item、duplicate ID、invalid Item；
- Reader close、iterator cancellation、session cleanup；
- Taiji `latest/dev` 只解析一次并固定；
- resolver 对显式/裸相对本地路径、Taiji URI、注册/未知 scheme 的严格分派。

消费者测试：

- `RunComposer` 与 `materialize_compose()` 对同一 Reader 产生相同 ID、RunInput merge
  和 metadata；
- `dataset show/items/get` 不再包含 Local/Taiji 业务分支；
- `dataset versions` 的 Taiji-only 和 Local unsupported 行为；
- `dataset fetch/publish` 现有测试保持通过。

### 13.2 Task-list/data-list 单元测试

- 无 task files、全部远程 ZIP：v2 且无 data-list；
- local directory、local ZIP：v3、正确 frame、task-list 无绝对路径；
- remote/local/none 混合：全部 v3，只有 local Item 有 data_ref；
- directory ZIP writer 的稳定顺序、symlink/special-file 拒绝；
- ZIP 声明 hash/size、损坏 index、危险 entry；
- NXDL golden bytes，header/frame/data_ref 精确 offset；
- bad magic/version/flags/reserved、截断、整数溢出、trailing bytes；
- 非 canonical metadata、item mismatch、frame/ZIP digest mismatch；
- frame/list/record hard limit 的等于上限通过与上限加 1 失败；
- 任意失败/取消后所有临时文件被清理。

### 13.3 Submission、Artifact 与 Batch

- v3 payload 同时上传并引用 `task_list` / `data_list`；
- ref/manifest kind、size、digest、frame_count cross-validation；
- Artifact owner/Environment scope 与现有 code/task-list 一致；
- v1/v2 不允许 data-list，v3 必须有 data-list；
- Batch Manager 是 whole-file download，测试 transport 不发 Range；
- Batch Manager restart 后重新下载/校验改变或缺失的本地文件；
- v3 materialization 写 deterministic TDSQL rows，crash 重放幂等；
- TDSQL writer 按聚合 bytes 分 chunk，大 record 单独写；
- terminal Control commit 先于 parent-key TDSQL cleanup，首次 cleanup 失败和
  delete-then-error 都能在下一次 reconcile 幂等重试；
- 无 Task Store 时精确报 `batch_task_store_required_for_data_list` 且 child count 为 0；
- v1/v2 无 Task Store 的 inline fallback regression。

### 13.4 Instance Manager

- v3 record 解码并注入标准 local ZIP `TaskFilesSpec`；
- `task_files_from_spec()` 实际创建 `LocalZipTaskFiles`，按 entry 读取正确内容；
- bad NXBR、identity、frame、ZIP 在 executor start 前失败；
- success/failure/cancel/start failure/reconcile/self-fence/manager stop 均清理 ZIP；
- start timeout 且 executor 状态未知时保留 ZIP，确认不存在后清理；
- 多 assignment 临时目录隔离；目录权限为 `0700`，`files.zip` 为 `0600`。

### 13.5 DevCloud E2E

最小 E2E 使用一个标准本地 Dataset，至少包含：

- 一个 `files.type=directory` Item；
- 一个 `files.type=zip` Item；
- task 实际通过 `ctx.task_files` 读取文件并把内容反映在结果中。

执行：

```bash
nexus submit batch \
  --generation local \
  --data /absolute/path/to/local-dataset \
  --use /absolute/path/to/run-part.yaml \
  --environment <devcloud-environment> \
  --wait
```

验收证据包括：

- submission manifest 为 task-list v3，存在 `data_list` Artifact；
- 上传后的 task-list 搜索不到提交机绝对路径；
- Batch Manager 下载并校验 whole data-list；
- TDSQL row `record_version=3`；
- Batch active 时能观察到对应 v3 rows，terminal 后由产品自动清零，不用测试 SQL 删除；
- 两个 child 均能读取预期 task file 并成功；
- assignment 结束后 manager 临时 ZIP 被清理。

4 GiB 不作为日常 E2E payload；边界通过小限制参数化测试覆盖。DevCloud E2E 只验证
真实 Artifact、TDSQL、Manager 和 runtime bridge 的跨组件链路。另保留一个远程
Taiji Dataset E2E，确认 v2 + remote Range 路径未回归。

## 14. Harbor stacked 适配边界

本分支只完成以下 Harbor 适配面，不复制 `feat/habor_quick_support` 的旁路实现：

1. 在 `dataset_source_registry` 注册 `t3-harbor` factory。
2. Reader 复用 Harbor discovery/compiler，输出标准 `DatasetDescriptor` 和
   `DatasetItem`。
3. `iter_item_ids()` 只做 discovery，不提前 compile 全部任务。
4. `iter_items()` 按需 compile，并把每个 Harbor task 目录声明为
   `DatasetItem.files = TaskFilesSpec(type='directory', path=...)`。
5. 编译后的 `body` 不得把源 `task_dir` 作为远端执行依赖；Harbor task 若调用
   `harbor.Task(task_dir)`，应从 `ctx.task_files` materialize 自己需要的临时目录并清理。

完成这些后，`RunComposer`、Dataset inspection CLI 和
`submit batch --generation local` 不需要 Harbor 特判；最终 local directory 会自动
进入 task-list v3/data-list 链路。

`dataset fetch t3-harbor://...` 由通用 Reader exporter 写标准
`dataset.yaml` / `item.yaml` / `files.zip`；该 exporter 不是 Reader、不是本 MVP 的
data-list writer，也不引入 Sink Registry。

## 15. MVP 验收标准

只有同时满足以下条件，MVP 才算完成：

- Local/Taiji/Harbor 读取逻辑通过 `open_dataset()` 收敛，三个目标消费者不再自行分派；
- Reader contract tests 覆盖确定的 split、limit、order、duplicate 和生命周期语义；
- 无本地文件的 Compose 保持 task-list v2，无额外 Artifact；
- 含任意本地 directory/ZIP 的 Compose 自动生成 task-list v3 + NXDL v1 data-list；
- 最终 task-list、Control DB 和 TDSQL identity 中没有提交机的 task-files 绝对路径；
- 远程 ZIP 仍走 signed URL + Range；
- Batch Manager whole-download、TDSQL record v3、Instance Manager 临时 ZIP 和
  `LocalZipTaskFiles` 链路通过 unit/integration/DevCloud E2E；
- 256 MiB frame、256 MiB record、4 GiB data-list 都是可配置的完整结构 hard cap；
- 无 TDSQL 的 v3 在 child 创建前明确失败，v1/v2 fallback 不回归；
- v3 parent 的 Task Store rows 在 Control terminal 后自动、幂等、可重试地清理；
- 所有 success/failure/cancel/start-failure 路径都有 assignment 临时文件清理；
- 没有 Harbor publish、COS store、Sink、BlobTaskFiles 或 data-list Range 的额外实现。
