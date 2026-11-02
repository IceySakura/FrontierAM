# Dataset CLI 使用指南

首次使用 JSONL/COS 发布功能，先阅读 [Dataset publish 使用指南](dataset-publish.md)，了解输入格式、导入规则和运行配置的区别。

`nexus dataset` 用于本地数据集编写、远程数据集查看、文件缓存和发布。数据集 item 存储的是
partial `RunInput`，可直接通过 `nexus run single --data ... --id ...` 或
`nexus run batch --data ...` 参与 run compose。

## 快速开始

```bash
# 创建一个本地 terminal 数据集模板
nexus dataset init ./workspace/datasets/my_terminal

# 查看本地数据集
nexus dataset show ./workspace/datasets/my_terminal
nexus dataset items ./workspace/datasets/my_terminal
nexus dataset get ./workspace/datasets/my_terminal case-001

# 首次使用 namespace 时创建权限记录；creator 由当前 Wedata 身份确定
nexus dataset namespace create nexus-test --admin your-name

# 上传并发布到你有权限的 namespace
nexus dataset publish ./workspace/datasets/my_terminal nexus-test/my_terminal@v1
```

## 从 JSONL 和附件发布到 COS

普通业务 JSONL 可以直接发布，不必提供 `--import-spec`。默认将整条记录放进
`body.task_config`，保留业务字段；有字符串 `id` 时沿用，没有 `id` 时生成
`line-00000001` 等行号 ID，并在 metadata 记录源文件 SHA256 和行号。
已存在但为空、null 或非字符串的 ID，以及重复 ID，都会报错。

例如每行输入：

```json
{"instance_id":"case-01","prompt":"完成任务","input_file":"case-01/abcd","bundle":"case-01"}
```

```bash
nexus dataset publish ./tasks.jsonl research/demo@v1 \
  --id-field instance_id --files-field bundle --files-to-cwd \
  --backend cos --store cos://example-bucket/datasets --region ap-guangzhou \
  --json > published.json

# 也可以先导入为标准本地 Dataset
nexus dataset import ./tasks.jsonl --id-field instance_id --files-field bundle \
  --files-to-cwd --output ./converted --json
```

`--id-field` 指定后，该字段必须存在且为非空字符串；即使 import-spec 设置了
`generate_ids: true`，也不会为缺失字段生成 ID。ID 字段保留在业务 `task_config` 中。
`--files-field` 将指定字段提取为附件声明，不保留在 `task_config` 中，路径相对 JSONL
解析，自动识别目录或 ZIP。没有附件时省略 `--files-field` 和 `--files-to-cwd`；
不指定附件字段时，业务中的路径字符串不会被自动打包。

两个字段参数都只读取顶层字段名，不解析点号路径，且不能指向同一个字段。
优先级为 CLI 显式参数、`--import-spec`、内置默认规则；覆盖附件字段名时保留 spec 中
明确设置的类型和 root_name_field。新指定的附件字段默认自动识别类型。

标准 JSONL 若包含 ID 字段和对象类型的 `body`，且其余字段只有 `files/merge/metadata`（以及显式
指定的 ID/附件字段），继续按 DatasetItem 解析。其他记录按业务输入包装，不丢弃额外字段。
业务记录恰好与标准 envelope 同形时，用 `record_kind: task_input` 明确指定；已有 RunInput
使用 `record_kind: run_input`。显式 import-spec 未写 record_kind 时仍按原有 dataset_item 模式处理。

发布时可以没有 `task_name`，运行时由 `--use` Profile 或其他 Compose 参数补齐。最终
RunInput 仍要求 task_name；publish 不推断执行任务，也不把缺失任务名写为 null。

需要更完整的转换规则时，仍可使用 import-spec，示例如下。

每行 JSONL 是一个任务记录。附件使用一个目录或一个 ZIP；例如：

```text
tasks.jsonl
import.yaml
case-01/abcd
```

```json
{"id":"case-01","input_file":"case-01/abcd","bundle":"case-01"}
```

转换配置 `import.yaml`：

```yaml
schema_version: 1
record_kind: task_input
task_name: my_task
id_field: id
split: test
files:
  path_field: bundle
  type: directory
compat:
  files_to_cwd: true
```

`task_input` 将业务字段放入 `task_config`，保留业务 ID，`bundle` 提取为附件声明。
已有 RunInput 使用 `record_kind: run_input`，已有 `id/body/files` 使用 `dataset_item`。
没有业务 ID 时可设置 `generate_ids: true`，生成行号 ID，并在 metadata 记录源文件摘要和行号。
附件路径相对 JSONL 所在目录解析；普通 ZIP 会转为 Nexus ZIP，已声明的远端 ZIP 会下载、校验并复制到发布目标。

```bash
# 密钥由 shell 或 CI 注入，无需注册密钥或创建 data-profile
export COS_SECRET_ID='<secret-id>'
export COS_SECRET_KEY='<secret-key>'
# 临时凭据额外设置 COS_TOKEN

nexus dataset publish ./tasks.jsonl research/demo@v1 \
  --import-spec ./import.yaml --backend cos \
  --store cos://example-bucket/datasets --region ap-guangzhou \
  --json > published.json

# 如需先检查转换结果，可单独导入为标准本地 Dataset
nexus dataset import ./tasks.jsonl --import-spec ./import.yaml -o ./converted --json
```

COS 中的 `research/demo@v1` 是发布路径与版本，不需要创建 Taiji namespace。
COS 发布目前要求桶未启用过版本控制，以保证禁止覆盖；同版本同内容可重复发布，不同内容报冲突。
`published.json` 的 `data_uri` 包含 region 和 Manifest SHA256，引用时保留完整字符串。
发布成功后无需保留原始 JSONL 或附件目录。

云端运行时，在新的提交环境提供读取凭据，并沿用 Cloud 的代码提交方式：

```bash
nexus submit batch --generation remote \
  --data "$(jq -r '.data_uri' published.json)" \
  --use ./run-profile.yaml --environment '<environment>'
```

运行 Profile 配置代码 `imports`、Agent、模型等运行参数，不配置数据密钥。
submit 从本地 env 获取凭据，随本次请求传递，云端用于生成和任务附件读取；不会写入 Dataset 或 Task List。
临时 token 到期后需使用新凭据重新提交。

开启 `files_to_cwd` 后，任务启动前得到 `pwd/case-01/abcd`，已有 `open('case-01/abcd')` 可直接工作。
也可以在 import/publish 时加 `--files-to-cwd`。目录名必须在同一 pwd 的并发任务间唯一；冲突会报错，任务结束后清理本次创建的目录。
ZIP 没有可推断的顶层目录时，可以在转换配置中设置 `files.root_name_field: id`，从记录字段指定落地目录名。
不启用该模式时，通过 `ctx.task_files.read_text('abcd')` 或 `download()` 按需访问。
需要在 sandbox 的 pwd 展开时，在运行 Profile 设置
`runner.task_files_materialization: {mode: cwd, target: 'runtime:main'}`。

发布到 Taiji 使用同一套转换配置和 `--backend taiji`，目标 namespace 仍需已有写权限。
云端 Taiji 读取使用 submit 环境中的 `WEDATA_KEY` JSON；原有 `--wedata-key-file` 入口继续兼容。
Harbor 可以直接发布 `t3-harbor:///path/to/tasks`；其运行 Profile 需要选择 `harbor_bridge_task`、加载对应模块，并配置 Runtime/Agent。

## URI 格式

远程数据集 URI 支持以下形式：

| 输入 | 解析结果 |
|------|----------|
| `swebench_verified` | `nexus-curated/swebench_verified@latest` |
| `swebench_verified@v1.0.0` | `nexus-curated/swebench_verified@v1.0.0` |
| `research1/swebench_verified` | `research1/swebench_verified@latest` |
| `taiji.nexus.research1/swebench_verified@v1.0.0` | `research1/swebench_verified@v1.0.0` |
| `cos://bucket/path/manifest.json?region=ap-guangzhou#sha256=<digest>` | 固定内容的 COS 发布引用 |

也可以通过 `?split=test` 指定 split，或者在命令中使用 `--split test`。

## 命令

### `nexus dataset init`

创建本地 authoring 数据集。

```bash
nexus dataset init <path> [--name NAME] [--force]
```

默认创建 `terminal` 模板，并生成 `case-001`、`case-002`、`case-003` 三个 item。
如果需要其它模板，可传 `--template generic|swe`；如果只想创建一个指定 id 的 item，可传
`--id ITEM_ID`。

生成的目录包含：

```text
dataset.yaml
items/case-001/item.yaml
items/case-001/files/
items/case-002/item.yaml
items/case-002/files/
items/case-003/item.yaml
items/case-003/files/
```

`terminal` 模板的 item 使用 `id/files/body/merge/metadata` schema，`body` 是 partial `RunInput`。
`files.path` 指向本地 `files` 目录；compose 时会转换成 `RunInput.task_files`。支持 task files 的
任务代码可通过 `ctx.task_files` 主动读取文件。现有 terminal-bench converter 仍保留
`task_config.task_dir` 的旧运行路线。

### `nexus dataset show`

查看本地或远程数据集元信息。

```bash
nexus dataset show ./workspace/datasets/my_terminal
nexus dataset show swebench_verified --json
```

`show` 对 Local、Taiji 和扩展 Source 统一输出类型化 Descriptor，包含
`reference/resolved_reference/provider/name/version/default_split/splits/metadata`。Taiji 的
`latest/dev` 会在 `resolved_reference` 中显示为实际版本；Local 和 Harbor 的
`metadata.consistency` 为 `live`。

### `nexus dataset versions`

查看版本列表。只有实现版本目录能力的 Provider（当前为 Taiji）支持该命令；Local、Harbor
等 live Reader 会明确报告不支持，请使用 `dataset show` 查看当前引用和一致性信息。
版本目录通过可选 `DatasetVersionCatalog` Protocol 暴露，不属于 `DatasetReader` 核心契约。

```bash
nexus dataset versions swebench_verified
```

### `nexus dataset items` / `nexus dataset list`

默认轻量列出 item id，不加载完整 Item body。使用 `--details` 时会读取完整 Item，并在 JSON
结果中包含 `metadata/files`。`list` 不传 dataset 时列出远程 UC 资产。

```bash
nexus dataset items swebench_verified --limit 5
nexus dataset list --keyword swebench --limit 20
nexus dataset list ./workspace/datasets/my_terminal --json
nexus dataset list ./workspace/datasets/my_terminal --details --json
```

### `nexus dataset get`

输出一个 item 的 JSON。

```bash
nexus dataset get nexus-bench/cobol_demo_0702@v1 cobol-modernization
nexus dataset get ./workspace/datasets/my_terminal case-001 -o /tmp/case-001.json
```

### `nexus dataset fetch`

将 Taiji 远程 dataset 或已注册的 Dataset Source 物化为标准本地布局。默认会导出所有 split；使用
`--split` 时只导出指定 split。fetch 会保存 item YAML，并把每个 item 的 files 完整写成本地
`files.zip`，输出目录本身可作为本地 dataset 传给 `--data` 使用。`--limit 0` 表示生成所选
split 的空 dataset；负数会在访问 Provider 前拒绝。

```bash
nexus dataset fetch swebench_verified --id astropy__astropy-12907 --json
nexus dataset fetch swebench_verified --limit 5
nexus dataset fetch swebench_verified -o ./workspace/datasets/swebench_verified
nexus dataset fetch 't3-harbor:///absolute/path/to/tasks?flat=true' \
  -o ./workspace/datasets/harbor-local
```

缓存路径默认是 `~/.nexus/datasets/<namespace>/<dataset>/<version>/`。发布后的
dataset version 不可变，重复 fetch 时已存在且校验匹配的 `files.zip` 会直接复用。使用
`-o/--output` 时，会把本地 dataset 写到指定目录；对 Source URI 该参数是必需的，且非空目录
需要 `--force`。该目录可以直接传给 `--data`。

### `nexus dataset namespace`

查看、创建和更新 Nexus namespace 的数据集创建权限：

```bash
# 查询；不存在时输出 exists: false
nexus dataset namespace show research

# 创建 namespace 权限记录
nexus dataset namespace create research --admin alice --admin bob

# 完整替换管理员列表
nexus dataset namespace update research --admin alice --admin carol
```

namespace 权限记录的 schema 固定为 `nexus`。创建人 `creator` 由 UC 根据当前 TAuth 身份确定，不能通过
命令行指定；只有 creator 可以更新记录。`update` 的 `--admin` 是完整覆盖，不是增量添加，因此需要把
希望保留的管理员全部传入。三个命令都需要 Wedata 凭证。

namespace 管理员控制谁可以在该 namespace 下创建数据集资产。它与 `dataset publish --admin` 指定的
单个数据集资产管理员是两层不同的权限，不会互相替代。

### `nexus dataset publish`

将本地数据集构建成平台标准文件布局，并上传、commit、publish。

```bash
nexus dataset publish <path> <namespace/dataset@version> [options]
nexus dataset publish <path> --namespace <namespace> --version v1.0.0 [options]
```

推荐使用目标 URI 语法，例如 `research/my_terminal@v1.0.0`。发布时必须明确指定
namespace；裸 `dataset@version` 不允许用于发布，避免误传到官方维护空间。
namespace 必须已通过 `dataset namespace create` 注册，且当前用户需要在 namespace 的管理员列表中；
发布命令不会隐式创建或认领 namespace。
上传前会先查找 UC Nexus asset；如果不存在，会自动注册后继续发布。自动注册的创建人默认使用
Wedata key 的 `subject`，也可以用 `--creator` 和 `--admin` 指定。

UC 资产接口可能以 HTTP 200 + `Response.Error` 返回“namespace 未注册”或“无权创建资产”等业务错误；
CLI 会将具体错误码和服务端消息原样展示。namespace 权限管理接口则使用普通的 HTTP
400/401/404/409 状态码。

常用参数：

| 参数 | 说明 |
|------|------|
| `<namespace/dataset@version>` | 推荐的发布目标写法，例如 `research/my_terminal@v1.0.0` |
| `--namespace` | 远程 namespace；不使用目标 URI 时必填 |
| `--name` | 远程 dataset 名称，默认来自 `dataset.yaml` 的 `name` |
| `--version` | 发布版本；目标 URI 未包含 `@version` 时必填 |
| `--split` | 只发布指定 split |
| `--import-spec` | 可选 JSONL 转换规则；省略时使用默认业务记录/标准 DatasetItem 识别 |
| `--id-field` | JSONL 顶层 ID 字段；显式指定后必须存在，覆盖 import-spec 的 ID 与生成规则 |
| `--files-field` | JSONL 顶层附件路径字段；提取为目录/ZIP 附件 |
| `--output` | 构建产物目录，默认 `<dataset>/_nexus_out`；目录非空时会自动覆盖 |
| `--snapshot` | dev snapshot 名称，默认 `dev-<UTC timestamp>` |
| `--with-parquet` | 生成 parquet；缺少 `pyarrow` 时会按需自动安装 |
| `--no-parquet` | 跳过 parquet；默认行为 |
| `--build-only` | 只构建本地产物，不上传 |
| `--upload-concurrency` | 上传文件并发数，默认 `5` |
| `--creator` / `--admin` | 自动注册 UC 资产时使用的创建人和管理员 |

构建产物位于 `<output>/files`，包含：

```text
<split>/raw/<item_id>.json
<split>/jsonl/<split>.jsonl
<split>/jsonl/<split>.jsonl.zip
<split>/parquet/<split>.parquet
<split>/blobs/<item_id>.zip
_meta/version.json
_meta/upload_manifest.json
```

默认不生成 parquet，便于在没有 `pyarrow` 的环境做本地结构检查；需要生成时显式使用
`--with-parquet`。

发布流程为：

```text
构建 files/ -> 按文件选择 uploads:sign 或 uploads:sign-multipart -> 上传 -> commit snapshot -> publish release version
```

### Python SDK 统一上传接口

`nexus.dataset.DatasetClient` 提供异步的统一文件上传接口。本地小文件和大文件使用相同调用方式，SDK 会根据
文件大小自动选择普通 COS PUT 或 multipart upload：

```python
import asyncio

from nexus.dataset import DatasetClient


async def main():
    async with DatasetClient() as client:
        upload = await client.upload_file(
            'research',
            'swebench_verified',
            'dev-202607240001',
            './large dataset.parquet',
        )
        print(upload)


asyncio.run(main())
```

远端文件名默认使用 `local_path` 的 basename；上例会上传为 `large dataset.parquet`。返回值保持为 IO 服务签名
接口的 `data` 字典。`upload_file` 只上传单个对象，不会自动 commit 或 publish snapshot；上传数据集标准布局时，
继续使用 `nexus dataset publish` 完成注册、上传、commit 和 publish 全流程。

SDK 使用保守的内部边界提前切换 multipart，避免贴近 COS 单次 PUT 上限。该边界属于实现细节，调用方不应依赖
具体数值。`nexus dataset publish` 内部也使用相同的自动选择逻辑。高级场景可以通过 `part_size`、
`max_concurrency` 和 `expires` 调整分片大小、并发数和预签名 URL
有效期；SDK 会校验 COS 的 part 大小限制，并在超大文件可能超过最多 part 数时自动提高分片大小。

上传行为和注意事项：

- 上传前会检查路径存在且是普通文件；上传期间文件大小、mtime 或 inode 变化会使操作失败。
- 单次 PUT 和各个 part 会对超时、限流和服务端临时错误进行重试；part 上传是按 part number 幂等覆盖的。
- multipart 的 part 或 complete 失败、任务取消时，SDK 会尽力调用预签名 abort URL 清理未完成上传。
- multipart ETag 是 COS 返回的 opaque 标识，不会被当作整个文件的 MD5；数据集完整性仍由 manifest 的
  SHA-256 和 commit 阶段校验。
- 需要有目标数据集的写权限和可用的 Wedata/TAuth 配置。COS 请求使用 IO 服务签发的 URL，不需要在客户端
  配置 COS 密钥。大文件应设置足够长的 `expires`，保证所有 part 和 complete 请求能在 URL 失效前完成。
- 普通 PUT 成功后若本地文件变化，或者后续 commit/publish 失败，远端对象可能已经存在。当前服务端没有提供
  对应的删除签名；可修复本地文件后用同一 snapshot 和文件名重传覆盖，或使用新的 snapshot 重试。

原有低层接口 `DatasetIoClient.sign_uploads(...)` 和
`DatasetIoClient.upload_file(signed_url, data, ...)` 保持兼容，适合需要批量签名或显式控制远端相对路径的代码。

## 平台环境变量

远程读取和发布使用 Taiji dataset IO 与 UC 服务。下面这些环境变量都不是必须的；未设置时会使用代码默认值。
Wedata key 从 `https://wedata.woa.com/security/user/keys` 获取，可以放进环境变量，也可以保存为本地文件。

| 环境变量 | 未设置时默认值 | 说明 |
|----------|----------------|------|
| `TAIJI_DATASET_SERVER_URL` | `http://taiji-nexus-dataset-io-dev.woa.com:8080` | Dataset IO 服务地址 |
| `UC_BASE_URL` | `http://taiji-nexus-dataset-uc-dev.woa.com:8080` | UC metadata 服务地址 |
| `WEDATA_KEY` | 继续尝试 `WEDATA_KEY_FILE`，再尝试 `~/.nexus/auth/wedata/key` | Wedata key 文件原文 |
| `WEDATA_KEY_FILE` | `~/.nexus/auth/wedata/key` | Wedata key 本地文件路径 |

## Run Compose 示例

```bash
nexus run single \
  --data swebench_verified \
  --id astropy__astropy-12907 \
  --use docker \
  --use openai
```

```bash
nexus run batch \
  --data swebench_verified \
  --use gongfeng \
  --use claude_code \
  --use litellm \
  --limit 2 \
  -d workspace/compose-smoke.db
```

命令行顺序就是 merge 顺序。后出现的 `--use` 或 `--set` 覆盖前面的同一路径：

```bash
nexus run single --data ./workspace/datasets/my_terminal --id case-001 \
  --set runtime_configs.main.worker_venv=false \
  --use worker_venv
```

这里最终以内置 `worker_venv` 为准；反过来把 `--set` 放最后，则以 `false` 为准。

### 扩展 DatasetReader

Nexus 通过 `open_dataset()` 将 Local、Taiji 和扩展 URI 统一解析为只读
`DatasetReader`。Reader 提供类型化 `describe()`、轻量 `iter_item_ids()` 和流式
`iter_items()`；`get/load/list` 是建立在流式接口上的辅助函数。

扩展模块可以通过 Source Registry 注册 URI scheme，Factory 返回统一 Reader，将现有外部
目录按需适配为 `DatasetItem`，无需预生成 `dataset.yaml`。例如安装 Harbor Bridge 后：

```bash
nexus dataset show 't3-harbor:///absolute/path/to/tasks?flat=true'
nexus dataset items 't3-harbor:///absolute/path/to/tasks?flat=true' --limit 20
nexus dataset get 't3-harbor:///absolute/path/to/tasks?flat=true' example-task
```

运行：

```bash
nexus run batch \
  --data 't3-harbor:///absolute/path/to/tasks?flat=true' \
  --use harbor_bridge \
  --use gongfeng \
  --use claude_code \
  --use litellm \
  --limit 20 \
  -d workspace/t3-harbor.db
```

Core 只按 URI scheme 从 `dataset_source_registry` 加载 Reader Factory；具体格式解析由扩展
模块实现。扩展 Reader 当前支持 `show/items/get/fetch` 和直接参与 Run Compose。`fetch` 使用
通用 Reader exporter 写出 `dataset.yaml`、`items/<split>/<id>/item.yaml` 与 `files.zip`，不会
调用 publish 或 COS。Harbor Reader 在 task package 接入可移植 `task_files` 前，会明确拒绝
cloud task-list materialization；先执行 `dataset fetch` 得到本地标准 Dataset，再提交即可。

### 在 `run-batch` generator 中使用 dataset

如果只是把 dataset 的每个 item 批量跑起来，优先使用 `nexus run batch --data ...`。当需要在
generator 里做自定义过滤、补充 metadata、按 item 动态改配置时，可以在 generator 中读取 dataset
item，再把 item 的 `body` 和 `files` 转成 `RunInput`。

```python
import asyncio

from nexus.dataset.client import DatasetClient
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


DATASET = 'nexus-test/terminal-bench-2@v3'


async def _load_items():
    async with DatasetClient() as client:
        return await client.load_items(DATASET, split='test', limit=20)


def _run_input_from_item(item):
    body = dict(item.body)
    task_files = item.resolved_task_files()
    if task_files is not None:
        body['task_files'] = task_files.model_dump(exclude_none=True)
    return RunInput.from_dict(body)


def generate_configs():
    for item in asyncio.run(_load_items()):
        yield BatchItem(
            run_id=item.id,
            config=_run_input_from_item(item),
            metadata={'dataset': DATASET, 'item_id': item.id, **item.metadata},
        )
```

然后用普通 generator 入口运行：

```bash
nexus run-batch \
  -g ./workspace/my_dataset_generator.py \
  -d ./workspace/my_dataset_batch.db \
  --max-concurrent 20
```

## 本地数据集结构

`dataset.yaml` 示例：

```yaml
schema_version: 1
name: my_terminal
authors:
  - name: your-name
    email: your-name@example.com
maintainers:
  - name: maintainer-name
    email: maintainer-name@example.com
default_split: test
splits:
  test:
    items:
      - items/case-001/item.yaml
      - items/case-002/item.yaml
      - items/case-003/item.yaml
```

`item.yaml` 示例：

```yaml
id: case-001
files:
  type: directory
  path: files
body:
  task_name: terminal_task
  task_config:
    message: Solve the task described in task_files/instruction.md.
    run_oracle: false
  runtime_configs:
    main:
      image: ubuntu:24.04
      working_dir: /workspace
      resource_spec:
        cpu: 2
        memory: 8
```
