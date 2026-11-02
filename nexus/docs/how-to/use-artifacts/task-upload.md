# 任务制品上传指南

本文介绍如何在 Nexus 任务中把沙盒产出的文件（如 PPT、PDF、报告等）上传到制品服务，
并把制品的 canonical URI 返回到任务输出，供平台层和用户使用。

制品上传使用**函数式 API**（`nexus.utils.artifact.uploader`），只需要几个顶层函数
和一个纯数据对象 `RunFolder`——无状态、易于测试、导入即用。

---

## 概念说明

Nexus 的制品上传链路分为两层，职责明确分离：

```
┌───────────────────────────────────────┐
│      任务层（业务方维护）              │
│  - 决定生成什么文件                │
│  - 决定生成几个文件                    │
│  - 调用 uploader 函数完成上传           │
│  - 在 TaskRunOutput 中返回 artifact_uri │
└───────────────┬───────────────────────┘
                │ 调用
┌───────────────▼───────────────────────┐
│         制品服务层（平台方维护）         │
│  - uploader 函数（open_run_folder_*│
│    / upload_file / upload_bytes / …）  │
│  - ArtifactClient（HTTP 客户端）        │
│  - 腾讯云 COS（实际存储）               │
│  - Taihu 网关（鉴权）                   │
└───────────────────────────────────────┘
```

**任务层只需关心业务逻辑**，制品服务层对任务层完全透明：
不管生成什么格式的文件、生成几个、output 字段怎么定义，制品服务代码都不需要修改。

---

## 环境准备

Host 进程需要配置 Taihu PAT（太湖个人访问令牌），沙盒内的上传凭证由 Host 自动注入，
无需在沙盒环境变量里单独配置。

```bash
# 方式一：持久化到本地文件（推荐，配置一次永久生效）
mkdir -p ~/.nexus/auth
echo "tai_pat_xxxxxxx" > ~/.nexus/auth/taihu
chmod 600 ~/.nexus/auth/taihu

# 方式二：环境变量
export ARTIFACT_TAIHU_PAT=tai_pat_xxxxxxx
export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
```

---

## `TaskRunOutput` 的制品字段约定

`TaskRunOutput` 基类**有意不定义任何字段**（仅含 `model_config`）——保持核心零业务契约。
任务只要产出文件到制品服务，就按下面的**约定**在自己的 output 子类里显式声明字段：

```python
from nexus.core import TaskRunOutput

class MyTaskOutput(TaskRunOutput):
    # 制品字段：CLI、文档与平台层均按 `artifact_uri` 这一名称识别
    artifact_uri: str | None = None

    # 其余业务字段可随需添加
    filename: str
    error: str | None = None
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `artifact_uri` | `str \| None` | canonical `artifact:///<owner>/<path>` URI，例如 `artifact:///alice/tasks/2026/08/05/my-task-.../result.pptx`。可直接粘贴给 `nexus artifact download / info / url`。**这是任务作者唯一需要关心的制品字段** |

**约定要点**：

- **字段名固定**：`artifact_uri`。CLI 展示、平台层解析都按这个名字识别，字段更名将导致识别失败
- **字段是可选的**：默认 `None` 表示"这次没产出制品"
- **多文件**：命名 `artifact_uri` / `artifact_uri2` / `artifact_uri3` …，参考[场景二](#场景二多个文件同一文件夹)

> **不产出制品的任务**：无需声明该字段。

---

## uploader API 参考

上传相关的对外符号全部在 `nexus.utils.artifact.uploader`（也可以从 `nexus.utils.artifact`
顶层导入）。核心是一个数据对象 + 两个构造入口 + 两个上传函数 + 两个 URI 组装函数，**没有任何有状态的类**。

### 数据对象：`RunFolder`

```python
@dataclass(frozen=True, slots=True)
class RunFolder:
    folder_id: str    # 制品服务的 folder id（可以作为 parent_folder_id 传给 ArtifactClient）
    folder_path: str  # 裸路径，如 /tasks/2026/08/06/nexus-abc-20260806-085532
    owner: str        # 制品所有者 login，用于组装 canonical URI
```

不可变值对象，由 `prepare_run_folder` / `open_run_folder_from_runtime` 生成，作为参数传给上传/URI 函数。

### 构造入口

#### `open_run_folder_from_runtime(...)` — 沙盒内使用（推荐）

```python
from nexus.utils.artifact.uploader import open_run_folder_from_runtime

async with open_run_folder_from_runtime(
    dt=None,           # 覆盖时间戳（通常无需指定）
    base_url=None,     # 覆盖制品服务 URL（通常无需指定，默认读取 $ARTIFACT_SERVICE_URL）
    dev_employee=None, # 开发环境直连时传员工号
) as (client, folder):
    ...
```

`asynccontextmanager`：进入时通过 Socket.IO 向 Host 请求 Taihu PAT、打开 `ArtifactClient` 的
aiohttp session、创建 `/tasks/YYYY/MM/DD/<hostname>-<ts>/` 文件夹；退出时自动关闭 session。
folder 名自动取沙盒 `$HOSTNAME`（格式 `nexus-<32hex>`）加 UTC 时间戳。

#### `prepare_run_folder(client, *, dt=None)` — Host 侧使用（测试 / CLI）

```python
from nexus.utils.artifact import ArtifactClient
from nexus.utils.artifact.uploader import prepare_run_folder

async with ArtifactClient(taihu_pat='tai_pat_...') as client:
    folder = await prepare_run_folder(client, dt=None)  # 返回 RunFolder
```

只做"创建文件夹并返回 `RunFolder`"，`client` 生命周期由调用方管理。适合测试代码 / CLI 工具。

### 上传函数

#### `upload_file(client, folder, local_path, *, name=None, content_type=None)`

上传沙盒本地文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `client` | `ArtifactClient` | 打开的客户端 |
| `folder` | `RunFolder` | 目标 run 文件夹 |
| `local_path` | `str \| Path` | 文件路径（沙盒内的绝对路径） |
| `name` | `str \| None` | 覆盖制品名称，默认取文件 basename |
| `content_type` | `str \| None` | 覆盖 MIME 类型，默认按扩展名自动推断 |

返回 `ArtifactInfo`，关键字段：`artifact_id`（`str`，见[进阶章节](#进阶如果你的平台需要长期持久化制品)）、`name`（`str`）、`status`（`'ready'` 表示可用）。

#### `upload_bytes(client, folder, data, name, *, content_type=None)`

上传内存中的字节数据。

| 参数 | 类型 | 说明 |
|------|------|------|
| `client` | `ArtifactClient` | 打开的客户端 |
| `folder` | `RunFolder` | 目标 run 文件夹 |
| `data` | `bytes` | 原始字节内容 |
| `name` | `str` | 制品名称（必填） |
| `content_type` | `str \| None` | MIME 类型，默认自动推断 |

### URI 组装函数

#### `artifact_uri_for(folder, artifact_name)`

根据制品名称拼出 canonical URI（含 owner 段），可直接放入任务输出的 `artifact_uri` 字段：

```python
artifact_uri_for(folder, 'result.pptx')
# → 'artifact:///junlinyin/tasks/2026/08/06/nexus-090fb6f8...-20260806-085532/result.pptx'
```

#### `folder_uri(folder)`

本次 run 文件夹的 canonical URI（末尾带 `/`），可直接给 `nexus artifact ls` 或
`nexus artifact download` 使用——传给 `download` 时会**递归下载整个文件夹**。

```python
folder_uri(folder)
# → 'artifact:///junlinyin/tasks/2026/08/06/nexus-090fb6f8...-20260806-085532/'
```

---

## 完整示例：从零写一个上传任务（嵌套目录）

以下示例演示**在一个 run folder 内构造多层目录树**：根目录一个文件、两个子文件夹，其中一个子文件夹里还嵌套一层孙文件夹。任务返回**整个 run folder 的 URI**（`folder_uri`），下游拿到后可以一条 `nexus artifact download <uri>` 递归还原整棵目录树——这也是"文件夹下载"能力的典型验证场景。

上传逻辑说明：

- 顶层文件使用辅助函数 `upload_file(client, folder, path)`（隐式落入 run folder 根目录）。
- **子文件夹**用底层 `client.create_folder(name, parent_folder_id=...)` 显式创建；
  子文件夹内的文件用 `client.upload_file(path, parent_folder_id=<subdir>.folder_id)` 上传。
  这就是"上传函数只服务 RunFolder 根，深层结构走底层 client"的分层用法。

```python
# nexus/extensions/tasks/my_task/task.py
import logging
from pathlib import Path

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    require_context,
    task_registry,
)

logger = logging.getLogger(__name__)
OUTPUT_DIR = '/tmp/my_task_output'


# ── 沙盒内执行的函数 ──────────────────────────────────────────────────────────

def _generate_nested_report(base_dir: str, title: str) -> str:
    """在沙盒内生成一棵三层目录树，返回根目录路径。

    树形：
        base_dir/
          index.md
          chapter1/
            intro.md
            figures/
              diagram.txt
          chapter2/
            body.md
    """
    import os
    os.makedirs(f'{base_dir}/chapter1/figures', exist_ok=True)
    os.makedirs(f'{base_dir}/chapter2', exist_ok=True)

    with open(f'{base_dir}/index.md', 'w', encoding='utf-8') as f:
        f.write(f'# {title}\n\n- chapter1/intro.md\n- chapter1/figures/diagram.txt\n- chapter2/body.md\n')
    with open(f'{base_dir}/chapter1/intro.md', 'w', encoding='utf-8') as f:
        f.write('# 第一章 引言\n\n由 Nexus 任务自动生成。')
    with open(f'{base_dir}/chapter1/figures/diagram.txt', 'w', encoding='utf-8') as f:
        f.write('ascii-diagram-placeholder')
    with open(f'{base_dir}/chapter2/body.md', 'w', encoding='utf-8') as f:
        f.write('# 第二章 正文\n\n由 Nexus 任务自动生成。')
    return base_dir


async def _upload_tree_from_sandbox(base_dir: str) -> str:
    """在沙盒内把 base_dir 下的整棵目录树上传到制品服务，返回 folder_uri。

    - 使用辅助函数上传根目录文件（upload_file 仅负责 run folder 根目录）。
    - 子文件夹用底层 client.create_folder 显式创建。
    - 子文件夹内的文件用 client.upload_file(..., parent_folder_id=<子文件夹 id>)。
    """
    from nexus.utils.artifact.uploader import (
        folder_uri,
        open_run_folder_from_runtime,
        upload_file,
    )

    base = Path(base_dir)
    async with open_run_folder_from_runtime() as (client, run_folder):
        # 根目录文件
        await upload_file(client, run_folder, str(base / 'index.md'))

        # chapter1/及其内容
        ch1 = await client.create_folder('chapter1', parent_folder_id=run_folder.folder_id)
        await client.upload_file(str(base / 'chapter1' / 'intro.md'), parent_folder_id=ch1.folder_id)

        # chapter1/figures/ 及其内容（孙文件夹）
        ch1_fig = await client.create_folder('figures', parent_folder_id=ch1.folder_id)
        await client.upload_file(str(base / 'chapter1' / 'figures' / 'diagram.txt'), parent_folder_id=ch1_fig.folder_id)

        # chapter2/ 及其内容
        ch2 = await client.create_folder('chapter2', parent_folder_id=run_folder.folder_id)
        await client.upload_file(str(base / 'chapter2' / 'body.md'), parent_folder_id=ch2.folder_id)

        return folder_uri(run_folder)


# ── 任务输入 / 输出 ────────────────────────────────────────────────────────────

class MyTaskInput(TaskRunInput):
    title: str = '我的报告'
    dirname: str = 'report'  # 沙盒内落盘的目录名，OUTPUT_DIR/<dirname>/


class MyTaskOutput(TaskRunOutput):
    # 本任务返回的是"整个 run folder 的 URI"，方便调用方递归下载整棵目录树。
    # 字段名可以自选：这里用 folder_uri，表示这是一个文件夹级URI（末尾带 /）。
    folder_uri: str | None = None
    error: str | None = None


# ── 任务函数 ───────────────────────────────────────────────────────────────────

@task_registry.register('my_task')
async def my_task(task_input: MyTaskInput) -> MyTaskOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime()
    sandbox_base_dir = f'{OUTPUT_DIR}/{task_input.dirname}'

    # Step 1：沙盒内生成整棵目录树
    await runtime.invoke(_generate_nested_report, sandbox_base_dir, task_input.title)

    # Step 2：沙盒内上传（自动组建 /tasks/YYYY/MM/DD/nexus-xxx-<ts>/ + 3 层子结构）
    try:
        uri = await runtime.invoke(_upload_tree_from_sandbox, sandbox_base_dir)
    except Exception as exc:
        logger.exception('upload failed')
        return MyTaskOutput(error=str(exc))

    return MyTaskOutput(folder_uri=uri)
```

> **必须创建 `__init__.py`**
>
> 任务目录必须有 `__init__.py`，否则 Python 将其视为 namespace package，
> `task.py` 中的 `@task_registry.register` 装饰器不会执行，运行时报
> `EntryNotFoundError: Entry 'my_task' not found`。
>
> ```python
> # nexus/extensions/tasks/my_task/__init__.py
> from nexus.core import Registry
>
> Registry.import_module('nexus.extensions.tasks.my_task.task')
> ```

对应的运行配置：

```yaml
# 将以下内容保存为 tests/e2e/config/my_task_gongfeng_example.yaml
# （参考已有示范：tests/e2e/config/artifact_output_gongfeng_example.yaml）
imports:
  - nexus.extensions.tasks.my_task

task_name: my_task
task_config:
  title: 季度报告
  dirname: q3_report

runtime_provider_name: gongfeng
runtime_provider_config:
  api_base_url: ${oc.env:GONGFENG_API_BASE_URL,"http://hyrl-sandbox.prod.woa.com:8052"}
  # 留空时自动读 ~/.nexus/auth/gongfeng/key
  token: ${oc.env:GONGFENG_TOKEN,""}
  resource_spec:
    cpu: 2
    memory: 4

runtime_configs:
  main:
    image: mirrors.tencent.com/hunyuan_yanguan/ubuntu:24.04
    worker_venv: true
    environment:
      # 沙盒只需知道服务地址；Taihu PAT 不注入沙盒环境，
      # 而是由 Host 通过 shared-value 机制按需下发（见「环境准备」）。
      ARTIFACT_SERVICE_URL: ${oc.env:ARTIFACT_SERVICE_URL,"https://artifact-nexus-agent.mcp.woa.com"}
```

运行：

```bash
uv run --frozen nexus run-single tests/e2e/config/my_task_gongfeng_example.yaml
```

输出示例：

```json
{
  "folder_uri": "artifact:///junlinyin/tasks/2026/08/06/nexus-090fb6f8...-20260806-085532/",
  "error": null
}
```

## 验证文件夹下载

拿到 `folder_uri` 后，直接递归下载整棵目录树到本地：

```bash
# -o 指定落地目录；不带 -o 会默认落至 ./<folder-name>/
nexus artifact download artifact:///junlinyin/tasks/2026/08/06/nexus-090fb6f8...-20260806-085532/ -o ./my-report
```

下载完成后，`./my-report/` 中应能看到还原后的三层目录结构：

```
./my-report/
  index.md
  chapter1/
    intro.md
    figures/
      diagram.txt
  chapter2/
    body.md
```

也可以只挑单文件（走 canonical URI 直达）：

```bash
nexus artifact download artifact:///junlinyin/tasks/2026/08/06/nexus-090fb6f8...-20260806-085532/chapter1/figures/diagram.txt -o ./diagram.txt
```


---

## 常见场景

### 场景一：单文件上传

最常见情况：output 只声明一个 `artifact_uri` 字段。

```python
from nexus.core import TaskRunOutput

class MyOutput(TaskRunOutput):
    artifact_uri: str | None = None
    error: str | None = None


async def _upload(path):
    from nexus.utils.artifact.uploader import (
        artifact_uri_for,
        open_run_folder_from_runtime,
        upload_file,
    )
    async with open_run_folder_from_runtime() as (client, folder):
        art = await upload_file(client, folder, path)
        return artifact_uri_for(folder, art.name)
```

结果目录：
```
/tasks/2026/08/05/my-task-XXXXXX/
  result.pptx
```

---

### 场景二：多个文件，同一文件夹

同一次 `open_run_folder_from_runtime()` 上下文内多次调用 `upload_file`，所有文件落入同一个 run 文件夹。
业务方可按主/次自定义命名：

```python
from nexus.core import TaskRunOutput

class MyOutput(TaskRunOutput):
    artifact_uri: str | None = None       # 主文件
    artifact_uri2: str | None = None      # 副文件（命名可任意，此处沿用编号后缀以便对应）


async def _upload_two(path1, path2):
    from nexus.utils.artifact.uploader import (
        artifact_uri_for,
        open_run_folder_from_runtime,
        upload_file,
    )
    async with open_run_folder_from_runtime() as (client, folder):
        art1 = await upload_file(client, folder, path1)
        art2 = await upload_file(client, folder, path2)
        return {
            'artifact_uri': artifact_uri_for(folder, art1.name),
            'artifact_uri2': artifact_uri_for(folder, art2.name),
        }
```

结果目录：
```
/tasks/2026/08/05/my-task-XXXXXX/
  slides.pptx          ← artifact_uri
  report.pdf           ← artifact_uri2
```

---

### 场景三：多个文件，分两个文件夹

两次顺序进入 `open_run_folder_from_runtime()` 上下文，每次创建一个独立文件夹（folder 名取 hostname + 调用时的 UTC 时间戳）。
只要两次调用间隔超过 1 秒，便会落入不同文件夹；否则将被视为同一秒，建议直接采用场景二（同一 folder）。

```python
async def _upload_two_folders(path1, path2):
    from nexus.utils.artifact.uploader import (
        artifact_uri_for,
        open_run_folder_from_runtime,
        upload_file,
    )

    async with open_run_folder_from_runtime() as (client1, folder1):
        art1 = await upload_file(client1, folder1, path1)
        uri_1 = artifact_uri_for(folder1, art1.name)

    async with open_run_folder_from_runtime() as (client2, folder2):
        art2 = await upload_file(client2, folder2, path2)
        uri_2 = artifact_uri_for(folder2, art2.name)

    return {'artifact_uri': uri_1, 'artifact_uri2': uri_2}
```

结果目录：
```
/tasks/2026/08/05/
  nexus-xxx-20260805-103000/    ← 第一次调用的文件夹
    slides.pptx
  nexus-xxx-20260805-103002/    ← 第二次调用（2 秒后）
    report.md
    report.pdf
```

---

### 场景四：上传内存中生成的内容（无需写临时文件）

若生成内容可直接获得字节数据（如动态生成的 JSON / HTML），可跳过写文件步骤：

```python
async def _upload_bytes_from_sandbox(data: bytes, name: str) -> str:
    from nexus.utils.artifact.uploader import (
        artifact_uri_for,
        open_run_folder_from_runtime,
        upload_bytes,
    )
    async with open_run_folder_from_runtime() as (client, folder):
        art = await upload_bytes(client, folder, data, name=name, content_type='application/json')
        return artifact_uri_for(folder, art.name)
```

---

## 获取下载链接

`artifact_uri` 即任务作者唯一需要提供给用户 / 平台的凭证——直接传入 CLI 或 SDK 即可获取文件。

### 下载单个文件

```bash
# CLI 方式（最常用）
uv run nexus artifact url artifact:///junlinyin/tasks/2026/08/05/my-task-XXXXXX/result.pptx
uv run nexus artifact download artifact:///junlinyin/tasks/2026/08/05/my-task-XXXXXX/result.pptx
```

```python
# SDK 方式
async with ArtifactClient(taihu_pat='tai_pat_...') as client:
    artifact_id = await client.resolve_resource_id(
        'artifact:///junlinyin/tasks/2026/08/05/my-task-XXXXXX/result.pptx', expect='artifact'
    )
    dl = await client.create_download_url(artifact_id)
    print(dl.url)   # 预签名 URL
```

> **提示**：预签名 URL 有效期约 300 秒。平台层展示下载按钮时，建议在用户点击时**实时生成** URL，请勿将 URL 存入数据库。保存 `artifact_uri` 即可，随时可重新签发。

### 下载整个 run 文件夹（递归）

当 `path` 指向一个文件夹时，`nexus artifact download` 会**自动切换到递归模式**，把文件夹的全部内容镜像到本地目录，逐级还原子目录结构（参照 `rsync -a src/ dest/`）。

直接把 `folder_uri(folder)`（末尾带 `/` 的 canonical URI）或文件夹路径传给 `download` 即可：

```bash
# 递归下载整个 run 文件夹至 ./run-output/
nexus artifact download artifact:///junlinyin/tasks/2026/08/05/my-task-XXXXXX -o ./run-output

# 不带 -o：默认落至 ./<folder-name>/
nexus artifact download artifact:///junlinyin/tasks/2026/08/05/my-task-XXXXXX

# 提高并发数（默认 4），加速大量小文件的下载
nexus artifact download artifact:///junlinyin/tasks/2026/08/05/my-task-XXXXXX -o ./run-output --concurrency 8
```

```python
# SDK 方式：递归下载到本地目录
async with ArtifactClient(taihu_pat='tai_pat_...') as client:
    folder_id = await client.resolve_resource_id(
        'artifact:///junlinyin/tasks/2026/08/05/my-task-XXXXXX', expect='folder'
    )
    successes, failures = await client.download_folder_to_dir(folder_id, './run-output', concurrency=4)
    print(f'downloaded {successes} file(s), {len(failures)} failed')
```

**失败处理**：递归模式下单文件失败不会中断整批，所有失败汇总到末尾日志；若有任何失败命令以非零退出码退出。

---

## 进阶：如果你的平台需要长期持久化制品

绝大多数任务只需要 `artifact_uri` —— 它本身包含 owner 和路径，能被 CLI / SDK 直接消费。

只有当你的**平台层**需要满足下面这类需求时，才需要额外持久化 `artifact_id`：

- 用户可能在数周或数月后再次下载，而 URI 中的路径 / owner 存在被更名的可能
- 大规模数据库外键，希望用一个短稳定 ID 而不是长 URI
- 需要频繁调用 `ArtifactClient.create_download_url(artifact_id)` 大量重签发（省去每次都要先 `resolve_resource_id` 沿文件夹树解析 URI 的开销）

这时output 里可以额外声明一个 `artifact_id: str | None = None` 字段，并把它从终端日志里隐藏（`artifact_id` 为内部句柄，对人阅读无意义）：

```python
from typing import ClassVar
from nexus.core import TaskRunOutput

class MyTaskOutput(TaskRunOutput):
    artifact_uri: str | None = None
    artifact_id: str | None = None      # 长期存档用
    filename: str

    # `__pformat_exclude__` 为通用机制：可用于隐藏任何不希望在终端日志中展示的字段
    __pformat_exclude__: ClassVar[frozenset[str]] = frozenset({'artifact_id'})
```

配套写法：

```python
async def _upload_and_return_both(path) -> dict[str, str]:
    from nexus.utils.artifact.uploader import (
        artifact_uri_for,
        open_run_folder_from_runtime,
        upload_file,
    )
    async with open_run_folder_from_runtime() as (client, folder):
        art = await upload_file(client, folder, path)
        return {
            'artifact_id':  art.artifact_id,
            'artifact_uri': artifact_uri_for(folder, art.name),
        }
```

`__pformat_exclude__` 只影响 `nexus run-single` / `run-batch` 终端 log 的可读性；`--output` JSON、`model_dump()`、程序化调用都能完整看到 `artifact_id`。

---

## 参考实现

- uploader 函数式API 源码：`nexus/utils/artifact/uploader.py`
- `TaskRunOutput`（零契约的输出基类）：`nexus/core/task.py`
- 完整示范任务：`nexus/extensions/tasks/artifact_output/task.py`
- 示范任务配置：`tests/e2e/config/artifact_output_gongfeng_example.yaml`
- E2E 测试脚本：`tests/e2e/artifact/run_uploader_e2e.py`
