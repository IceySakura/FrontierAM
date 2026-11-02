# Artifact CLI 使用指南

`nexus artifact` 用于把本地文件上传到 artifact 服务、在文件夹树中组织产物、生成下载链接，
以及把产物以只读方式共享给其他同事。artifact 服务把每个产物（artifact，即一个文件）放在文件夹
（folder）树下，上传和下载都通过预签名的腾讯云 COS URL 完成，CLI 只是
`nexus.utils.artifact.ArtifactClient` 的一层薄封装。

> **两种寻址形式**：所有命令都接受两种可互换的资源引用。
>
> * **裸路径**（如 `/models/v1/model.bin`）：始终指向调用者**自己**名下的资源，路径前导 `/` 可省略，`/a/b` 与 `a/b` 等价。
> * **`artifact:///<owner>/<path>` URI**：显式指定属主，可以是你自己（`artifact:///<你自己的 login>/...`）也可以是别人共享给你的资源（`artifact:///alice/models/model.bin`）。这样即使两位不同员工都授权给你一份同名文件，也能天然区分开。
>
> 客户端会沿文件夹树逐级把路径解析为对应的文件夹或制品。同一属主 + 同一父目录下的名称是唯一的，因此在指定属主后解析无歧义；名称按 NFC 归一化与服务端对齐。
>
> **两个保留URI 形态**（**仅供 `ls` 使用**，其他命令不接受）：
>
> * `artifact://~`：表示自己的目录。
> * `artifact:///`：所有共享给我的顶层资源聚合视图。它不是文件夹，不能往下钻——`artifact:///anything` 会被拒绝，改用 `artifact:///<真实 owner>/anything` 即可。

## 快速开始

下面以 test 环境为例。先配置太湖网关，用 `login` 把个人令牌保存到本地（只需一次），再执行核心链路：

```bash
export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com

# 交互式输入 PAT，保存到 ~/.nexus/auth/taihu（权限 0600），后续命令免传令牌
nexus artifact login

# 创建一个顶层文件夹
nexus artifact mkdir /models

# 上传本地文件到该文件夹
nexus artifact upload ./model.bin --folder /models

# 查看文件夹下的内容（默认只显示名称，文件夹带结尾 /）
nexus artifact ls /models

# 下载产物到本地
nexus artifact download /models/model.bin -o ./model.bin

# 共享给同事（当前仅支持员工只读授权）
nexus artifact grant /models/model.bin colleague-login-name

# 无参数列出：自己顶层（裸名）+ 所有共享给我的顶层（完整 URI）
nexus artifact ls
```

`nexus artifact login` 会安全提示输入 PAT（不进入 Shell 历史），保存后无需再设置环境变量。所有引用
资源的命令都接受斜杠路径。

## 认证与服务地址

artifact 服务使用太湖（Taihu）统一认证网关：调用方发送个人访问令牌（PAT），网关完成
认证后向后端注入携带员工身份的 `x-tai-identity` 头。

服务地址按 `--server` > `$ARTIFACT_SERVICE_URL` > 打包默认值解析；当前打包默认地址为
`https://artifact-nexus-agent.mcp.woa.com`。个人令牌按 `--pat` > `$ARTIFACT_TAIHU_PAT` > 本地令牌文件
（`~/.nexus/auth/taihu`）解析，前者覆盖后者。

### 保存令牌到本地（推荐）

用 `nexus artifact login` 把 PAT 保存到 `~/.nexus/auth/taihu`（以 `0600` 权限写入，仅属主可读），
之后所有命令都会自动读取，无需每次设置环境变量或传 `--pat`：

```bash
# 交互式安全输入（推荐，PAT 不进入 Shell 历史）
nexus artifact login

# 或直接传入（注意命令行 PAT 可能进入 Shell 历史）
nexus artifact login --pat tai_pat_xxxxxxx

# 删除已保存的令牌
nexus artifact logout
```

保存后即可直接使用：

```bash
export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
nexus artifact ls
```

如需临时覆盖本地令牌，仍可用 `$ARTIFACT_TAIHU_PAT` 或 `--pat`（优先级更高）。可选
`$ARTIFACT_TAIHU_PAT_FILE` 覆盖令牌文件路径。

### 太湖个人令牌（PAT）

在 https://tai.it.woa.com/user/pat 申请个人令牌。客户端通过
`Authorization: Bearer <PAT>` 发送令牌，网关校验后注入员工身份：

```bash
export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
read -rsp 'Taihu PAT: ' ARTIFACT_TAIHU_PAT && export ARTIFACT_TAIHU_PAT && echo
nexus artifact ls

# 显式参数可覆盖环境变量与本地令牌文件；注意命令行中的 PAT 可能进入 Shell 历史
nexus artifact ls \
  --server https://artifact-nexus-agent.mcp.woa.com \
  --pat tai_pat_xxxxxxx
```

支持的环境变量如下：

| 环境变量 | 说明 |
|----------|------|
| `ARTIFACT_SERVICE_URL` | artifact 服务地址（太湖网关入口）；未设置时默认 `https://artifact-nexus-agent.mcp.woa.com` |
| `ARTIFACT_TAIHU_PAT` | 太湖个人令牌（PAT），以 `Authorization: Bearer <PAT>` 发送 |
| `ARTIFACT_TAIHU_PAT_FILE` | 覆盖本地令牌文件路径；默认 `~/.nexus/auth/taihu` |

本地令牌文件（`nexus artifact login` 写入的 `~/.nexus/auth/taihu`）是 PAT 的最低优先级来源，
`--pat` 与 `$ARTIFACT_TAIHU_PAT` 会覆盖它。

所有子命令都支持以下通用参数覆盖上述来源：

| 参数 | 说明 |
|------|------|
| `--server` | 服务地址，覆盖 `$ARTIFACT_SERVICE_URL` |
| `--pat` | 太湖个人令牌，覆盖 `$ARTIFACT_TAIHU_PAT` 与本地令牌文件 |

未配置 PAT（参数、环境变量、本地令牌文件均无）时，客户端不会发送身份头，由目标服务决定是否允许访问。

## 数据模型

- **folder**：文件夹，可嵌套成树；顶层文件夹的 `parent` 为空。
- **artifact**：一个文件产物，挂在某个 folder 下（不指定 `--folder` 时挂在根）。
- **grant**：把某个 folder 或 artifact 的 `read` 权限授予指定员工 LoginName；文件夹上的授权会被其
  子资源继承。当前 Taihu 模型仅支持员工只读授权。
- **path（路径）/ URI**：CLI 用**裸路径**（如 `/models/v1/model.bin`）或 **`artifact:///<owner>/<path>` URI** 定位资源。裸路径隐式指向调用者自己的资源，URI 显式指定属主（自己或别人共享给你的）。客户端沿文件夹树逐级调用 `list_resources` 把引用解析为对应资源，因此无需关心内部编号。由于同一属主 + 同一父目录下同级名称唯一（服务端保证），指定属主后解析无歧义；名称按 NFC 归一化后与服务端比较。若某段不存在、或中间段不是文件夹，会报路径解析错误。**仅 `ls` 命令**还额外接受两个保留形态：`artifact:///~[/<path>]`（表示自己的目录）和 `artifact:///`（聚合共享视图别名，不能被下钻）。

## 命令

### `nexus artifact login` / `logout`

把太湖 PAT 保存到本地令牌文件（`~/.nexus/auth/taihu`，以 `0600` 权限写入），或删除它。保存后
其他命令会自动读取，无需每次设置 `$ARTIFACT_TAIHU_PAT` 或传 `--pat`。

```bash
# 交互式安全输入并保存（推荐）
nexus artifact login

# 直接传入（命令行 PAT 可能进入 Shell 历史）
nexus artifact login --pat tai_pat_xxxxxxx

# 删除已保存的令牌
nexus artifact logout
```

### `nexus artifact whoami`

打印当前 PAT（或 `--dev-employee`）对应的员工 login。拼 `artifact:///<owner>/...` URI 或用
`nexus artifact grant` 授权前，可先跑一次确认自己的身份。

```bash
nexus artifact whoami       # 打印 login，如 junlinyin
nexus artifact whoami --json   # {"employee_id": "junlinyin"}
```

### `nexus artifact mkdir` / `rmdir`

创建文件夹或删除一个空文件夹。`mkdir` 的参数是要创建的**完整文件夹路径**，其父路径必须已存在。

```bash
nexus artifact mkdir /models
nexus artifact mkdir /models/checkpoints   # 在已存在的 /models 下创建
nexus artifact rmdir /models/checkpoints
```

`rmdir` 只能删除服务端已判定为空的文件夹。`rm` 是异步删除：它接受删除请求后，后台维护任务才会
清理 COS 对象和数据库记录。因此刚执行 `rm` 就执行 `rmdir` 时，可能暂时收到 HTTP 409
`folder_not_empty`。这属于预期状态，应等待维护任务完成后重试；如果持续超时，再检查服务端
maintenance 日志和 COS 清理状态。

### `nexus artifact ls`

列出资源。可用的目标形态如下：

- **无参数**：同时列出自己名下的顶层（裸名）和所有共享给你的顶层（完整 URI），自己的资源在前、共享的资源在后。
- **裸路径** / **自己的 URI**：进入自己名下的子文件夹。
- **`artifact:///<owner>/[<subfolder>]`**：进入别人共享给你的命名空间。
- **`artifact:///`**：仅列出共享给你的顶层。

```bash
nexus artifact ls                                # 自己顶层 + 共享给我的顶层
nexus artifact ls /models                        # 自己 /models 下
nexus artifact ls artifact:///~                  # 等价于列自己的顶层
nexus artifact ls artifact://~                   # 简写形式，等价于上一行
nexus artifact ls artifact:///~/models           # 等价于 nexus artifact ls /models
nexus artifact ls artifact:///alice/             # alice 共享给我的顶层
nexus artifact ls artifact:///alice/reports/     # 进入 alice 的 reports/
nexus artifact ls artifact:///                   # 仅共享给我的顶层
nexus artifact ls artifact:/// --limit 20
nexus artifact ls /models --json
```

**无参数默认输出**：自己名下的资源显示裸名（文件夹带 `/`），来自其他人的资源显示完整 URI（如 `artifact:///alice/models/`），这样即便自己与他人存在同名文件，也能一目了然地加以区分；该 URI 可直接复制粘贴至 `download` / `info` / `rm`。**`artifact:///~[/...]` 视图**同样按"自己名下"打印裸名。**`artifact:///` 聚合视图**每行打印**完整 URI**，与无参数行为中共享部分格式相同。

分页时会在末尾打印 `# next_cursor: ...`，可传给 `--cursor` 取下一页。`--show-id` 追加资源 id 列；`--json` 输出会给每个 item 计算一个 `uri` 字段。

#### 查看别人授权（共享）给我的文件夹下的内容

`nexus artifact ls`（无参数）已在输出末尾列出所有共享给你的顶层资源（完整 URI）。获取 URI 后，即可继续向下钻取：

```bash
# 无参数 ls，输出末尾会包含共享资源，例如：artifact:///alice/sun/
nexus artifact ls
# 输出示例（后半部分为共享）：
#   models/
#   data.bin
#   artifact:///alice/sun/

# 进入 alice 的 sun/ 查看子项
nexus artifact ls artifact:///alice/sun/

# 继续往下钻（分页 / 看原始 id 都不变）
nexus artifact ls artifact:///alice/sun/子目录名
nexus artifact ls artifact:///alice/sun --limit 100
nexus artifact ls artifact:///alice/sun --json --show-id

# 直接下载：将 ls 输出的 URI 用于 download 即可
nexus artifact download artifact:///alice/sun/文件名 -o ./文件名
```

> 被授权的资源通常仅有只读权限：`ls`/`download` 可用，`upload`/`mkdir`/`rm` 等写操作会被拒绝（客户端识别到 `artifact:///<别人>/...` 时会提前报错，避免徒劳的服务端调用）。

### `nexus artifact artifacts`

跨文件夹平铺列出 artifact（不含文件夹层级）。默认输出自己名下的所有 artifact；传入 `artifact:///` 可得到聚合的、共享给我的 artifact 列表（每行是一条完整 URI）。

```bash
nexus artifact artifacts --limit 50
nexus artifact artifacts artifact:/// --json
```

默认非 JSON 输出为 `名称<TAB>status<TAB>size`。

### `nexus artifact upload`

上传本地文件。CLI 会计算 MD5、创建上传会话、把文件 PUT 到预签名的 COS URL，再 complete 会话。
`--folder` 传入目标文件夹**路径**（默认上传到根）。成功后打印 artifact 名称。

```bash
nexus artifact upload ./model.bin --folder /models
nexus artifact upload ./data.parquet --name dataset-v2.parquet
nexus artifact upload ./a.bin --content-type application/octet-stream --pool pool_x
```

| 参数 | 说明 |
|------|------|
| `--folder` | 目标文件夹路径（如 `/models`）；默认上传到根 |
| `--name` | artifact 名称；默认取本地文件名 |
| `--content-type` | 覆盖自动推断的 content type |
| `--pool` | 存储池 id；默认使用服务端默认池 |

### `nexus artifact upload-large`

大文件可通过该命令请求multipart 上传。服务端返回分片会话时，CLI 按会话中的 `part_size` 切分
文件，再并发 PUT 到各分片预签名 URL，最后提交分片编号与 ETag 列表；如果服务端返回普通上传会话，
客户端会回退到单次 PUT。整个过程仍由客户端直传 COS，无需在本地配置 COS 密钥。

```bash
# 使用服务端决定的分片大小和默认并发数 5
nexus artifact upload-large ./big-model.bin --folder /models

# 最多并发上传 8 片
nexus artifact upload-large ./big-model.bin \
  --folder /models \
  --concurrency 8
```

除普通上传支持的 `--folder`（文件夹路径）、`--name`、`--content-type`、`--pool` 外，`--concurrency`
可设置最大并发分片数（默认 5）。当前 CLI 不提供分片大小参数，分片大小由服务端在上传会话中返回。

分片上传失败时命令返回非零状态。未完成会话的过期和清理由服务端维护任务负责。CLI 当前只打印简化的
错误消息；排障时应记录执行时间、完整命令（隐藏 PAT）和目标环境，并通过服务端日志定位错误码和
request id。

### `nexus artifact publish`

把一次 task 的**产物目录**一键发布：新建一个文件夹，把`<local_dir>` 下的所有文件递归上传进去
（子目录会在服务端重建同样的层级），可选地把该文件夹只读共享给同事。它本质上是
`mkdir` + 批量 `upload` + `grant` 的封装，方便"跑完一个 task → 把产物集中归档并分享"。

```bash
# 把./task-out 下所有产物发布为一个新文件夹（名字默认取目录名 + 时间戳）
nexus artifact publish ./task-out

# 指定人可读标签，并把只读权限授予两名同事
nexus artifact publish ./task-out --task my-run --grant alice,bob

# 机器可读输出（含 folder 路径与上传清单）
nexus artifact publish ./task-out --task my-run --json
```

命名与可见性：

- 文件夹名固定为 `<label>-<UTC时间戳>-<短随机>`（时间戳格式 `YYYYmmdd-HHMMSS`，末尾追加 6 位随机十六进制
  防止同秒重跑撞名）。`label` 缺省取本地目录名，`--task` 只是覆盖这个标签，**不是精确名、也不是 id**。
- **每次运行都创建一个全新文件夹**——即使`--task` 相同，时间戳与随机后缀不同也会得到不同文件夹，因此这是
  "每次一份新产物"的语义，不做幂等复用。
- 文件夹默认**仅属主可见**（服务端 ACL，非本机文件权限）；`--grant` 会对列出的每个员工授予`read`，并
  被文件夹下所有已上传产物继承。被授权人用自己的 PAT 直接 `nexus artifact ls artifact:///<你的 login>/` 或 `artifact:///` 就能看到 / 下载。

| 参数 | 说明 |
|------|------|
| `local_dir` | 本地产物目录；其下所有普通文件都会被上传（递归，重建子目录层级） |
| `--task` | 文件夹名的人可读标签；默认取目录名。总会追加 UTC 时间戳与短随机后缀保证每次唯一 |
| `--grant` | 逗号分隔的员工 LoginName，授予只读并被子产物继承 |
| `--json` | 输出 JSON（含 `folder_path`、`uploaded` 清单与 `grants`） |

非 JSON 输出第一行是新文件夹路径（如`/my-run-20260804-153012-a1b2c3`），随后每行是一个已上传文件在
文件夹内的相对路径。目录不存在、不是目录、或目录内没有任何文件时命令直接返回非零且不做任何上传。

> **空文件会被跳过**：artifact 服务不接受 0 字节上传（会报 `size_bytes must be at least 1`），因此
> publish 会跳过目录中的空文件并打印 `Skipping empty file ...` 警告，其余文件正常上传；`--json` 输出的
> `skipped_empty` 字段会列出被跳过的相对路径。若目录里**所有**文件都为空，则命令返回非零且不创建文件夹。

### `nexus artifact download` / `url`

`download` 把产物下载到本地文件（默认文件名取 artifact 名）；`url` 只打印一个预签名下载 URL。二者的参数都是 artifact **路径或 URI**：裸路径指向自己名下的产物，`artifact:///<owner>/<path>` 指向别人共享给你的产物。

```bash
nexus artifact download /models/model.bin -o ./model.bin
nexus artifact download /models/model.bin                # 默认存成 artifact 名
nexus artifact url /models/model.bin                               # 打印预签名 URL

# 下载 / 打印别人共享给你的产物
nexus artifact download artifact:///alice/sun/文件名 -o ./本地文件名
nexus artifact url artifact:///alice/sun/文件名
```

| 参数 | 说明 |
|------|------|
| `path` | artifact 或文件夹的路径 / URI（例如 `/models/model.bin` 或 `artifact:///alice/model.bin`） |
| `-o` / `--output` | 输出路径。**下载 artifact 时**：本地文件名，默认取 artifact 名。**下载文件夹时**：目标目录，默认取 `./<folder-name>` |
| `--concurrency` | 递归下载文件夹时的并发数，默认 4。下载单个 artifact 时无效 |

客户端根据 URI 的属主段自动切换到 owned 或 shared 视图并做同名消歧。

#### 递归下载文件夹

如果 `path` 解析到一个文件夹，`download` 会**自动切换到递归模式**：把该文件夹的**全部内容**镜像到本地目录，逐级还原子文件夹结构。行为参照 `rsync -a src/ dest/`：`-o` 指定目标目录本身，不再多套一层同名子目录。

```bash
# 递归下载自己名下的整个 run 文件夹
nexus artifact download /tasks/2026/08/05/my-task-XXXXXX -o ./out
# 目录结构：./out/文件1、./out/子目录/文件2 ...

# 不带 -o：默认落至./<folder-name>/
nexus artifact download /tasks/2026/08/05/my-task-XXXXXX
# 目录结构：./my-task-XXXXXX/文件1 ...

# 下载别人共享给你的整个文件夹
nexus artifact download artifact:///alice/reports -o ./alice-reports --concurrency 8
```

**失败处理**：递归模式下**单文件失败不会中断整批**——所有失败会在结束时汇总到 `logger.error`（`--json` 模式下汇总到返回体的 `failures` 字段），若有任何失败命令以非零退出码退出。如此一来，单次网络抖动不会导致整批下载前功尽弃。

### `nexus artifact info` / `rm`

`info` 输出 artifact 元信息；`rm` 请求异步删除一个 artifact。参数都是 artifact **路径或 URI**。

```bash
nexus artifact info /models/model.bin
nexus artifact info artifact:///alice/sun/文件名--json      # 看别人共享给我的产物
nexus artifact rm /models/model.bin
```

> `rm` 只能删除**自己名下**的产物。如果传入 `artifact:///<别人>/...`，客户端会立刻拒绝（`owned by another employee`），不会请求服务端；被授权人无权删除他人产物，需要请属主自己删除或撤销授权。

`rm` 成功表示服务端已接受删除请求，不代表 COS 对象和数据库记录已经同步清理。需要立即删除父文件夹时，请按 `rmdir` 章节所述处理暂时的 `409 folder_not_empty`。

### `nexus artifact grants` / `grant` / `revoke`

管理某个 folder 或 artifact 的员工只读授权。资源用**路径**指定。

```bash
# 列出资源上的授权
nexus artifact grants /models

# 把只读权限授予员工 LoginName
nexus artifact grant /models colleague-login-name
nexus artifact grant /models/model.bin colleague-login-name --expires 2026-12-31T23:59:59

# 撤销员工只读授权
nexus artifact revoke /models colleague-login-name
```

| 参数 | 说明 |
|------|------|
| `path` | folder 或 artifact 路径 |
| `grantee_id` | 被授权员工的 LoginName |
| `--type` | 授权对象类型，当前仅支持 `employee`（默认） |
| `--permission` | 授权权限，当前仅支持 `read`（默认） |
| `--expires` | 可选的 ISO-8601 过期时间（仅 `grant`） |

授予文件夹读取权限会被其下的子文件夹和 artifact 继承；`grants` 输出中带
`(inherited_from ...)` 的即为继承而来的授权。当前不支持应用主体或写权限。

## Python SDK

`nexus.utils.artifact.ArtifactClient` 提供异步接口。作为 async context manager 使用，认证与服务地址的
解析顺序与 CLI 一致。走太湖网关时传入 PAT 即可：

```python
import asyncio
import os

from nexus.utils.artifact import ArtifactClient, TAIHU_GATEWAY_URL


async def main():
    async with ArtifactClient(
        base_url=TAIHU_GATEWAY_URL,
        taihu_pat=os.environ['ARTIFACT_TAIHU_PAT'],
    ) as client:
        folder = await client.create_folder('models')
        artifact = await client.upload_file('model.bin', parent_folder_id=folder.folder_id)
        print('uploaded:', artifact.name)

        # 共享给同事（只读）
        await client.put_grant(artifact.artifact_id, 'colleague-login-name', permission='read')

        # 下载回本地
        await client.download_to_file(artifact.artifact_id, 'restored.bin')


asyncio.run(main())
```

不传 `base_url` / `taihu_pat` 时会分别回退到 `$ARTIFACT_SERVICE_URL` / `$ARTIFACT_TAIHU_PAT`；PAT 未
经参数或环境变量提供时，还会回退到本地令牌文件（`nexus artifact login` 保存的
`~/.nexus/auth/taihu`）。

除高层的 `upload_file` / `upload_bytes` / `download_to_file` / `download_bytes` 外，SDK 也暴露
底层端点：`create_folder` / `delete_folder`、`list_resources` / `list_artifacts` /
`get_artifact` / `delete_artifact`、`list_grants` / `put_grant` / `delete_grant`、以及上传的
`create_upload` / `complete_upload` 和下载的 `create_download_url`。

上传行为与注意事项：

- 上传前会校验本地路径存在且为普通文件；上传通过预签名 COS URL 直传，客户端不需要配置 COS 密钥。
- 会为创建上传会话、创建文件夹和授予授权自动带上 `Idempotency-Key`（幂等），complete 与 delete
  操作不需要幂等键。
- 下载先写入临时 `.part` 文件再原子重命名，避免中断产生半个文件。
- 服务端错误（HTTP 4xx/5xx 或响应体中的 `error`）会转换成 `ArtifactClientError`，携带 `code`、
  `request_id` 和 `http_status`。

## test 环境手工验证

下面的最小链路验证太湖认证、Artifact API 和 COS 上传下载。请在可访问太湖 test 网关的网络环境中
执行以下脚本：

```bash
set -euo pipefail

export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
read -rsp 'Taihu PAT: ' ARTIFACT_TAIHU_PAT && export ARTIFACT_TAIHU_PAT && echo

RUN_ID=$(date +%Y%m%d-%H%M%S)
SRC=$(mktemp /tmp/nexus-artifact-src.XXXXXX)
DST=$(mktemp /tmp/nexus-artifact-dst.XXXXXX)
trap 'rm -f "$SRC" "$DST"' EXIT
printf 'nexus artifact manual test: %s\n' "$RUN_ID" > "$SRC"

# 1. 认证与列表（自己的顶层）
nexus artifact ls --limit 5

# 2. 创建、上传、查询（用路径寻址）
FOLDER="/manual-test-${RUN_ID}"
ARTIFACT="${FOLDER}/$(basename "$SRC")"
nexus artifact mkdir "$FOLDER"
nexus artifact upload "$SRC" --folder "$FOLDER" --content-type text/plain
nexus artifact info "$ARTIFACT" --json
nexus artifact ls "$FOLDER"

# 3. 下载回环校验；任何一步失败都会以非零状态退出
nexus artifact download "$ARTIFACT" -o "$DST"
cmp "$SRC" "$DST"
echo 'PASS: upload/download content matches'

# 4. 请求删除；后台清理完成前 rmdir 可能返回 409 folder_not_empty
nexus artifact rm "$ARTIFACT"
DELETED=0
for i in $(seq 1 45); do
  if nexus artifact rmdir "$FOLDER"; then
    DELETED=1
    break
  fi
  sleep 2
done

if [ "$DELETED" -ne 1 ]; then
  echo "FAIL: folder $FOLDER was not deleted within 90 seconds" >&2
  exit 1
fi
```

`info --json` 中 artifact 的 `status` 应为 `ready`，`cmp` 应返回 0。要验证共享链路，所有者可执行
`grant`，然后让被授权员工使用自己的 PAT 运行 `ls artifact:///` 和 `download`；最后由所有者执行
`revoke`。仅查看 `grants` 只能证明授权记录已创建，不能替代第二个员工身份的读取验证。

自动化 E2E 可在配置真实 test 环境和授权对象后运行：

```bash
export ARTIFACT_E2E=1
export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
read -rsp 'Taihu PAT: ' ARTIFACT_TAIHU_PAT && export ARTIFACT_TAIHU_PAT && echo
export ARTIFACT_GRANTEE=colleague-login-name  # 测试用员工 LoginName
uv run pytest tests/e2e/artifact/test_sdk_e2e.py -m e2e -q
```

该测试依赖真实服务、COS 存储桶和有效员工身份，不能作为无外部依赖的单元测试运行。现有 E2E 使用所有者
身份创建并检查 grant，下载也仍由所有者执行；它不验证被授权员工的 `shared` 列表和下载权限。完整 ACL
验证仍需按上文使用第二位员工自己的 PAT 手工执行。

## 常见错误排查

| 错误或现象 | 含义与处理 |
|------------|------------|
| 连接 `localhost:80` 被拒绝 | `--server` 或 `$ARTIFACT_SERVICE_URL` 被显式设置为 `http://localhost:80`，客户端无法连接到本地；请改用太湖网关 URL 或本地服务地址 |
| `missing_bearer_token` | 请求到达太湖网关但没有 PAT；检查 `$ARTIFACT_TAIHU_PAT` 或 `--pat` |
| `token_inactive` / `TOKEN_INVALID` | PAT 无效、已过期或被撤销；重新申请或更换 PAT |
| `Taihu verifier is not configured` | 后端 Taihu 校验配置不完整；检查部署配置和 Pod 是否已加载最新配置 |
| HTTP 502 或网关超时 | 太湖网关无法访问后端；检查路由、Service、Pod 就绪状态和网关到后端网络 |
| `COS head_object 403` | 请求已进入 Artifact 服务，但服务端 COS 凭据、Bucket、Region 或对象权限不正确 |
| `folder_not_empty` | artifact 删除仍在后台处理中；等待 maintenance 清理后重试 `rmdir` |
| `invalid_state` | artifact 或上传会话状态不允许当前操作；先用 `info --json` 检查 artifact 状态，再结合服务端日志排查 |

表中的错误码可能出现在网关响应、API 响应或服务端日志中。CLI 当前只输出异常消息，不保证展示
`code`、`request_id` 和 HTTP 状态；Python SDK 捕获到的 `ArtifactClientError` 则可读取这些属性。
使用 CLI 排障时请记录执行时间、命令（隐藏 PAT）和目标环境，再由服务维护者关联服务端日志；不要在
日志、工单或代码仓库中粘贴 PAT。
