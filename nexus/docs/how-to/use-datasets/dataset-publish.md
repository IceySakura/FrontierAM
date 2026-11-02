# Dataset publish 使用指南：从 JSONL 和附件发布到 COS

`nexus dataset publish` 将本地数据读取为 Nexus DatasetItem，打包声明的附件，上传到 COS，
返回可以用于读取、下载和提交任务的固定数据集引用。业务 JSONL 可以在 publish 时直接转换，
**不需要先执行一次 `dataset import`**。

本文适用于包含 `feat/dataset-jsonl-cos` 能力的提交端和云端服务。命令通过 `uv run nexus` 执行，
需要事先安装项目及任务所需依赖。`my_task`、桶名、发布前缀等示例值需要替换为自己的配置。

## 1. 先分清数据、转换规则和运行配置

| 内容 | 用途 | 是否必须单独写文件 |
| --- | --- | --- |
| 数据记录和附件 | 每条样本的业务输入，以及图片、代码等文件 | 必须提供可读取的数据来源 |
| 导入规则（`--import-spec`） | 说明 JSONL 属于哪种记录、哪个字段是 ID、哪个字段指向附件 | 可选；普通业务记录可用 `--id-field` / `--files-field`，复杂规则再用 spec |
| 运行配置（例如 submit 的 `--use`） | 提供数据中缺少的 imports、模型、Agent、Runtime 等参数 | 取决于数据是否已包含所需运行配置；不要求使用 YAML |

`import.yaml` 是导入规则文件的惯用名称，也可以使用 JSON，文件名没有特殊含义。
同一种数据格式可以复用同一份规则，不是每个人、每次运行都要写一份。

运行 Profile 不属于 publish 的要求。它可以来自已有配置文件、启动脚本调用原 generator
生成的配置，或者提交命令的 `--set` / `--set-json`。运行参数必须有来源，但不必都放在独立 YAML 中。

publish 本身**不会执行原 generator，不会自动提取算法运行配置，也不会启动任务**。
普通业务 JSONL 可直接发布；附件字段需显式指定。嵌套路径或算法特有格式仍需整理或配置规则。

## 2. 选择输入格式

| 手上的输入 | 用法 |
| --- | --- |
| 普通业务 JSONL，例如 `instance_id/prompt/bundle` | `--id-field instance_id --files-field bundle`；需要额外规则时再用 spec |
| 每行已经是运行配置 RunInput，另外带 ID | `--import-spec`，使用 `record_kind: run_input` |
| 每行已经是 `id/body/files` DatasetItem | 直接发布 JSONL；自动识别标准 envelope |
| 已有标准本地 Dataset 目录，包含 `dataset.yaml` 等 | 直接发布该目录，明确指定 `--backend cos` |

只有 `tasks.jsonl` 和图片子目录的文件夹，还不是标准本地 Dataset 目录。
此时应传 **JSONL 文件路径**，而不是它的父目录。目录叫 `input`、`input.offline` 或其他名字没有区别。

### 普通业务 JSONL：直接指定字段

```bash
uv run nexus dataset publish ./tasks.jsonl research/demo@v1 \
  --id-field instance_id --files-field bundle --files-to-cwd \
  --backend cos --store cos://your-bucket/datasets --region ap-guangzhou
```

不需要 `import.yaml`。附件字段提取到 `files`，其他字段放入 `body.task_config`；
运行时通过 `--use` 等方式补齐 `task_name`。没有附件时省略两个 `--files-*` 参数。
显式指定 ID 字段后，每条记录必须包含非空字符串 ID；两个字段参数仅读取顶层字段。
未指定 `--id-field` 时默认读 `id`，缺失才生成行号 ID，并在 metadata 中记录源文件摘要和行号。
空值、null、非字符串和重复 ID 都会报错。ID 字段保留在业务输入中。
CLI 字段参数覆盖 import-spec 中的同名映射；显式 `--id-field` 禁用缺失 ID 的自动生成。
`--files-field` 不带 spec 时自动识别目录或 ZIP；如果 spec 指定了附件类型或 root_name_field，仍保留这些设置。

### 方式 A：业务 JSONL，通过可选导入规则转换

假设数据目录为：

```text
my-data/
├── tasks.jsonl
├── import.yaml
└── case-001/
    └── reference.txt
```

`tasks.jsonl`：每行一个 JSON 对象，不要写 JSON 数组或空行。

```json
{"instance_id":"case-001","prompt":"请根据参考文件完成任务","task_dir":"case-001","bundle":"case-001"}
```

`import.yaml`：

```yaml
schema_version: 1
record_kind: task_input
task_name: my_task
id_field: instance_id
split: test
files:
  path_field: bundle
  type: directory
compat:
  files_to_cwd: true
```

这份规则会：

1. 使用 `instance_id` 作为 DatasetItem 的 `id`。
2. 取出 `bundle`，将它指向的目录作为附件；`bundle` 不进入 `task_config`。
3. 将其余业务字段放进 `body.task_config`，包括原来的 `instance_id`。
4. 设置 `body.task_name=my_task`，归入 `test` split。
5. 声明运行时将附件恢复到任务的 **host 工作目录**，以 `case-001` 为根目录。

转换后的结构示意如下；实际发布时 `files` 会变成带 SHA256、大小和 COS 引用的 ZIP 声明：

```json
{
  "id": "case-001",
  "body": {
    "task_name": "my_task",
    "task_config": {
      "instance_id": "case-001",
      "prompt": "请根据参考文件完成任务",
      "task_dir": "case-001"
    },
    "runner": {"task_files_materialization": {"mode": "cwd", "target": "host"}}
  },
  "files": {"type": "directory", "path": "case-001", "root_name": "case-001"}
}
```

`task_dir` 是示例任务使用的业务字段，不是所有任务的必填项。publish 不会自动改写业务字段中的
Ceph 绝对路径；任务在远端读取什么路径，需要与附件落地位置对应。

常见变体：

- 没有附件：删除 `files` 和 `compat.files_to_cwd`，只保留业务字段。
- 每条附件已经是 ZIP：设置 `files.type: zip`，让 `bundle` 指向 ZIP 文件。
  如需明确解包根目录，可设置 `files.root_name_field: instance_id`。
- 没有 ID：设置 `generate_ids: true`，自动生成 `line-00000001` 等 ID；行号随输入顺序变化。
- `path_field`、`id_field`、`root_name_field` 当前读取的是**顶层字段名**，不解析点号路径。
  例如原始数据的附件在 `extra.task_dir`，需要先提取成顶层字段，或改用下面的标准记录格式。

### 方式 B：标准 DatasetItem JSONL，不需要 import.yaml

将记录直接写成 `id/body/files` 格式，例如一行：

```json
{"id":"case-001","body":{"task_name":"my_task","task_config":{"instance_id":"case-001","prompt":"请根据参考文件完成任务","task_dir":"case-001"},"runner":{"task_files_materialization":{"mode":"cwd","target":"host"}}},"files":{"type":"directory","path":"case-001"}}
```

发布这个 JSONL 时省略 `--import-spec` 即可。无需附件时可以省略 `files`，同时不要声明附件解包。
自动识别要求记录包含 ID 字段、对象类型的 `body`，且其余字段只有 `files/merge/metadata`
以及显式指定的 ID/附件字段。带其他业务字段的记录会完整包装为 task_config；业务记录恰好
与标准 envelope 同形时，显式使用 `record_kind: task_input` 消除歧义。
如果要给标准 JSONL 指定非默认 split，仍可使用一份简短导入规则：
`record_kind: dataset_item`、`split: validation`。

### 方式 C：已有 RunInput JSONL

每行包含 `id` 以及 `task_name`、`task_config`、`agent_name` 等 RunInput 字段，导入规则为：

```yaml
schema_version: 1
record_kind: run_input
id_field: id
split: test
```

转换器会取出 `id`，其余运行字段放入 `body`；`task_files` 单独提取为附件声明。
也可通过 `files.path_field` 提取附件，但不能同时提供冲突的 `task_files`。
如果记录已包含完整可用的运行配置，submit 可以只传 `--data`；配置不完整时仍需补齐。
模型、沙箱等密钥应在提交时提供，不要为了保存完整 RunInput 而把密钥发布进数据集。

## 3. 附件路径与落地位置

相对附件路径按 **JSONL 所在目录**解析，不按执行命令时的工作目录解析。
发布进程必须能读取附件实际内容：例如 Ceph 路径必须在提交机器上已挂载且可读；
仅写一个远端机器上的绝对路径不能让 publish 自动访问那台机器。

每条记录支持一个目录或 ZIP，多个文件放进同一个目录即可。
普通 ZIP 会规范化为 Nexus ZIP；已声明的远端附件会读取、校验并复制到本次发布目标。

附件运行时有两种常见使用方式：

- 任务使用 `ctx.task_files` 按需读取：不必设置 `files_to_cwd`。
- 任务依赖工作目录中的文件：设置 `compat.files_to_cwd: true` 或 publish 的 `--files-to-cwd`。
  默认落到 host 的 `<工作目录>/<root_name>/...`，不是自动上传到所有沙箱的 `/workspace`。

如需 Runtime 直接接收附件，可以在运行配置中使用
`runner.task_files_materialization: {mode: cwd, target: 'runtime:main'}`，并提供对应 Runtime。
有些任务自己从 host 读取文件并上传沙箱，应沿用任务实际约定。
同一工作目录下的附件根目录不能相互冲突。

## 4. 配置 COS 发布目标与凭据

以下命令假设从 Nexus 仓库根目录执行，输入采用方式 A，并已在环境中提供 COS 凭据。
示例桶名、地域、前缀需要换成实际可访问的位置。

```bash
set -euo pipefail
umask 077

export DATA_ROOT=/absolute/path/to/my-data
export COS_STORE=cos://your-bucket-1234567890/datasets
export COS_REGION=ap-guangzhou
export DATASET_TARGET=research/my-dataset@v1
PUBLISH_DIR=$(mktemp -d)

# 从已有私有配置或 CI 注入，不在命令历史、文档中填写真实密钥。
: "${COS_SECRET_ID:?需要配置 COS_SECRET_ID}"
: "${COS_SECRET_KEY:?需要配置 COS_SECRET_KEY}"
export COS_SECRET_ID COS_SECRET_KEY
# 使用 STS 时还需 export COS_TOKEN。
```

发布需要对目标前缀的读写权限，以及查询桶版本控制状态的权限。
当前实现要求 **桶从未启用过版本控制**，曾开启后暂停的桶也不满足要求。
COS 的 `research/my-dataset@v1` 是路径和版本，不需要提前创建 Taiji namespace。
必须使用明确版本；COS 发布不接受 `latest`、`dev` 作为版本。

## 5. 本地检查，然后发布

先只构建，不上传：

```bash
uv run nexus dataset publish "$DATA_ROOT/tasks.jsonl" "$DATASET_TARGET" \
  --import-spec "$DATA_ROOT/import.yaml" \
  --backend cos --store "$COS_STORE" --region "$COS_REGION" \
  --build-only --output "$PUBLISH_DIR/build" --json
```

这里检查记录结构、ID、附件是否存在，并生成标准记录和 ZIP。
它不检查任务代码是否已注册、模型凭据是否有效或沙箱是否能启动。
纯本地输入的 build-only 不访问 COS；若输入引用远端附件，构建仍需下载它们。

正式发布：

```bash
uv run nexus dataset publish "$DATA_ROOT/tasks.jsonl" "$DATASET_TARGET" \
  --import-spec "$DATA_ROOT/import.yaml" \
  --backend cos --store "$COS_STORE" --region "$COS_REGION" \
  --json > "$PUBLISH_DIR/published.json"

DATA_URI=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["data_uri"])' \
  "$PUBLISH_DIR/published.json")
export DATA_URI
```

方式 B 或标准本地 Dataset 目录省略 `--import-spec`。发布后保留 `published.json`，
其中包含 `data_uri`、`content_sha256`、`manifest_sha256`、样本数和 split 统计。
`data_uri` 中的 `?region=...#sha256=...` 是完整引用的一部分，不要截掉。

默认构建使用临时目录，publish 结束后会清理；**不会默认留下一个可长期使用的本地标准数据集**。
如需保留构建文件，显式指定新的空 `--output` 目录。
build-only 的 `records/objects` 是构建产物，不等于可直接作为 `--data` 使用的本地 Dataset 目录。

如需先得到可独立使用的标准本地 Dataset，下面是可选步骤，不是 publish 的前置要求：

```bash
uv run nexus dataset import "$DATA_ROOT/tasks.jsonl" \
  --import-spec "$DATA_ROOT/import.yaml" \
  --output "$PUBLISH_DIR/converted" --json

uv run nexus dataset publish "$PUBLISH_DIR/converted" "$DATASET_TARGET" \
  --backend cos --store "$COS_STORE" --region "$COS_REGION" --json
```

## 6. 查看与下载

仍需在当前进程提供该桶的读取凭据：

```bash
uv run nexus dataset show "$DATA_URI" --json
uv run nexus dataset items "$DATA_URI" --limit 5 --json
uv run nexus dataset get "$DATA_URI" case-001
uv run nexus dataset fetch "$DATA_URI" --output "$PUBLISH_DIR/downloaded" --json
```

`fetch` 导出的是标准本地 Dataset 布局，不是 COS 对象路径的逐字节目录镜像。
如果要审计 COS 中实际保存的 manifest、分片和 ZIP，需按 manifest 的对象引用下载并校验。

COS 中的布局如下：

```text
<store-prefix>/research/my-dataset/
├── releases/v1/manifest.json
├── snapshots/<content_sha256>/test/items-00000.jsonl
└── objects/<zip_sha256>.zip
```

`manifest.json` 记录版本、split 和分片引用；`snapshots` 保存标准记录，`objects` 保存附件包。
因此 COS 中出现 snapshots 是正常的发布布局，不表示业务数据变成了另一套图片数据。
引用中的 SHA256 校验 manifest；manifest 和记录继续声明分片、附件的 SHA256 与大小。

同一版本、相同内容可以重试；同一版本、不同内容会冲突，需要使用新版本。
构建摘要会受到记录内容、顺序和附件包变化影响，不应仅凭 prompt 相同就认定整个版本相同。
上传中断可能留下尚未被发布 manifest 引用的对象；以发布命令成功返回的固定引用为准。

## 7. 将已发布 Dataset 提交到云端

publish 和 submit 是两个独立命令，可以在同一个 Bash 中连续执行。
当前 `submit --generation remote` 不会先替你发布本地 JSONL。

下面示例假设 `RUN_PROFILE` 已指向**可直接使用、凭据已解析的运行配置文件**。
它由算法原有配置或启动脚本提供，**不是 dataset publish 的产物**。
如果复用包含 `${oc.env:...}` 的普通 Profile，请在提交前由自己的启动逻辑解析所需字段；
不要假设 `--use` 会自动替换这些字符串。

```bash
: "${RUN_PROFILE:?指定已有的运行配置文件，或由原启动脚本生成}"
test -f "$RUN_PROFILE"

uv run nexus submit batch \
  --generation remote --data "$DATA_URI" --use "$RUN_PROFILE" \
  --environment tkex-nanjing-test \
  --control-url https://test-control-runner-nexus-agent.mcp.woa.com \
  --project-root "$PWD" --source "$PWD" \
  --taihu-token-file "$HOME/.nexus/auth/taihu-pat" \
  --json
```

根据任务实际需要补充 `--wheel-extra`、并发、资源、超时、结果上传配置等。
Task/Agent/Runtime 实现必须能从提交的代码和依赖中加载；仅发布数据不会部署算法代码。
提交会发送当前源码，建议使用干净 checkout，避免夹带无关本地修改。

如果数据本身包含完整运行配置，可以省略 `--use`；如果只发布 prompt 等业务字段，
省略运行配置不会让平台自动猜出模型或沙箱。
同一 Profile 可用于多个数据集，也可以对同一数据集更换 Profile 重跑，无需重新发布数据。

submit 从提交进程环境中读取 COS 凭据，随任务单独传递，远端用于读取分片和附件。
模型、沙箱凭据与 COS 凭据是不同用途；test 登录 PAT 也不能替代 COS 凭据。
远端子任务容器中 `env` 看不到 `COS_SECRET_ID`，不代表任务没有 COS 凭据：任务执行可通过
`DataAccessContext` 获取凭据。使用临时凭据时，要考虑生成、执行及结果上传的有效期。

## 8. 接到原算法启动脚本时，哪些工作仍需适配

最小流程是：**准备符合约定的输入 → publish → 获取 data_uri → submit**。

- 数据已经是标准 DatasetItem：直接 publish，可以没有 `import.yaml`。
- 普通业务记录：默认包装为 task_config；可用 `--id-field` / `--files-field` 指定字段，额外映射再用 import spec。
- 数据是任意嵌套业务格式：需要适配字段和附件声明；publish 不会分析任意 generator 推断格式。
- 原 generator 同时生成数据与模型配置：需要把数据和运行配置分开，或让它输出符合约定的记录。
- `--generator` 与 `--data/--use` 在当前 batch submit 中是互斥入口，不能简单同时传入。

对使用者可以封装成一条 Bash，但目前不存在自动适配所有算法 generator 的通用开关。

Taiji 也支持同一套 JSONL 导入规则，通过 `--backend taiji` 发布；其认证、namespace 权限与
COS 不同，不能照搬 COS 的桶和凭据。其他命令详见 [Dataset CLI](dataset-cli.md)。

## 9. 常见问题

| 现象 | 检查方向 |
| --- | --- |
| 提交时缺少 `task_name` | publish 允许省略；提交时由 Profile 或 Compose 参数补齐 |
| ID 缺失、重复或包含路径分隔符 | 提供稳定、合法且同 split 内唯一的字符串 ID；必要时启用自动行号 ID |
| 找不到附件 | 检查相对 JSONL 的路径、挂载和权限；嵌套路径字段是否已提取 |
| `files_to_cwd requires task files` | 无附件时不要启用自动解包 |
| COS 403 | 区分发布端与提交端凭据，检查目标桶/前缀和桶版本控制查询权限 |
| 桶版本控制校验失败 | 使用从未启用过版本控制的桶 |
| 同版本内容冲突 | 保持原版本不变，改用新版本发布 |
| publish 成功但任务缺模型、Agent 或 Runtime | 补充运行配置；发布成功只说明数据发布链路完成 |
| 文件在 host，但沙箱中没有 | 检查解包 target，以及任务是否实现 host → sandbox 上传 |
| 看见 `input.offline` 或额外 README | 这些是使用者本地目录命名/说明，不是平台要求的格式 |

实现依据：[记录转换](../../../nexus/dataset/records.py)、[构建与附件冻结](../../../nexus/dataset/build.py)、
[COS 发布与读取](../../../nexus/dataset/cos.py)、[CLI](../../../nexus/cli/commands/dataset.py)。
