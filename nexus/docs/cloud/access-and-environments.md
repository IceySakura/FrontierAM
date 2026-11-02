# 认证与运行环境

接入时分别确认三件事：访问哪套部署、使用什么身份、该身份在哪个 Environment 有权限和资源额度。
部署环境名称（如正式、测试）不是 `--environment` 所需的 Environment ID。

## 程序入口与地址

以下为正式环境的个人 API 入口。测试或其他部署使用该部署公布的成对 Control/Application 地址；不能只替换其中一个。
示例启用 Bash 严格错误检查，命令、变量检查或管道失败时停止执行，避免继续发送后续请求。

```bash
set -euo pipefail

export CONTROL_BASE_URL='https://control-runner-nexus-agent.mcp.woa.com'
export APPLICATION_BASE_URL='https://application-runner-nexus-agent.mcp.woa.com'
export TAIHU_TOKEN_FILE="$HOME/.nexus/auth/taihu-pat"
export NEXUS_ENVIRONMENT='<environment-id>'
```

本组文档中的 `CONTROL_BASE_URL`、`APPLICATION_BASE_URL` **不含末尾 `/v1` 和尾斜杠**，可以包含太湖发布路径前缀。
HTTP 示例统一追加 `/v1/...`；CLI 的 `--control-url` 直接使用 `CONTROL_BASE_URL`。

| 地址 | 用途 |
| --- | --- |
| [Web 控制台](https://cloud.nexus-agent.woa.com/v1/console) | 浏览器登录与页面操作 |
| Control API | 提交任务、查询状态、停止任务、查询授权 |
| Application API | 读取业务结果、Live 事件、提交交互命令、Runtime proxy |

提交响应中的 URL 应保存为任务元数据。只有当它属于当前身份可访问的 API 入口时才直接使用；应用订阅的发布前缀可能与响应中的
绝对 URL 不同。需要统一入口时，保存 `execution_id`，按本页的两个基址和 [API 参考](reference/api.md)中的路径构造地址。
不要把控制台地址、浏览器 Cookie 或个人身份头当作程序 API 的接入方式。

## 选择认证方式

<a id="web-与-cli-认证"></a>

| 调用场景 | 认证方式 | 准备条件 |
| --- | --- | --- |
| 浏览器控制台 | OA 登录 | 用自己的账号登录 |
| 个人 CLI | 太湖个人 PAT | PAT 已授权 `nexus-agent`，通过 `--taihu-token-file` 传入 |
| 个人 HTTP 脚本 | 同一个太湖个人 PAT | HTTP 请求携带 `Authorization: Bearer ...` |
| 线上应用 HTTP 调用 | 太湖应用签名 | 订阅 Control/Application API，并完成 Nexus 用户绑定 |

HTTP 是调用方式，不限定必须使用应用身份。选择与调用者对应的一种认证方式，不要混用。

### 个人 PAT

创建和保存 PAT 的步骤见[快速开始](quickstart.md#3-申请太湖-pat)。固定使用私有文件：

```bash
chmod 600 "$TAIHU_TOKEN_FILE"
```

不要在 `--taihu-token`、YAML、Generator 或日志中写明文 PAT。下面的请求函数供 HTTP 示例复用：

```bash
nexus_request() {
  local method="$1" url="$2"
  shift 2
  curl --fail-with-body -sS -X "$method" "$url" \
    -H "Authorization: Bearer $(cat "$TAIHU_TOKEN_FILE")" "$@"
}
```

### 应用身份

1. 用户先登录控制台，确认目标 Environment Grant 和额度。
2. 应用分别申请 [Control API](https://tai.it.woa.com/apps/mcp-l2m93f/apis/my/published/52725c12-2123-465e-8fcf-c08976cfe5a7/detail)
   和 [Application API](https://tai.it.woa.com/apps/mcp-l2m93f/apis/my/published/e5c783dd-c9c3-4e3f-b801-daca0aa0180e/detail) 权限。
3. 将 `app_id` 提交给 Nexus 管理员，在 Admin 用户列表中绑定到该用户的 system-default token。一个 `app_id` 只能绑定一个用户。

应用请求按绑定用户鉴权和计量：Execution 的 owner、Environment Grant 和资源用量均归该用户。应用没有独立配额，绑定不会
扩大用户权限。应用 Token 只保存在调用方，不提交给 Nexus。

把以下基址替换为太湖订阅结果中的发布前缀，去掉末尾 `/v1`，然后选择应用请求函数：

```bash
export CONTROL_BASE_URL='<control-published-api-prefix>'
export APPLICATION_BASE_URL='<application-published-api-prefix>'
export TAI_APP_ID='<your-tai-app-id>'
export TAI_APP_TOKEN_FILE="$HOME/.nexus/auth/tai-app-token"
chmod 600 "$TAI_APP_TOKEN_FILE"

nexus_request() {
  local method="$1" url="$2"
  shift 2
  local timestamp nonce token signature
  timestamp="$(date +%s)"
  nonce="$(openssl rand -hex 16)"
  token="$(cat "$TAI_APP_TOKEN_FILE")"
  signature="$(printf '%s' "${timestamp}${token}${nonce}${timestamp}" \
    | sha256sum | awk '{print toupper($1)}')"
  curl --fail-with-body -sS -X "$method" "$url" \
    -H "x-rio-paasid: $TAI_APP_ID" \
    -H "x-rio-timestamp: $timestamp" \
    -H "x-rio-nonce: $nonce" \
    -H "x-rio-signature: $signature" "$@"
}
```

签名示例使用 Bash、`openssl` 和 `sha256sum`。时间戳为 Unix 秒，与标准时间误差不能超过 180 秒；每次请求生成新的 nonce。
不要发送 Nexus 内部身份头，也不要自行构造后端 `timestamp`、`x-rio-seq`、`signature`。后端签名验证配置由 API 负责人维护。

### JSON 与流式请求

选定上述一种 `nexus_request` 后，在同一终端定义 JSON 包装函数。它同时检查 HTTP 失败，以及 SGW 可能通过 HTTP `200`
返回的 `errcode` 错误；适用于两种认证方式：

```bash
nexus_json() {
  local response status
  response="$(nexus_request "$@")" || {
    status=$?
    printf '%s\n' "$response" >&2
    return "$status"
  }
  if printf '%s' "$response" | jq -e 'type == "object" and has("errcode")' >/dev/null 2>&1; then
    printf '%s\n' "$response" | jq . >&2
    return 22
  fi
  printf '%s\n' "$response"
}

nexus_json GET "$CONTROL_BASE_URL/v1/me" | jq .
```

`/v1/me` 成功只证明接入身份有效，还需检查 Environment Grant。JSON API 使用 `nexus_json`；SSE 使用
`nexus_request ... -N`，不要用 JSON 包装函数缓存事件流。

<a id="领取-environment"></a>

## 申请 Environment 配额

1. 打开控制台的 **My Environments**。
2. 找到可用 Environment，查看 **Allocation** 和 **Default quota**。
3. **Access** 为 `available` 时点击 **Claim quota**。
4. 确认 **Access** 为 `granted`，并检查 **Granted limits**。
5. 提交时把 Environment ID 传给 `--environment`。

![My Environments 中申请配额的 Claim quota 按钮及申请后的 Granted limits](../assets/images/nexus-cloud/environment-quota.png)

示意图使用 dev 控制台界面和示例数据：`available` 行显示 **Claim quota**；申请后 **Access** 为 `granted`，
并显示 **Granted limits**。实际 Environment ID 和额度以自己所在部署的页面为准。

默认配额由部署和 Environment 配置决定，以页面的 **Default quota / Granted limits** 为准。常见默认配置为
5 managers、10 cores、20 GiB、100 slots、2 batches；如需更高额度，联系 Oncall 同学。

## Dynamic 与 Predefined

| 类型 | 资源方式 | 常见用途 |
| --- | --- | --- |
| Dynamic | 按提交申请 Manager，任务结束后回收 | Batch、需要自定义代码的任务 |
| Predefined | 使用预先存在的 Manager pool | 固定版本、低启动延迟任务 |

Dynamic 提交的资源不能超过个人 Grant 和 Environment 容量。Predefined 提交使用已有 pool，指定 `--pool` 和
`--config-only`，不申请新的 Manager。代码、镜像与依赖的详细约束保留在 [Batch 指南](submit-batch.md#代码与运行环境)。

## 资源、容量与隔离

- **Manager**：承载云端任务的执行进程。Single/Live 使用 Execution Manager；Batch 还使用 Batch Manager 管理子任务。
- **CPU cores / Memory**：所有 Manager 的 CPU、内存总量，单个 Manager 的规格与数量分别影响总用量。
- **Slots**：Manager 的逻辑并发容量。增加 slots 不会增加 CPU、内存，也不为每个任务提供独占资源。
- **Active batches**：同时运行的 Batch 数量，不是子任务数量。
- **Runtime**：任务使用的沙盒或容器，其服务凭据和资源额度单独管理。

Batch Manager 也占用一个 Manager 额度及自己的 CPU、内存；它不计入 child 执行容量。不同 Manager 分别分配 CPU 和内存，
同一 Manager 内的任务共享资源。Dynamic 创建并回收 Manager，Predefined 复用已有 pool，但两者都受运行环境和授权限制。

## 任务凭据

Nexus API 凭据用于提交和管理任务；模型、Runtime、Dataset 凭据用于任务执行；Trace Database 只读账号用于导出或查询轨迹。
这些凭据不能互相替代。

| 配置方式 | 何时读取凭据 | 提交前检查 |
| --- | --- | --- |
| `${oc.env:...}` | 在本地解析配置时读取 | 本地变量已配置；解析后的任务配置按敏感数据管理 |
| `token__env`、`api_key__env` | 在任务执行环境中读取 | 对应变量已按环境约定注入云端；本地 `export` 不会自动转发 |
| `--wedata-key-file` | 远程 Generator 读取 WeData 时使用 | 使用私有凭据文件，不写入 Generator |

`--host-env` 用于普通环境变量，其值会进入提交配置和 Pod 清单，不用于传递密码或 Token。任务运行时凭据的注入方式由所在环境
约定，未配置时先联系环境管理员；不要把真实值提交到 Git。
