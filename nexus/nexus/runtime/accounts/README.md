# nexus.runtime.accounts — 账号池异步 SDK

对 `nexus-account-pool-server` 的**纯异步**封装。核心类 `AccountPoolManager` 通过
统一的 `AsyncAccountPoolClient`（基于 `aiohttp`）与服务端交互，并提供
`AsyncReservation` / `AsyncMultiReservation` 两个 async context manager，负责
自动心跳与自动释放。

> 本 SDK 已经全面异步化：**不存在同步链路**。所有网络请求都走
> `ExecutionContext → AccountPoolManager → AsyncAccountPoolClient (aiohttp)`。

---

## 目录

- [1. 安装与初始化](#1-安装与初始化)
- [1.1 管理后台：建新池还是打 tag](#11-管理后台建新池还是打-tag)
- [2. 使用流程（快速上手）](#2-使用流程快速上手)
- [3. 服务端状态模型](#3-服务端状态模型)
- [4. 对外暴露的类与字段](#4-对外暴露的类与字段)
- [5. `AccountPoolManager` API 一览](#5-accountpoolmanager-api-一览)
- [6. 状态转换全景图](#6-状态转换全景图)
- [7. 账号可用性与 `invalidate` 语义](#7-账号可用性与-invalidate-语义)
- [8. 异常层级](#8-异常层级)
- [9. 参考用例](#9-参考用例)

---

## 1. 安装与初始化

SDK 已作为 `nexus/runtime/accounts` 子包内置，无需单独安装。`__init__.py` 只做
`re-export`，所有实现分散在 `manager.py` / `client.py` / `acquired.py` /
`reservation.py` / `models.py` / `exceptions.py`。

底层 HTTP 通信直接复用 `nexus.utils.async_http.http_client`（全局共享的
`aiohttp` 客户端，session 按 event loop 缓存并由 `atexit` 统一清理），SDK
本身不管理连接生命周期，也不再依赖 `requests`。

```python
from nexus.runtime.accounts import AccountPoolManager

mgr = AccountPoolManager(
    endpoint=None,        # 默认取 NEXUS_ACCOUNT_POOL_ENDPOINT
    ttl_seconds=60,       # 默认租约 TTL
)
```

大多数业务代码不必自己 `new`，直接：

```python
from nexus.core.context import ExecutionContext

async with ExecutionContext() as ctx:
    account = await ctx.get_account('github')
    # 使用 account.credentials / account.tokens
# ExecutionContext 退出时会自动 release_all() 并 aclose()
```

---

## 1.1 管理后台：建新池还是打 tag

池、账号、tag 的增删改在管理后台完成，任务代码里只领号，不创建资源：

**https://nexus-pool.woa.com/**

任务侧：`await ctx.get_account('github')` 或 `await ctx.get_account('gui-sandbox', tags=['chrome', 'vscode'])`。

**建新池**（管理页「新建池」），下面任一成立就拆：

- 凭证形态不同（GitHub token vs GUI 沙盒 cookie）
- 环境 / 租户 / 配额 / 默认 TTL / 启停要独立
- 调用方需要两个不能混领的稳定池名（`github` 和 `gui-sandbox`）

**给账号打 tag**（加号或编辑账号，逗号分隔如 `chrome, vscode`），当：

- 同一池、同一类资源，只是能力不同、可组合
- 一台号要能被多种请求领走：带 `chrome`+`vscode` 的号，既能给只要 chrome 的任务，也能给两个都要的任务

不要为每种 tag 组合再建一个池。库存会割裂：chrome 任务在饿，`gui-chrome-vscode` 池里却闲着。

若每个账号都只服务一种用途、任务里每次都写死同一组 `tags=`，那是假 tag，应拆成独立池。

不传 `tags` / 传 `None` / `[]`：池内任意 idle 都可领。传入则 AND 匹配，账号必须同时具备列表里每一个 tag；账号上多出来的 tag 不妨碍匹配。

---

## 2. 使用流程（快速上手）

### 2.1 三种典型场景

| 场景 | 推荐 API | 说明 |
|------|----------|------|
| 拿一个账号跑一段活，用完就还 | `mgr.acquire()` + `AcquiredAccount.release()` | 最常见 |
| 一次跑一批任务（如 3 个 crawler），需要独占 3 个账号 | `mgr.reservation()`（async CM） | 自动 cancel、支持中途 acquire/release 往返 |
| 跨 pool 原子拿号（github×2 + gitlab×1） | `mgr.multi_reservation()`（async CM） | 任一 pool 不够整体回滚 |

### 2.2 标准调用顺序

```
      ┌──────────────────────────────────────────────────────────────┐
      │ 1. 打开 ExecutionContext（或手工 new AccountPoolManager）    │
      └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ 2. 拿账号                                                    │
      │    ─ 单次：await mgr.acquire('github', ttl_seconds=60)       │
      │    ─ 批量：async with mgr.reservation('github', 3) as rsv:   │
      │             async with rsv.acquire() as act: ...             │
      └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ 3. 业务使用（心跳后台任务自动续期）                          │
      │    act.credentials / act.tokens                              │
      │    如果发现凭据不可用：await act.invalidate(reason='401')    │
      └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ 4. 归还                                                      │
      │    ─ 单次：await act.release()  或 await mgr.release(lease)  │
      │    ─ 批量：退出 async with 自动 cancel_reservation()         │
      └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ 5. 收尾：await mgr.release_all() ；await mgr.aclose()        │
      │    （ExecutionContext 退出时自动完成）                        │
      └──────────────────────────────────────────────────────────────┘
```

### 2.3 接口分组一览

| 分组 | 接口 | 用途 |
|------|------|------|
| **流转** | `acquire` / `release` / `release_all` / `heartbeat` | 单账号租借周期 |
| **批量预留（直接调用）** | `reserve` / `reserve_multi` / `cancel_reservation` | 手工控制批次 |
| **批量预留（context manager）** | `reservation` / `multi_reservation` | 推荐用法，自动 cancel |
| **管理** | `add_account` / `delete_account` / `list_accounts` / `invalidate` / `recover` / `get_pool_stats` | 池运维 |
| **生命周期** | `aclose`（异步） | 关闭 aiohttp session |
| **`AcquiredAccount` 实例方法** | `release` / `invalidate` | 用完归还或标记不可用 |

> ⚠️ 老版本的同步 `close()` 已废弃，现在统一是 `aclose()`（异步）。`ExecutionContext`
> 退出时会自动调用。

---

## 3. 服务端状态模型

一个账号只有 **4 种状态**（服务端 proto 定义）：

| 状态 | 含义 |
|------|------|
| `idle` | 空闲，可被 acquire 或 reserve |
| `reserved` | 被某个 reservation_token 预留住，可继续被同 token acquire |
| `acquired` | 已租出，正在使用；靠 heartbeat 续 expire_at |
| `invalid` | 账号级失效（凭据不可用），只有 `recover` 能救回 |

> 共享账号（`shared_cap > 0`）:
> - 槽位未满时 `status = idle`；批次内有 `reserved` 占位时 `status = reserved`；槽位占满时 `status = acquired`。
> - `shared_occupied` 是当前占用的槽位总数（含 `reserved` 占位与 `acquired` 实际持有）。
> - 共享账号也可参与 `reserve()`，会按份额向 `account_leases` 插入占位租约。
> - 共享账号的 `invalid` 是**账号级**（一份凭据挂了）——详见 §7。

---

## 4. 对外暴露的类与字段

`from nexus.runtime.accounts import ...` 可用符号（对应 `__init__.py` 的
`__all__`，纯 re-export）：

```
# Manager & data holders
AccountPoolManager   AcquiredAccount   AsyncAccountPoolClient
AsyncReservation     AsyncMultiReservation

# SDK types
Account   Lease   PoolStats   ReserveResult

# Exceptions
AccountPoolError   AcquireError   HeartbeatError   InvalidateError
NoAvailableAccountError   ReleaseError   ReservationError
```

### 4.1 `AcquiredAccount`（`acquire` 的返回值）

| 字段 | 类型 | 说明 |
|------|------|------|
| `lease_id` | `str` | 服务端下发的租约 ID，全局唯一 |
| `account_id` | `int` | 服务端账号 ID |
| `pool` | `str` | 池名（如 `github`） |
| `account_key` | `str` | 池内账号 key（如 `user1`） |
| `credentials_raw` | `str` | 原始 JSON 字符串 |
| `tokens_raw` | `str` | 原始 JSON 字符串 |
| `ttl_seconds` | `int` | 租约 TTL |
| `credentials` *(property)* | `dict` | `credentials_raw` 反序列化 |
| `tokens` *(property)* | `dict` | `tokens_raw` 反序列化 |
| `invalidated` *(property)* | `bool` | 是否已被 invalidate |
| `_reservation_token` | `str` | *internal*，来源预留时的 token（无预留则为 `''`）。**非空时，`release` 会自动把账号还回 `reserved`**（保留在批次内）；为空则 `release` 后回 `idle` |

方法：

| 方法 | 作用 |
|------|------|
| `await act.release()` | 停心跳 + release lease；若 `_reservation_token` 非空会自动带上 `reservation_token`，账号回 `reserved` 而非 `idle`，批次的最终释放由 `cancel_reservation` 完成 |
| `await act.invalidate(reason='')` | 停心跳 + 服务端标记账号 `invalid` |

### 4.2 `Account`（`list_accounts` 元素）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `int` | 账号 ID |
| `pool` | `str` | 池名 |
| `account_key` | `str` | 池内 key |
| `credentials_raw` / `tokens_raw` | `str` | 原始 JSON |
| `status` | `str` | `idle` / `reserved` / `acquired` / `invalid` |
| `shared_cap` | `int` | 共享容量，0 表示独占 |
| `shared_occupied` | `int` | 当前共享占用数 |
| `lease_id` | `str` | 若 `acquired`，当前 lease |
| `reservation_id` | `str` | 若 `reserved`，预留 ID |
| `expire_at` | `str` | 到期时间字符串 |
| `last_used_at` / `created_at` / `updated_at` | `str` | 审计字段 |
| `note` | `str` | 备注（≤ 512 字符） |
| `credentials` / `tokens` *(property)* | `dict` | 反序列化视图 |

### 4.3 `Lease`

`acquire` 底层 HTTP 返回的原始租约信息（一般用户不用直接接触，`AcquiredAccount`
已经完成封装）。字段：`lease_id`, `account_id`, `credentials_raw`,
`tokens_raw`, `expire_at`。

### 4.4 `PoolStats`（`get_pool_stats` 元素）

| 字段 | 说明 |
|------|------|
| `pool` | 池名 |
| `exclusive_total` / `exclusive_idle` / `exclusive_acquired` / `exclusive_reserved` / `exclusive_invalid` | 独占账号数量分布 |
| `shared_total` / `shared_capacity` / `shared_occupied` / `shared_invalid` | 共享账号数量分布 |

### 4.5 `ReserveResult`（`reserve` / `reserve_multi` 返回）

| 字段 | 说明 |
|------|------|
| `reservation_token` | 预留批次的凭证，后续 acquire/cancel 必须带 |
| `account_ids` | 打平的账号 ID 列表（单 pool 场景直接用） |
| `account_ids_by_pool` | `{pool: [ids]}`，多 pool 场景使用；单 pool 也会填 |

### 4.6 `AsyncReservation`（`mgr.reservation()` 上下文对象）

| 属性 | 说明 |
|------|------|
| `pool` | 预留的池名 |
| `token` | `reservation_token` |
| `account_ids` | 被预留的账号 ID 列表 |

方法：`async with rsv.acquire(ttl_seconds=None) as act: ...`

### 4.7 `AsyncMultiReservation`

| 属性 | 说明 |
|------|------|
| `token` | 跨池共享的 `reservation_token` |
| `account_ids_by_pool` | 按池分组的账号 ID |
| `pools` | 所有池名列表 |

方法：`async with rsv.acquire(pool, ttl_seconds=None) as act: ...`

### 4.8 `AsyncAccountPoolClient`

`AccountPoolManager` 底层调用的 HTTP 客户端，直接对应服务端 REST 接口。一般
业务代码不需要直接用；如果要自己写单元测试或做特殊控制（比如不想启心跳、
不想启用 asyncio task 管理），可以直接实例化它。

---

## 5. `AccountPoolManager` API 一览

按语义分四组。每个方法列出**完整签名 → 参数表 → 返回值 / 抛出异常**。所有方法
（除标注为同步的以外）都是 `async def`。

### 5.0 构造函数

```python
AccountPoolManager(endpoint: str | None = None, *, ttl_seconds: int = 60)
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `endpoint` | `str \| None` | `None` | 服务地址。`None` 时按 `NEXUS_ACCOUNT_POOL_ENDPOINT` 环境变量取，再退回 `http://nexus-pool.production.polaris:8080` |
| `ttl_seconds` | `int` | `60` | 默认租约 TTL，作用于所有 `acquire(ttl_seconds=None)` 场景 |

---

### 5.1 流转（circulation）

#### `acquire`
```python
async def acquire(
    pool: str,
    *,
    ttl_seconds: int | None = None,
    reservation_token: str = '',
    enable_heartbeat: bool = True,
    tags: list[str] | None = None,
) -> AcquiredAccount
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pool` | `str` | *必填* | 池名（如 `'github'`），须在 [管理后台](https://nexus-pool.woa.com/) 已创建 |
| `ttl_seconds` | `int \| None` | `None` | 租约 TTL，`None` 时用构造时的默认值。到期未心跳，服务端 reaper 会强制回收 |
| `reservation_token` | `str` | `''` | 非空表示从该批预留中取号；**传了 token 后，`release` 会自动把账号还回 `reserved`（保留在批次内）**，需要彻底释放整批账号请调 `cancel_reservation` |
| `enable_heartbeat` | `bool` | `True` | `True` 时启动后台 `asyncio` 心跳任务（间隔 = `ttl/3`，最小 1 秒） |
| `tags` | `list[str] \| None` | `None` | 可选。AND 匹配，账号须同时具备每一个 tag。示例：`tags=['chrome', 'vscode']`。不传 / `None` / `[]` 不按 tag 筛。何时打 tag、何时建新池见 [1.1](#11-管理后台建新池还是打-tag) |

- **返回**：`AcquiredAccount`
- **抛出**：`NoAvailableAccountError`（池空）、`AcquireError`（其他失败）
- **异常安全**：若拿到 lease 后 SDK 内部任何步骤失败（如查 account_key、起心跳协程），
  会主动 `release` 该 lease，避免服务端遗留孤儿 lease

#### `release`
```python
async def release(identifier: str) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `identifier` | `str` | 可以是 `AcquiredAccount.lease_id`，也可以是**池名**（按池名释放最近一次 acquire 的账号） |

- **副作用**：停对应心跳任务；若账号 `_reservation_token` 非空，会自动把 `reservation_token` 传给服务端，账号回 `reserved`；否则直接回 `idle`
- **不抛异常**（内部捕获 `AccountPoolError` 并日志告警）

#### `release_all`
```python
async def release_all() -> None
```
无参。批量对当前 manager 持有的每一个账号调 `release`。`ExecutionContext` 退出时会自动调用。

#### `heartbeat`
```python
async def heartbeat(lease_id: str, *, ttl_seconds: int = 60) -> str
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lease_id` | `str` | *必填* | acquire 时得到的 lease |
| `ttl_seconds` | `int` | `60` | 续期后的新 TTL（秒） |

- **返回**：服务端下发的新 `expire_at` 字符串
- **抛出**：`HeartbeatError`
- 一般不需要手工调，由 `enable_heartbeat=True` 时的后台任务自动完成

---

### 5.2 批量预留（reservation）

#### `reserve`
```python
async def reserve(pool: str, count: int, *, ttl_seconds: int = 300, tags: list[str] | None = None) -> ReserveResult
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pool` | `str` | *必填* | 池名 |
| `count` | `int` | *必填* | 预留数量，必须 > 0 |
| `ttl_seconds` | `int` | `300` | 整批预留的生存时长；到期未 acquire/cancel，服务端会自动回收 |
| `tags` | `list[str] \| None` | `None` | 与 `acquire` 相同，只筛该池 |

- **返回**：`ReserveResult`
- **抛出**：`ReservationError`（例如账号不足、参数非法）
- **语义**：all-or-nothing。数量不足时**不会**部分预留

#### `reserve_multi`
```python
async def reserve_multi(items: List[Dict[str, Any]], *, ttl_seconds: int = 300) -> ReserveResult
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `items` | `List[Dict]` | *必填* | 每项 `{'pool': str, 'count': int, 'tags': list[str]?}`；`pool` 非空、`count > 0`；`tags` 可选 |
| `ttl_seconds` | `int` | `300` | 整批（跨池共享同一 token）的 TTL |

- **返回**：`ReserveResult`（`account_ids_by_pool` 按池分组，`account_ids` 打平）
- **抛出**：`ReservationError`（任一池账号不足则整体回滚）

#### `cancel_reservation`
```python
async def cancel_reservation(reservation_token: str) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `reservation_token` | `str` | `reserve` / `reserve_multi` 返回的 token，不能为空 |

- **副作用**：批次里所有 `reserved` / `acquired`（未释放）账号回 `idle`
- **抛出**：`ReservationError`

#### `reservation`（async context manager）
```python
@asynccontextmanager
async def reservation(
    pool: str,
    count: int,
    *,
    ttl_seconds: int = 300,
    lease_ttl_seconds: int = 60,
    enable_heartbeat: bool = True,
) -> AsyncReservation
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pool` | `str` | *必填* | 池名 |
| `count` | `int` | *必填* | 预留数量 |
| `ttl_seconds` | `int` | `300` | 批次 TTL |
| `lease_ttl_seconds` | `int` | `60` | 后续 `rsv.acquire()` 的默认单次租约 TTL |
| `enable_heartbeat` | `bool` | `True` | 内层 acquire 是否启动心跳 |

- 内层 `rsv.acquire()` 释放时自动回 `reserved`（因为内部带了 token），保证账号不被批次外抢走
- 退出 `async with` 时自动 `cancel_reservation`，剩余账号全部回 `idle`
- 内部 `rsv.acquire()` 接受一个可选参数 `ttl_seconds: int | None`

#### `multi_reservation`（async context manager）
```python
@asynccontextmanager
async def multi_reservation(
    items: List[Dict[str, Any]],
    *,
    ttl_seconds: int = 300,
    lease_ttl_seconds: int = 60,
    enable_heartbeat: bool = True,
) -> AsyncMultiReservation
```

参数含义同 `reservation`，但 `items` 是多池数组。`rsv.acquire(pool, *, ttl_seconds=None)`
必须指明池名，且该池必须在 `items` 里，否则抛 `AccountPoolError`。

---

### 5.3 管理（admin）

#### `add_account`
```python
async def add_account(
    pool: str,
    account_key: str,
    credentials: Any = None,
    tokens: Any = None,
    *,
    shared_cap: int = 0,
    note: str = '',
    tags: list[str] | None = None,
) -> int
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pool` | `str` | *必填* | 池名 |
| `account_key` | `str` | *必填* | 池内唯一 key（重复会报错） |
| `credentials` | `dict \| str \| None` | `None` | 凭据；dict 会 JSON 序列化，字符串按原样发送；`None` → `"{}"` |
| `tokens` | `dict \| str \| None` | `None` | 同上 |
| `shared_cap` | `int` | `0` | `0`=独占；`>0`=共享账号，允许最大并发使用数 |
| `note` | `str` | `''` | 备注，服务端最长 512 字符 |
| `tags` | `list[str] \| None` | `None` | 创建时写入账号的 tags；空字符串会被忽略，没有有效 tag 时不发送该字段 |

- **返回**：新账号的 `account_id`
- **抛出**：`AccountPoolError`

#### `delete_account`
```python
async def delete_account(account_id: int) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `account_id` | `int` | 要物理删除的账号 ID |

- **抛出**：`AccountPoolError`

#### `list_accounts`
```python
async def list_accounts(
    *,
    pool: str = '',
    status: str = '',
    offset: int = 0,
    limit: int = 20,
) -> List[Account]
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pool` | `str` | `''` | 池名过滤；`''` 表示不过滤 |
| `status` | `str` | `''` | 状态过滤：`'idle' / 'reserved' / 'acquired' / 'invalid'`；`''` 表示不过滤 |
| `offset` | `int` | `0` | 分页偏移 |
| `limit` | `int` | `20` | 分页大小 |

- **返回**：`List[Account]`
- **抛出**：`AccountPoolError`

#### `invalidate`
```python
async def invalidate(account_id: int) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `account_id` | `int` | 要标记为 `invalid` 的账号 ID |

- **抛出**：`InvalidateError`
- 语义：**账号级失效**，参见 §7
- 注：一般在业务里用 `AcquiredAccount.invalidate(reason='...')`，会同时停心跳并写日志

#### `recover`
```python
async def recover(account_id: int) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `account_id` | `int` | `invalid` 状态的账号 ID |

- **抛出**：`InvalidateError`
- **效果**：`invalid` → `idle`

#### `get_pool_stats`
```python
async def get_pool_stats(pool: str = '') -> List[PoolStats]
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pool` | `str` | `''` | 池名；`''` 返回全部池的统计 |

- **返回**：`List[PoolStats]`
- **抛出**：`AccountPoolError`

---

### 5.4 生命周期

#### `aclose`（**异步**）
```python
async def aclose() -> None
```
停掉所有心跳任务，关闭底层 aiohttp session。**不会**自动 `release_all()`，若还有租约请先手工调 `release_all`。`ExecutionContext` 退出时会自动完成。

> ⚠️ 之前版本的同步 `close()` 已不复存在。

---

### 5.5 `AcquiredAccount` 实例方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `await act.release()` | 无 | 停心跳 + 释放；已释放/失效则幂等返回 |
| `await act.invalidate(reason: str = '')` | `reason` 仅用于日志 | 停心跳 + 服务端标记 `invalid`；`ExecutionContext` 退出时会跳过它 |

### 5.6 `AsyncReservation` / `AsyncMultiReservation` 实例方法

`AsyncReservation.acquire`：
```python
@asynccontextmanager
async def acquire(*, ttl_seconds: int | None = None) -> AcquiredAccount
```

`AsyncMultiReservation.acquire`：
```python
@asynccontextmanager
async def acquire(pool: str, *, ttl_seconds: int | None = None) -> AcquiredAccount
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pool` *(仅 multi)* | `str` | *必填* | 必须是构造 `multi_reservation` 时传入的池之一，否则 `AccountPoolError` |
| `ttl_seconds` | `int \| None` | `None` | 单次 acquire 的租约 TTL；`None` 时用 `lease_ttl_seconds` |

---

## 6. 状态转换全景图

```
                     ┌────────────────────────────────────────────────┐
                     │                                                │
                     │  recover(id)                                   │
                     │                                                │
                     ▼                                                │
                ┌────────┐  reserve() / reserve_multi()  ┌──────────┐ │
                │        │ ──────────────────────────▶ │          │ │
    add ──────▶ │  idle  │                             │ reserved │ │
                │        │◀── cancel_reservation / TTL │          │ │
                └────────┘                             └──────────┘ │
                    │  ▲                                     │      │
                    │  │                                     │      │
     acquire(pool)  │  │  release(lease)                     │ acquire(pool,
                    │  │  或 release_to_                     │  reservation_
                    │  │  reserved=False                     │  token=…)
                    │  │                                     │      │
                    ▼  │                                     ▼      │
                ┌────────────────────────────────────────────────┐  │
                │                    acquired                     │  │
                │ heartbeat 续 expire_at；超时守护协程回 idle；   │──┘
                │ invalidate() / act.invalidate() 转 invalid      │
                └────────────────────────────────────────────────┘
                                     │
                                     │  invalidate(id) / act.invalidate(reason)
                                     ▼
                               ┌──────────┐
                               │ invalid  │──── recover(id) ───▶ idle
                               └──────────┘
```

**完整迁移表**（括号内为服务端 `db.go` 行号，便于回溯）：

| 迁移 | 触发操作 |
|------|----------|
| `-` → `idle` | `add_account` |
| `idle` → `acquired` | `acquire(pool)`（不带 token） *(L158)* |
| `idle` → `reserved` | `reserve()` / `reserve_multi()` *(L361 / L431)* |
| `reserved` → `acquired` | `acquire(pool, reservation_token=…)` |
| `reserved` → `idle` | `cancel_reservation()` *(L451)* |
| `reserved` → `idle` | TTL 过期，服务端守护协程回收 *(L756–L757)* |
| `acquired` → `idle` | `release(lease)` **不带** `reservation_token` *(L210)* |
| `acquired` → `reserved` | `release(lease, reservation_token=…)` *(L204)*<br/>**SDK 侧自动：只要 acquire 时带了 `reservation_token`，`release` 就自动回 `reserved`** |
| `acquired` → `idle` | 心跳超时，服务端守护协程回收 *(L747–L748)* |
| any → `invalid` | `invalidate(id)` / `act.invalidate(reason)` *(L601)* |
| `invalid` → `idle` | `recover(id)` *(L622)* |
| `acquired` 保持 | `heartbeat(lease, ttl)` 刷新 `expire_at` *(L297)* |

---

## 7. 账号可用性与 `invalidate` 语义

### 7.1 acquire / reserve 自动跳过 `invalid`

所有选号入口都会**自动过滤掉 `status = 'invalid'` 的账号**，SDK/用户不需要在
外面再做一次可用性判断：

| 入口 | SQL 过滤条件（服务端） |
|------|------------------------|
| 独占 `acquire` | `status IN ('idle', 'reserved')` |
| 独占 `reserve` | `status = 'idle' AND reservation_id = '' AND shared_cap = 0` |
| 共享 `acquire` | `status IN ('idle', 'reserved')`（外层过滤后再翻牌/新增租约） |
| 共享 `reserve` | `shared_cap > 0 AND status != 'invalid' AND shared_used < shared_cap` |

一旦某个账号被标记 `invalid`，后续任何 `acquire` / `reserve` 都不会再把它选出来，
直到有人显式 `recover(id)`。

### 7.2 `invalidate` 是**账号级失效**

`invalidate(account_id)` 表达的是"**这份凭据挂了**"（例如密码被改、token 被撤销、
被平台风控），因此其行为是：

**对独占账号**：
- `status` → `invalid`，`lease_id` / `reservation_id` 清空，`expire_at` 归零
- 如果该账号原本正被某批 reservation 持有，那批 reservation 的 `account_ids` 里
  它就变得不可用；后续 `acquire(reservation_token=…)` 少一个候选，`cancel_reservation`
  正常回滚剩下的

**对共享账号**（`shared_cap > 0`）：
- `status` → `invalid`，`shared_used` 归零
- 服务端 `DELETE FROM account_leases WHERE account_id = ?`，即**该账号在所有
  batch 里的 `reserved` 占位与所有正在使用的 `acquired` 租约会一并被撤销**
- 这是 by design：**一份凭据挂了，所有共享它的持有者都得同时失效**，不存在"槽位级
  invalidate"

### 7.3 对已持有该账号的其他 caller 的影响

- 别人正在使用（`acquired`）的 lease 已经在服务端被删除，其后台 `heartbeat` 协程
  会开始报 `HeartbeatError` 并**自动退出**（详见 `manager.py` 的 `_heartbeat_loop`）。
- 对应的 SDK `AcquiredAccount` 对象后续 `await act.release()` 是幂等吞异常的，
  不会 crash，只会记 warning。
- **推荐做法**：业务代码若发现凭据可能失效（比如 401、captcha），应当调
  `await act.invalidate(reason='...')`，让池整体感知——而不是继续吃着这一份坏凭据往下跑。

---

## 8. 异常层级

```
AccountPoolError            # 基类
├── AcquireError
│   └── NoAvailableAccountError   # 池空
├── ReleaseError
├── HeartbeatError
├── ReservationError
└── InvalidateError
```

推荐捕获策略：

- `NoAvailableAccountError`：可等待/换池/降级。
- `HeartbeatError`：多半是账号被别人 invalidate 或 lease 过期，业务层要重新 acquire。
- 其他 `AccountPoolError`：网络或服务端错误，一般重试或告警。

---

## 9. 参考用例

### 9.1 最常见：单个账号，用完即还

```python
from nexus.runtime.accounts import AccountPoolManager, NoAvailableAccountError

mgr = AccountPoolManager(ttl_seconds=60)
try:
    act = await mgr.acquire('github', ttl_seconds=30)
    # 使用凭据
    do_something(act.credentials['username'], act.tokens['access_token'])
    await act.release()
except NoAvailableAccountError:
    print('池里没账号')
finally:
    await mgr.aclose()
```

### 9.2 用 `ExecutionContext` 自动清理（推荐）

```python
from nexus.core.context import ExecutionContext

async with ExecutionContext() as ctx:
    account = await ctx.get_account('github')
    gui = await ctx.get_account('gui-sandbox', tags=['chrome', 'vscode'])
    ...
# 退出时自动 release_all() + aclose()
```

### 9.3 使用中发现凭据失效，主动 invalidate

```python
act = await mgr.acquire('github')
try:
    resp = await call_github_api(act.tokens['access_token'])
    if resp.status == 401:
        await act.invalidate('401 Unauthorized')   # 账号级失效，池整体感知
    else:
        await act.release()
except Exception:
    await act.release()
    raise
```

### 9.4 批量预留 + async context manager（推荐）

```python
# 一次预留 3 个 github 账号，跑完自动 cancel
async with mgr.reservation('github', count=3, ttl_seconds=300,
                           lease_ttl_seconds=60) as rsv:
    for _ in range(3):
        async with rsv.acquire() as act:
            do_task(act.credentials, act.tokens)
        # 内层退出：account 由 acquired 回到 reserved（仍在批次内）
# 外层退出：批次 cancel，所有账号回 idle
```

关键点：`AsyncReservation.acquire()` 内部会把 `reservation_token` 带给
`AcquiredAccount`，而 `release` 看到 token 非空就自动把账号回到 `reserved`，
仍属于该批次，避免"归还后被其他调用者抢走"的问题。

### 9.5 多池原子预留

```python
async with mgr.multi_reservation(
    [{'pool': 'github', 'count': 2},
     {'pool': 'gitlab', 'count': 1}],
    ttl_seconds=300, lease_ttl_seconds=60,
) as rsv:
    async with rsv.acquire('github') as gh:
        async with rsv.acquire('gitlab') as gl:
            cross_sync(gh, gl)
```

如果任一池账号不够，`reserve_multi` 会**全部回滚**，抛 `ReservationError`。

### 9.6 手工预留：直接调用（不用 context manager）

对应 e2e case 12，适合不用 context manager 的场景。**只要传了 `reservation_token`，
`release` 就会自动把账号还回 `reserved`，无需额外开关：**

```python
result = await mgr.reserve('github', 2, ttl_seconds=120)
try:
    a = await mgr.acquire(
        'github',
        ttl_seconds=10,
        reservation_token=result.reservation_token,
    )
    do_task(a.credentials)
    await a.release()                  # 账号自动回 reserved，仍在批次里
    # 需要再取一次同批次账号
    b = await mgr.acquire(
        'github',
        reservation_token=result.reservation_token,
    )
    await b.release()
finally:
    await mgr.cancel_reservation(result.reservation_token)   # 这里才真正回 idle
```

### 9.7 管理接口

```python
# 新增独占账号
aid = await mgr.add_account(
    'github', 'user1',
    credentials={'username': 'user1', 'password': 'p1'},
    tokens={'access_token': 'ghp_xxx'},
    note='测试账号',
    tags=['linux', 'git'],
)

# 新增共享账号（允许 3 并发）
bid = await mgr.add_account('github', 'shared_bot', shared_cap=3)

# 列出所有 idle 的 github 账号
accounts = await mgr.list_accounts(pool='github', status='idle', limit=100)
for a in accounts:
    print(a.id, a.account_key, a.status, a.note)

# 强制失效再手工恢复
await mgr.invalidate(aid)
await mgr.recover(aid)

# 池统计
for s in await mgr.get_pool_stats('github'):
    print(s.pool, s.exclusive_idle, '/', s.exclusive_total)

# 删除
await mgr.delete_account(bid)
```

### 9.8 端到端评测脚本

`tests/test_accounts_e2e.py` 覆盖了本 README 提及的所有状态迁移和 API：

```bash
python3 nexus/tests/test_accounts_e2e.py
```

最新一次运行结果：**12 / 12 全部通过**。
