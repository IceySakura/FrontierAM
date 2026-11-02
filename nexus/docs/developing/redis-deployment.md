# Redis Cluster 与 FS 服务

`backend: redis` 使用 Redis 保存记录和索引，并通过 FS 存储大字段。FS 由 Controller 和 DataUnit 组成。统一入口 `nexus redis up` 在已有 Ray 集群上依次启动 Redis Cluster 与 FS，等待全部服务就绪后返回；业务客户端不会自动启动服务。

默认 Primary 数为 `max(可用主机数, 3)`，Replica 数为 `0`，至少需要一台可用主机。同一主机可以运行多个 Redis 实例。数据按 16384 个 hash slot 分布在 Primary 上。

`nexus --help`、`nexus redis --help`、`nexus redis cluster --help` 和 `nexus redis fs --help` 不加载 Ray/ZMQ、不连接服务，也不读取太极拓扑。

已有 Ray 集群时，一条命令即可启动完整服务：

```bash
nexus redis up
```

主机列表通过 `--redis-node-ips` 显式指定，否则使用太极的 `NODE_IP_LIST`；两者均未提供或为空时，自动读取存活的 Ray 节点并按 IP 去重、排序。Ray head 在该列表内时排到首位，不会因为它是 head 而将其加入专用 Redis 主机列表。默认零副本，Primary 数至少为 `3`，不足三台主机时通过同机多实例满足数量。显式设置 `--primary-count` 可覆盖 Primary 总数，仍要求至少为 `3`。

默认实例分配示例：

| 可用主机 | Primary 数 | 实例分配 |
| --- | --- | --- |
| 1 台：A | 3 | `A:16379`、`A:16380`、`A:16381` |
| 2 台：A、B | 3 | `A:16379`、`B:16379`、`A:16380` |

## 资源与网络

Redis 实例总数为：`Primary 数 × (1 + 每主 Replica 数) + 额外热备数`，不需要 GPU。实例按主机列表 round robin 分配：第 `i` 个实例（从 `0` 开始）使用第 `i % 主机数` 台主机，数据端口为 `16379 + i // 主机数`，bus 端口为数据端口加 `10000`。若还有剩余太极节点，测试 Client 自动使用第一个剩余节点；否则复用第一个 Redis 主机。节点间需双向开放所有实例的端口：

- `16379` 起的连续端口：Redis 数据端口；
- `26379` 起的连续端口：Redis Cluster bus 端口，仅供节点间通信；
- Ray head 的 `17000`：默认 Controller 端口；
- DataUnit 所在节点的动态 TCP 端口：blob 读写。

默认单主机示例使用数据端口 `16379`–`16381` 和 bus 端口 `26379`–`26381`；两主机示例中，A 使用数据端口 `16379`–`16380` 和 bus 端口 `26379`–`26380`，B 使用 `16379` 和 `26379`。同一主机上的实例共享机器故障风险；同机 Replica 可承接单个 Redis 进程故障，但不能保护整机故障。

默认零副本不提供自动故障切换；任一 Primary 故障时，Cluster 会因槽位不完整而不可用。需要副本时使用 `--replicas-per-primary 1`，它在 Primary 之外增加副本，不改变默认 Primary 数；`--extra-replicas` 同样不参与默认 Primary 数的计算。当前配置启用 AOF `everysec`。默认目录位于 `/tmp`，生产环境通过 `--redis-dir` 指向各节点的持久卷，并配置独立备份。DataUnit 的运行期 SSD offload 不提供重启恢复。

## 太极启动

使用 Nexus 仓库作为太极代码源。所有节点需要相同的 Nexus 环境、Redis/ZMQ 依赖和 Ray：

```bash
uv sync --extra redis
uv pip install 'ray>=2.9'
export PATH="$TAIJI_BASIC_CODE_PATH/.venv/bin:$PATH"
export PYTHONPATH="$TAIJI_BASIC_CODE_PATH${PYTHONPATH:+:$PYTHONPATH}"
```

Ray 集群由太极环境启动，本工具连接 `address=auto`。`prepare` 命令运行包内的 `redis_server_launcher_prepare.sh`，保留针对太极生成的 sysctl 命令的兼容处理；它只用于对应太极环境，不是通用本机安装器。

Init Command：

```bash
nexus redis cluster prepare
```

Start Command 只在 launcher 执行，同时启动 Redis Cluster 和 FS，发布 Redis endpoint 并持续监控：

```bash
nexus redis serve
```

Redis 配置通过命令行传入，不再读取 `NEXUS_REDIS_*` 服务配置环境变量。`up`、`serve`、`down` 和 `cluster` 管理命令共用以下 Redis 参数：

| 参数 | 默认值与含义 |
| --- | --- |
| `--redis-node-ips` | 可选逗号分隔主机列表；未指定时使用 `NODE_IP_LIST` 或发现存活 Ray 节点 |
| `--primary-count` | Primary 总数，最少 `3`；默认 `max(可用主机数, 3)`，与副本数量无关 |
| `--replicas-per-primary` | 每个 Primary 的基础 Replica 数，默认 `0` |
| `--extra-replicas` | 额外热备 Replica 数，默认 `0` |
| `--redis-client-ip` | 可选测试 Client 主机，必须在主机列表内 |
| `--redis-dir` | 默认 `/tmp/hykv-redis-cluster`，Redis 数据和 `nodes.conf` 目录 |
| `--redis-endpoint-file` | 仅 `serve`，指定发布的 Redis endpoint 文件 |

保留的平台变量为 `NODE_IP_LIST`（太极 `IP:slot,IP:slot,...` 列表）以及 `TAIJI_BASIC_OUTPUT_PATH`（太极输出目录）。Redis/FS 的默认地址和首次创建顺序从已连接的 Ray 集群发现 head，不再使用 `CHIEF_IP`；无法发现 head 时明确报错。

`up` 和 `serve` 同时支持 FS 参数 `--num-data-units`、`--controller-port`（默认 `17000`）、`--startup-timeout`（默认 `60` 秒）、`--ssd-offload-dir`、`--memory-waterline-bytes` 和 `--memory-safe-line-bytes`。`down` 使用相同 Redis 拓扑参数及 `--controller-port`，先停止 FS，再停止 Redis。非默认拓扑必须在后续状态、恢复、停止和测试命令中传入同样参数。

程序先用基础实例执行 `redis-cli --cluster create`，Primary/Replica 分配由 Redis CLI 完成，再将额外实例加入副本最少的 Primary。额外实例从加入时起就是正常 Replica，`cluster-migration-barrier 1` 允许它在某个 Primary 因故障切换失去副本后自动迁移。后续启动保留每个实例的 `nodes.conf`，不会重置故障切换后的角色和 config epoch。

每台主机的数据端口 `16379` 实例使用 `--redis-dir` 目录；其他实例使用该目录下以端口命名的子目录，例如 `16380/`。各实例分别保存配置、PID、日志、数据和 `nodes.conf`，互不覆盖。

重启已有集群时，保留原先的主机列表顺序和主副本配置；修改数量不提供已有集群的在线扩缩容。

旧规则下，8 台主机配合 `--replicas-per-primary 1` 默认创建 4 Primary + 4 Replica；新规则下，同样参数默认创建 8 Primary + 8 Replica。继续管理已有的 4 Primary + 4 Replica 集群时，需显式传入 `--primary-count 4 --replicas-per-primary 1`，不能省略 Primary 数将其当作在线扩容。

## 连接

`serve` 启动成功后，日志和 Redis endpoint 文件会输出：

- `redis_url`：包含全部 startup nodes 的 Cluster URL；
- `startup_nodes`：包含各实例的 `host:port`，同一主机可出现多个不同端口；
- 基础数据端口和基础 Cluster bus 端口。

文件路径优先使用 `--redis-endpoint-file`，其次使用 `${TAIJI_BASIC_OUTPUT_PATH}/redis-cluster.json`，两者均未设置时写入 Redis 数据目录的 `endpoints.json`。`up` 启动并检查服务后返回，不发布 Redis endpoint 文件。

业务客户端默认从当前进程已连接的 Ray 集群发现 head，使用其 `16379` 端口的 Redis 和 `17000` 端口的 Controller。客户端不会自动连接或启动 Ray；需要默认地址但未连接 Ray 或无法发现 head 时明确报错。显式 Redis URL 优先，Controller 地址仍独立配置，不从该 URL 推导。需要覆盖 Controller 时使用映射：

```python
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient

client = RedisClient(
    url='redis://10.0.0.1:16379/0',
    controller_endpoint={'host': '10.0.0.1', 'port': 17000},
)
```

`controller_endpoint` 支持 `Endpoint` 对象或映射，不接受字符串。映射省略 `host` 时发现 Ray head，省略 `port` 时使用 `17000`；省略整个配置时使用同样默认值。显式 Controller host/Endpoint 或借用的 `fs_client` 不触发 FS 地址发现；显式 Redis URL 或注入的 `redis_client` 不触发 Redis 地址发现。两侧均无需发现地址时，独立客户端无需安装或连接 Ray。只借用 FS、仍需补 Redis URL 时，仍要从已连接 Ray 发现 head。缺少可用 Controller 时按连接错误报错。

专用 Redis 主机列表不包含 Ray head 时，业务必须显式使用发布的 Redis URL；FS Controller 仍部署在 Ray head。

`redis://` 和 `rediss://` 在构造连接时通过 `INFO cluster` 检测；需要该命令的访问权限，连接、认证或检测失败均直接报错。检测到 Cluster 后保留原 URL 的认证、TLS 和连接参数，并从该节点发现完整拓扑。显式 `redis+cluster://host:port/0?nodes=host2:port2,host3:port3` 和 Sentinel URL 继续可用。Redis Cluster 只支持 DB 0，客户端必须能访问 Cluster 公布的所有节点。

Redis store 使用客户端生成的 UUID 字符串作为 Node ID，避免依赖复制计数器。Cluster 模式的批量写使用跨槽非事务 pipeline，批量读使用 `mget_nonatomic` 按槽拆分，因此跨分片批量操作不保证原子性。

生成的 Redis 配置只绑定回环和指定集群 IP，但不配置密码或 TLS，只能部署在严格隔离的内网/安全组中。主机被替换且本地 `nodes.conf` 丢失时，当前 `recover` 不会自动执行节点替换。

## 常用命令

```bash
uv run --no-sync nexus redis cluster prepare
uv run --no-sync nexus redis serve
uv run --no-sync nexus redis up
uv run --no-sync nexus redis cluster status
uv run --no-sync nexus redis cluster smoke
uv run --no-sync nexus redis cluster recover
uv run --no-sync nexus redis down
```

`serve` 中断后 Redis 和后台 FS 继续运行；停止完整服务使用 `nexus redis down`。`cluster` 子命令仅管理 Redis，`fs` 子命令可单独管理 FS；例如已有 Redis 时可使用 `nexus redis fs up`。

集群实现位于 `nexus.gateway.trajectory_store_backends.redis_store.cluster`。Redis 基础配置源文件为 `nexus/gateway/trajectory_store_backends/redis_store/redis_server_cluster.json`，源码和 wheel 都加载该包资源。命令行参数覆盖拓扑和部署路径；每次调用只解析一次，再传给 Ray workers。

`status` 要求超过配置 Primary 数一半的节点视图确认 `cluster_state=ok`、全部槽可用且所有 Primary 均存活；输出同时包含每个节点的角色、复制目标和槽位。完整健康检查还要求主副本数匹配配置、无异常标记，且配置了 Replica 时复制链路为 `up`。服务日志在健康状态下每 10 次检查输出一次（默认约 5 分钟），降级、不可用或状态变化会立即输出；Ray Worker 的重复 INFO 日志默认不转发。

## FS 集群启动

Redis 和 Ray 已运行时，在 HyTuner 根目录执行两条命令即可：

```bash
python3 -m pip install './third_party/nexus[redis]'
nexus redis fs up
```

`up` 从已连接的 Ray 集群发现 head，默认连接该主机 `16379` 端口的 Redis，Controller 固定部署在 head，默认端口为 `17000`。
按启动时存活的 Ray 节点数创建 DataUnit，每个节点一个（含 head）。显式指定 `--num-data-units` 时，从 head 开始轮流分配到可用节点；单节点 Ray 默认启动一个 DataUnit。
入口上传当前 Nexus 包到 Ray，并为 FS workers 安装 Redis/ZMQ extra；节点仍需具备 Nexus 基础环境及 Ray。
命令在 Controller 和全部 DataUnit 就绪后返回，服务由后台 Ray actor 持有，独立于启动命令和训练进程。
同样参数重复执行 `up` 会检查并复用现有服务，不重建数据；不同参数需要先停止现有服务。

训练使用 `backend: redis`，默认连接已连接 Ray 集群的 head 上 `17000` 端口的 Controller；自定义地址或端口时传入 `controller_endpoint` 映射。
训练 workers 仍需安装 Redis/ZMQ 依赖。停止 FS 使用：

```bash
nexus redis fs down
```

`up` 可通过 `--redis-url`、`--num-data-units`、`--controller-port`、`--startup-timeout` 覆盖默认值；
自定义端口时，`down` 使用相同的 `--controller-port`。Ray 被停止时 FS 同样停止，blob 不会自动恢复。

旧版本若将 FS 部署在非 Ray head 节点，需先用旧版本停止该 FS，再升级并在 head 重新启动；新版 `down` 按 Ray head 定位服务。

FS 默认启用本地 SSD offload，每个 DataUnit 的内存高水位为 `200G`，低水位未设置时取高水位的一半，即 `100G`。
可以按节点资源调整预算，例如每个 DU 高水位 1 GiB、低水位 512 MiB：

```bash
nexus redis fs up --ssd-offload-dir /mnt/local-ssd/nexus \
  --memory-waterline-bytes 1G --memory-safe-line-bytes 512M
```

`--ssd-offload-dir`、`--memory-waterline-bytes`、`--memory-safe-line-bytes` 适用于 `up` 和独立 `data-unit`；
`serve --config` 使用同名 JSON 字段。业务的 `trajectory_store.redis` 配置和 blob URL 保持原样。
`--request-workers`、`--ssd-workers`、`--copy-workers` 分别配置普通请求、SSD 准备、SSD/Ceph 文件复制的
独立线程池，三者默认均为 `8`，适用于 `up` 和独立 `data-unit`，JSON 使用下划线字段名。
内存准入由单独一个 worker 串行协调，不再叠加准入锁；控制请求和心跳使用独立通道。
内存 PUT、SSD 准备和复制返回 Future；冷 GET 在普通 worker 中直接读 SSD，释放 URL/metadata 锁后
提交回填并等待完成，整个 GET 结束前不腾出该 worker。因此 `request_workers` 也限制同时执行的冷 GET，
不另设冷读计数或等待队列。普通池满时热 GET、DELETE 也排队，stats/heartbeat 不进入普通请求池。
SSD 和复制并发分别在整个 DU 内限制；准备、复制及尚未执行的 GET 不预读 blob，普通 PUT 排队仍可能
持有已经接收的 payload。并发数限制请求数量，不构成整个 DU 的 RSS 或字节上限。
默认值是可调整的实现起点；增大并发会增加临时文件、在途引用及磁盘排队，不代表吞吐会按比例增长。
命令行的两个容量参数接受整数 bytes，或整数加 `K/M/G/T` 单字母后缀，忽略大小写，均按 1024 进制换算；
例如 `200G` 等于 `214748364800` bytes。负数、小数和其他后缀不支持。压测脚本沿用相同规则；
Python API 和 `serve --config` JSON 仍使用整数 bytes。
目录和两个水位的默认值只由 DataUnit 处理。命令行未指定或 JSON 省略的字段直接透传为未设置，
launcher 和部署入口不补默认项；显式提供的值原样传给 DataUnit 校验与归一。

需要前台托管或指定完整 Ray 拓扑时，继续使用 `serve --config`。所有参与节点安装同一版 Nexus；
`redis` extra 包含 Redis 和 ZMQ；`redis-fs` 保留为兼容 extra，Ray 单独安装：

```bash
uv sync --extra redis
uv pip install 'ray>=2.9'
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

先启动 Redis 和 Ray。复制 [redis_fs_cluster.json](../examples/configs/redis_fs_cluster.json) 并将 `redis_url` 改成所有节点都能访问的 Redis URL。普通 Redis、Sentinel 和 Redis Cluster URL 均沿用现有客户端规则；样例中的 `127.0.0.1` 仅适合同机 Redis。

| 配置项 | 默认值与含义 |
| --- | --- |
| `redis_url` | 必填，已运行的 Redis 连接 |
| `num_data_units` | `2`，1–256 个 DataUnit |
| `data_unit_node_ids` | `null`，或每个 DataUnit 对应的 Ray NodeID；长度需与数量一致 |
| `controller_node_id` | `null`，或 Controller 的 Ray NodeID |
| `controller_port` | `0`，自动分配；`up` 默认使用 `17000` |
| `startup_timeout` | `60` 秒，Ray 启动阶段等待上限 |
| `request_workers` | `8`，普通请求 worker 数，正整数 |
| `ssd_workers` | `8`，LRU 预写和导出副本准备的并发上限，正整数 |
| `copy_workers` | `8`，SSD 与共享 checkpoint 目录间逐文件复制的并发上限，正整数 |
| `ssd_offload_dir` | DataUnit 默认 `.hykvstore`，所在节点的 offload 根目录；相对路径基于 worker 启动工作目录 |
| `memory_waterline_bytes` | DataUnit 默认 `214748364800`（`200G`），必须为正整数 bytes，不接受 `null`；按每个 DU 的常驻 blob 字节计量 |
| `memory_safe_line_bytes` | 未指定或为 `null` 时，由 DataUnit 根据高水位计算 `H // 2`；显式设置整数时必须满足 `0 <= S <= H` |

每个 DU 在根目录内创建自己的唯一子目录，避免同节点多 DU 或不同运行互相覆盖文件。目录由 worker 创建；
要使用 SSD，必须在每个参与节点把根目录指向真实 SSD 挂载点。正常停止清理本次 DU 的文件；
进程被强制终止时可能留下孤儿目录，需要在确认旧运行已停止后由运维清理，启动不会扫描或恢复旧文件。

水位是每个 DU 的预算。同节点有多个 DU 时，要把各 DU 预算相加，并为 Redis、业务、Python 元数据、
排队请求、传输副本和文件页缓存留出空间。该配置限制常驻 blob 字节，不是进程 RSS 或整机内存的硬上限；
Ray 的 CPU 申请也不限制磁盘占用。SSD 容量不足和读写错误会直接报告给调用者，不回退到不受水位约束的内存写入。

新值字节数 `n` 大于低水位 `S` 时直接写 SSD；只有 `n <= S` 的值进入内存。
若已有常驻字节 `R` 加上新值达到高水位 `H`，则按 LRU 迁移内存对象，直到 `R <= S - n`，
再写入新值，使本次新增后的常驻字节不高于 `S`。SSD GET 只缓存 `n <= S` 的值，并保留 SSD 原文件；
缓存命中更新 LRU，后续淘汰该缓存只释放内存。`n > S` 的值每次从 SSD 读取，不缓存。
当 `S = H` 时，内存写入满足 `R + n = H` 已经符合目标，无需迁移；超过 `H` 才会实际淘汰。
读取缓存所需的迁移失败时 GET 报错，源 SSD 文件保留。

SSD 准备、LRU 候选落盘和 checkpoint 复制使用独立线程池；冷 GET 由普通请求 worker 直接读盘并等待回填。
PUT/GET 在相关迁移完成后返回；同 UUID 操作、内存准入及资源饱和时仍可能排队，stats 不等待文件 I/O。
超时可能发生在服务端操作已经完成后；现有客户端不会自动重放未知结果的 PUT/DELETE。
`stats` 保留 `objects`、`used_bytes` 表示逻辑总量，并增加 `resident_bytes`、`ssd_bytes`、
`resident_objects`、`ssd_objects`、`cached_bytes`、`cached_objects`；cached 表示同时拥有内存和 SSD 副本，
因此 `used_bytes = resident_bytes + ssd_bytes - cached_bytes`，对象计数遵循同一公式。

每个服务 actor 申请 1 个 CPU、0 个 GPU。未指定 NodeID 时由 Ray 调度；需要跨机器部署时显式指定不同主机的 NodeID。FS 数据通过 TCP 直连，Controller 和 DataUnit 自动绑定所在主机的可用端口，参与节点需允许这些端口互通。

```bash
uv run --no-sync nexus redis fs serve --config docs/examples/configs/redis_fs_cluster.json
# 太极 Start Command，指定自己的配置路径
nexus redis fs serve --config /path/to/redis_fs_cluster.json
```

launcher 只连接 `ray.init(address='auto')`。启动成功会输出一行 `redis_fs_endpoints` JSON，包含 `redis_url`、`controller_endpoint`、`topology` 和 DataUnit 地址。自定义 Controller 地址或端口时，将输出的 `controller_endpoint` 映射填入业务配置的 `trajectory_store.redis`；完整配置见 [redis_fs.yaml](../examples/configs/redis_fs.yaml)。

FS `serve` 在 SIGINT/SIGTERM 或运行异常退出时停止自己创建的服务，Redis 保持运行。先停止使用这些 blob 的业务，再停止 FS：
DataUnit 使用 DRAM 与 SSD 分层存储，SSD 是运行期容量扩展。索引仍在内存中，退出后没有重启恢复；
不提供持久化、副本、自动恢复或孤儿垃圾回收。使用新的业务 `key_prefix`，读写客户端版本保持一致。

## 单独启动 Controller / DataUnit

固定端口或不使用 Ray 时，准备所有进程共用的 topology JSON，内容包含 `cluster_id`、`data_units` 和 256 个 `slots`。可通过 `Topology.create([...]).dump()` 生成，或使用 `ClusterInfo.dump()` 中的 `topology` 字段。`Endpoint`、`DataUnitRegistration`、`Topology` 位于 `nexus.gateway.trajectory_store_backends.redis_store.file_storage.types`。例如：

```python
import json
from pathlib import Path
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import DataUnitRegistration, Endpoint, Topology

topology = Topology.create([
    DataUnitRegistration('unit-0', Endpoint('127.0.0.1', 17001)),
    DataUnitRegistration('unit-1', Endpoint('127.0.0.1', 17002)),
])
Path('topology.json').write_text(json.dumps(topology.dump()))
```

各进程分别运行；先启动 Controller，再启动 DataUnit：

```bash
uv run --no-sync nexus redis fs controller --topology topology.json \
  --redis-url redis://127.0.0.1:16379/0 --controller-host 127.0.0.1 --controller-port 17000
uv run --no-sync nexus redis fs data-unit --topology topology.json --unit-id unit-0 \
  --controller-host 127.0.0.1 --controller-port 17000
uv run --no-sync nexus redis fs data-unit --topology topology.json --unit-id unit-1 \
  --controller-host 127.0.0.1 --controller-port 17000
```

开发时可运行 [redis_fs.py](../examples/redis_fs.py)，它使用已运行的 Redis，创建 LocalCluster，并演示 FS blob 和轨迹 Store 的完整生命周期：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python docs/examples/redis_fs.py \
  --redis-url redis://127.0.0.1:16379/0 --large-mib 1
```

## FS 的两节点 CPU 验证

使用专用测试 Redis，URL 必须能被两台 Ray 节点访问。在恰好两个 CPU 节点的已有 Ray 集群中，先小规模检查，再测试 `128 × 65 MiB`；后者仅 blob 就约占 8.1 GiB，还需要客户端进程和传输副本的内存空间。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync python tests/stress/redis/check_redis_fs.py --redis-url "$TEST_REDIS_URL" --clients 4 --mib 1
uv run --no-sync python tests/stress/redis/check_redis_fs.py --redis-url "$TEST_REDIS_URL" --clients 128 --mib 65
# 同样的两节点测试，限制每个 DU 内存以确保经过 SSD（节点上须存在真实 SSD 挂载点）。
uv run --no-sync python tests/stress/redis/check_redis_fs.py --redis-url "$TEST_REDIS_URL" --clients 128 --mib 65 \
  --ssd-offload-dir /mnt/local-ssd/nexus --memory-waterline-bytes 512M --memory-safe-line-bytes 256M
```

脚本将客户端交替放到两节点，跨节点读取并校验数据、检查 blob 计数和 release 后的清理。
所有测试均验证常驻 blob 字节不高于高水位、内存与 SSD 计数守恒，以及 release 后两种存储计数归零。
默认负载低于 `200G` 高水位，允许 SSD 对象数为零；上面的 `512M`/`256M` 配置确保工作集超过内存容量，
脚本在工作集或单值大小保证会触发 offload 时要求出现 SSD 对象。
仅清理本次创建的 actors 和键，不停止或清空 Redis。缺少两节点或 SSD 环境时，对应测试应记为未验证。

## 故障测试

测试脚本位于 `tests/e2e/redis/`，通过以下命令在专用测试集群手动运行，常规 pytest 不会自动收集。故障切换场景需要副本，先启动完整 Redis/FS 服务，再对测试命令传入相同的 3 主、3 基础副本、1 额外副本参数：

```bash
nexus redis up --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
bash "$TAIJI_BASIC_CODE_PATH/tests/e2e/redis/run_redis_server_test.sh" \
  --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
```

测试覆盖：

- 3M+3R 拓扑和跨 Primary 数据分布；
- 单 Replica 故障时继续读写；
- 单 Primary 故障后对应 Replica 晋升，旧 KV 仍可读取；
- 额外热备 Replica 自动迁移到失去副本保护的新 Primary；
- 单分片负载下的故障恢复时间和延迟；
- async 与 `WAIT 1` 复制安全；
- 故障节点恢复后重新形成完整 3M+3R。

单项测试：

```bash
python3 -m tests.e2e.redis.run_cluster_tests test topology --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
python3 -m tests.e2e.redis.run_cluster_tests test replica --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
python3 -m tests.e2e.redis.run_cluster_tests test primary --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
python3 -m tests.e2e.redis.run_cluster_tests test migration --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
python3 -m tests.e2e.redis.run_cluster_tests test load --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
python3 -m tests.e2e.redis.run_cluster_tests test async --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
python3 -m tests.e2e.redis.run_cluster_tests test wait1 --primary-count 3 --replicas-per-primary 1 --extra-replicas 1
```

## 来源和切换

Redis Cluster 原先迁入自 HyKVStore `46b7d5aacae6818300a30fe1cfa9c64c5e156e34`；部署实现、默认配置和太极初始化脚本现在都位于 Nexus 包内。端口、AOF、健康检查与恢复机制沿用原实现；默认 Primary 数为 `max(可用主机数, 3)`，默认零副本，原先的 3 主、3 基础副本、1 额外副本需要显式配置。
FS 来自 `hybrid_client_dev@5853bb859e84842bee9271fbe030a1fff1051e92` 相对该提交的增量，并复用 Nexus 现有 Redis 客户端、记录类型和 CAS 实现。
持久目录默认值仍为 `/tmp/hykv-redis-cluster`，避免更名意外重建已有集群；生产环境继续通过 `--redis-dir` 指向原持久卷。
旧服务配置环境变量改为命令行参数，端点字段从 `hykv_redis_url` 改为 `redis_url`。
读写端升级及新 namespace/key prefix 要求见 [Redis trajectory store](redis-trajectory-store.md)。

上述故障测试会停止或杀死配置中的 Redis 节点，仅在专用测试集群执行。运行脚本已经显式加入仓库 `PYTHONPATH`；直接运行 Python 命令时也要设置它。
