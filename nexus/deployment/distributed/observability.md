# 平台指标与状态概览

平台状态分为三个独立读取路径：在线 Manager 和资源账本实时查询；复杂状态读取后台快照；历史曲线读取智研。
System 和环境详情的实时分区每 2 秒刷新，后台快照每 30 秒采集和刷新。
Metrics 的最近 1 小时/24 小时每分钟刷新，最近 7 天/30 天每 5 分钟刷新；业务统计由用户手动生成。
单任务状态每 3 秒读取 `summary=1`，Batch 父任务和 Batch 汇总每 10 秒读取；页面隐藏时暂停，同一分区不并发轮询。

## 配置

所有服务使用独立的 OTel MeterProvider；不会替换全局 Metrics 或 Trace Provider。
以下普通配置可放在全局启动配置中，修改数据库启动配置后需重启对应服务。写入配置前必须合并该范围现有完整对象。

```json
{
  "metrics": {
    "enabled": true,
    "endpoint": "http://opentelemetry.access.inner.monitor.zhiyan.woa.com:4318/v1/metrics",
    "app_mark": "21625_199057_nexus-online-service",
    "environment": "dev",
    "meter_name": "nexus_cloud",
    "export_interval_seconds": 60,
    "export_timeout_seconds": 5
  },
  "status_sampling": {
    "enabled": true,
    "interval_seconds": 30,
    "query_timeout_seconds": 1,
    "group_timeout_seconds": 3,
    "round_timeout_seconds": 10,
    "stale_after_seconds": 60,
    "live_cache_seconds": 1,
    "manager_summary_timeout_seconds": 3,
    "manager_summary_cache_seconds": 10,
    "release_grace_seconds": 300
  }
}
```

`metrics.enabled=false` 关闭指标导出；`status_sampling.enabled=false` 单独关闭后台快照生产。
实时概览不依赖采集者租约，也不依赖智研。监控初始化或导出失败不改变业务操作的结果。

Control Gateway 可配置独立的历史查询：

```json
{
  "metrics_history": {
    "enabled": true,
    "project_name": "Nexus-Agent",
    "project_id": 21625,
    "app_mark": "21625_199057_nexus-online-service",
    "environment": "dev",
    "group": "nexus_cloud",
    "max_segment_hours": 168,
    "upstream_requests_per_minute": 750,
    "retention_days": null,
    "histogram_aggregation_supported": true
  }
}
```

查询凭据放在 Control Gateway 的 Secret 配置 `metrics_query` 中，结构为 `{"username":"<staffname>","token":"<query-token>"}`。
它只用于智研 OpenAPI，不进入 OTLP 请求、指标维度或浏览器响应。普通配置不可保存这个字段。
当前智研图表接口的账号额度为 1000 次/分钟。每个 Control 实例默认最多发起 750 次/分钟（包括重试），
共用账号的多个实例和其他读取工具需共同分配额度，并验证对应的冷查询预算。
30 天窗口可能跨过 6 个周分段；3 个并行面板最多需要 54 次源请求，默认节奏为其保留 5 秒预算内的响应余量。
源 HTTP 请求预算为 1.5 秒，连接错误、超时、HTTP 429/5xx 和上游限流码 50001 最多重试一次。
短范围整份查询预算为 4 秒，长范围为 5 秒；失败分段保留具体原因，页面对包含失败分段的 HTTP 200 响应也执行快速重试。
快速重试后仍失败的面板先保留错误或已有样本；其他面板完成后等待 5 秒，仅对这些面板再恢复一次，每张卡每轮最多请求 3 次。
延迟恢复使用同一并发及整轮预算，切换筛选、隐藏或离开页面会取消等待，不为持续失败无限重试。
页面按最多 3 个面板并行逐步加载，30 天视图的整轮预算为 3 分钟，以覆盖全部展开的指标；每个面板的等待上限保持 6.5 秒。
`retention_days` 只填写已经核验的当前数据源保留期；留空表示尚未核验，不根据空曲线猜测保留天数。
若目标引擎不支持跨分钟的 Histogram 分布合并，将 `histogram_aggregation_supported` 设为 `false`，长区间 P95 会明确显示不支持。

Control Gateway 的业务统计使用独立数据库连接池（size=1、overflow=0），可通过普通配置控制预算：

```json
{
  "business_statistics": {
    "enabled": true,
    "query_timeout_seconds": 8,
    "request_timeout_seconds": 15,
    "queue_timeout_seconds": 1,
    "cache_seconds": 300,
    "cache_entries": 16,
    "max_name_groups": 1000,
    "max_response_bytes": 2097152
  }
}
```

统计超时、结果超限和繁忙均返回明确错误，不返回假零值或截断总数。配置变更在重启 Control Gateway 后生效。
取消请求会先核对数据库连接和本次 SELECT 的随机标记，再终止该查询；数据库账号需允许查看和终止自己的查询。
单 SQL 默认 8 秒，整份结果仍限制为 15 秒；该预算用于覆盖三倍月度数据的聚合开销。
此连接池隔离不等于数据库 CPU/IO 隔离；发布前仍需验证月度数据量和多副本下的总负载。

## 指标定义

`nexus.distributed.observability.schema.metric_definitions()` 生成目标指标组所需的定义。新增或修改指标时，以此函数的输出更新
目标环境的智研配置，并保留之前的配置以便回滚。

Histogram 的上报名与查询视图名不同。例如 SDK 上报 `gateway_duration`，查询视图为
`gateway_duration_distribution`、`gateway_duration_p95`、`gateway_duration_sum`。分桶最多 10 个边界，接收端配置与 SDK 一致。
Counter 与 Histogram 上报 DELTA；状态值使用 Gauge。P95 由智研合并分布后计算，页面不会平均不同分钟的 P95。
无样本保持空值；错误率要求分子、分母有相同的采样覆盖范围。
Metric explorer 可查询完整指标目录，包括连接获取、事件循环、采集新鲜度和资源量；可按数据源和操作类型筛选。
全局快照指标在未选择环境时固定查询 `environment=all`，不会混合全局与各环境的重复数值。

阶段尝试在真实操作结束时记录；数据库状态转换的计数在根事务提交成功后记录，回滚不产生成功事件。
依赖构建重试按每次尝试计时；Manager 启动从开始创建资源计时，不把前面的依赖等待计入启动耗时。
平台终态失败仅使用已有错误类型明确归因，任务错误、取消和未知分别保留；不采集模型、工具或 Agent 成功率。

运行中的操作在进程内维护有界状态，用于当前数量、持续时间、超预算数量和缺少预算的数量。无可靠预算时不导出虚假的零超时值。
Trace/结果/Artifact 在真实写入、读取或关闭边界采集；Trace 数据库的连接获取和 SQL 执行分别计时。

## 状态契约

- `/v1/system/summary`、`/v1/environments/{id}/stats`：轻量实时统计，返回 `queried_at`、`cache_age_seconds`、心跳时间与健康窗口。
  资源账本按环境统计，全局仅合并环境账本；池级可用槽位不改称池级 CPU/内存占用。
- `/v1/system/status`：只读取最新快照表，逐组返回 `sampled_at`、最近尝试、质量状态和错误码。
  失败保留最后成功样本，不刷新其时间；初始化前返回不可用。
- `/v1/managers/summary`：使用独立缓存，默认查询预算 3 秒、缓存 10 秒；每个 Control Gateway 最多并行执行 2 个汇总查询。
  可通过 `status_sampling.manager_summary_timeout_seconds`（最多 10 秒）和 `manager_summary_cache_seconds`（最多 60 秒）调整，
  在 `control-gateway` 角色配置中保留完整 `status_sampling` 对象后调整，修改后重启 Control Gateway；
  不将新字段写入仍有旧版本组件使用的全局配置。轻量状态查询仍保留 450 ms 预算和最多 1 秒缓存。
  超时或查询异常记录范围、耗时与失败原因；已有样本时返回 `data_status=stale`，页面显示旧样本提示，首次加载失败返回 503。
- `/v1/system/metrics`：管理员按固定分组查询最近 1/24/168/720 小时；分别为 1 分钟、1 分钟、30 分钟、1 小时聚合。
  长区间使用完整桶，展示实际截止时间；源端仍按 60 秒导出。查询、并发和缓存均有独立上限，不阻塞实时分区。
- `/v1/system/business-statistics`：管理员按带时区的 `start_at`、`end_at` 查询最多 31 天的业务任务统计。
  支持 `basis=finished|created|started`、`environment_id`、`task_kind=all|single|live|batch` 和完整 `task_name`。
- `/v1/executions/{id}?summary=1` 和列表的 `summary=1`：数据库只选白名单标量字段，详情、配置、日志与结果按需读取。

快照由 Reconciler 的 `observability-snapshot` 独立租约生产。专用连接池为 1，租约 TTL 60 秒、续租 10 秒、发现 5 秒。
发布时验证租约并短事务提交；失去所有权后停止返回全局 Gauge。全局 Gauge 使用稳定逻辑实例标记，进程诊断保留真实实例维度。
已删除环境的快照由有效采集者清理。

新增的业务时间字段只保存实际阶段起点或有效截止时间。历史记录没有可靠起点时标记未知，不用普通 `updated_at` 推断。
回收超时表示控制面在 `release_grace_seconds` 宽限期后尚未确认释放，不等同于独立证明物理资源仍存在。
`failed + next_retry_at` 的 Build 表示等待重试，`ready + next_retry_at` 表示等待清理。
CPU/内存账本是申请或预留量，实际利用率、OOM、磁盘和重启继续由现有容器、主机、数据库基础监控提供。

## Metrics 的长区间曲线

时间范围默认最近 1 小时，也可选择最近 24 小时、7 天、30 天；这里的 30 天是滚动窗口，不是自然月。
7 天与30天的截止时间分别对齐到最近完整半小时、整小时，避免旧数据的低分辨率造成边界误算。
接口返回 `granularity_seconds`、实际窗口、原请求截止时间、分段失败范围和样本覆盖信息，图表按这些时间绘制。
只查询核心图表和已展开的详情；折叠组、隐藏页面、未激活的 Observability Tab 不查询。

后台按已验证的单次跨度分段，默认每段最多 7 天。端点返回的重叠边界桶只计一次；历史稳定段可复用缓存，最近段重新查询。
取消后立即重查会创建新的有效请求，不复用正在取消的旧任务。分段失败保留缺失范围；全部失败时可保留上次完整结果并标记过期。
Counter 以增量合并，错误率按分子分母合并，Gauge 遵循指标定义，P95 由数据源合并分布后计算。
没有上报或已经不可读取的历史保持空值，不能因为范围选择了一个月就补出一个月的数据。

## 业务统计与历史回填

System 的“业务统计” Tab 不自动查询。修改筛选后点击“生成统计”，结果中保留原条件、生成时间和完整性说明。
页面名称为 `Business statistics`，筛选、按钮、图表、完整性提示和错误信息均使用英文。
小时/天趋势切换、表格排序分页、CSV/JSON 导出都使用当前结果；切换 Tab、隐藏页面或取消操作会停止未完成查询。

- 独立 Single/Live 各按一个 Execution 计数，Live 不按轮次计数；Batch 按 item 计数，父级编排和自动重试不重复计入。
- 默认按最终结束时间统计完成量；提交量使用记录创建时间，Batch 对应 item 物化入库时间；执行量使用首次开始时间。
- 时间范围为北京时间展示的左闭右开区间；提交量和执行量的状态取查询时状态，不重建过去时刻的状态。
- 失败包括 failed/lost/expired，取消包括 killed/cancelled；执行成功率只以成功和失败为分母，零分母显示“无已判定任务”。
- `task_name` 按完整注册名和大小写区分，保留 `_task`、数字等后缀。名称缺失单列并显示覆盖率；不读取 Agent 输出推断业务成功。

统计复用 `executions`、`distributed_batch_items` 和 `distributed_batch_item_attempts`，没有新增统计事实表或汇总表。
`0033_business_statistics` 增加 Execution/Batch item 的名称、Batch item 的首次开始/最终结束时间，以及时间范围索引。
新任务在业务事务内维护这些字段。`0034_batch_finish_backfill` 为历史终态 Batch item 自动回填结束时间：
已有值保持不变，缺失时优先取状态一致的当前子 Execution 的 `finished_at`，无法恢复时使用 item 原有 `updated_at`。
后者标为 `finished_at_estimated=true`；运行中及等待重试的 item 不回填，也不改变原来的 `updated_at`、状态或版本。
迁移每 1000 条主键范围提交一次，中断后重跑会跳过已填充值，全部完成后才更新 schema 版本。
这是保留来源标记的数据迁移，不支持直接 downgrade 丢弃估算标记；回滚需使用配套数据库恢复方案。
统计以 `coverage.estimated_time_records` 标注当前筛选结果中的估算条数，页面、CSV 与 JSON 保留该提示，不宣称时间精确。
下面的回填命令还可恢复名称、首次开始时间，并在真实来源重新可用时替换估算结束时间、清除估算标记。

以下命令沿用已有安全方式提供数据库连接配置，不在命令行中填写密码。先预览，再按相同范围应用：

```bash
nexus distributed statistics backfill --environment-id example-environment --batch-size 100 --max-pages 10
nexus distributed statistics backfill --environment-id example-environment --batch-size 100 --max-pages 10 --apply
```

每页输出计数和 `cursor`；将最后成功页的游标通过 `--cursor` 传入即可继续。默认每次只执行一页，输入读取每页最多 16 MiB，
单页有 30 秒预算；每页提交后才输出可续跑游标。重复执行不会覆盖已有有效字段，状态变化时通过条件更新避免覆盖新写入。

优先读取现有任务配置与 attempt 时间。需要读取仍存在的 Task Store 时，添加 `--with-task-store`，使用存储的 Reconciler 配置，
以只读模式连接，不自动创建 Task Store 表。也可用 `--task-list PARENT_EXECUTION_ID=LOCAL_PATH` 提供保留的任务列表，
回填会验证 item 身份和 SHA256 后提取注册名，不执行任务或生成器。

名称和首次开始时间的输入已删除、来源不可用时保持缺失；手动回填不会凭空生成真实结束时间。
没有时间的记录无法判断是否属于所选区间，覆盖提示会说明其环境/类型范围；
它们可能影响计数时，报表明确标为不完整。`updated_at` 只作为显式标记的历史估算，不使用父 Batch 完成时间代替子任务时间。
缺失时间核验与正常时间范围聚合分别选择读取路径：先核验有界 Batch 父记录集合，NULL 密集时避免二级索引逐行回表；
成功状态可直接证明任务曾开始，只有其他状态才需要检查历史 attempt。所有计数仍来自同一个只读事务视图。

## 部署与回滚

先运行 `nexus distributed db upgrade`，确认版本为 `0034_batch_finish_backfill`，再发布依赖新字段的服务。
本次增量迁移增加既有表的统计字段和索引，可重复执行；此前的 `0032_observability_snapshots` 保存监控快照。
已部署的旧扩展版本通过主线增量迁移衔接，不删除历史扩展表。

发布应记录准确源码 SHA、不可变镜像 digest、配置版本、旧配置完整内容以及验证时间。回滚时恢复旧镜像和旧启动配置；
旧版本允许读取较新 schema 时，新增兼容字段和快照表可以保留；要求精确 schema 版本的旧服务，必须配套恢复迁移版本。
需要撤销 DDL 时，先确认已停止所有依赖这些字段的服务，再执行迁移的 downgrade。

验收包括：已知计数跨周期回读、耗时分布、Gauge 回零与所有权切换、无快照和超时场景、授权隔离、实际 SQL 计划与并发性能、
真实任务结果及资源回收。OTLP HTTP 成功或 SDK 初始化成功本身不代表验收完成。
业务统计额外核验逻辑任务去重、三种时间口径、小时/日峰值、名称完整性、权限和缓存隔离、取消后 SQL 停止与连接恢复。
月度 Metrics 验证完整时间轴、分段边界、Counter/错误率/Gauge 聚合，以及已知跨分钟样本的 Histogram P95；回读数据源真实结果。
