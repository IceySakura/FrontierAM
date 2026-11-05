# 存储规划：本地盘与云盘

状态：草案（2026-11-05）。已按 iWiki 上的云磁盘（ITFS）文档核对过本机挂载（第 1.1 节），**还没有往云盘迁移任何东西**。

## 1. 现状

| 位置 | 容量 | 已用 | 主要内容 |
| --- | --- | --- | --- |
| `/`（vda1） | 100 GB | 27 GB | 系统；`/srv/frontieram-raw`（mcmc 题目原包，19.6 GB，下载中） |
| `/data`（vdb） | 98 GB | 73 GB（剩 20 GB） | 仓库与运行数据 25 GB，宿主机 docker 24 GB，4 个评测守护进程的镜像 23 GB，`FrontierAM-orig` 1.3 GB |
| `/anydisk`（FUSE，itfsproxy） | 显示 1 PB | — | `PersonalSpace/iceyghli`（个人空间，目前为空）、`DevSpace/Anydev`（空）、`AgentSpace`（21 个其他团队的共享盘，**不要动**） |
| `/agentdrive/Anydev`（FUSE，spaceclient） | 1 TB | 0 | 空 |

`/data` 的主要占用：

| 项 | 大小 | 说明 |
| --- | --- | --- |
| 宿主机 docker（`/data/docker`） | 24 GB | 18 个镜像：任务镜像、harness 镜像和构建中间层 |
| 评测守护进程（`/data/frontieram-grade`） | 23 GB | 每个 slot 一份同样的镜像，共 4 份 |
| `store/` | 8.2 GB | 全部提交、评测记录、日志；其中 11-04 的几个 campaign 目录已结束 |
| 8 份封装好的运行副本（`.runtime-*`） | 各约 0.77 GB，共 6.2 GB | 其中 `.runtime-packed-retired-*`、`.runtime-v51`、`.runtime-hy4` 已退役 |
| 其他 store（measure、iso、recal 等） | 1.2 GB | 锥优化重标定的测量记录 |
| `Raw_Resource/` | 0.66 GB | 出题方原包 |
| 指数题 v2 数据 | 增长中 | 正式规模的公开集和私有集用例加参考解，预计 2–4 GB |

两个 FUSE 守护进程都运行在 `system.slice`，绑定 CPU 0–3（调度核），所以往云盘读写不会占用评测或开发 CPU。CPU 0–3 和 grade-06 同在一个 L3 域，大批量传输时会争用这个域的缓存和内存带宽。

### 1.1 云盘的归属与用法（来源：iWiki「AnyDev 云磁盘 API 模式控制」4021193593、「ITFS Linux 客户端」4018506502、「AnyDev 云研发 CVM 挂载云磁盘」4022528563）

本机云盘客户端的本地状态接口（`http://127.0.0.1:49252/api/v1/status?full=1`，只读）显示登录用户为 iceyghli，可见的空间如下：

| 空间 | 挂载点 | 状态 | 权限 | 用途判断 |
| --- | --- | --- | --- | --- |
| Anydev（AnyDev 为用户创建的专属空间，标签 anydev/agentdrive） | `/agentdrive/Anydev` | 在线 | 管理员 | **首选目标**：属于本人，用于存放本项目的冷数据 |
| iceyghli（本人创建） | `/anydisk/PersonalSpace/iceyghli` | agentdrive 侧离线，经 `/anydisk` 可见 | 管理员 | 备选 |
| 约 20 个“团队共享磁盘”和 OpenClaw 专属盘 | `/anydisk/AgentSpace/…` | 离线 | 大多为普通成员 | **不用**：其他团队可见 |

文档要点：

- 客户端把本地缓存放在 `<base>/.cache/itfs`，本机 `base=/agentdrive`，**在根分区上**（当前 34 MB）。大批量上传时缓存会先占用根分区，所以单批上传量要低于根分区剩余空间（76 GB），并避开 mcmc 原包下载期间。
- 文档推荐用 `rsync` 往挂载目录同步。上传是异步的：应先用 `GET /api/v1/spaces/{space_id}/sync/status` 确认传输完成，并从云盘读回、核对 SHA256，再删除本地副本。
- 本机查不到配额（`any df` 提示“当前环境不支持查询磁盘配额”），`df` 显示的 1 TB 不一定是实际限额；可在 https://fs.it.woa.com/spaces 查看空间详情。

## 2. 原则

1. **计时路径上的东西一律留在本地盘**：正在使用的封装副本（`.runtime-suite`、`.runtime-recal`、`.runtime-ops`、`.runtime-exp2`）、各守护进程里的镜像、正在评测的数据、正在运行的 store 队列、各个 venv。FUSE 网络盘的延迟和抖动不可控，不能出现在评测路径上。
2. **私有评测材料不明文上云**：held-out 用例、参考解、私有清单、出题方包和锥优化出题方材料包，上传前先加密，密钥只放在 `/root/.frontieram/`（600 权限）。团队共享盘（`AgentSpace`）一律不用。
3. **迁出前先做清单和校验**：每个归档附 `SHA256SUMS`；从云盘取回后先校验，再删除本地副本。
4. **只在评测空闲时大批量传输**，并用 `nice`/`ionice` 降低优先级。

## 3. 分批计划

**第一批：不需要云盘，本地就能腾出空间（约 8–10 GB）**

- 删除 3 份已退役的封装副本（约 2.3 GB）；锥优化结果冻结后，再删 `.runtime-measure`、`.runtime-measure2`（约 1.5 GB）。删除前先把它们的 `runtime-lock.json` 归档到 `store/` 以便追溯。
- 宿主机 docker 中已不再使用的镜像和中间层，用 `docker image prune` 清理（先列清单）。
- `/data/workspace/FrontierAM-orig`（1.3 GB）打包，等可以上云后迁出。

**第二批：冷数据迁到专属云盘（`/agentdrive/Anydev/FrontierAM/`）**

| 内容 | 大小 | 方式 |
| --- | --- | --- |
| 已结束的 store 目录（11-04 的 campaign、overnight、interference 等） | 约 3–5 GB | 每个目录打成一个 tar 加 SHA256；含模型提交和日志，按私有材料加密 |
| 出题方原包（`Raw_Resource`、`/srv/frontieram-raw`） | 约 20 GB | 原样加密上传，本地只留已封装的副本 |
| 交付包（`dist/`） | 小 | 加密上传 |
| 冻结镜像的 `docker save` 备份 | 每个 1–3 GB | 冷备份；本地守护进程里的不删 |
| 指数题 v2 的母矩阵、用例和参考解 | 2–4 GB | 公开集可明文，私有集加密；参考解计算代价高（上百 CPU 小时），必须备份 |

**第三批：长期**

- 4 个评测守护进程各存一份相同的镜像（23 GB），可以研究让它们共享只读的镜像层，但这会改动已验证的隔离方案，等讨论后再定。
- 新题原包统一直接下载到云盘，用的时候再封装到本地。

## 4. 待确认

- 专属空间的实际配额和保留期（https://fs.it.woa.com/spaces），以及除本人外是否有其他人（例如云研发管理员）能读取；在确认之前，私有评测材料一律加密后再上传。
- 云盘吞吐量还没测；开始迁移前，先在 `/agentdrive/Anydev` 用 1 GB 测试文件测一次（评测空闲时）。

## 5. 工具

- iWiki 读取：knot 技能 1980（iwiki-doc），已安装在 `~/.cursor/skills/iwiki-doc`，CLI 为 `iwiki-cli`（v0.6.6）。
- 内网搜索：knot 技能 10064（isearch），已安装在 `~/.cursor/skills/isearch`。
- 两者都用太湖 PAT 鉴权。PAT 只保存在本机私有文件中（`~/.nexus/auth/taihu-pat`、`~/.iwiki/credentials`，均为 600 权限），不进仓库，也不进入任何参赛容器。
