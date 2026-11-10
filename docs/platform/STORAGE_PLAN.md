# 存储规划：本地盘与云盘

更新：2026-11-05。`/data` 已按第 2 节整理（只做了同一分区内的移动，没有删除或上传任何数据）；云盘的用法按 iWiki 上的 ITFS 文档核对（第 4 节）。

## 1. 原则

本地盘只放**会被读写的热数据**，云盘放**只为留存的冷数据**：

1. 计时路径上的东西必须在本地：正在使用的运行副本、各评测守护进程里的镜像、正在评测的用例、store 队列、venv。云盘是 FUSE 网络盘，延迟和抖动不可控，绝不出现在评测路径上。
2. 能重新生成的东西直接删，不上云：解压目录、构建缓存、可从原包重建的镜像缓存。
3. 生成代价高或不可再生的东西上云：原始题目包、退役的封装副本（追溯用）、跑完的 store、参考解（要算上百 CPU 小时）。
4. 私有评测材料（held-out 用例、参考解、私有清单、出题方包、模型提交）上传前先加密，密钥只放在 `/root/.frontieram/`；团队共享盘一律不用。
5. 每次迁出都附 `SHA256SUMS`；上传后先确认传输完成、读回校验，再删本地副本。大批量传输只在没有计时评测时进行。

## 2. 整理后的结构

`/data/README.md` 是盘面地图；`/data/archive/MOVES.tsv` 记录了每一次移动（原路径、新路径、大小）。

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `workspace/FrontierAM/` | 仓库（代码、文档）和正在使用的运行数据：`.runtime`（默认）、`.runtime-suite`、`.runtime-recal`、`.runtime-ops`、`.runtime-exp2`，`var/stores/store/`、`var/stores/store-recal/` | 约 14 GB |
| `workspace/agent-traffic-runtime/` | 开发 harness 依赖 | 0.1 GB |
| `archive/` | 12 份退役运行副本、3 个已结束的测量/实验 store、锥优化原始参考包、最初的仓库快照 | 约 11.6 GB |
| `trash/20261105/` | 调研时的解压目录、旧布局（grade-24/26/28 和最早的单守护进程）的评测守护进程目录 | 约 11 GB |
| `docker/lib/` | 宿主机 docker | 22 GB |
| `frontieram-grade/grade-{06,14,22,30}/` | 4 个评测守护进程 | 约 17 GB |

仓库根目录从 38 项减为 26 项。`scripts/analyze/conic-per-case-export.py` 已改为从 `archive/` 读取两个锥优化重测 store，已验证可读。

## 3. 哪些放云盘

| 类别 | 现在的位置 | 大小 | 处理 | 理由 |
| --- | --- | --- | --- | --- |
| 退役运行副本、已结束的 store、仓库快照 | `/data/archive/` | 11.6 GB | **上云**（打包成 tar.zst，加密） | 只用于追溯；12 份运行副本几乎相同，用 zstd 长窗口压缩后预计不到 2 GB |
| 题目原包 | `var/raw/`（0.7 GB）、`/srv/frontieram-raw/`（mcmc 19.6 GB） | 约 20 GB | **上云**（加密），本地只留封装好的副本 | 原包只在封装时读一次；以后新题直接下到云盘 |
| 交付包 | `dist/` | 20 MB | **上云**（加密） | 已交付 |
| 指数题 v2 的参考解和正式用例 | `tasks/exponential/versions/v2/dataset/cases/` | 预计 2–4 GB | **本地一份 + 云上备份** | 评测时要读，必须在本地；生成代价高，必须备份 |
| 冻结镜像 | docker | 每个 1.3–3.5 GB | **本地 + 云上 `docker save` 冷备份** | 运行必须在本地；备份防止镜像丢失后无法复现 |
| 已结束的开发记录 | `var/stores/store/dev-runs/`（2.5 GB，持续增长） | 2.5 GB 起 | 按批次**上云**，本地留索引和摘要 | 每次开发尝试都会留下工作区和日志，增长最快 |
| 已结束的评测明细 | `var/stores/store/results/`（5.7 GB） | 5.7 GB | 结论定稿后按套件**上云**，本地留汇总 JSON | 锥优化逐用例输出占大头，只在复核时用 |
| 可重建内容 | `/data/trash/`、无标签镜像、两个 ARM64 镜像 | 约 11 GB + 数 GB | **直接删** | 都能从原包或构建脚本重建 |
| 代码、正在使用的运行副本、store 队列、评测守护进程镜像、venv | 原处 | — | **留在本地** | 在评测路径上 |

预计效果：删除可重建内容约释放 13–15 GB，archive 上云释放 11.6 GB，`/data` 已用量从 76 GB 降到约 50 GB；之后开发记录和评测明细按批迁出，可以长期维持在 60 GB 以下。

执行顺序（都放在没有计时评测的时段）：

1. 删除 `/data/trash/20261105/` 和无用镜像（`docker image rm` 两个 ARM64 镜像与两个无标签镜像）。
2. 在云盘上用 1 GB 文件测一次吞吐。
3. `archive/` 打包、加密、上传、校验后删除本地副本。
4. 原包、交付包、参考解、镜像冷备份依次上云。

## 4. 云盘（来源：iWiki 4021193593、4018506502、4022528563）

本机云盘客户端的本地状态接口（`http://127.0.0.1:49252/api/v1/status?full=1`，只读）显示登录用户为 iceyghli：

| 空间 | 挂载点 | 状态 | 权限 | 用途 |
| --- | --- | --- | --- | --- |
| Anydev（AnyDev 为本人开的专属空间） | `/agentdrive/Anydev` | 在线 | 管理员 | **目标**：`/agentdrive/Anydev/FrontierAM/` |
| iceyghli（本人创建） | `/anydisk/PersonalSpace/iceyghli` | 经 `/anydisk` 可见 | 管理员 | 备选 |
| 约 20 个团队共享盘、OpenClaw 专属盘 | `/anydisk/AgentSpace/…` | — | 大多为普通成员 | **不用** |

- 两个 FUSE 守护进程（itfsproxy、spaceclient）都在 `system.slice`，绑定 CPU 0–3，传输不占评测或开发 CPU；但 CPU 0–3 与 grade-06 同在一个 L3 域。
- 客户端缓存在根分区的 `/agentdrive/.cache/itfs`，大批上传时会先占用根分区，单批上传量要低于根分区剩余空间。
- 上传是异步的：先用 `GET /api/v1/spaces/{space_id}/sync/status` 确认传输完成，再读回校验。文档推荐用 `rsync`。
- 本机查不到配额（`any df` 不支持），`df` 显示的 1 TB 不一定是实际限额；空间详情见 https://fs.it.woa.com/spaces。

## 5. 待确认

- 专属空间的实际配额和保留期，以及除本人外是否还有人（例如云研发管理员）能读取。在确认之前，私有材料一律先加密再上传。
- 第 3 节第 1–3 步已获同意（11-05 16:00）：由 `scripts/ops/data-maintenance.sh` 在新题基线全部完成、恢复锥优化评测之前的空闲窗口执行（`scripts/references/new-task-baselines.sh` 末尾调用），日志在 `var/stores/store/reruns/data-maintenance.log`。归档用 gpg AES256 加密，口令在 `/root/.frontieram/archive-passphrase`，**需要另存一份到安全的地方**，否则云上归档无法解密。

## 6. 工具

- iWiki：knot 技能 1980（iwiki-doc），`~/.cursor/skills/iwiki-doc`，CLI 为 `iwiki-cli` v0.6.6。
- 内网搜索：knot 技能 10064（isearch），`~/.cursor/skills/isearch`。
- 两者都用太湖 PAT 鉴权；PAT 只保存在本机私有文件中（`~/.nexus/auth/taihu-pat`、`~/.iwiki/credentials` 等，均为 600 权限），不进仓库，也不进入任何参赛容器。
