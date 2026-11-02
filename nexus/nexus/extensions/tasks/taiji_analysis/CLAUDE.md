# 工作区说明

本工作区集成了 Debug Agent 的全部 Skills，可以通过 Claude Code 直接调用。

## 使用方式（强制执行）

所有 skill 位于 `/data/workspace/.agent/skills/` 目录下。收到用户请求后：

1. 先查看技能列表，判断是否有适用的 skill
2. 如果恰好有一个技能明确适用：读取该 skill 的 SKILL.md，然后通过 Bash 执行对应脚本
3. 如果多个技能可能适用：选择最具体的一个，读取其 SKILL.md，然后通过 Bash 执行对应脚本
4. 如果没有明确适用的：不要加载任何 skill

约束：永远不要预先读取多个 skill；只在确定有用后读取。

## ⚡ 主动使用 Skill 原则（强制执行）

Agent 必须主动、优先使用 skill，而不是手动拼底层命令：

1. **Skill 优先**：任何操作如果有对应的 skill，必须走 skill 流程（读 SKILL.md → 按说明执行），禁止绕过 skill 直接调用底层工具
2. **包括中间步骤**：即使是在执行某个主任务过程中需要查 iwiki/gongfeng/tapd 等信息，也必须通过对应 skill 调用，不能因为"只是个小查询"就手动拼 mcporter-internal 命令
3. **MCP 调用必须走 skill**：所有 `mcporter-internal` 调用都应该先确认是否有封装好的 skill（如 iwiki、gongfeng、tapd），有则走 skill，没有才直接调用
4. **自主判断能力**：不需要用户明确说"用 xxx skill"，Agent 应该自己识别场景并主动选择合适的 skill 组合

## Debug Agent 核心技能

以下技能构成 Debug Agent 的诊断分析能力，按使用频率排序：

| 技能 | 用途 | 调用方式 |
|------|------|---------|
| taiji-log-analyzer | 太极训练实例日志分析（主力诊断工具） | `python3 /data/workspace/.agent/skills/taiji-log-analyzer/scripts/analyze_instance_log.py <instId> --task-id <taskID> --wsid <wsid> --full` |
| check-machine-status | 节点状态/异常检查（崩溃时必查） | `python3 /data/workspace/.agent/skills/check-machine-status/scripts/check_status.py --ips "<IPs>" --time "<YYYY-MM-DD HH:MM:SS>"` |
| ray-job-checker | Ray 集群任务日志/指标/AgentLoop 分析 | `python3 /data/workspace/.agent/skills/ray-job-checker/scripts/check_ray_job.py <host:port> <job_id> [--full\|--metrics\|--logs --filter agent]` |
| ray-code-explorer | 远程 Ray 节点代码定位和问题诊断 | `python3 /data/workspace/.agent/skills/ray-code-explorer/scripts/explore_ray_code.py <host:port> <job_id> [--tree\|--cat <path>\|--grep <keyword>]` |
| hytuner-nightly-ci | CI 任务列表/状态查询/指定 commit 跑 CI | `python3 /data/workspace/.agent/skills/hytuner-nightly-ci/scripts/list_ci_tasks.py [--status\|--name "<keyword>"]` |
| ci-bisect-debugger | 二分法定位引入 regression 的 commit | 流程型 skill，依赖 hytuner-nightly-ci 的 run_ci_task.py 执行实验 |
| taiji-config-checker | 太极任务平台配置校验（日志无错时用） | 通过 taiji-official-skills MCP 获取配置后分析 |
| taiji-cluster-task-checker | 集群/应用组任务巡检 | `python3 /data/workspace/.agent/skills/taiji-cluster-task-checker/scripts/check_cluster.py` |
| angelft-db-query | AngelFT 容错记录/误杀排查 | `python3 /data/workspace/.agent/skills/angelft-db-query/scripts/query_angelft.py <instance_id> [--table tickets\|metadata]` |
| swanlab-chart-viewer | 训练曲线/验证指标图表截图分析 | `python3 /data/workspace/.agent/skills/swanlab-chart-viewer/scripts/swanlab_screenshot.py --run-path "<user/project/run>" --section "<section>"` |
| taiji-official-skills | 太极平台 MCP 接口（任务详情/启停/克隆） | `python3 /data/workspace/.agent/skills/taiji-official-skills/scripts/connect_mcp.py call <tool> '<json_args>'` |
| taiji-token-provider | 内置只读 Token（免用户提供，仅限查询） | Token: `h3kcqqyYy5Uosyuyx1E5mw`，直接用于 taiji-official-skills 的查询操作 |

## 技能组合使用模式

| 场景 | 组合 |
|------|------|
| 太极任务失败分析 | taiji-log-analyzer → check-machine-status → (可选) angelft-db-query / ray-code-explorer |
| Ray 任务调试 | ray-job-checker → ray-code-explorer → (可选) swanlab-chart-viewer |
| 日志无明确错误 | taiji-log-analyzer → taiji-config-checker → taiji-official-skills |
| CI 状态查询/运行 | hytuner-nightly-ci |
| CI 退化定位 | ci-bisect-debugger（内部调用 hytuner-nightly-ci） |
| 集群资源排查 | taiji-cluster-task-checker |

## 其他通用技能

| 技能 | 用途 | 调用方式 |
|------|------|---------|
| weather | 天气查询 | `python3 /data/workspace/.agent/skills/weather/scripts/forecast.py <城市>` |
| gongfeng | 工蜂代码仓库操作 | `mcporter-internal call gongfeng.<tool> --args '{...}'` |
| iwiki | iWiki 文档查询 | `mcporter-internal call iwiki.<tool> --args '{...}'` |
| tapd | TAPD 项目管理 | `mcporter-internal call tapd.<tool> --args '{...}'` |

## MCP 工具调用

通过 `mcporter-internal` 调用内部 MCP 服务：
```bash
# 列出某个服务的所有工具
mcporter-internal list <server> --schema

# 调用工具
mcporter-internal call <server>.<tool> --args '{"key": "value"}'
```

配置文件：`/data/workspace/config/mcporter.json`

## 认证信息位置

- 太极 Token：`~/.taiji/config.json`
- MCP 服务配置：`/data/workspace/config/mcporter.json`

## 分析太极任务的完整流程

1. 用户提供太极平台 URL 或实例 ID → 从 URL 提取 `instId`、`taskID`、`wsId`
2. 运行 `taiji-log-analyzer` 获取全量日志分析
3. 如果有崩溃事件 → 运行 `check-machine-status` 检查节点状态
4. 如果日志无明确错误 → 运行 `taiji-config-checker` 检查平台配置
5. 如果怀疑 AngelFT 误杀/容错异常 → 运行 `angelft-db-query` 查看工单记录
6. 如果需要追踪代码逻辑 → 用 `gongfeng` clone 代码或 `ray-code-explorer` 远程查看
7. 综合日志分析 + 机器状态 + 配置检查给出根因诊断

## 分析 Ray Dashboard 任务的流程

1. 用户提供 Ray Dashboard URL（如 `http://host:port/#/jobs/job_id`）→ 提取 `host:port` 和 `job_id`
2. 运行 `ray-job-checker` 获取任务状态、日志、指标
3. 如果需要查看代码 → 运行 `ray-code-explorer` 定位源码
4. 如果需要查看训练曲线 → 运行 `swanlab-chart-viewer` 截取图表

## ⚠️ 输出格式要求（必须遵守）

分析太极任务时，**必须严格按照 `/data/workspace/.agent/skills/taiji-log-analyzer/SKILL.md` 中定义的报告格式输出**，不可自由发挥。报告结构如下：

1. **💡 一句话总结**（必须放最前面，直击要害，告诉用户问题是什么、该怎么改）
2. **基本信息**（一行紧凑格式：状态、实例ID、实验名、时间范围）
3. **📦 任务配置**
4. **🏷️ 实例状态**
5. **🔧 Pod 结构**
6. **💥 崩溃事件时间线**
7. **🔬 状态验证**
8. **🔍 根因分析**
9. **📋 Traceback 详情**
10. **❌ 错误统计**
11. **⚠️ Context 超限等辅助信息**
12. **🖥️ 机器状态检查**（必须执行 check-machine-status，不可跳过）

其他要求：
- 简洁不啰嗦，证据驱动，每个结论必须附带来源
- 如果没问题就直说没问题，不硬凑
- **根因要找真正的原因**：脚本输出只是线索，不能照搬。找不到就看代码（工蜂）、看机器（check-machine-status）、看更多日志（--raw），想办法
- **建议要精准可操作**：给具体改什么参数改成什么值，不要泛泛罗列一堆可能性

