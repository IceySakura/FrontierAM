# SWE Series

<script async id="asciicast-569727" src="https://asciinema.org/a/569727.js"></script>

## SuperSWE 平台退出状态

平台运行使用 [AgentBench 状态规范](https://doc.weixin.qq.com/doc/w3_AQgA3wbZAIMCNNeIZlvP7SGiV0J0G)、
[沙盒错误码契约](https://iwiki.woa.com/p/4041018198) 和
[沙盒归因最佳实践](https://iwiki.woa.com/p/4043497428)。基线已包含
[Nexus !1336](https://git.woa.com/nexus-agent/nexus/-/merge_requests/1336) 的结构化错误透传。

HY 一站式平台评估任务使用以下退出状态配置：

```yaml
imports:
  - nexus.extensions.tasks.swe.superswe
  - nexus.extensions.tasks.swe.platform
  - nexus.extensions.agents.swe_agent.agent
task_name: superswe_task
agent_name: swe_agent
agent_config:
  platform_exit_status: true
runner:
  hooks:
    superswe_platform:
      name: superswe_platform
```

| `agent_output.exit_code` | `agent_output.exit_status` |
| --- | --- |
| `0` | `finished`, `agent_timeout`, `max_iterations`, `context_length_exceeded`, `output_length_exceeded`, `runtime_error_by_agent` |
| `-1` | `inference_error`, `runtime_error_by_env`, `environment_error`, `unknown_error` |

沙盒归因只读取 `RuntimeUnhealthyError` / `SandboxApiError` 及其异常链携带的结构化字段。
Agent 执行期间的 `code=260422` 且 `reason` 精确等于 `Process.OOMKilled`、
`Process.NonZeroExit`、`Process.CrashLoopBackOff` 之一，才归为 `runtime_error_by_agent`。
启动失败、`Process.Killed`、`Queue.*`、未知或缺失 reason、错误码与 reason 不匹配均归环境。
模型文字、命令输出和容器退出码 `137` 单独出现时不能证明 Agent 导致了沙盒死亡。

`agent_timeout/0` 只由 `agent_config.timeout` 的整体期限触发。
模型请求超时（包括 `inference_time_timeout`）为 `inference_error/-1`。
单次工具超时默认结束当前 Agent。显式设置 `agent_config.continue_on_tool_timeout: true` 后，
只有框架自己的 `per_observation_timeout` 到期、且没有沙盒故障证据时，才将错误交还 Agent 继续执行。
此时轮次耗尽记录 `max_iterations/0`，正常结束记录 `finished/0`；单次工具超时不记为整体超时。
平台状态模式不接受累计 `observation_time_timeout`：请改用上述整体和单次期限；
普通 SWE / 训练的原生模式仍支持原有累计工具预算。

依赖导入、Provider/client 构造、输入/处理器准备，以及异步 Context/hook 准备失败，
即使发生在平台 hook 启动前，也保留标准失败输出。模型模式记录 `environment_error/-1`，
带结构化沙盒证据的启动失败记录 `runtime_error_by_env/-1`，Task 为 `agent_error`；
显式补丁/oracle 的纯评测准备失败记录 `judge_error/-1`，不虚构 Agent 执行。

`judge_output` 单独记录 `finished/0` 或 `judge_error/-1` 及实际错误。
Task 优先报告 Agent 的非零退出码；Agent 为零时再根据 Judge 退出码决定 `success` 或 `judge_error`。
`task_output.score` 保留实际得分，`task_output.error` 复制对应失败阶段的错误。
合法的 Agent 零码约束终止也保留实际错误诊断；`success` 不要求 `error` 为空。
失败结果同时设置 `RunOutput.success=false`，使原生 batch 的 `retry_failed` 也能识别。
正常判分时，所有必要组件仍使用同一份补丁，按各组件结果的 AND 得到二元分数。

对于确认由 Agent 导致、发生在 Judge 启动前的沙盒死亡，返回结果和直接抛异常两条路径
均按规范判 0 分，不再尝试从死亡沙盒提取补丁，也不重测该次 Agent。
该裁决的 `judge_output` 为 `finished/0`，额外记录
`scoring_method=agent_runtime_failure`、`tests_executed=false`；
`task_output.status=success`，`resolved=false`，`score=0`。
原生 `completed=false`、未执行组件和原始错误仍保留，表示测试没有执行完，不能解读成测试通过。
Judge 已经开始后的异常仍是 `judge_error`，不能借用这一裁决覆盖。

平台有效性规则使用 `nexus_run_valid_rule: task_output.status == "success"`。
平台的 `rule_engine` 将未加引号的 `success` 当作变量，字符串引号不能省略。
不要额外要求原生 `completed=true`，否则会把上述合法的零分裁决重新当成基础设施失败。
`platform_exit_status` 默认关闭，普通 SWE 和训练配置仍可使用原生退出状态。

`run single` JSON、batch DB 的可空 `judge_output` 列以及 JSONL / full 导出均保存 Judge 状态。
打开原有 v2 DB 时只补充该可空列，历史记录的 Judge 状态保持 NULL，不反推历史运行结果。

## SWE 清理与终端恢复

参考补丁的新增行也可能出现在正常依赖中。清理时，内容指纹只用于查找候选文件；
还需确认文件具有独立 unified diff 的文件头和 hunk，才按补丁副本删除。
额外文件名清理由 `task_config.artifact_cleanup_globs` 显式指定，默认不按通用文件名删除。
Nexus 运行环境和项目构建缓存受保护；Git 截断保留基线祖先上的标签和引用，删除后续提交引用，
并处理嵌套标签和不可达答案对象。
没有明确产物路径、也不具备补丁结构的源码文件会保留；题目镜像仍需独立检查答案残留。

只有提交新命令时才恢复死亡的终端会话。查看屏幕、查询命令状态保留故障现场，
向死亡会话发送按键会报错。同一会话的创建、恢复和销毁串行执行；恢复时先停止旧 PTY、
清空旧 capture 文件，再等待新 Shell 就绪并预热。预热失败按终端基础设施故障上报。

命令 ID 在会话内递增，重建后旧 ID 失效，不会查到新 Shell 的同号命令。
重建不恢复原 Shell 的工作目录、变量或后台任务。`command_id=0` 始终表示空命令。

## 调用超时与数据库连接

`NEXUS_CALL_DEADLINE` 和 `NEXUS_RETRY_TOTAL_TIMEOUT` 默认均为 `0`，不附加 Nexus 时限；
原有模型接口和整题时限仍有效。设为正值后，前者限制单次调用，后者在重试判断点检查累计耗时。
Responses 的普通、native 和归档调用遵循相同设置；外部任务取消立即传播。

命令后端默认 `terminal`，需要无状态执行时显式设置 `agent_config.bash_backend: stateless`。
防作弊提示默认关闭，开启 `task_config.enable_anti_hack_prompt` 后，提示按最终命令后端和违规处置生成。
参考答案、隐藏测试和判分专用字段继续隔离，`hints_text` 默认保留；用
`task_config.agent_instance_exclude_fields` 排除数据协议不允许提供的额外字段。
`Cargo.lock`、`composer.lock` 和 Go／PHP 的 `vendor/` 默认可进入提交；额外排除规则使用
`task_config.patch_exclude_patterns`，项目自身的 `.gitignore` 仍有效。

日志默认沿用每条 10,000 字符上限。需要保留更长的判分标记记录时，设置
`NEXUS_LOG_EXEMPT_SIZE=200000`；首尾文本保留方式不变。

Hycode 的 `disable_invoke_wait_timeout` 默认为 `null`：训练关闭控制端累计等待超时，
验证、独立调用和未知场景保留该超时。显式 `true`／`false` 优先；HTTP 超时和运行时限不受此字段影响。

打开已有 `judge_output` 列的 batch DB 不获取迁移写锁。缺列时才获取写锁并复查后迁移；
连接初始化失败或被取消时关闭连接，避免遗留 aiosqlite 后台线程。
