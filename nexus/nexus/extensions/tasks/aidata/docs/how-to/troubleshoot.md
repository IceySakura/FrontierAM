---
title: 常见故障排查
audience: 遇到 aidata 报错或异常行为的开发者
prerequisites:
  - 已经能跑单 case（[tutorial/first-case.md](../tutorial/first-case.md) ）
related:
  - ../reference/env-variables.md
  - ../explanation/architecture.md
---

# 常见故障排查

按 " 症状 → 根因 → 修法 " 组织 。 顺序按遇到概率排 。

## 1 ：`error: no such case` / case_id 找不到

症状 ：

```
error: no such case: nexus/extensions/tasks/aidata/cases/xxx
available cases:
```

根因 ：CLI 找的是 `AIDATA_CASES_ROOT`（若未设 → `<aidata-package>/cases/` ） 下的 case_id 目录 。
上游 nexus 仓 **不打包** case 数据 —— 若你没设 `AIDATA_CASES_ROOT` ，且仓内 cases 目录不存在或
只有一小部分 ，就会报这个错 。

修 ：把 `AIDATA_CASES_ROOT` 指到你的下游 cases 树 ：

```bash
export AIDATA_CASES_ROOT=/data/workspace/Tasks
./nexus/extensions/tasks/aidata/aidata.py list         # 现在能看到 case 列表了
```

若你的 cases 树同时含 `.env` ，CLI 会优先加载 `<cases-root>/../.env` —— 见
[reference/env-variables.md](../reference/env-variables.md) 。

## 2 ：Gongfeng 401 / 403 —— 凭据没进环境

症状 ：日志里出现 `POST http://hyrl-sandbox... 401` 或 `403` 。

根因 ：`GONGFENG_API_BASE_URL` / `GONGFENG_TOKEN` 没被加载 。

修 ：

- 检查 `.env` 是否存在 —— CLI 按 `<cases-root>/../.env` → `<repo>/.env` 顺序加载 。
- 若你在 IDE 里跑 ，IDE 的 env 未必包含 shell 的 `.env` 。 建议直接在 shell 里跑或者显式
  `export` 。
- 用 `env | grep GONGFENG` 确认 shell 里已经有 。

## 3 ：judge UNAVAILABLE

症状 ：某条 `judge_claim` 判定 detail 里出现 `judge_unavailable` ，或 orchestrator 频繁触发 infra
retry 。

根因 ：judge 后端网关 5xx / 超时 ，没在允许的重试次数内恢复 。

修 ：

- 单 case 手工 ：临时 `export NEXUS_JUDGE_MAX_RETRIES=10` 再跑一次 。
- Orchestrator 场景 ：`--judge-retries 10` （会自动写到子进程 env ） 。
- 如果 judge 网关短时间内不恢复 ，先跳过 judge 密集 case ，用 `--only` 白名单跑其它 case 。
- 若你想临时用不同 judge 模型 ：export `JUDGE_OPENAI_MODEL` / `JUDGE_OPENAI_BASE_URL` /
  `JUDGE_OPENAI_API_KEY` （见 [reference/env-variables.md](../reference/env-variables.md) ） 。

## 4 ：模板未找到 / `--template` 出错

症状 ：`FileNotFoundError: runtime.template.yaml` 或 `--template anthropic` 但跑起来仍是 litellm 。

根因 ：

- 模板只在 aidata package 目录里 `runtime.template.yaml` 与 `runtime.anthropic.template.yaml` 两个 。
- `--template` 支持别名 、绝对路径 、以及相对 aidata 目录的裸文件名 。 拼错了就会报找不到 。

修 ：

- 用 alias ：`--template litellm` / `--template anthropic` 。
- 或者传绝对路径 。
- 相对路径要相对 `aidata/` 目录 ，不是相对 shell 当前 pwd 。

## 5 ：`.env` 加载顺序不对 、生产凭据没生效

症状 ：跑生产 case 却跑到 dev 网关 ，或反过来 。

根因 ：CLI 用第一个存在的 `.env` ，顺序是 `<cases-root>/../.env` → `<repo>/.env` 。 只要前者存在
就不会 fallback 到后者 。

修 ：

- 明确规划 —— 生产凭据放到下游 `Tasks/../.env` ，开发凭据留在 nexus 仓 `.env` 。
- 若临时想强制某个凭据 ，直接 `export` 覆盖即可 —— 进程 env 优先级最高 。

## 6 ：`score == 0` 且 `error_message` 为 null

症状 ：`task_output.score == 0.0` ，但 `error_message` 是 null ；`main_trajectory` 长度正常 、
`rubric_results[]` 每条都填了 。

根因 ：agent 跑完了 、rubric 也评完了 —— 只是每条 check 都 FAIL 。 通常是 rubric 太严 、或者
agent 路径与 rubric 期望的不一样 。 不是 " 系统故障" ，是 " 建模问题" 。

修 ：

- 看 `rubric_results[].detail` —— 每条 check 说清楚了它期望什么 、看到什么 。
- 对比 `agent_output.main_trajectory` 里 agent 实际走的路径 。
- 如果这个 agent 表现算合理 ，rubric 需要放宽 。 如果 agent 明显跑歪了 ，那就是模型问题 。

## 7 ：`score == 0` 且 `error_message` 非 null

症状 ：任务提前退出 —— `error_message` 会精确指出是 step 0-6 里哪一步失败 。

根因 & 修 ：

- `case_config_error` / `missing_rubric` / `rubric_load_failed` —— case 结构 bug ，先跑
  `aidata.py validate <case>` 。
- `runtime_acquire_failed` —— gongfeng 侧 quota 或 token 问题 。 检查凭据 、看 gongfeng 队列 。
- `service_setup_failed` —— sidecar service 起不来 。 看 log 里的 startup script 输出 。 大概率是
  script 依赖的镜像里没某个二进制 ，或 ready_check 超时太短 。
- `bootstrap_failed` —— `bootstrap_script` 非零 exit 。 手工登进容器复现 ：
  `docker exec ... bash /workspace/.aidata/case/assets/start.sh` 。
- `agent_timeout` —— agent 循环超过 `agent_timeout` 秒 。 case 复杂度或模型差都会导致这个 。

## 8 ：dispatch 报 " unknown assert type"

症状 ：validate 或运行时 rubric 加载失败 ，`unknown assert type: xxx` 。

根因 ：rubric 里写了 `V42_ASSERT_TYPES` 之外的字符串 —— 拼错了 、或者用了旧版本 。

修 ：core primitive 全在 `case/schema.py::V42_ASSERT_TYPES` —— 目前是
`state_equals` / `text_contains` / `tool_called` / `tool_called_sequence` / `judge_claim` /
`test_result` / `script_probe` 。 参考 [author-a-rubric.md](./author-a-rubric.md) 。

## 9 ：v2 rubric DeprecationWarning

症状 ：跑一个还在用 v2 rubric（`verify: llm_judge/eq/regex/shell` ） 的 case ，出现
`DeprecationWarning: v2 rubric ... is deprecated` 。

根因 ：v2 pass-count 语义已冻结 ，只保留兼容路径 ，不再新增能力 。 详见
[explanation/v2-freeze.md](../explanation/v2-freeze.md) 。

修 ：这条只是警告 ，不影响运行 。 想真正修 —— 迁移到 v4.2 ，见
[`../../RUBRIC.md`](../../RUBRIC.md) §7 。 若只是想临时关掉 warning ，Python 端可以
`warnings.filterwarnings` 或 `-W ignore::DeprecationWarning` 。

## 10 ：sidecar service placeholder 没被替换

症状 ：agent 收到的 prompt 里出现字面的 `__AIDATA_JD_BACKEND_URL__` ，或 extra_mcp_servers 起
不来 。

根因 ：`services[].connections[].placeholder` 没在 `user_prompt` / `extra_mcp_servers` 里正确引用 ，
或者 service startup 失败 、URL 未解析出来 。

修 ：

- placeholder 名字必须跟 rubric / prompt / mcp 里出现的字符串 **完全一致** 。
- service ready_check 超时 ？ 抬 `ready_timeout_s` 。
- 在 orchestrator 里 ，`error_message` 会明确写 `service_setup_failed` 加 service name 。

## 找不到符合的症状 ？

- 打开 `workspace/mcp/tmp/aidata_<case>_<ts>.result.json` ，看 `task_output.error_message` 、
  `agent_output.main_trajectory` 、`task_output.rubric_results[].detail` 。
- 若在 orchestrator 里 ，进 `<out>/runs/<run_key>/attempts/N/` 看每次 attempt 的完整 log 。
- 打开 [explanation/architecture.md](../explanation/architecture.md) 、按 12 步骤逐个对齐 " 到哪一步坏" 。
