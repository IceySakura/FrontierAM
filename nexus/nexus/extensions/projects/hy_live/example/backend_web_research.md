# backend_web_research

让后脑调研 Typesafe 的 JEV，观察原生 `WebSearch` 和 `WebFetch` 是否真实联网成功。 名称保留用户原文，由后脑查证，不预设它实际指哪个项目。

## 数据与运行

数据：[`data/backend_web_research.json`](data/backend_web_research.json)。 LLM user 只批准调研权限、答复名称问题并等待报告完成。

```bash
python -m nexus.extensions.projects.hy_live.example \
  --case backend_web_research \
  --output <output-dir>/result.json
```

默认 WorkBuddy，用 `--backend codebuddy` 可观察 CodeBuddy 的同一任务。 原生搜索使用环境变量 `CODEBUDDY_API_KEY`，数据里只保存变量引用，见 [`../docs/impl/codebuddy_web_search.md`](../docs/impl/codebuddy_web_search.md)。 两个后端的认证配置不同，不能因其中一个通过就判定另一个也通过。

小脑 `web_search_provider=none`，需要写报告的完整请求交给后台，不以小脑元宝搜索代替后脑联网。

## 审查清单

1. 小脑调用 `delegate`，中脑创建一个调研 Task。 权限请求由用户批准；有补充问题时答复原 Run。
2. 检查 `result.json` 的 `task_output.tasks[].result.trajectory`：存在原生 `WebSearch`，查询包含 Typesafe、JEV 或查证后的候选名称。
3. `WebSearch` 的工具结果实际返回来源链接或摘要，不是 401、403、超时、占位凭据错误或空结果。 仅调用工具不算成功。
4. 后脑调用原生 `WebFetch` 读取至少两篇来源页面，工具结果有页面正文或摘要。 只输出链接、调用 Bash/curl、Fetch 报错后凭记忆写报告，都不算此项通过。
5. 报告说明名称查证结果，包含用途、设计、示例、维护情况及来源链接。 无法查证 JEV 时如实说明，不虚构一个同名项目。
6. Task 为 `done`，research.md 已登记，完成通知送达小脑。

CLI 自动检查原生 Search 返回来源、原生 Fetch 至少读取两篇不同来源、所有 Task 为 `done`。 最终审查还需记录工具名、查询、URL 和实际返回内容，并检查报告和产物。 研究内容的措辞、排版不作为偏差。
