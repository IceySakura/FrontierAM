# backend_web_fetch

回归后脑原生 WebFetch，不依赖搜索发现来源。 固定读取 TypeSafe 官方介绍页，整理实际页面内容并生成摘要。

## 数据与运行

数据：[`data/backend_web_fetch.json`](data/backend_web_fetch.json)。 默认 WorkBuddy，镜像与运行环境继承 example 默认配置，无需 worker venv 或额外安装包。

```bash
uv run python -m nexus.extensions.projects.hy_live.example \
  --case backend_web_fetch \
  --output <output-dir>/result.json
```

显式 `--backend codebuddy` 可对 CodeBuddy 运行同一回归。 凭据只通过 `CODEBUDDY_API_KEY` 环境变量提供。

## 审查清单

1. 小脑委托网页读取，中脑创建 Task，用户批准权限后继续。
2. 后脑原生 WebFetch 读取 `https://docs.typesafe.ai/introduction`，工具结果有页面正文，不是错误、空结果或只有 URL。
3. `fetch_summary.md` 包含页面实际介绍的 Jev 和三种原语，并附来源链接；不使用 Bash/curl 或模型记忆替代抓取。
4. Task 为 `done`，摘要文件登记为产物，完成通知送达小脑。

CLI 自动检查 `native_web_fetch_succeeded` 和 `tasks_done`；摘要内容和产物仍按本清单审查。
