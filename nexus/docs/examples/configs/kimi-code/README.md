# Kimi Code CLI：Nexus 接入和 SWE 验收

接入的是 Moonshot AI 的 TypeScript 项目 `MoonshotAI/kimi-code`，npm 包为
`@moonshot-ai/kimi-code`。它与已归档的 Python `kimi-cli` 是不同实现。

本适配使用官方 CLI 的工具、系统提示和 agent 循环。Nexus 负责安装、模型代理、
沙箱生命周期、轨迹记录和 SWE patch 评测。实现位于
`nexus/extensions/agents/kimi_code/`，没有修改公共代理或 SWE 判分代码。

## 已验证的版本与入口

2026-10-26 从 npm 镜像查询并固定版本 **2.3.2**；镜像记录发布时间为
2026-10-25。当前节点直连 GitHub 和官方 npm 请求超时，因此没有把镜像的
`latest` 标签当成 GitHub 最新 commit 的证明。发布包携带的源码、CLI help、
配置校验和沙箱实际版本都已检查。

- 安装：`npm install --prefix <isolated-prefix> @moonshot-ai/kimi-code@2.3.2`。
- Node.js：至少 **22.19.0**；优先使用 Nexus runtime 自带 Node，以兼容题目镜像的旧 Node。
- 注册名：`kimi_code`。
- 执行：`node _prompt_launcher.mjs <kimi-entry> <prompt.txt> --model nexus --output-format stream-json`。
- 状态目录：每次运行单独设置 `KIMI_CODE_HOME`。
- 自动更新和自身遥测关闭，保证版本固定。

`--prompt` 模式内部设置 `auto` 权限模式，并明确拒绝与 `--yolo` 或 `--auto`
同时使用。不要套用其他 CLI 的启动参数。

2.3.2 没有直接读取 prompt 文件或 stdin 的选项。Nexus 将任务按 UTF-8 原文写入
独立运行目录下权限为 `0600` 的 `prompt.txt`，启动层在同一 Node 进程内读取文件，
再交给官方 `--prompt` 入口。系统启动参数只包含文件路径，长任务不会触发 Linux
单参数大小限制；提示全文、换行和工具权限模式保留。`prompt.txt` 随运行产物保存，
不写入待评测的仓库；模型本身的上下文上限仍然适用。

## 模型代理与 thinking

Kimi 的 `config.toml` 中建立 `type = "openai"` 的 `nexus` provider，指向
沙箱内 `http://127.0.0.1:<port>/v1`。该代理把请求交给宿主的 Nexus client，
由 `openai_clients.main` 决定真实模型名、网关和参数。

```yaml
agent_name: kimi_code
agent_config:
  version: '2.3.2'
  working_dir: /testbed
  timeout: 12000
  context_window: 1000000
  max_tokens: 16384
  thinking_effort: high
```

`thinking_effort` 会声明模型支持 thinking，配置 `reasoning_key = "reasoning_content"`
及 `thinking.keep = "all"`。自部署 HY4 的真实 high 设置还需要在宿主 client
的 `enforced_request_args.extra_body.chat_template_kwargs` 中固定；不能只看 CLI 的显示值。

额外的环境变量可以通过 `extra_env` 设置。模型路由、状态目录、自动更新和
Node 注入参数受保护；冲突键会记录警告。不同 run 不共享 Kimi 会话。

## 工具 smoke

先提供沙箱和 Langfuse 环境变量。真实凭据仅存于 ignored 文件，不写进 YAML。

```bash
export MODEL_NAME='HY4-T3-A49B-SFT1008-RL1009-Penalty-step460-r200'
export OPENAI_BASE_URL='http://infer-proxy-yb-test.production.polaris:8000/openapi'
export OPENAI_API_KEY='EMPTY'

mkdir -p experiments/kimi-smoke/runtime_data experiments/kimi-smoke/temp
export TMPDIR="$PWD/experiments/kimi-smoke/temp"
export SQLITE_TMPDIR="$TMPDIR"
export NEXUS_SQLITE_NO_WAL=1

uv run nexus run single docs/examples/configs/kimi-code/smoke.yaml
```

验收不能只看进程退出码：
应检查真实文件值、shell 断言、实际模型请求和 Langfuse 工具反馈。

2026-10-26 的独立诊断实际完成 **5 次模型请求、6 次工具操作**：Write×3、Read×2、
Bash×1；沙箱独立读取 `sum.txt` 得到 `36`。Langfuse 已逐条核对原始请求、响应、
token 和工具输入输出，全部一致。输入102,259 token，输出697 token，缓存命中101,071 token。

项目 `zkc/cb-insider-verify`，trace ID：`be97c85fc0f13b08fea15a799394dad5`。

诊断成功后，CLI 退出清理阶段出现 `workspace ... is not materialized` 日志。
保留了该日志；实际断言成功、退出码0、完整结束标记和最终文件均已验证。
这条上游清理日志不是“整个链路无告警”的证明，SWE 验收需继续关注补丁保存。

## SWE smoke

使用现有 `swe_task` 与每题自己的镜像，开启 `worker_venv: {modules: []}`，
上传当前 checkout 的 wheel。在 main/eval worker 的 `extra_requirements` 中加入
`dataclasses-json>=0.6.7` 和 `unidiff>=0.7.5`。

两类题库均开启：

```yaml
task_config:
  enable_anti_hack_prompt: true
  enable_reward_hacking_detection: true
  enable_anti_hack: true
  hack_reward_override: 0.0
```

本次 pilot 固定三题，同时运行：

1. Multilingual：`preactjs__preact-3010`。
2. Multilingual：`nushell__nushell-12950`。
3. Pro：`instance_protonmail__webclients-5e815cfa518b223a088fa9bb232a5fc90ab15691`。

复用上面的模型、沙箱和 Langfuse 环境变量，指定完整实例 JSONL：

```bash
export SWE_DATASET_PATH='/absolute/path/to/cases.jsonl'
export SWE_LIMIT=3
export SWE_INSTANCE_IDS='preactjs__preact-3010,nushell__nushell-12950,instance_protonmail__webclients-5e815cfa518b223a088fa9bb232a5fc90ab15691'

uv run nexus run batch \
  --generator docs/examples/configs/kimi-code/swe_generator.py \
  --database experiments/kimi-smoke/runtime_data/swe.db \
  --max-concurrent 3
```

generator 默认最多3题，支持 `SWE_AGENT_TIMEOUT`、`SWE_RUNTIME_TIMEOUT`、
`SWE_RUNNER_TIMEOUT`、`SWE_EVAL_TIMEOUT` 和 `SWE_MAX_STEPS`。

三道 smoke 均为 `score=1.0`，并已逐条核对 Langfuse：

| case | score | 模型调用 | 工具操作 | trace ID |
| --- | ---: | ---: | ---: | --- |
| preactjs__preact-3010 | 1.0 | 76 | 122 | `1105914b2765f866f31526a4f903b645` |
| nushell__nushell-12950 | 1.0 | 91 | 97 | `2f7643a9f5f813b9d4434f82ecf35537` |
| protonmail/webclients（Pro） | 1.0 | 119 | 203 | `0f04e228597639b0aa84dc1ed0e19e4e` |

Pro 1/1、Multilingual 2/2通过；286次模型调用、422次工具操作完整核对。
独立判分日志回放3/3一致，10个实验沙箱全部停止。
完整验收见 [验收记录](acceptance-20261026.md)。仅有三题 smoke 不代表全量能力分数。

首轮实验脚本漏转义 `find -exec {}`，在模型调用前触发 SWE 的 `.format(**instance)`
错误；三条零调用原记录保留在 `pilot`，修正后独立运行 `pilot-config-fixed`。
这里的通用 generator 没有这条额外清理脚本，使用 Nexus 的默认 SWE 初始化。

## 输出与验收口径

`KimiCodeOutput.output` 保留完整 JSONL；`assistant_text`、`session_id`、`trace`
提供解析后的摘要。上游 stream-json 不输出 reasoning 文本，thinking 和 usage
应从 Nexus 捕获的真实模型响应及 Langfuse generation 中检查。

适配要求退出码0、有效 `session.resume_hint`、无结构化 error 和无损坏 JSONL
才标记成功。超时写入通用 `exit_status=timeout`，Harbor 可转换为 `TASK_TIMEOUT`。
工具调用和返回都有 ID；未返回工具的状态保持未知，不补造成功输出。

超时、取消和正常退出都会回收本次 CLI 的进程组及发现的子进程。原始 transcript
保存在沙箱运行目录，`artifact_dir` 返回路径；评测需要在释放沙箱前下载证据。

## 测试

```bash
uv run pytest tests/unit/extensions/agents/test_kimi_code.py -q
uv run ruff check nexus/extensions/agents/kimi_code tests/unit/extensions/agents/test_kimi_code.py
uv run ty check nexus/extensions/agents/kimi_code
```

单元测试覆盖错误结束、工具结果隔离、TOML作用域、模型环境隔离，以及真实子进程的
超时和取消回收。SWE 判分以原始 `task_output.score` 为准，不能拿 CLI 的退出成功代替。

长提示回归使用真实 2.3.2 CLI、Nexus 本地 HTTP 代理和固定模型回复，验证短提示、
超过 Linux 单参数上限的 ASCII/中文任务全文到达模型，以及步数耗尽时的非零退出。
该测试不请求线上模型；需要先安装对应 CLI，并指定入口：

```bash
export KIMI_CODE_TEST_ENTRY='/absolute/path/to/node_modules/@moonshot-ai/kimi-code/dist/main.mjs'
export KIMI_CODE_TEST_NODE='/absolute/path/to/node'
uv run pytest tests/integration/kimi_code/test_kimi_prompt_input.py -q
```
