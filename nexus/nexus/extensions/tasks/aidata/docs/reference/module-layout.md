---
title: 模块布局参考
audience: 想理解代码组织的开发者
prerequisites: 无
related:
  - ../explanation/architecture.md
  - ../explanation/refactor-history.md
---

# 模块布局参考

aidata 模块位于 `nexus/extensions/tasks/aidata/` 。 经历过几轮重构 ，当前分成
 五 个包 + 若干顶级脚本 。

## 顶层脚本

| 文件 | 作用 |
|---|---|
| `aidata.py` | 单 case CLI（`run` / `list` / `validate` / `package` ） 。 见 [cli.md](./cli.md) 。 |
| `aidata_orchestrator.py` | 批量编排器（`run` / `monitor` / `dashboard` / `status` / `stop` ） 。 见 [cli.md](./cli.md) 。 |
| `dataset_adapter.py` | Nexus Dataset 的 AIDATA instance → 现有 runner 认识的 case bundle 。 |
| `generator.py` | `nexus run-batch` 用的 batch generator ，扇出 cases 树 。 |
| `runtime.template.yaml` | LiteLLM 模板 —— 默认 、Gemini / GPT / OpenAI 兼容 。 |
| `runtime.anthropic.template.yaml` | vertex_anthropic passthrough 模板 —— Claude 系原生 prefix 缓存 。 |
| `pyproject.toml` | task extension 注册 + 依赖声明 。 |
| `README.md` | 模块级 README ，含目录结构 、常用 CLI 、 demo 段 。 |
| `RUBRIC.md` | rubric v4.2 完整字段字典 。 |
| `RUBRIC_AUTHORING.md` | rubric 编写风格 、推荐组合 。 |

## 五个子包

### `engine/` —— 任务入口 + 沙箱布局

- `task.py` —— `@task_registry.register('aidata')` 的入口函数 `run_aidata` 。 12 步 orchestration
  见 [`../explanation/architecture.md`](../explanation/architecture.md) 。
- `models.py` —— `AidataTaskInput` / `AidataTaskOutput` / `AidataService` 等 pydantic 模型 。
- `paths.py` —— 沙箱布局常量（`AGENT_DESKTOP` / `INTERNAL_ROOT` / `TESTS_DIR_IN` 等） +
  `AIDATA_MAIN_IMAGE` 覆盖点 + image 前缀 → MCP_PY / SHELL_PATH 映射 。
- `system_prompt.py` —— 默认 preamble ，可被 `/workspace/system_prompt.md` override 。
- `case_spec.py` / `context.py` / `sandbox.py` / `mcp_catalog.py` / `services.py` / `state.py` /
  `rubric.py` / `junit.py` / `check_map.py` / `prompt_builder.py` / `base_tools.py` —— 兼容
  shim ，实际实现已经迁到 `case/` 或 `runtime/` 下 。 老代码 `from ..engine.X import Y` 仍能工作 。
- `servers/` —— 兼容 shim ，实际实现在 `mock_servers/` 下 。

### `rubric/` —— 打分子系统

- `config.py` —— `RubricLimits` （trajectory 拍平上限） / `JunitLookupContext` /
  `_RUBRIC_LEGACY_MODE` 逃生阀 / 并发解析器 。
- `dispatch.py` —— DAG 顶层分发 。 `run_rubric_async` / `compute_scores` /
  `_ASSERT_EVALUATORS` 注册表 。 v4.2 与 v2 分派入口 。
- `fetch.py` —— Fetcher 门面 ，所有 primitive 的证据抓取都走它 。 `FileSource` /
  `ProbeSpec` dataclasses 、`file()` / `probe()` / `state_slice()` / `text_source()` /
  `trajectory()` 方法 。
- `judge.py` —— LLM judge client + `NEXUS_JUDGE_MAX_RETRIES` 。
- `junit.py` —— junit XML 解析 ，为 `test_result` primitive 提供 `JunitReport` 。
- `check_map.py` —— `check.id → junit testcase name` 映射加载 。
- `legacy_v2.py` —— v2 pass-count 分派 ，冻结中 。 见
  [`../explanation/v2-freeze.md`](../explanation/v2-freeze.md) 。
- `policy.py` —— `ScorePolicy` （threshold / continuous / use_threshold ） 。 见
  [`../explanation/rubric-scoring-model.md`](../explanation/rubric-scoring-model.md) 。
- `reward.py` —— 默认 ScorePolicy 与可选 `config.reward.script` 的最终 reward 聚合、脚本协议和静态校验。
- `selectors.py` —— `_gather_source` / `_resolve_state_source` / `_text_source_selector` 等
  底层选择器 。 Fetcher 门面在其上包了一层 。
- `sql.py` —— `state_sql:` selector（SQLite 探针） 。
- `jmespath_predicates.py` —— jmespath 谓词封装 。
- `tool_matcher.py` —— tool_call arguments 匹配（`contains_all_keys` / `equals` / regex ） 。
- `_impl.py` —— 遗留内部辅助 ；持续瘦身中 ，具体见
  [`../explanation/refactor-history.md`](../explanation/refactor-history.md) 。
- `primitives/` —— v4.2 assert 原语实现 ：`state_equals.py` / `text_contains.py` /
  `tool_called.py` / `judge_claim.py` / `test_result.py` / `script_probe.py` 。 每个原语一
  独立文件 ；均通过 Fetcher 拿证据 、返回 `VerifierResult = (passed, verdict, detail, raw_head)` 。
  每个 primitive 的 dispatch 树与 verdict 集见 [`./rubric-primitives.md`](./rubric-primitives.md) 。

### `case/` —— case 加载

- `spec.py` —— case bundle 加载入口 ：`resolve_case()` / `load_case_rubric()` /
  `case_rubric_path()` 。 处理 `task.yaml` / `case.yaml` / `rubric.yaml` / `data/` 、v4.2 与 v2 兼容
  、`extra_mcp_servers` 归一化 、metadata 校验 。
- `schema.py` —— 单一真相源 —— `V42_ASSERT_TYPES` / `LEGACY_VERIFY_KINDS` /
  `ALLOWED_DIMENSIONS` / `V2_VERSION_LITERALS` / `METADATA_DATE_RE` /
  `ALLOWED_AGENT_TOOLS_BY_CATEGORY` 。 CLI 校验器与运行时加载器
  都从这里导入 ，避免双份漂移 。
- `models.py` / `paths.py` —— 数据模型与路径常量的兼容 shim 。

### `runtime/` —— 沙箱 、上下文 、服务

- `sandbox.py` —— 上传 helpers 、bootstrap 执行 、confine 注入 。
- `services.py` —— `task.yaml::services` 的容器编排 ；每个 service 一个独立 nexus runtime
  、placeholder 替换 、env_name 注入 。
- `mcp_catalog.py` —— 默认 MCP 目录（`_BUNDLED_MOCKS` ） + per-case extras 合并 、
  local vs mock_server 双源 。
- `state.py` —— 运行结束后调所有 `*_dump_state` 工具 、下载 mock state 、生成
  `state_summary` + 提取 agent 最终答案 。
- `context.py` —— host 侧 context 文件 manifest 。
- `prompt_builder.py` —— 组装 agent 用户消息（清单 + 任务 + skill frontmatter ） 。
- `base_tools.py` —— aidata 特有的本地 base tool 注册 。

### `mock_servers/` —— 自带 mock 脚本

- `shell_server.py` —— `run_command` MCP tool 。 `bash -c` 薄包装 ，cwd pin 在
  `AGENT_DESKTOP` ，16 KiB 输出截断（head+tail 保留） ，对 `rm -rf` 等风险模式非阻塞警告 。
- `llm_client.py` —— mock LLM client ，被各个 mock server 复用 。
- `confine/` —— `libconfine.c` + `compile.sh` + `README.md` ；SIGKILL bomb 防护套件 ，
  在 bootstrap 时预编译到主容器 。
- `crontab.md` —— crontab mock 的 SKILL 说明 。

## 五层的边界

粗略讲 ：

```
CLI（aidata.py + aidata_orchestrator.py ）
      │
      ▼
task_registry.register('aidata') 入口 ── engine/task.py::run_aidata
      │
      ├── case/           解析 case bundle → AidataTaskInput
      ├── runtime/        准备沙箱 、拉起 services 、 加载 MCP 、 组装 prompt
      ├── mock_servers/   （被 runtime/mcp_catalog 上传到沙箱内运行 ）
      ├── engine/         orchestrate agent 运行 、 close MCP 、下载 state
      └── rubric/         对最终 state 打分
```

`engine/` 只做编排 ，重活分给 case / runtime / rubric 。 `case/` 与 `runtime/` 之间没有直接依赖 ；
两者都被 `engine/task.py` 调用 。 `rubric/` 独立 ，几乎不 import 上面几个包 —— 它只吃 state
dict + rubric dict 出结果 。

## 相关文档

- 一次运行的完整数据流 —— [`../explanation/architecture.md`](../explanation/architecture.md) 。
- 为什么会分成这五层 —— [`../explanation/refactor-history.md`](../explanation/refactor-history.md) 。
