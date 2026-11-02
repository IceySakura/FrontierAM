---
title: 结构重构时间线
audience: 想理解 " 为什么代码是现在这个组织方式 " 的开发者
prerequisites: 无
related:
  - ./architecture.md
  - ./v2-freeze.md
  - ./fetcher-and-primitives.md
---

# 结构重构时间线

aidata 在过去几个月经历了六次以上的结构重构 —— 每次都是 as-is 、每次都保留了 " 老 import path 仍
可用 " 的兼容承诺 。 本篇按时间倒序列出关键节点 、每步做了什么 、以及做出的兼容承诺 。

目的是 ：新加入的 reader 看到 `engine.rubric` 与 `rubric/` 、`engine.case_spec` 与 `case/spec.py`
两套文件同时存在时能理解 " 哪个是权威 、哪个是 shim " 。

## `097962d8` engine/ 拆成 rubric/ + case/ + runtime/ + mock_servers/

**做了什么** ：

`engine/` 内部原本有大约 15 个文件 ，混合着任务入口 、case 加载 、rubric 、沙箱管理 、mock 服务器
脚本 、系统提示 。 这次拆成 5 个包 ：

- `engine/` —— 只留 task entry 与沙箱布局常量 。
- `rubric/` —— 打分子系统 。 独立 、几乎不 import 上层 。
- `case/` —— case 加载 + schema 单一真相源 。
- `runtime/` —— 沙箱 、上下文 、服务编排 。
- `mock_servers/` —— shell / llm_client / confine 。

**兼容承诺** ：

`engine/case_spec.py` 、`engine/rubric.py` 、`engine/junit.py` 等每个原文件都保留 一 shim ，重新
导出新位置的名字 。 老代码 `from ..engine.case_spec import resolve_case` 仍能工作 。 shim 里
一行代码都没有业务逻辑 —— 纯 re-export 。

**为什么** ：`engine/` 单一目录随着 rubric 的 primitive 化和 sidecar service 支持已经膨胀到不可维
护 。 拆包让 " 打分 " 、" case 加载 " 、" 运行时 " 三条正交关注点各自演化 。

## `3b0d0d1c` v2 rubric 冻结 + schema 单一真相源

**做了什么** ：

- 建立 `case/schema.py` ，把 `V42_ASSERT_TYPES` / `LEGACY_VERIFY_KINDS` /
  `ALLOWED_DIMENSIONS` 等常量集中 。
- CLI 校验器 `aidata.py` 与运行时加载器 `case/spec.py` 都从 `schema.py` 导入 —— 之前两边各有一份 、
  多次发生漂移 。
- v2 rubric 分派路径整体转移到 `rubric/legacy_v2.py` ，冻结 —— 详见
  [`./v2-freeze.md`](./v2-freeze.md) 。
- validate 阶段拒绝新的 v2 写入 ；运行时加载 v2 打 DeprecationWarning 。

**兼容承诺** ：约 59 个生产 v2 case 无迁移压力 —— 继续跑 、pass-count 语义逐字节一致 。

## `e8f05874` Fetcher 门面 + `script_probe` primitive

**做了什么** ：

- `rubric/fetch.py::Fetcher` 引入 —— 中心化证据抓取层 。 详见
  [`./fetcher-and-primitives.md`](./fetcher-and-primitives.md) 。
- 加入 `FileSource` dataclass ，支持 `from: {file, service, parse_as}` dict 形式 ；
  `service` 可指向 sidecar 、`parse_as` 支持 json / yaml / csv 。
- 加入 `script_probe` primitive 、`ProbeSpec` dataclass 、`primitives/script_probe.py` 。
  运行结束后跑一段脚本 、看 stdout / exit_code 、支持主容器与 sidecar 。

**兼容承诺** ：

- 传统 `from: "file:/path"` 字符串形式 保持逐字节一致 —— 不改任何生产 case 的判定 。
- 通过 `NEXUS_AIDATA_RUBRIC_LEGACY=1` 逃生阀路由回 " 门面之前 " 的直接 state 访问路径 —— 若
  某个 primitive 在生产 case 上出现回归可以立刻切回 ，不用 revert 。

## `9cb64b19` `_impl.py` 拆分成内聚子模块

**做了什么** ：

`rubric/_impl.py` 曾是几百行的 " 什么 helper 都往里塞 " 文件 。 这次拆成 ：

- `rubric/config.py` —— `RubricLimits` / `JunitLookupContext` / `VerifierResult` typedef /
  `_resolve_concurrency` / `_RUBRIC_LEGACY_MODE` 逃生阀 / 常量 。
- `rubric/dispatch.py` —— DAG 顶层分发 、`_ASSERT_EVALUATORS` 注册表 、`compute_scores` /
  `run_rubric_async` / `run_rubric` 。
- `rubric/legacy_v2.py` —— v2 分派路径 。
- `rubric/selectors.py` —— `_gather_source` / `_resolve_state_source` / 等选择器 helper 。
- `rubric/sql.py` / `rubric/jmespath_predicates.py` / `rubric/tool_matcher.py` —— 各自一个内聚
  子域 。
- `rubric/primitives/*.py` —— 每个 v4.2 assert primitive 一个文件 。

**兼容承诺** ：`rubric/_impl.py` 保留 —— 里面剩一小部分尚未搬走的辅助 、以及少量兼容 re-export 。
持续瘦身中 。

## `e357bf19` 中文注释 i18n

**做了什么** ：把代码里的英文 docstring / 关键 comment 系统性地翻成中文 —— 与 `CLAUDE.md` 里的
Chinese punctuation 规范对齐 。 单纯注释改动 、无功能变化 。

**兼容承诺** ：无 —— 只是注释 。 不影响 API / 行为 。

## `d1f6ee4e` sync 3 upstream commits（orchestrator / prompt-caching / --template ）

**做了什么** ：从上游 nexus 主线合并三个能力 ：

- **orchestrator** —— `aidata_orchestrator.py` 独立整轮编排器（run / monitor / dashboard / status /
  stop 五子命令 ） 。 详见 [`../how-to/batch-run.md`](../how-to/batch-run.md) 。
- **prompt-caching** —— vertex_anthropic passthrough 支持原生 prefix 缓存 。 需要 `--template
  anthropic` 或 orchestrator 自动按模型族切模板 。
- **`--template`** —— CLI 层显式选 runtime 模板 ；alias `litellm` / `anthropic` / `vertex` /
  `passthrough` + 路径 。

**兼容承诺** ：`--template` 缺省 = `litellm` ，与之前的默认路径一致 。 不设 `--template` 的老脚本无
影响 。

## 总结的兼容原则

从这几次重构可以抽出三条原则 —— 也是未来重构应该遵守的 ：

1. **老 import 路径永远 shim** —— 新目录搬完之后 ，旧文件保留一层 re-export 。 不留 broken 悬挂 。
2. **YAML 表面语义永不改变** —— case bundle 是 spec ；重构可以改代码内部 ，但不能让同一份
   `task.yaml` / `rubric.yaml` 得出不同分数 。
3. **有逃生阀** —— 引入新抽象（如 Fetcher 门面） 时 ，加环境变量开关允许一次性回退 。 遇到生产
   回归时不必 git revert 。

这三条让重构可以频繁 、 增量 、 低风险 —— 而不是攒一次性大爆炸 。

## 相关文档

- 每个包现在做什么 —— [`../reference/module-layout.md`](../reference/module-layout.md) 。
- 数据流全景 —— [`./architecture.md`](./architecture.md) 。
- v2 冻结的具体承诺 —— [`./v2-freeze.md`](./v2-freeze.md) 。
- Fetcher 引入的动机与阶段 —— [`./fetcher-and-primitives.md`](./fetcher-and-primitives.md) 。
