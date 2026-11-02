---
title: Fetcher 门面与 primitive 层
audience: 想扩展 primitive 或理解证据抓取抽象的开发者
prerequisites:
  - 已经读过 [author-a-rubric.md](../how-to/author-a-rubric.md)
related:
  - ./architecture.md
  - ../how-to/use-fetcher-parse-as.md
---

# Fetcher 门面与 primitive 层

## 问题

在重构之前 ，每个 rubric primitive 直接读 `state[...]` —— 举例 ：

```python
# 简化后的 pre-refactor 面貌
async def _eval_state_equals(check, state, runtime, limits):
    source = check['assert']['from']
    ns, path = source.split('.', 1) if '.' in source else (source, '')
    slice_ = state.get(ns, {})
    for seg in path.split('.'):
        slice_ = slice_[int(seg)] if seg.isdigit() else slice_[seg]
    ...
```

这样写有几个具体问题 ：

1. **来源逻辑分散** —— 每个 primitive 各自写 `state[...]` 遍历 、jmespath 、SQL 分支 。 加一个新
   来源（比如 " 从 sidecar 读文件" ） 要改所有 primitive 。
2. **不能跨容器** —— `state[...]` 只能读 mock 收集的东西 ；agent 主容器和 sidecar 里的文件系统
   彻底看不见 。
3. **测试面爆炸** —— 每个 primitive 都要构造一份 fake state ，重复劳动 。

## 解 —— Fetcher 门面

`nexus/extensions/tasks/aidata/rubric/fetch.py::Fetcher` 是 rubric 子系统的中心化证据抓取层 。 所有
primitive 通过 Fetcher 拿证据 ，而不是直接触碰 state[...] 。

Fetcher 的方法 ：

| 方法 | 用途 |
|---|---|
| `trajectory(*, limits=)` | 拍平后的 trajectory 文本 。 与 `_gather_trajectory` 逐字节一致 。 |
| `state_slice(cfg)` | 结构化 state 访问 ；分发优先级同 `_resolve_state_source` 。 |
| `text_source(cfg, *, limits=)` | `_gather_source` 的门面 ，选择源并拍平 。 |
| `file(source)` | 抓取一个文件 ，可选从 sidecar 读 、可选解析 JSON / YAML / CSV 。 |
| `probe(spec: ProbeSpec)` | 运行结束后跑一段脚本 ，返回 `(stdout, exit_code)` 。 |

引入方式 ：`run_rubric_async` 在评估开始时创建一个 Fetcher 、放到 `state['_fetcher']` ，primitive
通过约定 key 拿到 —— 无需修改 primitive 签名 。 这让迁移可以逐个 primitive 、逐字节等价 、每一步
可回滚 。

## 阶段化演化

Fetcher 的引入是一个逐步演化的过程 ，不是一次性重写 。

### 第一阶段 —— 薄门面

Fetcher 最初是对已有 `_impl.py` 辅助函数（`_gather_source` / `_gather_trajectory` /
`_resolve_state_source` / `_text_source_selector`） 的薄包装 。 primitive 从直接调用这些辅助 ，切换到
调 Fetcher —— **逐字节等价** ，没引入新语义 。

这一阶段的价值 ：给后续能力扩展留出接入点 ，同时不冒改动生产 case 结果的风险 。

### 第二阶段 —— 结构化 dict 形式

`from: {file, service, parse_as}` 的 dict 形式 —— 见
[`../how-to/use-fetcher-parse-as.md`](../how-to/use-fetcher-parse-as.md) 。 关键新能力 ：

- `service` —— 从 sidecar 容器读 。 复用 SQL 运行时解析器的模式 —— 在 `ExecutionContext` 上查找
  已获取的 sidecar 运行时 ，然后 `run_command('cat <path>')` 。
- `parse_as` —— 自动解析 JSON / YAML / CSV ，让下游 jmespath / regex 可以直接用结构化值 。

传统 `from: "file:/path"` 字符串形式 **保持逐字节一致** —— 不迁移已有 case 、不产生 diff 。

### 第三阶段 —— `script_probe` primitive

`ProbeSpec` dataclass + `Fetcher.probe()` + `primitives/script_probe.py` 三件套一起加入 。 原语义
类似 " 后置钩子 shell " ，主要用于 " state 结构不方便结构化断言 ，但一句 shell 就能确定性判断 " 的
场景 。

## 逃生阀

`config.py::_RUBRIC_LEGACY_MODE` 从环境变量 `NEXUS_AIDATA_RUBRIC_LEGACY` 读取 。 设为 `1` 时
路由到 " 门面之前 " 的直接 state 访问路径 —— 若某个基于 Fetcher 的 primitive 在生产 case 上出现
回归 ，无需 git revert ，切换环境变量即可 。

移除目标 ：Fetcher 门面平稳落地后再过一个发布周期 。

## primitive 层的形状

每个 primitive 一个文件 ，位于 `rubric/primitives/` 。 命名与 assert type 对齐 ：

| Primitive 文件 | Assert type |
|---|---|
| `state_equals.py` | `state_equals` |
| `text_contains.py` | `text_contains` / `text_not_contains` |
| `tool_called.py` | `tool_called` / `tool_called_sequence` |
| `judge_claim.py` | `judge_claim` |
| `test_result.py` | `test_result` |
| `script_probe.py` | `script_probe` |

所有 primitive 都是 `async def _eval_...(check, state, *, runtime, junit_ctx, limits) -> VerifierResult` 。
分派表在 `dispatch.py::_ASSERT_EVALUATORS` 。 加一个新 primitive 三步 ：

1. 在 `case/schema.py::V42_ASSERT_TYPES` 里加名字（validate 层拒绝未知类型） 。
2. 在 `rubric/primitives/<name>.py` 写实现 。
3. 在 `rubric/dispatch.py::_ASSERT_EVALUATORS` 里加一行注册 。

如果你的新 primitive 需要一种新的证据来源 —— **在 Fetcher 上加方法** ，不要在 primitive 里直接
读 state 。 这条约束保证了 Fetcher 的中心化不被弱化 。

## 相关文档

- 使用侧文档 —— [`../how-to/use-fetcher-parse-as.md`](../how-to/use-fetcher-parse-as.md) 。
- 数据流全景 —— [`./architecture.md`](./architecture.md) 。
- 结构重构历史 —— [`./refactor-history.md`](./refactor-history.md) 。
