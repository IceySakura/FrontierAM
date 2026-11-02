---
title: 使用 Fetcher parse_as 与 script_probe
audience: rubric 作者 、需要跨容器读文件或跑脚本探针的开发者
prerequisites:
  - 熟悉 v4.2 assert 原语（[author-a-rubric.md](./author-a-rubric.md) ）
related:
  - ../explanation/fetcher-and-primitives.md
  - ../../RUBRIC.md
---

# 使用 Fetcher parse_as 与 script_probe

Fetcher 是 rubric 子系统的中心化证据抓取层 —— 所有 primitive 通过它读证据 ，而不是直接触碰
`state[...]` 。 它开出了两条新能力 ：

1. `from: {file, service, parse_as}` —— 结构化文件源 ，可从 sidecar 容器读 、可自动解析 JSON / YAML / CSV 。
2. `script_probe` primitive —— 运行结束后跑一段脚本 ，看 stdout / exit_code 。

本篇讲 " 什么时候用 " 与 " 怎么写 " 。 设计动机与门面演化见
[explanation/fetcher-and-primitives.md](../explanation/fetcher-and-primitives.md) 。

## 结构化文件源 `from: {...}`

传统写法 ：`from: "file:/tmp/output.json"` —— 从主容器读一个文件 ，返回原始字符串 。

新的 dict 形式 ：

```yaml
assert:
  type: state_equals
  from:
    file: /var/lib/jd_backend/orders.json
    service: jd-backend           # 可选 ：从某 sidecar 容器读 ；缺省 = main
    parse_as: json                # 可选 ：json | yaml | csv ；缺省 = 原始字符串
  path: 0.status
  equals: paid
```

三个字段的语义 ：

- `file` （必填） 目标文件路径 ，容器内 absolute 路径 。
- `service` （可选） sidecar service 名字（`task.yaml::services[].name` ） 。 缺省或 `main` = 主 agent
  容器 。 当声明的 service 未获取成功时 ，Fetcher 会静默返回空字符串 ，与 `from: file:...` 的
  on-missing 行为一致 —— rubric 判定为 FAIL ，不 crash 。
- `parse_as` （可选） 后续下游 primitive 消费的形状 ：
  - `json` 解析失败保留原始字符串 ，让下游 jmespath / regex 仍可继续处理 。
  - `yaml` 同上 。
  - `csv` 返回 `list[dict]`（DictReader ） ，方便 jmespath 遍历 。
  - 缺省 = 原始字符串 。

代码入口 ：`nexus/extensions/tasks/aidata/rubric/fetch.py::Fetcher.file` +
`FileSource` dataclass 。

## `script_probe` primitive

写在 `assert.type: script_probe` 里 ，让 rubric 在 agent 结束后跑一段脚本作为最终判定 。 与
`test_result` 的区别 ：`test_result` 走 pytest 生成的 junit XML ；`script_probe` 是一次性 shell 脚本 ，
不需要框架化 。

最小示例 ：

```yaml
assert:
  type: script_probe
  script: |
    sqlite3 /var/lib/backend/db.sqlite "select count(*) from orders where user_id='u42'"
  service: main
  timeout_sec: 15
  expect_exit: 0
  regex: '^3\s*$'
```

字段 ：

- `script` 会用 shell 执行，rubric 读取退出码和 stdout。
- `service` 缺省 `main` ；写某 sidecar 名字就在那个容器里跑 。
- `timeout_sec` 缺省 15 秒 ；超时 → 判定 FAIL 、`detail` 说明超时 。
- `expect_exit` 缺省 0；实际退出码不一致时判定失败。
- `contains` / `regex` 可选，用于检查 stdout。

当前实现不读取 `capture` / `equals`，也不会把 stdout 中的数值直接作为该 check 的连续分数。
每条 `script_probe` 仍只产出通过或失败，再由 rubric 的静态 `weight` 汇总总分。

代码入口 ：`nexus/extensions/tasks/aidata/rubric/fetch.py::Fetcher.probe` +
`ProbeSpec` dataclass 。

## 什么时候用 、什么时候不用

**用** ：

- 你的 case 起了 sidecar service 、要 assert 那个 sidecar 里的一个文件或数据库状态 。
- state dump 结构不方便结构化断言 ，但一句 shell 就能确定性判断 。
- 需要跨容器读文件（agent 主容器写不到 sidecar 内部） 。

**不用** ：

- state 已经通过 `*_dump_state` 工具汇总到 rubric 可读的 state slice —— 直接用 `state_equals` 更稳 。
- 你想 assert agent 说过什么 —— 用 `text_contains` / `judge_claim` 。
- 你已经有 pytest 、junit XML 生成逻辑 —— 用 `test_result` ，不要再包一层 script_probe 。

## 常见坑

- **script 里用了 shell 特性 ，忘了单容器可用的 shell 是什么** —— sidecar 通常是 alpine / debian ，
  能不能用 `jq` / `sqlite3` 依你自己的镜像 。 排障时先手动 `docker exec ... bash` 进去试一遍 。
- **timeout 设太短** —— 15 秒对 curl 一个健康的 sidecar 没问题 ，但如果 sidecar 需要一次冷启就不够 。
- **需要检查 stderr** —— 当前只读取 stdout；请让脚本显式使用 `2>&1` 合并输出。

## 相关文档

- 门面动机与 primitive 消费方式 —— [explanation/fetcher-and-primitives.md](../explanation/fetcher-and-primitives.md) 。
- 完整字段字典 —— [`../../RUBRIC.md`](../../RUBRIC.md) §2 。
- Fetcher 类源码 —— `nexus/extensions/tasks/aidata/rubric/fetch.py` 。
