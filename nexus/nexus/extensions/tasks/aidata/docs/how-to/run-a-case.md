---
title: 单 case 手工执行
audience: 想执行已有 case 的开发者
prerequisites:
  - 完成 [tutorial/first-case.md](../tutorial/first-case.md)
related:
  - ../reference/cli.md
  - ../reference/env-variables.md
  - ./troubleshoot.md
---

# 单 case 手工执行

`aidata.py` 是单 case 的手工入口 ，`aidata_orchestrator.py` 是批量入口 。 本篇只讲前者 。

用法归纳成一条 ：

```bash
./nexus/extensions/tasks/aidata/aidata.py run <case_id> [--template <alias|path>] [-- <extra args>]
```

## 常用旋钮

### `<case_id>`

路径相对 `AIDATA_CASES_ROOT` （若未设 ，回退到 `nexus/extensions/tasks/aidata/cases/` ） 。 支持嵌套目录 ：
`work/poc/email_http_poc` 是合法的 case_id 。 CLI 通过判断该目录下是否存在 `task.yaml` （优先 ） 或
`case.yaml` （legacy ） 来识别一个 case 。

### `--template` / `-t`

选择 runtime 模板 。 决定 " 用哪种 LLM 客户端起 agent " ：

| 别名 | 展开为 | 适用模型 |
|---|---|---|
| `litellm` （默认 ） | `runtime.template.yaml` | Gemini / GPT / OpenAI 兼容网关 |
| `default` | 同上 | 同上 |
| `anthropic` | `runtime.anthropic.template.yaml` | Claude / Anthropic 系（vertex_anthropic passthrough ） |
| `vertex` | 同 anthropic | 同上 |
| `passthrough` | 同 anthropic | 同上 |
| 一个路径 | 直接使用该文件 | 自定义模板 |

选错了会导致 agent 起不来 ，或起来了但吞掉 prefix caching 命中率 。 工业模型族与模板的对应关系
在 orchestrator 里已经自动化 —— 手工 `aidata.py run` 用默认 litellm 通吃即可 ，除非你在跑 Claude 系列 。

### `-v` / `-vv`

透传给底层 `nexus run-single` ，控制 nexus 侧日志 verbose 度 。 排障时 `-v` 起步 。

### `-- <extra>`

任何 `--` 之后的 tokens 会原样拼到 `nexus run-single` 的 argv 后 。 常见用途 ：追加 span log 、 强制关掉
tracing 、临时改超时 ，都直接透传到 nexus 层 。

## 场景

### 只想跑一次冒烟 ，看看是不是回归了

```bash
AIDATA_CASES_ROOT=/data/workspace/Tasks \
  ./nexus/extensions/tasks/aidata/aidata.py run work/poc/email_http_poc
```

推荐的 4 个 demo case （覆盖不同 rubric 原语组合 ） 见 [`../../README.md`](../../README.md) §"推荐 demo case" 。

### 想跑 Claude 系模型

先在 `.env` 里把 `SEARCH_OPENAI_MODEL` 改成 Claude 变体 ，然后 ：

```bash
./nexus/extensions/tasks/aidata/aidata.py run <case> --template anthropic -v
```

anthropic 模板走 vertex_anthropic passthrough ，能拿到原生 prefix cache 命中率 。 走错模板 Claude
仍能跑 ，只是每轮都是冷启 。

### 想只做 schema 检查 ，不真跑

```bash
./nexus/extensions/tasks/aidata/aidata.py validate <case>
```

秒级返回 ，纯本地 YAML 读 + schema 断言 ，不需要 gongfeng 也不需要 LLM 网关 。 CI 建议默认跑全量
validate 。

### 想列出可用 case

```bash
./nexus/extensions/tasks/aidata/aidata.py list
```

会读 `AIDATA_CASES_ROOT` 下所有含 `task.yaml` 的目录并按 case_id 排序输出 ，同时打印 case 摘要。

### 想把一个 case 打包分发

```bash
./nexus/extensions/tasks/aidata/aidata.py package <case> ./bundles/
```

产出 `<case>.tgz` ，包含整个 case 目录（`task.yaml` + `data/` + `assets/` + `rubric.yaml` + `solution/` ） 。
拿到 tarball 只要有 aidata engine 就能跑 —— 不依赖 cases root 、不依赖仓内相对路径 。

## 向后兼容

历史上大家习惯写 `./aidata.py <case_id>` （没有 `run` 子命令） 。 CLI 保留了这条捷径 ：如果第一个
argv 不是子命令名 ，直接当成 `run <case_id>` 处理 。 新脚本请用显式的 `run` 子命令 。

## 相关文档

- 完整命令语法与所有参数 —— [reference/cli.md](../reference/cli.md) 。
- 一次运行内部 12 步的详细数据流 —— [explanation/architecture.md](../explanation/architecture.md) 。
- 常见错误 —— [troubleshoot.md](./troubleshoot.md) 。
