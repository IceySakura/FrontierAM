---
title: 新建一个 case bundle
audience: 想给项目加一个新 case 的开发者
prerequisites:
  - 完成 [tutorial/first-case.md](../tutorial/first-case.md)
related:
  - ./author-a-rubric.md
  - ../reference/file-formats.md
  - ../../RUBRIC.md
  - ../../README.md
---

# 新建一个 case bundle

case 是 aidata 的最小分发单元 。 每个 case 是一个自包含的目录 ，含 manifest 、instruction 、
context fixtures 、rubric 、可选 bootstrap 与 tests 、可选参考解 。 打成 tarball 给只有 aidata engine 的
同事也能跑 。

本篇讲怎么从零造一个 case 。 rubric 部分只讲最小闭环 ，深入见
[author-a-rubric.md](./author-a-rubric.md) 与 [`RUBRIC.md`](../../RUBRIC.md) 。

## 目录骨架

```
cases/<case_id>/
├── task.yaml                      # 必需 ：manifest
├── instruction.md                 # 必需 ：agent 可见的 prompt
├── README.md                      # 必需 ：reviewer-facing 简述
├── rubric.yaml                    # 必需 ：v4.2 DSL
├── data/                          # 可选 ：agent 可见的 context fixtures
├── assets/                        # 可选 ：bootstrap 脚本 + seed 数据（agent 不可见）
├── tests/                         # 可选 ：test_result 用 ，见 RUBRIC.md §6
├── run-test.sh                    # 可选 ：case-root 版本 ，engine 优先它
└── solution/                      # 可选 ：reviewer-facing 参考物
    ├── golden_rollout.json
    ├── NOTES.md
    └── solve.py
```

`case_id` 支持嵌套目录 ，如 `work/poc/my_new_case` ；`AIDATA_CASES_ROOT` 之下第一个含 `task.yaml` 的
目录就是一个 case 。

## 步骤

### 1. 建目录

```bash
CASE=work/poc/my_new_case
CASES_ROOT=/data/workspace/Tasks    # 或者你自己的 cases 树
mkdir -p "$CASES_ROOT/$CASE/"{data,assets,solution}
```

### 2. 写 `task.yaml`

最小可跑版本 ：

```yaml
task_id: my_new_case_v1
metadata:
  created_at: 2026-06-30
  tags: [demo]

agent_timeout: 1800
max_iterations: 180

context_files:
  - some_input.pdf
```

字段字典见 [reference/file-formats.md](../reference/file-formats.md) 与
[`../../README.md`](../../README.md) §"task.yaml schema" 。

如果 case 需要辅助服务容器（sidecar service ） 、bootstrap 脚本 、image 覆盖 、extra_mcp_servers ，
都在 task.yaml 里声明 。 不写就走默认 —— 单容器 、无 bootstrap 、默认镜像 、默认 MCP 目录 。

### 3. 写 `instruction.md`

用自然语言描述 agent 应该做什么 。 简短明确 ，避免把校验逻辑写进来（那是 rubric 的事） 。 例如 ：

```markdown
你是一名办公助理 。 请从 desktop/inputs/some_input.pdf 中读出员工姓名与到岗日期 ，
把结果通过 email_send 工具发送给 hr@example.com 。
```

### 4. 写 `README.md`

给 reviewer 看 —— 这个 case 做什么 、为什么这样设计 、baseline 分数大约是多少 。 一段几十行足够 ，
帮助其他人快速判断这个 case 覆盖了哪种能力 。

### 5. 放 context fixtures

任何 `context_files` 里声明的文件都要放到 `cases/<case_id>/data/` 下 。 这些文件会被上传到 agent 的
`AGENT_INPUTS_DIR` （即 `/workspace/desktop/inputs/`） 。 fixtures 支持任意二进制 —— PDF / XLSX /
CSV / 图片 。

### 6. 静态校验

先跑 validate 检查 schema ：

```bash
./nexus/extensions/tasks/aidata/aidata.py validate $CASE
```

它会读 `task.yaml` + `rubric.yaml` （见下） 、断言必填字段 、tags 合法 、rubric 断言原语属于
`V42_ASSERT_TYPES` 。 有问题就修 ，通过后再往下走 。

### 7. 首次跑 —— 让 engine 生成 rubric.yaml 骨架

如果 rubric 你还没写 ，第一次运行会自动生成一份 v4.2 骨架 ，带 auto-gen banner ：

```bash
./nexus/extensions/tasks/aidata/aidata.py run $CASE -v
```

跑完打开 `cases/<case_id>/rubric.yaml` ，你会看到 LLM 猜出来的一版 checks 。 这只是 starting point 。
上线前必须人工审 —— 去掉 banner ，把 check 改成真正稳定 、可复现的 assert 。 rubric 编写规则见
[author-a-rubric.md](./author-a-rubric.md) 。

如果你的 case 是 benchmark 、想上线前就把 rubric 手写好 —— 直接写好 `rubric.yaml` 即可 ，engine 会
直接加载 ，跳过 builder 。

### 8. 把 rubric 打磨到稳定

反复跑 —— 观察 `rubric_results` —— 修 rubric —— 再跑 。 目标是 ：一个正常 agent 应该 PASS 全部
critical dimensions 的 check ，一个 buggy agent 会 FAIL 明确的一条 。 若发现 " 无论 agent 做什么都 PASS"
或 " 无论 agent 做什么都 FAIL" ，说明 rubric 写歪了 —— 前者太宽后者太严 。

对每条 rubric check 建议问自己 ：这条如果 FAIL ，我作为 reviewer 能否指着 detail 说清楚 " 因为 agent 没
做 X 所以扣分 " ？ 不能 → check 描述不够精确 。

### 9. 提交前 clean-up

- `assets/` 里的临时文件删掉 ，只留 bootstrap 脚本本身与它需要的 seed 数据 。
- `solution/` 建议给 reviewer 留一份参考 rollout 或 golden patch 。 不必 100% 精确 ，是给
  reviewer 判断新 case 是否有稳定解的锚点 。
- `README.md` 与 `task.yaml::metadata` 保持一致 。

## 常见做法

### 用 sidecar service 起一个 mock 后端

见 [`../../README.md`](../../README.md) §"task.yaml schema" 里 `services:` 段 。 每个 service 是一个独立的
runtime ，engine 会在 bootstrap + MCP 加载之前把它拉起 ，把它的公网 URL 通过 placeholder 或者
`env_name` 注入给 bootstrap 与 extra_mcp_servers 。

### 用 pytest 生成 junit 、rubric 走 test_result

见 [`../../RUBRIC.md`](../../RUBRIC.md) §6 。 简言之 ：case 里放 `run-test.sh` 与 `tests/` 目录
（pytest 或任意能生成 junit XML 的框架） ；rubric 用 `assert.type: test_result` 引用 testcase
name 。 engine 在 agent 结束后会调 run-test.sh ，解析 `report.junit.xml` ，把 pass/fail 映射成 rubric 判定 。

### 覆盖默认 system prompt

见 [`../../README.md`](../../README.md) §"workspace system prompt override" 。 在 bootstrap 里把
自定义 prompt 落到 `/workspace/system_prompt.md` 就会替换默认的 preamble 。 慎用 —— 只有当
runtime 语义与默认办公流冲突时才需要 。

## 相关文档

- rubric 编写深入 —— [author-a-rubric.md](./author-a-rubric.md) 与 [`../../RUBRIC.md`](../../RUBRIC.md) 。
- YAML 契约 —— [reference/file-formats.md](../reference/file-formats.md) 。
- fetcher parse_as / script_probe 新能力 —— [use-fetcher-parse-as.md](./use-fetcher-parse-as.md) 。
