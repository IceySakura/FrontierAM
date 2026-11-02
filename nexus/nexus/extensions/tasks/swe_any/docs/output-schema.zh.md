# SWE-Any 输出 schema

逐字段说明每个阶段写到 **host** 的所有产物。单个仓库的所有 artifacts 落在
`${SWE_ANY_OUTPUT_DIR}/<repo_name>/` 下（默认 `./outputs/swe_any/<repo_name>/`）。
本文是磁盘产物合同的**权威来源**；改了 `task.py` 里的 row builder，就要同步更新本文。

题型无关的横切框架（阶段模式、软/硬失败、git 原语、漏斗指标）用中文写在
[framework.zh.md](framework.zh.md)。Feature-implementation 的算法在
[feature-implementation.zh.md](feature-implementation.zh.md)。

## 约定

- **流水线阶段（当前 feature-implementation 接线）**：`Parse → Select → Mask → Eval`。
  Parse 每仓库跑一次；Select / Mask / Eval 按仓库内的 *item*（slice / feature）逐条迭代。
  后续题型可替换各阶段算法，但应保留 [framework.zh.md](framework.zh.md) 里的布局与 outcome/status 两轴。
- **Join key** —— 每条 per-item 行都带一个整数 `index`（Select 的 slice id，磁盘上是 `sgNN/` 子目录）。
  同一个 `index` 把一条 slice 串过 `select → mask → eval`，四份 `*.jsonl` 交付物据此 join。
- **Per-item 布局**（Select / Mask / Eval 共用）—— 见 [framework.zh.md §2](framework.zh.md)：
  `sgNN/<stage>_{heuristic,agentic}.json`、仓库聚合 `*_heuristic.jsonl` / `*_agentic.jsonl`、
  以及 unified `<stage>.jsonl`（下一阶段读它）。
- **Outcome vs status** —— 两根正交轴；完整规则见 [framework.zh.md §3](framework.zh.md)。
  简述：`success = (source != "")`；软失败进 `warnings`（`completed` 仍为 true）；
  `runtime_error` 只留给意外中断（`completed=false`）。patch 只有通过累积 `git apply` 校验才会写进 `source`。
- **Status 落在 unified 行。** `completed` / `runtime_error` 由 `<stage>.jsonl` 承载（没有单独的 summary 文件）；
  半部 `sgNN/*.json` + `<stage>_{heuristic,agentic}.jsonl` 保持**纯内容**。例外：Select 的 `success` / `error`
  属于它的 *agentic 内容*（Agent 自己的判定），所以也出现在 `select_agentic.json`；Mask / Eval 的
  `success` / `error` 是 host 侧推导的，只落在 unified 行。
- **`<stage>.jsonl`（unified）表的字段顺序** —— 下一阶段 / 最终数据集会消费的字段打 ★ 并排在前
  （`index` join key、门控用的 `source`、payload 节点集 / patch）；provenance、监控、status 排后面。
- **类型**用 JSON 类型：`str`、`int`、`float`、`bool`、`null`、`list[...]`、`object`。
  `X | null` 表示该 key 总是存在但可能为 JSON `null`。

## 目录布局（单个仓库）

```
<output_dir>/<repo_name>/
  parse_graph.json                 # Parse —— 实体/关系图
  parse_seed_policy.json           # Parse —— 种子策略排除项（可能缺失）
  select_heuristic.jsonl           # Select —— 半部聚合
  select_agentic.jsonl
  select.jsonl                     # Select —— UNIFIED（Mask 读它；带 status）
  mask_heuristic.jsonl             # Mask  —— 半部聚合
  mask_agentic.jsonl
  mask.jsonl                       # Mask  —— UNIFIED（Eval 读它；带 status）
  eval_heuristic.jsonl             # Eval  —— 半部聚合
  eval_agentic.jsonl
  eval.jsonl                       # Eval  —— UNIFIED（带 status）
  issue_agentic.jsonl              # Issue —— agentic 聚合（纯 agentic：无 heuristic 半部）
  issue.jsonl                      # Issue —— UNIFIED（problem_statement；Export 读它）
  export.jsonl                     # Export —— UNIFIED（instance_id + 评分出的 F2P/P2P + harbor 路径）
  swe_any_instances.jsonl          # Export —— SWEbench 形状记录（dataset_type='swe-any'）；swe_task 经 swe/swe_any handler 读取
  sgNN/                            # 每个 item 一个目录（sg00, sg01, ...）
    select_heuristic.json  select_agentic.json
    mask_heuristic.json    mask_agentic.json    mask_heuristic.patch
    eval_heuristic.json    eval_agentic.json    eval_agentic_{mask,gold}.patch
    eval_agentic_test_runner.sh    eval_agentic_verifier.json
    issue_agentic.json                                   # Issue（纯 agentic）

# 实验性：Harbor 格式适配尚未完成、仍在测试中；请勿用于评测。
<output_dir>/harbor/                # Export —— 暂不支持消费的 Harbor 产物
  dataset.toml
  swe_any.<repo>.sgNN/             # 每个「评分有效」实例一个目录
    task.toml  instruction.md
    environment/{Dockerfile, feature_mask.patch, test_mask.patch}
    solution/{solve.sh, feature_gold.patch}
    tests/{test.sh, test_gold.patch, test_runner.sh, verify_results.py, config.json}
```

**偏离项。** Issue 是**纯 agentic**（无 `issue_heuristic.*`——没有确定性草稿可打），
所以只有 agentic 聚合 + unified `issue.jsonl`。Export 是**确定性 + 终端**（无 per-item 半部）：
每个实例写一行 unified `export.jsonl` + Harbor 目录数据集（其 per-instance 产物是 harbor 任务目录，不是 `sgNN/` JSON）。

> **警告：** SWE-Any 的 Harbor 格式适配尚未完成，仍在测试中，请勿将 `harbor/` 用于评测。
> 当前支持的入口是由 `swe-any` dataset handler 消费的 `swe_any_instances.jsonl`。

**`.json` vs `.jsonl`** —— 两者是同一份数据的不同粒度，**不是**同一文件的两种格式：

- `sgNN/<stage>_heuristic.json` / `sgNN/<stage>_agentic.json` —— 一条 **per-item** JSON 对象，
  放在该 item 的 `sgNN/` 子目录，item 一跑完就写（流式；崩溃安全 / 可独立恢复）。
- `<stage>_heuristic.jsonl` / `<stage>_agentic.jsonl` —— 仓库根的**仓库级聚合**，一行一个 JSON 对象，
  最后一次性写。`.jsonl` 的第 `N` 行就是 `sg{N}/<stage>_heuristic.json` 的内容（同一 schema）。

所以下面每个 schema 只记一次，同时适用于 `sgNN/*.json` 文件和它 `.jsonl` 聚合的一行。
`.patch` 兄弟文件是 Agent 按路径读取的原始草稿 diff。

---

# Parse

每仓库单发（无 `sgNN/`、无 heuristic/agent 分轨）。确定性步骤 = SCIP 索引 → 实体图；agentic 步骤 = seed policy。

## `parse_graph.json`

实体/关系图（无 protobuf 的 JSON）。实体 = 一个 `function` 或 `class`（在其 SCIP 定义点）；
外部符号（被引用但仓库内未定义）以最小节点出现。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `info` | object | 图级元信息（见下）。 |
| `nodes` | list[object] | 实体节点（见下）。 |
| `edges` | list[object] | 关系边（见下）。 |

**`info`**（object）

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `project_root` | str | SCIP `metadata.project_root`（仓库根 URI）。 |
| `tool` | str | 索引器名 + 版本，如 `scip-python 0.x`。 |

**`nodes[]`**（object）

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `id` | str | SCIP symbol 字符串；节点唯一 id（Agent 选的就是这些 id，它们会成为 `agentic_nodes[].id` / 边端点）。 |
| `info` | object | 节点属性（见下）。 |

**`nodes[].info`**（object）—— 内部节点带全部 key；外部节点只带 `external: true`。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `kind` | str | `"function"` 或 `"class"`。 |
| `name` | str | 展示名（SCIP 最后一个 descriptor），如 `speak`。 |
| `module` | str | 点分模块/命名空间，如 `zoo.animals`。 |
| `file` | str | 仓库相对源码路径。 |
| `owner` | str \| null | 直接 owner（外层 class/function）的 SCIP symbol，否则 `null`。 |
| `def_line` | int \| null | 定义的 0-based 行号；未找到定义点时为 `null`。 |
| `enclosing_range` | list[int] \| null | 完整定义 span 的 `[start_line, start_col, end_line, end_col]`（0-based）；未知为 `null`。 |
| `is_test` | bool | 是否测试代码（严格按测试文件路径重新标注）。 |
| `test_evidence` | list[str] | 标注（或不标注）的依据：`"scip_role"` / `"path"` / `"name"` 的任意组合。 |
| `documentation` | str | Docstring / hover 文本（可能为空）。 |
| `external` | bool | 仓库内定义为 `false`；外部节点只带 `external: true`。 |

**`edges[]`**（object）

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `src` | str | 源节点 id。 |
| `dst` | str | 目标节点 id。 |
| `infos` | list[object] | 该 `(src, dst)` 上的并行关系事实（一条 DiGraph 边可承载多个）。 |

**`edges[].infos[]`**（object）

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `type` | str | `"enclosing"`（父→子）、`"implementation"`（impl→interface）或 `"reference"`（caller→callee）。 |
| `dst_external` | bool | `dst` 是否外部符号。 |
| `evidence` | str | 边的推导方式：`"descriptor"` / `"is_implementation"` / `"occurrence"`。 |

## `parse_seed_policy.json`

由 seed-policy agent 写（Select 读它来塑形种子池）。seed-policy pass 失败时缺失
（此时 `ParseTaskOutput.seed_policy_error` 说明原因）。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `exclude_globs` | list[str] | 禁止作为 slice *种子起点* 的仓库相对 glob（生成/vendored/build 代码）。用 `fnmatch` 匹配（故 `*` 也匹配 `/`）。 |
| `exclude_reasons` | object (str→str) | 每个 glob 一行理由。 |
| `notes` | str | 真正手写的产品逻辑在哪（自由文本）。 |

---

# Select

一个 item = 围绕一个采样 seed 的一条 BFS slice。确定性步骤 = 分层 BFS slice；
agentic 步骤 = Claude Code pass（这条 slice 是否构成一个连贯 feature？）。

## `sgNN/select_heuristic.json`（每 slice）+ `select_heuristic.jsonl`（仓库聚合）

BFS slice。heuristic 步骤失败时行降级为 `{index, seed, meta: {}, heuristic_nodes: []}`。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `index` | int | Item id（sgNN）。 |
| `seed` | str | slice 生长的起点节点 id。 |
| `meta` | object | slice 旋钮 + 计数（见下）。 |
| `heuristic_nodes` | list[object] | 选中的 slice 节点（形状见下；与 agentic 的 `agentic_nodes` 共用）。 |

**`meta`**（object）

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `max_bfs_nodes` | int | 用到的单 slice 节点上限。 |
| `drop_bfs_prob` | float | 用到的每层 Bernoulli 丢弃概率。 |
| `min_bfs_per_depth` | int | 每层 BFS 保留下限。 |
| `max_bfs_per_depth` | int | 每层 BFS 保留上限。 |
| `include_tests` | bool | 测试节点是否可入选。 |
| `include_external` | bool | 外部节点是否可入选。 |
| `rng_seed` | int \| null | 单 slice RNG 种子（`null` = 未播种抽样）。 |
| `component_size` | int | seed 所在连通分量大小。 |
| `node_count` | int | `len(heuristic_nodes)`。 |
| `file_count` | int | `heuristic_nodes` 跨越的不同文件数。 |

**`heuristic_nodes[]` / `agentic_nodes[]`**（object —— 一套共享节点形状，由 `node_entry` 构建）

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `id` | str | 节点 id（SCIP symbol）。 |
| `name` | str | 人类可读的限定名。 |
| `kind` | str \| null | `"function"` / `"class"`。 |
| `module` | str \| null | 点分模块路径。 |
| `file` | str | 仓库相对路径（可能为空）。 |
| `def_line` | int \| null | 0-based 定义行。 |
| `is_test` | bool | 是否测试代码。 |
| `documentation` | str | Docstring（可能为空）。 |

## `sgNN/select_agentic.json`（每 slice）+ `select_agentic.jsonl`（仓库聚合）

**只有** Agent 的最终输出 —— 原样写入。执行 status（`completed` / `runtime_error`）**不**在这里，
它在 unified `select.jsonl` 行上。`seed` 留在 heuristic 侧。Agent 只吐 id；host 把它们 enrich 成 `agentic_nodes` 对象。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `index` | int | Item id。 |
| `feature` | str | 解析出的 feature 的一行描述；非 `success` 时为空。 |
| `agentic_nodes` | list[object] | 解析出的 feature 的函数节点（形状同 `heuristic_nodes`，各带 `id`——**Mask 消费的 payload**）；非 `success` 时为空。 |
| `cover_rate` | float | `agentic_nodes` 中来自 heuristic slice 的比例（`final_in_slice / len(agentic_nodes)`；空时 `0.0`）。slice/feature 重叠度。 |
| `source` | str | Provenance：Agent 解析出连贯 feature 时为 `"agentic"`，否则 `""`（Select 无 heuristic 兜底，故永不为 `"heuristic"`）。 |
| `success` | bool | 业务 outcome：Agent 解析出一个连贯 feature（`== (source == "agentic")`）。 |
| `error` | str \| null | 没解析出的原因（业务，非 runtime）：`"abandoned: <reason>"` 或 `"schema: <check>"`；成功时 `null`。 |

## `select.jsonl`（unified —— Mask 读它）

精简 per-slice 记录：Agent 结果 + status，按 `index` join。heuristic 细节（`seed` / `meta` /
`heuristic_nodes`）与 `cover_rate` **不**内联——它们留在 `select_{heuristic,agentic}.jsonl`，按 `index`
回查即可。★ = Mask 消费的字段。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| ★ `index` | int | Item id —— Mask 据此 join。 |
| ★ `source` | str | Agent 解析出 feature 时为 `"agentic"`（**Mask 只挖这些**），否则 `""`。 |
| ★ `nodes` | list[object] | 解析出的 feature 的函数节点（agentic pass 的 `agentic_nodes`）——**Mask 消费的 payload**（它取其中的 id）。节点形状同 `select_heuristic.json`；非 `success` 时为空。 |
| `feature` | str | 一行 feature 描述（人类上下文；未解析出时为空）。 |
| `success` | bool | 交付 outcome：解析出了 feature（`== (source == "agentic")`）。 |
| `error` | str \| null | 没解析出的业务原因：`"abandoned: …"` / `"schema: …"`；Agent 反而崩溃时为 `null`（那算 `warning`），成功时也为 `null`。 |
| `warnings` | list[str] | 软诊断 —— 尤其是 Agent invoke/解析/下载失败（崩溃是软的，不算 `runtime_error`）。 |
| `completed` | bool | **Status**：本 slice 无意外中断。 |
| `runtime_error` | str \| null | **Status**：中断详情 —— heuristic 子图构建崩溃或未处理异常；软 Agent 失败时为 `null`（那算 `warning`）。 |

---

# Mask

一个 item = 一个解析出的 Select feature。确定性步骤（`run_mask`）= 保留 entry 签名、删 helper
→ 草稿 mask/gold patch；Agent 步骤 = 精修该草稿。

## `sgNN/mask_heuristic.json`（每 feature）+ `mask_heuristic.jsonl`（仓库聚合）

`run_mask` 草稿结果（经 invoke 返回，不下载）。run 失败时行降级为 `{index, feature}`。
唯一写到沙箱的文件是原始草稿 `.patch`，供 refine agent 按路径读；不归档。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `index` | int | Item id。 |
| `heuristic_mask_plan` | object | feature 分类 + 触及文件（见下）。 |
| `heuristic_mask_patch` | str | **草稿** unified diff：原仓库 → 挖空（stub entry、删 helper）。 |
| `heuristic_gold_patch` | str | 草稿 `heuristic_mask_patch` 的精确逆（挖空 → 原始）。 |
| `warnings` | list[str] | 草稿构建诊断 —— 分类 + apply-mask 警告合并（如无 entry；某节点因缺 span 被跳过）。 |
| `execution_time` | float | 沙箱内构建耗时（秒）。 |

**`heuristic_mask_plan`**（object）

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `nodes` | list[str] | feature 的可挖函数 id（仅 function，排除测试）。 |
| `entries` | list[str] | 调用图发现被 feature 外部调用的函数 → 草稿留签名、挖空 body。 |
| `deletes` | list[str] | 调用图未发现外部 caller 的函数 → 草稿整删；refine Agent 可从中保留重要接口。 |
| `classes` | list[str] | feature 里的 class id（不挖；仅上下文）。 |
| `files` | list[str] | 草稿触及的仓库相对文件。 |

## `sgNN/mask_agentic.json`（每 feature）+ `mask_agentic.jsonl`（仓库聚合）

refine agent 的输出（`use_agent=false` / 无草稿 / Agent 失败时为空）。调用图 boundary entries 是必须保留的安全下限，
但 Agent 可额外保留对 feature 重要的 caller-facing 签名；直接输出仍只有一份 `mask_patch`。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `index` | int | Item id。 |
| `cover_rate` | float | refine 触及文件中落在 feature 掩码文件内的比例（`1.0` = 完全在范围内；无 refine 时 `0.0`）。监控信号，非过滤 —— 越界文件名记进 `warnings`。 |
| `agentic_mask_patch` | str | refine 的 mask diff（可能为空）。`agentic_` 标记它是 Agent 原始输出 —— 区别于 unified `mask_patch`（*选定*的 patch，可能反而是草稿）。 |
| `agentic_gold_patch` | str | `agentic_mask_patch` 的 git 计算逆（沙箱内 `git apply` + `git diff -R`）；refine 不能 apply 时为空（此时 agentic 候选不被选用）。 |

## `mask.jsonl`（unified —— Eval 读它）

精简 per-feature 记录：选定的 feature mask/gold patch（refine 能 apply 时用它，否则用能 apply 的
heuristic 草稿）+ outcome + status。`feature_` 前缀区别于 Eval 的 `test_` patch。`deletes`（heuristic plan）
与 `cover_rate`（`mask_agentic.json`）**不**内联 —— 按 `index` 回查。一对 patch 只有通过**累积** apply-check
才提交（mask 打到 pristine、gold 再叠到挖空树上 —— 原始 → 挖空 → 原始的往返），故无可用 mask（`success=false`）
的行的 `feature_mask_patch` / `feature_gold_patch` 为空（原始候选在 `mask_heuristic` / `mask_agentic`）。★ = 数据集 payload / Eval 消费的字段。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| ★ `index` | int | Item id —— Eval 据此 join。 |
| ★ `nodes` | list[str] | feature 函数 id（Select `nodes` 的那些 `id`——这里缩成 id，非对象）；Eval 读它。 |
| ★ `feature_mask_patch` | str | **选定**的 feature mask diff（原始 → 挖空；**问题态** —— 数据集 payload）。 |
| ★ `feature_gold_patch` | str | 精确逆（挖空 → 原始；**参考解** —— 数据集 payload）。 |
| ★ `files` | list[str] | **选定**的 `feature_mask_patch` 实际触及的文件（真实掩码面 —— 含 agentic refine 越到的文件）。无可用 mask 提交时为空（`success=false`）。 |
| `source` | str | 选定 patch 的 provenance：`"agentic"`（用了 refine）、`"heuristic"`（回落草稿）或 `""`（无可用 mask）。 |
| `success` | bool | 交付 outcome `= source != ""`：存在可用、gold-ready 的 mask（只有 mask + gold 往返通过累积 apply-check 才提交）。与 status 解耦 —— refine agent 失败但 heuristic mask 仍算数。 |
| `error` | str \| null | 不可用的业务原因：`"no maskable functions in feature"` 或 `"no applicable mask patch …"`；成功时 `null`。 |
| `warnings` | list[str] | 软诊断 —— 含 refine agent 失败（软：heuristic 仍在）与越界 refine 文件。 |
| `completed` | bool | **Status**：本 feature 无意外中断。 |
| `runtime_error` | str \| null | **Status**：中断 —— `run_mask` 草稿 invoke 崩溃、apply-check 基础设施故障或未处理异常；软 refine 失败时为 `null`（那算 `warning`）。 |

---

# Eval

一个 item = 一个挖空的 feature。确定性步骤（`run_eval`）= 对**全部**候选原生测试构建删除草稿（HINT + 兜底）；
Agent 步骤 = 产出**两个非互逆**的 patch 和一个 grade-only runner：`test_mask_patch` 删除仓库中**所有**测到该 feature 的现存测试
（对 solver 隐藏 oracle），`test_gold_patch` 是**正向、泛化**的评分 oracle（覆盖 feature patch 中每个实质性、
外部可观察能力的黑盒测试 + 可选的与 feature 无关的 P2P 守卫）；`test_runner.sh` 将仓库真实测试框架/环境适配为结构化
per-test JSONL。语义约束（scope、覆盖、F2P≠∅、无 P2F）在 prompt 层由 Agent 自校验，
host 只做 apply-check。Eval 写*测试侧*；feature mask/gold 在 `mask.jsonl`（按 `index` join）。

## `sgNN/eval_heuristic.json`（每 feature）+ `eval_heuristic.jsonl`（仓库聚合）

`run_eval` 草稿结果（经 invoke 返回，不下载）。run 失败时行降级为 `{index}`，但不会阻断后续
author/refine agent；草稿只作为提示和诊断，绝不会直接进入 unified 结果。
`run_eval` 会在沙箱里写**两份**原始草稿 `.patch`（`eval_heuristic_mask.patch` = 删全部候选、
`eval_heuristic_gold.patch` = 原样恢复），供 Eval agent 按路径读；这两份不在此归档。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `index` | int | Item id。 |
| `heuristic_tests` | list[str] | 草稿删除的候选原生测试 id（全部）。 |
| `heuristic_mask_patch` | str | 删除**全部**候选原生测试的**草稿** diff。 |
| `heuristic_gold_patch` | str | 草稿的精确逆（原样恢复全部候选），仅供 Eval agent 参考。 |
| `heuristic_test_files` | list[str] | 草稿触及的测试文件。 |
| `warnings` | list[str] | 构建诊断。 |
| `execution_time` | float | 沙箱内构建耗时（秒）。 |

## `sgNN/eval_agentic.json`（每 feature）+ `eval_agentic.jsonl`（仓库聚合）

shaped 的 test-oracle 结果（必需的作者 Agent 失败时为空）。author/refine agent 写两个 diff
和 `eval_agentic_test_runner.sh`；apply-check 后，必需的独立 verifier agent 写
`eval_agentic_verifier.json`。host 下载并严格校验这些产物；任一 Agent 缺失或失败都会拒绝
该 item，不会提升 heuristic 草稿。`agentic_verifier` 记录语义质量门禁结果。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| `index` | int | Item id。 |
| `agentic_mask_patch` | str | 隐藏 Agent 找到的所有 feature 测试的删除 diff（extra-only oracle 时可能为空）。 |
| `agentic_gold_patch` | str | 正向 diff = 泛化的评分测试 + 可选 P2P 守卫（**不是** mask 的逆）。 |
| `agentic_runner_script` | str | 执行 oracle 并写结构化 JSONL 协议的 grade-only shell adapter。 |
| `agentic_test_files` | list[str] | 从 `patch_files(agentic_gold_patch)` 得到的候选评分目标；不包含 mask-only 路径。 |
| `agentic_verifier` | object \| null | 完整的独立 `verdict` / `rationale` / `findings` 对象；未完成验证时为 null。 |

verdict 编号稳定：`0` 无错误；`1` oracle 泄漏；`2` 越界/破坏性修改；`3` 过度约束；
`4` 测试过弱；`5` 易被投机；`6` runner 完整性；`7` 非 hermetic/flaky；`-1` 其它实质问题。
只有 `0` 通过。非零时 `findings` 给出具体证据，并记录全部发现的问题。

## `eval.jsonl`（unified）

每 feature 选定的测试侧 oracle（与 `mask.jsonl` 按 `index` join）。只有 agentic pair 带非空结构化 runner，
按评测时序（test_mask → test_gold，逐个叠加）通过**累积** apply-check，且独立 verifier 接受时，才提交进
`test_mask_patch` / `test_gold_patch` / `source`。`test_mask_patch` 与
`test_gold_patch` **互不为逆**：mask 隐藏现存 feature 测试，gold 是正向、泛化的评分 oracle。agentic 行由同一份
`test_runner_script` 执行 pregold/postgold 并产出稳定 opaque test id。Export 据此推导 FAIL_TO_PASS / PASS_TO_PASS
（掩码态 FAIL / gold 态 PASS，出现 P2F 则拒收）。feature nodes 不内联 —— 按 `index` 回查 `mask.jsonl`。
★ = 评分/导出步骤消费的数据集 payload。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| ★ `index` | int | Item id —— join 到 `mask.jsonl`。 |
| ★ `test_mask_patch` | str | **选定**的删除仓库中**所有**测到该 feature 的现存测试的 diff（**对 solver 隐藏 oracle**；extra-only oracle 时可能为空）。 |
| ★ `test_gold_patch` | str | **选定**的**正向**评分 oracle —— 泛化 / 新写的黑盒行为测试，覆盖 feature mask 移除的每个实质性外部可观察能力 + 可选的与 feature 无关 P2P 守卫。**不是** `test_mask_patch` 的逆。 |
| ★ `test_runner_script` | str | 必需的 grade-only 结构化 runner；缺失时 Eval 门禁失败；solver 不可见。 |
| `verifier` | object \| null | 完整保存在最终 `eval.jsonl` 的 verifier 对象：整数 `verdict`、字符串 `rationale`、字符串列表 `findings`。 |
| ★ `test_files` | list[str] | 从 `patch_files(test_gold_patch)` 得到的候选评分目标。Export 应用 patch 后过滤不存在的路径，再运行剩余测试推 F2P/P2P。 |
| `source` | str | Provenance：有可用 runner-backed oracle 时为 `"agentic"`，否则为 `""`。 |
| `success` | bool | 仅当 patch pair 可 apply、runner 非空且独立 verifier 接受时为 true。 |
| `error` | str \| null | 不可用的业务原因，包括 verifier 拒绝；成功时 `null`。 |
| `warnings` | list[str] | 软诊断，包括作者/verifier 调用或产物失败；这些失败都会拒绝该 oracle。 |
| `completed` | bool | **Status**：本 feature 无意外中断。 |
| `runtime_error` | str \| null | **Status**：中断 —— apply-check 基础设施故障或未处理异常。草稿构建和 Agent 失败是软 `warning`，但没有完整 refine + verify 结果仍然不能通过。 |

---

## `issue.jsonl`（unified）

每个「已挖空 + 已配 oracle」的 feature 一条问题描述（按 `index` join Mask `success` ⋈ Eval `source`）。
纯 agentic：Agent 产出非平凡描述时 `source` 为 `"agentic"`，否则 `""`。它被展示的参考 patch **不**内联 ——
按 `index` join Mask/Eval。★ = Export 读取的字段。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| ★ `index` | int | Item id —— join 到 `mask.jsonl` / `eval.jsonl`。 |
| ★ `problem_statement` | str | GitHub-issue 风格的题面文本（不泄漏）；非 `success` 时为空。 |
| `source` | str | 解析出时为 `"agentic"`，否则 `""`。 |
| `success` | bool | 交付 outcome `= source != ""` —— 非平凡问题描述（≥ 40 字符）。 |
| `error` | str \| null | 没产出的业务原因（如 `"problem_statement too short …"`）；成功时 `null`。 |
| `warnings` | list[str] | 软诊断 —— 含 issue agent 失败（invoke/解析/下载）。 |
| `completed` | bool | **Status**：无意外中断。 |
| `runtime_error` | str \| null | **Status**：中断（挖空态搭建 / 未处理），否则 `null`。 |

## `export.jsonl`（unified）

Export 处理的每个 feature 一行（按 `index` join Mask ⋈ Eval ⋈ Issue，以 issue `success` 门控）。
`success` = 该实例评分有效并写入了 Harbor。任务本体在 `<output_dir>/harbor/<instance_id>/`（见上文布局）。

| 字段 | 类型 | 含义 |
|-------|------|---------|
| ★ `index` | int | Item id。 |
| ★ `instance_id` | str | `swe_any.<repo>.sgNN` —— Harbor 任务名。 |
| ★ `success` | bool | 评分有效：`fail_to_pass` 非空且无 `PASS_TO_FAIL` 回归；已写 Harbor 目录。 |
| ★ `fail_to_pass` | list[str] | pre-gold FAIL、post-gold PASS 的测试（门控修复的 oracle）。 |
| `pass_to_pass` | list[str] | 两侧都 PASS 的测试（护栏）。 |
| `harbor_path` | str \| null | 产出的 `harbor/<instance_id>/` 目录（有效时），否则 `null`。 |
| `error` | str \| null | 未导出的可读原因。 |
| `error_kind` | str \| null | 机器可读分类，包括 runner 自报类别，以及 `runner_protocol_error`、`runner_exit_error`、`target_missing`、`no_postgold_pass`、`oracle_nongating`、`gold_regression`、`grade_setup_failed`。 |
| `warnings` | list[str] | 软诊断，包括在仍有真实目标时被忽略的、不存在的 test-gold 路径。 |
| `completed` | bool | **Status**：无意外中断。 |
| `runtime_error` | str \| null | **Status**：grade-invoke / 未处理中断，否则 `null`。 |

oracle 是**文件级**的：`test_files`（来自 `eval.jsonl`）整体跑，故 `fail_to_pass` / `pass_to_pass`
枚举的是这些文件里的全部测试 —— 恢复的原生测试与 Agent 新增测试**同等**（见 [framework.zh.md](framework.zh.md) §4b）。

---

# Per-run 任务输出（非文件）

每个阶段的 `run-batch` 还会为每仓库返回一个 `SweAny<Stage>Output` 对象（呈现给 batch DB / export，
不写在仓库目录下）。这些是仓库级聚合 —— `score`，外加如 Select 的 `num_subgraphs` / `num_agentic` /
`num_abandoned` / `cover_rate_mean`，Mask 的 `num_features` / `num_applicable` / `num_agentic` /
`num_agent_failed` / `cover_rate_mean`，Eval 的 `num_features` / `num_applicable` / `num_agentic` /
`num_agent_failed` / `num_verifier_rejected`。对 Eval，`num_applicable` 表示作者、apply-check 和
verifier 全部通过；`num_agent_failed` 统计作者/verifier 调用或产物失败，`num_verifier_rejected`
统计 verifier 正常完成但给出负面结论。它们都用于观测，score 仍随 `num_applicable > 0`。输出另含 unified 交付物的
host 路径。Issue 返回 `num_features` / `num_applicable` / `num_agent_failed`；Export 返回
`num_features` / `num_exported`（评分有效、写入 Harbor）+ `harbor_dir`。确切字段见 `task.py` 里的 `SweAny*Output` 模型。
