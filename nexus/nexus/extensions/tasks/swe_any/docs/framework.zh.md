# SWE-Any 框架约定

题型无关的横切约定：产物布局、heuristic / agentic 双轨、outcome / status、git 原语、漏斗指标。  
当前落地的题型是 **feature implementation**（算法见 [feature-implementation.md](feature-implementation.zh.md)）；后续 **bug fix** 等题型复用同一套框架，换的是各 stage 的启发式 / prompt / 门控，而不是目录与状态语义。

字段级 on-disk 合同见 [output-schema.md](output-schema.zh.md)。运行入口与配置见 [README.md](../README.md)。

---

## 1. 阶段模式

每个「造题」阶段通常是：

1. **Heuristic（确定性）** — 便宜、可复现，产出草稿结构或草稿 patch。
2. **Agentic（Claude Code）** — 在草稿上做判断 / 精修；失败默认是**软失败**（见 §3）。
3. **Choose + 校验** — 优先采用通过校验的 agentic 产物，否则回落 heuristic（若该 stage 有兜底）；写入 unified 交付物。

Parse 略特殊：图构建是启发式；seed policy 是 Agent，失败不挡图。Select 当前无 heuristic 兜底（只有 Agent 解析出的 feature 往下流）——这是**题型策略**，不是框架限制；bug fix 的 Select 可以另定规则。

阶段之间只通过 host 上的 **unified `<stage>.jsonl`** 衔接，用整数 **`index`**（磁盘上的 `sgNN/`）join。

### 1.1 当前 Pipeline 的定位

当前 `pipeline/` 实现是为了消除全局 stage barrier、支持 **feature-level parallelism**
而落地的专用编排，不是最终的通用 Pipeline 抽象。它暂时直接编码了 SWE-Any 的 stage 顺序、
字段映射、artifact 路径和失败传播规则，使同一仓库的各个 feature 可以独立向后推进。

后续会继续把这些衔接逻辑抽象为可复用的 stage handoff / execution protocol，让新的题型或不同
Pipeline 组合不需要重复实现整套编排逻辑。现有 `pre_feature → feature → post_feature` 应视为该
抽象之前的 feature 并行执行方案。

---

## 2. 产物布局

对每个仓库，artifacts 落在 `${SWE_ANY_OUTPUT_DIR}/<repo_name>/`。

### 2.1 仓库级共享（Parse）

Parse 破例：一次产出图，不是 per-item 双轨。

| 文件 | 角色 |
|------|------|
| `parse_graph.json` | 实体 / 关系图（下游共用基板） |
| `parse_seed_policy.json` | 可选；种子排除 globs |

### 2.2 Per-item 双轨（Select / Mask / Eval / …）

```
sgNN/<stage>_heuristic.json     # 单条启发式
sgNN/<stage>_agentic.json       # 单条 Agent
<stage>_heuristic.jsonl         # 仓库聚合（一行 = 一条 sgNN heuristic）
<stage>_agentic.jsonl
<stage>.jsonl                   # ★ UNIFIED：下一阶段只读这份
```

- **`.json` vs `.jsonl`**：同一内容的两种粒度，不是同文件两种格式；`sgNN/` 保证无聚合文件时仍可按条恢复。
- **Unified**：合并「选用结果 + 必要 payload + status」，不把 heuristic / agentic 全量再内联一遍；细节用 `index` 回查半部文件。
- 字段表、★ 下游消费字段：见 [output-schema.md](output-schema.zh.md)。

Sandbox 工件根目录约定为 `/workspace/swe_any`；烘焙仓库工作目录默认 `/testbed`。

---

## 3. Outcome 与 Status（两根正交轴）

每条 unified 行（以及部分 agentic 半部）共用这套语义：

| 字段 | 轴 | 含义 |
|------|----|------|
| `source` | Outcome | `"agentic"` / `"heuristic"` / `""` — 最终采用谁的产物 |
| `success` | Outcome | `= (source != "")` — 有可用交付物 |
| `error` | Outcome | 业务性「没产出」原因（abandoned / schema / no applicable …） |
| `warnings` | 诊断 | 软失败列表；**不影响** `success` |
| `completed` | Status | 无**意外中断** |
| `runtime_error` | Status | 意外中断详情 |

**软失败 → `warnings`**（`completed` 仍为 true）：

- Agent invoke / 解析 / 下载失败
- 候选 patch 未通过 `git apply`（换候选或最终无交付物）
- 越界 / drift 丢弃、回落 heuristic 的提示

**意外中断 → `runtime_error`**（`completed=false`）：

- 草稿构建的 `runtime.invoke` 崩溃（如 `run_mask` / `run_eval`）
- 累积 apply-check 的**基础设施**异常（通信失败等；fail-closed，不当成「校验通过」）
- 未捕获的 per-item 异常
- 题型里定义为硬失败的确定性步骤崩溃（如 Select 的 BFS 构建失败）

约定：`runtime_error` 与 `error` 不描述同一事件；有中断时通常来不及提交交付物，故很少出现 `success=true ∧ completed=false`。`success=true` 的行可以带 `warnings`（例如 Agent 挂了但 heuristic 兜住了）。

半部文件（`*_heuristic` / `*_agentic`）保持**纯内容**；`completed` / `runtime_error`（及多数 stage 的 `success` / `error`）落在 unified。Select 例外：`success` / `error` 也是 agentic 内容的一部分。

---

## 4. Git 原语（sandbox）

Host 经 `runtime.invoke` 调用（`utils/repo_git.py` → `runtime_setup.py`）：

底层由三个可组合的小原语拼成（`runtime_setup.py`）：`_reset`（reset --hard + clean -fd）、`_apply_patch`（单个 `git apply [--check]`）、`_apply_steps`（顺序叠加，`reset_before` / `reset_after` 控制 reset 书档）。对外的 target：

| 原语 | 行为 |
|------|------|
| `reset_repo` | `git reset --hard` + `git clean -fd`（保留 ignored）；每条 item 开始前 best-effort 调用，防 Agent 污染 |
| `apply_check_sequence` | 顺序叠加 `git apply`（reset_before + reset_after）**校验**能否 stack，末尾 reset**不留状态**；返回逐步 `{ok, empty, skipped, error}`；空 patch 为 no-op；空 results 视为失败 |
| `apply_patches`（`run_apply`） | 先回 pristine，再顺序 `git apply` 并**持久保留**（无尾随 reset）；Issue 用它搭"全挖空态"（apply feature_mask + test_mask，让 agent 浏览该树）。Export 不用它——评分走 `grade_instance` |
| `reverse_patch` | pristine → apply 正向 patch → `git add -A` → `git diff --cached -R` → reset；用于 **Agent 手写** mask 的规范逆 patch |
| `grade_instance`（`run_grade`） | 执行式评分：pregold / postgold 各 reset → apply steps → 执行必需的 agent-authored `test_runner.sh` 并校验结构化 JSONL → reset；无 runner 的行直接拒绝 |

启发式侧若能对「改前 / 改后」真实源码各做一次 diff（如 `build_edit_patch`），则逆 patch 不必再走 git。

校验策略（题型可选用）：**只把通过累积 apply-check 的候选写进 `source` / 交付 patch**；基础设施异常 fail-closed。

---

## 4a. Eval 独立 Agentic Verifier

Eval 的作者 Agent 产出 test mask/gold/runner 后，第二个独立 Agent 阅读完整 feature 与测试产物，
按稳定 taxonomy 输出 `{"verdict", "rationale", "findings"}`：`0` 表示 clean，`1..7`
分别表示 oracle 泄漏、越界/破坏、过度约束、测试过弱、易投机、runner 完整性、非 hermetic/flaky，
`-1` 表示其它实质问题。host 只接受 `verdict == 0`；verifier 调用失败、产物非法或任意非零
verdict 都 fail-closed。完整类别说明见 feature implementation 文档 §4.4。

---

## 4b. 执行式评分与 Harbor 导出（Export，题型无关）

「组装 → 执行评分 → 导出」这套机器是题型无关的（bug-fix 也复用），只是喂进去的 patch 不同：

- **评分（`grade_instance` → `stage/export`）**：给定「bug 态 patch」「gold patch」「露出测试 patch」和 `test_files`，在 baked 镜像里跑两遍 —— **pregold**（bug 态：实现挖空、测试露出 → 应 FAIL）与 **postgold**（叠 gold → 应 PASS）。同一份必需的 agent-authored `test_runner.sh` 为两侧输出结构化 `{test_id: status}`，据此归类 **FAIL_TO_PASS** / **PASS_TO_PASS** / **PASS_TO_FAIL**（回归）。**有效性门禁**：`FAIL_TO_PASS` 非空且无 `PASS_TO_FAIL`。

- **Harbor 导出（`stage/export/harbor.py`）**：Harbor 是**一实例一目录**的数据集（`harbor_bridge` 只消费不导出），Export 自己写 `<output_dir>/harbor/<instance_id>/`：`task.toml`（`[task]`/`[environment].docker_image,workdir`/`[agent]`/`[verifier]`）、`instruction.md`（problem_statement）、`environment/Dockerfile`（`FROM image` 并 `git apply` bug patch 烘进镜像）、`solution/{solve.sh, feature_gold.patch}`、`tests/{test.sh, test_gold.patch, config.json}`（reward 写到 `/logs/verifier/reward.txt`），外加根部一份 `dataset.toml`。patch 以 `*.patch` 兄弟文件落盘、由脚本 `git apply`（比把多行 diff 塞进 shell/Docker 稳）。

  **警告：SWE-Any 的 Harbor 格式适配尚未完成、仍在测试中，请勿使用。** 当前支持的评测入口是 `swe_any_instances.jsonl` + `swe-any` dataset handler。

## 4c. 纯 agentic 阶段（Issue，一处偏离）

多数 stage 是 heuristic + agentic 双轨；**Issue 是纯 agentic**（无 heuristic 半部、无 `sgNN/*_heuristic.json`），因为「写问题描述」没有确定性草稿可打。它照样遵守 §3 的 outcome/status 轴：`source='agentic'`（`success = source != ''`）当产出非平凡 `problem_statement`，否则空行 + 业务 `error`；agent 崩溃 → `warnings`、`completed` 保持 true（与 Select 一致）。

---

## 5. 漏斗指标

Mask / Eval 一类「有输入集合、有可用输出」的 stage，任务级输出对齐：

| 字段 | 含义 |
|------|------|
| `num_features`（或 Select 的 `num_subgraphs`） | 进入本 stage 的条目数 |
| `num_applicable` | `success=true`（有可用交付物）的条目数 |
| `num_agentic` | 其中最终 `source=='agentic'` 的条数 |
| `num_agent_failed`（Mask / Eval） | Agent 调用/产物软失败条数；Eval 同时统计作者与 verifier 失败 |
| `num_verifier_rejected`（Eval） | verifier 正常完成但判定会破坏已有功能或测试过弱的条数 |

关系：`num_features ≥ num_applicable ≥ num_agentic`。Construction score 通常为「任一 applicable > 0 → 1.0」——因此全员 Agent 挂掉但 heuristic 兜住时 score 仍可为 1.0，靠 `num_agent_failed` 发现。

---

## 6. 共享工具（题型可复用）

| 模块 | 职责 |
|------|------|
| `utils/graph/` | 轻量图模型 / JSON IO（无 SCIP 依赖） |
| `utils/scip/` | 索引与 protobuf |
| `utils/source_edit.py` | 按 `enclosing_range` stub/delete → unified diff；`patch_files` |
| `utils/classify_tests.py` | 测试路径 / `test_*` 判定 |
| `utils/agent_output.py` | 解析 Agent 写出的 JSON |
| `utils/sandbox_files.py` | Host ↔ sandbox 文本传输 |
| `utils/repo_git.py` | 上述 git 原语的 host 封装 |
| `utils/instance.py` | SWE-Env instance 水合与路径 |

题型差异应尽量落在 `stage/<name>/heuristics.py` + `prompts.py` + 门控函数，而不是另起一套产物布局或状态字段。

---

## 7. 文档怎么读

| 文档 | 内容 |
|------|------|
| **本文** | 框架：布局、状态轴、git、漏斗 |
| [output-schema.md](output-schema.zh.md) | 各文件字段类型与含义（on-disk 合同） |
| [feature-implementation.md](feature-implementation.zh.md) | Feature implementation 题型的 **算法**（Parse→Select→Mask→Eval） |
| [README.md](../README.md) | 模块地图、配置、如何跑 |
