# Feature Implementation：算法说明

本文只写 **feature implementation** 题型在 Parse → Select → Mask → Eval → Issue → Export 上的**算法与门控**：如何从图里抽出 feature、如何挖空实现、如何构造测试 oracle、如何写不泄漏的题面、如何执行评分并导出。  

框架层（产物布局、`success` / `completed` / `warnings`、累积 apply-check、漏斗指标等）见 [framework.zh.md](framework.zh.md)；字段表见 [output-schema.zh.md](output-schema.zh.md)。

后续 **bug fix** 等题型会共用框架，替换的是本节这类算法与门控，而不是目录约定。

---

## 0. 题型目标

构造「挖空一段 coherent feature 的实现，用原生 / 新增测试做 oracle」的题目。每个成功实例在逻辑上对应：

| 侧 | Patch | 含义 |
|----|-------|------|
| Feature | `feature_mask_patch` | 原仓库 → 挖空后的待实现态 |
| Feature | `feature_gold_patch` | 挖空态 → 原实现（参考解） |
| Test | `test_mask_patch` | 隐藏仓库中所有测到该 feature 的现存测试 |
| Test | `test_gold_patch` | 正向、泛化的评分 oracle（覆盖每个 entry + 可选 P2P 守卫；**非** mask 的逆） |

评测时序（Export 预期）：

```
pristine
  → apply feature_mask + test_mask      # 实现挖空 + oracle 隐藏
  → [solver]
  → apply feature_gold（或 solver patch）
  → apply test_gold                     # 露出正向评分 oracle
  → 在 test_files 内跑测试 → F2P / P2P
```

阶段数据流（算法视角）：

```
图 + seed policy
  → Select：切片 → Agent 收成 feature（仅 agentic 下流）
  → Mask：entry stub + helper 删除（± refine）→ 可用 mask/gold
  → Eval：隐藏所有 feature 测试（test_mask）+ 编写正向泛化 oracle（test_gold）→ 可用 test oracle
  → Issue：全挖空态 + 给 gold 参考 → Agent 写不泄漏的 problem_statement
  → Export：执行评分（pregold FAIL / postgold PASS）→ F2P/P2P → Harbor 目录
```

---

## 1. Parse：实体图与种子策略

**代码：** `stage/parse/heuristics.py`、`utils/scip/indexer.py`、`utils/classify_tests.py`。

### 1.1 SCIP 索引

```
scip-python index --project-name <name> --project-version 0.1.0 --output parse_index.scip .
```

产出 protobuf 索引；失败则整次 Parse 失败。

### 1.2 建图（`_build_graph`）

实体 = SCIP symbol 后缀为 `TYPE`（class）或 `METHOD`（function）的定义点。NetworkX `DiGraph`。

| Pass | 产出 | 要点 |
|------|------|------|
| 1 | 定义位点 | `ROLE_DEFINITION` → file / def_line / enclosing_range |
| 1b | 节点 | kind / name / module / file / owner / is_test / documentation / external |
| 2 | `enclosing` | owner → child（descriptor 层次） |
| 3 | `implementation` | `relationships.is_implementation` |
| 4 | `reference` | 在 function 的 `enclosing_range` 内找 occurrence；`innermost()` 取最小 span 的 caller → callee |

`callers(g, f)` = 沿 `reference` 的入边前驱。Mask 的 entry 判定、Eval 的覆盖测试都依赖它。

**局限：** 只看见函数 body 内 occurrence；模块 / class body 级引用可能漏。

### 1.3 测试标注（`annotate_tests`）

`is_test` **仅**由路径决定：目录含 `tests`/`test`/`testing`，或文件名 `test_*.py` / `*_test.py` / `conftest.py`。名字与 SCIP Test role 只进 `test_evidence`。

### 1.4 Seed Policy（Agent）

按目录聚合成 profile，让 Agent 给出**不应作为 Select 种子起点**的 `exclude_globs`（生成代码、vendor、build 等）。节点仍留在图中，slice 仍可 BFS 长进去。失败则 Select 无 exclusion。

---

## 2. Select：切片 → Feature

**代码：** `stage/select/heuristics.py`、`stage/select/schema.py`、`stage/select/prompts.py`。

本题型策略：**无 heuristic 兜底**——只有 Agent 解析出的 feature（`source=='agentic'`）进入 Mask。

### 2.1 默认旋钮

| 参数 | 默认 | 作用 |
|------|------|------|
| `max_bfs_nodes` | 100 | 单 slice 节点上限 |
| `drop_bfs_prob` | 0.3 | 每层 Bernoulli 丢弃概率 |
| `min_bfs_per_depth` / `max_bfs_per_depth` | 1 / 8 | 每层保留上下限 |
| `max_subgraphs` | 20 | 仓库级 slice 预算 |
| `min_component_nodes` | 20 | 可 seed 连通分量下限 |
| `min_final_nodes` / `max_final_nodes` | 1 / 100 | Agent `feature_node_ids` 长度窗 |

### 2.2 候选集与无向投影

- `candidate_nodes`：默认保留测试、去掉 external。
- `undirected_projection`：把 enclosing / implementation / reference 全部投影成无向边；BFS 走无向邻接。

### 2.3 种子池与预算

**`seed_components`：** `allowed − exclude_globs` 后求连通分量，保留 size 落在窗口内的分量（降序）。过小 island 不可 seed。

**分量配额：**

$$
\mathrm{cap}(C)=\max\bigl(1,\ \lceil |C|/\mathrm{max\_bfs\_nodes}\rceil\bigr)
$$

- $\sum\mathrm{cap}\le\mathrm{max\_subgraphs}$：每分量拿满 cap。
- 否则：按 cap 比例 + **最大余数法** 分到恰好 `max_subgraphs`；小分量可能为 0。

**`sample_seeds`：** 分量内节点等概率、无放回抽样。

### 2.4 分层 BFS（`_layered_bfs`）

```
selected ← [seed]; used ← {seed}; frontier ← [seed]

while frontier 且 |selected| < max_bfs_nodes:
  layer ← frontier 的未 used 邻居
  if layer 空: break
  used ← used ∪ layer          # 扫过即永久占用（含后续被丢弃的）
  shuffle; Bernoulli(1-p) 保留
  补到 min_bfs_per_depth；截到 max_bfs_per_depth 与全局剩余额度
  if kept 空: break
  selected.extend(kept); frontier ← kept
```

取舍：优先小而多样的切片，不为填满 `max_bfs_nodes` 回访 hub 溢出。

### 2.5 Agent 审片

二选一：

1. **Resolve：** `feature` + `feature_node_ids`（须从图中原文复制；可增删补全，禁止漂到无关 feature）。
2. **Abandon：** 含与候选几乎无交的 drift。

Host：`abandoned` / 非法 id / 长度出窗 → `success=false` + `error`；通过则 `source='agentic'`，id enrich 成 `agentic_nodes`。

`cover_rate = |agentic ∩ heuristic| / |agentic|`：重叠监控，非硬过滤。

**可测试性（prompt 要求）：** 只纳入「本环境可测」的函数——要么已被现有测试覆盖，要么能在当前环境写出并跑通新的黑盒测试；**排除**只靠外部二进制 / 服务 / 网络或加载大模型才能跑的函数（这些下游无法评分）。去掉不可测部分后若不成连贯 feature，则 abandon。这样 feature 范围收敛到「可评分核心」，避免挖了却测不到的死代码。

**下游门控：** Mask 只吃 `source=='agentic'` 且 `nodes` 非空。

---

## 3. Mask：挖空实现

**代码：** `stage/mask/heuristics.py`、`utils/source_edit.py`、`stage/mask/prompts.py`。

### 3.1 Entry vs delete（`classify_feature`）

可 mask：`kind==function` 且 `not is_test`（测试永不挖）。

$$
\mathrm{entries}=\{f\in F_{\mathrm{fn}}\mid \mathrm{callers}(f)\setminus F_{\mathrm{fn}}\neq\emptyset\}
$$

$$
\mathrm{deletes}=F_{\mathrm{fn}}\setminus\mathrm{entries}
$$

- **Entry**（调用图发现被 feature 外引用，含测试）→ 草稿留签名，body → `raise NotImplementedError`。
- **Delete**（调用图未发现 feature 外 caller）→ 草稿整函数删（含上方连续 `@decorator`）。
- Class 只记录，本轮不改。

这个分类只是 dangle-safe 的机械草稿，不等价于 public API 判断：动态调用、仓库外用户和静态分析漏边都可能让重要入口落入
`deletes`。无 entry 时 warning：整 feature 草稿被删、无 API 锚点；模块级漏检靠 apply-check / 执行暴露。

### 3.2 行级编辑（`build_edit_patch`）

按 `enclosing_range` 行 span（非 AST）：

| 类型 | 行为 |
|------|------|
| `STUB` | 保留到首个以 `:` 结尾的签名行，其后 stub |
| `DELETE` | 删 span + 上吞装饰器 |

同文件：重叠保留外层；自底向上应用。启发式：

- `mask_patch` = difflib(原始 → 挖空)
- `gold_patch` = difflib(挖空 → 原始)

### 3.3 Agent refine

读 Select feature 描述、草稿 patch、源码、测试与文档。heuristic entries 是必须保留的安全下限；Agent 可把
`deletes` 中对配置、调用、组合、序列化等 feature 使用方式重要的 callable 提升为留签名 stub，只整删真正的实现 helper。
输出仍只有可 apply 的 `mask_patch`。越界只记 `cover_rate` / warning，不整包拒绝。Gold 用框架的
`reverse_patch`（git）；refine 本身无法 apply 则 gold 空，不可提交。

### 3.4 选定规则（本题型）

1. agentic mask+gold 非空且累积校验通过 → `source='agentic'`
2. 否则启发式 pair 通过 → `source='heuristic'`
3. 否则无交付物

校验序列：`feature_mask → feature_gold`（往返）。Unified `files` = 最终 mask 的 `patch_files()`（真实掩码面）。

**下游门控：** Eval 只吃 `success=true`，直接读取最终 feature mask/gold patch 判断完整行为范围，不再消费
heuristic entries。

---

## 4. Eval：测试 Oracle

**代码：** `stage/eval/heuristics.py`、`stage/eval/prompts.py`。

### 4.1 候选原生测试（`find_candidate_tests`）

对 feature 内每个 function $f$，在 `callers(f)` 中保留同时满足：

1. `is_test`（测试路径）
2. 名匹配 `^test[_A-Z0-9]`
3. **叶子：** 无 caller（排除 helper）
4. 默认保留「已被 Select 拉进 feature 的测试」（Mask 不挖测试，仍可作 oracle）

按覆盖 feature 函数数降序。只跟直接 `reference`；fixture 间接覆盖会漏。

### 4.2 启发式草稿（仅 HINT）

对**全部**候选 `delete_functions_patch` → `heuristic_mask_patch`（删全部候选）+ `heuristic_gold_patch`（原样恢复）。`run_eval` 把这**两份**草稿都写进沙箱（`eval_heuristic_{mask,gold}.patch`）供 Agent 按路径读；草稿本身没有结构化 runner，不能独立进入 Export。

### 4.3 Agent：MASK（遮全部）+ GOLD（正向泛化）+ RUNNER（结构化执行）

Agent 产出**两个非互逆**的 patch 和一份 grade-only `test_runner.sh`；输入还包含 `feature_mask_patch` / `feature_gold_patch`（按路径读的只读上下文，说明「feature 是什么、正确长什么样」）：

1. **test_mask_patch**：删除仓库中**所有**测到该 feature 的现存测试（不止候选 HINT，需自己搜全）。整个 test class / test 文件都测该 feature 时，可整块删除。只碰测试文件、不碰 feature 源码，且能 `git apply`。
2. **test_gold_patch**：**正向**评分 oracle。要求（a）经**公有接口**黑盒覆盖每个 entry（对照被删原测试别漏行为）；（b）**泛化**——断言可观测行为而非实现内部，让合理的其它实现也能过（但必须在 gold 代码上 PASS，别过度泛化）；（c）可选新增少量与 feature 无关的稳定测试作 **P2P** 守卫；（d）能在当前环境 `git apply` 且能跑。
3. **test_runner.sh**：自行选择仓库真实可用的 Python/环境/测试框架；读取 `SWE_ANY_TEST_FILES_JSON`，将稳定 test id + `PASSED/FAILED/ERROR/SKIPPED/XFAIL/XPASS` 写入 `SWE_ANY_RESULTS_PATH`，最后写唯一 `completed` summary。runner/依赖故障写 `runner_error` summary，不能伪装成测试失败。
4. **一致性（mask ⋈ gold）**：评测跑的是 apply `test_mask` 再 `test_gold` 之后测试文件里存在的全部测试。想被评分的必须在 mask+gold 之后存在；想隐藏的必须被 mask 删。
5. **自校验（沙箱内，仅跑相关测试）**：apply `feature_mask + test_mask + test_gold` 后执行 runner → 至少一个非 PASS；再 apply `feature_gold` 后执行同一 runner → 全 PASS、无 P2F。

**无原生候选的 feature**（`candidates` 为空）也会调 Agent，走 **extra-only**：`test_mask_patch=""`，从零写 `test_gold_patch` 覆盖全部 entry。这样「可测但没有现成测试」的 feature 不会被直接判 `no candidate tests` 丢弃。

Host 校验**放宽**：scope / 覆盖 / 行为级 / 无 P2F 全交 prompt + Agent 自校验，host 只做 `git apply` 干净性（选定块）。整块删除、非原样恢复的泛化 gold 均合法。

### 4.4 独立 Agentic Verifier 门禁

作者产物通过 apply-check 后，第二个独立 Agent 阅读完整的 feature mask/gold、test mask/gold、
runner 和仓库现有用法，输出一个稳定的主问题编号：

| verdict | 已知问题 | 典型 hack / 失败方式 |
|---------|----------|----------------------|
| `0` | 无错误 | 没有实质问题，允许进入下游 |
| `1` | Oracle 泄漏 | `test_mask` 没藏全；solver 可见文件泄漏断言、gold 输出或参考答案 |
| `2` | 越界 / 破坏性修改 | 改实现、无关测试、共享 fixture/config，或 runner 扩大到无关测试，破坏已有功能 |
| `3` | 过度约束 | 断言私有 helper、内部表示、精确算法，导致正确的替代实现也失败 |
| `4` | 测试过弱 | 漏能力/边界，只测 smoke、存在性、happy path，mock/绕过 feature，挖空态也能过 |
| `5` | 易被投机 | 固定样例太少、暴露测试常量/路径、源码文本检查，solver 可 hard-code/special-case |
| `6` | Runner 完整性 | 漏跑、伪造 status、吞失败、id 不稳定/冲突、runner 自己改仓库或 apply patch |
| `7` | 非 hermetic / flaky | 依赖网络、服务、时钟、随机、顺序、全局状态或不稳定外部二进制 |
| `-1` | 其它 | 存在实质问题，但不属于以上已知类别 |

Verifier 只写 `{"verdict", "rationale", "findings"}`；多个问题时 `verdict` 取最直接、最可利用的
主问题，`findings` 记录全部证据。host 仅接受 `verdict == 0`。调用失败、产物非法和任意非零
verdict 都 fail closed。

### 4.5 选定规则（本题型）

`test_gold` 是评分锚（定义 F2P），故可用 agentic oracle 必须 `test_gold` 与 `test_runner.sh`
均非空，patch pair 通过累积 apply-check，并通过独立 verifier；`test_mask` 可空（extra-only）。
任一门禁失败都拒绝该 oracle，不回退 heuristic。

校验序列（评测时序，累积）：`test_mask → test_gold`（disjoint 于 feature 源码 patch，故静态门只需两 test patch 一起干净 apply；feature patch 的联合校验由 Export 执行评分负责）。

```
test_files = existing_after_apply(patch_files(test_gold))
```

`test_mask` 只定义 solver 不可见的旧测试，不定义评分目标；因此永久删除或改名的 mask-only
路径不会传给 pytest。Export 在应用完整 patch 序列后再次检查文件存在性，兼容修复前已落盘的
`eval.jsonl`。

Eval 不预存 per-test id；同一 runner 在 pregold/postgold 执行时产生稳定 opaque id，Export
据两份结构化结果推 F2P/P2P。缺少结构化 runner 的旧/heuristic 行直接拒绝，不再解析 pytest 日志。

---

## 5. Issue：写不泄漏的问题描述

**代码：** `stage/issue/prompts.py`、`stage/issue/schema.py`；`task.py:run_swe_any_issue`。**纯 agentic**（无 heuristic 半部）。

对每个在 Mask（`success`）∩ Eval（`source`）都可用的 feature：

1. **搭全挖空态**：`apply_patches` 先回 pristine，再 `git apply feature_mask_patch + test_mask_patch` 并**保留**（实现挖空 + 原生测试删除）。两者一起 apply 失败 → 业务 error（这也顺带做了 mask/test patch 的联合可 apply 校验）。
2. **给模型看 gold 参考**：`feature_gold_patch`（参考实现）+ `test_gold_patch`（Eval 的正向评分 oracle）—— host 把这两份 `upload_text` 到固定沙箱路径,prompt 让 agent **按路径读**（和 Mask/Eval 的草稿 patch 一致,prompt 保持小）,本轮 agent 调用结束即清理;外加 entry 函数限定名与 Select 的一行描述。
3. **产出**：Agent 写 `{problem_statement}`;`normalize_issue_agentic` 要求非平凡（≥40 字符）才算 `success`,`source='agentic'`。Agent 崩溃 → `warnings`（软）。

硬规则(prompt)：不得复述实现、不得提及任何测试名/断言、不得泄漏私有 helper/内部算法。

## 6. Export：组装 → 执行评分 → Harbor

**代码：** `stage/export/schema.py`、`stage/export/harbor.py`;`task.py:run_swe_any_export`。**确定性**（无 LLM）。

对每个在 Mask ∩ Eval ∩ Issue 都可用的 feature（gate 在 issue `success`）：

1. **执行评分**（`grade_instance`，baked 镜像内跑两遍）：
   - pregold steps = `feature_mask + test_mask + test_gold`（实现挖空、正向 oracle 露出 → 应 FAIL）
   - postgold steps = `feature_mask + test_mask + feature_gold + test_gold`（叠 gold 复原实现 → 应 PASS）
   - 各 phase：reset → 依次 `git apply` → 执行相同 `test_runner.sh` → 校验 JSONL → reset。runner 能适配 pytest/unittest/Django；缺少 runner 的数据直接拒绝。
   - 结构化推导以 **postgold 为锚**：postgold PASS 且 pregold 非 PASS（含未出现）→ F2P；两侧都 PASS → P2P；pregold PASS 但 postgold 非 PASS/未出现 → P2F。
   - **门禁**：F2P 非空且无 P2F(回归)。`test_gold` 里的 F2P 与 P2P 守卫**同等**参与(文件级,跑 `test_files` 全部);postgold 里其它挂掉的测试(如缺外部二进制)落入 FAIL_TO_FAIL,自动被排除、不进 oracle、也不算回归。
2. **组装** `SWEbenchInstance` 形状：除 patch/F2P/P2P 外写入**评分时同一份 `test_runner_script` + protocol version**，让 rollout 与 Harbor 复用完全相同的环境适配和 test id。
3. **落两种消费物**:
   - `<output_dir>/<repo>/swe_any_instances.jsonl` —— **nexus 评测的默认入口**:由 `swe_task` 经 `swe/swe_any` dataset handler 消费,在 **runtime init 时** `git apply feature_mask+test_mask` 重建挖空态(推理阶段再 `run_git_reinit` 清历史防作弊),模型 patch 打在其上,`get_eval_script` 揭示 `test_gold` 跑 `test_files`,按 F2P/P2P 判 resolved。**镜像零改动**(bug 态运行时重建,和 SWE-Smith 一致)。
   - `<output_dir>/harbor/<instance_id>/` —— **实验性产物，请勿使用**。SWE-Any 的 Harbor 格式适配尚未完成、仍在测试中；当前 `harbor_bridge` 不 build `environment/Dockerfile`，因此不会应用 bug patch。现阶段只使用上面的 `swe_any_instances.jsonl` 入口。
   - `use_hack_check` 为保留 hook（未实现,置 true **直接报错**,避免配置假装做过 hack 筛）。

评分同时充当 mask+gold+test patch 的**联合执行校验**：三者能一起 apply 且 F2P 非空,才导出。

### Feature 级并行入口

新增入口 `scripts/run_swe_any_feature_pipeline.sh` 保留上述六阶段语义，但改为三个已注册 task：

1. `swe_any_pipeline_pre_feature`：每 repo 复用一个 runtime，只执行 Parse 和确定性 Select heuristic；不调用 LLM。每个候选 slice 都进入 `feature_manifest.jsonl`；heuristic 失败也携带 `heuristic_error`，由 feature task 写出显式 Select 失败和下游 blocked rows。
2. `swe_any_pipeline_feature`：每个 manifest row 独占一个 runtime，顺序执行 Select agentic → Mask → Eval → Issue → Export；只写 `sgNN/feature_result.json`、该 feature 的诊断文件和独立 Harbor instance 目录，不写 repo/global aggregate 或 `dataset.toml`。
3. `swe_any_pipeline_post_feature`：纯 host reduce。它包含失败 batch record，并在 `feature_result.json` 缺失/损坏时用 `sgNN/select_heuristic.json` 重建显式 blocked rows；最后稳定重写 repo aggregate、全局 `swe_any_instances.jsonl` 和一次 `harbor/dataset.toml`。

旧的 `run_swe_any.sh` 和六个 `swe_any_{parse,select,mask,eval,issue,export}` task 不变。

---

## 7. 算法相关工具（本题型常用）

| 能力 | 做法 |
|------|------|
| `patch_files` | 按 hunk 行数跳过正文；整文件删除记 `---` 侧路径 |
| 启发式逆 patch | `build_edit_patch` 对真实前后文本各 diff 一次 |
| Agent 逆 patch | 框架 `reverse_patch`（git） |

---

## 8. 本题型已知局限

1. 引用图不完整 → entry 误判、测试漏检。  
2. 行级编辑粗糙 → 靠 refine。  
3. BFS `used` 永久占用 → hub 后 slice 可能偏小。  
4. `exclude_globs` 过宽 → 无合格分量。  
5. 无 entry 的全内部 feature → 无 API 锚点。  
6. Eval 仅直接引用 → fixture 间接覆盖进不了候选。  
7. Oracle 权威在文件级 → 同文件未选中测试仍会被 Export 跑到。  
8. 只接受带仓库专属 runner 的 agentic oracle；heuristic/旧数据缺少 runner 时不进入 Export。
