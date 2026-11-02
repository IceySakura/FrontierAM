# Auto Setup 输出示例

> 本文档展示 `auto_setup` 在不同语言/场景下的实际输出效果。
> 这些内容会被 `format_auto_setup_results()` 格式化后，通过 `{auto_setup_results}` 占位符注入到 agent 的 user prompt 中。

---

## 示例 1：Go 项目（标准路径，一切正常）

**场景**：`kubernetes/kubernetes` 仓库，Go 1.21，golden_patch 和 test_patch 都能 apply。

### auto_setup 内部执行流程

```
[AUTO_SETUP] Step 1: Exploring repository...
  → 检测到 go.mod: go 1.21
  → 包文件: go.mod, go.sum, Makefile
  → 测试目录: test/ (15 entries), pkg/kubelet/test/ (8 entries)

[AUTO_SETUP] Step 2: Determining runtime image...
  → _determine_image('go', data) → ('golang-local:1.21', '1.21')
  → 当前 image: ubuntu-git:22.04 → 需要切换
  → 切换到 golang-local:1.21 → 重新 clone + reset testbed

[AUTO_SETUP] Step 3: Probing patches...
  → golden_patch: git apply --check → exit_code=0 → clean (12 files)
  → test_patch: 先 apply golden，再 check test → exit_code=0 → clean on golden (3 files)
  → eval_strategy = standard

[AUTO_SETUP] Step 4: Running pre-install...
  → go mod download → exit_code=0, 32.1s
  → 总结: 1 ok, 0 failed, 32s total

[AUTO_SETUP] Complete in 58.3s
```

### 注入 prompt 的内容（format_auto_setup_results 输出）

```markdown
## Auto Setup Results

**Image:** `mirrors.tencent.com/.../swe_builder_eval_base_golang_local_1_21:latest` (switched from previous)
**Language:** go 1.21
**Package files:** go.mod, go.sum, Makefile
**Test dirs:** `test/` (15 entries), `pkg/kubelet/test/` (8 entries)

### Patch Probing
- golden_patch: **clean** — applies cleanly (12 files)
- test_patch: **clean** (on golden) — applies cleanly (3 files)
- Strategy: **standard**

### Pre-Install
| # | Command | Status | Time |
|---|---------|--------|------|
| 1 | `cd /testbed && go mod download 2>&1` | OK | 32.1s |
**Summary:** 1 ok, 0 failed, 32s total

### Warnings
- (none)
```

### Agent 看到的完整 prompt（base.md 渲染后，关键部分）

```markdown
# Environment Setup Task

## Environment

| Item | Value |
|------|-------|
| Repository | `kubernetes/kubernetes` |
| Base Commit | `a1b2c3d` |
| Language | `go` |
| Testbed | `/testbed` (cloned and reset to base_commit) |
| Base Image | `mirrors.tencent.com/.../swe_builder_eval_base_golang_local_1_21:latest` |

## Auto Setup Results

**Image:** `mirrors.tencent.com/.../swe_builder_eval_base_golang_local_1_21:latest` (switched from previous)
**Language:** go 1.21
**Package files:** go.mod, go.sum, Makefile
**Test dirs:** `test/` (15 entries), `pkg/kubelet/test/` (8 entries)

### Patch Probing
- golden_patch: **clean** — applies cleanly (12 files)
- test_patch: **clean** (on golden) — applies cleanly (3 files)
- Strategy: **standard**

### Pre-Install
| # | Command | Status | Time |
|---|---------|--------|------|
| 1 | `cd /testbed && go mod download 2>&1` | OK | 32.1s |
**Summary:** 1 ok, 0 failed, 32s total

### Warnings
- (none)

## Patch Information

**test_patch** — test cases that expose the bug (embed in eval.sh via heredoc):
<test_patch>
diff --git a/pkg/kubelet/kubelet_test.go b/pkg/kubelet/kubelet_test.go
...
</test_patch>

**golden_patch** — the reference fix:
<golden_patch>
diff --git a/pkg/kubelet/kubelet.go b/pkg/kubelet/kubelet.go
...
</golden_patch>

## Issue Description
...

---

## Workflow

### Phase 1 — Verify & Fix Setup (skip if auto-setup has no warnings)

Auto-setup has already explored the repo, selected the runtime image, probed patches,
and installed dependencies. Check the **Auto Setup Results** above.

1. **If no warnings** → skip directly to Phase 2     ← agent 看到 no warnings，直接跳到 Phase 2
...

### Phase 2 — Write eval.sh & Submit

Based on **Patch Probing** results from auto-setup:

| Strategy | Action |
|----------|--------|
| **standard** | Embed test_patch as heredoc, run test command from test_patch |  ← strategy=standard
...
```

**Agent 预期行为**：看到 no warnings + strategy=standard，直接写 eval.sh（嵌入 test_patch heredoc + `go test` 命令），然后 submit_eval。 **预计 2-3 次迭代**。

---

## 示例 2：Python 项目（安装部分失败）

**场景**：`django/django` 仓库，Python 项目，pip install 部分失败。

### 注入 prompt 的内容

```markdown
## Auto Setup Results

**Image:** `mirrors.tencent.com/.../swe_builder_eval_base_ubuntu_git_22_04:latest`
**Language:** python unknown
**Package files:** setup.py, setup.cfg, pyproject.toml, tox.ini, requirements.txt
**Test dirs:** `tests/` (42 entries)

### Patch Probing
- golden_patch: **clean** — applies cleanly (3 files)
- test_patch: **clean** (on golden) — applies cleanly (2 files)
- Strategy: **standard**

### Pre-Install
| # | Command | Status | Time |
|---|---------|--------|------|
| 1 | `cd /testbed && pip install -e ".[dev,test]" 2>/dev/null || pip ins...` | FAILED | 45.2s |
| 2 | `pip install tox 2>/dev/null || true` | OK | 8.1s |
**Summary:** 1 ok, 1 failed, 53s total

### Warnings
- install failed: cd /testbed && pip install -e ".[dev,test]" — error: subprocess-exited-with-error
```

**Agent 预期行为**：
1. 看到 install warning → Phase 1 查漏补缺
2. 运行 `bash` 查看具体错误，可能需要 `pip install -e .` 或 `pip install -r requirements/tests.txt`
3. 修好后进入 Phase 2 写 eval.sh
4. **预计 5-8 次迭代**（原先 20-40 次）

---

## 示例 3：Rust 项目（test_patch 无法 apply）

**场景**：`tokio-rs/tokio` 仓库，Rust 1.75，test_patch context mismatch。

### 注入 prompt 的内容

```markdown
## Auto Setup Results

**Image:** `mirrors.tencent.com/.../swe_builder_eval_base_rust_git_1_75_slim:latest` (switched from previous)
**Language:** rust 1.75
**Package files:** Cargo.toml, Cargo.lock
**Test dirs:** `tests/` (12 entries)

### Patch Probing
- golden_patch: **clean** — applies cleanly (5 files)
- test_patch: **context_mismatch** (on golden) — fails on both golden and base: error: patch failed: tokio/tests/...
- Strategy: **custom_test**

### Pre-Install
| # | Command | Status | Time |
|---|---------|--------|------|
| 1 | `cd /testbed && cargo fetch 2>&1` | OK | 28.5s |
**Summary:** 1 ok, 0 failed, 29s total

### Warnings
- (none)
```

**Agent 预期行为**：
1. 看到 no warnings → 跳过 Phase 1
2. 看到 strategy=**custom_test** → 不嵌入 test_patch，而是分析 golden_patch，写自定义测试
3. 从 golden_patch 找到新增的函数/修改的逻辑，写一个 `cargo test` 命令
4. **预计 5-10 次迭代**（原先需要 agent 自己探测 patch、发现 apply 失败、反复调试）

---

## 示例 4：C++ 项目（cmake，需要额外安装）

**场景**：`opencv/opencv` 仓库，C++ cmake 项目。

### 注入 prompt 的内容

```markdown
## Auto Setup Results

**Image:** `mirrors.tencent.com/.../swe_builder_eval_base_ubuntu_git_22_04:latest`
**Language:** c++ unknown
**Package files:** CMakeLists.txt, Makefile
**Test dirs:** `tests/` (8 entries)
**Build systems:** cmake, make

### Patch Probing
- golden_patch: **clean** — applies cleanly (2 files)
- test_patch: **clean** (on golden) — applies cleanly (1 files)
- Strategy: **standard**

### Pre-Install
| # | Command | Status | Time |
|---|---------|--------|------|
| 1 | `apt-get install -y build-essential cmake libgtk-3-dev libavcodec-d...` | OK | 120.3s |
**Summary:** 1 ok, 0 failed, 120s total

### Warnings
- (none)
```

**Agent 预期行为**：
1. 跳过 Phase 1
2. 写 eval.sh：cmake configure → cmake build → ctest（嵌入 test_patch heredoc）
3. **预计 5-10 次迭代**

---

## 示例 5：Java 项目（golden_patch CRLF 问题）

**场景**：某 Windows 风格的 Java 项目，golden_patch 有 CRLF。

### 注入 prompt 的内容

```markdown
## Auto Setup Results

**Image:** `mirrors.tencent.com/.../swe_builder_eval_base_java_git_17:latest` (switched from previous)
**Language:** java 17
**Package files:** pom.xml, mvnw

### Patch Probing
- golden_patch: **crlf** — CRLF/trailing whitespace detected
- test_patch: **clean** (on base) — applies cleanly (2 files)
- Strategy: **crlf_fix**

### Pre-Install
| # | Command | Status | Time |
|---|---------|--------|------|
| 1 | `cd /testbed && mvn dependency:resolve -q 2>&1 || true` | OK | 85.6s |
**Summary:** 1 ok, 0 failed, 86s total

### Warnings
- (none)
```

**Agent 预期行为**：
1. 跳过 Phase 1
2. 看到 strategy=**crlf_fix** → 在 eval.sh 开头添加 `git -C /testbed config core.autocrlf input`
3. 然后标准流程：嵌入 test_patch + mvn test
4. **预计 3-5 次迭代**

---

## 示例 6：submit_eval 失败的结构化返回

### 旧版返回（~4000 chars，淹没 context）

```
❌ Full state FAILED (base + golden_patch): Fail: full state (base+golden_patch) did not pass (OMNIGRIL_EXIT_CODE=1) — likely environment/setup issue

This means the environment is not correctly set up — the test cannot even pass when the fix is applied.
Fix the environment or test command first.

[Full state output (exit=1)]:
+ cd /testbed
+ git config --global --add safe.directory /testbed
+ git apply --ignore-whitespace <<'PATCH_EOF'
... (test_patch 内容 ~2000 chars)
PATCH_EOF
+ pytest tests/test_parser.py::test_validate_input -xvs
FAILED tests/test_parser.py::test_validate_input - ModuleNotFoundError: No module named 'lxml'
=========================== short test summary info ============================
FAILED tests/test_parser.py::test_validate_input - ModuleNotFoundError: No module named 'lxml'
============================== 1 failed in 0.42s ===============================
+ rc=1
+ echo OMNIGRIL_EXIT_CODE=1
OMNIGRIL_EXIT_CODE=1
+ exit 1

(Note: /testbed has been reset to base_commit — you can run eval.sh directly to debug.)
```

### 新版返回（~400 chars，结构化）

```
## submit_eval FAILED: full_state_fails (attempt 2/30)

### Root Cause
FAILED tests/test_parser.py::test_validate_input - ModuleNotFoundError: No module named 'lxml'

### Suggested Fix
pip install lxml in eval.sh before test command

### History
[1: full_state_fails, 2: full_state_fails]

### Output (last 500 chars)
```
FAILED tests/test_parser.py::test_validate_input - ModuleNotFoundError: No module named 'lxml'
============================== 1 failed in 0.42s ===============================
+ rc=1
+ echo OMNIGRIL_EXIT_CODE=1
```
```

### pass2pass 预检拦截返回（~300 chars）

```
## submit_eval FAILED: base_state_passes (attempt 1/30)

### Root Cause
eval.sh PASSES on base_commit (OMNIGRIL_EXIT_CODE=0) — no fail2pass signal.

### Suggested Fix
Write a custom test from golden_patch that exercises the specific code change (fails on base, passes with fix). Do NOT use test_patch heredoc.

### History
[1: base_state_passes]
```

**对比**：
- 旧版 ~4000 chars（~1000 tokens），大量原始输出淹没关键信息
- 新版 ~300-500 chars（~100 tokens），直接给出 root cause + 修复建议
- pass2pass 预检在 submit_eval 开头就拦截，**不需要执行 golden_patch apply + full state 运行**，节省 60-180s

---

## 端到端流程对比

### 旧流程（Go 项目示例）

```
迭代 1:  agent 调用 explore_repo                    → 返回 ~3000 chars 报告
迭代 2:  agent 解读报告，调用 switch_runtime_image   → 切换到 golang-local:1.21
迭代 3:  agent 编写 install.sh                      → str_replace_editor
迭代 4:  agent 运行 bash /tmp/install.sh            → go mod download（输出 ~2000 chars）
迭代 5:  agent 检查安装结果                          → bash echo "install ok"
迭代 6:  agent 探测 golden_patch                     → bash git apply --check（输出 ~500 chars）
迭代 7:  agent 探测 test_patch                       → bash git apply --check（输出 ~500 chars）
迭代 8:  agent 重置 testbed                          → bash git reset --hard
迭代 9:  agent 开始写 eval.sh                        → str_replace_editor
迭代 10: agent 调用 submit_eval                      → 成功
───────────────────────────────────────────
总计: 10 次迭代，context ~30K tokens 到 submit_eval
```

### 新流程（同一个 Go 项目）

```
Phase 0: auto_setup（不消耗迭代，~58s）
  ├─ explore_repo → 检测 Go 1.21
  ├─ switch_runtime_image → golang-local:1.21
  ├─ probe_patches → golden=clean, test=clean, strategy=standard
  └─ pre_install → go mod download OK

迭代 1: agent 看到 auto_setup 结果（no warnings + standard）
         直接写 eval.sh                              → str_replace_editor
迭代 2: agent 调用 submit_eval                       → 成功
───────────────────────────────────────────
总计: 2 次迭代，context ~5K tokens 到 submit_eval
```

**节省：8 次迭代，~25K tokens context**
