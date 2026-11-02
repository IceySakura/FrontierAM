# Environment Setup Task

## Environment

| Item | Value |
|------|-------|
| Repository | `{repo}` |
| Base Commit | `{base_commit}` |
| Language | `{language}` |
| Testbed | `/testbed` (cloned and reset to base_commit) |
| Base Image | `{base_image}` |

{auto_setup_results}

## Patch Information

**test_patch** — test cases that expose the bug (embed in eval.sh via heredoc):
<test_patch>
{test_patch}
</test_patch>

**golden_patch** — the reference fix (read to understand what changed; NOT applied to /testbed):
<golden_patch>
{golden_patch}
</golden_patch>

## Issue Description

{problem_statement}

---

## Task

Set up this environment so that a **fail2pass eval script** can be validated and the environment image uploaded.

**Deliverables:**
1. `/tmp/eval.sh` — self-contained script that **FAILS on base_commit** and **PASSES on base_commit + golden_patch**
2. (optional) Additional install steps if auto-setup was incomplete

**Success criteria:** call `submit_eval` and pass validation.

**eval.sh format:**
- Apply test_patch via heredoc (if non-empty and applicable)
- Run the targeted test command
- End with `rc=$?; echo "OMNIGRIL_EXIT_CODE=$rc"; exit $rc`

---

## Workflow

### Phase 1 — Verify & Fix Setup (skip if auto-setup has no warnings)

Auto-setup has already explored the repo, selected the runtime image, probed patches, and installed dependencies. Check the **Auto Setup Results** above.

1. **If no warnings** → skip directly to Phase 2
2. **If install failed** → fix the specific failing install command with `bash`
3. **If image wrong** → call `switch_runtime_image` (max 3 switches)
4. **If missing packages** → install them with `bash`
5. **If golden_patch is `hunk_failed` or `already_exists`** → try `switch_runtime_image`; if still broken, proceed with Phase 2 anyway (submit_eval retries with fallback apply methods)

**Budget:** ≤5 bash calls for this phase. Do not re-install packages that auto-setup already installed successfully. You may use `bash` to explore the repo or gather additional information if needed.

---

### Phase 2 — Write eval.sh & Submit

Based on **Patch Probing** results from auto-setup:

| Strategy | Action |
|----------|--------|
| **standard** | Embed test_patch as heredoc, run test command from test_patch |
| **custom_test** | Write custom test from golden_patch analysis (test_patch broken or empty) |
| **crlf_fix** | Add `git -C /testbed config core.autocrlf input` at top of eval.sh, then standard |

#### Step 1: Write eval.sh

**Standard eval.sh template (strategy=standard):**
```bash
#!/bin/bash
set -xo pipefail
cd /testbed
git config --global --add safe.directory /testbed

# Source runtime env (docker commit does NOT preserve session exports):
# [ -f /root/venv/bin/activate ] && source /root/venv/bin/activate
# export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
# [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
# export BUNDLE_PATH=/root/bundle

# Apply test_patch via heredoc (NOT from /tmp — those files are deleted before eval.sh runs)
git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# Run the targeted test (exact path and name from test_patch)
<YOUR TEST COMMAND>

rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Custom test eval.sh template (strategy=custom_test):**
```bash
#!/bin/bash
set -xo pipefail
cd /testbed
git config --global --add safe.directory /testbed

# No test_patch heredoc — write a custom test targeting golden_patch changes
<WRITE CUSTOM TEST FILE OR INLINE TEST>

# Run the custom test via the project's test runner
<YOUR TEST COMMAND>

rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

#### Step 2: Submit

```
submit_eval(eval_script_path="/tmp/eval.sh", explanation="what the test checks and why it fails on base")
```

On failure, read the submit_eval output, make ONE targeted fix to eval.sh, and resubmit.

---

## Reference

### How submit_eval Works

1. Post-processes eval.sh (removes `set -u`, injects GRADLE_OPTS/CARGO settings)
2. Static checks (rejects scripts referencing `/tmp` patch files or hardcoded exit codes)
3. **Pre-check:** runs eval.sh on base state first — if PASSES (exit 0), rejects immediately as pass2pass (no golden_patch applied)
4. Deletes `/tmp/golden_patch.patch` and `/tmp/test_patch.patch` (prevents cheating)
5. Applies golden_patch to /testbed → runs eval.sh → **full state must PASS**
6. Resets /testbed to base_commit → runs eval.sh → **base state must FAIL**
7. LLM validation (checks that eval.sh runs real tests, not faked results)
8. Checks /testbed is clean, commits and uploads the environment image
9. On failure: restores /testbed to base_commit and restores /tmp patch files

### Key Rules

**Three rules to follow:**
- `/testbed` is at `base_commit` — golden_patch is NOT applied. Tests on base_commit will fail; this is expected, not a problem to debug
- **Never run test commands outside eval.sh** — output is wrong on base_commit and wastes iterations
- **Never apply golden_patch permanently** — submit_eval applies it automatically

**eval.sh must be self-contained (NO /tmp patch files):**
- `submit_eval` deletes `/tmp/golden_patch.patch` and `/tmp/test_patch.patch` **before** running eval.sh — you MUST embed test_patch content as a heredoc in eval.sh, never `git apply /tmp/test_patch.patch`
- Source all runtime environments explicitly (docker commit does not preserve session exports)
- Use `set -xo pipefail` (not `set -euxo` — `set -u` is stripped automatically)
- End with `rc=$?; echo "OMNIGRIL_EXIT_CODE=$rc"; exit $rc`

**Paths:**
- Always use `/testbed` as repo root
- Build dirs outside /testbed: C/C++ → `/tmp/project_build`; Python venv → `/root/venv`; Ruby gems → `/root/bundle`

**Install safety** (violations block golden_patch application):
- Never modify `.gitignore`
- Never create source/config files inside `/testbed` from install scripts
- npm: `npm install --ignore-scripts`
- bundler: `BUNDLE_PATH=/root/bundle bundle install`

**Auto-injected by submit_eval** (no action needed):
- `GRADLE_OPTS` proxy config when eval.sh uses `gradlew`
- `CARGO_HTTP_MULTIPLEXING=false` when eval.sh uses `cargo`

### Diagnosing Failed submit_eval

After each failure: /testbed is auto-reset; /tmp patch files are restored. Run at most ONE diagnostic bash before fixing.

#### golden_patch failed to apply (eval.sh was NOT run)

| Error | Cause | Fix |
|-------|-------|-----|
| `already exists` | install created a file in /testbed that golden_patch also creates | Remove from install (`--ignore-scripts`, `BUNDLE_PATH=/root/bundle`), resubmit |
| `trailing whitespace` / CRLF | CRLF in repo | `git -C /testbed config core.autocrlf input`, resubmit |
| `does not apply` / `hunk FAILED` | Data quality | Resubmit (system retries with fallbacks); if persists, try switch_runtime_image |

#### Full state FAILS

Read the **first** error line in submit_eval output, match to this table, make ONE fix:

| Error | Cause | Fix |
|-------|-------|-----|
| `command not found` / missing library | Runtime env not sourced in eval.sh | Add sourcing (see eval.sh template) |
| test_patch `git apply` fails inside eval.sh | Context doesn't match after golden_patch | Add `--3way` or `--ignore-whitespace` to `git apply` in eval.sh; if still fails, write custom test |
| Test runner reports failures | golden_patch doesn't fix what this test checks | Pick a test closer to golden_patch changes |
| `No tests ran` / `collected 0 items` | Wrong path or filter | Fix test command |
| Build error (`CMakeError`, `error[E`, `BUILD FAILURE`) | Not compiled | Add build step before test |
| DB connection error | Database not started | Add `service postgresql start; sleep 2` in eval.sh |

#### Base state PASSES (pass2pass)

The test passes even without the fix — it cannot detect the bug.

**Strategy:** Design a custom test from golden_patch that exercises the specific code change:

| Golden patch change | Test approach | Why it fails on base |
|---------------------|--------------|---------------------|
| Adds new function/method | Call that function directly | Symbol not found |
| Return value / validation changed | Assert the new value | Old value returned |
| New class/type added | Import or instantiate it | Import error |
| Error path removed | Assert no-throw on formerly-failing input | Old code throws |

**Write the custom test using the project's test runner** (pytest, cargo test, go test, vendor/bin/phpunit, etc.), not bare scripts. Then call submit_eval immediately.

**Stop condition:** 5+ submissions with base still passing → no fail2pass signal exists. Stop immediately.

### Rejected Patterns (validator will reject)

| Pattern | Reason |
|---------|--------|
| `grep -q "func" src/file.c` as sole test | Source text inspection, not behavioral test |
| `pytest ... \|\| true` | Always exits 0 |
| `cargo build` only (no `cargo test`) | Compiles but doesn't test |
| `git apply /tmp/test_patch.patch` in eval.sh | File deleted before eval runs — use heredoc |

### Handling Broken test_patch

When test_patch cannot be applied (context mismatch, corruption), do NOT try to fix the patch itself. Instead:

1. Read golden_patch to understand what production code changed
2. Write a custom test that exercises the specific change
3. Use the project's test runner, not bare scripts
4. Embed the custom test directly in eval.sh

### Binary Files in test_patch

If test_patch contains binary file entries, exclude them from `git apply` and restore manually:
```bash
git apply --ignore-whitespace --exclude='**/*.bin' --exclude='**/*.png' <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF
mkdir -p path/to/ && git cat-file blob abc1234 > path/to/file.bin
```

### External Services

All services are stopped before eval.sh runs. Start required services inside eval.sh:
```bash
service postgresql start; sleep 2
```

---

{language_guide}
