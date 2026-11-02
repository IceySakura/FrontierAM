# SWE-Any output schema

Field-by-field reference for everything each stage writes to the **host**. All artifacts
for one repo live under `${SWE_ANY_OUTPUT_DIR}/<repo_name>/` (default
`./outputs/swe_any/<repo_name>/`). This is the source of truth for the on-disk contract;
if you change a row builder in `task.py`, update this file.

Cross-cutting framework (stage pattern, soft/hard failure, git primitives, funnel metrics)
is documented in Chinese in [framework.md](framework.zh.md). Feature-implementation algorithms
are in [feature-implementation.md](feature-implementation.zh.md).

## Conventions

- **Pipeline stages (current feature-implementation wiring)**: `Parse → Select → Mask → Eval`.
  Parse runs once per repo; Select / Mask / Eval iterate over per-repo *items* (slices /
  features). Future problem types may swap stage algorithms but should keep the layout and
  outcome/status axes in [framework.md](framework.zh.md).
- **Join key** — every per-item row carries an integer `index` (the Select slice id, shown
  on disk as the `sgNN/` subdir). The same `index` threads a slice through
  `select → mask → eval`, so the four `*.jsonl` deliverables join on it.
- **Per-item layout** (Select / Mask / Eval all share it) — see [framework.md §2](framework.zh.md):
  `sgNN/<stage>_{heuristic,agentic}.json`, repo aggregates `*_heuristic.jsonl` /
  `*_agentic.jsonl`, and unified `<stage>.jsonl` (what the next stage reads).
- **Outcome vs status** — two independent axes; full rules in [framework.md §3](framework.zh.md).
  Short form: `success = (source != "")`; soft failures go to `warnings` (`completed` stays
  true); `runtime_error` is reserved for unexpected interruptions (`completed=false`).
  A patch is committed to `source` only when it passes cumulative `git apply` validation.
- **Status lives on the unified row.** `completed` / `runtime_error` are carried by
  `<stage>.jsonl` (there is no separate summary file); the per-half `sgNN/*.json` +
  `<stage>_{heuristic,agentic}.jsonl` stay **pure content**. Exception: Select's `success` /
  `error` are part of its *agentic content* (the agent's own verdict), so they also appear in
  `select_agentic.json`; Mask / Eval derive `success` / `error` host-side, so those two live only
  on the unified row.
- **Field order in the `<stage>.jsonl` (unified) tables** — fields the next stage / final
  dataset consumes are marked ★ and listed first (the `index` join key, the gating `source`,
  and the payload node-sets / patches); provenance, monitoring, and status follow.
- **Types** use JSON types: `str`, `int`, `float`, `bool`, `null`, `list[...]`, `object`.
  `X | null` means the key is always present but may be JSON `null`.

## Directory layout (one repo)

```
<output_dir>/<repo_name>/
  parse_graph.json                 # Parse — entity/relation graph
  parse_seed_policy.json           # Parse — seed-policy exclusions (may be absent)
  select_heuristic.jsonl           # Select — per-half aggregates
  select_agentic.jsonl
  select.jsonl                     # Select — UNIFIED (Mask reads this; carries status)
  mask_heuristic.jsonl             # Mask  — per-half aggregates
  mask_agentic.jsonl
  mask.jsonl                       # Mask  — UNIFIED (Eval reads this; carries status)
  eval_heuristic.jsonl             # Eval  — per-half aggregates
  eval_agentic.jsonl
  eval.jsonl                       # Eval  — UNIFIED (carries status)
  issue_agentic.jsonl              # Issue — agentic aggregate (agentic-only: no heuristic half)
  issue.jsonl                      # Issue — UNIFIED (problem_statement; Export reads this)
  export.jsonl                     # Export — UNIFIED (instance_id + graded F2P/P2P + harbor path)
  swe_any_instances.jsonl          # Export — SWEbench-shaped records (dataset_type='swe-any'); swe_task reads via the swe/swe_any handler
  sgNN/                            # one dir per item (sg00, sg01, ...)
    select_heuristic.json  select_agentic.json
    mask_heuristic.json    mask_agentic.json    mask_heuristic.patch
    eval_heuristic.json    eval_agentic.json    eval_agentic_{mask,gold}.patch
    eval_agentic_test_runner.sh    eval_agentic_verifier.json
    issue_agentic.json                                   # Issue (agentic-only)

# EXPERIMENTAL: Harbor adaptation is incomplete and under test; do not use for evaluation.
<output_dir>/harbor/                # Export — unsupported Harbor artifact (directory-per-instance)
  dataset.toml
  swe_any.<repo>.sgNN/             # one dir per graded-VALID instance
    task.toml  instruction.md
    environment/{Dockerfile, feature_mask.patch, test_mask.patch}
    solution/{solve.sh, feature_gold.patch}
    tests/{test.sh, test_gold.patch, test_runner.sh, verify_results.py, config.json}
```

**Deviations.** Issue is **agentic-only** (no `issue_heuristic.*` — nothing deterministic to draft),
so it has just the agentic aggregate + the unified `issue.jsonl`. Export is **deterministic + terminal**
(no per-item halves): it writes one unified `export.jsonl` row per instance and the Harbor directory
dataset (the per-instance artifact is the harbor task dir, not a `sgNN/` JSON).

> **Warning:** the Harbor format adaptation is incomplete and still under testing. Do not use
> `harbor/` as an evaluation input. The supported path is `swe_any_instances.jsonl` through the
> `swe-any` dataset handler.

**`.json` vs `.jsonl`** — the two are the same data at different granularity, **not** two
formats of one file:

- `sgNN/<stage>_heuristic.json` / `sgNN/<stage>_agentic.json` — one **per-item** JSON object,
  under that item's `sgNN/` subdir, written as soon as the item finishes (streaming;
  crash-safe / independently recoverable).
- `<stage>_heuristic.jsonl` / `<stage>_agentic.jsonl` — the **repo-level aggregate** at the repo
  root, one JSON object per line, written once at the end. Line `N` of the `.jsonl` is exactly
  the content of `sg{N}/<stage>_heuristic.json` (same schema).

So each schema below is documented once and applies to both the `sgNN/*.json` file and one
line of its `.jsonl` aggregate. The `.patch` siblings are the raw draft diffs the agent reads
by path.

---

# Parse

Single-shot per repo (no `sgNN/`, no heuristic/agent split). Deterministic step = SCIP index
→ entity graph; agentic step = seed policy.

## `parse_graph.json`

The entity/relation graph (protobuf-free JSON). Entity = a `function` or `class` at its SCIP
definition site; external symbols (referenced but not defined in the repo) appear as
minimal nodes.

| Field | Type | Meaning |
|-------|------|---------|
| `info` | object | Graph-level metadata (see below). |
| `nodes` | list[object] | Entity nodes (see below). |
| `edges` | list[object] | Relation edges (see below). |

**`info`** (object)

| Field | Type | Meaning |
|-------|------|---------|
| `project_root` | str | SCIP `metadata.project_root` (repo root URI). |
| `tool` | str | Indexer name + version, e.g. `scip-python 0.x`. |

**`nodes[]`** (object)

| Field | Type | Meaning |
|-------|------|---------|
| `id` | str | The SCIP symbol string; the node's unique id (the agent selects these ids; they become `agentic_nodes[].id` / edge endpoints). |
| `info` | object | Node attributes (see below). |

**`nodes[].info`** (object) — internal nodes carry all keys; external nodes carry only `external: true`.

| Field | Type | Meaning |
|-------|------|---------|
| `kind` | str | `"function"` or `"class"`. |
| `name` | str | Display name (last SCIP descriptor), e.g. `speak`. |
| `module` | str | Dotted module/namespace, e.g. `zoo.animals`. |
| `file` | str | Repo-relative source path. |
| `owner` | str \| null | SCIP symbol of the immediate owner (enclosing class/function), else `null`. |
| `def_line` | int \| null | 0-based line of the definition, or `null` if the def site was not found. |
| `enclosing_range` | list[int] \| null | `[start_line, start_col, end_line, end_col]` (0-based) of the full definition span; `null` if unknown. |
| `is_test` | bool | Whether this entity is test code (re-tagged strictly by test-file path). |
| `test_evidence` | list[str] | Why it was (or wasn't) tagged: any of `"scip_role"`, `"path"`, `"name"`. |
| `documentation` | str | Docstring / hover text (may be empty). |
| `external` | bool | `false` for repo-defined entities; external nodes instead have only `external: true`. |

**`edges[]`** (object)

| Field | Type | Meaning |
|-------|------|---------|
| `src` | str | Source node id. |
| `dst` | str | Destination node id. |
| `infos` | list[object] | Parallel relation facts on this `(src, dst)` (a plain DiGraph edge can hold several). |

**`edges[].infos[]`** (object)

| Field | Type | Meaning |
|-------|------|---------|
| `type` | str | `"enclosing"` (parent→child), `"implementation"` (impl→interface), or `"reference"` (caller→callee). |
| `dst_external` | bool | Whether `dst` is an external symbol. |
| `evidence` | str | How the edge was derived: `"descriptor"`, `"is_implementation"`, or `"occurrence"`. |

## `parse_seed_policy.json`

Written by the seed-policy agent (Select reads it to shape its seed pool). Absent when the
seed-policy pass failed (then `ParseTaskOutput.seed_policy_error` explains why).

| Field | Type | Meaning |
|-------|------|---------|
| `exclude_globs` | list[str] | Repo-relative globs barred from *seeding* a slice (generated/vendored/build code). Applied with `fnmatch` (so `*` also matches `/`). |
| `exclude_reasons` | object (str→str) | Per-glob one-line justification. |
| `notes` | str | Where the real hand-written product logic lives (free text). |

---

# Select

One item = one BFS slice around a sampled seed. Deterministic step = layered-BFS slice;
agentic step = the Claude Code pass (does the slice form one coherent feature?).

## `sgNN/select_heuristic.json` (per slice) + `select_heuristic.jsonl` (repo aggregate)

The BFS slice. On a heuristic-step failure the row degrades to `{index, seed, meta: {}, heuristic_nodes: []}`.

| Field | Type | Meaning |
|-------|------|---------|
| `index` | int | Item id (sgNN). |
| `seed` | str | Node id the slice grew from. |
| `meta` | object | Slice knobs + counts (see below). |
| `heuristic_nodes` | list[object] | The selected slice nodes (shape below; shared with agentic `agentic_nodes`). |

**`meta`** (object)

| Field | Type | Meaning |
|-------|------|---------|
| `max_bfs_nodes` | int | Per-slice node cap used. |
| `drop_bfs_prob` | float | Per-depth Bernoulli dropout probability used. |
| `min_bfs_per_depth` | int | Min nodes kept per BFS depth. |
| `max_bfs_per_depth` | int | Max nodes kept per BFS depth. |
| `include_tests` | bool | Whether test nodes were eligible. |
| `include_external` | bool | Whether external nodes were eligible. |
| `rng_seed` | int \| null | Per-slice RNG seed (`null` = unseeded draw). |
| `component_size` | int | Size of the connected component the seed came from. |
| `node_count` | int | `len(heuristic_nodes)`. |
| `file_count` | int | Distinct files spanned by `heuristic_nodes`. |

**`heuristic_nodes[]` / `agentic_nodes[]`** (object — one shared node shape, built by `node_entry`)

| Field | Type | Meaning |
|-------|------|---------|
| `id` | str | Node id (SCIP symbol). |
| `name` | str | Human-readable qualified name. |
| `kind` | str \| null | `"function"` / `"class"`. |
| `module` | str \| null | Dotted module path. |
| `file` | str | Repo-relative path (may be empty). |
| `def_line` | int \| null | 0-based definition line. |
| `is_test` | bool | Whether it is test code. |
| `documentation` | str | Docstring (may be empty). |

## `sgNN/select_agentic.json` (per slice) + `select_agentic.jsonl` (repo aggregate)

The agent's final output only — written verbatim. Execution status (`completed` /
`runtime_error`) is **not** here; it lives on the unified `select.jsonl` row. `seed` stays on
the heuristic side. The agent emits ids; the host enriches them into `agentic_nodes` objects.

| Field | Type | Meaning |
|-------|------|---------|
| `index` | int | Item id. |
| `feature` | str | One-line description of the resolved feature; empty unless `success`. |
| `agentic_nodes` | list[object] | The resolved feature's function nodes (same shape as `heuristic_nodes`, each carrying `id` — **the payload Mask consumes**); empty unless `success`. |
| `cover_rate` | float | Fraction of `agentic_nodes` that came from the heuristic slice (`final_in_slice / len(agentic_nodes)`; `0.0` when empty). Slice/feature overlap. |
| `source` | str | Provenance: `"agentic"` when the agent resolved a coherent feature, else `""` (Select has no heuristic fallback, so never `"heuristic"`). |
| `success` | bool | Business outcome: the agent resolved a coherent feature (`== (source == "agentic")`). |
| `error` | str \| null | Why nothing resolved (business, not runtime): `"abandoned: <reason>"` or `"schema: <check>"`; `null` on success. |

## `select.jsonl` (unified — Mask reads this)

The lean per-slice record: the agent's result + status, joined on `index`. Heuristic detail
(`seed` / `meta` / `heuristic_nodes`) and `cover_rate` are **not** inlined — they stay in
`select_{heuristic,agentic}.jsonl`, recoverable by joining on `index`. ★ = what Mask consumes.

| Field | Type | Meaning |
|-------|------|---------|
| ★ `index` | int | Item id — Mask joins on it. |
| ★ `source` | str | `"agentic"` when the agent resolved a feature (**Mask masks only these**), else `""`. |
| ★ `nodes` | list[object] | The resolved feature's function nodes (the agentic pass's `agentic_nodes`) — **the payload Mask consumes** (it takes the ids). Same node shape as `select_heuristic.json`; empty unless `success`. |
| `feature` | str | One-line feature description (human context; empty unless resolved). |
| `success` | bool | Deliverable outcome: a feature was resolved (`== (source == "agentic")`). |
| `error` | str \| null | Business reason nothing resolved: `"abandoned: …"` / `"schema: …"`; `null` when the agent instead crashed (that is a `warning`) or on success. |
| `warnings` | list[str] | Soft diagnostics — notably a failed agent invoke/parse/download (a crash is soft, not a `runtime_error`). |
| `completed` | bool | **Status**: no unexpected interruption on this slice. |
| `runtime_error` | str \| null | **Status**: an interruption — the heuristic subgraph build crashing or an unhandled failure; `null` for a soft agent failure (that is a `warning`). |

---

# Mask

One item = one resolved Select feature. Deterministic step (`run_mask`) = keep entry
signatures, delete helpers → draft mask/gold patch; agent step = refine that draft.

## `sgNN/mask_heuristic.json` (per feature) + `mask_heuristic.jsonl` (repo aggregate)

The `run_mask` draft result (returned via the invoke result, not downloaded). On a run
failure the row degrades to `{index, feature}`. The only sandbox file written is the raw draft
`.patch`, which the refine agent reads by path; it is not archived.

| Field | Type | Meaning |
|-------|------|---------|
| `index` | int | Item id. |
| `heuristic_mask_plan` | object | Classification of the feature + the files it touches (see below). |
| `heuristic_mask_patch` | str | **Draft** unified diff: original repo → masked (stub entries, delete helpers). |
| `heuristic_gold_patch` | str | Exact inverse of the draft `heuristic_mask_patch` (masked → original). |
| `warnings` | list[str] | Draft build diagnostics — classification + apply-mask warnings, merged (e.g. no entry points; a node skipped for a missing span). |
| `execution_time` | float | Seconds the in-sandbox build took. |

**`heuristic_mask_plan`** (object)

| Field | Type | Meaning |
|-------|------|---------|
| `nodes` | list[str] | The feature's maskable function ids (functions only, tests excluded). |
| `entries` | list[str] | Functions with a graph-observed caller outside the feature → draft keeps signature and stubs body. |
| `deletes` | list[str] | Functions with no graph-observed outside caller → draft deletes them; the refine agent may preserve important interfaces from this list. |
| `classes` | list[str] | Class ids in the feature (not masked; context only). |
| `files` | list[str] | Repo-relative files the draft touches. |

## `sgNN/mask_agentic.json` (per feature) + `mask_agentic.jsonl` (repo aggregate)

The refine agent's output (empty when `use_agent=false` / no draft / agent failed). Graph-boundary
entries are a mandatory safety minimum, but the agent may preserve additional important
caller-facing feature signatures; its direct output remains one `mask_patch`.

| Field | Type | Meaning |
|-------|------|---------|
| `index` | int | Item id. |
| `cover_rate` | float | Fraction of the refine's touched files that stayed within the feature's masked files (`1.0` = fully in scope; `0.0` when no refine). Monitoring signal, not a filter — out-of-scope filenames are logged to `warnings`. |
| `agentic_mask_patch` | str | The refine's mask diff (may be empty). `agentic_` marks it as the agent's raw output — distinct from the unified `mask_patch` (the *chosen* patch, which may instead be the draft). |
| `agentic_gold_patch` | str | Git-computed inverse of `agentic_mask_patch` (`git apply` + `git diff -R` in the sandbox); empty if the refine did not apply (then the agentic candidate is not chosen). |

## `mask.jsonl` (unified — Eval reads this)

The lean per-feature record: the chosen feature mask/gold patch (agentic refine when it applies,
else the heuristic draft when it applies) + outcome + status. `feature_` distinguishes these from
Eval's `test_` patches. `deletes` (heuristic plan) and `cover_rate` (`mask_agentic.json`) are not
inlined — join on `index` for them. A pair is committed only when it passes the **cumulative**
apply-check (mask applies to pristine, then gold applies on top of the masked tree — the
original → masked → original round-trip), so a row with no usable mask (`success=false`) leaves
`feature_mask_patch` / `feature_gold_patch` empty (the raw candidates live in `mask_heuristic` /
`mask_agentic`). ★ = the dataset payload / what Eval consumes.

| Field | Type | Meaning |
|-------|------|---------|
| ★ `index` | int | Item id — Eval joins on it. |
| ★ `nodes` | list[str] | Feature function ids (the `id`s of Select's `nodes` — here reduced to ids, not objects); Eval reads it. |
| ★ `feature_mask_patch` | str | **Chosen** feature mask diff (original → masked; **the problem state** — the dataset payload). |
| ★ `feature_gold_patch` | str | Exact inverse (masked → original; **the reference solution** — the dataset payload). |
| ★ `files` | list[str] | Files the CHOSEN `feature_mask_patch` actually touches (the real masked surface — including anything an agentic refine strayed onto). Empty when no usable mask was committed (`success=false`). |
| `source` | str | Chosen-patch provenance: `"agentic"` (refine used), `"heuristic"` (draft fallback), or `""` (no usable mask). |
| `success` | bool | Deliverable outcome `= source != ""`: a usable, gold-ready mask exists (committed only if the mask + gold round-trip passed the cumulative apply-check). Decoupled from status — a heuristic mask still counts even if the refine agent failed. |
| `error` | str \| null | Business reason not usable: `"no maskable functions in feature"` or `"no applicable mask patch …"`; `null` on success. |
| `warnings` | list[str] | Soft diagnostics — incl. a failed refine agent (soft: the heuristic still stands) and out-of-scope refine files. |
| `completed` | bool | **Status**: no unexpected interruption on this feature. |
| `runtime_error` | str \| null | **Status**: an interruption — the `run_mask` draft invoke crashing, an apply-check infra failure, or an unhandled error; `null` for a soft refine failure (that is a `warning`). |

---

# Eval

One item = one masked feature. Deterministic step (`run_eval`) = build a draft deletion patch
over **all** candidate native tests (an optional HINT, never a deliverable or fallback); the required
author/refine agent then writes TWO patches that are
**not inverses** plus one grade-only runner: `test_mask_patch` deletes EVERY existing repo test that exercises the feature
(hiding the oracle from the solver) and `test_gold_patch` is the FORWARD, generalized grade-time
oracle (black-box tests covering every substantive observable capability in the feature patches +
optional feature-independent P2P guards);
`test_runner.sh` adapts the repository's real framework/environment to structured per-test JSONL.
Semantic constraints are author self-checked and independently reviewed; the host apply-checks the
patch sequence and accepts only verifier verdict `0`. Eval writes the *test side*; the feature
mask/gold lives in `mask.jsonl` (join on `index`).

## `sgNN/eval_heuristic.json` (per feature) + `eval_heuristic.jsonl` (repo aggregate)

The `run_eval` draft result (returned via the invoke result, not downloaded). On a run
failure the row degrades to `{index}`. `run_eval` also writes BOTH raw draft `.patch` files in the
sandbox (`eval_heuristic_mask.patch` = delete all candidates, `eval_heuristic_gold.patch` = restore
them verbatim), which the Eval agent reads by path; those are not archived here.

| Field | Type | Meaning |
|-------|------|---------|
| `index` | int | Item id. |
| `heuristic_tests` | list[str] | The candidate native test ids the draft deletes (all of them). |
| `heuristic_mask_patch` | str | **Draft** diff deleting ALL candidate native tests. |
| `heuristic_gold_patch` | str | Exact inverse of the draft (restore all candidates verbatim); verifier-author context only. |
| `heuristic_test_files` | list[str] | Test files the draft touches. |
| `warnings` | list[str] | Build diagnostics. |
| `execution_time` | float | Seconds the in-sandbox build took. |

## `sgNN/eval_agentic.json` (per feature) + `eval_agentic.jsonl` (repo aggregate)

The shaped test-oracle result (empty when the required author fails). The author/refine pass writes
two RAW `.patch` files plus `eval_agentic_test_runner.sh`; after apply-check, a required independent
verifier writes `eval_agentic_verifier.json`. The host downloads and validates all artifacts.
Missing or failed agent output rejects the item; the heuristic draft is never promoted.
`agentic_` marks the author's raw output, while `agentic_verifier` records the semantic quality gate.

| Field | Type | Meaning |
|-------|------|---------|
| `index` | int | Item id. |
| `agentic_mask_patch` | str | Deletion diff hiding all feature tests the agent found (may be empty for an extra-only oracle). |
| `agentic_gold_patch` | str | Forward diff = the generalized grade-time tests + optional P2P guards (NOT the inverse of the mask). |
| `agentic_runner_script` | str | Grade-only shell adapter that executes the oracle and writes the structured JSONL protocol. |
| `agentic_test_files` | list[str] | Candidate grade targets from `patch_files(agentic_gold_patch)`; mask-only paths are excluded. |
| `agentic_verifier` | object \| null | Complete independent `verdict` / `rationale` / `findings` object. Null when verification did not complete. |

Verifier codes are stable: `0` clean; `1` oracle leakage; `2` out-of-scope/destructive;
`3` overconstrained; `4` too weak; `5` easily gamed; `6` runner integrity;
`7` non-hermetic/flaky; `-1` other material issue. Only `0` passes. For a non-zero verdict,
`findings` contains concrete evidence (all detected issues, even when `verdict` names only the
primary one).

## `eval.jsonl` (unified)

The chosen test-side oracle per feature (joins with `mask.jsonl` on `index`). A candidate is committed
to `test_mask_patch` / `test_gold_patch` / `source` only when the pair passes the **cumulative**
apply-check in grade order (test_mask → test_gold, stacked) and the independent verifier accepts it.
`test_mask_patch` and
`test_gold_patch` are **not inverses**: mask hides the existing feature tests, gold is the forward,
generalized grade oracle. The required `test_runner_script` executes pregold/postgold
and emits stable opaque test ids. Export derives
FAIL_TO_PASS / PASS_TO_PASS from the two executions (rejecting on P2F). Feature nodes are not inlined
— join `mask.jsonl` on `index`. ★ = the dataset payload the grading/export step consumes.

| Field | Type | Meaning |
|-------|------|---------|
| ★ `index` | int | Item id — joins to `mask.jsonl`. |
| ★ `test_mask_patch` | str | **Chosen** diff deleting EVERY existing repo test that exercises the feature (**hides the oracle from the solver**; may be empty for an extra-only oracle). |
| ★ `test_gold_patch` | str | **Chosen** FORWARD grade-time oracle — generalized / newly-authored black-box tests covering every substantive observable capability removed by the feature mask + optional feature-independent P2P guards. NOT the inverse of `test_mask_patch`. |
| ★ `test_runner_script` | str | Required grade-only structured runner. Rows without one fail the Eval gate. It is never exposed to the solver. |
| `verifier` | object \| null | The complete verifier object, preserved in final `eval.jsonl`: integer `verdict`, string `rationale`, and string-list `findings`. |
| ★ `test_files` | list[str] | Candidate grade targets from `patch_files(test_gold_patch)`. Export filters out paths that do not exist after patch application, then runs all remaining tests to derive F2P/P2P. |
| `source` | str | Provenance: `"agentic"` for a usable runner-backed oracle, otherwise `""`. |
| `success` | bool | True only when the patch pair applies, the structured runner is non-empty, and the independent verifier accepts the oracle. |
| `error` | str \| null | Business reason not usable, including verifier rejection; `null` on success. |
| `warnings` | list[str] | Soft diagnostics, including author/verifier invocation or artifact failures; all such failures reject the oracle. |
| `completed` | bool | **Status**: no unexpected interruption on this feature. |
| `runtime_error` | str \| null | **Status**: an interruption — an apply-check infrastructure failure or an unhandled error. Draft-build and agent failures are soft warnings, but still cannot produce an accepted oracle without clean refine + verify output. |

---

## `issue.jsonl` (unified)

One problem statement per masked+oracled feature (join Mask `success` ⋈ Eval `source` on `index`).
Agentic-only: `source` is `"agentic"` when the agent produced a non-trivial statement, else `""`.
The reference patches it was shown are NOT inlined — join Mask/Eval on `index`. ★ = what Export reads.

| Field | Type | Meaning |
|-------|------|---------|
| ★ `index` | int | Item id — joins to `mask.jsonl` / `eval.jsonl`. |
| ★ `problem_statement` | str | The GitHub-issue-style task text (non-leaking); empty unless `success`. |
| `source` | str | `"agentic"` when resolved, else `""`. |
| `success` | bool | Deliverable outcome `= source != ""` — a non-trivial problem statement (≥ 40 chars). |
| `error` | str \| null | Business reason none produced (e.g. `"problem_statement too short …"`); `null` on success. |
| `warnings` | list[str] | Soft diagnostics — incl. a failed issue agent (invoke/parse/download). |
| `completed` | bool | **Status**: no unexpected interruption. |
| `runtime_error` | str \| null | **Status**: an interruption (masked-state setup / unhandled), else `null`. |

## `export.jsonl` (unified)

One row per feature Export processed (join Mask ⋈ Eval ⋈ Issue on `index`, gated on issue `success`).
`success` = the instance graded VALID and was written to Harbor. The task itself lives under
`<output_dir>/harbor/<instance_id>/` (see the layout above).

| Field | Type | Meaning |
|-------|------|---------|
| ★ `index` | int | Item id. |
| ★ `instance_id` | str | `swe_any.<repo>.sgNN` — the Harbor task name. |
| ★ `success` | bool | Graded VALID: non-empty `fail_to_pass` and no `PASS_TO_FAIL` regression; Harbor dir written. |
| ★ `fail_to_pass` | list[str] | Tests that FAIL pre-gold and PASS post-gold (the oracle that gates the fix). |
| `pass_to_pass` | list[str] | Tests that PASS in both (guardrail). |
| `harbor_path` | str \| null | The emitted `harbor/<instance_id>/` dir (when valid), else `null`. |
| `error` | str \| null | Human-readable reason the row was not exported. |
| `error_kind` | str \| null | Machine-readable category, including runner-declared kinds plus `runner_protocol_error`, `runner_exit_error`, `target_missing`, `no_postgold_pass`, `oracle_nongating`, `gold_regression`, or `grade_setup_failed`. |
| `warnings` | list[str] | Soft diagnostics, including nonexistent test-gold paths that were ignored because other real targets remained. |
| `completed` | bool | **Status**: no unexpected interruption. |
| `runtime_error` | str \| null | **Status**: a grade-invoke / unhandled interruption, else `null`. |

The oracle is **file-level**: `test_files` (from `eval.jsonl`) is run in full, so `fail_to_pass` /
`pass_to_pass` enumerate whatever tests those files contain — restored-native and agent-added
**equally** (see [framework.md](framework.zh.md) §4b).

---

# Per-run task outputs (not files)

Each stage's `run-batch` also returns a `SweAny<Stage>Output` object per repo (surfaced to the
batch DB / export, not written under the repo dir). These are repo-level aggregates —
`score`, plus e.g. Select `num_subgraphs` / `num_agentic` / `num_abandoned` / `cover_rate_mean`,
Mask `num_features` / `num_applicable` / `num_agentic` / `num_agent_failed` / `cover_rate_mean`, Eval `num_features` /
`num_applicable` / `num_agentic` / `num_agent_failed` / `num_verifier_rejected`.
For Eval, `num_applicable` means author + apply-check + verifier all passed; `num_agent_failed`
counts author/verifier invocation or artifact failures, while `num_verifier_rejected` counts valid
negative verdicts. These are observability counters; score still follows `num_applicable > 0`. The output also carries the host path of the
unified deliverable. Issue returns `num_features` / `num_applicable` / `num_agent_failed`; Export
returns `num_features` / `num_exported` (graded-valid, written to Harbor) + `harbor_dir`. See the
`SweAny*Output` models in `task.py` for the exact fields.
