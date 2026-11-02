# SWE-Any

Large-scale SWE dataset construction, integrated into nexus as ordinary
sampling tasks (runtime provider + `ctx.get_runtime` + downloaded artifacts) —
**not** an isolated sub-pipeline.

Pipeline: **Parse → Select → Mask → Eval → Issue → Export**.

> Docs:
> - [docs/framework.zh.md](docs/framework.zh.md) — shared framework (layout, outcome/status, git primitives, grading/harbor)
> - [docs/feature-implementation.zh.md](docs/feature-implementation.zh.md) — feature-implementation algorithms
> - [docs/parse.zh.md](docs/parse.zh.md) — SCIP raw index, `parse_graph.json` schema, and conversion pipeline
> - [docs/output-schema.md](docs/output-schema.md) (English) / [docs/output-schema.zh.md](docs/output-schema.zh.md) (中文) — on-disk field contract

> [!WARNING]
> The SWE-Any Harbor format adaptation is incomplete and still under testing. Do not use the
> generated `harbor/` directories for evaluation. Use `swe_any_instances.jsonl` through the
> `swe-any` dataset handler.

| Stage | What | Status |
|-------|------|--------|
| **Parse** | SCIP → entity graph (`parse_graph.json`) → agentic seed policy (`parse_seed_policy.json`) | ✅ migrated |
| **Select** | layered BFS × N → Claude Code agentic pass → `sgNN/*.json` + `*.jsonl` | ✅ implemented |
| **Mask** | graph-boundary draft → agent keeps important interfaces and deletes helpers → re-implement-it hole | ✅ implemented |
| **Eval** | agent refines/authors the hidden forward oracle and structured runner; an independent agent verifies it before acceptance | ✅ implemented |
| **Issue** | agentic-only: in a fully masked tree, shown the gold patches, write a non-leaking `problem_statement` (`issue.jsonl`) | ✅ implemented |
| **Export** | deterministic execution grading and `swe_any_instances.jsonl`; Harbor task directories remain experimental | ⚠️ Harbor adaptation under test |

Deterministic work may seed an agentic pass, but only **Mask** currently promotes a heuristic draft
when its refine is unavailable or unusable. **Select** and **Eval** fail closed without accepted
agent output; **Issue** is agentic-only; **Export** is deterministic. Parse's optional seed policy
soft-falls to no exclusions, without replacing its deterministic graph.

```
baked repo image ──► Parse: SCIP index (heuristic) ──► parse_graph.json (host)
                            seed policy (claude_code) ──► parse_seed_policy.json (host)
                                   │
        Select (one task / repo, max_workers=1 currently):
          seed pool = graph − seed_policy exclusions, kept to a component-size window
          max_subgraphs seeds (20) split over components, each capped at size/max_bfs_nodes
          per seed: heuristic (layered BFS) ──► sgNN/select_heuristic.json
                    agentic (claude_code)    ──► sandbox select_agentic.json → host sgNN/select_agentic.json
          then: parent select_heuristic.jsonl + select_agentic.jsonl, unified select.jsonl (Mask reads)
                                   │
        Mask (one task / repo; only source=='agentic' rows from select.jsonl):
          per feature: classify entry/delete → draft mask/gold ──► sgNN/mask_heuristic.json
                       refine (claude_code) + git reverse gold ──► sgNN/mask_agentic.json
          apply-check (mask→gold); choose agentic else heuristic → unified mask.jsonl (Eval reads)
                                   │
        Eval (one task / repo; only success==true rows from mask.jsonl):
          per feature: candidate leaf tests → draft delete-all ──► sgNN/eval_heuristic.json
                       AUTHOR mask + gold + runner (agent)     ──► sgNN/eval_agentic.json
                       REVIEW regression risk + weakness       ──► sgNN/eval_agentic_verifier.json
          accept only apply-checked AUTHOR output with verifier verdict 0 → unified eval.jsonl
                                   │
        Issue (agentic-only; mask.success ∩ eval.source rows):
          per feature: run_apply feature_mask+test_mask (persist, fully masked tree)
                       show gold patches → claude_code writes problem_statement ──► issue.jsonl
                                   │
        Export (deterministic, no LLM; mask ∩ eval ∩ issue rows):
          per feature: run_grade pregold(FAIL)/postgold(PASS) → derive+validate F2P/P2P
                       → harbor/<instance_id>/ (task.toml, instruction.md, environment, solution, tests)
                       + unified export.jsonl
```

## Layout

Files are grouped into four subpackages — `stage/`, `utils/`, `generator/`, `scripts/` — plus root orchestration (`task.py`, `runtime_setup.py`) and `config/`.

| Path | Role |
|------|------|
| `stage/parse/` | Parse stage: `heuristics` builds the entity graph from a SCIP index (`build_graph`, SCIP/protobuf-dependent), plus `prompts` for the seed-policy pass. The `is_test` re-tagging it runs lives in the shared `utils/classify_tests`. |
| `stage/select/` | Select stage: `heuristics` = seed pool (`seed_components` / `sample_seeds`) + layered-BFS extraction (`select_subgraph`), plus `prompts` for the slice agentic pass. |
| `stage/mask/` | Mask stage: `heuristics` keeps signatures of graph-boundary entries and deletes the rest (tests untouched) (`classify_feature`), rewriting source via `utils/source_edit` (`apply_mask`) into a draft mask/gold patch; `prompts` lets the refine agent preserve additional important feature interfaces before writing the final mask. The host derives the refined mask's matching gold patch (its git-computed inverse via `repo_git.reverse_patch` — `git apply` + `git diff -R` in the sandbox) and stores the pair as `agentic_mask_patch` / `agentic_gold_patch` in `mask_agentic.json`; the chosen patch per feature (the refine when it applies, else the heuristic draft when it applies) is collected as `feature_mask_patch` / `feature_gold_patch` in the unified `mask.jsonl` that Eval reads. |
| `stage/eval/` | Eval stage: `heuristics.find_candidate_tests` lists leaf native tests as a hint. One agent authors `test_mask_patch`, forward `test_gold_patch`, and `test_runner.sh`; a second agent classifies oracle leakage, destructive scope, over-constraint, weak/gameable tests, runner integrity, and flaky dependencies. Only verdict `0` passes. |
| `stage/issue/` | Issue stage (**agentic-only**, no `heuristics`): `prompts.build_issue_prompt` shows the agent a fully masked tree + the gold patches (`feature_gold_patch` = reference impl, `test_gold_patch` = Eval's forward grade oracle) and asks for a non-leaking `problem_statement`; `outputs` joins Mask (`success`) ⋈ runner-backed Eval (`success`) and shapes the unified `issue.jsonl`. |
| `stage/export/` | Export stage (**deterministic**, no LLM): joins Mask ⋈ Eval ⋈ Issue, execution-grades the required structured runner to derive F2P/P2P, emits a `SWEbenchInstance`-shaped record, and writes the directory-per-instance Harbor task. |
| `utils/graph/` | Lightweight, protobuf-free entity-graph model (`model`) + SCIP symbol parsing (`symbol`) + JSON IO. Shared by every stage; importing it never pulls the SCIP/protobuf toolchain. |
| `utils/scip/` | SCIP toolchain (protobuf side): `indexer` runs `scip-python` (plain subprocess); `scip_pb2` (+ `scip_proto/`) is the committed, pre-generated `.scip` protobuf bindings. |
| `utils/source_edit.py` | Graph-span source editing: stub/delete a node by its `enclosing_range` line span → unified diff + exact inverse, with git's `\ No newline at end of file` handling (`build_edit_patch`, `delete_functions_patch`). Reads repo source (so it sits outside `graph`) and is stage-agnostic — shared by `mask` (stub+delete entries/helpers) and `eval` (mask/restore the selected native tests). |
| `utils/classify_tests.py` | Python test classification over the graph. `annotate_tests` re-tags `is_test` **strictly by file path** (a test-looking name alone never qualifies); `is_test_function` is the `test_*` predicate. Protobuf-free, shared by `parse` (enrichment after the build) and `eval` (find final test cases) so neither imports the other. |
| `utils/agent_output.py` | Reads back the JSON file either agentic pass wrote, tolerating prose or fences around the object. |
| `utils/batch_config.py` | The `run-batch` **config-merge pipeline**: default+patch config merging, per-backend LLM client patches, and lifting/resolving the `instances:` spec (used by the generators). |
| `utils/instance.py` | Everything about a SWE-Env **instance**: schema readers (`load_single_instance` / `load_instances`), task-input hydration (`apply_instance`), batch selection (`selected_instances` + filters), and per-instance host paths (`repo_name_of` / `output_dir` / `host_*_path`). |
| `utils/sandbox_files.py` | Host↔sandbox single-file text transfer (`download_text_artifact` / `upload_text` / `clear_sandbox_file`). |
| `utils/repo_git.py` | Host-side sandbox git ops via `runtime.invoke`, all built from composable in-sandbox primitives: `apply_check_sequence` (cumulative `git apply`, validate-then-reset) / `apply_patches` (`run_apply`, persist — Issue's masked tree, no trailing reset) / `reset_repo` / `reverse_patch` (git-canonical inverse of an agent patch) / `grade_instance` (`run_grade`, pregold/postgold pytest runs for Export). |
| `utils/constants.py` | Shared sandbox paths, artifact filenames, and Select defaults. |
| `generator/stage/{parse,select,mask,eval,issue,export}.py` | Legacy `nexus run-batch` generators, one per stage (each gated on the prior stage's host artifact; Export skips the LLM-client patch because it has no agent). |
| `generator/pipeline/{pre_feature,feature,post_feature}.py` | Feature-pipeline batch generators, mirroring `config/pipeline/`. |
| `scripts/run_batch.swe_any_{parse,select,mask,eval,issue,export}.sh` | Batch wrappers; consume outer `DB` / `DB_PATH`. |
| `scripts/run_single.swe_any_{parse,select,mask,eval,issue,export}.sh` | One-instance wrappers over the configs; supply only what a config cannot (secrets, the instance). |
| `task.py` | Host tasks `swe_any_{parse,select,mask,eval,issue,export}` (registered in `pyproject.toml`). |
| `runtime_setup.py` | Sandbox-side targets (`run_parse` / `run_mask` / `run_eval` / `run_apply` / `run_apply_sequence` / `run_reverse_patch` / `run_grade` / `reset_repo`) invoked via `runtime.invoke`. |
| `config/` | All pipeline YAML: `stage/{parse,select,mask,eval,issue,export}.yaml`, `pipeline/{pre_feature,feature,post_feature}.yaml`, `client/<name>.yaml` LLM-backend patches, and `custom/*.yaml` example patches (`SWE_ANY_CONFIG_PATCH`). See the "Config" section below for how the layers stack. |

## Entity / relation catalogue

- **Entity** = a `function` or a `class` (SCIP definition site).
- **Relations** (edge `type`):

| ID | type | direction | evidence |
|----|------|-----------|----------|
| E01 | `enclosing` | parent → child entity | SCIP descriptor hierarchy |
| E02 | `implementation` | entity → entity | `relationships.is_implementation` |
| E03 | `reference` | function → entity | occurrence inside a function's `enclosing_range` |

## Parse (index / seed policy)

**Index — SCIP** (sandbox, no LLM). `run_parse` installs scip-python when
needed, indexes the repo and builds `parse_graph.json`, which the host downloads.

**Seed policy — Claude Code** (host+agent, same sandbox). The host builds a
per-directory profile of the graph and asks the agent which paths must not be
used as a *starting point* for feature extraction — generated clients, vendored
trees, build output. The agent writes `parse_seed_policy.json`; the host reads back the
globs and rewrites the archive from them:

```json
{"exclude_globs": ["symphony/bdk/gen/*"],
 "exclude_reasons": {"symphony/bdk/gen/*": "OpenAPI-generated models and API clients"},
 "notes": "Hand-written logic lives under symphony/bdk/core."}
```

Excluded paths stay in the graph — generated code is a bad place to *start* a
slice but a legitimate part of the feature a slice grows into. The globs are
applied as written: an over-broad policy is not second-guessed here, it just
leaves Select's seed pool small enough that Select runs fewer slices.

This pass is soft in every direction: agent failure, a missing file or
unparseable JSON all leave `parse_graph.json` intact and record `seed_policy_error` on
the output. Select then runs unfiltered rather than paying for a re-index. It
always runs — Parse is never index-only — so a Parse run needs LLM credentials
like Select does.

## Select (heuristic / agentic)

**Seed pool.** Seeds are always sampled — there is no explicit-seed input; use
`rng_seed` to reproduce a run. The pool is the filtered graph minus the seed
policy's `exclude_globs`, restricted to connected components sized within
`[min_component_nodes, max_component_nodes]` (default `[20, 1e9]`), measured
*after* the exclusions split the graph. Peripheral islands — a util plus its
only test, or a mock whose real callers are all external — otherwise yield 2-3
node slices that the agent cannot turn into a feature.

Draws are uniform over the nodes of a component, so a package contributes seeds in
proportion to its entity count. The component upper bound is effectively off by
default: a real repo has one large component plus dust (finos: 1141 nodes after
dropping generated code, proksee: 230), so a concrete ceiling empties the pool and
fails preflight. Each slice records its `meta.component_size` for post-hoc analysis.

**Slice count is derived, not requested.** Each eligible component is capped at
`ceil(size / max_bfs_nodes)` slices — the number of disjoint full-size slices it could
hold — and `max_subgraphs` (default 20) is a repo-level budget those caps compete for.
When the caps fit in the budget every component gets its cap; otherwise the budget is
apportioned in proportion to them (largest-remainder, so the total lands exactly on
`max_subgraphs`), and a component can get 0 when there are more components than
slices. So finos yields 20 slices and proksee 4, instead of both being asked for the
same fixed number. Draws inside a component are without replacement, since two
identical seeds only buy two identical agentic calls.

**Heuristic — layered BFS** (host, no LLM). Its knobs carry `bfs` in the middle of
their name (`max_bfs_nodes`, `drop_bfs_prob`, `min_bfs_per_depth`,
`max_bfs_per_depth`) to keep them apart from the seed-pool and final-list windows.
Per depth: shuffle → `drop_bfs_prob` → raise to `min_bfs_per_depth`
(take-as-available) → cap `max_bfs_per_depth` → global `max_bfs_nodes`.
Scanned-but-dropped nodes stay in a permanent `used` set, including nodes removed by
caps. This intentionally favors smaller, more diverse slices over revisiting overflow
from hub layers just to fill `max_bfs_nodes`. Defaults keep tests
(`include_tests=true`).

**Agentic pass — Claude Code** (`agent_name=claude_code`, one-shot `--print`). Before
the agent runs, Select materializes a working graph from Parse's full
`parse_graph.json` using `include_tests` / `include_external`. Heuristic BFS and the
agent both see only that filtered view (uploaded once into the sandbox), and the
prompt states which symbol classes were removed so the agent knows every `feature_node_ids`
id must come from that file. The agent writes one agentic JSON per slice to a
pinned sandbox path (`/workspace/swe_any/select_agentic.json`). The host
downloads each file, validates/normalizes into `sgNN/select_agentic.json`, then appends
a parent JSONL. Intermediate agent traces go through processors (e.g.
`chat_tracker`).

The prompt allows exactly two outcomes. Either the slice resolves into one
coherent feature, or the agent **abandons** it with a one-sentence reason — it is
explicitly told not to go hunting for some other feature to report, because a
slice of generated boilerplate should cost one cheap look, not a repo-wide
search. Drift counts as abandonment too: the prompt says that a `feature_node_ids` list
sharing few or none of its functions with the candidates means the agent walked
onto a different feature, and must be reported as unresolved however coherent
that other feature is. Anything short of a resolved feature archives empty
`feature` / `agentic_nodes`. (The agent emits ids; the host enriches the survivors into
`agentic_nodes` objects — the same node shape as the heuristic `heuristic_nodes`.)

`min_final_nodes` / `max_final_nodes` (default `[1, 100]`, the ceiling matching
`max_bfs_nodes`) bound how many functions `feature_node_ids` may hold. The active bounds are
stated in the prompt — which also tells the agent to abandon rather than pad or
truncate a feature to fit — and re-checked on the returned agentic output, after dedupe
and id validation. A list outside the window is a schema failure like any other: the
row keeps `completed=true` but `success=false` and records `error: schema: feature_node_ids
size N outside a..b`.

Select preflight errors (missing/corrupt graph, empty filtered graph, or an empty
seed pool) propagate as input/configuration failures. Only per-slice
heuristic/agentic failures are soft and retained in the output archives.

Archives (one repo):

- `output_dir/<repo>/sgNN/select_heuristic.json` — `seed` + `meta` (knobs, `component_size`) + `heuristic_nodes`
- `output_dir/<repo>/sgNN/select_agentic.json` — the agent's final output only (fields below), pure content
- `output_dir/<repo>/select_heuristic.jsonl` / `select_agentic.jsonl` — per-half aggregates (one object / line)
- `output_dir/<repo>/select.jsonl` — **unified** deliverable (Mask reads this): the lean per-slice record — the agent's result (`source` / `nodes` / `feature` / `success` / `error`) + the per-item status. Heuristic detail (`seed` / `meta` / `heuristic_nodes`) and `cover_rate` are not inlined — join on `index` for them.

The per-half `sgNN/*.json` + `*_agentic.jsonl` are pure content; the per-item status lives on the
unified `select.jsonl` row. The two failure channels stay separate — *the run failed* (status) vs
*there was nothing to find* (content):

| Field | Where | Meaning |
|-------|-------|---------|
| `success` | agentic + unified | Outcome channel: the agent resolved one coherent feature (only then are `feature` / `nodes` non-empty). `success ⟹ completed` |
| `source` | agentic + unified | Provenance, kept in sync with `success`: `agentic` when resolved, else `""` (Select has no heuristic fallback) |
| `error` | agentic + unified | Outcome channel: why nothing resolved (business, not runtime) — `abandoned: <reason>` or `schema: <check>`; set only when `completed` but not `success` |
| `nodes` | unified (← agentic `agentic_nodes`) | The resolved feature's function nodes (objects); Mask takes their ids. Empty unless `success` |
| `cover_rate` | agentic only | Fraction of `agentic_nodes` that came from the heuristic slice (`final_in_slice / len(agentic_nodes)`); not inlined in unified |
| `completed` | unified | Status channel: no runtime error on this slice (agent / download / JSON parse) |
| `runtime_error` | unified | Status channel: that runtime failure, if any |

`cover_rate` measures slice/feature overlap at the id level, which also tracks how
dense the slice was: in the current smoke the resolved features scored 0.17-0.39 there
while ~100% of their functions still lived in files the slice already covered. Judge a
future BFS variant on both numbers, not on this one alone.

Aggregated on the task output as `num_completed` / `num_agentic` /
`num_abandoned` / `cover_rate_mean` (mean `cover_rate` over agentic slices).

```python
from nexus.extensions.tasks.swe_any.utils.graph import read_json
from nexus.extensions.tasks.swe_any.stage.select import sample_seeds, seed_components, select_subgraph

graph = read_json('parse_graph.json')
components = seed_components(graph, set(graph.nodes), exclude_globs=['pkg/gen/*'])
seeds = sample_seeds(graph, components=components, max_bfs_nodes=100, max_subgraphs=20)
slice_ = select_subgraph(graph, seed=seeds[0], drop_bfs_prob=0.3, min_bfs_per_depth=1, max_bfs_per_depth=8)
```

## Mask (heuristic draft / agentic refine)

**Input gate.** Only `select.jsonl` rows with `source == 'agentic'` and non-empty `nodes`
are masked (`load_resolved_features`). Cap with `max_features` (0 = all).

**Heuristic — entry stub + helper delete** (sandbox `run_mask`, no LLM). Feature functions
with outside callers keep their signature and get a `raise NotImplementedError` body
(*entries*); internal-only helpers are deleted entirely (*deletes*). Tests are never
masked. The draft pair is `heuristic_mask_patch` (original → hole) +
`heuristic_gold_patch` (exact inverse from the before/after source).

**Agentic refine — Claude Code.** The agent reads the Select feature summary, draft
`.patch`, repo source, tests, and docs. Graph-boundary entries are a mandatory safety
minimum; the agent may also keep signatures of important caller-facing feature interfaces,
then deletes only genuine implementation helpers and writes one refined `mask_patch`.
Out-of-scope files are logged (`cover_rate` / `warnings`) but do not reject the patch.
The host then derives `agentic_gold_patch` via sandbox `reverse_patch` (`git apply` +
`git diff --cached -R`); if the refine itself does not apply, gold is empty and the
agentic candidate is discarded.

**Choose + apply-check.** Priority: agentic pair when both patches are non-empty and the
cumulative apply-check passes (`pristine → feature_mask → feature_gold → reset`); else the
heuristic pair when it applies; else `source=""`, `success=false`. Soft refine failures
fall back to the draft (`warnings`, `completed=true`). The unified `mask.jsonl` row carries
the chosen `feature_mask_patch` / `feature_gold_patch` plus `files` (the files the
**chosen** mask touches). Eval reads the patches themselves as the authoritative feature scope.

Archives (one repo):

- `output_dir/<repo>/sgNN/mask_heuristic.json` / `mask_agentic.json`
- `output_dir/<repo>/mask_heuristic.jsonl` / `mask_agentic.jsonl`
- `output_dir/<repo>/mask.jsonl` — **unified** deliverable (Eval reads this)

Aggregated on the task output as the funnel `num_features ≥ num_applicable ≥ num_agentic`
plus `cover_rate_mean` over agentic features.

```python
from nexus.extensions.tasks.swe_any.utils.graph import read_json
from nexus.extensions.tasks.swe_any.stage.mask import classify_feature
from nexus.extensions.tasks.swe_any.utils.source_edit import build_edit_patch

graph = read_json('parse_graph.json')
plan = classify_feature(graph, feature_node_ids)  # entries / deletes / files
# build_edit_patch(repo_dir, stubs=plan.entries, deletes=plan.deletes) → mask + gold
```

## Eval (heuristic draft / agentic test-oracle)

**Input gate.** Only `mask.jsonl` rows with `success == true` (a committed, apply-checked
feature mask) are eval targets — **not** merely non-empty `files` (that list can stay
populated from the heuristic plan even when no patch applied).

**Heuristic — delete-all candidates** (sandbox `run_eval`). For each feature function,
`find_candidate_tests` walks direct `callers` and keeps leaf `test_*` functions in test
files (helpers that other tests call are dropped). Default `include_feature_tests=true`
keeps tests Select pulled into the feature (Mask left them intact). The draft deletes
**all** candidates → `heuristic_mask_patch` / `heuristic_gold_patch`; it is a HINT the
agent may extend, not an independently exportable oracle.

**Agentic pass — MASK all + author the GOLD + runner.** Reading the feature mask/gold patches
(by path, read-only context), the agent authors TWO patches that are **not inverses** plus
`test_runner.sh`:
`test_mask_patch` deletes EVERY existing repo test that exercises the feature (beyond the
candidate hint — it searches the repo, may drop whole classes/files), and `test_gold_patch`
is the forward, generalized grade-time oracle (black-box tests covering every substantive
observable capability in the feature patches + optional feature-independent P2P guards). The runner selects the real repository environment
and emits stable per-test JSONL. Semantic constraints (scope, coverage, F2P≠∅,
no-P2F) are prompt-level and author-self-verified; the host adds apply-check and the independent
semantic verifier. Extra-only
(empty mask, feature with no native tests) is legal.

**Independent verifier gate.** After the patch pair passes apply-check, a second agent reads the
full feature mask/gold, authored test mask/gold, runner, and repository. It emits one stable integer
verdict plus rationale/findings: `0` clean; `1..7` known hack/quality failures (leakage, destructive
scope, over-constraint, weakness, easy gaming, runner integrity, flakiness); `-1` other. Its strict
JSON verdict is host-validated; only `0` passes.

**Choose + apply-check.** `test_gold` grades, so agentic is usable only when `test_gold` and the runner
are non-empty, the pair passes `pristine → test_mask → test_gold → reset`, and the independent
verifier returns verdict `0`.
Unified `eval.jsonl` stores `test_mask_patch` / `test_gold_patch`, the required `test_runner_script`,
the verifier verdict, and `test_files = patch_files(test_gold)` — mask-only paths are solver-hiding inputs, not grade
targets. Export keeps only files that actually exist after applying the patches, then runs every
test in those files. The runner emits structured stable ids used to derive F2P / P2P.

Archives (one repo):

- `output_dir/<repo>/sgNN/eval_heuristic.json` / `eval_agentic.json` / `eval_agentic_verifier.json`
- `output_dir/<repo>/eval_heuristic.jsonl` / `eval_agentic.jsonl`
- `output_dir/<repo>/eval.jsonl` — **unified** deliverable (Export reads this)

The Eval output also reports `num_verifier_rejected` separately from verifier invocation failures.

```python
from nexus.extensions.tasks.swe_any.utils.graph import read_json
from nexus.extensions.tasks.swe_any.stage.eval import find_candidate_tests

graph = read_json('parse_graph.json')
candidates = find_candidate_tests(graph, feature_node_ids, include_feature_tests=True)
```

## Tasks

```bash
uv run nexus modules install swe_any
```

Workload image = SWE-Env instance `image` via `SWE_ANY_IMAGE`; repo at
`SWE_ANY_REPO_DIR` (default `/testbed`). **scip-python indexes Python only.**

There are two entry points. `run_single.*.sh` drives one instance through
`config/stage/*.yaml` (whose `oc.env` defaults are that path's control
panel); `run_batch.*.sh` drives a whole JSONL through the generators.

```bash
# One instance through the full chain (Parse → Select → Mask → Eval), one-row JSONL
SWE_ANY_INSTANCE_PATH=path/to/one_row.jsonl \
  bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_parse.sh
SWE_ANY_INSTANCE_PATH=path/to/one_row.jsonl \
  bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_select.sh
SWE_ANY_INSTANCE_PATH=path/to/one_row.jsonl \
  bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_mask.sh
SWE_ANY_INSTANCE_PATH=path/to/one_row.jsonl \
  bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_eval.sh
```

### Config: default + custom, and env just fills the YAML

Both entry points share one base — the stage's `config/stage/*.yaml` — and
`run-batch` deep-merges over it any patches you name in `SWE_ANY_CONFIG_PATCH`.
Patches are **parallel groups** (a budget, a backend, …) under `config/custom/`:
stack as many as you like, comma-separated, applied left→right, **later wins**. A
bare name resolves under `config/custom/` (`smoke` / `smoke.yaml` both hit
`config/custom/smoke.yaml`), else a relative/absolute path. Merge is RFC 7386:
dicts recurse, `null` deletes a key (e.g. `processors: {chat: null}`),
lists/scalars replace.

```bash
SWE_ANY_CONFIG_PATCH=smoke.yaml              # default + cheap budget
SWE_ANY_CONFIG_PATCH=smoke.yaml,vertex.yaml  # + vertex backend
```

`config/client/<name>.yaml` holds each backend's transport bits (model name,
`thinking`, …) and is applied automatically from the chosen
`openai_clients.main.name` (which a patch like `vertex.yaml` sets) — you never list
it, and adding a backend is one new file. `run-single` uses only `default/*.yaml`
(one instance, no stacking).

**env just fills the YAML.** Every value is `${oc.env:VAR,default}`: the YAML holds
the default and exporting `$VAR` overrides it at every layer at once
(`SWE_ANY_MAX_SUBGRAPHS=4 bash run_batch...sh`). Secrets (`GONGFENG_TOKEN`,
`OPENAI_API_KEY`) are the exception — no default, so a missing one fails loudly.
Which instances a batch runs is the `instances:` block (`path` / `ids` /
`language` / `skip` / `limit`) in the YAML, same env rule (`SWE_ANY_LIMIT=1` wins;
the seed draw is `task_config.rng_seed`, a run knob, not an `instances` field).

The `run_batch` wrappers carry only what must exist before Python starts: secrets,
the instances file, `SWE_ANY_OUTPUT_DIR` (DB/export paths are built from it in
shell) and `MAX_CONCURRENT` / `WORKER_PROCESSES`.

```bash
# Whole pipeline: outer script → all six batch .sh in order (Parse → Select → Mask → Eval → Issue → Export)
SWE_ANY_INSTANCES=path/to/instances.jsonl bash nexus/extensions/tasks/swe_any/scripts/run_swe_any.sh
# or pass the instances file as $1:
bash nexus/extensions/tasks/swe_any/scripts/run_swe_any.sh path/to/instances.jsonl
# run a subset (earlier artifacts must already be on disk):
STAGES="issue export" bash nexus/extensions/tasks/swe_any/scripts/run_swe_any.sh path/to/instances.jsonl
```

All six stages at scale — `run_swe_any.sh` just chains each `run_batch.swe_any_<stage>.sh`, and each
inner `.sh` runs `nexus run-batch` with its own per-stage DB under `SWE_ANY_OUTPUT_DIR`
(all key `run_id` on `instance_id`). Stages share one `SWE_ANY_OUTPUT_DIR` so each reads the prior
stage's host artifacts. A stage's non-zero exit (expected per-instance HARD failures at scale) is
logged and the chain continues — only the instances that produced artifacts flow downstream.
`GONGFENG_TOKEN` is always required; `OPENAI_API_KEY` whenever a non-Export (LLM) stage runs.

```bash
# Parse batch (index + seed policy)
SWE_ANY_INSTANCES=./instances.jsonl \
DB_PATH=./outputs/swe_any/parse_batch.db \
MAX_CONCURRENT=10 WORKER_PROCESSES=5 \
  bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_parse.sh

# Select batch (needs host parse_graph.json from Parse under SWE_ANY_OUTPUT_DIR;
# picks up parse_seed_policy.json from the same dir when present)
SWE_ANY_INSTANCES=./instances.jsonl \
DB_PATH=./outputs/swe_any/select_batch.db \
MAX_CONCURRENT=4 WORKER_PROCESSES=2 \
  bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_select.sh

# Mask batch (needs host select.jsonl + parse_graph.json under SWE_ANY_OUTPUT_DIR)
SWE_ANY_INSTANCES=./instances.jsonl \
DB_PATH=./outputs/swe_any/mask_batch.db \
MAX_CONCURRENT=4 WORKER_PROCESSES=2 \
  bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_mask.sh

# Eval batch (needs host mask.jsonl + parse_graph.json under SWE_ANY_OUTPUT_DIR)
SWE_ANY_INSTANCES=./instances.jsonl \
DB_PATH=./outputs/swe_any/eval_batch.db \
MAX_CONCURRENT=4 WORKER_PROCESSES=2 \
  bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_eval.sh
```

`WORKER_PROCESSES` (default 8, as in SWE-Cosmos) shards the batch across processes and
**splits** `MAX_CONCURRENT` between them, so it buys host-side parallelism rather than
more concurrent sandboxes — the shard count is `min(WORKER_PROCESSES, MAX_CONCURRENT,
instances)`. A sharded DB cannot be resumed with `WORKER_PROCESSES=1`, so keep the value
stable for a given `DB_PATH`.

### Feature-level pipeline

`scripts/run_swe_any_feature_pipeline.sh` is an additive launcher that removes
global agent-stage barriers:

> [!NOTE]
> The current `pipeline/` package is a purpose-built scheduling layout for
> **feature-level parallelism**, not the final general pipeline abstraction. It
> deliberately encodes SWE-Any's current stage handoffs and artifact conventions
> so each feature can advance independently. A later refactor will extract a
> reusable handoff/execution protocol for other pipeline shapes.

1. `swe_any_pipeline_pre_feature` runs Parse and deterministic Select heuristic per repo.
2. `swe_any_pipeline_feature` runs one manifest slice through Select agentic, Mask, Eval,
   Issue, and Export in one reused sandbox.
3. `swe_any_pipeline_post_feature` deterministically reconstructs repo aggregates and the
   global dataset, including explicit blocked rows for failed features.

The legacy six-stage launcher remains available.

## Notes

- **Repo source = baked image.** No whole-repo upload at runtime.
- **Parse deps** (`networkx` / `protobuf`) ship via `worker_venv: {modules: ["swe_any"]}`.
- **`scip/scip_pb2.py`** is committed; regenerate with `grpc_tools.protoc` from `scip/` (see earlier docs).
- **Concurrency guard (Select / Mask / Eval):** the semaphore/gather orchestration remains, but
  `max_workers` is temporarily required to be `1`. Concurrent Claude Code calls in one runtime
  currently register multiple handlers for the same `llm_request` event and can duplicate model
  calls. Non-1 values fail immediately with an explanatory error.
- **Timeouts:** provider `request_timeout` must exceed Claude `agent_timeout` (rule of thumb: `+600s`).
  Parse holds one sandbox for indexing *and* the seed-policy agent, so its floor is
  `index + npm install + agent + 600s`; the `.sh` wrappers and the batch generator compute this.
- **`is_test` is path-based** (`utils/classify_tests.annotate_tests`): only files under a
  `test`/`tests`/`testing` directory segment (or `test_*.py` / `*_test.py` / `conftest.py`)
  count. A `test_*` function name alone never qualifies. SCIP's Test role is recorded as
  evidence only.
- **Downstream gates:** Mask reads only `select.jsonl` rows with `source=='agentic'`; Eval reads
  only `mask.jsonl` rows with `success==true` (committed apply-checked feature mask).
- **Issue / Export:** Issue generates the non-leaking problem statement; Export execution-grades
  the assembled instance and emits its Harbor task directory.
