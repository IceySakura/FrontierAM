"""Shared constants for the SWE-Any pipeline (Parse → Select → Mask → Eval).

Absolute sandbox paths are pinned here and reused by the runtime-side setup,
the host-side tasks, and the Claude Code prompt template, so the summary
artifacts always land where the host expects to download them from and where a
later Mask stage expects to read them.
"""

from __future__ import annotations


# Pipeline artifacts live under one sandbox dir so a later Mask/Eval stage can
# find them after Parse/Select release control.
SANDBOX_SWE_ANY_DIR = '/workspace/swe_any'

# Repo source location inside the baked per-repo image (SWE-Env ``workdir``).
SANDBOX_REPO_DIR = '/testbed'

# Parse artifacts (single-shot per repo; exempt from the {stage}_heuristic/agent
# convention below — the graph is the shared substrate, not a per-item result).
GRAPH_FILENAME = 'parse_graph.json'
# Parse's agentic pass writes this next to the graph; Select reads it to shape the seed pool.
SEED_POLICY_FILENAME = 'parse_seed_policy.json'
INDEX_FILENAME = 'parse_index.scip'

# Per-item stages (Select / Mask / Eval) share ONE artifact layout, named
# ``{stage}_{heuristic,agentic}`` for the deterministic + agentic halves:
#   ``sgNN/{stage}_heuristic.json`` + ``{stage}_heuristic.jsonl`` (per-item + repo aggregate)
#   ``sgNN/{stage}_agentic.json``   + ``{stage}_agentic.jsonl``
#   ``{stage}.jsonl``    — repo-level UNIFIED deliverable: heuristic⋈agentic joined per item with
#                          all patch bodies inlined + the per-item status (``completed`` /
#                          ``runtime_error``). The canonical hand-off downstream reads.
# The per-``sgNN`` JSON keeps each item independently recoverable even without the aggregates.

# Select
SELECT_HEURISTIC_FILENAME = 'select_heuristic.json'
SELECT_AGENTIC_FILENAME = 'select_agentic.json'
SELECT_HEURISTIC_JSONL_FILENAME = 'select_heuristic.jsonl'
SELECT_AGENTIC_JSONL_FILENAME = 'select_agentic.jsonl'
SELECT_JSONL_FILENAME = 'select.jsonl'  # unified — Mask reads this

# Mask
MASK_HEURISTIC_FILENAME = 'mask_heuristic.json'  # run_mask draft: MaskPlan + draft mask/gold patches
MASK_AGENTIC_FILENAME = 'mask_agentic.json'  # agent-refined mask patch + its inverse gold
MASK_HEURISTIC_JSONL_FILENAME = 'mask_heuristic.jsonl'
MASK_AGENTIC_JSONL_FILENAME = 'mask_agentic.jsonl'
MASK_JSONL_FILENAME = 'mask.jsonl'  # unified — chosen mask/gold patch per feature; Eval reads this

# Eval
# Per-slice host archives (JSON), one per half — the shaped record for that half, aggregated into the
# ``*.jsonl`` below. NOT sandbox paths.
EVAL_HEURISTIC_FILENAME = 'eval_heuristic.json'  # run_eval draft archive (tests + heuristic mask/gold patch bodies)
EVAL_AGENTIC_FILENAME = 'eval_agentic.json'  # authored oracle + structured runner + verifier verdict
EVAL_HEURISTIC_JSONL_FILENAME = 'eval_heuristic.jsonl'
EVAL_AGENTIC_JSONL_FILENAME = 'eval_agentic.jsonl'
EVAL_JSONL_FILENAME = 'eval.jsonl'  # unified — test mask/gold + grade-only runner per feature
# Transient .patch files under swe_any_dir (NOT the repo), cleared after the agent call. Eval juggles
# THREE mask/gold pairs, so each patch file carries a ``{who}_{mask,gold}`` qualifier (a stage may emit
# more than one file, so the name says which):
#   - feature context, uploaded from Mask's row, read-only "what is the feature": eval_feature_{mask,gold}.patch
#   - heuristic test drafts (run_eval writes the mask; host uploads its inverse gold): eval_heuristic_{mask,gold}.patch
#   - agentic test oracle, the author WRITES both raw diffs: eval_agentic_{mask,gold}.patch
#   - independent verifier verdict (strict JSON): eval_agentic_verifier.json
EVAL_FEATURE_MASK_FILENAME = 'eval_feature_mask.patch'
EVAL_FEATURE_GOLD_FILENAME = 'eval_feature_gold.patch'
EVAL_HEURISTIC_MASK_FILENAME = 'eval_heuristic_mask.patch'
EVAL_HEURISTIC_GOLD_FILENAME = 'eval_heuristic_gold.patch'
EVAL_AGENTIC_MASK_FILENAME = 'eval_agentic_mask.patch'
EVAL_AGENTIC_GOLD_FILENAME = 'eval_agentic_gold.patch'
EVAL_AGENTIC_RUNNER_FILENAME = 'eval_agentic_test_runner.sh'
EVAL_AGENTIC_VERIFIER_FILENAME = 'eval_agentic_verifier.json'

# Issue (agentic-only: one masked+oracled feature -> a non-leaking problem statement). No heuristic
# half, so there is only the agentic archive + the unified deliverable.
ISSUE_AGENTIC_FILENAME = 'issue_agentic.json'
ISSUE_AGENTIC_JSONL_FILENAME = 'issue_agentic.jsonl'
ISSUE_JSONL_FILENAME = 'issue.jsonl'  # unified — problem_statement per feature; Export reads this
# Transient gold references the Issue agent reads BY PATH. The mask patches are already applied to the
# working tree and are not uploaded as readable context.
ISSUE_FEATURE_GOLD_FILENAME = 'issue_feature_gold.patch'  # feature_gold_patch (a reference implementation)
ISSUE_TEST_GOLD_FILENAME = 'issue_test_gold.patch'  # eval's test_gold_patch (the grading tests / acceptance)

# Export (assemble + execution-grade + Harbor emit). Deterministic; no per-item heuristic/agentic
# halves — one unified row per graded instance, plus the Harbor directory-per-instance dataset.
EXPORT_JSONL_FILENAME = 'export.jsonl'  # unified — instance_id + graded F2P/P2P + harbor path
# SWE-bench-shaped records for graded-valid instances (dataset_type='swe-any'); consumed by swe_task
# via the swe/swe_any dataset handler (reconstructs the hole at runtime init).
EXPORT_INSTANCES_FILENAME = 'swe_any_instances.jsonl'
HARBOR_DIR = 'harbor'  # <output_dir>/harbor/<instance_id>/ (directory-per-instance dataset)

# Feature-level pipeline hand-off artifacts.
FEATURE_MANIFEST_FILENAME = 'feature_manifest.jsonl'
FEATURE_RESULT_FILENAME = 'feature_result.json'


def sandbox_path(name: str, *, swe_any_dir: str = SANDBOX_SWE_ANY_DIR) -> str:
    """Join *name* onto the sandbox SWE-Any directory (POSIX join)."""
    return f'{swe_any_dir.rstrip("/")}/{name}'


def slice_dirname(index: int) -> str:
    """Return the per-slice subdirectory name (``sg00``, ``sg01``, ...)."""
    return f'sg{index:02d}'


# Sentinel for "no ceiling": large enough that no real graph reaches it, so a knob
# left at this value never constrains anything and is not stated in the prompt.
UNBOUNDED_NODES = 1_000_000_000

# Layered-BFS defaults for the Select heuristic. These knobs carry ``bfs`` in the
# middle of their name to keep them apart from the seed-pool and final-list windows.
DEFAULT_MAX_BFS_NODES = 100
DEFAULT_DROP_BFS_PROB = 0.3
DEFAULT_MIN_BFS_PER_DEPTH = 1
DEFAULT_MAX_BFS_PER_DEPTH = 8
# Repo-level slice budget. Each component is capped at ceil(size / max_bfs_nodes)
# slices; when the caps fit in this budget every component gets its cap, otherwise the
# budget is apportioned in proportion to them. So a big repo gets more slices than a
# small one instead of every repo getting the same fixed count.
DEFAULT_MAX_SUBGRAPHS = 20
# Seeds are drawn only from connected components sized within this window, so a
# slice cannot degenerate into an isolated 2-3 node island. The upper bound is
# effectively off by default: in practice a repo has ONE large component (finos
# 1188 nodes after dropping generated code, proksee 230) plus dust, so any
# concrete ceiling would empty the pool and fail Select preflight.
DEFAULT_MIN_COMPONENT_NODES = 20
DEFAULT_MAX_COMPONENT_NODES = UNBOUNDED_NODES
# Size window for the agent's `feature_node_ids` list, stated in the prompt and enforced on the
# returned agentic output. The ceiling matches max_bfs_nodes: a feature far larger than the
# slice it grew from means the agent drifted off the slice.
DEFAULT_MIN_FINAL_NODES = 1
DEFAULT_MAX_FINAL_NODES = 100
# Keep the concurrent orchestration code, but run serially until installed
# agents share one LLMRequestHandler per runtime. Multiple handlers currently
# consume the same runtime-level llm_request event and can duplicate model calls.
DEFAULT_MAX_WORKERS = 1
