# Benchmarks

Benchmark assets live under `eval/`.

All paths below are relative to the repo root (the directory that contains the top-level `nexus/` folder).

## HLE

### 1) Dataset

Use either:
- HLE parquet dataset root (contains `data/test-*.parquet`)
- Canonical JSONL file

HLE dataset (Hugging Face):
- Dataset page:
  ```
  https://huggingface.co/datasets/cais/hle
  ```
- Pull via `datasets`:
  ```bash
  python - <<'PY'
  from datasets import load_dataset
  load_dataset("cais/hle", split="test")
  PY
  ```

HLE-200 dataset (aligned with HLE, internal repo master branch):
- Repo:
  ```bash
  git clone git@git.woa.com:felixjjiang/hle_200.git
  cd hle_200
  git lfs pull
  ```
- Dataset path:
  ```
  hle_200/data/test-00000-of-00001.parquet
  ```

HLE-200 source dataset (internal repo data/hle-200-jsonl branch):
- Local source JSONL with the built-in HLE adapter:
  ```bash
  cd "$(git rev-parse --show-toplevel)"
  git fetch git@git.woa.com:felixjjiang/hle_200.git data/hle-200-jsonl
  mkdir -p nexus/extensions/tasks/stem_mas/benchmark/data
  git show FETCH_HEAD:hle_200.jsonl > nexus/extensions/tasks/stem_mas/benchmark/data/hle_200.source.jsonl
  ```

### 1.5) Runtime Image

Use the normal Nexus runtime image. Benchmark configs set `runtime_configs.main.worker_venv: true`, so Python changes
under `nexus/` are uploaded into a fresh worker venv without rebuilding the base runtime image.

Rebuild the runtime image only when changing `deployment/runtime`, base runtime dependencies, or image-level system
packages. Run the evaluation on an IDC machine, or inside a Taiji job, with dataset/config paths and required env vars
available in the job workspace.

### 2) Config

Edit the single base config:
- `nexus/extensions/tasks/stem_mas/benchmark/eval/config/base.yaml`

Required env vars:
- `SEARCH_USER` / `SEARCH_API_KEY` (search tool)
- `SCIENCE_API_KEY` (LLM)
- `CODE_INTERPRETER` (must be `"none"`, `"python"`, or `"jupyter"`)

Example bash exports:

```bash
# Base (required)
export SEARCH_USER="your_user_id"
export SEARCH_API_KEY="your_search_api_key"
export SCIENCE_API_KEY="your_llm_api_key"
export CODE_INTERPRETER="none"  # or "python" / "jupyter"

# Optional: override the base runtime image
export STEM_MAS_IMAGE="mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest"
export STEM_MAS_RUNTIME_IMAGE="mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest"

# Optional: Taiji judge (only when using eval/config/hle/judge_gpt_oss.yaml)
export JUDGE_MODEL="gpt-oss-120b-sci-128k-new_daisyyi_for_RL_R2_zwfy"
```

### 3) Run

Copy `runners/benchmark.env.example` to ignored `runners/benchmark.env`, fill credentials and
`STEM_MAS_DATASET_PATH`, then choose one experiment config:

- Codex: `CONFIG=nexus/extensions/tasks/stem_mas/benchmark/eval/config/hle/codex_v7.yaml`
- Claude Code: `CONFIG=nexus/extensions/tasks/stem_mas/benchmark/eval/config/hle/claude_code_v7.yaml`
- main_agent: `CONFIG=nexus/extensions/tasks/stem_mas/benchmark/eval/config/hle/main_agent_api.yaml`

Reusable bash templates live under `runners/`:
- `runners/run_batch_template.sh`: run locally or inside a platform job with `nexus run batch`.
- `runners/submit_platform_template.sh`: submit the current checkout to a Ray Jobs platform and run the batch template.
- `runners/benchmark.env.example`: copy to ignored `runners/benchmark.env`, then fill credentials/paths.

Background run (nohup):

```bash
mkdir -p outputs
nohup bash nexus/extensions/tasks/stem_mas/benchmark/runners/run_batch_template.sh \
  > outputs/hle_200_taiji_judge.log 2>&1 &

tail -f outputs/hle_200_taiji_judge.log
```

### 4) Parameters

- `ENV_FILE` / `CONF_PATH`: optional override for the env file. By default runner scripts source
  `runners/benchmark.env`.
- `CONFIG`: benchmark config path. If unset, the runner uses `eval/config/base.yaml`.
- `STEM_MAS_DATASET_PATH`: HLE-200 parquet file/root or JSONL path. The SOTA configs resolve `dataset.path`
  from this env var and fail fast when it is empty or missing.
- `--database`: SQLite checkpoint DB.
- `--export-path`: JSONL export path.
- `--max-concurrent`: max concurrent tasks.

### 4.5) Codex MCP routing

Keep the Codex HLE config pinned when reproducing historical HLE runs. The current legacy-alignment version is:

```bash
export CODEX_VERSION=0.131.0-alpha.22
```

Do not rely on `CODEX_VERSION=latest` for reproducibility. On this machine, a full Codex HLE-200 run with `latest`
has produced malformed MCP call names such as `search_toolssearch` in Codex stderr.

Historical platform probes in this checkout also showed that MCP routing is Codex-version-sensitive: Codex
`0.131.0-alpha.22` emitted real `mcp__search_tools__search` and `mcp__search_tools__open` calls without
`unsupported call` errors. Codex `0.114.0` should be treated as a separate comparison because it has shown
sub-agent-unavailable and routing-risk behavior in prior probes. For legacy result alignment, prefer
`0.131.0-alpha.22` plus the config prompt guard that treats planner/researcher/solver/etc. as prompt labels rather
than unlisted `agent_type` values. If `search_toolssearch` or `unsupported call` appears, treat it as a known
version/routing risk to track rather than silently changing the benchmark prompt.

For current fast Codex HLE-200 alignment tests after raising the DeepSeek Flash route capacity, use
`MAX_CONCURRENT=20` as a starting point for a single full-200 run, then tune against current platform throttling and
runtime-container capacity.

### 5) Output

- JSONL: one result per line at `--export-path`.
- SQLite DB: run state and checkpoints at `--database`.

### 6) Resume

- Re-run with the same `--database` to resume and skip succeeded items.
- Set `RETRY_FAILED=2` in the runner env to retry failed rows after the initial pass.
- For retry-only / no-resume behavior, use `nexus run batch` with `--retry-only` or `--no-resume`.

## Dataset Item Schema

Dataset adapters normalize source rows directly into Nexus `DatasetItem` objects. Each item contains only portable,
benchmark-owned data:

- `id`: stable benchmark task id.
- `task_config.question` and `task_config.prompt`: normalized task input.
- `task_config.answer`: reference answer or proof.
- `task_config.evaluator_type`: evaluator route owned by the dataset contract when required.
- `task_config.metadata`: source identity, split, verifier contract, and other benchmark metadata.

Model credentials, agent selection, runtime settings, retry policy, timeouts, and cache routing are run-owned fields and
must not be stored in a published dataset. Dataset-specific conversion belongs to its adapter. For example,
`HLEDataset` maps `origin_question` / `ref_answer` directly to a `DatasetItem`; math adapters additionally attach the
Lean verifier contract. Hytuner is an optional one-way export format derived from a `DatasetItem`, not the evaluation
execution model; its `reward_model.ground_truth` is derived from canonical `task_config.answer` during export.

## Math Benchmarks

The math route is separate from HLE. The benchmark is split into three layers:

1. Benchmark contract: dataset rows, prompt/answer shape, task id mapping, and official verifier route.
2. Solver strategy: `no_agent`, `codex`, `codex_lean`, or a future official/paper solver.
3. Tools and services: Lean LSP MCP, search/retrieval, and the final Lean4 verifier.

All currently supported math datasets in this directory are formal Lean tasks. The final score must come from the
Lean4 verifier route, not from an LLM judge. Lean MCP is optional solving assistance for solver strategies such as
`codex_lean`; it is not the scoring authority.

### Official Alignment Boundary

Official benchmark evaluation usually defines the dataset, accepted candidate shape, and verifier/scoring rule. It
does not require a specific solver implementation. Paper or model-report scores may use stronger solver algorithms:
pass@k sampling, search, retrieval, Lean feedback, refinement loops, or specialized provers.

Current `stem_mas` math configs implement a direct pass@1-style baseline:

```text
dataset row -> benchmark prompt profile -> one solver output -> Lean4 verifier
```

Future official-method comparisons should be added as solver strategies at the same level as `no_agent`, `codex`, and
`codex_lean`; they should still emit the same candidate shape and be judged by the same verifier path. Do not put solver
algorithm logic into dataset adapters or evaluators.

The current RunPart profiles emit one candidate per dataset item. References to pass@4, pass@8, or pass@32 below
describe planned repeated-sampling runs; repeated datasets must use stable sample-suffixed item IDs before those
numbers can be reported with the benchmark's pass@k estimator.

### Dataset Inventory

`stem_mas` has a math dataset adapter for the following names:

- `minif2f`
- `putnam672`
- `putnam-2025`
- `fatem` / `fateh` / `fatex`

The adapter accepts `.jsonl`, `.json`, `.parquet`, or one-problem-per-file `.txt` inputs. Each dataset adapter owns its
conversion into the shared Nexus `DatasetItem` contract. Formal Lean rows are routed to `lean4_verifier`; non-formal
rows are treated as unscored unless a future dataset adapter defines a separate evaluator policy.

Default Hugging Face / authority sources and intended evaluator route:

| Dataset | Primary source | Count | Difficulty / scoring route |
| --- | --- | ---: | --- |
| `minif2f` | `roozbeh-yz/miniF2F_v2`, file `miniF2F_v2c.jsonl`, pinned HF revision | 488 rows (244 valid + 244 test) | Formal Lean route. The current matrix intentionally runs all v2c rows; report the split and do not compare the 488-row percentage directly with traditional 244-row miniF2F-test leaderboards. |
| `putnam672` | `ChristianZ97/PutnamBench-lean4`, file `putnam_bench_lean4.jsonl`, pinned HF revision | 672 rows | Formal PutnamBench source, including the 12 Putnam 2025 statements. The current verifier contract is a task-bound lower-bound diagnostic, not yet an official PutnamBench score; see the candidate-contract notes below. |
| `putnam-2025` | `ChristianZ97/PutnamBench-lean4`, file `putnam_bench_lean4.jsonl` filtered to `putnam_2025_*` | 12 rows | Formal Putnam 2025 subset. It currently has the same partial-hole diagnostic boundary as `putnam672`, so do not report it as an official pass@k score yet. |
| `fatem` | Official `frenzymath/FATE-Eval` GitHub fallback, file `data/FATE-M.json` | 150 rows | Formal Lean route. Score with FATE-Eval's Lean verifier / pass@k. No Hugging Face mirror was found, so this default source is explicitly non-HF. |
| `fateh` | Official `frenzymath/FATE-Eval` GitHub fallback, file `data/FATE-H.json` | 100 rows | Formal Lean route. Score with FATE-Eval's Lean verifier / pass@k. |
| `fatex` | Official `frenzymath/FATE-Eval` GitHub fallback, file `data/FATE-X.json` | 100 rows | Formal Lean route. Score with FATE-Eval's Lean verifier / pass@k. |

External baselines are useful for calibration only. In particular, reports of DeepSeek-V4-Flash improving on
PutnamBench with Lean verifier plus Mathlib search describe a broader Goedel-Architect-style loop with blueprinting,
Lean compiler feedback, and Mathlib refinement. That is an upper-bound reference, not the same as ordinary
multi-agent solving plus a single MCP endpoint.

Evaluator policy is dataset-owned. Official-style reporting follows the benchmark:

- formal Lean datasets (`minif2f`, `putnam672`, `putnam-2025`, `fatem`, `fateh`, `fatex`) -> Lean verifier /
  compile success.

Current implementation status: the math dataset adapter records `score_route`, `reference_quality`,
`verifier_backend`, `lean4_verifier_benchmark`, `lean4_verifier_task_id`, and `lean_prompt_profile` in each formal
sample's metadata. Formal math configs default to `evaluator_type=lean4_verifier`, which submits the final proof body
or complete Lean file required by the dataset contract to `lean4-verifier-v0d2d0.production.polaris`.

Deprecated natural-language Putnam routes and LLM-judge math routes are not part of the current formal math matrix.
Use the Lean4 verifier route for these datasets. If a future dataset is truly natural-language only, add a separate
dataset adapter and evaluator policy instead of overloading the formal math path.

The candidate contract remains benchmark-specific so results stay comparable with the validated experiment routes:

- `minif2f` generates and verifies a manifest-bound `proof_body`.
- `fatem`, `fateh`, and `fatex` generate official-style complete Lean code; the client deterministically locates the
  target declaration and submits its self-contained proof body to the pinned FATE-Eval manifest. This is sound but
  stricter than FATE-Eval's full-code static-precheck pipeline because helper declarations are not submitted.
- `putnam672` and `putnam-2025` generate the strict PutnamBench complete-file format, then locate the exact target
  declaration and submit its proof body to the pinned PutnamBench manifest. Unrelated compilable declarations are
  rejected before verifier submission. However, 346/672 Putnam files contain a factored answer hole in addition to the
  final theorem hole (6/12 for Putnam 2025). The current proof-body manifest replaces only the final hole, so these runs
  are not equivalent to either official with-solution or no-solution evaluation.

Earlier Putnam runs recorded with `candidate.kind=full_lean_file` are compile-only diagnostics, not task-bound scores.
The current Putnam proof-body contract is marked `putnam_partial_hole_contract` until the verifier exposes a
statement-bound with-solution manifest or a multi-hole submission contract.

The client rejects transport/system errors and manifest-version mismatches instead of converting them to benchmark
score 0. Only a successful Lean compile without error diagnostics counts as correct; aggregate repeated samples with
pass@k.

### Math Config Matrix

Math configs are grouped by model family:

```text
nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/deepseek_v4_flash/no_agent.yaml
nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/deepseek_v4_flash/codex.yaml
nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/deepseek_v4_flash/codex_lean.yaml
nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/a20b_high_260609_v5_step320_192k/no_agent.yaml
nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/a20b_high_260609_v5_step320_192k/codex.yaml
nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/a20b_high_260609_v5_step320_192k/codex_lean.yaml
```

Configs define model family, agent surface, and tool mode. Dataset selection is a run parameter:

```bash
export CONFIG=nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/deepseek_v4_flash/codex_lean.yaml
export STEM_MAS_DATASET_NAME=minif2f  # minif2f | putnam672 | putnam-2025 | fatem | fateh | fatex
export STEM_MAS_DATASET_PATH=""       # empty uses the adapter's default source
```

Each model config owns its default model, token limit, timeout, and reasoning effort. Environment variables are
explicit run-level overrides; do not source DeepSeek overrides when selecting an A20B config, or vice versa.

Every math config must declare a top-level `solver_strategy`:

- `no_agent`: direct single-turn LLM baseline, requires `agent_name=single_turn`.
- `codex`: Codex runtime without Lean MCP, requires `agent_name=stem_mas_codex`.
- `codex_lean`: Codex runtime with Lean LSP MCP, requires `agent_name=stem_mas_codex`.
- `official_solver`: external official pipeline, requires `agent_name=single_turn`; the runtime infers the concrete
  pipeline from the benchmark unless `official_pipeline` overrides it.

The benchmark RunPart validates this pairing and writes `solver_strategy`, `solver_family`, `solver_agent_surface`,
`solver_tool_mode`, and `solver_official_method` into task metadata. Official runs also set the execution-owned
`task_config.official_solver_pipeline`. Official strategies are command adapters, not built-in copies of paper
algorithms: the command emits the proof candidate and the final score still comes from Lean4 verifier.

Use a shell loop or platform job wrapper to run the dataset matrix. The `codex_lean` variant adds the Lean LSP MCP
tool server, while `codex` is the no-tool Codex baseline.

### Nexus Dataset And RunPart

The source adapters can materialize normalized rows as standard Nexus `DatasetItem` objects. These items contain only
the benchmark-owned task payload: prompt, reference, verifier route, and source metadata. Model credentials, agent,
runtime, solver strategy, and run-level timeouts remain in a reusable RunPart loaded from the checked-in profile.

Create a local authoring dataset, inspect it, and run it with compose:

```bash
python nexus/extensions/tasks/stem_mas/benchmark/runners/tools/materialize_nexus_dataset.py \
  --config nexus/extensions/tasks/stem_mas/benchmark/eval/config/math/deepseek_v4_flash/no_agent.yaml \
  --output workspace/datasets/minif2f

nexus dataset show workspace/datasets/minif2f
nexus run batch \
  --data workspace/datasets/minif2f \
  --use stem_mas_benchmark:profile=math/deepseek_v4_flash/no_agent \
  -d workspace/minif2f_no_agent.db
```

The local directory can be published with `nexus dataset publish` and reused by URI. The generic official profile
infers `minif2f`, `putnambench`, or `fate_eval` from its configured dataset; `official_pipeline` remains an explicit
override for ad hoc compose calls.

`run_batch_template.sh` uses this same path: it materializes the selected upstream source under the run directory and
invokes `nexus run batch --data ... --use stem_mas_benchmark:profile=...`. There is no separate benchmark generator
contract. Gateway cache routing is execution-owned and is therefore not embedded into portable dataset items.

`no_agent` uses the shared `single_turn` agent surface: one direct LLM completion, no tools, no sub-agents, and no
multi-turn agent loop. Only visible assistant content is submitted to the evaluator; hidden provider reasoning is not
promoted to a final answer. Use it as the direct-model baseline against Codex.

The supported solver-strategy matrix is:

| Solver strategy | Meaning | Official-score interpretation |
| --- | --- | --- |
| `no_agent` | One direct LLM completion. | Direct pass@1 baseline only. |
| `codex` | Codex runtime without Lean MCP tools. | Agent-surface baseline; not an official prover pipeline. |
| `codex_lean` | Codex runtime with Lean LSP MCP tools. | Tool-enabled agent baseline; final score still comes from Lean4 verifier. |
| `official_solver` | Dataset-selected external official prover command. | Official-method comparison if the command implements the intended prover pipeline. |

Official solver commands receive one JSON object on stdin and must print exactly one JSON object of the form
`{"candidate": "<non-empty string>"}` on stdout. Configure commands with `task_config.official_solver_command` or the
pipeline-specific environment variable:

- `STEM_MAS_MINIF2F_OFFICIAL_SOLVER_COMMAND`
- `STEM_MAS_PUTNAMBENCH_OFFICIAL_SOLVER_COMMAND`
- `STEM_MAS_FATE_EVAL_OFFICIAL_SOLVER_COMMAND`

DeepSeek-V4-Flash configs use the distillation-platform route:

```bash
export GENERATION_MODEL=openai/api_deepseek_deepseek-v4-flash
export GENERATION_REASONING_EFFORT=max
```

The DeepSeek math configs set `extra_body.reasoning_effort` to `${GENERATION_REASONING_EFFORT:-max}` and keep
`thinking.type=enabled` / `enable_thinking=true`. This requests max reasoning on the distillation platform; verify
backend acceptance from request logs before treating it as a guaranteed model-side setting.

A20B configs use the Taiji route:

```bash
export GENERATION_MODEL=A20B-High-260609-V5-step320-192K
export GENERATION_TAIJI_POLARIS_SERVICE=A20B-High-260609-V5-step320-192K_AIDE
export GENERATION_REASONING_EFFORT=high
```

These configs keep the current Taiji high-thinking / interleaved-thinking settings in `client_args`.

### Lean MCP Integration

Lean MCP endpoint:

```bash
export LEAN_LSP_MCP_URL=http://lean-lsp-mcp-v0d27d0.production.polaris:8080/mcp
```

Current math experiments only cover no-agent and Codex. Codex uses `extra_config_toml` with `[mcp_servers.lean_lsp]`
and `npx -y mcp-remote ...`; the MCP server and enabled tools belong to the YAML config. The stem_mas Codex patch
derives namespace tools from the Responses request and only accepts standard `mcp__namespace__tool` call names. If a
Codex/model/provider combination emits malformed names such as `lean_lsplean_run_code`, treat that as a tool-calling
compatibility failure to fix in the Codex/provider path rather than silently repairing it in the benchmark. No-agent has
no Lean MCP variant because it intentionally performs a direct LLM completion.

Codex no-tool baselines are required for every model. The Lean MCP version should only change tool availability and
the minimal prompt text needed to use the verifier; it should not change the target answer/proof objective.
