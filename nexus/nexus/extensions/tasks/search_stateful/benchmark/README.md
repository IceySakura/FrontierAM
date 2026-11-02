# Benchmarks

Benchmark assets live under `eval/` and `docker/`.

All paths below are relative to the repo root (the directory that contains the top-level `nexus/` folder).

## HLE

### 1) Dataset

Use either:
- HLE parquet dataset root (contains `data/test-*.parquet`)
- Naive JSONL file

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

HLE-200 dataset (naive JSONL, internal repo data/hle-200-jsonl branch):
- Local JSONL:
  ```bash
  cd "$(git rev-parse --show-toplevel)"
  git fetch git@git.woa.com:felixjjiang/hle_200.git data/hle-200-jsonl
  mkdir -p nexus/extensions/tasks/search_stateful/benchmark/data
  git show FETCH_HEAD:hle_200.jsonl > \
    nexus/extensions/tasks/search_stateful/benchmark/data/hle_200.jsonl
  ```

### 1.5) Build Image (optional)

One-click build (and push) the benchmark image:

```bash
cd nexus/extensions/tasks/search_stateful/benchmark/docker
bash build.sh
```

What it does:
- Builds a Docker image from the repo root using `docker/Dockerfile`.
- Tags it as `mirrors.tencent.com/asi/nexus-eval:tmp-<LOCAL_IP>` and pushes it by default.
- Before pushing, it tries to delete the existing remote tag with the same name (to avoid tag conflicts).

### 1.6) Where to Build / Where to Run

- Build the benchmark image on a **DevCloud** machine or an **IDC** machine (i.e. any host that can run Docker and push to
  `mirrors.tencent.com`).
- Run the evaluation on an **IDC** machine (recommended), or run it inside a **Taiji job** (which executes in IDC).
  In Taiji, ensure the dataset/config paths are available in the job workspace and all required env vars are set.

### 2) Config

Edit the single base config:
- `nexus/extensions/tasks/search_stateful/benchmark/eval/config/base.yaml`

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

# Optional: set images (only if you built your own image)
export HLE_IMAGE="mirrors.tencent.com/asi/nexus-eval:tmp-<LOCAL_IP>"
export HLE_RUNTIME_IMAGE="mirrors.tencent.com/asi/nexus-eval:tmp-<LOCAL_IP>"

# Optional: Taiji judge (only when using eval/config/gpt_oss_judge.yaml)
export JUDGE_MODEL="gpt-oss-120b-sci-128k-new_daisyyi_for_RL_R2_zwfy"
```

### 3) Run

Background run (nohup):

```bash
mkdir -p outputs
nohup python nexus/extensions/tasks/search_stateful/benchmark/eval/run.py \
  --dataset nexus/extensions/tasks/search_stateful/benchmark/data/hle_200.jsonl \
  --format naive \
  --out outputs/hle_200_taiji_judge.jsonl \
  --database outputs/hle_200_taiji_judge.db \
  --max-concurrent 50 \
  --config nexus/extensions/tasks/search_stateful/benchmark/eval/config/gpt_oss_judge.yaml \
  > outputs/hle_200_taiji_judge.log 2>&1 &

tail -f outputs/hle_200_taiji_judge.log
```

### 4) Parameters

- `--dataset` (required): dataset path (HLE parquet dir/file or naive JSONL).
- `--format` (default: `hle`): dataset format, `hle` or `naive`.
- `--out` (default: `outputs/hle_results.jsonl`): JSONL export path.
- `--database` (default: `outputs/hle_results_{format}.db`): SQLite checkpoint DB.
- `--max-concurrent` (default: `4`): max concurrent tasks.
- `--config` (default: `eval/config/base.yaml`): base config path.

### 5) Output

- JSONL: one result per line at `--out`.
- SQLite DB: run state and checkpoints at `--database`.

### 6) Resume

- Re-run with the same `--database` to resume and skip succeeded items.
- For retry-only / no-resume behavior, use `nexus run-batch` with `--retry-only` or `--no-resume`.

## Formats

- `hle`: parquet with `question` + `answer`
- `naive`: JSONL with aliases:
  - question: `origin_question`, `question`, `prompt`, `input`, `query`
  - answer: `ref_answer`, `answer`, `output`, `response`, `completion`
