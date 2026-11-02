# Taiji Analysis Judge Benchmark

评估 judge LLM 的准确性：用当前 judge 的输出（经人工清洗）作为 GT，跑新 judge 模型对比。

## 流程

```
1. build_judge_benchmark.py    → 生成评测集 (parquet)
2. run_judge_benchmark.py      → 跑新 judge 模型，输出 score + reasoning
3. eval_judge_benchmark.py     → 评估：score 一致 + reasoning 一致 = pass
```

## Step 1: 生成评测集

```bash
# 全量（不等人标注）
python3 build_judge_benchmark.py --results-dir /path/to/nexus_results

# 只用人确认过的 case（标注完成后）
python3 build_judge_benchmark.py --results-dir /path/to/nexus_results --only-correct
```

输出 `judge_benchmark.parquet`，每行包含：
- `run_id`, `experiment` — case 标识
- `question` — 原始问题
- `prediction` — agent 模型输出的报告
- `reference` — 标准答案报告
- `gt_score` — 当前 judge 给的分 (0 或 1)
- `gt_reasoning` — 当前 judge 给的理由

## Step 2: 跑新 judge

所有参数均可通过环境变量配置（推荐），也可通过命令行参数覆盖。

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TAIJI_JUDGE_AGENT_BASE_URL` | API 地址 | `http://llm-api.model-eval.woa.com/v1` |
| `TAIJI_JUDGE_AGENT_MODEL` | 模型名 | `openai/api_google_claude-opus-4-8` |
| `TAIJI_JUDGE_AGENT_API_KEY` | API Key | (空) |
| `TAIJI_JUDGE_AGENT_API_STYLE` | `chat` 或 `responses` | `chat` |
| `TAIJI_JUDGE_AGENT_REASONING_EFFORT` | `none/low/medium/high/xhigh/max` | (空) |

### 示例

```bash
# Chat Completions API（默认）
python3 run_judge_benchmark.py \
    --input judge_benchmark.parquet \
    --model gpt-4o \
    --api-base https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY

# 用自定义 prompt
python3 run_judge_benchmark.py \
    --input judge_benchmark.parquet \
    --model hy3-high \
    --api-base http://localhost:8000/v1 \
    --prompt-file my_judge_prompt.txt

# 快速测试
python3 run_judge_benchmark.py --input judge_benchmark.parquet --model gpt-4o --limit 20

# GPT-5.6-sol via Responses API（推荐用环境变量配置）
export TAIJI_JUDGE_AGENT_API_STYLE=responses
export TAIJI_JUDGE_AGENT_MODEL=api_azure_openai_gpt-5.6-sol
export TAIJI_JUDGE_AGENT_REASONING_EFFORT=xhigh
python3 run_judge_benchmark.py --input judge_benchmark.parquet
```

输出 `judge_results_{model}.jsonl`，每行：
- `run_id`, `experiment`
- `new_score` — 新 judge 给的分
- `new_reasoning` — 新 judge 给的理由
- `raw_output` — 原始输出

## Step 3: 评估

所有参数均可通过环境变量配置（推荐），也可通过命令行参数覆盖。

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TAIJI_JUDGE_JUDGE_BASE_URL` | API 地址 | `http://llm-api.model-eval.woa.com/v1` |
| `TAIJI_JUDGE_JUDGE_MODEL` | 模型名 | `openai/api_google_claude-opus-4-8` |
| `TAIJI_JUDGE_JUDGE_API_KEY` | API Key | (空) |
| `TAIJI_JUDGE_JUDGE_API_STYLE` | `chat` 或 `responses` | `chat` |
| `TAIJI_JUDGE_JUDGE_REASONING_EFFORT` | `none/low/medium/high/xhigh/max` | (空) |

### 示例

```bash
# 完整评估（score + reasoning 一致性）
python3 eval_judge_benchmark.py \
    --benchmark judge_benchmark.parquet \
    --results judge_results_gpt-4o.jsonl \
    --model gpt-4o \
    --output eval_detail.jsonl

# 只看 score 一致率（不调 LLM）
python3 eval_judge_benchmark.py \
    --benchmark judge_benchmark.parquet \
    --results judge_results_gpt-4o.jsonl \
    --score-only

# GPT-5.6-sol via Responses API（推荐用环境变量配置）
export TAIJI_JUDGE_JUDGE_API_STYLE=responses
export TAIJI_JUDGE_JUDGE_MODEL=api_azure_openai_gpt-5.6-sol
export TAIJI_JUDGE_JUDGE_REASONING_EFFORT=xhigh
python3 eval_judge_benchmark.py \
    --benchmark judge_benchmark.parquet \
    --results judge_results.jsonl
```

输出 metrics：
- **Score accuracy** — 新 judge 分数和 GT 一致的比率
- **Reasoning consistency** — judge-judge LLM 判定理由一致的比率
- **Overall pass rate** — 两者都一致的比率 (最严格)

## API 风格说明

| 风格 | 适用模型 | 请求方式 |
|------|----------|----------|
| `chat` | GPT-4o, Claude, HY3 等 | `/v1/chat/completions`，`messages` 数组 |
| `responses` | GPT-5.6-sol | `/v1/responses`，`input` + `instructions`，`reasoning: {"effort": "..."}` |

## 评判逻辑

```
                     ┌─────────────────────┐
                     │  judge_benchmark    │
                     │  (question,         │
                     │   prediction,       │
                     │   reference,        │
                     │   gt_score,         │
                     │   gt_reasoning)     │
                     └──────────┬──────────┘
                                │
                    ┌───────────▼───────────┐
                    │  New Judge Model      │
                    │  → new_score          │
                    │  → new_reasoning      │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                   ▼
    ┌─────────────────┐              ┌────────────────────┐
    │ Score Match?    │              │ Judge-Judge LLM    │
    │ new == gt       │              │ reasoning_a vs     │
    │ → 0/1           │              │ reasoning_b        │
    └────────┬────────┘              │ → 0/1              │
             │                       └─────────┬──────────┘
             └───────────────┬─────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Overall Pass    │
                    │ = score AND     │
                    │   reasoning     │
                    └─────────────────┘
```
