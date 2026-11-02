# Taiji Replay — 运行指南

**前置条件**：任意有网络的 Linux 机器（不需要 GPU），Python 3.13+。

---

## 1. 环境准备

```bash
git clone git@git.woa.com:nexus-agent/nexus.git
cd nexus
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]" litellm boto3
```

---

## 2. 准备数据集

```bash
# 如果调度节点不是数据所在机器，先拷过去
scp episodes/dataset.json <调度节点>:/data/workspace/taiji_replay/episodes/

cd /data/workspace/taiji_replay
python3 convert_dataset.py
# dataset.json → episodes/dataset.parquet
```

---

## 3. 配置环境变量

```bash
# ── Agent LLM（hy3-preview，被评估模型）──
export TAIJI_OPENAI_API_KEY="<APP_ID>:<APP_KEY>"
export TAIJI_OPENAI_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_OPENAI_MODEL="openai/api_taiji_hy3-preview"

# ── Judge LLM（评分）──
export TAIJI_JUDGE_API_KEY="<APP_ID>:<APP_KEY>"
export TAIJI_JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_JUDGE_MODEL="openai/api_azure_openai_gpt-5.6-sol"
export TAIJI_JUDGE_API_STYLE="responses"
export TAIJI_JUDGE_REASONING_EFFORT="xhigh"

# ── 数据集 ──
export TAIJI_DATASET_PATH="/data/workspace/taiji_replay/episodes/dataset.parquet"

# ── Runtime ──
export GONGFENG_RUNTIME_TOKEN="<your-gongfeng-sandbox-token>"

# ── Skill Token ──
export TAIJI_API_TOKEN="h3kcqqyYy5Uosyuyx1E5mw"
export GF_TOKEN="<your-gongfeng-pat>"
export SWANLAB_API_KEY="<your-swanlab-key>"

# ── MCP Token（注入容器，组装为 mcporter.json）──
export MCPORTER_IWIKI_TOKEN="<your-iwiki-mcp-token>"
export MCPORTER_GONGFENG_TOKEN="<your-gongfeng-mcp-token>"
export MCPORTER_TAPD_TOKEN="<your-tapd-mcp-token>"

# ── 自动安装缺失依赖（兜底） ──
export NEXUS_AUTO_INSTALL=1
```

---

## 4. 运行

```bash
uv run nexus run-batch \
    -g nexus/extensions/tasks/taiji_analysis/generator.py \
    -d /data/workspace/taiji_replay/results/taiji_bench.db \
    --max-concurrent 100 \
    --timeout 600
```

---

## 5. 监控进度

```bash
uv run nexus inspect-batch /data/workspace/taiji_replay/results/taiji_bench.db
```

---

## 6. 断点续跑

```bash
# 续跑（自动跳过已成功的，不需要 -g）
uv run nexus run-batch \
    -d /data/workspace/taiji_replay/results/taiji_bench.db \
    --max-concurrent 100

# 只重跑失败的
uv run nexus run-batch \
    -d /data/workspace/taiji_replay/results/taiji_bench.db \
    --retry-only \
    --max-concurrent 100
```

---

## 7. 计算指标

```bash
python -m nexus.extensions.tasks.taiji_analysis.compute_metrics \
    --db /data/workspace/taiji_replay/results/taiji_bench.db \
    --output /data/workspace/taiji_replay/results/metrics.json
```

---

## 8. 导出结果

```bash
uv run nexus export --format jsonl /data/workspace/taiji_replay/results/taiji_bench.db
uv run nexus export --format full /data/workspace/taiji_replay/results/taiji_bench.db
```

---

## Token 获取

| Token | 获取方式 |
|-------|---------|
| `TAIJI_OPENAI_API_KEY` | https://model-eval.woa.com/model → 申请账号 → 获取 APP_ID:APP_KEY |
| `GONGFENG_RUNTIME_TOKEN` | 工蜂 sandbox 资源平台申请；如已跑过 SWE-bench 可复用 |
| `GF_TOKEN` | 工蜂 → Settings → Access Tokens → 创建 PAT |
| `TAIJI_API_TOKEN` | 固定值 `h3kcqqyYy5Uosyuyx1E5mw` |
| `TAIJI_JUDGE_API_KEY` | 同 TAIJI_OPENAI_API_KEY（可复用同一个 APP_ID:APP_KEY） |
| `MCPORTER_IWIKI_TOKEN` | Knot 平台 → iWiki MCP 应用 → 创建 PAT |
| `MCPORTER_GONGFENG_TOKEN` | Knot 平台 → 工蜂 MCP 应用 → 创建 PAT |
| `MCPORTER_TAPD_TOKEN` | Knot 平台 → TAPD MCP 应用 → 创建 PAT |
| `SWANLAB_API_KEY` | SwanLab 平台 → Settings → API Keys（可选） |

---

## 预估时间

100 并发 × N 条 × 平均 3-5 分钟/条，按当前数据量约 **30-45 分钟**

---

## 常见问题

**中途失败了？** 直接用断点续跑命令，自动跳过已成功的。

**想换模型？** 改 `TAIJI_OPENAI_MODEL`，用新的 `-d` 路径重跑。

**容器配额不够？** 降低 `--max-concurrent`。
