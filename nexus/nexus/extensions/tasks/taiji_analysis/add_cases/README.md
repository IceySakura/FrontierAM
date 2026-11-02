# 添加新 Case 到 Benchmark

## 完整流程

```
准备 case → 初始化数据集 → 跑 GT → 写入 GT → 跑对比模型 → Judge 评分 → 合并入库
```

---

## Step 1：准备 case

创建 `new_cases.json`，每条包含 `url` 和 `question`：

```json
[
  {
    "url": "https://hunyuanaide.taiji.woa.com/web/pre_training_inst_list?taskId=446976&taskID=basic_train_xxx&wsId=10103&instId=abc123",
    "question": "帮我看下这个实例为什么挂了：https://hunyuanaide.taiji.woa.com/web/pre_training_inst_list?taskId=446976&taskID=basic_train_xxx&wsId=10103&instId=abc123"
  },
  {
    "url": "https://hunyuanaide.taiji.woa.com/web/pre_training_inst_list?...",
    "question": "这个任务训了几步之后卡住不动了，帮忙看看：https://..."
  }
]
```

说明：
- `url`：太极平台实例链接，必须包含 `instId=`
- `question`：用户原始问题（会原样发给 Agent）
- `inst_id`/`task_id`/`ws_id`：可选，不填会自动从 URL 提取

---

## Step 2：初始化数据集

```bash
python3 nexus/extensions/tasks/taiji_analysis/add_cases/init_cases.py \
  --input new_cases.json \
  --output ./new_benchmark/
```

产出：
```
new_benchmark/
├── dataset.json      # 标准格式，answer 为空
└── dataset.parquet   # 供 nexus run-batch 使用
```

---

## Step 3：跑 GT（最强模型，跳过 Judge）

```bash
export TAIJI_SKIP_JUDGE=1
export TAIJI_DATASET_PATH=./new_benchmark/dataset.parquet
export TAIJI_OPENAI_API_KEY="<APP_ID>:<APP_KEY>"
export TAIJI_OPENAI_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_OPENAI_MODEL="api_google_claude-opus-4-8"
export TAIJI_REASONING_EFFORT="max"
export GONGFENG_RUNTIME_TOKEN="<token>"
export MCPORTER_IWIKI_TOKEN="<token>"
export MCPORTER_GONGFENG_TOKEN="<token>"
export MCPORTER_TAPD_TOKEN="<token>"
export TAIJI_API_TOKEN="h3kcqqyYy5Uosyuyx1E5mw"
export GF_TOKEN="<token>"

uv run nexus run-batch \
  -g nexus/extensions/tasks/taiji_analysis/generator.py \
  -d ./new_benchmark/gt_bench.db \
  --max-concurrent 10
```

跑完后 DB 中每条记录的 `prediction` 就是 GT 候选（`score=-1` 表示未 Judge）。

---

## Step 4：写入 GT

将跑出的 prediction 写入 dataset.json 的 answer 字段：

```bash
python3 nexus/extensions/tasks/taiji_analysis/gt_management/replace_gt.py \
  --db ./new_benchmark/gt_bench.db \
  --dataset ./new_benchmark/dataset.json
```

可先 `--dry-run` 预览替换数量。

---

## Step 5：跑对比模型

取消 `TAIJI_SKIP_JUDGE`，为每个模型分别跑：

```bash
unset TAIJI_SKIP_JUDGE
export TAIJI_DATASET_PATH=./new_benchmark/dataset.parquet

# 模型 1：hy3_high
export TAIJI_OPENAI_MODEL="openai/api_taiji_hy3"
export TAIJI_REASONING_EFFORT="high"
uv run nexus run-batch \
  -g nexus/extensions/tasks/taiji_analysis/generator.py \
  -d ./new_benchmark/results_hy3_high.db \
  --max-concurrent 10

# 模型 2：hy3_nothink
export TAIJI_REASONING_EFFORT=""
uv run nexus run-batch \
  -g nexus/extensions/tasks/taiji_analysis/generator.py \
  -d ./new_benchmark/results_hy3_nothink.db \
  --max-concurrent 10

# 模型 3：opus4.8_max
export TAIJI_OPENAI_MODEL="api_google_claude-opus-4-8"
export TAIJI_REASONING_EFFORT="max"
uv run nexus run-batch \
  -g nexus/extensions/tasks/taiji_analysis/generator.py \
  -d ./new_benchmark/results_opus4.8_max.db \
  --max-concurrent 10
```

注意：Step 5 需要配置 Judge LLM 环境变量（`TAIJI_JUDGE_*`），评分会自动执行。

---

## Step 6：合并入库

将各模型结果整理到 eval_frontend 可读取的目录结构：

```bash
# 计算各模型指标
python3 -m nexus.extensions.tasks.taiji_analysis.compute_metrics \
  --db ./new_benchmark/results_hy3_high.db \
  --output ./new_benchmark/results_hy3_high/metrics.json

# 重复对每个模型执行...
```

最终目录结构放到 eval_frontend 的 `episodes/` 下即可在前端查看。

---

## 快速参考

| 步骤 | 脚本 | 关键参数 |
|------|------|---------|
| 初始化 | `add_cases/init_cases.py` | `--input`, `--output` |
| 跑 GT | `nexus run-batch` + generator.py | `TAIJI_SKIP_JUDGE=1` |
| 写入 GT | `gt_management/replace_gt.py` | `--db`, `--dataset` |
| 跑对比 | `nexus run-batch` + generator.py | 改 `TAIJI_OPENAI_MODEL` |
| 只换 GT 重 Judge | `gt_management/rejudge_with_new_gt.py` | `--source-results`, `--new-gt` |

---

## 注意事项

1. **GT 质量**：跑完 GT 建议人工抽查几条，确认分析质量合格再继续
2. **实例过期**：太极实例有 TTL，过期后 Agent 无法拉到日志。尽早跑 GT
3. **并发数**：API 有 RPM 限制，opus4.8 建议 ≤10 并发
4. **断点续跑**：任何步骤中断都可以直接重跑同一个 DB 路径，自动跳过已成功的
5. **只加模型不重跑 GT**：跳过 Step 3-4，直接从 Step 5 开始
