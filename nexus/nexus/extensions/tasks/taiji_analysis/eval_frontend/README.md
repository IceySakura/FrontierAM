# Taiji Analysis Benchmark — 标注前端 & 过滤脚本

对 benchmark 数据进行浏览、多模型对比、人工标注、GT 选择，然后用脚本过滤数据集重跑。

---

## 快速启动

```bash
pip install flask pandas
python -m nexus.extensions.tasks.taiji_analysis.eval_frontend --port 8090
```

默认读取 `/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/nexus_results/`，可通过 `--results-dir` 指定。

---

## Benchmark 版本

动态扫描 `EPISODES_DIR` 下的 `dataset_v*/` 目录，自动识别版本。版本标签是**多选**的（取并集），在 Case Browser 可按版本筛选。

---

## 页面功能

### Dashboard

首页总览：

- 模型数、case 数、已标注数、标注进度
- Benchmark 版本列表（动态扫描）
- 各模型 pass rate 对比

### Case Browser

展示当前 benchmark 每条 case 的**具体内容**：

| 列 | 说明 |
|----|------|
| ID | run_id |
| Ver | 版本标签（多选，如 `v0` `v1` `v2`）|
| Pass | 模型通过率标签：`all_pass` / `all_fail` / `3/5` |
| Question | 原始问题（截断） |
| Reference | 参考答案摘要 |
| Sender | 提问者 |
| Label | 人工标注状态 |
| Exclude | 排除原因（inaccessible/bad_gt）|

每列都有筛选器（文本输入或下拉），可叠加过滤。点击行打开 Case Detail Modal 查看全文和打标注。

### Compare & Annotate

对比 + 标注合一页面：

- **筛选**：按 pass 标签（all_fail/partial/all_pass）+ 标注状态（unlabeled/correct/...）
- **导航**：Prev / Next 逐条浏览筛选结果
- **内容**：Reference 全文 + 各模型 score 摘要 + 各模型完整 prediction + Judge reasoning
- **标注**：直接在页面内标注（label + note + per-model feedback）
- **GT 选择**：对比 online GT / newrun GT / 各模型 prediction，选择最优 GT 来源

典型用法：选 "All Fail" + "Unlabeled"，逐条审阅标注。

### Hard Case Analysis

专注于所有模型都答错的 case：

- 自动筛选 all-fail case
- 展示 prediction vs reference 摘要
- 用于 judge 校准和数据质量排查

### 页面互跳

| 从 | 到 | 操作 |
|----|----|----|
| Case Browser | Compare | 点击行 |
| Compare | Case Browser | 点 "Back to Cases"（高亮该行）|

---

## 标注数据

所有标注数据存储在 `--results-dir` 目录下。

### annotations.json

```json
{
  "889": {
    "label": "judge_too_strict",
    "note": "",
    "annotator": "lunahyhuang",
    "timestamp": "2026-07-14T11:30:00.000Z",
    "gt_source": "newrun",
    "benchmark_quality": "good",
    "model_feedback": {
      "opus4.8_xhigh": {"score": 1, "note": "模型其实答对了", "judge_comment": "...", "annotator": "..."}
    }
  }
}
```

基础字段：
- `label`：`correct`、`judge_too_strict`、`judge_too_lenient`、`ambiguous`、`data_issue`
- `note`：自由文本备注

高级字段：
- `gt_source`：GT 来源选择（`online` / `newrun` / `<experiment_name>`）
- `benchmark_quality`：题目质量评价
- `model_feedback`：逐模型打分（覆盖 judge 结果）

### exclude_reasons.json

记录被排除的 case 及原因（如 `inaccessible`、`bad_gt`），过滤脚本默认自动排除这些 case。

---

## 回填 Judge Reasoning

对已有结果补充 judge reasoning（Compare 页面展示用），读取 `results.jsonl` 中缺少 reasoning 的 case，调用 Judge LLM 生成后回写到 `results.jsonl` 和 DB。

```bash
# 配置 Judge LLM
export TAIJI_JUDGE_API_KEY="<APP_ID>:<APP_KEY>"
export TAIJI_JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_JUDGE_MODEL="api_azure_openai_gpt-5.6-sol"
export TAIJI_JUDGE_API_STYLE="responses"
export TAIJI_JUDGE_REASONING_EFFORT="xhigh"

# 运行回填
python3 nexus/extensions/tasks/taiji_analysis/eval_frontend/backfill_judge_reasoning.py \
    --results-dir /path/to/experiment_results \
    --concurrency 10
```

参数：

| 参数 | 作用 |
|------|------|
| `--results-dir` | 包含 `results.jsonl` + `taiji_bench*.db` 的实验目录 |
| `--concurrency` | 并发数（GPT-5.6-sol 限 30 RPM，建议 ≤ 10） |
| `--model` | 覆盖 Judge 模型（默认读 `TAIJI_JUDGE_MODEL`） |
| `--reasoning-effort` | 覆盖 reasoning effort（默认读 `TAIJI_JUDGE_REASONING_EFFORT`） |

---

## 过滤脚本

标注完成后，用 `filter_by_annotation.py` 根据 `annotations.json` 中的 label 过滤数据集：

```bash
cd nexus/extensions/tasks/taiji_analysis/eval_frontend

# 去掉无效数据（data_issue + inaccessible + bad_gt）
python3 filter_by_annotation.py --exclude-labels data_issue,inaccessible,bad_gt

# 也去掉质量"不适合"的 case
python3 filter_by_annotation.py --exclude-labels data_issue,inaccessible,bad_gt --exclude-quality 不适合

# 只保留高质量 case
python3 filter_by_annotation.py --include-quality 非常适合

# 先看效果不写文件
python3 filter_by_annotation.py --exclude-labels data_issue,inaccessible,bad_gt --exclude-quality 不适合 --dry-run
```

参数：

| 参数 | 作用 |
|------|------|
| `--input` | 输入 parquet（默认 `episodes/dataset.parquet`）|
| `--output` | 输出路径（默认加 `_filtered` 后缀）|
| `--output-format` | `parquet`（默认）或 `ids`（纯文本 ID 列表）|
| `--results-dir` | annotations.json 所在目录（默认 nexus_results） |
| `--exclude-labels` | 排除这些 label（可选值：`data_issue`/`inaccessible`/`bad_gt`/`ambiguous`/`others`）|
| `--include-labels` | 只保留这些 label（如 `correct`）|
| `--exclude-quality` | 排除这些 benchmark_quality（可选值：`不适合`/`一般`/`非常适合`）|
| `--include-quality` | 只保留这些 benchmark_quality |
| `--no-exclude-reasons` | 不自动排除 exclude_reasons.json 中的 case |
| `--dry-run` | 只打印统计 |

---

## 完整工作流

### Step 1: 跑 benchmark

```bash
nexus run-batch ...
```

### Step 2: 启动前端 & 标注

```bash
python -m nexus.extensions.tasks.taiji_analysis.eval_frontend --port 8090
```

打开浏览器，逐条审阅并标注：

- **Case Browser** — 看每条题目内容，快速定位问题 case
- **Compare & Annotate** — 筛 `all_fail` → 逐条对比模型输出 vs 参考答案，打标注 + 逐模型打分
- **GT 选择** — 对比 online GT / newrun GT / 各模型 prediction，选择最优 GT 来源
- **Trace 查看** — 查看 Agent 执行过程的 span 详情

可用标注 label：

| Label | 含义 | 是否保留在 benchmark |
|-------|------|-------------------|
| `correct` | judge 判对了，模型确实答错 | 保留 |
| `judge_too_strict` | 模型其实答对了，judge 误判为错 | 过滤掉 |
| `judge_too_lenient` | 模型答错了但 judge 放过了 | 保留 |
| `ambiguous` | 题目本身有歧义，无法公平评判 | 过滤掉 |
| `data_issue` | 数据有问题（格式错、信息缺失等） | 过滤掉 |
| `inaccessible` | 平台任务已删除，无法复现评测 | 过滤掉 |
| `bad_gt` | ground truth 本身就是错的 | 过滤掉 |

### Step 3: 过滤数据集 & 重跑

标注完成后，用脚本过滤掉无效 case，生成校准后的 benchmark：

```bash
# 去掉不该算分的 case
python3 filter_by_annotation.py \
    --exclude-labels data_issue,judge_too_strict,ambiguous

# exclude_reasons.json 里的 inaccessible/bad_gt 默认自动排除
# 查看过滤效果
python3 filter_by_annotation.py --exclude-labels data_issue,judge_too_strict,ambiguous --dry-run
```

用过滤后的数据集重跑：

```bash
nexus run-batch --dataset episodes/dataset_hard_filtered.parquet ...
```

重跑后的 pass rate 就是**校准过的**——去掉了 judge 误判和无效数据，反映模型真实能力。

---

## 用户认证

打开页面需选择/输入用户名。预设：`lunahyhuang`、`nickklau`，支持自定义。存 localStorage。

---

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/users` | GET | 预设用户列表 |
| `/api/versions` | GET | benchmark 版本列表（动态扫描） |
| `/api/experiments` | GET | 所有实验及 metrics |
| `/api/all_cases` | GET | 全部 case 内容 + 各模型得分 + 版本标签 |
| `/api/results/<exp>` | GET | 某实验的全部结果 |
| `/api/case/<exp>/<run_id>` | GET | 单 case 详情（含 config、agent_output）|
| `/api/compare/<run_id>` | GET | 一个 run_id 在所有实验中的结果 |
| `/api/hard_cases` | GET | 所有模型都答错的 case 列表 |
| `/api/trace/<exp>/<run_id>` | GET | Agent 执行 trace（从 DB 读取 span） |
| `/api/annotations` | GET/POST | 标注读写（label、note、gt_source、model_feedback） |
| `/api/model_feedback` | POST | 保存逐模型打分 |
| `/api/judge_reasoning` | GET | Judge 推理过程 |
| `/api/online_gt/<run_id>` | GET | 原始 online GT 答案 |
| `/api/newrun_gt/<run_id>` | GET | 重跑 GT 答案 |
| `/api/exclude_reasons` | GET/POST | 排除原因管理 |
