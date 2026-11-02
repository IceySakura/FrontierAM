# ACE 评测脚本说明

## 使用镜像
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:ace

## 功能概览

脚本会完成以下流程：

1. 配置环境变量： `数据集配置`、`Agent 配置`、`User Simulator 配置`、`评测控制参数`
2. 按类别执行 `nexus run-batch`
3. 将生成的 `.db` 结果导出为 `.jsonl`
4. 使用 `stasis.py` 生成统计分析结果（含 csv）

支持单类评测，也支持一次性评测多个类别。

---

## 目录要求

输入数据集目录：

```bash
/workspace/ACEBench
```


结果输出目录：

```bash
/workspace/result
```

---

## 环境变量说明

### 1. 数据集配置

```bash
DATASET_PATH='/workspace/ACEBench'
export ACE_BENCH_DATASET_PATH='$DATASET_PATH'
```

- `ACE_BENCH_DATASET_PATH`：ACEBench 数据集根目录

---

### 2. Agent 配置

```bash
export ACE_AGENT_API_KEY='${ACE_AGENT_API_KEY:-}'
export ACE_AGENT_BASE_URL='${ACE_AGENT_BASE_URL:-}'
export ACE_AGENT_MODEL='${ACE_AGENT_MODEL:-deepseek-v3}'
export ACE_AGENT_MAX_TOKENS='${ACE_AGENT_MAX_TOKENS:-4096}'
export ACE_AGENT_TEMPERATURE='${ACE_AGENT_TEMPERATURE:-0.0}'
export ACE_AGENT_TOP_P='${ACE_AGENT_TOP_P:-1.0}'
```

用于控制被评测模型（agent）的调用参数：

- `ACE_AGENT_API_KEY`：模型 API Key
- `ACE_AGENT_BASE_URL`：模型服务地址
- `ACE_AGENT_MODEL`：模型名，默认 `deepseek-v3`
- `ACE_AGENT_MAX_TOKENS`：最大生成 token 数
- `ACE_AGENT_TEMPERATURE`：采样温度
- `ACE_AGENT_TOP_P`：top-p 参数

额外透传参数：

```bash
export ACE_AGENT_REQUEST_ARGS='{'stream': false}'
export ACE_AGENT_EXTRA_BODY='{}'
export ACE_AGENT_EXTRA_HEADERS='{}'
```

- `ACE_AGENT_REQUEST_ARGS`：请求参数透传
- `ACE_AGENT_EXTRA_BODY`：额外请求体
- `ACE_AGENT_EXTRA_HEADERS`：额外请求头

可选 reasoning 参数：

```bash
export ACE_AGENT_REASONING_EFFORT='${ACE_AGENT_REASONING_EFFORT:-}'
```

如果模型支持推理强度控制，可通过该变量配置。

---

### 3. User Simulator 配置

```bash
export ACE_USER_API_KEY='${ACE_USER_API_KEY:-}'
export ACE_USER_BASE_URL='${ACE_USER_BASE_URL:-}'
export ACE_USER_MODEL='${ACE_USER_MODEL:-deepseek-v3}'

export ACE_USER_MAX_TOKENS='${ACE_USER_MAX_TOKENS:-4096}'
export ACE_USER_TEMPERATURE='${ACE_USER_TEMPERATURE:-0.7}'
export ACE_USER_TOP_P='${ACE_USER_TOP_P:-1.0}'

export ACE_USER_REQUEST_ARGS='{}'
export ACE_USER_EXTRA_BODY='{}'
export ACE_USER_EXTRA_HEADERS='{}'
```

用于控制 user simulator：

- `ACE_USER_API_KEY`
- `ACE_USER_BASE_URL`
- `ACE_USER_MODEL`
- `ACE_USER_MAX_TOKENS`
- `ACE_USER_TEMPERATURE`
- `ACE_USER_TOP_P`

---

### 4. 评测控制参数

```bash
export ACE_TASK_IDS='${ACE_TASK_IDS:-}'
export ACE_SKIP='${ACE_SKIP:-0}'
export ACE_LIMIT='${ACE_LIMIT:-}'

export ACE_AGENT_NAME='${ACE_AGENT_NAME:-ace_agent}'
export ACE_LANGUAGE='${ACE_LANGUAGE:-zh}'
export ACE_CATEGORY='${ACE_CATEGORY:-agent}'
```

含义如下：

- `ACE_TASK_IDS`：指定评测 task id，多个用逗号分隔（可为空）
- `ACE_SKIP`：跳过前 N 条样本（可为空）
- `ACE_LIMIT`：限制样本数量（可为空）
- `ACE_AGENT_NAME`：运行时使用的 agent 名称（固定为ace_agent）
- `ACE_LANGUAGE`：评测语言（zh=中文评测集，en=英文评测集）
- `ACE_CATEGORY`：评测类别 （存在三种评测类别: agent、special、normal，或者test_all(全部三种类别)）

也就是说：

- `agent`：只评测 agent 类
- `special`：只评测 special 类
- `normal`：只评测 normal 类
- `test_all`：评测 `agent / special / normal`

---

## 输出文件命名规则

脚本会基于以下信息自动生成输出文件名：

- 评测类别
- agent model
- user model
- 时间戳
- hash

示例：

```bash
./workspace/result/ace_category_agent_AGENT_MODEL_deepseek-v3_USER_MODEL_deepseek-v3_20260413_175200_123_abcd1234.db
```

同名 `.jsonl` 和 `.csv` 文件也会在同目录生成。

---

## 执行流程说明

### 1. 运行 batch 评测

```bash
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/ace/generator.py \
  -d "$DB_PATH" \
  --max-concurrent "$MAX_CONCURRENT"
```

参数说明：

- `-g`：指定 generator 脚本
- `-d`：指定 sqlite db 输出路径
- `--max-concurrent`：并发数，默认 20

---

### 2. 导出 db 为 jsonl

脚本当前写法为：

```bash
uv run nexus export "$DB_PATH" -o "${DB_PATH%.db}.jsonl"
```

---

### 3. 运行 stasis 分析

```bash
uv run python ./nexus/extensions/tasks/ace/stasis.py "${DB_PATH%.db}.jsonl" \
  --verbose --csv "${DB_PATH%.db}.csv"
```

作用：

- 读取导出的 jsonl 结果
- 输出详细分析
- 生成 csv 文件

---

## 完整示例

```bash
#!/usr/bin/env bash
set -euo pipefail


# =========================
# dataset path
# =========================
DATASET_PATH="/workspace/ACEBench"
export ACE_BENCH_DATASET_PATH="$DATASET_PATH"


# =========================
# Agent config
# =========================
export ACE_AGENT_API_KEY="${ACE_AGENT_API_KEY:-}"
export ACE_AGENT_BASE_URL="${ACE_AGENT_BASE_URL:-}"
export ACE_AGENT_MODEL="${ACE_AGENT_MODEL:-deepseek-v3}"
export ACE_AGENT_MAX_TOKENS="${ACE_AGENT_MAX_TOKENS:-4096}"
export ACE_AGENT_TEMPERATURE="${ACE_AGENT_TEMPERATURE:-0.0}"
export ACE_AGENT_TOP_P="${ACE_AGENT_TOP_P:-1.0}"

export ACE_AGENT_REQUEST_ARGS="{"stream": false}"
export ACE_AGENT_EXTRA_BODY="{}"
export ACE_AGENT_EXTRA_HEADERS="{}"

export ACE_AGENT_REASONING_EFFORT="${ACE_AGENT_REASONING_EFFORT:-}"

# =========================
# User simulator config
# =========================
export ACE_USER_API_KEY="${ACE_USER_API_KEY:-}"
export ACE_USER_BASE_URL="${ACE_USER_BASE_URL:-}"
export ACE_USER_MODEL="${ACE_USER_MODEL:-deepseek-v3}"

export ACE_USER_MAX_TOKENS="${ACE_USER_MAX_TOKENS:-4096}"
export ACE_USER_TEMPERATURE="${ACE_USER_TEMPERATURE:-0.7}"
export ACE_USER_TOP_P="${ACE_USER_TOP_P:-1.0}"

export ACE_USER_REQUEST_ARGS="{}"
export ACE_USER_EXTRA_BODY="{}"
export ACE_USER_EXTRA_HEADERS="{}"

# =========================
# eval config
# =========================
export ACE_TASK_IDS="${ACE_TASK_IDS:-}"
export ACE_SKIP="${ACE_SKIP:-0}"
export ACE_LIMIT="${ACE_LIMIT:-}"
export ACE_AGENT_NAME="${ACE_AGENT_NAME:-ace_agent}"

export ACE_LANGUAGE="${ACE_LANGUAGE:-zh}"
export ACE_CATEGORY="${ACE_CATEGORY:-normal}"


# =========================
# 输出结果地址：ace_category_评测类别_AGENT_MODEL_使用的AGENT_MODEL_USER_MODEL_使用的USER_MODEL_当前时间_md5值
# =========================
NOW=$(date +'%Y%m%d_%H%M%S_%3N')

SAFE_AGENT_MODEL=$(echo $ACE_AGENT_MODEL | tr '/' '_')
SAFE_USER_MODEL=$(echo $ACE_USER_MODEL | tr '/' '_')

RAW_STR="${ACE_CATEGORY}_${SAFE_AGENT_MODEL}_${SAFE_USER_MODEL}_${NOW}"
HASH=$(echo -n "$RAW_STR" | md5sum | cut -c1-8)

result_path='/workspace/result'

# 如果目录不存在就创建
mkdir -p "$result_path"


DB_PATH="${ACE_DB_PATH:-./workspace/result/ace_category_${ACE_CATEGORY}_AGENT_MODEL_${SAFE_AGENT_MODEL}_USER_MODEL_${SAFE_USER_MODEL}_${NOW}_${HASH}.db}"


# =========================
# 执行流程
# =========================

# step1: run batch

# 并发数
MAX_CONCURRENT="${ACE_MAX_CONCURRENT:-20}"

uv run nexus run-batch \
-g ./nexus/extensions/tasks/ace/generator.py \
-d "$DB_PATH" \
--max-concurrent "$MAX_CONCURRENT"

# step2: convert
uv run nexus export "$DB_PATH" -o "${DB_PATH%.db}.jsonl"

# step3: run stasis analysis
echo "\n评估类别：$DB_PATH"
uv run python ./nexus/extensions/tasks/ace/stasis.py "${DB_PATH%.db}.jsonl" \
--verbose --csv "${DB_PATH%.db}.csv"
```