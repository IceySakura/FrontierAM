# WebArena-Verified Task

WebArena-Verified 是 WebArena benchmark 的验证版本，具有以下特点：

- **确定性评估**：不依赖 LLM-as-Judge，使用类型感知的值归一化和结构化比较
- **812 个验证任务**：经过人工审计的参考答案和评估器
- **Hard 子集 (258 个)**：高难度任务，用于快速评估核心能力

## 核心改进

### 1. 确定性评分

传统 WebArena 使用 LLM 来判断答案正确性，这导致：
- 评分结果不可复现
- 不同运行结果可能不同
- 依赖外部 LLM 服务

WebArena-Verified 使用确定性方法：

| 特性 | 传统方法 | WebArena-Verified |
|------|----------|-------------------|
| 数字比较 | `"$100.00" == "$100"` ❌ | 标准化后 `100.0 == 100.0` ✅ |
| 布尔比较 | `"yes" == "true"` ❌ | 标准化后 `True == True` ✅ |
| 日期比较 | `"2024-04-08" == "04/08/2024"` ❌ | 标准化后相等 ✅ |
| 时间格式 | `"00:15:00" == "15min"` ❌ | 标准化后 `900s == 900s` ✅ |
| 州名缩写 | `"PA" == "Pennsylvania"` ❌ | 标准化后相等 ✅ |
| JSON 顺序 | `{"a":1,"b":2} == {"b":2,"a":1}` ❌ | key 顺序无关 ✅ |
| 备选答案 | 需要 LLM 判断 | `NormalizedType(["a","b"])` 匹配任一 |

### 2. 类型感知归一化

支持的数据类型 (`data_types.py`)：

| 类型 | 类名 | 说明 |
|------|------|------|
| `string` | `NormalizedString` | 大小写不敏感、空白规范化 |
| `number/integer/float` | `NormalizedNumber` | 容差比较 (1e-6)，支持百分比 |
| `currency` | `NormalizedCurrency` | 移除符号，精度对齐 (0.01) |
| `boolean` | `NormalizedBoolean` | yes/no/true/false/1/0 解析 |
| `date` | `NormalizedDate` | 多格式支持，ISO 输出 |
| `url` | `NormalizedUrl` | 尾斜杠处理，大小写规范 |
| `duration` | `NormalizedDuration` | HH:MM:SS ↔ Xmin ↔ Xhr Ymin (120s 容差) |
| `distance` | `NormalizedDistance` | km/miles/feet → 米 (50m 容差) |
| `location-name` | `NormalizedLocationName` | 缩写扩展 (St.→Street, Ave.→Avenue) |
| `full_address` | `NormalizedFullAddress` | 州名缩写 (PA→Pennsylvania)，地址规范化 |
| `coordinates` | `NormalizedCoordinates` | 坐标容差 (0.001°) |

### 3. 结构化比较

支持嵌套对象和数组的递归比较：
- 有序/无序数组比较
- JSON key 顺序无关
- 忽略额外字段
- 备选值匹配

### 4. 评估器类型

| 评估器 | 数量 | 说明 |
|--------|------|------|
| `AgentResponseEvaluator` | 324 | 检查 agent 返回的数据 |
| `NetworkEventEvaluator` | 488 | 检查 agent 是否访问了特定 URL |

**NetworkEventEvaluator 支持**：
- 单个 URL 匹配
- 多 URL 列表 (OR 关系)
- 正则表达式模式 (`^...$`)
- 忽略的 query 参数模式

## 使用方法

### 环境变量配置

```bash
# 数据集路径
export WEBARENA_VERIFIED_DATASET_PATH='./workspace/webarena-verified.json'

# 指定任务 ID（可选）
export WEBARENA_VERIFIED_TASK_IDS='0,1,2,3'

# 跳过任务数量
export WEBARENA_VERIFIED_SKIP=0

# 最大任务数量
export WEBARENA_VERIFIED_LIMIT=100

# 只运行 Hard 子集
export WEBARENA_VERIFIED_HARD_ONLY=true

# 模型配置
export OPENAI_API_KEY='your_api_key'
export OPENAI_BASE_URL='http://your-api-endpoint'
export OPENAI_MODEL='your_model_name'

# 评估配置
export WEBARENA_USE_DETERMINISTIC_EVAL=true  # 使用确定性评估
```

### 运行单个任务

```bash
cd /data/workspace/nexus/nexus

uv run nexus run \
  --task webarena_verified \
  --task-config '{"task_id": 0, "task_config": {...}}' \
  --agent browser_gym_agent
```

### 批量运行

```bash
cd /data/workspace/nexus/nexus

# 运行所有 812 个任务
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/webarena_verified/generator.py \
  -d ./outputs/webarena_verified.db \
  --max-concurrent 5

# 只运行 Hard 子集 (258 个)
WEBARENA_VERIFIED_HARD_ONLY=true uv run nexus run-batch \
  -g ./nexus/extensions/tasks/webarena_verified/generator.py \
  -d ./outputs/webarena_verified_hard.db \
  --max-concurrent 5
```

### 查看结果

```python
import sqlite3
import json

conn = sqlite3.connect('outputs/webarena_verified.db')
cursor = conn.cursor()

# 统计准确率
cursor.execute('''
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN json_extract(task_output, '$.score') > 0 THEN 1 ELSE 0 END) as success,
        AVG(json_extract(task_output, '$.score')) as avg_score
    FROM runs 
    WHERE status = 'success'
''')
total, success, avg_score = cursor.fetchone()
print(f'Total: {total}, Success: {success}, Accuracy: {success/total*100:.2f}%')

# 按网站分类统计
cursor.execute('''
    SELECT 
        json_extract(metadata, '$.sites') as sites,
        COUNT(*) as count,
        AVG(json_extract(task_output, '$.score')) as avg_score
    FROM runs 
    WHERE status = 'success'
    GROUP BY sites
''')
for sites, count, score in cursor.fetchall():
    print(f'{sites}: {count} tasks, {score*100:.2f}% accuracy')

conn.close()
```

## 数据集格式

WebArena-Verified 使用新的评估格式：

```json
{
  "task_id": 0,
  "sites": ["shopping_admin"],
  "start_urls": ["__SHOPPING_ADMIN__/admin/dashboard"],
  "intent": "Get the top-1 best-selling product name(s) in 2022",
  "intent_template": "Get the top-{{n}} best-selling {{entity}} in {{period}}",
  "instantiation_dict": {"n": 1, "entity": "product name(s)", "period": "2022"},
  "eval": [
    {
      "evaluator": "AgentResponseEvaluator",
      "expected": {
        "task_type": "retrieve",
        "status": "SUCCESS",
        "retrieved_data": ["Quest Lumaflex™ Band"]
      },
      "results_schema": {
        "type": "array",
        "items": {"type": "string"}
      }
    }
  ]
}
```

## 文件结构

```
webarena_verified/
├── __init__.py          # 模块入口
├── task.py              # 任务实现
├── generator.py         # 批量生成器
├── evaluator.py         # 确定性评估器
├── data_types.py        # 归一化数据类型
└── README.md            # 文档
```

## 与原版 WebArena 的区别

| 特性 | WebArena | WebArena-Verified |
|------|----------|-------------------|
| 评估方式 | LLM-as-Judge | 确定性评估 |
| 任务数量 | 812 | 812 (+ 258 Hard 子集) |
| 数据格式 | `eval.reference_answers` | `eval[].expected.retrieved_data` |
| URL 字段 | `start_url` | `start_urls` (数组) |
| 评估器 | `string_match`, `url_match` | `AgentResponseEvaluator`, `StringMatchEvaluator` |

## 参考

- [WebArena-Verified GitHub](https://github.com/ServiceNow/webarena-verified)
- [原版 WebArena](https://github.com/web-arena-x/webarena)
