# Verifier Module - 网站操作任务验证器

验证网站操作任务执行结果是否符合预期。

## 模块结构

```
verifier/
├── __init__.py           # 模块入口
├── base.py               # 基础类和工具函数
├── llm_client.py         # LLM 客户端（语义比较）
├── navigation_verifier.py # 导航验证器
├── state_verifier.py     # 状态验证器
├── retrieval_verifier.py # 检索验证器
├── action_verifier.py    # 动作验证器
├── composite_verifier.py # 复合验证器
└── README.md
```

## 整体流程

```
                    ┌─────────────────────────────────────┐
                    │         CompositeVerifier           │
                    │   (按 type 匹配 groundtruth/response) │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│NavigationVerifier│    │  StateVerifier  │    │ ActionVerifier  │
│  (页面导航验证)  │    │  (状态变更验证)  │    │  (动作序列验证)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                   │
                                   ▼
                       ┌─────────────────┐
                       │RetrievalVerifier│
                       │ (内容检索验证)   │
                       │   + LLMClient   │
                       └─────────────────┘
```

---

## Groundtruth & Response 完整结构说明

### 1. Navigation（导航验证）

验证页面导航路径。

| Groundtruth 字段 | 类型 | 必填 | 说明 |
|-----------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"navigation"` |
| `expected_pages` | array | ✅ | 期望的页面路径列表 |
| `options.mode` | string | ❌ | `"final"`(默认) / `"contains"` / `"exact"` |

| Response 字段 | 类型 | 必填 | 说明 |
|--------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"navigation"` |
| `executed_pages` | array | ✅ | 实际导航的页面路径列表 |

**验证模式：**
| mode | 说明 |
|------|-----|
| `final` | 只比较末元素，验证最终页面 |
| `contains` | expected_pages 是 executed_pages 的有序子序列 |
| `exact` | 两个列表完全相等 |

**示例：**
```python
# Groundtruth
{
    'type': 'navigation',
    'expected_pages': ['/home', '/category', '/merchant/chayanyuese'],
    'options': {'mode': 'contains'},
}

# Response
{'type': 'navigation', 'executed_pages': ['/home', '/search', '/category', '/merchant/chayanyuese']}
```

---

### 2. State（状态验证）

验证应用状态（如购物车、表单、用户信息等）。

| Groundtruth 字段 | 类型 | 必填 | 说明 |
|-----------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"state"` |
| `expected_states` | object | ✅ | 期望的状态对象（值为 `null` 表示该键应不存在） |
| `options.mode` | string | ❌ | `"include"`(默认) / `"exact"` |

| Response 字段 | 类型 | 必填 | 说明 |
|--------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"state"` |
| `executed_states` | object | ✅ | 当前状态对象 |

**验证模式：**
| mode | 说明 |
|------|-----|
| `include` | 部分匹配：验证 expected_states 中的键值（`null` 表示该键应不存在） |
| `exact` | 精确匹配：两者完全相等 |

**示例 - 新增/更新：**
```python
# Groundtruth
{'type': 'state', 'expected_states': {'cart': [{'name': '幽兰拿铁', 'quantity': 1}]}, 'options': {'mode': 'include'}}

# Response
{
    'type': 'state',
    'executed_states': {
        'cart': [{'name': '幽兰拿铁', 'quantity': 1}, {'name': '声声乌龙', 'quantity': 2}],
        'user': {'logged_in': true},
    },
}
# ✅ 通过：包含期望的 cart 内容
```

**示例 - 删除验证：**
```python
# Groundtruth
{'type': 'state', 'expected_states': {'cart': null}, 'options': {'mode': 'include'}}

# Response
{'type': 'state', 'executed_states': {'user': {'logged_in': true}}}
# ✅ 通过：cart 键不存在
```

**示例 - 混合场景（同时验证存在和不存在）：**
```python
# Groundtruth
{
    'type': 'state',
    'expected_states': {
        'cart': [{'name': '幽兰拿铁', 'quantity': 1}],  # 应存在
        'temp_data': null,  # 应不存在
        'user': {'logged_in': true},  # 应存在
    },
    'options': {'mode': 'include'},
}

# Response
{
    'type': 'state',
    'executed_states': {
        'cart': [{'name': '幽兰拿铁', 'quantity': 1}],
        'user': {'logged_in': true},
        'other_field': 'xxx',
    },
}
# ✅ 通过：cart 和 user 匹配，temp_data 不存在
```

**示例 - 精确匹配：**
```python
# Groundtruth
{'type': 'state', 'expected_states': {'cart': [{'name': '幽兰拿铁', 'quantity': 1}]}, 'options': {'mode': 'exact'}}

# Response
{'type': 'state', 'executed_states': {'cart': [{'name': '幽兰拿铁', 'quantity': 1}]}}
# ✅ 通过：完全相等
```

---

### 3. Retrieval（内容检索验证）

验证检索到的文本内容（语义相似度匹配）。

| Groundtruth 字段 | 类型 | 必填 | 说明 |
|-----------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"retrieval"` |
| `expected_content` | string | ✅ | 期望的文本内容 |
| `options.similarity_threshold` | float | ❌ | 相似度阈值，默认 0.8 |

| Response 字段 | 类型 | 必填 | 说明 |
|--------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"retrieval"` |
| `executed_content` | string | ✅ | 实际检索到的内容 |

**示例：**
```python
# Groundtruth
{'type': 'retrieval', 'expected_content': '幽兰拿铁：18元，冰/热可选', 'options': {'similarity_threshold': 0.8}}

# Response
{'type': 'retrieval', 'executed_content': '幽兰拿铁，售价18元，支持冰饮或热饮'}
```

---

### 4. Action（动作验证）

验证执行的操作动作序列。

| Groundtruth 字段 | 类型 | 必填 | 说明 |
|-----------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"action"` |
| `expected_actions` | array | ✅ | 期望的动作序列 |
| `options.mode` | string | ❌ | `"final"`(默认) / `"contains"` / `"exact"` |

| Response 字段 | 类型 | 必填 | 说明 |
|--------------|------|-----|------|
| `type` | string | ✅ | 固定值 `"action"` |
| `executed_actions` | array | ✅ | 实际执行的动作序列 |

**验证模式：**
| mode | 说明 |
|------|-----|
| `final` | 只比较末元素，验证最后一个动作 |
| `contains` | expected_actions 是 executed_actions 的有序子序列 |
| `exact` | 两个列表完全相等 |

**示例：**
```python
# Groundtruth
{
    'type': 'action',
    'expected_actions': ['click_product', 'click_add_to_cart', 'click_checkout'],
    'options': {'mode': 'contains'},
}

# Response
{
    'type': 'action',
    'executed_actions': ['click_product', 'scroll_down', 'click_add_to_cart', 'click_cart', 'click_checkout'],
}
```

---

## 字段对照总表

| 类型 | Groundtruth 字段 | Response 字段 |
|-----|-----------------|---------------|
| navigation | `expected_pages` | `executed_pages` |
| state | `expected_states` | `executed_states` |
| retrieval | `expected_content` | `executed_content` |
| action | `expected_actions` | `executed_actions` |

## Options 汇总

| 类型 | options.mode 可选值 | 默认值 | 其他 options |
|-----|-------------------|-------|-------------|
| navigation | `final` / `contains` / `exact` | `final` | - |
| state | `include` / `exact` | `include` | - |
| retrieval | - | - | `similarity_threshold` (默认 0.8) |
| action | `final` / `contains` / `exact` | `final` | - |

---

## 完整示例

**Groundtruth:**
```python
{
    'groundtruth': [
        {'type': 'navigation', 'expected_pages': ['/home', '/merchant/chayanyuese'], 'options': {'mode': 'contains'}},
        {
            'type': 'state',
            'expected_states': {'cart': [{'name': '幽兰拿铁', 'quantity': 1}], 'temp_data': null},
            'options': {'mode': 'include'},
        },
        {'type': 'retrieval', 'expected_content': '幽兰拿铁：18元', 'options': {'similarity_threshold': 0.8}},
        {'type': 'action', 'expected_actions': ['click_product', 'click_add_to_cart'], 'options': {'mode': 'contains'}},
    ],
}
```

**Response:**
```python
{
    'response': [
        {'type': 'navigation', 'executed_pages': ['/home', '/search', '/merchant/chayanyuese']},
        {
            'type': 'state',
            'executed_states': {'cart': [{'name': '幽兰拿铁', 'quantity': 1}], 'user': {'logged_in': true}},
        },
        {'type': 'retrieval', 'executed_content': '幽兰拿铁，售价18元'},
        {'type': 'action', 'executed_actions': ['click_product', 'scroll_down', 'click_add_to_cart']},
    ],
}
```

---

## 使用方法

### 单类型验证

```python
from verifier import NavigationVerifier, StateVerifier, ActionVerifier, RetrievalVerifier

# 导航验证
nav_verifier = NavigationVerifier()
result = nav_verifier.verify(
    groundtruth={'expected_pages': ['/home', '/merchant']},
    response={'executed_pages': ['/home', '/search', '/merchant']},
    options={'mode': 'contains'},
)
print(result.success, result.score, result.message)

# 状态验证
state_verifier = StateVerifier()
result = state_verifier.verify(
    groundtruth={'expected_states': {'cart': [{'name': '拿铁', 'quantity': 1}]}},
    response={'executed_states': {'cart': [{'name': '拿铁', 'quantity': 1}]}},
    options={'mode': 'include'},
)

# 动作验证
action_verifier = ActionVerifier()
result = action_verifier.verify(
    groundtruth={'expected_actions': ['click_add', 'click_checkout']},
    response={'executed_actions': ['click_add', 'scroll', 'click_checkout']},
    options={'mode': 'contains'},
)
```

### 复合验证

```python
from verifier import CompositeVerifier

verifier = CompositeVerifier()

groundtruth = {
    'groundtruth': [
        {'type': 'navigation', 'expected_pages': ['/merchant/chayanyuese'], 'options': {'mode': 'final'}},
        {
            'type': 'state',
            'expected_states': {'cart': [{'name': '幽兰拿铁', 'quantity': 1}]},
            'options': {'mode': 'include'},
        },
    ],
}

response = {
    'response': [
        {'type': 'navigation', 'executed_pages': ['/home', '/merchant/chayanyuese']},
        {'type': 'state', 'executed_states': {'cart': [{'name': '幽兰拿铁', 'quantity': 1}]}},
    ],
}

result = verifier.verify(groundtruth, response)
print(f'通过: {result.success}, 得分: {result.score}')
print(f'详情: {result.details}')
```

### 单项便捷验证

```python
from verifier import CompositeVerifier

verifier = CompositeVerifier()

# 使用 verify_single 方法直接验证单个类型
result = verifier.verify_single(
    item_type='navigation',
    groundtruth={'expected_pages': ['/home', '/merchant']},
    response={'executed_pages': ['/home', '/merchant']},
    options={'mode': 'final'},
)
```

---

## 测试方法

```bash
# 在项目根目录运行
cd websites/tasks

# 运行单元测试
python -m pytest verifier/ -v

# 测试单个验证器
python -c "
from verifier import NavigationVerifier

v = NavigationVerifier()
result = v.verify(
    {'expected_pages': ['/home']},
    {'executed_pages': ['/home']},
    {'mode': 'final'}
)
print(result)
"
```

---

## 文件功能说明

| 文件 | 功能 |
|-----|------|
| `base.py` | 定义 `VerificationResult` 结果类和 `BaseVerifier` 抽象基类 |
| `llm_client.py` | LLM 客户端，用于 RetrievalVerifier 的语义比较 |
| `navigation_verifier.py` | 页面导航验证，支持 final/contains/exact 三种模式 |
| `state_verifier.py` | 状态验证，支持 include/exact 模式，null 值表示删除 |
| `retrieval_verifier.py` | 内容检索验证，使用 LLM 进行语义相似度比较 |
| `action_verifier.py` | 动作序列验证，支持 final/contains/exact 三种模式 |
| `composite_verifier.py` | 复合验证器，按 type 匹配并组合多个验证项 |

---

## VerificationResult 结构

```python
@dataclass
class VerificationResult:
    success: bool  # 是否通过
    score: float  # 得分 (0.0 - 1.0)
    message: str  # 结果描述
    details: Dict[str, Any]  # 详细信息
```

**创建方法：**
- `VerificationResult.passed(message, details)` - 创建通过结果 (score=1.0)
- `VerificationResult.failed(message, score, details)` - 创建失败结果
- `VerificationResult.partial(score, message, details)` - 创建部分通过结果
