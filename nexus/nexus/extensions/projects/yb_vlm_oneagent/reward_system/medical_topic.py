"""Medical topic rubric scoring.

This scorer reuses the shared ``yb_rubric_base`` infrastructure and only
customizes the judge prompt plus overall-score weighting for medical data.
"""

import json

from typing import Any

from .yb_rubric_base import (
    render_history,
    yb_rubric_score_impl,
)


SCORE_SYSTEM_PROMPT = """
你是一位严谨的按评分清单逐条评判的评分教师。你的任务是根据【评分清单】中每条考点的 conclusion，判定学生回答对每个考点的满足情况，并据此打分。

**核心原则**：
1. 每条考点的满足状态是固定类型，不得创建新的满足类型
2. 对数字、范围、时间等事实点保持高度敏感，答案须与考点在数值上一致（允许等价表达、合理近似和精度容忍，但不允许实质性数值偏差）
3. 严禁引入外部世界知识，审查完全基于给定考点
4. 每条考点独立判定，互不影响
""".strip()

SCORE_USER_PROMPT_TEMPLATE = """
## 一、输入信息

### 1.1 用户问题
{question}

### 1.2 提问时间
{date}

### 1.3 历史对话
{history}

### 1.4 评分清单（checklist）
{checklist}

### 1.5 学生回答
{answer}

---

## 二、满足状态判定

### 2.1 主需（need）—— 用户核心需求，最关键

**医疗类主需严格判定原则**：conclusion 中如果明确列出了多个维度/要点（如用①②③编号、分号分隔、或"必含以下N项"等方式列举），则答案必须**逐一覆盖所有列出的维度**才能判为完全满足。仅提及部分维度应按覆盖比例降低得分，而非直接判为满足。

| 状态 | 判定规则 |
|------|---------|
| **提及-满足** | 答案**完整覆盖** conclusion 中列出的所有维度/要点且符合要求。含义相同即可；conclusion 中用"或称/或/又称/也叫"及括号列出的多个名称互为等价，答案匹配**任一**即可；举例类（"如xxx"）类似即可；概念类言之有理即可；枚举类只要已列举项**全部无错误**即属此状态。**注意**：如果 conclusion 明确要求覆盖 N 个要点，答案必须全部覆盖才属此状态 |
| **提及-不满足** | 答案提及但存在**事实错误**（数字/范围/时间/结论等与考点不一致），或对核心定性方向判断错误。枚举类仅当列举项中存在事实错误时才属此状态 |
| **未提及-不满足** | 答案完全未提及该考点 |

#### 主需类型（need_type）

每条主需根据其在回答链路中的角色，分为三类：

| need_type | 含义 | 判定口诀 |
|-----------|------|---------|
| **entity** | 核心实体识别：回答"是什么/是谁"的身份判定 | "认错了东西" |
| **answer** | 核心答案：直接回答用户问的数值/结论/操作 | "东西认对了但信息错了" |
| **supporting** | 支撑补充：识别依据、背景信息、注意事项等 | "答案对了但不够完整" |

> **注意**：不是所有问题都有 entity 类型。医疗报告解读类问题通常以 answer 和 supporting 为主。

### 2.2 延展（extension）—— 补充信息，有则加分

| 状态 | 判定规则 |
|------|---------|
| **未提及-满足** | 未提及该考点，或仅提及主体名词但未做实质性展开或错误解释 |
| **提及-满足** | 提及且内容符合考点要求（允许部分覆盖，按覆盖程度在 0.85-1.0 区间内取值） |
| **提及-不满足** | 提及但存在事实错误 |

> **排序原则：答错 < 不答 < 正确作答**

### 2.3 陷阱（pitfall）—— 常见错误检验

| 状态 | 判定规则 |
|------|---------|
| **未命中** | 未出现陷阱提及的任何问题 |
| **命中** | 命中陷阱中**部分或全部**问题 |

### 2.4 判定约束

1. **不得创建上述之外的新满足类型**
2. **严禁引入外部知识**
3. **数字敏感**：数字/范围/时间要求与考点一致（受下述等价和容忍规则约束）
4. **区间子集原则（仅数值属性）**：当考点给出的范围表示**数值属性**（如"正常范围3.5-5.5mmol/L""参考值0-40U/L"），答案区间 ⊆ 考点区间才算满足
5. **边界词语义**：
   - "超过/多于/大于X"：答案须 > X
   - "至少/不少于/≥X"：答案须 ≥ X
   - "不到/少于/低于X"：答案须 < X
   - "最多/不超过/≤X"：答案须 ≤ X
   - "X以上"/"X以下"：含X本身
6. **等价表达认可**（视为数值一致）：
   - 单位换算：1kg=1000g，100cm=1m，1.5小时=90分钟
   - 格式等价：1999=1,999=一千九百九十九
   - 百分比等价：50%=0.5=五成
   - 近似表达：答案含"约/大约/左右/近/接近"时，若考点精确值落在合理近似范围内视为满足
   - 精度/日期容忍：答案可用更低精度表述，已表述精度内须一致，不得导致数值实质性改变
7. **名称等价（conclusion 内的别称/同义表述）**：当 conclusion 中使用"或称/或/又称/也叫/即/又名"以及**括号内**列出多个名称时，这些名称**互为等价**，答案匹配其中**任何一个**即视为满足。判定时须逐一核对 conclusion 中列出的所有等价名称。
8. **数量要求宽松原则**：当考点给出的范围表示**数量要求**（如"提供2-3个""列举2-3个"），答案数量≥下限即满足（多答不扣分，少于下限才不满足）
9. **举例/枚举类规则**：当 conclusion 中以举例形式给出答案（含"如""例如""等"标志词）且附带数量要求时：
   - **内容开放性**：答案**不要求与示例完全一致**，只要逻辑自洽且解释合理即可满足
   - **数量基准**：覆盖比例以 conclusion 中**明确要求的数量**为基准，而非 conclusion 中列举的所有示例数量
10. **医疗免责声明宽松判定**：对于涉及"医疗免责"类的考点，判定标准应从宽。只要回答中包含任何形式的免责相关表述即视为满足，包括但不限于：建议就医/咨询医生、仅供参考、不构成医疗建议、以医生诊断为准、建议进一步检查、请遵医嘱等。即使措辞不够正式或未使用"免责声明"字样，只要传达了"本回答不替代专业医疗意见"的含义即可判为"提及-满足"。

---

## 三、评分映射

### 3.1 主需（need）

#### 提及-满足 → 按覆盖程度（0.5 - 1.0）

| 覆盖程度 | score | 标准 |
|---------|-------|------|
| 完全覆盖 | 1.0 | 所有要点正确覆盖（枚举类：达到要求数量且全对） |
| 大部分覆盖 | 0.8 | 核心全覆盖，缺个别次要细节（枚举类：达到要求数量下限且全对） |
| 部分覆盖 | 0.5 | 覆盖部分要点，有缺失但无错误（枚举类：未达要求但≥1个全对） |

> **枚举类基准**：覆盖比例以 conclusion 中**要求的数量**为基准（非示例总数）。0个正确→未提及-不满足。

#### 提及-不满足 → 基础分 × 惩罚系数（上限 0.4）

**基础分**（该考点中正确部分的比例）：

| 档位 | 基础分 | 标准 |
|------|-------|------|
| 其余全对 | 0.8 | 除错误点外其余内容全部正确 |
| 大部分对 | 0.5 | 多数内容正确，少数错误 |
| 少量对 | 0.3 | 少数内容正确，多数错误或缺失 |
| 几乎无 | 0.0 | 几乎无正确内容，或考点仅含单一核心数据且该数据错误 |

**惩罚系数**（错误的严重程度）：

| 错误程度 | 惩罚系数 | 标准 |
|---------|---------|------|
| 轻微 | ×0.5 | 次要细节不一致，核心正确 |
| 中等 | ×0.25 | 重要数据/属性错误，实体正确 |
| 严重 | ×0.0 | 核心实体错误或结论完全相反 |

> **注意**：need_type="entity" 且状态为"提及-不满足"时，错误程度固定为**严重**（×0.0），得分为 0.0。

#### 未提及-不满足 → 0.0

---

### 3.2 延展（extension）

#### 核心排序：答错(≤0.75) < 不答(0.8) < 正确作答(0.85-1.0)，三档互不交叉。

| 状态 | score |
|------|-------|
| 提及-满足（完全覆盖） | 1.0 |
| 提及-满足（大部分覆盖） | 0.9 |
| 提及-满足（部分覆盖） | 0.85 |
| 未提及-满足 | 0.8（固定） |
| 提及-不满足（轻微） | 0.75 |
| 提及-不满足（中等） | 0.4 |
| 提及-不满足（严重） | 0.0 |

---

### 3.3 陷阱（pitfall）

| 状态 | score |
|------|-------|
| 未命中 | 1.0 |
| 命中 | 0.0 |

---

## 四、输出格式

### 4.1 推理链路与字段顺序

输出 JSON 时，每条考点的字段**严格按推理链路排序**——先思考再结论：

```
步骤1：读取考点信息    → kind, id, statement, conclusion
步骤2：判定主需类型    → need_type（仅need）
步骤3：分析推理过程    → score_reason（完整的思考链路）
步骤4：给出判定结论    → satisfaction_status
步骤5：给出最终分数    → score
```

### 4.2 JSON 结构

**JSON 安全规则**：所有字段值中的引号统一使用中文引号（""或''），禁止在字段值内部出现英文双引号（"），以确保 JSON 合法可解析。

```json
{{
  "scored_checklist": [
    {{
      "kind": "原始kind值",
      "id": "原始id值",
      "statement": "原始statement值",
      "conclusion": "原始conclusion值",
      "need_type": "entity/answer/supporting/null",
      "score_reason": "推理过程：need_type判定理由 → 答案与考点对比分析（逐一核对conclusion中列出的各维度/要点）→ 满足状态判定 → 得分推导（提及-不满足时须显式写出基础分档位和惩罚系数档位的查表过程）",
      "satisfaction_status": "满足状态",
      "score": 0.0-1.0
    }}
  ]
}}
```

### 4.3 字段约束

**need_type**：

| kind | 允许取值 | 说明 |
|------|---------|------|
| need | **"entity"** / **"answer"** / **"supporting"** | 必填 |
| extension | **null** | 延展不设类型 |
| pitfall | **null** | 陷阱不设类型 |

**satisfaction_status**：

| kind | 允许取值 |
|------|---------|
| need | "提及-满足" / "提及-不满足" / "未提及-不满足" |
| extension | "未提及-满足" / "提及-满足" / "提及-不满足" |
| pitfall | "未命中" / "命中" |

### 4.4 score_reason 要求

score_reason 应体现完整推理链路：**① 类型判定 → ② 对比分析 → ③ 状态判定 → ④ 得分推导**

**关键要求**：
- 涉及名称等价时，必须列出 conclusion 中的**所有等价名称/别称**，逐一与答案核对
- 涉及多维度/多要点时，必须**逐一列出 conclusion 要求的各维度**，逐一判断答案是否覆盖
- "提及-不满足"的得分推导中，必须**显式写出基础分档位→数值、惩罚系数档位→数值的完整查表过程**，不得跳步

---

## 五、自检清单

- [ ] 每条考点已判定并评分，无遗漏；need 已判定 need_type 并在 score_reason 中说明理由
- [ ] entity + 提及-不满足 → score 固定 0.0；extension/pitfall 的 need_type 为 null
- [ ] 涉及名称等价时已逐一核对 conclusion 中所有等价名称，未遗漏
- [ ] 涉及多维度要求时已逐一核对各维度覆盖情况，按覆盖比例给分
- [ ] 提及-不满足的得分推导已显式写出基础分档位和惩罚系数档位的查表过程
- [ ] 举例/枚举类：覆盖比例以**要求数量**为基准（非示例总数）；内容合理即可，不要求与示例完全一致
- [ ] 分数区间：need提及-满足∈[0.5,1.0]，提及-不满足≤0.4；extension答错(≤0.75)<不答(0.8)<正确(≥0.85)
- [ ] 未引入外部知识；JSON格式有效，字段顺序按推理链路排列

---

请严格按照上述规则，对学生回答进行逐条判定和评分，输出完整 JSON。
""".strip()


_WEIGHT_MAP = {
    4: 0.5,
    3: 3.0,
    2: 2.0,
    1: 1.0,
    0: 0.2,
}


def _is_pitfall_violated(item: dict[str, Any]) -> bool:
    """Return whether a medical pitfall item was hit."""
    satisfaction_status = item.get('satisfaction_status', '') or ''
    status_lower = satisfaction_status.replace(' ', '').replace('-', '').lower()
    if '未命中' in satisfaction_status or 'notviolated' in status_lower or 'not_violated' in status_lower:
        return False
    if '命中' in satisfaction_status or 'violated' in status_lower:
        return True
    score = item.get('score', 0.0)
    try:
        score = float(score)
    except Exception:
        score = 0.0
    return score <= 0.0


def _get_calc_weight(kind: str, item_weight: object) -> float:
    if kind == 'pitfall':
        return 5.0
    if kind == 'extension':
        return 0.5
    if item_weight is not None and item_weight in _WEIGHT_MAP:
        return _WEIGHT_MAP[item_weight]
    return 2.0


def get_overall_score(checklist: list[dict[str, Any]]) -> float:
    """Compute medical overall score with kind-specific safety weights."""
    total_weighted_score = 0.0
    total_weight = 0.0

    for item in checklist:
        kind = item.get('kind', 'need')
        item_weight = item.get('weight', None)
        calc_weight = _get_calc_weight(kind, item_weight)

        score = item.get('score', 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0

        if kind == 'pitfall':
            score = 0.0 if _is_pitfall_violated(item) else 1.0

        total_weighted_score += score * calc_weight
        total_weight += calc_weight

    if total_weight == 0.0:
        return 0.0
    return round(total_weighted_score / total_weight, 4)


def build_scoring_prompt(
    question: str,
    messages: list[dict[str, Any]],
    checklist: list[dict[str, Any]],
    answer: str,
    date: str = '未知',
) -> str:
    """Build the medical scoring user prompt from the rubric checklist and answer."""
    history = render_history(messages)
    checklist_text = json.dumps(checklist, ensure_ascii=False, indent=2)
    return SCORE_USER_PROMPT_TEMPLATE.format(
        question=question,
        date=date,
        history=history,
        checklist=checklist_text,
        answer=answer,
    )


async def medical_rubric_score(obj: dict[str, Any]) -> dict[str, Any]:
    """Rubric-based scoring function for the Medical topic."""
    return await yb_rubric_score_impl(
        obj=obj,
        build_scoring_prompt_fn=build_scoring_prompt,
        get_overall_score_fn=get_overall_score,
        system_prompt=SCORE_SYSTEM_PROMPT,
        version_tag='medical',
    )
