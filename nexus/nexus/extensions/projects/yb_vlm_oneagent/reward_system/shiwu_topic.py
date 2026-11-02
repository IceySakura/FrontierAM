"""
yb_rubric scoring — V4 prompts and weighting (enhanced version, 0510).

Uses rubric-based evaluation: sends the model's response along with a rubric checklist
to a judge LLM, which scores each rubric item. The overall score is a weighted average.

Enhancements over earlier rubric prompts:
  - need_type classification (entity / answer / supporting)
  - Name equivalence rules (或称/又称/括号内别称)
  - Refined extension scoring: 答错(≤0.75) < 不答(0.8) < 正确作答(0.85-1.0)
  - Explicit reasoning chain in score_reason

Weight strategy: need_type based
  - need (entity/answer): 1.0
  - need (supporting): 0.15
  - pitfall: 0.3
  - extension: 0
"""

import json

from typing import Any

from .yb_rubric_base import (
    render_history,
    yb_rubric_score_impl,
)


# ---------------------------------------------------------------------------
# Prompts (V4 — enhanced from VLM-Rubrics/src/tasks/judge_score.py)
# ---------------------------------------------------------------------------

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

| 状态 | 判定规则 |
|------|---------|
| **提及-满足** | 答案提及且符合要求。含义相同即可；conclusion 中用"或称/或/又称/也叫"及括号列出的多个名称互为等价，答案匹配**任一**即可；举例类（"如xxx"）类似即可；概念类言之有理即可；枚举类只要已列举项**全部无错误**即属此状态（含数量不足，详见评分映射中的枚举类规则） |
| **提及-不满足** | 答案提及但存在**事实错误**（数字/范围/时间/结论等与考点不一致）。枚举类仅当列举项中存在事实错误时才属此状态 |
| **未提及-不满足** | 答案完全未提及该考点 |

#### 主需类型（need_type）

每条主需根据其在回答链路中的角色，分为三类：

| need_type | 含义 | 判定口诀 | 答错后果 |
|-----------|------|---------|---------|
| **entity** | 核心实体识别：回答"是什么/是谁"的身份判定 | "认错了东西" | 全链路崩溃，后续考点连带出错 |
| **answer** | 核心答案：直接回答用户问的数值/结论/操作，或答错会导致后续连带出错的前置依赖 | "东西认对了但信息错了"或"前置条件错了" | 没回答对问题，但实体判断不受影响 |
| **supporting** | 支撑补充：识别依据、背景信息、注意事项等 | "答案对了但不够完整" | 缺失不致命，不影响核心答案正确性 |

**entity 的核心实体范围**：品牌+型号、生物品种、人名、地名、组织名、书名、药名、股票名等。
**口诀**："这是什么/谁"错了→entity；"它有什么特征/多少钱"错了→answer 或 supporting。

**典型示例**：

| 用户问题 | 考点 | need_type | 理由 |
|---------|------|-----------|------|
| "这是什么花" | 识别为郁金香 | entity | 核心身份 |
| "这是什么花" | 依据：杯状花型、对生叶片 | supporting | 识别依据，缺失不影响答案 |
| "这是什么花" | 花语是博爱 | supporting | 补充信息 |
| "这本书价格多少" | 识别为《三体》 | entity | 核心身份，错了→价格也错 |
| "这本书价格多少" | 价格59元 | answer | 直接答案 |
| "这本书价格多少" | 作者是刘慈欣 | supporting | 补充信息 |
| "这个药怎么吃" | 识别为阿莫西林 | entity | 核心身份，错了→用法也错 |
| "这个药怎么吃" | 每日3次每次1片 | answer | 直接答案 |
| "这个药怎么吃" | 避免与酒精同服 | supporting | 补充注意事项 |
| "为什么报错" | 第15行空指针异常 | answer | 直接答案（根因诊断） |
| "为什么报错" | 建议添加非空判断 | supporting | 补充建议 |
| "这张发票金额多少" | 金额328元 | answer | 直接答案（无实体识别环节） |
| "这张发票金额多少" | 开具日期2024年1月 | supporting | 补充信息 |
| "帮我算一下这道题" | 答案是42 | answer | 直接答案 |
| "帮我算一下这道题" | 解题步骤 | supporting | 过程展示 |

> **注意**：不是所有问题都有 entity 类型。无实体识别环节的问题（如计算、翻译）可以只有 answer 和 supporting。

### 2.2 延展（extension）—— 补充信息，有则加分

| 状态 | 判定规则 |
|------|---------|
| **未提及-满足** | 未提及该考点，或仅提及主体名词但未做实质性展开或错误解释 |
| **提及-满足** | 提及且内容符合考点要求（允许部分覆盖，按覆盖程度在 0.85-1.0 区间内取值） |
| **提及-不满足** | 提及但存在事实错误 |

> **排序原则：答错 < 不答 < 正确作答**
> 延展类鼓励作答：正确作答（按覆盖度得分）> 不答 > 有错误。不答不是最优选择，但好过答错。

### 2.3 陷阱（pitfall）—— 常见错误检验

| 状态 | 判定规则 |
|------|---------|
| **未命中** | 未出现陷阱提及的任何问题 |
| **命中** | 命中陷阱中**部分或全部**问题 |

### 2.4 判定约束

1. **不得创建上述之外的新满足类型**
2. **严禁引入外部知识**
3. **数字敏感**：数字/范围/时间要求与考点一致（受下述等价和容忍规则约束）
4. **区间子集原则（仅数值属性）**：当考点给出的范围表示**数值属性**（如"价格160-220元""重量30-60克"），答案区间 ⊆ 考点区间才算满足
5. **数量要求宽松原则**：当考点给出的范围表示**数量要求**（如"提供2-3个""列举2-3个"），答案数量≥下限即满足（多答不扣分，少于下限才不满足）
6. **边界词语义**：
   - "超过/多于/大于X"：答案须 > X
   - "至少/不少于/≥X"：答案须 ≥ X
   - "不到/少于/低于X"：答案须 < X
   - "最多/不超过/≤X"：答案须 ≤ X
   - "X以上"/"X以下"：含X本身
7. **等价表达认可**（视为数值一致）：
   - 单位换算：1kg=1000g，100cm=1m，1.5小时=90分钟
   - 格式等价：1999=1,999=一千九百九十九
   - 百分比等价：50%=0.5=五成
   - 序数等价：第1名=排名第一
   - 倍数等价：增长100%=翻倍=翻了一番；是去年的3倍=增长200%
   - 近似表达：答案含"约/大约/左右/近/接近"时，若考点精确值落在合理近似范围内视为满足（如考点1999元，答案约2000元→满足）
   - 精度/日期容忍：答案可用更低精度表述（年>月>日），已表述精度内须一致，不得导致数值实质性改变（如2022年6月6日→2022年6月✅，3.14→3.1✅，99.7%→100%❌，2022年6月→2022年7月❌）
8. **名称等价（conclusion 内的别称/同义表述）**：当 conclusion 中使用"或称/或/又称/也叫/即/又名"以及**括号内**列出多个名称时，这些名称**互为等价**，答案匹配其中**任何一个**即视为满足。判定时须逐一核对 conclusion 中列出的所有等价名称，**不得仅锚定第一个或最显眼的名称而忽略其余**。
9. **粒度容忍（层级部分匹配）**：当考点包含多层级信息（如品牌+型号+子型号、类目+品种+规格等），答案正确覆盖了上层信息但缺少更细粒度的下层信息时，视为"提及-满足"并按覆盖程度给分（不低于0.5）。但若答案给出了同等粒度的**错误**信息，则视为"提及-不满足"。判定逻辑：
   - 答案覆盖全部层级且正确 → 提及-满足，1.0
   - 答案覆盖上层正确但缺少下层细节 → 提及-满足，按覆盖层级比例给分（≥0.5）
   - 答案覆盖上层正确但下层信息**错误** → 提及-不满足
   - 答案上层信息就已错误 → 提及-不满足
10. **举例/枚举类规则**：当 conclusion 中以举例形式给出答案（含“如”“例如”“等”标志词）且附带数量要求时：
   - **内容开放性**：答案**不要求与示例完全一致**，只要逻辑自洽且解释合理即可满足
   - **数量基准**：覆盖比例以 conclusion 中**明确要求的数量**为基准（如“提供2-3个”则基准为2-3），而非 conclusion 中列举的所有示例数量。答案达到要求数量且正确即满足，不因未覆盖全部示例而扣分
11. **style_1 复述判定规则**：若考点 id 为 style_1，则按以下规则判定——若学生回答直接复现原题原文内容，视为 **不满足**，否则视为 **满足**。注意：即使输入区没有提供原题全文，也不能因此默认判为满足；只要学生回答自己写出了完整题干、整段设问、整张表格、整句选项或原题固定表述，就应按复述处理。以下算违规复述：单独设置“题目内容”“题干”“题目信息提取”“选项如下”等小节，集中抄写或近乎原句改写题干、设问、表格、誓词；直接把题目正文完整打一遍；完整列出多个选项；完整复现带条件、限定语的题目句子；或把较长且信息完整的原选项整句重新写出。像“题目：……”“题目信息提取：……”“选项：A…B…C…D…”这类先复述再分析，即使后面跟了解析，也算违规。以下不算违规：只用一句话概括题型或任务；为推理而提及少量必要条件、变量、公式；只引用选项字母或短关键词；对原选项做明显压缩、改写后的分析性转述。若只是概括题意或对选项含义做分析性改写，不按复述处理。

**综合示例**：

| 考点 | 答案 | 判定 | 规则 |
|------|------|------|------|
| 160-220元 | 160-185元 | ✅ | 区间子集 |
| 160-220元 | 160-230元 | ❌ | 上限超出 |
| 2-3个 | 2个 | ✅ | 范围取值 |
| 2-3个 | 4个 | ✅ | 多答不扣分，满足最低要求即可 |
| 2-3个 | 1个 | ❌ | 未达最低要求 |
| 超过100万 | 120万 | ✅ | >100 |
| 超过100万 | 100万 | ❌ | 不满足>100 |
| 至少3个 | 3个 | ✅ | ≥3 |
| 1kg | 1000g | ✅ | 单位等价 |
| 50% | 五成 | ✅ | 百分比等价 |
| 增长100% | 翻倍 | ✅ | 倍数等价 |
| 增长100% | 增长两倍 | ❌ | 两倍=200%≠100% |
| 1999元 | 约2000元 | ✅ | 近似表达 |
| 99.7% | 100% | ❌ | 精度容忍：数值实质性改变 |
| 生产日期2022年6月6日 | 2022年6月 | ✅ | 日期精度容忍 |
| 生产日期2022年6月6日 | 2022年7月 | ❌ | 月份不一致 |
| 列举相似公众人物至少2个（如金高银-单眼皮清秀等） | "赵露思-甜美邻家感，沈月-清秀圆脸"（不在示例中但合理） | ✅ | 举例性考点，合理且数量达标 |
| 同上 | "赵露思-甜美邻家感"（1个，合理但不足） | ✅ 但0.5 | 举例性考点，合理但未达"至少2个" |
| 同上 | "奥巴马-长得像"（无合理解释） | ❌ | 举例性考点，解释不合理 |
| 品牌尤尼克斯(YONEX)，型号NANOFLARE 700 | "尤尼克斯NANOFLARE 700" | ✅ 1.0 | 全部层级正确 |
| 品牌尤尼克斯(YONEX)，型号NANOFLARE 700 | "尤尼克斯NANOFLARE" | ✅ 0.8 | 粒度容忍：品牌+系列正确，缺少具体型号 |
| 品牌尤尼克斯(YONEX)，型号NANOFLARE 700 | "尤尼克斯" | ✅ 0.5 | 粒度容忍：仅品牌正确，缺少型号信息 |
| 品牌尤尼克斯(YONEX)，型号NANOFLARE 700 | "尤尼克斯NANOFLARE 800" | ❌ | 型号错误（700 vs 800），非粒度缺失而是事实错误 |
| 品牌尤尼克斯(YONEX)，型号NANOFLARE 700 | "李宁" | ❌ | 品牌错误，上层信息已错 |
| 用途和习性（如四季常青、耐修剪、庭院绿篱、盆景观赏、净化空气、驱蚊、入药等，提供2-3个） | "四季常青、耐修剪、可作盆景"（3个全对） | ✅ 1.0 | 枚举类数量基准：要求2-3个，答案3个达到上限且正确 |
| 同上 | "四季常青、耐修剪"（2个全对） | ✅ 0.8 | 枚举类数量基准：要求2-3个，答案2个达到下限且正确 |
| 同上 | "四季常青"（1个全对） | ✅ 0.5 | 枚举类数量基准：未达要求但有正确内容 |
| 同上 | "四季常青、可食用"（1对1错） | ❌ | 存在事实错误→提及-不满足 |

---

## 三、评分映射

### 3.1 主需（need）

#### 提及-满足 → 按覆盖程度（0.5 - 1.0）

| 覆盖程度 | score | 标准 |
|---------|-------|------|
| 完全覆盖 | 1.0 | 所有要点正确覆盖（枚举类：达到要求数量且全对） |
| 大部分覆盖 | 0.8 | 核心全覆盖，缺个别次要细节（枚举类：达到要求数量下限且全对） |
| 部分覆盖 | 0.5 | 覆盖部分要点，有缺失但无错误（枚举类：未达要求但≥1个全对） |

> **枚举类基准**：覆盖比例以 conclusion 中**要求的数量**为基准（非示例总数），详见判定约束第10条。0个正确→未提及-不满足。

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

> **注意**：need_type="entity" 且状态为"提及-不满足"时，错误程度固定为**严重**（×0.0），得分为 0.0。因为实体识别错误意味着整个回答基础崩塌。

> **单一数据考点**：当考点仅包含一个核心数据（如一个数值、一个范围），且该数据错误时，基础分档位为"几乎无"（0.0），因为不存在"其余正确"的部分。

**示例**：
- "售价1999元,身高135-160cm,价格稳定" vs "售价2999元,其余对" → 基础分档位=大部分对(0.5)，错误程度=中等(×0.25) → 0.5×0.25=0.1
- "该车JAVA安宝ANDA" vs "捷安特ATX"（entity类型）→ 错误程度固定=严重(×0.0) → 0.0
- "日常价160-220元" vs "160-230元" → 基础分档位=其余全对(0.8)，错误程度=中等(×0.25) → 0.8×0.25=0.2
- "牛奶重量30-60克" vs "100-120克"（考点仅含单一数据） → 基础分档位=几乎无(0.0)，错误程度=中等(×0.25) → 0.0×0.25=0.0

#### 未提及-不满足 → 0.0

---

### 3.2 延展（extension）

#### 核心排序：答错(≤0.75) < 不答(0.8) < 正确作答(0.85-1.0)，三档互不交叉。

#### 提及-满足 → 按覆盖程度（0.85 - 1.0）

| 覆盖程度 | score | 标准 |
|---------|-------|------|
| 完全覆盖 | 1.0 | 所有要点正确覆盖 |
| 大部分覆盖 | 0.9 | 核心全覆盖，缺个别次要细节 |
| 部分覆盖 | 0.85 | 覆盖部分要点，有缺失但无错误 |

> 实际可在 0.85-1.0 区间内灵活取值。枚举类基准同 need（详见判定约束第10条）。

#### 未提及-满足 → 0.8（固定）

#### 提及-不满足 → 按错误程度（0.0 - 0.75）

| 错误程度 | score | 标准 |
|---------|-------|------|
| 轻微 | 0.75 | 次要细节不一致，核心方向正确 |
| 中等 | 0.4 | 重要数据/属性错误 |
| 严重 | 0.0 | 核心内容完全错误或结论相反 |

---

### 3.3 陷阱（pitfall）

| 状态 | score |
|------|-------|
| 未命中 | 1.0 |
| 命中 | 0.0 |

---

### 3.4 评分区间总表

| kind | 状态 | score | need_type | 逻辑 |
|------|------|-------|-----------|------|
| need | 提及-满足 | **0.5 - 1.0** | entity/answer/supporting | 无错误，按覆盖程度 |
| need | 提及-不满足 | **0.0 - 0.4** | entity/answer/supporting | 有错误；entity类型固定0.0 |
| need | 未提及-不满足 | **0.0** | entity/answer/supporting | 缺失 |
| extension | 提及-满足 | **0.85 - 1.0** | — | 无错误，按覆盖程度连续取值 |
| extension | 未提及-满足 | **0.8** | — | 未涉及，固定基础分 |
| extension | 提及-不满足 | **0.0 - 0.75** | — | 有错误，按错误程度 |
| pitfall | 未命中 | **1.0** | — | — |
| pitfall | 命中 | **0.0** | — | — |

**设计原则**：
- need：有错上限(0.4) < 无错下限(0.5)，无交叉；entity答错固定0.0
- extension：答错上限(0.75) < 不答(0.8) < 正确作答下限(0.85)，三档无交叉
- need_type 仅适用于 need，extension 和 pitfall 不设 need_type

---

## 四、输出格式

### 4.1 推理链路与字段顺序

输出 JSON 时，每条考点的字段**严格按推理链路排序**——先思考再结论：

```
步骤1：读取考点信息    → kind, id, timeliness_required, enumeration_required, statement, conclusion
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
      "timeliness_required": "原始值",
      "enumeration_required": "原始值",
      "statement": "原始statement值",
      "conclusion": "原始conclusion值",
      "need_type": "entity/answer/supporting/null",
      "score_reason": "推理过程：need_type判定理由 → 答案与考点对比分析（逐一核对conclusion中所有等价名称）→ 满足状态判定 → 得分推导（提及-不满足时须显式写出基础分档位和惩罚系数档位的查表过程）",
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
- "提及-不满足"的得分推导中，必须**显式写出基础分档位→数值、惩罚系数档位→数值的完整查表过程**，不得跳步
- 不涉及名称等价的考点，无需强制写等价名称步骤

---

## 五、自检清单

- [ ] 每条考点已判定并评分，无遗漏；need 已判定 need_type 并在 score_reason 中说明理由
- [ ] entity + 提及-不满足 → score 固定 0.0；extension/pitfall 的 need_type 为 null
- [ ] 涉及名称等价时已逐一核对 conclusion 中所有等价名称，未遗漏
- [ ] 提及-不满足的得分推导已显式写出基础分档位和惩罚系数档位的查表过程
- [ ] 举例/枚举类：覆盖比例以**要求数量**为基准（非示例总数）；内容合理即可，不要求与示例完全一致
- [ ] 数值区间用区间子集原则，数量要求用≥下限原则（多答不扣分）
- [ ] 粒度容忍：上层正确缺下层→"提及-满足"(≥0.5)；同等粒度错误→"提及-不满足"
- [ ] 分数区间：need提及-满足∈[0.5,1.0]，提及-不满足≤0.4；extension答错(≤0.75)<不答(0.8)<正确(≥0.85)
- [ ] 未引入外部知识；JSON格式有效，字段顺序按推理链路排列

---

请严格按照上述规则，对学生回答进行逐条判定和评分，输出完整 JSON。
""".strip()


# ---------------------------------------------------------------------------
# Overall score computation (V4: need_type based weights)
# ---------------------------------------------------------------------------


def get_overall_score(checklist: list[dict[str, Any]]) -> float:
    """
    计算总分（加权平均）
    权重规则：
      - need（need_type 为 entity/answer）：核心主需，权重 1.0
      - need（need_type 为 supporting）：支撑主需，权重 0.15
      - pitfall：权重 0.3
      - extension：权重 0
    特殊规则：若所有核心主需（entity/answer）得分均为 0，则总分为 0.0
      - 优先检查 entity 类型，如果没有 entity 则检查 answer 类型
    """
    # 检查核心主需是否全为 0
    # 优先检查 entity 类型，如果没有 entity 则检查 answer 类型
    entity_items = [
        item for item in checklist if item.get('kind', '') == 'need' and item.get('need_type', '') == 'entity'
    ]
    if entity_items:
        # 有 entity 类型，检查 entity 是否全为 0
        if all(item.get('score', 0.0) == 0.0 for item in entity_items):
            return 0.0
    else:
        # 没有 entity 类型，检查 answer 类型是否全为 0
        answer_items = [
            item for item in checklist if item.get('kind', '') == 'need' and item.get('need_type', '') == 'answer'
        ]
        if answer_items and all(item.get('score', 0.0) == 0.0 for item in answer_items):
            return 0.0

    total_weighted_score = 0.0
    total_weight = 0.0
    style1_penalty = 1.0  # 是否命中 style_1，发生了重复图片内容的现象
    for item in checklist:
        kind = item.get('kind', '')
        need_type = item.get('need_type', '')
        item_id = item.get('id', '')
        if item_id == 'style_1':
            if item.get('score', 1.0) < 1.0:
                style1_penalty = 0.8
            continue

        if kind == 'need':
            # 核心主需 (entity/answer) 权重 1.0；其它 (supporting 等) 权重 0.15
            weight = 1.0 if need_type in ('entity', 'answer') else 0.15
        elif kind == 'pitfall':
            weight = 0.3
        elif kind == 'extension':
            weight = 0
        else:
            weight = 1.0  # 未知类型默认权重

        total_weighted_score += item.get('score', 0.0) * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return round(total_weighted_score * style1_penalty / total_weight, 4)


# ---------------------------------------------------------------------------
# Prompt builder (V4)
# ---------------------------------------------------------------------------


def build_scoring_prompt(
    question: str,
    messages: list[dict[str, Any]],
    checklist: list[dict[str, Any]],
    answer: str,
    date: str = '未知',
) -> str:
    """Build the scoring user prompt from the rubric checklist and model answer."""
    history = render_history(messages)
    checklist_text = json.dumps(checklist, ensure_ascii=False, indent=2)
    return SCORE_USER_PROMPT_TEMPLATE.format(
        question=question,
        date=date,
        history=history,
        checklist=checklist_text,
        answer=answer,
    )


# ---------------------------------------------------------------------------
# Main scoring function (public API)
# ---------------------------------------------------------------------------


async def shiwu_rubric_score(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Rubric-based scoring function for the Shiwu topic.

    Expects obj to contain:
      - "response": the model's answer text
      - "rubrics": rubric checklist list from the ground_truth envelope
      - "prompt": the user question text
      - "history": the conversation history list (optional)

    Sets on obj:
      - "reward": float score in [0, 1], or -1.0 on failure
      - "critic_resp": raw judge response for debugging
      - "model_output": same as critic_resp
      - "scoring_mode": "yb_rubric" or "yb_rubric_failed"
      - "original_request": reference to obj itself
    """
    return await yb_rubric_score_impl(
        obj=obj,
        build_scoring_prompt_fn=build_scoring_prompt,
        get_overall_score_fn=get_overall_score,
        system_prompt=SCORE_SYSTEM_PROMPT,
        version_tag='shiwu',
    )
