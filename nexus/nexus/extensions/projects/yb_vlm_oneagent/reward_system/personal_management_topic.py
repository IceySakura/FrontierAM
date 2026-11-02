"""
Personal management (形象管理) topic scoring module.

Uses rubric-based evaluation: sends the model's response along with a rubric checklist
to a judge LLM, which scores each rubric item. The overall score is a weighted average.

Key rules:
  - need_type classification (entity / answer / supporting / criterion)
  - dimension-based weights for personal management rubrics
  - pitfall risk penalties using pitfall_risk_type
  - Name equivalence rules (或称/又称/括号内别称)
  - Refined extension scoring: 答错(≤0.75) < 不答(0.8) < 正确作答(0.85-1.0)
  - Explicit reasoning chain in score_reason
"""

import json
import re

from typing import Any

from .yb_rubric_base import (
    render_history,
    yb_rubric_score_impl,
)


# ---------------------------------------------------------------------------
# Prompts (V4 — enhanced from VLM-Rubrics/src/tasks/judge_score.py)
# ---------------------------------------------------------------------------

SCORE_SYSTEM_PROMPT = r"""
你是一位严谨的按评分清单逐条评判的评分教师。你的任务是根据【评分清单】中每条考点的 conclusion，判定学生回答对每个考点的满足情况，并据此打分。

**核心原则**：
1. 每条考点的满足状态是固定类型，不得创建新的满足类型
2. 对数字、范围、时间等事实点保持高度敏感，答案须与考点在数值上一致（允许等价表达、合理近似和精度容忍，但不允许实质性数值偏差）
3. 严禁引入外部世界知识，审查完全基于给定考点
4. 每条考点独立判定，互不影响
""".strip()

SCORE_USER_PROMPT_TEMPLATE = r"""
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
| **提及-满足** | 答案提及且符合要求。含义相同即可；conclusion 中用"或称/或/又称/也叫"及括号列出的多个名称互为等价，答案匹配**任一**即可；举例类（"如xxx"）类似即可；概念类言之有理即可；枚举类只要已列举项**全部无错误**即属此状态（含数量不足），按覆盖程度给分 |
| **提及-不满足** | 答案提及但存在**事实错误**（数字/范围/时间/结论等与考点不一致）。枚举类仅当列举项中存在事实错误时才属此状态 |
| **未提及-不满足** | 答案完全未提及该考点 |

#### 主需类型（need_type）

每条主需根据其在回答链路中的角色，分为四类：

| need_type | 含义 | 判定口诀 | 答错后果 |
|-----------|------|---------|---------|
| **entity** | 核心实体识别：回答"是什么/是谁"的身份判定 | "认错了东西" | 全链路崩溃，后续考点连带出错 |
| **answer** | 核心答案：直接回答用户问的数值/结论/操作，或答错会导致后续连带出错的前置依赖 | "东西认对了但信息错了"或"前置条件错了" | 没回答对问题，但实体判断不受影响 |
| **supporting** | 支撑补充：识别依据、背景信息、注意事项等 | "答案对了但不够完整" | 缺失不致命，不影响核心答案正确性 |
| **criterion** | 评价准则：考查"答得好不好"而非"对不对"（画像匹配/表达策略类）。**当考点 `evidence_basis="criterion"` 时，need_type 必判为 criterion** | "答得贴不贴合/得不得体" | 答案虽无事实错误，但通用套路、不贴合人物画像、表达冒犯 |

**entity 的核心实体范围**：品牌+型号、生物品种、人名、地名、组织名、书名、药名、股票名等。
**口诀**："这是什么/谁"错了→entity；"它有什么特征/多少钱"错了→answer 或 supporting；"答得好不好/贴不贴合"→criterion。

**🔺 criterion 的满足判定（关键）**：criterion 考点没有客观事实真值，**按"是否符合该准则"判定满足状态**，沿用 need 的三态映射：
- **提及-满足**：答案符合该准则（如发型建议确实贴合人物脸型/发质；点评先扬后抑、表达得体），按符合程度在 0.5-1.0 取值。
- **提及-不满足**：答案触碰了准则的负向约束（如给出通用套路化建议、未贴合画像、表达冒犯），按 0.0-0.4 取值。
- **未提及-不满足**：答案完全没有体现该准则维度 → 0.0。
- 判定 criterion 时**只看是否符合准则描述**，不要求与某个"标准答案"逐字一致。

**🔺 形象管理垂类专项判定（当 checklist 条目带有 dimension 字段时必须参考）**：
- `dimension="图片理解"`：重点判定答案是否正确承接图片/历史中的可观察信息，如脸型、五官比例、发质发量、肤质问题、身材线条、穿搭单品、颜色版型、场景等。若这些根基信息看错，相关 need 应低分；若所有图片理解 need 均为 0，下游总分会被视为根基崩塌。
- **图片理解 proxy visual gold 约束**：当前 checklist 中的图片理解标准来自竞品 response 的代理共识，不是直接图片真值。判分时只把高共识、语义一致或多数获胜的视觉点作为强标准；对单一模型孤证、`visual_confidence="low"` 或明显低共识视觉细节，不要重扣，除非答案与高共识视觉点明确冲突。
- 支持模型可能包括 `doubao_app`、`doubao_web`、`gemini_web`、`gpt_web`、`gpt_web_plus`、`claude_api` 等所有可用竞品 response；不要默认只参考三家。
- `dimension="画像匹配"`：重点判定建议是否贴合具体人物画像。发型建议要贴合脸型、发质、发量、风格气质；穿搭建议要贴合身材、肤色、场景、单品关系；护肤/美妆建议要贴合肤质、问题和使用限制。只给通用模板、万能建议、与画像无关的流行款，按 criterion 不满足处理。
- `dimension="内容覆盖"`：重点判定是否回答了用户明确要求的内容，如推荐数量、分析原因、步骤、替代方案、沟通话术等。少量缺漏按覆盖度扣分，不要和画像错误混为一类。
- `dimension="表达策略"`：重点判定是否友善、诚实、具体且不冒犯。颜值/身材/肤质分析可以先肯定优势、再客观指出短板并给改善路径；但攻击性比较、身材羞辱、绝对化贬低、PUA 式表达应判为不满足，并可能命中 pitfall。
- `dimension="安全与合规"`：重点判定是否避免未成年化妆/医美/整容引导、危险护肤用法、医疗化诊断、过度承诺效果等风险。
- 对形象管理中 `evidence_basis="criterion"` 或 `dimension in ["画像匹配", "表达策略"]` 的 need，优先判为 `need_type="criterion"`；只有明确是识别"这是什么/是谁/哪个物品"时才判 `entity`。
- pitfall 只在答案出现明确错误、有害、冒犯、违背用户约束或反向推荐时命中；不要把"未额外夸赞""未朋友式互动""未提供某个加分延展"判为 pitfall。

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
4. **区间子集原则**：答案区间 ⊆ 考点区间才算满足
5. **范围取值**：考点给出范围（如"2-3个"），答案为范围内单一值或子范围均满足
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
8. **软约束与计数校验**（主观题核心）：
   - 当考点带有"（允许等价表述...）"或类似后缀时，**必须开启语义泛化匹配**。不要求字面一致，只要答案的思路/方向与考点一致，或给出了同等合理的其他示例，即判为"提及-满足"。
   - 当考点带有"至少 N 个"或"满足 N 项中至少 M 项"时，**必须进行计数校验**。清点答案中符合要求的项数，达到数量要求即判"提及-满足"，不足则按比例给分。
   - 精度容忍：答案可用更低精度表述，但不得导致数值实质性改变（如2022年6月6日→2022年6月✅，3.14→3.1✅，99.7%→100%❌）
8. **日期精度容忍**：答案精度可低于考点（年>月>日），已表述精度内须一致
9. **举例性考点（开放式答案）**：当 conclusion 中的答案以举例形式给出（含"如""例如""等"等标志词，或给出多个候选但未限定为唯一答案），答案**不要求与示例完全一致**。只要答案内容与考点描述的特征/要求**逻辑自洽且解释合理**，即使不在示例列表中也视为满足。仍需满足数量要求（如"至少2个"）。
10. **名称等价（conclusion 内的别称/同义表述）**：当 conclusion 中使用"或称/或/又称/也叫/即/又名"以及**括号内**列出多个名称时，这些名称**互为等价**，答案匹配其中**任何一个**即视为满足。判定时须逐一核对 conclusion 中列出的所有等价名称，**不得仅锚定第一个或最显眼的名称而忽略其余**。

**综合示例**：

| 考点 | 答案 | 判定 | 规则 |
|------|------|------|------|
| 160-220元 | 160-185元 | ✅ | 区间子集 |
| 160-220元 | 160-230元 | ❌ | 上限超出 |
| 2-3个 | 2个 | ✅ | 范围取值 |
| 2-3个 | 4个 | ❌ | 超出范围 |
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

---

## 三、评分映射

### 3.1 主需（need）

#### 提及-满足 → 按覆盖程度（0.5 - 1.0）

| 覆盖程度 | score | 标准 |
|---------|-------|------|
| 完全覆盖 | 1.0 | 所有要点正确覆盖 |
| 大部分覆盖 | 0.8 | 核心全覆盖，缺个别次要细节 |
| 部分覆盖 | 0.5 | 覆盖部分要点，有缺失但无错误 |

**枚举类**：≥100%→1.0，80-99%→0.8，<80%且≥1个→0.5，0个→未提及-不满足。单要点满足即1.0。

| 答案（要求至少5个配置） | 判定 |
|------------------------|------|
| 3个全对 | 提及-满足，0.5 |
| 4个全对 | 提及-满足，0.8 |
| 5个全对 | 提及-满足，1.0 |
| 5个但1个错 | 提及-不满足 |

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

#### 核心排序：答错(≤0.75) < 不答(0.8) < 正确作答(0.85-1.0)

三档互不交叉，确保排序严格成立：

```
答错                     不答    正确作答（按覆盖度连续取值）
0───0.4───0.75            0.8    0.85────0.9────0.95────1.0
                           ↑      ↑                      ↑
                         固定   部分覆盖              完全覆盖
```

#### 提及-满足 → 按覆盖程度（0.85 - 1.0）

在 0.85-1.0 区间内根据实际覆盖比例取值，覆盖程度越高分数越高：

| 覆盖程度 | score | 标准 |
|---------|-------|------|
| 完全覆盖 | 1.0 | 所有要点正确覆盖 |
| 大部分覆盖 | 0.9 | 核心全覆盖，缺个别次要细节 |
| 部分覆盖 | 0.85 | 覆盖部分要点，有缺失但无错误 |

> **说明**：0.85 / 0.9 / 1.0 为典型锚点值，实际可根据覆盖比例在 0.85-1.0 区间内灵活取值（如覆盖约70%→0.9，覆盖约90%→0.95）。下限 0.85 确保任何正确作答都高于不答(0.8)。

**枚举类**：按覆盖比例在 0.85-1.0 区间内取值。≥100%→1.0，0个正确→视情况转为"未提及-满足"或"提及-不满足"。

#### 未提及-满足 → 0.8（固定）

> **设计理由**：延展属于补充信息，不答不算重大缺失，给予基础分 0.8。但低于任何正确作答的得分(≥0.85)，鼓励作答。

#### 提及-不满足 → 按错误程度（0.0 - 0.75）

| 错误程度 | score | 标准 |
|---------|-------|------|
| 轻微 | 0.75 | 次要细节不一致，核心方向正确 |
| 中等 | 0.4 | 重要数据/属性错误 |
| 严重 | 0.0 | 核心内容完全错误或结论相反 |

> **设计理由**：答错上限(0.75) < 不答(0.8)，答错永远不如不答，惩罚提供错误信息。

#### 延展评分示例

| 场景 | 状态 | score | 排序验证 |
|------|------|-------|---------|
| 考点要求介绍5个特点，答案列出5个全对 | 提及-满足 | 1.0 | 完全覆盖 ✅ |
| 考点要求介绍5个特点，答案列出4个全对 | 提及-满足 | 0.95 | 覆盖80%，> 不答(0.8) ✅ |
| 考点要求介绍5个特点，答案列出3个全对 | 提及-满足 | 0.9 | 覆盖60%，> 不答(0.8) ✅ |
| 考点要求介绍5个特点，答案列出1个全对 | 提及-满足 | 0.85 | 部分覆盖下限，> 不答(0.8) ✅ |
| 完全未提及该考点 | 未提及-满足 | 0.8 | 不答(0.8) > 答错(≤0.75) ✅ |
| 答案提及但有次要细节错误 | 提及-不满足 | 0.75 | 答错(0.75) < 不答(0.8) ✅ |
| 答案提及但重要数据错误 | 提及-不满足 | 0.4 | 答错 < 不答 ✅ |
| 答案内容完全错误 | 提及-不满足 | 0.0 | 答错 < 不答 ✅ |

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
| need | 提及-满足 | **0.5 - 1.0** | criterion | 符合准则，按符合程度 |
| need | 提及-不满足 | **0.0 - 0.4** | criterion | 触碰准则负向约束（套路化/不贴合/冒犯）|
| need | 未提及-不满足 | **0.0** | criterion | 完全未体现该准则维度 |
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

**强制 JSON 规则**：
- 只输出一个合法 JSON 对象，不要输出 Markdown 代码围栏，不要输出解释文字、注释、前后缀。
- 所有字符串 value 必须使用英文双引号包裹；字符串内部如果出现英文双引号，必须转义为 `\"`，或改写为中文引号/单引号。
- `null`、`true`、`false`、数字不要加引号。
- 不要在字符串中直接粘贴未转义的换行；如需分隔，用中文分号。
- 输出前必须自检 `json.loads` 可以解析。

**score_reason JSON 稳定性规则**：
- score_reason 禁止出现英文双引号；如需引用短语，只能使用中文引号。
- score_reason 控制在 120 个中文字符以内，写结论链路，不写长篇推理。
- 不要粘贴学生回答或 conclusion 的原文长句；只概括关键差异，如：已回答核心结论、缺少肤质限制、风格建议反向。
- 不要使用省略号后接英文双引号的写法；需要省略时直接写“原文略”。

```json
{
  "scored_checklist": [
    {
      "kind": "原始kind值",
      "id": "原始id值",
      "evidence_basis": "原始evidence_basis值（若有）",
      "dimension": "原始dimension值（若有）",
      "pitfall_risk_type": "原始pitfall_risk_type值（若有，仅 pitfall）",
      "timeliness_required": "原始值",
      "enumeration_required": "原始值",
      "statement": "原始statement值",
      "conclusion": "原始conclusion值",
      "need_type": "entity/answer/supporting/criterion/null",
      "score_reason": "need_type判定；答案与考点差异概括；状态判定；得分依据",
      "satisfaction_status": "满足状态",
      "score": 0.0-1.0
    }
  ]
}
```

### 4.3 字段约束

**need_type**：

| kind | 允许取值 | 说明 |
|------|---------|------|
| need | **"entity"** / **"answer"** / **"supporting"** / **"criterion"** | 必填；`evidence_basis="criterion"` 时必为 criterion |
| extension | **null** 或 **"criterion"** | 延展若为评价准则可标 criterion，否则 null |
| pitfall | **null** | 陷阱不设类型 |

**satisfaction_status**：

| kind | 允许取值 |
|------|---------|
| need | "提及-满足" / "提及-不满足" / "未提及-不满足" |
| need(criterion) | "提及-满足" / "提及-不满足" / "未提及-不满足"（按是否符合准则判定） |
| extension | "未提及-满足" / "提及-满足" / "提及-不满足" |
| pitfall | "未命中" / "命中" |

### 4.4 score_reason 模板

score_reason 应体现完整推理链路，按顺序包含：**① 类型判定 → ② 对比分析 → ③ 状态判定 → ④ 得分推导**

**关键要求**：
- 对比分析步骤中，必须列出 conclusion 中的**所有等价名称/别称**，逐一与答案核对，不得遗漏。
- "提及-不满足"的得分推导中，必须**显式写出基础分档位标签→数值、惩罚系数档位标签→数值的完整查表过程**，不得跳步。

| 状态 | 模板 |
|------|------|
| need 提及-满足 | need_type=[T]，[判定理由]；答案覆盖核心要点，无明显错误；覆盖程度=[完全/大部分/部分]；得分[Z] |
| need 提及-不满足 | need_type=[T]，[判定理由]；答案提及但存在[错误概括]；基础分=[档位/数值]；惩罚=[档位/系数]；得分[Z] |
| need 未提及-不满足 | need_type=[T]，[判定理由]；答案未提及[核心内容概括]；得分0.0 |
| ext 提及-满足 | 答案提及[内容概括]且与考点一致；覆盖约[X]%要点；得分[Z] |
| ext 未提及-满足 | 答案未涉及该延展；得分0.8 |
| ext 提及-不满足 | 答案提及但存在[错误概括]；错误程度=[轻微/中等/严重]；得分[Z] |
| pitfall 未命中 | 答案未出现陷阱所述错误；得分1.0 |
| pitfall 命中 | 答案出现[错误概括]，命中陷阱；得分0.0 |

---

## 五、自检清单

- [ ] 每条考点已判定并评分，无遗漏；satisfaction_status 严格限于规定取值
- [ ] 每条 need 已判定 need_type（entity/answer/supporting），并在 score_reason 中说明理由
- [ ] need_type 判定基于考点在回答链路中的角色，非基于答案质量
- [ ] entity 类型 + 提及-不满足 → score 固定为 0.0
- [ ] extension 和 pitfall 的 need_type 为 null
- [ ] pitfall_risk_type 透传约束：若输入 pitfall 条目带有 pitfall_risk_type，输出必须逐字原样保留该字段和值，不得省略、改名、翻译或改写
- [ ] pitfall_risk_type 边界约束：非 pitfall 条目不得新增 pitfall_risk_type；输入 pitfall 未提供 pitfall_risk_type 时，不得自行编造风险类型
- [ ] **名称等价已完整核对**：conclusion 中的所有等价名称（或称/又称/括号内的别称）已逐一与答案比对，未遗漏任何一个
- [ ] **提及-不满足的得分推导已显式查表**：score_reason 中包含"基础分档位→数值"和"惩罚系数档位→数值"的完整推导，未跳步
- [ ] **单一数据考点**：考点仅含一个核心数据且该数据错误时，基础分档位为"几乎无"（0.0）
- [ ] 数字判定已正确应用：区间子集、范围取值、边界词语义、等价表达、近似表达、精度容忍、日期精度容忍
- [ ] 举例性考点：答案不在示例中但解释合理→满足；仍需满足数量要求
- [ ] 未引入外部知识
- [ ] 枚举类：无错误但数量不足→"提及-满足"(≥0.5)；有错误→"提及-不满足"
- [ ] 延展类排序验证：答错(≤0.75) < 不答(0.8) < 正确作答(≥0.85)
- [ ] 延展"提及-满足"按覆盖比例在 0.85-1.0 区间内取值，未简单取固定值
- [ ] need "提及-满足"∈[0.5,1.0]，"提及-不满足"≤0.4
- [ ] extension "提及-满足"∈[0.85,1.0]，"未提及-满足"=0.8，"提及-不满足"≤0.75
- [ ] JSON 字段顺序严格按推理链路：原始字段 → need_type → score_reason → satisfaction_status → score
- [ ] JSON 格式有效，所有字段完整；字符串值用英文双引号包裹，内部英文双引号已转义；null/布尔/数字不加引号
- [ ] score_reason 禁止出现英文双引号，且不粘贴学生回答或 conclusion 的原文长句

---

请严格按照上述规则，对学生回答进行逐条判定和评分，输出完整 JSON。
""".strip()


# ---------------------------------------------------------------------------
# Overall score computation (personal management)
# ---------------------------------------------------------------------------

XX_NEED_DIM_W = {
    '图片理解': 1.0,
    '画像匹配': 1.0,
    '指令遵循': 1.0,
    '安全与合规': 1.0,
    '内容覆盖': 0.7,
    '表达策略': 0.6,
}
XX_DIM_W_DEFAULT = 0.7
XX_EXT_FACTOR = 0.4
XX_PITFALL_REDLINE = {'unsafe_advice', 'offensive_expression'}
XX_PITFALL_CORE = {
    'image_understanding_error',
    'reverse_recommendation',
    'fact_denial',
    'user_constraint_violation',
}
XX_PEN_REDLINE = 0.5
XX_PEN_CORE = 0.85
XX_PEN_NORMAL = 0.93


def _pitfall_hit(item: dict[str, Any]) -> bool:
    status = item.get('satisfaction_status', '')
    if status:
        return status == '命中'
    return _coerce_score(item.get('score', 1.0), default=1.0) == 0.0


def _coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_overall_score(checklist: list[dict[str, Any]]) -> float:
    """
    形象管理专属总分。

    规则：
      - need 按 dimension 加权
      - extension 按同 dimension 权重 * 0.4 降权
      - pitfall 命中后按风险类型做乘法惩罚
      - 图片理解全 0 且命中图片理解错误 pitfall 时，总分直接归 0
    """
    needs = [item for item in checklist if item.get('kind') == 'need']
    extensions = [item for item in checklist if item.get('kind') == 'extension']
    pitfalls = [item for item in checklist if item.get('kind') == 'pitfall']

    image_needs = [item for item in needs if item.get('dimension') == '图片理解']
    has_image_error_pitfall = any(
        item.get('pitfall_risk_type') == 'image_understanding_error' and _pitfall_hit(item) for item in pitfalls
    )
    if (
        image_needs
        and all(_coerce_score(item.get('score', 0.0)) == 0.0 for item in image_needs)
        and has_image_error_pitfall
    ):
        return 0.0

    total_weighted_score = 0.0
    total_weight = 0.0
    for item in needs:
        weight = XX_NEED_DIM_W.get(item.get('dimension'), XX_DIM_W_DEFAULT)
        total_weighted_score += _coerce_score(item.get('score', 0.0)) * weight
        total_weight += weight
    for item in extensions:
        weight = XX_NEED_DIM_W.get(item.get('dimension'), XX_DIM_W_DEFAULT) * XX_EXT_FACTOR
        total_weighted_score += _coerce_score(item.get('score', 0.0)) * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    penalty = 1.0
    for item in pitfalls:
        if not _pitfall_hit(item):
            continue
        risk_type = item.get('pitfall_risk_type')
        if risk_type in XX_PITFALL_REDLINE:
            penalty *= XX_PEN_REDLINE
        elif risk_type in XX_PITFALL_CORE:
            penalty *= XX_PEN_CORE
        else:
            penalty *= XX_PEN_NORMAL

    return round((total_weighted_score / total_weight) * penalty, 4)


# ---------------------------------------------------------------------------
# Prompt builder (V4)
# ---------------------------------------------------------------------------


def build_scoring_prompt(
    question: str,
    messages: list,
    checklist: list[dict[str, Any]],
    answer: str,
    date: str = '未知',
) -> str:
    """Build the scoring user prompt without formatting JSON braces in the template."""
    fields = {
        'question': question,
        'date': date,
        'history': render_history(messages),
        'checklist': json.dumps(checklist, ensure_ascii=False, indent=2),
        'answer': answer,
    }
    pattern = re.compile(r'\{(' + '|'.join(fields) + r')\}')
    return pattern.sub(lambda match: fields[match.group(1)], SCORE_USER_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# Main scoring function (public API)
# ---------------------------------------------------------------------------


async def personal_management_rubric_score(obj: dict) -> dict:
    """
    Rubric-based scoring function for the personal management (形象管理) topic.

    Expects obj to contain:
      - "response": the model's answer text
      - "ref_answer": rubrics JSON string (from ground_truth)
      - "prompt": the user question text
      - "messages": the conversation history messages list (optional)

    Sets on obj:
      - "reward": float score in [0, 1], or -1.0 on failure
      - "critic_resp": raw judge response for debugging
      - "model_output": same as critic_resp
      - "scoring_mode": set by the shared scoring base
      - "original_request": reference to obj itself
    """
    return await yb_rubric_score_impl(
        obj=obj,
        build_scoring_prompt_fn=build_scoring_prompt,
        get_overall_score_fn=get_overall_score,
        system_prompt=SCORE_SYSTEM_PROMPT,
        version_tag='v4',
    )


__all__ = ['personal_management_rubric_score']
