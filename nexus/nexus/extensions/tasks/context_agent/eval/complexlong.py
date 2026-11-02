"""ComplexLong evaluator — one LLM judge for the four structured long-context sets.

``ComplexLongTable`` / ``Graph`` / ``Tree`` / ``Doc`` all have a determinate
ground truth, so grading collapses to a single question: is the model's final
conclusion equivalent to the gold answer? They deliberately do NOT reuse the
per-family rule-based scorers the source repos ship (table's ``score.py``,
graph's ``graph_bench_eval.py``), because ``tree`` questions carry no output
format contract — unlike table ("answer on the last line") and graph
(``Answer: [...]``) — so regex extraction would produce false negatives there.

The one thing that varies per family is a single sentence in the prompt: the
*comparison rule*. Whether the gold answer is a scalar, an order-free set, an
ordered sequence or an integer decides whether the same model answer is right or
wrong, so collapsing them into one rule would skew the verdict. The rule texts
follow the convention each source repo already validated: table's three
``answer_form`` values, graph's ``SCHEMA`` dispatch table, tree's eight
``answer_type`` values, and doc's open-ended semantic equivalence.

Ported from ``hy_unify_eval`` ``T1_eval/ComplexLong/judge.py``; the prompt, the
rule table and the score parsing are kept byte-identical so numbers stay
comparable with that harness.
"""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

MAX_RETRIES = 10

# Per-field cap on what reaches the judge. The longest gold answer measured is
# ~13k characters (a long tree sequence), so this only ever truncates an
# oversized model answer, keeping the request under the judge's input limit.
MAX_FIELD_CHARS = 24000

DATASET_NAMES = (
    'ComplexLongTable',
    'ComplexLongGraph',
    'ComplexLongTree',
    'ComplexLongDoc',
)

# --------------------------------------------------------------- comparison rules
RULE_SCALAR = (
    '标准答案是一个整体（单个数值、单个 id、一个名称或一段原文），必须整段核对，'
    '不要按分隔符把它拆成多项来比较——标准答案里出现的顿号、逗号可能是原文的一'
    '部分，不代表它是一个多元素集合。'
)
RULE_SET = (
    '标准答案是一个集合，顺序无关：元素完全相同就判 1 分，学生的排列顺序与标准答案'
    '不同不扣分。但元素必须逐个核对，缺少、多出或写错任何一个元素都判 0 分——'
    '「大部分元素都对」不是 1 分。元素很多时请耐心逐个比对，不要目测个数就下结论。'
)
RULE_ORDERED = (
    '标准答案是有序序列，顺序本身就是考点：元素相同但顺序不同判 0 分。必须逐位核对，'
    '任何一位缺失、多出、错位或写错都判 0 分——「大部分位置都对」不是 1 分。'
    '序列很长时请耐心逐位比对，不要目测长度或只看首尾就下结论。'
    '如果序列的元素本身又是一个数组（嵌套的二维结构），那么外层和内层都要逐位核对，'
    '内层的顺序同样是考点。'
)
RULE_INT = (
    '标准答案是一个整数，数值必须完全相等才判 1 分，差 1 也判 0 分，不做四舍五入、'
    '不接受近似值和区间。允许千分位和会计记法的写法差异（-118,522 与 -118522 等价，'
    '(123) 与 -123 等价）。'
)
RULE_ADJ = (
    '标准答案是一个邻接表（键为节点、值为邻居列表）。每个键都必须存在，且每个键'
    '对应的邻居集合必须完全一致才判 1 分；邻居列表内部的排列顺序不同不扣分，'
    '但缺少、多出或写错任何一个键或任何一个邻居都判 0 分。'
)
RULE_PATH = (
    '标准答案由一条有序路径和一个总成本组成，两者都必须正确才判 1 分。路径顺序敏感，'
    '逐位核对，任何一跳缺失、多出或错位都判 0 分；总成本必须数值完全相等。'
    '只答对路径没答对成本，或只答对成本没答对路径，都判 0 分。'
)
RULE_STRUCT = (
    '标准答案是一个结构化对象（若干个键值对）。每个键都必须出现，且每个键的值都必须'
    '与标准答案一致，才判 1 分；漏答、多答或答错任何一个键的值都判 0 分。'
    '学生用 JSON 还是自然语言表述这些键值不影响评分。'
)
RULE_OPEN = (
    '标准答案是一段结论性文本。核对学生答案的最终结论是否与它表达同一事实，'
    '允许措辞、语言和详略不同；标准答案中的每个关键点都必须被覆盖，漏答或答错'
    '任一关键点判 0 分。'
)

# graph: mirrors the SCHEMA dispatch table in ``graph_bench_eval.py``.
# ``*`` means every mode of that task is graded the same way.
_GRAPH_RULE = {
    ('G-L1', '*'): RULE_SET,  # id set
    ('G-L2', '*'): RULE_SET,  # prefixed id set
    ('G-A1', '*'): RULE_INT,
    ('G-A2', 'MAXCNT'): RULE_INT,
    ('G-A2', '*'): RULE_SET,
    ('G-C1', '*'): RULE_SET,  # triple set
    ('G-C2', 'ADJ'): RULE_ADJ,
    ('G-C2', 'EDGES'): RULE_SET,  # pair set
    ('G-C2', 'COUNT'): RULE_INT,
    ('G-C2', '*'): RULE_SET,
    ('G-R1', '*'): RULE_SET,
    ('G-R2', 'COST'): RULE_INT,
    ('G-R2', '*'): RULE_PATH,  # path AND cost
}

# tree: the eight ``extra_info.answer_type`` values.
_TREE_RULE = {
    'number': RULE_INT,
    'node': RULE_SCALAR,
    'text': RULE_SCALAR,
    'structure_shape': RULE_STRUCT,
    'node_sequence': RULE_ORDERED,
    'sequence': RULE_ORDERED,
    'text_sequence': RULE_ORDERED,
    'value_sequence': RULE_ORDERED,
}

# table: the three ``answer_form`` values its README requires to be kept apart.
_TABLE_RULE = {
    '标量': RULE_SCALAR,
    '集合': RULE_SET,
    '有序列表': RULE_ORDERED,
}

# The two G-R2 modes whose question asks for the path AND the total cost.
_GRAPH_NEED_COST = ('G-R2/BOTH', 'G-R2/HOPCAP')


def compare_rule(dataset_name: str | None, answer_form: object) -> str:
    """Pick the comparison rule for a dataset / answer shape."""
    answer_form = str(answer_form or '')

    if dataset_name == 'ComplexLongTable':
        rule = _TABLE_RULE.get(answer_form)
    elif dataset_name == 'ComplexLongTree':
        rule = _TREE_RULE.get(answer_form)
    elif dataset_name == 'ComplexLongGraph':
        task, _, mode = answer_form.partition('/')
        rule = _GRAPH_RULE.get((task, mode)) or _GRAPH_RULE.get((task, '*'))
    elif dataset_name == 'ComplexLongDoc':
        rule = RULE_OPEN
    else:
        rule = None

    if rule is None:
        logger.warning(
            'ComplexLong: unrecognised (dataset_name=%r, answer_form=%r); falling back to the open-ended rule',
            dataset_name,
            answer_form,
        )
        return RULE_OPEN
    return rule


# --------------------------------------------------------------------- prompt
PROMPT_TEMPLATE = """你是一名严格、客观、稳定的判卷老师。你的任务是判断「学生答案」的最终结论是否与「标准答案」一致。

【评分规则】
1. 只评 0 分或 1 分。1 分表示最终结论正确，0 分表示不正确。
2. 学生答案通常会先复述题目、再展开推理、最后作答。请忽略复述和推理过程，忽略排版、语言和输出格式，只核对学生给出的最终结论。学生没有按题目要求的格式输出、用了另一种语言，都不影响评分。
3. 本题的比对规则：{compare_rule}
4. 只允许写法层面的等价改写：1/2 与 0.5、-118,522 与 -118522、(123) 与 -123、加引号与不加引号、JSON 与自然语言列举同一批元素。不允许同义词替换、不做近似匹配、不做四舍五入。
5. 最终结论缺失、答错、答成另一个问题、只复述了题目却没有作答、或答案为空，一律 0 分。
6. 不要因为学生的解释更长、更细、包含了部分正确的中间步骤，或者包含了标准答案的片段，就给 1 分；最终结论不对就是 0 分。
7. 标准答案为空集时（`[]`，或明确表示「没有符合条件的结果」），学生也明确回答空集才给 1 分；学生没有作答，或答出了任何元素，都是 0 分。
8. 学生答案里若出现多个互相矛盾的候选结论，以最后给出的那个为准。
9. 拿不准就给 0 分。宁可漏判一道答对的，也不要放过一道答错的。

【输出格式，必须严格遵守】
【评分依据】：用一两句话说明最终结论是否与标准答案等价。
【评分】：0分 或 1分
{question_block}
<标准答案>：
{ground_truth}

<学生答案>：
{student_answer}
"""


def _clip(text: object) -> str:
    """Truncate an oversized field so the judge request stays within its limit."""
    text = '' if text is None else str(text)
    if len(text) <= MAX_FIELD_CHARS:
        return text
    return text[:MAX_FIELD_CHARS] + f'\n……（学生答案过长，此处已截断，原文共 {len(text)} 字符）'


def strip_think(response: str) -> str:
    """Drop chain-of-thought, keeping only what follows ``</think>``."""
    if '</think>' in response:
        match = re.search(r'</think>(.*)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response


def ground_truth_text(task: dict) -> str:
    """Render the gold answer sent to the judge.

    The ``G-R2/BOTH`` and ``G-R2/HOPCAP`` questions ask for a path *and* a total
    cost, but ``answer`` holds only the path; the cost lives in
    ``metadata.orig.cost``. Without it the judge would grade half the question.
    """
    answer = task.get('answer', '')
    meta = task.get('metadata') or {}

    if meta.get('answer_form') in _GRAPH_NEED_COST:
        cost = (meta.get('orig') or {}).get('cost')
        if cost is not None:
            return f'路径（顺序敏感）：{answer}\n总成本 cost：{cost}'
        logger.warning(
            'ComplexLong: %s is %s but carries no orig.cost; grading the path only',
            meta.get('task_id'),
            meta.get('answer_form'),
        )
    return str(answer)


def build_prompt(model_output: str, task: dict) -> str:
    """Assemble the judge prompt, including the question when the task has one."""
    meta = task.get('metadata') or {}
    question = task.get('context_question') or ''
    question_block = f'\n<题目>：\n{_clip(question)}\n' if str(question).strip() else ''
    return PROMPT_TEMPLATE.format(
        compare_rule=compare_rule(meta.get('dataset_name'), meta.get('answer_form')),
        question_block=question_block,
        ground_truth=_clip(ground_truth_text(task)),
        student_answer=_clip(model_output),
    )


# Try strict ("【评分】：1分") first, then looser variants, taking the LAST match
# so a trailing score line wins over any score mentioned mid-reasoning.
_SCORE_PATTERNS = (
    re.compile(r'【评分】\s*[：:]\s*([01])\s*分'),
    re.compile(r'【评分】\s*[：:]?\s*([01])'),
    re.compile(r'([01])\s*分\s*$'),
)


def parse_score(text: str) -> int | None:
    """Extract the binary score from a judge reply, or ``None`` if absent."""
    if not text:
        return None
    for pattern in _SCORE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return int(matches[-1])
    return None


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Grade *model_output* for a ComplexLong task with the shared judge."""
    if not model_output or not str(model_output).strip():
        return {
            'score': 0,
            'rationale': 'No model output',
            'requirement_status': [],
            'judge_prompt': [],
        }

    prompt = build_prompt(strip_think(str(model_output)), task)
    messages = [{'role': 'user', 'content': prompt}]
    task_id = (task.get('metadata') or {}).get('task_id', '')

    for attempt in range(MAX_RETRIES):
        try:
            raw = await llm.chat(messages)
            if isinstance(raw, dict):
                raw = raw.get('response', '')
            score = parse_score(raw or '')
            if score is None:
                raise ValueError('no 【评分】 score line found in judge output')
            return {
                'score': score,
                'rationale': (raw or '').strip(),
                'requirement_status': [],
                'judge_prompt': messages,
            }
        except ValueError as exc:
            logger.warning(
                'ComplexLong judge parse failed (task_id=%s, attempt %s/%s): %s',
                task_id,
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                'ComplexLong judge call failed (task_id=%s, attempt %s/%s): %s',
                task_id,
                attempt + 1,
                MAX_RETRIES,
                exc,
            )

    return {
        'score': 0,
        'rationale': f'Judge failed after {MAX_RETRIES} attempts',
        'requirement_status': [],
        'judge_prompt': messages,
    }
