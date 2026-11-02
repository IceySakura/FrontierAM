"""
Rubric JSON → 结构化文本转换器。

对齐 baseline: General_badcase_RL/code/PE/rubric_formatter.py

将 labeling 阶段产出的 rubrics JSON 数组转为可读文本，供 Judge 模型输入使用。
"""

# 权重 → 可读标签
WEIGHT_LABELS: dict[int, str] = {
    3: '核心必要标准',
    2: '次要必要标准',
    1: '重要标准',
    -1: '陷阱标准',
}


def format_single_rubric(idx: int, rubric: dict) -> str:
    """将单条 rubric dict 格式化为可读文本块。

    Args:
        idx: 从 1 开始的序号
        rubric: 单条 rubric，包含 标题/描述/权重/满足条件/不满足条件

    Returns:
        格式化后的文本
    """
    title = rubric.get('标题', '未命名标准')
    desc = rubric.get('描述', '')
    weight = rubric.get('权重', 1)
    positive = rubric.get('满足条件', [])
    negative = rubric.get('不满足条件', [])

    weight_label = WEIGHT_LABELS.get(weight, f'权重{weight}')

    lines = [f'【评估标准 {idx}】{title}（权重: {weight} - {weight_label}）']
    lines.append(f'描述: {desc}')

    if positive:
        lines.append('✅ 满足条件:')
        for item in positive:
            lines.append(f'  - {item}')

    if negative:
        lines.append('❌ 不满足条件:')
        for item in negative:
            lines.append(f'  - {item}')

    return '\n'.join(lines)


def format_rubrics(rubrics: list[dict]) -> str:
    """将 rubrics JSON 数组转为完整的结构化文本。

    Args:
        rubrics: labeling 输出中的 rubrics 列表

    Returns:
        可直接嵌入 judge prompt 的结构化文本
    """
    if not rubrics:
        return '（无评估标准）'

    blocks = []
    for i, rubric in enumerate(rubrics, start=1):
        blocks.append(format_single_rubric(i, rubric))

    return '\n\n'.join(blocks)
