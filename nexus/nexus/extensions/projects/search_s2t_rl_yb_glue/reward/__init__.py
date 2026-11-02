"""S2T yuanbao-glue 奖励计算模块。

提供基于 examples/yuanbao_search（agts）、面向必须搜索数据的奖励评分：
- 基于规则的格式检查（planner 门控）
- 有效搜索门控（没有包含非空 query 的搜索 → 0.0，并跳过 LLM Judge）
- 四个硬门控（format / safety / length / natural，复用 audio_grm 门控）：任一判负 → 0.0
- LLM Judge 语义评分（完全匹配 / 部分匹配 / 不匹配 → 1.0 / 0.5 / 0.0）
- 总奖励计算（format_failed ⇒ -1.0；否则搜索后三档语义分减去轻量 query 成本，下界为 0.0）
"""

from .gate_judge import evaluate_gates as gate_judge_evaluate
from .llm_judge import compute_score as llm_judge_score
from .rule_based_judge import compute_score as rule_based_score
from .total_reward import compute_score, compute_score_async


__all__ = [
    'compute_score',
    'compute_score_async',
    'gate_judge_evaluate',
    'llm_judge_score',
    'rule_based_score',
]
