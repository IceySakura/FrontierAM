"""
AGTS 奖励计算模块。

提供多种奖励评分方式：
- 基于规则的格式检查
- LLM Judge 语义评分
- 格式协议检查（LLM）
- Pairwise RM 评分
- GRM (Generalized Reward Model) 评分
- Rubric-based 评分（badcase rubric judge）
- 总奖励计算（整合多个评分）
"""

from .closed_qa_judge import compute_score as closed_qa_judge_score
from .closed_qa_rubric_judge import compute_score as closed_qa_rubric_judge_score
from .format_judge import compute_score as format_judge_score
from .grm_judge import compute_score as grm_judge_score
from .hy_reward import compute_score as hy_reward_score
from .pairwise_judge import compute_score as pairwise_judge_score
from .pairwise_rubrics import compute_score as pairwise_rubrics_score
from .rubric_formatter import format_rubrics
from .rubric_judge import compute_score as rubric_judge_score
from .rule_based_judge import compute_score as rule_based_score
from .total_reward import compute_score, compute_score_async


__all__ = [
    'closed_qa_judge_score',
    'closed_qa_rubric_judge_score',
    'compute_score',
    'compute_score_async',
    'format_judge_score',
    'grm_judge_score',
    'hy_reward_score',
    'pairwise_judge_score',
    'pairwise_rubrics_score',
    'rubric_judge_score',
    'format_rubrics',
    'rule_based_score',
]
