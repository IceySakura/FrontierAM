"""Deterministic GSM8K scoring used by the unified reward-system task.

Migrated from hytuner/third_party/open_verl/verl/utils/reward_score/gsm8k.py

Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning."
Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 2024.
"""

import re


# Optimization: Only search in the last N characters for the answer
_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str, method: str = 'strict') -> str | None:
    """Extract the final answer from a solution string.

    Args:
        solution_str: The full solution text from the model
        method: Extraction method
            - 'strict': Look for "#### <number>" format (tests model formatting)
            - 'flexible': Extract the last number in the text (more lenient)

    Returns:
        The extracted answer string, or None if no answer found
    """
    assert method in ['strict', 'flexible']

    # Optimization: For math problems, the final answer is usually at the end.
    # Only match on the last 300 characters.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == 'strict':
        # Look for "#### <number>" format
        solutions = re.findall(r'#### (\-?[0-9\.\,]+)', solution_str)
        if len(solutions) == 0:
            return None
        # Take the last solution, clean up formatting
        return solutions[-1].replace(',', '').replace('$', '')

    if method == 'flexible':
        # Find all numbers in the text
        answers = re.findall(r'(\-?[0-9\.\,]+)', solution_str)
        if len(answers) == 0:
            return None
        # Find the last valid number (not empty or just a dot)
        invalid_str = ['', '.']
        for final_answer in reversed(answers):
            if final_answer not in invalid_str:
                return final_answer
        return None

    return None


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = 'strict',
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Compute the reward score for a GSM8K solution.

    Args:
        solution_str: The full solution text from the model
        ground_truth: The correct answer
        method: Answer extraction method ('strict' or 'flexible')
        format_score: Score to return if answer format is wrong
        score: Score to return if answer is correct

    Returns:
        score if answer matches ground_truth, format_score otherwise
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return format_score
    if answer == ground_truth:
        return score
    return format_score


__all__ = ['compute_score', 'extract_solution']
