"""Answer evaluation for BrowseComp benchmark.

BrowseComp supports multiple evaluation strategies:
- exact: Exact string match after normalization
- fuzzy: Fuzzy matching with tolerance for minor variations
"""

import re

from difflib import SequenceMatcher
from typing import Any, Literal


def evaluate_answer(
    predicted: str | None,
    ground_truth: str,
    evaluation_type: Literal['exact', 'fuzzy'] = 'fuzzy',
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, float, dict]:
    """Evaluate if predicted answer matches ground truth.

    Args:
        predicted: The answer from the agent
        ground_truth: The expected answer
        evaluation_type: Type of evaluation to use
        metadata: Optional metadata for custom evaluation logic

    Returns:
        Tuple of (is_correct, score, evaluation_details)
        - is_correct: Boolean indicating if answer is correct
        - score: Float between 0.0 and 1.0
        - evaluation_details: Dict with additional evaluation information
    """
    if predicted is None:
        return False, 0.0, {'reason': 'No prediction'}

    if evaluation_type == 'exact':
        is_match, score = _exact_match(predicted, ground_truth)
    elif evaluation_type == 'fuzzy':
        is_match, score = _fuzzy_match(predicted, ground_truth)
    else:
        raise ValueError(f'Unknown evaluation type: {evaluation_type}')

    eval_details = {
        'ground_truth': ground_truth,
        'evaluation_type': evaluation_type,
        'similarity_score': score,
    }

    return is_match, score, eval_details


def _exact_match(predicted: str, ground_truth: str) -> tuple[bool, float]:
    """Exact match after normalization."""
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)

    # Check exact match
    if pred_normalized == gt_normalized:
        return True, 1.0

    # Check for numeric equivalence
    if is_numeric_match(pred_normalized, gt_normalized):
        return True, 1.0

    return False, 0.0


def _fuzzy_match(predicted: str, ground_truth: str) -> tuple[bool, float]:
    """Fuzzy match using sequence similarity."""
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)

    # Check exact match first
    if pred_normalized == gt_normalized:
        return True, 1.0

    # Check numeric equivalence
    if is_numeric_match(pred_normalized, gt_normalized):
        return True, 1.0

    # Calculate sequence similarity
    similarity = SequenceMatcher(None, pred_normalized, gt_normalized).ratio()

    # Check if ground truth is contained in prediction or vice versa
    if gt_normalized in pred_normalized or pred_normalized in gt_normalized:
        similarity = max(similarity, 0.85)

    # Check for date variations (e.g., "1988-96" vs "1988-1996")
    if _is_date_variation(pred_normalized, gt_normalized):
        similarity = 1.0

    is_match = similarity >= 0.8
    similarity = 1.0 if is_match else 0.0

    return is_match, similarity


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation at start/end
    - Strip quotes
    """
    text = text.lower().strip()

    # Remove surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # Remove punctuation at boundaries
    text = text.strip('.,!?;:')

    # Normalize whitespace
    return ' '.join(text.split())


def is_numeric_match(pred: str, gt: str) -> bool:
    """Check if two strings represent the same number."""
    try:
        # Remove commas and percentage signs
        pred_clean = pred.replace(',', '').replace('%', '')
        gt_clean = gt.replace(',', '').replace('%', '')

        pred_num = float(pred_clean)
        gt_num = float(gt_clean)

        # Handle percentage conversion
        if '%' in pred and '%' not in gt:
            pred_num = pred_num / 100
        elif '%' in gt and '%' not in pred:
            gt_num = gt_num / 100

        # Use relative tolerance for comparison
        return abs(pred_num - gt_num) < 1e-6 * max(abs(pred_num), abs(gt_num), 1.0)
    except (ValueError, TypeError):
        return False


def _is_date_variation(pred: str, gt: str) -> bool:
    """Check if two strings represent the same date/year range with variations.

    Examples:
    - "1988-96" == "1988-1996"
    - "aug 26 1999" == "august 26, 1999"
    """
    # Check year range variations (e.g., 1988-96 vs 1988-1996)
    year_range_pattern = r'(\d{4})-(\d{2,4})'

    pred_match = re.search(year_range_pattern, pred)
    gt_match = re.search(year_range_pattern, gt)

    if pred_match and gt_match:
        pred_start = pred_match.group(1)
        pred_end = pred_match.group(2)
        gt_start = gt_match.group(1)
        gt_end = gt_match.group(2)

        # Expand 2-digit year to 4-digit
        if len(pred_end) == 2:
            pred_end = pred_start[:2] + pred_end
        if len(gt_end) == 2:
            gt_end = gt_start[:2] + gt_end

        return pred_start == gt_start and pred_end == gt_end

    # Check date variations (remove punctuation and normalize spaces)
    pred_date = re.sub(r'[,\-/]', ' ', pred)
    gt_date = re.sub(r'[,\-/]', ' ', gt)
    pred_date = ' '.join(pred_date.split())
    gt_date = ' '.join(gt_date.split())

    return pred_date == gt_date
