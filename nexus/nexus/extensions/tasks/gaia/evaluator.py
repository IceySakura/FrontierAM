"""Answer evaluation for GAIA benchmark.

GAIA uses exact match or normalized string comparison for evaluation.
"""

from typing import Any


def evaluate_answer(
    predicted: str | None,
    ground_truth: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Evaluate if predicted answer matches ground truth.

    Args:
        predicted: The answer from the agent
        ground_truth: The expected answer
        metadata: Optional metadata for custom evaluation logic

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False

    # Normalize both answers
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)

    # Exact match after normalization
    if pred_normalized == gt_normalized:
        return True

    # Check for numeric equivalence
    return bool(is_numeric_match(pred_normalized, gt_normalized))


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
    """Check if two strings represent the same number.

    Handles:
    - Different decimal precision (3.14 == 3.140)
    - Scientific notation (1e3 == 1000)
    - Comma separators (1,000 == 1000)
    """
    try:
        # Remove commas
        pred_clean = pred.replace(',', '')
        gt_clean = gt.replace(',', '')

        pred_num = float(pred_clean)
        gt_num = float(gt_clean)

        # Use relative tolerance for comparison
        return abs(pred_num - gt_num) < 1e-6 * max(abs(pred_num), abs(gt_num), 1.0)
    except (ValueError, TypeError):
        return False
