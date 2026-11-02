"""Answer evaluation for HLE benchmark.

HLE supports multiple evaluation strategies:
- exact: Exact string match after normalization
- fuzzy: Fuzzy matching with tolerance
- semantic: Semantic similarity using embeddings (optional)
"""

from difflib import SequenceMatcher
from typing import Any, Literal


def evaluate_answer(
    predicted: str | None,
    ground_truth: str | list[str],
    evaluation_type: Literal['exact', 'fuzzy', 'semantic'] = 'fuzzy',
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, float, dict]:
    """Evaluate if predicted answer matches ground truth.

    Args:
        predicted: The answer from the agent
        ground_truth: The expected answer(s)
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

    # Handle multiple ground truths
    ground_truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]

    best_score = 0.0
    eval_details = {}

    for gt in ground_truths:
        if evaluation_type == 'exact':
            is_match, score = _exact_match(predicted, gt)
        elif evaluation_type == 'fuzzy':
            is_match, score = _fuzzy_match(predicted, gt)
        elif evaluation_type == 'semantic':
            is_match, score = _semantic_match(predicted, gt, metadata)
        else:
            raise ValueError(f'Unknown evaluation type: {evaluation_type}')

        if score > best_score:
            best_score = score
            eval_details = {
                'matched_ground_truth': gt,
                'evaluation_type': evaluation_type,
                'similarity_score': score,
            }

    # Consider correct if score >= 0.8
    is_correct = best_score >= 0.8

    return is_correct, best_score, eval_details


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

    is_match = similarity >= 0.8
    similarity = 1.0 if is_match else 0.0
    return is_match, similarity


def _semantic_match(predicted: str, ground_truth: str, metadata: dict[str, Any] | None) -> tuple[bool, float]:
    """Semantic similarity match using embeddings.

    Note: This is a placeholder. In production, you would use:
    - OpenAI embeddings API
    - Sentence transformers
    - Other embedding models
    """
    # For now, fall back to fuzzy match
    # TODO: Implement semantic similarity using embeddings
    return _fuzzy_match(predicted, ground_truth)


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
    - Percentages (50% == 0.5)
    """
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
