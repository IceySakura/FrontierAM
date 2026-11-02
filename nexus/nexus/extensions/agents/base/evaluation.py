import logging

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from nexus.utils.reward_score import ReasoningReward


logger = logging.getLogger(__name__)


# Negative status/failure sentinels from ReasoningReward that should be passed
# through without being clamped by evaluate_output. This lets downstream consumers
# (e.g. hytuner training loops) distinguish genuine 0.0 scores from evaluator
# or reward-service failures. All other negative values are still clamped to 0.0.
_PASSTHROUGH_SENTINELS: frozenset[float] = frozenset(
    float(member.value) for member in ReasoningReward if member.value < 0
)


@dataclass
class EvaluationDetails:
    """Base class for structured evaluation details.

    Subclass this to carry mode-specific metrics alongside the scalar score.
    """

    status: str = 'success'
    msg: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Evaluator(Protocol):
    """
    Interface for evaluating agent outputs.
    """

    async def evaluate(
        self,
        prediction: Any,
        reference: Any,
        question: Any | None = None,
    ) -> float:
        """
        Evaluate prediction against reference.
        """
        ...

    async def evaluate_with_details(
        self,
        prediction: Any,
        reference: Any,
        question: Any | None = None,
    ) -> tuple[float, EvaluationDetails | None]:
        """
        Evaluate prediction against reference with structured details.

        Default implementation delegates to evaluate() and returns None for details.
        Subclasses can override this to provide mode-specific evaluation details.
        """
        ...


async def evaluate_output(
    prediction: Any,
    reference: Any,
    question: Any | None = None,
    *,
    evaluator: Evaluator,
    need_details: bool = False,
) -> tuple[float, EvaluationDetails | None, bool]:
    """
    Sanitizes inputs and executes the evaluation strategy.

    Always returns (score, evaluation_details, is_valid_reward).
    - When need_details is False, evaluation_details is always None.
    - is_valid_reward is False when the evaluator failed (exception) or returned
      a negative sentinel from ReasoningReward (e.g. UNJUDGED/FAILED/RS_FAILED),
      so downstream consumers can distinguish genuine 0.0 scores from infra
      failures. Input-validation early returns (None/empty prediction or
      reference) are treated as valid 0.0 scores (is_valid_reward=True),
      because the score reflects a real dataset/agent state, not a reward
      service failure.
    """
    # 1. System Guard: Missing or empty reference is a data issue, score 0.0.
    if reference is None:
        error_msg = 'Reference is None; this is likely a dataset bug. Defaulting to 0.0.'
        logger.error(error_msg)
        if need_details:
            return 0.0, EvaluationDetails(status='failed', msg=error_msg), True
        return 0.0, None, True

    # 2. Sanitization: Normalize strings, pass-through others.
    if isinstance(prediction, str) and not (prediction := prediction.strip()):
        error_msg = 'Prediction is empty string; defaulting to 0.0.'
        logger.warning(error_msg)
        if need_details:
            return 0.0, EvaluationDetails(status='failed', msg=error_msg), True
        return 0.0, None, True

    if isinstance(reference, str):
        reference = reference.strip()
        if not reference:
            error_msg = 'Reference is empty string; this is likely a dataset bug. Defaulting to 0.0.'
            logger.error(error_msg)
            if need_details:
                return 0.0, EvaluationDetails(status='failed', msg=error_msg), True
            return 0.0, None, True

    if prediction is None:
        error_msg = 'Prediction is None; defaulting to 0.0.'
        logger.warning(error_msg)
        if need_details:
            return 0.0, EvaluationDetails(status='failed', msg=error_msg), True
        return 0.0, None, True

    # 3. Execute
    is_valid_reward = True
    try:
        details: EvaluationDetails | None = None
        if need_details:
            score, details = await evaluator.evaluate_with_details(
                prediction=prediction,
                reference=reference,
                question=question,
            )
        else:
            score = await evaluator.evaluate(prediction=prediction, reference=reference, question=question)
        if score in _PASSTHROUGH_SENTINELS:
            is_valid_reward = False
        # 4. Normalize
        raw_score = float(score)
        # Preserve ReasoningReward failure/status sentinels (e.g. FAILED=-200,
        # RS_FAILED=-201, TASK_FAILED=-300) so the training layer can tell them
        # apart from a legitimate 0.0. is_valid_reward is already False for
        # sentinels (set above), so downstream consumers get both signals.
        # Everything else is clamped to [0.0, 1.0] as a safety net against
        # buggy evaluators.
        if raw_score in _PASSTHROUGH_SENTINELS:
            return raw_score, details, is_valid_reward
        normalized_score = max(0.0, min(1.0, raw_score))
        return normalized_score, details, is_valid_reward

    except (ValueError, TypeError) as e:
        error_msg = f'Invalid score from {evaluator.__class__.__name__}: {e}'
        logger.error(error_msg)
        if need_details:
            return 0.0, EvaluationDetails(status='failed', msg=error_msg), False
        return 0.0, None, False
    except Exception as e:
        error_msg = f'Crash in {evaluator.__class__.__name__}: {e}'
        logger.error(error_msg, exc_info=True)
        if need_details:
            return 0.0, EvaluationDetails(status='failed', msg=error_msg), False
        return 0.0, None, False
