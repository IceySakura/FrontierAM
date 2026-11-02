from enum import Enum


class ReasoningReward(Enum):
    """Reward values aligned with HyTuner."""

    CHOSEN = 1
    REJECTED = 0
    FORMAT = 0.1
    THRESHOLD = -99
    UNJUDGED = -100
    FAILED = -200
    RS_FAILED = -201
    TASK_FAILED = -300
