"""Rule: trajectory_truncation."""

from ._termination_reason import termination_reason_rule


RULE = termination_reason_rule(
    'trajectory_truncation',
    '轨迹截断',
    {
        'TRUNCATION',
        'OUTPUT_LENGTH_EXCEEDED',
    },
)
