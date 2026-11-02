from __future__ import annotations

from collections.abc import Sequence
from enum import Enum


class TokenBudgetComponent(str, Enum):
    """Mutually exclusive token sources that may consume a trajectory budget.

    Gateway's token-only budget tracker treats strict prompt extensions after a
    committed sampled stream as external observations and accounts them under
    ``TOOL_OBSERVATION``. Tool outputs are one such producer; the tracker cannot
    distinguish message roles from token IDs alone.
    """

    AGENT_OUTPUT = 'agent_output'
    TOOL_OBSERVATION = 'tool_observation'
    COMPACTION_OUTPUT = 'compaction_output'


DEFAULT_TOKEN_BUDGET_COMPONENTS = frozenset(TokenBudgetComponent)


def parse_token_budget_include(value: Sequence[str] | None) -> frozenset[TokenBudgetComponent]:
    """Parse ``token_budget_include`` into validated components."""
    if value is None:
        return DEFAULT_TOKEN_BUDGET_COMPONENTS
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError('token_budget_include must be a list of token component names')

    try:
        components = [TokenBudgetComponent(component) for component in value]
    except ValueError as error:
        valid_values = ', '.join(component.value for component in TokenBudgetComponent)
        raise ValueError(f'Invalid token_budget_include value; choose from: {valid_values}') from error
    if len(components) != len(set(components)):
        raise ValueError('token_budget_include must not contain duplicate components')
    if TokenBudgetComponent.AGENT_OUTPUT not in components:
        raise ValueError('token_budget_include must contain agent_output')
    return frozenset(components)
