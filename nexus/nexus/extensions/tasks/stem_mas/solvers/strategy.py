"""Solver strategy contracts for stem_mas benchmark runs."""

from __future__ import annotations

from typing import Any, ClassVar


class SolverStrategy:
    """Describe how a benchmark candidate is generated before evaluation."""

    registry: ClassVar[dict[str, type[SolverStrategy]]] = {}

    name: ClassVar[str] = ''
    family: ClassVar[str] = ''
    agent_surface: ClassVar[str] = ''
    tool_mode: ClassVar[str] = ''
    official_method: ClassVar[bool] = False
    allowed_agent_names: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not cls.name:
            return
        if cls.name in SolverStrategy.registry:
            registered = SolverStrategy.registry[cls.name]
            raise ValueError(
                f'Duplicate solver strategy {cls.name!r}: '
                f'{registered.__module__}.{registered.__name__} and {cls.__module__}.{cls.__name__}'
            )
        SolverStrategy.registry[cls.name] = cls

    def metadata(self) -> dict[str, Any]:
        return {
            'solver_strategy': self.name,
            'solver_family': self.family,
            'solver_agent_surface': self.agent_surface,
            'solver_tool_mode': self.tool_mode,
            'solver_official_method': self.official_method,
        }


class NoAgentSolverStrategy(SolverStrategy):
    name = 'no_agent'
    family = 'direct'
    agent_surface = 'single_turn'
    tool_mode = 'none'
    allowed_agent_names = ('single_turn',)


class CodexSolverStrategy(SolverStrategy):
    name = 'codex'
    family = 'agent'
    agent_surface = 'codex'
    tool_mode = 'none'
    allowed_agent_names = ('codex', 'stem_mas_codex')


class CodexLeanSolverStrategy(SolverStrategy):
    name = 'codex_lean'
    family = 'agent'
    agent_surface = 'codex'
    tool_mode = 'lean_lsp_mcp'
    allowed_agent_names = ('stem_mas_codex',)


class OfficialSolverStrategy(SolverStrategy):
    name = 'official_solver'
    family = 'official'
    agent_surface = 'external'
    tool_mode = 'benchmark_specific'
    official_method = True
    allowed_agent_names = ('single_turn',)


SOLVER_STRATEGIES = SolverStrategy.registry


def resolve_solver_strategy(config: dict[str, Any]) -> SolverStrategy:
    """Resolve and validate the configured candidate-generation strategy."""

    raw_strategy = config.get('solver_strategy')
    if not isinstance(raw_strategy, str) or not raw_strategy.strip():
        raise ValueError('solver_strategy must be a non-empty string.')
    strategy_name = raw_strategy.strip()

    try:
        strategy_cls = SOLVER_STRATEGIES[strategy_name]
    except KeyError as exc:
        known = ', '.join(sorted(SOLVER_STRATEGIES))
        raise ValueError(f'Unknown solver_strategy {strategy_name!r}. Known strategies: {known}.') from exc

    strategy = strategy_cls()
    validate_solver_strategy(config, strategy)
    return strategy


def validate_solver_strategy(config: dict[str, Any], strategy: SolverStrategy) -> None:
    agent_name = str(config.get('agent_name') or '').strip()
    if strategy.allowed_agent_names and agent_name not in strategy.allowed_agent_names:
        allowed = ', '.join(strategy.allowed_agent_names)
        raise ValueError(f'solver_strategy={strategy.name} requires agent_name in [{allowed}].')


__all__ = [
    'CodexLeanSolverStrategy',
    'CodexSolverStrategy',
    'NoAgentSolverStrategy',
    'OfficialSolverStrategy',
    'SOLVER_STRATEGIES',
    'SolverStrategy',
    'resolve_solver_strategy',
    'validate_solver_strategy',
]
