"""Solver runtime adapters for stem_mas tasks."""

from nexus.extensions.tasks.stem_mas.solvers.official import (
    OfficialSolverRequest,
    OfficialSolverResult,
    solve_with_official_pipeline,
)
from nexus.extensions.tasks.stem_mas.solvers.strategy import (
    SOLVER_STRATEGIES,
    SolverStrategy,
    resolve_solver_strategy,
)


__all__ = [
    'OfficialSolverRequest',
    'OfficialSolverResult',
    'SOLVER_STRATEGIES',
    'SolverStrategy',
    'resolve_solver_strategy',
    'solve_with_official_pipeline',
]
