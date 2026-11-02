"""swe_any dataset handler — evaluate feature-implementation instances via the shared swe_task flow.

Re-exports the module-level handler functions so ``ModuleDatasetHandler(swe_any)`` (registered for
``dataset_type = 'swe-any'`` in :mod:`nexus.extensions.tasks.swe.dataset_handler`) can dispatch to
them. See :mod:`.dataset_handler` for the reconstruct-the-hole-at-runtime design.
"""

from __future__ import annotations

from .dataset_handler import (
    eval_runtime_init,
    get_eval_report_map,
    get_eval_script,
    get_problem_statement,
    get_repo_language,
    infer_runtime_init,
)


__all__ = [
    'eval_runtime_init',
    'get_eval_report_map',
    'get_eval_script',
    'get_problem_statement',
    'get_repo_language',
    'infer_runtime_init',
]
