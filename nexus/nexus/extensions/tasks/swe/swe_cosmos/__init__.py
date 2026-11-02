from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from nexus.extensions.tasks.swe.swe_cosmos.dataset_handler import (
        eval_runtime_init,
        get_eval_report_map,
        get_eval_script,
        get_problem_statement,
        get_repo_language,
        infer_runtime_init,
    )


__all__ = [
    'get_eval_script',
    'get_eval_report_map',
    'get_repo_language',
    'get_problem_statement',
    'eval_runtime_init',
    'infer_runtime_init',
]


# Resolved on first access so importing a sibling module (e.g. generator) does not pull in the eval path and its deps.
def __getattr__(name: str) -> Any:
    if name in __all__:
        from nexus.extensions.tasks.swe.swe_cosmos import dataset_handler

        return getattr(dataset_handler, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
