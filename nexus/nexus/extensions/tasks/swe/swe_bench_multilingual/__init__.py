"""
SWE-bench Multilingual dataset handler.

This module provides the interface for handling SWE-bench Multilingual dataset instances
across multiple programming languages including Java, Python, JavaScript, Go, Rust, PHP, Ruby, and C.
"""

from nexus.extensions.tasks.swe.swe_bench_multilingual.dataset_handler import (
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
    'infer_runtime_init',
    'eval_runtime_init',
]
