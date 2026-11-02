import asyncio

from nexus.extensions.tools.code_interpreter.python_code_interpreter import PythonCodeInterpreter
from nexus.extensions.tools.code_interpreter.safe_python_executor import (
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)


EXEC_AND_PRINT_CODE = 'x = 40\nprint(x + 2)\nx + 2'
UNAUTHORIZED_IMPORT_CODE = 'import os'
AUTHORIZED_IMPORT_EVAL_CODE = 'import math\nmath.sqrt(16)'
AUTHORIZED_IMPORT_PRINT_CODE = 'import math\nprint(math.sqrt(16))'
TIMEOUT_CODE = 'import time\ntime.sleep(2)'
INVALID_SYNTAX_CODE = 'def broken(:\n    pass'


def test_evaluate_python_code_executes_and_captures_logs() -> None:
    safe_state: dict[str, object] = {}
    python_interpreter = PythonCodeInterpreter()

    safe_result = evaluate_python_code(
        code=EXEC_AND_PRINT_CODE,
        state=safe_state,
    )
    python_result = python_interpreter.run(EXEC_AND_PRINT_CODE, timeout=None)

    assert safe_result['success'] is True
    assert python_result['success'] is True
    assert safe_result['output'].strip() == python_result['output'].strip() == '42'
    # safe executor currently runs in isolated subprocess and does not mutate caller state.
    assert safe_state == {}


def test_evaluate_python_code_rejects_unauthorized_import() -> None:
    python_interpreter = PythonCodeInterpreter()
    safe_result = evaluate_python_code(UNAUTHORIZED_IMPORT_CODE)
    python_result = python_interpreter.run(UNAUTHORIZED_IMPORT_CODE, timeout=None)

    assert safe_result['success'] is False
    assert 'Import of os is not allowed' in safe_result['output']
    assert python_result['success'] is True
    assert safe_result['success'] != python_result['success']


def test_evaluate_python_code_allows_authorized_import() -> None:
    python_interpreter = PythonCodeInterpreter()
    safe_result_eval = evaluate_python_code(
        code=AUTHORIZED_IMPORT_EVAL_CODE,
        authorized_imports=['math'],
    )
    safe_result_print = evaluate_python_code(
        code=AUTHORIZED_IMPORT_PRINT_CODE,
        authorized_imports=['math'],
    )
    python_result = python_interpreter.run(AUTHORIZED_IMPORT_PRINT_CODE, timeout=None)

    assert safe_result_eval['success'] is True
    assert safe_result_print['success'] is True
    assert python_result['success'] is True
    # Only stdout is returned, expression values are not surfaced in latest API.
    assert safe_result_eval['output'] == ''
    assert safe_result_print['output'].strip() == '4.0'
    assert python_result['output'].strip() == '4.0'


def test_evaluate_python_code_times_out_with_concurrent_asyncio_tasks() -> None:
    async def run_all_safe() -> list[dict[str, object] | BaseException]:
        safe_tasks = [
            asyncio.to_thread(
                evaluate_python_code,
                code=TIMEOUT_CODE,
                timeout_seconds=1,
            )
            for _ in range(50)
        ]
        return await asyncio.gather(*safe_tasks, return_exceptions=True)

    async def run_all_python() -> list[dict[str, object] | BaseException]:
        python_interpreter = PythonCodeInterpreter()
        python_tasks = [
            asyncio.to_thread(
                python_interpreter.run,
                TIMEOUT_CODE,
                timeout=1,
            )
            for _ in range(50)
        ]
        return await asyncio.gather(*python_tasks, return_exceptions=True)

    safe_results = asyncio.run(run_all_safe())
    python_results = asyncio.run(run_all_python())

    assert len(safe_results) == 50
    assert len(python_results) == 50
    for safe_result, python_result in zip(safe_results, python_results):
        assert not isinstance(safe_result, BaseException)
        assert not isinstance(python_result, BaseException)

        assert safe_result['success'] is False
        assert python_result['success'] is False
        assert safe_result['output'] == python_result['output'] == 'Execution timed out'
        assert safe_result['timeout'] is True
        assert python_result['timeout'] is True


def test_nested_evaluate_python_code_is_not_blocked_by_the_inherited_start_lock() -> None:
    """A tool callback may re-enter the executor from inside the forked child.

    ``start_process`` forks while holding the spawn lock, so the child inherits it
    locked. Unless that lock is reset after the fork, this nested call blocks until the
    outer timeout fires instead of returning its result.
    """

    def inner_eval() -> str:
        return evaluate_python_code(code='print(42)', timeout_seconds=5)['output']

    result = evaluate_python_code(
        code='print(inner_eval())',
        static_tools={**BASE_PYTHON_TOOLS, 'inner_eval': inner_eval},
        timeout_seconds=5,
    )

    assert result.get('timeout') is not True, 'nested call deadlocked on the inherited lock'
    assert result['success'] is True
    assert '42' in result['output']


def test_evaluate_python_code_returns_error_on_syntax_error_without_timeout() -> None:
    result = evaluate_python_code(code=INVALID_SYNTAX_CODE, timeout_seconds=None)

    assert result['success'] is False
    assert 'Code parsing failed on line' in result['output']
    assert 'SyntaxError' in result['output']


def test_evaluate_python_code_returns_error_on_syntax_error_with_timeout() -> None:
    result = evaluate_python_code(code=INVALID_SYNTAX_CODE, timeout_seconds=5)

    assert result['success'] is False
    assert 'Code parsing failed on line' in result['output']
    assert 'SyntaxError' in result['output']


