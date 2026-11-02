"""
开发信息: DrakosXiao(2025.12)
工具名称: PYTHON代码解释器工具
工具类型: RUNTIME
"""

# 基础导入
import asyncio
import logging
import multiprocessing
import re
import sys

from io import StringIO
from typing import Any

from pydantic import BaseModel, Field

# 项目导入
from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.code_interpreter.safe_python_executor import (
    BASE_BUILTIN_MODULES,
    evaluate_python_code,
    start_process,
)


logger = logging.getLogger(__name__)


# ============== 原有 PythonCodeInterpreter 实现==============
class PythonCodeInterpreter:
    def __init__(self):
        self.globals_dict: dict[str, Any] = {}
        self.locals_dict: dict[str, Any] = {}

    @staticmethod
    def sanitize_input(query: str) -> str:
        query = re.sub(r'^(\s|`)*(?i:python)?\s*', '', query)
        return re.sub(r'(\s|`)*$', '', query)

    @classmethod
    def worker(
        cls,
        command: str,
        globals_dict: dict | None,
        locals_dict: dict | None,
        queue: multiprocessing.Queue,
    ) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            cleaned_command = cls.sanitize_input(command)
            exec(cleaned_command, globals_dict, locals_dict)
            sys.stdout = old_stdout
            queue.put({'success': True, 'output': mystdout.getvalue()})
        except Exception as e:
            sys.stdout = old_stdout
            queue.put({'success': False, 'output': repr(e)})

    def run(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        queue: multiprocessing.Queue = multiprocessing.Queue()

        if timeout is not None:
            p = multiprocessing.Process(target=self.worker, args=(command, self.globals_dict, self.locals_dict, queue))
            start_process(p)
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                return {'success': False, 'output': 'Execution timed out', 'timeout': True}
        else:
            self.worker(command, self.globals_dict, self.locals_dict, queue)

        return queue.get()


# ============== 全局 PythonCodeInterpreter 实例（保持会话状态） ==============
_PythonCodeInterpreter = PythonCodeInterpreter()
_SafePythonInterpreterState: dict[str, Any] = {'__name__': '__main__'}


# ============== Nexus 工具适配层 ==============
class PythonCodeInterpreterInput(BaseModel):
    code: str = Field(description='The python code to execute to do further analysis or calculation.')
    timeout: int | None = Field(default=30, description='Timeout in seconds. Default: 30')


@tool_registry.register(  # pyright: ignore[reportArgumentType]
    'python_code_interpreter',
    meta={
        'name': 'python_code_interpreter',
        'description': (
            'Use this to execute python code and do data analysis or calculation. '
            'If you want to see the output of a value, you should print it out with `print(...)`. '
            'Variables persist across calls.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def python_code_interpreter_tool(params: PythonCodeInterpreterInput, context: ToolContext) -> ToolResponse:
    code = params.code
    if not code or not code.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty code provided.'}],
            meta={'success': False, 'exit_code': 1},
        )

    # 读取工具配置中的全局最大超时
    cfg = context.tool_configs.get('python_code_interpreter', {})
    max_timeout = cfg.get('max_timeout') if isinstance(cfg, dict) else None
    timeout = params.timeout
    timeout_capped = False
    if max_timeout is not None and timeout is not None and timeout > max_timeout:
        timeout = max_timeout
        timeout_capped = True

    result = await asyncio.to_thread(_PythonCodeInterpreter.run, code, timeout=timeout)

    timeout_notice = (
        f'\n[Notice] Requested timeout ({params.timeout}s) exceeded the configured max_timeout ({max_timeout}s), '
        f'clamped to {max_timeout}s.'
        if timeout_capped
        else ''
    )

    is_error = result == 'Execution timed out' or 'Error' in result or 'Exception' in result
    if is_error:
        output = f'Error executing code:\nError: {result}{timeout_notice}'
        return ToolResponse(
            output=[{'type': 'text', 'text': output}],
            meta={'success': False, 'exit_code': 1, 'stderr': result},
        )
    output = f'Successfully executed:\nStdout: {result}{timeout_notice}'
    return ToolResponse(
        output=[{'type': 'text', 'text': output}],
        meta={'success': True, 'exit_code': 0, 'stdout': result},
    )


@tool_registry.register(  # pyright: ignore[reportArgumentType]
    'safe_python_code_interpreter',
    meta={
        'name': 'safe_python_code_interpreter',
        'description': (
            'Execute Python code with a sandboxed AST evaluator for analysis and calculations. '
            'Constraints: only safe built-in tools are available, imports are restricted to an allow-list '
            f'({", ".join(BASE_BUILTIN_MODULES)}), and filesystem/process/network/system access is blocked '
            '(e.g. open, os, pathlib, subprocess, socket). '
            'If you need a value in output, use print(...). Variables DO NOT persist across calls in this safe interpreter.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def safe_python_code_interpreter_tool(params: PythonCodeInterpreterInput, context: ToolContext) -> ToolResponse:
    code = params.code
    if not code or not code.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty code provided.'}],
            meta={'success': False, 'exit_code': 1},
        )

    # 直接调用原有逻辑执行代码解释器
    result = await asyncio.to_thread(evaluate_python_code, code)  # TODO: add back timeout

    success = result['success']
    py_output = result['output']

    if not success:
        return ToolResponse(
            output=[{'type': 'text', 'text': py_output}],
            meta={'success': False, 'exit_code': 1, 'stderr': py_output},
        )

    if not py_output:
        py_output = 'No Stdout is generated. Please use print(...) to print the output.'
    else:
        py_output = f'Stdout: {py_output}'

    return ToolResponse(
        output=[{'type': 'text', 'text': py_output}],
        meta={'success': True, 'exit_code': 0, 'stdout': py_output},
    )
