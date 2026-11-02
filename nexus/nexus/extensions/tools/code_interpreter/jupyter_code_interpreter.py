"""
Local Jupyter code interpreter tool.
Aligns with python_code_interpreter usage but executes inside a persistent Jupyter kernel.

Configuration:
    tool_configs:
      jupyter_code_interpreter:
        python_path: /path/to/python  # Optional: specify which Python to use for the Jupyter kernel
        timeout: 60                   # Optional: default timeout in seconds
"""

import asyncio
import logging
import re
import weakref

from typing import Any

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.runtime.executors.jupyter import AsyncDirectPythonKernel


logger = logging.getLogger(__name__)


class JupyterCodeInterpreterInput(BaseModel):
    code: str = Field(description='The python code to execute in the jupyter environment.')


_CODE_BLOCK_RE = re.compile(r'```python(.*?)```', re.DOTALL)


class _JupyterKernelRunner:
    """Manages a persistent Jupyter kernel for code execution."""

    def __init__(self, python_path: str | None = None) -> None:
        """Initialize the kernel runner.

        Args:
            python_path: Path to the Python executable. If None, uses sys.executable
                        (the Python running the nexus runtime server).
        """
        self._kernel = AsyncDirectPythonKernel(python_path)
        self._python_path = python_path
        self._started = False
        self._lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._lock:
            if not self._started:
                await self._kernel.start()
                self._started = True
                if self._python_path:
                    logger.debug('Jupyter kernel started with custom python_path: %s', self._python_path)

    async def run(self, code: str, timeout: float | None = None):
        async with self._run_lock:
            await self._ensure_started()
            return await self._kernel.execute(code, timeout=timeout)

    async def close(self) -> None:
        async with self._run_lock:
            async with self._lock:
                if not self._started:
                    return
                await self._kernel.close()
                self._started = False


# Key: event loop -> {python_path -> _JupyterKernelRunner}
# Why use loop object as key (not id(loop)):
# - In tests/dev, event loops are often created and destroyed repeatedly.
# - id(old_loop) can be reused by a new loop object.
# - If cache key is id(loop), a new loop may accidentally hit an old runner bound
#   to a different (already closed) loop, causing cross-loop runtime errors.
_KERNELS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str | None, _JupyterKernelRunner]] = (
    weakref.WeakKeyDictionary()
)


def _get_kernel_runner(python_path: str | None = None) -> _JupyterKernelRunner:
    """Get or create a kernel runner for the given python_path.

    Args:
        python_path: Path to the Python executable. If None, uses sys.executable.

    Returns:
        A _JupyterKernelRunner instance for the given python_path.
    """
    loop = asyncio.get_running_loop()
    runners = _KERNELS.setdefault(loop, {})
    runner = runners.get(python_path)
    if runner is None:
        runner = _JupyterKernelRunner(python_path)
        runners[python_path] = runner
    return runner


async def _clear_kernel_runners(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Close and clear cached kernel runners for a loop (defaults to current loop)."""
    target_loop = loop or asyncio.get_running_loop()
    runners = _KERNELS.pop(target_loop, {})
    for runner in runners.values():
        try:
            await runner.close()
        except Exception:
            logger.warning('Failed to close Jupyter kernel runner during cleanup', exc_info=True)


async def _clear_all_kernel_runners() -> None:
    """Close and clear all cached kernel runners across loops."""
    all_runners: list[_JupyterKernelRunner] = []
    for loop in list(_KERNELS.keys()):
        all_runners.extend(_KERNELS.pop(loop, {}).values())
    for runner in all_runners:
        try:
            await runner.close()
        except Exception:
            logger.warning('Failed to close Jupyter kernel runner during global cleanup', exc_info=True)


def _normalize_code(code: str) -> str:
    matches = _CODE_BLOCK_RE.findall(code)
    if matches:
        return matches[0].strip()
    return code.strip()


def _format_output(result: Any) -> str:
    parts: list[str] = []
    if result.stdout:
        parts.append(f'--- stdout ---\n{result.stdout}')
    if result.stderr:
        parts.append(f'--- stderr ---\n{result.stderr}')
    if result.display:
        for item in result.display:
            if isinstance(item, dict):
                if 'text/plain' in item:
                    parts.append(f'--- display (text/plain) ---\n{item["text/plain"]}')
                if 'image/png' in item:
                    parts.append('--- display (image/png) ---\n[Image data received, not displayed]')
    if result.error:
        for err in result.error:
            traceback_text = '\n'.join(err.get('traceback', []))
            parts.append(f'--- error ---\n{traceback_text}')
    return '\n'.join(parts) if parts else 'No output.'


@tool_registry.register(
    'jupyter_code_interpreter',
    meta={
        'name': 'jupyter_code_interpreter',
        'description': (
            'Use this tool to execute Python code. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files). When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after configured timeout.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def jupyter_code_interpreter_tool(params: JupyterCodeInterpreterInput, context: ToolContext) -> ToolResponse:
    code = params.code
    if not code or not code.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty code provided.'}],
            meta={'success': False, 'exit_code': 1},
        )

    # Read python_path from tool_configs if configured
    # Example configuration:
    #   tool_configs:
    #     jupyter_code_interpreter:
    #       python_path: /root/miniconda3/bin/python
    tool_config = context.tool_configs.get('jupyter_code_interpreter', {})
    if not isinstance(tool_config, dict):
        tool_config = {}
    python_path = tool_config.get('python_path')
    timeout = tool_config.get('timeout', 30)

    normalized_code = _normalize_code(code)
    result = await _get_kernel_runner(python_path).run(normalized_code, timeout=timeout)

    success = result.status == 'ok'
    formatted_output = _format_output(result)
    if success:
        output = f'Successfully executed:\nOutput:\n{formatted_output}'
        return ToolResponse(
            output=[{'type': 'text', 'text': output}],
            meta={'success': True, 'exit_code': 0, 'stdout': result.stdout, 'stderr': result.stderr},
        )

    error_output = f'Error executing code:\nOutput:\n{formatted_output}'
    return ToolResponse(
        output=[{'type': 'text', 'text': error_output}],
        meta={'success': False, 'exit_code': 1, 'stdout': result.stdout, 'stderr': result.stderr},
    )
