"""
CodeInterpreter tool - Execute Python code in a stateful Jupyter kernel.
Adapted for the nexus tool framework using AsyncDirectPythonKernel.

Configuration:
    tool_configs:
      code_interpreter:
        python_path: /path/to/python  # Optional: Python inside the container
        work_dir: /tmp/jupyter_work   # Optional: image output dir inside the container
"""

import ast
import asyncio
import base64
import logging
import os
import re
import time
import uuid
import weakref

from typing import Any

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel
from nexus.runtime.executors.jupyter import AsyncDirectPythonKernel


logger = logging.getLogger(__name__)

# Default work directory inside the container for image output
_DEFAULT_RUNTIME_WORK_DIR = '/tmp/jupyter_work'


# ============== Input Model ==============


class CodeInterpreterInput(RobustBaseModel):
    """Code interpreter tool input parameters."""

    code: str = Field(description='Python code to execute in the Jupyter kernel.')
    timeout: float | None = Field(default=30, description='Timeout in seconds. Default: 30')


# ============== Code Normalization ==============

_CODE_BLOCK_RE = re.compile(r'```[^\n]*\n(.+?)```', re.DOTALL)


def _normalize_code(code: str) -> str:
    """Extract code from markdown code blocks if present."""
    matches = _CODE_BLOCK_RE.findall(code)
    if matches:
        return matches[0].strip()
    return code.strip()


# ============== Kernel Management ==============


class _JupyterKernelRunner:
    """Manages a persistent Jupyter kernel for stateful code execution.

    Each runner has its own work_dir for saving output images.
    """

    def __init__(self, python_path: str | None = None, work_dir: str | None = None, run_id: str | None = None) -> None:
        self._kernel = AsyncDirectPythonKernel(python_path)
        self._python_path = python_path
        self._started = False
        self._lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        self._run_id = run_id or str(uuid.uuid4())
        # Per-runner work directory for image output, named by run_id + timestamp for isolation
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_work_dir = work_dir or _DEFAULT_RUNTIME_WORK_DIR
        self.work_dir = os.path.join(base_work_dir, f'{self._run_id}_{timestamp}')
        os.makedirs(self.work_dir, exist_ok=True)
        self._tool_image_counter = 0

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._lock:
            if not self._started:
                await self._kernel.start()
                # Set the kernel's working directory to work_dir so that
                # relative-path saves (e.g. img.save("out.png")) land there.
                await self._kernel.execute(
                    f'import os; os.chdir({self.work_dir!r})',
                )
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


# Key: event loop -> {(run_id, python_path) -> _JupyterKernelRunner}
# run_id comes from ToolContext.run_id, ensuring each agent instance gets its own kernel.
_KERNELS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str | None, str | None], _JupyterKernelRunner]
] = weakref.WeakKeyDictionary()


def _get_kernel_runner(
    run_id: str | None = None,
    python_path: str | None = None,
    work_dir: str | None = None,
) -> _JupyterKernelRunner:
    """Get or create a kernel runner keyed by (run_id, python_path).

    Different agents (different run_id) get separate kernels with isolated state.
    """
    loop = asyncio.get_running_loop()
    runners = _KERNELS.setdefault(loop, {})
    key = (run_id, python_path)
    runner = runners.get(key)
    if runner is None:
        runner = _JupyterKernelRunner(python_path, work_dir=work_dir, run_id=run_id)
        runners[key] = runner
        logger.debug('Created new kernel runner for run_id=%s, python_path=%s', run_id, python_path)
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


# ============== Image Saving ==============


def _serve_image(image_base64: str, runner: _JupyterKernelRunner) -> tuple[str, str]:
    """Save a base64-encoded PNG image to disk and return (absolute_path, base64_data).

    The file is named using the runner's run_id and incrementing counter
    (e.g. ``<run_id>_tool_image_3.png``) for readability and traceability.
    """
    runner._tool_image_counter += 1
    image_file = f'{runner._run_id}_tool_image_{runner._tool_image_counter}.png'
    local_path = os.path.join(runner.work_dir, image_file)
    png_bytes = base64.b64decode(image_base64)
    with open(local_path, 'wb') as f:
        f.write(png_bytes)
    return os.path.abspath(local_path), image_base64


def _image_file_to_base64(file_path: str) -> str:
    """Read an image file from disk and return its base64-encoded string."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ============== Output Formatting ==============


def _format_output(
    result: Any,
    runner: _JupyterKernelRunner,
) -> tuple[str, list[dict[str, Any]], set[str]]:
    """Format kernel execution result into a readable string and interleaved image entries.

    Returns:
        (formatted_text, image_entries, served_image_basenames)
        - formatted_text: human-readable text output
        - image_entries: list of interleaved text/image_url dicts for each image
        - served_image_basenames: set that tracks images already saved from
          display_data / execute_result so they are not double-counted during
          the post-execution file scan.
    """
    parts: list[str] = []
    image_entries: list[dict[str, Any]] = []
    served_images: set[str] = set()

    if result.stdout:
        parts.append(f'--- stdout ---\n{result.stdout}')
    if result.stderr:
        parts.append(f'--- stderr ---\n{result.stderr}')
    if result.display:
        for item in result.display:
            if isinstance(item, dict):
                if 'image/png' in item:
                    # Save the displayed image to disk (counter incremented inside _serve_image)
                    image_path, b64_data = _serve_image(item['image/png'], runner)
                    image_basename = os.path.basename(image_path)
                    served_images.add(image_basename)
                    # Append interleaved text + image_url entries
                    image_entries.append({'type': 'text', 'text': f'\n\n![{image_basename}]({image_path})'})
                    image_entries.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{b64_data}'},
                    })
                elif 'text/plain' in item:
                    parts.append(f'--- display (text/plain) ---\n{item["text/plain"]}')
    if result.error:
        for err in result.error:
            traceback_text = '\n'.join(err.get('traceback', []))
            parts.append(f'--- error ---\n{traceback_text}')
    return '\n'.join(parts) if parts else '', image_entries, served_images


def _scan_new_images(
    runner: _JupyterKernelRunner,
    existing_files: set[str],
    served_images: set[str],
) -> list[dict[str, Any]]:
    """Scan the work directory for newly saved image files.

    Returns:
        image_entries: list of interleaved text/image_url dicts suitable for
        ToolResponse output. Each detected image produces two entries:
        a text entry with the image path and an image_url entry with base64 data.
    """
    image_entries: list[dict[str, Any]] = []
    if not os.path.exists(runner.work_dir):
        return []
    for f in sorted(os.listdir(runner.work_dir)):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue
        if f in existing_files or f in served_images:
            continue
        abs_path = os.path.join(runner.work_dir, f)
        runner._tool_image_counter += 1
        b64_data = _image_file_to_base64(abs_path)
        # Determine mime type from extension
        ext = f.rsplit('.', 1)[-1].lower()
        mime = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
        }.get(ext, 'image/png')
        image_entries.append({'type': 'text', 'text': f'\n\n![{f}]({abs_path})'})
        image_entries.append({
            'type': 'image_url',
            'image_url': {'url': f'data:{mime};base64,{b64_data}'},
        })
        logger.debug('Auto-detected saved image: %s -> %s', f, abs_path)
    return image_entries


# ============== Nexus Tool Registration ==============


@tool_registry.register(
    'jupyter_with_image',
    meta={
        'name': 'code_interpreter',
        'description': (
            'Executes Python code in a stateful Jupyter kernel and returns results.\n\n'
            'Capabilities:\n'
            '- Use PIL, NumPy, or OpenCV to process or analyze images to improve understanding.\n'
            '- Perform complex mathematical calculations or data manipulation.\n'
            '- Code execution is persistent: variables and functions are stored and reusable in subsequent calls.\n\n'
            'Image Handling:\n'
            '- Image file paths are provided directly in the conversation. Use PIL to open them:\n'
            '  ```python\n'
            '  from PIL import Image\n'
            '  img = Image.open("/path/to/image.png")\n'
            '  ```\n\n'
            'Visualization & Output:\n'
            '- To view generated images, return the object, use plt.show(), or save it (e.g., image.save("out.png")).\n'
            '- DO NOT use image.show().\n\n'
            'Pre-installed packages:\n'
            '- PIL, NumPy, OpenCV, Matplotlib, SciPy, Scikit-learn, Scikit-image, Pandas, SymPy'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def jupyter_with_image_tool(params: CodeInterpreterInput, context: ToolContext) -> ToolResponse:
    """Execute Python code in a stateful Jupyter kernel (runtime/sandbox version)."""
    code = params.code
    if not code or not code.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty code provided.'}],
            meta={'success': False, 'exit_code': 1},
        )

    # Read configuration from tool_configs
    tool_config = context.tool_configs.get('code_interpreter', {})
    python_path = tool_config.get('python_path', os.environ.get('PYTHON_PATH', None))
    work_dir = tool_config.get('work_dir', os.environ.get('CODE_INTERPRETER_WORK_DIR', _DEFAULT_RUNTIME_WORK_DIR))

    # Use run_id from context for kernel isolation — each agent gets its own kernel
    run_id = context.run_id

    # Normalize code (extract from markdown blocks if needed)
    normalized_code = _normalize_code(code)

    # Basic validation
    if 'exit(' in normalized_code.lower():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Using exit() is not allowed.'}],
            meta={'success': False, 'exit_code': 1},
        )

    try:
        ast.parse(normalized_code)
    except SyntaxError as e:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'SyntaxError: {e}'}],
            meta={'success': False, 'exit_code': 1},
        )

    # Get or create a kernel runner isolated by run_id
    runner = _get_kernel_runner(run_id=run_id, python_path=python_path, work_dir=work_dir)

    # Snapshot existing image files before execution (to detect newly saved ones)
    existing_image_files: set[str] = set()
    if os.path.exists(runner.work_dir):
        for f in os.listdir(runner.work_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                existing_image_files.add(f)

    # Execute code in the persistent kernel
    try:
        result = await runner.run(normalized_code, timeout=params.timeout)
    except Exception as e:
        error_msg = f'Error executing code: {e}'
        logger.error(error_msg)
        return ToolResponse(
            output=[{'type': 'text', 'text': error_msg}],
            meta={'success': False, 'exit_code': 1},
        )

    success = result.status == 'ok'
    formatted_output, display_image_entries, served_images = _format_output(result, runner)

    # Scan for newly saved image files (e.g. image.save('out.png'))
    new_image_entries = _scan_new_images(runner, existing_image_files, served_images)

    # Build interleaved output: text first, then images from display_data, then newly saved images
    all_image_entries = display_image_entries + new_image_entries

    if success:
        output_parts = []
        if formatted_output:
            output_parts.append({'type': 'text', 'text': formatted_output})
        output_parts.extend(all_image_entries)
        return ToolResponse(
            output=output_parts,
            meta={'success': True, 'exit_code': 0, 'stdout': result.stdout, 'stderr': result.stderr},
        )

    output_parts = [{'type': 'text', 'text': formatted_output}]
    output_parts.extend(all_image_entries)
    return ToolResponse(
        output=output_parts,
        meta={'success': False, 'exit_code': 1, 'stdout': result.stdout, 'stderr': result.stderr},
    )
