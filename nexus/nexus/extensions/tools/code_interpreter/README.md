# Code Interpreter Tools

This directory provides two runtime code execution tools.

## Tools

- `python_code_interpreter`: lightweight `exec`-based interpreter.
- `jupyter_code_interpreter`: Jupyter-kernel-based interpreter with richer outputs.

## Persistence & State

**python_code_interpreter**
- State container: module-level singleton (`globals_dict/locals_dict`).
- Default behavior: runs in a **new subprocess** when `timeout` is set (default `30s`).
- Persistence only if `timeout=None` (in-process execution, no timeout safety).

**jupyter_code_interpreter**
- State container: **Jupyter kernel process** (`AsyncDirectPythonKernel`).
- Persistence: variables persist **across tool calls** within the same runtime process and event loop.
- Kernel/runtime restarts reset state.
- Outputs include `stdout`, `stderr`, and rich display data.

## Execution Placement (LLM vs Tool)

In typical `nexus run-single` flows:
- LLM requests are made by the runner/agent process.
- Runtime tools execute inside the runtime container.
They are different processes and usually different containers.

## Testing (Manual Export + Unit + E2E)

From the Nexus repo root:

Manually export environment variables:

```bash
export SEARCH_API_KEY="<your-key>"
export SEARCH_API_USER="<your-user>"
export OPENAI_API_KEY="<your-openai-key>"
export CODE_INTERPRETER_TOOL="jupyter_code_interpreter"  # or python_code_interpreter
```

Unit test (new tool):

```bash
PYTHONPATH=$PWD uv run pytest tests/unit/extensions/tools/test_code_interpreter.py -q
```

End-to-end test (real request, no mock):

```bash
PYTHONPATH=$PWD uv run pytest tests/e2e/test_e2e_code_interpreter.py -q
```

Notes for E2E:
- Requires Docker and will auto-build the runtime image on first run.
- Build uses `make docker-runtime`, then tags a **temporary** image based on git state.
- Git state only considers changes under `deployment/`, `nexus/`, plus `pyproject.toml` and `Makefile`.
- Temporary tags look like: `nexus-runtime:<tag>-e2e-tmp-<sha>` (dirty trees include `-dirty-<hash>`).
- Only the latest temporary tag is kept; older temp images are removed automatically.
- Subsequent runs reuse the temp image (faster) unless you change relevant files or force rebuild.
- Perf log is appended to `workspace/research/e2e_perf.log` for each tool run.

Controls:
- `NEXUS_E2E_RUNTIME_IMAGE_SKIP_BUILD=1` to skip auto-build (test will skip if image missing).
- `NEXUS_E2E_RUNTIME_IMAGE_FORCE_REBUILD=1` to force rebuild even if the image exists.

## Simple Demo (Config)

Minimal demo config (uses env vars to pick tool):

```yaml
task_config:
  question: "Please call ${oc.env:CODE_INTERPRETER_TOOL} to execute `print(1+1)` and report the output."

processors:
  trace:
    args:
      file_path: "workspace/research/trace_infoseek.jsonl"
```

## Input Parameters

- `code` (string): code to execute.
- `timeout` (seconds, optional): execution timeout.
