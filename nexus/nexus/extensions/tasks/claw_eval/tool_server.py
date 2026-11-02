"""MCP Tool Server bridge for claw-eval mock services.

Bridges claw-eval's HTTP mock services into the MCP protocol so that
nexus agents can call them as standard MCP tools. For each claw-eval task,
this module:

1. Starts the required mock services (gmail, calendar, etc.) as local processes
2. Launches an MCP stdio server that exposes each tool defined in task.yaml
3. When the agent calls an MCP tool, the bridge forwards the request to the
   corresponding HTTP mock service endpoint and returns the result

The MCP server script is generated dynamically and run as a subprocess,
allowing nexus's MCPToolManager to connect via stdio transport.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service lifecycle helpers (adapted from claw_eval.runner.services)
# ---------------------------------------------------------------------------


@dataclass
class ServiceHandle:
    """Tracks a running mock service subprocess."""

    name: str
    port: int
    proc: subprocess.Popen
    reset_endpoint: str | None = None


def _is_healthy(url: str, method: str = 'POST') -> bool:
    """Probe a health-check endpoint. Returns True if status < 500."""
    try:
        import httpx

        headers = {'X-Health-Check': '1'}
        with httpx.Client(trust_env=False, timeout=2.0) as client:
            resp = client.get(url, headers=headers) if method.upper() == 'GET' else client.post(url, headers=headers)
        return resp.status_code < 500
    except Exception as e:
        logger.debug('Health check failed for %s: %s', url, e)
        return False


def start_mock_services(
    services: list[dict[str, Any]],
    cwd: Path,
    port_offset: int = 0,
) -> list[ServiceHandle]:
    """Start mock service subprocesses declared in task.yaml.

    Args:
        services: List of ServiceDef dicts from task.yaml
        cwd: Working directory (claw-eval project root)
        port_offset: Offset to add to each port for parallel execution

    Returns:
        List of ServiceHandle for started services
    """
    handles: list[ServiceHandle] = []

    for svc in services:
        name = svc['name']
        port = svc['port'] + port_offset
        command = svc['command']
        health_check = svc['health_check']
        health_check_method = svc.get('health_check_method', 'POST')
        ready_timeout = svc.get('ready_timeout', 10)
        reset_endpoint = svc.get('reset_endpoint')
        env_extra = svc.get('env', {})

        # Apply port offset to URLs
        if port_offset:
            import re

            def _shift_url(url: str) -> str:
                return re.sub(
                    r'localhost:(\d+)',
                    lambda m: f'localhost:{int(m.group(1)) + port_offset}',
                    url,
                )

            health_check = _shift_url(health_check)
            if reset_endpoint:
                reset_endpoint = _shift_url(reset_endpoint)

        # Check if already running
        if _is_healthy(health_check, health_check_method):
            logger.debug(f"Service '{name}' already running on port {port}")
            handles.append(ServiceHandle(name, port, None, reset_endpoint))  # type: ignore[arg-type]
            continue

        # Build command
        cmd = shlex.split(command)
        if cmd and cmd[0] in ('python', 'python3'):
            cmd[0] = sys.executable

        # Build env: strip proxy vars
        base_env = dict(os.environ)
        for proxy_key in (
            'http_proxy',
            'https_proxy',
            'all_proxy',
            'HTTP_PROXY',
            'HTTPS_PROXY',
            'ALL_PROXY',
        ):
            base_env.pop(proxy_key, None)
        env = {**base_env, **(env_extra or {})}
        if port_offset:
            env['PORT'] = str(port)

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for health
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ''
                raise RuntimeError(f"Service '{name}' exited immediately (rc={proc.returncode}): {stderr[:500]}")
            if _is_healthy(health_check, health_check_method):
                handles.append(ServiceHandle(name, port, proc, reset_endpoint))
                logger.debug(f"Service '{name}' started on port {port}")
                break
            time.sleep(0.3)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            raise RuntimeError(f"Service '{name}' did not become ready within {ready_timeout}s")

    return handles


def stop_mock_services(handles: list[ServiceHandle]) -> None:
    """Stop all mock service subprocesses."""
    for handle in reversed(handles):
        if handle.proc is not None:
            handle.proc.terminate()
            try:
                handle.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait()
            logger.debug(f"Service '{handle.name}' stopped")


def reset_mock_services(handles: list[ServiceHandle]) -> None:
    """Reset all mock services to initial state."""
    import httpx

    for handle in handles:
        if handle.reset_endpoint:
            try:
                httpx.post(handle.reset_endpoint, timeout=5.0)
            except Exception as exc:
                logger.warning(f"Reset failed for service '{handle.name}': {exc}")


# ---------------------------------------------------------------------------
# MCP bridge server script generation
# ---------------------------------------------------------------------------


def generate_mcp_bridge_script(
    tools: list[dict[str, Any]],
    tool_endpoints: list[dict[str, Any]],
    port_offset: int = 0,
) -> str:
    """Generate a Python script that acts as an MCP stdio server.

    The generated script bridges tool calls to the HTTP mock service endpoints.
    It uses the `mcp` Python package for MCP protocol implementation.

    Args:
        tools: List of ToolSpec dicts from task.yaml (name, description, input_schema)
        tool_endpoints: List of ToolEndpoint dicts (tool_name, url, method)
        port_offset: Port offset for parallel execution

    Returns:
        Python script content as a string
    """
    # Build endpoint map with port offset applied
    endpoint_map = {}
    for ep in tool_endpoints:
        url = ep['url']
        method = ep.get('method', 'POST')
        if port_offset:
            import re

            url = re.sub(
                r'localhost:(\d+)',
                lambda m: f'localhost:{int(m.group(1)) + port_offset}',
                url,
            )
        endpoint_map[ep['tool_name']] = {'url': url, 'method': method}

    # Serialize JSON data as compact strings (no newlines).
    # Use repr() to produce a safely-escaped Python string literal
    # that handles single quotes, backslashes, and unicode correctly.
    tools_repr = repr(json.dumps(tools, ensure_ascii=False))
    endpoints_repr = repr(json.dumps(endpoint_map, ensure_ascii=False))

    # Build the script without textwrap.dedent to avoid indentation issues
    # caused by interpolated multi-line content.
    #
    # IMPORTANT: We use the low-level MCP server API instead of FastMCP's
    # decorator-based registration to preserve the original tool input_schema
    # exactly as defined in task.yaml.  FastMCP's @mcp.tool() decorator
    # infers schema from function signatures, which corrupts parameter
    # names (e.g. wrapping everything under a "kwargs" key).  The low-level
    # approach passes tool schemas verbatim so the agent sees the correct
    # parameter names (e.g. "message_id", not "kwargs").
    return (
        '#!/usr/bin/env python3\n'
        '"""Auto-generated MCP bridge server for claw-eval mock services."""\n'
        '\n'
        'import json\n'
        'import sys\n'
        '\n'
        'import httpx\n'
        'from mcp.server.lowlevel import Server\n'
        'from mcp.types import Tool, TextContent\n'
        'import mcp.server.stdio\n'
        'import asyncio\n'
        '\n'
        '# Tool definitions from task.yaml\n'
        f'TOOLS = json.loads({tools_repr})\n'
        '\n'
        '# Endpoint mapping\n'
        f'ENDPOINTS = json.loads({endpoints_repr})\n'
        '\n'
        '_http_client = httpx.Client(timeout=30.0, trust_env=False)\n'
        'server = Server("claw-eval-tools")\n'
        '\n'
        '\n'
        'def _dispatch(tool_name: str, arguments: dict) -> tuple[str, bool]:\n'
        '    """Forward a tool call to the corresponding HTTP mock service.\n'
        '    Returns (response_text, is_error).\n'
        '    """\n'
        '    endpoint = ENDPOINTS.get(tool_name)\n'
        '    if endpoint is None:\n'
        '        return json.dumps({"error": f"Unknown tool: {tool_name}"}), True\n'
        '\n'
        '    try:\n'
        '        resp = _http_client.request(\n'
        '            method=endpoint["method"],\n'
        '            url=endpoint["url"],\n'
        '            json=arguments,\n'
        '        )\n'
        '        return resp.text, resp.status_code >= 400\n'
        '    except Exception as exc:\n'
        '        return json.dumps({"error": str(exc)}), True\n'
        '\n'
        '\n'
        '# Build a name -> schema lookup for list_tools\n'
        '_TOOL_SPECS = []\n'
        'for _tool_def in TOOLS:\n'
        '    _TOOL_SPECS.append(Tool(\n'
        '        name=_tool_def["name"],\n'
        '        description=_tool_def.get("description", ""),\n'
        '        inputSchema=_tool_def.get("input_schema", {"type": "object", "properties": {}}),\n'
        '    ))\n'
        '\n'
        '\n'
        '@server.list_tools()\n'
        'async def handle_list_tools():\n'
        '    return _TOOL_SPECS\n'
        '\n'
        '\n'
        '@server.call_tool()\n'
        'async def handle_call_tool(name: str, arguments: dict | None):\n'
        '    text, is_error = _dispatch(name, arguments or {})\n'
        '    return [TextContent(type="text", text=text)]\n'
        '\n'
        '\n'
        'async def main():\n'
        '    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):\n'
        '        await server.run(\n'
        '            read_stream,\n'
        '            write_stream,\n'
        '            server.create_initialization_options(),\n'
        '        )\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    asyncio.run(main())\n'
    )


def build_mcp_server_spec(
    task_id: str,
    tools: list[dict[str, Any]],
    tool_endpoints: list[dict[str, Any]],
    port_offset: int = 0,
    bridge_script_path: str | None = None,
) -> dict[str, Any]:
    """Build an MCPServerSpec-compatible dict for the claw-eval tool bridge.

    If bridge_script_path is not provided, a temp file will be created.

    Args:
        task_id: Task identifier for naming
        tools: ToolSpec list from task.yaml
        tool_endpoints: ToolEndpoint list from task.yaml
        port_offset: Port offset for parallel execution
        bridge_script_path: Path to save the bridge script

    Returns:
        Dict compatible with MCPServerSpec for use with ctx.load_mcp_tools()
    """
    import tempfile

    script_content = generate_mcp_bridge_script(tools, tool_endpoints, port_offset)

    if bridge_script_path is None:
        fd, bridge_script_path = tempfile.mkstemp(
            prefix=f'claw_eval_mcp_{task_id}_',
            suffix='.py',
        )
        os.close(fd)

    with open(bridge_script_path, 'w') as f:
        f.write(script_content)

    return {
        'name': f'claw-eval-{task_id}',
        # Use a short, fixed prefix to avoid AWS Bedrock Converse API's
        # 64-character tool name limit.  Without this, MCPToolManager would
        # auto-generate "claw-eval-<task_id>_" which can easily exceed 64
        # chars for tasks with long IDs (e.g. T31zh_escalation_budget_triage).
        'prefix': 'ce_',
        'client': {
            'transport': 'stdio',
            'command': sys.executable,
            'args': [bridge_script_path],
            'env': {},
        },
    }


# ---------------------------------------------------------------------------
# Audit snapshot collection
# ---------------------------------------------------------------------------


def collect_audit_snapshots(
    services: list[dict[str, Any]],
    port_offset: int = 0,
) -> dict[str, Any]:
    """Collect audit data from all running mock services.

    Most claw-eval mock services expose a /audit endpoint that returns
    a log of all operations performed during the task execution.

    Args:
        services: List of ServiceDef dicts from task.yaml
        port_offset: Port offset for parallel execution

    Returns:
        Dict mapping service names to their audit data
    """
    import re

    import httpx

    audits: dict[str, Any] = {}

    for svc in services:
        name = svc['name']
        port = svc['port'] + port_offset

        # Construct audit URL: replace the path portion with /audit
        health_url = svc['health_check']
        if port_offset:
            health_url = re.sub(
                r'localhost:(\d+)',
                lambda m: f'localhost:{int(m.group(1)) + port_offset}',
                health_url,
            )

        # Most services have audit at /<service_name>/audit
        base_url = f'http://localhost:{port}'
        audit_url = f'{base_url}/{name}/audit'

        try:
            with httpx.Client(trust_env=False, timeout=5.0) as client:
                resp = client.get(audit_url)
                if resp.status_code < 400:
                    audits[name] = resp.json()
                else:
                    logger.debug(f"No audit endpoint for service '{name}' (status {resp.status_code})")
        except Exception as exc:
            logger.debug(f"Failed to collect audit for service '{name}': {exc}")

    return audits


# ---------------------------------------------------------------------------
# Sandbox MCP bridge (for T100-T104 container-based tasks)
# ---------------------------------------------------------------------------

# Sandbox tool definitions matching claw-eval's sandbox_tools.py
SANDBOX_TOOL_SPECS: list[dict[str, Any]] = [
    {
        'name': 'sandbox_shell_exec',
        'description': 'Execute a shell command inside the sandbox and return stdout/stderr.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The shell command to execute.',
                },
                'timeout_seconds': {
                    'type': 'integer',
                    'description': 'Max seconds before the command is killed (default: 30).',
                    'default': 30,
                },
            },
            'required': ['command'],
        },
    },
    {
        'name': 'sandbox_file_read',
        'description': 'Read the contents of a file at the given path.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Absolute or relative path to the file.',
                },
            },
            'required': ['path'],
        },
    },
    {
        'name': 'sandbox_file_write',
        'description': 'Write content to a file at the given path (creates parent directories if needed).',
        'input_schema': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Absolute or relative path to the file.',
                },
                'content': {
                    'type': 'string',
                    'description': 'The content to write to the file.',
                },
            },
            'required': ['path', 'content'],
        },
    },
]


def generate_sandbox_mcp_bridge_script(runtime_base_url: str) -> str:
    """Generate an MCP stdio server script for sandbox tools.

    The generated script bridges sandbox tool calls (sandbox_shell_exec,
    sandbox_file_read, sandbox_file_write) to the nexus runtime's HTTP API.

    The nexus runtime exposes executors via ``POST /api/v1/call_executor/{name}``
    with JSON request bodies.  This bridge translates the claw-eval sandbox
    tool semantics to the corresponding nexus executor calls:

    - ``sandbox_shell_exec`` → ``run_command`` executor
    - ``sandbox_file_read``  → ``download_file`` executor
    - ``sandbox_file_write`` → ``upload_file`` executor

    Args:
        runtime_base_url: Base URL of the nexus runtime HTTP server
                          (e.g. ``http://localhost:12345``)

    Returns:
        Python script content as a string
    """
    tools_repr = repr(json.dumps(SANDBOX_TOOL_SPECS, ensure_ascii=False))
    url_repr = repr(runtime_base_url)

    return (
        '#!/usr/bin/env python3\n'
        '"""Auto-generated MCP bridge for claw-eval sandbox tools → nexus runtime."""\n'
        '\n'
        'import json\n'
        'import sys\n'
        'import base64\n'
        '\n'
        'import httpx\n'
        'from mcp.server.lowlevel import Server\n'
        'from mcp.types import Tool, TextContent\n'
        'import mcp.server.stdio\n'
        'import asyncio\n'
        '\n'
        f'RUNTIME_BASE_URL = {url_repr}\n'
        f'TOOLS = json.loads({tools_repr})\n'
        '\n'
        '_http_client = httpx.Client(timeout=120.0, trust_env=False)\n'
        'server = Server("claw-eval-sandbox")\n'
        '\n'
        '\n'
        'def _call_executor(executor_name: str, payload: dict) -> dict:\n'
        '    """Call a nexus runtime executor via HTTP."""\n'
        '    url = f"{RUNTIME_BASE_URL}/api/v1/call_executor/{executor_name}"\n'
        '    resp = _http_client.post(url, json=payload, timeout=120.0)\n'
        '    resp.raise_for_status()\n'
        '    return resp.json()\n'
        '\n'
        '\n'
        'def _dispatch(tool_name: str, arguments: dict) -> tuple[str, bool]:\n'
        '    """Route sandbox tool calls to nexus runtime executors."""\n'
        '    try:\n'
        '        if tool_name == "sandbox_shell_exec":\n'
        '            command = arguments.get("command", "")\n'
        '            timeout = arguments.get("timeout_seconds", 30)\n'
        '            result = _call_executor("run_command", {\n'
        '                "command": command,\n'
        '                "cwd": "/workspace",\n'
        '                "timeout": float(timeout),\n'
        '            })\n'
        '            # Format like claw-eval sandbox server: {exit_code, stdout, stderr}\n'
        '            output = {\n'
        '                "exit_code": result.get("return_code", -1),\n'
        '                "stdout": result.get("stdout", ""),\n'
        '                "stderr": result.get("stderr", ""),\n'
        '            }\n'
        '            return json.dumps(output), False\n'
        '\n'
        '        elif tool_name == "sandbox_file_read":\n'
        '            path = arguments.get("path", "")\n'
        '            if not path.startswith("/"):\n'
        '                path = f"/workspace/{path}"\n'
        '            result = _call_executor("download_file", {\n'
        '                "files": [{"path": path, "encoding": "utf8"}],\n'
        '            })\n'
        '            files = result.get("files", [])\n'
        '            errors = result.get("errors")\n'
        '            if errors:\n'
        '                return json.dumps({"error": errors[0].get("message", "read failed")}), True\n'
        '            if files:\n'
        '                return json.dumps({"path": path, "content": files[0].get("content", "")}), False\n'
        '            return json.dumps({"error": "file not found"}), True\n'
        '\n'
        '        elif tool_name == "sandbox_file_write":\n'
        '            path = arguments.get("path", "")\n'
        '            content = arguments.get("content", "")\n'
        '            if not path.startswith("/"):\n'
        '                path = f"/workspace/{path}"\n'
        '            result = _call_executor("upload_file", {\n'
        '                "files": [{"path": path, "content": content, "encoding": "utf8"}],\n'
        '                "overwrite": True,\n'
        '            })\n'
        '            results = result.get("results", [])\n'
        '            if results and results[0].get("success"):\n'
        '                return json.dumps({"status": "ok", "path": path}), False\n'
        '            msg = results[0].get("message", "write failed") if results else "write failed"\n'
        '            return json.dumps({"error": msg}), True\n'
        '\n'
        '        else:\n'
        '            return json.dumps({"error": f"Unknown tool: {tool_name}"}), True\n'
        '\n'
        '    except Exception as exc:\n'
        '        return json.dumps({"error": str(exc)}), True\n'
        '\n'
        '\n'
        '_TOOL_SPECS = []\n'
        'for _tool_def in TOOLS:\n'
        '    _TOOL_SPECS.append(Tool(\n'
        '        name=_tool_def["name"],\n'
        '        description=_tool_def.get("description", ""),\n'
        '        inputSchema=_tool_def.get("input_schema", {"type": "object", "properties": {}}),\n'
        '    ))\n'
        '\n'
        '\n'
        '@server.list_tools()\n'
        'async def handle_list_tools():\n'
        '    return _TOOL_SPECS\n'
        '\n'
        '\n'
        '@server.call_tool()\n'
        'async def handle_call_tool(name: str, arguments: dict | None):\n'
        '    text, is_error = _dispatch(name, arguments or {})\n'
        '    return [TextContent(type="text", text=text)]\n'
        '\n'
        '\n'
        'async def main():\n'
        '    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):\n'
        '        await server.run(\n'
        '            read_stream,\n'
        '            write_stream,\n'
        '            server.create_initialization_options(),\n'
        '        )\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    asyncio.run(main())\n'
    )


def build_sandbox_mcp_server_spec(
    task_id: str,
    runtime_base_url: str,
    bridge_script_path: str | None = None,
) -> dict[str, Any]:
    """Build an MCPServerSpec for sandbox tools backed by nexus runtime.

    Args:
        task_id: Task identifier for naming
        runtime_base_url: Base URL of the nexus runtime (e.g. http://localhost:12345)
        bridge_script_path: Optional explicit path for the bridge script

    Returns:
        Dict compatible with MCPServerSpec for use with ctx.load_mcp_tools()
    """
    import tempfile

    script_content = generate_sandbox_mcp_bridge_script(runtime_base_url)

    if bridge_script_path is None:
        fd, bridge_script_path = tempfile.mkstemp(
            prefix=f'claw_eval_sandbox_mcp_{task_id}_',
            suffix='.py',
        )
        os.close(fd)

    with open(bridge_script_path, 'w') as f:
        f.write(script_content)

    return {
        'name': f'claw-eval-sandbox-{task_id}',
        # Use a short, fixed prefix to avoid:
        # 1. AWS Bedrock's 64-char tool name limit
        # 2. LLMs replacing hyphens with underscores in tool names, causing
        #    "claw-eval-sandbox-..." to become "claw-eval-sandbox_..." and
        #    failing tool resolution.
        'prefix': 'ce_',
        'client': {
            'transport': 'stdio',
            'command': sys.executable,
            'args': [bridge_script_path],
            'env': {},
        },
    }
