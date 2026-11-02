"""Taiji Analysis task — orchestrates expiry check, agent run, and evaluation."""

import json
import logging
import os
import re

from pathlib import Path

import httpx

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry
from nexus.extensions.tasks.skills.utils import read_local_directory_to_file_items, read_local_file_to_file_item
from nexus.extensions.tasks.taiji_analysis.evaluator import evaluate_root_cause


logger = logging.getLogger(__name__)

TAIJI_MCP_URL = os.getenv('TAIJI_MCP_URL', 'http://21.46.122.32:80')
TAIJI_API_TOKEN = os.getenv('TAIJI_API_TOKEN', 'h3kcqqyYy5Uosyuyx1E5mw')


class TaijiTaskInput(TaskRunInput, WithAgentRunner):
    """Input for a single Taiji analysis episode."""

    question: str
    answer: str
    inst_id: str
    task_id: str
    ws_id: str
    url: str


class TaijiTaskOutput(TaskRunOutput, WithScore):
    """Output of a single Taiji analysis episode."""

    prediction: str = ''
    reference: str = ''
    status: str = ''  # completed | expired | error
    metadata: dict | None = None


async def check_instance_expired(inst_id: str, task_id: str, ws_id: str) -> bool:
    """Check whether the Taiji instance still exists via MCP Server.

    Protocol: POST /mcp, JSON-RPC 2.0, token via HTTP header 'taiji_api_token'.
    Returns True if the instance is expired (no longer exists).
    """
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        'taiji_api_token': TAIJI_API_TOKEN,
    }
    payload = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {
            'name': 'query_task_instance_list',
            'arguments': {'task_id': task_id, 'ws_id': ws_id},
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f'{TAIJI_MCP_URL}/mcp', headers=headers, json=payload)
            if resp.status_code != 200:
                return False

            data = resp.json()
            content = data.get('result', {}).get('content', [])
            for item in content:
                if item.get('type') == 'text':
                    instances_data = json.loads(item['text'])
                    instances = (
                        instances_data if isinstance(instances_data, list) else instances_data.get('instances', [])
                    )
                    return not any(
                        inst.get('inst_id') == inst_id or inst.get('instId') == inst_id for inst in instances
                    )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        pass

    return False


def _extract_skills_used(agent_output) -> list[str]:
    """Extract skill names from agent tool call history."""
    skills = set()
    for tool_call in getattr(agent_output, 'tool_calls', []):
        command = getattr(tool_call, 'command', '') or ''
        match = re.search(r'/\.agent/skills/([^/]+)/', command)
        if match:
            skills.add(match.group(1))
    return sorted(skills)


_THIS_DIR = Path(__file__).resolve().parent
AGENT_SKILLS_DIR = Path(os.getenv('TAIJI_AGENT_SKILLS_DIR', str(_THIS_DIR / '.agent')))
CLAUDE_MD_PATH = Path(os.getenv('TAIJI_CLAUDE_MD_PATH', str(_THIS_DIR / 'CLAUDE.md')))


async def _setup_container(runtime) -> None:
    """Upload workspace files and install dependencies in the container."""
    # Upload .agent/ skills and CLAUDE.md
    files = []
    if AGENT_SKILLS_DIR.exists():
        files.extend(read_local_directory_to_file_items(AGENT_SKILLS_DIR, '/data/workspace/.agent'))
    if CLAUDE_MD_PATH.exists():
        files.append(read_local_file_to_file_item(CLAUDE_MD_PATH, '/data/workspace/CLAUDE.md'))
    if files:
        resp = await runtime.upload_file(files, overwrite=True)
        logger.debug('Uploaded %d workspace files to container', len(resp.results))

    # Install mcporter-internal
    result = await runtime.run_command(
        'export PATH="/nix/runtime/nix-env/bin:/usr/local/bin:/usr/bin:/bin:$PATH" && '
        'npm install -g --prefix /usr/local @tencent/mcporter-internal --registry=https://mirrors.tencent.com/npm/',
        timeout=120,
    )
    if result.return_code != 0:
        logger.warning('npm install mcporter-internal failed (rc=%d): %s', result.return_code, result.stderr)

    # Write mcporter.json MCP config
    mcporter_json = json.dumps({
        'mcpServers': {
            'iwiki': {
                'baseUrl': 'https://prod.mcp.it.woa.com/app_iwiki_mcp/mcp3',
                'headers': {'Authorization': f'Bearer {os.getenv("MCPORTER_IWIKI_TOKEN", "")}'},
            },
            'gongfeng': {
                'baseUrl': 'https://mcpgw.knot.woa.com/gongfeng',
                'headers': {'Authorization': f'Bearer {os.getenv("MCPORTER_GONGFENG_TOKEN", "")}'},
            },
            'tapd': {
                'baseUrl': 'http://mcp-idc.tapd.woa.com/mcp',
                'headers': {'Authorization': f'Bearer {os.getenv("MCPORTER_TAPD_TOKEN", "")}'},
            },
        }
    })
    await runtime.run_command(
        f"mkdir -p /data/workspace/config && cat > /data/workspace/config/mcporter.json <<'EOFJ'\n{mcporter_json}\nEOFJ",
        timeout=10,
    )


@task_registry.register('taiji_analysis')
async def taiji_analysis_task(task_input: TaijiTaskInput) -> TaijiTaskOutput:
    """Run a single Taiji analysis episode: expiry check → agent → evaluate."""
    # Step 1: Check if instance is expired
    if await check_instance_expired(task_input.inst_id, task_input.task_id, task_input.ws_id):
        logger.info('Instance %s is expired, skipping.', task_input.inst_id)
        return TaijiTaskOutput(score=0.0, prediction='', status='expired')

    # Step 2: Setup container (upload files + install deps)
    ctx = require_context()
    runtime = await ctx.get_runtime()
    await _setup_container(runtime)

    # Step 3: Run agent
    if not task_input.question:
        logger.error('Empty question for inst_id=%s, skipping', task_input.inst_id)
        return TaijiTaskOutput(score=0.0, prediction='', status='error')
    agent_output = await task_input.run_agent(user_prompt=task_input.question)
    prediction = getattr(agent_output, 'output', '') or ''

    # Step 4: Evaluate (skip if TAIJI_SKIP_JUDGE=1, useful for GT generation runs)
    if os.getenv('TAIJI_SKIP_JUDGE', '0') == '1':
        return TaijiTaskOutput(
            score=-1.0,
            prediction=prediction,
            reference=task_input.answer,
            status='completed',
            metadata={
                'skills_used': _extract_skills_used(agent_output),
                'latency_seconds': getattr(agent_output, 'elapsed_seconds', None),
                'turns_used': getattr(agent_output, 'turns_used', None),
                'judge_reasoning': 'skipped',
            },
        )

    judge_result = await evaluate_root_cause(
        prediction=prediction,
        reference=task_input.answer,
        question=task_input.question,
    )

    return TaijiTaskOutput(
        score=judge_result.score,
        prediction=prediction,
        reference=task_input.answer,
        status='completed',
        metadata={
            'skills_used': _extract_skills_used(agent_output),
            'latency_seconds': getattr(agent_output, 'elapsed_seconds', None),
            'turns_used': getattr(agent_output, 'turns_used', None),
            'judge_reasoning': judge_result.reasoning,
        },
    )
