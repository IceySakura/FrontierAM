"""Six-mode V2.1 config assembly; no model-specific evaluator or runtime fork."""

import json
import os

from nexus import BatchItem, RunInput
from nexus.extensions.tasks.os_world_v2.adapter.source import COMMIT, RELEASE, task_id
from nexus.extensions.tasks.os_world_v2.generators import policy as shared
from nexus.extensions.tasks.os_world_v2.generators.policy import env_bool, env_float, env_int


def selected_release() -> str:
    value = os.getenv('OSWORLD_BENCHMARK_RELEASE', RELEASE).strip()
    if value != RELEASE:
        raise ValueError(f'Independent os_world_v2 supports {RELEASE}, not {value!r}')
    return value


def tasks(selected: str | None = None) -> list[str]:
    from nexus.extensions.tasks.os_world_v2.data.index import dataset_index

    ids = list(dataset_index())
    requested = (
        {task_id(selected)}
        if selected
        else {task_id(value.strip()) for value in os.getenv('OSWORLD_TASK_IDS', '').split(',') if value.strip()}
    )
    if requested - set(ids):
        raise ValueError('Selected V2.1 task not found in pinned manifest')
    result = [value for value in ids if not requested or value in requested]
    if selected is None:
        skip, limit = env_int('OSWORLD_SKIP', 0) or 0, env_int('OSWORLD_LIMIT', None)
        if skip < 0 or (limit is not None and limit < 0):
            raise ValueError('OSWorld skip/limit must be non-negative')
        result = result[skip:][:limit]
    if not result:
        raise ValueError('No V2.1 tasks selected')
    return result


def build_item(selected: str, choice) -> BatchItem:
    from nexus.extensions.tasks.os_world_v2.generators.policy import agent_config, model_client, required

    shared.validate_tool_id(required('AGS_OSWORLD_TOOL_ID'))
    from nexus.extensions.tasks.os_world_v2.data.index import dataset_index

    record = dataset_index()[selected]
    agent_name, agent = agent_config(choice)
    client = model_client(selected, choice)
    required('AGS_SECRET_ID')
    required('AGS_SECRET_KEY')
    if shared._ags_provision_mode() != 'control_plane':
        raise ValueError('V2.1 adapter currently requires an explicit AGS tool image')
    runtime = shared._runtime_configs()
    runtime['main'].update(provision_mode='control_plane', timeout=env_int('OSWORLD_SANDBOX_TIMEOUT', 10800))
    provider = shared._runtime_provider_config()
    provider['osworld_cdp_proxy_mode'] = os.getenv('OSWORLD_AGS_CDP_PROXY_MODE', 'image')
    task_config = {
        'benchmark_release': RELEASE,
        'task_id': selected,
        'domain': 'tasks',
        'skip_gui_start': True,
        'command_as_user': False,
        'client_password': os.getenv('OSWORLD_CLIENT_PASSWORD', 'osworld-public-evaluation'),
        'action_space': 'pyautogui',
        'include_initial_screenshot': choice.harness == 'bare',
        'setup_wait_time': env_float('OSWORLD_SETUP_WAIT_TIME', 5),
        'step_pause': env_float('OSWORLD_STEP_PAUSE', 1),
        'artifact_taihu_pat': '${oc.env:ARTIFACT_TAIHU_PAT}' if os.getenv('ARTIFACT_TAIHU_PAT') else None,
        'artifact_service_url': os.getenv('ARTIFACT_SERVICE_URL') or None,
        'bundle_sha256': record['metadata']['bundle_sha256'],
        'wedata_key': '${oc.env:WEDATA_KEY}',
        'v21': {
            'source_commit': COMMIT,
            'website_host_suffix': os.getenv('WEBSITE_HOST_SUFFIX', 'localhost'),
            'official_timeout': env_float('OSWORLD_OFFICIAL_TIMEOUT', 900),
            'setup_only': env_bool('OSWORLD_SETUP_ONLY', False),
            'website_ready_timeout': env_float('OSWORLD_WEBSITE_READY_TIMEOUT', 900),
            'gitlab_initial_wait': env_float('OSWORLD_GITLAB_INITIAL_WAIT', 600),
            'gitlab_ready_timeout': env_float('OSWORLD_GITLAB_READY_TIMEOUT', 1800),
            'cad_grader': {
                'backend': os.getenv('OSWORLD_CAD_GRADER_BACKEND', 'ags'),
                'tool_id': os.getenv('OSWORLD_CAD_GRADER_TOOL_ID') or None,
                'install_dependencies': env_bool('OSWORLD_CAD_GRADER_INSTALL_DEPENDENCIES', True),
                'timeout': env_int('OSWORLD_CAD_GRADER_TIMEOUT', 900),
            },
            'pdf_grader': {
                'backend': os.getenv('OSWORLD_PDF_GRADER_BACKEND', 'ags'),
                'tool_id': os.getenv('OSWORLD_PDF_GRADER_TOOL_ID') or None,
                'install_dependencies': env_bool('OSWORLD_PDF_GRADER_INSTALL_DEPENDENCIES', True),
                'timeout': env_int('OSWORLD_PDF_GRADER_TIMEOUT', 1800),
            },
            'environment': json.loads(os.getenv('OSWORLD_V21_ENV_JSON', '{}')),
        },
    }
    imports = [
        'nexus.extensions.tasks.os_world_v2.task',
        'nexus.extensions.agents.codex.agent'
        if choice.harness == 'codex'
        else 'nexus.extensions.agents.claude_code.agent'
        if choice.harness == 'cc'
        else f'nexus.extensions.agents.cua.{agent_name}.agent',
        f'nexus.extensions.openai_clients.{client["name"]}',
    ]
    config = RunInput.model_validate({
        'imports': imports,
        'task_name': 'os_world_v2',
        'task_config': task_config,
        'runner': {'hooks': {'output_contract': {'name': 'osworld_v2_output_contract', 'args': {}}}},
        'task_files': record['files'],
        'runtime_provider_name': 'ags',
        'runtime_provider_config': provider,
        'runtime_configs': runtime,
        'agent_name': agent_name,
        'agent_config': agent,
        'openai_clients': {'main': client},
        'processors': {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}},
    })
    return BatchItem(
        config=config,
        run_id=f'v21__{selected}',
        metadata={
            'benchmark_release': RELEASE,
            'source_commit': COMMIT,
            'task_id': selected,
            'model_family': choice.model_family,
            'harness': choice.harness,
        },
    )
