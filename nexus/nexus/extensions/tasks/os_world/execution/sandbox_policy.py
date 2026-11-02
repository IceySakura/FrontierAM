"""Versioned V1 sandbox allocation; task/evaluator inputs are not changed."""

from typing import Any

from nexus.extensions.tasks.os_world.data.task_source import OSWorldBatchTask


POLICY_VERSION = 'v1-instructor-gimp-large-20260908'
INSTRUCTOR_TASK_ID = '69acbb55-d945-4927-a87b-8480e1a5bb7e'
GIMP_UPSCALE_TASK_ID = '8ea73f6f-9689-42ad-8c60-195bbf06a7ba'
LARGE_TOOL_ID = 'sdt-ruk8welf'
LARGE_PROVIDER = 'osworld_large'
LARGE_PROVIDER_MODULE = 'nexus.extensions.tasks.os_world.execution.large_sandbox'
LARGE_TASK_PROFILES = {
    ('multi_apps', INSTRUCTOR_TASK_ID): 'instructor-large',
    ('gimp', GIMP_UPSCALE_TASK_ID): 'gimp-upscale-large',
}


def uses_large_sandbox(task: OSWorldBatchTask) -> bool:
    return (task.domain, task.task_id) in LARGE_TASK_PROFILES


def apply_sandbox_policy(task: OSWorldBatchTask, config: dict[str, Any]) -> dict[str, Any]:
    """Route approved tasks only, identically for Claude and Hunyuan."""
    if uses_large_sandbox(task):
        config['runtime_configs']['main']['tool_id'] = LARGE_TOOL_ID
        config['runtime_configs']['main']['tool_name'] = 'osworld-v2-8c-32g'
        config['runtime_provider_name'] = LARGE_PROVIDER
        harness = {
            'codex_native_bridge': 'codex',
            'claude_code_native_bridge': 'cc',
            'cua_gpt_agent': 'bare',
            'cua_anthropic_agent': 'bare',
        }[config['agent_name']]
        config['runtime_provider_config'] = {**config['runtime_provider_config'], 'cli_harness': harness}
        if LARGE_PROVIDER_MODULE not in config['imports']:
            config['imports'].append(LARGE_PROVIDER_MODULE)
    return config


def sandbox_metadata(task: OSWorldBatchTask, config: dict[str, Any]) -> dict[str, Any]:
    return {
        'sandbox_policy_version': POLICY_VERSION,
        'sandbox_profile': LARGE_TASK_PROFILES.get((task.domain, task.task_id), 'v1-default'),
        'sandbox_tool_id': config['runtime_configs']['main']['tool_id'],
        'sandbox_changed_from_historical': uses_large_sandbox(task),
    }
