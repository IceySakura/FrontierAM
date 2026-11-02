"""Worker-local compatibility for the fork's declared Nexus transport imports.

These are transport/error contracts only. The fork owns all task repairs.
No retired V2 task-specific dispatch is installed.
"""

import json
import sys

from contextlib import contextmanager
from types import ModuleType


class EvaluatorNetworkError(ConnectionError):
    pass


class EvaluatorDependencyError(RuntimeError):
    pass


def execute_guest_json(env, source, *, timeout=90, label='evaluator'):
    result = env.controller.run_python_script(source, timeout=timeout)
    if not isinstance(result, dict) or result.get('status') != 'success':
        raise EvaluatorNetworkError(f'{label}: guest command failed')
    try:
        return json.loads(result.get('output') or '')
    except ValueError as exc:
        raise EvaluatorNetworkError(f'{label}: guest response is not JSON') from exc


def validate_browser_websocket(env, *, query=execute_guest_json):
    value = query(
        env,
        'import json,websocket; print(json.dumps({"ok": callable(websocket.create_connection)}))',
        timeout=30,
        label='browser websocket dependency',
    )
    if value.get('ok') is not True:
        raise EvaluatorDependencyError('Missing guest websocket-client')


def install(failures):
    parent_name = 'nexus.extensions.tasks.os_world_v2'
    parent = sys.modules[parent_name]

    @contextmanager
    def audit(*args):
        failures.clear()
        yield
        if failures:
            raise EvaluatorNetworkError('Unresolved required evaluator transport failure')

    async def run_setup_command(runtime, command, **kwargs):
        return await runtime.run_command(command=command, **kwargs)

    exports = {
        'evaluator_network': {
            'EvaluatorNetworkError': EvaluatorNetworkError,
            'EvaluatorDependencyError': EvaluatorDependencyError,
            'evaluator_network_audit': audit,
            'current_evaluator_audit': lambda: failures,
        },
        'evaluator_transport': {
            'execute_guest_json': execute_guest_json,
            'validate_browser_websocket': validate_browser_websocket,
        },
        'setup_transport': {'guest_pip_index_option': lambda: ''},
        'setup_recovery': {'run_setup_command': run_setup_command},
    }
    for name, values in exports.items():
        module = ModuleType(f'{parent_name}.{name}')
        module.__dict__.update(values)
        sys.modules[module.__name__] = module
        setattr(parent, name, module)
