"""Isolated fork lifecycle; bidirectional RPC delegates setup runtime commands."""

import asyncio
import base64
import json
import os
import sys
import traceback

from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from nexus.extensions.tasks.os_world_v2.adapter import loader, support
from nexus.extensions.tasks.os_world_v2.adapter.transport import install_requests_transport, local_gitlab_origin


def safe_error_message(exc):
    message = str(exc)
    for key, value in os.environ.items():
        if value and any(part in key.upper() for part in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY')):
            message = message.replace(value, '[REDACTED]')
    # Task-generated accounts may be mentioned by requests/GitLab errors.
    import re

    message = re.sub(r'(?i)(private-token|authorization|password)([\'"\s:=]+)[^,\s\'"}]+', r'\1\2[REDACTED]', message)
    return message[:4000]


class WorkerRuntime:
    def __init__(self, protocol):
        self.protocol = protocol

    async def _call(self, method, **kwargs):
        self.protocol.write(json.dumps({'runtime_call': method, 'kwargs': kwargs}) + '\n')
        self.protocol.flush()
        result = json.loads(sys.stdin.readline())
        while method == 'pdf_grade' and result.get('model_call'):
            from desktop_env.evaluators.model_client import generate_text

            request = result['model_call']
            try:
                # aws_third is synchronous and owns its asyncio.run internally.
                # This RPC handler already runs inside asyncio.run(_call(...)).
                text = await asyncio.to_thread(generate_text, request['prompt'], **request['kwargs'])
                reply = {'model_result': {'ok': True, 'text': text}}
            except Exception as exc:
                reply = {'model_result': {'ok': False, 'error_type': type(exc).__name__}}
            self.protocol.write(json.dumps(reply) + '\n')
            self.protocol.flush()
            result = json.loads(sys.stdin.readline())
        if not result.get('ok'):
            raise RuntimeError('Runtime setup RPC failed')
        return SimpleNamespace(**result['result'])

    async def run_command(self, **kwargs):
        return await self._call('run_command', **kwargs)

    async def setup_upload_bytes(self, remote_path, content, *, filename='file'):
        return await self._call(
            'setup_upload_bytes',
            remote_path=remote_path,
            content=base64.b64encode(content).decode(),
            filename=filename,
        )

    async def setup_upload_file(self, local_path, remote_path):
        return await self.setup_upload_bytes(remote_path, Path(local_path).read_bytes(), filename=Path(local_path).name)


class Environment:
    def __init__(self, task, host, cache, password):
        from desktop_env.controllers.python import PythonController
        from desktop_env.controllers.setup import SetupController
        from desktop_env.user_simulator import create_user_simulator

        self.task_config = task
        self.task_id = task.id
        self.instruction = task.get('instruction', '')
        self.cache_dir = cache
        self.client_password = password
        self.vm_ip, self.server_port, self.chromium_port, self.vnc_port, self.vlc_port = host, 5000, 9222, 5910, 8080
        self.enable_proxy = self.current_use_proxy = False
        self.is_environment_used = False
        self.action_history = []
        self._step_no, self._traj_no = 0, -1
        self.controller = PythonController(host, 5000)
        self.setup_controller = SetupController(
            vm_ip=host,
            server_port=5000,
            chromium_port=9222,
            vlc_port=8080,
            cache_dir=cache,
            client_password=password,
        )
        self.user_simulator = create_user_simulator(task.get('user_simulator')) if task.get('user_simulator') else None
        if self.user_simulator:
            self.user_simulator.reset(instruction=self.instruction)

    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()


def serve():
    protocol = sys.stdout
    runtime = WorkerRuntime(protocol)
    state = {}
    with redirect_stdout(sys.stderr):
        for raw in sys.stdin:
            boundary = None
            try:
                request = json.loads(raw)
                boundary = request.get('stderr_boundary')
                if boundary:
                    print(f'NEXUS_WORKER_STDERR:{boundary}:start', file=sys.stderr, flush=True)
                op, args = request['op'], request.get('args', {})
                if op == 'load':
                    root = Path(args['root'])
                    os.environ.update(args['environment'])
                    os.environ['OSWORLD_AUX_DIAGNOSTIC_DIR'] = args['cache']
                    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
                    os.chdir(root)
                    compatibility = [None]
                    failures = install_requests_transport(
                        args['host'],
                        args['sites'],
                        args['suffix'],
                        compatibility,
                        gitlab_origin=local_gitlab_origin(args['environment']),
                        gitlab_pages_origin=local_gitlab_origin(args['environment'], 'GITLAB_PAGES_URL'),
                        task_id=args['task_id'],
                    )
                    support.install(failures)
                    task = loader.load(root, args['task_id'])
                    if args.get('isolated_cad'):
                        from nexus.extensions.tasks.os_world_v2.adapter.cad_worker import install

                        install(task, runtime)
                    if args.get('isolated_pdf'):
                        from nexus.extensions.tasks.os_world_v2.adapter.pdf_worker import install

                        install(task, runtime)
                    from nexus.extensions.tasks.os_world_v2.adapter.aws_third import register

                    register()
                    from evaluation_examples.setup_support import create_setup_plan

                    plan = create_setup_plan(task)
                    env = Environment(task, args['host'], args['cache'], args['password'])
                    phases = task.get_phases() if callable(getattr(task, 'get_phases', None)) else []
                    state.update(
                        task=task, plan=plan, env=env, phases=phases, compatibility=compatibility, failures=failures
                    )
                    result = {
                        'instruction': task.get('instruction', ''),
                        'related_apps': task.get('related_apps', []),
                        'user_simulator': bool(env.user_simulator),
                        'requires_action_history': plan.requires_executed_action_history,
                        'task_owned_cdp': plan.task_owned_cdp,
                        'task_current_date': task.get('task_current_date'),
                        'phases': [
                            {k: v for k, v in phase.items() if k not in {'setup', 'evaluate'}} for phase in phases
                        ],
                    }
                elif op == 'prepare':
                    plan, env = state['plan'], state['env']
                    preparation_runtime = runtime
                    if args.get('dependency_profile'):
                        from nexus.extensions.tasks.os_world_v2.adapter.task033 import install_profile

                        if state['task'].id != '033':
                            raise ValueError('Dependency profile is only valid for Task033')
                        preparation_runtime = install_profile(args['dependency_profile'], runtime)
                    asyncio.run(
                        plan.prepare(
                            preparation_runtime,
                            run_setup=args['run_setup'],
                            client_password=env.client_password,
                            dependency_root=args.get('dependency_root') or None,
                        )
                    )
                    state['compatibility'][0] = plan.compatibility(image_owns_cdp_proxy=args['image_cdp'])
                    asyncio.run(plan.prepare_environment(runtime, cache_dir=env.cache_dir, run_setup=args['run_setup']))
                    result = None
                elif op == 'setup':
                    plan, env = state['plan'], state['env']
                    index = args.get('phase', 0)
                    if index:
                        env._step_no = 0
                        env._traj_no += 1
                        env.action_history.clear()
                        state['phases'][index]['setup'](env.setup_controller, use_proxy=False)
                    else:
                        state['task'].setup(env.setup_controller, use_proxy=False)
                    env.is_environment_used = True
                    env.instruction = state['task'].get('instruction', '')
                    asyncio.run(plan.validate_service(runtime))
                    if not state['phases']:
                        asyncio.run(plan.validate_standard_setup(runtime))
                    result = {'instruction': env.instruction}
                elif op == 'evaluate':
                    task, env = state['task'], state['env']
                    env.instruction = args['instruction']
                    env.action_history = args['history']
                    state['failures'].clear()
                    preflight = getattr(task, 'evaluator_preflight', None)
                    if callable(preflight):
                        preflight(env)
                    scope = getattr(task, 'evaluator_scope', None)
                    index = args.get('phase')
                    evaluator = task.evaluate if index is None else state['phases'][index]['evaluate']
                    from nexus.extensions.tasks.os_world_v2.adapter.evaluation_diagnostics import record_evaluation
                    from nexus.extensions.tasks.os_world_v2.adapter.task074_capture import capture_scope

                    with record_evaluation(env.task_id, env.cache_dir) as diagnostic:
                        with scope(env, 'upstream-v2.1') if callable(scope) else nullcontext():
                            with capture_scope(task, env):
                                result = evaluator(env)
                        diagnostic['score'] = result.get('score') if isinstance(result, dict) else result
                    if state['failures']:
                        raise support.EvaluatorNetworkError('Required website request failed during evaluation')
                    validate = getattr(task, 'validate_evaluator_result', None)
                    if callable(validate):
                        validate(result)
                elif op == 'respond':
                    result = state['env'].user_simulator.respond(args['question'])
                elif op == 'teardown':
                    function = getattr(state['task'], 'teardown', None)
                    if callable(function):
                        function(state['env'])
                    result = None
                else:
                    raise ValueError('Unknown worker operation')
                response = {'ok': True, 'result': result}
                encoded = json.dumps(response, allow_nan=False)
            except BaseException as exc:
                encoded = json.dumps({
                    'ok': False,
                    'error_type': type(exc).__name__,
                    'message': safe_error_message(exc),
                    'missing_module': exc.name if isinstance(exc, ModuleNotFoundError) else None,
                    'frames': [
                        f'{Path(frame.filename).name}:{frame.lineno}:{frame.name}'
                        for frame in traceback.extract_tb(exc.__traceback__)
                    ],
                    'attribute': getattr(exc, 'name', None) if isinstance(exc, AttributeError) else None,
                })
            if boundary:
                print(f'NEXUS_WORKER_STDERR:{boundary}:end', file=sys.stderr, flush=True)
            protocol.write(encoded + '\n')
            protocol.flush()


if __name__ == '__main__':
    serve()
