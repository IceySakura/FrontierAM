"""Real installed CLI + Nexus proxy + protocol; only model and package installation are substituted."""

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(os.environ['HYNESS_REFERENCE_WORKSPACE']).resolve()
OUT = Path(os.environ['HYNESS_EVIDENCE_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'nexus'))
spec = importlib.util.spec_from_file_location(
    'fixture', ROOT / 'nexus/tests/integration/anthropic_proxy/test_hyness_apps.py'
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
PREFIX = Path(os.environ['HYNESS_TEST_PREFIX']).resolve()
NODE = os.environ.get('HYNESS_TEST_NODE', 'node')


class Bridge:
    def __init__(self, actions):
        self.actions, self.requests = actions, []

    async def request(self, params, timeout):
        self.requests.append(deepcopy(params))
        n = len(self.requests)
        message = {'role': 'assistant', 'content': 'FINISHED'}
        if n <= len(self.actions):
            name, args = self.actions[n - 1]
            message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {'id': f'call-{n}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
                ],
            }
        return {
            'id': f'reply-{n}',
            'object': 'chat.completion',
            'created': 1,
            'model': params['model'],
            'choices': [
                {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if 'tool_calls' in message else 'stop'}
            ],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
        }


def text(content):
    return content if isinstance(content, str) else '\n'.join(x.get('text', '') for x in content if isinstance(x, dict))


async def check(app_id, scenario):
    label = f'{app_id}-{scenario}'
    with (
        tempfile.TemporaryDirectory(prefix=f'hyness-pkg-{app_id}-') as temporary,
        pytest.MonkeyPatch.context() as patch,
    ):
        work = Path(temporary) / 'work'
        work.mkdir()
        for args in [
            ['init', '-q'],
            ['config', 'user.name', 'Fixture'],
            ['config', 'user.email', 'fixture@example.test'],
        ]:
            subprocess.run(['git', *args], cwd=work, capture_output=True, check=True)
        (work / 'value.txt').write_text('before\n')
        subprocess.run(['git', 'add', 'value.txt'], cwd=work, check=True)
        subprocess.run(['git', 'commit', '-qm', 'base'], cwd=work, check=True)
        for key in list(os.environ):
            if key.startswith(('HYNESS_', 'ANTHROPIC_', 'CLAUDE_')) or key.lower() in [
                'http_proxy',
                'https_proxy',
                'all_proxy',
            ]:
                patch.delenv(key, raising=False)
        patch.setenv('HOME', str(Path(temporary) / 'home'))
        patch.setenv('NO_PROXY', '*')
        patch.setenv('no_proxy', '*')
        engine = PREFIX / 'node_modules/@tencent/hyness'
        version = json.loads((engine / 'package.json').read_text())['version']
        package = PREFIX / f'node_modules/@tencent/hyness-{app_id}'
        app_version = json.loads((package / 'package.json').read_text())['version']

        async def install(_transport, packages, _registry):
            assert packages == [f'@tencent/hyness@{version}', f'@tencent/hyness-{app_id}@{app_version}']
            return {
                'prefix': str(PREFIX),
                'entry': str(engine / 'apps/cli/dist/hyness.js'),
                'node': NODE,
                'version': version,
            }

        patch.setattr(fixture.runtime_setup, 'install_hyness', install)
        patch.setattr(fixture.context, 'require_runtime_control', lambda: fixture.context.RuntimeControl(sio=object()))
        actions = []
        if scenario == 'submit':
            actions = (
                [
                    ('bash', {'command': "printf 'after\\n' > value.txt"}),
                    ('bash', {'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}),
                ]
                if app_id == 'mini-swe-agent'
                else [
                    ('str_replace_editor', {'command': 'view', 'path': 'value.txt'}),
                    (
                        'str_replace_editor',
                        {'command': 'str_replace', 'path': 'value.txt', 'old_str': 'before', 'new_str': 'after'},
                    ),
                    ('submit', {}),
                    ('submit', {}),
                ]
            )
        elif scenario == 'native':
            actions = [
                ('Read', {'file_path': str(work / 'value.txt')}),
                ('Edit', {'file_path': str(work / 'value.txt'), 'old_string': 'before', 'new_string': 'after'}),
            ]
        elif scenario in {'permission-denied', 'long-command'}:
            # The pinned native CLI refuses recursive deletion. Verify that a
            # completed refusal does not poison subsequent independent tools.
            (work / 'node_modules').mkdir()
            (work / 'node_modules/sentinel').write_text('must remain\n')
            actions = [
                ('TaskCreate', {'subject': 'Keep this task', 'description': 'State must survive a native refusal.'}),
                ('Read', {'file_path': str(work / 'value.txt')}),
                ('Bash', {'command': 'rm -rf node_modules'}),
                ('Edit', {'file_path': str(work / 'value.txt'), 'old_string': 'before', 'new_string': 'after'}),
                ('TaskList', {}),
            ]
            if scenario == 'long-command':
                actions[2] = (
                    'Bash',
                    {
                        'command': 'node -e \'setTimeout(() => {require("fs").appendFileSync("long-command-receipt", "done\\n"); console.log("finished");}, 305000)\'',
                        'timeout': 600000,
                        'description': 'Complete a bounded native timing fixture',
                    },
                )
        options = fixture.process_options()
        if app_id == 'workbuddy':
            options['ENABLE_TOOL_SEARCH'] = 'true'
        config = fixture.HynessInput(
            user_prompt='Change value.txt before to after, then review and submit.',
            working_dir=str(work),
            packages=[f'@tencent/hyness@{version}', f'@tencent/hyness-{app_id}@{app_version}'],
            app_config=f'@tencent/hyness-{app_id}/apps/{app_id}/app.yaml',
            harness_id=app_id,
            preserve_tool_schema=True,
            session_persistence=app_id == 'workbuddy',
            max_turns=8,
            timeout=480 if scenario == 'long-command' else 240,
            extra_env=options,
        )
        bridge = Bridge(actions)
        result = await fixture.launch(config, bridge, patch)
        (OUT / f'{label}.requests.json').write_text(json.dumps(bridge.requests, ensure_ascii=False, indent=2))
        (OUT / f'{label}.result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        summary = {
            key: result.get(key)
            for key in ['success', 'exit_code', 'process_exit_code', 'execution_termination', 'failure_stage']
        }
        summary.update(
            app_id=app_id,
            scenario=scenario,
            engine=version,
            app=app_version,
            calls=len(bridge.requests),
            file=(work / 'value.txt').read_text(),
        )
        if scenario in {'permission-denied', 'long-command'}:
            tool_results = {
                message['tool_call_id']: text(message['content'])
                for request in bridge.requests
                for message in request['messages']
                if message.get('role') == 'tool' and message.get('tool_call_id')
            }
            output_events = []
            for line in result.get('process_output', '').splitlines():
                try:
                    output_events.append(json.loads(line))
                except ValueError:
                    continue
            blocks = [
                block
                for event in output_events
                for block in event.get('message', {}).get('content', [])
                if isinstance(block, dict) and block.get('type') == 'tool_result'
            ]
            summary.update(
                denied_operation_unchanged=(work / 'node_modules/sentinel').read_text() == 'must remain\n',
                permission_denied_visible='WORKBUDDY_PERMISSION_DENIED' in tool_results.get('call-3', ''),
                permission_denied_is_error=any(
                    block.get('tool_use_id') == 'call-3' and block.get('is_error') is True for block in blocks
                ),
                retained_task='Keep this task' in tool_results.get('call-5', ''),
                unknown_outcomes=sum(
                    'outcome is unknown' in content or 'NATIVE_OUTCOME_UNKNOWN' in content
                    for content in tool_results.values()
                ),
            )
            if scenario == 'long-command':
                summary.update(
                    long_command_completed='finished' in tool_results.get('call-3', ''),
                    long_command_executed_once=(work / 'long-command-receipt').read_text() == 'done\n'
                    if (work / 'long-command-receipt').exists()
                    else False,
                )
        if bridge.requests:
            first = bridge.requests[0]
            system = '\n\n'.join(text(m['content']) for m in first['messages'] if m['role'] == 'system')
            user = '\n\n'.join(text(m['content']) for m in first['messages'] if m['role'] == 'user')
            summary.update(
                system_chars=len(system),
                user_chars=len(user),
                tools=[t['function']['name'] for t in first.get('tools', [])],
            )
            expected_system = (ROOT / f'hyness/app-definitions/{app_id}/prompts/system.md').read_text().strip()
            expected_user = (
                (ROOT / f'hyness/app-definitions/{app_id}/prompts/task.md')
                .read_text()
                .replace('{{cwd}}', str(work))
                .replace('{{task}}', str(config.user_prompt))
            )
            summary.update(
                system_equal=system == expected_system,
                system_all_equal=all(
                    '\n\n'.join(text(m['content']) for m in request['messages'] if m['role'] == 'system')
                    == expected_system
                    for request in bridge.requests
                ),
                system_sha256=hashlib.sha256(system.encode()).hexdigest(),
                user_equal=user == expected_user,
            )
        (OUT / f'{label}.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if not result.get('execution_termination'):
            print(result.get('stderr', '')[-3000:], flush=True)
        assert bridge.requests, 'No model request'
        assert summary['system_equal'], summary
        assert summary['system_all_equal'], summary
        assert summary['user_equal'], summary
        if scenario == 'permission-denied':
            assert summary['denied_operation_unchanged'], summary
            assert summary['permission_denied_visible'], summary
            assert summary['permission_denied_is_error'], summary
            assert summary['retained_task'], summary
            assert summary['unknown_outcomes'] == 0, summary
        if scenario == 'long-command':
            assert summary['long_command_completed'], summary
            assert summary['long_command_executed_once'], summary
            assert summary['retained_task'], summary
            assert summary['unknown_outcomes'] == 0, summary
        if scenario == 'no-tool':
            assert result['execution_termination']['reason'] == 'tool_error', summary
            assert result['exit_code'] == 65, summary
            assert len(bridge.requests) == 1
        else:
            assert result['success'], summary
            assert summary['file'] == 'after\n'
            assert result['execution_termination']['reason'] == (
                'completed' if app_id == 'workbuddy' else 'hook_stopped'
            ), summary
            assert len(bridge.requests) == len(actions) + (app_id == 'workbuddy'), summary
        return summary


async def main():
    ids = sys.argv[1:] or ['mini-swe-agent', 'swe-agent', 'workbuddy']
    results = []
    for app_id in ids:
        scenarios = ['native', 'permission-denied'] if app_id == 'workbuddy' else ['submit', 'no-tool']
        if app_id == 'workbuddy' and os.environ.get('HYNESS_LONG_TOOL_CHECK') == '1':
            scenarios.append('long-command')
        for scenario in scenarios:
            try:
                results.append(await check(app_id, scenario))
            except Exception as error:
                print('FAILED', app_id, scenario, str(error), flush=True)
                results.append({'app_id': app_id, 'scenario': scenario, 'error': str(error)})
    (OUT / 'package-checks.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if any('error' in result for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
