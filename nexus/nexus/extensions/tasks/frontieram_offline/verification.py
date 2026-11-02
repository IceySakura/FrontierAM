"""Local fake HTTP + real Nexus/Codex/Docker integration, with zero model calls."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import threading
import time

from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from nexus.runner.single import RunInput, run

from .runtime import FrontierAMOfflineProvider


async def inspect_released(source_root: Path, docker_host: str, identity: str | None) -> bool:
    if not identity:
        return False
    process = await asyncio.create_subprocess_exec(
        str(source_root / 'runtime/bin/docker'),
        '--host',
        docker_host,
        'inspect',
        '--type',
        'container',
        identity,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
    return process.returncode != 0 and b'no such' in stderr.lower()


def write_frozen_submission(source_root: Path) -> str:
    source = source_root / 'campaigns/20261102_offline_30min_v2/01-opus5-5/source'
    manifest = json.loads((source.parent / 'source_manifest.json').read_text())
    payload = {}
    for name, digest in manifest['files'].items():
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        payload[name] = base64.b64encode(data).decode()
    return (
        "python - <<'PY'\nimport base64\nfrom pathlib import Path\n"
        "root=Path('/workspace/submission'); root.mkdir(exist_ok=True)\n"
        f'files={payload!r}\n'
        'for name,data in files.items():\n'
        ' p=root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(base64.b64decode(data)); p.chmod(0o755)\n'
        "print('FIXTURE_WRITE_OK')\nPY"
    )


def successful_tool_outputs(path: Path) -> list[str]:
    outputs = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        item = event.get('item') or {}
        if event.get('type') != 'item.completed' or item.get('type') != 'mcp_tool_call':
            continue
        for content in (item.get('result') or {}).get('content', []):
            if content.get('type') != 'text':
                continue
            try:
                result = json.loads(content['text'])
            except (ValueError, KeyError):
                continue
            if isinstance(result, dict) and result.get('exit_code') == 0 and not result.get('timed_out'):
                outputs.append(result.get('output', ''))
    return outputs


async def verify_parallel_runtime(source_root: Path, output: Path, docker_host: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    provider = FrontierAMOfflineProvider(source_root, docker_host)
    identities = []
    checks = {}
    try:
        acquired = await asyncio.gather(
            *(provider.acquire(name, output_dir=str(output / name)) for name in ('left', 'right')),
            return_exceptions=True,
        )
        for value in acquired:
            if isinstance(value, BaseException):
                raise value
        left, right = [value for value in acquired if not isinstance(value, BaseException)]
        identities = [(await value.inspect())['Id'] for value in (left, right)]
        checks['different_container_ids'] = identities[0] != identities[1]
        for value in (left, right):
            value.set_deadline(deadline_monotonic=time.monotonic() + 60)
        result = await asyncio.gather(
            left._run_command({'command': 'printf left > /workspace/marker; cat /workspace/marker', 'timeout': 10}),
            right._run_command(
                {
                    'command': 'test ! -e /workspace/marker && printf right > /workspace/marker && cat /workspace/marker',
                    'timeout': 10,
                }
            ),
        )
        checks['isolated_writes'] = (
            result[0]['stdout'].strip() == 'left'
            and result[1]['stdout'].strip() == 'right'
            and all(value['return_code'] == 0 for value in result)
        )
        checks['all_resource_contracts'] = all(left.configuration_checks.values()) and all(
            right.configuration_checks.values()
        )
    finally:
        await provider.release_all()
    checks['both_released'] = len(identities) == 2 and all(
        await asyncio.gather(*(inspect_released(source_root, docker_host, value) for value in identities))
    )
    report = {'mode': 'runtime_parallel', 'passed': all(checks.values()), 'checks': checks, 'real_model_calls': 0}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


class Fixture(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode: str, command: str):
        self.mode, self.command = mode, command
        self.requests = []
        self.failures = []
        self.normalizations_paired = False
        super().__init__(('127.0.0.1', 0), Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, status, body):
        encoded = json.dumps(body).encode()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.send_header('Retry-After', '0')
            self.end_headers()
            self.wfile.write(encoded)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw)
        fixture = cast(Fixture, self.server)
        number = len(fixture.requests) + 1
        fixture.requests.append({'request': number, 'payload_sha256': hashlib.sha256(raw).hexdigest()})
        if self.path != '/v1/chat/completions' or self.headers.get('Authorization') != 'Bearer local-fixture-only':
            fixture.failures.append('unexpected_endpoint_or_credential')
            self.respond(401, {'error': {'code': 9999}})
            return
        if fixture.mode == 'retry_cap' or (fixture.mode == 'parameter_retry' and number == 1):
            self.respond(500, {'error': {'code': 3006, 'message': 'synthetic upstream failure'}})
            return
        tools = body.get('tools') or []
        if len(tools) != 1:
            fixture.failures.append('unexpected_tool_set')
            self.respond(400, {'error': {'code': 9999}})
            return
        name = tools[0]['function']['name']
        step = number - 1 if fixture.mode == 'parameter_retry' else number
        if fixture.mode == 'deadline':
            time.sleep(10)
        arguments = {'command': fixture.command, 'timeout_seconds': 30, 'max_output_chars': 2000}
        if fixture.mode == 'parameter_retry' and step == 1:
            arguments.update(
                command='mkdir -p /workspace/submission; touch /workspace/submission/FORBIDDEN', timeout_seconds=121
            )
        is_tool = step == 1 or (fixture.mode == 'parameter_retry' and step == 2)
        if (
            fixture.mode == 'parameter_retry'
            and step == 2
            and 'invalid_timeout' not in json.dumps(body.get('messages'))
        ):
            fixture.failures.append('tool_error_not_returned_to_history')
        if not is_tool:
            tool_messages = json.dumps([m for m in body.get('messages', []) if m.get('role') == 'tool'])
            if 'FIXTURE_WRITE_OK' not in tool_messages:
                fixture.failures.append('tool_success_not_returned_to_history')
            if 'FIXTURE_ISOLATION_OK' not in tool_messages:
                fixture.failures.append('isolation_probe_not_returned_to_history')
            if fixture.mode in {'missing_id', 'object_arguments'}:
                messages = body['messages']
                assistant = next((m for m in reversed(messages) if m.get('tool_calls')), {})
                tool = next((m for m in reversed(messages) if m.get('role') == 'tool'), {})
                call_id = ((assistant.get('tool_calls') or [{}])[0]).get('id')
                fixture.normalizations_paired = bool(call_id and call_id == tool.get('tool_call_id'))
            message = {'role': 'assistant', 'content': 'LOCAL_FIXTURE_COMPLETE'}
            finish_reason = 'stop'
        else:
            call: dict[str, Any] = {
                'id': 'fixture_call_' + str(step),
                'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)},
            }
            if fixture.mode == 'missing_id':
                call.pop('id')
            if fixture.mode == 'object_arguments':
                call['function']['arguments'] = arguments
            if fixture.mode == 'unknown_tool':
                call['function']['name'] = 'forbidden_host_tool'
            message = {'role': 'assistant', 'content': None, 'tool_calls': [call]}
            finish_reason = 'length' if fixture.mode == 'length' else 'tool_calls'
        self.respond(
            200,
            {
                'id': 'local-fixture-' + str(number),
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': body['model'],
                'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
                'usage': {'prompt_tokens': 5, 'completion_tokens': 5, 'total_tokens': 10},
            },
        )


async def verify_case(source_root: Path, output: Path, docker_host: str, mode: str) -> dict:
    command = (
        write_frozen_submission(source_root)
        if mode == 'full_numerical'
        else (
            'mkdir -p /workspace/submission; printf verified > /workspace/submission/probe.txt; printf FIXTURE_WRITE_OK'
        )
    )
    if mode in {'length', 'unknown_tool', 'deadline'}:
        command = 'mkdir -p /workspace/submission; touch /workspace/submission/FORBIDDEN'
    else:
        command = (
            "python - <<'PY'\nimport os,socket\nfrom pathlib import Path\n"
            'assert os.getuid()==1000\n'
            "assert not Path('/var/run/docker.sock').exists()\n"
            f'assert not Path({str(source_root)!r}).exists()\n'
            "assert not Path('/workspace/private_pilot').exists()\n"
            "assert not any('API_KEY' in k or 'AIHUB' in k or 'TAIHU' in k for k in os.environ)\n"
            "s=socket.socket(); s.settimeout(1); assert s.connect_ex(('1.1.1.1',53)) != 0; s.close()\n"
            "print('FIXTURE_ISOLATION_OK')\nPY\n" + command
        )
    fixture = Fixture(mode, command)
    if mode == 'network_boundary':
        # Exercise a script file and compiled program: command-name filtering
        # cannot enforce this boundary, while the tool's execution namespace can.
        program = (
            '#include <arpa/inet.h>\n#include <errno.h>\n#include <stdio.h>\n#include <sys/socket.h>\n'
            'int main(void) { int s=socket(AF_INET,SOCK_STREAM,0); '
            'struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(443); '
            'inet_pton(AF_INET,"203.0.113.1",&a.sin_addr); '
            'int r=connect(s,(struct sockaddr*)&a,sizeof(a)); '
            'if(r==0 || (errno!=ENETUNREACH && errno!=EACCES && errno!=EPERM)) return 1; '
            'puts("COMPILED_NETWORK_BLOCKED"); return 0; }\n'
        )
        probe = (
            'import errno,json,socket\n'
            'checks={}\n'
            f'targets=[("external_tcp","203.0.113.1",443),("host_model_proxy","127.0.0.1",{fixture.server_port})]\n'
            'for label,host,port in targets:\n'
            ' s=socket.socket(); s.settimeout(1); rc=s.connect_ex((host,port)); s.close()\n'
            ' checks[label]=rc in (errno.ENETUNREACH,errno.EACCES,errno.EPERM,errno.ECONNREFUSED)\n'
            's=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n'
            'try:\n'
            ' s.sendto(b"probe",("203.0.113.1",9)); checks["external_udp"]=False\n'
            'except OSError as e: checks["external_udp"]=e.errno in (errno.ENETUNREACH,errno.EACCES,errno.EPERM)\n'
            'finally: s.close()\n'
            'print(json.dumps(checks,sort_keys=True)); assert all(checks.values())\n'
            'print("SCRIPT_NETWORK_BLOCKED")\n'
        )
        fixture.command = (
            'set -e\n' + fixture.command + '\n'
            "cat > /workspace/network_probe.py <<'PY'\n" + probe + 'PY\n'
            'python /workspace/network_probe.py\n'
            "cat > /workspace/network_probe.c <<'C'\n" + program + 'C\n'
            'cc /workspace/network_probe.c -o /workspace/network_probe\n'
            '/workspace/network_probe\n'
        )
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    configuration = {
        'imports': ['nexus.extensions.tasks.frontieram_offline'],
        'task_name': 'frontieram_exponential_offline',
        'task_config': {
            'output_dir': str(output / 'task'),
            'evaluate': mode == 'full_numerical',
            'verification_only': True,
        },
        'runtime_provider_name': 'frontieram_offline',
        'runtime_provider_config': {'source_root': str(source_root), 'docker_host': docker_host},
        'agent_name': 'frontieram_codex_offline',
        'agent_config': {
            'base_url': f'http://127.0.0.1:{fixture.server_port}/v1',
            'model': 'claude-opus-5-5',
            'development_seconds': 5 if mode == 'deadline' else 60,
            'max_http_attempts': 8,
        },
        'processors': {'token_counter': {'name': 'token_counter'}},
        'runner': {'cache': {'enabled': False}, 'timeout': 1200},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'input.json').write_text(json.dumps(configuration, indent=2) + '\n')
    started = time.monotonic()
    try:
        if mode == 'full_numerical':
            with (output / 'nexus_cli.log').open('wb') as log:
                process = await asyncio.create_subprocess_exec(
                    str(Path(sys.executable).with_name('nexus')),
                    'run-single',
                    str(output / 'input.json'),
                    '-o',
                    str(output / 'nexus_result.json'),
                    stdout=log,
                    stderr=log,
                )
                await asyncio.wait_for(process.wait(), timeout=1200)
            data = json.loads((output / 'nexus_result.json').read_text())
        else:
            result = await run(RunInput.from_dict(configuration))
            data = result.model_dump(mode='json')
    finally:
        await asyncio.to_thread(fixture.shutdown)
        fixture.server_close()
        thread.join(timeout=2)
    (output / 'nexus_result.json').write_text(json.dumps(data, indent=2) + '\n')
    task = data.get('task_output') or {}
    agent = data.get('agent_output') or {}
    checks = {
        'nexus_returned': data['success'],
        'fixture_history_checks': not fixture.failures,
        'forbidden_marker_absent': not (output / 'task/source/FORBIDDEN').exists(),
        'workspace_container_released': await inspect_released(
            source_root,
            docker_host,
            task.get('provenance', {}).get('container_id'),
        ),
        'all_isolation_checks_passed': bool(task.get('provenance', {}).get('isolation'))
        and all(task['provenance']['isolation'].values()),
    }
    if mode in {'length', 'unknown_tool', 'retry_cap'}:
        checks['failure_not_algorithm_zero'] = task.get('score') is None and task.get('status') == 'unmeasured'
        checks['expected_http_attempts'] = len(fixture.requests) == (4 if mode == 'retry_cap' else 1)
    elif mode == 'deadline':
        checks['shared_deadline'] = (
            agent.get('stop_reason') == 'development_deadline' and agent.get('development_seconds', 999) < 9
        )
    else:
        checks['normal_agent_final'] = (
            agent.get('stop_reason') == 'turn_completed' and agent.get('evaluation_eligible') is True
        )
        checks['output_frozen'] = (output / 'task/source_manifest.json').exists()
        actual_outputs = successful_tool_outputs(output / 'task/agent/codex_events.jsonl')
        checks['real_tool_completed_successfully'] = any(
            'FIXTURE_WRITE_OK' in result and 'FIXTURE_ISOLATION_OK' in result for result in actual_outputs
        )
        expected_completions = 3 if mode == 'parameter_retry' else 2
        usage = data.get('processor_results', {}).get('token_counter', {}).get('main', {})
        checks['native_usage_trace_attached'] = (
            usage.get('llm_call_count') == expected_completions
            and usage.get('prompt_tokens') == 5 * expected_completions
            and usage.get('completion_tokens') == 5 * expected_completions
        )
        if mode in {'missing_id', 'object_arguments'}:
            checks['normalization_history_pairing'] = fixture.normalizations_paired
        if mode == 'parameter_retry':
            checks['same_payload_retry'] = (
                len(fixture.requests) >= 2
                and fixture.requests[0]['payload_sha256'] == fixture.requests[1]['payload_sha256']
            )
            checks['four_actual_requests'] = len(fixture.requests) == 4 and agent.get('http_retries') == 1
        if mode == 'full_numerical':
            checks['all_69_numerical_runs_valid'] = task.get('valid_runs') == 69 and task.get('total_runs') == 69
        if mode == 'network_boundary':
            checks['script_and_compiled_network_blocked'] = any(
                'SCRIPT_NETWORK_BLOCKED' in result and 'COMPILED_NETWORK_BLOCKED' in result for result in actual_outputs
            )
    report = {
        'mode': mode,
        'passed': all(checks.values()),
        'checks': checks,
        'fixture_failures': fixture.failures,
        'local_http_requests': fixture.requests,
        'real_model_calls': 0,
        'elapsed_seconds': time.monotonic() - started,
        'score_scope': 'integration diagnostic only; not a new model trial',
    }
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for mode in args.cases:
        if mode == 'runtime_parallel':
            report = await verify_parallel_runtime(args.source_root, args.output / mode, args.docker_host)
        else:
            report = await verify_case(args.source_root, args.output / mode, args.docker_host, mode)
        reports.append(report)
        print(json.dumps({'mode': mode, 'passed': report['passed'], 'checks': report['checks']}), flush=True)
    summary = {
        'passed': all(r['passed'] for r in reports),
        'cases': reports,
        'real_model_calls': 0,
        'framework_commit': '75409ac247b1c46cedd3466f4b860b5f30e5ce07',
    }
    (args.output / 'verification_result.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--docker-host', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--cases',
        nargs='+',
        choices=[
            'network_boundary',
            'runtime_parallel',
            'full_numerical',
            'parameter_retry',
            'missing_id',
            'object_arguments',
            'length',
            'unknown_tool',
            'retry_cap',
            'deadline',
        ],
        default=[
            'runtime_parallel',
            'network_boundary',
            'full_numerical',
            'parameter_retry',
            'missing_id',
            'object_arguments',
            'length',
            'unknown_tool',
            'retry_cap',
            'deadline',
        ],
    )
    raise SystemExit(asyncio.run(main(parser.parse_args())))
