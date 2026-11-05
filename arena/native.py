"""Protocol v6 development: each model in its own native harness at its highest verified settings, still offline.

    host:       atr serve  (holds the credential, highest reasoning tier, largest output cap, records traffic)
                  ^ TCP 127.0.0.1
                socat UNIX-LISTEN  run/atr.sock
                  ^ the only path out of the container (a bind-mounted socket, no network)
    container:  socat TCP 127.0.0.1:18080 -> /run/atr/atr.sock
                ATR client transport + Harness.setup() -> native CLI (Claude Code, Codex, DSH, Step, Antigravity)
                the CLI's native tools and all candidate code run here: --network none, one pinned CPU

The upstream key never enters the container; the CLI only gets a local proxy token. The development deadline is
enforced from the host by pausing the container, and the submission is frozen with the same reviewed exporter as
protocols v4/v5, so grading is unchanged.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import time

from pathlib import Path

from .tasks import TASKS, Placement, write


ATR_ROOT = Path(os.environ.get('FRONTIERAM_ATR', '/data/workspace/agent-traffic-runtime'))
ATR_PYTHON = ATR_ROOT / '.venv/bin/python'
HARNESS_IMAGES = {'exponential': 'frontieram-exponential-harness:v6', 'conic': 'frontieram-conic-harness:v6',
                  'galaxy-gravity': 'frontieram-galaxy-gravity-harness:v6',
                  'polarization-demosaicking': 'frontieram-polarization-demosaicking-harness:v6',
                  'species-tree': 'frontieram-species-tree-harness:v6',
                  'trajectory-covariance': 'frontieram-trajectory-covariance-harness:v6'}
# The vendor's own client where ATR has one; Claude Code for Messages models without one, Codex for Responses ones.
HARNESS = {
    'claude-opus-5-5': 'claude',
    'gpt-6.1-sol': 'codex',
    'gpt-6-astra': 'codex',
    'deepseek-v4-pro': 'dsh',
    'step-5-preview': 'step',
    'gemini-4-argon': 'antigravity',
    'hy4': 'claude',
    'minimax-m3.1-flash': 'codex',
}
CREDENTIALS = {
    'hy4': ('HY4_TAIJI_MESSAGE_API_TOKEN', Path('/root/.frontieram/hy4-taiji.token')),
}
DEFAULT_CREDENTIAL = ('AIHUB_API_KEY', Path('/root/.frontieram/aihub.key'))
CONTAINER_PORT = 18080
MEMORY = 4 * 2**30
# Not the grader's UID 1000: RLIMIT_NPROC is per real UID host-wide, and the conic verifier caps UID 1000 at 128
# tasks; native harness threads under UID 1000 made concurrent grading fail with "Cannot fork".
DEV_UID = 2000

LAUNCHER = r'''
import json, subprocess, sys
from contextlib import ExitStack
from pathlib import Path
from atr.harness import Endpoint, Task, resolve
from atr.harness.nonstream import client_transport

spec = json.loads(sys.argv[1])
endpoint = Endpoint(base_url=spec["base_url"], token=spec["token"], model=spec["model"],
                    context_window=spec["context_window"], max_output=spec["max_output"], effort=spec["effort"],
                    dialect=spec["dialect"], streaming=False)
# The client transport answers only once the whole upstream response exists, so a client idle timeout shorter than
# the longest max-effort response ends the attempt; every client gets the full development window instead.
WINDOW_MS = int(spec["client_timeout_seconds"] * 1000)
options = {"step": {"httpIdleTimeoutMs": 0, "retry": {"provider": {"timeoutMs": WINDOW_MS}}}}
task = Task(prompt=spec["prompt"], workspace=Path("/workspace"), harness_options=options.get(spec["harness"], {}))
harness = resolve(spec["harness"])
refused = harness.refuses(task)
if refused:
    print("ATR_REFUSED " + refused, flush=True)
    sys.exit(97)
with ExitStack() as stack:
    endpoint = stack.enter_context(client_transport(endpoint, timeout=None))
    home = Path("/home/agent/atr-home")
    launch = harness.setup(home, endpoint, task)
    environment = launch.environment()
    if spec["harness"] == "claude":
        environment["API_TIMEOUT_MS"] = str(WINDOW_MS)
    if spec["harness"] == "codex":
        config = home / ".codex" / "config.toml"
        text = config.read_text()
        marker = 'wire_api = "responses"\n'
        if marker not in text:
            print("ATR_REFUSED codex config has no Responses provider table to extend", flush=True)
            sys.exit(98)
        config.write_text(text.replace(marker, marker + f"stream_idle_timeout_ms = {WINDOW_MS}\n", 1))
    print("ATR_LAUNCH " + json.dumps({"argv": list(launch.argv)[:3], "stdin": launch.prompt_via_stdin}), flush=True)
    process = subprocess.Popen(list(launch.argv), env=environment, cwd="/workspace",
                               stdin=subprocess.PIPE if launch.prompt_via_stdin else subprocess.DEVNULL)
    if launch.prompt_via_stdin:
        process.stdin.write(task.prompt.encode())
        process.stdin.close()
    sys.exit(process.wait())
'''


def resolve_route(model: str, harness: str) -> dict:
    """Highest verified reasoning tier, largest output cap and the client window, from ATR's manual catalog."""
    script = (
        'import json,sys\nfrom atr.upstream import aihub\nfrom atr.harness import resolve\n'
        'model, harness = sys.argv[1:]\nroute = aihub.resolve(model)\nclient = resolve(harness)\n'
        'dialect = client.dialect_for(route) if hasattr(client, "dialect_for") else client.dialect\n'
        'print(json.dumps({"dialect": dialect, "context_window": route.context_length,'
        ' "max_output": aihub.output_limit(model, "max", dialect=dialect, route=route),'
        ' "effort": aihub.effort_level(model, "", dialect=dialect, route=route)}))\n'
    )
    result = subprocess.run([str(ATR_PYTHON), '-c', script, model, harness], capture_output=True, text=True,
                            cwd=ATR_ROOT, check=False, timeout=60)
    if result.returncode:
        raise RuntimeError(f'ATR cannot route {model} through {harness}: {result.stderr.strip()[-400:]}')
    return json.loads(result.stdout)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def _wait_port(port: int, process: subprocess.Popen, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('atr serve exited during startup')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                return
        time.sleep(0.25)
    raise RuntimeError('atr serve did not start listening')


def freeze(controller, docker, name: str, output: Path, label: str = 'source') -> dict:
    """controller.freeze with the exporter under DEV_UID: /proc/1/root of the paused container is only readable by
    its own UID. The export script, size caps and archive validation are the reviewed ones, unchanged."""
    docker.pause(name)
    raw = output / f'{label}.docker.tar'
    image = docker.inspect(name)['Image']
    exporter = 'frontieram-export-' + secrets.token_hex(16)
    command = [
        *docker.prefix, 'run', '--rm', '--name', exporter, '--network', 'none', '--pid', 'container:' + name,
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true', '--user', f'{DEV_UID}:{DEV_UID}',
        *controller.placement_options(docker.cpuset, docker.cgroup_parent),
        '--memory', str(controller.MEMORY), '--memory-swap', str(controller.MEMORY), '--pids-limit', '32',
        '--ulimit', 'nofile=128:128', '--ipc', 'none', '--log-driver', 'none', '--entrypoint', 'python3',
        image, '-c', controller.EXPORT_SCRIPT,
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total = 0
    try:
        with raw.open('xb') as stream:
            started = time.monotonic()
            while block := process.stdout.read(1024 * 1024):
                total += len(block)
                if total > controller.RAW_TAR_CAP:
                    raise controller.SubmissionFailure('export exceeds artifact plus archive overhead limit')
                if time.monotonic() - started > 90:
                    raise controller.InfrastructureFailure('paused workspace export timed out')
                stream.write(block)
        error = process.stderr.read(4096)
        code = process.wait(timeout=30)
        if code == 42:
            raise controller.SubmissionFailure(error.decode(errors='replace')[:500])
        if code:
            raise controller.InfrastructureFailure('paused workspace export failed: ' + error.decode(errors='replace')[:500])
        result = controller.validate_and_extract(raw, output / label, output / f'{label}.tar')
        result.update(paused_before_export=True, frozen_at_utc=controller.now(), raw_archive_bytes=total,
                      exporter_uid=DEV_UID)
        controller.save(output / f'{label}_manifest.json', result)
        raw.unlink()
        docker.remove(name)
        return result
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        subprocess.run([*docker.prefix, 'rm', '--force', exporter], capture_output=True, timeout=15)


# v6.1: development cores are packed more densely than grading cores, so timings measured while developing are noisy.
TIMING_NOTE = (
    '\n\n## Development timing is approximate\n'
    'Your development core shares its L3 cache and memory bandwidth with other concurrent attempts, so times you '
    'measure here can be 10-30% slower than in verification and vary from run to run. Use them only for relative '
    'comparisons: interleave the variants you compare (A, B, A, B, ...) and compare medians. Verification runs on an '
    'isolated grading core; only the verification result is authoritative. Each case is timed over the whole process '
    'from launch to exit, including interpreter startup and imports, so keep a safety margin under every limit.\n'
)


SUBMISSION_NOTE = (
    'Put the complete source, an executable run.sh and README.md in /workspace/submission; the evaluator runs '
    './run.sh from that directory in a clean copy of this environment without the agent tools.'
)


def prompt(task_name: str, runtime: dict, protocol: str = 'v6') -> str:
    task = TASKS[task_name]
    return (
        task.instruction.replace(' Use only the offline workspace tool.', '')
        + ' The machine is offline: no network access and no package downloads; use what is installed. '
        + getattr(task, 'submission_note', SUBMISSION_NOTE) + ' '
        'One attempt, at most 7200 wall-clock seconds including model requests and tool time. Finishing your turn, '
        'or the deadline, freezes /workspace/submission for evaluation; there is no later repair or hidden feedback.'
        + runtime['architecture_override_prompt']
        + (TIMING_NOTE if protocol == 'v6.1' else '')
    )


def develop(spec: dict, runtime: dict) -> dict:
    """One v6 attempt; returns the same summary fields as a Nexus v4/v5 attempt."""
    task = TASKS[spec['task']]
    model = spec['model']
    harness = HARNESS[model]
    placement = Placement(**spec['placement'])
    output = Path(spec['output'])
    frozen_dir = output / 'task'
    frozen_dir.mkdir(parents=True, exist_ok=True)
    route = resolve_route(model, harness)
    seconds = spec.get('development_seconds') or 7200
    assets = task.assets(runtime)
    docker = assets.controller.Docker(runtime['docker_path'], placement.docker_host, placement.cpuset,
                                      placement.cgroup_parent)
    write(output / 'v6_conditions.json', {'harness': harness, 'route': route, 'image': HARNESS_IMAGES[task.name],
                                          'development_seconds': seconds, 'memory_bytes': MEMORY})
    credential_env, credential_file = CREDENTIALS.get(model, DEFAULT_CREDENTIAL)
    token = secrets.token_urlsafe(32)
    port = _free_port()
    socket_dir = Path('/run/frontieram/atr') / output.name
    socket_dir.mkdir(parents=True, exist_ok=False)
    socket_dir.chmod(0o777)
    name = f'frontieram-v6-{secrets.token_hex(8)}'
    processes: list[subprocess.Popen] = []
    summary: dict = {'harness': harness, 'route': route}
    started = time.monotonic()
    try:
        environment = {**os.environ, credential_env: credential_file.read_text().strip()}
        serve = subprocess.Popen(
            [str(ATR_ROOT / '.venv/bin/atr'), 'serve', '--model', model, '--host', '127.0.0.1', '--port', str(port),
             '--token', token, '--record-dir', str(output / 'atr'), '--max-output', 'max'],
            cwd=ATR_ROOT, env=environment, stdout=(output / 'atr-serve.log').open('wb'), stderr=subprocess.STDOUT,
        )
        processes.append(serve)
        _wait_port(port, serve)
        relay = subprocess.Popen(
            ['socat', f'UNIX-LISTEN:{socket_dir}/atr.sock,fork,mode=666', f'TCP:127.0.0.1:{port}'],
            stdout=subprocess.DEVNULL, stderr=(output / 'socat.log').open('wb'),
        )
        processes.append(relay)
        create = [
            *docker.prefix, 'create', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges:true', '--cgroupns', 'private',
            '--cpus', '1', '--cpuset-cpus', placement.cpuset, '--cgroup-parent', placement.cgroup_parent,
            '--memory', str(MEMORY), '--memory-swap', str(MEMORY),
            # Node-based harnesses share this container with the candidate's code; v4's 32 processes and 128 file
            # descriptors cannot hold a native client, so v6 raises both and records it in v6_conditions.json.
            '--pids-limit', '1024', '--ulimit', 'nofile=16384:16384', '--ulimit', 'core=0:0',
            '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g,mode=1777',
            '--tmpfs', f'/workspace:rw,exec,nosuid,nodev,size=1g,uid={DEV_UID},gid={DEV_UID},mode=0755',
            '--tmpfs', f'/home/agent:rw,exec,nosuid,nodev,size=1g,uid={DEV_UID},gid={DEV_UID},mode=0700',
            '--mount', f'type=bind,src={socket_dir},dst=/run/atr',
            '--env', 'HOME=/home/agent', '--log-driver', 'none', '--user', f'{DEV_UID}:{DEV_UID}',
            '--workdir', '/workspace',
            '--entrypoint', '/bin/sleep', HARNESS_IMAGES[task.name], 'infinity',
        ]
        subprocess.run(create, check=True, capture_output=True, timeout=60)
        docker.run('start', name)
        docker.run('exec', name, '/bin/mkdir', '-p', '/workspace/submission')
        with Path(assets.public_tar).open('rb') as stream:
            subprocess.run([*docker.prefix, 'exec', '-i', '--workdir', '/workspace', name, '/bin/tar',
                            '--no-same-owner', '-xf', '-'], stdin=stream, check=True, capture_output=True, timeout=120)
        docker.run('exec', '-d', name, 'socat', f'TCP-LISTEN:{CONTAINER_PORT},bind=127.0.0.1,fork,reuseaddr',
                   'UNIX-CONNECT:/run/atr/atr.sock')
        launcher_spec = {
            'base_url': f'http://127.0.0.1:{CONTAINER_PORT}', 'token': token, 'model': model, 'harness': harness,
            'prompt': prompt(task.name, runtime, spec['protocol']), 'client_timeout_seconds': seconds, **route,
        }
        deadline = started + seconds
        with (output / 'cli.log').open('wb') as log:
            client = subprocess.Popen(
                [*docker.prefix, 'exec', name, '/opt/atr-venv/bin/python', '-c', LAUNCHER, json.dumps(launcher_spec)],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
            try:
                code = client.wait(timeout=max(1.0, deadline - time.monotonic()))
                stop_reason = 'turn_completed' if code == 0 else 'harness_exited'
            except subprocess.TimeoutExpired:
                docker.pause(name)
                client.kill()
                client.wait(timeout=30)
                code, stop_reason = None, 'development_deadline'
        stopped = time.monotonic()
        docker.pause(name)
        summary.update(exit_code=code, agent_stop_reason=stop_reason, development_seconds=stopped - started)
        eligible = stop_reason in ('turn_completed', 'development_deadline')
        try:
            manifest = freeze(assets.controller, docker, name, frozen_dir)
            summary.update(source_frozen=True, files=len(manifest['files']))
            status = 'verification_completed' if eligible else 'unmeasured'
        except assets.controller.SubmissionFailure as error:
            docker.remove(name)
            summary.update(source_frozen=False, freeze_error=str(error)[:300])
            status = 'invalid_submission' if eligible else 'unmeasured'
    except Exception as error:
        subprocess.run([*docker.prefix, 'rm', '--force', name], capture_output=True)
        summary.update(error=f'{type(error).__name__}: {str(error)[:500]}', agent_stop_reason='infrastructure_error')
        status = 'unmeasured'
    finally:
        for process in reversed(processes):
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
    records = list((output / 'atr').rglob('events.jsonl'))
    requests = sum(1 for path in records for line in path.read_text().splitlines() if '"request"' in line)
    summary.update(task_status=summary.get('task_status', status), http_attempts=requests,
                   agent_eligible=status in ('verification_completed', 'invalid_submission'))
    if summary.get('source_frozen') and not (frozen_dir / 'source.tar').exists():
        summary.update(task_status='unmeasured', error='frozen archive missing')
    write(output / 'develop_result.json', summary)
    return summary
