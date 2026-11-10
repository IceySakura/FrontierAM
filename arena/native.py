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

import contextlib
import fcntl
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time

from pathlib import Path

from .tasks import TASKS, Placement, write


ATR_ROOT = Path(os.environ.get('FRONTIERAM_ATR', '/data/workspace/agent-traffic-runtime'))
ATR_PYTHON = ATR_ROOT / '.venv/bin/python'
HARNESS_IMAGES = {'exponential': 'frontieram-exponential-harness:v6', 'conic': 'frontieram-conic-harness:v6',
                  'exponential-v2': 'frontieram-exponential-harness:v6',
                  'galaxy-gravity': 'frontieram-galaxy-gravity-harness:v6',
                  'galaxy-gravity-v2': 'frontieram-galaxy-gravity-harness:v6',
                  'galaxy-gravity-v3': 'frontieram-galaxy-gravity-harness:v6',
                  'polarization-demosaicking': 'frontieram-polarization-demosaicking-harness:v6',
                  'species-tree': 'frontieram-species-tree-harness:v6',
                  'species-tree-v2': 'frontieram-species-tree-harness:v6',
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


def freeze(controller, docker, name: str, output: Path, label: str = 'source', remove: bool = True,
           pause: bool = True) -> dict:
    """controller.freeze with the exporter under DEV_UID: /proc/1/root of the paused container is only readable by
    its own UID. The export script, size caps and archive validation are the reviewed ones, unchanged. A v6.2 test
    submission keeps the (still paused) container; a v6.3 test snapshot exports from the running container
    (pause=False): a file changed during the export fails the snapshot, and the agent may retry."""
    output.mkdir(parents=True, exist_ok=True)
    if pause:
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
        result.update(paused_before_export=pause, frozen_at_utc=controller.now(), raw_archive_bytes=total,
                      exporter_uid=DEV_UID)
        controller.save(output / f'{label}_manifest.json', result)
        raw.unlink()
        if remove:
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
# Exponential tasks time a session from the first input byte, not from process launch (their API.md is authoritative).
TIMING_NOTE_SESSION = TIMING_NOTE.replace(
    'Each case is timed over the whole process from launch to exit, including interpreter startup and imports, so keep '
    'a safety margin under every limit.',
    'Timing and limits follow the task\'s own interface rules exactly as written in its documents; keep a safety '
    'margin under every limit.')


SUBMISSION_NOTE = (
    'Put the complete source, an executable run.sh and README.md in /workspace/submission; the evaluator runs '
    './run.sh from that directory in a clean copy of this environment without the agent tools.'
)

# v6.2: what the final standardised score rewards, per task, without the mapping function (results/PROTOCOL_V62.md).
SCORING_DISCLOSURE = {
    'conic': (
        'An instance counts as solved only if all three runs of it, each a fresh process timed from launch to exit '
        '(interpreter startup and imports included), finish within that instance\'s own time limit and return a '
        'result that passes the independent checker: an optimal solution or a valid primal or dual infeasibility '
        'certificate; "unresolved" never counts. The final score depends only on the fraction of held-out instances '
        'solved, compared with a strong reference solver on the same instances and limits: well below the reference '
        'scores zero, matching it is the midpoint of the scale and solving everything is the maximum. Being faster '
        'within a limit earns nothing extra. All five cone families (linear, quadratic, second-order cone, '
        'semidefinite, exponential) count; sizes range from tens to about a million variables, and most instances '
        'have a one-second limit, so robustness across the whole range matters more than peak speed on a few.'
    ),
    'exponential-v2': (
        'Every case, scored sessions and controls alike, must return results within its limits and accuracy '
        'tolerance; a single failed case makes the whole submission score zero. Given that, the score depends on '
        'your time per session relative to the fastest of a panel of published methods on that session (Expokit, '
        'KIOPS, expmv, SciPy expm_multiply, a Chebyshev method, SLEPc MFN, ExponentialUtilities.jl), averaged '
        'with equal weight over matrices, structure families, scenarios and accuracy levels: matching the per-case '
        'fastest published method is the midpoint of the scale, and the faster you are than that the higher the '
        'score. Timing follows the task\'s own interface rules (API.md): it starts before the first matrix byte '
        'is delivered and ends when all results are out and the process has exited; imports and input-independent '
        'initialisation before READY are not timed.'
    ),
}

SCORING_DISCLOSURE['species-tree-v2'] = (
    'Each held-out instance is run once in a fresh process within its time limit (startup included); a timeout, crash '
    'or invalid tree counts as the worst error for that instance. The score depends only on the mean normalised '
    'Robinson-Foulds error over all held-out instances, compared with two published reference methods measured on the '
    'same instances and limits: matching the weaker one scores zero, matching the stronger one is the midpoint of the '
    'scale and recovering every true split is the maximum. Running time does not count beyond the limit. Instances '
    'vary in how much the gene trees conflict, so a method must stay accurate across that range.'
)

SCORING_DISCLOSURE['galaxy-gravity-v2'] = (
    'Every call on every held-out snapshot must stay within the CPU, wall-clock and memory limits and return finite '
    'accelerations for all particles; a single failed call makes the whole submission score zero. Given that, the '
    'score depends only on the overall relative acceleration error, compared with two tree-code references measured '
    'on the same snapshots and budget: matching the weaker one scores zero, matching the stronger one is the midpoint '
    'of the scale and exact accelerations are the maximum. Using less CPU time earns nothing extra, so spend the budget '
    'on accuracy while keeping a safety margin under the limit.'
)

SCORING_DISCLOSURE['galaxy-gravity-v3'] = (
    'Every call on every held-out snapshot must stay within the CPU, wall-clock and memory limits and return finite '
    'accelerations for all particles; a single failed call makes the whole submission score zero. Given that, the '
    'score depends only on the overall relative acceleration error, on a logarithmic scale: matching a classical tree '
    'code scores zero, matching a high-order fast multipole reference is the midpoint of the scale, and an error of '
    '1e-15 (double precision) is the maximum. Using less CPU time earns nothing extra.'
)

SCORING_DISCLOSURE['polarization-demosaicking'] = (
    'Every held-out scene must be reconstructed within its wall-clock limit (startup and imports included), memory '
    'limit and output contract; a single failed scene makes the whole submission score zero. Given that, the score '
    'depends only on the mean RMSE over all scenes and the four angles, compared with two reference methods measured on '
    'the same scenes and limits: matching the weaker one scores zero, matching the stronger one is the midpoint of the '
    'scale and zero error is the maximum. Speed within the limit earns nothing extra, so keep a safety margin under it.'
)
SCORING_DISCLOSURE['trajectory-covariance'] = (
    'The score depends only on the mean whitened error over all held-out inputs, averaged with equal weight over the '
    'settings and trajectory lengths; a failed or invalid output counts as the zero matrix for that input. It is '
    'compared with two reference estimators measured on the same inputs: matching the weaker one scores zero, matching '
    'the stronger one is the midpoint of the scale and zero error is the maximum. Running time does not count beyond '
    'the limit. The settings differ strongly in their autocorrelation structure, so a method must be accurate across all '
    'of them rather than excel on one.'
)

SETTING_SINGLE = (
    'One attempt, at most {seconds} wall-clock seconds including model requests and tool time. Finishing your '
    'turn, or the deadline, freezes /workspace/submission for evaluation; the evaluator gives no feedback during '
    'development and there is no later repair.'
)

SETTING_ITERATIVE = (
    'One attempt, at most {seconds} wall-clock seconds including model requests, tool time and evaluations. '
    'At any time you may run `/opt/frontieram/bin/submit --test` (optionally `--groups G1,G2` or `--cases ID1,ID2`): '
    'it freezes the current /workspace/submission, pauses this environment, grades the frozen copy on the public '
    'development set with the formal evaluator on isolated grading cores (one timed run per case, the formal per-case '
    'limits, cases with short limits first), and then writes per-case results, a per-case comparison with the baseline '
    'and a public-set score on the 100-point scale (baseline = 50) to /workspace/feedback/test-<n>/report.md (and '
    'report.json), and prints a summary. Each test evaluation is capped at {cap} minutes: when the cap is reached the '
    'cases finished so far are reported, the rest are marked not evaluated, and the score covers the evaluated cases '
    'only. {typical} The evaluation time counts against your budget; restrict a test with --groups or --cases when you '
    'need only part of the set. You may test as often as you like, one test at a time. `/opt/frontieram/bin/submit '
    '--status` shows the remaining budget without using it. `/opt/frontieram/bin/submit --final` freezes '
    '/workspace/submission as your final submission and ends the attempt; if you never submit --final, '
    '/workspace/submission as it stands when the budget runs out is graded (not the last tested version). Only the '
    'hidden held-out set determines your score; the public-set score is feedback.'
)
TEST_CAP_SECONDS = int(os.environ.get('FRONTIERAM_V62_TEST_CAP') or 900)  # override only for validation runs
# Measured full public evaluations of baseline-speed solutions on this machine (results/PROTOCOL_V62.md).
TYPICAL_EVALUATION = {
    'conic': 'A full public evaluation of a solver about as fast and robust as the baseline takes about 6 minutes; '
             'a solver that runs to many of the long limits takes longer and may reach the cap.',
    'exponential-v2': 'A full public evaluation of a solution about as fast as the baseline takes about 4 minutes; '
                      'slower or failing solutions take longer and may reach the cap.',
}

# Protocol v6.3 (results/PROTOCOL_V63.md): the limit and the scoring rule are stated; nothing suggests using all of it.
V63_SECONDS = {'single': 3600, 'iterative': 14400}
SETTING_SINGLE_V63 = (
    'One attempt with a limit of {minutes} minutes of wall-clock time, including model requests and tool time. When '
    'you finish your turn, or when the limit is reached, /workspace/submission is frozen and graded once on a hidden '
    'held-out set; only that score counts. The evaluator gives no feedback during development.'
)
SETTING_ITERATIVE_V63 = (
    'One attempt with a limit of {minutes} minutes of wall-clock time, including model requests and tool time. '
    '`/opt/frontieram/bin/submit --test` takes a snapshot of /workspace/submission and returns at once; the snapshot '
    'is evaluated on a hidden public set of cases on a separate grading core while you keep working (the clock keeps '
    'running). When the evaluation is done, /workspace/feedback/test-<n>/report.md gives the public-set score on the '
    '100-point scale (reference = 50, the same mapping as the final score) and {detail}; it never lists individual '
    'cases. One test is evaluated at a time, each capped at {cap} minutes. `/opt/frontieram/bin/submit --status` '
    'shows the time left and the state of your tests. `/opt/frontieram/bin/submit --final` freezes '
    '/workspace/submission as your final submission and ends the attempt; otherwise /workspace/submission as it '
    'stands when the limit is reached is graded. Only the final score on the hidden held-out set counts; the '
    'public-set score is feedback.'
)
FEEDBACK_DETAIL = {
    True: 'a table with one row per category of cases: the numbers of passed, wrong, timed-out and crashed cases, your '
          '{metrics} relative to the reference, and the category score',
    False: 'the numbers of passed, wrong, timed-out and crashed cases and your {metrics} relative to the reference',
}

SUBMIT_CLIENT_V63 = r'''#!/usr/bin/env python3
"""FrontierAM protocol v6.3: --test snapshots /workspace/submission for a public-set evaluation in the background;
--status shows the time left and the tests; --final ends the attempt with /workspace/submission as it stands."""
import argparse, json, socket, sys

parser = argparse.ArgumentParser(description=__doc__)
kind = parser.add_mutually_exclusive_group(required=True)
kind.add_argument("--test", action="store_true", help="snapshot and evaluate on the hidden public set (returns at once)")
kind.add_argument("--status", action="store_true", help="time left, tests so far, the running test")
kind.add_argument("--final", action="store_true", help="freeze /workspace/submission and end the attempt")
args = parser.parse_args()
request = {"kind": "test" if args.test else "final" if args.final else "status"}
connection = socket.socket(socket.AF_UNIX)
connection.connect("/run/frontieram-eval/eval.sock")
connection.sendall((json.dumps(request) + "\n").encode())
data = b""
while True:
    chunk = connection.recv(65536)
    if not chunk:
        break
    data += chunk
reply = json.loads(data or b'{"ok": false, "message": "no reply"}')
print(reply["message"])
sys.exit(0 if reply.get("ok") else 1)
'''

SUBMIT_CLIENT = r'''#!/usr/bin/env python3
"""FrontierAM protocol v6.2: --test grades /workspace/submission on the public set; --final ends the attempt."""
import argparse, json, socket, sys

parser = argparse.ArgumentParser(description=__doc__)
kind = parser.add_mutually_exclusive_group(required=True)
kind.add_argument("--test", action="store_true", help="public-set evaluation with per-case baseline comparison")
kind.add_argument("--final", action="store_true", help="freeze /workspace/submission and end the attempt")
kind.add_argument("--status", action="store_true", help="show the remaining budget and tests so far")
parser.add_argument("--groups", help="comma-separated groups (or families) to evaluate")
parser.add_argument("--cases", help="comma-separated case IDs to evaluate")
args = parser.parse_args()
request = {"kind": "test" if args.test else "final" if args.final else "status", "groups": args.groups, "cases": args.cases}
connection = socket.socket(socket.AF_UNIX)
connection.connect("/run/frontieram-eval/eval.sock")
connection.sendall((json.dumps(request) + "\n").encode())
data = b""
while True:
    chunk = connection.recv(65536)
    if not chunk:
        break
    data += chunk
reply = json.loads(data or b'{"ok": false, "message": "no reply"}')
print(reply["message"])
sys.exit(0 if reply.get("ok") else 1)
'''

ROOT = Path(__file__).resolve().parents[1]
# The public-feedback graders (results/PROTOCOL_V62.md): their own runtimes or seals and their own stores.
PUBLIC_EVAL = {
    'conic': {'env': {'FRONTIERAM_RUNTIME': str(ROOT / '.runtime-conic-public')}, 'store': ROOT / 'var/stores/store-public-conic'},
    'exponential-v2': {'env': {'FRONTIERAM_EXP2_SEAL': str(ROOT / '.runtime-exp2-public')},
                       'store': ROOT / 'var/stores/store-public-exp2'},
}


def feedback_detail(task_name: str) -> str:
    from .public_v63 import config

    cfg = config(task_name)
    metrics = ' and '.join(m for m in (cfg.get('accuracy'), 'running time' if cfg.get('timed') else None) if m)
    return FEEDBACK_DETAIL[bool(cfg.get('category'))].format(metrics=metrics or 'results')


def prompt(task_name: str, runtime: dict, protocol: str = 'v6', setting: str = 'single', seconds: float = 7200) -> str:
    task = TASKS[task_name]
    if protocol == 'v6.3':
        minutes = int(round(seconds / 60))
        budget = (SETTING_ITERATIVE_V63.format(minutes=minutes, detail=feedback_detail(task_name), cap=TEST_CAP_SECONDS // 60)
                  if setting == 'iterative' else SETTING_SINGLE_V63.format(minutes=minutes))
        budget += ' How the final score is determined: ' + SCORING_DISCLOSURE[task_name]
    elif protocol == 'v6.2':
        budget = (SETTING_ITERATIVE.format(seconds=int(seconds), cap=TEST_CAP_SECONDS // 60,
                                           typical=TYPICAL_EVALUATION.get(task_name, ''))
                  if setting == 'iterative' else SETTING_SINGLE.format(seconds=int(seconds)))
        budget += ' How the final score is determined: ' + SCORING_DISCLOSURE[task_name]
    else:
        budget = ('One attempt, at most 7200 wall-clock seconds including model requests and tool time. Finishing '
                  'your turn, or the deadline, freezes /workspace/submission for evaluation; there is no later repair '
                  'or hidden feedback.')
    return (
        task.instruction.replace(' Use only the offline workspace tool.', '')
        + ' The machine is offline: no network access and no package downloads; use what is installed. '
        + getattr(task, 'submission_note', SUBMISSION_NOTE) + ' '
        + budget
        + runtime['architecture_override_prompt']
        + ((TIMING_NOTE_SESSION if task_name.startswith('exponential') and protocol in ('v6.2', 'v6.3') else TIMING_NOTE)
           if protocol in ('v6.1', 'v6.2', 'v6.3') else '')
    )


class EvalService:
    """v6.2 iterative setting: answers /opt/frontieram/bin/submit over a socket bind-mounted into the container.

    A test pauses the container, freezes /workspace/submission with the reviewed exporter (keeping the container),
    grades the copy on the public set (`bench public-eval` against the task's public-feedback grader), copies the
    report into /workspace/feedback/test-<n>/ and resumes. A final request ends the attempt. Everything counts against
    the deadline; at the deadline the attempt is frozen at once and an unfinished evaluation is left to complete.
    """

    def __init__(self, task_name, docker, controller, name: str, output: Path, started: float, deadline: float):
        self.task, self.docker, self.controller, self.name = task_name, docker, controller, name
        self.output, self.started, self.deadline = output / 'evaluations', started, deadline
        self.final, self.ended = threading.Event(), threading.Event()
        self.busy, self.container = threading.Lock(), threading.Lock()
        self.records: list[dict] = []
        self.processes: list[subprocess.Popen] = []
        self.server = None

    def start(self, socket_path: Path) -> None:
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(str(socket_path))
        socket_path.chmod(0o666)
        self.server.listen(4)
        self.server.settimeout(1.0)
        threading.Thread(target=self._serve, daemon=True).start()

    def end(self) -> None:
        """Called by develop() before its final pause: no test may resume the container after this."""
        self.ended.set()
        with self.container:
            self.docker.pause(self.name)
        if self.server is not None:
            self.server.close()

    def drain(self, timeout: float = 7200) -> None:
        """Wait for abandoned evaluations, so their grading jobs never overlap the next iterative attempt."""
        for process in self.processes:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()

    def _serve(self) -> None:
        while not self.ended.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection) -> None:
        with connection:
            try:
                request = json.loads(connection.makefile().readline())
                kind = request.get('kind')
                reply = self._final() if kind == 'final' else self._status() if kind == 'status' else self._test(request)
            except Exception as error:
                reply = {'ok': False, 'message': f'evaluation service error: {type(error).__name__}: {error}'}
            try:
                connection.sendall(json.dumps(reply).encode())
            except OSError:
                pass

    def _now(self) -> float:
        return round(time.monotonic() - self.started, 1)

    def _status(self) -> dict:
        tests = [r for r in self.records if r['kind'] == 'test']
        running = self.busy.locked()
        return {'ok': True, 'message': f'Remaining budget: {max(0, int(self.deadline - time.monotonic()))} s. Tests so far: '
                                       f'{len(tests)}{" (one is running)" if running else ""}. Each test is capped at '
                                       f'{TEST_CAP_SECONDS // 60} minutes.'}

    def _final(self) -> dict:
        self.records.append({'kind': 'final', 'at_seconds': self._now()})
        self.final.set()
        return {'ok': True, 'message': 'Final submission received: /workspace/submission is frozen and the attempt ends.'}

    def _test(self, request: dict) -> dict:
        if not self.busy.acquire(blocking=False):
            return {'ok': False, 'message': 'Another test evaluation is still running; wait for its result.'}
        try:
            n = sum(r['kind'] == 'test' for r in self.records) + 1
            record = {'kind': 'test', 'n': n, 'requested_at_seconds': self._now(),
                      'groups': request.get('groups'), 'cases': request.get('cases')}
            self.records.append(record)
            if self.ended.is_set() or self.deadline - time.monotonic() < 30:
                record['status'] = 'refused_no_time'
                return {'ok': False, 'message': 'Too little time is left for an evaluation.'}
            directory = self.output / f'test-{n:02d}'
            try:
                with self.container:
                    if self.ended.is_set():
                        return {'ok': False, 'message': 'The attempt has ended.'}
                    manifest = freeze(self.controller, self.docker, self.name, directory, remove=False)
                record['files'] = len(manifest['files'])
            except self.controller.SubmissionFailure as error:
                record['status'] = 'invalid_submission'
                self._resume(None, n)
                return {'ok': True, 'message': f'Test {n}: /workspace/submission cannot be frozen as a submission: '
                                               f'{str(error)[:300]}'}
            except Exception as error:
                record.update(status='evaluation_error', reason=f'freeze: {type(error).__name__}: {str(error)[:300]}')
                self._resume(None, n)
                return {'ok': False, 'message': f'Test {n} failed on the evaluator side while freezing (not a verdict '
                                                f'on your submission); you may retry.'}
            config = PUBLIC_EVAL[self.task]
            command = [sys.executable, '-m', 'arena.cli', '--store', str(config['store']), 'public-eval',
                       '--task', self.task, '--source', str(directory / 'source'), '--out', str(directory / 'report'),
                       '--label', f'{self.output.parent.name} test {n}',
                       '--cap', str(int(max(30, min(TEST_CAP_SECONDS, self.deadline - time.monotonic() - 30))))]
            for option in ('groups', 'cases'):
                if request.get(option):
                    command += [f'--{option}', str(request[option])]
            log = (directory / 'public-eval.log').open('wb')
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **config['env'], 'FRONTIERAM_PUBLIC_EVAL': '1'}, stdout=log,
                                       stderr=subprocess.STDOUT)
            self.processes.append(process)
            try:
                code = process.wait(timeout=max(1.0, self.deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                record['status'] = 'deadline_during_evaluation'
                return {'ok': False, 'message': 'The budget ran out during the evaluation.'}
            report_path = directory / 'report/report.json'
            if code or not report_path.exists():
                record.update(status='evaluation_error', exit_code=code)
                self._resume(None, n)
                return {'ok': False, 'message': f'Test {n} failed on the evaluator side (not a verdict on your '
                                                f'submission); you may retry.'}
            report = json.loads(report_path.read_text())
            record.update(status=report['status'], score=report.get('score'), finished_at_seconds=self._now(),
                          cases=len(report.get('cases') or []))
            if not self._resume(directory / 'report', n):
                return {'ok': False, 'message': 'The attempt has ended.'}
            return {'ok': True, 'message': self._summary(report, n)}
        finally:
            self.busy.release()

    def _resume(self, report: Path | None, n: int) -> bool:
        with self.container:
            if self.ended.is_set():
                return False
            self.docker.run('unpause', self.name)
        if report is not None:
            target = f'/workspace/feedback/test-{n:02d}'
            self.docker.run('exec', '--user', f'{DEV_UID}:{DEV_UID}', self.name, '/bin/mkdir', '-p', target)
            payload = subprocess.run(['tar', '-C', str(report), '-cf', '-', 'report.md', 'report.json'],
                                     capture_output=True, check=True).stdout
            self.docker.run('exec', '-i', '--user', f'{DEV_UID}:{DEV_UID}', '--workdir', target, self.name,
                            '/bin/tar', '--no-same-owner', '-xf', '-', input=payload, timeout=60)
        return True

    def _summary(self, report: dict, n: int) -> str:
        score = report.get('score')
        lines = [f'Test {n}: status {report["status"]}, public-set score '
                 f'{"none" if score is None else f"{score:.1f}"} (100-point scale, baseline = 50), '
                 f'{len(report.get("cases") or [])} cases' + (' (subset)' if report.get('subset') else '') + '.']
        if self.task == 'conic' and report.get('success_rate') is not None:
            lines.append(f'Solved {report["success_rate"]:.3f} of them; the baseline solved '
                         f'{report["baseline_success_rate"]:.3f}.')
            lines += [f'  {g}: {v["solved"]}/{v["cases"]} (baseline {v["baseline_solved"]})'
                      for g, v in sorted(report['groups'].items())]
        elif report.get('failed_cases'):
            lines.append(f'{len(report["failed_cases"])} failed cases (any failure scores the formal submission 0).')
        if report.get('not_evaluated'):
            lines.append(f'{len(report["not_evaluated"])} cases were not evaluated before the {TEST_CAP_SECONDS // 60}-minute '
                         f'cap; the score covers the evaluated cases only.')
        lines.append(f'Details: /workspace/feedback/test-{n:02d}/report.md. Remaining budget: '
                     f'{max(0, int(self.deadline - time.monotonic()))} s.')
        return '\n'.join(lines)


class EvalServiceV63:
    """v6.3 iterative setting: /opt/frontieram/bin/submit over a bind-mounted socket, without stopping the attempt.

    A test snapshots /workspace/submission from the running container (reviewed exporter, no pause) and returns at
    once; the snapshot is graded on the public set in the background on one grade slot, ahead of held-out grading
    (`bench public-eval-v63`), and the report is copied into /workspace/feedback/test-<n>/. One test at a time, each
    capped. A final request ends the attempt. Evaluations still running when the attempt ends finish unread.
    """

    def __init__(self, task_name, docker, controller, name: str, output: Path, started: float, deadline: float):
        self.task, self.docker, self.controller, self.name = task_name, docker, controller, name
        self.output, self.started, self.deadline = output / 'evaluations', started, deadline
        self.final, self.ended = threading.Event(), threading.Event()
        self.busy, self.container = threading.Lock(), threading.Lock()
        self.records: list[dict] = []
        self.threads: list[threading.Thread] = []
        self.server = None

    start = EvalService.start
    _serve = EvalService._serve
    _handle = EvalService._handle
    _now = EvalService._now
    _final = EvalService._final

    def end(self) -> None:
        """No report may be copied into the container after this (develop() pauses it next)."""
        self.ended.set()
        with self.container:
            pass
        if self.server is not None:
            self.server.close()

    def drain(self, timeout: float = TEST_CAP_SECONDS + 600) -> None:
        """Wait for the last evaluation, so the attempt's record holds every test's outcome."""
        for thread in self.threads:
            thread.join(timeout=timeout)

    def _remaining(self) -> int:
        return max(0, int(self.deadline - time.monotonic()))

    def _status(self) -> dict:
        tests = [r for r in self.records if r['kind'] == 'test']
        lines = [f'Time left: {self._remaining()} s. Tests so far: {len(tests)}.']
        for r in tests[-5:]:
            state = r.get('status', 'running')
            score = '' if r.get('score') is None else f', public-set score {r["score"]:.1f}'
            lines.append(f'  test {r["n"]} (submitted at {r["requested_at_seconds"]:.0f} s): {state}{score}')
        return {'ok': True, 'message': '\n'.join(lines)}

    def _test(self, request: dict) -> dict:
        if not self.busy.acquire(blocking=False):
            running = next(r for r in reversed(self.records) if r['kind'] == 'test')
            return {'ok': False, 'message': f'Test {running["n"]} is still being evaluated; one test at a time '
                                            f'(`submit --status` shows its state).'}
        release = True
        try:
            n = sum(r['kind'] == 'test' for r in self.records) + 1
            record = {'kind': 'test', 'n': n, 'requested_at_seconds': self._now()}
            self.records.append(record)
            if self.ended.is_set():
                record['status'] = 'refused_attempt_ended'
                return {'ok': False, 'message': 'The attempt has ended.'}
            directory = self.output / f'test-{n:02d}'
            try:
                with self.container:
                    manifest = freeze(self.controller, self.docker, self.name, directory, remove=False, pause=False)
                record['files'] = len(manifest['files'])
            except self.controller.SubmissionFailure as error:
                record.update(status='invalid_submission', score=0.0, finished_at_seconds=self._now())
                report = {'task': self.task, 'status': 'invalid_submission', 'score': 0.0, 'reason': str(error)[:300],
                          'categories': {}}
                self._deliver(directory / 'report', report, n)
                return {'ok': True, 'message': f'Test {n}: /workspace/submission cannot be frozen as a submission: '
                                               f'{str(error)[:300]}'}
            except Exception as error:
                record.update(status='snapshot_failed', reason=f'{type(error).__name__}: {str(error)[:300]}')
                return {'ok': False, 'message': f'Test {n}: the snapshot failed (files may have changed while it was '
                                                f'taken); not a verdict on your submission, you may retry.'}
            thread = threading.Thread(target=self._evaluate, args=(record, directory), daemon=True)
            self.threads.append(thread)
            thread.start()
            release = False
            return {'ok': True, 'message': f'Test {n} submitted at {record["requested_at_seconds"]:.0f} s and is being '
                                           f'evaluated in the background (at most {TEST_CAP_SECONDS // 60} minutes). '
                                           f'The report will appear in /workspace/feedback/test-{n:02d}/report.md; '
                                           f'`submit --status` shows its state. Time left: {self._remaining()} s.'}
        finally:
            if release:
                self.busy.release()

    def _evaluate(self, record: dict, directory: Path) -> None:
        n = record['n']
        try:
            command = [sys.executable, '-m', 'arena.cli', 'public-eval-v63', '--task', self.task,
                       '--source', str(directory / 'source'), '--out', str(directory / 'report'), '--cap', str(TEST_CAP_SECONDS), '--label', f'{self.output.parent.name} test {n}']
            with (directory / 'public-eval.log').open('wb') as log:
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=TEST_CAP_SECONDS + 600).returncode
            path = directory / 'report/report.json'
            report = json.loads(path.read_text()) if path.exists() else None
            if code or report is None:
                report = {'task': self.task, 'status': 'unmeasured', 'score': None, 'categories': {},
                          'reason': 'evaluator-side failure; you may submit another test'}
                record['exit_code'] = code
            record.update(status=report['status'], score=report.get('score'), finished_at_seconds=self._now())
            self._deliver(directory / 'report', report, n, write=code != 0 or not path.exists())
        except Exception as error:
            record.update(status='evaluation_error', reason=f'{type(error).__name__}: {str(error)[:300]}',
                          finished_at_seconds=self._now())
        finally:
            self.busy.release()

    def _deliver(self, folder: Path, report: dict, n: int, write: bool = True) -> None:
        """Write report.json/report.md (if not written by the evaluator) and copy them into the running container."""
        from .feedback import render

        folder.mkdir(parents=True, exist_ok=True)
        if write or not (folder / 'report.md').exists():
            (folder / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
            (folder / 'report.md').write_text(render(report))
        with self.container:
            if self.ended.is_set():
                return
            target = f'/workspace/feedback/test-{n:02d}'
            self.docker.run('exec', '--user', f'{DEV_UID}:{DEV_UID}', self.name, '/bin/mkdir', '-p', target)
            payload = subprocess.run(['tar', '-C', str(folder), '-cf', '-', 'report.md', 'report.json'],
                                     capture_output=True, check=True).stdout
            self.docker.run('exec', '-i', '--user', f'{DEV_UID}:{DEV_UID}', '--workdir', target, self.name,
                            '/bin/tar', '--no-same-owner', '-xf', '-', input=payload, timeout=60)


def develop(spec: dict, runtime: dict) -> dict:
    """One v6 attempt; returns the same summary fields as a Nexus v4/v5 attempt."""
    task = TASKS[spec['task']]
    model = spec['model']
    harness = HARNESS[model]
    placement = Placement(**spec['placement'])
    output = Path(spec['output'])
    frozen_dir = output / 'task'
    frozen_dir.mkdir(parents=True, exist_ok=True)
    script = spec.get('agent_script')  # validation only: a shell script stands in for the model and its harness
    route = {'script': True} if script else resolve_route(model, harness)
    protocol = spec['protocol']
    setting = spec.get('setting') or 'single'
    seconds = spec.get('development_seconds') or (V63_SECONDS[setting] if protocol == 'v6.3' else 7200)
    if protocol == 'v6.2' and setting == 'iterative':
        raise ValueError('the v6.2 iterative setting is retired; use --protocol v6.3 (results/PROTOCOL_V63.md)')
    iterative = protocol == 'v6.3' and setting == 'iterative'
    if iterative:
        from .public_v63 import PUBLIC, baseline_path
        if task.name not in PUBLIC or not baseline_path(task.name).exists():
            raise ValueError(f'{task.name} has no v6.3 public baseline (bench public-baseline-v63)')
    assets = task.assets(runtime)
    docker = assets.controller.Docker(runtime['docker_path'], placement.docker_host, placement.cpuset,
                                      placement.cgroup_parent)
    write(output / 'v6_conditions.json', {'harness': harness, 'route': route, 'image': HARNESS_IMAGES[task.name],
                                          'development_seconds': seconds, 'memory_bytes': MEMORY,
                                          'protocol': protocol,
                                          'setting': setting if protocol in ('v6.2', 'v6.3') else None,
                                          'test_cap_seconds': TEST_CAP_SECONDS if iterative else None})
    credential_env, credential_file = CREDENTIALS.get(model, DEFAULT_CREDENTIAL)
    token = secrets.token_urlsafe(32)
    port = _free_port()
    socket_dir = Path('/run/frontieram/atr') / output.name
    socket_dir.mkdir(parents=True, exist_ok=False)
    socket_dir.chmod(0o777)
    name = f'frontieram-v6-{secrets.token_hex(8)}'
    processes: list[subprocess.Popen] = []
    summary: dict = {'harness': harness, 'route': route, 'protocol': protocol, 'setting': setting}
    service = None
    attempt_marker = contextlib.ExitStack()
    mounts = []
    if iterative:
        from .cli import context

        attempt_marker.enter_context(context()[1].iterative_attempt(f'v6.3 iterative {output.name}'))
        eval_dir = Path('/run/frontieram/eval') / output.name
        eval_dir.mkdir(parents=True, exist_ok=False)
        eval_dir.chmod(0o777)
        client_dir = output / 'client'
        (client_dir / 'bin').mkdir(parents=True)
        (client_dir / 'bin/submit').write_text(SUBMIT_CLIENT_V63)
        (client_dir / 'bin/submit').chmod(0o755)
        mounts = ['--mount', f'type=bind,src={eval_dir},dst=/run/frontieram-eval',
                  '--mount', f'type=bind,src={client_dir},dst=/opt/frontieram,readonly']
    started = time.monotonic()
    try:
        if not script:
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
            '--mount', f'type=bind,src={socket_dir},dst=/run/atr', *mounts,
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
            'prompt': prompt(task.name, runtime, protocol, setting, seconds), 'client_timeout_seconds': seconds, **route,
        }
        (output / 'prompt.txt').write_text(launcher_spec['prompt'])
        deadline = started + seconds
        if iterative:
            service = EvalServiceV63(task.name, docker, assets.controller, name, output, started, deadline)
            service.start(eval_dir / 'eval.sock')
        launch = (['/bin/sh', '-c', script] if script else
                  ['/opt/atr-venv/bin/python', '-c', LAUNCHER, json.dumps(launcher_spec)])
        with (output / 'cli.log').open('wb') as log:
            client = subprocess.Popen([*docker.prefix, 'exec', name, *launch],
                                      stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            while True:
                try:
                    code = client.wait(timeout=1.0)
                    stop_reason = 'turn_completed' if code == 0 else 'harness_exited'
                    break
                except subprocess.TimeoutExpired:
                    formal = service is not None and service.final.is_set()
                    if formal or time.monotonic() >= deadline:
                        if service is not None:
                            service.end()
                        docker.pause(name)
                        client.kill()
                        client.wait(timeout=30)
                        code, stop_reason = None, 'formal_submission' if formal else 'development_deadline'
                        break
        stopped = time.monotonic()
        if service is not None:
            service.end()
        docker.pause(name)
        summary.update(exit_code=code, agent_stop_reason=stop_reason, development_seconds=stopped - started)
        eligible = stop_reason in ('turn_completed', 'development_deadline', 'formal_submission')
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
        if service is not None:
            summary['evaluations'] = service.records
            summary['test_submissions'] = sum(r['kind'] == 'test' for r in service.records)
            service.drain()
        attempt_marker.close()
    records = list((output / 'atr').rglob('events.jsonl'))
    requests = sum(1 for path in records for line in path.read_text().splitlines() if '"request"' in line)
    summary.update(task_status=summary.get('task_status', status), http_attempts=requests,
                   agent_eligible=status in ('verification_completed', 'invalid_submission'))
    if summary.get('source_frozen') and not (frozen_dir / 'source.tar').exists():
        summary.update(task_status='unmeasured', error='frozen archive missing')
    write(output / 'develop_result.json', summary)
    return summary
