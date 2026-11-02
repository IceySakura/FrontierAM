"""Emit an experimental Harbor directory-per-instance artifact for a graded SWE-Any instance.

WARNING — DO NOT USE THIS FORMAT YET. The SWE-Any Harbor adaptation is incomplete and still under
testing; its generated directories are not a supported evaluation input.

``harbor_bridge`` boots the prebuilt ``docker_image`` and
does NOT build a Harbor ``Dockerfile`` (it discards its ``RUN``/``COPY``), so the bug state baked
into ``environment/Dockerfile`` below is never applied — a task loaded through the bridge would ship
the feature already implemented. The path actually consumed today is ``swe_any_instances.jsonl`` +
the ``swe-any`` ``DatasetHandler`` (``nexus/extensions/tasks/swe/swe_any/``), which reconstructs the
bug hole at runtime init. Treat this Harbor emission as a best-effort artifact pending native Harbor
image-build support; when that lands, move the bug ``git apply`` out of the Dockerfile into a step
the bridge runs (e.g. the top of ``tests/test.sh``).

Export writes the layout itself:

    <harbor_root>/
      dataset.toml                      # shared dataset metadata
      <instance_id>/
        task.toml                       # [task]/[environment]/[agent]/[verifier]
        instruction.md                  # the problem_statement (all the solver sees)
        environment/Dockerfile + *.patch  # FROM image, apply BUG state (feature_mask + test_mask), reinit git
        solution/{solve.sh, feature_gold.patch}   # oracle: restore the implementation
        tests/{test.sh, test_gold.patch, test_runner.sh, verify_results.py, config.json}

Patch bodies are written as sibling ``*.patch`` files and ``git apply``-ed by the scripts (robust vs.
embedding multi-line diffs in shell/Docker). Empty patches are skipped.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any


# Drop git history after the bug state is applied, so a solver cannot recover the reference
# implementation (or the original PR) by walking back revisions — collapse the masked working tree
# into a single "Initial commit". Mirrors ``swe/constants.py::GIT_REINIT_SCRIPT`` used by the
# consumed dataset-handler path (kept inline so swe_any does not import the ``swe`` extension).
_GIT_REINIT_RUN = (
    'RUN set -e; \\\n'
    '    find . -name .git -exec rm -rf {} + ; \\\n'
    '    find . -name .gitmodules -type f -exec rm -f {} + ; \\\n'
    '    git config --global init.defaultBranch main; \\\n'
    '    git config --global --add safe.directory "$(pwd)"; \\\n'
    '    git init; \\\n'
    '    git config --local user.name "nexus"; \\\n'
    '    git config --local user.email "nexus@anonymous.com"; \\\n'
    '    git config --local commit.gpgsign false; \\\n'
    '    git add -A; \\\n'
    '    git commit --no-verify --allow-empty -m "Initial commit"'
)

_VERIFY_STRUCTURED_RESULTS = """\
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
result_path = Path(sys.argv[2])
runner_rc = int(sys.argv[3])
reward_path = Path('/logs/verifier/reward.txt')
reward = 0
try:
    records = [json.loads(line) for line in result_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    tests = {}
    summaries = []
    valid_statuses = {'PASSED', 'FAILED', 'ERROR', 'SKIPPED', 'XFAIL', 'XPASS'}
    for index, record in enumerate(records):
        if record.get('type') == 'test':
            test_id, status = record.get('id'), record.get('status')
            if not isinstance(test_id, str) or not test_id or test_id in tests or status not in valid_statuses:
                raise ValueError('invalid/duplicate test record')
            tests[test_id] = status
        elif record.get('type') == 'summary':
            if index != len(records) - 1:
                raise ValueError('summary must be final')
            summaries.append(record)
        else:
            raise ValueError('unknown record type')
    expected = list(config.get('FAIL_TO_PASS') or []) + list(config.get('PASS_TO_PASS') or [])
    if (
        runner_rc == 0
        and len(summaries) == 1
        and summaries[0].get('status') == 'completed'
        and summaries[0].get('count') == len(tests)
        and config.get('FAIL_TO_PASS')
        and all(tests.get(test_id) == 'PASSED' for test_id in expected)
    ):
        reward = 1
except Exception as exc:
    print(f'invalid SWE-Any runner results: {exc}', file=sys.stderr)
reward_path.write_text(f'{reward}\\n', encoding='utf-8')
"""


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    if executable:
        path.chmod(0o755)


_TOML_CTRL = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\x08': '\\b', '\x0c': '\\f'}


def _toml_escape(s: str) -> str:
    """Escape a TOML basic-string value: ``\\`` / ``"`` AND C0 control chars.

    A raw newline/CR/NUL in e.g. ``image`` or ``workdir`` would otherwise split the value mid-line;
    ``harbor_bridge``'s ``tomllib.load`` then raises and aborts loading the WHOLE dataset (not just
    the one bad task), so reject the controls here rather than emit an unparseable ``task.toml``.
    """
    out: list[str] = []
    for ch in s:
        if ch == '\\':
            out.append('\\\\')
        elif ch == '"':
            out.append('\\"')
        elif ch in _TOML_CTRL:
            out.append(_TOML_CTRL[ch])
        elif ch < ' ':  # any other C0 control (NUL, etc.) — forbidden raw in a TOML basic string
            out.append(f'\\u{ord(ch):04X}')
        else:
            out.append(ch)
    return ''.join(out)


def _toml_str_list(items: list[str]) -> str:
    return '[' + ', '.join(f'"{_toml_escape(i)}"' for i in items) + ']'


def write_dataset_toml(harbor_root: Path, *, name: str = 'swe_any', version: str = '0.1.0') -> None:
    """Write the shared ``dataset.toml`` at the Harbor root (idempotent)."""
    _write(
        harbor_root / 'dataset.toml',
        f'[dataset]\nname = "{_toml_escape(name)}"\nversion = "{version}"\n'
        'description = "swe_any feature-implementation tasks (auto-constructed)."\n',
    )


def write_harbor_instance(
    harbor_root: Path,
    swebench: dict[str, Any],
    *,
    agent_timeout: float = 1800.0,
    verifier_timeout: float = 1800.0,
    build_timeout: float = 1800.0,
) -> Path:
    """Write one experimental ``<harbor_root>/<instance_id>/`` task tree.

    The Harbor adaptation is incomplete and under test; callers must not treat this artifact as a
    supported evaluation input. Returns the instance directory.
    """
    instance_id = str(swebench['instance_id'])
    workdir = str(swebench.get('workdir') or '/testbed')
    image = str(swebench.get('image') or '')
    inst_dir = harbor_root / instance_id

    fail_to_pass = list(swebench.get('FAIL_TO_PASS') or [])
    pass_to_pass = list(swebench.get('PASS_TO_PASS') or [])
    test_files = list(swebench.get('test_files') or [])
    test_runner_script = str(swebench.get('test_runner_script') or '').strip()
    if not test_runner_script:
        raise ValueError('SWE-Any Harbor export requires test_runner_script')

    # --- task.toml ---
    _write(
        inst_dir / 'task.toml',
        '\n'.join([
            'version = "1.0"',
            '',
            '[task]',
            f'name = "{_toml_escape(instance_id)}"',
            '',
            '[metadata]',
            'category = "programming"',
            'tags = ["swe_any", "feature-implementation"]',
            '',
            '[environment]',
            f'docker_image = "{_toml_escape(image)}"',
            f'workdir = "{_toml_escape(workdir)}"',
            f'build_timeout_sec = {build_timeout}',
            '',
            '[agent]',
            f'timeout_sec = {agent_timeout}',
            '',
            '[verifier]',
            f'timeout_sec = {verifier_timeout}',
            '',
        ]),
    )

    # --- instruction.md ---
    _write(inst_dir / 'instruction.md', (swebench.get('problem_statement') or '').rstrip() + '\n')

    # --- environment/ : bake the BUG state into the image, then drop git history ---
    env_dir = inst_dir / 'environment'
    dockerfile = [f'FROM {image}', f'WORKDIR {workdir}']
    bug_patches = [
        ('feature_mask.patch', swebench.get('feature_mask_patch') or ''),
        ('test_mask.patch', swebench.get('test_mask_patch') or ''),
    ]
    present_bug = [(fname, body) for fname, body in bug_patches if body.strip()]
    for fname, body in present_bug:
        _write(env_dir / fname, body if body.endswith('\n') else body + '\n')
    if present_bug:
        copied = ' '.join(fname for fname, _ in present_bug)
        dockerfile.append(f'COPY {copied} /tmp/')
        applies = ' && '.join(f'git apply --ignore-whitespace /tmp/{fname}' for fname, _ in present_bug)
        dockerfile.append(f'RUN {applies}')
    dockerfile.append(_GIT_REINIT_RUN)  # collapse to a single commit — no recoverable history
    _write(env_dir / 'Dockerfile', '\n'.join(dockerfile) + '\n')

    # --- solution/ : oracle restores the implementation ---
    _write(inst_dir / 'solution' / 'feature_gold.patch', _ensure_nl(swebench.get('patch') or ''))
    _write(
        inst_dir / 'solution' / 'solve.sh',
        '\n'.join([
            '#!/bin/bash',
            # No `set -u`: conda's activate dereferences vars unset in a non-interactive shell, which
            # would abort before the command runs (swe_env strips it too via postprocess_eval_script).
            'set -xo pipefail',
            f'cd {_sh_quote(workdir)}',
            'git apply --ignore-whitespace /solution/feature_gold.patch',
            '',
        ]),
        executable=True,
    )

    # --- tests/ : reveal the oracle (apply the forward gold tests), run it, write reward ---
    tests_dir = inst_dir / 'tests'
    _write(tests_dir / 'test_gold.patch', _ensure_nl(swebench.get('test_gold_patch') or ''))
    _write(
        tests_dir / 'config.json',
        json.dumps(
            {
                'FAIL_TO_PASS': fail_to_pass,
                'PASS_TO_PASS': pass_to_pass,
                'test_files': test_files,
                'test_runner_protocol': swebench.get('test_runner_protocol'),
            },
            indent=2,
        )
        + '\n',
    )
    reveal = []
    if (swebench.get('test_gold_patch') or '').strip():
        reveal.append('git apply --ignore-whitespace /tests/test_gold.patch || true')
    _write(tests_dir / 'test_runner.sh', _ensure_nl(test_runner_script), executable=True)
    _write(tests_dir / 'verify_results.py', _VERIFY_STRUCTURED_RESULTS)
    run_oracle = [
        'results=/logs/verifier/test_results.jsonl',
        'rm -f "$results"',
        (
            f'SWE_ANY_REPO_DIR={_sh_quote(workdir)} '
            f'SWE_ANY_RESULTS_PATH="$results" '
            f'SWE_ANY_RUNNER_PROTOCOL={int(swebench.get("test_runner_protocol") or 1)} '
            f'SWE_ANY_TEST_FILES_JSON={_sh_quote(json.dumps(test_files, separators=(",", ":")))} '
            'bash /tests/test_runner.sh'
        ),
        'runner_rc=$?',
        'if command -v python3 >/dev/null 2>&1; then verifier_python=python3; else verifier_python=python; fi',
        '"$verifier_python" /tests/verify_results.py /tests/config.json "$results" "$runner_rc"',
    ]
    _write(
        tests_dir / 'test.sh',
        '\n'.join([
            '#!/bin/bash',
            'set -xo pipefail',  # no `set -u` — see solve.sh (conda activate needs unset vars)
            'mkdir -p /logs/verifier',
            f'cd {_sh_quote(workdir)}',
            *reveal,
            *run_oracle,
            '',
        ]),
        executable=True,
    )
    return inst_dir


def _ensure_nl(text: str) -> str:
    return text if (not text or text.endswith('\n')) else text + '\n'


def _sh_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)
