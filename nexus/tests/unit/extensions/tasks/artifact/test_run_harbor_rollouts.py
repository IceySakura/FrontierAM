from __future__ import annotations

import json
import os
import subprocess
import sys

from pathlib import Path

import tomllib


REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / 'nexus/extensions/tasks/artifact/run_harbor_rollouts.sh'


def test_expansion_writes_render_token_before_appending_missing_metadata(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    task = source / 'sample'
    task.mkdir(parents=True)
    (task / 'task.toml').write_text(
        'version = "T3-RL-v1.0"\n[verifier.env]\nT3_JUDGE_NAME = "web_judger"\nT3_INSTANCE_ID = "source-id"\n',
        encoding='utf-8',
    )

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    uv.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'args = sys.argv[1:]\n'
        'assert args[:3] == ["run", "--with", "harbor==0.20.0"]\n'
        'command = args[3:]\n'
        'if command[0] == "python":\n'
        '    sys.stdin.read()\n'
        '    sys.exit(0)\n'
        'assert command[:3] == ["nexus", "run", "batch"]\n'
        'Path(os.environ["CAPTURE_ARGS"]).write_text(json.dumps(command))\n',
        encoding='utf-8',
    )
    uv.chmod(0o755)

    out_root = tmp_path / 'out'
    env = {
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'NEXUS_REPO': str(REPO),
        'NEXUS_AUTO_INSTALL': '0',
        'PYTHONDONTWRITEBYTECODE': '1',
        'OUT_ROOT': str(out_root),
        'RUN_NAME': 'case',
        'ROLLOUTS': '1',
        'CONCURRENT': '1',
        'WORKER_PROCESSES': '1',
        'EXPAND_JOBS': '1',
        'OPENAI_API_KEY': 'fake-inference-key',
        'JUDGE_API_KEY': 'fake-judge-key',
        'GONGFENG_RUNTIME_TOKEN': 'fake-runtime-token',
        'CAPTURE_ARGS': str(tmp_path / 'args.json'),
    }

    result = subprocess.run(
        ['bash', str(SCRIPT), str(source)],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    expanded = out_root / 'case/tasks/sample__r0/task.toml'
    text = expanded.read_text(encoding='utf-8')
    config = tomllib.loads(text)
    assert config['verifier']['env']['GONGFENG_RUNTIME_TOKEN'] == '${GONGFENG_RUNTIME_TOKEN}'
    assert config['metadata'] == {
        'instance_id': 'sample__r0',
        'source_instance_id': 'sample',
        'rollout_index': 0,
    }
    assert text.index('GONGFENG_RUNTIME_TOKEN') < text.index('[metadata]')
    assert json.loads((tmp_path / 'args.json').read_text())[0:3] == ['nexus', 'run', 'batch']
