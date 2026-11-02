"""RewardKit OpenCode backend with persistent JSONL trajectory capture."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import uuid

from pathlib import Path

from rewardkit.agents import AgentCLI, register_agent


class OpenCodeCLI(AgentCLI):
    name = 'opencode'
    cli_name = 'opencode'
    install_script = 'opencode --version'

    def __init__(self) -> None:
        self._trace_id = uuid.uuid4().hex
        self._attempt = 0
        trace_root = os.environ.get('REWARDKIT_OPENCODE_TRACE_DIR', '/logs/verifier/opencode_judge')
        self._trace_dir = Path(trace_root)
        self._trace_dir.mkdir(parents=True, exist_ok=True)

    def build_command(self, prompt, schema, allowed_tools=()):
        del allowed_tools
        request_path = self._trace_dir / f'{self._trace_id}.request.json'
        stdout_path = self._trace_dir / f'{self._trace_id}.latest.jsonl'
        stderr_path = self._trace_dir / f'{self._trace_id}.latest.stderr.log'
        request_path.write_text(
            json.dumps({'prompt': prompt, 'schema': schema}, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        prompt = (
            prompt
            + '\n\nYour final message must be exactly one JSON object matching this '
            + 'JSON Schema, with no surrounding prose or code fences:\n'
            + json.dumps(schema)
        )
        model = os.environ.get('REWARDKIT_MODEL', '').strip()
        variant = os.environ.get('REWARDKIT_OPENCODE_VARIANT', 'high').strip()
        argv = [
            'opencode',
            'run',
            '--format=json',
            '--dangerously-skip-permissions',
            '--agent',
            'build',
            '--thinking',
            '--title',
            'rewardkit-judge',
        ]
        if model:
            argv.append(f'--model={model}')
        if variant:
            argv.extend(['--variant', variant])
        argv.append(prompt)
        shell_command = ' '.join(shlex.quote(part) for part in argv)
        # OpenCode stores session state in a SQLite database under
        # XDG_DATA_HOME. Give every invocation (including RewardKit retries)
        # its own directory so concurrent judges cannot contend on one DB.
        data_dir = tempfile.mkdtemp(prefix='opencode-judge-')
        # tee preserves the exact OpenCode event stream even when the CLI exits
        # non-zero and RewardKit therefore never calls parse_output().
        return [
            'bash',
            '-lc',
            (
                'set -o pipefail; '
                f'XDG_DATA_HOME={shlex.quote(data_dir)} '
                f'python3 /tests/judge_runtime.py stage opencode:{self._trace_id} {shell_command} '
                f'2> >(tee {shlex.quote(str(stderr_path))} >&2) '
                f'| tee {shlex.quote(str(stdout_path))}'
            ),
        ]

    def model_args(self, model):
        del model
        # build_command reads REWARDKIT_MODEL directly so the command can be
        # wrapped with tee while preserving OpenCode's exit status.
        return []

    def parse_output(self, raw):
        self._attempt += 1
        (self._trace_dir / f'{self._trace_id}.attempt-{self._attempt:02d}.jsonl').write_text(
            raw,
            encoding='utf-8',
        )
        texts = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('type') == 'text':
                text = event.get('part', {}).get('text', '')
                if text:
                    texts.append(text)
        return texts[-1] if texts else raw


register_agent(OpenCodeCLI)
