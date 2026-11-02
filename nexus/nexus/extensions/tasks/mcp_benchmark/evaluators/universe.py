from __future__ import annotations

import json

from typing import Any

from nexus.extensions.tasks.mcp_benchmark.types import EvalResult
from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.executors.file import FileItem


class UniverseRuntimeEvaluator:
    async def evaluate(
        self,
        *,
        runtime,
        helper_script: str,
        input_payload: dict[str, Any],
        work_dir: str,
        timeout_s: float,
    ) -> EvalResult:
        input_path = f'{work_dir}/universe_eval_input.json'
        output_path = f'{work_dir}/universe_eval_output.json'
        await runtime.upload_file([
            FileItem(path=input_path, content=json.dumps(input_payload, ensure_ascii=False, indent=2), encoding='utf8')
        ])
        result = await runtime.run_command(
            command=f'python3 {helper_script} --input {input_path} --output {output_path}',
            timeout=timeout_s,
        )
        if result.status != CommandRunStatus.Finished or result.return_code != 0:
            return EvalResult(
                passed=False,
                score=0.0,
                details={
                    'status': result.status.value,
                    'return_code': result.return_code,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                },
            )

        output = await runtime.download_file([FileItem(path=output_path)])
        if not output.files:
            return EvalResult(
                passed=False,
                score=0.0,
                details={
                    'error': 'universe evaluator produced no output',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                },
            )
        payload = json.loads(output.files[0].content or '{}')
        return EvalResult.model_validate(payload)
