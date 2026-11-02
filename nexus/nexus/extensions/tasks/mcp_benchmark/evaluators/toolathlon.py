from __future__ import annotations

import json
import shlex

from nexus.extensions.tasks.mcp_benchmark.types import EvalResult
from nexus.runtime.executors.command import CommandRunStatus


class ToolathlonRuntimeEvaluator:
    async def evaluate(
        self,
        *,
        runtime,
        command: str,
        cwd: str | None,
        env: dict[str, str],
        timeout_s: float,
        reward_json_path: str | None = None,
    ) -> EvalResult:
        if reward_json_path is not None:
            quoted_reward_path = shlex.quote(reward_json_path)
            cleanup = await runtime.run_command(
                command=f'rm -f -- {quoted_reward_path}',
                timeout=10.0,
            )
            if cleanup.status != CommandRunStatus.Finished or cleanup.return_code != 0:
                raise RuntimeError(
                    f'failed to reset evaluator reward file {reward_json_path}: '
                    f'{(cleanup.stderr or cleanup.stdout or "")[:2000]}'
                )

        result = await runtime.run_command(command=command, cwd=cwd, extra_env=env, timeout=timeout_s)
        command_passed = result.status == CommandRunStatus.Finished and result.return_code == 0
        score = 1.0 if command_passed else 0.0
        reward_error: str | None = None

        # Only evaluator types that explicitly opt into the Harbor continuous
        # reward contract may read reward.json. Native Toolathlon scripts use
        # their process exit code and never inspect this agent-writable path.
        if reward_json_path is not None:
            reward_result = await runtime.run_command(
                command=f'cat -- {quoted_reward_path}',
                timeout=10.0,
            )
            try:
                if reward_result.status != CommandRunStatus.Finished or reward_result.return_code != 0:
                    raise ValueError((reward_result.stderr or 'reward file missing').strip())
                payload = json.loads((reward_result.stdout or '').strip())
                reward = payload.get('reward')
                if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                    raise ValueError('reward must be a number')
                score = max(0.0, min(1.0, float(reward)))
            except Exception as exc:
                # Continuous evaluators are defined by reward.json. Missing or
                # malformed output must fail closed instead of treating an exit
                # code 0 as a perfect score.
                score = 0.0
                reward_error = str(exc)

        passed = score >= 1.0
        details = {
            'status': result.status.value,
            'return_code': result.return_code,
            'stdout': (result.stdout or '')[:4000],
            'stderr': (result.stderr or '')[:4000],
        }
        if reward_json_path is not None:
            details['reward_json_path'] = reward_json_path
        if reward_error is not None:
            details['reward_error'] = reward_error
        return EvalResult(
            passed=passed,
            score=score,
            details=details,
        )
