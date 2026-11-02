"""rubric 最终 reward 脚本测试。"""

import asyncio
import json

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.aidata import aidata as aidata_cli
from nexus.extensions.tasks.aidata.rubric.reward import RewardScriptError, resolve_reward, validate_reward_config


class StubRuntime:
    def __init__(self, stdout: str = '0', *, return_code: int = 0, status: str = 'Finished') -> None:
        self.stdout = stdout
        self.return_code = return_code
        self.status = status
        self.calls: list[dict[str, Any]] = []

    async def run_command(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({'command': command, **kwargs})
        return SimpleNamespace(
            stdout=self.stdout,
            stderr='',
            return_code=self.return_code,
            status=self.status,
        )


class TimeoutRuntime:
    async def run_command(self, command: str, **kwargs: Any) -> SimpleNamespace:
        raise asyncio.TimeoutError


def _results(*, passed: int, total: int = 30) -> list[dict[str, Any]]:
    return [
        {
            'id': f'check_{i + 1}',
            'passed': i < passed,
            'verdict': 'Y' if i < passed else 'N',
            'weight': 1,
            'dimension': 'correctness',
        }
        for i in range(total)
    ]


def _summary(*, passed: int, total: int = 30, hard_gate_triggered: bool = False) -> dict[str, Any]:
    weighted = passed / total
    return {
        'score': weighted,
        'raw_score': weighted,
        'dimension_scores': {'correctness': weighted},
        'passed': passed,
        'total': total,
        'hard_gate_failed': [],
        'hard_gate_triggered': hard_gate_triggered,
    }


def test_reward_script_can_turn_twenty_of_thirty_checks_into_one() -> None:
    runtime = StubRuntime(stdout='1\n')
    rubric = {'version': '4.2', 'config': {'reward': {'script': 'reward.py'}}, 'checks': []}
    rows = _results(passed=20)

    outcome = asyncio.run(
        resolve_reward(
            rubric,
            rows,
            _summary(passed=20),
            runtime=runtime,
            cwd='/workspace/.aidata/case',
            state_summary={'orders': 2},
        )
    )

    assert outcome.score == 1.0
    assert outcome.continuous_score == 1.0
    assert outcome.policy == {'mode': 'script', 'script': 'reward.py', 'timeout_sec': 15.0}
    assert runtime.calls[0]['command'] == 'python3 reward.py'
    assert runtime.calls[0]['cwd'] == '/workspace/.aidata/case'
    assert runtime.calls[0]['timeout'] == 15.0
    payload = json.loads(runtime.calls[0]['stdin'])
    assert payload == {
        'version': '1.0',
        'rubric_results': rows,
        'score_summary': _summary(passed=20),
        'state_summary': {'orders': 2},
    }


def test_reward_script_accepts_json_score_output() -> None:
    runtime = StubRuntime(stdout='{"score": 0.37}\n')
    rubric = {
        'version': '4.2',
        'config': {'reward': {'script': 'scoring/reward.sh', 'timeout_sec': 3}},
        'checks': [],
    }

    outcome = asyncio.run(resolve_reward(rubric, _results(passed=1), _summary(passed=1), runtime=runtime, cwd='/case'))

    assert outcome.score == 0.37
    assert outcome.continuous_score == 0.37
    assert runtime.calls[0]['command'] == 'bash scoring/reward.sh'
    assert runtime.calls[0]['timeout'] == 3.0


def test_reward_script_cannot_bypass_a_failed_hard_gate() -> None:
    runtime = StubRuntime(stdout='0.9')
    rubric = {
        'version': '4.2',
        'config': {'hard_gate_cap': 0.25, 'reward': {'script': 'reward.py'}},
        'checks': [],
    }
    summary = _summary(passed=25, hard_gate_triggered=True)
    summary['hard_gate_failed'] = ['required_check']

    outcome = asyncio.run(resolve_reward(rubric, _results(passed=25), summary, runtime=runtime, cwd='/case'))

    assert outcome.score == 0.25
    assert outcome.continuous_score == 0.25


@pytest.mark.parametrize('stdout', ['not-json', '1.01', '-0.1', 'NaN', 'true', '{"value": 1}'])
def test_reward_script_rejects_invalid_score_output(stdout: str) -> None:
    runtime = StubRuntime(stdout=stdout)
    rubric = {'version': '4.2', 'config': {'reward': {'script': 'reward.py'}}, 'checks': []}

    with pytest.raises(RewardScriptError, match='score'):
        asyncio.run(resolve_reward(rubric, [], _summary(passed=0), runtime=runtime, cwd='/case'))


def test_reward_script_rejects_nonzero_exit() -> None:
    runtime = StubRuntime(stdout='1', return_code=7)
    rubric = {'version': '4.2', 'config': {'reward': {'script': 'reward.py'}}, 'checks': []}

    with pytest.raises(RewardScriptError, match='return_code=7'):
        asyncio.run(resolve_reward(rubric, [], _summary(passed=0), runtime=runtime, cwd='/case'))


def test_reward_script_timeout_is_reported_without_fallback() -> None:
    rubric = {'version': '4.2', 'config': {'reward': {'script': 'reward.py', 'timeout_sec': 2}}, 'checks': []}

    with pytest.raises(RewardScriptError, match='timed out') as exc_info:
        asyncio.run(resolve_reward(rubric, [], _summary(passed=0), runtime=TimeoutRuntime(), cwd='/case'))

    assert exc_info.value.kind == 'timeout'


def test_default_reward_path_still_uses_score_policy() -> None:
    rubric = {
        'version': '4.2',
        'config': {'score': {'continuous': False, 'use_threshold': True, 'threshold': 0.8}},
        'checks': [],
    }

    outcome = asyncio.run(resolve_reward(rubric, [], _summary(passed=20), runtime=None, cwd='/case'))

    assert outcome.score == 0.0
    assert outcome.continuous_score == pytest.approx(2 / 3)
    assert outcome.policy == {'threshold': 0.8, 'continuous': False, 'use_threshold': True}


def test_reward_script_config_requires_a_shipped_case_file(tmp_path: Path) -> None:
    rubric = {'version': '4.2', 'config': {'reward': {'script': 'reward.py'}}, 'checks': []}

    assert validate_reward_config(rubric, case_dir=tmp_path) == [
        'rubric.yaml: config.reward.script file not found: reward.py'
    ]
    (tmp_path / 'reward.py').write_text('print(1)\n', encoding='utf-8')
    assert validate_reward_config(rubric, case_dir=tmp_path) == []


@pytest.mark.parametrize(
    ('reward', 'message'),
    [
        ({'script': '../reward.py'}, 'relative path'),
        ({'script': '/tmp/reward.py'}, 'relative path'),
        ({'script': 'reward.txt'}, '.py or .sh'),
        ({'script': 'reward.py', 'timeout_sec': 0}, 'positive number'),
    ],
)
def test_reward_script_config_rejects_unsafe_or_invalid_values(reward: dict[str, Any], message: str) -> None:
    rubric = {'version': '4.2', 'config': {'reward': reward}, 'checks': []}

    assert message in validate_reward_config(rubric)[0]


@pytest.mark.parametrize('script', ['_reward.py', 'files/reward.py', 'data/reward.sh'])
def test_reward_script_config_rejects_assets_skipped_by_upload(script: str, tmp_path: Path) -> None:
    script_path = tmp_path / script
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text('print(1)\n', encoding='utf-8')
    rubric = {'version': '4.2', 'config': {'reward': {'script': script}}, 'checks': []}

    assert 'not a shippable case asset' in validate_reward_config(rubric, case_dir=tmp_path)[0]


def test_reward_script_and_score_policy_are_mutually_exclusive() -> None:
    rubric = {
        'version': '4.2',
        'config': {
            'reward': {'script': 'reward.py'},
            'score': {'continuous': False, 'use_threshold': True},
        },
        'checks': [],
    }

    assert 'mutually exclusive' in validate_reward_config(rubric)[0]


def test_aidata_validate_checks_reward_script_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir = tmp_path / 'case_1'
    case_dir.mkdir()
    (case_dir / 'task.yaml').write_text(
        'metadata:\n  created_at: 2026-07-18\nuser_prompt: test\n',
        encoding='utf-8',
    )
    (case_dir / 'rubric.yaml').write_text(
        """
version: "4.2"
config:
  reward:
    script: reward.py
checks:
  - id: answer_present
    description: answer exists
    dimension: correctness
    weight: 1
    assert:
      type: text_contains
      from: agent_answer
      contains: ok
""".strip(),
        encoding='utf-8',
    )
    monkeypatch.setattr(aidata_cli, 'CASES_DIR', tmp_path)

    problems = aidata_cli._validate_case('case_1')

    assert 'rubric.yaml: config.reward.script file not found: reward.py' in problems
