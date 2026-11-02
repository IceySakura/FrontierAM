"""RL reward-contract semantics for the artifact task (archive/RL mode).

Mirrors ``test_frontend_task_reward_semantics.py`` for the built-in-judge
artifact task: verifies the empty/incomplete short-circuit, the zero_reason →
judge_code/training_action mapping, and that ``extra_fields.reward_extra_info``
carries the correct training_action + judge metrics.

These are pure-function tests — no runtime, no sandbox, no LLM.
"""

import base64
import inspect
import io
import re
import tarfile

from pathlib import Path
from types import SimpleNamespace

from nexus.extensions.tasks.artifact import reward_protocol as rp
from nexus.extensions.tasks.artifact import task as artifact
from nexus.utils.reward_score import ReasoningReward


def _ti() -> SimpleNamespace:
    return SimpleNamespace(instance_id='case-1')


def _archive(files: dict[str, str]) -> str:
    """Build a base64 tar.gz of ``{path: content}`` (what the infer phase emits)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for path, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


# ── Short-circuit: empty artifact ─────────────────────────────────────────────


def test_no_files_written_is_penalty() -> None:
    """Agent exited cleanly (ENV_DONE) yet wrote nothing → the model's own failure.

    It must PARTICIPATE in training (negative reward, no maskout termination), unlike
    every other empty cause, which means the rollout never finished.
    """
    out = artifact._archive_short_circuit(
        _ti(),
        {},
        '',
        [],
        {'agent_termination_reason': 'ENV_DONE'},
        archive_cos_url='cos://bucket/empty.tar.gz',
    )
    assert out is not None
    assert out.archive_cos_url == 'cos://bucket/empty.tar.gz'
    assert out.score == rp.PENALTY_MISSING  # -0.2
    assert out.completed is True
    assert out.termination_reason is None  # NOT a maskout — must train
    assert out.zero_reason == 'ARTIFACT_MISSING'
    ri = out.extra_fields['reward_extra_info']
    assert ri['training_action'] == 'penalty'
    assert ri['judge_code'] == rp.Judge.ARTIFACT_MISSING
    assert ri['extra_reward_metrics']['judge/artifact_missing'] == 1.0
    assert ri['extra_reward_metrics']['judge/model_fault'] == 1.0
    assert ri['extra_reward_metrics']['artifact/empty/no_files_written'] == 1.0


def test_other_empty_causes_stay_maskout() -> None:
    """Every empty cause EXCEPT no_files_written means the rollout was cut short.

    Those must stay masked out — the model cannot be held responsible for a harness
    timeout or a crashed agent. This is the gate that stops the penalty from being
    widened to all empty artifacts.
    """
    # 'unknown' is the real shape seen in production: hycode returns success=False
    # with no termination_reason (an infra burst produced 178 of these in one step).
    for meta, expect_cause in (
        ({'agent_exit_status': 'failed'}, 'unknown'),
        ({'agent_termination_reason': 'TIMEOUT'}, 'agent_timeout'),
        ({'agent_termination_reason': 'MAX_STEPS'}, 'max_steps'),
        ({'artifact_collect_error': 'tar failed'}, 'collect_failed'),
    ):
        out = artifact._archive_short_circuit(_ti(), {}, '', [], meta)
        assert out is not None, meta
        assert out.score == float(ReasoningReward.THRESHOLD.value), meta
        assert out.completed is False, meta
        assert out.termination_reason == 'REWARD_ERROR', meta
        assert out.zero_reason == f'EMPTY:{expect_cause}', meta
        ri = out.extra_fields['reward_extra_info']
        assert ri['training_action'] == 'maskout', meta
        assert ri['judge_code'] == rp.Judge.ROLLOUT_UNFINISHED, meta
        assert ri['extra_reward_metrics'][f'artifact/empty/{expect_cause}'] == 1.0, meta


def test_nonempty_artifact_without_required_stack_proceeds() -> None:
    assert artifact._archive_short_circuit(_ti(), {}, 'x', ['index.html'], {}) is None


def test_required_stack_with_package_json_proceeds() -> None:
    inst = {'judge_config': {'required_stack': 'vite-react'}}
    archive = _archive({'package.json': '{}', 'src/App.tsx': 'x'})
    assert (
        artifact._archive_short_circuit(
            _ti(),
            inst,
            archive,
            ['package.json', 'src/App.tsx'],
            {'agent_termination_reason': 'ENV_DONE'},
        )
        is None
    )


def test_svg_single_clean_proceeds() -> None:
    inst = {'judge_config': {'required_stack': 'svg'}}
    archive = _archive({'art.svg': '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'})
    assert (
        artifact._archive_short_circuit(
            _ti(),
            inst,
            archive,
            ['art.svg'],
            {'agent_termination_reason': 'ENV_DONE'},
        )
        is None
    )


def test_classify_all_zero_is_valid_use_score() -> None:
    # A legitimate all-zero (bad code) is a real training signal, not a maskout.
    assert artifact._classify_zero_reason('ALL_ZERO') == (rp.Judge.SUCCESS, '')
    assert artifact._classify_zero_reason(None) == (rp.Judge.SUCCESS, '')


def test_classify_infra_failures_are_maskout() -> None:
    assert artifact._classify_zero_reason('EVAL_DEPLOY_SANDBOX_UNAVAILABLE') == (
        rp.Judge.EVAL_DEPLOY_SANDBOX_UNAVAILABLE,
        'REWARD_ERROR',
    )
    assert artifact._classify_zero_reason('EVAL_VLM_CALL_FAILED') == (rp.Judge.EVAL_VLM_CALL_FAILED, 'REWARD_ERROR')
    assert artifact._classify_zero_reason('EVAL_PARSE_FAILED') == (rp.Judge.EVAL_VLM_CALL_FAILED, 'REWARD_ERROR')
    assert artifact._classify_zero_reason('EVAL_NO_SANDBOX_CTX') == (rp.Judge.EVAL_AGENT_CALL_FAILED, 'REWARD_ERROR')
    assert artifact._classify_zero_reason('EVAL_ERROR:boom') == (rp.Judge.EVAL_AGENT_CALL_FAILED, 'REWARD_ERROR')


def test_classify_config_task_and_unknown_failures_are_distinct() -> None:
    for zero_reason in ('EVAL_NO_ARTIFACT', 'EVAL_NO_PIPELINE', 'EVAL_UNKNOWN_PIPELINE:missing'):
        assert artifact._classify_zero_reason(zero_reason) == (
            rp.Judge.EVAL_CONFIG_FAILED,
            'REWARD_ERROR',
        )

    assert artifact._classify_zero_reason('EVAL_TASK_ERROR:TASK_TIMEOUT') == (
        rp.Judge.NEXUS_TASK_TIMEOUT,
        'TASK_TIMEOUT',
    )
    assert artifact._classify_zero_reason('EVAL_TASK_ERROR:UNHANDLED_ERROR') == (
        rp.Judge.NEXUS_TASK_FAILED,
        'UNHANDLED_ERROR',
    )
    assert artifact._classify_zero_reason('EVAL_NEW_FAILURE') == (
        rp.Judge.EVAL_UNKNOWN_FAILED,
        'REWARD_ERROR',
    )


# ── cross-repo contract: termination_reason carries maskout ───────────────────
#
# HyTuner does NOT read judge_code to decide maskout -- it only logs it. The
# load-bearing signal is ``termination_reason``: hytuner matches it by exact value
# against MASKOUT_TERMINATION_REASONS (hytuner/utils/termination_reason.py:52-69).
# A hit keeps the sample out of the gradient (V1 drops the response from the
# training group via is_maskout; V2 zeroes loss_mask in trajectory_ops.py:433).
#
# Judge-side failures therefore map to ``REWARD_ERROR`` -- an existing hytuner
# member, so no hytuner change is needed. A value hytuner does not define would
# silently not match and the failure would train as a real low score.
#
# The two tests below lock both directions of that contract. Changing a maskout
# term_reason to an unrecognized value, or adding a term_reason to a penalty path,
# silently changes whether a rollout trains -- with no error anywhere. These are
# the tests that go red first.


def _emitted_zero_reasons() -> list[str]:
    """Every ``zero_reason`` literal the artifact task can actually emit.

    **Derived from the source, not hand-maintained.** An earlier hand-written copy of
    this list silently rotted: it still held the pre-rename ``ARTIFACT_EMPTY:`` /
    ``JUDGE_NO_PIPELINE`` spellings, which no longer match any branch in
    ``_classify_zero_reason`` and so fell through to the EVAL_UNKNOWN_FAILED catch-all.
    The contract test below kept passing (the fallback is also maskout) while never
    once exercising ROLLOUT_UNFINISHED — the single largest maskout bucket in
    production. Grepping the source keeps that from recurring.
    """
    src = Path(inspect.getsourcefile(artifact)).read_text(encoding='utf-8')
    # zero_reason='LITERAL' and zero_reason=f'PREFIX:{placeholder}'
    literals = re.findall(r"zero_reason=(f?)'([^']*)'", src)
    reasons = {
        # substitute a realistic value for the f-string placeholder
        re.sub(r'\{[^}]*\}', 'max_steps', text) if is_f else text
        for is_f, text in literals
    }
    # _zero_reason_from_results builds these dynamically rather than via zero_reason=
    reasons |= {'ALL_ZERO', 'EVAL_VLM_CALL_FAILED', 'EVAL_PARSE_FAILED', 'EVAL_ERROR:agent_failed'}
    assert len(reasons) >= 8, f'zero_reason grep found only {reasons} -- did the assignment style change?'
    return sorted(reasons)


def test_every_emitted_zero_reason_round_trips_to_the_right_action() -> None:
    """Each emitted ``zero_reason`` must classify to a code whose action matches intent.

    ``ARTIFACT_*`` means the model shipped something bad → penalty, must train.
    Everything else is a rollout/eval fault → maskout, must be excluded.

    This is the test that would have caught the stale-fixture drift: it asserts on the
    strings the code really produces, so a rename that forgets a call site fails here.
    """
    for zero_reason in _emitted_zero_reasons():
        judge_code, term_reason = artifact._classify_zero_reason(zero_reason)
        action = rp.Judge(judge_code).action
        if zero_reason == 'ALL_ZERO':
            assert action == 'use_score', f'{zero_reason!r} -> {action}'
        elif zero_reason.startswith('ARTIFACT_'):
            assert action == 'penalty', f'{zero_reason!r} -> {action}, penalty would be lost'
            assert term_reason == '', f'{zero_reason!r} got term_reason={term_reason!r}: masks the penalty'
        else:
            assert action == 'maskout', f'{zero_reason!r} -> {action}, expected maskout'

    # the rollout-truncation family must reach ROLLOUT_UNFINISHED, not the catch-all
    for zero_reason in ('EMPTY:no_files_written', 'EMPTY:unknown', 'INCOMPLETE:max_steps'):
        code, _ = artifact._classify_zero_reason(zero_reason)
        assert code == rp.Judge.ROLLOUT_UNFINISHED, f'{zero_reason!r} -> {rp.Judge(code).name}'


def test_all_maskout_codes_emit_hytuner_maskout_reason() -> None:
    """Every maskout verdict must carry a termination_reason hytuner masks on.

    hytuner matches termination_reason against MASKOUT_TERMINATION_REASONS by exact
    value. Emit anything outside that set and the sample is never masked, so an
    eval-side failure trains as if it were the model's real score.
    """
    hytuner_maskout_reasons = {'REWARD_ERROR', 'TASK_TIMEOUT', 'UNHANDLED_ERROR', 'INITIALIZATION_ERROR'}
    checked_maskout = 0
    for zero_reason in _emitted_zero_reasons():
        judge_code, term_reason = artifact._classify_zero_reason(zero_reason)
        if rp.Judge(judge_code).action != 'maskout':
            continue
        checked_maskout += 1
        assert term_reason, f'{zero_reason!r} produced an empty termination_reason -- it would train'
        assert term_reason in hytuner_maskout_reasons, (
            f'{zero_reason!r} -> term_reason={term_reason!r} is not in hytuner'
            f"'s MASKOUT_TERMINATION_REASONS; maskout will silently not apply"
        )
    assert checked_maskout >= 5, f'only {checked_maskout} maskout reasons exercised -- list went stale again'


def test_penalty_codes_never_emit_maskout_reason() -> None:
    """Penalty verdicts must carry an empty termination_reason, or they become maskouts.

    A penalty only reaches the gradient because its termination_reason is empty (see
    ``_stack_penalty_output``). Give it any hytuner maskout reason and the sample is
    masked -> the penalty is silently discarded.
    """
    penalty_codes = [c for c in rp.Judge if c.action == 'penalty']
    assert penalty_codes, 'no penalty codes found -- the taxonomy lost its penalty path'
    for code in penalty_codes:
        for zero_reason in (None, 'ALL_ZERO'):
            judge_code, term_reason = artifact._classify_zero_reason(zero_reason)
            assert not term_reason, f'use_score/penalty path for {code.name} got term_reason={term_reason!r}'


def test_judge_action_is_total() -> None:
    """Every Judge member maps to exactly one of the three actions, and outcome works.

    Guards against a newly added code silently defaulting to 'maskout' via the
    fall-through in ``Judge.action``.
    """
    for code in rp.Judge:
        assert code.action in ('use_score', 'penalty', 'maskout'), f'{code.name} -> {code.action}'
        assert code.outcome in ('success', 'penalty', 'masked'), f'{code.name} -> {code.outcome}'


# ── reward_extra_info shape ──────────────────────────────────────────────────


def test_rl_extra_fields_success_use_score() -> None:
    ef = artifact._rl_extra_fields(
        rp.Judge.SUCCESS,
        weighted_scores={'web_app': 0.7},
        infer_metadata={'artifact_file_count': 12},
    )
    ri = ef['reward_extra_info']
    assert ri['training_action'] == 'use_score'
    assert ri['extra_reward_metrics']['judge/success'] == 1.0
    assert ri['weighted_scores'] == {'web_app': 0.7}
    assert ri['artifact']['file_count'] == 12


def test_rl_extra_fields_timeout_is_maskout() -> None:
    ri = artifact._rl_extra_fields(rp.Judge.NEXUS_TASK_TIMEOUT, term_reason='TASK_TIMEOUT')['reward_extra_info']
    assert ri['training_action'] == 'maskout'
    assert ri['extra_reward_metrics']['judge/eval_timeout'] == 1.0
    assert ri['extra_reward_metrics']['judge/nexus_task_timeout'] == 1.0
    assert ri['extra_reward_metrics']['judge/eval_config_failed'] == 0.0


def test_finalize_eval_preserves_archive_cos_url(tmp_path) -> None:
    result = artifact._finalize_eval(
        _ti(),
        judge_results=[artifact.JudgeResult(judge_name='web_judger', score=0.5)],
        final_score=0.5,
        resolved=False,
        zero_reason=None,
        archive_b64='archive',
        archive_cos_url='cos://bucket/final.tar.gz',
        file_list=['index.html'],
        infer_trajectories={'steps': []},
        infer_metadata={'artifact_file_count': 1},
        save_dir=str(tmp_path),
        trajectory_id=None,
    )

    assert result.archive_cos_url == 'cos://bucket/final.tar.gz'
    assert result.archive_b64 == 'archive'
    assert result.file_list == ['index.html']


def test_term_reason_table_is_consistent() -> None:
    """``_TERM_REASON_BY_CODE`` must agree with each code's ``action``, for every code.

    This is the structural version of the two tests above: instead of relying on branch
    coverage through ``_classify_zero_reason``, it checks the table directly, so a newly
    added Judge member that forgets its term_reason fails here immediately.

    Contract (hytuner/utils/termination_reason.py MASKOUT_TERMINATION_REASONS):
      maskout → a reason hytuner already defines, matched by exact value
      penalty / use_score → term_reason MUST be '' (anything else kills the gradient)
    """
    hytuner_maskout_reasons = {'REWARD_ERROR', 'TASK_TIMEOUT', 'UNHANDLED_ERROR', 'INITIALIZATION_ERROR'}
    for code in rp.Judge:
        term = rp.term_reason_for(code)
        if code.action == 'maskout':
            assert term in hytuner_maskout_reasons, f'{code.name} is maskout but term_reason={term!r}'
        else:
            assert term == '', f'{code.name} is {code.action} but term_reason={term!r} would mask it'

    # every member is present -- term_reason_for must not be silently falling back
    assert set(rp._TERM_REASON_BY_CODE) == set(rp.Judge), 'a Judge member has no term_reason entry'


def test_failure_metrics_keep_config_task_and_unknown_buckets_separate() -> None:
    config_metrics = rp.build_judge_metrics(rp.Judge.EVAL_CONFIG_FAILED)
    task_metrics = rp.build_judge_metrics(rp.Judge.NEXUS_TASK_FAILED)
    timeout_metrics = rp.build_judge_metrics(rp.Judge.NEXUS_TASK_TIMEOUT, term_reason='TASK_TIMEOUT')
    unknown_metrics = rp.build_judge_metrics(rp.Judge.EVAL_UNKNOWN_FAILED)

    assert config_metrics['judge/eval_config_failed'] == 1.0
    assert task_metrics['judge/nexus_task_failed'] == 1.0
    assert timeout_metrics['judge/nexus_task_timeout'] == 1.0
    assert timeout_metrics['judge/eval_timeout'] == 1.0
    assert unknown_metrics['judge/eval_unknown_failed'] == 1.0

    for metrics in (task_metrics, timeout_metrics, unknown_metrics):
        assert metrics['judge/eval_config_failed'] == 0.0


def test_undeployable_artifact_is_penalty() -> None:
    """Artifact that will not deploy is the model's fault → penalty, stays in training.

    Infra-side deploy failures take the ``no_sandbox`` path instead and are masked out
    (see test_infra_deploy_failure_is_maskout).
    """
    code, term = artifact._classify_zero_reason('ARTIFACT_UNDEPLOYABLE')
    assert code == rp.Judge.ARTIFACT_UNDEPLOYABLE
    assert term == '', 'any maskout reason here would silently convert the penalty into a maskout'
    assert rp.Judge(code).action == 'penalty'
    metrics = rp.build_judge_metrics(code)
    assert metrics['judge/artifact_undeployable'] == 1.0
    assert metrics['judge/success'] == 0.0, 'must no longer be miscounted as a success'
    assert metrics['judge/model_fault'] == 1.0


def test_infra_deploy_failure_is_maskout() -> None:
    """``no_sandbox`` (deploy service unreachable) is eval-side → maskout, own code."""
    zero_reason = artifact._zero_reason_from_results([
        artifact.JudgeResult(judge_name='web_judger', score=0.0, error='boom', failure_kind='no_sandbox')
    ])
    assert zero_reason == 'EVAL_DEPLOY_SANDBOX_UNAVAILABLE'
    code, term = artifact._classify_zero_reason(zero_reason)
    assert code == rp.Judge.EVAL_DEPLOY_SANDBOX_UNAVAILABLE
    assert term == 'REWARD_ERROR'
    assert rp.build_judge_metrics(code)['judge/eval_deploy_sandbox_unavailable'] == 1.0


def test_agent_status_translates_installed_agent_timeout() -> None:
    """hycode has no ``termination_reason`` field, so it must be inferred.

    ``InstalledAgentOutput`` only carries success / exit_code / stderr. Without this
    translation ``artifact_empty_cause`` can never return 'agent_timeout' and every
    hycode timeout lands in the 'unknown' bucket (measured: 19456/19456 rollouts had an
    empty agent_termination_reason).
    """
    timed_out = SimpleNamespace(success=False, exit_code=-1, stderr='boom\n[timeout]')
    assert artifact._agent_status(timed_out)['agent_termination_reason'] == 'TIMEOUT'
    assert rp.artifact_empty_cause(artifact._agent_status(timed_out)) == 'agent_timeout'

    # exit_code=-1 alone must NOT imply timeout -- the installer failure path reuses it.
    install_failed = SimpleNamespace(success=False, exit_code=-1, stderr='Failed to install HyCode: x')
    assert artifact._agent_status(install_failed)['agent_termination_reason'] == ''
    assert rp.artifact_empty_cause(artifact._agent_status(install_failed)) == 'unknown'

    # clean exit stays clean, and is attributed to the model when no files were written
    ok = SimpleNamespace(success=True, exit_code=0, stderr='')
    status = artifact._agent_status(ok)
    assert status['agent_termination_reason'] == ''
    assert status['agent_exit_status'] == 'success'
    assert rp.artifact_empty_cause(status) == 'no_files_written'

    # dynamicagent multiturn reports a turn limit explicitly
    maxed = SimpleNamespace(success=True, exit_code=0, stderr='', reached_max_turns=True)
    assert artifact._agent_status(maxed)['agent_termination_reason'] == 'MAX_STEPS'

    # an agent that DOES report termination_reason (claude_code) is passed through
    native = SimpleNamespace(exit_status='failed', termination_reason='MAX_STEPS', execution_status='')
    assert artifact._agent_status(native)['agent_termination_reason'] == 'MAX_STEPS'


def test_agent_execution_status_is_forwarded_to_rl() -> None:
    """exit_code must reach the RL dump, or a crash looks identical to a clean no-op."""
    extra = rp.build_reward_extra_info(
        request_id='',
        judge_code=rp.Judge.ROLLOUT_UNFINISHED,
        judge_response={},
        infer_metadata={'agent_exit_status': 'failed', 'agent_execution_status': 'exit_code=-1'},
    )
    assert extra['agent_execution_status'] == 'exit_code=-1'
    assert extra['agent_exit_status'] == 'failed'


# ══════════════════════════════════════════════════════════════════════════
# 失败优先于分数（J01）
#
# 复合 judge 可以在一条腿判分失败时仍返回非零均分。那个分数掺了评测故障，
# 不是有效成绩 —— 两端都必须把它判成 maskout，否则 task 侧把它当成绩训练、
# Harbor 侧判 -1，同一个故障两端语义相反。
# ══════════════════════════════════════════════════════════════════════════


def _judge_result(**kwargs):
    from nexus.extensions.tasks.artifact.judges.base import JudgeResult

    return JudgeResult(**kwargs)


def _action_for(judge_result) -> str:
    """跑 task 侧的完整决策链：outcome → zero_reason → judge_code → action。"""
    _score, _resolved, zero_reason = artifact.resolve_eval_outcome(judge_result)
    code, _term = artifact._classify_zero_reason(zero_reason)
    return rp.Judge(code).action


def test_composite_partial_score_with_failure_is_masked_out() -> None:
    """视觉 0.8 + general parse_failed → 复合 0.4 带 failure_kind。

    修复前：只在 score==0 时查 failure_kind，所以 0.4 走 SUCCESS 进训练，
    而 Harbor 侧读 failure_kind 判 -1。
    """
    composite = _judge_result(
        judge_name='aesthetic_general',
        score=0.4,
        error='parse: no valid <score>',
        failure_kind='parse_failed',
        child_results=[
            _judge_result(judge_name='aesthetic', score=0.8),
            _judge_result(judge_name='general_agent', score=0.0, error='parse: boom', failure_kind='parse_failed'),
        ],
    )

    score, _resolved, zero_reason = artifact.resolve_eval_outcome(composite)

    assert score == 0.4, '分数保留用于诊断'
    assert zero_reason == 'EVAL_PARSE_FAILED'
    assert _action_for(composite) == 'maskout', '评测故障必须 maskout，不能当成绩'


def test_both_entrypoints_agree_on_composite_failure() -> None:
    """同一个结果，task 判 maskout、Harbor 判 -1 —— 两端语义必须一致。"""
    from nexus.extensions.tasks.artifact.harbor_entry import verdict_to_reward

    composite = _judge_result(
        judge_name='aesthetic_general',
        score=0.35,
        error='vlm timeout',
        failure_kind='vlm_failed',
    )

    assert _action_for(composite) == 'maskout'
    assert verdict_to_reward(composite, judge_name='aesthetic_general')['reward'] == -1.0


def test_legitimate_score_is_untouched() -> None:
    """没有 failure_kind 的正分就是有效成绩，不能被误判。"""
    ok = _judge_result(judge_name='web_judger', score=0.72)

    score, _resolved, zero_reason = artifact.resolve_eval_outcome(ok)

    assert score == 0.72
    assert zero_reason is None
    assert _action_for(ok) == 'use_score'


def test_legitimate_zero_still_trains() -> None:
    """合法零分（代码写得差）是真实训练信号，不是 maskout。"""
    bad = _judge_result(judge_name='web_judger', score=0.0)

    score, _resolved, zero_reason = artifact.resolve_eval_outcome(bad)

    assert score == 0.0
    assert zero_reason == 'ALL_ZERO'
    assert _action_for(bad) == 'use_score'


def test_undeployable_artifact_is_a_penalty_not_a_maskout() -> None:
    """产物跑不起来 → 惩罚分进训练（模型该学到这比不做更糟）。"""
    undeployable = _judge_result(
        judge_name='web_judger',
        score=0.0,
        passed=False,
        evidence='artifact does not deploy: npm build failed',
        metadata={'deploy_failed': True, 'deploy_error': 'artifact: npm build failed'},
    )

    score, _resolved, zero_reason = artifact.resolve_eval_outcome(undeployable)

    assert score == artifact.PENALTY_UNDEPLOYABLE
    assert zero_reason == 'ARTIFACT_UNDEPLOYABLE'
    assert _action_for(undeployable) == 'penalty'


def test_evaluator_fault_wins_over_deploy_penalty() -> None:
    """两个信号同时在时，评测故障优先。

    把 maskout 改写成产物惩罚等于把「我们挂了」记成「模型交了个烂东西」。
    """
    both = _judge_result(
        judge_name='web_judger',
        score=0.0,
        error='render service unreachable',
        failure_kind='no_sandbox',
        metadata={'deploy_failed': True},
    )

    score, _resolved, zero_reason = artifact.resolve_eval_outcome(both)

    assert score == 0.0, '不该被改写成 PENALTY_UNDEPLOYABLE'
    assert zero_reason == 'EVAL_DEPLOY_SANDBOX_UNAVAILABLE'
    assert _action_for(both) == 'maskout'


def test_child_deploy_failed_does_not_override_the_composite() -> None:
    """子腿的 deploy_failed 不能推翻顶层结论。

    复合 judge 可能已经把某条腿的部署失败降级处理了（可选腿失败仍给分）；
    task 越过顶层去扫子项会推翻那个明确的降级决策。
    """
    composite = _judge_result(
        judge_name='aesthetic_general',
        score=0.6,
        child_results=[
            _judge_result(judge_name='aesthetic', score=0.6),
            _judge_result(judge_name='general_agent', score=0.0, metadata={'deploy_failed': True}),
        ],
    )

    score, _resolved, zero_reason = artifact.resolve_eval_outcome(composite)

    assert score == 0.6, '顶层分数是权威的'
    assert zero_reason is None
    assert _action_for(composite) == 'use_score'


def test_resolved_comes_from_the_top_level_passed_flag() -> None:
    """``resolved`` 读顶层 passed；没有 passed 时回落到 score>=1.0。"""
    explicit = _judge_result(judge_name='baseline_check', score=0.0, passed=True)
    assert artifact.resolve_eval_outcome(explicit)[1] is True

    implicit_full = _judge_result(judge_name='svg', score=1.0)
    assert artifact.resolve_eval_outcome(implicit_full)[1] is True

    implicit_partial = _judge_result(judge_name='svg', score=0.5)
    assert artifact.resolve_eval_outcome(implicit_partial)[1] is False
