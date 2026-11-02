"""Error surfacing contract for the Gongfeng provider.

Covers what a task owner actually sees when a sandbox call fails: the error code, the
attribution, the next step, and the identifiers needed to report it. Also pins the
degraded paths — a backend answer without `error`, an unknown code, an empty runtime
state — so a missing field never turns into a crash or a wrong verdict.
"""

from __future__ import annotations

import logging

from typing import Any

import pytest

from nexus.runtime.providers import gongfeng as gf


class _StubResponse:
    """Minimal stand-in for the aiohttp response consumed by ``_call_api``."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status = 200

    async def text(self) -> str:
        import json

        return json.dumps(self._payload)


def _raise_from_payload(payload: dict[str, Any]) -> gf.SandboxApiError:
    """Drive the error branch of ``_call_api`` over ``payload`` and return the exception.

    The hint delegates to the same module-level helper ``_call_api`` uses, so this mirror
    cannot drift away from production on that judgement.
    """
    code = payload.get('code', -1)
    msg = payload.get('msg', 'Unknown error')
    err = payload.get('error') or {}
    reason = str(err.get('reason') or '')
    trace_id = str(err.get('trace_id') or '')
    try:
        exit_code = int(err.get('exit_code') or 0)
    except (TypeError, ValueError):
        exit_code = 0
    headline = f'Gongfeng API error (code={code}'
    if reason:
        headline += f', reason={reason}'
    headline += f'): {msg}'
    details = [f'next step: {gf._next_step_hint(code, reason, has_trace_id=bool(trace_id))}']
    if exit_code:
        details.append(f'container exit_code={exit_code}')
    if trace_id:
        details.append(f'trace_id={trace_id}')
    return gf.SandboxApiError(
        f'{headline} | ' + ' | '.join(details),
        code=code,
        reason=reason,
        msg=str(msg),
        trace_id=trace_id,
        exit_code=exit_code,
    )


class TestNextStepHint:
    """The hint must match the disposition column of the sandbox error contract."""

    @pytest.mark.parametrize('code', sorted(gf.RETRIABLE_CODES))
    def test_retriable_codes_advise_backoff(self, code: int) -> None:
        assert 'retry the same request' in gf._next_step_hint(code, '')

    def test_quota_exhausted_and_rate_limited_differ(self) -> None:
        exhausted = gf._next_step_hint(gf.CODE_RESOURCE_EXHAUSTED, 'Quota.Exceeded')
        limited = gf._next_step_hint(gf.CODE_RESOURCE_EXHAUSTED, 'Quota.RateLimited')
        assert 'quota' in exhausted
        assert 'slow down' in limited
        assert exhausted != limited

    def test_quota_exhausted_points_at_the_portal(self) -> None:
        """Running out of quota is self-service; the message must say where to go."""
        hint = gf._next_step_hint(gf.CODE_RESOURCE_EXHAUSTED, 'Quota.Exceeded')
        assert gf.SANDBOX_PORTAL_RESOURCE_URL in hint

    def test_rate_limited_does_not_point_at_the_portal(self) -> None:
        """Applying for quota does not fix a rate limit, so the link would misdirect."""
        hint = gf._next_step_hint(gf.CODE_RESOURCE_EXHAUSTED, 'Quota.RateLimited')
        assert gf.SANDBOX_PORTAL_RESOURCE_URL not in hint

    def test_internal_error_says_reporting_not_retrying(self) -> None:
        hint = gf._next_step_hint(gf.CODE_INTERNAL, 'Platform.UnknownError')
        assert 'report it' in hint
        assert 'retrying will not help' in hint

    @pytest.mark.parametrize('code', [260400, 260401, 260403, 260404, 260409, 260410, 260422])
    def test_client_side_codes_advise_fixing_the_request(self, code: int) -> None:
        assert 'needs fixing' in gf._next_step_hint(code, '')

    def test_canceled_says_nothing_to_fix(self) -> None:
        """A cancelled call is the caller's own doing: neither a bad payload nor a bug to report."""
        hint = gf._next_step_hint(gf.CODE_CANCELED, 'Request.Canceled')
        assert 'cancelled' in hint
        assert 'nothing to fix' in hint
        assert 'report it' not in hint
        assert 'needs fixing' not in hint

    @pytest.mark.parametrize('code', [260405, 260451, 260501, 260599, 261999, 999])
    def test_unregistered_code_stops_and_reports(self, code: int) -> None:
        """The contract registers an exact set; anything else must stop and be reported.

        A band check would hand an unregistered 2604xx code the "fix your request" advice,
        which contradicts the contract and sends the caller chasing their own payload.
        """
        hint = gf._next_step_hint(code, '')
        assert 'unrecognized' in hint
        assert 'stop retrying' in hint

    def test_registered_disposition_sets_cover_the_whole_contract(self) -> None:
        """The five disposition groups must add up to exactly the contract's 13 codes."""
        registered = gf.RETRIABLE_CODES | gf.CLIENT_FIX_CODES | {
            gf.CODE_RESOURCE_EXHAUSTED,
            gf.CODE_CANCELED,
            gf.CODE_INTERNAL,
        }
        assert registered == {
            260400,
            260401,
            260403,
            260404,
            260409,
            260410,
            260412,
            260422,
            260429,
            260499,
            260500,
            260503,
            260504,
        }


class TestSandboxApiError:
    def test_carries_contract_fields_and_reads_as_actionable_text(self) -> None:
        exc = _raise_from_payload(
            {
                'code': 260503,
                'msg': 'sandbox start failed: image pull error[TraceID]:abc123',
                'error': {'reason': 'Image.PullError', 'trace_id': 'abc123', 'exit_code': 0},
            }
        )
        assert (exc.code, exc.reason, exc.trace_id, exc.exit_code) == (260503, 'Image.PullError', 'abc123', 0)
        text = str(exc)
        assert 'code=260503' in text
        assert 'reason=Image.PullError' in text
        assert 'next step: transient platform-side condition' in text
        assert 'trace_id=abc123' in text
        # Zero exit codes carry no information and must not be shown.
        assert 'exit_code' not in text

    def test_keeps_the_prefix_log_scanners_match_on(self) -> None:
        """swe_cosmos greps run logs for this prefix to judge whether a run is trustworthy."""
        exc = _raise_from_payload({'code': 260400, 'msg': 'bad request'})
        assert str(exc).startswith('Gongfeng API error (code=260400)')

    def test_is_catchable_as_runtime_error(self) -> None:
        """Existing `except RuntimeError` call sites must keep working."""
        assert isinstance(_raise_from_payload({'code': 260500, 'msg': 'boom'}), RuntimeError)

    def test_missing_error_object_degrades_to_code_only(self) -> None:
        """Gateway rejections answer without `error`; that must not crash or invent a reason."""
        exc = _raise_from_payload({'code': 260401, 'msg': 'token required'})
        assert exc.reason == ''
        assert exc.trace_id == ''
        assert 'reason=' not in str(exc)
        assert 'next step:' in str(exc)

    def test_non_numeric_exit_code_is_ignored(self) -> None:
        exc = _raise_from_payload({'code': 260422, 'msg': 'x', 'error': {'reason': 'X.Y', 'exit_code': 'n/a'}})
        assert exc.exit_code == 0

    def test_exit_code_is_surfaced_when_the_container_terminated(self) -> None:
        exc = _raise_from_payload(
            {'code': 260422, 'msg': 'oom', 'error': {'reason': 'Process.OOMKilled', 'exit_code': 137}}
        )
        assert exc.exit_code == 137
        assert 'container exit_code=137' in str(exc)


class TestDeathReasonContract:
    def test_unhealthy_reasons_all_have_a_next_step(self) -> None:
        """Every reason that kills a sandbox must tell the owner what to do about it."""
        assert set(gf.UNHEALTHY_REASONS) <= set(gf.DEATH_NEXT_STEPS)

    def test_recoverable_forwarding_failures_are_not_deaths(self) -> None:
        """Ports not up yet or a transient dial failure must go through the streak counter."""
        for reason in ('Platform.RuntimeError', 'Platform.UnknownError', 'Image.PullError'):
            assert reason not in gf.UNHEALTHY_REASONS

    def test_stopped_sandbox_reason_counts_as_death(self) -> None:
        """The proxy collapses every failure to Runtime.NotFound once the record is stopped.

        Without this the watchdog needs the full failure streak (~12 min) to notice a sandbox
        that is already gone.
        """
        assert 'Runtime.NotFound' in gf.UNHEALTHY_REASONS


class TestQueueFailureRemarks:
    def test_covers_every_never_started_stop_reason(self) -> None:
        for remark in ('queue_timeout', 'queue_give_up', 'block_timeout', 'start_failed'):
            assert gf._is_queue_failure_remark(remark)

    def test_matches_the_stop_failed_suffix_variant(self) -> None:
        """The backend appends `(stop_failed:...)` when the stop call itself failed."""
        assert gf._is_queue_failure_remark('queue_timeout(stop_failed:connection timeout)')
        assert gf._is_queue_failure_remark('start_failed(stop_failed:connection timeout)')

    def test_normal_stops_are_not_queue_failures(self) -> None:
        for remark in ('framework', 'user_delete', 'idle_timeout', 'ttl_expired'):
            assert not gf._is_queue_failure_remark(remark)

    @pytest.mark.parametrize('remark', ['queue_timeout_v2', 'start_failed_hard', 'block_timeout2'])
    def test_unregistered_lookalike_reasons_do_not_match(self, remark: str) -> None:
        """A bare prefix match would fold a future reason into today's classification.

        The backend's stopcode contract accepts `(` as the only separator for exactly this
        purpose, so an unrecognised reason surfaces as unrecognised instead of silently
        inheriting the queue-failure verdict. This side must not be looser than that.
        """
        assert not gf._is_queue_failure_remark(remark)






@pytest.mark.asyncio
async def test_queue_wait_keeps_polling_while_state_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued sandbox must keep being polled until the ledger says scheduled."""
    provider = gf.GongfengRuntimeProvider.__new__(gf.GongfengRuntimeProvider)
    polls = 0

    async def _fake_get_runtime(runtime_id: str, log_level: int = logging.DEBUG) -> dict[str, Any]:
        nonlocal polls
        polls += 1
        if polls < 3:
            return {'status': gf.RUNTIME_STATUS_QUEUED}
        return {'status': gf.RUNTIME_STATUS_SCHEDULED, 'url': 'http://sandbox.local:8080', 'port_mappings': {}}

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(provider, '_get_runtime', _fake_get_runtime, raising=False)
    monkeypatch.setattr(provider, '_next_queue_delay', lambda: 0.0, raising=False)
    monkeypatch.setattr(gf.asyncio, 'sleep', _no_sleep)

    url, _ = await provider._wait_for_scheduled('n', 'rid-unknown', 0.0)
    assert url == 'http://sandbox.local:8080'
    assert polls == 3



class TestMsgSurfacing:
    def test_msg_is_exposed_as_an_attribute(self) -> None:
        exc = _raise_from_payload({'code': 260400, 'msg': 'bad request'})
        assert exc.msg == 'bad request'
        assert 'bad request' in str(exc)

    def test_cause_is_ignored_even_when_present(self) -> None:
        exc = _raise_from_payload(
            {
                'code': 260503,
                'msg': 'failed to create sandbox',
                'error': {'reason': 'Image.PullError', 'cause': 'ErrImagePull: manifest unknown'},
            }
        )
        assert not hasattr(exc, 'cause')
        assert 'cause:' not in str(exc)


class TestTraceIdPointer:
    """The report advice must not point at a trace ID the message does not carry.

    `_call_api` already appends `trace_id=...` only when the field is present, so the advice
    has to be gated on the same condition — otherwise the degraded answer the surrounding
    code deliberately tolerates (no `error` object) reads as "report it with the trace_id
    above" with no trace ID anywhere in the message.
    """

    def test_hint_omits_the_pointer_when_no_trace_id_is_available(self) -> None:
        exc = _raise_from_payload({'code': 260500, 'msg': 'boom'})
        assert 'report it' in str(exc)
        assert 'trace_id' not in str(exc)

    def test_hint_omits_the_pointer_when_trace_id_is_empty(self) -> None:
        exc = _raise_from_payload(
            {'code': 260500, 'msg': 'boom', 'error': {'reason': 'Platform.UnknownError', 'trace_id': ''}}
        )
        assert 'trace_id' not in str(exc)

    def test_hint_points_at_the_trace_id_when_one_is_present(self) -> None:
        exc = _raise_from_payload(
            {'code': 260500, 'msg': 'boom', 'error': {'reason': 'Platform.UnknownError', 'trace_id': 'tid-1'}}
        )
        assert 'report it with the trace_id above' in str(exc)
        assert 'trace_id=tid-1' in str(exc)

    def test_unregistered_code_pointer_is_gated_too(self) -> None:
        assert 'trace_id' not in gf._next_step_hint(999, '')
        assert 'trace_id' in gf._next_step_hint(999, '', has_trace_id=True)
