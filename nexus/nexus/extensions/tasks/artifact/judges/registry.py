"""Registration, lazy loading, and top-level execution of artifact judges.

Strategy names match modules under ``judges.strategies``. The wrapper checks
inputs, prepares declared capabilities, and validates the returned verdict.
Composite judges call sub-judge functions directly and own failure aggregation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nexus.core.registry import (
    EntryNotFoundError,
    ProcessorResult,
    Registry,
    RegistryError,
)
from nexus.extensions.tasks.artifact.judges.base import (
    JudgeExecutionError,
    JudgeFailureKind,
    JudgeInput,
    JudgeResult,
)


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient


logger = logging.getLogger(__name__)


JudgeFn = Callable[[JudgeInput], Awaitable[JudgeResult]]
ValidateFn = Callable[[JudgeInput], str | None]
"""Cheap input check: return a reason to reject, ``None`` to accept."""


# Screenshot implies deploy; supplied resources are reused.
_CAPABILITY_TOKENS: frozenset[str] = frozenset({'deploy', 'screenshot'})


# Evaluator faults are masked out; no_artifact remains trainable.
EVALUATOR_FAULT_KINDS = frozenset[JudgeFailureKind]({
    'no_client',
    'no_sandbox',
    'vlm_failed',
    'parse_failed',
    'agent_failed',
    'config_error',
    'eval_timeout',
    'internal_error',
})


# Explicit strategy budget, or a budget derived from the verifier limit.
JUDGE_BUDGET_ENV = 'ARTIFACT_JUDGE_BUDGET_SECONDS'
VERIFIER_TIMEOUT_ENV = 'T3_VERIFIER_TIMEOUT_SEC'

# Reserve proportional headroom with a minimum for verifier finalization.
JUDGE_BUDGET_HEADROOM_RATIO = 0.05
JUDGE_BUDGET_MIN_HEADROOM = 60.0


# Strategy names must match their module or package name.
STRATEGIES_PACKAGE = 'nexus.extensions.tasks.artifact.judges.strategies'


@dataclass(frozen=True)
class JudgeSpec:
    """A resolved judge: the callable plus everything the wrapper needs."""

    name: str
    fn: JudgeFn
    requires: tuple[str, ...] = ('agent',)
    capabilities: tuple[str, ...] = ()
    validate: ValidateFn | None = None
    """Optional cheap input check, run BEFORE clients and capabilities.

    Return a reason string to reject (becomes ``config_error``), ``None`` to
    accept. Must not deploy, call a model, or touch the runtime — it exists so
    a data error is rejected before anything expensive happens. See
    :func:`run_judge` for the ordering.
    """
    model_files: Mapping[str, str] = field(default_factory=dict)


def _positional_param_count(sig: inspect.Signature) -> tuple[int, bool]:
    """Return the number of positional parameters and whether *args is accepted."""
    positional = 0
    has_varargs = False
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            positional += 1
        elif param.kind is param.VAR_POSITIONAL:
            has_varargs = True
    return positional, has_varargs


def _validate_registration(obj: JudgeFn, name: str, meta: dict) -> ProcessorResult | None:
    """Validate strategy declarations when the module registers them."""
    if not inspect.iscoroutinefunction(obj):
        raise TypeError(f'@register_judge({name!r}) must decorate an async function; got {obj!r}')
    sig = inspect.signature(obj)
    positional, has_varargs = _positional_param_count(sig)
    if positional != 1 or has_varargs or len(sig.parameters) != 1:
        raise TypeError(
            f'@register_judge({name!r}): compute_score must take exactly one positional '
            f'JudgeInput parameter (the wrapper calls fn(inp)); got signature {sig}'
        )

    # Preserve explicit empty tuples; defaults apply only to missing keys.
    requires = meta.get('requires', ('agent',))
    if not isinstance(requires, tuple) or not all(isinstance(r, str) for r in requires):
        raise TypeError(f'@register_judge({name!r}): requires must be a tuple[str, ...]; got {requires!r}')
    capabilities = meta.get('capabilities', ())
    if not isinstance(capabilities, tuple) or not all(isinstance(c, str) for c in capabilities):
        raise TypeError(f'@register_judge({name!r}): capabilities must be a tuple[str, ...]; got {capabilities!r}')
    unknown = set(capabilities) - _CAPABILITY_TOKENS
    if unknown:
        raise ValueError(
            f'@register_judge({name!r}): unknown capabilities {sorted(unknown)}; allowed = {sorted(_CAPABILITY_TOKENS)}'
        )

    model_files = meta.get('model_files', {})
    if not isinstance(model_files, dict):
        raise TypeError('model_files must map Judge purposes to YAML filenames')
    if model_files:
        import re

        if set(model_files) != set(requires):
            raise ValueError('model_files must bind every required Judge purpose')
        if not all(
            isinstance(v, str) and re.fullmatch(r'[a-z0-9][a-z0-9.-]*-aihub-2\.0\.yaml', v)
            for v in model_files.values()
        ):
            raise ValueError('model_files values must be bundled AIHub 2.0 YAML filenames')

    validate = meta.get('validate')
    if validate is not None:
        if not callable(validate) or inspect.iscoroutinefunction(validate):
            raise TypeError(
                f'@register_judge({name!r}): validate must be a sync callable '
                f'(JudgeInput) -> str | None; got {validate!r}'
            )
        v_sig = inspect.signature(validate)
        v_positional, v_varargs = _positional_param_count(v_sig)
        if v_positional != 1 or v_varargs:
            raise TypeError(
                f'@register_judge({name!r}): validate must take exactly one positional '
                f'JudgeInput parameter; got {v_sig}'
            )
        # Additional required parameters would make validate(inp) uncallable.
        missing_defaults = [
            p.name
            for p in v_sig.parameters.values()
            if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
            and p.name not in list(v_sig.parameters)[:1]
        ]
        if missing_defaults:
            raise TypeError(
                f'@register_judge({name!r}): validate has required parameter(s) '
                f'{missing_defaults} beyond the JudgeInput; the wrapper calls '
                f'validate(inp) with nothing else, so give them defaults or drop them'
            )
    return None  # keep obj + meta unchanged


judge_registry: Registry[JudgeFn] = Registry('artifact_judge', processor=_validate_registration)


def register_judge(
    name: str,
    *,
    requires: tuple[str, ...] = ('agent',),
    capabilities: tuple[str, ...] = (),
    validate: ValidateFn | None = None,
    model_files: Mapping[str, str] | None = None,
) -> Callable[[JudgeFn], JudgeFn]:
    """Register a strategy under its module or package name.

    agent is the default role for tool-using agent execution; vlm is for
    direct model judgments without an agent loop. requires=() declares no clients.
    Roles require an exact binding, with no fallback to inference main.
    Profiles are selected by entrypoints, never by composite child calls.
    Capabilities are prepared before execution; screenshot implies deploy.
    The synchronous validate hook returns None to accept or a reason to reject.
    """
    # Validate declaration types before coercion.
    return judge_registry.register(
        name,
        meta={
            'requires': tuple(requires) if isinstance(requires, tuple) else requires,
            'capabilities': tuple(capabilities) if isinstance(capabilities, tuple) else capabilities,
            'validate': validate,
            'model_files': dict(model_files) if model_files is not None else {},
        },
    )


def get_judge(name: str) -> JudgeSpec:
    """Resolve a strategy by name, importing its module or package on demand.

    Import errors propagate to the caller.
    """
    if name not in judge_registry:
        Registry.import_module(f'{STRATEGIES_PACKAGE}.{name}')
    fn = judge_registry.get(name)
    meta = getattr(fn, 'nexus_meta', {}).get('meta', {}) or {}
    # Preserve explicitly empty dependency declarations.
    return JudgeSpec(
        name=name,
        fn=fn,
        requires=tuple(meta.get('requires', ('agent',))),
        capabilities=tuple(meta.get('capabilities', ())),
        validate=meta.get('validate'),
        model_files=dict(meta.get('model_files', {})),
    )


def resolve_judge_name(row: Mapping[str, Any], *, default: str = '') -> str:
    """Resolve the judge_name field for data-row entrypoints."""
    judge_name = str(row.get('judge_name') or '').strip() or default.strip()
    if not judge_name:
        raise ValueError('no judge_name in the row')
    return judge_name


def _artifact_precheck(judge_name: str, inp: JudgeInput) -> JudgeResult | None:
    """Classify a confirmed empty deliverable before resource preparation.

    Missing file_list means unknown, not empty. A clean finish yields no_artifact;
    an unfinished rollout yields agent_failed. Content policies belong to strategies.
    """
    from nexus.extensions.tasks.artifact.reward_protocol import Judge, artifact_empty_cause

    row = dict(inp.instance or {})
    # Both entrypoints normalize agent outcome into infer_metadata.
    infer_metadata = row.get('infer_metadata')
    infer_metadata = dict(infer_metadata) if isinstance(infer_metadata, dict) else {}

    # Missing file_list means unknown, not empty.
    if 'file_list' not in row:
        return None

    file_list = row.get('file_list')
    file_list = list(file_list) if isinstance(file_list, (list, tuple)) else []

    if not file_list:
        cause = artifact_empty_cause(infer_metadata)
        finished_cleanly = cause == 'no_files_written'
        return JudgeResult(
            judge_name=judge_name,
            score=0.0,
            error=(
                'agent exited cleanly but wrote no files'
                if finished_cleanly
                else f'no files collected from working_dir (cause={cause})'
            ),
            # Clean finish → the model's own doing (stays in training).
            # Anything else → the rollout never completed (maskout).
            failure_kind='no_artifact' if finished_cleanly else 'agent_failed',
            metadata={
                'artifact_precheck': 'empty',
                'empty_cause': cause,
                'judge_code': str(Judge.ARTIFACT_MISSING if finished_cleanly else Judge.ROLLOUT_UNFINISHED),
            },
        )

    return None


def resolve_judge_budget() -> float | None:
    """Return the strategy execution budget in seconds, or None if unbounded.

    ARTIFACT_JUDGE_BUDGET_SECONDS takes precedence; non-positive values disable it.
    Otherwise derive a budget from T3_VERIFIER_TIMEOUT_SEC with finalization headroom.
    """
    explicit = (os.environ.get(JUDGE_BUDGET_ENV) or '').strip()
    if explicit:
        try:
            value = float(explicit)
        except ValueError:
            logger.warning('%s=%r is not a number; ignoring', JUDGE_BUDGET_ENV, explicit)
        else:
            return None if value <= 0 else value

    verifier = (os.environ.get(VERIFIER_TIMEOUT_ENV) or '').strip()
    if verifier:
        try:
            total = float(verifier)
        except ValueError:
            logger.warning('%s=%r is not a number; ignoring', VERIFIER_TIMEOUT_ENV, verifier)
        else:
            if total > 0:
                # Reserve time for evidence, reward output, and log flushing.
                headroom = max(JUDGE_BUDGET_MIN_HEADROOM, total * JUDGE_BUDGET_HEADROOM_RATIO)
                budget = total - headroom
                if budget <= 0:
                    logger.warning(
                        '%s=%.0fs leaves no room for judging after %.0fs of headroom; running unbounded',
                        VERIFIER_TIMEOUT_ENV,
                        total,
                        headroom,
                    )
                    return None
                return budget

    return None


def _fail(spec_name: str, error: str, kind: JudgeFailureKind) -> JudgeResult:
    """Uniform ``JudgeResult`` for wrapper-side failures."""
    return JudgeResult(judge_name=spec_name, score=0.0, error=error, failure_kind=kind)


def _deploy_failure(
    spec_name: str,
    deploy_error: str,
    classify: Callable[[str], tuple[bool, str]],
) -> JudgeResult:
    """Keep artifact deployment failures trainable; mask infrastructure failures."""
    is_artifact_fault, reason = classify(deploy_error)
    if is_artifact_fault:
        return JudgeResult(
            judge_name=spec_name,
            score=0.0,
            passed=False,
            evidence=f'artifact does not deploy: {reason}'[:400],
            metadata={'deploy_failed': True, 'deploy_error': reason},
        )
    return JudgeResult(
        judge_name=spec_name,
        score=0.0,
        error=f'deploy failed: {reason}',
        failure_kind='no_sandbox',
        metadata={'deploy_error': reason},
    )


def resolve_client_role(
    clients: Mapping[str, OpenAICompatibleClient],
    role: str,
) -> OpenAICompatibleClient | None:
    """Require an exact role binding; never substitute another model implicitly."""
    return clients.get(role)


async def _prepare_capabilities(
    spec: JudgeSpec,
    inp: JudgeInput,
) -> JudgeInput | JudgeResult:
    """Prepare declared resources, reusing any supplied sandbox or screenshots.

    Screenshot implies deploy. Return augmented input or a structured failure.
    """
    caps = set(spec.capabilities)
    if not caps:
        return inp

    # Local imports keep the registry module light — a judge that declares no
    # capabilities never pulls the deploy/screenshot code path.
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import (
        classify_deploy_error,
        prepare_sandbox_context,
        sandbox_context_from_screenshot,
    )

    sandbox_ctx = inp.sandbox_ctx

    # 1. deploy — screenshot implies this. Only run when the caller has not
    #    already pre-filled a SandboxContext with a URL.
    if (caps & {'deploy', 'screenshot'}) and (sandbox_ctx is None or not sandbox_ctx.ok):
        if not inp.archive_b64:
            return _fail(
                spec.name,
                f'{spec.name} declares deploy but JudgeInput.archive_b64 is empty',
                'no_artifact',
            )
        try:
            sandbox_ctx = await prepare_sandbox_context(inp.archive_b64, instance_id=inp.instance_id)
        except Exception as exc:  # noqa: BLE001
            # An exception (as opposed to a returned ``deploy_error``) means the
            # deploy path itself broke, not the artifact — evaluator fault.
            return _fail(spec.name, f'deploy failed: {type(exc).__name__}: {exc}', 'no_sandbox')
        if sandbox_ctx.deploy_error:
            return _deploy_failure(spec.name, sandbox_ctx.deploy_error, classify_deploy_error)

    # 2. screenshot — sandbox is up (from step 1 or pre-filled). Skip if the
    #    caller already handed us screenshots (composite path).
    if 'screenshot' in caps and sandbox_ctx is not None and sandbox_ctx.screenshots.is_empty():
        from nexus.extensions.tasks.artifact.judges.capabilities.screenshot_in_container import (
            capture_screenshot_in_container,
        )

        try:
            shot = await capture_screenshot_in_container(inp.runtime, sandbox_ctx.sandbox_url or '')
        except Exception as exc:  # noqa: BLE001
            return _fail(spec.name, f'screenshot failed: {type(exc).__name__}: {exc}', 'no_sandbox')
        if not shot:
            return _fail(spec.name, 'screenshot returned no data', 'no_sandbox')
        sandbox_ctx = sandbox_context_from_screenshot(
            sandbox_url=sandbox_ctx.sandbox_url or '',
            screenshot_b64=shot,
            sandbox_proxy_url=sandbox_ctx.sandbox_proxy_url,
        )

    from dataclasses import replace

    return replace(inp, sandbox_ctx=sandbox_ctx)


async def run_judge(judge_name: str, inp: JudgeInput) -> JudgeResult:
    """Run one top-level judge and return a validated, structured verdict.

    Check the artifact and strategy input before preparing clients and resources.
    Limit strategy execution to its configured budget, classify execution errors,
    and enforce the registered result name. Composites dispatch their own children.
    """
    # 1. Resolve.
    try:
        spec = get_judge(judge_name)
    except (EntryNotFoundError, ImportError, RegistryError) as exc:
        logger.exception('failed to load judge %r', judge_name)
        return _fail(
            judge_name,
            f'unknown or broken judge {judge_name!r}: {type(exc).__name__}: {exc}',
            'config_error',
        )

    # Classify empty artifacts before strategy validation and resource preparation.
    artifact_failure = _artifact_precheck(spec.name, inp)
    if artifact_failure is not None:
        return artifact_failure

    # 3. Cheap input validation, before clients / deploy / screenshots.
    if spec.validate is not None:
        try:
            reason = spec.validate(inp)
        except Exception as exc:  # noqa: BLE001
            # A crash in the validator is our bug, not bad data.
            logger.exception('judge %s validate hook raised', spec.name)
            return _fail(spec.name, f'validate hook raised {type(exc).__name__}: {exc}', 'internal_error')
        # Invalid validator return types are implementation errors.
        if reason is not None and not isinstance(reason, str):
            logger.error(
                'judge %s validate hook returned %r (%s); expected None or a reason string',
                spec.name,
                reason,
                type(reason).__name__,
            )
            return _fail(
                spec.name,
                f'validate hook returned {type(reason).__name__} {reason!r}; '
                'expected None (accept) or a non-empty reason string (reject)',
                'internal_error',
            )
        if reason:
            logger.info('judge %s rejected its input: %s', spec.name, reason)
            return _fail(spec.name, reason, 'config_error')

    # Child strategies reuse the same explicit client bindings.
    for role in spec.requires:
        if resolve_client_role(inp.openai_clients, role) is None:
            return _fail(
                spec.name,
                f'missing client for role {role!r}; available = {sorted(inp.openai_clients)}',
                'no_client',
            )

    # 5. Capabilities.
    prep = await _prepare_capabilities(spec, inp)
    if isinstance(prep, JudgeResult):
        return prep
    inp = prep

    # 6-8. Run the judge, under a wall-clock budget.
    budget = resolve_judge_budget()
    started = time.monotonic()
    try:
        if budget is None:
            result = await spec.fn(inp)
        else:
            result = await asyncio.wait_for(spec.fn(inp), timeout=budget)
    except (TimeoutError, asyncio.TimeoutError):
        elapsed = time.monotonic() - started
        # Return a structured timeout while the verifier can still finalize.
        logger.error('judge %s exceeded its %gs budget (elapsed %.1fs)', spec.name, budget, elapsed)
        return _fail(
            spec.name,
            f'judging exceeded its {budget:g}s budget (elapsed {elapsed:.1f}s); '
            f'raise {JUDGE_BUDGET_ENV} / [verifier] timeout_sec, or make this judge cheaper',
            'eval_timeout',
        )
    except JudgeExecutionError as exc:
        logger.info('judge %s failed structurally (%s): %s', spec.name, exc.kind, exc)
        return _fail(spec.name, str(exc), exc.kind)
    except Exception as exc:  # noqa: BLE001
        logger.exception('judge %s raised an unexpected exception', spec.name)
        return _fail(spec.name, f'{type(exc).__name__}: {exc}', 'internal_error')

    # Validate actual return values, not just annotations.
    if not isinstance(result, JudgeResult):
        logger.error('judge %s returned %s, not JudgeResult', spec.name, type(result).__name__)
        return _fail(
            spec.name,
            f'judge returned {type(result).__name__}, expected JudgeResult',
            'internal_error',
        )
    try:
        # Revalidate to catch mutations after model construction.
        result = JudgeResult.model_validate(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        logger.error('judge %s returned an invalid JudgeResult: %s', spec.name, exc)
        return _fail(spec.name, f'invalid JudgeResult: {exc}', 'internal_error')

    # The registered top-level name is authoritative.
    if result.judge_name != spec.name:
        logger.warning(
            'judge %s returned judge_name=%s; enforcing registered name',
            spec.name,
            result.judge_name,
        )
        result = result.model_copy(update={'judge_name': spec.name})

    return result


__all__ = [
    'EVALUATOR_FAULT_KINDS',
    'JudgeFn',
    'JudgeSpec',
    'STRATEGIES_PACKAGE',
    'get_judge',
    'judge_registry',
    'register_judge',
    'run_judge',
]
