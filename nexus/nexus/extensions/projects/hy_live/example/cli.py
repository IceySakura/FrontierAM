"""Run one frozen hy-live sandbox example through nexus.runner.single."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

from pathlib import Path
from typing import Any, Literal

from nexus.extensions.projects.hy_live.example.cases import BACKENDS, CASES, get_case
from nexus.extensions.projects.hy_live.example.harness import (
    CEREBELLUM_SAMPLING,
    DEFAULT_API_BASE,
    DEFAULT_BACKEND,
    DEFAULT_CLIENT_QUESTIONS,
    DEFAULT_MODEL,
    DEFAULT_RELAY_UPDATES,
    DEFAULT_RUNTIME_IMAGE,
    DEFAULT_SANDBOX_API,
    DEFAULT_WORKSPACE_IMAGE,
    MIDBRAIN_SAMPLING,
    LayerSampling,
    build_run_input,
)
from nexus.extensions.projects.hy_live.example.validate import validate_run
from nexus.runner.single import RunHooks, run
from nexus.trace import Span, SpanEndEvent, SpanProcessor, SpanStartEvent
from nexus.utils.log import setup_logging
from nexus.utils.provider_io import redact_provider_io


class _PolicyRequestProcessor(SpanProcessor):
    """Persist logical policy calls, including responses before the agent can rewrite them."""

    def __init__(self, output_path: Path, *, sensitive_source: dict[str, str | None]) -> None:
        descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._stream = os.fdopen(descriptor, 'a', encoding='utf-8')
        os.fchmod(self._stream.fileno(), 0o600)
        self._sensitive_source = sensitive_source

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        self._write_record(span, 'start')

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        self._write_record(span, 'end')

    def finalize(self) -> None:
        self._stream.close()

    def _write_record(self, span: Span, phase: Literal['start', 'end']) -> None:
        if span.name != 'openai_completion' or span.attributes.get('_client_key') != 'main':
            return
        record: dict[str, Any] = {
            'trace_id': span.trace_id,
            'span_id': span.span_id,
            'phase': phase,
            'start_time_unix_nano': span.start_time_unix_nano,
            'status': span.status.code,
        }
        if phase == 'start':
            inputs = span.attributes['inputs']
            request_kwargs = inputs.get('kwargs', {})
            request_meta = inputs.get('request_meta') or {}
            record.update(
                messages=inputs['messages'],
                tools=request_kwargs.get('tools'),
                tool_choice=request_kwargs.get('tool_choice'),
                parallel_tool_calls=request_kwargs.get('parallel_tool_calls'),
                request_meta={
                    key: request_meta[key] for key in ('invocation_kind', 'turn_step') if key in request_meta
                },
            )
            if isinstance(inputs.get('model'), str):
                record['model'] = inputs['model']
        else:
            record['end_time_unix_nano'] = span.end_time_unix_nano
            record['status_message'] = span.status.message
            if 'outputs' in span.attributes:
                outputs = span.attributes['outputs']
                record['completion'] = {
                    key: outputs[key] for key in ('id', 'model', 'choices', 'usage') if key in outputs
                }
        sanitized = redact_provider_io(record, sensitive_source=self._sensitive_source)
        self._stream.write(json.dumps(sanitized, ensure_ascii=False) + '\n')
        self._stream.flush()


def trajectory_line(record: dict[str, Any]) -> str:
    """Serialize one record as a compact JSONL line, the format of ``cerebellum.jsonl``."""
    return json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'


def _write_trajectories(
    output_dir: Path, task_output: Any, *, sensitive_source: dict[str, str | None]
) -> dict[str, dict[str, Any]]:
    """Write ``cerebellum.jsonl``, ``midbrain.jsonl`` and ``dialogue.jsonl``, one record each.

    Returns:
        The records written, keyed by layer; a layer the task output lacks is skipped.
    """
    written: dict[str, dict[str, Any]] = {}
    if not isinstance(task_output, dict):
        return written
    for layer in ('cerebellum', 'midbrain', 'dialogue'):
        record = task_output.get(f'{layer}_trajectory')
        if not isinstance(record, dict) or not record:
            continue
        sanitized = redact_provider_io(record, sensitive_source=sensitive_source)
        descriptor = os.open(output_dir / f'{layer}.jsonl', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(trajectory_line(sanitized))
        written[layer] = sanitized
    return written


def _layer_sampling(args: argparse.Namespace, layer: str, default: LayerSampling) -> LayerSampling:
    """Read one layer's sampling flags; a caller-built namespace without them keeps the defaults."""
    options = vars(args)
    return LayerSampling(
        reasoning_effort=options.get(f'{layer}_reasoning_effort', default.reasoning_effort),
        temperature=options.get(f'{layer}_temperature', default.temperature),
        top_p=options.get(f'{layer}_top_p', default.top_p),
        max_tokens=options.get(f'{layer}_max_tokens', default.max_tokens),
    )


def _list_cases() -> int:
    for case in CASES.values():
        backends = ','.join(sorted(case.backends))
        print(f'{case.name:30} {backends:20} {case.data_relpath:48} {case.spec_relpath}')
    return 0


async def _run_case(args: argparse.Namespace) -> int:
    case = get_case(args.case)
    try:
        run_input = build_run_input(
            case,
            run_id=args.run_id or f'hy-live-{case.name}',
            api_base_url=args.api_base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            sandbox_api_url=args.sandbox_api_url,
            runtime_image=args.runtime_image,
            workspace_image=args.workspace_image,
            timeout=args.timeout,
            worker_venv=args.worker_venv,
            sandbox_purpose=args.sandbox_purpose or None,
            relay_updates=args.relay_updates,
            client_questions=vars(args).get('client_questions', DEFAULT_CLIENT_QUESTIONS),
            cerebellum_sampling=_layer_sampling(args, 'cerebellum', CEREBELLUM_SAMPLING),
            midbrain_sampling=_layer_sampling(args, 'midbrain', MIDBRAIN_SAMPLING),
            web_search_provider=vars(args).get('web_search'),
            backend=vars(args).get('backend', DEFAULT_BACKEND),
            workbuddy_package=vars(args).get('workbuddy_package'),
            workbuddy_package_sha256=vars(args).get('workbuddy_package_sha256'),
            pass_env=vars(args).get('pass_env') or (),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if vars(args).get('record_events'):
        run_input.task_config['record_events_path'] = str(args.output.parent / 'events.jsonl')
    sensitive_source = {
        name: os.getenv(name)
        for name in (
            'LIVE_AGENT_POLICY_API_KEY',
            'LIVE_AGENT_EXTERNAL_API_KEY',
            'GONGFENG_RUNTIME_TOKEN',
            'CODEBUDDY_API_KEY',
            *(vars(args).get('pass_env') or ()),
        )
    }
    capture = _PolicyRequestProcessor(args.output.parent / 'policy_requests.jsonl', sensitive_source=sensitive_source)
    try:
        result = await run(run_input, hooks=RunHooks(span_processors=[capture]))
    finally:
        capture.finalize()
    payload = result.model_dump(mode='json')
    _write_trajectories(args.output.parent, payload.get('task_output'), sensitive_source=sensitive_source)
    validation = validate_run(result.task_output, output_dir=args.output.parent, case_checks=case.case_checks)
    payload['example_validation'] = validation
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    summary = {
        'case': case.name,
        'backend': run_input.task_config.get('backend_adapter', DEFAULT_BACKEND),
        'success': result.success and validation['passed'],
        'error': result.error.model_dump(mode='json') if result.error else None,
        'output_path': str(args.output),
        'validation': validation,
        'checklist': str(case.spec_path),
    }
    print(json.dumps(redact_provider_io(summary, sensitive_source=sensitive_source), ensure_ascii=False, indent=2))
    return 0 if result.success and validation['passed'] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Print registered cases and exit')
    parser.add_argument('--case', choices=sorted(CASES), help='Runnable example case name')
    parser.add_argument('--output', type=Path, help='Where to write result.json')
    parser.add_argument('--run-id', default='')
    parser.add_argument(
        '--api-base-url',
        default=os.getenv('LIVE_AGENT_POLICY_BASE_URL', DEFAULT_API_BASE),
        help='OpenAI-compatible endpoint shared by all model clients',
    )
    parser.add_argument(
        '--model', default=os.getenv('LIVE_AGENT_POLICY_MODEL', DEFAULT_MODEL), help='Model served at --api-base-url'
    )
    parser.add_argument('--reasoning-effort', default='high', help='Working agent (external client) and user simulator')
    parser.add_argument(
        '--temperature', type=float, default=0, help='Working agent (external client) and user simulator'
    )
    parser.add_argument(
        '--backend',
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help='Working agent: codebuddy (ACP v1) or workbuddy (wb --acp, ACP v2).',
    )
    parser.add_argument(
        '--workbuddy-package',
        default=os.getenv('HY_LIVE_WORKBUDDY_PACKAGE') or None,
        help='Optional Host wb .tgz to upload; omit to use the runtime image preinstall with --backend workbuddy.',
    )
    parser.add_argument(
        '--workbuddy-package-sha256',
        default=os.getenv('HY_LIVE_WORKBUDDY_PACKAGE_SHA256') or None,
        help='Expected sha256 of --workbuddy-package; the sandbox checks the reassembled package against it.',
    )
    for layer, sampling in (('cerebellum', CEREBELLUM_SAMPLING), ('midbrain', MIDBRAIN_SAMPLING)):
        parser.add_argument(
            f'--{layer}-reasoning-effort',
            default=sampling.reasoning_effort,
            help=f'{layer} reasoning_effort; an empty value sends none',
        )
        parser.add_argument(f'--{layer}-temperature', type=float, default=sampling.temperature)
        parser.add_argument(f'--{layer}-top-p', type=float, default=sampling.top_p)
        parser.add_argument(f'--{layer}-max-tokens', type=int, default=sampling.max_tokens)
    parser.add_argument(
        '--sandbox-api-url',
        default=os.getenv('GONGFENG_API_BASE_URL', DEFAULT_SANDBOX_API),
        help='Gongfeng sandbox API',
    )
    parser.add_argument(
        '--sandbox-purpose',
        choices=('', 'prod', 'eval', 'test'),
        default=os.getenv('GONGFENG_RUNTIME_PURPOSE', ''),
        help='Gongfeng sandbox purpose. Required when the token is scoped to one purpose, e.g. prod.',
    )
    parser.add_argument(
        '--runtime-image',
        default=os.getenv('NEXUS_RUNTIME_IMAGE', DEFAULT_RUNTIME_IMAGE),
        help='Gongfeng sandbox /nix runtime with the CodeBuddy host relay (tag: docs/impl/runtime_image.md).',
    )
    parser.add_argument(
        '--workspace-image',
        default=os.getenv('LIVE_AGENT_RUNTIME_IMAGE', DEFAULT_WORKSPACE_IMAGE),
        help='Task/workspace container image. Must differ from --runtime-image: different image, different warm-up lane.',
    )
    parser.add_argument(
        '--web-search',
        choices=('yuanbao', 'none'),
        default=None,
        help="Cerebellum web_search backend; omit to keep the case's web_search_provider (default none).",
    )
    parser.add_argument(
        '--pass-env',
        action='append',
        default=[],
        metavar='NAME',
        help='Hand this host environment variable to the working agent, e.g. TMAP_WEBSERVICE_KEY; repeatable. '
        'Only the name enters the run data, and the value is redacted from the outputs.',
    )
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument(
        '--record-events',
        action='store_true',
        help="Write events.jsonl next to --output: every exchange with the outside, in hylive's contract.",
    )
    parser.add_argument(
        '--worker-venv',
        action='store_true',
        help='Upload current nexus into a sandbox venv. Default is false; rebuild the runtime image instead.',
    )
    parser.add_argument(
        '--relay-updates',
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RELAY_UPDATES,
        help='Have the runtime push live ACP progress; needs a runtime image (or --worker-venv) with relay_updates.',
    )
    parser.add_argument(
        '--client-questions',
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CLIENT_QUESTIONS,
        help='Let the working agent ask the user (AskUserQuestion / elicitation); needs a runtime image with client_questions.',
    )
    args = parser.parse_args(argv)
    if args.list:
        return args
    if not args.case or args.output is None:
        parser.error('--case and --output are required unless --list')
    return args


def main() -> None:
    setup_logging(log_level=logging.INFO, trim_threshold=(300, 300))
    args = parse_args()
    if args.list:
        raise SystemExit(_list_cases())
    raise SystemExit(asyncio.run(_run_case(args)))


if __name__ == '__main__':
    main()
