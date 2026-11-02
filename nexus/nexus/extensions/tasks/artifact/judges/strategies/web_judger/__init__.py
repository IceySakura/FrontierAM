"""Score Web artifacts from flow-based evidence and aesthetic judgments.

The registry prepares deployment; this package owns collection, validation,
coverage gates, and aggregation.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.judges.base import JudgeFailureKind, JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.registry import register_judge, resolve_client_role
from nexus.extensions.tasks.artifact.judges.strategies.visual_anchored import resolve_anchors_from_instance
from nexus.extensions.tasks.artifact.judges.strategies.web_judger import executor, verifier


logger = logging.getLogger(__name__)

NAME = 'web_judger'
MIN_WEIGHTED_COVERAGE = 0.5

# This strategy owns the Harbor contract path; the shared entrypoint does not.
HARBOR_FLOW_PATHS = ('/tests/flow.json',)


def _maskout(error: str, kind: JudgeFailureKind, **metadata: Any) -> JudgeResult:
    """构造由上层屏蔽的零分结果。"""
    return JudgeResult(judge_name=NAME, score=0.0, error=error, failure_kind=kind, metadata=metadata)


def _collection_meta(evidence: list[dict[str, Any]], code_meta: dict[str, Any]) -> dict[str, Any]:
    """Summarize collection diagnostics for successful and failed evaluations."""
    return {
        'status': _tally(evidence),
        # Keep the traceback tail, where the root exception is reported.
        'script_errors': [
            f'{e.get("fid", "?")}[{e.get("status", "?")}]: {str(e.get("error") or "")[-400:]}'
            for e in evidence
            if e.get('status') not in executor.OK_STATUSES
        ][:10],
        'missing_fids': code_meta.get('missing_fids') or [],
        'n_suppressed': sum(len(e.get('suppressed') or []) for e in evidence),
        'shot_calls': sum(int(e.get('shot_calls') or 0) for e in evidence),
        'shot_successes': sum(1 for e in evidence for shot in (e.get('shots') or []) if shot.get('b64')),
        'shot_capture_failures': sum(int(e.get('shot_capture_failures') or 0) for e in evidence),
        'shot_errors': [msg for e in evidence for msg in (e.get('shot_errors') or [])][:10],
        'shot_download_failures': sum(
            1 for e in evidence for shot in (e.get('shots') or []) if shot.get('download_failed')
        ),
        'visual_evidence_missing': sum(
            1
            for e in evidence
            if int(e.get('shot_calls') or 0) > 0 and not any(shot.get('b64') for shot in (e.get('shots') or []))
        ),
        'flow2code_seconds': round(code_meta.get('execution_time') or 0.0, 1),
    }


def _load_harbor_flow() -> tuple[dict[str, Any] | None, str]:
    """Read the strategy-owned Harbor contract; return (contract, error).

    Absent files return (None, empty string); unreadable or invalid files report an error.
    """
    for candidate in HARBOR_FLOW_PATHS:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:  # noqa: BLE001
            return None, f'{candidate} 读取失败: {type(exc).__name__}: {exc}'
        if not isinstance(payload, dict):
            return None, f'{candidate} 必须是 JSON object，得到 {type(payload).__name__}'
        return payload, ''
    return None, ''


def _flow_contract(row: dict[str, Any]) -> tuple[Any, str]:
    """Resolve an inline contract before the Harbor file.

    An explicitly invalid flow field must not fall back to another source.
    """
    if 'flow' in row:
        candidate = row['flow']
        if isinstance(candidate, dict) and candidate.get('flows'):
            return candidate, ''
        return None, (
            f"instance['flow'] 显式给了但不是带 flows 的 object（得到 "
            f'{type(candidate).__name__}）—— 不回落到其他来源，避免打错的字段静默生效'
        )

    if row.get('flows'):
        # Accept a contract supplied directly as instance data.
        return row, ''

    contract, error = _load_harbor_flow()
    if error:
        return None, error
    return contract, ''


def _validate_input(inp: JudgeInput) -> str | None:
    """Validate the contract and aesthetic configuration before deployment."""
    row = dict(inp.instance or {})

    flow, load_error = _flow_contract(row)
    if load_error:
        # Preserve file-read errors instead of reporting a missing contract.
        return load_error
    if flow is None:
        return (
            '没有 flow 契约：web_judger 需要它。'
            f'把契约放到 {HARBOR_FLOW_PATHS[0]}（即任务目录的 tests/flow.json，'
            'Verifier 会上传整个 tests/）。'
        )
    if not isinstance(flow, dict):
        return f'flow 契约必须是 object，得到 {type(flow).__name__}'
    flows = flow.get('flows')
    if not flows:
        return 'flow 契约里没有 flows'
    # Require an array of flow objects before execution.
    if not isinstance(flows, list):
        return f'flow 契约的 flows 必须是数组，得到 {type(flows).__name__}'
    seen_fids: set[str] = set()
    for index, item in enumerate(flows):
        if not isinstance(item, dict):
            return f'flows[{index}] 必须是 object，得到 {type(item).__name__}'
        fid = str(item.get('fid') or '').strip()
        if not fid:
            # Collection and verification index flows by fid.
            return f'flows[{index}] 缺 fid —— 采证与判定都按 fid 索引'
        if fid in seen_fids:
            # Reject duplicate IDs before dictionary indexing can merge rules.
            return f'flows[{index}] 的 fid {fid!r} 重复 —— 按 fid 索引会静默覆盖同名 flow'
        seen_fids.add(fid)

    # Per-instance aesthetic settings override shared configuration.
    aesthetic_mode = str(row.get('aesthetic_mode') or config.get().judge.aesthetic_mode)
    # Reject unsupported modes before deployment.
    if aesthetic_mode not in config.AESTHETIC_MODES:
        return f'未知 aesthetic_mode {aesthetic_mode!r}，合法值：{sorted(config.AESTHETIC_MODES)}'
    if aesthetic_mode == 'pairwise' and not resolve_anchors_from_instance(row):
        return 'pairwise 模式但 instance 里没有 anchor_screenshot(s)'

    return None


@register_judge(
    'web_judger',
    requires=('agent', 'vlm'),
    capabilities=('deploy',),
    validate=_validate_input,
    model_files={'agent': 'deepseek-flash-aihub-2.0.yaml', 'vlm': 'gemini-3.8-flash-aihub-2.0.yaml'},
)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Collect evidence, verify flows and aesthetics, and apply coverage gates.

    No images yields a valid zero; evaluator failures carry a structured failure kind.
    """
    row = dict(inp.instance or {})
    flow, _ = _flow_contract(row)  # _validate_input 已确认可读且带 flows
    aesthetic_mode = str(row.get('aesthetic_mode') or config.get().judge.aesthetic_mode)

    # Deployment is prepared and classified by the wrapper.
    if inp.sandbox_ctx is None or not inp.sandbox_ctx.sandbox_url:
        return _maskout('wrapper 没有提供部署好的 sandbox_url', 'no_sandbox')
    page_url = inp.sandbox_ctx.sandbox_url

    # Use the agent client for scripts and the VLM client for functional and aesthetic verdicts.
    coder = resolve_client_role(inp.openai_clients, 'agent')
    judger = resolve_client_role(inp.openai_clients, 'vlm')
    if coder is None or judger is None:
        return _maskout('agent 或 vlm 的 client 取不到', 'no_client')

    # Script generation and execution share the runtime and source directory.
    source_dir = str(row.get('working_dir') or '/workspace')
    scripts, code_meta = await executor.flow2code(
        flow,
        runtime=inp.runtime,
        runtime_name=inp.instance_id,
        archive_b64=inp.archive_b64,
        problem_statement=inp.problem_statement,
        file_list=row.get('file_list') or [],
        openai_client=coder,
        source_dir=source_dir,
        save_dir=inp.save_dir,
        model_config=inp.model_configs.get('agent'),
    )
    if not scripts:
        return _maskout(
            code_meta.get('upload_error')
            or code_meta.get('agent_error')
            or code_meta.get('read_error')
            or 'flow2code 没产出任何脚本',
            'agent_failed',
            flow2code=code_meta,
        )

    # 3. Execute the generated scripts, gather evidence.
    evidence = await executor.execute(
        scripts,
        runtime=inp.runtime,
        page_url=page_url,
        flow=flow,
        save_dir=inp.save_dir,
        source_dir=source_dir,
    )

    # Judge functionality and aesthetics from the collected evidence.
    result = await verifier.verify(
        evidence,
        flow=flow,
        judger=judger,
        problem_statement=inp.problem_statement,
        aesthetic_mode=aesthetic_mode,
        instance=row,
        archive_b64=inp.archive_b64,
        save_dir=inp.save_dir,
    )

    summary = result['summary']
    # Preserve collection diagnostics on failure paths too.
    collection = _collection_meta(evidence, code_meta)
    # An unavailable functional verdict invalidates the combined score.
    if not summary['func_available']:
        return _maskout(
            '没有任何 flow 判出结论（全部 inconclusive）',
            'agent_failed',
            summary=summary,
            flow2code=code_meta,
            collection=collection,
        )
    # Insufficient coverage is a contract-quality error.
    if summary['weighted_coverage'] < MIN_WEIGHTED_COVERAGE:
        return _maskout(
            f'有效 flow 加权覆盖率过低: {summary["weighted_coverage"]:.3f} < {MIN_WEIGHTED_COVERAGE:.3f}',
            'config_error',
            summary=summary,
            flow2code=code_meta,
            collection=collection,
        )
    # No images yields a valid zero; a failed VLM call is an evaluator fault.
    if not summary['aesthetic_available']:
        aesthetic = result.get('aesthetic') or {}
        reason = aesthetic.get('reason')
        detail = aesthetic.get('error') or '未知原因'
        if reason == 'no_images':
            return JudgeResult(
                judge_name=NAME,
                score=0.0,
                passed=False,
                evidence=f'视觉检查无证据: {detail}',
                metadata={
                    'summary': summary,
                    'flow2code': code_meta,
                    'collection': collection,
                    'aesthetic': aesthetic,
                    'zero_reason': 'no_visual_evidence',
                },
            )
        return _maskout(
            f'美学分不可用: {detail}',
            'vlm_failed',
            summary=summary,
            flow2code=code_meta,
            collection=collection,
            aesthetic=aesthetic,
        )

    logger.info(
        '[%s] web_judger final=%.3f func=%.3f aes=%.3f(%s) %s scripts=%d/%d',
        inp.instance_id[:12],
        summary['final_score'],
        summary['func_score'],
        summary['aesthetic_score'],
        aesthetic_mode,
        summary['counts'],
        len(scripts),
        len(flow['flows']),
    )
    return JudgeResult(
        judge_name=NAME,
        score=summary['final_score'],
        metadata={
            # Keep the scoring URL and the public inspection URL distinct.
            'real_url': page_url,
            'proxy_url': inp.sandbox_ctx.sandbox_proxy_url or '',
            'summary': summary,
            # Keep mode and anchor provenance; full aesthetic output is saved separately.
            'aesthetic': {
                key: value
                for key, value in (result['aesthetic'] or {}).items()
                if key in ('mode', 'prompt_variant', 'verdict', 'anchor_sha256', 'dims', 'per_dimension')
            },
            # Retain per-flow verdicts and evidence references.
            'verdicts': [
                {
                    key: verdict.get(key)
                    for key in ('fid', 'verdict', 'reason', 'evidence_refs', 'inconclusive_reason', 'overridden')
                }
                for verdict in result['verdicts']
            ],
            # Retain collection status and timing diagnostics.
            'collection': collection,
        },
    )


def _tally(evidence: list[dict[str, Any]]) -> dict[str, int]:
    """Count evidence items by collection status."""
    out: dict[str, int] = {}
    for item in evidence:
        value = str(item.get('status') or 'unknown')
        out[value] = out.get(value, 0) + 1
    return out


__all__ = ['compute_score']
