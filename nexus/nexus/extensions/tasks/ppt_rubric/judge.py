from __future__ import annotations

import base64
import contextlib
import json
import logging
import math
import os
import uuid

from typing import Any

from nexus.extensions.tasks.mm_agent.reward_system.reward_platform_client import (
    build_expected_sub_service_classes,
    extract_sub_service_data,
    get_reward_platform_token,
    get_reward_platform_url,
    parse_reward_platform_router_id,
    post_to_reward_platform,
)
from nexus.utils.reward_score import ReasoningReward

from .ppt_judge_prompt import (
    PPT_JUDGE_SYSTEM,
    SCALE_MAX,
    SCALE_MIN,
    SUBJECTIVE_DIMENSION_WEIGHTS,
    aggregate_subjective,
    build_judge_prompt,
    normalise_dimension_score,
)
from .ppt_metrics import PPTRubricConfig
from .ppt_paste_gate import count_code_chars, decide_from_counts
from .ppt_pdf_layout import fit_image_long_edge


logger = logging.getLogger(__name__)


FORMAT_FAILURE_REWARD = 0.0

NOT_MEASURED = -1.0
JUDGE_NOT_JUDGEABLE = -1.0


FusionWeights = PPTRubricConfig


DEFAULT_JUDGE_LONG_EDGE = 1280


ACCURACY_MODE_CODES: dict[str, float] = {
    'none': 0.0,
    'ref_code': 1.0,
    'diff_report': 2.0,
    'structure_listing': 3.0,
}
UNKNOWN_ACCURACY_MODE = -1.0

JUDGE_SIZE_KEYS = ('accuracy_prompt_chars', 'ref_listing_chars', 'cand_listing_chars')


def _judge_payload(sub_data: dict[str, Any]) -> dict[str, Any]:

    judged = sub_data.get('details')

    if not isinstance(judged, dict):
        judged = sub_data.get('res')
        if isinstance(judged, str):
            with contextlib.suppress(TypeError, ValueError):
                judged = json.loads(judged)

    return dict(judged) if isinstance(judged, dict) else dict(sub_data)


def _parse_judge_payload(res: Any) -> tuple[dict[str, float | None], dict[str, Any]]:
    dims: dict[str, float | None] = {}
    detail: dict[str, Any] = {}

    if not isinstance(res, dict):
        return dims, detail

    payload = res

    mode = payload.get('accuracy_mode')
    if mode is not None:
        detail['judge_accuracy_mode'] = ACCURACY_MODE_CODES.get(str(mode), UNKNOWN_ACCURACY_MODE)
    for key in JUDGE_SIZE_KEYS:
        if key not in payload:
            continue
        with contextlib.suppress(TypeError, ValueError):
            size = float(payload[key])
            if math.isfinite(size):
                detail[f'judge_{key}'] = size

    if any(f'subj_{d}' in payload for d in SUBJECTIVE_DIMENSION_WEIGHTS):
        for dim in SUBJECTIVE_DIMENSION_WEIGHTS:
            key = f'subj_{dim}'
            if key not in payload:
                continue
            if isinstance(payload[key], bool):
                dims[dim] = None
                detail[key] = JUDGE_NOT_JUDGEABLE
                continue
            try:
                norm = float(payload[key])
            except (TypeError, ValueError):
                dims[dim] = None
                detail[key] = JUDGE_NOT_JUDGEABLE
                continue

            if not math.isfinite(norm) or norm < 0 or norm > 1:
                dims[dim] = None
                detail[key] = JUDGE_NOT_JUDGEABLE
                continue
            value = norm
            dims[dim] = value
            detail[key] = round(value, 4)
            raw = payload.get(f'{key}_raw')
            if raw is not None:
                with contextlib.suppress(TypeError, ValueError):
                    raw_value = float(raw)
                    if math.isfinite(raw_value):
                        detail[f'{key}_raw'] = raw_value
        for dim in SUBJECTIVE_DIMENSION_WEIGHTS:
            reason = payload.get(f'subj_{dim}_reason')
            if reason:
                detail[f'subj_{dim}_reason'] = str(reason)[:300]
        if 'source_seen' in payload:
            with contextlib.suppress(TypeError, ValueError):
                detail['judge_source_seen'] = float(payload['source_seen'])
        if dims:
            skip_reason = payload.get('visual_skip')
            if skip_reason:
                detail['subj_visual_skip'] = str(skip_reason)[:64]
            if payload.get('render_png_rejected'):
                detail['subj_render_png_rejected'] = 1.0
            return dims, detail

    for dim in SUBJECTIVE_DIMENSION_WEIGHTS:
        if dim not in payload:
            continue
        entry = payload[dim]
        raw = entry.get('score') if isinstance(entry, dict) else entry
        norm = normalise_dimension_score(raw)
        dims[dim] = norm
        if norm is not None:
            detail[f'subj_{dim}_raw'] = raw if math.isfinite(float(raw)) else None
            detail[f'subj_{dim}'] = round(norm, 4)
        else:
            detail[f'subj_{dim}'] = JUDGE_NOT_JUDGEABLE

    return dims, detail


def fuse(
    *,
    objective: float | None,
    fidelity: float | None,
    subjective: float | None,
    canvas: float | None,
    cfg: Any,
    out: dict[str, Any],
    text: float | None = None,
    baked: float | None = None,
) -> float:
    for name, value in {
        'objective': objective,
        'fidelity': fidelity,
        'subjective': subjective,
        'canvas': canvas,
        'text': text,
        'baked': baked,
    }.items():
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value))):
            raise ValueError(f'{name} must be finite or None, got {value!r}; reject the scorer response')
    parts: list[tuple[float, float]] = []
    if objective is not None:
        parts.append((objective, cfg.w_objective))
    if fidelity is not None:
        parts.append((fidelity, cfg.w_fidelity))
    if subjective is not None:
        parts.append((subjective, cfg.w_subjective))
    w_text = float(getattr(cfg, 'w_text', 0.0) or 0.0)
    if text is not None and w_text > 0:
        parts.append((text, w_text))

    if not parts:
        logger.error('[ppt_rubric] neither objective nor subjective available; scoring 0')
        out['ppt_scoring_failed'] = 1.0
        out['ppt_fused_score'] = FORMAT_FAILURE_REWARD
        return FORMAT_FAILURE_REWARD

    if any(not math.isfinite(float(weight)) or weight < 0 for _, weight in parts):
        raise ValueError('Fusion weights must be finite and nonnegative; correct rubric_config.weights')
    total_w = sum(w for _, w in parts)
    if total_w <= 0:
        raise ValueError('At least one available fusion component must have positive weight')
    fused = sum(v * w for v, w in parts) / total_w

    if canvas is not None:
        if not math.isfinite(cfg.canvas_gate_floor) or not 0 <= cfg.canvas_gate_floor <= 1:
            raise ValueError('canvas_gate_floor must be finite and between 0 and 1')
        gate = cfg.canvas_gate_floor + (1.0 - cfg.canvas_gate_floor) * max(0.0, min(1.0, canvas))
        out['ppt_canvas_gate'] = round(gate, 4)
        fused *= gate

    start = float(getattr(cfg, 'baked_soft_start', 1.0))
    floor = float(getattr(cfg, 'baked_floor', 1.0))
    if not math.isfinite(start) or not math.isfinite(floor) or not 0 <= start <= 1 or not 0 <= floor <= 1:
        raise ValueError('Baked gate start and floor must be finite and between 0 and 1')
    if baked is not None and baked > start:
        t = min(1.0, (baked - start) / max(1e-6, 0.5 - start))
        bgate = 1.0 - (1.0 - floor) * t
        out['ppt_baked_gate'] = round(bgate, 4)
        fused *= bgate

    fused = max(0.0, min(1.0, fused))

    out['ppt_fused_score'] = fused
    return fused


async def subjective_score(
    *,
    prompt_str: str,
    response: str,
    router_id: str,
    ref_texts: list[str],
    ref_png_path: str = '',
    render_png: bytes | None = None,
    ref_code: str = '',
    judge_long_edge_px: int | None = None,
    generated_listing: str = '',
    reference_listing: str = '',
    require_source: bool = True,
) -> tuple[float | None, dict[str, Any]]:
    if not router_id:
        return None, {'subjective_skip': 'no_router_id'}

    token = get_reward_platform_token()
    if not token:
        return None, {'subjective_skip': 'no_user_token'}

    assistant_text = (generated_listing or response or '').strip()
    judge_extra: dict[str, Any] = {
        'scoring_mode': 'ppt_subjective',
        'dimensions': list(SUBJECTIVE_DIMENSION_WEIGHTS),
        'scale': [SCALE_MIN, SCALE_MAX],
        'ref_texts': ref_texts,
        'ref_png_path': ref_png_path,
        'judge_system': PPT_JUDGE_SYSTEM,
        'gold': {
            'ref_texts': ref_texts,
            'diff_report': '',
            'ref_structure_listing': reference_listing,
            'require_source': bool(require_source),
        },
    }

    if render_png:
        uri = 'data:image/png;base64,' + base64.b64encode(render_png).decode('ascii')
        judge_extra['render_png'] = uri
        judge_extra['render_png_bytes'] = len(render_png)
        judge_extra['gold']['render_png'] = uri
    detail_png_bytes = float(len(render_png)) if render_png else 0.0

    source_uri = ''
    if ref_png_path and os.path.exists(ref_png_path):
        long_edge = judge_long_edge_px or DEFAULT_JUDGE_LONG_EDGE
        fitted, fit_diag = fit_image_long_edge(ref_png_path, long_edge)
        if fitted:
            source_uri = 'data:image/png;base64,' + base64.b64encode(fitted).decode('ascii')
            judge_extra['source_png_bytes'] = len(fitted)
            judge_extra['judge_long_edge'] = long_edge

            judge_extra['gold']['source_png'] = source_uri
        else:
            logger.warning('[ppt_rubric] source image fit failed: %s', fit_diag.get('fit_error'))

    judge_extra['judge_prompt'] = build_judge_prompt(
        generated_listing=assistant_text,
        reference_listing=reference_listing,
        source_available=bool(source_uri),
    )

    if ref_code:
        judge_extra['ref_code'] = ref_code
        judge_extra['gold']['ref_code'] = ref_code

    user_content: Any = prompt_str
    if source_uri:
        user_content = [
            {'type': 'image_url', 'image_url': {'url': source_uri}},
            {'type': 'text', 'text': prompt_str or 'Reproduce this slide.'},
        ]

    try:
        class1, class2, router_version = parse_reward_platform_router_id(router_id)
    except ValueError as exc:
        return None, {'subjective_error': str(exc)[:200], 'judge_png_bytes': detail_png_bytes}

    payload = {
        'request_id': f'ppt_judge_{uuid.uuid4().hex}',
        'exp_id': os.environ.get('REWARD_PLATFORM_EXP_ID', 'ppt_topic'),
        'source': 'dpo',
        'model_type': os.environ.get('REWARD_PLATFORM_MODEL_TYPE', 'hy_turbo_v2.07'),
        'messages': [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_text},
        ],
        'ref_answer': ref_code,
        'class1': class1,
        'class2': class2,
        'router_version': router_version,
        'extra': judge_extra,
        'public_test': [],
    }

    headers = {'Content-Type': 'application/json', 'token': token}
    try:
        res = await post_to_reward_platform(get_reward_platform_url(), payload, headers)
    except Exception as exc:  # noqa: BLE001 - a judge outage must not kill the rollout
        logger.warning('[ppt_rubric] call failed: %s', exc)
        return None, {'subjective_error': f'{type(exc).__name__}: {exc}'[:200], 'judge_png_bytes': detail_png_bytes}

    sub_data, extract_error, sub_class = extract_sub_service_data(
        res,
        build_expected_sub_service_classes(class1, class2),
        fallback_class_suffix=class2,
    )
    if sub_data is None:
        logger.warning('[ppt_rubric] judge unavailable: %s', extract_error)
        return None, {
            'subjective_error': f'judge error: {extract_error}'[:200],
            'judge_png_bytes': detail_png_bytes,
        }

    res = _judge_payload(sub_data)
    dims, detail = _parse_judge_payload(res)
    detail['subjective_sub_service'] = str(sub_class or '')
    subjective, agg_info = aggregate_subjective(dims)
    if subjective is not None:
        detail['subj_dims_used'] = float(len(agg_info['used']))
        detail['judge_png_bytes'] = detail_png_bytes
        return subjective, detail

    raw = res.get('score') if isinstance(res, dict) else res
    if isinstance(res, dict) and raw is None:
        raw = res.get('reward_score', res.get('final_score'))
    if isinstance(raw, bool):
        return None, {'subjective_error': 'boolean judge score', 'judge_png_bytes': detail_png_bytes}
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return None, {'subjective_error': f'unparseable judge score: {raw!r}', 'judge_png_bytes': detail_png_bytes}
    if not math.isfinite(raw):
        return None, {'subjective_error': f'nonfinite judge score: {raw!r}', 'judge_png_bytes': detail_png_bytes}

    if raw <= ReasoningReward.THRESHOLD.value:
        return None, {'subjective_error': f'judge sentinel {raw}', 'judge_png_bytes': detail_png_bytes}
    detail['judge_png_bytes'] = detail_png_bytes

    if raw > 1.0:
        return None, {'subjective_error': f'judge score {raw} exceeds [0, 1]', 'judge_png_bytes': detail_png_bytes}
    detail['subjective_raw'] = round(raw, 4)
    detail['subjective_scalar_fallback'] = 1.0
    if raw == 0.0:
        detail['subjective_bare_zero'] = 1.0
        detail['subjective_error'] = 'bare 0.0 without dimensions (empty answer / no judgement)'
        return None, detail
    return max(0.0, min(1.0, raw)), detail


_METRIC_NAMES = (
    ('text_density', None),
    ('imbalance', None),
    ('text_overflow', None),
    ('whitespace', None),
    ('collision', None),
    ('contrast', None),
)


_FIDELITY_NAMES = ('canvas_fidelity', 'bounds_fidelity', 'element_iou')

_ELEMENT_IOU_FACTORS = ('align_iou', 'coverage', 'matched', 'missed_from_ref', 'hallucinated', 'model_n', 'ref_n')


def decode_judge_png(encoded: Any) -> bytes | None:

    if not encoded or not isinstance(encoded, str):
        return None
    try:
        return base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001 - a bad render must not fail scoring
        logger.warning('[ppt_rubric] could not decode sandbox judge png: %s', exc)
        return None


def paste_gate_columns(diagnostics: dict, ref_code: str) -> dict[str, Any]:
    gate = {key[len('paste_') :]: value for key, value in diagnostics.items() if key.startswith('paste_')}
    columns: dict[str, Any] = {}
    for key in ('gold_text_chars', 'code_chars', 'pdf_chars', 'text_chars_used'):
        if isinstance(gate.get(key), (int, float)):
            columns[f'ppt_{key}'] = float(gate[key])
    if gate.get('gate'):
        columns['ppt_text_gate'] = str(gate['gate'])
    if gate.get('text_source'):
        columns['ppt_text_source'] = str(gate['text_source'])

    if gate.get('gate') != 'no_gold_text_chars' or not ref_code:
        return columns

    gold_chars, gold_how = count_code_chars(ref_code)
    if gold_chars < 0:
        columns['ppt_text_gate'] = f'host_ref_code_{gold_how}'
        return columns
    columns['ppt_gold_text_chars'] = float(gold_chars)

    stripped, detail = decide_from_counts(
        gold_chars,
        float(gate.get('code_chars', -1)),
        float(gate.get('pdf_chars', -1)),
        full_page_picture=bool(gate.get('full_page_picture')),
    )
    for key in ('gold_text_chars', 'code_chars', 'pdf_chars', 'text_chars_used', 'model_floor'):
        if isinstance(detail.get(key), (int, float)):
            columns[f'ppt_{key}'] = float(detail[key])
    if detail.get('text_source'):
        columns['ppt_text_source'] = f'host_{detail["text_source"]}'

    verdict = str(detail.get('gate') or '')
    if verdict == 'stripped':
        columns['ppt_text_gate'] = 'host_stripped'
        columns['stripped'] = True
        columns['ppt_text_stripped_reason'] = str(detail.get('reason') or '')[:200]
    elif verdict in ('ok', 'undecidable', 'gold_below_min'):
        columns['ppt_text_gate'] = f'host_{verdict}'
    return columns


_BARE_DETAIL_KEYS = frozenset(
    [f'subj_{dim}' for dim in SUBJECTIVE_DIMENSION_WEIGHTS]
    + [f'subj_{dim}_raw' for dim in SUBJECTIVE_DIMENSION_WEIGHTS]
)


def detail_column(key: str) -> str:

    if key in _BARE_DETAIL_KEYS:
        return key
    return key if key.startswith('ppt_') else f'ppt_{key}'


def mirror_into_reward_extra_info(extra_fields: dict[str, Any]) -> None:
    reward_extra_info = extra_fields.get('reward_extra_info')
    if not isinstance(reward_extra_info, dict):
        reward_extra_info = {}
        extra_fields['reward_extra_info'] = reward_extra_info
    for key, value in extra_fields.items():
        if key == 'reward_extra_info' or not key.startswith(('ppt_', 'subj_')):
            continue

        if isinstance(value, (int, float)):
            reward_extra_info[key] = float(value)


def stable_columns() -> dict[str, Any]:
    cols: dict[str, Any] = {
        'ppt_render_ok': 0.0,
        'ppt_raster_fallback': 0.0,
        'ppt_pdf_measured': 0.0,
        'ppt_n_text_measured': 0.0,
        'ppt_objective': NOT_MEASURED,
        'ppt_subjective': NOT_MEASURED,
        'ppt_subjective_ok': 0.0,
        'ppt_metrics_used': 0.0,
        'ppt_subj_dims_used': 0.0,
        'ppt_judge_png_bytes': 0.0,
        'ppt_judge_accuracy_mode': NOT_MEASURED,
    }
    for key in JUDGE_SIZE_KEYS:
        cols[f'ppt_judge_{key}'] = 0.0
    for name, _ in _METRIC_NAMES:
        if f'ppt_{name}' in cols:
            raise RuntimeError(f'generated column ppt_{name} collides with an explicit key')
        cols[f'ppt_{name}'] = NOT_MEASURED
        if f'ppt_rel_{name}' in cols:
            raise RuntimeError(f'generated column ppt_rel_{name} collides with an explicit key')
        cols[f'ppt_rel_{name}'] = NOT_MEASURED
    cols['ppt_objective_relative'] = NOT_MEASURED
    cols['ppt_gold_objective'] = NOT_MEASURED
    cols['ppt_relative_metrics_used'] = 0.0
    cols['ppt_decorative_inkless'] = NOT_MEASURED
    for name in _FIDELITY_NAMES:
        if f'ppt_{name}' in cols:
            raise RuntimeError(f'generated column ppt_{name} collides with an explicit key')
        cols[f'ppt_{name}'] = NOT_MEASURED
    cols['ppt_fidelity'] = NOT_MEASURED
    cols['ppt_fidelity_used'] = 0.0
    for name in _ELEMENT_IOU_FACTORS:
        cols[f'ppt_element_{name}'] = NOT_MEASURED
    cols['ppt_canvas_gate'] = NOT_MEASURED

    cols['ppt_fused_score'] = NOT_MEASURED

    cols['ppt_unrenderable'] = 0.0

    cols['ppt_text_stripped'] = 0.0
    cols['ppt_gold_text_chars'] = NOT_MEASURED
    cols['ppt_code_chars'] = NOT_MEASURED
    cols['ppt_pdf_chars'] = NOT_MEASURED
    cols['ppt_text_chars_used'] = NOT_MEASURED
    cols['ppt_model_floor'] = NOT_MEASURED

    cols['ppt_text_gate'] = ''
    cols['ppt_text_source'] = ''
    for dim in SUBJECTIVE_DIMENSION_WEIGHTS:
        cols[f'subj_{dim}'] = NOT_MEASURED
        cols[f'subj_{dim}_raw'] = NOT_MEASURED

    cols['ppt_subj_visual_skip'] = ''

    cols['ppt_subj_render_png_rejected'] = 0.0

    cols['ppt_subjective_sub_service'] = ''

    cols['ppt_subjective_error'] = ''
    cols['ppt_subjective_skip'] = ''

    cols['ppt_subjective_scalar_fallback'] = 0.0
    cols['ppt_subjective_raw'] = NOT_MEASURED

    cols['ppt_subjective_bare_zero'] = 0.0

    cols['ppt_scoring_failed'] = 0.0
    cols['ppt_scoring_error'] = ''
    cols['ppt_text_stripped_reason'] = ''

    for name in TEXT_FIDELITY_COLUMNS:
        if name in cols:
            raise RuntimeError(f'generated column {name} collides with an explicit key')
        cols[name] = NOT_MEASURED

    cols['ppt_diff_report'] = ''

    cols['ppt_judge_source_seen'] = NOT_MEASURED

    for dim in SUBJECTIVE_DIMENSION_WEIGHTS:
        cols[f'ppt_subj_{dim}_reason'] = ''

    cols['ppt_infra_missing'] = 0.0
    cols['ppt_text_truncated'] = 0.0

    cols['ppt_baked_text_ratio'] = NOT_MEASURED
    cols['ppt_baked_gate'] = 1.0

    cols['ppt_generator_found'] = 0.0
    return cols


TEXT_FIDELITY_COLUMNS = (
    'ppt_text_fidelity',
    'ppt_text_recall',
    'ppt_text_precision',
    'ppt_text_position',
    'ppt_text_size',
    'ppt_text_color',
    'ppt_text_hidden_chars',
)


def text_fidelity_columns(result: dict[str, Any]) -> dict[str, Any]:

    detail = result.get('text_detail') or {}
    score = result.get('text_fidelity')
    cols: dict[str, Any] = {'ppt_text_fidelity': -1.0 if score is None else float(score)}
    for key in ('recall', 'precision', 'position', 'size', 'color'):
        value = detail.get(key)
        cols[f'ppt_text_{key}'] = float(value) if isinstance(value, (int, float)) else -1.0
    hidden = detail.get('hidden_chars')
    cols['ppt_text_hidden_chars'] = float(hidden) if isinstance(hidden, (int, float)) else -1.0
    cols['ppt_text_truncated'] = float(bool(detail.get('truncated')))
    cols['ppt_diff_report'] = str(result.get('diff_report') or '')[:4000]
    return cols
