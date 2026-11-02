from __future__ import annotations

import base64
import logging
import os

from typing import Any

from .ppt_fidelity import FidelityConfig, compute_fidelity_metrics
from .ppt_metrics import (
    OBJECTIVE_METRICS,
    PPTRubricConfig,
    aggregate_objective,
    compute_objective_metrics,
    relative_objective_scores,
    weighted_mean,
)
from .ppt_paste_gate import GOLD_MIN_CHARS, MODEL_MIN_CHARS
from .ppt_pdf_layout import judge_long_edge_px, normalise_source_size
from .ppt_render import count_inkless_decorations
from .ppt_structure import (
    baked_text_ratio,
    extract_elements,
    extract_text_lines,
    lines_from_elements,
    render_pair,
    render_structure_listing,
    summarise,
)
from .ppt_text_diff import compare_text, normalise


logger = logging.getLogger(__name__)


LAYOUT_METRIC_NAMES = tuple(name for name, _ in OBJECTIVE_METRICS)

FIDELITY_METRIC_NAMES = ('canvas_fidelity', 'bounds_fidelity', 'element_iou')

ELEMENT_IOU_FACTORS = ('align_iou', 'coverage', 'matched', 'missed_from_ref', 'hallucinated', 'model_n', 'ref_n')


BAKED_TEXT_GATE = 0.5


FIDELITY_WEIGHTS = {
    'canvas_fidelity': 1.000,
    'element_iou': 0.900,
    'bounds_fidelity': 0.700,
}


def _aggregate_fidelity(scores: dict[str, float | None]) -> tuple[float | None, int]:
    score, info = weighted_mean(scores, FIDELITY_WEIGHTS)
    return score, int(info['n_used'])


_ENVIRONMENT_CACHE: dict[str, Any] | None = None


def render_environment() -> dict[str, Any]:

    global _ENVIRONMENT_CACHE
    if _ENVIRONMENT_CACHE is not None:
        return _ENVIRONMENT_CACHE

    import shutil  # noqa: PLC0415 - probe-only imports
    import subprocess  # noqa: PLC0415

    environment: dict[str, Any] = {}

    soffice = shutil.which('soffice') or shutil.which('libreoffice')
    environment['soffice_path'] = soffice or ''
    if soffice:
        try:
            done = subprocess.run([soffice, '--version'], capture_output=True, text=True, timeout=60)
            environment['soffice_version'] = (done.stdout or done.stderr or '').strip()[:120]
        except Exception as exc:  # noqa: BLE001 - a probe must not fail scoring
            environment['soffice_version'] = f'probe_failed: {type(exc).__name__}'

    if shutil.which('fc-list'):
        try:
            done = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True, timeout=60)
            environment['cjk_font_count'] = len([ln for ln in done.stdout.splitlines() if ln.strip()])
        except Exception as exc:  # noqa: BLE001 - a probe must not fail scoring
            environment['cjk_font_count'] = f'probe_failed: {type(exc).__name__}'
    else:
        environment['cjk_font_count'] = 'fc-list_missing'

    _ENVIRONMENT_CACHE = environment
    return environment


def _gold_ref_texts(reference_geometry: Any, limit: int = 200) -> list[str]:

    if reference_geometry is None:
        return []
    texts: list[str] = []
    for element in getattr(reference_geometry, 'elements', None) or []:
        text = str(getattr(element, 'text', '') or '').strip()
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _visible_chars(lines: list | None) -> int:

    if lines is None:
        return -1
    return sum(len(normalise(ln.text)) for ln in lines if ln.visible)


def _visible_text_gate(
    gold_chars: int,
    gold_source: str,
    cand_lines: list | None,
) -> tuple[bool, dict[str, Any]]:

    detail: dict[str, Any] = {'gold_source': gold_source}
    if gold_chars < 0:
        detail['gate'] = 'no_gold_text_chars'
        return False, detail
    detail['gold_text_chars'] = float(gold_chars)
    if gold_chars < GOLD_MIN_CHARS:
        detail['gate'] = 'gold_below_min'
        return False, detail
    cand = _visible_chars(cand_lines)
    detail['pdf_chars'] = float(cand)
    if cand < 0:
        detail['gate'] = 'undecidable'
        return False, detail
    detail['text_chars_used'] = float(cand)
    detail['text_source'] = 'pdf_visible'
    if cand >= MODEL_MIN_CHARS:
        detail['gate'] = 'ok'
        return False, detail
    detail['gate'] = 'stripped'
    detail['reason'] = (
        f'reference slide shows {gold_chars} characters but the output shows {cand} '
        f'(visible text, gate {MODEL_MIN_CHARS}); the text is not reproduced as text'
    )
    return True, detail


def _select_objective_calibre(
    requested: str, relative_objective: float | None, *, has_ref: bool, reference_parsed: bool
) -> tuple[str, str | None]:
    if requested != 'gold_relative':
        return 'absolute', None
    if relative_objective is not None:
        return 'gold_relative', None
    if not has_ref:
        return 'absolute', 'no reference deck'
    if not reference_parsed:
        return 'absolute', 'the reference deck could not be parsed'
    return 'absolute', 'no layout metric was measured on both decks'


def score_pptx(
    pptx_path: str,
    source_size: list | None = None,
    rubric_config: dict | None = None,
    ref_pptx_path: str = '',
    want_judge_png: bool = True,
    gold_text_chars: int | None = None,
    code_path: str = '',
) -> dict[str, Any]:
    config = dict(rubric_config or {})

    result: dict[str, Any] = {
        'completed': False,
        'error': None,
        'metrics': dict.fromkeys(LAYOUT_METRIC_NAMES + FIDELITY_METRIC_NAMES),
        'objective': None,
        'fidelity': None,
        'text_stripped': False,
        'metrics_used': 0,
        'fidelity_used': 0,
        'unrenderable': False,
        'evaluator_failed': False,
        'diagnostics': {},
        'judge_png_b64': None,
        'ref_texts': [],
        'generator_code': '',
        'objective_relative': None,
        'relative_metrics': {},
        'text_fidelity': None,
        'text_detail': {},
        'diff_report': '',
        'structure': {},
    }

    result['diagnostics'].update(render_environment())

    if not pptx_path or not os.path.exists(pptx_path):
        result['error'] = f'pptx not found: {pptx_path or "<empty>"}'
        result['unrenderable'] = True
        return result

    try:
        agent_code = ''
        if code_path and os.path.exists(code_path):
            try:
                with open(code_path, encoding='utf-8', errors='replace') as handle:
                    agent_code = handle.read()
            except OSError as exc:
                result['diagnostics']['code_read_error'] = f'{type(exc).__name__}'

        weights_override = config.get('weights') or {}
        try:
            cfg = PPTRubricConfig.from_mapping(weights_override)
        except (TypeError, ValueError) as exc:
            logger.error('[ppt_rubric] invalid rubric config: %s', exc)
            result['diagnostics']['config_error'] = str(exc)[:200]
            result['error'] = f'invalid rubric config: {exc}'[:500]
            result['evaluator_failed'] = True
            return result

        geometry, diagnostics = _build_geometry(
            pptx_path,
            enable_pdf_measure=bool(config.get('enable_pdf_measure', True)),
            enable_raster=bool(config.get('enable_raster', False)),
            pdf_timeout=float(config.get('pdf_timeout', 120.0)),
            raster_timeout=float(config.get('raster_timeout', 60.0)),
            judge_long_edge=judge_long_edge_px(source_size),
        )
        result['diagnostics'].update(diagnostics)

        if geometry is None:
            result['error'] = diagnostics.get('geometry_error', 'geometry parse failed')
            result['unrenderable'] = True
            return result

        if diagnostics.get('n_slides') != 1:
            result['error'] = f'expected exactly one slide, got {diagnostics.get("n_slides")!r}'
            result['unrenderable'] = True
            return result

        if config.get('enable_pdf_measure', True) and geometry.judge_png is None:
            result['evaluator_failed'] = True
            result['error'] = str(
                diagnostics.get('raster_error') or diagnostics.get('pdf_error') or 'no rendered page'
            )[:200]
            return result

        if diagnostics.get('pdf_path'):
            inkless, inkless_diag = count_inkless_decorations(
                pptx_path, geometry, diagnostics['pdf_path'], timeout=float(config.get('pdf_timeout', 120.0))
            )
            result['diagnostics'].update(inkless_diag)
            if inkless is not None:
                result['diagnostics']['decorative_inkless'] = float(inkless)

        layout_scores, layout_details = compute_objective_metrics(geometry, cfg)
        objective, objective_info = aggregate_objective(layout_scores, cfg)
        for name, value in layout_scores.items():
            result['metrics'][name] = None if value is None else float(value)
            error = (layout_details.get(name) or {}).get('error')
            if error:
                result['diagnostics'][f'{name}_error'] = str(error)[:200]
        result['objective'] = None if objective is None else float(objective)
        result['metrics_used'] = int(objective_info.get('n_used', 0))

        reference_geometry = None
        ref_diagnostics: dict[str, Any] = {}
        has_ref = bool(ref_pptx_path and os.path.exists(ref_pptx_path))
        if has_ref:
            reference_geometry, ref_diagnostics = _build_geometry(
                ref_pptx_path,
                enable_pdf_measure=bool(config.get('ref_pdf_measure', True)),
                enable_raster=False,
                pdf_timeout=float(config.get('pdf_timeout', 120.0)),
                raster_timeout=float(config.get('raster_timeout', 60.0)),
                judge_long_edge=None,
            )
            result['diagnostics']['ref_render_ok'] = reference_geometry is not None
            if reference_geometry is None:
                result['diagnostics']['ref_render_error'] = str(ref_diagnostics.get('geometry_error', 'unknown'))[:200]

        relative_objective = None
        relative_used = 0
        if reference_geometry is not None:
            gold_scores, gold_details = compute_objective_metrics(reference_geometry, cfg)
            relative_scores, relative_info = relative_objective_scores(gold_details, layout_details, cfg)
            relative_objective, relative_agg = weighted_mean(relative_scores, cfg.objective_weights())
            relative_used = int(relative_agg.get('n_used', 0))
            result['objective_relative'] = None if relative_objective is None else float(relative_objective)
            result['relative_metrics'] = {
                name: (None if value is None else float(value)) for name, value in relative_scores.items()
            }
            result['diagnostics']['relative_raw_pairs'] = relative_info.get('raw_pairs') or {}
            result['diagnostics']['relative_metrics_used'] = relative_used
            result['diagnostics']['gold_objective'] = (
                None if (gold_objective := aggregate_objective(gold_scores, cfg)[0]) is None else float(gold_objective)
            )
        calibre, fallback = _select_objective_calibre(
            cfg.objective_mode, relative_objective, has_ref=has_ref, reference_parsed=reference_geometry is not None
        )
        result['diagnostics']['objective_mode'] = calibre
        if fallback:
            result['diagnostics']['objective_mode_fallback'] = fallback
        if calibre == 'gold_relative':
            assert relative_objective is not None, 'gold_relative is selected only when a relative objective exists'
            result['objective'] = float(relative_objective)
            result['metrics_used'] = relative_used

        cand_structure = extract_elements(pptx_path)
        cand_lines, cand_renders = _rendered_lines(
            diagnostics.get('pdf_path'), cand_structure, result['diagnostics'], 'cand'
        )
        ref_structure = extract_elements(ref_pptx_path) if has_ref else None
        ref_lines, ref_renders = (
            _rendered_lines(ref_diagnostics.get('pdf_path'), ref_structure, result['diagnostics'], 'ref')
            if ref_structure is not None
            else (None, None)
        )
        if config.get('enable_pdf_measure', True) and (
            cand_lines is None or cand_renders is None or (has_ref and (ref_lines is None or ref_renders is None))
        ):
            result['error'] = 'rendered text visibility evidence unavailable'
            result['evaluator_failed'] = True
            return result
        cand_summary = summarise(cand_structure)
        ref_summary = summarise(ref_structure) if ref_structure is not None else None
        result['structure'] = {'cand': cand_summary, 'ref': ref_summary}
        cand_visible_chars = _visible_chars(cand_lines)
        ref_visible_chars = _visible_chars(ref_lines) if ref_lines is not None else -1
        result['diagnostics']['cand_visible_chars'] = float(cand_visible_chars)
        result['diagnostics']['ref_visible_chars'] = float(ref_visible_chars)
        result['cand_structure_listing'] = render_structure_listing(cand_structure, visible_chars=cand_visible_chars)
        result['ref_structure_listing'] = (
            render_structure_listing(ref_structure, visible_chars=ref_visible_chars)
            if ref_structure is not None
            else ''
        )

        if ref_lines:
            result['ref_texts'] = [ln.text for ln in ref_lines if ln.visible][:200]
        else:
            result['ref_texts'] = _gold_ref_texts(reference_geometry)
        result['diagnostics']['ref_texts_n'] = len(result['ref_texts'])
        result['generator_code'] = agent_code
        result['diagnostics']['generator_chars'] = len(agent_code)

        if cfg.w_text > 0 and ref_lines is not None:
            diff = compare_text(
                ref_lines,
                cand_lines or [],
                aspect=_canvas_aspect(source_size, cand_structure.slide_aspect),
                gold_summary=ref_summary,
                cand_summary=cand_summary,
            )
            result['text_fidelity'] = None if diff.score is None else float(diff.score)
            result['text_detail'] = diff.detail
            result['diff_report'] = diff.report
            if diff.detail.get('truncated'):
                result['error'] = 'text comparison exceeded its limit'
                result['evaluator_failed'] = True
                return result

        fidelity_scores, fidelity_details = compute_fidelity_metrics(
            geometry,
            ref_geometry=reference_geometry,
            source_size=normalise_source_size(source_size),
            cfg=FidelityConfig(),
        )
        for name, value in fidelity_scores.items():
            result['metrics'][name] = None if value is None else float(value)
            error = (fidelity_details.get(name) or {}).get('error')
            if error:
                result['diagnostics'][f'{name}_error'] = str(error)[:200]
        element_detail = fidelity_details.get('element_iou') or {}
        for key in ELEMENT_IOU_FACTORS:
            value = element_detail.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result['diagnostics'][f'element_{key}'] = float(value)
        fidelity, fidelity_used = _aggregate_fidelity(fidelity_scores)
        result['fidelity'] = None if fidelity is None else float(fidelity)
        result['fidelity_used'] = fidelity_used

        if isinstance(gold_text_chars, (int, float)) and int(gold_text_chars) >= 0:
            gold_chars, gold_source = int(gold_text_chars), 'dataset'
        else:
            gold_chars, gold_source = _visible_chars(ref_lines), 'reference_render'
        stripped, gate_detail = _visible_text_gate(gold_chars, gold_source, cand_lines)

        baked = None
        if ref_lines and ref_renders is not None and cand_renders is not None:
            baked = baked_text_ratio(ref_lines, ref_renders, cand_renders)
        result['baked_text_ratio'] = baked
        result['diagnostics']['baked_text_ratio'] = baked
        if baked is not None and baked >= BAKED_TEXT_GATE and gold_chars >= GOLD_MIN_CHARS and not stripped:
            stripped = True
            gate_detail['gate'] = 'baked_text'
            gate_detail['reason'] = (
                f"{baked:.0%} of the reference text appears as pixels inside the output's pictures "
                f'(gate {BAKED_TEXT_GATE:.0%}); the slide is pasted, not reconstructed'
            )
        result['diagnostics'].update({f'paste_{k}': v for k, v in gate_detail.items()})
        result['text_stripped'] = bool(stripped)
        if stripped:
            result['objective'] = 0.0
            result['fidelity'] = 0.0
            result['text_fidelity'] = 0.0
            for name in LAYOUT_METRIC_NAMES:
                result['metrics'][name] = 0.0
            if result['objective_relative'] is not None:
                result['objective_relative'] = 0.0
            result['relative_metrics'] = dict.fromkeys(result['relative_metrics'], 0.0)
            logger.warning('[ppt_rubric] paste gate fired: %s', gate_detail.get('reason', ''))

        if want_judge_png and geometry.judge_png:
            result['judge_png_b64'] = base64.b64encode(geometry.judge_png).decode('ascii')

        result['completed'] = True
        return result
    except Exception as exc:  # noqa: BLE001 - the contract is to never raise
        logger.exception('[ppt_rubric] scoring failed for %s', pptx_path)
        result['error'] = f'{type(exc).__name__}: {exc}'[:500]
        result['evaluator_failed'] = True
        return result


def _rendered_lines(
    pdf_path: str | None, structure: Any, diagnostics: dict[str, Any], side: str
) -> tuple[list | None, Any]:

    if pdf_path and os.path.exists(pdf_path):
        renders, rdiag = render_pair(pdf_path)
        diagnostics.update({f'{side}_{k}': v for k, v in rdiag.items()})
        lines, diag = extract_text_lines(pdf_path, renders=renders)
        diagnostics.update({f'{side}_{k}': v for k, v in diag.items()})
        if lines is not None:
            diagnostics[f'{side}_lines_source'] = 'pdf'
            return lines, renders
    if structure is not None and not structure.error:
        diagnostics[f'{side}_lines_source'] = 'structure'
        return lines_from_elements(structure), None
    diagnostics[f'{side}_lines_source'] = 'unavailable'
    if structure is not None and structure.error:
        diagnostics[f'{side}_structure_error'] = structure.error[:200]
    return None, None


def _canvas_aspect(source_size: list | None, slide_aspect: float | None) -> float:

    normalised = normalise_source_size(source_size)
    if normalised is not None:
        return normalised[0] / normalised[1]
    return float(slide_aspect) if slide_aspect and slide_aspect > 0 else 16.0 / 9.0


def _build_geometry(
    pptx_path: str,
    *,
    enable_pdf_measure: bool,
    enable_raster: bool,
    pdf_timeout: float,
    raster_timeout: float,
    judge_long_edge: int | None,
) -> tuple[Any, dict[str, Any]]:

    from .ppt_pdf_layout import extract_text_layout, pptx_to_pdf, render_pdf_page_png
    from .ppt_render import geometry_from_pptx, rasterise_pptx

    diagnostics: dict[str, Any] = {}
    workdir = os.path.dirname(os.path.abspath(pptx_path))

    pdf_layout = None
    judge_png = None
    if enable_pdf_measure:
        try:
            pdf_path, pdf_diagnostics = pptx_to_pdf(pptx_path, workdir, timeout=pdf_timeout)
            diagnostics.update(pdf_diagnostics)
            if pdf_path:
                diagnostics['pdf_path'] = pdf_path
                pdf_layout, layout_diagnostics = extract_text_layout(pdf_path)
                diagnostics.update(layout_diagnostics)
                judge_png, png_diagnostics = render_pdf_page_png(pdf_path, target_long_edge=judge_long_edge)
                diagnostics.update(png_diagnostics)
        except Exception as exc:  # noqa: BLE001
            diagnostics['pdf_stage_error'] = f'{type(exc).__name__}: {exc}'[:200]

    slide_png = None
    if enable_raster:
        try:
            slide_png, raster_diagnostics = rasterise_pptx(pptx_path, workdir, timeout=raster_timeout)
            diagnostics.update(raster_diagnostics)
        except Exception as exc:  # noqa: BLE001
            diagnostics['raster_stage_error'] = f'{type(exc).__name__}: {exc}'[:200]

    try:
        geometry, geometry_diagnostics = geometry_from_pptx(
            pptx_path, pdf_layout=pdf_layout, local_bg_pdf_path=diagnostics.get('pdf_path')
        )
        diagnostics.update(geometry_diagnostics)
    except Exception as exc:  # noqa: BLE001 - the contract is to never raise
        diagnostics['geometry_error'] = f'{type(exc).__name__}: {exc}'[:200]
        return None, diagnostics

    if geometry is None:
        diagnostics.setdefault('geometry_error', 'geometry_from_pptx returned None')
        return None, diagnostics

    if slide_png is not None:
        geometry.image_bytes = slide_png
    if judge_png is not None:
        geometry.judge_png = judge_png
        if geometry.image_bytes is None:
            geometry.image_bytes = judge_png
            diagnostics['raster'] = 'from_pdf'
    diagnostics['n_elements'] = len(geometry.elements)
    diagnostics['pdf_measured'] = pdf_layout is not None
    diagnostics['n_text_measured'] = int(geometry_diagnostics.get('n_text_measured', 0) or 0)
    return geometry, diagnostics


__all__ = [
    'FIDELITY_METRIC_NAMES',
    'LAYOUT_METRIC_NAMES',
    'score_pptx',
]
