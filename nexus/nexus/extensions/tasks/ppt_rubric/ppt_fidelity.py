from __future__ import annotations

import math

from dataclasses import dataclass

from .ppt_metrics import SlideGeometry, smoothstep_reward


@dataclass
class FidelityConfig:
    canvas_ratio_tol: float = 0.02
    canvas_ratio_bad: float = 0.30

    bounds_slack: float = 0.02
    bounds_good: float = 0.01
    bounds_bad: float = 0.20

    iou_match_floor: float = 0.20

    iou_good: float = 0.70
    iou_bad: float = 0.30

    coverage_good: float = 0.80
    coverage_bad: float = 0.20

    count_log_tol: float = 0.5
    count_log_bad: float = 1.4

    count_floor: float = 0.25

    smoothstep_gamma: float = 1.0


def _norm_boxes(geo: SlideGeometry) -> list[tuple[float, float, float, float, str]]:

    w = max(1.0, geo.view_width)
    h = max(1.0, geo.view_height)
    out = []
    for e in geo.elements:
        if e.area <= 0:
            continue
        out.append((e.x / w, e.y / h, e.width / w, e.height / h, e.kind))
    return out


def _iou(a: tuple[float, float, float, float, str], b: tuple[float, float, float, float, str]) -> float:
    ax, ay, aw, ah, _ = a
    bx, by, bw, bh, _ = b
    left = max(ax, bx)
    right = min(ax + aw, bx + bw)
    top = max(ay, by)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _greedy_match(
    model: list, ref: list, *, floor: float, kind_must_match: bool = True
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    pairs: list[tuple[float, int, int]] = []
    for i, m in enumerate(model):
        for j, r in enumerate(ref):
            if kind_must_match and m[4] != r[4]:
                continue
            v = _iou(m, r)
            if v >= floor:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)

    used_m: set[int] = set()
    used_r: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for v, i, j in pairs:
        if i in used_m or j in used_r:
            continue
        used_m.add(i)
        used_r.add(j)
        matched.append((i, j, v))

    unmatched_m = set(range(len(model))) - used_m
    unmatched_r = set(range(len(ref))) - used_r
    return matched, unmatched_m, unmatched_r


def metric_canvas_fidelity(
    geo: SlideGeometry, source_size: tuple[float, float] | None, cfg: FidelityConfig
) -> tuple[float | None, dict]:
    if not source_size or source_size[0] <= 0 or source_size[1] <= 0:
        return None, {'reason': 'no_source_size'}
    got = getattr(geo, 'slide_aspect', None)
    if got is None or got <= 0:
        return None, {'reason': 'renderer_reported_no_slide_aspect'}

    want = float(source_size[0]) / float(source_size[1])
    got = float(got)

    err = abs(got - want) / want

    score = smoothstep_reward(
        err,
        lower=cfg.canvas_ratio_tol,
        upper=cfg.canvas_ratio_bad,
        gamma=cfg.smoothstep_gamma,
        invert=True,
    )
    return score, {
        'source_ratio': round(want, 4),
        'slide_ratio': round(got, 4),
        'ratio_rel_err': round(err, 4),
        'looks_like_4_3_default': abs(got - 4.0 / 3.0) < 0.01,
    }


def metric_bounds_fidelity(geo: SlideGeometry, cfg: FidelityConfig) -> tuple[float | None, dict]:

    items = [e for e in geo.elements if e.area > 0 and not e.is_decorative]
    if not items:
        return None, {'reason': 'no_elements'}

    cw, ch = max(1.0, geo.view_width), max(1.0, geo.view_height)
    total_area = 0.0
    outside_area = 0.0
    offenders = 0
    for e in items:
        total_area += e.area
        vis_l = max(e.x, 0.0)
        vis_t = max(e.y, 0.0)
        vis_r = min(e.x + e.width, cw)
        vis_b = min(e.y + e.height, ch)
        visible = max(0.0, vis_r - vis_l) * max(0.0, vis_b - vis_t)
        out = max(0.0, e.area - visible)
        outside_area += out
        if e.area > 0 and (out / e.area) > cfg.bounds_slack:
            offenders += 1

    if total_area <= 0:
        return None, {'reason': 'zero_area'}

    ratio = outside_area / total_area
    score = smoothstep_reward(
        ratio,
        lower=cfg.bounds_good,
        upper=cfg.bounds_bad,
        gamma=cfg.smoothstep_gamma,
        invert=True,
    )
    return score, {
        'outside_area_ratio': round(ratio, 4),
        'offending_elements': offenders,
        'elements_used': len(items),
    }


def metric_element_iou(geo: SlideGeometry, ref: SlideGeometry | None, cfg: FidelityConfig) -> tuple[float | None, dict]:
    if ref is None:
        return None, {'reason': 'no_reference_geometry'}

    model_boxes = _norm_boxes(geo)
    ref_boxes = _norm_boxes(ref)
    if not ref_boxes:
        return None, {'reason': 'reference_has_no_elements'}
    if not model_boxes:
        return 0.0, {'reason': 'model_has_no_elements', 'ref_n': len(ref_boxes)}

    matched, unmatched_m, unmatched_r = _greedy_match(model_boxes, ref_boxes, floor=cfg.iou_match_floor)

    n_m, n_r = len(model_boxes), len(ref_boxes)

    align = (sum(v for _, _, v in matched) / len(matched)) if matched else 0.0

    coverage = len(matched) / n_r
    count_err = abs(n_m - n_r) / max(1, n_r)

    s_align = smoothstep_reward(align, lower=cfg.iou_bad, upper=cfg.iou_good, gamma=cfg.smoothstep_gamma)
    s_cover = smoothstep_reward(coverage, lower=cfg.coverage_bad, upper=cfg.coverage_good, gamma=cfg.smoothstep_gamma)
    s_count = smoothstep_reward(
        math.log10(1.0 + count_err),
        lower=cfg.count_log_tol,
        upper=cfg.count_log_bad,
        gamma=cfg.smoothstep_gamma,
        invert=True,
    )

    s_count = max(0.0, max(cfg.count_floor, s_count))
    score = (max(0.0, s_align) * max(0.0, s_cover) * s_count) ** (1.0 / 3.0)

    return score, {
        'align_iou': round(align, 4),
        'coverage': round(coverage, 4),
        'matched': len(matched),
        'missed_from_ref': len(unmatched_r),
        'hallucinated': len(unmatched_m),
        'model_n': n_m,
        'ref_n': n_r,
        'count_rel_err': round(count_err, 4),
        'sub_scores': {
            'align': round(s_align, 4),
            'coverage': round(s_cover, 4),
            'count': round(s_count, 4),
        },
    }


def compute_fidelity_metrics(
    geo: SlideGeometry,
    *,
    ref_geometry: SlideGeometry | None = None,
    source_size: tuple[float, float] | None = None,
    cfg: FidelityConfig | None = None,
) -> tuple[dict[str, float | None], dict[str, dict]]:
    cfg = cfg or FidelityConfig()
    scores: dict[str, float | None] = {}
    details: dict[str, dict] = {}

    for name, fn in (
        ('canvas_fidelity', lambda: metric_canvas_fidelity(geo, source_size, cfg)),
        ('bounds_fidelity', lambda: metric_bounds_fidelity(geo, cfg)),
        ('element_iou', lambda: metric_element_iou(geo, ref_geometry, cfg)),
    ):
        try:
            score, detail = fn()
        except Exception as exc:  # noqa: BLE001
            score, detail = None, {'error': f'{type(exc).__name__}: {exc}'}
        scores[name] = score
        details[name] = detail

    return scores, details


__all__ = [
    'FidelityConfig',
    'compute_fidelity_metrics',
    'metric_canvas_fidelity',
    'metric_bounds_fidelity',
    'metric_element_iou',
]
