from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SlideElement:
    x: float
    y: float
    width: float
    height: float
    kind: str = 'shape'
    text: str = ''
    text_len: int = 0
    font_size_px: float = 0.0
    content_width: float | None = None
    content_height: float | None = None
    fg_rgb: tuple[int, int, int] | None = None
    bg_rgb: tuple[int, int, int] | None = None
    bg_filled: bool = True
    group_id: int = 0
    is_decorative: bool = False
    shape_id: int = 0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    @property
    def is_text(self) -> bool:
        return self.kind == 'text' and self.text_len > 0


@dataclass
class SlideGeometry:
    elements: list[SlideElement] = field(default_factory=list)
    view_width: float = 1280.0
    view_height: float = 720.0
    image_bytes: bytes | None = None

    pdf_layout: Any | None = None

    slide_aspect: float | None = None

    judge_png: bytes | None = None

    @property
    def view_area(self) -> float:
        return max(1.0, self.view_width * self.view_height)


def smoothstep_reward(
    x: float,
    *,
    lower: float,
    upper: float,
    gamma: float = 1.0,
    invert: bool = False,
) -> float:
    if upper <= lower:
        raise ValueError(f'upper must be > lower, got {lower!r} >= {upper!r}')
    if gamma <= 0:
        raise ValueError(f'gamma must be positive, got {gamma!r}')

    if x <= lower:
        base = 0.0
    elif x >= upper:
        base = 1.0
    else:
        t = (x - lower) / (upper - lower)
        a = t**gamma
        b = (1.0 - t) ** gamma
        u = a / (a + b)
        base = 3.0 * u * u - 2.0 * u * u * u

    return 1.0 - base if invert else base


def _rect_intersection_area(a: SlideElement, b: SlideElement) -> float:
    left = max(a.x, b.x)
    right = min(a.x + a.width, b.x + b.width)
    top = max(a.y, b.y)
    bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:

    def _channel(c: int) -> float:
        s = max(0.0, min(1.0, c / 255.0))
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> float:

    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def metric_imbalance(geo: SlideGeometry, cfg: PPTRubricConfig) -> tuple[float | None, dict]:

    items = [e for e in geo.elements if e.area > 0]
    if not items:
        return None, {'reason': 'no_weighted_elements'}

    total_w = sum(e.area for e in items)
    if total_w <= 0:
        return None, {'reason': 'zero_total_area'}

    cx = sum(e.center[0] * e.area for e in items) / total_w
    cy = sum(e.center[1] * e.area for e in items) / total_w

    off_x = abs(cx - geo.view_width / 2.0) / geo.view_width
    off_y = abs(cy - geo.view_height / 2.0) / geo.view_height

    sx = smoothstep_reward(
        off_x, lower=cfg.imbalance_x_tol, upper=cfg.imbalance_x_bad, gamma=cfg.smoothstep_gamma, invert=True
    )
    sy = smoothstep_reward(
        off_y, lower=cfg.imbalance_y_tol, upper=cfg.imbalance_y_bad, gamma=cfg.smoothstep_gamma, invert=True
    )
    score = min(sx, sy)

    return score, {
        'centroid_norm': [round(cx / geo.view_width, 4), round(cy / geo.view_height, 4)],
        'offset_norm': [round(off_x, 4), round(off_y, 4)],
        'axis_scores': [round(sx, 4), round(sy, 4)],
        'nodes_used': len(items),
    }


def metric_text_overflow(geo: SlideGeometry, cfg: PPTRubricConfig) -> tuple[float | None, dict]:
    measurable = [e for e in geo.elements if e.is_text and e.content_height is not None and e.height > 0]
    if not measurable:
        return None, {'reason': 'renderer_reported_no_content_size'}

    worst_ratio = 0.0
    n_overflow = 0
    for e in measurable:
        v = (e.content_height or 0.0) / max(1.0, e.height)
        h = (e.content_width or 0.0) / max(1.0, e.width) if e.width > 0 else 0.0
        ratio = max(v, h)
        if ratio > 1.0 + cfg.overflow_slack:
            n_overflow += 1
        worst_ratio = max(worst_ratio, ratio)

    score = smoothstep_reward(
        worst_ratio,
        lower=1.0 + cfg.overflow_slack,
        upper=cfg.overflow_bad_ratio,
        gamma=cfg.smoothstep_gamma,
        invert=True,
    )
    return score, {
        'worst_ratio': round(worst_ratio, 4),
        'n_overflow': n_overflow,
        'n_checked': len(measurable),
    }


def metric_whitespace(geo: SlideGeometry, cfg: PPTRubricConfig) -> tuple[float | None, dict]:
    content_ratio: float | None = None
    source = 'geometry'

    if geo.image_bytes:
        try:
            content_ratio = _content_ratio_from_raster(geo.image_bytes, cfg)
            source = 'raster_variance'
        except Exception as exc:  # noqa: BLE001 - fall back, but say so loudly
            source = f'geometry_fallback({type(exc).__name__})'

    if content_ratio is None:
        covered = _approx_covered_area(geo, cfg)
        content_ratio = covered / geo.view_area

    score = smoothstep_reward(
        content_ratio,
        lower=cfg.whitespace_min_content,
        upper=cfg.whitespace_good_content,
        gamma=cfg.smoothstep_gamma,
    )
    return score, {'content_ratio': round(content_ratio, 4), 'source': source}


def _content_ratio_from_raster(image_bytes: bytes, cfg: PPTRubricConfig) -> float:

    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('failed to decode rendered slide bytes')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, cfg.ws_gaussian_ksize, 0)

    kh = cfg.ws_box_ksize_h | 1
    kv = cfg.ws_box_ksize_v | 1
    ksize = (kh, kv)

    local_mean = cv2.boxFilter(blurred, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    local_mean_sq = cv2.boxFilter(blurred**2, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    std_dev = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 0.0))

    norm = np.clip(std_dev, 0.0, cfg.ws_std_clip) / cfg.ws_std_clip
    return float(np.mean(norm > cfg.ws_binary_threshold))


def _approx_covered_area(geo: SlideGeometry, cfg: PPTRubricConfig) -> float:

    n = cfg.coverage_grid
    cw = geo.view_width / n
    ch = geo.view_height / n
    full_page = geo.view_area * cfg.collision_max_area_ratio
    occupied = 0
    for gy in range(n):
        for gx in range(n):
            cell_x = gx * cw
            cell_y = gy * ch
            for e in geo.elements:
                if e.area <= 0 or e.is_decorative:
                    continue
                if e.kind == 'image' and e.area >= full_page:
                    continue
                if e.x < cell_x + cw and e.x + e.width > cell_x and e.y < cell_y + ch and e.y + e.height > cell_y:
                    occupied += 1
                    break
    return (occupied / (n * n)) * geo.view_area


def _is_legitimate_overlay(top: SlideElement, bottom: SlideElement) -> bool:
    if not top.is_text:
        return False

    if bottom.kind != 'shape':
        return False
    if bottom.area < top.area * 1.5:
        return False
    inter = _rect_intersection_area(top, bottom)
    if top.area <= 0:
        return False
    return (inter / top.area) >= 0.85


def metric_collision(geo: SlideGeometry, cfg: PPTRubricConfig) -> tuple[float | None, dict]:

    items = [e for e in geo.elements if e.area > 0 and e.area < geo.view_area * cfg.collision_max_area_ratio]
    if len(items) < 2:
        return None, {'reason': 'fewer_than_two_collidable_elements'}

    n = len(items)
    total_pairs = n * (n - 1) // 2

    severity_sum = 0.0
    n_collisions = 0
    n_overlays = 0
    n_intra_group = 0
    worst = 0.0
    colliding: set[int] = set()

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            inter = _rect_intersection_area(a, b)
            if inter <= 0:
                continue

            if a.group_id and a.group_id == b.group_id:
                n_intra_group += 1
                continue
            if _is_legitimate_overlay(a, b) or _is_legitimate_overlay(b, a):
                n_overlays += 1
                continue
            min_area = min(a.area, b.area)
            if min_area <= 0:
                continue
            ratio = inter / min_area
            worst = max(worst, ratio)
            if ratio > cfg.collision_slack:
                n_collisions += 1
                severity_sum += ratio
                colliding.update((i, j))

    density = severity_sum / max(1, len(colliding))

    if density <= 0.0:
        score = 1.0
    else:
        lo = math.log10(max(cfg.collision_density_good, 1e-9))
        hi = math.log10(max(cfg.collision_density_bad, 1e-9))
        score = smoothstep_reward(
            math.log10(density),
            lower=lo,
            upper=hi,
            gamma=cfg.smoothstep_gamma,
            invert=True,
        )
    return score, {
        'collision_density': round(density, 5),
        'n_collisions': n_collisions,
        'n_pairs': total_pairs,
        'n_legit_overlays': n_overlays,
        'n_intra_group_skipped': n_intra_group,
        'worst_overlap_ratio': round(worst, 4),
        'n_checked': n,
    }


def metric_contrast(geo: SlideGeometry, cfg: PPTRubricConfig) -> tuple[float | None, dict]:

    checkable = [e for e in geo.elements if e.is_text and e.fg_rgb is not None and e.bg_rgb is not None]
    if not checkable:
        return None, {'reason': 'no_colour_info_on_text_elements'}

    worst_ratio = math.inf
    worst_detail: dict[str, Any] = {}
    for e in checkable:
        cr = contrast_ratio(e.fg_rgb, e.bg_rgb)  # type: ignore
        threshold = cfg.contrast_large_text_good if e.font_size_px >= cfg.large_text_px else cfg.contrast_good

        normalised = cr / threshold
        if normalised < worst_ratio:
            worst_ratio = normalised
            worst_detail = {
                'contrast_ratio': round(cr, 2),
                'threshold': threshold,
                'font_size_px': e.font_size_px,
            }

    score = smoothstep_reward(
        worst_ratio,
        lower=cfg.contrast_bad / cfg.contrast_good,
        upper=1.0,
        gamma=cfg.smoothstep_gamma,
    )
    return score, {'worst_normalised': round(worst_ratio, 4), **worst_detail}


def metric_text_density(geo: SlideGeometry, cfg: PPTRubricConfig) -> tuple[float | None, dict]:

    layout = geo.pdf_layout
    if layout is None or getattr(layout, 'spans', None) is None:
        return None, {'reason': 'needs_pdf_measurement'}

    page_area = max(1.0, float(layout.page_width) * float(layout.page_height))
    spans = [s for s in layout.spans if s.text.strip()]
    if not spans:
        return None, {'reason': 'no_text_spans'}

    ink = sum(s.width * s.height for s in spans) / page_area
    chars = sum(len(s.text.strip()) for s in spans)

    buckets: dict[int, int] = {}
    for s in spans:
        key = int(round(s.y0 / cfg.density_line_bucket_pt))
        buckets[key] = buckets.get(key, 0) + len(s.text.strip())
    line_count = len(buckets)
    chars_per_line = chars / max(1, line_count)

    s_ink = smoothstep_reward(
        ink,
        lower=cfg.density_ink_good,
        upper=cfg.density_ink_bad,
        gamma=cfg.smoothstep_gamma,
        invert=True,
    )
    s_lines = smoothstep_reward(
        float(line_count),
        lower=cfg.density_lines_good,
        upper=cfg.density_lines_bad,
        gamma=cfg.smoothstep_gamma,
        invert=True,
    )
    s_cpl = smoothstep_reward(
        chars_per_line,
        lower=cfg.density_cpl_good,
        upper=cfg.density_cpl_bad,
        gamma=cfg.smoothstep_gamma,
        invert=True,
    )

    score = min(s_ink, s_lines, s_cpl)
    return score, {
        'ink_ratio': round(ink, 4),
        'line_count': line_count,
        'chars_per_line': round(chars_per_line, 1),
        'chars': chars,
        'sub_scores': {
            'ink': round(s_ink, 3),
            'lines': round(s_lines, 3),
            'chars_per_line': round(s_cpl, 3),
        },
    }


@dataclass
class PPTRubricConfig:
    w_text_density: float = 1.000
    w_imbalance: float = 1.000
    w_text_overflow: float = 1.000

    w_whitespace: float = 0.0
    w_collision: float = 1.000
    w_contrast: float = 1.000

    w_objective: float = 0.15
    w_fidelity: float = 0.4
    w_subjective: float = 0.5

    w_text: float = 0.0

    objective_mode: str = 'gold_relative'

    canvas_gate_floor: float = 0.35

    baked_soft_start: float = 0.3
    baked_floor: float = 0.3

    smoothstep_gamma: float = 1.0

    density_ink_good: float = 0.12
    density_ink_bad: float = 0.28
    density_lines_good: float = 12.0
    density_lines_bad: float = 30.0
    density_cpl_good: float = 26.0
    density_cpl_bad: float = 55.0
    density_line_bucket_pt: float = 4.0

    imbalance_x_tol: float = 0.0075
    imbalance_x_bad: float = 0.15
    imbalance_y_tol: float = 0.145
    imbalance_y_bad: float = 0.35

    overflow_slack: float = 0.02
    overflow_bad_ratio: float = 1.5

    whitespace_min_content: float = 0.08
    whitespace_good_content: float = 0.35
    ws_box_ksize_h: int = 201
    ws_box_ksize_v: int = 151
    ws_std_clip: float = 50.0
    ws_gaussian_ksize: tuple[int, int] = (21, 21)
    ws_binary_threshold: float = 0.05
    coverage_grid: int = 32

    collision_slack: float = 0.02
    collision_max_area_ratio: float = 0.9

    collision_density_good: float = 0.01
    collision_density_bad: float = 0.25
    collision_bad_ratio: float = 0.35

    contrast_good: float = 4.5
    contrast_large_text_good: float = 3.0
    contrast_bad: float = 1.5
    large_text_px: float = 24.0

    def objective_weights(self) -> dict[str, float]:
        return {
            'text_density': self.w_text_density,
            'imbalance': self.w_imbalance,
            'text_overflow': self.w_text_overflow,
            'whitespace': self.w_whitespace,
            'collision': self.w_collision,
            'contrast': self.w_contrast,
        }

    @classmethod
    def from_mapping(cls, overrides: dict[str, Any] | None) -> PPTRubricConfig:

        if not overrides:
            return cls()
        valid = set(cls.__dataclass_fields__)
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f'[ppt_rubric] unknown config keys: {sorted(unknown)}. Valid keys: {sorted(valid)}')
        for name, value in overrides.items():
            if name.startswith('w_'):
                try:
                    weight = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'[ppt_rubric] weight {name} must be numeric, got {value!r}') from exc
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(f'[ppt_rubric] weight {name} must be finite and nonnegative, got {value!r}')
            elif name == 'objective_mode':
                if value not in ('absolute', 'gold_relative'):
                    raise ValueError(
                        f'[ppt_rubric] objective_mode must be "absolute" or "gold_relative", got {value!r}'
                    )
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'[ppt_rubric] {name} must be finite, got {value!r}')
        return cls(**overrides)


OBJECTIVE_METRICS: tuple[tuple[str, Any], ...] = (
    ('text_density', metric_text_density),
    ('imbalance', metric_imbalance),
    ('text_overflow', metric_text_overflow),
    ('whitespace', metric_whitespace),
    ('collision', metric_collision),
    ('contrast', metric_contrast),
)


RELATIVE_RAW: dict[str, tuple[str, bool]] = {
    'text_density': ('ink_ratio', True),
    'imbalance': ('offset_max', True),
    'text_overflow': ('worst_ratio', True),
    'whitespace': ('content_ratio', False),
    'collision': ('collision_density', True),
    'contrast': ('worst_normalised', False),
}

RELATIVE_TOLERANCE = 0.05
RELATIVE_BAD = 0.4
RELATIVE_EPSILON = 1e-3


def _relative_raw(detail: dict, key: str) -> float | None:
    if key == 'offset_max':
        offsets = detail.get('offset_norm')
        if not isinstance(offsets, (list, tuple)) or not offsets:
            return None
        values = [float(v) for v in offsets if isinstance(v, (int, float))]
        return max(values) if values else None
    value = detail.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def relative_objective_scores(
    gold_details: dict[str, dict], cand_details: dict[str, dict], cfg: PPTRubricConfig | None = None
) -> tuple[dict[str, float | None], dict[str, Any]]:
    cfg = cfg or PPTRubricConfig()
    scores: dict[str, float | None] = {}
    pairs: dict[str, Any] = {}
    for name, (key, higher_is_worse) in RELATIVE_RAW.items():
        gold_raw = _relative_raw(gold_details.get(name) or {}, key)
        cand_raw = _relative_raw(cand_details.get(name) or {}, key)
        if gold_raw is None or cand_raw is None:
            scores[name] = None
            continue
        signed = math.log10((cand_raw + RELATIVE_EPSILON) / (gold_raw + RELATIVE_EPSILON))
        penalty = max(0.0, signed if higher_is_worse else -signed)
        scores[name] = smoothstep_reward(
            penalty, lower=RELATIVE_TOLERANCE, upper=RELATIVE_BAD, gamma=cfg.smoothstep_gamma, invert=True
        )
        pairs[name] = {'gold': round(gold_raw, 5), 'cand': round(cand_raw, 5), 'log_ratio': round(signed, 4)}
    return scores, {'raw_pairs': pairs, 'n_used': sum(1 for v in scores.values() if v is not None)}


def compute_objective_metrics(
    geo: SlideGeometry, cfg: PPTRubricConfig | None = None
) -> tuple[dict[str, float | None], dict[str, dict]]:
    cfg = cfg or PPTRubricConfig()
    scores: dict[str, float | None] = {}
    details: dict[str, dict] = {}
    for name, fn in OBJECTIVE_METRICS:
        try:
            score, detail = fn(geo, cfg)
        except Exception as exc:  # noqa: BLE001
            score, detail = None, {'error': f'{type(exc).__name__}: {exc}'}
        scores[name] = score
        details[name] = detail
    return scores, details


def weighted_mean(scores: dict[str, float | None], weights: dict[str, float]) -> tuple[float | None, dict[str, Any]]:
    used: dict[str, float] = {}
    total_w = 0.0
    acc = 0.0
    for name, w in weights.items():
        s = scores.get(name)
        if s is None:
            continue
        s = float(s)

        if not math.isfinite(s):
            continue
        s = max(0.0, min(1.0, s))
        used[name] = w
        total_w += w
        acc += s * w

    if total_w <= 0:
        return None, {'n_used': 0, 'skipped': sorted(weights)}

    return acc / total_w, {
        'n_used': len(used),
        'used_weight_sum': round(total_w, 4),
        'skipped': sorted(set(weights) - set(used)),
    }


def aggregate_objective(
    scores: dict[str, float | None], cfg: PPTRubricConfig | None = None
) -> tuple[float | None, dict[str, Any]]:
    cfg = cfg or PPTRubricConfig()
    return weighted_mean(scores, cfg.objective_weights())


__all__ = [
    'SlideElement',
    'SlideGeometry',
    'PPTRubricConfig',
    'OBJECTIVE_METRICS',
    'compute_objective_metrics',
    'aggregate_objective',
    'weighted_mean',
    'relative_objective_scores',
    'RELATIVE_RAW',
    'smoothstep_reward',
    'contrast_ratio',
    'metric_text_density',
    'metric_imbalance',
    'metric_text_overflow',
    'metric_whitespace',
    'metric_collision',
    'metric_contrast',
]
