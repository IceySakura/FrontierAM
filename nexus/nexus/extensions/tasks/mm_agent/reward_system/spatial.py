"""Spatial reward functions for MM-Agent (point / bbox / seg).

These rewards score model predictions against ground-truth annotations
produced by ``examples/mm_agent/data_scripts/convert_messages_to_rldata.py``.

Ground-truth formats (stored as plain strings in ``reward_model.ground_truth``):

* bbox : ``"... [x1, y1, x2, y2] ..."``  (exactly 4 integers)
* point: ``"... [x, y] ..."``            (exactly 2 integers)
* seg  : ``"... [[[x1, y1], [x2, y2], ...]] ..."``  (nested polygon(s),
         supports multi-ring ``[[[...],[...]], [[...]]]``)

Coordinates are assumed to live in the Qwen-VL normalized 0~999 canvas
unless ``image_size=(W, H)`` is explicitly provided via ``kwargs``.

All reward functions share the same ``async`` signature used by
``vl_reward_system`` in :mod:`nexus.extensions.tasks.mm_agent.reward_system.reward`.

Return value: ``(score: float, details: dict)``.  If the prediction fails
the *format reward* (cannot be parsed, wrong cardinality, etc.) the score
is ``0.0`` and ``details['format_ok'] = False``.
"""

from __future__ import annotations

import ast
import logging
import re

from typing import Any

import numpy as np

from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)

# Default reference canvas (Qwen-VL style 0~999). Overridable via kwargs.
_DEFAULT_IMAGE_SIZE: tuple[int, int] = (1000, 1000)

# Mask IoU threshold τ — rewards below τ are zeroed out (paper-style).
_DEFAULT_IOU_THRESHOLD: float = 0.1

# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

# Matches innermost bracket groups containing only numbers/commas/whitespace,
# e.g. "[12, 34]" or "[1, 2, 3, 4]".  Used for bbox/point.
_FLAT_LIST_RE = re.compile(r'\[\s*-?\d+(?:\s*,\s*-?\d+)+\s*\]')

# Matches a triple-nested polygon block, e.g. "[[[x1,y1],[x2,y2], ...]]".
# Non-greedy and DOTALL so it spans newlines.
_NESTED_POLY_RE = re.compile(r'\[\s*\[\s*\[.*?\]\s*\]\s*\]', re.S)


def _parse_flat_list(text: str, expected_len: int) -> list[int] | None:
    """Return the *last* flat numeric list in ``text`` with ``expected_len`` items.

    Taking the **last** match is deliberate: model responses often restate the
    question first ("Locate the bbox of [object] ...") and the true answer is
    produced afterwards.  Returns ``None`` on any failure (format reward = 0).
    """
    if not isinstance(text, str) or not text:
        return None
    matches = _FLAT_LIST_RE.findall(text)
    if not matches:
        return None
    for candidate in reversed(matches):
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if (
            isinstance(parsed, list)
            and len(parsed) == expected_len
            and all(isinstance(v, (int, float)) for v in parsed)
        ):
            return [int(round(v)) for v in parsed]
    return None


def _parse_polygons(text: str) -> list[list[tuple[int, int]]] | None:
    """Parse ``[[[x,y], ...]]`` (or multi-ring ``[[[...], [...]]]``) from ``text``.

    Returns a list of rings, where each ring is a list of ``(x, y)`` integer
    tuples.  Returns ``None`` if no valid polygon with ≥ 3 vertices is found.
    """
    if not isinstance(text, str) or not text:
        return None
    matches = _NESTED_POLY_RE.findall(text)
    if not matches:
        return None

    for candidate in reversed(matches):
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        rings = _coerce_to_rings(parsed)
        if rings:
            return rings
    return None


def _coerce_to_rings(obj: Any) -> list[list[tuple[int, int]]] | None:
    """Coerce a parsed object into a list of rings of ``(x, y)`` tuples.

    Accepts both ``[[[x,y], ...]]`` (single polygon, one ring) and
    ``[[[x,y], ...], [[x,y], ...]]`` (multi-ring).
    """
    if not isinstance(obj, list) or not obj:
        return None

    # Single ring wrapped once: [[[x,y], ...]]  -> obj[0] is a ring.
    # Multi-ring:               [[[...], [...]]] -> obj is a list of rings.
    # Detect by peeking at obj[0][0].
    first = obj[0]
    if not isinstance(first, list) or not first:
        return None

    if isinstance(first[0], (int, float)):
        # obj is a single ring: [[x,y], [x,y], ...]
        rings_raw = [obj]
    elif isinstance(first[0], list):
        # obj is a list of rings
        rings_raw = obj
    else:
        return None

    rings: list[list[tuple[int, int]]] = []
    for ring in rings_raw:
        if not isinstance(ring, list) or len(ring) < 3:
            continue
        pts: list[tuple[int, int]] = []
        ok = True
        for pt in ring:
            if not isinstance(pt, list) or len(pt) != 2 or not all(isinstance(v, (int, float)) for v in pt):
                ok = False
                break
            pts.append((int(round(pt[0])), int(round(pt[1]))))
        if ok and len(pts) >= 3:
            rings.append(pts)
    return rings or None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _bbox_iou(a: list[int], b: list[int]) -> float:
    """Axis-aligned bbox IoU for ``[x1, y1, x2, y2]`` pairs."""
    ax1, ax2 = min(a[0], a[2]), max(a[0], a[2])
    ay1, ay2 = min(a[1], a[3]), max(a[1], a[3])
    bx1, bx2 = min(b[0], b[2]), max(b[0], b[2])
    by1, by2 = min(b[1], b[3]), max(b[1], b[3])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _rasterize(rings: list[list[tuple[int, int]]], size: tuple[int, int]) -> np.ndarray:
    """Rasterize polygon rings (+ holes approximated as XOR) into a binary mask.

    ``size`` is ``(W, H)``; output shape is ``(H, W)`` bool.
    The first ring is treated as the outer contour and subsequent rings are
    XOR-ed in — this matches how COCO-style polygons are typically drawn.
    """
    w, h = size
    img = Image.new('L', (w, h), 0)
    if not rings:
        return np.zeros((h, w), dtype=bool)
    draw = ImageDraw.Draw(img)
    # outer
    draw.polygon(rings[0], fill=1)
    # holes / extra rings: XOR toggle
    for ring in rings[1:]:
        hole = Image.new('L', (w, h), 0)
        ImageDraw.Draw(hole).polygon(ring, fill=1)
        img = Image.fromarray(
            np.asarray(img, dtype=np.uint8) ^ np.asarray(hole, dtype=np.uint8),
            mode='L',
        )
    return np.asarray(img, dtype=bool)


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------------------
# Reward functions (async — to match vl_reward_system contract)
# ---------------------------------------------------------------------------


async def point_reward(
    question: str,
    response: str,
    target_answer: str,
    **kwargs: Any,
) -> tuple[float, dict]:
    """Point reward: negative normalized MSE between predicted / GT points.

    Score = max(0, 1 - ||pred - gt||^2 / diag^2)  ∈ [0, 1].

    Format reward: if no ``[x, y]`` with 2 ints can be parsed from ``response``,
    returns ``(0.0, {format_ok: False})``.
    """
    pred = _parse_flat_list(response, expected_len=2)
    gt = _parse_flat_list(target_answer, expected_len=2)
    details: dict[str, Any] = {
        'reward_func': 'point',
        'prediction': pred,
        'reference': gt,
        'format_ok': pred is not None,
    }
    if pred is None or gt is None:
        return 0.0, details

    w, h = 1000, 1000
    diag_sq = float(w * w + h * h)
    dx, dy = pred[0] - gt[0], pred[1] - gt[1]
    mse = (dx * dx + dy * dy) / diag_sq  # normalized squared distance
    score = max(0.0, 1.0 - mse)
    details.update({'normalized_mse': mse, 'image_size': (w, h), 'score': score})
    return score, details


async def bbox_reward(
    question: str,
    response: str,
    target_answer: str,
    **kwargs: Any,
) -> tuple[float, dict]:
    """Bounding-box reward: IoU between predicted and GT bboxes.

    Score = IoU if IoU ≥ τ else 0.0.  τ defaults to ``_DEFAULT_IOU_THRESHOLD``
    and can be overridden via ``kwargs['iou_threshold']``.
    """
    pred = _parse_flat_list(response, expected_len=4)
    gt = _parse_flat_list(target_answer, expected_len=4)
    details: dict[str, Any] = {
        'reward_func': 'bbox',
        'prediction': pred,
        'reference': gt,
        'format_ok': pred is not None,
    }
    if pred is None or gt is None:
        return 0.0, details

    iou = _bbox_iou(pred, gt)
    tau = float(kwargs.get('iou_threshold', _DEFAULT_IOU_THRESHOLD))
    score = iou if iou >= tau else 0.0
    details.update({'iou': iou, 'threshold': tau, 'score': score})
    return score, details


async def seg_reward(
    question: str,
    response: str,
    target_answer: str,
    **kwargs: Any,
) -> tuple[float, dict]:
    """Segmentation reward: mask IoU + thresholding + normalized centroid MSE.

    Pipeline:
      1. Parse triple-nested polygon lists from both ``response`` and
         ``target_answer``.  Any failure ⇒ format reward zero.
      2. Rasterize each into a binary mask using ``image_size``.
      3. Compute mask IoU; zero it if below τ.
      4. Also record the negative normalized centroid MSE for diagnostics
         (not added into ``score`` by default).
    """
    pred_rings = _parse_polygons(response)
    gt_rings = _parse_polygons(target_answer)
    details: dict[str, Any] = {
        'reward_func': 'seg',
        'format_ok': pred_rings is not None,
        'pred_rings': len(pred_rings) if pred_rings else 0,
        'gt_rings': len(gt_rings) if gt_rings else 0,
    }
    if not pred_rings or not gt_rings:
        return 0.0, details

    w, h = 1000, 1000
    try:
        pred_mask = _rasterize(pred_rings, (w, h))
        gt_mask = _rasterize(gt_rings, (w, h))
    except Exception as exc:  # extremely defensive — bad coords, etc.
        logger.warning(f'seg_reward rasterize failed: {exc}')
        details.update({'format_ok': False, 'error': str(exc)})
        return 0.0, details

    iou = _mask_iou(pred_mask, gt_mask)
    tau = float(kwargs.get('iou_threshold', _DEFAULT_IOU_THRESHOLD))
    score = iou if iou >= tau else 0.0

    # diagnostic: normalized centroid distance
    centroid_mse: float | None = None
    p_c = _mask_centroid(pred_mask)
    g_c = _mask_centroid(gt_mask)
    if p_c is not None and g_c is not None:
        diag_sq = float(w * w + h * h)
        dx, dy = p_c[0] - g_c[0], p_c[1] - g_c[1]
        centroid_mse = (dx * dx + dy * dy) / diag_sq

    details.update({
        'iou': iou,
        'threshold': tau,
        'image_size': (w, h),
        'centroid_mse': centroid_mse,
        'score': score,
    })
    return score, details


__all__ = ['point_reward', 'bbox_reward', 'seg_reward']
