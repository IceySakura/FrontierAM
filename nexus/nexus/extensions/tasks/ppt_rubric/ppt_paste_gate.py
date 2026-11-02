from __future__ import annotations

import ast
import re

from typing import Any


MODEL_MIN_CHARS = 8
GOLD_MIN_CHARS = 24


MODEL_CHARS_RATIO = 0.10


PICTURE_FLOOR_RATIO = 0.25


_FULL_PAGE_PICTURE_SHARE = 0.70

_ASSET_SUFFIXES = (
    '.png',
    '.jpg',
    '.jpeg',
    '.pptx',
    '.pdf',
    '.emf',
    '.svg',
    '.gif',
    '.bmp',
    '.webp',
    '.tiff',
    '.js',
    '.json',
    '.txt',
)


_IDENT_LOWER = re.compile(r'[a-z][a-z0-9_]*\Z')
_IDENT_UPPER = re.compile(r'[A-Z][A-Z0-9_]*\Z')
_API_TOKEN_MAX = 14


def _looks_like_api_token(s: str) -> bool:

    if ' ' in s or not s.isascii() or len(s) > _API_TOKEN_MAX:
        return False
    if _IDENT_LOWER.match(s) or _IDENT_UPPER.match(s):
        return True
    if len(s) <= 2:
        return True
    return s.replace('.', '').isdigit()


def _literal_chars(s: str) -> int:

    s = s.strip()
    if not s:
        return 0
    if s.lower().endswith(_ASSET_SUFFIXES):
        return 0
    if '/' in s or '\\' in s:
        return 0

    if s.startswith('#') and len(s) <= 9:
        return 0
    if len(s) in (3, 6, 8) and all(c in '0123456789abcdefABCDEF' for c in s):
        return 0

    if re.fullmatch(r'\d+(\.\d+)?%', s):
        return 0
    if _looks_like_api_token(s):
        return 0
    return len(s)


def _js_literal_chars(code: str) -> int:

    total = 0
    i, n = 0, len(code)
    quote = ''
    buf: list[str] = []
    while i < n:
        ch = code[i]
        if quote:
            if ch == '\\' and i + 1 < n:
                buf.append(code[i + 1])
                i += 2
                continue
            if ch == quote:
                total += _literal_chars(''.join(buf))
                buf, quote = [], ''
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < n and code[i + 1] == '/':
            nl = code.find('\n', i)
            if nl < 0:
                break
            i = nl
            continue
        if ch == '/' and i + 1 < n and code[i + 1] == '*':
            end = code.find('*/', i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch in '"\'`':
            quote = ch
            i += 1
            continue
        i += 1
    return total


def count_code_chars(code: str) -> tuple[int, str]:
    if not code or not code.strip():
        return -1, 'empty'
    try:
        tree = ast.parse(code)
    except SyntaxError:
        if 'pptxgen' in code or 'require(' in code or 'const ' in code:
            return _js_literal_chars(code), 'js'
        return -1, 'unparseable'

    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            total += _literal_chars(node.value)
    return total, 'ast'


def count_pdf_chars(geometry: Any) -> tuple[int, str]:
    layout = getattr(geometry, 'pdf_layout', None)
    if layout is None:
        return -1, 'pdf_unavailable'
    spans = getattr(layout, 'spans', None)
    if not spans:
        return 0, 'pdf_empty_text_layer'
    return sum(len((getattr(sp, 'text', '') or '').strip()) for sp in spans), 'pdf'


def _has_full_page_picture(geometry: Any) -> bool:

    elements = getattr(geometry, 'elements', None) or []
    view_width = float(getattr(geometry, 'view_width', 0) or 0)
    view_height = float(getattr(geometry, 'view_height', 0) or 0)
    if view_width <= 0 or view_height <= 0:
        return False
    canvas = view_width * view_height
    for element in elements:
        if getattr(element, 'kind', '') != 'image':
            continue
        if (element.width * element.height) >= canvas * _FULL_PAGE_PICTURE_SHARE:
            return True
    return False


def decide_from_counts(
    gold_chars: int,
    code_chars: float,
    pdf_chars: float,
    *,
    model_min: int = MODEL_MIN_CHARS,
    gold_min: int = GOLD_MIN_CHARS,
    full_page_picture: bool = False,
) -> tuple[bool, dict[str, Any]]:
    detail: dict[str, Any] = {'gold_text_chars': float(gold_chars)}

    if gold_chars < 0:
        detail['gate'] = 'no_gold_text_chars'
        return False, detail

    if gold_chars < gold_min:
        detail['gate'] = 'gold_below_min'
        return False, detail

    floor = max(model_min, int(gold_chars * MODEL_CHARS_RATIO))
    if full_page_picture:
        floor = max(floor, int(gold_chars * PICTURE_FLOOR_RATIO))
    detail['model_floor'] = float(floor)
    detail['full_page_picture'] = 1.0 if full_page_picture else 0.0

    detail['code_chars'] = float(code_chars)
    detail['pdf_chars'] = float(pdf_chars)

    available = [c for c in (code_chars, pdf_chars) if c >= 0]
    if not available:
        detail['gate'] = 'undecidable'
        return False, detail

    if pdf_chars >= 0:
        best, source = pdf_chars, 'pdf'
    else:
        best, source = code_chars, 'code'
    detail['text_chars_used'] = float(best)
    detail['text_source'] = source
    if best >= floor:
        detail['gate'] = 'ok'
        return False, detail

    detail['gate'] = 'stripped'
    detail['reason'] = (
        f'gold slide carries {gold_chars} characters but the answer puts '
        f'{best} on the page (via {source}, gate {floor}); that hands back '
        f'the screenshot rather than reproducing the slide'
    )
    return True, detail


def check_text_stripped(
    code: str | None,
    geometry: Any,
    gold: dict,
    *,
    model_min: int = MODEL_MIN_CHARS,
    gold_min: int = GOLD_MIN_CHARS,
) -> tuple[bool, dict[str, Any]]:
    detail: dict[str, Any] = {}

    gold_chars = gold.get('text_chars')
    if not isinstance(gold_chars, (int, float)) or isinstance(gold_chars, bool):
        detail['gate'] = 'no_gold_text_chars'
        return False, detail
    gold_chars = int(gold_chars)

    code_chars, _ = count_code_chars(code or '')
    pdf_chars, _ = count_pdf_chars(geometry)
    return decide_from_counts(
        gold_chars,
        code_chars,
        pdf_chars,
        model_min=model_min,
        gold_min=gold_min,
        full_page_picture=_has_full_page_picture(geometry),
    )


__all__ = [
    'MODEL_MIN_CHARS',
    'GOLD_MIN_CHARS',
    'MODEL_CHARS_RATIO',
    'PICTURE_FLOOR_RATIO',
    'check_text_stripped',
    'count_code_chars',
    'count_pdf_chars',
    'decide_from_counts',
]
