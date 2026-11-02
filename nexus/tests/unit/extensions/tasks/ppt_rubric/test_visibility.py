
from __future__ import annotations

import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from nexus.extensions.tasks.ppt_rubric import ppt_structure as ps  # noqa: E402


def _line(**kw):
    base = {'text': 'x', 'x0': 0.10, 'y0': 0.10, 'x1': 0.20, 'y1': 0.16, 'size': 20.0}
    base.update(kw)
    return ps.TextLine(**base)


def _renders(h=500, w=400):
    full = np.zeros((h, w, 3), dtype=np.uint8)
    return ps.PageRenders(full=full, notext=full.copy())


def _ink_renders(h=500, w=400):
    full = np.zeros((h, w, 3), dtype=np.uint8)
    notext = full.copy()
    y0, y1 = int(0.10 * h), int(0.16 * h)
    x0, x1 = int(0.10 * w), int(0.20 * w)
    full[y0:y1, x0:x1] = 128
    return ps.PageRenders(full=full, notext=notext)


def test_invisible_text_line_visible_without_renders():
    lines = [_line(invisible_text=True)]
    diag: dict = {}
    ps._mark_visibility(lines, None, diag)
    assert lines[0].visible
    assert lines[0].hidden_reason == ''


def test_invisible_text_off_canvas_stays_hidden():
    lines = [_line(x0=1.05, x1=1.15, invisible_text=True)]
    ps._mark_visibility(lines, _renders(), {})
    assert not lines[0].visible
    assert lines[0].hidden_reason == 'off_canvas'


def test_invisible_text_line_ignores_causal_ink():
    lines = [_line(invisible_text=True)]
    ps._mark_visibility(lines, _renders(), {})
    assert lines[0].visible
    assert lines[0].hidden_reason == ''


def test_normal_line_without_ink_hidden():
    lines = [_line()]
    ps._mark_visibility(lines, _renders(), {})
    assert not lines[0].visible
    assert lines[0].hidden_reason == 'no_ink'


def test_normal_line_with_ink_visible():
    lines = [_line()]
    ps._mark_visibility(lines, _ink_renders(), {})
    assert lines[0].visible


def test_diag_counts_hidden_lines():
    lines = [_line(), _line(), _line(invisible_text=True)]
    diag: dict = {}
    ps._mark_visibility(lines, _renders(), diag)
    assert diag['n_hidden_lines'] == 2
