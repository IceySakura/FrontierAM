"""Tests that vendored runtime sys.path edits stay contained.

The review found that the import-time cleanup in font_tools is undone at run
time: parallel._work() re-inserts fontid/ on every serial identify_font call
and fontsize_api._load_faces() re-inserts bx/ on every estimate_font_size call.
The adapter now wraps both entry points so sys.path is exactly restored around
each call. These tests pin that without needing real font files.
"""

from __future__ import annotations

import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nexus.extensions.tasks.ppt_rubric.ppt_tools import font_tools as ft  # noqa: E402


def test_vendored_call_sites_restore_sys_path():
    ft._reset_for_test()
    mods = ft._import_vendored()
    par, fz = mods['PAR'], mods['FZ']

    # The fork pool stays defused (first-round fix, still expected).
    assert par._POOL is False

    saved = list(sys.path)

    # parallel._work is the serial-path entry identify() lands on with the
    # pool defused; empty items means no rendering, but the sys.path insert
    # at its top still runs -- which is exactly what must be undone.
    assert par._work((None, None, [])) == []
    assert sys.path == saved, 'fontid/ leaked onto sys.path via _work()'

    # Repeated calls must not accumulate entries either.
    par._work((None, None, []))
    assert sys.path == saved

    # fontsize_api._ink_box inserts bx/ each call (its `from text_blocks
    # import _glyph_mask` needs it); wrapped the same way. A blank array keeps
    # it on its real code path (mask empty -> None) without font files.
    import numpy as np  # noqa: PLC0415 - exercised inside the call under test

    assert fz._ink_box(np.zeros((4, 4, 3), dtype=np.uint8), [0, 0, 2, 2]) is None
    assert sys.path == saved, 'bx/ leaked onto sys.path via _ink_box()'

    # And nothing vendored is left on the path after import-time cleanup.
    for p in (ft._HERE, str(Path(ft._HERE) / 'fontid'), str(Path(ft._HERE) / 'sizeid'), str(Path(ft._HERE) / 'bx')):
        assert p not in sys.path
