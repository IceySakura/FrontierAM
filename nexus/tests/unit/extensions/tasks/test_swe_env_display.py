"""Tests for DISPLAY sanitisation on the SWE task's subprocess paths.

gui_stack.sh exports DISPLAY at its top level and start.sh sources it
unconditionally, but Xvfb only starts under NEXUS_GUI_AUTO_START=1. The resulting
dangling DISPLAY breaks browser-driven eval scripts in ways that never name
DISPLAY: Electron aborts at startup, while chromium's GPU process crash-loops and
browser.close() hangs ~30s, so Playwright suites go flaky rather than failing.

These drive a real temporary socket_dir rather than a patched os.path, so they
exercise the same filesystem check production takes.
"""

from __future__ import annotations

import pytest

from nexus.extensions.tasks.swe.env import local_display_number, strip_dangling_display


LOCAL_DISPLAY_NUM_99 = '99'


@pytest.fixture
def x11_dir(tmp_path):
    """Stand-in for /tmp/.X11-unix. Empty by default; call it to add live sockets."""

    def live(*numbers: str) -> str:
        for n in numbers:
            (tmp_path / f'X{n}').touch()
        return str(tmp_path)

    live.path = str(tmp_path)  # type: ignore[attr-defined]
    return live


def test_dangling_display_is_dropped(x11_dir):
    """DISPLAY=:99 with no X server on :99 is removed; the rest of the env survives."""
    env = strip_dangling_display({'DISPLAY': ':99', 'HOME': '/root'}, socket_dir=x11_dir.path)

    assert 'DISPLAY' not in env
    assert env['HOME'] == '/root'


def test_live_display_is_preserved(x11_dir):
    """A display backed by a real X server is left alone.

    This is the NEXUS_GUI_AUTO_START=1 / gui_start path: Xvfb is running, so a
    script that wants the desktop must still find DISPLAY.
    """
    socket_dir = x11_dir('99')

    assert strip_dangling_display({'DISPLAY': ':99'}, socket_dir=socket_dir)['DISPLAY'] == ':99'


def test_liveness_is_judged_per_display_not_hardcoded_99(x11_dir):
    """:7 is dropped even while :99 is live -- the socket decides, not the number."""
    socket_dir = x11_dir('99')

    assert 'DISPLAY' not in strip_dangling_display({'DISPLAY': ':7'}, socket_dir=socket_dir)


@pytest.mark.parametrize('value', ['somehost:0', '192.168.1.5:0'])
def test_remote_display_is_never_judged(x11_dir, value):
    """A remote display is preserved: its reachability is not ours to decide."""
    assert strip_dangling_display({'DISPLAY': value}, socket_dir=x11_dir.path)['DISPLAY'] == value


def test_absent_display_is_not_synthesised(x11_dir):
    """No DISPLAY in the env -> none invented."""
    assert 'DISPLAY' not in strip_dangling_display({'HOME': '/root'}, socket_dir=x11_dir.path)


@pytest.mark.parametrize('value', [':99.0', 'unix:99'])
def test_screen_suffix_and_unix_prefix_follow_the_socket(tmp_path, value):
    """':99.0' and 'unix:99' both name local display 99, so both track its liveness."""
    dead, alive = tmp_path / 'dead', tmp_path / 'alive'
    dead.mkdir()
    alive.mkdir()
    (alive / 'X99').touch()

    assert strip_dangling_display({'DISPLAY': value}, socket_dir=str(alive))['DISPLAY'] == value
    assert 'DISPLAY' not in strip_dangling_display({'DISPLAY': value}, socket_dir=str(dead))


@pytest.mark.parametrize('value', ['', 'not-a-display', ':abc'])
def test_malformed_display_is_left_untouched(x11_dir, value):
    """An unparseable DISPLAY is not second-guessed."""
    assert strip_dangling_display({'DISPLAY': value}, socket_dir=x11_dir.path).get('DISPLAY') == value


def test_caller_env_is_not_mutated(x11_dir):
    """The input dict is left alone; only the returned copy loses DISPLAY."""
    original = {'DISPLAY': ':99'}

    returned = strip_dangling_display(original, socket_dir=x11_dir.path)

    assert original == {'DISPLAY': ':99'}
    assert 'DISPLAY' not in returned


def test_local_display_number_parsing():
    """Only local forms yield a number; remote and malformed yield None."""
    assert local_display_number(':99') == LOCAL_DISPLAY_NUM_99
    assert local_display_number(':0.0') == '0'
    assert local_display_number('unix:99') == LOCAL_DISPLAY_NUM_99
    assert local_display_number(' :99 ') == LOCAL_DISPLAY_NUM_99
    assert local_display_number('host:0') is None
    assert local_display_number('192.168.1.5:0') is None
    assert local_display_number('no-colon') is None
    assert local_display_number(':abc') is None
    assert local_display_number('') is None


def test_eval_and_infer_both_apply_the_filter():
    """Both in-runtime subprocess paths must route their env through the filter.

    Guards against a future edit reintroducing the leak on one path while the
    other stays clean -- the failure mode there is silent and instance-specific.
    """
    from pathlib import Path

    for module in ('eval.py', 'infer.py'):
        source = Path('nexus/extensions/tasks/swe') / module
        text = source.read_text()
        assert 'strip_dangling_display(os_envs)' in text, module
