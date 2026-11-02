"""Environment sanitisation shared by the SWE task's in-runtime subprocesses.

`get_clean_sandbox_env()` in nexus.utils.shell is used by ~20 callers across agents
and executors, so display handling lives here instead: it is specific to what the
SWE task shells out to (dataset-supplied eval and install scripts, which routinely
drive real browsers) and must not change behaviour for the GUI agents.
"""

import logging
import os


logger = logging.getLogger(__name__)

# An X server creates /tmp/.X11-unix/X<n> for display :<n> as it starts, so the
# absence of that socket is a reliable "nothing is listening" signal for a local
# display.
_X11_SOCKET_DIR = '/tmp/.X11-unix'


def local_display_number(display: str) -> str | None:
    """Return the display number when *display* names a local X display.

    Accepts the forms an X client treats as local -- ``:99``, ``:99.0``, ``unix:99``
    -- and returns ``'99'``. Returns ``None`` for a remote display (``host:0``) or
    anything unparseable, so callers leave those untouched.
    """
    display = display.strip()
    if not display:
        return None

    host, sep, rest = display.rpartition(':')
    if not sep or host not in ('', 'unix'):
        return None

    number = rest.split('.', 1)[0]
    return number if number.isdigit() else None


def strip_dangling_display(env: dict[str, str], *, socket_dir: str = _X11_SOCKET_DIR) -> dict[str, str]:
    """Return *env* without a ``DISPLAY`` that no X server is listening on.

    The runtime server process holds ``DISPLAY=:99`` even when no X server exists:
    ``deployment/runtime/start.sh`` sources ``gui_stack.sh`` unconditionally and that
    script exports ``DISPLAY`` at its top level, but Xvfb is only launched under
    ``NEXUS_GUI_AUTO_START=1``. ``get_clean_sandbox_env()`` copies ``os.environ`` and
    sanitises only PATH/LD_LIBRARY_PATH, so scripts we spawn inherit that value.

    A dangling ``DISPLAY`` is worse than none, because X clients trust it and then
    fail in ways that never mention ``DISPLAY``:

    - Electron/Cypress abort during startup (``Missing X server or $DISPLAY``, then
      SIGSEGV) even with ``--headless``, because ozone selects the X11 backend from
      ``DISPLAY`` alone.
    - Chromium under Playwright still launches and the test itself passes, but ANGLE
      keeps retrying ``xcb_connect()`` against the dead display, its GPU process dies
      and respawns in a loop, and ``browser.close()`` intermittently blocks ~30s --
      past the suite's own timeout. The run then reports a teardown timeout, so the
      suite goes flaky rather than failing outright, which is much harder to
      attribute. Measured on sveltejs__kit-14261 in a single container:
      ``DISPLAY=:99`` failed 7/10 runs, ``DISPLAY`` unset 0/10.

    Only local displays are judged, and only when their socket is missing:

    - A live local display is preserved, so scripts still see it when the GUI stack
      really is up (``NEXUS_GUI_AUTO_START=1`` / the ``gui_start`` executor).
    - A remote ``host:0`` is preserved: its reachability is not ours to decide, and a
      wrong "no server" verdict would strip a display that works.

    ``socket_dir`` is a seam for tests; production callers use the default.
    """
    display = env.get('DISPLAY')
    if display is None:
        return env

    number = local_display_number(display)
    if number is None or os.path.exists(f'{socket_dir}/X{number}'):
        return env

    logger.debug('Unsetting DISPLAY=%s: no X server is listening on it', display)
    env = dict(env)
    del env['DISPLAY']
    return env
