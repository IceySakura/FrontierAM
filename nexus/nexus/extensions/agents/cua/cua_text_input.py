"""Shared CUA text input snippets, self-contained in the desktop process.

Select paste keys from the focused X11 application's class, not model text or
window titles. Never try both shortcuts: Ctrl+V changes terminal input state.
"""

from __future__ import annotations

import base64
import inspect


def _paste_text(text: str) -> None:
    import sys
    import time

    import pyautogui
    import pyperclip

    keys = ('ctrl', 'v')
    if sys.platform == 'darwin':
        keys = ('command', 'v')
    elif sys.platform.startswith('linux'):
        from Xlib import display

        connection = display.Display()
        try:
            window = connection.get_input_focus().focus
            classes = None
            # Input focus may belong to a child widget rather than a top-level
            # application. Integer focus values (None/PointerRoot) are not windows.
            for _ in range(16):
                if not hasattr(window, 'get_wm_class'):
                    break
                classes = window.get_wm_class()
                if classes:
                    break
                parent = window.query_tree().parent
                if not parent or parent.id == window.id:
                    break
                window = parent
            if not classes:
                raise RuntimeError('CUA paste cannot identify focused X11 application; no keys sent')
            terminal_classes = {
                'gnome-terminal',
                'gnome-terminal-server',
                'org.gnome.terminal',
                'xfce4-terminal',
                'mate-terminal',
                'tilix',
                'terminator',
                'konsole',
                'org.kde.konsole',
                'alacritty',
                'kitty',
                'org.wezfurlong.wezterm',
            }
            if any(str(value).lower() in terminal_classes for value in classes):
                keys = ('ctrl', 'shift', 'v')
            elif any(str(value).lower() in {'xterm', 'uxterm', 'urxvt', 'rxvt'} for value in classes):
                # These terminals conventionally paste the X11 PRIMARY selection
                # with Shift+Insert, not CLIPBOARD. Do not silently paste stale text.
                raise RuntimeError('CUA clipboard paste is not configured for this terminal; no keys sent')
        finally:
            connection.close()
    pyperclip.copy(text)
    if pyperclip.paste() != text:
        raise RuntimeError('CUA clipboard roundtrip failed; no paste keys sent')
    time.sleep(0.1)
    pyautogui.hotkey(*keys)
    time.sleep(0.2)


def text_input_code(text: str) -> str:
    """Type safe ASCII directly; paste Unicode, controls and shell-special text.

    Newlines are text, not implicit submit actions. A model must request an
    explicit Enter when the application's paste behaviour requires submission.
    """
    if not text:
        return 'time.sleep(0.1)\n'
    special = set('<>|;&{}~`"\'\\$()[]*?!')
    if text.isascii() and text.isprintable() and not any(c in special for c in text):
        return f'import pyautogui\npyautogui.typewrite({text!r}, interval=0.03)\n'
    encoded = base64.b64encode(text.encode('utf-8')).decode('ascii')
    # Guest images need not have this version of Nexus installed.
    return (
        inspect.getsource(_paste_text)
        + '\nimport base64\n'
        + f"_paste_text(base64.b64decode('{encoded}').decode('utf-8'))\n"
    )
