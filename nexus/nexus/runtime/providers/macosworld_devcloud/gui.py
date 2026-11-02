"""Python snippets for macOS GUI actions, executed inside the desktop runtime."""


def mac_multi_click(x: int | None, y: int | None, button: str, clicks: int) -> str:
    # PyAutoGUI's macOS backend emits separate single clicks without click-state counts.
    return (
        'import Quartz\n'
        f'pyautogui.moveTo({x!r}, {y!r})\n'
        'point = tuple(pyautogui.position())\n'
        f'button = {{"left": 0, "right": 1, "middle": 2}}[{button!r}]\n'
        'down, up = {0: (1, 2), 1: (3, 4), 2: (25, 26)}[button]\n'
        f'for count in range(1, {clicks + 1}):\n'
        ' for kind in (down, up):\n'
        '  event = Quartz.CGEventCreateMouseEvent(None, kind, point, button)\n'
        '  Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, count)\n'
        '  Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)\n'
        '  time.sleep(0.05)'
    )
