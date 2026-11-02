import pytest


@pytest.mark.parametrize(
    'code',
    [
        'pyautogui.click(100, 200)',
        'pyautogui.hotkey("ctrl", "l")\ntime.sleep(0.5)\npyautogui.write("hello")',
        'pyautogui.scroll(-5)',
        'pyautogui.position()',
        'pyautogui.hotkey("ctrl", "alt", "t")',
        'pyautogui.write("http://localhost:3000")',
        'if platform.system() == "Linux":\n    pyautogui.press("enter")\nelse:\n    time.sleep(0.1)',
        'platform.system()',
        'pyautogui.dragTo(300, 400, duration=0.2, button="left")',
        'pyautogui.click(1920 / 2, 1080 / 2)',
        'pyautogui.hotkey(*["ctrl", "shift", "esc"])',
    ],
)
def test_pyautogui_guard_allows_official_action_subset(code):
    from nexus.runtime.pyautogui_guard import validate_pyautogui_code

    validate_pyautogui_code(code)


@pytest.mark.parametrize(
    'code',
    [
        'import os\npyautogui.click(1, 2)',
        'open("/tmp/x", "w").write("x")',
        'requests.post("http://127.0.0.1:18000/post")',
        'print(pyautogui.position())',
        'value = pyautogui.position()',
        'for _ in range(2):\n    pyautogui.click(1, 2)',
        'pyautogui.screenshot()',
        'pyautogui.locateCenterOnScreen("button.png")',
        'pyautogui.getActiveWindow()',
        'pyautogui.getAllWindows()',
        'pyautogui.getWindowsWithTitle("Chrome")',
        'pyautogui.write("javascript:document.body.innerText")',
        'pyautogui.write("data:text/html,<h1>test</h1>")',
        'getattr(pyautogui, "screenshot")()',
        'pyautogui.__dict__["click"](1, 2)',
        'time.time()',
        'platform.os.system("true")',
    ],
)
def test_pyautogui_guard_rejects_code_outside_action_subset(code):
    from nexus.runtime.pyautogui_guard import PyAutoGUICodeValidationError, validate_pyautogui_code

    with pytest.raises(PyAutoGUICodeValidationError):
        validate_pyautogui_code(code)
