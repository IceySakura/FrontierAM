"""macOS task instructions for the shared installed CLI agents."""

from typing import Any
from urllib.parse import urlparse


def build_harness_prompt(instruction: str, *, home_dir: str | None, runtime_url: str) -> str:
    port = urlparse(runtime_url).port or 5000
    return f"""You are operating a real macOS desktop. Complete the task on this Mac.
Your shell, file tools and CLI processes run on the same Mac as the desktop.
Use shell commands, file operations, AppleScript, GUI interaction, or a combination.
The desktop user's home is {home_dir or 'the current user home (resolve with echo ~)'}.
Task input files, when present, are under that home's Documents/benchmark_files.
Do not change system language, screen resolution, or unrelated application settings.

Seeing the screen:
Before acting, capture the current desktop with:
curl --fail --silent --show-error --max-time 120 http://127.0.0.1:{port}/screenshot -o /tmp/macosworld-screen.png
Then inspect /tmp/macosworld-screen.png with your image-reading tool (Read or view_image).
Use the actual screenshot dimensions; do not assume a fixed resolution. Refresh the
screenshot after GUI actions before deciding what to do next. Do not use Linux tools
such as gnome-screenshot, xdotool, apt, or DISPLAY.

GUI actions can use /usr/local/bin/python3 with pyautogui, or osascript.
macOS uses command rather than ctrl for shortcuts; applications can be launched with open -a.
Shell/API operations are allowed when they implement the task's requested changes.
Verify the actual resulting files/settings/UI against the task description.
Do not read, modify, search for, or execute evaluator/grader scripts or reference answers.
Do not modify the controller server, native bridge, or other evaluation infrastructure.
Do not stop after describing a plan: use tools until the task is completed or genuinely infeasible.
Finish with TASK_DONE after verifying completion, or TASK_INFEASIBLE with a concrete reason.
These markers do not determine your score; the environment is independently evaluated.

Task:
{instruction}
"""


def harness_error(output: Any) -> str | None:
    """Do not let a failed CLI invocation masquerade as a valid zero-score run."""
    if isinstance(output, dict):
        success, detail = output.get('success'), output.get('stderr')
    else:
        success, detail = getattr(output, 'success', None), getattr(output, 'stderr', None)
    if success is not True:
        return f'harness execution failed: {str(detail or "missing successful CLI result")[:2000]}'
    return None
