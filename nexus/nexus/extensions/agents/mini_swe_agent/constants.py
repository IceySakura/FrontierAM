"""Runtime-safe constants shared by the host agent loop and the sandbox bash tool.

This module is deliberately kept free of any host-only dependency — in
particular ``jinja2``, which ``mini_core`` needs for template rendering. The
RUNTIME ``tools.py`` is imported *inside the sandbox* to execute the bash tool
(``import_from_path`` -> ``importlib.import_module`` on the runtime side), so if
it transitively imported ``mini_core`` (and thus ``jinja2``), the sandbox would
fail to initialize unless ``jinja2`` were installed there. Keeping these plain
constants here lets both ``tools.py`` (sandbox) and ``mini_core``/``agent.py``
(host) share them without coupling the sandbox to ``jinja2``.
"""

SUBMIT_SENTINEL = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
"""First stdout line (rc==0) that signals the agent submitted its patch."""

DEFAULT_ENV_TIMEOUT = 300
"""Per-command timeout in seconds (mirrors upstream mini ``environment.timeout``)."""

DEFAULT_MINI_ENV: dict[str, str] = {
    'PAGER': 'cat',
    'MANPAGER': 'cat',
    'LESS': '-R',
    'PIP_PROGRESS_BAR': 'off',
    'TQDM_DISABLE': '1',
}
"""Env vars injected into every bash command (mirrors upstream mini ``environment.env``)."""
