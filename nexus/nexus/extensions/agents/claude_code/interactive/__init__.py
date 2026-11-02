"""Claude Code interactive agent — runs the native TUI behind a PTY.

Unlike the ``claude_code`` agent (which runs ``claude --print`` over pipes),
this drives the real interactive terminal UI, so Claude Code's interactive-only
agent branch (different tools, system prompt, and features like
artifact/workflow/output-style) is exercised.

Architecture:
  * ``hooks.py`` — replaceable interaction transport. Correlates
    AskUserQuestion / ExitPlanMode requests and responses via files.
  * ``session.py`` — framework-agnostic PTY driver
    (start / send_turn / respond_and_wait / close).
  * ``screen.py`` — pyte-based virtual terminal for ``latest_screen``.
  * ``events.py`` — event extraction, truncation, and incremental buffer.
  * ``proxy.py`` — ``PtyAnthropicProxy`` subclass for event tracking.
  * ``runtime.py`` — nexus worker-side glue: installs Claude, starts proxy,
    manages sessions, formats results.
"""
