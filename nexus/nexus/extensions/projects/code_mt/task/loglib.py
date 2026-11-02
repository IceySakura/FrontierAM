"""ANSI-colored, emoji-tagged logging helpers for code_mt phases.

Goal: make setup / verifier / rule-judge / llm-judge / cleanup transitions easy
to spot in long log streams. Honors NO_COLOR; falls back to plain text when the
log stream isn't a TTY-capable handler (anything writing plain text is fine
since we only emit escape sequences, never overwrite/cursor moves).
"""

import logging
import os


_USE_COLOR = os.environ.get('NO_COLOR') is None

_C = {
    'reset': '\x1b[0m',
    'bold': '\x1b[1m',
    'dim': '\x1b[2m',
    'red': '\x1b[31m',
    'green': '\x1b[32m',
    'yellow': '\x1b[33m',
    'blue': '\x1b[34m',
    'magenta': '\x1b[35m',
    'cyan': '\x1b[36m',
    'white': '\x1b[37m',
    'bg_green': '\x1b[42m',
    'bg_red': '\x1b[41m',
    'bg_blue': '\x1b[44m',
}


def _c(color: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f'{_C[color]}{text}{_C["reset"]}'


def _bold(color: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f'{_C["bold"]}{_C[color]}{text}{_C["reset"]}'


# === Markers ===

OK = _bold('green', '✅')
FAIL = _bold('red', '❌')
WARN = _bold('yellow', '⚠️ ')
INFO = _bold('cyan', '▶')
ARROW = _bold('blue', '→')


def banner(logger: logging.Logger, color: str, title: str) -> None:
    """Print a 78-wide colored banner: `══ TITLE ══`."""
    text = f' {title} '
    pad = max(0, 78 - len(text))
    left = pad // 2
    right = pad - left
    line = '═' * left + text + '═' * right
    logger.debug(_bold(color, line))


def phase_start(logger: logging.Logger, tag: str, msg: str) -> None:
    """`[TAG] ▶ msg`"""
    logger.debug('%s %s %s', _bold('cyan', f'[{tag}]'), INFO, msg)


def phase_ok(logger: logging.Logger, tag: str, msg: str) -> None:
    """`[TAG] ✅ msg` (bold green tag)"""
    logger.info('%s %s %s', _bold('green', f'[{tag}]'), OK, msg)


def phase_fail(logger: logging.Logger, tag: str, msg: str) -> None:
    """`[TAG] ❌ msg` (bold red tag)"""
    logger.info('%s %s %s', _bold('red', f'[{tag}]'), FAIL, msg)


def phase_warn(logger: logging.Logger, tag: str, msg: str) -> None:
    logger.warning('%s %s %s', _bold('yellow', f'[{tag}]'), WARN, msg)


def phase_info(logger: logging.Logger, tag: str, msg: str) -> None:
    """`[TAG]   msg` (neutral, indented detail line)"""
    logger.debug('%s   %s', _c('dim', f'[{tag}]'), msg)


def score_line(logger: logging.Logger, name: str, kind: str, score: float, detail: str = '') -> None:
    """`[JUDGE] ✅ fail2pass     (rule)        score=1.0  detail`"""
    ok = score >= 1.0
    marker = OK if ok else FAIL
    name_col = _bold('green' if ok else 'red', f'{name:<18}')
    kind_col = _c('cyan', f'({kind})').ljust(16 + len(_C['cyan']) + len(_C['reset']))
    score_col = _bold('green' if ok else 'red', f'score={score:.2f}')
    extra = f'  {_c("dim", detail)}' if detail else ''
    logger.info('%s %s %s %s %s%s', _bold('magenta', '[JUDGE]'), marker, name_col, kind_col, score_col, extra)
