from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import os
import queue
import re
import shlex
import subprocess
import sys
import threading

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any


logger = logging.getLogger(__name__)

# bashlex is used to robustly split a raw shell command into its constituent
# sub-commands (one per ``kind == 'command'`` node in the bash AST). When the
# package is unavailable the whole tool-call statistics pipeline is disabled
# (see :func:`is_tool_stats_runtime_ready`).
try:
    import bashlex  # type: ignore

    _BASHLEX_AVAILABLE = True
except Exception as _e:  # pragma: no cover - defensive
    bashlex = None  # type: ignore[assignment]
    _BASHLEX_AVAILABLE = False
    logger.warning(
        'tool_analyzer: bashlex is unavailable (%s); tool-call statistics '
        'collection and hytuner upload of tool metrics will be disabled. '
        'Install bashlex (>=0.18) to re-enable them.',
        _e,
    )


# ============================================================================
# Tunables
# ============================================================================

# Truncated-raw command length kept on each command record.
RAW_COMMAND_TRUNC_LEN = 200


_BASHLEX_EXPANSION_LIMIT = int(os.environ.get('NEXUS_BASHLEX_EXPANSION_LIMIT', '10'))
_BASHLEX_MAX_INPUT_LEN = int(os.environ.get('NEXUS_BASHLEX_MAX_INPUT_LEN', '10000'))
_BASHLEX_MAX_SUBSTITUTIONS = int(os.environ.get('NEXUS_BASHLEX_MAX_SUBSTITUTIONS', '50'))
_BASHLEX_TIMEOUT_SECONDS = float(os.environ.get('NEXUS_BASHLEX_TIMEOUT_SECONDS', '1.0'))
_BASHLEX_WORKER_MAX_RSS_MB = float(os.environ.get('NEXUS_BASHLEX_WORKER_MAX_RSS_MB', '512'))
_BASHLEX_CACHE_SIZE = int(os.environ.get('NEXUS_BASHLEX_CACHE_SIZE', '4096'))
# Long commands (heredocs, inline scripts) rarely repeat, yet would dominate the cache's memory.
_BASHLEX_CACHE_MAX_INPUT_LEN = int(os.environ.get('NEXUS_BASHLEX_CACHE_MAX_INPUT_LEN', '2000'))

_BASHLEX_WORKER_PATH = Path(__file__).with_name('_bashlex_worker.py')


class _BashlexParserUnavailableError(Exception):
    """The parser process timed out, died or could not start; the outcome must not be cached."""

    def __init__(self, reason: str, *, timed_out: bool = False) -> None:
        super().__init__(reason)
        self.timed_out = timed_out


def _read_parser_responses(stdout: IO[bytes], responses: queue.Queue[bytes]) -> None:
    for line in iter(stdout.readline, b''):
        responses.put(line)
    responses.put(b'')


def _write_parser_requests(stdin: IO[bytes], requests: queue.Queue[bytes | None]) -> None:
    while (payload := requests.get()) is not None:
        try:
            stdin.write(payload)
            stdin.flush()
        except (OSError, ValueError):
            return


@dataclasses.dataclass
class _ParserSession:
    process: subprocess.Popen[bytes]
    requests: queue.Queue[bytes | None]
    responses: queue.Queue[bytes]
    writer: threading.Thread
    reader: threading.Thread


class _BashlexParserProcess:
    """A long-lived ``_bashlex_worker.py`` process shared by every parse in this process.

    Starting one interpreter per command made each parse cost a full Python startup, which on
    a loaded node alone could exceed the timeout; the process is therefore reused. A request
    that overruns ``_BASHLEX_TIMEOUT_SECONDS`` still kills it, so a pathological input can
    neither hang the caller nor keep growing, and the next parse starts a fresh process.
    Requests are serialized because the pipe carries one request at a time.

    Requests are written by a writer thread, so the timeout also bounds a request larger than
    the pipe buffer sent to a parser that is not reading stdin (still importing, or stuck).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: _ParserSession | None = None
        self._inherited_after_fork: list[_ParserSession] = []

    def parse(self, text: str) -> list[list[str]] | None:
        """Return the command words of ``text``, or None when bashlex rejects it.

        Raises:
            _BashlexParserUnavailableError: The process timed out, died or could not start.
        """
        with self._lock:
            session = self._ensure_started_locked()
            try:
                return self._request_locked(session, text)
            except BaseException:
                self._stop_locked()
                raise

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def reset_after_fork(self) -> None:
        """Forget the parent's process in a forked child, which starts its own on demand.

        The parent's pipes stay referenced and are never closed here: closing them could block
        on a buffer lock held by the parent's reader or writer thread, which does not exist in
        the child.
        """
        if self._session is not None:
            self._inherited_after_fork.append(self._session)
        self._lock = threading.Lock()
        self._session = None

    def _ensure_started_locked(self) -> _ParserSession:
        if self._session is not None and self._session.process.poll() is not None:
            self._stop_locked()
        if self._session is None:
            try:
                process = subprocess.Popen(
                    [sys.executable, str(_BASHLEX_WORKER_PATH), str(_BASHLEX_WORKER_MAX_RSS_MB)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as error:
                raise _BashlexParserUnavailableError(f'cannot start parser: {error}') from error
            requests: queue.Queue[bytes | None] = queue.Queue()
            responses: queue.Queue[bytes] = queue.Queue()
            writer = threading.Thread(
                target=_write_parser_requests,
                args=(process.stdin, requests),
                name='bashlex-parser-writer',
                daemon=True,
            )
            reader = threading.Thread(
                target=_read_parser_responses,
                args=(process.stdout, responses),
                name='bashlex-parser-reader',
                daemon=True,
            )
            writer.start()
            reader.start()
            self._session = _ParserSession(process, requests, responses, writer, reader)
        return self._session

    def _request_locked(self, session: _ParserSession, text: str) -> list[list[str]] | None:
        payload = json.dumps({'text': text, 'expansionlimit': _BASHLEX_EXPANSION_LIMIT}).encode('utf-8') + b'\n'
        session.requests.put(payload)
        try:
            line = session.responses.get(timeout=_BASHLEX_TIMEOUT_SECONDS)
        except queue.Empty:
            raise _BashlexParserUnavailableError('timed out', timed_out=True) from None
        if not line:
            raise _BashlexParserUnavailableError(f'parser closed its output (exit code {session.process.poll()})')
        try:
            response = json.loads(line)
            commands = response['commands']
            exiting = response.get('exiting', False)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise _BashlexParserUnavailableError(f'malformed response: {error}') from error
        if exiting:
            self._stop_locked()
        return commands

    def _stop_locked(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        process = session.process
        session.requests.put(None)
        if process.poll() is None:
            process.kill()
        process.wait()
        # The kill fails a write blocked on the full pipe; stdin cannot be closed while the
        # writer still holds its buffer lock.
        session.writer.join(timeout=1.0)
        session.reader.join(timeout=1.0)
        for stream in (process.stdin, process.stdout):
            # Closing a broken stdin re-raises the unflushed request as BrokenPipeError.
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()


_bashlex_parser = _BashlexParserProcess()
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_bashlex_parser.reset_after_fork)


def _parse_command_words(text: str) -> tuple[tuple[str, ...], ...] | None:
    commands = _bashlex_parser.parse(text)
    return None if commands is None else tuple(tuple(words) for words in commands)


# A _BashlexParserUnavailableError propagates, so only deterministic outcomes are cached.
_parse_command_words_cached = functools.lru_cache(maxsize=_BASHLEX_CACHE_SIZE)(_parse_command_words)


def _safe_bashlex_parse(text: str) -> list[list[str]] | None:
    """Parse and collect command words in the long-lived parser process, killed on timeout."""
    if bashlex is None:  # pragma: no cover - defensive
        return None

    n = len(text)
    if n > _BASHLEX_MAX_INPUT_LEN:
        logger.debug(
            'tool_analyzer: skip bashlex.parse (input length %d > %d)',
            n,
            _BASHLEX_MAX_INPUT_LEN,
        )
        return None
    # Cheap O(n) scans; both must pass.
    if text.count('$(') > _BASHLEX_MAX_SUBSTITUTIONS or text.count('`') > _BASHLEX_MAX_SUBSTITUTIONS:
        logger.debug(
            'tool_analyzer: skip bashlex.parse (excessive substitutions in input of length %d)',
            n,
        )
        return None

    parse = _parse_command_words_cached if n <= _BASHLEX_CACHE_MAX_INPUT_LEN else _parse_command_words
    try:
        commands = parse(text)
    except _BashlexParserUnavailableError as error:
        if error.timed_out:
            logger.warning(
                'tool_analyzer: bashlex.parse timed out after %.1fs; '
                'killing parser and dropping command from tool-call statistics (input length %d)',
                _BASHLEX_TIMEOUT_SECONDS,
                n,
            )
        else:
            logger.debug(
                'tool_analyzer: bashlex parser unavailable (%s); this command will be excluded from tool-call statistics.',
                error,
            )
        return None
    if commands is None:
        logger.debug(
            'tool_analyzer: bashlex rejected the input; this command will be excluded from tool-call statistics.'
        )
        return None
    return [list(words) for words in commands]


_STR_REPLACE_EDITOR_VALID_COMMANDS: frozenset[str] = frozenset({
    'view',
    'create',
    'str_replace',
    'insert',
    'undo_edit',
})

_DEFAULT_TOOL_STATS_TASKS: tuple[str, ...] = (
    'harbor_bridge_task',
    'harbor_trial_task',
    'swe_task',
    'terminal_task',
)

_env_tasks = os.environ.get('NEXUS_TOOL_STATS_TASKS')
TOOL_STATS_TASK_ALLOWLIST: frozenset[str] = frozenset(
    s.strip() for s in (_env_tasks.split(',') if _env_tasks else _DEFAULT_TOOL_STATS_TASKS) if s.strip()
)


def is_task_eligible_for_tool_stats(task_name: str | None) -> bool:
    """Return True iff tool-call statistics should be collected for *task_name*."""
    return bool(task_name) and task_name in TOOL_STATS_TASK_ALLOWLIST


def is_tool_stats_runtime_ready() -> bool:
    """Return True iff the runtime has the dependencies needed for tool stats.

    Currently this means the ``bashlex`` package must be importable. When this
    returns ``False`` the trace processor must skip both metric computation and
    the hytuner upload of tool metrics, regardless of any task allowlist.
    """
    return _BASHLEX_AVAILABLE


def should_collect_tool_stats(task_name: str | None) -> bool:
    """Combined gate for tool-call statistics collection.

    Tool stats are collected only when **both** independent gates pass:
      1. :func:`is_tool_stats_runtime_ready` (bashlex available);
      2. :func:`is_task_eligible_for_tool_stats` (task in allowlist).

    A ``task_name`` of ``None`` skips gate (2) so unscoped callers (e.g. unit
    tests) still get stats when bashlex is available.
    """
    if not is_tool_stats_runtime_ready():
        return False
    if task_name is None:
        return True
    return is_task_eligible_for_tool_stats(task_name)


# Failure keywords used by both SWE tool-result detection and Terminus2 terminal
# output detection.  Matched case-insensitively as substrings.
FAILURE_KEYWORDS = (
    'error',
    'traceback',
    'command not found',
    'permission denied',
    'no such file',
    'fatal:',
    'syntax error',
    'segmentation fault',
)

# Python interpreter aliases (exact match).  Versioned binaries such as
# ``python3.11`` are detected by :func:`_is_python_head` instead.
PYTHON_ALIASES: frozenset[str] = frozenset({'python', 'python2', 'python3'})


def _is_python_head(head: str) -> bool:
    """Return True if *head* is a python interpreter, including versioned
    binaries like ``python3.11`` / ``python2.7``.
    """
    if not head:
        return False
    if head in PYTHON_ALIASES:
        return True
    # ``python3.11``, ``python3.12.0``, ``python2.7`` ...
    if head.startswith('python'):
        rest = head[len('python') :]
        if not rest:
            return True
        # rest must be all digits and dots and start with a digit
        if rest[0].isdigit() and all(c.isdigit() or c == '.' for c in rest):
            return True
    return False


# Test framework module names that, when passed as a *word* to a python
# interpreter (e.g. ``python -m pytest``), classify the command as
# ``test_execution``. Matched as exact word equality (no substring).
_PYTHON_TEST_MODULES: frozenset[str] = frozenset({
    'pytest',
    'unittest',
    'nose',
    'nose2',
    'tox',
    'doctest',
})


# ----------------------------------------------------------------------------
# Functional categories: head -> category lookup
# ----------------------------------------------------------------------------
# A single sub-command is classified by looking up the basename of its head
# token (``words[0]``) in this table.  Heads not present here fall back to
# 'unknown'.  Test runners with a dedicated CLI (mocha / jest / rspec / ...)
# are mapped directly to ``test_execution`` here; ``go test`` / ``cargo test``
# are handled as a (head, words[1]) special-case in :func:`_classify_words`.

_HEAD_TO_CATEGORY: dict[str, str] = {}


def _register(category: str, *heads: str) -> None:
    for h in heads:
        _HEAD_TO_CATEGORY[h] = category


_register(
    'text_editing',
    'view',
    'create',
    'str_replace',
    'insert',
    'undo_edit',
    'vi',
    'vim',
    'nvim',
    'nano',
    'emacs',
    'ed',
)
_register(
    'process_management',
    'kill',
    'pkill',
    'killall',
    'c-c',
    'c-d',
    'c-z',
    'c-\\',
    'c-b',
)
_register(
    'search_tools',
    'grep',
    'egrep',
    'fgrep',
    'rg',
    'ag',
)
_register(
    'file_operations',
    'cat',
    'head',
    'tail',
    'less',
    'more',
    'strings',
)
_register(
    'find_tools',
    'find',
    'locate',
    'which',
    'whereis',
)
_register(
    'text_processing',
    'awk',
    'sed',
    'cut',
    'tr',
    'sort',
    'uniq',
    'wc',
    'diff',
    'patch',
)
_register(
    'file_management',
    'cd',
    'pushd',
    'popd',
    'cp',
    'mv',
    'rm',
    'mkdir',
    'rmdir',
    'touch',
    'ls',
    'stat',
    'tree',
    'pwd',
    'chmod',
    'chown',
    'chgrp',
    'ln',
)
_register(
    'system_info',
    'ps',
    'uname',
    'whoami',
    'id',
    'hostname',
    'who',
    'w',
    'last',
    'df',
    'du',
    'free',
    'top',
    'htop',
    'iostat',
    'vmstat',
    'lsof',
    'mount',
    'umount',
    'sysctl',
    'dmesg',
    'journalctl',
    'date',
    'cal',
)
_register(
    'network_tools',
    'curl',
    'wget',
    'pip',
    'pip3',
    'pipx',
    'poetry',
    'uv',
    'npm',
    'yarn',
    'pnpm',
    'apt',
    'apt-get',
    'yum',
    'dnf',
    'brew',
    'ssh',
    'scp',
    'rsync',
    'sftp',
    'ftp',
    'telnet',
    'nc',
    'ncat',
    'socat',
    'ping',
    'traceroute',
    'nslookup',
    'dig',
    'host',
    'netstat',
    'ss',
    'ip',
    'ifconfig',
    'route',
    'arp',
)
_register(
    'dev_tools',
    'git',
    'svn',
    'hg',
    'npx',
    'go',
    'node',
    'cargo',
    'rustc',
    'rustup',
    'make',
    'cmake',
    'ninja',
    'c++',
    'gcc',
    'g++',
    'clang',
    'clang++',
    'ld',
    'as',
    'ar',
    'ranlib',
    'strip',
    'objdump',
    'nm',
    'readelf',
    'ldd',
    'mvn',
    'gradle',
    'perl',
    'tsc',
    'docker',
    'docker-compose',
    'podman',
    'kubectl',
    'helm',
    'aws',
    'gcloud',
    'az',
    'conda',
    'virtualenv',
    'pyenv',
    'mysql',
    'jupyter',
    'ipython',
    'twine',
    'wheel',
    'sqlite3',
    'redis-cli',
)
_register(
    'python',
    'python',
    'python2',
    'python3',
)
_register(
    'binary_tools',
    'hexdump',
    'od',
    'xxd',
)
_register(
    'control_flow',
    'echo',
    'printf',
    'export',
    'true',
    'false',
    'sleep',
    'tmux',
    'env',
    'eval',
    'exec',
    'source',
    '.',
    'set',
    'unset',
    'readonly',
    'declare',
    'typeset',
    'break',
    'continue',
    'return',
    'exit',
    'logout',
    'shift',
    'getopt',
    'getopts',
    'wait',
    'jobs',
    'fg',
    'bg',
    'disown',
    'trap',
    '[',
    '[[',
    'expr',
    'bc',
    'dc',
    'time',
    'timeout',
    'nohup',
    'sudo',
    'su',
    'doas',
    'bash',
    'sh',
    'zsh',
    'fish',
    'csh',
    'tcsh',
    'dash',
    'ksh',
    'history',
    'alias',
    'unalias',
    # archive utilities
    'tar',
    'zip',
    'unzip',
    'gzip',
    'gunzip',
    'bzip2',
    'bunzip2',
    'xz',
    'unxz',
    '7z',
    'rar',
    'unrar',
    # data utilities
    'tee',
    'yes',
    'seq',
    'shuf',
    'rev',
    'paste',
    'join',
    'split',
    'csplit',
    'column',
    'expand',
    'unexpand',
    'fold',
    'fmt',
    'pr',
    'nl',
    'xargs',
    'parallel',
    'base64',
    'md5sum',
    'sha1sum',
    'sha256sum',
    'sha512sum',
    'cksum',
    # misc
    'man',
    'help',
    'systemctl',
    'service',
    'crontab',
    'at',
)
_register(
    'test_execution',
    # Python
    'pytest',
    'py.test',
    'unittest',
    'nose',
    'nose2',
    'tox',
    # JS / TS
    'mocha',
    'jest',
    'jasmine',
    'karma',
    'ava',
    'tap',
    'vitest',
    # Ruby
    'rspec',
    'minitest',
    # PHP
    'phpunit',
    'pest',
    # Java / JVM
    'junit',
    # .NET
    'nunit',
    'xunit',
    'dotnet-test',
    # Other
    'ginkgo',
    'gotestsum',
    'ctest',
    'bats',
    'test',
)
# Submission heads are agent actions, not test runs.
_register('control_flow', 'submit')


# ----------------------------------------------------------------------------
# Exec-failure exemption list (read-only / query-style commands)
# ----------------------------------------------------------------------------
# These tools frequently echo source code or log content that contains
# FAILURE_KEYWORDS without the command itself having failed. They are
# exempted from the keyword-based exec-failure check.

EXEC_FAILURE_EXEMPT_TOOLS: frozenset[str] = frozenset({
    # text_editing - viewer/editor sub-commands
    'view',
    'vi',
    'vim',
    'nvim',
    'nano',
    'emacs',
    'ed',
    # file_operations - read-only viewers
    'cat',
    'head',
    'tail',
    'less',
    'more',
    'strings',
    # search_tools - keyword search
    'grep',
    'egrep',
    'fgrep',
    'rg',
    'ag',
    # find_tools - filename matching
    'find',
    'locate',
    'which',
    'whereis',
    # text_processing
    'awk',
    'sed',
    'cut',
    'tr',
    'sort',
    'uniq',
    'wc',
    'diff',
    'patch',
    # file_management - read-only listing/metadata
    'ls',
    'stat',
    'tree',
    'pwd',
    # binary_tools - hex viewers
    'hexdump',
    'od',
    'xxd',
    # system_info / read-only utilities
    'man',
    'help',
    'history',
    'alias',
    'date',
    'cal',
    'hostname',
    'id',
    'groups',
    'who',
    'w',
    'last',
    'df',
    'du',
    'free',
    'top',
    'htop',
    'iostat',
    'vmstat',
    'lsof',
    'mount',
    'sysctl',
    'dmesg',
    'journalctl',
    'base64',
    'md5sum',
    'sha1sum',
    'sha256sum',
    'sha512sum',
    'cksum',
    'tee',
    'yes',
    'seq',
    'shuf',
    'rev',
    'paste',
    'join',
    'split',
    'csplit',
    'column',
    'expand',
    'unexpand',
    'fold',
    'fmt',
    'pr',
    'nl',
    'ping',
    'traceroute',
    'nslookup',
    'dig',
    'host',
    'netstat',
    'ss',
    'ip',
    'ifconfig',
    'route',
    'arp',
    'printf',
    'echo',
    'xargs',
    # task submission
    'submit',
})

# Read-only ``git`` sub-commands. ``git`` itself is NOT in the exemption list
# above because mutating sub-commands (``git apply`` / ``git commit`` /
# ``git checkout``) should still be checked for genuine execution failures.
GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset({
    'log',
    'show',
    'diff',
    'blame',
    'grep',
    'branch',
    'status',
})


def is_exempt_from_failure_check(primary_tool: str, raw_command: str = '') -> bool:
    """Return True iff *primary_tool* is on the exec-failure exemption list.

    For ``git`` the decision additionally depends on the sub-command extracted
    from *raw_command*: only sub-commands in ``GIT_READONLY_SUBCOMMANDS`` are
    exempt.
    """
    if not primary_tool:
        return False
    tool = primary_tool.lower()
    if tool == 'git':
        return _git_subcommand_from_raw(raw_command) in GIT_READONLY_SUBCOMMANDS
    return tool in EXEC_FAILURE_EXEMPT_TOOLS


def _git_subcommand_from_raw(raw_command: str) -> str:
    """Return the first non-flag word after ``git`` in *raw_command*.

    Skips global flags like ``-C /path`` / ``-c key=val``. Uses ``shlex`` for
    quote-aware tokenisation; returns ``''`` if no sub-command is present.
    """
    if not raw_command:
        return ''
    try:
        tokens = shlex.split(raw_command, posix=True)
    except Exception:
        tokens = raw_command.split()
    # Caller may pass the sub-command's raw text directly; in that case
    # tokens[0] *is* ``git``. Otherwise nothing to parse.
    if (len(tokens) < 2 or os.path.basename(tokens[0]).lower() != 'git') and (
        not tokens or os.path.basename(tokens[0]).lower() != 'git'
    ):
        return ''
    j = 1
    while j < len(tokens) and tokens[j].startswith('-'):
        if tokens[j] in ('-C', '-c') and j + 1 < len(tokens):
            j += 2
            continue
        j += 1
    return tokens[j].lower() if j < len(tokens) else ''


def is_command_exempt_from_failure_check(
    sub_cmds: list[dict] | None,
    raw_command: str,
    primary_tool_fallback: str = '',
) -> bool:
    """Return True iff a *whole* command is exempt from exec-failure check.

    A command can be composed of multiple sub-commands joined by ``&&`` /
    ``||`` / ``;`` / ``|`` / sub-shells / command-substitutions. Per the
    bashlex-refactor decision, we only exempt a command when **every**
    sub-command is itself exempt (avoids over-exempting compound commands
    such as ``grep foo file && python script.py``).

    Falls back to ``is_exempt_from_failure_check(primary_tool_fallback,
    raw_command)`` when ``sub_cmds`` is empty.
    """
    if not sub_cmds:
        return is_exempt_from_failure_check(primary_tool_fallback, raw_command)
    for sc in sub_cmds:
        if not isinstance(sc, dict):
            return False
        name = (sc.get('name') or '').lower()
        if not name:
            return False
        if name == 'git':
            sub_raw = sc.get('raw') or raw_command
            if _git_subcommand_from_raw(sub_raw) not in GIT_READONLY_SUBCOMMANDS:
                return False
            continue
        if name not in EXEC_FAILURE_EXEMPT_TOOLS:
            return False
    return True


# ============================================================================
# Word-level classifier (no regex)
# ============================================================================


def _basename_lower(token: str) -> str:
    """Return ``os.path.basename(token).lower()`` or '' for empty/None."""
    if not token:
        return ''
    return os.path.basename(token).lower()


def _looks_like_python_test(words: list[str]) -> bool:
    """AST-derived heuristic: do *words* describe a python test invocation?

    Triggers when the head is a python interpreter (including versioned
    binaries such as ``python3.11``) and any later word is:
      - a known test framework module (e.g. ``python -m pytest``);
      - a file whose basename starts with ``test_`` or ends with ``_test.py``
        (e.g. ``python tests/test_foo.py``);
      - the literal directory name ``tests`` (e.g. ``python tests/``).
    """
    if not words:
        return False
    head = _basename_lower(words[0])
    if not _is_python_head(head):
        return False
    for w in words[1:]:
        wl = w.lower()
        if wl in _PYTHON_TEST_MODULES:
            return True
        base = os.path.basename(wl).rstrip('/')
        if not base:
            continue
        if base == 'tests':
            return True
        if base.startswith('test_') and base.endswith('.py'):
            return True
        if base.endswith('_test.py'):
            return True
    return False


def _classify_words(words: list[str]) -> str:
    """Classify a sub-command (given as its bashlex word list) into a category.

    Pure word-level lookup; never invokes regex. Returns 'unknown' when the
    head token is not recognised.
    """
    if not words:
        return 'unknown'
    head = _basename_lower(words[0])
    if not head:
        return 'unknown'
    # Python test detection (head=python*, with test-shaped argument).
    if _looks_like_python_test(words):
        return 'test_execution'
    # Versioned python binaries (e.g. ``python3.11 script.py``) -> 'python'.
    if _is_python_head(head):
        return 'python'
    # ``go test`` / ``cargo test`` / ``dotnet test`` (head + words[1] == 'test').
    if head in ('go', 'cargo', 'dotnet') and len(words) >= 2 and words[1].lower() == 'test':
        return 'test_execution'
    # ``npm test`` / ``yarn test`` / ``pnpm test``.
    if head in ('npm', 'yarn', 'pnpm') and len(words) >= 2 and words[1].lower() in ('test', 'run'):
        if words[1].lower() == 'test':
            return 'test_execution'
        # ``npm run test`` style
        if len(words) >= 3 and words[2].lower() == 'test':
            return 'test_execution'
    # Direct head -> category lookup.
    return _HEAD_TO_CATEGORY.get(head, 'unknown')


# ============================================================================
# bashlex-driven sub-command extraction
# ============================================================================


# Match a heredoc opener: ``<<`` or ``<<-`` optionally followed by whitespace
# and a tag that is either bare (``EOF``) or wrapped in single/double quotes
# (``'EOF'`` / ``"EOF"``). Only ASCII identifier-style tags are recognised; the
# opener match drives a manual scan for the matching tag line in
# :func:`_strip_heredocs`.
_HEREDOC_OPEN_RE = re.compile(
    r"""<<(?P<dash>-?)\s*(?P<q>['"]?)(?P<tag>[A-Za-z_][\w-]*)(?P=q)""",
)


def _strip_heredocs(raw_command: str) -> str:
    """Remove heredoc bodies (and their openers) from *raw_command*..
    The function is conservative: if the regex doesn't find any heredoc
    opener it returns *raw_command* unchanged.
    """
    if not raw_command or '<<' not in raw_command:
        return raw_command

    out = raw_command
    # Iterate left-to-right; each iteration strips one heredoc and shrinks
    # the string. Bound the loop to avoid pathological cases.
    for _ in range(32):
        m = _HEREDOC_OPEN_RE.search(out)
        if not m:
            break
        tag = m.group('tag')
        dash = m.group('dash') == '-'
        opener_start = m.start()
        body_start = m.end()
        # Find the closing TAG line. The closing tag must appear on its own
        # line (optionally preceded by whitespace -- only tabs when ``<<-``,
        # but real shells accept any whitespace; we accept any whitespace
        # to be lenient).
        if dash:
            close_re = re.compile(rf'\n[\t ]*{re.escape(tag)}[ \t]*(?:\n|\Z)')
        else:
            close_re = re.compile(rf'\n{re.escape(tag)}[ \t]*(?:\n|\Z)')
        cm = close_re.search(out, body_start)
        if cm:
            # Strip [opener_start, cm.end()) and replace with single space so
            # adjacent tokens stay separated (e.g. ``cat <<EOF\n...\nEOF &&
            # ls`` -> ``cat   && ls``).
            out = out[:opener_start] + ' ' + out[cm.end() :]
        else:
            # Truncated: drop everything from the opener onward.
            out = out[:opener_start] + ' '
            break
    return out


def extract_sub_commands(raw_command: str) -> list[dict]:
    """Split *raw_command* into sub-commands via bashlex AST traversal.

    Each returned dict has the shape::

        {'name': '<basename of words[0]>',
         'category': '<category>',
         'raw':  '<reconstructed sub-command text>'}

    Pure-assignment AST nodes (``words == []``) are skipped.

    **bashlex parse failure:** returns ``[]``. The caller is expected to
    treat this as "skip this command" (do not increment parse_total /
    exec_total / category counts) so the metric denominator shrinks rather
    than absorbs misclassified data. A debug-level log is emitted for
    visibility into which commands bashlex cannot parse.

    Returns ``[]`` also when *raw_command* is empty or bashlex is missing.
    """
    if not raw_command or not raw_command.strip():
        return []
    if not _BASHLEX_AVAILABLE:
        return []
    # bashlex cannot parse heredocs with quoted tags (``<< 'EOF'``) which are
    # very common in SWE-style ``cat > file << 'EOF' ... EOF`` writes. Strip
    # heredoc bodies first so bashlex only sees the surrounding shell.
    sanitized = _strip_heredocs(raw_command)
    if not sanitized.strip():
        return []

    commands = _safe_bashlex_parse(sanitized)
    if commands is None:
        return []

    sub_cmds: list[dict] = []
    for words in commands:
        name = _basename_lower(words[0]) or words[0]
        sub_cmds.append({
            'name': name,
            'category': _classify_words(words),
            'raw': ' '.join(words),
        })
    return sub_cmds


# ============================================================================
# Failure detection helpers
# ============================================================================


def _truncate_raw(text: str, limit: int = RAW_COMMAND_TRUNC_LEN) -> str:
    if text is None:
        return ''
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + '...'


def text_contains_failure(text: str) -> bool:
    """Return True iff ``text`` contains any of the FAILURE_KEYWORDS (ci)."""
    if not text:
        return False
    low = text.lower()
    return any(kw in low for kw in FAILURE_KEYWORDS)


# ============================================================================
# SWE turn statistics
# ============================================================================


def _codex_shell_command_text(command: Any) -> str:
    """Recover a bashlex-parseable command string from a Codex ``shell`` argv.

    Codex's ``shell`` function tool passes ``command`` as an argv **array** of
    strings (e.g. ``["bash", "-lc", "<script>"]``), not the single ``command``
    string that swe_agent's ``bash`` tool uses. bashlex expects a shell string,
    so:

      * unwrap the canonical ``bash -lc <script>`` / ``sh -c <script>`` form to
        the inner ``<script>`` (the actual shell text the model wrote), so the
        sub-command decomposition sees the real commands rather than the
        ``bash`` wrapper;
      * otherwise shell-quote-join the argv (a bare ``["ls", "-la"]`` becomes
        ``ls -la``, which bashlex parses directly).

    Returns '' for empty / non-list / non-str inputs (caller excludes it from
    stats, matching the existing unparseable-command policy).
    """
    if isinstance(command, str):
        return command
    if not isinstance(command, list) or not command:
        return ''
    parts = [str(x) for x in command]
    head = os.path.basename(parts[0]).lower()
    if head in ('bash', 'sh', 'zsh', 'dash') and len(parts) >= 3 and parts[1] in ('-c', '-lc', '-lic', '-ic'):
        # The shell script is the argument right after the -c flag.
        return parts[2]
    try:
        return ' '.join(shlex.quote(p) for p in parts)
    except Exception:
        return ' '.join(parts)


def _swe_extract_command_text(tool_name: str, tool_args: dict) -> tuple[str, str]:
    """Extract a representative command string and override-category from a
    SWE tool call.

    Returns (command_string, override_category_or_empty).

      * ``str_replace_editor`` -> sub-command string + ``text_editing`` override.
      * ``review_submit`` / ``submit`` / ``finish`` -> ``submit`` keyword
        with no category override.
      * ``bash`` / ``Bash`` -> raw ``command`` argument, no override.
      * ``shell`` (Codex) -> argv array unwrapped to a shell string, no override.
      * ``apply_patch`` (Codex freeform tool) -> literal ``apply_patch`` token
        with ``text_editing`` override (the args are a raw patch body, not a
        shell command; the fixed token is counted without shell parsing).
      * other tools -> excluded; never parse serialized tool arguments as shell.
    """
    if not isinstance(tool_args, dict):
        return '', ''
    name = (tool_name or '').rsplit('/', 1)[-1]
    if name in ('bash', 'Bash'):
        cmd = tool_args.get('command') or tool_args.get('cmd') or ''
        return (cmd, '') if isinstance(cmd, str) else ('', '')
    # Codex ``shell`` tool: command is an argv array (see responses_proxy
    # converter). swe_agent never emits this name, so the branch is additive.
    if name == 'shell':
        return _codex_shell_command_text(tool_args.get('command')), ''
    # Codex ``apply_patch`` freeform tool: arguments are {'input': '*** Begin
    # Patch ...'} (a raw patch, not shell). Classify as text_editing; the bare
    # ``apply_patch`` token keeps the call countable without parsing the patch.
    if name == 'apply_patch':
        return 'apply_patch', 'text_editing'
    if name in ('str_replace_editor', 'str_replace_based_edit_tool', 'editor'):
        sub = tool_args.get('command') or tool_args.get('action') or ''
        sub_str = str(sub)
        # Only force the ``text_editing`` category when the sub-command is
        # one of the editor's whitelisted actions. If the model produced an
        # invalid value (e.g. ``change``, ``edit``, ``modify``), Pydantic
        # rejects the call before any editing happens, so we must NOT
        # mis-classify it as text_editing -- fall back to the generic
        # category resolution (which will land on ``unknown``).
        if sub_str in _STR_REPLACE_EDITOR_VALID_COMMANDS:
            return sub_str, 'text_editing'
        return sub_str, ''
    if name in ('review_submit', 'submit', 'finish'):
        return 'submit', ''
    return '', ''


def compute_swe_turn_stats(
    assistant_message: dict,
    tool_messages: list[dict],
) -> dict:
    """Compute per-turn tool-call statistics for a SWE-agent turn.

    Counting rules:
      * ``parse_total`` / ``parse_success`` / ``exec_total`` / ``exec_success``
        are command-level. A command whose JSON arguments fail to parse is
        counted in ``parse_total`` only (not exec).
      * **A command whose bashlex sub-command extraction returns [] is
        excluded from every counter** so metric denominators shrink rather
        than be polluted by unparseable inputs.
      * ``commands[i].sub_cmds`` / ``sub_categories`` carry the bashlex
        decomposition; aggregation accumulates category counts at the
        sub-command granularity.
    """
    stats: dict[str, Any] = {
        'parse_total': 0,
        'parse_success': 0,
        'exec_total': 0,
        'exec_success': 0,
        'commands': [],
    }
    try:
        tool_calls = (assistant_message or {}).get('tool_calls') or []

        # No tool_calls at all: count as a single parse failure.
        if not tool_calls:
            stats['parse_total'] = 1
            stats['parse_success'] = 0
            return stats

        # tool_call_id -> tool message content map
        tool_result_by_id: dict[str, str] = {}
        for tm in tool_messages or []:
            if not isinstance(tm, dict):
                continue
            tcid = tm.get('tool_call_id') or ''
            content = tm.get('content', '')
            if isinstance(content, list):
                content = ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
            tool_result_by_id[tcid] = str(content or '')

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get('function') or {}
            tool_name = func.get('name') or ''
            raw_args = func.get('arguments')

            # JSON arg parse
            parse_ok = False
            tool_args: dict = {}
            if isinstance(raw_args, dict):
                parse_ok = True
                tool_args = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                    if isinstance(parsed, dict):
                        tool_args = parsed
                        parse_ok = True
                except Exception:
                    parse_ok = False

            command_str, override_category = _swe_extract_command_text(tool_name, tool_args)

            # Only shell tools feed arguments into bashlex. Existing non-shell
            # tools keep their command labels and category overrides.
            name = tool_name.rsplit('/', 1)[-1]
            if not command_str:
                sub_cmds = []
            elif name in ('bash', 'Bash', 'shell'):
                sub_cmds = extract_sub_commands(command_str)
            else:
                sub_cmds = [
                    {
                        'name': _basename_lower(command_str) or command_str,
                        'category': override_category or _classify_words([command_str]),
                        'raw': command_str,
                    }
                ]
            if not command_str or not sub_cmds:
                logger.debug(
                    'tool_analyzer: SWE tool-call excluded from stats (tool=%r, command_str=%r, parse_ok=%s)',
                    tool_name,
                    _truncate_raw(command_str),
                    parse_ok,
                )
                # Still record a placeholder command entry so downstream
                # debugging can see the call existed, but do NOT bump any
                # of the four counters.
                stats['commands'].append({
                    'tool_name': tool_name,
                    'parse_ok': parse_ok,
                    'exec_ok': None,
                    'raw': _truncate_raw(command_str),
                    'sub_cmds': [],
                    'sub_categories': [],
                    'excluded_from_stats': True,
                })
                continue

            if override_category:
                for sc in sub_cmds:
                    sc['category'] = override_category

            # Now this command participates in counting.
            stats['parse_total'] += 1
            if parse_ok:
                stats['parse_success'] += 1

            sub_names = [sc.get('name', '') for sc in sub_cmds]
            sub_categories = [sc.get('category', 'unknown') for sc in sub_cmds]

            tcid = tc.get('id') or ''
            exec_ok: bool | None
            if not parse_ok:
                exec_ok = None
            else:
                stats['exec_total'] += 1
                tool_result = tool_result_by_id.get(tcid, '')
                primary_tool_fallback = sub_names[0] if sub_names else tool_name
                if is_command_exempt_from_failure_check(sub_cmds, command_str, primary_tool_fallback):
                    exec_ok = True
                    stats['exec_success'] += 1
                elif text_contains_failure(tool_result):
                    exec_ok = False
                else:
                    exec_ok = True
                    stats['exec_success'] += 1

            stats['commands'].append({
                'tool_name': tool_name,
                'parse_ok': parse_ok,
                'exec_ok': exec_ok,
                'raw': _truncate_raw(command_str),
                'sub_cmds': sub_names,
                'sub_categories': sub_categories,
            })
    except Exception as e:  # pragma: no cover - defensive
        logger.debug('tool_analyzer: compute_swe_turn_stats failed: %s', e)
        stats.setdefault('error', str(e))
    return stats


# ============================================================================
# Terminus2 turn statistics
# ============================================================================

_TERMINUS_JSON_PARSER = None
_TERMINUS_XML_PARSER = None


def _get_terminus_parsers() -> tuple[Any, Any]:
    global _TERMINUS_JSON_PARSER, _TERMINUS_XML_PARSER
    if _TERMINUS_JSON_PARSER is None or _TERMINUS_XML_PARSER is None:
        try:
            from nexus.extensions.agents.terminus2.parsing import TerminusJSONParser  # type: ignore
            from nexus.extensions.agents.terminus2.xml_parsing import TerminusXMLParser  # type: ignore

            _TERMINUS_JSON_PARSER = TerminusJSONParser()
            _TERMINUS_XML_PARSER = TerminusXMLParser()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug('tool_analyzer: failed to import terminus2 parsers: %s', e)
    return _TERMINUS_JSON_PARSER, _TERMINUS_XML_PARSER


def _parse_terminus_response(content: str, parser_name: str = 'json') -> list[str]:
    if not content:
        return []
    json_parser, xml_parser = _get_terminus_parsers()
    parser = xml_parser if parser_name == 'xml' else json_parser
    if parser is None:
        return []
    try:
        result = parser.parse_response(content)
    except Exception as e:
        logger.debug('tool_analyzer: terminus parse_response raised: %s', e)
        return []
    keystrokes: list[str] = []
    for cmd in getattr(result, 'commands', []) or []:
        try:
            keystrokes.append(str(getattr(cmd, 'keystrokes', '') or ''))
        except Exception:
            continue
    return keystrokes


def compute_terminus2_turn_stats(
    assistant_content: str,
    next_user_content: str,
    parser_name: str = 'json',
) -> dict:
    """Compute per-turn stats for a Terminus2 turn.

    Counting rules mirror :func:`compute_swe_turn_stats`: a keystroke whose
    bashlex sub-command extraction returns ``[]`` is excluded from every
    counter (does NOT increment parse_total / exec_total).
    """
    stats: dict[str, Any] = {
        'parse_total': 0,
        'parse_success': 0,
        'exec_total': 0,
        'exec_success': 0,
        'commands': [],
    }
    try:
        keystrokes = _parse_terminus_response(assistant_content, parser_name)

        if not keystrokes:
            stats['parse_total'] = 1
            stats['parse_success'] = 0
            return stats

        batch_failed = text_contains_failure(next_user_content)
        for ks in keystrokes:
            ks_parse_ok = bool(ks and ks.strip())
            sub_cmds = extract_sub_commands(ks) if ks_parse_ok else []
            if not ks_parse_ok or not sub_cmds:
                logger.debug(
                    'tool_analyzer: Terminus2 keystroke excluded from stats (parse_ok=%s, raw=%r)',
                    ks_parse_ok,
                    _truncate_raw(ks),
                )
                stats['commands'].append({
                    'tool_name': 'tmux_keystrokes',
                    'parse_ok': ks_parse_ok,
                    'exec_ok': None,
                    'raw': _truncate_raw(ks),
                    'sub_cmds': [],
                    'sub_categories': [],
                    'excluded_from_stats': True,
                })
                continue

            stats['parse_total'] += 1
            stats['parse_success'] += 1
            stats['exec_total'] += 1

            sub_names = [sc.get('name', '') for sc in sub_cmds]
            sub_categories = [sc.get('category', 'unknown') for sc in sub_cmds]
            primary_tool_fallback = sub_names[0] if sub_names else ''

            if is_command_exempt_from_failure_check(sub_cmds, ks, primary_tool_fallback):
                exec_ok: bool | None = True
            else:
                exec_ok = not batch_failed
            if exec_ok:
                stats['exec_success'] += 1

            stats['commands'].append({
                'tool_name': 'tmux_keystrokes',
                'parse_ok': True,
                'exec_ok': exec_ok,
                'raw': _truncate_raw(ks),
                'sub_cmds': sub_names,
                'sub_categories': sub_categories,
            })
    except Exception as e:  # pragma: no cover - defensive
        logger.debug('tool_analyzer: compute_terminus2_turn_stats failed: %s', e)
        stats.setdefault('error', str(e))
    return stats


# ============================================================================
# Aggregation / summary
# ============================================================================


def aggregate_tool_call_stats(per_turn_stats: Iterable[dict]) -> dict:
    """Aggregate per-turn stats into a trajectory-level summary.

    ``category_counts`` is computed at the **sub-command** granularity:
    a tool-call composed of ``cd /x && grep foo`` contributes one count to
    ``file_management`` (cd) and one to ``search_tools`` (grep). Commands
    excluded from stats (no sub_cmds) contribute nothing.
    """
    per_turn_stats = list(per_turn_stats or [])
    summary: dict[str, Any] = {
        'turns': len(per_turn_stats),
        'parse_total': 0,
        'parse_success': 0,
        'exec_total': 0,
        'exec_success': 0,
        'category_counts': defaultdict(int),
    }
    for ts in per_turn_stats:
        if not isinstance(ts, dict):
            continue
        summary['parse_total'] += int(ts.get('parse_total', 0) or 0)
        summary['parse_success'] += int(ts.get('parse_success', 0) or 0)
        summary['exec_total'] += int(ts.get('exec_total', 0) or 0)
        summary['exec_success'] += int(ts.get('exec_success', 0) or 0)
        for cmd in ts.get('commands', []) or []:
            if not isinstance(cmd, dict):
                continue
            if cmd.get('excluded_from_stats'):
                continue
            for cat in cmd.get('sub_categories') or []:
                summary['category_counts'][cat or 'unknown'] += 1
    summary['category_counts'] = dict(summary['category_counts'])
    return summary


_EMPTY_TURN_STATS: dict[str, Any] = {
    'parse_total': 0,
    'parse_success': 0,
    'exec_total': 0,
    'exec_success': 0,
    'commands': [],
}


def _coerce_text(content: Any) -> str:
    """Flatten an OpenAI-style content field (str | list[dict|str]) to text."""
    if content is None:
        return ''
    if isinstance(content, list):
        return ''.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def build_turn_stats_from_messages(
    new_msgs: list[dict] | None,
    task_name: str | None = None,
) -> dict | None:
    """Derive a single per-turn stats record from a message-slice.

    Returns:
        - ``None``: gating rejected the turn (task not eligible / bashlex missing).
        - ``dict``: a turn-stats record (possibly the empty placeholder when no
          assistant message is present, so the index alignment with
          ``num_turns`` is preserved on the caller side).
    """
    if not should_collect_tool_stats(task_name):
        return None

    msgs = list(new_msgs) if isinstance(new_msgs, list) else []
    if not msgs:
        return dict(_EMPTY_TURN_STATS)

    assistant_msg = None
    assistant_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, dict) and m.get('role') == 'assistant':
            assistant_msg = m
            assistant_idx = i
            break

    if assistant_msg is None:
        return dict(_EMPTY_TURN_STATS)

    downstream = msgs[assistant_idx + 1 :]
    has_tool_calls = bool(assistant_msg.get('tool_calls'))
    tool_msgs = [m for m in downstream if isinstance(m, dict) and m.get('role') == 'tool']

    if has_tool_calls or tool_msgs:
        return compute_swe_turn_stats(assistant_msg, tool_msgs)

    content = _coerce_text(assistant_msg.get('content'))
    next_user_content = ''
    for m in downstream:
        if isinstance(m, dict) and m.get('role') == 'user':
            next_user_content = _coerce_text(m.get('content'))
            break
    return compute_terminus2_turn_stats(content, next_user_content)


def simplify_turn_stats(turn_stats: dict) -> dict:
    """Compress a per-turn record into a category-keyed summary.

    The verbose ``commands`` list is collapsed into
    ``{category: {'count': n, 'parse_ok': n, 'exec_ok': n}}``:
      * ``count`` reflects how many **sub-commands** in that category appeared
        in this turn (a compound tool-call with N sub-commands contributes N
        counts);
      * ``parse_ok`` / ``exec_ok`` keep their **command-level** semantics:
        each command contributes 1 to every distinct sub-category it touches.

    Commands excluded from stats (``excluded_from_stats=True``) are skipped
    entirely. Other turn-level counters pass through unchanged.
    """
    if not isinstance(turn_stats, dict):
        return {}
    simplified: dict[str, Any] = {k: v for k, v in turn_stats.items() if k != 'commands'}
    cat_counts: dict[str, dict[str, int]] = {}
    for cmd in turn_stats.get('commands') or []:
        if not isinstance(cmd, dict):
            continue
        if cmd.get('excluded_from_stats'):
            continue
        sub_categories = cmd.get('sub_categories') or ['unknown']
        unique_cats: set[str] = set()
        for cat in sub_categories:
            cat = cat or 'unknown'
            bucket = cat_counts.setdefault(cat, {'count': 0, 'parse_ok': 0, 'exec_ok': 0})
            bucket['count'] += 1
            unique_cats.add(cat)
        for cat in unique_cats:
            bucket = cat_counts[cat]
            if cmd.get('parse_ok'):
                bucket['parse_ok'] += 1
            if cmd.get('exec_ok'):
                bucket['exec_ok'] += 1
    simplified['commands'] = cat_counts
    return simplified


def summarize_tool_call_stats(per_turn_stats: Iterable[dict]) -> dict:
    """Trajectory-level summary with a defensive fallback."""
    per_turn_list = list(per_turn_stats or [])
    try:
        return aggregate_tool_call_stats(per_turn_list)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug('tool_analyzer: summarize_tool_call_stats failed: %s', e)
        return {
            'turns': len(per_turn_list),
            'parse_total': 0,
            'parse_success': 0,
            'exec_total': 0,
            'exec_success': 0,
            'category_counts': {},
        }


def inject_tool_call_into_extra(
    extra_fields: dict | None,
    per_turn_stats: Iterable[dict] | None,
) -> dict | None:
    """Attach simplified tool-call stats to an ``extra_fields`` dict.

    Behavior:
      * If ``per_turn_stats`` is empty/None, ``extra_fields`` is returned
        unchanged.
      * Otherwise a new dict is returned with the original keys plus
        ``tool_call_stats`` (per-turn simplified records) and
        ``tool_call_summary`` (trajectory-level aggregate).
    """
    per_turn_list = list(per_turn_stats or [])
    if not per_turn_list:
        return extra_fields
    merged: dict[str, Any] = dict(extra_fields) if isinstance(extra_fields, dict) else {}
    merged['tool_call_stats'] = [simplify_turn_stats(ts) for ts in per_turn_list]
    merged['tool_call_summary'] = summarize_tool_call_stats(per_turn_list)
    return merged
