"""Adapters: nexus source parsing -> the frozen Trajectory model.

parse_nexus_messages(messages, task_output) turns nexus's in-memory OpenAI-chat
trajectory into a Trajectory; action_view(trajectory) is the flat bridge the rules
consume. The rules depend only on those, never on the source shape. Stdlib only.
"""

from __future__ import annotations

import json
import re

from .trajectory_models import ToolCall, ToolCallStatus, Trajectory, Turn


# ---------------------------------------------------------------- repo inference
def derive_repo(instance_id: str | None) -> str | None:
    """SWE-bench ids look like '<org>__<name>-<number|sha>' (optionally 'instance_' prefixed)."""
    if not instance_id:
        return None
    iid = instance_id
    if iid.startswith('instance_'):
        iid = iid[len('instance_') :]
    m = re.match(r'([A-Za-z0-9_.-]+?)__([A-Za-z0-9_.-]+?)-(?:[0-9]+|[0-9a-f]{6,})', iid)
    if m:
        return f'{m.group(1)}/{m.group(2)}'.lower()
    if '__' in iid:
        org, rest = iid.split('__', 1)
        name = re.split(r'-(?:[0-9]|v[0-9a-f])', rest)[0]
        return f'{org}/{name}'.lower()
    return None


# ---------------------------------------------------------------- nexus (live training trajectory)
def _coerce_command_str(command) -> str:
    """Codex's ``shell`` tool passes ``command`` as an argv array, whereas
    swe_agent's ``bash`` passes a single string. ``action_view`` and the rules
    downstream expect a string, so join arrays here.

    ``["bash", "-lc", "<script>"]`` (and ``sh -c`` etc.) unwrap to the inner
    ``<script>`` -- the actual shell text the model wrote -- so online-fetch
    detection sees the real command; other argvs are space-joined; a string is
    returned as-is.
    """
    if isinstance(command, str):
        return command
    if not isinstance(command, list) or not command:
        return ''
    parts = [str(x) for x in command]
    head = parts[0].rsplit('/', 1)[-1].lower()
    if head in ('bash', 'sh', 'zsh', 'dash') and len(parts) >= 3 and parts[1] in ('-c', '-lc', '-lic', '-ic'):
        return parts[2]
    return ' '.join(parts)


def _normalize_nexus_tool(name: str, args: dict) -> dict:
    """Map a nexus tool call -> our ToolCall shape.

    Scaffold-aware: besides swe_agent's ``bash``/``command`` shape it recognizes
    the codex shell tool (``shell``/``exec_command``/``exec`` with the script under
    ``command`` OR ``cmd``, possibly an argv array) and codex's ``apply_patch``
    freeform tool. Without this, every codex command is dropped here and
    action_view() hands the rules an empty command list -- so anti-hack never
    fires on the codex/mini_swe/hycode scaffolds. mini_swe already uses the plain
    ``bash``/``command`` shape and passes through the first branch.
    """
    n = (name or '').lower()
    # Codex ``apply_patch`` freeform tool: args are {'input': '*** Begin Patch...'}.
    # Route to Edit so patch-shaped hacks (str_replace old==new etc.) can see it;
    # not a shell command, so it must not fall through to the Bash branch.
    if n == 'apply_patch':
        return {'tool_name': 'Edit', 'input': {'command': 'apply_patch', 'patch': args.get('input', '')}}
    # bash / execute_bash / shell / run_command ... plus codex, which sends the shell tool as
    # ``exec_command`` with the script under ``cmd`` -- neither the name nor the arg key matches
    # the others, so both spellings have to be accepted or every codex command is dropped here.
    if 'bash' in n or n in {'shell', 'exec_command', 'exec'} or 'command' in args or 'cmd' in args:
        command = args.get('command')
        if command in (None, ''):
            command = args.get('cmd', '')
        return {'tool_name': 'Bash', 'input': {'command': _coerce_command_str(command)}}
    if 'str_replace' in n or 'edit' in n or 'write' in n:  # str_replace_editor / edit / write
        cmd = args.get('command', '')
        tn = 'Read' if cmd == 'view' else ('Write' if cmd == 'create' else 'Edit')
        return {
            'tool_name': tn,
            'input': {
                'command': cmd,
                'file_path': args.get('path') or args.get('file_path'),
                'old_str': args.get('old_str', ''),
                'new_str': args.get('new_str') or args.get('file_text', ''),
            },
        }
    return {
        'tool_name': name or 'Tool',
        'input': {
            'command': args.get('command', ''),
            **{k: v for k, v in args.items() if isinstance(v, (str, int, float))},
        },
    }


def parse_nexus_messages(messages: list, task_output) -> Trajectory:
    """nexus's in-memory trajectory -> Trajectory. The training/eval hook calls this.

    messages: OpenAI-style chat (assistant carries `tool_calls`; a `role='tool'`
              message carries the tool observation, linked by tool_call_id).
    task_output: nexus SWETaskOutput (read instance_id / repo / patch / score).
    dicts or pydantic objects both work.
    """

    def g(o, k, d=None):
        return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)

    iid = g(task_output, 'instance_id')
    results, positional = tool_observation_maps(messages)

    turns: list[Turn] = []
    for i, m in enumerate(messages):
        if g(m, 'role') != 'assistant':
            continue
        tool_calls: list[ToolCall] = []
        for k, tc in enumerate(g(m, 'tool_calls') or []):
            fn = g(tc, 'function')
            try:
                a = (
                    json.loads(g(fn, 'arguments'))
                    if isinstance(g(fn, 'arguments'), str)
                    else (g(fn, 'arguments') or {})
                )
            except (json.JSONDecodeError, TypeError):
                a = {}
            tid = g(tc, 'id')
            norm = _normalize_nexus_tool(g(fn, 'name'), a)
            result = results.get(tid) if tid else None
            if result is None:
                result = positional.get((i, k))
            tool_calls.append(
                ToolCall(
                    tool_use_id=tid or f't_{i}',
                    tool_name=norm['tool_name'],
                    input=norm['input'],
                    result=result,
                    status=ToolCallStatus.SUCCESS,
                )
            )
        if tool_calls:
            turns.append(Turn(turn_number=len(turns), tool_calls=tool_calls))

    return Trajectory(
        id=iid or 'unknown',
        task_id=iid or 'unknown',
        scaffold='nexus_swe',
        turns=turns,
        repo=g(task_output, 'repo') or derive_repo(iid),
        task_type=g(task_output, 'task_type'),
        base_commit=g(task_output, 'base_commit'),
        history_truncated=g(task_output, 'history_truncated'),
        score=g(task_output, 'score'),
        final_patch=g(task_output, 'patch') or g(task_output, 'patch_text') or '',
    )


def tool_observation_maps(messages: list) -> tuple[dict[str, str], dict[tuple[int, int], str]]:
    """Map tool-call IDs and positions to observations for Nexus chat messages."""

    def g(o, k, d=None):
        return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)

    results = {}  # tool_call_id -> observation
    # (assistant index, position within its tool_calls) -> observation. Training clients may
    # emit tool calls with no id (HyTuner mints none: the RL path never goes through an
    # OpenAI server), and an id lookup then finds nothing, leaving every rule to run on
    # commands with no output. Results follow their calls in order, so position recovers the
    # pairing. Pending calls are dropped at any non-tool message, so a turn that produced no
    # observation (a format error, whose feedback arrives as a user message) cannot shift the
    # observations of the turns after it.
    positional: dict[tuple[int, int], str] = {}
    pending: list[tuple[int, int]] = []
    for i, m in enumerate(messages):
        role = g(m, 'role')
        if role == 'assistant':
            pending = [(i, k) for k in range(len(g(m, 'tool_calls') or []))]
            continue
        if role != 'tool':
            pending = []
            continue
        c = g(m, 'content')
        c = c if isinstance(c, str) else json.dumps(c)
        tid = g(m, 'tool_call_id')
        if tid:
            results[tid] = c
        if pending:
            positional[pending.pop(0)] = c

    return results, positional


# ---------------------------------------------------------------- the bridge
def action_view(traj: Trajectory) -> dict:
    """Trajectory -> the flat action view that the online_answer_fetch rule consumes.

    Only the fields that rule needs survive: the bash commands (and their ordered
    command/observation steps for per-fetch downstream inference) and the joined
    observation text (used to derive the package-under-fix in the repo-independent
    fallback), plus base_commit, which anchors the local git-history checks, and
    history_truncated, which settles whether those checks have anything to find. Patch /
    edit fields were dropped with the patch-based rules.
    """
    bash: list[str] = []
    bash_steps: list[dict] = []  # ordered (command, observation) for per-fetch downstream inference
    obs: list[str] = []
    for turn in traj.turns:
        for tc in turn.tool_calls:
            if tc.tool_name == 'Bash':
                cmd = tc.input.get('command', '')
                if cmd:
                    bash.append(cmd)
                    bash_steps.append({'command': cmd, 'observation': tc.result or ''})
            if tc.result:
                obs.append(tc.result)
    return {
        'run_id': traj.id,
        'scaffold': traj.scaffold,
        'repo': traj.repo,
        'task_type': traj.task_type,
        'base_commit': traj.base_commit,
        'history_truncated': traj.history_truncated,
        'resolved': traj.resolved,
        'score': traj.score,
        'bash_commands': bash,
        'bash_steps': bash_steps,
        'observations_text': ('\n'.join(obs) + '\n' + (traj.metadata.get('observations_text') or ''))[:60000],
    }
