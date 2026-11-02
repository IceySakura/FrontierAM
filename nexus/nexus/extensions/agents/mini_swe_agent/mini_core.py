"""Faithful, transparent port of the mini-swe-agent core pieces.

This module reimplements the small set of upstream mini-swe-agent behaviors that
determine what the model sees and how actions are parsed, so this white-box
agent is byte-for-byte equivalent to upstream mini-swe-agent on SWE-bench:

- The prompt/observation/format-error templates (copied verbatim from upstream
  mini-swe-agent's SWE-bench config).
- The ``BASH_TOOL`` function schema (copied verbatim from upstream
  ``minisweagent/models/utils/actions_toolcall.py``) — passed literally to the
  LLM so the tool schema matches upstream exactly.
- ``parse_toolcall_actions`` (tool-call -> action, with ``FormatError`` on
  malformed output) and the observation formatting, mirroring upstream.
- The submit sentinel detection from ``minisweagent/environments/local.py``.

Templates are rendered with Jinja2 (``StrictUndefined``), exactly like upstream,
so whitespace control and conditionals produce identical text.
"""

from __future__ import annotations

import json

from typing import Any

from jinja2 import StrictUndefined, Template

# Runtime-safe constants live in the dependency-free ``constants`` module so the
# sandbox-side ``tools.py`` can share them without importing this (jinja2-based)
# module. ``check_submitted`` below uses ``SUBMIT_SENTINEL``.
from nexus.extensions.agents.mini_swe_agent.constants import SUBMIT_SENTINEL


# ============== Templates (verbatim from upstream SWE-bench config) ==============

# ``minisweagent`` SWE-bench system template.
SYSTEM_TEMPLATE = 'You are a helpful assistant that can interact with a computer shell to solve programming tasks.'

# ``minisweagent`` SWE-bench instance template.
INSTANCE_TEMPLATE = """\
<uploaded_files>
{{cwd}}
</uploaded_files>
I've uploaded a code repository in the directory {{cwd}}. Consider the following PR description:

<pr_description>
{{task}}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {{cwd}} directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it using the bash tool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long.

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
"""

# ``minisweagent`` SWE-bench observation template (note the curly apostrophe in
# "command's" is intentional — kept verbatim).
OBSERVATION_TEMPLATE = (
    '{% if output.exception_info -%}\n'
    '<exception>{{output.exception_info}}</exception>\n'
    '{% endif -%}\n'
    '<returncode>{{output.returncode}}</returncode>\n'
    '{% if output.output | length < 10000 -%}\n'
    '<output>\n'
    '{{ output.output -}}\n'
    '</output>\n'
    '{%- else -%}\n'
    '<warning>\n'
    'The output of your last command was too long.\n'
    'Please try a different command that produces less output.\n'
    'If you really need to see something from the full command’s output, you can redirect output to a file and then search in that file.\n'
    '</warning>\n'
    '{%- set elided_chars = output.output | length - 10000 -%}\n'
    '<output_head>\n'
    '{{ output.output[:5000] }}\n'
    '</output_head>\n'
    '<elided_chars>\n'
    '{{ elided_chars }} characters elided\n'
    '</elided_chars>\n'
    '<output_tail>\n'
    '{{ output.output[-5000:] }}\n'
    '</output_tail>\n'
    '{%- endif -%}'
)

# ``LitellmModelConfig`` default ``format_error_template`` — upstream's SWE-bench
# config does not override it, so a format error is surfaced to the model as the
# raw error text.
FORMAT_ERROR_TEMPLATE = '{{ error }}'


# ============== Tool schema (verbatim from upstream actions_toolcall.py) ==============

BASH_TOOL: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'bash',
        'description': 'Execute a bash command',
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute',
                }
            },
            'required': ['command'],
        },
    },
}

TOOL_NAME = 'bash'
"""LLM-visible name of the single tool exposed to the model."""


# ============== Rendering ==============


def render(template: str, **template_vars: Any) -> str:
    """Render a Jinja2 template with ``StrictUndefined`` (same engine as upstream)."""
    return Template(template, undefined=StrictUndefined).render(**template_vars)


# ============== Action parsing ==============


class FormatError(Exception):
    """Raised when the model output is not in the expected format.

    Mirrors ``minisweagent.exceptions.FormatError``: carries the message(s) that
    should be appended to the conversation so the model can correct itself.
    """

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


def parse_toolcall_actions(
    tool_calls: list,
    *,
    format_error_template: str = FORMAT_ERROR_TEMPLATE,
    finish_reason: str | None = None,
) -> list[dict]:
    """Parse tool calls into actions. Raises ``FormatError`` on malformed output.

    Faithful port of upstream ``parse_toolcall_actions``: no tool calls, invalid
    JSON arguments, an unknown tool name, or a missing ``command`` field each
    produce a ``FormatError`` whose message is the rendered
    ``format_error_template`` (role ``user``).
    """
    if not tool_calls:
        raise FormatError({
            'role': 'user',
            'content': render(
                format_error_template,
                error='No tool calls found in the response. Every response MUST include at least one tool call.',
                actions=[],
                has_tool_calls=False,
                finish_reason=finish_reason,
            ),
        })

    actions: list[dict] = []
    for tool_call in tool_calls:
        error_msg = ''
        args: Any = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:  # noqa: BLE001 - mirror upstream: any parse failure is a format error
            error_msg = f'Error parsing tool call arguments: {e}.'
        if tool_call.function.name != TOOL_NAME:
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if not isinstance(args, dict) or 'command' not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError({
                'role': 'user',
                'content': render(
                    format_error_template,
                    actions=[],
                    error=error_msg.strip(),
                    has_tool_calls=True,
                    finish_reason=finish_reason,
                ),
            })
        actions.append({'command': args['command'], 'tool_call_id': tool_call.id})
    return actions


# ============== Observation formatting ==============


def format_observation(action: dict, output: dict, *, observation_template: str = OBSERVATION_TEMPLATE) -> dict:
    """Render one execution ``output`` dict into a tool-result message.

    ``output`` has ``output`` (combined stdout+stderr), ``returncode`` and
    ``exception_info`` keys — mirroring upstream ``LocalEnvironment.execute``.
    """
    content = render(observation_template, output=output)
    return {'role': 'tool', 'tool_call_id': action['tool_call_id'], 'content': content}


# ============== Submit sentinel ==============


def check_submitted(raw_output: str, returncode: int) -> str | None:
    """Return the submission text if ``raw_output`` signals a submit, else ``None``.

    Faithful port of ``LocalEnvironment._check_finished``: after left-stripping
    the output, the first line (stripped) must equal ``SUBMIT_SENTINEL`` and the
    return code must be ``0``; the submission is everything after that line.
    """
    lines = raw_output.lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == SUBMIT_SENTINEL and returncode == 0:
        return ''.join(lines[1:])
    return None
