"""PreToolUse hook script for AskUserQuestion — auto-selects the first (recommended) option.

This script is deployed into the runtime container and invoked by Claude Code's hook system.
It denies the tool call and provides the selected answers via permissionDecisionReason,
matching the exact format used by interactive mode.

Copied verbatim from ``frontend/askuser_hook.py`` (v1) so the artifact module stays
self-contained. See ``agent_setup.build_askuser_hook_writer`` for how it is deployed.
"""

import json
import sys


data = json.load(sys.stdin)
tool_input = data.get('tool_input') or data.get('toolInput') or {}
questions = tool_input.get('questions', [])
parts = []
for q in questions:
    opts = q.get('options', [])
    selected = opts[0].get('label', 'first option') if opts else 'first option'
    parts.append(f'"{q.get("question", "?")}"="{selected}"')
reason = (
    'User has answered your questions: ' + ', '.join(parts) + ". You can now continue with the user's answers in mind."
)
output = {
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': reason,
    }
}
print(json.dumps(output))
