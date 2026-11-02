"""Standalone guest hook: no synthetic user answer or plan approval."""

import json


def main():
    print(
        json.dumps({
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': (
                    'This task has no user simulator. No user answer or plan approval is available. '
                    'Continue autonomously using the task instructions and visible environment; '
                    'do not ask the user or enter plan mode.'
                ),
            }
        })
    )


if __name__ == '__main__':
    main()
