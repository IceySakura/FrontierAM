"""Shared Android World mobile action-space descriptions, aligned with HyData's mobile action set.

The exposed actions are the subset of HyData mobile actions the Android World backend can execute.
Coordinates are HyData-canonical **normalized ``[0, 1]``** (fractions of screen width/height); the
``mobile/*`` tools scale them to real device pixels internally.
"""

# HyData-aligned mobile action names (tool ``meta['name']`` values), gui_core subset + mobile.*.
MOBILE_ACTION_TYPES = (
    'click',
    'type',
    'press',
    'wait',
    'answer',
    'terminate',
    'mobile.swipe',
    'mobile.long_press',
    'mobile.double_click',
    'mobile.button_press',
    'mobile.open_app',
)

TERMINATION_ACTIONS = ('answer', 'terminate')

MOBILE_ACTION_SPACE = """\
Mobile action space (HyData-aligned). Coordinates are normalized [0, 1] fractions of the screen
(top-left = [0, 0], bottom-right = [1, 1]):
- click: tap at point [x, y].
- type: type text into the focused field (optionally tap a point to focus first).
- press: press and release a single key (e.g. enter).
- wait: wait for the UI to settle.
- answer: submit an answer to the user's query and finish.
- terminate: end the task with a status (success/failure/infeasible).
- mobile.swipe: swipe/fling from point [x1, y1] to [x2, y2] (also used to scroll).
- mobile.long_press: touch and hold at point [x, y].
- mobile.double_click: double-tap at point [x, y].
- mobile.button_press: press a system button (back/home/enter).
- mobile.open_app: open an app by package/name.
"""

MOBILE_PARAMETER_GUIDE = """\
Parameters (coordinates are normalized [0, 1]):
- click: {"point": [x, y]}
- type: {"text": string, "point": [x, y] (optional)}
- press: {"key": string}
- wait: {"duration": seconds}
- answer: {"text": string}
- terminate: {"status": "success"|"failure"|"infeasible", "reason": string (optional)}
- mobile.swipe: {"from": [x1, y1], "to": [x2, y2]}
- mobile.long_press: {"point": [x, y]}
- mobile.double_click: {"point": [x, y]}
- mobile.button_press: {"button": "back"|"home"|"enter"}
- mobile.open_app: {"package": string}
"""

MOBILE_TERMINATION_GUIDANCE = (
    'When the task is complete, call answer (with the answer text) or terminate (with a status). '
    'Do not finish by replying in natural language only; every step must call a mobile tool.'
)


__all__ = [
    'MOBILE_ACTION_SPACE',
    'MOBILE_ACTION_TYPES',
    'MOBILE_PARAMETER_GUIDE',
    'MOBILE_TERMINATION_GUIDANCE',
    'TERMINATION_ACTIONS',
]
