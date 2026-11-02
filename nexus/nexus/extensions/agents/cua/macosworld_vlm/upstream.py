# ruff: noqa: W293, SIM108
# Adapted from showlab/macosworld agent/openai.py at
# e2ca8334b3765537e5c1c428ce57990248a1bb5f (Show Lab).
# Licensed CC BY-NC 4.0; see UPSTREAM_LICENSE.txt.
# Prompt unchanged; parser method lifted into a standalone function.
# Preserve upstream supported-action whitelist, including its omissions.

GPT_SYSTEM_PROMPT = """
You are an agent that performs Mac desktop computer tasks by controlling mouse and keyboard through VNC. For each step, you will receive a screenshot observation of the computer screen and should predict the next action.

Your output must be raw text commands with the following structure:
```
<action_name> <parameter_1> <parameter_2>
<action_name> <parameter_1> <parameter_2>
...
```

For example:
```
move_to 0.25 0.5
key_press command-c
left_click
```

Available actions and their parameters:

1. Mouse Actions:
- "move_to": Move cursor to normalized coordinates
  Required params: {"x": float 0-1, "y": float 0-1}
  
- "left_click": Perform left mouse click
  No params required
  
- "middle_click": Perform middle mouse click
  No params required
  
- "right_click": Perform right mouse click
  No params required
  
- "double_click": Perform double left click
  No params required

- "triple_click": Perform triple left click
  No params required

- "drag_to": Drag with the left mouse button to a specified coordinate.
  Required params: {"x": float 0-1, "y": float 0-1}

- "mouse_down": Press and hold a mouse button.
  Required params: {"button": string ("left", "middle", "right")}

- "mouse_up": Release a mouse button.
  Required params: {"button": string ("left", "middle", "right")}

- "scroll_down": Scroll down by proportion of screen height
  Required params: {"amount": float 0-1}
  
- "scroll_up": Scroll up by proportion of screen height
  Required params: {"amount": float 0-1}

- "scroll_left": Scroll up by proportion of screen width
  Required params: {"amount": float 0-1}

- "scroll_right": Scroll up by proportion of screen width
  Required params: {"amount": float 0-1}

2. Keyboard Actions:
- "type_text": Type ASCII text
  Required params: {"text": string}
  Everything after `type_text ` will be parsed as parameter 1, including spaces. No need to escape any characters.
  
- "key_press": Press a key or key combination.
  Required params: {"key": string}
  Available keys: ctrl, command, option, backspace, tab, enter, esc, del, left, up, right, down, or single ASCII characters
  When pressing a combination of keys simultaneously, connect the keys using `-`, for example, `command-c` or `ctrl-alt-del`

3. Control Actions:
- "wait": Wait for specified seconds
  Required params: {"seconds": float}
  
- "fail": Indicate task cannot be completed
  No params required
  
- "done": Indicate task is already finished
  No params required

Important Notes:
- Your username is "ec2-user" and password is "000000"
- All coordinates (x,y) should be normalized between 0 and 1
- All scroll amounts should be normalized between 0 and 1
- Only ASCII characters are allowed for text input
- The control commands (wait, fail, done) must be the only command issued in a round. If one of these commands is used, no other actions should be provided alongside it.
- Return only the actions in a backtick-wrapped plaintext code block, one line per action, no other text
"""


def parse_agent_output(agent_output):
    """
    Parse the raw output string from the GUI agent into a list of actions.
    Each action is a dict with an "action" key and any required parameters.

    This function is robust to:
    - Extra surrounding backticks or triple backticks
    - Extra spaces and non-action text lines
    - Parameters provided with "key=value" format
    - Incomplete or misformatted lines (which will print an error and skip that line)
    """
    valid_actions = {
        'move_to',
        'left_click',
        'middle_click',
        'right_click',
        'double_click',
        'scroll_down',
        'scroll_up',
        'type_text',
        'key_press',
        'wait',
        'fail',
        'done',
    }
    actions = []
    agent_output = agent_output.strip()
    if agent_output.startswith('```') and agent_output.endswith('```'):
        agent_output = agent_output[3:-3].strip()
    agent_output = agent_output.strip('`').strip()
    lines = agent_output.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            tokens = line.split()
            if not tokens:
                continue
            action_cmd = tokens[0].strip().lower()
            if action_cmd not in valid_actions:
                continue
            action_dict = {'action': action_cmd}
            if action_cmd in ['move_to', 'drag_to']:
                if len(tokens) < 3:
                    print(f'Error parsing line (move_to requires 2 parameters): {line}')
                    continue

                def parse_float(token):
                    if '=' in token:
                        token = token.split('=')[-1]
                    return float(token)

                action_dict['x'] = parse_float(tokens[1])
                action_dict['y'] = parse_float(tokens[2])
            elif action_cmd in ['mouse_down', 'mouse_up']:
                if len(tokens) < 2:
                    print(f'Error parsing line ({action_cmd} requires a button parameter): {line}')
                    continue
                button = tokens[1]
                if '=' in button:
                    button = button.split('=')[-1]
                action_dict['button'] = button.lower()
            elif action_cmd in ['scroll_down', 'scroll_up']:
                if len(tokens) < 2:
                    print(f'Error parsing line ({action_cmd} requires 1 parameter): {line}')
                    continue
                token = tokens[1]
                if '=' in token:
                    token = token.split('=')[-1]
                try:
                    action_dict['amount'] = float(token)
                except Exception as e:
                    print(f'Error parsing parameter for {action_cmd}: {line} - {e}')
                    continue
            elif action_cmd == 'wait':
                if len(tokens) < 2:
                    print(f'Error parsing line (wait requires 1 parameter): {line}')
                    continue
                token = tokens[1]
                if '=' in token:
                    token = token.split('=')[-1]
                try:
                    action_dict['seconds'] = float(token)
                except Exception as e:
                    print(f'Error parsing parameter for wait: {line} - {e}')
                    continue
            elif action_cmd == 'type_text':
                raw_text = line[len(tokens[0]) :]
                if raw_text.strip() == '':
                    text = raw_text
                else:
                    text = ' '.join(raw_text.split())
                action_dict['text'] = text
            elif action_cmd == 'key_press':
                if len(tokens) < 2:
                    print(f'Error parsing line (key_press requires a key parameter): {line}')
                    continue
                key = tokens[1]
                if '=' in key:
                    key = key.split('=')[-1]
                action_dict['key'] = key
            actions.append(action_dict)
        except Exception as e:
            print(f'Error parsing line: {line} - {e}')
            continue
    return actions
