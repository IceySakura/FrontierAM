"""Guest-only stdlib stream-json supervisor; no Nexus runtime server is required."""

import json
import os
import subprocess
import sys
import uuid

from pathlib import Path


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    process = subprocess.Popen(
        spec['command'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
        env={**os.environ, **spec['env']},
    )
    assert process.stdin is not None
    assert process.stdout is not None
    stdin = process.stdin

    def send(value):
        stdin.write(json.dumps(value, ensure_ascii=False) + '\n')
        stdin.flush()

    init_id = uuid.uuid4().hex
    send({'type': 'control_request', 'request_id': init_id, 'request': {'subtype': 'initialize'}})
    sent = False
    result = None
    try:
        for line in process.stdout:
            event = json.loads(line)
            print(line, end='', flush=True)
            if event.get('type') == 'control_response' and not sent:
                response = event.get('response', {})
                if response.get('request_id') != init_id:
                    continue
                if response.get('subtype') == 'error':
                    raise RuntimeError('CC stream-json initialization failed')
                prompt = Path(spec['prompt_file']).read_text(encoding='utf-8')
                send({
                    'type': 'user',
                    'message': {'role': 'user', 'content': prompt},
                    'session_id': spec['session_id'],
                    'parent_tool_use_id': None,
                })
                sent = True
            elif event.get('type') == 'control_request':
                # User-input tools have already been answered by explicit
                # PreToolUse hooks. A remaining permission callback must not
                # invent an answer or silently approve an unknown operation.
                response = {'behavior': 'deny', 'message': 'Unanswered permission request in V2 session'}
                send({
                    'type': 'control_response',
                    'response': {
                        'subtype': 'success',
                        'request_id': event['request_id'],
                        'response': response,
                    },
                })
            elif (
                event.get('type') == 'result'
                and not event.get('parent_tool_use_id')
                and event.get('session_id') in (None, spec['session_id'])
            ):
                result = event
                break
        stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        if result is None:
            raise RuntimeError('CC stream-json process ended without a result')
        return 1 if result.get('is_error') else 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == '__main__':
    sys.exit(main())
