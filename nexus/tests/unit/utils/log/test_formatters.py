from __future__ import annotations

import json
import logging


def test_json_line_formatter_accepts_injected_context_provider():
    from nexus.utils.log.formatters import JsonLineFormatter

    record = logging.LogRecord(
        name='nexus.test.json',
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg='hello %s',
        args=('json',),
        exc_info=None,
    )
    record.execution_id = 'exec-record'
    record.custom_payload = {'ok': True}

    formatter = JsonLineFormatter(
        component='test-component',
        identity='test-identity',
        context_provider=lambda: {'execution_id': 'exec-context', 'assignment_id': 'assign-context'},
        top_level_extra_fields={'execution_id', 'assignment_id'},
    )

    payload = json.loads(formatter.format(record))

    assert payload['message'] == 'hello json'
    assert payload['component'] == 'test-component'
    assert payload['identity'] == 'test-identity'
    assert payload['execution_id'] == 'exec-record'
    assert payload['assignment_id'] == 'assign-context'
    assert payload['extra']['custom_payload'] == {'ok': True}
