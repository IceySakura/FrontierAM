"""The dumped trajectory keeps the reason a task reported something else in place of."""

from types import SimpleNamespace

from nexus.trace.processors.hytuner import HytunerProcessor


class _Span:
    """A non-LLM span, which is what carries task-reported attributes."""

    name = 'task_phase'

    def __init__(self, attributes: dict) -> None:
        self.attributes = attributes


def _dump(attributes: dict) -> dict:
    processor = HytunerProcessor(file_path=None, trajectory_id='7_abc')
    processor.on_end(_Span(attributes), SimpleNamespace())
    return processor.get_dump_result()


def test_dump_records_the_reason_that_was_replaced() -> None:
    dump = _dump({
        'hytuner.termination_reason': 'ENV_DONE',
        'hytuner.cleared_termination_reason': 'TASK_TIMEOUT',
    })

    # Both halves are needed: the first is what the trainer acts on, the second is the only
    # thing separating a rescued rollout from an ordinary success.
    assert dump['termination_reason'] == 'ENV_DONE'
    assert dump['cleared_termination_reason'] == 'TASK_TIMEOUT'


def test_dump_omits_the_field_when_nothing_was_replaced() -> None:
    dump = _dump({'hytuner.termination_reason': 'ENV_DONE'})

    assert dump['termination_reason'] == 'ENV_DONE'
    # Absent rather than None, so counting occurrences is a key check.
    assert 'cleared_termination_reason' not in dump


def test_an_unreplaced_reason_is_dumped_unchanged() -> None:
    dump = _dump({'hytuner.termination_reason': 'TASK_TIMEOUT'})

    assert dump['termination_reason'] == 'TASK_TIMEOUT'
    assert 'cleared_termination_reason' not in dump
