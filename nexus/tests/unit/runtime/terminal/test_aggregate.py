from nexus.runtime.executors.terminal.capture import CommandRecord
from nexus.runtime.executors.terminal.session import TerminalSession


def _rec(cid, batch, *, command, stdout='', stderr='', output='', end=True, exit_code=0):
    return CommandRecord(
        command_id=cid,
        batch=batch,
        command=command,
        exit_code=exit_code if end else None,
        start_time=float(cid),
        end_time=float(cid) + 0.1 if end else None,
        stdout=stdout,
        stderr=stderr,
        output=output,
    )


def test_aggregate_single_record_is_passthrough():
    rec = _rec(1, 1, command='echo hi', stdout='hi\n', output='hi\n')
    info = TerminalSession._aggregate([rec])
    assert info.command_id == 1
    assert info.command == 'echo hi'
    assert info.stdout == 'hi\n'
    assert info.output == 'hi\n'
    assert info.exit_code == 0
    assert info.end_time is not None


def test_aggregate_multi_record():
    recs = [
        _rec(1, 1, command='echo A', stdout='A\n', output='A\n', exit_code=0),
        _rec(2, 1, command='echo B', stdout='B\n', output='B\n', exit_code=0),
        _rec(3, 1, command='echo C', stdout='C\n', output='C\n', exit_code=7),
    ]
    info = TerminalSession._aggregate(recs)
    assert info.command_id == 1                 # first id
    assert info.command == 'echo A\necho B\necho C'
    assert info.stdout == 'A\nB\nC\n'
    assert info.output == 'A\nB\nC\n'
    assert info.exit_code == 7                   # last record's exit code
    assert info.start_time == 1.0                # first
    assert info.end_time == 3.1                  # last


def test_aggregate_in_progress_when_any_unfinished():
    recs = [
        _rec(1, 1, command='echo A', stdout='A\n', output='A\n', exit_code=0),
        _rec(2, 1, command='sleep 9', end=False),
    ]
    info = TerminalSession._aggregate(recs)
    assert info.command_id == 1
    assert info.stdout == 'A\n'        # partial output visible
    assert info.exit_code is None      # not finished
    assert info.end_time is None       # not finished
