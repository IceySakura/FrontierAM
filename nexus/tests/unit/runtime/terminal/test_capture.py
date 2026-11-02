import json
import os

from nexus.runtime.executors.terminal.capture import CaptureReader, normalize_cr


def _write_cmd(d, cid, batch, *, end=True, exit_code=0, content=None, stdout='', stderr='', output=''):
    meta = {'command_id': cid, 'batch': batch, 'start_time': float(cid)}
    if end:
        meta['end_time'] = float(cid) + 0.1
        meta['exit_code'] = exit_code
    with open(os.path.join(d, f'cmd_{cid}.json'), 'w') as f:
        json.dump(meta, f)
    if content is not None:
        with open(os.path.join(d, f'cmd_content_{cid}'), 'w') as f:
            f.write(content)
    for name, val in (('stdout', stdout), ('stderr', stderr), ('output', output)):
        with open(os.path.join(d, f'{name}_{cid}'), 'w') as f:
            f.write(val)


def test_normalize_cr():
    assert normalize_cr('a\r\nb\rc') == 'a\nb\nc'


async def test_read_status_missing(tmp_path):
    assert await CaptureReader(str(tmp_path)).read_status() is None


async def test_read_status_present(tmp_path):
    (tmp_path / 'status.json').write_text('{"status":"idle","command_id":0}')
    status = await CaptureReader(str(tmp_path)).read_status()
    assert status is not None
    assert status.status == 'idle'
    assert status.command_id == 0


async def test_read_status_invalid_json_returns_none(tmp_path):
    (tmp_path / 'status.json').write_text('{bad json')
    assert await CaptureReader(str(tmp_path)).read_status() is None


async def test_read_command_missing_meta(tmp_path):
    assert await CaptureReader(str(tmp_path)).read_command(1) is None


async def test_read_command_includes_batch(tmp_path):
    d = str(tmp_path)
    _write_cmd(d, 1, 7, content='echo hi', stdout='hi\r\n', output='hi\r\n')
    rec = await CaptureReader(d).read_command(1)
    assert rec is not None
    assert rec.command_id == 1
    assert rec.batch == 7
    assert rec.command == 'echo hi'
    assert rec.exit_code == 0
    assert rec.end_time is not None
    assert rec.stdout == 'hi\n'
    assert rec.output == 'hi\n'


async def test_read_command_invalid_meta_returns_none(tmp_path):
    (tmp_path / 'cmd_1.json').write_text('{bad json')
    assert await CaptureReader(str(tmp_path)).read_command(1) is None


async def test_read_batch_collects_same_batch(tmp_path):
    d = str(tmp_path)
    _write_cmd(d, 1, 1, content='echo A', stdout='A\n', output='A\n')
    _write_cmd(d, 2, 1, content='echo B', stdout='B\n', output='B\n')
    _write_cmd(d, 3, 1, content='echo C', stdout='C\n', output='C\n')
    batch = await CaptureReader(d).read_batch(1)
    assert [r.command_id for r in batch.records] == [1, 2, 3]
    assert batch.all_finished is True


async def test_read_batch_stops_at_batch_change(tmp_path):
    d = str(tmp_path)
    _write_cmd(d, 1, 1, content='echo A', stdout='A\n', output='A\n')
    _write_cmd(d, 2, 1, content='echo B', stdout='B\n', output='B\n')
    _write_cmd(d, 3, 2, content='echo C', stdout='C\n', output='C\n')  # next batch
    batch = await CaptureReader(d).read_batch(1)
    assert [r.command_id for r in batch.records] == [1, 2]
    assert batch.all_finished is True


async def test_read_batch_unfinished(tmp_path):
    d = str(tmp_path)
    _write_cmd(d, 1, 1, content='echo A', stdout='A\n', output='A\n')
    _write_cmd(d, 2, 1, end=False, content='sleep 9')  # still running
    batch = await CaptureReader(d).read_batch(1)
    assert [r.command_id for r in batch.records] == [1, 2]
    assert batch.all_finished is False


async def test_read_batch_missing_first(tmp_path):
    batch = await CaptureReader(str(tmp_path)).read_batch(1)
    assert batch.records == []
    assert batch.all_finished is False


async def test_read_batch_none_batch_returns_single(tmp_path):
    import json as _json
    import os as _os

    d = str(tmp_path)
    # two records with NO batch field (regression scenario)
    for cid in (1, 2):
        with open(_os.path.join(d, f'cmd_{cid}.json'), 'w') as f:
            _json.dump({'command_id': cid, 'start_time': float(cid), 'end_time': float(cid) + 0.1, 'exit_code': 0}, f)
        for name in ('stdout', 'stderr', 'output'):
            with open(_os.path.join(d, f'{name}_{cid}'), 'w') as f:
                f.write('')
    batch = await CaptureReader(d).read_batch(1)
    # batch is None -> must NOT merge id 1 and id 2; return just the first record
    assert [r.command_id for r in batch.records] == [1]
    assert batch.all_finished is True
