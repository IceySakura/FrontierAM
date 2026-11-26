import json

from arena.prune import MIN_BYTES, prune_candidate


def test_prune_removes_only_bulky_non_text_files(tmp_path):
    candidate = tmp_path / 'grade/candidate'
    (candidate / 'scene/worker').mkdir(parents=True)
    (candidate / 'rounds/b4/0/handoff').mkdir(parents=True)
    big = b'x' * MIN_BYTES
    (candidate / 'scene/raw.npy').write_bytes(big)
    (candidate / 'scene/worker/pred.npy').write_bytes(big + b'y')
    (candidate / 'rounds/b4/0/handoff/output.bin').write_bytes(big)
    (candidate / 'rounds/b4/0/run.json').write_bytes(b'{"ok": true}' + b' ' * MIN_BYTES)
    (candidate / 'summary.json').write_text('{"status": "ok"}')
    (candidate / 'small.bin').write_bytes(b'z' * (MIN_BYTES - 1))
    (candidate / 'table.h').write_bytes(big)
    (candidate / 'link.npy').symlink_to(candidate / 'scene/raw.npy')

    result = prune_candidate(candidate, 'test')

    assert result == {'files': 3, 'bytes': 3 * MIN_BYTES + 1}
    assert not (candidate / 'scene/raw.npy').exists()
    assert not (candidate / 'scene/worker/pred.npy').exists()
    assert not (candidate / 'rounds/b4/0/handoff/output.bin').exists()
    for kept in ('rounds/b4/0/run.json', 'summary.json', 'small.bin', 'table.h'):
        assert (candidate / kept).exists()
    record = json.loads((candidate / 'PRUNED.json').read_text())
    assert record['files'] == 3 and record['reason'] == 'test'
    assert sorted(r['path'] for r in record['removed']) == ['rounds/b4/0/handoff/output.bin', 'scene/raw.npy',
                                                          'scene/worker/pred.npy']


def test_prune_without_candidate_directory(tmp_path):
    assert prune_candidate(tmp_path / 'missing', 'test') == {'files': 0, 'bytes': 0}
    assert not (tmp_path / 'missing').exists()
