"""Curator only: preserve the source ZIP and append a separately named evaluator overlay."""
from pathlib import Path
import argparse, hashlib, json, os, time, zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'tools', 'manifest', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    expected = manifest['source_archive_sha256']
    assert manifest['revision'] == 'balanced-split-20261117'
    assert a.source.resolve() != a.out.resolve()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    partial = a.out.with_suffix(a.out.suffix + '.partial')
    receipt_path = a.out.with_suffix('.receipt.json')
    progress_path = a.out.with_suffix('.progress.json')
    start = time.time()

    def progress(stage, **values):
        progress_path.write_text(json.dumps(dict(stage=stage, elapsed_seconds=time.time()-start, **values), indent=2)+'\n')

    with zipfile.ZipFile(a.source) as original:
        originals = {i.filename: (i.CRC, i.file_size, i.compress_size, i.header_offset) for i in original.infolist()}
        assert len(originals) == len(original.infolist())
        assert not any(n.startswith('balanced-evaluator/') for n in originals)
        for row in manifest['cases']:
            data_id = hashlib.sha256(original.read('data/'+row['case']+'/COMPLETE.json')).hexdigest()
            assert data_id == row['data_id'], row['case']

    progress('copying_source')
    h = hashlib.sha256()
    with a.source.open('rb') as src, partial.open('xb') as dst:
        copied = 0
        for chunk in iter(lambda: src.read(8 << 20), b''):
            h.update(chunk); dst.write(chunk); copied += len(chunk)
            if copied % (1 << 30) < len(chunk):
                progress('copying_source', copied_bytes=copied)
        dst.flush(); os.fsync(dst.fileno())
    assert h.hexdigest() == expected, 'source archive fingerprint mismatch; partial preserved'
    progress('adding_overlay')
    guide = '''# Balanced replay data: start here

The original data/ cases are byte-preserved. Original directory names and meta.json split labels are source identities, not the current split. Use balanced-evaluator/split_manifest.json for the 15-public/30-private assignment.

The original root-level tools and reports are archival. Use the new balanced-evaluator/ directory and its TESTER_PROMPT.md. Never expose this full archive or that directory to the model; copy only its model_materials/ five-file allowlist into an isolated model workspace.

Example (after dependency installation and native extension build):

    cd balanced-evaluator
    python tools/replay.py --data ../data --split-manifest split_manifest.json --split public --out var/balanced-public
    python tools/replay.py --data ../data --split-manifest split_manifest.json --split private --out var/balanced-private
    python tools/calibrate_anchors.py --results var/balanced-public var/balanced-private --split-manifest split_manifest.json --out var/balanced-calibration

The split was balanced using only two fixed CG references, with random choice among accepted assignments. These are existing, previously examined trajectories. This is a rebalanced replay evaluation, not a new independent holdout. The current reference identities, loss, 30-second budget and 256-MiB worker limit remain unchanged. Formal v4 and its scores remain separate.
'''
    with zipfile.ZipFile(partial, 'a', compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr('START_BALANCED_EVALUATION.md', guide)
        with zipfile.ZipFile(a.tools) as tools:
            for item in tools.infolist():
                assert not item.is_dir()
                output.writestr('balanced-evaluator/'+item.filename, tools.read(item))
        output.writestr('balanced-evaluator/split_manifest.json', a.manifest.read_bytes())
    with zipfile.ZipFile(partial) as output:
        members = output.infolist()
        assert len({i.filename for i in members}) == len(members)
        for name, record in originals.items():
            i = output.getinfo(name)
            assert (i.CRC, i.file_size, i.compress_size, i.header_offset) == record
        for i in members:
            if i.filename not in originals:
                output.read(i)  # CRC validation of all added entries.
    progress('hashing_final')
    final_sha = digest(partial)
    partial.rename(a.out)
    receipt = dict(revision=manifest['revision'], file=a.out.name, bytes=a.out.stat().st_size,
                   sha256=final_sha, source_archive_sha256=expected,
                   split_manifest_sha256=digest(a.manifest), tools_zip_sha256=digest(a.tools),
                   preserved_original_members=len(originals), added_members=len(members)-len(originals),
                   original_data_unchanged=True, seconds=time.time()-start)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    progress('complete', **receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
