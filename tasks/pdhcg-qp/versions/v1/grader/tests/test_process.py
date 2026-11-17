"""Exercise the real UID boundary and supervisor on tiny analytic QPs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

from qpbench.evaluate import evaluate
from qpbench.process import run, worker_pids

UID = 59137
pytestmark = pytest.mark.skipif(os.geteuid() != 0 or sys.platform != "linux",
                                reason="UID isolation requires Linux root")


@pytest.fixture(autouse=True)
def dedicated_uid():
    if worker_pids(UID):
        pytest.skip("test UID already in use")
    yield
    assert not worker_pids(UID), "supervisor left a worker behind"


def submission(path, body):
    path.mkdir()
    entry = path / "run.sh"
    entry.write_text("#!/usr/bin/python3\n" + body)
    entry.chmod(0o755)
    return path


def test_timeout_and_detached_background_are_cleaned():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        path.chmod(0o777)
        report = run(["/usr/bin/python3", "-c", "import time; time.sleep(30)"], path, 0.15, uid=UID)
        assert report["reason"] == "time_limit"
        report = run(["/usr/bin/python3", "-c",
                      "import os,time; child=os.fork(); "
                      "(os.setsid(),time.sleep(30)) if child==0 else time.sleep(0.05)"], path, 2, uid=UID)
        assert report["reason"] == "background_process"


def test_real_grading_input_is_readonly_and_data_remains_private(tmp_path):
    trusted = tmp_path / "secret"
    trusted.write_text("not available to worker")
    # All data below pytest's private directory remains inaccessible to the worker.
    source = submission(tmp_path / "source", f'''
import argparse, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
assert os.getuid() == {UID}
assert not os.access({str(trusted)!r}, os.R_OK)
assert os.path.basename(os.path.dirname(a.input)) == 'current'
assert not Path('previous-run').exists()
Path('previous-run').write_text('must not survive into another solve')
try:
    open(a.input, 'wb')
except PermissionError:
    pass
else:
    raise RuntimeError('input was writable')
np.savez(a.output, x=np.array([1.]), y=np.array([1.]),
         metadata_json=np.asarray(json.dumps({{'schema_version':1,'status':'solved'}})))
''')
    path = tmp_path / "input.npz"
    np.savez(path, q_data=np.array([1.]), q_indices=np.array([0]), q_indptr=np.array([0, 1]), q_shape=np.array([1, 1]),
             a_data=np.array([1.]), a_indices=np.array([0]), a_indptr=np.array([0, 1]), a_shape=np.array([1, 1]),
             c=np.array([-2.]), b=np.array([1.]),
             metadata_json=np.asarray(json.dumps({"schema_version": 1, "objective_offset": 0})))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"id": "one", "family": "analytic", "path": "input.npz",
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    result = evaluate(source, manifest, {"one": 3.0}, uid=UID)
    assert result["valid"] and result["p"] == 1.0, result
    assert len(result["raw_cases"][0]["repeats"]) == 3


def test_submission_link_is_candidate_failure(tmp_path):
    from qpbench.evaluate import copy_submission
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.sh").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="link"):
        copy_submission(source, tmp_path / "copy", UID)


def test_self_contained_reference_passes_a_real_sample(tmp_path):
    import shutil
    from arena.qp_runtime import VERSION, reference_source
    source = reference_source(tmp_path / "reference")
    samples = VERSION / "data/samples"
    row = json.loads((samples / "manifest.json").read_text())["cases"][0]
    shutil.copyfile(samples / row["path"], tmp_path / "input.npz")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{**row, "path": "input.npz"}]}))
    result = evaluate(source, manifest, {row["id"]: 10.0}, uid=UID)
    assert result["valid"] and result["p"] == 1.0, result
