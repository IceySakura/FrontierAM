"""Compare two isolated reference submissions bitwise with the pre-Harbor implementation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGE = "sha256:f0c4eae19633c71f9a218cca9f70dc4ad30672519e165a14c0979b2ed0b79ffb"
CONTROL = """
import hashlib, json, sys
import numpy as np
mode, method = sys.argv[1:]
sys.path.insert(0, '/reference/full-frame-leic' if mode == 'historic' else '/workspace/submission')
if mode == 'historic':
    from full_frame import reconstruct as impl
    def run(raw):
        return impl(raw, method)
else:
    from solution import reconstruct as run
rng = np.random.default_rng(1350)
images = {'constant': np.full((18,24),73,dtype=np.uint8),
          'tile': np.tile(np.array([[59,47],[71,83]],dtype=np.uint8), (11,13)),
          'thin': rng.integers(0,256,(7,32),dtype=np.uint8),
          'odd': rng.integers(0,256,(25,37),dtype=np.uint8),
          'full': rng.integers(0,256,(1024,1024),dtype=np.uint8)}
result = {}
for name, raw in images.items():
    raw.flags.writeable = False
    pred = run(raw)
    assert pred.shape == (4,*raw.shape) and pred.dtype == np.float64 and np.isfinite(pred).all()
    for c,(i,j) in enumerate([(1,1),(0,1),(0,0),(1,0)]):
        np.testing.assert_allclose(pred[c,i::2,j::2],raw[i::2,j::2],atol=1e-10,rtol=0)
    result[name] = hashlib.sha256(pred.tobytes(order='C')).hexdigest()
print(json.dumps(result, sort_keys=True))
"""


def main():
    sources = json.loads((HERE/"source-manifest.json").read_text())
    for name, item in sources.items():
        for path in (ROOT/item["upstream"], HERE/"bilinear-edge"/name, HERE/"reflect"/name):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path

    historical = ROOT/"results/polarization-demosaicking"
    results = {}
    for method in ("bilinear", "reflect"):
        for mode in ("historic", "submission"):
            files = [(historical/p, "/reference/"+p)
                     for p in ("full-frame-leic/full_frame.py", "probe/methods/liu.py", "probe/metrics.py")
                     ] if mode == "historic" else [(HERE/method, "/workspace/submission")]
            cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--cpus", "1",
                   "--cpuset-cpus", "0", "--memory", "8g", "--memory-swap", "8g",
                   "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_NUM_THREADS=1"]
            for path, dest in files:
                cmd.extend(["-v",f"{path}:{dest}:ro"])
            cmd.extend(["--entrypoint","python",IMAGE,"-I","-B","-c",CONTROL,mode,method])
            output = json.loads(subprocess.check_output(cmd))
            if mode == "historic":
                results[method] = output
            else:
                assert output == results[method], method
    return dict(status="passed", synthetic_only=True, image=IMAGE, methods=results,
                independent_containers=4, matching_output_pairs=10, output_dtype="float64")


if __name__ == "__main__":
    result = main()
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+"\n")
    print("10 paired synthetic predictions identical across isolated historical and submission processes")
