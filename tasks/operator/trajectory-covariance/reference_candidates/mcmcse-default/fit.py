"""Pass the C-order NPY payload to R (mcmcse 1.5.0) and convert the output back to NPY."""
import subprocess
import sys
from pathlib import Path

import numpy as np

x = np.load(sys.argv[1], allow_pickle=False)
n, d = x.shape
payload = subprocess.run(
    ["Rscript", "--vanilla", str(Path(__file__).with_name("estimate.R")), str(n), str(d)],
    input=x.tobytes(order="C"), stdout=subprocess.PIPE, check=True,
).stdout
estimate = np.frombuffer(payload, dtype="<f8").reshape(d, d)
estimate = (estimate + estimate.T) / 2
np.save(sys.argv[2], np.ascontiguousarray(estimate, dtype="<f8"), allow_pickle=False)
