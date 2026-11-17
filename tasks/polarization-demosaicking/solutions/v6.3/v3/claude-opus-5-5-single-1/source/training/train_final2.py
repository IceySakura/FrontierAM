import numpy as np, sys
from lmmse4 import run
from lmmse2 import PH
K=int(sys.argv[1]); mode=sys.argv[2]
W=run(K,mode)
np.save(f'/workspace/dev/weights_K{K}{mode}.npy', np.stack([W[ph] for ph in PH]).astype(np.float64))
