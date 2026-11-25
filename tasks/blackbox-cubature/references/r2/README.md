# blackbox-cubature v3 references (r2)

Both speak the v3 protocol (`versions/v3/workspace/problem.md`) through `cubature_protocol.py` and run in the task
image (numpy 2.1.3, SciPy 1.17.1); `run.sh` is the entry point.

| Role | Directory | Method |
| --- | --- | --- |
| 50 points (strong) | `gpca_rqmc/` | GPCA-RQMC (Xiao and Wang 2019): finite-difference gradients, then one scrambled Sobol' net in the eigenbasis of \(E[\nabla f\nabla f^\top]\); a port of `versions/v3/baselines/methods.py:as_rqmc`, the method that measured strongest among the literature methods once every query counts (`analysis/DUAL_PLAN.md` section 9.1) |
| 0 points (weak) | `mc/` | plain Monte Carlo on the whole budget, as in v2 |

Both draw all randomness from the instance seed of the init message, so a rerun reproduces their estimates exactly.
They are frozen for v3 and are not tuned on the formal set.
