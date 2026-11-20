# Black-box cubature

`./run.sh` starts the JSON Lines solver. It uses the installed Python, NumPy,
and SciPy. It does not need a build, a network connection, or any evaluator
files. The launcher limits the numerical libraries to one thread.

`common.py` owns the oracle exchange, point cache, and budget. `solver.py`
selects from numerical rules and learned approximations using only queried
values. The structural models are checked on scattered points and, when
useful, near inferred peaks or boundaries. Unresolved cases use adaptive
quadrature or scrambled Sobol estimates of the remaining residual.

Run the public development pilot with:

```sh
python3 /workspace/environment/public-dev/pilot.py /workspace/submission
```
