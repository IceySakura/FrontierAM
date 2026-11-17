# Cubature Submission

Run `./run.sh`. The program reads an initialization JSON line, requests batches
of at most 512 points, and writes a finite integral estimate as a result JSON
line. All requested points, including pilots and native-library callbacks,
share the initialization budget.

The solver uses validated analytic controls, learned ridge approximations,
randomized digital nets, and a budget-wrapped p-adaptive cubature fallback.
Numerical fitting has an internal wall-clock deadline. A failed numerical fit
falls back to estimates computed from the actual sampled values.

Dependencies are the installed Python 3, NumPy, SciPy, and QMCPy packages.
No network or package installation is used.

The provided Steven G. Johnson cubature source is in `vendor/cubature`, with
its license. Its Clenshaw-Curtis tables retain the original numeric constants
through level 12, which exceeds all levels reachable with this point budget.
The included Linux amd64 shared library can be rebuilt locally:

```sh
cc -O2 -fPIC -shared vendor/cubature/pcubature.c -lm -o libpcubature.so
```

`run.sh` performs this build if the shared library is absent.
