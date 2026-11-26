# Conic solver

Offline solver for the candidate-v1 NPZ conic QP interface.

## Run

```sh
./build.sh
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`build.sh` compiles `src/solver.cpp` with the system C++ compiler and links the
OpenBLAS/LAPACK shared library copied into `lib/` from the preinstalled SciPy
numerical stack. zlib is used from the base system. The build also warms the
preinstalled Python/NumPy/HiGHS imports used by the fast path. No network access
or package installation is required.

## Method

The entry point is `wrapper.py`. Problems containing only zero/nonnegative
cones, and problems whose only second-order cones have dimension two, are sent
to the bundled HiGHS LP/QP solver. Dimension-two second-order cones are
transformed exactly into paired nonnegative rows. Farkas certificates for
linear/quadratic infeasibility are obtained from auxiliary HiGHS LPs.

Remaining conic problems use a standalone C++17 homogeneous primal-dual
interior-point method. Zero cones are eliminated by a dense equality presolve
when practical, otherwise represented as paired nonnegative rows. Nonnegative
and second-order blocks use Nesterov-Todd scaling; semidefinite blocks use a
matrix geometric-mean scaling; exponential blocks use logarithmic-barrier
Hessian scaling. Dense BLAS/LAPACK factorizations are used for small reduced
normal-equation systems and a matrix-free CG path for larger sparse systems.
Results are checked in original coordinates before a status is emitted.

References: HiGHS (LP/QP and Farkas certificates); Nesterov-Todd scaling and
Mehrotra predictor-corrector interior-point methods; homogeneous self-dual
embedding for conic feasibility certificates.
