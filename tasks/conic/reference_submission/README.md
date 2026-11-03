# Private reference submission

This package runs the frozen local Clarabel 0.11.1 reference through the submission interface. It is not part of the anonymous public task, and the version is not asserted to be the exact historical paper build.

## Offline preparation and execution

The base environment supplies CPython 3.12, pip, NumPy 2.3.5 and SciPy 1.16.3. Before isolating the build, place target-platform wheels in `dependencies/wheels/` for the three exact pins in `requirements.txt`: Clarabel 0.11.1, cffi 2.0.0 and pycparser 2.23. Linux deployment requires Linux wheels for the selected architecture; a macOS installation cannot substitute for them. Clarabel uses a CPython stable-ABI wheel; cffi must support CPython 3.12. Include the wheel artifacts and their hashes in the evaluator's frozen dependency inventory.

From this directory, run once:

```sh
./build.sh
```

The script installs only those three packages into writable `.deps/`, uses no package index or source build, verifies the base package versions, and solves a tiny 2×2 PSD problem to check the actual native wheel. It records versions, wheel hashes and that smoke result in `build_manifest.json`. There is no fallback to another solver or platform. Offline build success must be established in the target environment; supplying wheel filenames alone does not establish SDP support.

For each evaluation process:

```sh
./run.sh --input /absolute/input.npz --output /absolute/result.npz
```

Only `--input` and `--output` are accepted, in either order, once each. Paths may contain spaces. The output is a candidate NPZ produced during execution, to be checked independently by the evaluator. Build time is separate; the evaluator externally measures the full run process and enforces its resource limits. The solver's internal 1000-second limit does not replace the external 1000-second process limit or the memory watchdog.

## Frozen implementation and configuration

`conicbench/__init__.py`, `model.py` and `baseline.py` are byte-identical copies from the project's frozen local implementation. `source_manifest.json` records their original relative paths and SHA-256 hashes. No input data, checker, test-set mapping or precomputed solution is included.

The wrapper always invokes the copied baseline with `--time-limit 1000 --tol 1e-9`. All cases use one configuration: one thread; `direct_solve_method="auto"`; presolve, equilibration and chordal decomposition enabled; compact chordal decomposition, completed original-row duals and `clique_graph` merging enabled. The five full-accuracy gap/feasibility/infeasibility tolerances are `1e-9`. Other Clarabel 0.11.1 defaults, including reduced tolerances and `tol_ktratio`, remain unchanged. There is no per-instance tuning. The full effective settings and actual linear solver backend are recorded in each candidate's diagnostics.

The code passes the upper triangle of the input symmetric `P` to Clarabel and returns vectors in original input coordinates. Solver statuses are claims: even `Solved` and `AlmostSolved` require external validation. The private reference does not run or replace the independent checker.

The copied implementation documents its relationship to `ClarabelBenchmarks@3679912c6bbd3f64c5c962f9d1c09d524561c412`. This package preserves that implementation; it does not claim paper-protocol equivalence or transfer prior macOS timings into a new Linux score.
