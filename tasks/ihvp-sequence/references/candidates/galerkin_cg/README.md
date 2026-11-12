# Solution-history augmented CG candidate

This portable baseline uses the public `make_solver(d, seed, options)` interface.
It tests whether a sequence is easy because previous approximate solutions span
the new solution. It is a new, research-informed implementation, not a complete
reproduction of any cited solver and not a qualified SOTA reference.

Use the same `solver.py` with these options:

| Variant | Options |
| --- | --- |
| Warm CG | `{"history_size":0,"warm_start":true,"wall_budget":20.0}` |
| History 16 augmented CG | `{"history_size":16,"warm_start":true,"projected":true,"wall_budget":20.0}` |
| History 48 augmented CG | `{"history_size":48,"warm_start":true,"projected":true,"wall_budget":20.0}` |

`maxiter` can impose a fixed iteration cap; otherwise `maxiter_factor*d` is used
(factor 50 by default) and the deadline usually controls the run. The internal
20-second sequence budget starts at the first `solve` call. The external grader
also charges startup/import/protocol time. Actual measured total time, memory,
quality, and repeated-run stability are needed for reference qualification; the
internal cap alone does not establish the required headroom. Remaining internal
time is divided over remaining steps. `rtol`, `residual_refresh` (128), `basis_rtol`
(1e-10), and `coarse_rtol` (1e-12) are configurable but must be frozen before
private evaluation. There is no random sampling or private-data access.

For a basis U from past approximate solutions, the solver refreshes C=H_t U and
E=U^T C using the **current** paid oracle. The initial correction

    x <- x + U E^-1 U^T (b-H_t x)

minimizes the current SPD quadratic over x+span(U). Projected CG then searches in
the H_t-orthogonal complement of U, using P=I-U E^-1 C^T. After a paid true-residual
replacement it repeats the coarse correction before restarting, preserving the
required U^T r=0 condition. A warm point is validated with a current HVP and
compared to zero by its quadratic objective. Nearly dependent history directions
are removed by pivoted QR; unstable coarse directions are removed or the solver
falls back to warm CG. Storage is O(dk), including U and H_t U; no dense d-by-d
matrix is allocated. Phase changes retain only legal past vectors and always
refresh their products; no transport or future operator is assumed.

All history, QR, HVP refresh, small factorization, projections and iterations are
inside the timed worker. A 48-dimensional basis refresh costs 48 HVP vectors each
step; it is not free. `projected=false` is an optional coarse-correction-only CG
ablation. L-BFGS preconditioning is intentionally not part of this candidate.

The method design follows established public work:

- [Carlberg, Forstall and Tuminaro, POD-augmented CG](https://arxiv.org/abs/1512.05820): recycling solution-oriented spaces for changing SPD matrices/RHSs.
- [Gaul, Gutknecht, Liesen and Nabben, deflation/augmentation framework](https://arxiv.org/abs/1206.1506): Galerkin projection and correction equivalence.
- [Gaul and Schlömer, preconditioned self-adjoint recycling](https://arxiv.org/abs/1208.0264): recycled spectral spaces and self-adjoint deflation.
- [KryPy RecyclingCg](https://krypy.readthedocs.io/en/latest/krypy.recycling.html) and [Trilinos Belos RCGSolMgr](https://trilinos.github.io/docs/belos/class_belos_1_1_r_c_g_sol_mgr.html) are research implementations of related methods; their benchmark results are not claimed for this code.

Low-dimensional generator controls do not mathematically imply a solution space
of the same dimension: H(s)^-1 b(s) is generally nonlinear in s. Public screening
of this candidate is an empirical test of that potential weakness. Freeze code
and options after public screening and before any private evaluation.
