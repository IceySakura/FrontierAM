# Host-only hard-sequence construction

These files are curator/evaluator tools. Do not include them, teacher factors,
reference vectors or local seed plans in candidate sample mounts.

For each step, define

\[
H_t=Q_tD_t^{1/2}B_t^2D_t^{1/2}Q_t^\top,
\quad b_t=Q_tD_t^{1/2}B_ty_t,
\quad x_t^*=Q_tD_t^{-1/2}B_t^{-1}y_t.
\]

`B_t` contains full-rank, randomly paired 2x2 SPD blocks with eigenvalues in
`[0.8,1.2]`. `D_t` has a continuous broad or many-band spectrum and bounded,
full-dimensional diagonal drift. The known spectral envelope is derived from
these bounds. No rank-12 approximation or six-dimensional forcing is used.

Two fixed orthogonal architectures are represented: a signed/permuted random
butterfly, and compositions of two or three independent signed/permuted DCTs.
Two additional time-varying rotation layers change every coordinate. Their
pairings stay inside nearby log-spectral buckets, so small angle changes do not
mix the most extreme diagonal scales and artificially destroy warm starts.

The whitened right-hand process has independent full-dimensional AR innovations,
with correlation and mean-reversion/phase-change behavior varying by family.
The construction is exogenous, not a trajectory of a genuine outer optimizer.
It guarantees `x_star.T H x_star = ||y||^2`, balancing target energy rather than
using an ambient isotropic solution that concentrates energy in large eigenvalues.

`HardCase(path)` exposes host-side `rhs(t)`, `reference(t)`, `phase(t)`, `meta`,
`dimension`, `steps`, `mu`, `L`, `hvp(t,v)` and `score(t,x)`. HVPs accept a vector
or a block with at most 64 columns; the public protocol must charge/control each
column independently. `score` evaluates the relative energy loss using
`||F(x-reference)||^2/||F reference||^2`, where `F=B sqrt(D) Q.T`, and also
reports Euclidean/residual diagnostics. This preserves exact zero/one loss for
the stored reference/zero prediction. Invalid shapes and nonfinite vectors raise
`ValueError`; arithmetic overflow raises `FloatingPointError`. A nonpositive or
nonfinite teacher energy is an `ArithmeticError`, not a candidate failure.

Run the small dense reference check before generating public feedback data:

```sh
python -B tasks/ihvp-sequence/versions/v3/curator/generate.py --self-test
python -B tasks/ihvp-sequence/versions/v3/curator/generate.py \
  --split public --out var/raw/ihvp-hard-v3/public
```

The fixed five-family/three-level plan is sealed before any solver screening;
its actual random seeds remain in ignored storage. Initial condition scales are
`1e5`, `1e6`, `1e7`. Public pilot measurements may motivate an explicitly recorded
global design revision before final freezing. Do not drop individual cases based
on which method wins. Preserve the previous pilot and its plan on revision.

Private generation requires `--split private --design-freeze <ignored-file>`.
The receipt must have `status: FROZEN` and the current `design_sha256`. Generate
private cases only after the public design is qualified and frozen by the
maintainer. Actual data, answers and seeds must remain outside tracked paths.

Neither construction nor reference validation proves that generic history methods
cannot solve the cases. Public qualification must measure true current-system
Galerkin histories of sizes 16/48, fixed-metric rank, actual warm-start energy,
and baseline time/memory. A known structured family may legitimately be inferred
from oracle responses. Do not claim mathematical unlearnability or universal SOTA.
