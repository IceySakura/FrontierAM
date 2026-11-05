# Dataset-v2 generator smoke report

Generated 2026-11-05 with all CPU work pinned through
`nice -n 19 taskset -c 2,3`. This is a reduced-size public format smoke, not a
truth-qualified benchmark release.

## Standard matrix fetch

No SuiteSparse archive could be fetched from this machine:

* `HB/bcsstk17`, `HB/bcsstk18`
* `DIMACS10/fe_4elt2`, `DIMACS10/wing_nodal`

The Heroku endpoint failed TLS negotiation (`SSL: WRONG_VERSION_NUMBER`) and
the TAMU endpoint returned HTTP 403 for all four archives. The required URL
attempts and errors are frozen in `suitesparse_public_provenance.json`.
The declared `fe_4elt2`/`wing_nodal` mutual replacement also could not be
applied because both endpoints failed. No substitute was silently relabeled as
a standard matrix. The smoke therefore uses eight independent public synthetic
mothers; a later network-enabled run can rerun the same fetch script and use
two standard plus six synthetic mothers.

Raw archives, when available, are stored under `raw/`, which ignores everything
except its `.gitignore`.

## Implemented synthetic families

`synth.py` implements:

* 2-D anisotropic 5/9-point variable-coefficient diffusion with contrast and
  nonzero potential;
* 3-D anisotropic 7-point diffusion with mixed coefficient blocks;
* random geometric graph Laplacians with log-uniform edge weights;
* heavy-tailed preferential-attachment/configuration-model Laplacians;
* signed-incidence `B.T W B + diag(v)` operators with both off-diagonal signs;
* log-spaced diagonal spectra under sparse orthogonal 2x2 block rotations.

Public and private realization seeds are in separate sealed-label JSON files.
The private generators and seeds were not executed.

All eight smoke mothers were exactly symmetric. Eight random Rayleigh checks
were positive for every mother. Smallest-eigenvalue Lanczos estimates were
nonnegative when ARPACK converged; it did not converge within the short smoke
budget for one high-contrast diffusion and the deliberately ill-conditioned
log-spectrum operator. Their PSD property follows directly from their
construction.

## Public smoke distribution

* 8 mothers, 48 scored sessions, and 8 analytic controls.
* Six sessions per mother: `p4/p7/p10` crossed with `one_shot/reuse`.
* Scored `n`: 1,000–2,000; median 1,350.
* Scored `nnz`: 4,000–16,842; median 10,823.
* Exact target eta strata: `1e-2, 1, 10, 1e2, 1e3, 1e4, 3e4, 1e5`, one mother
  (six sessions) per stratum.
* Eta broad buckets: 18 low, 12 middle, 18 high sessions.
* Family sessions: mesh diffusion 18, geometric graph 12, network 6,
  signed-incidence 6, diagonal/local-spectrum 6.
* Compressed case files total 6,719,857 bytes.

The 16-RHS sessions use exactly 6 dense Rademacher, 4 localized impulse,
3 graph-smooth, 2 high-frequency, and 1 near-zero category. `b0` and `b1` use
independent draws and include scored zero-coefficient variations.

Every one of the 56 NPZ files loaded successfully through
`tasks/exponential/public/tools/run_local.py::_load_case`. Array names and
dtypes match the pilot format: float64 `data/b0/b1/y_ref/t/rtol/atol`, uint32
`indices`, and uint64 `indptr/shape`.

## Truth status and cost

The eight controls have analytic dense-eigendecomposition truth. Scored
`y_ref` arrays were generated as finite zero placeholders
(`placeholder_zero_unqualified`). `truth.py` has since replaced them in the
public smoke: all 48 are `chebyshev_audited_qualified`. See `truth_plan.md` for
the method, validation and measured cost. The planning estimates below predate
that run.

For the reduced smoke, the Chebyshev planning model estimates 1.90 CPU-minutes
total primary work. The hardest session is `syn_signed_p10_k16`: degree 3,081,
1.296 billion estimated sparse floating operations, or 0.43 CPU-minutes at the
declared 50 million operations/second planning rate. Independent audit is
budgeted at 1–2x primary, rising to 2–3x for eta at least `1e4`.

For the eventual full 104-session dataset, retain the DESIGN budget of
20–60 CPU-hours primary plus 20–80 CPU-hours independent audit; the smoke sizes
are intentionally too small to extrapolate hardware throughput reliably.
`truth_plan.md` specifies the high-accuracy Chebyshev/Clenshaw computation and
independent Lanczos plus rational/scaling-and-Taylor audit. Per-case smoke
estimates are in `cases/public_smoke/truth_costs.json`.
