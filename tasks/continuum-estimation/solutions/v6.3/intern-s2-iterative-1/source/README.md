# estimate — extrapolation to the limit

Estimates the limiting value `t` as the discretization level `x -> 0` from a
small number (4 or 5) of positive-level observations `y(x)`.

## Model

The observations are assumed to follow a smooth asymptotic expansion

    y(x) = t + c1*x^p + c2*x^q + ...      with p, q > 0, x -> 0

The program does not know the convergence orders `p, q, ...` and recovers them
from the data, then extrapolates to `x = 0`. Five independent estimators are
computed and combined by a trimmed mean so that no single bad model dominates:

1. **Polynomial extrapolation**, degree `n-1` (Vandermonde in increasing powers
   so the constant term is the value at `x = 0`). A degree `n-2` fit is also
   available (`estimate.POLY_N2`) as a non-interpolating second opinion, but on
   the public dev set it consistently drags the aggregate away from the power
   series answer, so it is disabled by default.
2. **Two-term power series** `t + c1*x^p + c2*x^q` on a 160x160 log-spaced grid
   of exponents. The three coefficients per grid slot come from the analytic
   Cramer solution of the 3x3 normal equations, which makes the whole grid
   vectorised and cheap (~3 ms) compared with looping `lstsq` calls.
3. **Local two-point Richardson**: `|dy/dx| ~ x^(p-1)` on a log-log plot, so the
   dominant order is `1 + slope`, feeding a two-point Richardson estimate.
4. **Three-term refinement** of the best power-series slots, gated on the
   two-term residual being decisively smaller than the `n-2` polynomial
   residual — otherwise we would be overfitting.

### Why the `p -> 0` region is excluded from the grid argmin

The exponent grid reaches down to `pmin = 0.015`. For such a small exponent,
`x^p` is nearly constant over the observed `x` range, so the two-term model
collapses onto the constant term and can fit the data exactly (residual
*exactly* zero). The global argmin of the grid residual therefore often lives
in that degenerate region and returns a `t` that is essentially a weighted mean
of the observations rather than the limit.

Consequences that shaped the implementation:

* Exponents below `ND_LO` (1.0) and above `ND_HI` (12) are excluded when
  picking the grid argmin/median/weighted values. The non-degenerate region is
  deliberately narrow: exponents outside it either collapse onto the constant
  term or are too high to be identifiable from 4–5 points.
* A residual band that contains only a few slots is almost always just the
  degenerate slot, so band-derived candidates (and the 3-term gate) are only
  emitted when at least `NBAND_MIN` (2) slots survive.
* The low-exponent slots are still computed and still contribute to the
  residual band when enough of them survive — they are not thrown away.
* The 3-term refinement only fires when the best 2-term residual is below
  `GATE` (0.1) of the genuine degree-`n-2` polynomial residual, and at most the
  best `KC` (3) band slots are refined. Without the gate, the 3-term model
  overfits the few points and drags the aggregate.
* `TRIM` (0.25) of the candidate estimates are dropped at each tail before
  averaging, so a single degenerate slot that slips through cannot dominate.
* When the grid's best residual is essentially zero relative to the range of
  the data, the power-series model is not identifiable: the surviving slots
  have collapsed onto the constant term and their limits point at the data,
  not at `x = 0`. `DEGEN_FLOOR` (1e-12) detects this and suppresses the band
  and 3-term candidates in that case, leaving the polynomial and Richardson
  fallbacks.

## Usage

```sh
./run.sh                                    # default paths
./run.sh INPUT=INPUT.json OUTPUT=OUTPUT.json
```

Default input `/workspace/inputs/input.json`, output
`/workspace/result/output.json`. `INPUT`/`OUTPUT` are optional for local
development; the verifier invokes the no-arg form.

The script is executed as `/bin/sh /workspace/submission/run.sh` with
`/workspace/runtime/bin` first on `PATH`, so the bare `python3` resolves to the
bundled Python 3.11.15 / NumPy 1.26.4 / SciPy 1.12.0.

Input schema:

    {"x": [x1, x2, ...], "y": [y1, y2, ...]}

Output schema (exactly one field):

    {"estimate": <finite number>}

## Files

* `estimate.py`   — the estimator (library and CLI, see `estimate.main`)
* `run.sh`        — driver; reads input, writes output
* `README.md`     — this file

## Timing

The whole grid is vectorised; a single case runs in well under a second
(including interpreter startup) on one logical CPU with BLAS pinned to one
thread via the environment variables set at the top of `estimate.py`. No
threads are spawned.

## References

* Richardson extrapolation / Aitken–Neville polynomial extrapolation
* Separable-variable fitting with a grid over unknown convergence orders
