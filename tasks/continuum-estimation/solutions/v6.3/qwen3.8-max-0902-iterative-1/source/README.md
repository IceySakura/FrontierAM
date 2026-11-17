# Continuum estimation from positive discretization levels

Estimates `t = lim_{x->0} y(x)` from the four or five positive-level
observations in `/workspace/inputs/input.json` and writes
`{"estimate": t}` to `/workspace/result/output.json`.

## Run

```sh
/bin/sh /workspace/submission/run.sh                    # graded paths
/bin/sh /workspace/submission/run.sh INPUT OUTPUT       # local development
```

No build step, no network access, single thread, NumPy only (SciPy is not
imported). Measured wall clock is ~2 s per case including interpreter start-up,
against a 60 s limit (up to ~3 s when the development core is shared).

## Method

### 1. The simulator form is recoverable from the public pairs

`/workspace/public/dev` ships eight input/truth pairs whose truth files carry
`family`, `p` and `q`. That makes it possible to *test* candidate functional
forms instead of guessing them. `dev/model_identification.md` records the
derivation; the summary is that for every `smooth-primary` example

```
y(x) = t + a*x**p + b*x**q + c*x**(q + 3/2)            0 < p < q
```

reproduces the supplied `y` to within 4e-15 absolute once `p`, `q` and `t` are
taken from the truth file. The `q + 3/2` offset was recovered to nine
significant figures independently on all seven of those examples, and the fit
residual at machine precision rules out any further term of comparable size.
The one `oscillatory-audit` example does not follow this form (residual 1.2e-2);
it is handled by the adverse-regime branch described in section 4.

### 2. The model is exact but not identified

The model has six unknowns and there are only four or five observations, of
which four (`t,a,b,c`) enter linearly once `(p,q)` are fixed. So:

* **n = 4** — the 4x4 linear system is square: *every* `(p,q)` interpolates the
  data exactly. `t` is a smooth function `t(p,q)` and the answer is decided
  entirely by which `(p,q)` are plausible a priori.
* **n = 5** — one scalar condition survives,
  `det[1, x**p, x**q, x**(q+3/2), y] = 0`,
  whose solution set is a curve in the `(p,q)` plane along which `t` drifts.
  On dev case 003, for instance, that curve carries `t` from -2.48 to -2.90
  (truth -2.8650) as `p` runs from 0.06 to 0.82.

No amount of numerical care removes this; it has to be closed with a prior.

### 3. Bayesian estimate over the exactly fitting set

`solver.py` evaluates the posterior over parameter values that fit the data
exactly and reports its median:

1. **Candidate generation.** For `n = 5` the determinant above is sampled on a
   dense `p x q` lattice (340 x 2200); every sign change is refined by
   vectorised bisection to ~1e-13, which traces the entire exact-fit manifold
   including every branch. For `n = 4` the `(p,q)` plane is covered by a
   300 x 340 lattice instead, since all of it fits exactly.
2. **Weighting.** Each candidate `(p,q)` gets `theta(p,q)` from a batched QR
   least-squares solve and is weighted by a prior of smooth
   (logistic-tapered) boxes on `p`, `q`, `q - p`, `|t|`, `a`, `b`, `c`, times a
   two-component likelihood (sharp, plus a wide floor that keeps the answer
   sane if a case is only approximately of the assumed form). Box ranges come
   from the public examples, widened by roughly 30-50%; the boxes are
   deliberately asymmetric because `b` is negative in 7/7 public draws and `c`
   is at most +0.05.
3. **Local refinement.** Two zoom rounds re-trace the manifold on a fine
   lattice around the best candidate found globally, which removes the grid
   quantisation from the answer. The refined candidates are then *pooled* with
   the global ones and a single posterior is evaluated over the union — the
   separate lattices have different densities, so their total masses are not
   comparable, but their pointwise log-weights are.
4. **Summary — the posterior median.** The scoring loss
   `|z - t| / max(1, |t|)` is an L1 loss, so the median is its Bayes estimator.
   The mode was measured against it on both data sets: the two are a wash on
   135 synthetic draws (1.309e-3 vs 1.304e-3) but the median is better on the
   seven real `smooth-primary` dev pairs (2.34e-4 vs 2.56e-4), and it is the
   stable choice where the posterior is flat, so it is what ships. Weighted
   means and tempered medians were also measured and were consistently worse.

   The theoretically motivated `1/|dF/dq|` manifold-measure factor is computed
   but switched off (`use_slope=False`). It is the correct delta-function
   Jacobian for turning the 2-d `(p,q)` integral into an integral along the
   exact-fit curve, but it is a poor estimator ingredient here: it sharpened
   two dev cases and degraded a third by more, and on synthetic draws it was
   actively harmful (5.76e-3 vs 1.31e-3, worst case 0.28 vs 0.03), because the
   factor diverges at turning points of the manifold and so over-weights
   isolated spikes.

### 4. Adverse-regime branch

For data that is *not* of the identified form, the exact-fit manifold still
exists but only at absurd coefficients — dev case 004 needs `b = -15.8`,
`c = +32.9` to fit. The best prior log-weight over the candidate set is a clean
detector of this: -120.2 for case 004 against [-0.94, -0.12] for the seven
`smooth-primary` cases, and [-1.99, -0.12] over 210 synthetic draws from
deliberately widened parameter ranges. When that statistic falls below -6 the
three-term answer is discarded and replaced by a multi-start grid-zoom
least-squares fit of the smooth two-power model `y = t + a*x**p + b*x**q`,
which averages an oscillatory component instead of chasing it. On case 004
this cuts the loss from 6.1e-2 to 1.7e-3; if the detector ever misfired on a
conforming case the two-power fit still lands around 1e-3, so the downside is
bounded.

### 5. Safety

If the Bayesian path yields nothing usable, `estimate()` falls back to
degree-`n-1` polynomial extrapolation to `x = 0`, then to a two-point
Richardson scan over assumed orders, then to `y` at the smallest level. The
result is range-checked against the observed span, coerced to a finite float,
written atomically, and the process always exits 0. `main()` wraps parsing,
estimation and writing so that a malformed input still produces a valid
`{"estimate": ...}` file.

## Files

| file | purpose |
| --- | --- |
| `run.sh` | entry point; picks an interpreter that has NumPy |
| `solver.py` | the whole estimator (one frozen program, no state) |
| `dev/model_identification.md` | derivation of the model form and the `q + 3/2` offset |
| `dev/selftest.py` | reproduces the identification numbers and scores the dev pairs |

## Measured accuracy

Mean `|z - t| / max(1, |t|)`:

| set | mean loss |
| --- | --- |
| 7 `smooth-primary` dev pairs | 2.34e-4 |
| all 8 dev pairs (includes the audit case) | 4.11e-4 |
| 45 synthetic draws matched to the dev parameter ranges | 9.70e-4 |
| 45 synthetic draws from deliberately widened ranges | 2.42e-3 |
| 45 synthetic draws from deliberately narrowed ranges | 5.35e-4 |
| all 135 synthetic draws (median 1.15e-4, worst 3.1e-2) | 1.31e-3 |

Per-case loss on the eight public pairs:

| case | n | family | loss |
| --- | --- | --- | --- |
| 000 | 5 | smooth-primary | 1.34e-4 |
| 001 | 5 | smooth-primary | 1.32e-5 |
| 002 | 4 | smooth-primary | 2.47e-5 |
| 003 | 5 | smooth-primary | 1.34e-3 |
| 004 | 5 | oscillatory-audit | 1.65e-3 |
| 005 | 4 | smooth-primary | 1.12e-5 |
| 006 | 4 | smooth-primary | 7.90e-5 |
| 007 | 5 | smooth-primary | 3.22e-5 |

Case 003 dominates what is left: its exact-fit manifold carries `t` over a
0.4-wide interval on which the prior varies by less than 0.1 nats, so the truth
and the estimate are statistically indistinguishable under any prior this size
of sample can support.

For comparison on the eight dev pairs: two-point Richardson extrapolation at
an assumed first order averages 2.2e-2, and degree-`n-1` polynomial
extrapolation to `x = 0` averages 4.5e-3.
