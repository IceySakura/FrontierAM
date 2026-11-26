# Continuum estimation from positive discretization levels

## Method

The public development cases are reproduced exactly by the smooth convergent
error model

```
y(x) = t + a x^p + b x^q + c x^(q + 3/2),      p, q > 0,
```

with an unknown limit `t`, unknown convergence orders `p,q` and unknown
amplitudes.  Four or five samples cannot identify six parameters pointwise:
with five samples the set of exact fits is a curve in `(p,q)`, with four
samples it is the whole `(p,q)` domain.  The program therefore performs
Bayesian *model averaging* over that manifold instead of trusting one fit.

For fixed `(p,q)` the amplitudes `(t,a,b,c)` are linear and are eliminated
with a batched QR least-squares solve.  The marginal weight of a hypothesis is

```
prior(p,q) * prior(a,b,c) / |det R|^alpha * [delta(residual) for n=5]
```

where `R` is the triangular QR factor, `alpha = 0.5` tempers the volume
factor, and the five-sample delta contributes the coarea factor
`1/|d residual/dq|` along the exact-fit curve.  The prior on the location `t`
is flat, which makes the estimate translation-equivariant; soft-box priors on
`p`, `q-p` and the amplitudes encode the smooth convergent regimes of the
public examples while keeping heavy enough tails for out-of-prior inputs.

The five-sample exact-fit curve is located by scanning the signed orthogonal
residual for sign changes and polishing all brackets with vectorised
bisection; the four-sample case is integrated on a dense `(p, q-p)` grid.
Everything is batched NumPy linear algebra, single-threaded, and takes a few
seconds per case (well inside the 60-second budget, startup included).

### Ensemble

The structured posterior and a model-free two-power nonlinear least-squares
fit (`y = c0 + c1 x^p + c2 x^q`, fitted from a spread of starts) have
complementary biases: the former uses the third-order term, the latter absorbs
it into an effective order.  Their average (0.6 / 0.4) has a substantially
smaller error than either component alone on the evaluation data.  A
scale-invariant formulation of the structured posterior (normalising `x` by
`max(x)` and `y` by its range, with a correspondingly scaled amplitude prior)
is mixed in when the absolute-scale prior is inconsistent with the data or
when the levels exceed unity.

### Robustness

* Arrays are sorted; constant `y` returns that value; non-monotone (adverse
  oscillatory) inputs use the model-free fit.
* Sanity bounds derived from the observed `y` range reject pathological
  extrapolations.
* If any stage fails, a wider-prior posterior, the two-power fit, and a small
  ensemble of Richardson-style extrapolations are tried in that order.
* Finite-value checks, an atomic output write, and a shell-level fallback
  guarantee exactly one finite JSON number is always produced.

## Files

* `solve.py` – complete estimator and I/O.
* `run.sh` – entry point; `/bin/sh /workspace/submission/run.sh` (optionally
  `run.sh INPUT OUTPUT`).  Selects a Python interpreter that has NumPy.
* `README.md` – this file.

## Runtime

Python 3.11 with the bundled offline NumPy 1.26 / SciPy 1.12 runtime.  No
network access, no build step, no cross-invocation state, one thread.
