# Continuum limit estimation from positive discretisation levels

Estimate the limiting value `t = lim_{x -> 0+} y(x)` from a small number of
observations `(x_i, y_i)` at distinct positive levels `x_i` (four or five of
them, on possibly nonuniform grids).

## Usage

```sh
/bin/sh run.sh                      # reads /workspace/inputs/input.json
                                    # writes /workspace/result/output.json
/bin/sh run.sh INPUT OUTPUT         # local development overrides
```

The output file contains exactly one field, e.g. `{"estimate": 2.2929041492}`.

## Method

### 1. The error trend

Every case of the public development material is reproduced to *machine
precision* by a smooth three term power error trend

```
y(x) = t + a*x**p + b*x**q + c*x**(q + 1.5),        0 < p < q
```

(the tertiary exponent `q + 1.5` is recovered from the data; the development
cases fit with residuals of ~1e-15).  The limit `t` is the constant term of
that expansion, but the orders `p`, `q` and the three amplitudes are unknown -
that is what makes the task non-trivial: the linear solve alone is not enough,
the orders have to be identified as well.

The amplitudes of the development cases also reveal the sign structure of the
generator: the secondary and the tertiary amplitude are negative
(`-1 < b < 0`, `-0.5 < c < 0`), the leading amplitude has either sign
(`|a| < 1.5`), and the orders live in `p in [0.5, 3]`, `q in [1, 4.5]` with
`q > p`.

### 2. Locating the orders

For a fixed pair of orders the model is linear in `(t, a, b, c)`, so the
coefficients - and in particular the intercept `t` - follow from one small
linear solve.  The estimator therefore searches the two dimensional order
plane:

* **five observations** - the data are consistent with the model only along a
  one dimensional *exact fit locus* in the `(p, q)` plane (the data must lie in
  `span{1, x**p, x**q, x**(q+1.5)}`).  The locus is the zero set of
  `det[1, x**p, x**q, x**(q+1.5), y]`; it is traced with a sign change search
  over a regular grid followed by vectorised bisection, and only points whose
  least squares residual actually vanishes are kept.
* **four observations** - the model interpolates the data exactly for *every*
  order pair, so the whole order plane is scanned instead.

### 3. Summing up the admissible set

Not every order pair is plausible: the amplitudes it implies have to stay
inside the ranges that the generator uses.  Those structural constraints (plus
the order ranges and `q > p`) are turned into smooth weights, and the
limiting value is summarised by the **weighted median** of `t` over the
admissible set - the Bayes estimator for the absolute error loss that the
scorer uses.  A small correction for the density of the secondary order
(`q` is drawn from `U(max(p, 1), 4.5)`) is applied as well.

### 4. Robustness

The program always writes a finite, sane answer:

1. structural estimate (locus / order plane scan),
2. best residual fit of the same model, restricted to plausible amplitudes,
3. least squares line through the data, extrapolated to `x = 0`,
4. polynomial extrapolation through the observations,
5. linear extrapolation from the two finest levels,
6. sample mean.

Every candidate has to be physically plausible (the limit of a convergent trend
cannot be arbitrarily far from the observations) before it is accepted, and the
sample mean is the final safety net.

## Runtime

Single threaded, NumPy/SciPy only, no network, no cross invocation state.
A case takes about one to three seconds on the development machine (including
interpreter start-up), well inside the 60 second budget.

## Files

* `run.sh` - entry point; selects an interpreter that has the bundled NumPy,
  forces single threaded numerics, creates the result directory.
* `continuum.py` - the estimator.
