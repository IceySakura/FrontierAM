# Long-run covariance from one trajectory

`./run.sh INPUT.npy OUTPUT.npy` reads one `(M, d)` float64 trajectory and writes the
`(d, d)` estimate `H` of `Sigma = lim n Cov(mean_n)`: symmetric, PSD, `<f8`, C order.

## Dependencies

- Python 3 with NumPy. Tested with Python 3.11 and NumPy 1.26.
- No other packages and no build step, so there is no `build.sh`.
- Runs single-threaded: `run.sh` sets the BLAS thread variables to 1.
- Randomness comes only from `numpy.random.default_rng(SEED)`. `SEED` defaults to 1, so
  results are deterministic.
- All code is original and lives in `lrcov.py`.

## Method

Every step is affine-equivariant (`x -> A x + b` gives `H -> A H A^T`). The program never uses
a model label: it chooses between two general model classes using only the data.

**0. Whitening.** Centre the trajectory and whiten it with the symmetric inverse square root of
its sample covariance. This keeps the numerics well conditioned whatever coordinates the input
arrives in.

**A. Finite-state branch.** Taken only when all of the following hold:

- the trajectory visits few distinct points: `K <= 200` and `M >= 20 K`;
- every point is re-entered many times. This rules out continuous chains whose repeated rows
  come from Metropolis rejections;
- a parametric-bootstrap G test does not reject a first-order Markov chain against a
  second-order one. The test uses 20 chains simulated from the fitted transition matrix and
  rejects at z > 4. Lumped or hidden-state chains are rejected here.

When it is taken:

1. Put a Dirichlet posterior on the rows of the transition matrix, restricted to the observed
   transition support.
2. For each draw `P`, compute the exact long-run covariance:
   `F_c^T (D Z + Z^T D - D) F_c`, where
   - `Z = (I - P + 1 pi^T)^{-1}` is the fundamental matrix,
   - `D = diag(pi)`,
   - `F_c` holds the centred state values.
3. Combine the draws with the Bayes step below.

**B. VAR branch** (everything else):

1. Fit a VAR(p) by least squares. The order `p` is chosen by AIC with `p <= min(40, M/(10 d))`.
2. Draw from the Jeffreys-prior posterior: `Sigma_u ~ inverse Wishart`, then
   `B | Sigma_u ~ matrix normal` around the OLS fit.
3. When `p = 1`, first move the OLS centre by half of the Pope (1990) first-order bias
   correction (persistent chains are otherwise under-estimated). The correction is shrunk if
   needed to keep the fit stationary.
4. Drop non-stationary draws (companion spectral radius at least 0.9995).
5. Map each kept draw to `(I - sum A_j)^{-1} Sigma_u (I - sum A_j)^{-T}`.

**Bayes step (both branches).** Return the `H` that minimises the posterior mean of the
evaluation loss `|| S^{-1/2} (H - S) S^{-1/2} ||_F` over the draws `S_s`.

- It is computed by iteratively reweighted least squares. Each step solves the `d^2 x d^2`
  system `sum_s w_s P_s H P_s = sum_s w_s P_s`, where `P_s = S_s^{-1}` and
  `w_s = 1 / loss_s`.
- The effect is to shrink the estimate exactly where it is uncertain. This matters most for
  slowly mixing short chains, where the plug-in estimate is heavy-tailed. In development tests
  it cut the mean error by up to 40% on such chains, and it never made things worse elsewhere.

**Back-transform.** Map `H` back to the input coordinates, then symmetrise and clip its
eigenvalues at 0.

If anything fails, the program falls back to multivariate batch means with batch size
`floor(sqrt(M))`. If even that fails, it writes the zero matrix.

The code also keeps some switches that are turned off because they did not help in development:
a multi-resolution variant (VAR on m-block means with a Lepski choice of m) and alternative
order rules (`P_RULE`).

Background:

- VAR spectral estimation of long-run covariances: den Haan and Levin (1997), "VARHAC".
- Bias of OLS VAR(1) estimates: Nicholls and Pope (1988) and Pope (1990).
- Normal–inverse-Wishart / Jeffreys posterior for VARs: standard Bayesian VAR results.
- Long-run covariance of a finite Markov chain via the fundamental matrix: Kemeny and Snell.
- Testing Markov order with the G statistic: Anderson and Goodman (1957); here it is calibrated
  by parametric bootstrap.
- Bayes estimation under the loss `tr((S^{-1}H - I)^2)`: Haff (1980) and Olkin and Selliah
  (1977).

## Running

```sh
./run.sh /abs/input.npy /abs/pred.npy
python3 check.py reference.npy /abs/pred.npy   # optional self-check with the dev pack
```

Run time is usually 0.3–3 s for `M <= 80000` and `d <= 6`. Peak memory is a few hundred MB.
