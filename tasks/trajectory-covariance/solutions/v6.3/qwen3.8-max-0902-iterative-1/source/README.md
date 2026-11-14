# Long-run covariance from a single MCMC trajectory

Estimates `H`, the `d x d` long-run covariance

```
Sigma = lim_{n->inf} n * Cov( mean_{t<=n} x_t )
```

from one trajectory of shape `(M, d)`, so that `H/M` estimates the covariance of
the sample mean and `sqrt(H_jj / M)` the standard error of component `j`.

## Run

```sh
./run.sh /absolute/input/input.npy /absolute/output/pred.npy
```

## Build

`build.sh` takes no arguments; it only byte-compiles the Python sources. There is
nothing to compile — the estimator is pure Python 3 + NumPy.

## Dependencies

* Python 3 (developed and tested on 3.11)
* NumPy (tested on 1.26)

No SciPy, no compiled extension, no network access, no third-party code.
Single-threaded; BLAS threading is pinned to 1 at import. Runs in well under
0.5 s for `M = 80000`, inside the 5 s budget including interpreter start-up.

## Method

`Sigma = sum_k Gamma(k)`. Truncating that sum with a kernel needs a bandwidth and
pays a bias/variance price, and for the strongly autocorrelated / oscillatory
dynamics here a flat or Bartlett truncation is either badly biased or wildly
variable. Instead the trajectory is modelled as a vector autoregression and the
model's autocovariance sequence is summed analytically to infinity:

```
y_t = sum_{i=1..p} A_i y_{t-i} + e_t,   Cov(e_t) = Om
H(p) = B^{-1} Om B^{-T},                B = I - sum_i A_i
```

* `G[k] = (1/M) sum_t y_t y_{t+k}^T` for `k = 0..K`, `K = min(0.3 M, 1200)`, all
  lags from one zero-padded FFT (`O(M log M)`).
* `A_i`, `Om` from the multivariate Yule-Walker equations. An empirical
  autocovariance sequence is positive definite, so the Yule-Walker fit is always
  stationary: `B` is invertible and `H(p)` is symmetric PSD by construction.
* The order `p` minimises `M log|Om_p| + 1.8 d^2 p` over `p = 0 .. pmax`
  (`pmax = min(48, 0.05 M / d)`). The penalty constant was swept over
  0.5 ... log(M): an AIC-scale value (~2) measured best on linear-dynamics,
  finite-state-chain and MCMC-style trajectories, while BIC under-fits chains
  whose autocovariance is a sum of many exponentials - exactly the
  finite-state / data-augmentation-Gibbs case. `K`, `KMAX` and `PMAX_FRAC` were
  also swept and the result is insensitive to all of them (the selected order
  never approaches the caps).
* The data are whitened by the lag-0 covariance for conditioning only and the
  result is mapped back, so the estimator is equivariant under any invertible
  affine change of coordinates (the scoring loss is invariant under the same
  change).
* Safety net: in the whitened frame `H` is a matrix of integrated
  autocorrelation times, so eigenvalues are clipped at `0.2 M`; no direction can
  be estimated beyond a fraction of the trajectory length.

Output is written as one C-order, little-endian float64 `(d, d)` array, exactly
symmetric and positive semi-definite. If anything at all goes wrong the program
falls back to the lag-0 sample covariance (and then to zeros) and still exits 0.

## Alternatives benchmarked and rejected

All comparisons were run on trajectories simulated to match each evaluation
category (VAR(1)/VAR(2) with positive and mixed-sign coefficients, oscillatory
VARs with complex eigenvalues, Metropolis-Hastings on finite state spaces, and
random-walk-Metropolis / leapfrog-HMC / Polya-Gamma-Gibbs chains on a Bayesian
regression posterior), scored with the same whitened Frobenius error and with
references computed analytically or from many parallel chains.

* lag-0 sample covariance (ignores autocorrelation): far worse, and catastrophic
  on mixed-sign / oscillatory dynamics.
* Kernel HAC - flat, Bartlett, Parzen, Tukey, quadratic-spectral - with
  threshold, `M^{1/3}`, or Newey-West plug-in bandwidths: 2-20x worse. Flat
  truncation is the specific failure mode on oscillatory dynamics, where it can
  do worse than returning the zero matrix.
* Blending the VAR estimate with a kernel estimate: worse than the VAR alone at
  every blend weight and bandwidth rule tried.
* Selecting among kernel/VAR candidates by minimising the apparent error
  `||H - I||_F^2` in a whitened frame (an unbiased risk estimate, since the cross
  term has zero mean): the frame has to come from a pilot estimate, and the
  pilot is itself a candidate, so the criterion collapses onto the pilot.
  Cross-fitting a frame from one half of the trajectory against the other half
  would break this but was not adopted - the measured upside was small and
  confined to the shortest trajectories.
* Mean-subtraction bias correction (`G[k] += H/M`, and the `1/(M-k)`
  normalisation): no measurable gain.
* Forward/backward (time-reversed) VAR averaging: a no-op - the Yule-Walker fit
  is exactly time-reversal equivariant and the long-run covariance is invariant.
* Model averaging over orders with criterion-derived weights: no measurable gain.

## Sources

Original code written for this task; no third-party code is bundled. Standard
references for the method: Yule-Walker estimation of VAR models and the identity
`Sigma = B^{-1} Om B^{-T}` (e.g. Hamilton, *Time Series Analysis*, ch. 10);
kernel HAC and automatic bandwidth selection as the benchmarked alternative
(Andrews 1991, Newey-West 1994, Geyer 1992).
