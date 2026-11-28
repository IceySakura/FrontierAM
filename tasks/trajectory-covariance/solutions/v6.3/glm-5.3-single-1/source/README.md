# Long-run covariance estimator from a single MCMC trajectory

`./run.sh /absolute/path/input.npy /absolute/path/output.npy`

reads a `(M, d)` float64 trajectory (rows = time, columns = components) and writes
a `(d, d)` finite, symmetric, positive semidefinite estimate `H` of the long-run
covariance

```
Sigma = lim_{n->inf} n Cov(bar X_n) = Gamma_0 + sum_{k>=1} (Gamma_k + Gamma_k^T)
```

so that `H / M` estimates the covariance of the mean vector.

## Method (single generic algorithm, no per-case tuning)

1. **Autocovariances by FFT.** The trajectory is centred by its sample mean and
   all cross-autocovariances `Gamma_k = (1/M) sum_t Y_t Y_{t+k}^T` for
   `k = 0 .. min(M-1, M/2)` are obtained at once from the FFT of the padded
   series (`O(d^2 M log M)`, ~0.6 s for `M = 80000, d = 6`, well inside the 5 s
   budget including interpreter start-up).

2. **Scale-free whitening.** The input coordinates may have been pushed through
   an arbitrary invertible affine map, so every statistical decision is made in
   the space whitened by the lag-zero covariance `T = Gamma_0^{-1/2}` (eigenvalue
   floored at `1e-10 * lambda_max` for numerical safety). In that space all
   statistics below are invariant under any invertible affine change of the
   input, and the final estimate is mapped back exactly with
   `H = Gamma_0^{1/2} S Gamma_0^{1/2}`, so the estimator is affine-equivariant
   by construction.

3. **Noise level.** The lag-`k` whitened autocovariance trace `a_k` is pure noise
   beyond the correlation length. Its standard deviation is estimated robustly
   from second differences (`E[(a_{k-1}-2a_k+a_{k+1})^2] = 6 sigma^2`), which is
   insensitive to a slowly decaying signal.

4. **Lag window.** The truncation lag `K` is chosen from the risk of the
   truncated autocovariance sum, `bias(K)^2 + variance(K)`: for a decay rate
   `tau` the relative variance grows like `K / M` and the relative truncation
   bias like `exp(-K/tau)`, whose optimum is `K* ~ (tau/2) log(M/tau)`. `tau`
   is obtained self-consistently from the cumulative whitened autocovariance
   trace (the average integrated autocorrelation inside the current window),
   iterated to a fixed point; this handles processes whose autocovariance is a
   mixture of fast and slow modes, where a plain decay fit on `a_k` only sees
   the fast modes. `K` is additionally capped at twice the last lag whose raw
   signal still clears the noise band (short-memory processes such as MA(1)
   must not accumulate pure noise lags), and a significance guard sends `K` to
   zero when neither any single early lag nor the accumulated autocovariance
   stands clear of the noise, which keeps the estimator at the
   sample-covariance optimum for effectively independent draws.

5. **Assemble.** The whitened autocovariances are summed up to `K` with a
   half-cosine taper on the last half of the window, mapped back, symmetrised
   and projected on the PSD cone (negative eigenvalues clipped at zero), which
   guarantees the acceptance conditions (symmetry `1e-10`, PSD `1e-10`).

The estimator is purely a function of the delivered trajectory: no table, no
model identification, no extra sampling, no state carried across calls.

## Dependencies

Python 3 with NumPy (both preinstalled; no network, no build step, hence no
`build.sh`). Randomness is not used; `SEED=1` is exported by `run.sh` for
completeness. Single-threaded: `OPENBLAS/OMP/MKL/NUMEXPR` thread counts are
pinned to 1 inside `run.sh`.

## Files

- `run.sh` — entry point, `./run.sh INPUT OUTPUT`
- `estimator.py` — the complete implementation (also usable as a module:
  `estimate(X)`)

Self-check with the provided development samples:

```
./run.sh "$DEV/01/input.npy" "$PWD/pred.npy"
python3 "$DEV/check.py" "$DEV/01/reference.npy" "$PWD/pred.npy"
```
