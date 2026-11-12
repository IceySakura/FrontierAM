# Long-run covariance from one trajectory

Entry point: `./run.sh /abs/input.npy /abs/output.npy` (no build step).
Dependencies: Python 3 and NumPy only (both preinstalled; no other packages, no network).
All source code (`lrc.py`, `run.sh`) was written for this task. Randomness is seeded with
`SEED` (default 1), so results are deterministic. Typical run time is 0.2-1.3 s per input.

## Method (`lrc.py`)
The input is centred; all steps are affine equivariant, so a per-trajectory linear change of
coordinates and shift do not change the error.

1. **VAR(p) model.** AIC is computed for orders 0..pmax (pmax <= 40) on a common sample. Every
   order with Akaike weight above 1% is fitted by least squares and its normal / inverse-Wishart
   posterior (flat prior) is sampled, the number of draws proportional to the weight (400 total,
   explosive draws discarded). Each draw gives S_j = A(1)^{-1} Sigma_e A(1)^{-T}; the draws of all
   orders are pooled with their Akaike weights (Bayesian averaging over the order).
2. **Empirical-Bayes isotropy shrinkage.** In coordinates whitened by the sample covariance,
   the log-eigenvalues of every draw are shrunk toward their mean. The shrinkage factor
   tau^2/(tau^2+s^2) compares the eigenvalue dispersion of the point estimate with the posterior
   noise s^2 of those directions (noise inflated by d/2 for eigenvalue spreading).
3. **Bayes decision.** The returned matrix minimises the posterior mean of the evaluation loss
   ||S_j^{-1/2}(H - S_j)S_j^{-1/2}||_F over the draws (iteratively reweighted least squares on a
   d^2 x d^2 linear system, solved in whitened coordinates). This shrinks noisy, heavy-tailed
   directions (slow modes) instead of reporting the unstable plug-in value.
4. **Rejection chains.** If more than 30% of consecutive rows are exact repeats (Metropolis
   rejections), a scalar-coefficient AR(q) model x_t = sum_i a_i x_{t-i} + e_t (shared AR
   polynomial, full innovation covariance, q by AIC, computed from windowed lag cross-products)
   is also fitted; unless the VAR is better by a log-likelihood margin of 30 + 0.005 M, its
   Bayes-decision estimate (posterior over a and Sigma_e) gets weight 0.7 and the VAR estimate 0.3.
5. **Finite-valued chains.** If the trajectory has K <= 2000 distinct rows, K <= M/4, and at
   least 3K changes of value, it is also treated as a finite Markov chain: transition counts give
   the plug-in transition matrix (circular closure keeps it irreducible) and the exact long-run
   covariance follows from the fundamental matrix (I - P + 1 pi^T)^{-1}. The result is the equal
   average of this Markov estimate, the VAR estimate and the scalar-AR estimate (or the VAR
   estimate again when the scalar model is rejected by the margin rule).
6. The output is symmetrised and projected onto the PSD cone. If the posterior/decision step
   fails, the plain least-squares VAR estimate is used, and as a last resort the sample covariance.
