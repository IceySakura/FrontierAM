# Literature-informed recycling solvers (research candidates)

This directory contains independent, bounded-window HVP adaptations. It is not
an official reference, does not replace r1, and is not a verbatim reproduction
of the complete authors' implementations. No frozen agent source was used to
implement these methods. The frozen agent programs are comparison subjects only.

- `ritz`: augmented CG with current-operator coarse refresh and Rayleigh-Ritz
  selection of small Ritz values. The trial space combines the retained basis
  and a bounded window of previous CG search directions. Cached H-images are
  transported through two-pass MGS. This is a truncated recycling variant,
  not a claim to reproduce every restart/locking feature of incremental eigCG.
- `pod`: energy-metric POD compression of retained weighted modes, the current
  approximate solution, and CG corrections weighted by their known coefficients.
  Previous compressed modes are exponentially forgotten (0.8 by default).
  The next system refreshes the retained modes and uses augmented CG. This uses
  the energy/solution-oriented POD principle; it does not reproduce the paper's
  large-space three-stage solver, output-oriented metric, or RBF weighting.
- `scaled`: PCG with a fixed SPD spectral inverse during each solve,
  B=I+U(theta diag(lambda)^(-1)-I)U^T. U is Euclidean orthonormal and comes from
  actual HVPs of the previous system. Small-mode and large-mode variants use
  a practical endpoint of the observed Ritz spectrum for theta. Old spectral
  information is reused without a k-HVP refresh before solving; new Ritz data
  is harvested from that solve. This measures the cost of acquiring spectra,
  unlike demonstrations supplied with exact eigenvectors in advance.

All share scalar quadratic-optimal warm starts and a predeclared cold-step
weight 3. There are no private-factor, future-RHS, hidden-label, or solution-table
inputs. Every construction, compression and HVP is inside the sequence clock.
The three programs start empty for every sequence and discard their spectral
basis at an announced phase change.

Large work arrays are allocated once and reused. This avoids the RSS growth
observed with short-lived wide arrays in the initial macOS engineering runs.
The step-boundary allocator hook releases only already-free allocator pages;
actual live and peak RSS remain monitored normally. Both that work and imports
are charged. Memory failures from earlier engineering versions are retained
under ignored var/ and are not mixed into the final frozen-source run.

Sources:
- Saad, Yeung, Erhel, Guyomarc'h, *A Deflated Version of the Conjugate Gradient
  Algorithm*, SISC 21(5), 1909–1926, 2000. DOI 10.1137/S1064829598339761.
- Carlberg, Forstall, Tuminaro, *Krylov-Subspace Recycling via the POD-Augmented
  Conjugate-Gradient Method*, SIMAX 37(3), 1304–1336, 2016.
  https://arxiv.org/abs/1512.05820
- Diouane, Gürol, Mouhtal, Orban, *A Spectral Preconditioner for the Conjugate
  Gradient Method with Iteration Budget*, 2026 preprint.
  https://arxiv.org/abs/2603.28969
- Same authors, *Deflation-Based Spectral Preconditioning for the Conjugate
  Gradient Method on Perturbed Systems*, 2026 preprint.
  https://arxiv.org/abs/2610.28808

Selection uses public data only. Final comparisons must run identical frozen
source snapshots, report all resource failures, and keep historical official
scores unchanged. Local Apple M3 Max results are not Linux grade-slot results.
