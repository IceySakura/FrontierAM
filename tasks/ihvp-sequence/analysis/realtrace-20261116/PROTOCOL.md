> Current replay calibration / 当前回放标定（2026-11-17）：Warm CG＝0，Cold CG＝50；public/private分别重测，旧CG/L-BFGS标定说明作为归档保留。详见[评分修订](reports/ANCHOR_CALIBRATION_20261117.md)。本轮属于对已见结果的事后标定，未修改正式v4。

# Real HERO trajectory candidate dataset, version 2026-11-16

This is a new dataset candidate, not a replacement of official v4 scores.

Each sequence has 50 systems and d=16384. There are 15 public and 30 private
sequences, independently initialized. Each of CIFAR-10, CIFAR-100 and Fashion-MNIST
contributes 5 public + 10 private sequences. Five predeclared schedules cover fixed
batches, resampled batches, medium/short episode resets and faster P updates.

The underlying network is a bias-free two-layer ReLU classifier:

    logits(x) = C ReLU((W0 + B P^T)x),  P^T P = I.

The joint lower variable is z=(vec B, vec C). CIFAR-100 uses hidden=128,
rank=28 and classes=100. The other datasets use hidden=256, rank=54 and classes=10.
These are complete smaller neural models, not projections of the old 671744-dimensional
Hessian. Input normalization is calculated from each official training split.

At each actual P-update checkpoint:

    b_t = grad_z f_query(P_t,z_t)
    H_t = Hessian_zz g_support(P_t,z_t) + (rho_t + gamma) I,
    rho_t = max_h ||(XP)^T diag(mask_h) E||_2, gamma=1e-3.

The Hessian is the true joint CE Hessian, including the B/C cross terms. It is
not replaced by the Gauss-Newton matrix. The explicit damping is necessary because
a neural Hessian can be indefinite. The negative part is bounded by rho_t;
therefore H_t >= gamma I. A Jacobian norm bound supplies a finite certified L.
The same damping is used by both the fixed teacher driver and recorded system.
No dense Hessian is formed. Reference solutions are certified by a true residual:

    ||b-Hv||^2 / [gamma (2 b^T v - v^T H v)] <= 1e-18.

An independent NumPy CPU replay must pass <=1e-16. Operator tests compare to
autograd and dense factorization on a separate tiny model. All recorded systems
also undergo symmetry, linearity, curvature, dimension and finite-value checks.

Lower updates: Nesterov SGD, lr=.016, momentum=.9, joint gradient clip=1.
P updates: normalized horizontal implicit gradient, polar retraction and a fixed
four-trial scalar query-loss safeguard. The driver uses 32 Jacobi-PCG iterations.
This teacher is fixed independently of all evaluated submissions. Initial weights
follow the prior HERO implementation (Kaiming-uniform W0, Xavier-uniform C).
Each sequence starts with independently sampled parameters/data streams; the model
may receive 0,1 or3 actual warm-up episodes. Candidate outputs never affect future
systems. The stored float64 operator is recomputed at a float32 training checkpoint.

The curvature batch is4096 and query batch128. The continuous case uses K=510
lower updates with a P update every10 steps, yielding50 recorded checkpoints.
Short episodes retain the original K=50 shape (four P calls), concatenated until50.
A phase change is announced at each B/P reset. Within an episode, continuity is
empirical, not a promise of low-rank changes or a low-dimensional solution span.

Evaluation follows the existing v4 binary HVP protocol: one fresh worker per sequence,
single CPU thread, wall time30s and peak worker RSS256MiB. Startup, IPC, HVP calls,
coarse-space work and outputs all consume the clock. Teacher arrays/reference
preparation and post-run scoring are excluded, as in the existing task. No future
right-hand side or operator is exposed. Loss is mean relative squared energy error:

    loss = mean_t (vhat_t-vstar_t)^T H_t (vhat_t-vstar_t)
                  / (vstar_t^T H_t vstar_t).

Compare the frozen cold CG, warm CG, L-BFGS-PCG, Ritz/POD/spectral recycling
adaptations and four original frozen agent submissions. All source/options and
case hashes are retained. Methods are not selected or retuned from private results.
Report invalid/time/memory failures explicitly. Raw losses take priority: the old
CG=0 / L-BFGS-PCG=50 score calibration is not automatically valid on new data.

Model visibility: the model may read only the public task statement, worker-side
API/protocol documentation, its own submission and a released public feedback interface.
It may not read this curator document, code/, var/, plans/seeds, operator caches,
reference solutions, teacher logs, validation outputs, private feedback, or other
submissions. These belong in a physically separate evaluator environment. The
local development harness provides protocol separation and UID protection, not a
complete hostile-code/network sandbox.
