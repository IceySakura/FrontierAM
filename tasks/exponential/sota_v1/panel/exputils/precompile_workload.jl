using ExponentialUtilities
using LinearAlgebra
using SparseArrays

A = sparse([1, 2], [1, 2], [-1.0, -2.0], 2, 2)
B = [1.0 0.25; -0.5 0.75]
y = zeros(2)
caches = ExponentialUtilities._phiv_timestep_caches(@view(B[:, 1]), 2, 1)
phiv_timestep!(
    y, 0.01, A, B; adaptive = true, tol = 1.0e-9, m = 2,
    opnorm = 2.0, ishermitian = true, NA = nnz(A), caches = caches
)
