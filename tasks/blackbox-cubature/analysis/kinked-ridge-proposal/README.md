# Proposal: replace polynomial_ridge with a sum of kinked ridges

polynomial_ridge (1 + a2 t^2 + a4 t^4, t = v.(x - u)) is a degree-4 polynomial in x: a least-squares fit on the
495 monomials of degree <= 4 (d = 8) or the ridge direction plus exact moments integrates it to machine precision with
2048 points, so the family only tells whether a solver tried that one step. Summing polynomial ridges does not help
(the sum is still a degree-4 polynomial).

Prototype (`oracle.py`): f(x) = 1 + sum_{j=1..m} c_j max(0, v_j.(x - u_j))^{p_j}, m in {2, 3}, p_j in {2, 3}, v_j random
unit vectors, u_j ~ U(0.25, 0.75)^d, c_j ~ U(0.5, 3). Exact integral of each term by the truncated-power formula

    int_[0,1]^d max(0, b + w.x)^p dx = sum_S (-1)^(d-|S|) max(0, b + sum_{i in S} w_i)^(p+d) / (prod_{k=1..d}(p+k) prod_i w_i)

in exact rational arithmetic (checked against adaptive quadrature in d = 1, 2 to 1e-15 and 2^20-point Sobol in d = 3, 5, 8).

Measured on 36 cases (grade slot, 30 s per case; median log10 relative error):

| d | Monte Carlo | Sobol | HumanPortfolio | author's GPT-6.1-sol |
| --- | --- | --- | --- | --- |
| 2 | -2.58 | -6.49 | -7.13 | -6.49 |
| 3 | -2.80 | -5.20 | -5.26 | -5.99 |
| 5 | -2.92 | -4.36 | -4.21 | -4.50 |
| 8 | -2.64 | -4.07 | -3.85 | -4.15 |

On polynomial_ridge the same solvers give -15 (sol in every dimension, HumanPortfolio at d = 2). On the prototype
no solver saturates and errors vary continuously; a solver that identifies every hyperplane and power could still
integrate exactly, which is the structure-discovery skill the family is meant to test. With such a family the
guard rule (gain capped at 0) can be dropped.
