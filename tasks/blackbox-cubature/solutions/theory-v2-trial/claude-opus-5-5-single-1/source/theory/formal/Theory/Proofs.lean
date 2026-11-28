import Theory.Defs
import Theory.Main
import Theory.Kink
import Theory.Quadric
import Theory.Hcond

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof.
The work is done in `Theory/OneDim.lean` (the one-dimensional building blocks),
`Theory/Tail.lean` (the preintegrated profile and its loss of smoothness),
`Theory/MultiDim.lean` (iterated partial derivatives), `Theory/Main.lean` (assembly of R1) and
`Theory/Kink.lean` (R2), `Theory/Hcond.lean` (GKLS's condition (h2), final assembly of R1 and
R2) and `Theory/Quadric.lean` (R3).
-/

open Filter MeasureTheory ProbabilityTheory Real
open scoped ENNReal

namespace GKLS

/-- **R1 (the condition `r_j ≥ ∑_{k≠j} r_k` of GKLS Theorem 3 cannot be weakened).**
Let `d = n + 1`, let `j` be the preintegration coordinate and `r` a smoothness multi-index with
`1 ≤ r_j < ∑_{k ≠ j} r_k`. Then there is `φ : ℝ^d → ℝ` such that, with `θ ≡ 1`,
`f = θ · ind(φ) = ind ∘ φ` and `ρ` the standard normal density, all hypotheses of GKLS Theorem 3
other than the condition on `r` hold:

* (H0) `θ ≡ 1 ∈ C^r_mix`, all its partial derivatives of positive order vanish identically, and
  `θ ∈ L^p(ρ_d)` for every `p`;
* (H1) `φ ∈ C^r_mix(ℝ^d)`; in fact every mixed derivative with at most `r_j` differentiations in
  `x_j` (and any number in the other variables) exists everywhere and is continuous;
* (H2) `D_j φ ≥ 1/2 > 0` everywhere and `φ(x_j, y) → ∞` as `x_j → ∞`, for every `y`;
* (H3) every line in the direction `x_j` meets `{φ = 0}`, i.e. GKLS's set `U_j` is all of
  `ℝ^{d-1}` (so their boundary condition is vacuous);
* (H4) `|φ(x)| ≤ 1 + ∑ |x_i|` and `|D^α φ| ≤ 3/2` for every `α ≠ 0` with `α_j ≤ r_j`;
* (H5) the weighted integrability that defines `W^r_{d,p,ζ,mix}`, for the classical derivatives:
  every `D^α φ` (`α_j ≤ r_j`) lies in `L^p(ρ_d)` for all `p < ∞`, and for `α ≠ 0`,
  `∫ |D^α φ(x)|^p ∏_i ζ_{α_i}(x_i) dx < ∞` for every family of integrable weights `ζ_0, ζ_1, …`
  and (H5w) these classical derivatives are also the weak derivatives:
  `∫ φ · D_{reverse l} v = (-1)^{|l|} ∫ (D_l φ) · v` for every smooth compactly supported `v`;
* (H6) GKLS's condition (h2), literally and for all parameters: for the root `ψ` of
  `φ(ψ(y), y) = 0`, all `η`, all lists `γ⁽¹⁾, …, γ⁽ᵃ⁾` of nonempty multi-indices with
  `γ⁽ⁱ⁾_j ≤ r_j`, and all `b, c`, the function
  `h(y) = D^η θ(ψ(y), y) ∏ᵢ D^{γ⁽ⁱ⁾} φ(ψ(y), y) / (D_j φ(ψ(y), y))^b · ρ^{(c)}(ψ(y))`
  satisfies `∫ |h(y)|^p ∏_k ζ_{α_k}(y_k) dy < ∞` for all `p ≥ 0`, all integrable weights and all
  `α` (condition (h1) is vacuous because of (H3));

but (C) the conclusion of GKLS Theorem 3 fails (its classical part): `P_j f ∉ C^{r_{-j}}_mix(ℝ^{d-1})`. Moreover (E) the
mixed smoothness of `P_j f` is exactly "total order `r_j`": every iterated partial derivative of
`P_j f` with at most `r_j` differentiations (any directions, any order) exists everywhere and is
continuous, and no iterated partial derivative with more than `r_j` differentiations exists
everywhere. -/
theorem gkls_mixed_condition_sharp (n : ℕ) (j : Fin (n + 1)) (r : Fin (n + 1) → ℕ)
    (hrj : 1 ≤ r j) (hsum : r j < ∑ k : Fin n, r (j.succAbove k)) :
    ∃ φ : (Fin (n + 1) → ℝ) → ℝ,
      -- (H0) `θ ≡ 1 ∈ C^r_mix`, its derivatives of positive order vanish, and `θ ∈ L^p(ρ_d)`
      ContDiffMixed r (fun _ : Fin (n + 1) → ℝ => (1 : ℝ)) ∧
      (∀ l : List (Fin (n + 1)), l ≠ [] →
        iterPartial l (fun _ : Fin (n + 1) → ℝ => (1 : ℝ)) = fun _ => 0) ∧
      (∀ p : ℝ≥0∞, MemLp (fun _ : Fin (n + 1) → ℝ => (1 : ℝ)) p
        (Measure.pi fun _ : Fin (n + 1) => gaussianReal 0 1)) ∧
      -- (H1) `φ ∈ C^r_mix`, and `φ` is `C^∞` in the variables `x_{-j}`
      ContDiffMixed r φ ∧
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        IterPartialExists l φ ∧ Continuous (iterPartial l φ)) ∧
      -- (H2) monotone in `x_j`, and `φ → ∞` as `x_j → ∞`
      (∀ x, 1 / 2 ≤ partialDeriv j φ x) ∧
      (∀ y : Fin n → ℝ, Tendsto (fun t => φ (Fin.insertNth j t y)) atTop atTop) ∧
      -- (H3) `U_j = ℝ^{d-1}`
      (∀ y : Fin n → ℝ, ∃ t, φ (Fin.insertNth j t y) = 0) ∧
      -- (H4) growth bounds
      (∀ x, |φ x| ≤ 1 + ∑ i, |x i|) ∧
      (∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j → ∀ x, |iterPartial l φ x| ≤ 3 / 2) ∧
      -- (H5) weighted Sobolev membership
      (∀ p : ℝ≥0∞, p ≠ ∞ → ∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        MemLp (iterPartial l φ) p (Measure.pi fun _ : Fin (n + 1) => gaussianReal 0 1)) ∧
      (∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
        ∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j →
          Integrable (fun x => |iterPartial l φ x| ^ p * ∏ i, ζ (l.count i) (x i))) ∧
      -- (H5w) the classical derivatives `D_l φ` are the weak derivatives of `φ`
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        ∀ v : (Fin (n + 1) → ℝ) → ℝ, ContDiff ℝ (⊤ : ℕ∞) v → HasCompactSupport v →
          ∫ x, φ x * iterPartial l.reverse v x = (-1) ^ l.length * ∫ x, iterPartial l φ x * v x) ∧
      -- (H6) GKLS's condition (h2), with `θ ≡ 1` (condition (h1) is vacuous by (H3))
      (∀ ψ : (Fin n → ℝ) → ℝ, (∀ y, φ (Fin.insertNth j (ψ y) y) = 0) →
        ∀ (η : List (Fin (n + 1))) (γs : List (List (Fin (n + 1)))) (b c : ℕ),
          (∀ γ ∈ γs, γ ≠ [] ∧ γ.count j ≤ r j) →
          ∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ) (α : Fin n → ℕ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
            Integrable (fun y => |iterPartial η (fun _ => (1 : ℝ)) (Fin.insertNth j (ψ y) y) *
                (γs.map fun γ => iterPartial γ φ (Fin.insertNth j (ψ y) y)).prod /
                partialDeriv j φ (Fin.insertNth j (ψ y) y) ^ b *
                iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i))) ∧
      -- (C) the conclusion `P_j f ∈ C^{r_{-j}}_mix` of GKLS Theorem 3 fails
      ¬ ContDiffMixed (fun k => r (j.succAbove k)) (preintegrate j (fun x => ind (φ x))) ∧
      -- (E) exact mixed smoothness of `P_j f`: total order `r_j`, and no more
      (∀ l : List (Fin n), l.length ≤ r j →
        IterPartialExists l (preintegrate j (fun x => ind (φ x))) ∧
        Continuous (iterPartial l (preintegrate j (fun x => ind (φ x))))) ∧
      (∀ l : List (Fin n), r j < l.length →
        ¬ IterPartialExists l (preintegrate j (fun x => ind (φ x)))) :=
  main_counterexample_h n j r hrj hsum

/-- **R2 (kinks, `θ = φ`: the condition can be relaxed by at most one).**
With the same notation, let `θ = φ`, so that `f = φ · ind(φ) = max(φ, 0)` has a kink. For every
multi-index `r` with `1 ≤ r_j` and `r_j + 1 < ∑_{k ≠ j} r_k` there is `φ` satisfying (H1)–(H5w)
of R1 (so `θ = φ` satisfies them as well) and (H6) GKLS's condition (h2) for `θ = φ`, but
`P_j f ∉ C^{r_{-j}}_mix(ℝ^{d-1})`. More precisely `P_j f` has exact
total mixed order `r_j + 1`: every iterated partial derivative with at most `r_j + 1`
differentiations exists everywhere and is continuous, and none with more exists everywhere. -/
theorem gkls_mixed_condition_kink (n : ℕ) (j : Fin (n + 1)) (r : Fin (n + 1) → ℕ)
    (hrj : 1 ≤ r j) (hsum : r j + 1 < ∑ k : Fin n, r (j.succAbove k)) :
    ∃ φ : (Fin (n + 1) → ℝ) → ℝ,
      -- (H1)–(H5w) for `φ` (hence for `θ = φ`), exactly as in R1
      ContDiffMixed r φ ∧
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        IterPartialExists l φ ∧ Continuous (iterPartial l φ)) ∧
      (∀ x, 1 / 2 ≤ partialDeriv j φ x) ∧
      (∀ y : Fin n → ℝ, Tendsto (fun t => φ (Fin.insertNth j t y)) atTop atTop) ∧
      (∀ y : Fin n → ℝ, ∃ t, φ (Fin.insertNth j t y) = 0) ∧
      (∀ x, |φ x| ≤ 1 + ∑ i, |x i|) ∧
      (∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j → ∀ x, |iterPartial l φ x| ≤ 3 / 2) ∧
      (∀ p : ℝ≥0∞, p ≠ ∞ → ∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        MemLp (iterPartial l φ) p (Measure.pi fun _ : Fin (n + 1) => gaussianReal 0 1)) ∧
      (∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
        ∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j →
          Integrable (fun x => |iterPartial l φ x| ^ p * ∏ i, ζ (l.count i) (x i))) ∧
      -- (H5w) the classical derivatives `D_l φ` are the weak derivatives of `φ`
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        ∀ v : (Fin (n + 1) → ℝ) → ℝ, ContDiff ℝ (⊤ : ℕ∞) v → HasCompactSupport v →
          ∫ x, φ x * iterPartial l.reverse v x = (-1) ^ l.length * ∫ x, iterPartial l φ x * v x) ∧
      -- (H6) GKLS's condition (h2), with `θ = φ` (condition (h1) is vacuous by (H3))
      (∀ ψ : (Fin n → ℝ) → ℝ, (∀ y, φ (Fin.insertNth j (ψ y) y) = 0) →
        ∀ (η : List (Fin (n + 1))) (γs : List (List (Fin (n + 1)))) (b c : ℕ),
          η.count j ≤ r j → (∀ γ ∈ γs, γ ≠ [] ∧ γ.count j ≤ r j) →
          ∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ) (α : Fin n → ℕ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
            Integrable (fun y => |iterPartial η φ (Fin.insertNth j (ψ y) y) *
                (γs.map fun γ => iterPartial γ φ (Fin.insertNth j (ψ y) y)).prod /
                partialDeriv j φ (Fin.insertNth j (ψ y) y) ^ b *
                iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i))) ∧
      -- the conclusion of GKLS Theorem 3 fails for the kink `f = φ · ind(φ)`
      ¬ ContDiffMixed (fun k => r (j.succAbove k)) (preintegrate j (fun x => φ x * ind (φ x))) ∧
      -- exact mixed smoothness of `P_j f`: total order `r_j + 1`
      (∀ l : List (Fin n), l.length ≤ r j + 1 →
        IterPartialExists l (preintegrate j (fun x => φ x * ind (φ x))) ∧
        Continuous (iterPartial l (preintegrate j (fun x => φ x * ind (φ x))))) ∧
      (∀ l : List (Fin n), r j + 1 < l.length →
        ¬ IterPartialExists l (preintegrate j (fun x => φ x * ind (φ x)))) :=
  kink_counterexample_h n j r hrj hsum

/-- **R3 (Target A, minor: explicit Hölder-1/2 bounds after preintegration across a quadric
jump).** Let `a ≠ 0` and let `f = ind(a x_j² + β(x_{-j}) x_j + κ(x_{-j}))` (`quadricJump`) be the
jump across a hypersurface that is quadratic — hence *not monotone* — in the preintegration
variable `x_j`; write `D(y) = β(y)² - 4 a κ(y)` (`β, κ` arbitrary). Then:

* (U) `P_j f` is `1/2`-Hölder in terms of the data, with an explicit constant:
  `|P_j f(y) - P_j f(y')| ≤ (|β(y) - β(y')| + √|D(y) - D(y')|) / (|a| √(2π))`;
* (Z) if `a < 0` (the surface is closed in the `x_j` direction), `P_j f(y) = 0` wherever `D(y) ≤ 0`;
* (L) if `a < 0` and both roots `(-β(y) ± √D(y))/(2a)` lie in `[-R, R]` (implied by
  `|β(y)| + √D(y) ≤ 2|a|R`), then `P_j f(y) ≥ ρ(R) √D(y) / |a|`.

(U) is a global regularity upper bound matching, at fold points where `D` changes sign with
`∇D ≠ 0`, the square-root singularity of Gilbert–Kuo–Sloan's Theorem 1; (Z)+(L) are explicit-constant
versions of their lower bound. -/
theorem quadric_preintegration_sqrt (n : ℕ) (j : Fin (n + 1)) (a : ℝ) (ha : a ≠ 0)
    (β κ : (Fin n → ℝ) → ℝ) :
    -- (U) Hölder-1/2 upper bound
    (∀ y y' : Fin n → ℝ,
      |preintegrate j (quadricJump j a β κ) y - preintegrate j (quadricJump j a β κ) y'| ≤
        (|β y - β y'| + √|(β y ^ 2 - 4 * a * κ y) - (β y' ^ 2 - 4 * a * κ y')|) /
          (|a| * √(2 * π))) ∧
    -- (Z) vanishing where the fibre misses the surface
    (a < 0 → ∀ y : Fin n → ℝ, β y ^ 2 - 4 * a * κ y ≤ 0 →
      preintegrate j (quadricJump j a β κ) y = 0) ∧
    -- (L) square-root lower bound
    (a < 0 → ∀ (R : ℝ) (y : Fin n → ℝ), |β y| + √(β y ^ 2 - 4 * a * κ y) ≤ 2 * |a| * R →
      gaussianPDFReal 0 1 R * √(β y ^ 2 - 4 * a * κ y) / |a| ≤
        preintegrate j (quadricJump j a β κ) y) :=
  quadric_assembly n j a ha β κ

end GKLS
