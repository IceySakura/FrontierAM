import Theory.Sanity

/-!
# The separate-chains hard instance

Coordinates of `ℝ^{m K}` are split into `m` blocks of length `K`; block `ξ` carries a Nesterov-type
chain quadratic with the boundary weight `1 - q` that makes the geometric vector `c q^j` the exact
minimiser. Component `ξ` of one node is (a multiple of) the chain of block `ξ`; every other component
is zero, and `h = (μ/2)‖·‖²`.
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63
namespace Chain

variable {m K : ℕ}

/-- Position `j` of block `ξ`. -/
def idx (ξ : Fin m) (j : Fin K) : Fin (m * K) := finProdFinEquiv (ξ, j)

/-- Coordinate `j` of block `ξ`, and `0` beyond the end of the chain. -/
def xc (x : E (m * K)) (ξ : Fin m) (j : ℕ) : ℝ :=
  if h : j < K then x (idx ξ ⟨j, h⟩) else 0

lemma xc_add (x y : E (m * K)) (ξ : Fin m) (j : ℕ) : xc (x + y) ξ j = xc x ξ j + xc y ξ j := by
  unfold xc; split_ifs <;> simp

lemma xc_sub (x y : E (m * K)) (ξ : Fin m) (j : ℕ) : xc (x - y) ξ j = xc x ξ j - xc y ξ j := by
  unfold xc; split_ifs <;> simp

lemma xc_smul (a : ℝ) (x : E (m * K)) (ξ : Fin m) (j : ℕ) : xc (a • x) ξ j = a * xc x ξ j := by
  unfold xc; split_ifs <;> simp

lemma xc_zero (ξ : Fin m) (j : ℕ) : xc (0 : E (m * K)) ξ j = 0 := by
  unfold xc; split_ifs <;> simp

lemma xc_of_le (x : E (m * K)) (ξ : Fin m) {j : ℕ} (h : K ≤ j) : xc x ξ j = 0 := by
  unfold xc; rw [dif_neg (by omega)]

/-- Sums over the coordinates, block by block. -/
lemma sum_coords (g : Fin (m * K) → ℝ) :
    ∑ c, g c = ∑ ξ : Fin m, ∑ j ∈ Finset.range K,
      (if h : j < K then g (idx ξ ⟨j, h⟩) else 0) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun ξ _ => ?_
  rw [← Fin.sum_univ_eq_sum_range (fun j => if h : j < K then g (idx ξ ⟨j, h⟩) else 0) K]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [dif_pos j.2]
  rfl

lemma norm_sq_eq (x : E (m * K)) : ‖x‖ ^ 2 = ∑ ξ : Fin m, ∑ j ∈ Finset.range K, xc x ξ j ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  simp only [Real.norm_eq_abs, sq_abs]
  rw [sum_coords]
  refine Finset.sum_congr rfl fun ξ _ => Finset.sum_congr rfl fun j _ => ?_
  unfold xc; split_ifs <;> simp

lemma inner_eq (x y : E (m * K)) :
    ⟪x, y⟫ = ∑ ξ : Fin m, ∑ j ∈ Finset.range K, xc x ξ j * xc y ξ j := by
  have : ⟪x, y⟫ = ∑ c, x c * y c := by simp [PiLp.inner_apply, mul_comm]
  rw [this, sum_coords]
  refine Finset.sum_congr rfl fun ξ _ => Finset.sum_congr rfl fun j _ => ?_
  unfold xc; split_ifs <;> simp

lemma sq_xc_le_norm_sq (x : E (m * K)) (ξ : Fin m) (j : ℕ) : xc x ξ j ^ 2 ≤ ‖x‖ ^ 2 := by
  by_cases hj : j < K
  · rw [norm_sq_eq]
    have h1 : xc x ξ j ^ 2 ≤ ∑ j ∈ Finset.range K, xc x ξ j ^ 2 :=
      Finset.single_le_sum (f := fun j => xc x ξ j ^ 2) (fun _ _ => sq_nonneg _)
        (Finset.mem_range.2 hj)
    exact h1.trans (Finset.single_le_sum (f := fun ξ => ∑ j ∈ Finset.range K, xc x ξ j ^ 2)
      (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ ξ))
  · rw [xc_of_le x ξ (by omega)]; simp

/-- The chain weights: `1` except `1 - q` on the last link. -/
def w (K : ℕ) (q : ℝ) (j : ℕ) : ℝ := if j + 1 = K then 1 - q else 1

lemma w_nonneg {q : ℝ} (hq1 : q ≤ 1) (j : ℕ) : 0 ≤ w K q j := by
  unfold w; split_ifs <;> linarith

lemma w_le_one {q : ℝ} (hq : 0 ≤ q) (j : ℕ) : w K q j ≤ 1 := by
  unfold w; split_ifs <;> linarith

/-- The chain bilinear form of block `ξ`:
`x₀ v₀ + ∑_{j<K} w_j (x_j - x_{j+1}) (v_j - v_{j+1})`, with `x_K = v_K = 0`. -/
def bil (q : ℝ) (ξ : Fin m) (x v : E (m * K)) : ℝ :=
  xc x ξ 0 * xc v ξ 0 +
    ∑ j ∈ Finset.range K, w K q j * (xc x ξ j - xc x ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1))

lemma bil_comm (q : ℝ) (ξ : Fin m) (x v : E (m * K)) : bil q ξ x v = bil q ξ v x := by
  unfold bil
  congr 1
  · ring
  · exact Finset.sum_congr rfl fun j _ => by ring

lemma bil_add_right (q : ℝ) (ξ : Fin m) (x v v' : E (m * K)) :
    bil q ξ x (v + v') = bil q ξ x v + bil q ξ x v' := by
  unfold bil
  simp only [xc_add]
  have : ∀ j ∈ Finset.range K,
      w K q j * (xc x ξ j - xc x ξ (j + 1)) * (xc v ξ j + xc v' ξ j - (xc v ξ (j + 1) + xc v' ξ (j + 1))) =
      w K q j * (xc x ξ j - xc x ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1)) +
        w K q j * (xc x ξ j - xc x ξ (j + 1)) * (xc v' ξ j - xc v' ξ (j + 1)) := fun j _ => by ring
  rw [Finset.sum_congr rfl this, Finset.sum_add_distrib]
  ring

lemma bil_smul_right (q : ℝ) (ξ : Fin m) (x v : E (m * K)) (a : ℝ) :
    bil q ξ x (a • v) = a * bil q ξ x v := by
  unfold bil
  simp only [xc_smul]
  rw [mul_add, Finset.mul_sum]
  congr 1
  · ring
  · exact Finset.sum_congr rfl fun j _ => by ring

lemma bil_sub_left (q : ℝ) (ξ : Fin m) (x y v : E (m * K)) :
    bil q ξ (x - y) v = bil q ξ x v - bil q ξ y v := by
  unfold bil
  simp only [xc_sub]
  rw [add_sub_add_comm, ← Finset.sum_sub_distrib]
  congr 1
  · ring
  · exact Finset.sum_congr rfl fun j _ => by ring

/-- The chain quadratic of block `ξ`. -/
def Q (q : ℝ) (ξ : Fin m) (x : E (m * K)) : ℝ := bil q ξ x x

lemma Q_nonneg {q : ℝ} (hq1 : q ≤ 1) (ξ : Fin m) (x : E (m * K)) : 0 ≤ Q q ξ x := by
  unfold Q bil
  refine add_nonneg (mul_self_nonneg _) (Finset.sum_nonneg fun j _ => ?_)
  rw [mul_assoc]
  exact mul_nonneg (w_nonneg hq1 j) (mul_self_nonneg _)

lemma Q_add (q : ℝ) (ξ : Fin m) (x v : E (m * K)) :
    Q q ξ (x + v) = Q q ξ x + 2 * bil q ξ x v + Q q ξ v := by
  unfold Q bil
  simp only [xc_add]
  have : ∀ j ∈ Finset.range K,
      w K q j * (xc x ξ j + xc v ξ j - (xc x ξ (j + 1) + xc v ξ (j + 1))) *
        (xc x ξ j + xc v ξ j - (xc x ξ (j + 1) + xc v ξ (j + 1))) =
      w K q j * (xc x ξ j - xc x ξ (j + 1)) * (xc x ξ j - xc x ξ (j + 1)) +
        2 * (w K q j * (xc x ξ j - xc x ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1))) +
        w K q j * (xc v ξ j - xc v ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1)) := fun j _ => by ring
  rw [Finset.sum_congr rfl this, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

lemma Q_smul (q : ℝ) (ξ : Fin m) (x : E (m * K)) (a : ℝ) : Q q ξ (a • x) = a ^ 2 * Q q ξ x := by
  unfold Q
  rw [bil_smul_right, bil_comm, bil_smul_right]
  ring

lemma Q_sub (q : ℝ) (ξ : Fin m) (x y : E (m * K)) :
    Q q ξ (x - y) = Q q ξ x - 2 * bil q ξ x y + Q q ξ y := by
  have : x - y = x + (-1 : ℝ) • y := by simp [sub_eq_add_neg]
  rw [this, Q_add, Q_smul, bil_smul_right]
  ring

/-- Convex combination identity: `Q(a x + (1-a) y) = a Q x + (1-a) Q y - a (1-a) Q (x - y)`. -/
lemma Q_convex_comb (q : ℝ) (ξ : Fin m) (x y : E (m * K)) (a : ℝ) :
    Q q ξ (a • x + (1 - a) • y) = a * Q q ξ x + (1 - a) * Q q ξ y - a * (1 - a) * Q q ξ (x - y) := by
  rw [Q_add, Q_smul, Q_smul, Q_sub, bil_smul_right, bil_comm, bil_smul_right, bil_comm]
  ring

/-- `Q v ≤ 4 ‖v‖²`. -/
lemma Q_le {q : ℝ} (hq : 0 ≤ q) (hq1 : q ≤ 1) (ξ : Fin m) (v : E (m * K)) :
    Q q ξ v ≤ 4 * ‖v‖ ^ 2 := by
  -- the shifted sum
  have hshift : ∑ j ∈ Finset.range K, xc v ξ (j + 1) ^ 2 + xc v ξ 0 ^ 2 =
      ∑ j ∈ Finset.range K, xc v ξ j ^ 2 := by
    have h1 := Finset.sum_range_succ' (fun j => xc v ξ j ^ 2) K
    have h2 := Finset.sum_range_succ (fun j => xc v ξ j ^ 2) K
    rw [xc_of_le v ξ le_rfl] at h2
    linarith
  have hterm : ∀ j ∈ Finset.range K,
      w K q j * (xc v ξ j - xc v ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1)) ≤
        2 * xc v ξ j ^ 2 + 2 * xc v ξ (j + 1) ^ 2 := by
    intro j _
    have hw0 := w_nonneg (K := K) hq1 j
    have hw1 := w_le_one (K := K) hq j
    have hsq : 0 ≤ (xc v ξ j - xc v ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1)) := mul_self_nonneg _
    calc w K q j * (xc v ξ j - xc v ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1))
        ≤ 1 * ((xc v ξ j - xc v ξ (j + 1)) * (xc v ξ j - xc v ξ (j + 1))) := by
          rw [mul_assoc]; exact mul_le_mul_of_nonneg_right hw1 hsq
      _ ≤ 2 * xc v ξ j ^ 2 + 2 * xc v ξ (j + 1) ^ 2 := by
          nlinarith [sq_nonneg (xc v ξ j + xc v ξ (j + 1))]
  have hblock : Q q ξ v ≤ 4 * ∑ j ∈ Finset.range K, xc v ξ j ^ 2 := by
    unfold Q bil
    have := Finset.sum_le_sum hterm
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum] at this
    nlinarith [sq_nonneg (xc v ξ 0)]
  have hle : ∑ j ∈ Finset.range K, xc v ξ j ^ 2 ≤ ‖v‖ ^ 2 := by
    rw [norm_sq_eq]
    exact Finset.single_le_sum (f := fun ξ => ∑ j ∈ Finset.range K, xc v ξ j ^ 2)
      (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ ξ)
  linarith

/-- Cauchy–Schwarz for the chain form. -/
lemma bil_sq_le {q : ℝ} (hq1 : q ≤ 1) (ξ : Fin m) (x v : E (m * K)) :
    bil q ξ x v ^ 2 ≤ Q q ξ x * Q q ξ v := by
  have key : ∀ t : ℝ, 0 ≤ Q q ξ x - 2 * t * bil q ξ x v + t ^ 2 * Q q ξ v := by
    intro t
    have h := Q_nonneg hq1 ξ (x + (-t) • v)
    rw [Q_add] at h
    have e1 : bil q ξ x ((-t) • v) = -t * bil q ξ x v := bil_smul_right q ξ x v (-t)
    have e2 : Q q ξ ((-t) • v) = t ^ 2 * Q q ξ v := by
      unfold Q
      rw [bil_smul_right, bil_comm, bil_smul_right]
      ring
    rw [e1, e2] at h
    linarith
  rcases (Q_nonneg hq1 ξ v).lt_or_eq with hpos | hzero
  · have := key (bil q ξ x v / Q q ξ v)
    have hne : Q q ξ v ≠ 0 := hpos.ne'
    field_simp at this
    nlinarith
  · -- `Q v = 0` forces `bil x v = 0`
    by_contra hcon
    have hb : bil q ξ x v ≠ 0 := by
      intro hb; rw [hb] at hcon; nlinarith [Q_nonneg hq1 ξ x, Q_nonneg hq1 ξ v]
    have h1 := key ((Q q ξ x + 1) / (2 * bil q ξ x v))
    rw [← hzero] at h1
    have h2 : 2 * ((Q q ξ x + 1) / (2 * bil q ξ x v)) * bil q ξ x v = Q q ξ x + 1 := by
      field_simp
    linarith

/-! ## The gradient of the chain quadratic -/

/-- `v ↦ bil q ξ x v` as a linear functional. -/
def bilLin (q : ℝ) (ξ : Fin m) (x : E (m * K)) : E (m * K) →ₗ[ℝ] ℝ where
  toFun v := bil q ξ x v
  map_add' v v' := bil_add_right q ξ x v v'
  map_smul' a v := by simp [bil_smul_right]

/-- The gradient of `½ Q_ξ` at `x`: the Riesz vector of `bil q ξ x`. -/
def G (q : ℝ) (ξ : Fin m) (x : E (m * K)) : E (m * K) :=
  (InnerProductSpace.toDual ℝ (E (m * K))).symm (LinearMap.toContinuousLinearMap (bilLin q ξ x))

lemma inner_G (q : ℝ) (ξ : Fin m) (x v : E (m * K)) : ⟪G q ξ x, v⟫ = bil q ξ x v := by
  simp [G, InnerProductSpace.toDual_symm_apply, bilLin]

lemma G_sub (q : ℝ) (ξ : Fin m) (x y : E (m * K)) : G q ξ x - G q ξ y = G q ξ (x - y) := by
  apply ext_inner_right ℝ
  intro v
  rw [inner_sub_left, inner_G, inner_G, inner_G, bil_sub_left]

lemma norm_G_le {q : ℝ} (hq : 0 ≤ q) (hq1 : q ≤ 1) (ξ : Fin m) (x : E (m * K)) :
    ‖G q ξ x‖ ≤ 4 * ‖x‖ := by
  set g := G q ξ x
  have h1 : ‖g‖ ^ 2 = bil q ξ x g := by rw [← inner_G, real_inner_self_eq_norm_sq]
  have h2 : bil q ξ x g ^ 2 ≤ Q q ξ x * Q q ξ g := bil_sq_le hq1 ξ x g
  have h3 : Q q ξ x ≤ 4 * ‖x‖ ^ 2 := Q_le hq hq1 ξ x
  have h4 : Q q ξ g ≤ 4 * ‖g‖ ^ 2 := Q_le hq hq1 ξ g
  have hQx := Q_nonneg hq1 ξ x
  have hQg := Q_nonneg hq1 ξ g
  have h5 : ‖g‖ ^ 4 ≤ 16 * ‖x‖ ^ 2 * ‖g‖ ^ 2 := by
    calc ‖g‖ ^ 4 = (bil q ξ x g) ^ 2 := by rw [← h1]; ring
      _ ≤ Q q ξ x * Q q ξ g := h2
      _ ≤ (4 * ‖x‖ ^ 2) * (4 * ‖g‖ ^ 2) := mul_le_mul h3 h4 hQg (by positivity)
      _ = 16 * ‖x‖ ^ 2 * ‖g‖ ^ 2 := by ring
  rcases (norm_nonneg g).lt_or_eq with hg | hg
  · have hg2 : 0 < ‖g‖ ^ 2 := by positivity
    have h6 : ‖g‖ ^ 2 ≤ 16 * ‖x‖ ^ 2 := by nlinarith
    by_contra hcon
    push Not at hcon
    nlinarith [norm_nonneg x]
  · rw [← hg]; positivity

/-- A function with an exact quadratic remainder has the stated gradient. -/
lemma hasGradientAt_of_sq_bound {d : ℕ} {f : E d → ℝ} {g x : E d} {C : ℝ}
    (h : ∀ v, |f (x + v) - f x - ⟪g, v⟫| ≤ C * ‖v‖ ^ 2) : HasGradientAt f g x := by
  rw [hasGradientAt_iff_hasFDerivAt, hasFDerivAt_iff_isLittleO_nhds_zero]
  have hO : (fun v => f (x + v) - f x - (InnerProductSpace.toDual ℝ (E d) g) v) =O[nhds 0]
      (fun v : E d => ‖v‖ ^ 2) := by
    refine Asymptotics.IsBigO.of_bound C (Filter.Eventually.of_forall fun v => ?_)
    rw [InnerProductSpace.toDual_apply_apply, Real.norm_eq_abs, Real.norm_eq_abs,
      abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖v‖ ^ 2)]
    exact h v
  exact hO.trans_isLittleO (Asymptotics.isLittleO_norm_pow_id one_lt_two)

/-- Component `ξ` of the chain node: `N (α/2 Q_ξ x - β x_{ξ,0})`. -/
def comp (q N α β : ℝ) (ξ : Fin m) (x : E (m * K)) : ℝ := N * (α / 2 * Q q ξ x - β * xc x ξ 0)

/-- The unit vector at position `0` of block `ξ`. -/
def e0 (hK : 0 < K) (ξ : Fin m) : E (m * K) := EuclideanSpace.single (idx ξ ⟨0, hK⟩) 1

lemma inner_e0 (hK : 0 < K) (ξ : Fin m) (v : E (m * K)) : ⟪e0 hK ξ, v⟫ = xc v ξ 0 := by
  simp [e0, EuclideanSpace.inner_single_left, xc, hK]

/-- The gradient of `comp`. -/
def gcomp (q N α β : ℝ) (hK : 0 < K) (ξ : Fin m) (x : E (m * K)) : E (m * K) :=
  N • (α • G q ξ x - β • e0 hK ξ)

lemma hasGradientAt_comp {q : ℝ} (hq : 0 ≤ q) (hq1 : q ≤ 1) (N α β : ℝ) (hK : 0 < K) (ξ : Fin m)
    (x : E (m * K)) : HasGradientAt (comp q N α β ξ) (gcomp q N α β hK ξ x) x := by
  apply hasGradientAt_of_sq_bound (C := |N * α / 2| * 4)
  intro v
  have hexp : comp q N α β ξ (x + v) - comp q N α β ξ x - ⟪gcomp q N α β hK ξ x, v⟫ =
      N * α / 2 * Q q ξ v := by
    simp only [comp, gcomp, real_inner_smul_left, inner_sub_left, inner_G, inner_e0, Q_add, xc_add]
    ring
  rw [hexp, abs_mul, abs_of_nonneg (Q_nonneg hq1 ξ v), mul_assoc]
  exact mul_le_mul_of_nonneg_left (Q_le hq hq1 ξ v) (abs_nonneg _)

lemma gradient_comp {q : ℝ} (hq : 0 ≤ q) (hq1 : q ≤ 1) (N α β : ℝ) (hK : 0 < K) (ξ : Fin m)
    (x : E (m * K)) : gradient (comp q N α β ξ) x = gcomp q N α β hK ξ x :=
  (hasGradientAt_comp hq hq1 N α β hK ξ x).gradient

lemma comp_smooth {q : ℝ} (hq : 0 ≤ q) (hq1 : q ≤ 1) {N α β L : ℝ} (hN : 0 ≤ N) (hα : 0 ≤ α)
    (hL : 4 * N * α ≤ L) (hK : 0 < K) (ξ : Fin m) : LSmooth L (comp (K := K) q N α β ξ) := by
  refine ⟨fun x => (hasGradientAt_comp hq hq1 N α β hK ξ x).differentiableAt, fun x y => ?_⟩
  rw [gradient_comp hq hq1 N α β hK, gradient_comp hq hq1 N α β hK]
  have : gcomp q N α β hK ξ x - gcomp q N α β hK ξ y = (N * α) • G q ξ (x - y) := by
    simp only [gcomp, ← G_sub, smul_sub, smul_smul]
    abel
  rw [this, norm_smul, Real.norm_eq_abs, abs_of_nonneg (mul_nonneg hN hα)]
  calc N * α * ‖G q ξ (x - y)‖ ≤ N * α * (4 * ‖x - y‖) :=
        mul_le_mul_of_nonneg_left (norm_G_le hq hq1 ξ (x - y)) (mul_nonneg hN hα)
    _ = 4 * N * α * ‖x - y‖ := by ring
    _ ≤ L * ‖x - y‖ := mul_le_mul_of_nonneg_right hL (norm_nonneg _)

/-! ## Supports -/

/-- The coordinates at positions `< k ξ` of every block `ξ`. -/
def Box (k : Fin m → ℕ) : Set (Fin (m * K)) :=
  {c | ((finProdFinEquiv.symm c).2 : ℕ) < k (finProdFinEquiv.symm c).1}

lemma idx_mem_Box {k : Fin m → ℕ} {ξ : Fin m} {j : Fin K} : idx ξ j ∈ Box k ↔ (j : ℕ) < k ξ := by
  simp only [Box, idx, Set.mem_setOf_eq, Equiv.symm_apply_apply]

lemma Box_mono {k k' : Fin m → ℕ} (h : k ≤ k') : Box (K := K) k ⊆ Box k' := by
  intro c hc
  exact lt_of_lt_of_le hc (h _)

lemma xc_eq_zero_of_supp {k : Fin m → ℕ} {x : E (m * K)} (hx : supp x ⊆ Box k) (ξ : Fin m)
    {j : ℕ} (hj : k ξ ≤ j) : xc x ξ j = 0 := by
  unfold xc
  split_ifs with h
  · by_contra hne
    have := idx_mem_Box.1 (hx hne)
    simp at this
    omega
  · rfl

lemma xc_single (ξ' : Fin m) (j' : Fin K) (ξ : Fin m) (j : ℕ) :
    xc (EuclideanSpace.single (idx ξ' j') (1 : ℝ)) ξ j = if ξ = ξ' ∧ j = (j' : ℕ) then 1 else 0 := by
  unfold xc
  split_ifs with h1 h2 h2
  · simp [PiLp.single_apply, idx, h2.1, h2.2]
  · simp only [EuclideanSpace.single, PiLp.single_apply, idx, Equiv.apply_eq_iff_eq, Prod.mk.injEq]
    rw [if_neg]
    intro h
    apply h2
    refine ⟨h.1, ?_⟩
    have := congrArg Fin.val h.2
    simpa using this
  · exact absurd (h2.2 ▸ j'.2) h1
  · rfl

lemma G_coord (q : ℝ) (ξ : Fin m) (x : E (m * K)) (c : Fin (m * K)) :
    G q ξ x c = bil q ξ x (EuclideanSpace.single c 1) := by
  rw [← inner_G, EuclideanSpace.inner_single_right]
  simp

/-- The gradient of the chain quadratic advances block `ξ` by at most one position. -/
lemma G_supp {q : ℝ} {k : Fin m → ℕ} {x : E (m * K)} (hx : supp x ⊆ Box k) (ξ : Fin m) :
    supp (G q ξ x) ⊆ Box (k + Pi.single ξ 1) := by
  intro c hc
  by_contra hnot
  apply hc
  obtain ⟨⟨ξ', j'⟩, rfl⟩ : ∃ p, finProdFinEquiv p = c := ⟨_, finProdFinEquiv.apply_symm_apply c⟩
  change G q ξ x (idx ξ' j') = 0
  have hnot' : ¬ (j' : ℕ) < (k + Pi.single ξ 1 : Fin m → ℕ) ξ' := fun h => hnot (idx_mem_Box.2 h)
  rw [G_coord]
  unfold bil
  simp only [xc_single]
  by_cases hξ : ξ = ξ'
  · subst hξ
    simp only [Pi.add_apply, Pi.single_eq_same, not_lt] at hnot'
    have h0 : (0 : ℕ) ≠ j' := by omega
    simp only [true_and, h0, if_false, mul_zero, zero_add]
    refine Finset.sum_eq_zero fun j _ => ?_
    by_cases hj : j < k ξ
    · have a1 : j ≠ (j' : ℕ) := by omega
      have a2 : j + 1 ≠ (j' : ℕ) := by omega
      simp [a1, a2]
    · rw [xc_eq_zero_of_supp hx ξ (by omega), xc_eq_zero_of_supp hx ξ (by omega)]
      ring
  · simp [hξ]

lemma supp_smul_subset {d : ℕ} (a : ℝ) (x : E d) : supp (a • x) ⊆ supp x := by
  intro c hc h
  apply hc
  simp [h]

lemma supp_sub_subset {d : ℕ} {S : Set (Fin d)} {x y : E d} (hx : supp x ⊆ S) (hy : supp y ⊆ S) :
    supp (x - y) ⊆ S := by
  intro c hc
  by_contra hcS
  apply hc
  have h1 : x c = 0 := by by_contra h; exact hcS (hx h)
  have h2 : y c = 0 := by by_contra h; exact hcS (hy h)
  simp [h1, h2]

lemma gcomp_supp {q N α β : ℝ} (hK : 0 < K) {k : Fin m → ℕ} {x : E (m * K)} (hx : supp x ⊆ Box k)
    (ξ : Fin m) : supp (gcomp q N α β hK ξ x) ⊆ Box (k + Pi.single ξ 1) := by
  refine (supp_smul_subset _ _).trans (supp_sub_subset ((supp_smul_subset _ _).trans (G_supp hx ξ))
    ((supp_smul_subset _ _).trans ?_))
  intro c hc
  have : c = idx ξ ⟨0, hK⟩ := by
    by_contra hne
    apply hc
    simp [e0, PiLp.single_apply, hne]
  rw [this, idx_mem_Box]
  simp

/-! ## The minimiser and the objective -/

/-- `α = μ q / (1-q)²`: the chain coefficient for which `q` is the decay rate of the minimiser. -/
def alpha (μ q : ℝ) : ℝ := μ * q / (1 - q) ^ 2

/-- The geometric vector `x⋆_{ξ,j} = c q^j`. -/
def xstar (m K : ℕ) (c q : ℝ) : E (m * K) :=
  WithLp.toLp 2 (fun i => c * q ^ ((finProdFinEquiv.symm i).2 : ℕ))

lemma xc_xstar (c q : ℝ) (ξ : Fin m) (j : ℕ) :
    xc (xstar m K c q) ξ j = if j < K then c * q ^ j else 0 := by
  unfold xc xstar
  split_ifs <;> simp [idx]

/-- The objective `φ = f + h` in closed form. -/
def Phi (q α β μ : ℝ) (x : E (m * K)) : ℝ :=
  ∑ ξ : Fin m, (α / 2 * Q q ξ x - β * xc x ξ 0) + μ / 2 * ‖x‖ ^ 2

lemma Phi_eq_sum (q α β μ : ℝ) (x : E (m * K)) :
    Phi q α β μ x =
      ∑ ξ : Fin m, (α / 2 * Q q ξ x - β * xc x ξ 0 + μ / 2 * ∑ j ∈ Finset.range K, xc x ξ j ^ 2) := by
  unfold Phi
  rw [norm_sq_eq, Finset.mul_sum, ← Finset.sum_add_distrib]

/-- First-order optimality of `x⋆` on block `ξ`, tested against `v`. -/
lemma block_identity {μ q c : ℝ} (hK : 0 < K) (hq1 : q < 1) (ξ : Fin m) (v : E (m * K)) :
    alpha μ q * bil q ξ (xstar m K c q) v +
        μ * ∑ j ∈ Finset.range K, xc (xstar m K c q) ξ j * xc v ξ j =
      μ * c / (1 - q) ^ 2 * xc v ξ 0 := by
  have h1q : (1 - q) ≠ 0 := by
    intro h; linarith
  set S := ∑ j ∈ Finset.range K, q ^ j * xc v ξ j with hS
  set S' := ∑ j ∈ Finset.range K, q ^ j * xc v ξ (j + 1) with hS'
  have hshift : q * S' = S - xc v ξ 0 := by
    have e1 := Finset.sum_range_succ' (fun j => q ^ j * xc v ξ j) K
    have e2 := Finset.sum_range_succ (fun j => q ^ j * xc v ξ j) K
    rw [xc_of_le v ξ le_rfl, mul_zero, add_zero] at e2
    have e3 : ∑ j ∈ Finset.range K, q ^ (j + 1) * xc v ξ (j + 1) = q * S' := by
      rw [hS', Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    simp only [pow_zero, one_mul] at e1
    linarith
  have hw : ∀ j ∈ Finset.range K,
      w K q j * (xc (xstar m K c q) ξ j - xc (xstar m K c q) ξ (j + 1)) = c * (1 - q) * q ^ j := by
    intro j hj
    rw [Finset.mem_range] at hj
    rw [xc_xstar, xc_xstar, if_pos hj]
    unfold w
    by_cases hj1 : j + 1 < K
    · rw [if_pos hj1, if_neg (by omega)]; ring
    · rw [if_neg hj1, if_pos (by omega)]; ring
  have hbil : bil q ξ (xstar m K c q) v = c * xc v ξ 0 + c * (1 - q) * (S - S') := by
    unfold bil
    rw [xc_xstar, if_pos hK, pow_zero, mul_one]
    congr 1
    rw [hS, hS', mul_sub, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [mul_assoc (w K q j), ← mul_assoc, hw j hj]
    ring
  have hsum : ∑ j ∈ Finset.range K, xc (xstar m K c q) ξ j * xc v ξ j = c * S := by
    rw [hS, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [xc_xstar, if_pos (Finset.mem_range.1 hj)]
    ring
  rw [hbil, hsum]
  have key : μ * q * (c * xc v ξ 0 + c * (1 - q) * (S - S')) + μ * (c * S) * (1 - q) ^ 2 =
      μ * c * xc v ξ 0 := by
    linear_combination (-(μ * c * (1 - q))) * hshift
  unfold alpha
  have hpos : (1 - q) ^ 2 ≠ 0 := pow_ne_zero 2 h1q
  field_simp
  linear_combination key

lemma Phi_expand {μ q c : ℝ} (hK : 0 < K) (hq1 : q < 1) (v : E (m * K)) :
    Phi q (alpha μ q) (μ * c / (1 - q) ^ 2) μ (xstar m K c q + v) =
      Phi q (alpha μ q) (μ * c / (1 - q) ^ 2) μ (xstar m K c q) +
        (∑ ξ : Fin m, alpha μ q / 2 * Q q ξ v + μ / 2 * ‖v‖ ^ 2) := by
  rw [Phi_eq_sum, Phi_eq_sum, norm_sq_eq, Finset.mul_sum, ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun ξ _ => ?_
  have hb := block_identity (μ := μ) (c := c) hK hq1 ξ v
  have hs : ∑ j ∈ Finset.range K, xc (xstar m K c q + v) ξ j ^ 2 =
      ∑ j ∈ Finset.range K, xc (xstar m K c q) ξ j ^ 2 +
        2 * ∑ j ∈ Finset.range K, xc (xstar m K c q) ξ j * xc v ξ j +
        ∑ j ∈ Finset.range K, xc v ξ j ^ 2 := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by rw [xc_add]; ring
  rw [Q_add, xc_add, hs]
  linear_combination hb

lemma Phi_ge {μ q c : ℝ} (hμ : 0 ≤ μ) (hK : 0 < K) (hq : 0 ≤ q) (hq1 : q < 1) (x : E (m * K)) :
    Phi q (alpha μ q) (μ * c / (1 - q) ^ 2) μ (xstar m K c q) + μ / 2 * ‖x - xstar m K c q‖ ^ 2 ≤
      Phi q (alpha μ q) (μ * c / (1 - q) ^ 2) μ x := by
  have h := Phi_expand (μ := μ) (c := c) hK hq1 (x - xstar m K c q)
  rw [add_sub_cancel] at h
  rw [h]
  have hα : 0 ≤ alpha μ q := by unfold alpha; positivity
  have : 0 ≤ ∑ ξ : Fin m, alpha μ q / 2 * Q q ξ (x - xstar m K c q) :=
    Finset.sum_nonneg fun ξ _ => mul_nonneg (by positivity) (Q_nonneg hq1.le ξ _)
  linarith

lemma Phi_xstar {μ q c : ℝ} (hK : 0 < K) (hq1 : q < 1) :
    Phi q (alpha μ q) (μ * c / (1 - q) ^ 2) μ (xstar m K c q) =
      -(m * (μ * c / (1 - q) ^ 2) * c / 2) := by
  rw [Phi_eq_sum]
  have : ∀ ξ : Fin m, alpha μ q / 2 * Q q ξ (xstar m K c q) -
      μ * c / (1 - q) ^ 2 * xc (xstar m K c q) ξ 0 +
      μ / 2 * ∑ j ∈ Finset.range K, xc (xstar m K c q) ξ j ^ 2 = -((μ * c / (1 - q) ^ 2) * c / 2) := by
    intro ξ
    have hb := block_identity (μ := μ) (c := c) hK hq1 ξ (xstar m K c q)
    have h0 : xc (xstar m K c q) ξ 0 = c := by rw [xc_xstar, if_pos hK, pow_zero, mul_one]
    rw [h0] at hb ⊢
    have hsq : ∑ j ∈ Finset.range K, xc (xstar m K c q) ξ j ^ 2 =
        ∑ j ∈ Finset.range K, xc (xstar m K c q) ξ j * xc (xstar m K c q) ξ j :=
      Finset.sum_congr rfl fun j _ => by ring
    unfold Q
    rw [hsq]
    linear_combination hb / 2
  rw [Finset.sum_congr rfl fun ξ _ => this ξ, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  ring

lemma Phi_zero (q α β μ : ℝ) : Phi q α β μ (0 : E (m * K)) = 0 := by
  unfold Phi Q bil
  simp [xc_zero]

/-- Gap on a box: if `x` vanishes at position `k ξ < K` of every block, its gap is at least
`(μ c²/2) ∑_ξ q^{2 k ξ}`. -/
lemma Phi_gap_box {μ q c : ℝ} (hμ : 0 ≤ μ) (hK : 0 < K) (hq : 0 ≤ q) (hq1 : q < 1)
    {k : Fin m → ℕ} (hk : ∀ ξ, k ξ < K) {x : E (m * K)} (hx : supp x ⊆ Box k) :
    μ * c ^ 2 / 2 * ∑ ξ : Fin m, (q ^ k ξ) ^ 2 ≤
      Phi q (alpha μ q) (μ * c / (1 - q) ^ 2) μ x -
        Phi q (alpha μ q) (μ * c / (1 - q) ^ 2) μ (xstar m K c q) := by
  have h := Phi_ge (c := c) hμ hK hq hq1 x
  have hnorm : ∑ ξ : Fin m, xc (x - xstar m K c q) ξ (k ξ) ^ 2 ≤ ‖x - xstar m K c q‖ ^ 2 := by
    rw [norm_sq_eq]
    exact Finset.sum_le_sum fun ξ _ =>
      Finset.single_le_sum (f := fun j => xc (x - xstar m K c q) ξ j ^ 2) (fun _ _ => sq_nonneg _)
        (Finset.mem_range.2 (hk ξ))
  have hterm : ∀ ξ, xc (x - xstar m K c q) ξ (k ξ) ^ 2 = c ^ 2 * (q ^ k ξ) ^ 2 := by
    intro ξ
    rw [xc_sub, xc_eq_zero_of_supp hx ξ le_rfl, xc_xstar, if_pos (hk ξ)]
    ring
  simp only [hterm] at hnorm
  rw [← Finset.mul_sum] at hnorm
  nlinarith

/-! ## The instance -/

/-- The hard instance: node `0` holds the `m` chains (scaled by `N = n m`), all other components
vanish, `h = (μ/2)‖·‖²`, `x₀ = 0`. -/
def inst (n m K : ℕ) (hn : 0 < n) (q α β μ : ℝ) : Instance n m (m * K) where
  F i ξ := if i = ⟨0, hn⟩ then comp q ((n : ℝ) * m) α β ξ else fun _ => 0
  h x := ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal)
  x0 := 0
  prox v η := (1 + η * μ)⁻¹ • v

lemma inst_f {n : ℕ} (hn : 0 < n) (hm : 0 < m) (q α β μ : ℝ) (x : E (m * K)) :
    (inst n m K hn q α β μ).f x = ∑ ξ : Fin m, (α / 2 * Q q ξ x - β * xc x ξ 0) := by
  unfold Instance.f inst
  simp only
  rw [Finset.sum_eq_single (⟨0, hn⟩ : Fin n)]
  · simp only [if_true, comp]
    rw [← Finset.mul_sum]
    have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
    have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
    field_simp
  · intro i _ hi
    simp [hi]
  · simp

lemma inst_φ {n : ℕ} (hn : 0 < n) (hm : 0 < m) (q α β μ : ℝ) (x : E (m * K)) :
    (inst n m K hn q α β μ).φ x = ((Phi q α β μ x : ℝ) : EReal) := by
  unfold Instance.φ
  rw [inst_f hn hm]
  simp only [inst, Phi]
  rw [← EReal.coe_add]

lemma convex_chainPart {q α β : ℝ} (hq1 : q ≤ 1) (hα : 0 ≤ α) :
    ConvexOn ℝ Set.univ (fun x : E (m * K) => ∑ ξ : Fin m, (α / 2 * Q q ξ x - β * xc x ξ 0)) := by
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  obtain rfl : b = 1 - a := by linarith
  simp only [smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun ξ _ => ?_
  rw [Q_convex_comb, xc_add, xc_smul, xc_smul]
  have : 0 ≤ α / 2 * (a * (1 - a) * Q q ξ (x - y)) :=
    mul_nonneg (by positivity) (mul_nonneg (mul_nonneg ha hb) (Q_nonneg hq1 ξ _))
  nlinarith

end Chain
end ULB63
