import Theory.PathSeq
import Theory.Model

/-!
# A path of cliques

The nodes `0, …, n-1` are cut into `K` consecutive groups (`grp i = min (i / s) (K-1)`,
`s = n / K`). The gossip matrix `cqW n K` averages inside groups and moves weight `θ = s/4`
between consecutive groups; on group means it acts as a lazy path. It lies in `𝓦_{n,ρ}` for
`ρ ≥ √(1 - 1/(4K²))`, and its support graph joins only nodes of equal or consecutive groups.
-/

open Finset

noncomputable section

namespace ULB63
namespace Cliques

open _root_.ULB63.Path

/-- The group of node `i`. -/
def grp (n K : ℕ) (i : Fin n) : ℕ := min ((i : ℕ) / (n / K)) (K - 1)

/-- Group sizes. -/
def sz (n K g : ℕ) : ℕ := (univ.filter (fun i : Fin n => grp n K i = g)).card

/-- Group sums. -/
def gsum {n : ℕ} (K : ℕ) (v : Fin n → ℝ) (g : ℕ) : ℝ :=
  ∑ i ∈ univ.filter (fun i : Fin n => grp n K i = g), v i

/-- Group means, as a sequence. -/
def ybar {n : ℕ} (K : ℕ) (v : Fin n → ℝ) (g : ℕ) : ℝ := gsum K v g / sz n K g

/-- The inter-group weight `θ = s/4`. -/
def θ (n K : ℕ) : ℝ := ((n / K : ℕ) : ℝ) / 4

def cm (n K g : ℕ) : ℝ := if g = 0 then 0 else θ n K
def cp (n K g : ℕ) : ℝ := if g + 1 = K then 0 else θ n K

/-- Entry of the matrix between groups `g` and `h`. -/
def coef (n K g h : ℕ) : ℝ :=
  if g = h then (1 - (cm n K g + cp n K g) / sz n K g) / sz n K g
  else if g + 1 = h ∨ h + 1 = g then θ n K / (sz n K g * sz n K h) else 0

/-- The path-of-cliques gossip matrix. -/
def cqW (n K : ℕ) : Matrix (Fin n) (Fin n) ℝ := fun i j => coef n K (grp n K i) (grp n K j)

variable {n K : ℕ}

lemma grp_lt (hK : 1 ≤ K) (i : Fin n) : grp n K i < K := by
  unfold grp; have := min_le_right ((i : ℕ) / (n / K)) (K - 1); omega

lemma grp_mono {i j : Fin n} (h : (i : ℕ) ≤ j) : grp n K i ≤ grp n K j := by
  unfold grp
  exact min_le_min (Nat.div_le_div_right h) le_rfl

/-- Every group has at least `s = n / K` nodes. -/
lemma sz_ge (hK : 1 ≤ K) {g : ℕ} (hg : g < K) : n / K ≤ sz n K g := by
  set s := n / K with hs
  rcases Nat.eq_zero_or_pos s with hs0 | hspos
  · rw [hs0]; exact Nat.zero_le _
  have hsK : s * K ≤ n := Nat.div_mul_le_self n K
  have hgs : (g + 1) * s ≤ n := by
    calc (g + 1) * s ≤ K * s := Nat.mul_le_mul_right _ (by omega)
      _ = s * K := by ring
      _ ≤ n := hsK
  have hn : 0 < n := by
    have : 0 < s * K := Nat.mul_pos hspos (by omega)
    omega
  have hlt : ∀ t, t < s → g * s + t < n := by
    intro t ht
    have : g * s + t < (g + 1) * s := by rw [add_mul, one_mul]; omega
    omega
  unfold sz
  apply Finset.le_card_of_inj_on_range (fun t => (⟨(g * s + t) % n, Nat.mod_lt _ hn⟩ : Fin n))
  · intro t ht
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    show min (((g * s + t) % n) / (n / K)) (K - 1) = g
    rw [Nat.mod_eq_of_lt (hlt t ht), ← hs]
    have : (g * s + t) / s = g := by
      rw [show g * s + t = t + s * g by ring, Nat.add_mul_div_left _ _ hspos,
        Nat.div_eq_of_lt ht, zero_add]
    rw [this]; omega
  · intro a ha b hb hab
    simp only [Fin.mk.injEq] at hab
    rw [Nat.mod_eq_of_lt (hlt a ha), Nat.mod_eq_of_lt (hlt b hb)] at hab
    omega

lemma sz_pos (hK : 1 ≤ K) (hKn : K ≤ n) {g : ℕ} (hg : g < K) : 0 < sz n K g := by
  have h1 := sz_ge (n := n) hK hg
  have h2 : 1 ≤ n / K := (Nat.one_le_div_iff (by omega)).2 hKn
  omega

/-- Sums over nodes of functions of the group. -/
lemma sum_fiber (hK : 1 ≤ K) (F : ℕ → ℝ) :
    ∑ i : Fin n, F (grp n K i) = ∑ g ∈ range K, (sz n K g : ℝ) * F g := by
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range K) (g := grp n K)
    (fun i _ => Finset.mem_range.2 (grp_lt hK i))]
  refine Finset.sum_congr rfl fun g _ => ?_
  rw [Finset.sum_congr rfl fun i hi => by rw [(Finset.mem_filter.1 hi).2], Finset.sum_const,
    nsmul_eq_mul]
  rfl

lemma sum_v (hK : 1 ≤ K) (v : Fin n → ℝ) : ∑ i, v i = ∑ g ∈ range K, gsum K v g := by
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range K) (g := grp n K)
    (fun i _ => Finset.mem_range.2 (grp_lt hK i))]
  rfl

lemma sum_sz (hK : 1 ≤ K) : ∑ g ∈ range K, (sz n K g : ℝ) = n := by
  have := sum_v (n := n) hK (fun _ => (1 : ℝ))
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at this
  rw [this]
  refine Finset.sum_congr rfl fun g _ => ?_
  simp [gsum, sz]

lemma gsum_eq (hK : 1 ≤ K) (hKn : K ≤ n) (v : Fin n → ℝ) {g : ℕ} (hg : g < K) :
    gsum K v g = sz n K g * ybar K v g := by
  have := sz_pos hK hKn hg
  unfold ybar
  field_simp

/-- The action of `cqW` on a vector: group mean minus a lazy path Laplacian of the means. -/
lemma cqW_mulVec (hK : 1 ≤ K) (hKn : K ≤ n) (v : Fin n → ℝ) (i : Fin n) :
    (cqW n K).mulVec v i = ybar K v (grp n K i) -
      θ n K / sz n K (grp n K i) * lap (ybar K v) K (grp n K i) := by
  set g := grp n K i with hg
  have hgK : g < K := grp_lt hK i
  have hszg := sz_pos hK hKn hgK
  have h1 : (cqW n K).mulVec v i = ∑ h ∈ range K, coef n K g h * gsum K v h := by
    simp only [Matrix.mulVec, dotProduct, cqW]
    rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range K) (g := grp n K)
      (fun j _ => Finset.mem_range.2 (grp_lt hK j))]
    refine Finset.sum_congr rfl fun h _ => ?_
    rw [gsum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [(Finset.mem_filter.1 hj).2]
  have hsplit : ∀ h, coef n K g h * gsum K v h =
      (if g = h then (1 - (cm n K g + cp n K g) / sz n K g) / sz n K g * gsum K v h else 0) +
      (if g + 1 = h then θ n K / (sz n K g * sz n K h) * gsum K v h else 0) +
      (if h + 1 = g then θ n K / (sz n K g * sz n K h) * gsum K v h else 0) := by
    intro h
    unfold coef
    by_cases e1 : g = h
    · subst e1; simp
    · by_cases e2 : g + 1 = h
      · have e3 : ¬ (h + 1 = g) := by omega
        simp [e1, e2, e3]
      · by_cases e3 : h + 1 = g
        · simp [e1, e2, e3]
        · simp [e1, e2, e3]
  rw [h1, Finset.sum_congr rfl fun h _ => hsplit h, Finset.sum_add_distrib,
    Finset.sum_add_distrib, Finset.sum_ite_eq, if_pos (Finset.mem_range.2 hgK),
    Finset.sum_ite_eq]
  have hC : ∑ h ∈ range K, (if h + 1 = g then θ n K / (sz n K g * sz n K h) * gsum K v h else 0) =
      if g = 0 then 0 else θ n K / sz n K g * ybar K v (g - 1) := by
    by_cases hg0 : g = 0
    · rw [if_pos hg0, Finset.sum_eq_zero]
      intro h _; rw [if_neg (by omega)]
    · rw [if_neg hg0, Finset.sum_eq_single (g - 1)]
      · rw [if_pos (by omega), gsum_eq hK hKn v (by omega : g - 1 < K)]
        have := sz_pos hK hKn (by omega : g - 1 < K)
        field_simp
      · intro h _ hh; rw [if_neg (by omega)]
      · intro hh; exact absurd (Finset.mem_range.2 (by omega)) hh
  rw [hC, gsum_eq hK hKn v hgK]
  unfold lap cm cp
  by_cases hgK1 : g + 1 = K
  · have : g + 1 ∉ range K := by rw [Finset.mem_range]; omega
    rw [if_neg this]
    by_cases hg0 : g = 0
    · simp only [if_pos hg0, if_pos hgK1]
      field_simp; ring
    · simp only [if_neg hg0, if_pos hgK1]
      field_simp; ring
  · have : g + 1 ∈ range K := by rw [Finset.mem_range]; omega
    rw [if_pos this, gsum_eq hK hKn v (by omega : g + 1 < K)]
    have := sz_pos hK hKn (by omega : g + 1 < K)
    by_cases hg0 : g = 0
    · simp only [if_pos hg0, if_neg hgK1]
      field_simp; ring
    · simp only [if_neg hg0, if_neg hgK1]
      field_simp; ring

lemma lap_shift_seq (U : ℕ → ℝ) (c : ℝ) (N g : ℕ) : lap (fun h => U h - c) N g = lap U N g := by
  unfold lap; split_ifs <;> ring

lemma θ_nonneg : 0 ≤ θ n K := by unfold θ; positivity

lemma coef_symm (g h : ℕ) : coef n K g h = coef n K h g := by
  unfold coef
  by_cases e : g = h
  · subst e; rfl
  · rw [if_neg e, if_neg (Ne.symm e)]
    by_cases e2 : g + 1 = h ∨ h + 1 = g
    · rw [if_pos e2, if_pos (by tauto), mul_comm]
    · rw [if_neg e2, if_neg (by tauto)]

lemma ybar_one (hK : 1 ≤ K) (hKn : K ≤ n) {g : ℕ} (hg : g < K) :
    ybar K (fun _ : Fin n => (1 : ℝ)) g = 1 := by
  have := sz_pos hK hKn hg
  unfold ybar gsum sz
  rw [Finset.sum_const, nsmul_eq_mul, mul_one, div_self]
  unfold sz at this
  exact_mod_cast this.ne'

lemma s_pos (hK : 1 ≤ K) (hKn : K ≤ n) : 1 ≤ n / K := (Nat.one_le_div_iff (by omega)).2 hKn

/-- `cqW n K ∈ 𝓦_{n,ρ}` for `ρ ≥ √(1 - 1/(4K²))`. -/
theorem cqW_commClass (hK : 1 ≤ K) (hKn : K ≤ n) {ρ : ℝ}
    (hρ : Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ ρ) : CommClass n ρ (cqW n K) where
  nonneg i j := by
    have hgK := grp_lt hK i
    have hsz := sz_pos (n := n) hK hKn hgK
    have hsz' : ((n / K : ℕ) : ℝ) ≤ sz n K (grp n K i) := by exact_mod_cast sz_ge hK hgK
    unfold cqW coef
    split_ifs with h1 h2
    · have hcm : cm n K (grp n K i) ≤ θ n K := by unfold cm; split_ifs <;> simp [θ_nonneg]
      have hcp : cp n K (grp n K i) ≤ θ n K := by unfold cp; split_ifs <;> simp [θ_nonneg]
      have hcm0 : 0 ≤ cm n K (grp n K i) := by unfold cm; split_ifs <;> simp [θ_nonneg]
      have hcp0 : 0 ≤ cp n K (grp n K i) := by unfold cp; split_ifs <;> simp [θ_nonneg]
      have hszr : (0 : ℝ) < sz n K (grp n K i) := by exact_mod_cast hsz
      have : (cm n K (grp n K i) + cp n K (grp n K i)) / sz n K (grp n K i) ≤ 1 := by
        rw [div_le_one hszr]
        unfold θ at hcm hcp
        linarith
      apply div_nonneg (by linarith) hszr.le
    · exact div_nonneg θ_nonneg (by positivity)
    · exact le_rfl
  symm := by
    ext i j
    simp only [Matrix.transpose_apply, cqW]
    exact coef_symm _ _
  row_sum i := by
    have h := cqW_mulVec hK hKn (fun _ => (1 : ℝ)) i
    have hgK := grp_lt hK i
    have hlap : lap (ybar K (fun _ : Fin n => (1 : ℝ))) K (grp n K i) = 0 := by
      unfold lap
      rw [ybar_one hK hKn hgK]
      split_ifs with h1 h2 h2
      · ring
      · rw [ybar_one hK hKn (by omega)]; ring
      · rw [ybar_one hK hKn (by omega)]; ring
      · rw [ybar_one hK hKn (by omega), ybar_one hK hKn (by omega)]; ring
    rw [hlap, mul_zero, sub_zero, ybar_one hK hKn hgK] at h
    simpa [Matrix.mulVec, dotProduct] using h
  contraction v := by
    have hn' : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
    have hK' : (0 : ℝ) < K := by exact_mod_cast (by omega : 0 < K)
    set s : ℕ := n / K with hs
    have hs1 : 1 ≤ s := s_pos hK hKn
    have hs' : (1 : ℝ) ≤ s := by exact_mod_cast hs1
    have hθ : θ n K = (s : ℝ) / 4 := rfl
    set vbar : ℝ := (∑ j, v j) / n with hvbar
    set z : ℕ → ℝ := fun g => ybar K v g - vbar with hz
    have hJ : ∀ i, (J n).mulVec v i = vbar := by
      intro i
      simp only [J, Matrix.mulVec, dotProduct, hvbar]
      rw [← Finset.mul_sum]; field_simp
    have hcomp : ∀ i, ((cqW n K - J n).mulVec v) i =
        z (grp n K i) - θ n K / sz n K (grp n K i) * lap z K (grp n K i) := by
      intro i
      rw [Matrix.sub_mulVec, Pi.sub_apply, cqW_mulVec hK hKn, hJ, hz, lap_shift_seq]
      ring
    have hszpos : ∀ g ∈ range K, (0 : ℝ) < sz n K g := fun g hg => by
      exact_mod_cast sz_pos hK hKn (Finset.mem_range.1 hg)
    have hszge : ∀ g ∈ range K, (s : ℝ) ≤ sz n K g := fun g hg => by
      exact_mod_cast sz_ge hK (Finset.mem_range.1 hg)
    have hnorm : ∑ i, (((cqW n K - J n).mulVec v) i) ^ 2 =
        ∑ g ∈ range K, (sz n K g : ℝ) * (z g - θ n K / sz n K g * lap z K g) ^ 2 := by
      rw [← sum_fiber hK (fun g => (z g - θ n K / sz n K g * lap z K g) ^ 2)]
      exact Finset.sum_congr rfl fun i _ => by rw [hcomp]
    set E := edgeSum z K with hE
    have hE0 : 0 ≤ E := Finset.sum_nonneg fun k _ => sq_nonneg _
    have hexp : ∑ g ∈ range K, (sz n K g : ℝ) * (z g - θ n K / sz n K g * lap z K g) ^ 2 =
        ∑ g ∈ range K, (sz n K g : ℝ) * z g ^ 2 - 2 * θ n K * ∑ g ∈ range K, z g * lap z K g +
          θ n K ^ 2 * ∑ g ∈ range K, lap z K g ^ 2 / sz n K g := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun g hg => ?_
      have := hszpos g hg
      field_simp
      ring
    have hI : ∑ g ∈ range K, z g * lap z K g = E := lap_inner z K
    have hS : ∑ g ∈ range K, lap z K g ^ 2 / sz n K g ≤ 4 * E / s := by
      have h1 : ∑ g ∈ range K, lap z K g ^ 2 / sz n K g ≤ ∑ g ∈ range K, lap z K g ^ 2 / s :=
        Finset.sum_le_sum fun g hg => div_le_div_of_nonneg_left (sq_nonneg _) (by positivity)
          (hszge g hg)
      have h2 : ∑ g ∈ range K, lap z K g ^ 2 / (s : ℝ) = (∑ g ∈ range K, lap z K g ^ 2) / s := by
        rw [Finset.sum_div]
      have h3 := lap_sq_le z K
      rw [h2] at h1
      calc _ ≤ (∑ g ∈ range K, lap z K g ^ 2) / s := h1
        _ ≤ 4 * E / s := div_le_div_of_nonneg_right h3 (by positivity)
    -- the weighted Poincaré inequality
    have hsum0 : ∑ g ∈ range K, (sz n K g : ℝ) * z g = 0 := by
      have h1 : ∀ g ∈ range K, (sz n K g : ℝ) * z g = gsum K v g - sz n K g * vbar := by
        intro g hg
        rw [hz, mul_sub, ← gsum_eq hK hKn v (Finset.mem_range.1 hg)]
      rw [Finset.sum_congr rfl h1, Finset.sum_sub_distrib, ← sum_v hK, ← Finset.sum_mul,
        sum_sz hK, hvbar]
      field_simp
      ring
    have hP : ∑ g ∈ range K, (sz n K g : ℝ) * z g ^ 2 ≤ n * K * E / 2 := by
      have hid : ∑ g ∈ range K, ∑ h ∈ range K, (sz n K g : ℝ) * sz n K h * (z g - z h) ^ 2 =
          2 * n * ∑ g ∈ range K, (sz n K g : ℝ) * z g ^ 2 -
            2 * (∑ g ∈ range K, (sz n K g : ℝ) * z g) ^ 2 := by
        have : ∀ g h, (sz n K g : ℝ) * sz n K h * (z g - z h) ^ 2 =
            sz n K h * (sz n K g * z g ^ 2) + sz n K g * (sz n K h * z h ^ 2) -
              2 * ((sz n K g * z g) * (sz n K h * z h)) := fun g h => by ring
        simp only [this, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
          ← Finset.sum_mul, sum_sz hK]
        ring
      rw [hsum0] at hid
      have hle : ∑ g ∈ range K, ∑ h ∈ range K, (sz n K g : ℝ) * sz n K h * (z g - z h) ^ 2 ≤
          ∑ g ∈ range K, ∑ h ∈ range K, (sz n K g : ℝ) * sz n K h * (K * E) := by
        refine Finset.sum_le_sum fun g hg => Finset.sum_le_sum fun h hh => ?_
        have := pair_sq_le z K g h (Finset.mem_range.1 hg) (Finset.mem_range.1 hh)
        have hp : (0 : ℝ) ≤ sz n K g * sz n K h := by positivity
        exact mul_le_mul_of_nonneg_left this hp
      have hrhs : ∑ g ∈ range K, ∑ h ∈ range K, (sz n K g : ℝ) * sz n K h * (K * E) =
          (n : ℝ) ^ 2 * (K * E) := by
        simp only [← Finset.sum_mul, ← Finset.mul_sum, sum_sz hK]
        ring
      rw [hrhs] at hle
      have : 2 * (n : ℝ) * ∑ g ∈ range K, (sz n K g : ℝ) * z g ^ 2 ≤ (n : ℝ) ^ 2 * (K * E) := by
        linarith
      rw [le_div_iff₀ (by norm_num)]
      nlinarith
    -- the group means are dominated by the vector
    have hzv : ∑ g ∈ range K, (sz n K g : ℝ) * z g ^ 2 ≤ ∑ i, v i ^ 2 := by
      have hgrp : ∀ g ∈ range K, (sz n K g : ℝ) * z g ^ 2 ≤
          ∑ i ∈ univ.filter (fun i : Fin n => grp n K i = g), (v i - vbar) ^ 2 := by
        intro g hg
        have hpos := hszpos g hg
        have hcs := sq_sum_le_card_mul_sum_sq (s := univ.filter (fun i : Fin n => grp n K i = g))
          (f := fun i => v i - vbar)
        have hsumg : ∑ i ∈ univ.filter (fun i : Fin n => grp n K i = g), (v i - vbar) =
            sz n K g * z g := by
          rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
          rw [hz]
          simp only
          rw [show (∑ i ∈ univ.filter (fun i : Fin n => grp n K i = g), v i) = gsum K v g from rfl,
            gsum_eq hK hKn v (Finset.mem_range.1 hg)]
          unfold sz
          ring
        rw [hsumg] at hcs
        have : ((sz n K g : ℝ) * z g) ^ 2 = sz n K g * (sz n K g * z g ^ 2) := by ring
        rw [this] at hcs
        unfold sz at hcs ⊢
        have hpos' : (0 : ℝ) < ((univ.filter (fun i : Fin n => grp n K i = g)).card : ℝ) := by
          unfold sz at hpos; exact hpos
        exact le_of_mul_le_mul_left hcs hpos'
      have h1 := Finset.sum_le_sum hgrp
      rw [Finset.sum_fiberwise_of_maps_to (s := univ) (t := range K) (g := grp n K)
        (fun i _ => Finset.mem_range.2 (grp_lt hK i))] at h1
      have h2 : ∑ i, (v i - vbar) ^ 2 = ∑ i, v i ^ 2 - n * vbar ^ 2 := by
        have e1 : ∀ i, (v i - vbar) ^ 2 = v i ^ 2 - 2 * vbar * v i + vbar ^ 2 := fun i => by ring
        simp only [e1, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
          Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        have : ∑ i, v i = n * vbar := by rw [hvbar]; field_simp
        rw [this]; ring
      rw [h2] at h1
      nlinarith [sq_nonneg vbar]
    -- conclusion
    have hfac : 1 / (4 * (K : ℝ) ^ 2) ≤ 2 * θ n K / (n * K) := by
      rw [hθ, div_le_div_iff₀ (by positivity) (by positivity)]
      have hlt : n < K * (s + 1) := by
        have := Nat.lt_mul_div_succ n (show 0 < K by omega)
        rw [← hs] at this; exact this
      have hlt' : (n : ℝ) < K * (s + 1) := by exact_mod_cast hlt
      nlinarith
    have hkey : ∑ i, (((cqW n K - J n).mulVec v) i) ^ 2 ≤ (1 - 1 / (4 * K ^ 2)) * ∑ i, v i ^ 2 := by
      rw [hnorm, hexp, hI]
      set Z := ∑ g ∈ range K, (sz n K g : ℝ) * z g ^ 2 with hZ
      have hZ0 : 0 ≤ Z := Finset.sum_nonneg fun g hg => mul_nonneg (hszpos g hg).le (sq_nonneg _)
      have h1 : Z - 2 * θ n K * E + θ n K ^ 2 * ∑ g ∈ range K, lap z K g ^ 2 / sz n K g ≤
          Z - θ n K * E := by
        have : θ n K ^ 2 * ∑ g ∈ range K, lap z K g ^ 2 / sz n K g ≤ θ n K ^ 2 * (4 * E / s) :=
          mul_le_mul_of_nonneg_left hS (sq_nonneg _)
        have h4 : θ n K ^ 2 * (4 * E / s) = θ n K * E := by rw [hθ]; field_simp
        linarith
      have h2 : 2 * θ n K / (n * K) * Z ≤ θ n K * E := by
        rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
        have := mul_le_mul_of_nonneg_left hP (show 0 ≤ 2 * θ n K by linarith [θ_nonneg (n := n) (K := K)])
        nlinarith
      have h3 : (1 - 2 * θ n K / (n * K)) * Z ≤ (1 - 1 / (4 * K ^ 2)) * Z :=
        mul_le_mul_of_nonneg_right (by linarith) hZ0
      have h5 : 0 ≤ 1 - 1 / (4 * (K : ℝ) ^ 2) := by
        have : (1 : ℝ) ≤ K := by exact_mod_cast hK
        rw [sub_nonneg, div_le_one (by positivity)]
        nlinarith
      calc _ ≤ Z - θ n K * E := h1
        _ ≤ (1 - 2 * θ n K / (n * K)) * Z := by nlinarith
        _ ≤ (1 - 1 / (4 * K ^ 2)) * Z := h3
        _ ≤ (1 - 1 / (4 * K ^ 2)) * ∑ i, v i ^ 2 := mul_le_mul_of_nonneg_left hzv h5
    have hρ0 : 0 ≤ ρ := (Real.sqrt_nonneg _).trans hρ
    unfold norm2
    rw [show ρ * Real.sqrt (∑ i, v i ^ 2) = Real.sqrt (ρ ^ 2 * ∑ i, v i ^ 2) by
      rw [Real.sqrt_mul (sq_nonneg ρ), Real.sqrt_sq hρ0]]
    apply Real.sqrt_le_sqrt
    have hρ2 : 1 - 1 / (4 * (K : ℝ) ^ 2) ≤ ρ ^ 2 := by
      have h5 : 0 ≤ 1 - 1 / (4 * (K : ℝ) ^ 2) := by
        have : (1 : ℝ) ≤ K := by exact_mod_cast hK
        rw [sub_nonneg, div_le_one (by positivity)]
        nlinarith
      calc 1 - 1 / (4 * (K : ℝ) ^ 2) = Real.sqrt (1 - 1 / (4 * K ^ 2)) ^ 2 := (Real.sq_sqrt h5).symm
        _ ≤ ρ ^ 2 := pow_le_pow_left₀ (Real.sqrt_nonneg _) hρ 2
    calc _ ≤ (1 - 1 / (4 * K ^ 2)) * ∑ i, v i ^ 2 := hkey
      _ ≤ ρ ^ 2 * ∑ i, v i ^ 2 :=
        mul_le_mul_of_nonneg_right hρ2 (Finset.sum_nonneg fun i _ => sq_nonneg _)

/-- The support graph of `cqW` only joins nodes of equal or consecutive groups. -/
lemma cqW_ne_zero {i j : Fin n} (h : cqW n K i j ≠ 0) :
    grp n K i ≤ grp n K j + 1 ∧ grp n K j ≤ grp n K i + 1 := by
  unfold cqW coef at h
  by_cases e : grp n K i = grp n K j
  · omega
  · rw [if_neg e] at h
    by_cases e2 : grp n K i + 1 = grp n K j ∨ grp n K j + 1 = grp n K i
    · omega
    · rw [if_neg e2] at h; exact absurd rfl h

end Cliques
end ULB63
