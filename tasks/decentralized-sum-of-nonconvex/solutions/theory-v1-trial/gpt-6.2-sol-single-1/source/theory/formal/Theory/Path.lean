import Theory.Chain
import Theory.Network

open Matrix

set_option maxHeartbeats 1000000

noncomputable section

namespace ULB63
namespace Path

variable {n : ℕ}

def next (i : Fin n) : Fin n := ⟨min (i.val+1) (n-1),
  (min_le_right _ _).trans_lt (by have := i.isLt; omega)⟩

/-- Lazy walk on the path, with the missing edge at either endpoint set to a self-loop. -/
def matrix (n : ℕ) : Matrix (Fin n) (Fin n) ℝ := fun i j =>
  (if j=i then 1/2 else 0) + (if j=Chain.prev i then 1/4 else 0) +
    (if j=next i then 1/4 else 0)

theorem nonneg (i j : Fin n) : 0 ≤ matrix n i j := by
  unfold matrix
  split <;> split <;> split <;> norm_num

theorem row_sum (i : Fin n) : ∑ j, matrix n i j = 1 := by
  simp only [matrix, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  norm_num

theorem symm_entry (i j : Fin n) : matrix n i j = matrix n j i := by
  by_cases h : i=j
  · subst j; rfl
  have hji : j ≠ i := fun he => h he.symm
  have hp : j = Chain.prev i ↔ i = next j := by
    simp only [Fin.ext_iff, Chain.prev, next]
    have hi := i.isLt
    have hj := j.isLt
    have hh : i.val ≠ j.val := fun h' => h (Fin.ext h')
    omega
  have hn : j = next i ↔ i = Chain.prev j := by
    simp only [Fin.ext_iff, Chain.prev, next]
    have hi := i.isLt
    have hj := j.isLt
    have hh : i.val ≠ j.val := fun h' => h (Fin.ext h')
    omega
  simp only [matrix, hp, hn, h, hji, ite_false, zero_add]
  exact add_comm _ _

theorem support (i j : Fin n) (h : matrix n i j ≠ 0) :
    i.val ≤ j.val+1 ∧ j.val ≤ i.val+1 := by
  unfold matrix at h
  by_cases h₀ : j = i
  · subst j; omega
  by_cases h₁ : j = Chain.prev i
  · subst j; dsimp [Chain.prev]; omega
  by_cases h₂ : j = next i
  · subst j
    have hi := i.isLt
    dsimp [next]; omega
  simp [h₀, h₁, h₂] at h

theorem apply_vec (v : Fin n → ℝ) (i : Fin n) :
    (matrix n).mulVec v i = (1/2)*v i+(1/4)*v (Chain.prev i)+(1/4)*v (next i) := by
  unfold Matrix.mulVec dotProduct matrix
  simp [add_mul, ite_mul, Finset.sum_add_distrib]

private theorem sum_mix (v : Fin n → ℝ) : ∑ i, (matrix n).mulVec v i = ∑ i, v i := by
  change ∑ i, ∑ j, matrix n i j * v j = _
  rw [Finset.sum_comm]
  simp_rw [symm_entry, ← Finset.sum_mul, row_sum, one_mul]

def edgeEnergy (v : Fin n → ℝ) : ℝ := ∑ i, (v i-v (Chain.prev i))^2

theorem norm2_sq (v : Fin n → ℝ) : norm2 v ^ 2 = ∑ j, v j^2 := by
  unfold norm2
  rw [Real.sq_sqrt (Finset.sum_nonneg (fun _ _ => sq_nonneg _))]

theorem norm2_nonneg (v : Fin n → ℝ) : 0 ≤ norm2 v := Real.sqrt_nonneg _

theorem dissipate (v : Fin n → ℝ) :
    norm2 ((matrix n).mulVec v)^2 + edgeEnergy v / 8 ≤ norm2 v ^ 2 := by
  have H (i : Fin n) :
      ((matrix n).mulVec v i)^2 +
        (v i-v (Chain.prev i))^2 / 8 ≤ (matrix n).mulVec (fun j => (v j)^2) i := by
    rw [apply_vec, apply_vec]
    have h1 := sq_nonneg (v i - v (next i))
    have h2 := sq_nonneg (v (Chain.prev i) - v (next i))
    nlinarith
  have h := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin n))) (fun i _ => H i)
  rw [Finset.sum_add_distrib, ← Finset.sum_div, sum_mix] at h
  simpa only [norm2_sq, edgeEnergy] using h

private def diffsTo (n a : ℕ) : Finset (Fin n) :=
  Finset.univ.filter (fun j => 0 < j.val ∧ j.val ≤ a)

private theorem telescope (hn : 0 < n) (v : Fin n → ℝ) (a : ℕ) (ha : a < n) :
    v ⟨a,ha⟩ - v ⟨0,hn⟩ = ∑ j ∈ diffsTo n a, (v j-v (Chain.prev j)) := by
  induction a with
  | zero =>
    have he : diffsTo n 0 = ∅ := by ext x; simp [diffsTo]; omega
    rw [he, Finset.sum_empty]
    exact sub_self _
  | succ a ih =>
    let j : Fin n := ⟨a+1,ha⟩
    have ht : diffsTo n (a+1) = insert j (diffsTo n a) := by
      ext i
      simp only [diffsTo, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
        Fin.ext_iff]
      change (0 < i.val ∧ i.val ≤ a+1) ↔ i.val=a+1 ∨ 0 < i.val ∧ i.val≤a
      omega
    have hnot : j ∉ diffsTo n a := by simp [diffsTo,j]
    have hprev : Chain.prev j = ⟨a,by omega⟩ := Fin.ext (by simp [Chain.prev,j])
    rw [ht, Finset.sum_insert hnot, ← ih (by omega), hprev]
    change v j-v ⟨0,hn⟩ = (v j - v ⟨a,by omega⟩)+(v ⟨a,by omega⟩-v ⟨0,hn⟩)
    ring

theorem point_sq (hn : 0 < n) (v : Fin n → ℝ) (i : Fin n) :
    (v i-v ⟨0,hn⟩)^2 ≤ (n:ℝ)*edgeEnergy v := by
  rw [telescope hn v i.val i.isLt]
  have H := sq_sum_le_card_mul_sum_sq (s := diffsTo n i.val) (f := fun j => v j-v (Chain.prev j))
  refine H.trans (mul_le_mul ?_ ?_ (Finset.sum_nonneg (fun _ _ => sq_nonneg _)) (by positivity))
  · exact_mod_cast (show (diffsTo n i.val).card ≤ n by simpa using Finset.card_le_univ (diffsTo n i.val))
  · exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) (fun _ _ _ => sq_nonneg _)

private theorem poincare (hn : 0 < n) (v : Fin n → ℝ) (hzero : ∑ i, v i = 0) :
    norm2 v ^ 2 ≤ (n:ℝ)^2 * edgeEnergy v := by
  let z := v ⟨0,hn⟩
  have bound (i : Fin n) : (v i-z)^2 ≤ (n:ℝ)*edgeEnergy v := point_sq hn v i
  have hsum := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin n))) (fun i _ => bound i)
  have hs : ∑ i, (v i-z)^2 = ∑ i, v i^2 + (n:ℝ)*z^2 := by
    calc
      _ = ∑ i, (v i^2-2*z*v i+z^2) := Finset.sum_congr rfl fun _ _ => by ring
      _ = _ := by
        rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hzero, mul_zero, sub_zero]
        simp [Finset.sum_const, nsmul_eq_mul]
  rw [hs] at hsum
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
  rw [norm2_sq]
  nlinarith [show 0 ≤ (n:ℝ)*z^2 by positivity]

theorem commClass (hn : 0 < n) {ρ : ℝ}
    (hρ : 1-(1/(16*(n:ℝ)^2)) ≤ ρ) : CommClass n ρ (matrix n) where
  nonneg := nonneg
  symm := by ext i j; exact symm_entry _ _
  row_sum := row_sum
  contraction v := by
    have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    have hn₁ : (1 : ℝ) ≤ n := by exact_mod_cast hn
    let y := (1/(n:ℝ))*∑ i, v i
    let u : Fin n → ℝ := fun i => v i-y
    have hnorm : (n:ℝ)*y = ∑ i, v i := by dsimp [y]; field_simp
    have hz : ∑ i, u i = 0 := by
      simp only [u, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, hnorm, sub_self]
    have hbound : norm2 u ≤ norm2 v := by
      apply (sq_le_sq₀ (norm2_nonneg u) (norm2_nonneg v)).1
      rw [norm2_sq, norm2_sq]
      have H : ∑ i, (u i)^2 = ∑ i, v i^2 - (n:ℝ)*y^2 := by
        calc
          _ = ∑ i, (v i^2-2*y*v i+y^2) := Finset.sum_congr rfl fun _ _ => by dsimp [u]; ring
          _ = _ := by
            rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← hnorm]
            simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
            ring
      rw [H]
      exact sub_le_self _ (by positivity)
    have hcenter : (matrix n-J n).mulVec v = (matrix n).mulVec u := by
      ext i
      rw [Matrix.sub_mulVec]
      have hJ : (J n).mulVec v i = y := by
        change (∑ j, (1/(n:ℝ))*v j) = y
        exact (Finset.mul_sum _ _ _).symm
      simp only [Pi.sub_apply,hJ,apply_vec,u]
      ring
    have hh := dissipate u
    have hp := poincare hn u hz
    have hn₂ : 0 < (n:ℝ)^2 := by positivity
    let w := 1/(16*(n:ℝ)^2)
    have hw : 0 ≤ w := by dsimp [w]; positivity
    have hw₁ : w ≤ 1 := by
      dsimp [w]
      exact (div_le_one (by positivity)).2 (by nlinarith)
    have hh₀ : norm2 u ^ 2 / (n:ℝ)^2 ≤ edgeEnergy u := by
      rw [div_le_iff₀ hn₂, mul_comm]
      exact hp
    have he : norm2 u ^ 2 / (n:ℝ)^2 / 8 = 2*w*norm2 u ^ 2 := by dsimp [w]; ring
    have hh₁ := div_le_div_of_nonneg_right hh₀ (show (0:ℝ) ≤ 8 by norm_num)
    rw [he] at hh₁
    have hnorm' : norm2 ((matrix n).mulVec u) ≤ (1-w) * norm2 u := by
      apply (sq_le_sq₀ (norm2_nonneg _) (mul_nonneg (sub_nonneg.mpr hw₁) (norm2_nonneg u))).1
      have hres := mul_nonneg (sq_nonneg w) (sq_nonneg (norm2 u))
      nlinarith
    rw [hcenter]
    exact hnorm'.trans (le_trans (mul_le_mul_of_nonneg_left hbound (sub_nonneg.mpr hw₁))
      (mul_le_mul_of_nonneg_right hρ (norm2_nonneg _)))

end Path
end ULB63
