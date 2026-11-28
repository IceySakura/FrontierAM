import Theory.Acceleration
import Theory.BlockRun

/-! The local history rules and exact mixing on two connected nodes. -/

open Finset MeasureTheory
open scoped ENNReal

noncomputable section

namespace ULB63.TwoNode
open Acceleration

variable {m d : ℕ}

def other (i : Fin 2) : Fin 2 := i.rev

theorem sum_two {V : Type*} [AddCommMonoid V] (f : Fin 2 → V) (i : Fin 2) :
    ∑ j, f j = f i + f (other i) := by
  rw [Fin.sum_univ_two]
  fin_cases i
  · rfl
  · exact add_comm _ _

theorem other_ne (i : Fin 2) : other i ≠ i := by fin_cases i <;> decide

theorem other_other (i : Fin 2) : other (other i) = i := by simp [other]

theorem offdiag_ne_zero {ρ : ℝ} {W : Matrix (Fin 2) (Fin 2) ℝ}
    (hW : CommClass 2 ρ W) (hρ : ρ < 1) (i : Fin 2) : W i (other i) ≠ 0 := by
  intro hzero
  have hw₁ : W (other i) i = 0 := by
    have hh := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M i (other i)) hW.symm
    exact hh.trans hzero
  have hd₁ := hW.row_sum i
  have hd₂ := hW.row_sum (other i)
  rw [sum_two _ i, hzero, add_zero] at hd₁
  rw [sum_two _ (other i), other_other, hw₁, add_zero] at hd₂
  let v : Fin 2 → ℝ := fun j => if j = i then 1 else -1
  have hi : v i = 1 := by simp [v]
  have ho : v (other i) = -1 := by simp [v, other_ne]
  have hv : (W-J 2).mulVec v = v := by
    funext j
    have hji : j = i ∨ j = other i := by fin_cases j <;> fin_cases i <;> decide
    simp only [Matrix.mulVec, dotProduct, Matrix.sub_apply, J]
    rcases hji with rfl | rfl
    · rw [sum_two _ j, hd₁, hzero, hi, ho]
      norm_num
    · rw [sum_two _ (other i), other_other, hd₂, hw₁, hi, ho]
      norm_num
  have hp : 0 < norm2 v := by
    unfold norm2
    rw [sum_two _ i, hi, ho]
    norm_num
  have h := hW.contraction v
  rw [hv] at h
  nlinarith only [h, mul_pos (sub_pos.mpr hρ) hp]

def decodeMean (i : Fin 2) (row : Fin 2 → ℝ) (own mix : E d) : E d :=
  (1/2 : ℝ) • (own + (row (other i))⁻¹ • (mix - row i • own))

theorem decodeMean_exact {W : Matrix (Fin 2) (Fin 2) ℝ} (i : Fin 2) (h : W i (other i) ≠ 0)
    (g : Fin 2 → E d) :
    decodeMean i (W i) (g i) (∑ j, W i j • g j) = (1/2 : ℝ) • ∑ j, g j := by
  unfold decodeMean
  rw [sum_two (fun j => W i j • g j) i, sum_two g i, add_sub_cancel_left, smul_smul,
    inv_mul_cancel₀ h, one_smul]

def cycle (m : ℕ) : ℕ := m + 2

def stride (m : ℕ) : ℕ := microSteps * cycle m

def gradAt (m k ξ : ℕ) : ℕ := stride m * k + microSteps * ξ

def mixAt (m k : ℕ) : ℕ := stride m * k + microSteps * m + commSlot

def proxAt (m k : ℕ) : ℕ := stride m * k + microSteps * (m + 1)

def blockIndex (m s : ℕ) : ℕ := s / microSteps / cycle m

def phase (m s : ℕ) : ℕ := s / microSteps % cycle m

theorem schedule {m k j r : ℕ} (hj : j < cycle m) (hr : r < microSteps) :
    let s := stride m * k + microSteps * j + r
    blockIndex m s = k ∧ phase m s = j ∧ s % microSteps = r := by
  let s := stride m * k + microSteps * j + r
  have he : s = r + microSteps*(j+cycle m*k) := by unfold s stride; ring
  have h₁ : s / microSteps = j + cycle m*k := by
    rw [he, Nat.add_mul_div_left _ _ (by decide : 0 < microSteps), Nat.div_eq_of_lt hr, zero_add]
  have hc : 0 < cycle m := by unfold cycle; omega
  refine ⟨?_, ?_, ?_⟩
  · rw [blockIndex, h₁, Nat.add_mul_div_left _ _ hc, Nat.div_eq_of_lt hj, zero_add]
  · rw [phase, h₁, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hj]
  · change s % microSteps = r
    rw [he, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]

def readVec (H : List (Obs d 1)) (s : ℕ) : E d :=
  match H[s]? with
  | some (.grad g) => g
  | some (.prox p) => p
  | some (.recv r) => r 0
  | _ => 0

def xFrom {Ω : Type} (m : ℕ) (info : Info 2 d Ω) (H : List (Obs d 1)) : ℕ → E d
  | 0 => info.x0
  | k+1 => readVec H (proxAt m k)

def zFrom {Ω : Type} (m p : ℕ) (info : Info 2 d Ω) (H : List (Obs d 1)) (k : ℕ) : E d :=
  if k % p = 0 then xFrom m info H k
  else momentum (k % p - 1) (xFrom m info H (k-1)) (xFrom m info H k)

def yFrom {Ω : Type} (m p : ℕ) (info : Info 2 d Ω) (H : List (Obs d 1)) (k : ℕ) : E d :=
  extrapolate (k % p) (xFrom m info H k) (zFrom m p info H k)

def gradFrom (m : ℕ) (H : List (Obs d 1)) (k : ℕ) : E d :=
  (1/m : ℝ) • ∑ ξ : Fin m, readVec H (gradAt m k ξ)

def callRule (m p : ℕ) (L : ℝ) {Ω : Type} (info : Info 2 d Ω) (H : List (Obs d 1)) : Call m d :=
  let s := H.length
  let k := blockIndex m s
  let j := phase m s
  if s % microSteps = 0 then
    if h : j < m then .grad ⟨j, h⟩ (yFrom m p info H k)
    else if j = m + 1 then
      .prox (yFrom m p info H k -
        L⁻¹ • decodeMean info.node info.row (gradFrom m H k) (readVec H (mixAt m k))) L⁻¹
    else .idle
  else .idle

def sendRule (m : ℕ) (H : List (Obs d 1)) : E d :=
  if phase m H.length = m then gradFrom m H (blockIndex m H.length) else 0

def algorithm (m p : ℕ) (L : ℝ) : Algorithm 2 m 1 where
  Ω := Unit
  P := Measure.dirac ()
  call _ info H := callRule m p L info H
  send _ _ H _ := sendRule m H
  out _ info H := xFrom m info H (blockIndex m H.length)

instance (m p : ℕ) (L : ℝ) : Subsingleton (algorithm m p L).Ω :=
  inferInstanceAs (Subsingleton Unit)

private theorem outside_read {Ω : Type} {info : Info 2 d Ω} {H : List (Obs d 1)} {j : Fin d}
    (hj : j ∉ allowed info H) (s : ℕ) : (readVec H s) j = 0 := by
  classical
  have hh (o : Obs d 1) (ho : H[s]? = some o) : j ∉ o.supp := by
    intro hz
    exact hj (Or.inr (Set.mem_iUnion.mpr ⟨o, Set.mem_iUnion.mpr ⟨List.mem_of_getElem? ho, hz⟩⟩))
  unfold readVec
  cases hs : H[s]? with
  | none => rfl
  | some o =>
    have hz := hh o hs
    cases o with
    | idle => rfl
    | grad g => exact not_not.mp hz
    | prox p => exact not_not.mp hz
    | recv r =>
      exact not_not.mp (fun hg => hz (Set.mem_iUnion.mpr ⟨0, hg⟩))

private theorem outside_x {Ω : Type} {info : Info 2 d Ω} {H : List (Obs d 1)} {j : Fin d}
    (hj : j ∉ allowed info H) (k : ℕ) : (xFrom m info H k) j = 0 := by
  classical
  cases k with
  | zero => exact not_not.mp (fun hz => hj (Or.inl hz))
  | succ k => exact outside_read hj _

private theorem outside_y {Ω : Type} {p : ℕ} {info : Info 2 d Ω} {H : List (Obs d 1)} {j : Fin d}
    (hj : j ∉ allowed info H) (k : ℕ) : (yFrom m p info H k) j = 0 := by
  unfold yFrom extrapolate zFrom
  split <;> simp [momentum, outside_x (m := m) hj]

private theorem outside_grad {Ω : Type} {info : Info 2 d Ω} {H : List (Obs d 1)} {j : Fin d}
    (hj : j ∉ allowed info H) (k : ℕ) : (gradFrom m H k) j = 0 := by
  change ((EuclideanSpace.proj (𝕜 := ℝ) j) (gradFrom m H k)) = 0
  simp [gradFrom, map_sum, outside_read hj]

private theorem slot_zero_current_empty {c : ℕ} {H : List (Obs d c)}
    (h : H.length % microSteps = 0) : currentIter H = [] := by
  have he : microSteps * (H.length / microSteps) = H.length := by
    have hh := Nat.mod_add_div H.length microSteps
    rw [h, zero_add] at hh
    exact hh
  simp [currentIter, he]

theorem algorithm_admissible {p : ℕ} {L M μ Δ ρ : ℝ} (hL : 0 < L) :
    (algorithm m p L).Admissible M μ Δ ρ := by
  classical
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro d info H j hj
    by_contra hb
    change j ∈ (callRule m p L info H).supp at hj
    dsimp only [callRule] at hj
    split at hj
    · split at hj
      · exact hj (outside_y hb _)
      · split at hj
        · apply hj
          simp [decodeMean, outside_y hb, outside_grad hb, outside_read hb]
        · exact hj
    · exact hj
  · intro d info H k j hj
    by_contra hb
    change (sendRule m H) j ≠ 0 at hj
    unfold sendRule at hj
    split at hj
    · exact hj (outside_grad hb _)
    · exact hj rfl
  · intro d info H j hj
    by_contra hb
    exact hj (outside_x hb _)
  · intro d info H v η h
    change callRule m p L info H = _ at h
    dsimp only [callRule] at h
    split at h
    · split at h
      · contradiction
      · split at h
        · have hh := (Call.prox.inj h).2
          rw [← hh]
          positivity
        · contradiction
    · contradiction
  · intro d info H h
    change (callRule m p L info H).isGrad = false
    dsimp only [callRule]
    split
    · rename_i hs
      simp [slot_zero_current_empty hs] at h
    · rfl
  · intro d info H h
    change (callRule m p L info H).isProx = false
    dsimp only [callRule]
    split
    · rename_i hs
      simp [slot_zero_current_empty hs] at h
    · rfl
  · intros; exact Subsingleton.measurable
  · intros; exact Subsingleton.measurable
  · intros; exact Subsingleton.measurable

end ULB63.TwoNode
