import Theory.TwoNodeAlgorithm

/-! Trace verification for the two-node algorithm. -/

open Finset
open scoped ENNReal

noncomputable section

namespace ULB63

namespace Algorithm

variable {n m c d : ℕ} (A : Algorithm n m c) (I : Instance n m d)
  (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω)

theorem hist_prefix {s S : ℕ} (h : s ≤ S) (i : Fin n) : A.hist I W ω s i <+: A.hist I W ω S i := by
  induction h with
  | refl => exact List.prefix_rfl
  | step h ih => exact ih.trans (by rw [hist]; split <;> apply List.prefix_append)

end Algorithm

namespace TwoNode
open Acceleration

variable {m d p : ℕ} {L M μ Δ ρ : ℝ} (I : Instance 2 m d)
  (W : Matrix (Fin 2) (Fin 2) ℝ)

def reference (p : ℕ) (L : ℝ) (I : Instance 2 m d) : ℕ → E d × E d
  | 0 => (I.x0, I.x0)
  | k+1 =>
    let st := step I L (k % p) (reference p L I k)
    if (k+1) % p = 0 then (st.1, st.1) else st

theorem reference_succ_x (k : ℕ) :
    (reference p L I (k+1)).1 = (step I L (k%p) (reference p L I k)).1 := by
  rw [reference]
  split <;> rfl

theorem reference_epoch (hp : 0 < p) (K : ℕ) :
    reference p L I (p*K) = (restart I L p K, restart I L p K) := by
  have ha (k j : ℕ) (hj : j < p) : (p*k+j)%p = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hj]
  induction K with
  | zero => simp [reference, restart]
  | succ K ih =>
    have hb (j : ℕ) (hj : j < p) :
        reference p L I (p*K+j) = stage I L (restart I L p K) j := by
      induction j with
      | zero => simpa [stage] using ih
      | succ j ij =>
        have hlt : j < p := by omega
        have hz : (p*K+(j+1)) % p ≠ 0 := by rw [ha _ _ hj]; omega
        have hh : p*K+(j+1) = (p*K+j)+1 := by omega
        rw [hh, reference, ite_eq_right (by simpa only [← hh] using hz), ha _ _ hlt, ij hlt, stage]
    obtain ⟨j, hj⟩ := Nat.exists_eq_succ_of_ne_zero hp.ne'
    have hj' : j < p := by omega
    have hh : p*(K+1) = (p*K+j)+1 := by rw [Nat.mul_succ, hj]; omega
    have hz : ((p*K+j)+1) % p = 0 := by rw [← hh, Nat.mul_mod_right]
    rw [hh, reference, ite_eq_left hz, ha _ _ hj', hb j hj', restart, hj, stage]

open Algorithm

private theorem read_stable (ω : (algorithm m p L).Ω) (i : Fin 2)
    {s S v : ℕ} (hv : v < s) (hs : s ≤ S) :
    readVec ((algorithm m p L).hist I W ω S i) v =
      readVec ((algorithm m p L).hist I W ω s i) v := by
  let A := algorithm m p L
  have hpre := A.hist_prefix I W ω hs i
  have hv₁ : v < (A.hist I W ω s i).length := by rwa [hist_length]
  have hv₂ := hv₁.trans_le hpre.length_le
  unfold readVec
  rw [List.getElem?_eq_getElem hv₁, List.getElem?_eq_getElem hv₂, hpre.getElem hv₁]

private theorem x_stable (ω : (algorithm m p L).Ω) (i : Fin 2)
    {k s S : ℕ} (hs : stride m*k ≤ s) (hS : s ≤ S) :
    xFrom m ((algorithm m p L).info I W ω i) ((algorithm m p L).hist I W ω S i) k =
      xFrom m ((algorithm m p L).info I W ω i) ((algorithm m p L).hist I W ω s i) k := by
  cases k with
  | zero => rfl
  | succ k =>
    have hh : proxAt m k < stride m*(k+1) := by
      rw [Nat.mul_succ]
      unfold proxAt stride cycle microSteps
      omega
    exact read_stable I W ω i (hh.trans_le hs) hS

private theorem y_stable (ω : (algorithm m p L).Ω) (i : Fin 2)
    {k s S : ℕ} (hs : stride m*k ≤ s) (hS : s ≤ S) :
    yFrom m p ((algorithm m p L).info I W ω i) ((algorithm m p L).hist I W ω S i) k =
      yFrom m p ((algorithm m p L).info I W ω i) ((algorithm m p L).hist I W ω s i) k := by
  have hprev : stride m*(k-1) ≤ s := (Nat.mul_le_mul_left _ (Nat.sub_le ..)).trans hs
  simp only [yFrom, zFrom, x_stable I W ω i hs hS, x_stable I W ω i hprev hS]

private theorem entry (ω : (algorithm m p L).Ω) (i : Fin 2) {s S : ℕ} (hs : s < S) :
    let A := algorithm m p L
    (A.hist I W ω S i)[s]? =
    if s % microSteps = commSlot then
      some (.recv fun k => ∑ j, W i j • A.send d (A.info I W ω j) (A.hist I W ω s j) k)
    else some (Algorithm.answer I i (A.call d (A.info I W ω i) (A.hist I W ω s i))) := by
  let A := algorithm m p L
  have hpre := A.hist_prefix I W ω (s := s+1) hs i
  have hlo : s < (A.hist I W ω (s+1) i).length := by rw [hist_length]; omega
  have hhi := hlo.trans_le hpre.length_le
  change (A.hist I W ω S i)[s]? = _
  rw [List.getElem?_eq_getElem hhi, ← hpre.getElem hlo, ← List.getElem?_eq_getElem hlo, hist]
  split <;> rw [List.getElem?_append_right (le_of_eq (A.hist_length I W ω s i)),
    hist_length, Nat.sub_self] <;> rfl

private theorem localGrad_match (hF : ∀ i ξ, Differentiable ℝ (I.F i ξ))
    (ω : (algorithm m p L).Ω) (k : ℕ) (i : Fin 2)
    {S : ℕ} (hS : stride m*k+microSteps*m ≤ S)
    (hY : ∀ i, yFrom m p ((algorithm m p L).info I W ω i)
      ((algorithm m p L).hist I W ω (stride m*k) i) k =
      extrapolate (k%p) (reference p L I k).1 (reference p L I k).2) :
    gradFrom m ((algorithm m p L).hist I W ω S i) k =
      gradient (I.localAverage i)
        (extrapolate (k%p) (reference p L I k).1 (reference p L I k).2) := by
  let A := algorithm m p L
  let y := extrapolate (k%p) (reference p L I k).1 (reference p L I k).2
  change gradFrom m (A.hist I W ω S i) k = gradient (I.localAverage i) y
  rw [show gradient (I.localAverage i) y = _ from
    (hasGradientAt_fin_average (fun ξ => hF i ξ y)).gradient]
  unfold gradFrom
  congr 1
  apply sum_congr rfl
  intro ξ _
  let s := gradAt m k ξ
  have hlt : s < S := by
    have h := Nat.mul_lt_mul_of_pos_left ξ.isLt (by decide : 0 < microSteps)
    dsimp [s, gradAt]
    omega
  obtain ⟨hblock, hphase, hslot⟩ : blockIndex m s = k ∧ phase m s = ξ.val ∧ s%microSteps = 0 := by
    simpa [s, gradAt] using schedule (m := m) (k := k) (j := ξ.val) (r := 0)
      (by unfold cycle; omega) (by decide : 0 < microSteps)
  have hb : stride m*k ≤ s := by unfold s gradAt; omega
  have hcall : A.call d (A.info I W ω i) (A.hist I W ω s i) = .grad ξ y := by
    change callRule m p L (A.info I W ω i) (A.hist I W ω s i) = _
    simp only [callRule, hist_length, hslot, hblock, hphase, ite_true, dite_eq_left ξ.isLt]
    rw [y_stable I W ω i le_rfl hb, hY i]
  change readVec (A.hist I W ω S i) s = _
  unfold readVec
  rw [entry I W ω i hlt, hslot, ite_eq_right (by decide : 0 ≠ commSlot), hcall]
  rfl

private theorem mod_succ_sub_one {k p : ℕ} (hp : 0 < p) (hz : (k+1)%p ≠ 0) :
    (k+1)%p-1 = k%p := by
  have he : (k+1)%p = (k%p+1)%p := by
    conv_lhs => rw [Nat.add_mod]
    conv_rhs => rw [Nat.add_mod, Nat.mod_mod]
  have hlt : k%p+1 < p := by
    have ha := Nat.mod_lt k hp
    by_contra! hc
    have hx : k%p+1 = p := by omega
    rw [he, hx, Nat.mod_self] at hz
    exact hz rfl
  rw [he, Nat.mod_eq_of_lt hlt, Nat.add_sub_cancel]

private theorem y_matches {k : ℕ} {Ω : Type} (info : Info 2 d Ω)
    (H : List (Obs d 1)) (hp : 0 < p)
    (hx : xFrom m info H k = (reference p L I k).1)
    (hpast : xFrom m info H (k-1) = (reference p L I (k-1)).1) :
    yFrom m p info H k = extrapolate (k%p) (reference p L I k).1 (reference p L I k).2 := by
  unfold yFrom zFrom
  rw [hx, hpast]
  congr 1
  cases k with
  | zero => simp [reference]
  | succ k =>
    by_cases hz : (k+1)%p = 0
    · rw [ite_eq_left hz, reference, ite_eq_left hz]
    · rw [ite_eq_right hz, Nat.add_sub_cancel, mod_succ_sub_one hp hz, reference_succ_x,
        reference, ite_eq_right hz]
      rfl

private theorem cycle_step_matches (hF : ∀ i ξ, Differentiable ℝ (I.F i ξ))
    (hW : ∀ i, W i (other i) ≠ 0)
    (ω : (algorithm m p L).Ω) (k : ℕ)
    (hY : ∀ i, yFrom m p ((algorithm m p L).info I W ω i)
      ((algorithm m p L).hist I W ω (stride m*k) i) k =
      extrapolate (k%p) (reference p L I k).1 (reference p L I k).2) (i : Fin 2) :
    xFrom m ((algorithm m p L).info I W ω i)
      ((algorithm m p L).hist I W ω (stride m*(k+1)) i) (k+1) =
      (reference p L I (k+1)).1 := by
  let A := algorithm m p L
  let y := extrapolate (k%p) (reference p L I k).1 (reference p L I k).2
  let g := fun i => gradient (I.localAverage i) y
  let s := mixAt m k
  let t := proxAt m k
  have hS : stride m*k + microSteps*m ≤ s := by unfold s mixAt; omega
  have fst : s < t := by
    unfold s t mixAt proxAt commSlot microSteps
    omega
  have hlast : t < stride m*(k+1) := by
    rw [Nat.mul_succ]
    unfold t proxAt stride cycle microSteps
    omega
  obtain ⟨hbs, hps, hss⟩ : blockIndex m s = k ∧ phase m s = m ∧ s%microSteps = commSlot := by
    exact schedule (m := m) (k := k) (j := m) (r := commSlot)
      (by unfold cycle; omega) (by decide : commSlot < microSteps)
  have hg (j : Fin 2) (a : Fin 1) : A.send d (A.info I W ω j) (A.hist I W ω s j) a = g j := by
    change sendRule m (A.hist I W ω s j) = _
    rw [sendRule, hist_length, hps, ite_eq_left rfl, hbs]
    exact localGrad_match I W hF ω k j hS hY
  have hmix : readVec (A.hist I W ω t i) s = ∑ j, W i j • g j := by
    unfold readVec
    rw [entry I W ω i fst, ite_eq_left hss]
    change (∑ j, W i j • A.send d (A.info I W ω j) (A.hist I W ω s j) 0) = _
    simp only [hg]
  obtain ⟨hbt, hpt, hst⟩ : blockIndex m t = k ∧ phase m t = m+1 ∧ t%microSteps = 0 := by
    simpa only [t, proxAt, add_zero] using schedule (m := m) (k := k) (j := m+1) (r := 0)
      (by unfold cycle; omega) (by decide : 0 < microSteps)
  have hgrad : gradient I.f y = (1/2 : ℝ) • ∑ j, g j := by
    have hloc (j : Fin 2) : DifferentiableAt ℝ (I.localAverage j) y :=
      (hasGradientAt_fin_average (fun ξ => hF j ξ y)).differentiableAt
    change gradient (fun x => (1 / ((2 : ℕ) : ℝ)) * ∑ j : Fin 2, I.localAverage j x) y = _
    simpa only [Nat.cast_ofNat, g] using (hasGradientAt_fin_average hloc).gradient
  have htbase : stride m*k ≤ t := by unfold t proxAt; omega
  have hcall : A.call d (A.info I W ω i) (A.hist I W ω t i) = .prox (y-L⁻¹ • gradient I.f y) L⁻¹ := by
    change callRule m p L (A.info I W ω i) (A.hist I W ω t i) = _
    have hn : ¬ (m+1 < m) := by omega
    simp only [callRule, hist_length, hbt, hpt, hst, ite_true, dite_eq_right hn]
    have hd : gradFrom m (A.hist I W ω t i) k = g i :=
      localGrad_match I W hF ω k i (hS.trans fst.le) hY
    rw [hd, y_stable I W ω i le_rfl htbase, hY i, hmix]
    change Call.prox (y-L⁻¹ • decodeMean i (W i) (g i) (∑ j, W i j • g j)) L⁻¹ = _
    rw [decodeMean_exact i (hW i), hgrad]
  rw [reference_succ_x I k]
  change readVec (A.hist I W ω (stride m*(k+1)) i) t = _
  unfold readVec
  rw [entry I W ω i hlast, hst, ite_eq_right (by decide : 0 ≠ commSlot), hcall]
  rfl

theorem run_matches (hp : 0 < p) (hF : ∀ i ξ, Differentiable ℝ (I.F i ξ))
    (hW : ∀ i, W i (other i) ≠ 0) (ω : (algorithm m p L).Ω) (k : ℕ) (i : Fin 2) :
    xFrom m ((algorithm m p L).info I W ω i)
      ((algorithm m p L).hist I W ω (stride m*k) i) k = (reference p L I k).1 := by
  induction k using Nat.strong_induction_on generalizing i with
  | h k ih =>
    cases k with
    | zero => rfl
    | succ k =>
      apply cycle_step_matches I W hF hW
      intro j
      apply y_matches I _ _ hp (ih k (by omega) j)
      have hle : stride m*(k-1) ≤ stride m*k := Nat.mul_le_mul_left _ (Nat.sub_le ..)
      rw [x_stable I W ω j le_rfl hle]
      exact ih (k-1) (by omega) j

theorem averaged_output_matches (hp : 0 < p) (hF : ∀ i ξ, Differentiable ℝ (I.F i ξ))
    (hW : ∀ i, W i (other i) ≠ 0) (ω : (algorithm m p L).Ω) (k : ℕ) :
    (algorithm m p L).avgOutput I W ω (cycle m*k) = (reference p L I k).1 := by
  let A := algorithm m p L
  have ht : microSteps * (cycle m*k) = stride m*k := by unfold stride; ring
  have hb : blockIndex m (stride m*k) = k := by
    simpa only [mul_zero, add_zero] using
      (schedule (m := m) (k := k) (j := 0) (r := 0) (by unfold cycle; omega)
        (by decide : 0 < microSteps)).1
  change (1/2 : ℝ) • ∑ i, A.output I W ω (cycle m*k) i = _
  have hx (i : Fin 2) : A.output I W ω (cycle m*k) i = (reference p L I k).1 := by
    unfold Algorithm.output
    rw [ht]
    change xFrom m (A.info I W ω i) (A.hist I W ω (stride m*k) i)
      (blockIndex m (A.hist I W ω (stride m*k) i).length) = _
    rw [hist_length, hb]
    exact run_matches I W hp hF hW ω k i
  simp only [hx, Fin.sum_univ_two]
  module

end TwoNode

end ULB63
