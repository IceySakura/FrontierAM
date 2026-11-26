import Theory.Recovery
import Theory.ValueTree

open Set
open scoped Pointwise

namespace Cubature

noncomputable def probePoint {d : ℕ} (ρ : ℝ) (u : Point d) (s : ℝ) : Point d :=
  (ρ/s) • u

theorem probe_allowed {d : ℕ} {ρ : ℝ} (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20)
    {u : Point d} (hu : ∀ i, |u i| ≤ 1) {s : ℝ} (hs : 1/2 ≤ s) :
    allowed (probePoint ρ u s) := by
  have hs0 : 0 < s := by linarith
  have hl : ρ/s ≤ 40 := (div_le_iff₀ hs0).2 (by linarith)
  intro i
  simp only [probePoint,Pi.smul_apply,smul_eq_mul,abs_mul,
    abs_of_nonneg (div_nonneg hρ hs0.le)]
  exact (mul_le_mul_of_nonneg_left (hu i) (div_nonneg hρ hs0.le)).trans (by simpa using hl)

theorem probe_difference {d : ℕ} {ρ : ℝ} (hρ : 0 ≤ ρ) {u : Point d}
    (hu : ∀ i, |u i| ≤ 1) {s t : ℝ} (hs : 1/2 ≤ s) (ht : 1/2 ≤ t) (i : Fin d) :
    |probePoint ρ u t i - probePoint ρ u s i| ≤ 4*ρ*|t-s| := by
  have hs' : 0 < s := by linarith
  have ht' : 0 < t := by linarith
  have hz : 0 < t*s := mul_pos ht' hs'
  have htS : 1/4 ≤ t*s := by nlinarith [mul_le_mul ht hs (by norm_num : (0:ℝ) ≤ 1/2) ht'.le]
  have hh : probePoint ρ u t i - probePoint ρ u s i = (ρ*(s-t)/(t*s))*u i := by
    simp only [probePoint,Pi.smul_apply,smul_eq_mul]
    field_simp
    <;> ring
  rw [hh,abs_mul,abs_div,abs_mul,abs_of_nonneg hρ,abs_of_pos hz,abs_sub_comm s t]
  apply (mul_le_mul_of_nonneg_left (hu i) (div_nonneg (mul_nonneg hρ (abs_nonneg _)) hz.le)).trans
  rw [mul_one]
  apply (div_le_iff₀ hz).2
  have hbase : 1 ≤ 4*(t*s) := by linarith
  have H := mul_le_mul_of_nonneg_right hbase (mul_nonneg hρ (abs_nonneg (t-s)))
  nlinarith only [H]

theorem payoff_as_step {d : ℕ} {ρ : ℝ} (hρ : 0 < ρ) (E : Ellipsoid d) (u : Point d)
    (hu : 0 < quadratic E.A u) (h : Point d → ℝ)
    {r : ℝ} (hr : 0 < r) (he : r^2=quadratic E.A u + linear E.b u*r)
    {s : ℝ} (hs : 0 < s) :
    augmentedPayoff ρ E h (probePoint ρ u s) =
      h (probePoint ρ u s)+(if r<s then 1 else 0) := by
  unfold augmentedPayoff payoff probePoint
  have H := inverse_radius_answer hρ.ne' E u hu hr he s hs
  by_cases ha : r < s
  · rw [ite_eq_left (H.2 ha),ite_eq_left ha]
  · rw [ite_eq_right (fun hv => ha (H.1 hv)),ite_eq_right ha]

noncomputable def gridRadius (k j : ℕ) : ℝ :=
  1/2 + (7/2)*((j:ℝ)/k)

theorem grid_left (k : ℕ) : gridRadius k 0 = 1/2 := by simp [gridRadius]

theorem grid_right {k : ℕ} (hk : 0 < k) : gridRadius k k = 4 := by
  have hk0 : (k:ℝ) ≠ 0 := by exact_mod_cast hk.ne'
  unfold gridRadius
  rw [div_self hk0]
  norm_num

theorem grid_order {k i j : ℕ} (h : i ≤ j) : gridRadius k i ≤ gridRadius k j := by
  unfold gridRadius
  gcongr

theorem grid_mem {k j : ℕ} (hk : 0 < k) (hj : j ≤ k) :
    gridRadius k j ∈ Icc (1/2 : ℝ) 4 := by
  refine ⟨?_,? _⟩
  · rw [← grid_left k]
    exact grid_order (Nat.zero_le _)
  · rw [← grid_right hk]
    exact grid_order hj

theorem grid_step {k : ℕ} (hk : 0 < k) (j : ℕ) :
    gridRadius k (j+1)-gridRadius k j ≤ 4/(k:ℝ) := by
  have hk' : (0:ℝ)<k := by exact_mod_cast hk
  have h : gridRadius k (j+1)-gridRadius k j = (7/2 : ℝ)/(k:ℝ) := by
    simp only [gridRadius,Nat.cast_succ,div_eq_mul_inv]
    ring
  rw [h]
  exact div_le_div_of_nonneg_right (by norm_num) hk'.le

theorem grid_bracket {k : ℕ} (hk : 0 < k) {r : ℝ} (h1 : 1/2 ≤ r) (h2 : r < 4) :
    ∃ j : Fin k, gridRadius k j.val ≤ r ∧ r < gridRadius k (j.val+1) := by
  classical
  have ex : ∃ n : ℕ, r < gridRadius k n := ⟨k,by simpa [grid_right hk] using h2⟩
  let n := Nat.find ex
  have hn : r < gridRadius k n := Nat.find_spec ex
  have hk' : n ≤ k := Nat.find_min' ex (by simpa [grid_right hk] using h2)
  have hn0 : n ≠ 0 := by
    intro H
    rw [H,grid_left] at hn
    linarith
  have hm : n-1<n := by omega
  have hj : n-1<k := hm.trans_le hk'
  refine ⟨⟨n-1,hj⟩,le_of_not_gt (Nat.find_min ex hm),?_⟩
  change r < gridRadius k (n-1+1)
  simpa [Nat.sub_add_cancel (by omega : 1 ≤ n)] using hn

theorem inverse_radius_lt_four {d : ℕ} (E : Ellipsoid d) (u : Point d)
    (hu : normSq u ≤ 2) {r : ℝ} (hr : r ≤ 4)
    (he : r^2 = quadratic E.A u + linear E.b u*r) : r < 4 := by
  by_contra! h
  have h' : r=4 := le_antisymm hr h
  rw [h'] at he
  have hA := E.high u
  have hb := (abs_le.mp (E.bias u)).2
  norm_num at he
  linarith

theorem local_step_test {d k : ℕ} (hk : 0 < k)
    {ρ L : ℝ} (hρ : 0 < ρ) (hL : 0 ≤ L) (hscale : 64*L*ρ ≤ k)
    (E : Ellipsoid d) (u : Point d) (hu : ∀ i, |u i| ≤ 1) (hq : 0 < quadratic E.A u)
    (h : Point d → ℝ) (hlip : CoordinateLipschitz L h) {r : ℝ} (hr : 0 < r)
    (he : r^2 = quadratic E.A u + linear E.b u*r) {s t : ℝ}
    (hs : 1/2 ≤ s) (hst : s ≤ t) (hmesh : t-s ≤ 4/(k:ℝ)) :
    1/2 < augmentedPayoff ρ E h (probePoint ρ u t) - augmentedPayoff ρ E h (probePoint ρ u s)
      ↔ s ≤ r ∧ r < t := by
  have hs0 : 0 < s := by linarith
  have ht0 := hs0.trans_le hst
  have HK : 0 < (k:ℝ) := by exact_mod_cast hk
  have h0 : 0 ≤ 4*ρ*|t-s| := by positivity
  have HL := hlip (probePoint ρ u t) (probePoint ρ u s) _ h0
    (probe_difference hρ.le hu hs (hs.trans hst))
  have HB : L*(4*ρ*|t-s|) ≤ 1/4 := by
    rw [abs_of_nonneg (sub_nonneg.mpr hst)]
    have hm := mul_le_mul_of_nonneg_left hmesh (show 0 ≤ L*4*ρ by positivity)
    have HS : L*4*ρ*(4/(k:ℝ)) ≤ 1/4 := by
      rw [← mul_div_assoc,div_le_iff₀ HK]
      linarith only [hscale]
    nlinarith only [hm,HS]
  have H := abs_le.mp (HL.trans HB)
  rw [payoff_as_step hρ E u hq h hr he ht0, payoff_as_step hρ E u hq h hr he hs0]
  by_cases h1 : r<s
  · have h2 := h1.trans_le hst
    simp [h1,h2,h1.not_ge] <;> linarith [H.1,H.2]
  · by_cases h2 : r<t
    · simp [h1,h2,le_of_not_gt h1] <;> linarith [H.1,H.2]
    · simp [h1,h2] <;> linarith [H.1,H.2]

/-- Use the saved lower-end value to interpret one real midpoint query. -/
noncomputable def valueBisect {d : ℕ} (ρ : ℝ) (u : Point d) :
    ℕ → ℝ → ℝ → ℝ → ValueTree (Point d) ℝ
  | 0, l, h, _ => .ret ((l+h)/2)
  | m+1, l, h, y =>
    let a := (l+h)/2
    .ask (probePoint ρ u a) (fun v =>
      if v-y>1/2 then valueBisect ρ u m l a y else valueBisect ρ u m a h v)

theorem valueBisect_within {d : ℕ} (ρ : ℝ) (u : Point d)
    (m : ℕ) (l h y : ℝ) : (valueBisect ρ u m l h y).within m := by
  induction m generalizing l h y with
  | zero => trivial
  | succ m ih =>
    intro v
    simp only [valueBisect] at *
    split <;> apply ih

theorem valueBisect_allowed {d : ℕ} {ρ : ℝ} (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20)
    {u : Point d} (hu : ∀ i, |u i| ≤ 1) (m : ℕ)
    {l h : ℝ} (hl : 1/2 ≤ l) (hlh : l ≤ h) (y : ℝ) :
    (valueBisect ρ u m l h y).admissible allowed := by
  induction m generalizing l h y with
  | zero => trivial
  | succ m ih =>
    let a := (l+h)/2
    have ha : l ≤ a := by dsimp [a]; linarith
    have ha' : a ≤ h := by dsimp [a]; linarith
    refine ⟨probe_allowed hρ hρ' hu (hl.trans ha), ?_⟩
    intro v
    change (if v-y>1/2 then valueBisect ρ u m l a y
      else valueBisect ρ u m a h v).admissible allowed
    split
    · exact ih hl ha y
    · exact ih (hl.trans ha) ha' v

theorem valueBisect_eq {d : ℕ} (ρ : ℝ) (u : Point d) (f : Point d → ℝ)
    (bit : Point d → Bool) (r W : ℝ)
    (hb : ∀ s > 0, bit (probePoint ρ u s) = true ↔ r < s)
    (hgap : ∀ l a : ℝ, 1/2 ≤ l → l ≤ a → a-l ≤ W →
      (1/2 < f (probePoint ρ u a)-f (probePoint ρ u l) ↔ l ≤ r ∧ r < a))
    (m : ℕ) {l h : ℝ} (hl : 1/2 ≤ l) (hlr : l ≤ r) (hrh : r ≤ h) (hw : h-l ≤ W) :
    (valueBisect ρ u m l h (f (probePoint ρ u l))).run f = (bisect ρ u m l h).run bit := by
  induction m generalizing l h with
  | zero => rfl
  | succ m ih =>
    let a := (l+h)/2
    have ha : l ≤ a := by dsimp [a]; linarith
    have ha' : a ≤ h := by dsimp [a]; linarith
    have ha0 : 0 < a := by linarith
    have hg := hgap l a hl ha (by linarith)
    have hs : 1/2 < f (probePoint ρ u a)-f (probePoint ρ u l) ↔ r < a :=
      hg.trans (and_iff_right hlr)
    change (if f (probePoint ρ u a)-f (probePoint ρ u l)>1/2 then
        valueBisect ρ u m l a (f (probePoint ρ u l))
      else valueBisect ρ u m a h (f (probePoint ρ u a))).run f =
      (if bit (probePoint ρ u a) then (bisect ρ u m l a).run bit
      else (bisect ρ u m a h).run bit)
    by_cases h : r < a
    · rw [ite_eq_left (hs.2 h),ite_eq_left ((hb a ha0).2 h)]
      exact ih hl hlr h.le (by linarith)
    · have hn : bit (probePoint ρ u a) ≠ true := fun he => h ((hb a ha0).1 he)
      rw [ite_eq_right (fun he => h (hs.1 he)),ite_eq_right hn]
      exact ih (hl.trans ha) (le_of_not_gt h) hrh (by linarith)

/-- A scanner only reads the fixed `k+1` grid points. -/
noncomputable def rayScan {d : ℕ} (ρ : ℝ) (u : Point d) (k : ℕ) :
    ValueTree (Point d) (Fin (k+1) → ℝ) :=
  ValueTree.collect (fun i : Fin (k+1) => ValueTree.observe (probePoint ρ u (gridRadius k i.val)))
    (Finset.univ : Finset (Fin (k+1))).toList

noncomputable def detectedCell {k : ℕ} (hk : 0 < k) (y : Fin (k+1) → ℝ) : Fin k :=
  if h : ∃ i : Fin k, 1/2 < y i.succ-y i.castSucc then Classical.choose h else ⟨0,hk⟩

noncomputable def learnRay {d k : ℕ} (ρ : ℝ) (u : Point d) (hk : 0 < k) (m : ℕ) :
    ValueTree (Point d) ℝ :=
  (rayScan ρ u k).bind fun y =>
    let j := detectedCell hk y
    valueBisect ρ u m (gridRadius k j.val) (gridRadius k (j.val+1)) (y j.castSucc)

theorem learnRay_within {d k : ℕ} (ρ : ℝ) (u : Point d) (hk : 0 < k) (m : ℕ) :
    (learnRay ρ u hk m).within (k+1+m) := by
  unfold learnRay
  apply ValueTree.within_bind
  · unfold rayScan
    have H := ValueTree.within_collect
      (fun i : Fin (k+1) => ValueTree.observe (probePoint ρ u (gridRadius k i.val)))
      (Finset.univ : Finset (Fin (k+1))).toList 1 (fun _ => fun _ => trivial)
    simpa using H
  · exact fun _ => valueBisect_within _ _ _ _ _ _

theorem learnRay_allowed {d k : ℕ} {ρ : ℝ} (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20)
    {u : Point d} (hu : ∀ i, |u i| ≤ 1) (hk : 0 < k) (m : ℕ) :
    (learnRay ρ u hk m).admissible allowed := by
  unfold learnRay
  apply ValueTree.admissible_bind
  · unfold rayScan
    apply ValueTree.admissible_collect
    intro i
    exact ⟨probe_allowed hρ hρ' hu (grid_mem hk (Nat.le_of_lt_succ i.isLt)).1,fun _ => trivial⟩
  · intro y
    exact valueBisect_allowed hρ hρ' hu m (grid_mem hk (Nat.le_of_lt (detectedCell hk y).isLt)).1
      (grid_order (Nat.le_succ _)) _

theorem learnRay_correct {d k : ℕ} (hk : 0 < k) {ρ L : ℝ}
    (hρ : 0 < ρ) (hL : 0 ≤ L) (hscale : 64*L*ρ ≤ k)
    (E : Ellipsoid d) (u : Point d) (hu : ∀ i, |u i| ≤ 1)
    (hn : 1 ≤ normSq u ∧ normSq u ≤ 2) (h : Point d → ℝ) (hlip : CoordinateLipschitz L h)
    (m : ℕ) {r : ℝ} (hr : r ∈ Icc (1/2 : ℝ) 4)
    (he : r^2=quadratic E.A u + linear E.b u*r) :
    (learnRay ρ u hk m).run (augmentedPayoff ρ E h) ∈ Icc (1/2:ℝ) 4 ∧
    |(learnRay ρ u hk m).run (augmentedPayoff ρ E h) - r| ≤ 2/(2:ℝ)^m := by
  let f := augmentedPayoff ρ E h
  let Y := (rayScan ρ u k).run f
  have hy (i : Fin (k+1)) : Y i = f (probePoint ρ u (gridRadius k i.val)) :=
    ValueTree.run_collect _ _ _ (by simp)
  have hr0 : 0 < r := by linarith [hr.1]
  have hq : 0 < quadratic E.A u := by linarith [hn.1,E.low u]
  have hg (s t : ℝ) := local_step_test hk hρ hL hscale E u hu hq h hlip hr0 he (s := s) (t := t)
  have hobs (j : Fin k) : 1/2 < Y j.succ-Y j.castSucc ↔
      gridRadius k j.val ≤ r ∧ r < gridRadius k (j.val+1) := by
    simp only [hy,Fin.val_succ,Fin.val_castSucc]
    exact hg _ _ (grid_mem hk (Nat.le_of_lt j.isLt)).1
      (grid_order (Nat.le_succ _)) (grid_step hk _)
  obtain ⟨j,hj⟩ := grid_bracket hk hr.1 (inverse_radius_lt_four E u hn.2 hr.2 he)
  have he0 : ∃ i : Fin k, 1/2 < Y i.succ - Y i.castSucc := ⟨j,(hobs j).2 hj⟩
  let i := detectedCell hk Y
  have hh : gridRadius k i.val ≤ r ∧ r < gridRadius k (i.val+1) := by
    apply (hobs i).mp
    unfold i detectedCell
    rw [dif_pos he0]
    exact Classical.choose_spec he0
  have Hi := grid_mem hk (Nat.le_of_lt i.isLt)
  have Hi' := grid_mem hk (Nat.succ_le_of_lt i.isLt)
  have hb (s : ℝ) (hs : 0 < s) :
      answerBit (region ρ E) (probePoint ρ u s) = true ↔ r < s :=
    inverse_radius_answer hρ.ne' E u hq hr0 he s hs
  have Hrun : (learnRay ρ u hk m).run f =
      (bisect ρ u m (gridRadius k i.val) (gridRadius k (i.val+1))).run (answerBit (region ρ E)) := by
    rw [learnRay,ValueTree.run_bind]
    change (valueBisect ρ u m (gridRadius k i.val) (gridRadius k (i.val+1)) (Y i.castSucc)).run f = _
    rw [hy]
    exact valueBisect_eq ρ u f _ r _ hb hg m Hi.1 hh.1 hh.2.le (grid_step hk _)
  rw [Hrun]
  have R := bisect_correct ρ u _ r hb m (by linarith [Hi.1]) hh.1 hh.2.le
  refine ⟨⟨Hi.1.trans R.1.1,R.1.2.trans Hi'.2⟩,R.2.trans ?_⟩
  rw [pow_succ,mul_comm ((2:ℝ)^m) (2:ℝ),← div_div]
  exact div_le_div_of_nonneg_right (by linarith [Hi.1,Hi'.2] :
    (gridRadius k (i.val+1)-gridRadius k i.val)/2 ≤ 2) (pow_nonneg (by norm_num) _)

end Cubature
