import Theory.Kink

namespace Cubature

private lemma hinge_tent (a h : ℝ) (hh : 0 ≤ h) :
    max 0 (a+h) - 2*max 0 a + max 0 (a-h) = max 0 (h-|a|) := by
  rcases le_total 0 a with ha|ha
  · rw [abs_of_nonneg ha, max_eq_right ha, max_eq_right (by linarith : 0 ≤ a+h)]
    rcases le_total h a with h1|h1
    · rw [max_eq_right (sub_nonneg.mpr h1), max_eq_left (sub_nonpos.mpr h1)]
      ring
    · rw [max_eq_left (sub_nonpos.mpr h1), max_eq_right (sub_nonneg.mpr h1)]
      ring
  · rw [abs_of_nonpos ha, max_eq_left ha, max_eq_left (by linarith : a-h ≤ 0)]
    rcases le_total 0 (a+h) with h1|h1
    · rw [max_eq_right h1, max_eq_right (by linarith : 0 ≤ h- -a)]
      ring
    · rw [max_eq_left h1, max_eq_left (by linarith : h- -a ≤ 0)]
      ring

private lemma kink_center_max (f : ℝ → ℝ) (c h : ℝ) :
    (kinkCenter f c h = c-h ∨ kinkCenter f c h = c ∨ kinkCenter f c h = c+h) ∧
    |secondDiff f (c-h) h| ≤ |secondDiff f (kinkCenter f c h) h| ∧
    |secondDiff f c h| ≤ |secondDiff f (kinkCenter f c h) h| ∧
    |secondDiff f (c+h) h| ≤ |secondDiff f (kinkCenter f c h) h| := by
  let L := |secondDiff f (c-h) h|
  let M := |secondDiff f c h|
  let U := |secondDiff f (c+h) h|
  let r := kinkCenter f c h
  change (r = c-h ∨ r = c ∨ r = c+h) ∧
    L ≤ |secondDiff f r h| ∧ M ≤ |secondDiff f r h| ∧ U ≤ |secondDiff f r h|
  by_cases hl : M ≤ L
  · by_cases hu : U ≤ L
    · have he : r=c-h := by change (if M≤L then (if U≤L then c-h else c+h) else _) = _; simp [hl, hu]
      rw [he]
      exact ⟨Or.inl rfl, le_rfl, hl, hu⟩
    · have he : r=c+h := by change (if M≤L then (if U≤L then c-h else c+h) else _) = _; simp [hl, hu]
      rw [he]
      have hv := le_of_not_ge hu
      exact ⟨Or.inr (Or.inr rfl), hv, hl.trans hv, le_rfl⟩
  · have hl' := le_of_not_ge hl
    by_cases hu : U ≤ M
    · have he : r=c := by change (if M≤L then _ else (if U≤M then c else c+h)) = _; simp [hl, hu]
      rw [he]
      exact ⟨Or.inr (Or.inl rfl), hl', le_rfl, hu⟩
    · have he : r=c+h := by change (if M≤L then _ else (if U≤M then c else c+h)) = _; simp [hl, hu]
      rw [he]
      have hv := le_of_not_ge hu
      exact ⟨Or.inr (Or.inr rfl), hl'.trans hv, hv, le_rfl⟩

private lemma noise_second (noise : ℝ → ℝ) {e : ℝ}
    (hne : ∀ x, |noise x| ≤ e) (x h : ℝ) :
    |secondDiff noise x h| ≤ 4*e := by
  unfold secondDiff
  apply abs_le.mpr
  have a := abs_le.mp (hne (x+h))
  have b := abs_le.mp (hne x)
  have c := abs_le.mp (hne (x-h))
  constructor <;> linarith [a.1,a.2,b.1,b.2,c.1,c.2]

private lemma signal_second (f g noise : ℝ → ℝ) (w J : ℝ)
    (hf : ∀ x, f x = g x + J*max 0 (x-w) + noise x)
    (x h : ℝ) (hh : 0 ≤ h) :
    secondDiff f x h =
      J*max 0 (h-|x-w|) + (secondDiff g x h + secondDiff noise x h) := by
  have H : max 0 (x+h-w) - 2*max 0 (x-w) + max 0 (x-h-w) = max 0 (h-|x-w|) := by
    simpa only [show x+h-w = (x-w)+h by ring, show x-h-w = (x-w)-h by ring]
      using hinge_tent (x-w) h hh
  unfold secondDiff
  rw [hf (x+h), hf x, hf (x-h), ← H]
  ring

private lemma nearest_score (f : ℝ → ℝ) (w c h : ℝ)
    (_hh : 0 ≤ h) (hw : |w-c| ≤ 3*h/2) :
    ∃ x, |x-w| ≤ h/2 ∧ |secondDiff f x h| ≤ |secondDiff f (kinkCenter f c h) h| := by
  have range := abs_le.mp hw
  have M := (kink_center_max f c h).2
  by_cases h1 : w ≤ c-h/2
  · exact ⟨c-h, abs_le.mpr ⟨by linarith, by linarith [range.1]⟩, M.1⟩
  by_cases h2 : w ≤ c+h/2
  · exact ⟨c, abs_le.mpr ⟨by linarith, by linarith⟩, M.2.1⟩
  · exact ⟨c+h, abs_le.mpr ⟨by linarith [range.2], by linarith⟩, M.2.2⟩

lemma kink_step_bound (f g noise : ℝ → ℝ) (w J B e c h : ℝ)
    (hf : ∀ x, f x = g x + J*max 0 (x-w) + noise x)
    (hbg : ∀ x s, 0 ≤ s → |secondDiff g x s| ≤ B*s^2)
    (hne : ∀ x, |noise x| ≤ e) (hJ : J ≠ 0) (hh : 0 < h)
    (hw : |w-c| ≤ 3*h/2)
    (hrob : 8*(B*h^2+4*e) ≤ |J| * h) :
    |w-kinkCenter f c h| ≤ 3*(h/2)/2 := by
  let r := kinkCenter f c h
  let bad := B*h^2+4*e
  have ha : 0 < |J| := abs_pos.mpr hJ
  have hb (x : ℝ) :
      |secondDiff f x h - J*max 0 (h-|x-w|)| ≤ bad := by
    rw [signal_second f g noise w J hf x h hh.le, add_sub_cancel_left]
    exact (abs_add_le _ _).trans
      (add_le_add (hbg x h hh.le) (noise_second noise hne x h))
  obtain ⟨x,hx,hm⟩ := nearest_score f w c h hh.le hw
  have A (y : ℝ) : |J*max 0 (h-|y-w|)| = |J| * max 0 (h-|y-w|) := by
    rw [abs_mul, abs_of_nonneg (le_max_left 0 (h-|y-w|))]
  have triangle1 : |J| * (h/2) ≤ |secondDiff f x h|+bad := by
    calc
      _ ≤ |J| * max 0 (h-|x-w|) := mul_le_mul_of_nonneg_left
        ((by linarith : h/2 ≤ h-|x-w|).trans (le_max_right _ _)) ha.le
      _ = |J*max 0 (h-|x-w|)| := (A x).symm
      _ ≤ |secondDiff f x h|+|J*max 0 (h-|x-w|)-secondDiff f x h| := by
        calc
          _ = |secondDiff f x h+(J*max 0 (h-|x-w|)-secondDiff f x h)| := by
            congr 1
            ring
          _ ≤ _ := abs_add_le _ _
      _ ≤ _ := by rw [abs_sub_comm]; exact add_le_add le_rfl (hb x)
  have triangle2 : |secondDiff f r h| ≤ |J| * max 0 (h-|r-w|)+bad := by
    calc
      _ = |J*max 0 (h-|r-w|)+(secondDiff f r h-J*max 0 (h-|r-w|))| := by
        congr 1
        ring
      _ ≤ _ := (abs_add_le _ _).trans
        (by rw [A r]; exact add_le_add le_rfl (hb r))
  have H : |J| * (h/4) ≤ |J| * max 0 (h-|r-w|) := by
    change |secondDiff f x h| ≤ |secondDiff f r h| at hm
    linarith only [triangle1,triangle2,hm,hrob]
  have hc := (mul_le_mul_iff_right₀ ha).mp H
  have hc' : h/4 ≤ h-|r-w| := by
    by_cases p : 0 ≤ h-|r-w|
    · simpa [max_eq_right p] using hc
    · rw [max_eq_left (le_of_not_ge p)] at hc
      linarith
  rw [abs_sub_comm]
  linarith

lemma kink_search_bound (f g noise : ℝ → ℝ) (w J B e : ℝ)
    (hf : ∀ x, f x = g x + J*max 0 (x-w) + noise x)
    (hbg : ∀ x s, 0 ≤ s → |secondDiff g x s| ≤ B*s^2)
    (hne : ∀ x, |noise x| ≤ e) (hJ : J ≠ 0)
    (n : ℕ) (c h : ℝ) (hh : 0 < h) (hw : |w-c| ≤ 3*h/2)
    (hrob : ∀ j < n, let s := h/(2 : ℝ)^j; 8*(B*s^2+4*e) ≤ |J| * s) :
    let z := kinkSearch f c h n
    z.2.length = 5*n ∧ |w-z.1| ≤ 3*(h/(2 : ℝ)^n)/2 ∧
      ∀ x ∈ z.2, c-2*h ≤ x ∧ x ≤ c+2*h := by
  induction n generalizing c h with
  | zero => simpa [kinkSearch] using hw
  | succ n ih =>
    let r := kinkCenter f c h
    have hs : 0 < h/2 := by positivity
    have h0 : 8*(B*h^2+4*e) ≤ |J| * h := by
      simpa using hrob 0 (Nat.zero_lt_succ n)
    have hw' := kink_step_bound f g noise w J B e c h hf hbg hne hJ hh hw h0
    have hc : r = c-h ∨ r = c ∨ r = c+h := (kink_center_max f c h).1
    have lr : c-2*h ≤ r-2*(h/2) := by rcases hc with hc|hc|hc <;> rw [hc] <;> linarith
    have ur : r+2*(h/2) ≤ c+2*h := by rcases hc with hc|hc|hc <;> rw [hc] <;> linarith
    have e (j : ℕ) : (h/2)/(2 : ℝ)^j = h/(2 : ℝ)^(j+1) := by
      rw [div_div, pow_succ, mul_comm (2 : ℝ)]
    obtain ⟨hb,ha,hq⟩ := ih r (h/2) hs hw' (fun j hj ↦ by
      dsimp only
      rw [e j]
      exact hrob (j+1) (Nat.succ_lt_succ hj))
    simp only [kinkSearch]
    refine ⟨?_,?_,?_⟩
    · rw [List.length_append, hb, Nat.mul_succ]
      simp [kinkNodes, Nat.add_comm]
    · simpa only [e n] using ha
    · intro x hx
      rcases List.mem_append.mp hx with hnow|hlater
      · simp [kinkNodes] at hnow
        rcases hnow with (rfl|rfl|rfl|rfl|rfl) <;> constructor <;> linarith
      · obtain ⟨lo,hi⟩ := hq x hlater
        exact ⟨lr.trans lo,hi.trans ur⟩

end Cubature
