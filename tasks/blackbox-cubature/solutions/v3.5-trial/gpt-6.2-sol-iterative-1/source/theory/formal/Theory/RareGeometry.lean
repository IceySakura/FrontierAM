import Theory.BallModel

open Set MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal

namespace Cubature

noncomputable section

def shiftedBall {d : ℕ} (a : Vec d) (r : ℝ) : Set (Vec d) :=
  {z | ∑ i, (z i - a i)^2 ≤ r^2}

def shiftedBallMass {d : ℕ} (a : Vec d) (r : ℝ) : ℝ :=
  (gaussian d).real (shiftedBall a r)

def rareCenter {d m : ℕ} (j : Fin d → Fin m) : Vec d :=
  fun i => ((j i : ℕ) + (1 : ℝ)/2) / m

def rareOuter (m : ℕ) : ℝ := 1 / (3 * (m : ℝ))

def rareInner (d m : ℕ) : ℝ :=
  rareOuter m * (gaussianPDFReal 0 1 2 / (4 * ((d : ℝ)+1)))

def rareLowerMass (d m : ℕ) : ℝ :=
  (2 * gaussianPDFReal 0 1 2 * rareInner d m / ((d : ℝ)+1))^d

lemma pdf_two_pos : 0 < gaussianPDFReal 0 1 2 :=
  gaussianPDFReal_pos _ _ _ one_ne_zero

lemma rareOuter_pos {m : ℕ} (hm : 0 < m) : 0 < rareOuter m := by
  have hm0 : 0 < (m : ℝ) := by exact_mod_cast hm
  unfold rareOuter
  positivity

lemma rareOuter_le_one {m : ℕ} (hm : 0 < m) : rareOuter m ≤ 1 := by
  have h : (1 : ℝ) ≤ m := by exact_mod_cast hm
  unfold rareOuter
  exact (div_le_iff₀ (by positivity)).2 (by linarith)

lemma rareInner_pos {d m : ℕ} (hm : 0 < m) : 0 < rareInner d m := by
  have h := rareOuter_pos hm
  have hp := pdf_two_pos
  unfold rareInner
  positivity

lemma rareInner_le_outer {d m : ℕ} (hm : 0 < m) : rareInner d m ≤ rareOuter m := by
  unfold rareInner
  exact mul_le_of_le_one_right (rareOuter_pos hm).le
    ((div_le_iff₀ (by positivity)).2 (by linarith [standard_pdf_le_one 2]))

lemma shiftedBall_mono {d : ℕ} {a : Vec d} {s r : ℝ} (hs : 0 ≤ s) (hsr : s ≤ r) :
    shiftedBall a s ⊆ shiftedBall a r := by
  intro z hz
  exact (show ∑ i, (z i - a i)^2 ≤ s^2 from hz).trans (pow_le_pow_left₀ hs hsr 2)

lemma shiftedBall_coordinate {d : ℕ} {a z : Vec d} {r : ℝ} (hr : 0 ≤ r)
    (hz : z ∈ shiftedBall a r) (i : Fin d) : |z i-a i| ≤ r := by
  have h1 : (z i-a i)^2 ≤ r^2 :=
    (Finset.single_le_sum (fun (j : Fin d) _ => sq_nonneg (z j-a j)) (Finset.mem_univ i)).trans hz
  exact (sq_le_sq₀ (abs_nonneg _) hr).mp (by simpa [sq_abs] using h1)

private lemma interval_gaussian_lower {a w : ℝ}
    (ha : a ∈ Icc (0 : ℝ) 1) (hw : 0 ≤ w) (hw1 : w ≤ 1) :
    ENNReal.ofReal (2 * gaussianPDFReal 0 1 2 * w) ≤ gaussianReal 0 1 (Icc (a-w) (a+w)) := by
  let s := Icc (a-w) (a+w)
  rw [gaussianReal_apply_eq_integral _ one_ne_zero]
  apply ENNReal.ofReal_le_ofReal
  have hV : (volume : Measure ℝ) s ≠ ∞ := by dsimp [s]; rw [Real.volume_Icc]; exact ENNReal.ofReal_ne_top
  apply le_trans _ (setIntegral_mono_on (integrableOn_const hV) (integrable_gaussianPDFReal 0 1).integrableOn
    measurableSet_Icc (fun x hx => ?_) :
    (∫ _ in s, gaussianPDFReal 0 1 2) ≤ ∫ x in s, gaussianPDFReal 0 1 x)
  · rw [setIntegral_const, Measure.real, show s = Icc (a-w) (a+w) from rfl, Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith), smul_eq_mul]
    ring_nf
    exact le_rfl
  · have h2 : x ^ 2 ≤ 4 := by
      have hxp : -2 ≤ x := by linarith only [ha.1, hx.1, hw1]
      have hxm : x ≤ 2 := by linarith only [ha.2, hx.2, hw1]
      nlinarith only [mul_nonneg (sub_nonneg.mpr hxm) (show 0 ≤ x+2 by linarith only [hxp])]
    simp only [gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
    exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith only [h2])) (by positivity)

lemma shiftedBallMass_lower {d : ℕ} (a : Vec d) (r : ℝ)
    (ha : ∀ i, a i ∈ Icc (0 : ℝ) 1) (hr : 0 ≤ r) (hr1 : r ≤ 1) :
    (2 * gaussianPDFReal 0 1 2 * r / ((d : ℝ)+1))^d ≤ shiftedBallMass a r := by
  let w : ℝ := r/((d : ℝ)+1)
  have hw : 0 ≤ w := div_nonneg hr (by positivity)
  have hw' : w ≤ r := (div_le_iff₀ (by positivity)).2 (by nlinarith only [hr, mul_nonneg hr (Nat.cast_nonneg d)])
  have hdw : (d : ℝ)*w ≤ r := by
    unfold w
    rw [← mul_div_assoc, div_le_iff₀ (by positivity)]
    nlinarith only [hr]
  have hbox : univ.pi (fun i => Icc (a i-w) (a i+w)) ⊆ shiftedBall a r := by
    intro z hz
    simp only [mem_univ_pi, mem_Icc] at hz
    have hin (i : Fin d) : (z i-a i)^2 ≤ w^2 := by
      have h1 : 0 ≤ w - (z i-a i) := by linarith only [(hz i).2]
      have h2 : 0 ≤ (z i-a i)+w := by linarith only [(hz i).1]
      nlinarith only [mul_nonneg h1 h2]
    change ∑ i, (z i-a i)^2 ≤ r^2
    calc
      _ ≤ ∑ _ : Fin d, w^2 := Finset.sum_le_sum fun i _ => hin i
      _ = ((d : ℝ)*w)*w := by simp; ring
      _ ≤ r*r := mul_le_mul hdw hw' hw hr
      _ = r^2 := by ring
  have hp : 0 ≤ 2 * gaussianPDFReal 0 1 2 * w := by
    have ht := pdf_two_pos
    positivity
  have h1 : ENNReal.ofReal ((2 * gaussianPDFReal 0 1 2 * w)^d) ≤
      (gaussian d) (univ.pi (fun i => Icc (a i-w) (a i+w))) := by
    rw [gaussian, Measure.pi_pi, ENNReal.ofReal_pow hp]
    calc
      _ = ∏ _ : Fin d, ENNReal.ofReal (2 * gaussianPDFReal 0 1 2 * w) := by simp
      _ ≤ _ := Finset.prod_le_prod' (fun i _ => interval_gaussian_lower (ha i) hw (hw'.trans hr1))
  have h2 := ENNReal.toReal_mono (measure_ne_top _ _) (h1.trans (measure_mono hbox))
  have h3 : 0 ≤ (2 * gaussianPDFReal 0 1 2 * w)^d := pow_nonneg hp _
  simpa [shiftedBallMass, measureReal_def, ENNReal.toReal_ofReal h3, w, mul_div_assoc] using h2

lemma shiftedBallMass_upper {d : ℕ} (a : Vec d) (r : ℝ) (hr : 0 ≤ r) :
    shiftedBallMass a r ≤ (2*r)^d := by
  have hbox : shiftedBall a r ⊆ univ.pi (fun i => Icc (a i-r) (a i+r)) := by
    intro z hz
    apply mem_univ_pi.mpr
    intro i
    obtain ⟨h1,h2⟩ := abs_le.mp (shiftedBall_coordinate hr hz i)
    exact ⟨by linarith only [h1],by linarith only [h2]⟩
  have hu : (gaussian d) (univ.pi (fun i => Icc (a i-r) (a i+r))) ≤ ENNReal.ofReal ((2*r)^d) := by
    rw [gaussian, Measure.pi_pi, ENNReal.ofReal_pow (by positivity)]
    calc
      _ ≤ ∏ _ : Fin d, ENNReal.ofReal (2*r) := by
        apply Finset.prod_le_prod'
        intro i _
        have ht := gaussian_le_volume (Icc (a i-r) (a i+r))
        simpa only [Real.volume_Icc, show a i+r-(a i-r) = 2*r by ring] using ht
      _ = _ := by simp
  have ht := ENNReal.toReal_mono ENNReal.ofReal_ne_top ((measure_mono hbox).trans hu)
  simpa [shiftedBallMass, measureReal_def, ENNReal.toReal_ofReal (pow_nonneg (by positivity : 0 ≤ 2*r) _)] using ht

lemma rareCenter_mem {d m : ℕ} (hm : 0 < m) (j : Fin d → Fin m) :
    ∀ i, rareCenter j i ∈ Icc (0 : ℝ) 1 := by
  intro i
  have hm0 : 0 < (m : ℝ) := by exact_mod_cast hm
  have hj : ((j i : ℕ) : ℝ)+1 ≤ m := by exact_mod_cast Nat.succ_le_of_lt (j i).isLt
  unfold rareCenter
  exact ⟨by positivity, (div_le_iff₀ hm0).2 (by linarith only [hj])⟩

lemma rareBalls_disjoint {d m : ℕ} (hm : 0 < m) {i j : Fin d → Fin m} (hij : i ≠ j) :
    Disjoint (shiftedBall (rareCenter i) (rareOuter m))
      (shiftedBall (rareCenter j) (rareOuter m)) := by
  apply Set.disjoint_left.mpr
  intro z hi hj
  obtain ⟨k,hk⟩ : ∃ k, i k ≠ j k := by
    by_contra! h
    exact hij (funext h)
  have hm0 : 0 < (m : ℝ) := by exact_mod_cast hm
  have hdiff : (1 : ℝ) ≤ |((i k : ℕ) : ℝ) - (j k : ℕ)| := by
    rcases lt_or_gt_of_ne hk with hh | hh
    · have hh' : ((i k : ℕ) : ℝ)+1 ≤ (j k : ℕ) := by exact_mod_cast Nat.succ_le_of_lt hh
      exact le_abs.mpr (Or.inr (by linarith only [hh']))
    · have hh' : ((j k : ℕ) : ℝ)+1 ≤ (i k : ℕ) := by exact_mod_cast Nat.succ_le_of_lt hh
      exact le_abs.mpr (Or.inl (by linarith only [hh']))
  have hsep : (1 : ℝ)/m ≤ |rareCenter i k - rareCenter j k| := by
    unfold rareCenter
    rw [← sub_div, abs_div, abs_of_pos hm0,
      show ((i k : ℕ) : ℝ) + 1/2 - ((j k : ℕ) + (1 : ℝ)/2) = (i k : ℕ) - ((j k : ℕ) : ℝ) by ring]
    exact div_le_div_of_nonneg_right hdiff hm0.le
  have h0 := (rareOuter_pos hm).le
  have hrad₁ := shiftedBall_coordinate h0 hi k
  have hrad₂ := shiftedBall_coordinate h0 hj k
  have hmax : |rareCenter i k - rareCenter j k| ≤ 2 * rareOuter m := by
    rw [show rareCenter i k - rareCenter j k = -(z k-rareCenter i k) + (z k-rareCenter j k) by ring]
    exact (abs_add_le _ _).trans (by rw [abs_neg]; linarith only [hrad₁,hrad₂])
  have hc := hsep.trans hmax
  unfold rareOuter at hc
  have hh : (1 : ℝ)/(3*m)*2 = (1/(m : ℝ)) * (2/3) := by field_simp
  rw [mul_comm 2, hh] at hc
  have hinv : 0 < (1 : ℝ)/m := by positivity
  linarith only [hc, hinv]

lemma rare_inner_mass_gap {d m : ℕ} (hd : 0 < d) (hm : 0 < m) (a : Vec d)
    (ha : ∀ i, a i ∈ Icc (0 : ℝ) 1) :
    0 < shiftedBallMass a (rareOuter m) ∧
      shiftedBallMass a (rareInner d m) ≤ shiftedBallMass a (rareOuter m) / 4 := by
  have hpos := rareOuter_pos hm
  have hp := pdf_two_pos
  have outLo := shiftedBallMass_lower a (rareOuter m) ha hpos.le (rareOuter_le_one hm)
  have inHi := shiftedBallMass_upper a (rareInner d m) (rareInner_pos hm).le
  refine ⟨lt_of_lt_of_le (by positivity) outLo, inHi.trans ?_⟩
  have heq : 2*rareInner d m =
      (2 * gaussianPDFReal 0 1 2 * rareOuter m / ((d : ℝ)+1))/4 := by
    unfold rareInner
    field_simp
  rw [heq, div_pow]
  have h4 : (4 : ℝ) ≤ 4^d := by
    obtain ⟨i,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hd.ne'
    rw [pow_succ]
    have h : (1 : ℝ) ≤ 4^i := one_le_pow₀ (by norm_num)
    linarith only [h]
  exact (div_le_div_of_nonneg_left (by positivity) (by norm_num) h4).trans
    (div_le_div_of_nonneg_right outLo (by norm_num))

end

end Cubature
