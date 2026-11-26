import Theory.Geometry

open MeasureTheory Set
open scoped BigOperators Pointwise ENNReal

namespace Cubature

theorem probability_pos_of_open {d : ℕ} {s : Set (Point d)}
    (hs : IsOpen s) (hne : s.Nonempty) : 0 < probability s := by
  have hvol := hs.measure_ne_zero (μ := (volume : Measure (Point d))) hne
  have hG : gaussian d s ≠ 0 := fun h ↦ hvol (volume_absolutelyContinuous_gaussian d h)
  exact ENNReal.toReal_pos hG (measure_ne_top _ _)

theorem arithmeticBall_mono {d : ℕ} {r s : ℝ} (hr : 0 ≤ r) (hrs : r ≤ s) :
    arithmeticBall (d := d) r ⊆ arithmeticBall s := by
  intro x hx
  change normSq x < r^2 at hx
  exact hx.trans_le (pow_le_pow_left₀ hr hrs _)

theorem ballMass_mono {d : ℕ} {r s : ℝ} (hr : 0 ≤ r) (hrs : r ≤ s) :
    ballMass d r ≤ ballMass d s := probability_mono (arithmeticBall_mono hr hrs)

theorem ballMass_lt {d : ℕ} (hd : 0 < d) {r s : ℝ} (hr : 0 ≤ r) (hrs : r < s) :
    ballMass d r < ballMass d s := by
  let shell : Set (Point d) := {x | r ^ 2 < normSq x ∧ normSq x < s ^ 2}
  have hS : IsOpen shell :=
    (isOpen_lt continuous_const continuous_normSq).inter (isOpen_lt continuous_normSq continuous_const)
  have hne : shell.Nonempty := by
    let i : Fin d := ⟨0,hd⟩
    have hrs0 : 0 ≤ (r+s)/2 := by linarith
    refine ⟨((r+s)/2) • axis i, ?_⟩
    change r^2 < normSq (((r+s)/2) • axis i) ∧ normSq (((r+s)/2) • axis i) < s^2
    rw [normSq_smul, normSq_axis, mul_one]
    exact ⟨pow_lt_pow_left₀ (by linarith) hr (by decide),
      pow_lt_pow_left₀ (by linarith) hrs0 (by decide)⟩
  have hsub : shell ⊆ arithmeticBall s \ arithmeticBall r :=
    fun _ h ↦ ⟨h.2, fun hh ↦ (not_lt_of_ge h.1.le) hh⟩
  have hpos := (probability_pos_of_open hS hne).trans_le (probability_mono hsub)
  unfold probability at hpos
  rw [measureReal_sdiff (arithmeticBall_mono hr hrs.le)
    (isOpen_arithmeticBall r).measurableSet] at hpos
  exact sub_pos.mp hpos

theorem dilate_arithmeticBall {d : ℕ} (r : ℝ) {t : ℝ} (ht : 0 < t) :
    t • arithmeticBall (d := d) r = arithmeticBall (t*r) := by
  rw [← preimage_smul_inv₀ ht.ne']
  ext x
  change normSq (t⁻¹ • x) < r^2 ↔ normSq x < (t*r)^2
  rw [normSq_smul, inv_pow, mul_pow, inv_mul_lt_iff₀ (pow_pos ht 2)]

theorem ballMass_ratio {d : ℕ} {r s : ℝ} (hr : 0 < r) (hrs : r ≤ s) :
    ballMass d s ≤ (s/r)^d * ballMass d r := by
  have h1 : 1 ≤ s/r := (le_div_iff₀ hr).2 (by simpa using hrs)
  have hp := probability_dilate_le h1 (isOpen_arithmeticBall (d := d) r).measurableSet
  rw [dilate_arithmeticBall r (by linarith), div_mul_cancel₀ _ hr.ne'] at hp
  exact hp

theorem ballMass_diff_le {d : ℕ} {r s : ℝ} (hr : 0 < r) (hs : 0 < s) :
    |ballMass d s-ballMass d r| ≤ (max r s/min r s)^d-1 := by
  have step (a b : ℝ) (ha : 0 < a) (hab : a ≤ b) :
      ballMass d b-ballMass d a ≤ (b/a)^d-1 := by
    have h := ballMass_ratio (d := d) ha hab
    have h1 : 1 ≤ b/a := (le_div_iff₀ ha).2 (by simpa using hab)
    have h2 : 0 ≤ (b/a)^d - 1 := sub_nonneg.mpr (one_le_pow₀ h1)
    have h3 := mul_le_mul_of_nonneg_left (probability_le_one (arithmeticBall (d := d) a)) h2
    change ((b/a)^d-1)*ballMass d a ≤ _ at h3
    linarith only [h,h3]
  rcases le_total r s with h | h
  · rw [max_eq_right h,min_eq_left h,abs_of_nonneg (sub_nonneg.mpr (ballMass_mono hr.le h))]
    exact step r s hr h
  · rw [max_eq_left h,min_eq_right h,abs_of_nonpos (sub_nonpos.mpr (ballMass_mono hs.le h)),neg_sub]
    exact step s r hs h

theorem ballMass_continuousOn (d : ℕ) : ContinuousOn (ballMass d) (Ioi 0) := by
  intro r hr
  have hr' : 0 < r := hr
  let g := fun s : ℝ ↦ (max r s/min r s)^d-1
  have hg : ContinuousAt g r := by
    dsimp [g]
    exact (((continuous_const.max continuous_id).continuousAt.div
      (continuous_const.min continuous_id).continuousAt (by simpa using hr'.ne')).pow d).sub
        continuousAt_const
  have hg0 : g r = 0 := by simp [g,hr'.ne']
  apply Metric.continuousWithinAt_iff.mpr
  intro ε hε
  obtain ⟨δ,hδ,hgδ⟩ := Metric.continuousAt_iff.mp hg ε hε
  refine ⟨δ,hδ, fun s hs hsr ↦ ?_⟩
  specialize hgδ hsr
  rw [hg0, Real.dist_eq, sub_zero] at hgδ
  rw [Real.dist_eq]
  exact (ballMass_diff_le hr' hs).trans_lt ((le_abs_self (g s)).trans_lt hgδ)

theorem ballMass_every_level {d : ℕ} {a b I : ℝ} (ha : 0 < a) (hab : a ≤ b)
    (hI : I ∈ Icc (ballMass d a) (ballMass d b)) :
    ∃ r ∈ Icc a b, ballMass d r = I :=
  intermediate_value_Icc hab ((ballMass_continuousOn d).mono fun _ h ↦ ha.trans_le h.1) hI

def scalarMatrix {d : ℕ} (v : ℝ) : Coefficients d :=
  fun i j ↦ if i=j then v else 0

theorem quadratic_scalarMatrix {d : ℕ} (v : ℝ) (x : Point d) :
    quadratic (scalarMatrix v) x = v * normSq x := by
  simp only [quadratic, normSq, scalarMatrix, ite_mul, zero_mul, Finset.sum_ite_eq,
    Finset.mem_univ, ite_true, Finset.mul_sum, pow_two]
  apply Finset.sum_congr rfl
  intro i _
  ring

noncomputable def ballEllipsoid (d : ℕ) {ρ r : ℝ} (hρ : 0 < ρ)
    (hr : r ∈ Icc (3*ρ/4) ρ) : Ellipsoid d := by
  have hr0 : 0 < r := by have h := hr.1; linarith
  have h : 1 ≤ ρ/r := (le_div_iff₀ hr0).2 (by simpa using hr.2)
  have h2 : ρ/r ≤ 4/3 := (div_le_iff₀ hr0).2 (by have h := hr.1; linarith)
  have hsq1 : 1 ≤ (ρ/r)^2 := by nlinarith [sq_nonneg (ρ/r-1)]
  have hsq2 : (ρ/r)^2 ≤ 2 := by nlinarith [pow_le_pow_left₀ (by linarith : 0 ≤ ρ/r) h2 2]
  refine ⟨scalarMatrix ((ρ/r)^2), 0, ?_, ?_, ?_, ?_⟩
  · intro i j
    simp [scalarMatrix,eq_comm]
  · intro x
    rw [quadratic_scalarMatrix]
    simpa using mul_le_mul_of_nonneg_right hsq1 (normSq_nonneg x)
  · intro x
    rw [quadratic_scalarMatrix]
    exact mul_le_mul_of_nonneg_right hsq2 (normSq_nonneg x)
  · intro x
    simp only [linear, Pi.zero_apply, zero_mul, Finset.sum_const_zero, abs_zero]
    linarith [normSq_nonneg x]

theorem region_ballEllipsoid (d : ℕ) {ρ r : ℝ} (hρ : 0 < ρ)
    (hr : r ∈ Icc (3*ρ/4) ρ) :
    region ρ (ballEllipsoid d hρ hr) = arithmeticBall r := by
  have hr0 : 0 < r := by have h := hr.1; linarith
  ext x
  change polynomial (scalarMatrix ((ρ/r)^2)) 0 (ρ⁻¹ • x) < 1 ↔ normSq x < r^2
  simp only [polynomial, quadratic_scalarMatrix, normSq_smul, linear, Pi.zero_apply,
    zero_mul, Finset.sum_const_zero, add_zero, ← mul_assoc, ← mul_pow]
  have hh : ρ / r * ρ⁻¹ = r⁻¹ := by field_simp
  rw [hh, inv_pow, inv_mul_lt_iff₀ (pow_pos hr0 2),mul_one]

theorem massGap_pos {d : ℕ} (hd : 0 < d) {ρ : ℝ} (hρ : 0 < ρ) :
    0 < massGap d ρ := sub_pos.mpr (ballMass_lt hd (by linarith) (by linarith))

end Cubature
