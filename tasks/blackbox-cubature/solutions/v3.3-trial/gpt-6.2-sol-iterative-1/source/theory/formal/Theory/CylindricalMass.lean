import Theory.CylindricalLearner

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

theorem entry_error_quadratic {d : ℕ} {A B : Coefficients d} {ε : ℝ} (hε : 0≤ε)
    (h : ∀ i j, |B i j-A i j|≤ε) (x : Point d) :
    |quadratic B x-quadratic A x|≤ε*d*normSq x := by
  have H : quadratic B x-quadratic A x= ∑ i, ∑ j, (B i j-A i j)*x i*x j := by
    simp only [quadratic,← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [H]
  calc
    _ ≤ ∑ i, ∑ j, |(B i j-A i j)*x i*x j| := by
      apply (Finset.abs_sum_le_sum_abs _ _).trans
      exact Finset.sum_le_sum fun i _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, ε*(|x i| * |x j|) := by
      apply Finset.sum_le_sum
      intro i _
      apply Finset.sum_le_sum
      intro j _
      simp only [abs_mul,mul_assoc]
      exact mul_le_mul_of_nonneg_right (h i j) (mul_nonneg (abs_nonneg _) (abs_nonneg _))
    _ = ε*(∑ i,∑ j, |x i| * |x j|) := by simp [Finset.mul_sum]
    _ ≤ _ := by
      rw [mul_assoc]
      exact mul_le_mul_of_nonneg_left (sum_abs_cross x) hε

noncomputable def cylinderTruncation {d : ℕ} (ρ T : ℝ) (B : Coefficients d) :
    Set (Point d) :=
  {x | normSq (ρ⁻¹ • x)≤T^2 ∧ quadratic B (ρ⁻¹ • x)<1}

theorem cylinderRegion_measurable {d : ℕ} (ρ : ℝ) (G : RankTwoCap d) :
    MeasurableSet (cylinderRegion ρ G) := by
  have hc : Continuous (fun x : Point d => quadratic G.matrix (ρ⁻¹ • x)) := by
    unfold quadratic
    fun_prop
  exact (isOpen_lt hc continuous_const).measurableSet

theorem cylinderTail_measurable {d : ℕ} (ρ T : ℝ) :
    MeasurableSet (cylinderTail (d := d) ρ T) :=
  (isOpen_lt continuous_const (continuous_normSq.comp (continuous_const_smul _))).measurableSet

theorem cylinder_contraction {d : ℕ} (G : RankTwoCap d) (x : Point d) {s : ℝ}
    (hs0 : 0≤s) (hs1 : s≤1) :
    quadratic G.matrix (s • x)≤s*quadratic G.matrix x := by
  rw [quadratic_smul]
  have H : s^2≤s := by nlinarith [mul_nonneg hs0 (sub_nonneg.mpr hs1)]
  exact mul_le_mul_of_nonneg_right H (by rw [quadratic_gram]; positivity)

theorem cylinder_sandwich {d : ℕ} (ρ T : ℝ) (G : RankTwoCap d) (B : Coefficients d)
    {δ : ℝ} (hδ0 : 0≤δ) (hδ : δ<1)
    (hfit : ∀ x, normSq x≤T^2 → |quadratic B x-quadratic G.matrix x|≤δ) :
    (1-δ)^d*(probability (cylinderRegion ρ G)-probability (cylinderTail (d := d) ρ T)) ≤
        probability (cylinderTruncation ρ T B) ∧
    probability (cylinderTruncation ρ T B) ≤ (1+δ)^d*probability (cylinderRegion ρ G) := by
  let S := cylinderRegion ρ G
  let U := cylinderTail (d := d) ρ T
  let V := S ∩ Uᶜ
  let F := cylinderTruncation ρ T B
  have hS : MeasurableSet S := cylinderRegion_measurable _ _
  have hU : MeasurableSet U := cylinderTail_measurable _ _
  have hV : MeasurableSet V := hS.inter hU.compl
  have h0 : 0<1-δ := by linarith
  have h1 : 0<1+δ := by linarith
  have hin : (1-δ) • V ⊆ F := by
    rintro x ⟨y,hy,rfl⟩
    have hn : normSq (ρ⁻¹ • y)≤T^2 := le_of_not_gt hy.2
    have hs : quadratic G.matrix (ρ⁻¹ • y)<1 := hy.1
    have hsub : normSq (ρ⁻¹ • ((1-δ) • y))≤T^2 := by
      rw [smul_comm (ρ⁻¹ : ℝ) (1-δ) y,normSq_smul]
      have h : (1-δ)^2≤1 := by nlinarith [mul_nonneg hδ0 h0.le]
      calc
        _ ≤ 1*normSq (ρ⁻¹ • y) := mul_le_mul_of_nonneg_right h (normSq_nonneg _)
        _ ≤ _ := by simpa using hn
    have H := (abs_le.mp (hfit _ hsub)).2
    refine ⟨hsub,?_⟩
    rw [smul_comm (ρ⁻¹ : ℝ) (1-δ) y] at H ⊢
    have hb := cylinder_contraction G (ρ⁻¹ • y) h0.le (by linarith : 1-δ≤1)
    linarith only [H,hb,mul_lt_mul_of_pos_left hs h0]
  have hout : F ⊆ (1+δ) • S := by
    intro x hx
    have hx' : normSq (ρ⁻¹ • x)≤T^2 ∧ quadratic B (ρ⁻¹ • x)<1 := hx
    have ht : quadratic G.matrix (ρ⁻¹ • x)<1+δ := by
      linarith [(abs_le.mp (hfit _ hx'.1)).1,hx'.2]
    let t := (1+δ)⁻¹
    have hp : 0<t := inv_pos.mpr h1
    have ht1 : t≤1 := inv_le_one_of_one_le₀ (by linarith)
    refine ⟨t • x,?_ ,?_⟩
    · change quadratic G.matrix (ρ⁻¹ • (t • x))<1
      rw [smul_comm (ρ⁻¹ : ℝ) t x]
      exact (cylinder_contraction G _ hp.le ht1).trans_lt
        (by simpa [t,inv_mul_cancel₀ h1.ne'] using mul_lt_mul_of_pos_left ht hp)
    · change (1+δ) • (t • x) = x
      rw [smul_smul,show (1+δ)*t=1 from mul_inv_cancel₀ h1.ne',one_smul]
  have hVlb : probability S-probability U≤probability V := by
    have sub : S\V⊆U := by
      intro x hx
      by_contra h
      exact hx.2 ⟨hx.1,h⟩
    have H := probability_mono sub
    unfold probability at H
    rw [measureReal_sdiff inter_subset_left hV] at H
    change probability S-probability V≤probability U at H
    linarith
  exact ⟨(mul_le_mul_of_nonneg_left hVlb (pow_nonneg h0.le d)).trans
    ((probability_dilate_ge h0 (by linarith) hV).trans (probability_mono hin)),
    (probability_mono hout).trans (probability_dilate_le (by linarith : 1≤1+δ) hS)⟩

noncomputable def cylinderCubature {d : ℕ} (hd : 0<d) (ρ : ℝ) (w : Point d)
    (m : ℕ) (T : ℝ) : QueryTree (Point d) (Coefficients d × ℝ) :=
  (cylinderLearner hd ρ w m).map (fun B => (B,probability (cylinderTruncation ρ T B)))

theorem cylinderCubature_budget {d : ℕ} (hd : 0<d) (ρ : ℝ) (w : Point d)
    (m : ℕ) (T : ℝ) : (cylinderCubature hd ρ w m T).depth ≤ (3*d+1)*m :=
  (QueryTree.depth_map_le _ _).trans (cylinderLearner_depth hd ρ w m)

theorem cylinderCubature_allowed {d : ℕ} (hd : 0<d) {ρ : ℝ} (hρ : 0≤ρ) (hρ' : ρ≤16)
    (w : Point d) (hw : ∀ i, |w i|≤1) (m : ℕ) (T : ℝ) :
    (cylinderCubature hd ρ w m T).admissible allowed :=
  QueryTree.admissible_map _ _ _ (cylinderLearner_allowed hd hρ hρ' w hw m)

theorem cylinderCubature_bounds {d : ℕ} (hd : 0<d) {ρ : ℝ} (hρ : 0<ρ)
    (w : Point d) (m : ℕ) (T : ℝ) (hnoise : cylinderNoise m≤1/8)
    (hδ : cylinderDilationError d m T<1)
    (G : RankTwoCap d) (ha₁ : 1≤quadratic G.matrix w) (ha₂ : quadratic G.matrix w≤2) :
    let p := (cylinderCubature hd ρ w m T).run (answerBit (cylinderRegion ρ G))
    (∀ i j, |p.1 i j-G.matrix i j|≤cylinderMatrixError m) ∧
    (1-cylinderDilationError d m T)^d *
        (probability (cylinderRegion ρ G)-probability (cylinderTail (d := d) ρ T)) ≤ p.2 ∧
    p.2 ≤ (1+cylinderDilationError d m T)^d*probability (cylinderRegion ρ G) := by
  simp only [cylinderCubature,QueryTree.run_map,Prod.fst,Prod.snd]
  have h₁ := cylinderLearner_precision hd hρ G w ha₁ ha₂ m hnoise
  refine ⟨h₁,?_⟩
  have h0 : 0≤cylinderMatrixError m := by unfold cylinderMatrixError cylinderNoise; positivity
  apply cylinder_sandwich ρ T G _ (by unfold cylinderDilationError; positivity) hδ
  intro x hx
  exact (entry_error_quadratic h0 h₁ x).trans
    (mul_le_mul_of_nonneg_left hx (by positivity : 0≤cylinderMatrixError m*d))

end Cubature
