import Theory.CylindricalAlgebra
import Theory.CylindricalModel

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

noncomputable def rankShift {d : ℕ} (w v : Point d) : Point d :=
  w + (1/16 : ℝ) • v

theorem rankShift_bound {d : ℕ} {w v : Point d} (hw : ∀ j, |w j|≤1)
    (hv : ∀ j, |v j|≤2) (j : Fin d) :
    |rankShift w v j|≤9/8 := by
  simp only [rankShift,Pi.add_apply,Pi.smul_apply,smul_eq_mul]
  have H := abs_add_le (w j) ((1/16:ℝ)*v j)
  rw [abs_mul,abs_of_nonneg (by norm_num : 0≤(1/16:ℝ))] at H
  linarith [hv j,hw j]

theorem rankShift_curvature {d : ℕ} (G : RankTwoCap d) (w v : Point d)
    (hw1 : 1≤quadratic G.matrix w) (hw2 : quadratic G.matrix w≤2) (hv : normSq v≤4) :
    1/2≤quadratic G.matrix (rankShift w v) ∧ quadratic G.matrix (rankShift w v)≤3 := by
  let u := (1/16 : ℝ) • v
  have hu : quadratic G.matrix u≤1/32 := by
    dsimp [u]
    rw [quadratic_smul,quadratic_gram]
    linarith [G.upper v]
  rw [quadratic_gram] at hw1 hw2 hu
  let a := linear G.u w
  let b := linear G.v w
  let c := linear G.u u
  let e := linear G.v u
  change 1≤a^2+b^2 at hw1
  change a^2+b^2≤2 at hw2
  change c^2+e^2≤1/32 at hu
  have Hprod := mul_le_mul hw2 hu (by positivity : 0≤c^2+e^2) (by norm_num : (0:ℝ)≤2)
  have hab : (a*c+b*e)^2≤(1/4:ℝ)^2 := by nlinarith only [Hprod,sq_nonneg (a*e-b*c)]
  have ha : |a*c+b*e|≤1/4 := (sq_le_sq₀ (abs_nonneg _) (by norm_num)).1 (by rwa [sq_abs])
  have ha' := abs_le.mp ha
  rw [quadratic_gram,rankShift,linear_add_coord,linear_add_coord]
  change 1/2≤(a+c)^2+(b+e)^2 ∧ (a+c)^2+(b+e)^2≤3
  exact ⟨by nlinarith only [hw1,ha'.1,sq_nonneg c,sq_nonneg e],
    by nlinarith only [hw2,hu,ha'.2]⟩

theorem rankShift_plus_formula {d : ℕ} (G : RankTwoCap d) (w : Point d) (i : Fin d) :
    quadratic G.matrix (rankShift w (axis i)) =
      quadratic G.matrix w+(1/8)*curvatureProduct G w i+(1/256)*G.matrix i i := by
  simp only [rankShift,quadratic_gram,linear_add_coord,linear_smul,linear_axis,
    RankTwoCap.matrix,curvatureProduct]
  ring

theorem rankShift_minus_formula {d : ℕ} (G : RankTwoCap d) (w : Point d) (i : Fin d) :
    quadratic G.matrix (rankShift w (-axis i)) =
      quadratic G.matrix w-(1/8)*curvatureProduct G w i+(1/256)*G.matrix i i := by
  have H (b : Point d) : linear b (-axis i) = -b i := by
    rw [← neg_one_smul ℝ (axis i),linear_smul,linear_axis,neg_one_mul]
  simp only [rankShift,quadratic_gram,linear_add_coord,linear_smul,H,
    RankTwoCap.matrix,curvatureProduct]
  ring

theorem rankShift_pair_formula {d : ℕ} (G : RankTwoCap d) (w : Point d) (i j : Fin d) :
    quadratic G.matrix (rankShift w (axis i+axis j)) -
      quadratic G.matrix (rankShift w (axis i)) -
      quadratic G.matrix (rankShift w (axis j)) + quadratic G.matrix w =
      (1/128)*G.matrix i j := by
  simp only [rankShift,quadratic_gram,linear_add_coord,linear_smul,linear_axis,RankTwoCap.matrix]
  ring

theorem axis_pair_norm_bound {d : ℕ} (i j : Fin d) : normSq (axis i+axis j)≤4 := by
  by_cases h : i=j
  · subst j
    rw [← two_smul ℝ (axis i),normSq_smul,normSq_axis]
    norm_num
  · rw [normSq_two_axes h]
    norm_num

theorem axis_pair_coord_bound {d : ℕ} (i j k : Fin d) : |(axis i+axis j) k|≤2 := by
  rw [Pi.add_apply]
  linarith [abs_add_le (axis i k) (axis j k),axis_bound i k,axis_bound j k]

noncomputable def cylindricalProbe {d : ℕ} (ρ : ℝ) (v : Point d) (m : ℕ) :
    QueryTree (Point d) ℝ :=
  (bisect ((9/8)*ρ) ((8/9 : ℝ) • v) m (1/2) 4).map (fun s => s^2)

theorem cylindricalProbe_depth {d : ℕ} (ρ : ℝ) (v : Point d) (m : ℕ) :
    (cylindricalProbe ρ v m).depth≤m := by
  unfold cylindricalProbe
  exact (QueryTree.depth_map_le _ _).trans_eq (bisect_depth _ _ _ _ _)

theorem cylindricalProbe_allowed {d : ℕ} {ρ : ℝ} (hρ : 0≤ρ) (hρ' : ρ≤16)
    (v : Point d) (hv : ∀ i, |v i|≤9/8) (m : ℕ) :
    (cylindricalProbe ρ v m).admissible allowed := by
  have h0 : ∀ i, |((8/9 : ℝ) • v) i|≤1 := by
    intro i
    rw [Pi.smul_apply,smul_eq_mul,abs_mul,abs_of_nonneg (by norm_num : 0≤(8/9:ℝ))]
    linarith [hv i]
  exact QueryTree.admissible_map _ _ _
    (bisect_admissible (by positivity) (by linarith) h0 m le_rfl (by norm_num))

theorem cylindricalProbe_precision {d : ℕ} {ρ : ℝ} (hρ : 0<ρ)
    (G : RankTwoCap d) (v : Point d) (hv : 1/2≤quadratic G.matrix v ∧ quadratic G.matrix v≤3)
    (m : ℕ) :
    |(cylindricalProbe ρ v m).run (answerBit (cylinderRegion ρ G)) -
       quadratic G.matrix v| ≤ cylinderNoise m := by
  let a := quadratic G.matrix v
  let r := Real.sqrt a
  have ha : 0≤a := by linarith [hv.1]
  have hr0 : 0≤r := Real.sqrt_nonneg a
  have hr : r^2=a := Real.sq_sqrt ha
  have hr1 : 1/2≤r := (sq_le_sq₀ (by norm_num) hr0).1 (by rw [hr]; linarith [hv.1])
  have hr2 : r≤4 := (sq_le_sq₀ hr0 (by norm_num)).1 (by rw [hr]; linarith [hv.2])
  have hp (t : ℝ) (ht : 0<t) :
      answerBit (cylinderRegion ρ G) ((((9/8)*ρ)/t) • ((8/9 : ℝ) • v)) = true ↔ r<t := by
    have hc : ρ⁻¹ * ((((9/8)*ρ)/t)*(8/9))=t⁻¹ := by field_simp
    rw [answerBit_iff]
    change quadratic G.matrix (ρ⁻¹ • ((((9/8)*ρ)/t) • ((8/9:ℝ) • v)))<1 ↔ r<t
    rw [smul_smul ((((9/8)*ρ)/t)) (8/9:ℝ) v,smul_smul,hc,quadratic_smul,inv_pow,
      inv_mul_lt_iff₀ (pow_pos ht 2),mul_one]
    change a<t^2 ↔ r<t
    rw [← hr,sq_lt_sq₀ hr0 ht.le]
  have hB := bisect_correct ((9/8)*ρ) ((8/9 : ℝ) • v) (answerBit (cylinderRegion ρ G))
    r hp m (by norm_num) hr1 hr2
  let S := (bisect ((9/8)*ρ) ((8/9 : ℝ) • v) m (1/2) 4).run (answerBit (cylinderRegion ρ G))
  have hS1 : 1/2≤S := hB.1.1
  have hS2 : S≤4 := hB.1.2
  have he : |S-r|≤2/(2:ℝ)^m := by
    refine hB.2.trans ?_
    rw [pow_succ,mul_comm ((2:ℝ)^m) 2,← div_div]
    exact div_le_div_of_nonneg_right (by norm_num : (4-1/2:ℝ)/2≤2) (by positivity)
  rw [cylindricalProbe,QueryTree.run_map]
  change |S^2-a|≤cylinderNoise m
  rw [← hr,show S^2-r^2=(S-r)*(S+r) by ring,abs_mul]
  have hh : |S+r|≤8 := by rw [abs_of_nonneg (by linarith)]; linarith
  have H := mul_le_mul he hh (abs_nonneg _) (by positivity : 0 ≤ 2/(2:ℝ)^m)
  exact H.trans_eq (by unfold cylinderNoise; ring)

end Cubature
