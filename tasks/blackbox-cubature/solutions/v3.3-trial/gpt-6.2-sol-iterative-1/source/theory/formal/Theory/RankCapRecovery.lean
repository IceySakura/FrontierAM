import Theory.RankCapModel
import Theory.KinkGeometry

open MeasureTheory Set
open scoped BigOperators Pointwise

namespace Cubature

theorem quadratic_gram {d : ℕ} (G : RankTwoCap d) (x : Point d) :
    quadratic G.matrix x = (linear G.u x)^2+(linear G.v x)^2 := by
  unfold RankTwoCap.matrix quadratic linear
  simp only [add_mul, Finset.sum_add_distrib]
  congr 1 <;> (
    rw [pow_two,Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring)

theorem gram_symmetric {d : ℕ} (G : RankTwoCap d) (i j : Fin d) :
    G.matrix i j=G.matrix j i := by simp [RankTwoCap.matrix,mul_comm]

theorem gram_pair {d : ℕ} (G : RankTwoCap d) (i j : Fin d) :
    quadratic G.matrix (axis i+axis j) = G.matrix i i+G.matrix j j+2*G.matrix i j := by
  simp only [quadratic,Pi.add_apply,mul_add,add_mul,Finset.sum_add_distrib]
  simp only [axis,mul_ite,mul_one,mul_zero,Finset.sum_ite_eq',Finset.mem_univ,ite_true]
  rw [gram_symmetric G j i]
  ring

theorem rankCap_integrable {d : ℕ} (ρ : ℝ) (G : RankTwoCap d) :
    Integrable (rankCap ρ G) (gaussian d) := by
  have hc : Continuous (rankCap ρ G) := by
    unfold rankCap matrixCap quadratic
    fun_prop
  apply Integrable.mono' (integrable_const (1:ℝ)) hc.aestronglyMeasurable
  exact ae_of_all _ fun x => by
    change ‖max (1-quadratic G.matrix (ρ⁻¹ • x)) 0‖≤1
    rw [Real.norm_eq_abs,abs_of_nonneg (le_max_right _ _),quadratic_gram]
    apply max_le _ zero_le_one
    nlinarith [sq_nonneg (linear G.u (ρ⁻¹ • x)), sq_nonneg (linear G.v (ρ⁻¹ • x))]

noncomputable def rankProbe {d : ℕ} (ρ : ℝ) (v : Point d) : ValueTree (Point d) ℝ :=
  (ValueTree.observe ((ρ/4) • v)).map (fun f => 16*(1-f))

theorem rankProbe_allowed {d : ℕ} {ρ : ℝ} (hρ : 0 ≤ ρ) (hb : ρ≤20)
    {v : Point d} (hv : ∀ j, |v j|≤1) : (rankProbe ρ v).admissible allowed := by
  apply ValueTree.admissible_map
  refine ⟨?_,fun _ => trivial⟩
  intro j
  simp only [Pi.smul_apply,smul_eq_mul,abs_mul,abs_of_nonneg (by positivity : 0 ≤ ρ/4)]
  linarith [mul_le_mul_of_nonneg_left (hv j) (show 0 ≤ ρ/4 by positivity)]

theorem rankProbe_exact {d : ℕ} {ρ : ℝ} (hρ : 0 < ρ) (G : RankTwoCap d) (v : Point d)
    (hv : normSq v≤2) : (rankProbe ρ v).run (rankCap ρ G) = quadratic G.matrix v := by
  have hq : quadratic G.matrix v≤4 := by rw [quadratic_gram]; linarith [G.upper v]
  have hh : (ρ⁻¹ : ℝ)*(ρ/4)=1/4 := by field_simp
  rw [rankProbe,ValueTree.run_map]
  change 16*(1-max (1-quadratic G.matrix (ρ⁻¹ • ((ρ/4) • v))) 0)=_
  rw [smul_smul,hh,quadratic_smul,max_eq_left (by nlinarith)]
  ring

noncomputable def rankDiagonal {d : ℕ} (ρ : ℝ) : ValueTree (Point d) (Point d) :=
  ValueTree.collect (fun i : Fin d => rankProbe ρ (axis i)) (Finset.univ : Finset (Fin d)).toList

noncomputable def rankRow {d : ℕ} (ρ : ℝ) (D : Point d) (i : Fin d) :
    ValueTree (Point d) (Point d) :=
  ValueTree.collect (fun j : Fin d =>
    if i=j then .ret (D i)
    else (rankProbe ρ (ray (rayIndex i j))).map (fun a => (a-D i-D j)/2))
      (Finset.univ : Finset (Fin d)).toList

theorem rankDiagonal_exact {d : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : RankTwoCap d) :
    (rankDiagonal ρ).run (rankCap ρ G) = fun i => G.matrix i i := by
  funext i
  rw [rankDiagonal,ValueTree.run_collect _ _ _ (by simp),
    rankProbe_exact hρ G _ (by rw [normSq_axis];norm_num),quadratic_axis]

theorem rankRow_exact {d : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : RankTwoCap d) (i : Fin d) :
    (rankRow ρ (fun j => G.matrix j j) i).run (rankCap ρ G) = G.matrix i := by
  funext j
  rw [rankRow,ValueTree.run_collect _ _ _ (by simp)]
  by_cases he : i=j
  · subst j; simp [ValueTree.run]
  · rw [ite_eq_right he,ValueTree.run_map,rankProbe_exact hρ _ _ (ray_norm _).2,
      mixed_ray he,gram_pair]
    ring

theorem rankDiagonal_within {d : ℕ} (ρ : ℝ) :
    (rankDiagonal (d := d) ρ).within d := by
  unfold rankDiagonal
  have H := ValueTree.within_collect (fun i : Fin d => rankProbe ρ (axis i))
    (Finset.univ : Finset (Fin d)).toList 1
    (fun _ => ValueTree.within_map _ _ (by exact fun _ => trivial))
  simpa using H

theorem rankRow_within {d : ℕ} (ρ : ℝ) (D : Point d) (i : Fin d) :
    (rankRow ρ D i).within d := by
  unfold rankRow
  have H := ValueTree.within_collect (fun j : Fin d =>
    if i=j then .ret (D i)
    else (rankProbe ρ (ray (rayIndex i j))).map (fun a => (a-D i-D j)/2))
      (Finset.univ : Finset (Fin d)).toList 1 (fun j => by
        split
        · trivial
        · exact ValueTree.within_map _ _ (fun _ => trivial))
  simpa using H

theorem rankDiagonal_allowed {d : ℕ} {ρ : ℝ} (hρ : 0 ≤ ρ) (hb : ρ≤20) :
    (rankDiagonal (d := d) ρ).admissible allowed :=
  ValueTree.admissible_collect _ _ _ fun i => rankProbe_allowed hρ hb (axis_bound i)

theorem rankRow_allowed {d : ℕ} {ρ : ℝ} (hρ : 0 ≤ ρ) (hb : ρ≤20) (D : Point d) (i : Fin d) :
    (rankRow ρ D i).admissible allowed := by
  apply ValueTree.admissible_collect
  intro j
  split
  · trivial
  · exact ValueTree.admissible_map _ _ _ (rankProbe_allowed hρ hb (ray_bound _))

noncomputable def rankOnePart {d : ℕ} (a : Point d) (p : ℝ) : Coefficients d :=
  fun i j => a i*a j/p

noncomputable def residualOf {d : ℕ} (G : RankTwoCap d) (k : Fin d) : Coefficients d :=
  fun i j => G.matrix i j-rankOnePart (G.matrix k) (G.matrix k k) i j

def crossVector {d : ℕ} (G : RankTwoCap d) (k : Fin d) : Point d :=
  fun j => G.u k*G.v j-G.v k*G.u j

theorem residual_formula {d : ℕ} (G : RankTwoCap d) (k : Fin d) (hk : G.matrix k k≠0)
    (i j : Fin d) :
    residualOf G k i j = crossVector G k i*crossVector G k j/(G.matrix k k) := by
  unfold residualOf rankOnePart crossVector
  field_simp [hk]
  unfold RankTwoCap.matrix
  ring

theorem gram_zero {d : ℕ} (G : RankTwoCap d) (h : ∀ i, G.matrix i i≤0) :
    G.matrix=0 := by
  have hz (i : Fin d) : G.u i=0 ∧ G.v i=0 := by
    have h0 : G.u i*G.u i+G.v i*G.v i≤0 := h i
    exact ⟨by nlinarith [sq_nonneg (G.v i)],by nlinarith [sq_nonneg (G.u i)]⟩
  funext i j
  simp [RankTwoCap.matrix,(hz i).1,(hz i).2]

theorem residual_zero {d : ℕ} (G : RankTwoCap d) (k : Fin d) (hk : 0<G.matrix k k)
    (h : ∀ i, residualOf G k i i≤0) :
    G.matrix=rankOnePart (G.matrix k) (G.matrix k k) := by
  have hz (i : Fin d) : crossVector G k i=0 := by
    have hi := h i
    rw [residual_formula G k hk.ne'] at hi
    have H := (div_le_iff₀ hk).1 hi
    nlinarith only [H]
  funext i j
  have H := residual_formula G k hk.ne' i j
  rw [hz i,zero_mul,zero_div] at H
  exact sub_eq_zero.mp H

theorem residual_reconstruct {d : ℕ} (G : RankTwoCap d) (k l : Fin d) (hk : 0<G.matrix k k)
    (hl : 0<residualOf G k l l) (i j : Fin d) :
    G.matrix i j = rankOnePart (G.matrix k) (G.matrix k k) i j+
      residualOf G k l i*residualOf G k l j/(residualOf G k l l) := by
  have h0 : crossVector G k l≠0 := by
    intro h
    rw [residual_formula G k hk.ne',h,zero_mul,zero_div] at hl
    exact (lt_irrefl _ hl)
  have hF : residualOf G k i j =
      residualOf G k l i*residualOf G k l j/(residualOf G k l l) := by
    simp only [residual_formula G k hk.ne']
    field_simp [hk.ne',h0]
    <;> ring
  rw [← hF, residualOf]
  ring

noncomputable def rankSecond {d : ℕ} (ρ : ℝ) (Diag a : Point d) (p : ℝ) :
    ValueTree (Point d) (Coefficients d) :=
  let S := fun j => Diag j-rankOnePart a p j j
  if h : ∃ j, 0 < S j then
    let l := Classical.choose h
    (rankRow ρ Diag l).map fun b =>
      let y := fun j => b j-rankOnePart a p l j
      fun i j => rankOnePart a p i j+y i*y j/(S l)
  else .ret (rankOnePart a p)

noncomputable def rankTwoLearner (d : ℕ) (ρ : ℝ) : ValueTree (Point d) (Coefficients d) :=
  (rankDiagonal ρ).bind fun Diag =>
    if h : ∃ k, 0 < Diag k then
      let k := Classical.choose h
      let p := Diag k
      (rankRow ρ Diag k).bind fun a => rankSecond ρ Diag a p
    else .ret 0

theorem rankTwoLearner_within (d : ℕ) (ρ : ℝ) : (rankTwoLearner d ρ).within (3*d) := by
  unfold rankTwoLearner
  have H : 3*d=d+(d+d) := by omega
  rw [H]
  refine ValueTree.within_bind _ _ (rankDiagonal_within ρ) ?_
  intro a
  split
  · refine ValueTree.within_bind _ _ (rankRow_within _ _ _) ?_
    intro b
    unfold rankSecond
    dsimp only
    split
    · exact ValueTree.within_map _ _ (rankRow_within _ _ _)
    · trivial
  · trivial

theorem rankTwoLearner_allowed (d : ℕ) {ρ : ℝ} (hρ : 0≤ρ) (hb : ρ≤20) :
    (rankTwoLearner d ρ).admissible allowed := by
  unfold rankTwoLearner
  refine ValueTree.admissible_bind _ _ _ (rankDiagonal_allowed hρ hb) ?_
  intro a
  split
  · refine ValueTree.admissible_bind _ _ _ (rankRow_allowed hρ hb _ _) ?_
    intro b
    unfold rankSecond
    dsimp only
    split
    · exact ValueTree.admissible_map _ _ _ (rankRow_allowed hρ hb _ _)
    · trivial
  · trivial

theorem rankTwoLearner_exact {d : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : RankTwoCap d) :
    (rankTwoLearner d ρ).run (rankCap ρ G) = G.matrix := by
  rw [rankTwoLearner,ValueTree.run_bind,rankDiagonal_exact hρ G]
  by_cases h : ∃ k, 0 < G.matrix k k
  · rw [dif_pos h,ValueTree.run_bind,rankRow_exact hρ G]
    let k := Classical.choose h
    have hp : 0<G.matrix k k := Classical.choose_spec h
    change (rankSecond ρ (fun j => G.matrix j j) (G.matrix k) (G.matrix k k)).run
      (rankCap ρ G) = _
    unfold rankSecond
    dsimp only
    split
    · rename_i h'
      rw [ValueTree.run_map,rankRow_exact hρ G]
      let l := Classical.choose h'
      have hl : 0<residualOf G k l l := Classical.choose_spec h'
      funext i j
      exact (residual_reconstruct G k l hp hl i j).symm
    · rename_i h'
      rw [ValueTree.run]
      exact (residual_zero G k hp (fun i => le_of_not_gt (fun ht => h' ⟨i,ht⟩))).symm
  · rw [dif_neg h,ValueTree.run]
    exact (gram_zero G (fun i => le_of_not_gt (fun ht => h ⟨i,ht⟩))).symm

theorem rank_two_cap_upper (d : ℕ) {ρ : ℝ} (hρ : 0<ρ) (hb : ρ≤20) :
    ∃ T : ValueTree (Point d) (Coefficients d × ℝ),
      T.within (3*d) ∧ T.admissible allowed ∧
      ∀ G : RankTwoCap d, T.run (rankCap ρ G) = (G.matrix,rankCapIntegral ρ G) := by
  let out : Coefficients d → Coefficients d × ℝ := fun A => (A,∫ x,matrixCap ρ A x ∂(gaussian d))
  refine ⟨(rankTwoLearner d ρ).map out,ValueTree.within_map _ _ (rankTwoLearner_within d ρ),
    ValueTree.admissible_map _ _ _ (rankTwoLearner_allowed d hρ.le hb),?_⟩
  intro G
  rw [ValueTree.run_map,rankTwoLearner_exact hρ G]
  rfl

end Cubature
