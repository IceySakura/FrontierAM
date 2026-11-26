import Theory.RankCapRecovery
import Theory.LowerBounds
import Mathlib.Probability.Distributions.Gaussian.Multivariate

open MeasureTheory ProbabilityTheory Set WithLp
open scoped BigOperators Pointwise ProbabilityTheory RealInnerProductSpace

namespace Cubature

theorem linear_comm {d : ℕ} (u v : Point d) : linear u v=linear v u := by
  simp [linear,mul_comm]

theorem linear_smul_left_coord {d : ℕ} (t : ℝ) (u v : Point d) :
    linear (t • u) v=t*linear u v := by rw [linear_comm,linear_smul,linear_comm v u]

def linearReadings {d n : ℕ} (z : Fin n → Point d) : Point d →ₗ[ℝ] (Fin n → ℝ) where
  toFun c i := linear c (z i)
  map_add' a b := by
    funext i
    simp [linear,Pi.add_apply,add_mul,Finset.sum_add_distrib]
  map_smul' t a := by
    funext i
    simp [linear_smul_left_coord,Pi.smul_apply,smul_eq_mul]

theorem invisible_unit {d n : ℕ} (h : n<d) (z : Fin n → Point d) :
    ∃ w : Point d, normSq w=1 ∧ ∀ i, linear w (z i)=0 := by
  let L := linearReadings z
  have hn : ¬Function.Injective L := by
    intro hh
    have H := LinearMap.finrank_le_finrank_of_injective hh
    simp only [Module.finrank_fintype_fun_eq_card,Fintype.card_fin] at H
    omega
  obtain ⟨a,b,hval,hne⟩ := Function.not_injective_iff.mp hn
  let x := a-b
  have hx : x≠0 := sub_ne_zero.mpr hne
  have hy : 0<normSq x := by
    have hex : ∃ i, x i ≠ 0 := by
      by_contra! H
      exact hx (funext H)
    obtain ⟨i,hi⟩ := hex
    exact (sq_pos_of_ne_zero hi).trans_le (sq_le_normSq x i)
  let t := (Real.sqrt (normSq x))⁻¹
  have h0 : L x = 0 := by change L (a-b) = 0; rw [map_sub,hval,sub_self]
  have h1 : L (t • x)=0 := by rw [map_smul,h0,smul_zero]
  refine ⟨t • x,?_,fun i => congrFun h1 i⟩
  rw [normSq_smul,show t^2=(normSq x)⁻¹ by dsimp [t]; rw [inv_pow,Real.sq_sqrt hy.le],
    inv_mul_cancel₀ hy.ne']

theorem euclidean_linear {d : ℕ} (w x : Point d) :
    (innerSL ℝ (toLp 2 w : EuclideanSpace ℝ (Fin d))) (toLp 2 x)=linear w x := by
  rw [innerSL_apply_apply,PiLp.inner_apply]
  simp [linear,Real.inner_apply,mul_comm]

theorem standard_projection {d : ℕ} (w : Point d) (h : normSq w=1) :
    (gaussian d).map (linear w) = gaussianReal 0 1 := by
  let L : StrongDual ℝ (EuclideanSpace ℝ (Fin d)) := innerSL ℝ (toLp 2 w)
  have hm : (gaussian d).map (toLp 2) = stdGaussian (EuclideanSpace ℝ (Fin d)) := by
    rw [gaussian_eq_pi]
    exact map_pi_eq_stdGaussian
  have ha : linear w = L ∘ toLp 2 := funext (fun x => (euclidean_linear w x).symm)
  rw [ha,← Measure.map_map (by fun_prop) (by fun_prop),hm,IsGaussian.map_eq_gaussianReal L,
    integral_strongDual_stdGaussian L,variance_dual_stdGaussian L]
  have hn : ‖L‖^2 = 1 := by
    dsimp [L]
    rw [innerSL_apply_norm,EuclideanSpace.real_norm_sq_eq]
    exact h
  rw [hn,Real.toNNReal_one]

def unitCap {d : ℕ} (w : Point d) (hw : normSq w=1) : RankTwoCap d where
  u := w
  v := 0
  upper := fun x => by
    have h : ‖(toLp 2 w : EuclideanSpace ℝ (Fin d))‖=1 := by
      have H : ‖(toLp 2 w : EuclideanSpace ℝ (Fin d))‖^2=1 := by
        rw [EuclideanSpace.real_norm_sq_eq]
        exact hw
      nlinarith [norm_nonneg (toLp (2:ENNReal) w : EuclideanSpace ℝ (Fin d))]
    have hc := norm_inner_le_norm (𝕜 := ℝ) (toLp 2 w : EuclideanSpace ℝ (Fin d))
      (toLp 2 x : EuclideanSpace ℝ (Fin d))
    rw [← innerSL_apply_apply ℝ,euclidean_linear,Real.norm_eq_abs,h,one_mul] at hc
    have he := pow_le_pow_left₀ (abs_nonneg _) hc 2
    rw [sq_abs,EuclideanSpace.real_norm_sq_eq] at he
    have hz : linear (0 : Point d) x=0 := by simp [linear]
    rw [hz,zero_pow (by decide),add_zero]
    linarith [show (linear w x)^2≤normSq x from he, normSq_nonneg x]

noncomputable def scalarCap (ρ : ℝ) (t : ℝ) : ℝ :=
  max (1-(t/ρ)^2) 0

theorem unitCap_integral {d : ℕ} (ρ : ℝ) (w : Point d) (hw : normSq w=1) :
    rankCapIntegral ρ (unitCap w hw) = ∫ t, scalarCap ρ t ∂(gaussianReal 0 1) := by
  have hf : rankCap ρ (unitCap w hw) = scalarCap ρ ∘ linear w := by
    funext x
    have hz : linear (0 : Point d) (ρ⁻¹ • x)=0 := by simp [linear]
    simp only [rankCap,matrixCap,quadratic_gram,unitCap,hz,linear_smul,
      zero_pow (by decide : (2:ℕ) ≠ 0),add_zero,Function.comp_apply,scalarCap,div_eq_mul_inv]
    rw [mul_comm]
  have hm : Measurable (linear w) := by unfold linear; fun_prop
  have hs : Measurable (scalarCap ρ) := by unfold scalarCap; fun_prop
  unfold rankCapIntegral
  rw [hf]
  calc
    _ = ∫ t, scalarCap ρ t ∂((gaussian d).map (linear w)) :=
      (integral_map hm.aemeasurable hs.aestronglyMeasurable).symm
    _ = _ := by rw [standard_projection w hw]

theorem scalarCap_gap (ρ : ℝ) (hρ : 0<ρ) :
    0<rankTailTolerance ρ ∧
      (∫ t, scalarCap ρ t ∂(gaussianReal 0 1)) ≤ 1-2*rankTailTolerance ρ := by
  let μ := gaussianReal 0 1
  let S := Ioi ρ
  have hpos : 0 < μ.real S := by
    have h : (volume : Measure ℝ) ≪ μ := gaussianReal_absolutelyContinuous' 0 one_ne_zero
    have hn : μ S≠0 := fun H =>
      (isOpen_Ioi.measure_ne_zero (μ := (volume : Measure ℝ))
        ⟨ρ+1,show ρ<ρ+1 by linarith⟩) (h H)
    exact ENNReal.toReal_pos hn (measure_ne_top _ _)
  refine ⟨div_pos hpos (by norm_num),?_⟩
  have hCap : Integrable (scalarCap ρ) μ := by
    refine Integrable.mono' (integrable_const (1:ℝ)) ?_
      (ae_of_all _ (fun x => ?_))
    · apply Measurable.aestronglyMeasurable
      unfold scalarCap
      fun_prop
    · rw [Real.norm_eq_abs,abs_of_nonneg (le_max_right _ _)]
      exact max_le (by nlinarith [sq_nonneg (x/ρ)]) zero_le_one
  have hMeas : MeasurableSet Sᶜ := measurableSet_Ioi.compl
  have hh : Integrable (Sᶜ.indicator (fun _ => (1 : ℝ))) μ := (integrable_const _).indicator hMeas
  have H (x : ℝ) : scalarCap ρ x ≤ Sᶜ.indicator (fun _ => (1 : ℝ)) x := by
    by_cases hx : x∈S
    · rw [indicator_of_notMem (by exact fun hg => hg hx)]
      unfold scalarCap
      have hh : 1≤x/ρ := (le_div_iff₀ hρ).2 (by simpa using (le_of_lt hx))
      have hb : 1≤(x/ρ)^2 := by nlinarith [sq_nonneg (x/ρ-1)]
      exact max_le (by linarith) le_rfl
    · rw [indicator_of_mem (by exact hx)]
      unfold scalarCap
      exact max_le (by nlinarith [sq_nonneg (x/ρ)]) zero_le_one
  have hb := integral_mono hCap hh H
  rw [integral_indicator hMeas,setIntegral_const,smul_eq_mul,mul_one,
    probReal_compl_eq_one_sub measurableSet_Ioi] at hb
  change _ ≤ 1-2*(μ.real S/2)
  linarith

theorem rank_two_cap_lower (ρ : ℝ) (hρ : 0<ρ) :
    0<rankTailTolerance ρ ∧
      ∀ (d n : ℕ), n<d → ∀ T : ValueTree (Point d) ℝ, T.within n →
        ∃ G : RankTwoCap d, rankTailTolerance ρ ≤ |T.run (rankCap ρ G)-rankCapIntegral ρ G| := by
  let δ := rankTailTolerance ρ
  obtain ⟨hδ,hgap⟩ := scalarCap_gap ρ hρ
  refine ⟨hδ,?_⟩
  intro d n hn T hT
  let G₀ : RankTwoCap d := ⟨0,0,by intro x; simp [linear]; linarith [normSq_nonneg x]⟩
  let f := rankCap ρ G₀
  let L := T.trace f
  have hL := (ValueTree.trace_length T f hT).trans_lt hn
  obtain ⟨w,hw,hann⟩ := invisible_unit hL (fun i : Fin L.length => ρ⁻¹ • L.get i)
  let G := unitCap w hw
  have hF : f = fun _ => 1 := by
    funext x
    simp [f,rankCap,matrixCap,quadratic_gram,G₀,linear]
  have hZ : rankCapIntegral ρ G₀=1 := by
    change (∫ x, f x ∂(gaussian d))=1
    rw [hF]
    simp
  have hI : rankCapIntegral ρ G≤1-2*δ := by rwa [unitCap_integral]
  have heq : T.run (rankCap ρ G)=T.run f := by
    apply ValueTree.run_eq_of_agree
    intro x hx
    obtain ⟨i,hi⟩ := List.get_of_mem hx
    have H : linear w (ρ⁻¹ • x)=0 := by rw [← hi]; exact hann i
    have hz : linear (0 : Point d) (ρ⁻¹ • x)=0 := by simp [linear]
    rw [hF]
    norm_num [rankCap,matrixCap,quadratic_gram,G,unitCap,H,hz]
  have HE := abs_two_targets (c := T.run f) (a := rankCapIntegral ρ G)
    (b := 1) (w := 1-rankCapIntegral ρ G) rfl
  have HH : δ≤(1-rankCapIntegral ρ G)/2 := by linarith
  rcases HE with h | h
  · refine ⟨G,?_⟩
    rw [heq]
    exact HH.trans h
  · refine ⟨G₀,?_⟩
    rw [hZ]
    exact HH.trans h

end Cubature
