import Theory.Composite
import Theory.Chain

open Set
open scoped RealInnerProductSpace

set_option maxHeartbeats 1400000

noncomputable section

namespace ULB63
namespace Regularized

open Composite

variable {n m d : ℕ} {I : Instance n m d} {L μ Δ ν : ℝ}

def quad (ν : ℝ) (y x : E d) : ℝ := ν/2 * ‖x-y‖^2

def shift (ν : ℝ) (y v : E d) (η : ℝ) : E d :=
  (1+η*ν)⁻¹ • (v+(η*ν) • y)

def stepSize (ν η : ℝ) : ℝ := η/(1+η*ν)

def instanceAt (I : Instance n m d) (ν : ℝ) (y x : E d) : Instance n m d where
  F := I.F
  h v := I.h v + (quad ν y v : EReal)
  x0 := x
  prox v η := I.prox (shift ν y v η) (stepSize ν η)

@[simp] theorem f_eq (ν : ℝ) (y x : E d) : (instanceAt I ν y x).f = I.f := rfl

theorem phi_eq (ν : ℝ) (y x v : E d) :
    (instanceAt I ν y x).φ v = I.φ v + (quad ν y v : EReal) := by
  change (I.f v:EReal)+(I.h v+(quad ν y v:EReal)) = ((I.f v:EReal)+I.h v)+(quad ν y v:EReal)
  rw [add_assoc]

theorem feasible_iff (ν : ℝ) (y x v : E d) :
    feasible (instanceAt I ν y x) v ↔ feasible I v := by
  change I.h v + (quad ν y v : EReal) ≠ ⊤ ↔ I.h v ≠ ⊤
  exact EReal.add_ne_top_iff_ne_top_left (EReal.coe_ne_bot _) (EReal.coe_ne_top _)

theorem value_eq (hI : I.InClass L μ Δ) (ν : ℝ) (y x : E d) {v : E d} (hv : feasible I v) :
    value (instanceAt I ν y x) v = value I v+quad ν y v := by
  change I.f v +(I.h v+(quad ν y v:EReal)).toReal = _
  rw [EReal.toReal_add hv (hI.h_ne_bot v) (EReal.coe_ne_top _) (EReal.coe_ne_bot _), EReal.toReal_coe]
  unfold value
  ring

private theorem add_real_lsc {θ : E d → EReal} {g : E d → ℝ}
    (hθ : LowerSemicontinuous θ) (hg : Continuous g) :
    LowerSemicontinuous (fun z => θ z + (g z : EReal)) :=
  hθ.add' (continuous_coe_real_ereal.comp hg).lowerSemicontinuous
    (fun _ => EReal.continuousAt_add (.inr (EReal.coe_ne_bot _)) (.inr (EReal.coe_ne_top _)))

private theorem quad_cont (ν : ℝ) (y : E d) : Continuous (quad ν y) := by
  unfold quad
  fun_prop

private theorem sigma_nonbot (hI : I.InClass L μ Δ) (v : E d) : I.φ v ≠ ⊥ := by
  exact EReal.add_ne_bot_iff.mpr ⟨EReal.coe_ne_bot _,hI.h_ne_bot _⟩

private theorem addQuad_strong {θ : E d → EReal} {a : ℝ}
    (hθ : StronglyConvexE a θ) (hbot : ∀ x, θ x ≠ ⊥) (ν : ℝ) (y : E d) :
    StronglyConvexE (a+ν) (fun x => θ x+(quad ν y x : EReal)) := by
  intro x z t ht ht₁
  by_cases h0 : t=0
  · subst t; simp
  by_cases h1 : t=1
  · subst t; simp
  have ht' : 0 < t := lt_of_le_of_ne ht (fun h => h0 h.symm)
  have ht₁' : 0 < 1-t := by
    have := lt_of_le_of_ne ht₁ h1
    linarith
  have hn (v : E d) : θ v + (quad ν y v:EReal) ≠ ⊥ :=
    EReal.add_ne_bot_iff.mpr ⟨hbot _,EReal.coe_ne_bot _⟩
  have hp (v : E d) (r : ℝ) (hr : 0 ≤ r) :
      (r:EReal)*(θ v + (quad ν y v:EReal)) ≠ ⊥ := by
    exact (EReal.mul_ne_bot _ _).2
      ⟨.inl (EReal.coe_ne_bot _), .inr (hn v), .inl (EReal.coe_ne_top _),
        .inl (by exact_mod_cast hr)⟩
  by_cases hx : θ x = ⊤
  · have he : (t:EReal)*(θ x + (quad ν y x:EReal)) = ⊤ := by
      rw [hx,EReal.top_add_coe,mul_comm,EReal.top_mul_coe_of_pos ht']
    simp only [he,EReal.top_add_of_ne_bot (hp z (1-t) ht₁'.le),le_top]
  by_cases hz : θ z = ⊤
  · have he : ((1-t:ℝ):EReal)*(θ z + (quad ν y z:EReal)) = ⊤ := by
      rw [hz,EReal.top_add_coe,mul_comm,EReal.top_mul_coe_of_pos ht₁']
    simp only [he,EReal.add_top_of_ne_bot (hp x t ht),le_top]
  let v := t • x+(1-t) • z
  have H := hθ x z t ht ht₁
  change θ v + ((a/2*t*(1-t)*‖x-z‖^2:ℝ):EReal) ≤ _ at H
  rw [← EReal.coe_toReal hx (hbot x), ← EReal.coe_toReal hz (hbot z)] at H
  have H' : θ v + ((a/2*t*(1-t)*‖x-z‖^2:ℝ):EReal) ≤
      ((t*(θ x).toReal +(1-t)*(θ z).toReal:ℝ):EReal) := by
    push_cast
    exact H
  have hv : θ v ≠ ⊤ := by
    intro ht
    rw [ht,EReal.top_add_coe,top_le_iff] at H'
    exact EReal.coe_ne_top _ H'
  rw [← EReal.coe_toReal hv (hbot v), ← EReal.coe_add, EReal.coe_le_coe_iff] at H'
  have he : v-y = t • (x-y)+(1-t) • (z-y) := by dsimp [v]; module
  have hdiff : (x-y)-(z-y) = x-z := by abel
  have hnq := Chain.normSq_bary t ht ht₁ (x-y) (z-y)
  rw [hdiff, ← he] at hnq
  have he' := congrArg (fun r : ℝ => ν/2*r) hnq
  change θ v + (quad ν y v:EReal) + _ ≤
    (t:EReal)*(θ x+(quad ν y x : EReal)) + ((1-t:ℝ):EReal)*(θ z+(quad ν y z:EReal))
  rw [← EReal.coe_toReal hv (hbot v), ← EReal.coe_toReal hx (hbot x),
    ← EReal.coe_toReal hz (hbot z), ← EReal.coe_add, ← EReal.coe_add,
    ← EReal.coe_add, ← EReal.coe_add, ← EReal.coe_mul, ← EReal.coe_mul, ← EReal.coe_add,
    EReal.coe_le_coe_iff]
  unfold quad
  nlinarith

theorem shift_size_pos {η : ℝ} (hν : 0 ≤ ν) (hη : 0 < η) : 0 < stepSize ν η := by
  unfold stepSize
  positivity

private theorem complete (hν : 0 ≤ ν) {η : ℝ} (hη : 0 < η) (y v x : E d) :
    quad ν y x + ‖x-v‖^2/(2*η) =
      ‖x-shift ν y v η‖^2/(2*stepSize ν η) + (ν/(2*(1+η*ν)))*‖v-y‖^2 := by
  let a := 1+η*ν
  have ha : 0 < a := by dsimp [a]; positivity
  have hshift : shift ν y v η = y+a⁻¹ • (v-y) := by
    unfold shift
    change a⁻¹ • (v+(η*ν) • y) = _
    rw [show v+(η*ν) • y = a • y+(v-y) by dsimp [a]; module,
      smul_add,smul_smul,inv_mul_cancel₀ ha.ne',one_smul]
  rw [show x-v = (x-y)-(v-y) by abel,
    show x-shift ν y v η = (x-y)-a⁻¹ • (v-y) by rw [hshift]; abel,
    norm_sub_sq_real (x-y) (v-y), norm_sub_sq_real (x-y) (a⁻¹ • (v-y)),
    real_inner_smul_right, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (inv_nonneg.mpr ha.le)]
  unfold quad stepSize
  change ν/2*‖x-y‖^2+_ = _/(2*(η/a))+ν/(2*a)*‖v-y‖^2
  field_simp
  ring

theorem prox_exact (hI : I.InClass L μ Δ) (hν : 0 ≤ ν) (y x v : E d) {η : ℝ} (hη : 0 < η) :
    IsProx (instanceAt I ν y x).h η v ((instanceAt I ν y x).prox v η) := by
  let p := I.prox (shift ν y v η) (stepSize ν η)
  have hp := hI.prox_exact (shift ν y v η) _ (shift_size_pos hν hη)
  intro w
  change (I.h p+(quad ν y p:EReal)) + ((‖p-v‖^2/(2*η):ℝ):EReal) ≤
    (I.h w+(quad ν y w:EReal)) + ((‖w-v‖^2/(2*η):ℝ):EReal)
  rw [add_assoc,add_assoc,← EReal.coe_add,← EReal.coe_add,
    complete hν hη, complete hν hη, EReal.coe_add,EReal.coe_add,← add_assoc,← add_assoc]
  exact add_le_add_left (hp w) _

theorem min_exists (hI : I.InClass L μ Δ) (hν : 0 < ν) (y : E d) {x : E d}
    (hx : feasible I x) : ∃ u, feasible I u ∧
      ∀ v, (instanceAt I ν y x).φ u ≤ (instanceAt I ν y x).φ v := by
  let Q := instanceAt I ν y x
  obtain ⟨u,hu⟩ := hI.minimizer
  have huF := optimal_feasible hI hu
  have hφc : LowerSemicontinuous I.φ := by
    have H := add_real_lsc hI.h_closed (differentiable_f hI).continuous
    change LowerSemicontinuous (fun v => (I.f v:EReal)+I.h v)
    simpa only [add_comm] using H
  have hQc : LowerSemicontinuous Q.φ := by
    rw [show Q.φ = fun v => I.φ v+(quad ν y v : EReal) from funext (phi_eq ν y x)]
    exact add_real_lsc hφc (quad_cont ν y)
  let S := {v : E d | Q.φ v ≤ Q.φ x}
  have hs : IsClosed S := hQc.isClosed_preimage _
  let r := value I x+quad ν y x-value I u
  have hQc_eq {v : E d} (hv : feasible I v) : Q.φ v = ((value I v+quad ν y v:ℝ):EReal) := by
    change (instanceAt I ν y x).φ v = _
    rw [phi_eq,value_coe hI hv,← EReal.coe_add]
  have hf {v : E d} (hv : Q.φ v ≤ Q.φ x) : feasible I v := by
    intro htop
    have hV : I.φ v = ⊤ := by simp [Instance.φ,htop]
    change (instanceAt I ν y x).φ v ≤ Q.φ x at hv
    rw [phi_eq, hV,EReal.top_add_coe,hQc_eq hx,top_le_iff] at hv
    exact EReal.coe_ne_top _ hv
  have hbound : S ⊆ Metric.closedBall y (Real.sqrt (2*r/ν)) := by
    intro v hv
    have hh : Q.φ v ≤ Q.φ x := hv
    have hvF := hf hh
    rw [hQc_eq hvF,hQc_eq hx,EReal.coe_le_coe_iff] at hh
    have hval := loss_nonneg hI hu hvF
    have hnorm : ‖v-y‖^2 ≤ 2*r/ν := by
      apply (le_div_iff₀ hν).2
      unfold r
      unfold quad at hh ⊢
      unfold loss at hval
      linarith
    rw [Metric.mem_closedBall,dist_eq_norm]
    exact Real.le_sqrt_of_sq_le hnorm
  have hc := (isCompact_closedBall y (Real.sqrt (2*r/ν))).of_isClosed_subset hs hbound
  obtain ⟨z,hz,hmin⟩ := LowerSemicontinuousOn.exists_isMinOn
    (show S.Nonempty from ⟨x, by change Q.φ x ≤ Q.φ x; exact le_rfl⟩)
    hc (hQc.lowerSemicontinuousOn S)
  refine ⟨z,hf hz,fun v => ?_⟩
  by_cases hv : v∈S
  · exact hmin hv
  · exact (show Q.φ z ≤ Q.φ x from hz).trans (le_of_lt (not_le.mp hv))

theorem inClass_at (hI : I.InClass L μ Δ) (hν : 0 ≤ ν) (y : E d) {x u : E d}
    (hx : feasible I x) (hu : feasible I u)
    (hU : ∀ v, (instanceAt I ν y x).φ u ≤ (instanceAt I ν y x).φ v) :
    (instanceAt I ν y x).InClass L (μ+ν) (loss (instanceAt I ν y x) u x) where
  smooth := hI.smooth
  f_convex := hI.f_convex
  h_ne_bot v := EReal.add_ne_bot_iff.mpr ⟨hI.h_ne_bot v,EReal.coe_ne_bot _⟩
  h_proper := ⟨x, (feasible_iff ν y x x).2 hx⟩
  h_closed := add_real_lsc hI.h_closed (quad_cont ν y)
  h_convex := by
    intro u v t ht ht₁
    have H := addQuad_strong hI.h_convex hI.h_ne_bot ν y u v t ht ht₁
    rw [zero_add] at H
    have h : (0:ℝ)/2*t*(1-t)*‖u-v‖^2 ≤ ν/2*t*(1-t)*‖u-v‖^2 := by
      simp only [zero_div,zero_mul]
      positivity
    exact le_trans (add_le_add le_rfl (EReal.coe_le_coe_iff.mpr h)) H
  prox_exact v η hη := prox_exact hI hν y x v hη
  φ_strongly_convex := by
    rw [funext (phi_eq ν y x)]
    exact addQuad_strong hI.φ_strongly_convex (sigma_nonbot hI) ν y
  minimizer := ⟨u,hU⟩
  initial_gap := by
    rw [star_eq _ hU]
    change (instanceAt I ν y x).φ x - (instanceAt I ν y x).φ u ≤ _
    unfold loss
    rw [phi_eq,phi_eq,value_coe hI hx,value_coe hI hu,← EReal.coe_add,← EReal.coe_add,
      ← EReal.coe_sub,value_eq hI ν y x hx,value_eq hI ν y x hu]

end Regularized
end ULB63
