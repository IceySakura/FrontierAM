import Theory.Acceleration

/-! The local-average-only relaxation, phrased using the fixed class for the unchanged geometry. -/

open Finset
noncomputable section

namespace ULB63.Instance

variable {n m d : ℕ} (I : Instance n m d)

/-- For verifying geometry only: replace each component at node `i` by its local average.
Algorithms still query the original instance, not this verification instance. -/
def collapsed : Instance n m d where
  F i _ := I.localAverage i
  h := I.h
  x0 := I.x0
  prox := I.prox

/-- Same geometry and shared prox as `InClass`, differentiable components with no individual
Lipschitz bound, and `L`-smooth local averages. Used with `m ≥ 1`. -/
def InLocalClass (L μ Δ : ℝ) : Prop :=
  (∀ i ξ, Differentiable ℝ (I.F i ξ)) ∧ I.collapsed.InClass L μ Δ

theorem collapsed_local_eq (hm : m ≠ 0) (i : Fin n) :
    I.collapsed.localAverage i = I.localAverage i := by
  funext x
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  simp [localAverage, collapsed, nsmul_eq_mul, one_div, hm']

theorem collapsed_f_eq (hm : m ≠ 0) : I.collapsed.f = I.f := by
  change (fun x => (1/n : ℝ) * ∑ i, I.collapsed.localAverage i x) =
    (fun x => (1/n : ℝ) * ∑ i, I.localAverage i x)
  simp only [collapsed_local_eq I hm]

theorem collapsed_phi_eq (hm : m ≠ 0) : I.collapsed.φ = I.φ := by
  unfold φ
  rw [collapsed_f_eq I hm]
  rfl

theorem collapsed_gap_eq (hm : m ≠ 0) : I.collapsed.gap = I.gap := by
  unfold gap φstar
  simp only [collapsed_phi_eq I hm]

theorem inLocalClass_of_local {M L μ Δ : ℝ} (hm : m ≠ 0) (hI : I.InClass M μ Δ)
    (hlocal : ∀ i, LSmooth L (I.localAverage i)) : I.InLocalClass L μ Δ := by
  have hφ := I.collapsed_phi_eq hm
  refine ⟨fun i ξ => (hI.smooth i ξ).1, ?_⟩
  refine ⟨fun i _ => hlocal i, ?_, hI.h_ne_bot, hI.h_proper, hI.h_closed, hI.h_convex,
    hI.prox_exact, ?_, ?_, ?_⟩
  · rw [I.collapsed_f_eq hm]; exact hI.f_convex
  · rw [hφ]; exact hI.φ_strongly_convex
  · rw [hφ]; exact hI.minimizer
  · unfold φstar
    rw [hφ]
    exact hI.initial_gap

theorem inLocalClass_of_standard {L μ Δ : ℝ} (hm : m ≠ 0) (hI : I.InClass L μ Δ) :
    I.InLocalClass L μ Δ :=
  I.inLocalClass_of_local hm hI (localSmooth_of_components hm hI.smooth)

theorem collapsed_proxGrad_eq (hm : m ≠ 0) (L : ℝ) (y : E d) :
    proxGrad I.collapsed L y = proxGrad I L y := by
  unfold proxGrad
  rw [I.collapsed_f_eq hm]
  rfl

theorem collapsed_stage_eq (hm : m ≠ 0) (L : ℝ) (x : E d) (j : ℕ) :
    Acceleration.stage I.collapsed L x j = Acceleration.stage I L x j := by
  induction j with
  | zero => rfl
  | succ j ih => simp only [Acceleration.stage, ih, Acceleration.step, I.collapsed_proxGrad_eq hm]

theorem collapsed_restart_eq (hm : m ≠ 0) (L : ℝ) (p K : ℕ) :
    Acceleration.restart I.collapsed L p K = Acceleration.restart I L p K := by
  induction K with
  | zero => rfl
  | succ K ih => simp only [Acceleration.restart, ih, I.collapsed_stage_eq hm]

end ULB63.Instance
