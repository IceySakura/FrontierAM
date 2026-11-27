import Theory.Schedule
import Theory.Spectral

open MeasureTheory hiding average average_const average_sub
open scoped ENNReal

set_option maxHeartbeats 1200000
set_option linter.style.haveILetI false

noncomputable section

namespace ULB63
namespace Upper

open Protocol Composite FiniteMean Epoch Schedule

variable {n m d : ℕ} {L μ Δ ρ ε : ℝ} {I : Instance n m d}

private theorem control_error (hI : I.InClass L μ Δ) (b y : E d) (s : NodeDraw n m) :
    control I b y s - gradient I.f y =
      average (fun i => let v := fun j : Fin m => gradient (I.F i j) y - gradient (I.F i j) b
                        v (s i) - average v) := by
  simp only [Schedule.control, fullGrad hI b, fullGrad hI y, average_sub, localGrad]
  abel

theorem control_mean (hI : I.InClass L μ Δ) [Nonempty (Fin m)] (b y : E d) :
    average (fun s : NodeDraw n m => control I b y s - gradient I.f y) = 0 := by
  let v := fun i (j : Fin m) => gradient (I.F i j) y - gradient (I.F i j) b
  let z := fun i j => v i j - average (v i)
  have h (i : Fin n) : average (z i) = 0 := by
    simp only [z, average_sub, average_const, sub_self]
  simp only [control_error hI]
  change average (fun s : NodeDraw n m => (Fintype.card (Fin n) : ℝ)⁻¹ • ∑ i, z i (s i)) = 0
  rw [average_smul, indep_sum_centered n z h, smul_zero]

theorem control_variance (hI : I.InClass L μ Δ) [Nonempty (Fin n)] [Nonempty (Fin m)]
    (b y : E d) :
    average (fun s : NodeDraw n m => ‖control I b y s - gradient I.f y‖^2) ≤
      (L^2 / n) * ‖y-b‖^2 := by
  let v := fun i (j : Fin m) => gradient (I.F i j) y - gradient (I.F i j) b
  let z := fun i j => v i j - average (v i)
  have h (i : Fin n) : average (z i) = 0 := by
    simp only [z, average_sub, average_const, sub_self]
  have hb (i : Fin n) : average (fun j => ‖z i j‖^2) ≤ L^2 * ‖y-b‖^2 := by
    change average (fun j => ‖v i j - average (v i)‖^2) ≤ _
    rw [centered_square]
    refine (sub_le_self _ (sq_nonneg _)).trans ?_
    rw [← average_const (ι := Fin m) (L^2 * ‖y-b‖^2)]
    apply average_mono
    intro j
    rw [← mul_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) ((hI.smooth i j).2 y b) 2
  have hnr : (0 : ℝ) < n := by
    have := Fintype.card_pos (α := Fin n)
    simpa using (show (0 : ℝ) < (Fintype.card (Fin n) : ℝ) by exact_mod_cast this)
  simp only [control_error hI]
  change average (fun s : NodeDraw n m => ‖(Fintype.card (Fin n) : ℝ)⁻¹ • ∑ i, z i (s i)‖^2) ≤ _
  simp only [Fintype.card_fin, norm_smul, Real.norm_eq_abs, abs_of_nonneg (inv_nonneg.mpr hnr.le),
    mul_pow, average_mul, indep_sum_square n z h]
  calc
    _ ≤ ((n : ℝ)⁻¹)^2 * ∑ _ : Fin n, (L^2 * ‖y-b‖^2) :=
      mul_le_mul_of_nonneg_left (Finset.sum_le_sum (fun i _ => hb i)) (sq_nonneg _)
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp

theorem fromSchedule (hn : 0 < n) (B : ℕ) (σ : Type) [Fintype σ] [Nonempty σ]
    (p : (d : ℕ) → Info n d σ → Plan m d (E d))
    (f : (d : ℕ) → Instance n m d → σ → E d)
    (hvalid : ∀ d (info : Info n d σ), (p d info).Valid Supported (supp info.x0))
    (hexec : ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ), I.InClass L μ Δ →
      CommClass n ρ W → ∀ s,
        Plan.Executes I W B (fun i => p d ⟨i, W i, I.x0, s⟩) (fun _ => f d I s))
    (hfeas : ∀ d (I : Instance n m d), I.InClass L μ Δ → ∀ s, feasible I (f d I s))
    (hbound : ∀ d (I : Instance n m d), I.InClass L μ Δ → ∀ u, (∀ x, I.φ u ≤ I.φ x) →
      average (fun s => loss I u (f d I s)) ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W → A.hittingTime I W ε ≤ (B : ℕ∞) := by
  letI : MeasurableSpace σ := ⊤
  let A := compile (uniform σ) p
  refine ⟨A, compile_admissible _ _ hvalid, ?_⟩
  letI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  intro d I W hI hW
  obtain ⟨u, hu⟩ := hI.minimizer
  have hout (s : σ) (i : Fin n) : A.output I W s B i = f d I s :=
    compiled_output_of_executes (uniform σ) p I W s (hexec d I W hI hW s) i
  have havg (s : σ) : A.avgOutput I W s B = f d I s := by
    unfold Algorithm.avgOutput
    simp only [hout, one_div]
    simpa only [average, Fintype.card_fin] using average_const (ι := Fin n) (f d I s)
  have hgap : A.expectedGap I W B ≤ ENNReal.ofReal ε := by
    unfold Algorithm.expectedGap
    simp_rw [havg, fun s => gap_eq hI hu (hfeas d I hI s)]
    change (∫⁻ s, ENNReal.ofReal (loss I u (f d I s)) ∂uniform σ) ≤ _
    rw [uniform_lintegral σ _ (fun s => loss_nonneg hI hu (hfeas d I hI s))]
    exact ENNReal.ofReal_le_ofReal (hbound d I hI u hu)
  exact iInf₂_le B hgap

theorem rate_parameters (hμ : 0 < μ) (hL : μ ≤ L) :
    0 < Real.sqrt (μ/(8*L)) ∧ Real.sqrt (μ/(8*L)) ≤ 1 ∧
      2*L*(Real.sqrt (μ/(8*L)))^2 = μ/4 := by
  have hLp : 0 < L := hμ.trans_le hL
  have hq : 0 < μ/(8*L) := by positivity
  have hle : μ/(8*L) ≤ 1 := (div_le_one (by positivity)).2 (by linarith)
  refine ⟨Real.sqrt_pos.2 hq, ?_, ?_⟩
  · simpa using Real.sqrt_le_sqrt hle
  · rw [Real.sq_sqrt hq.le]
    field_simp
    ring

theorem exact_bound (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D R H : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hH : (1/2 : ℝ)^H * Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤ ((H * (R * (m+4*D+1)) : ℕ) : ℕ∞) := by
  let root : Fin n := ⟨0, by omega⟩
  let a := Real.sqrt (μ/(8*L))
  have hLp := hμ.trans_le hL
  obtain ⟨ha, ha₁, ha₂⟩ := rate_parameters hμ hL
  letI : Nonempty (Fin n) := ⟨root⟩
  letI : Nonempty (Fin m) := ⟨⟨0, by omega⟩⟩
  let g (d : ℕ) (I : Instance n m d) (y : E d) (_ : Unit) := gradient I.f y
  let once (d : ℕ) (I : Instance n m d) (x : E d) (s : Draws Unit R) :=
    (evaluate I L a (g d I) R s x x).mean
  let single (d : ℕ) (i : Fin n) (x : E d) (s : Draws Unit R) : Plan m d (E d) :=
    inner (exactOracle D root i) L a R s x x
  apply fromSchedule (by omega : 0 < n) _ (Draws (Draws Unit R) H)
    (fun d info => repeating (single d info.node) H info.seed info.x0)
    (fun d I s => repeatEval (once d I) H s I.x0)
  · intro d info
    apply repeating_valid _ _ _ _ (subset_refl _)
    intro S x hx s
    exact inner_valid _ hLp a (B := S) (fun _ _ _ hy j => exactOracle_valid D root info.node hy j)
      R s (subset_refl _) hx hx
  · intro d I W hI hW s
    have hp := Network.certified_radius (by omega) hρ hρ₁ hW hD
    apply repeating_exec _ (once d I) _
    intro y t
    exact inner_exec _ L a (g d I) _ (fun x j => exactOracle_exec hI hW D hp root x j) R t y y
  · intro d I hI s
    obtain ⟨u, hu⟩ := hI.minimizer
    exact repeat_feasible (once d I)
      (fun x hx t => (evaluate_feasible hI hLp a (g d I) R t hx x).2) H s (initial_feasible hI hu)
  · intro d I hI u hu
    have hf (x : E d) (hx : feasible I x) (s : Draws Unit R) : feasible I (once d I x s) :=
      (evaluate_feasible hI hLp a (g d I) R s hx x).2
    have hg (x : E d) (hx : feasible I x) :
        average (fun s => loss I u (once d I x s)) ≤ loss I u x / 2 :=
      contract hI hLp hμ hu hx ha ha₁ ha₂ (show (0:ℝ) ≤ 0 by rfl) (g d I)
        (by intro y; simp [g, average_const])
        (by intro y; simp [g, average_const]) R hR (by simp)
    exact (repeat_contract (once d I) hf hg H (initial_feasible hI hu)).trans
      ((mul_le_mul_of_nonneg_left (initial_loss hI hu) (by positivity)).trans hH)

theorem parallel_bound (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D R H : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hparallel : 8*(L/μ)*(R : ℝ) ≤ n)
    (hH : (1/2 : ℝ)^H * Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤ ((H * (m+4*D+R*(2+4*D)) : ℕ) : ℕ∞) := by
  let root : Fin n := ⟨0, by omega⟩
  let a := Real.sqrt (μ/(8*L))
  have hLp := hμ.trans_le hL
  obtain ⟨ha, ha₁, ha₂⟩ := rate_parameters hμ hL
  letI : Nonempty (Fin n) := ⟨root⟩
  letI : Nonempty (Fin m) := ⟨⟨0, by omega⟩⟩
  let τ := Draws (NodeDraw n m) R
  let once (d : ℕ) (I : Instance n m d) (x : E d) (s : τ) :=
    (evaluate I L a (control I x) R s x x).mean
  have hnr : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hnoise : (2*(L^2/(n:ℝ))/(L*μ))*(R:ℝ) ≤ 1/4 := by
    have he : (2*(L^2/(n:ℝ))/(L*μ))*(R:ℝ) = (8*(L/μ)*(R:ℝ))/((4:ℝ)*n) := by
      field_simp
      ring
    rw [he]
    exact (div_le_iff₀ (by positivity)).2 (by convert hparallel using 1; ring)
  apply fromSchedule (by omega : 0 < n) _ (Draws τ H)
    (fun _ info => repeating (parallelEpoch D root info.node L a R) H info.seed info.x0)
    (fun d I s => repeatEval (once d I) H s I.x0)
  · intro d info
    exact repeating_valid _ (fun _ _ hx s => parallelEpoch_valid D root info.node L a hLp R hx s)
      H info.seed (subset_refl _)
  · intro d I W hI hW s
    have hp := Network.certified_radius (by omega) hρ hρ₁ hW hD
    apply repeating_exec _ (once d I) _
    intro x t
    exact parallelEpoch_exec hI hW D hp root L a R x t
  · intro d I hI s
    obtain ⟨u, hu⟩ := hI.minimizer
    exact repeat_feasible (once d I)
      (fun x hx t => (evaluate_feasible hI hLp a (control I x) R t hx x).2) H s (initial_feasible hI hu)
  · intro d I hI u hu
    have hf (x : E d) (hx : feasible I x) (s : τ) : feasible I (once d I x s) :=
      (evaluate_feasible hI hLp a (control I x) R s hx x).2
    have hg (x : E d) (hx : feasible I x) :
        average (fun s => loss I u (once d I x s)) ≤ loss I u x / 2 :=
      contract hI hLp hμ hu hx ha ha₁ ha₂ (show 0 ≤ L^2/(n:ℝ) by positivity)
        (control I x) (control_mean hI x) (control_variance hI x) R hR hnoise
    exact (repeat_contract (once d I) hf hg H (initial_feasible hI hu)).trans
      ((mul_le_mul_of_nonneg_left (initial_loss hI hu) (by positivity)).trans hH)

end Upper
end ULB63
