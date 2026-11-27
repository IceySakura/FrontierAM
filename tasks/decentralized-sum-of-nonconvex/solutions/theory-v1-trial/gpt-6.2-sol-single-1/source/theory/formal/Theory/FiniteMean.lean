import Theory.Composite

open MeasureTheory hiding average average_const
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63
namespace FiniteMean

open Composite

variable {σ τ V : Type*} [Fintype σ] [Fintype τ] [AddCommMonoid V] [Module ℝ V]

theorem average_add (f g : σ → V) :
    average (fun s => f s + g s) = average f + average g := by
  simp [average, Finset.sum_add_distrib, smul_add]

theorem average_smul (a : ℝ) (f : σ → V) :
    average (fun s => a • f s) = a • average f := by
  simp only [average, Finset.smul_sum, smul_comm a (Fintype.card σ : ℝ)⁻¹]

theorem average_mul (a : ℝ) (f : σ → ℝ) :
    average (fun s => a * f s) = a * average f :=
  average_smul a f

theorem average_sub {V : Type*} [AddCommGroup V] [Module ℝ V] (f g : σ → V) :
    average (fun s => f s - g s) = average f - average g := by
  simp [average, Finset.sum_sub_distrib, smul_sub]

theorem average_mono {f g : σ → ℝ} (h : ∀ s, f s ≤ g s) :
    average f ≤ average g :=
  mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun s _ => h s) (by positivity)

theorem average_equiv (e : σ ≃ τ) (f : τ → V) :
    average f = average (fun s => f (e s)) := by
  unfold average
  rw [Fintype.card_congr e, ← Fintype.sum_equiv e (fun s => f (e s)) f (fun _ => rfl)]

theorem average_prod (f : σ × τ → V) :
    average f = average (fun s => average (fun t => f (s,t))) := by
  simp only [average, Fintype.card_prod, Nat.cast_mul, mul_inv, mul_smul, Finset.smul_sum,
    Fintype.sum_prod_type]

theorem average_inner_left {d : ℕ} (f : σ → E d) (v : E d) :
    average (fun s => ⟪f s,v⟫) = ⟪average f,v⟫ := by
  simp [average, real_inner_smul_left, sum_inner]

theorem average_inner_right {d : ℕ} (v : E d) (f : σ → E d) :
    average (fun s => ⟪v,f s⟫) = ⟪v,average f⟫ := by
  simp [average, real_inner_smul_right, inner_sum]

theorem centered_square {d : ℕ} [Nonempty σ] (f : σ → E d) :
    average (fun s => ‖f s - average f‖ ^ 2) =
      average (fun s => ‖f s‖ ^ 2) - ‖average f‖ ^ 2 := by
  simp only [norm_sub_sq_real, average_add, average_sub, average_mul, average_inner_left,
    average_const, real_inner_self_eq_norm_sq]
  ring

private theorem indep_split {s : ℕ} (f : (Fin (s+1) → σ) → V) :
    average f = average (fun a : σ => average (fun b : Fin s → σ => f (Fin.cons a b))) := by
  rw [average_equiv (Fin.consEquiv (fun _ : Fin (s+1) => σ)), average_prod]
  rfl

/-- Independent local draws, with separate centering at every node. -/
theorem indep_sum_centered {d : ℕ} [Nonempty σ] (n : ℕ) (v : Fin n → σ → E d)
    (hv : ∀ i, average (v i) = 0) :
    average (fun s : Fin n → σ => ∑ i, v i (s i)) = 0 := by
  induction n with
  | zero => simp [average]
  | succ n ih =>
    have ht := ih (fun i => v i.succ) (fun i => hv i.succ)
    rw [indep_split]
    simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ, average_add, average_const, ht,
      add_zero]
    exact hv 0

theorem indep_sum_square {d : ℕ} [Nonempty σ] (n : ℕ) (v : Fin n → σ → E d)
    (hv : ∀ i, average (v i) = 0) :
    average (fun s : Fin n → σ => ‖∑ i, v i (s i)‖ ^ 2) =
      ∑ i, average (fun u => ‖v i u‖ ^ 2) := by
  induction n with
  | zero => simp [average]
  | succ n ih =>
    have hx := indep_sum_centered n (fun i => v i.succ) (fun i => hv i.succ)
    have hy := ih (fun i => v i.succ) (fun i => hv i.succ)
    rw [indep_split, Fin.sum_univ_succ]
    simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ, norm_add_sq_real,
      average_add, average_mul, average_const, average_inner_right, hx, inner_zero_right,
      mul_zero, add_zero, hy]

/-- A finite product of independent uniform draws, in chronological order. -/
def Draws (σ : Type) : ℕ → Type
  | 0 => Unit
  | k+1 => σ × Draws σ k

instance drawsFintype {σ : Type} [Fintype σ] : (k : ℕ) → Fintype (Draws σ k)
  | 0 => inferInstanceAs (Fintype Unit)
  | k+1 =>
    letI : Fintype (Draws σ k) := @drawsFintype σ _ k
    inferInstanceAs (Fintype (σ × Draws σ k))

instance drawsNonempty {σ : Type} [Nonempty σ] : (k : ℕ) → Nonempty (Draws σ k)
  | 0 => inferInstanceAs (Nonempty Unit)
  | k+1 =>
    letI : Nonempty (Draws σ k) := @drawsNonempty σ _ k
    inferInstanceAs (Nonempty (σ × Draws σ k))

def drawHead {σ : Type} {k : ℕ} (r : Draws σ (k+1)) : σ :=
  (show σ × Draws σ k from r).1

def drawTail {σ : Type} {k : ℕ} (r : Draws σ (k+1)) : Draws σ k :=
  (show σ × Draws σ k from r).2

theorem average_draws_succ {σ V : Type} [Fintype σ] [AddCommMonoid V] [Module ℝ V]
    {k : ℕ} (f : Draws σ (k+1) → V) :
    average f = average (fun s : σ => average (fun t : Draws σ k =>
      f (show Draws σ (k+1) from (s,t)))) :=
  average_prod (show σ × Draws σ k → V from f)

section Measure

variable (σ : Type) [Fintype σ] [Nonempty σ] [MeasurableSpace σ] [DiscreteMeasurableSpace σ]

def uniform : Measure σ := (Fintype.card σ : ℝ≥0∞)⁻¹ • Measure.count

instance uniform_prob : IsProbabilityMeasure (uniform σ) where
  measure_univ := by
    have h := Fintype.card_ne_zero (α := σ)
    simp only [uniform, Measure.smul_apply, Measure.count_univ, ENat.card_eq_coe_fintype_card,
      smul_eq_mul]
    exact ENNReal.inv_mul_cancel (by exact_mod_cast h) (by simp)

theorem uniform_lintegral (f : σ → ℝ) (hf : ∀ s, 0 ≤ f s) :
    (∫⁻ s, ENNReal.ofReal (f s) ∂uniform σ) = ENNReal.ofReal (average f) := by
  have hcard : 0 < (Fintype.card σ : ℝ) := by exact_mod_cast Fintype.card_pos
  rw [uniform, lintegral_smul_measure, lintegral_count, tsum_fintype, average]
  simp only [smul_eq_mul]
  rw [ENNReal.ofReal_mul (inv_nonneg.mpr hcard.le), ENNReal.ofReal_inv_of_pos hcard,
    ENNReal.ofReal_natCast, ENNReal.ofReal_sum_of_nonneg (fun i _ => hf i)]

end Measure

end FiniteMean
end ULB63
