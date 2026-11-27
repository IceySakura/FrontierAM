import Theory.Batch
import Theory.Inexact

open scoped ENNReal

set_option maxHeartbeats 1500000
set_option linter.style.haveILetI false

noncomputable section

namespace ULB63
namespace Nested

open Composite Epoch FiniteMean Regularized Protocol Schedule Upper

variable {n m d B : ℕ} {L μ Δ ρ ν ε : ℝ}
variable {I : Instance n m d} {W : Matrix (Fin n) (Fin n) ℝ}

def translate {α : Type} (ν : ℝ) (y : E d) : Plan m d α → Plan m d α
  | .done x => .done x
  | .grad j x k => .grad j x (fun r => translate ν y (k r))
  | .mix x k => .mix x (fun r => translate ν y (k r))
  | .prox v η k => .prox (shift ν y v η) (stepSize ν η) (fun r => translate ν y (k r))

theorem translate_valid {α : Type} {P : Set (Fin d) → α → Prop} {p : Plan m d α}
    (hν : 0 ≤ ν) {y : E d} {S : Set (Fin d)} (hy : Supported S y) (hp : p.Valid P S) :
    (translate ν y p).Valid P S := by
  induction p generalizing S with
  | done x => exact hp
  | grad j x k ih =>
    exact ⟨hp.1,fun v => ih v (hy.mono Set.subset_union_left) (hp.2 v)⟩
  | mix x k ih =>
    exact ⟨hp.1,fun v => ih v (hy.mono Set.subset_union_left) (hp.2 v)⟩
  | prox v η k ih =>
    exact ⟨⟨(hp.1.1.add (hy.smul _)).smul _,shift_size_pos hν hp.1.2⟩,
      fun w => ih w (hy.mono Set.subset_union_left) (hp.2 w)⟩

private theorem translate_messages {α : Type} (ν : ℝ) (y : E d) (p : Plan m d α) :
    (translate ν y p).message = p.message := by cases p <;> rfl

private theorem translate_active {α : Type} (ν : ℝ) (y : E d) (p : Plan m d α) :
    (translate ν y p).active ↔ p.active := by cases p <;> rfl

private theorem translate_step {α : Type} (ν : ℝ) (x y : E d)
    (p : Fin n → Plan m d α) :
    Plan.advance I W (fun i => translate ν y (p i)) =
      fun i => translate ν y (Plan.advance (instanceAt I ν y x) W p i) := by
  funext i
  unfold Plan.advance
  cases h : p i <;> simp only [h,translate,translate_messages,instanceAt]

theorem translate_exec {α : Type} (ν : ℝ) (x y : E d) {r : ℕ}
    {p : Fin n → Plan m d α} {v : Fin n → α}
    (h : Plan.Executes (instanceAt I ν y x) W r p v) :
    Plan.Executes I W r (fun i => translate ν y (p i)) v := by
  induction h with
  | done => exact .done _
  | next hh _ ih =>
    refine .next (fun i => (translate_active ν y _).2 (hh i)) ?_
    rw [translate_step ν x y]
    exact ih

abbrev InnerSeed (n m B R H : ℕ) := Draws (Draws (Batch.Choice n m B) R) H

def innerIdeal (I : Instance n m d) (L a : ℝ) (R H : ℕ) (x : E d)
    (s : InnerSeed n m B R H) : E d :=
  repeatEval (fun b t => (Epoch.evaluate I L a (Batch.estimate I b) R t b b).mean) H s x

theorem inner_feasible (hI : I.InClass L μ Δ) (hL : 0 < L) (a : ℝ)
    (R H : ℕ) {x : E d} (hx : feasible I x) (s : InnerSeed n m B R H) :
    feasible I (innerIdeal I L a R H x s) :=
  repeat_feasible _ (fun x hx t => (Epoch.evaluate_feasible hI hL a (Batch.estimate I x) R t hx x).2)
    H s hx

theorem inner_contract (hI : I.InClass L μ Δ) [Nonempty (Fin n)] [Nonempty (Fin m)] [Nonempty (Fin B)]
    (hL : 0 < L) (hμ : 0 < μ) {a : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1) (ha₂ : 2*L*a^2=μ/4)
    (R H : ℕ) (hR : 6 ≤ a*(R:ℝ)) (hnoise : 2*(L^2/((n:ℝ)*(B:ℝ)))/(L*μ)*(R:ℝ) ≤ 1/4)
    {u x : E d} (hu : ∀ w, I.φ u ≤ I.φ w) (hx : feasible I x) :
    average (fun s : InnerSeed n m B R H => loss I u (innerIdeal I L a R H x s)) ≤
      (1/2:ℝ)^H*loss I u x := by
  change average (fun s => loss I u (repeatEval
    (fun b t => (Epoch.evaluate I L a (Batch.estimate I b) R t b b).mean) H s x)) ≤ _
  apply repeat_contract _ _ _ H hx
  · intro x hx t
    exact (Epoch.evaluate_feasible hI hL a (Batch.estimate I x) R t hx x).2
  · intro x hx
    exact Epoch.contract hI hL hμ hu hx ha ha₁ ha₂ (show 0 ≤ L^2/((n:ℝ)*(B:ℝ)) by positivity)
      (Batch.estimate I x) (Batch.mean hI x) (Batch.variance hI x) R hR hnoise

def oracle (D : ℕ) (r i : Fin n) (ν L a : ℝ) (R H : ℕ)
    (x y : E d) (s : InnerSeed n m B R H) : Plan m d (E d) :=
  translate ν y (repeating (Batch.epoch D r i L a R) H s x)

theorem oracle_valid (D : ℕ) (r i : Fin n) (hν : 0 ≤ ν) (hL : 0 < L) (a : ℝ)
    (R H : ℕ) {S : Set (Fin d)} {x y : E d} (hx : Supported S x) (hy : Supported S y)
    (s : InnerSeed n m B R H) :
    (oracle D r i ν L a R H x y s).Valid Supported S :=
  translate_valid hν hy
    (repeating_valid _ (fun _ _ hx s => Batch.epoch_valid D r i L a hL R hx s) H s hx)

theorem oracle_exec [Nonempty (Fin B)] (hW : CommClass n ρ W)
    (D : ℕ) (hD : ∀ i j, 0 < (Network.lazy W ^ D) i j) (r : Fin n)
    (hI : I.InClass L μ Δ) (ν a : ℝ) (R H : ℕ)
    (x y : E d) (s : InnerSeed n m B R H) :
    Plan.Executes I W (H*(m+4*D+R*(B+1+4*D))) (fun i => oracle D r i ν L a R H x y s)
      (fun _ => innerIdeal (instanceAt I ν y x) L a R H x s) := by
  let Q := instanceAt I ν y x
  apply translate_exec ν x y
  apply repeating_exec _ (fun x s => (Epoch.evaluate Q L a (Batch.estimate Q x) R s x x).mean)
  intro b t
  apply Batch.epoch_exec_from_full (I := Q) ?_ hW D hD r L a R b t
  intro v
  change gradient I.f v = average (fun i => localGrad I i v)
  exact fullGrad hI v

variable {σ : Type}

def outerPlan (f : E d → E d → σ → Plan m d (E d)) (a : ℝ) :
    (r : ℕ) → Draws σ r → E d → E d → Plan m d (E d)
  | 0, _, x, _ => .done x
  | r+1, s, x, z =>
    let y := blend a x z
    (f x y (drawHead s)).bind fun p =>
    (outerPlan f a r (drawTail s) p (momentum a y p z)).bind
      (fun x' => .done (blend (fraction r) x x'))

theorem outer_valid (f : E d → E d → σ → Plan m d (E d)) (a : ℝ)
    (hf : ∀ S x y, Supported S x → Supported S y → ∀ s, (f x y s).Valid Supported S)
    (r : ℕ) (s : Draws σ r) {S : Set (Fin d)} {x z : E d}
    (hx : Supported S x) (hz : Supported S z) : (outerPlan f a r s x z).Valid Supported S := by
  induction r generalizing S x z with
  | zero => exact hx
  | succ r ih =>
    have hy := supported_blend hx hz a
    unfold outerPlan
    apply Plan.valid_bind (hf _ _ _ hx hy _)
    intro T hST p hp
    apply Plan.valid_bind (ih _ hp (supported_momentum hp (hy.mono hST) (hz.mono hST) a))
    intro U hTU w hw
    exact supported_blend (hx.mono (hST.trans hTU)) hw _

theorem outer_exec (f : Fin n → E d → E d → σ → Plan m d (E d))
    (g : E d → E d → σ → E d) (a : ℝ) (C : ℕ)
    (hg : ∀ x y s, Plan.Executes I W C (fun i => f i x y s) (fun _ => g x y s))
    (r : ℕ) (s : Draws σ r) (x z : E d) :
    Plan.Executes I W (r*C) (fun i => outerPlan (f i) a r s x z)
      (fun _ => (Inexact.evaluate a g r s x z).mean) := by
  induction r generalizing x z with
  | zero => simpa only [Nat.zero_mul,outerPlan,Inexact.evaluate] using
      Plan.Executes.done (I := I) (W := W) (fun _ => x)
  | succ r ih =>
    let y := blend a x z
    let p := g x y (drawHead s)
    let zz := momentum a y p z
    have h := (hg x y (drawHead s)).bind
      (f := fun i p => (outerPlan (f i) a r (drawTail s) p (momentum a y p z)).bind
        (fun v => .done (blend (fraction r) x v)))
      ((ih (drawTail s) p zz).bind (f := fun _ v => .done (blend (fraction r) x v)) (.done _))
    rw [show (r+1)*C = C+(r*C+0) by rw [Nat.succ_mul]; omega]
    exact h

private theorem outer_params (hμ : 0 < μ) (hν : μ ≤ ν) :
    0 < Real.sqrt (μ/(2*ν)) ∧ Real.sqrt (μ/(2*ν)) ≤ 1 ∧
      (ν/2)*(Real.sqrt (μ/(2*ν)))^2 = μ/4 := by
  have hν₀ := hμ.trans_le hν
  have h : 0 < μ/(2*ν) := by positivity
  refine ⟨Real.sqrt_pos.2 h, ?_, ?_⟩
  · have h' : μ/(2*ν) ≤ 1 := (div_le_one (by positivity)).2 (by linarith)
    simpa using Real.sqrt_le_sqrt h'
  · rw [Real.sq_sqrt h.le]
    field_simp
    ring

theorem bound (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hν : μ ≤ ν) (hL : μ+ν ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D R H B S K : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt ((μ+ν)/(8*L)) * (R : ℝ))
    (hB : 1 ≤ B) (hbatch : 8*(L/(μ+ν))*(R : ℝ) ≤ (n : ℝ)*(B : ℝ))
    (hS : 6 ≤ Real.sqrt (μ/(2*ν))*(S:ℝ))
    (hH : 3*(1+ν/μ)*(1/2:ℝ)^H*(S:ℝ) ≤ 1/4)
    (hK : (1/2 : ℝ)^K * Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤ ((K*(S*(H*(m+4*D+R*(B+1+4*D)))) : ℕ) : ℕ∞) := by
  let root : Fin n := ⟨0, by omega⟩
  let ai := Real.sqrt ((μ+ν)/(8*L))
  let a := Real.sqrt (μ/(2*ν))
  have hν₀ := hμ.trans_le hν
  have hq : 0 < μ+ν := by positivity
  have hLp := hq.trans_le hL
  obtain ⟨hai,hai₁,hai₂⟩ := rate_parameters hq hL
  obtain ⟨ha,ha₁,ha₂⟩ := outer_params hμ hν
  letI : Nonempty (Fin n) := ⟨root⟩
  letI : Nonempty (Fin m) := ⟨⟨0, by omega⟩⟩
  letI : Nonempty (Fin B) := ⟨⟨0, by omega⟩⟩
  let τ := InnerSeed n m B R H
  let g (d : ℕ) (I : Instance n m d) (x y : E d) (s : τ) :=
    innerIdeal (instanceAt I ν y x) L ai R H x s
  let once (d : ℕ) (I : Instance n m d) (x : E d) (s : Draws τ S) :=
    (Inexact.evaluate a (g d I) S s x x).mean
  let plan (d : ℕ) (i : Fin n) (x : E d) (s : Draws τ S) : Plan m d (E d) :=
    outerPlan (oracle D root i ν L ai R H) a S s x x
  have hnr : (0:ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hbr : (0:ℝ) < B := by exact_mod_cast (show 0 < B by omega)
  have hnoise : 2*(L^2/((n:ℝ)*(B:ℝ)))/(L*(μ+ν))*(R:ℝ) ≤ 1/4 := by
    have he : 2*(L^2/((n:ℝ)*(B:ℝ)))/(L*(μ+ν))*(R:ℝ) =
        (8*(L/(μ+ν))*(R:ℝ))/((4:ℝ)*n*B) := by field_simp; ring
    rw [he]
    exact (div_le_iff₀ (by positivity)).2 (by convert hbatch using 1; ring)
  have hfeas (d : ℕ) (I : Instance n m d) (hI : I.InClass L μ Δ)
      (x : E d) (hx : feasible I x) (y : E d) (s : τ) : feasible I (g d I x y s) := by
    obtain ⟨v,hvf,hv⟩ := min_exists hI hν₀ y hx
    have hQ := inClass_at hI hν₀.le y hx hvf hv
    exact (feasible_iff ν y x _).1
      (inner_feasible hQ hLp ai R H ((feasible_iff _ _ _ _).2 hx) s)
  have hbound (d : ℕ) (I : Instance n m d) (hI : I.InClass L μ Δ)
      (x : E d) (hx : feasible I x) (y v : E d) (hvf : feasible I v)
      (hv : ∀ w, (instanceAt I ν y x).φ v ≤ (instanceAt I ν y x).φ w) :
      average (fun s => loss (instanceAt I ν y x) v (g d I x y s)) ≤
        (1/2:ℝ)^H * loss (instanceAt I ν y x) v x := by
    have hQ := inClass_at hI hν₀.le y hx hvf hv
    exact inner_contract (B := B) hQ hLp hq hai hai₁ hai₂ R H hR hnoise hv ((feasible_iff _ _ _ _).2 hx)
  apply fromSchedule (by omega : 0 < n) _ (Draws (Draws τ S) K)
    (fun d info => repeating (plan d info.node) K info.seed info.x0)
    (fun d I s => repeatEval (once d I) K s I.x0)
  · intro d info
    apply repeating_valid _ _ _ _ (subset_refl _)
    intro T x hx s
    exact outer_valid _ a (fun _ _ _ hv hw t => oracle_valid D root info.node hν₀.le hLp ai R H hv hw t)
      S s hx hx
  · intro d I W hI hW s
    have hpos := Network.certified_radius (by omega) hρ hρ₁ hW hD
    -- The executable subproblem does not depend on a feasibility test. The execution proof
    -- uses minimizers only in the separate convergence argument.
    apply repeating_exec _ (once d I) _
    intro x t
    apply outer_exec _ (g d I)
    intro v y s
    exact oracle_exec hW D hpos root hI ν ai R H v y s
  · intro d I hI s
    obtain ⟨u,hu⟩ := hI.minimizer
    exact repeat_feasible (once d I)
      (fun x hx t => (Inexact.evaluate_feasible hI a (g d I) (hfeas d I hI) S t hx x).2)
      K s (initial_feasible hI hu)
  · intro d I hI u hu
    have hf (x : E d) (hx : feasible I x) (s : Draws τ S) : feasible I (once d I x s) :=
      (Inexact.evaluate_feasible hI a (g d I) (hfeas d I hI) S s hx x).2
    have hg (x : E d) (hx : feasible I x) :
        average (fun s => loss I u (once d I x s)) ≤ loss I u x / 2 :=
      Inexact.epoch_contract hI hμ hν₀ ha ha₁ ha₂ (by positivity) (g d I)
        (hfeas d I hI) (hbound d I hI) hu hx S hS hH
    exact (repeat_contract (once d I) hf hg K (initial_feasible hI hu)).trans
      ((mul_le_mul_of_nonneg_left (initial_loss hI hu) (by positivity)).trans hK)

end Nested
end ULB63
