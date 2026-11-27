import Theory.Regularized
import Theory.Epoch

open scoped RealInnerProductSpace

set_option maxHeartbeats 1200000

noncomputable section

namespace ULB63
namespace Inexact

open Composite Epoch FiniteMean Regularized

variable {n m d : ℕ} {I : Instance n m d} {L μ Δ ν : ℝ}

private theorem approx_three_point (hI : I.InClass L μ Δ) (hμ : 0 < μ) (hν : 0 < ν)
    (y : E d) {x u p w : E d} (hx : feasible I x) (hu : feasible I u)
    (hp : feasible I p) (hw : feasible I w)
    (hm : ∀ v, (instanceAt I ν y x).φ u ≤ (instanceAt I ν y x).φ v) :
    value I p-value I w ≤
      ν/2 * (‖w-y‖^2-‖w-p‖^2-‖p-y‖^2) +
        (1+ν/μ)*loss (instanceAt I ν y x) u p := by
  let Q := instanceAt I ν y x
  have hQ := inClass_at hI hν.le y hx hu hm
  have hpQ : feasible Q p := (feasible_iff _ _ _ _).2 hp
  have hwQ : feasible Q w := (feasible_iff _ _ _ _).2 hw
  let t := μ/(μ+ν)
  have ht : 0 < t := by dsimp [t]; positivity
  have ht₁ : t ≤ 1 := (div_le_one (by positivity)).2 (by linarith)
  obtain ⟨hv,hc⟩ := phi_blend hQ hpQ hwQ ht.le ht₁
  have hnb := loss_nonneg hQ hm hv
  let δ := loss Q u p
  have hfactor : (μ+ν)/2*t*(1-t) = t*(ν/2) := by
    dsimp [t]
    field_simp
    ring
  rw [hfactor] at hc
  have H : t*(value Q p-value Q w+(ν/2)*‖w-p‖^2) ≤ δ := by
    dsimp only [δ,loss]
    unfold loss at hnb
    linarith
  have hh : value Q p-value Q w+ν/2*‖w-p‖^2 ≤ δ/t := by
    apply (le_div_iff₀ ht).2
    rw [mul_comm _ t]
    exact H
  have hf : δ/t = (1+ν/μ)*δ := by
    dsimp [t]
    field_simp
  change value (instanceAt I ν y x) p-value (instanceAt I ν y x) w+ν/2*‖w-p‖^2 ≤ δ/t at hh
  rw [value_eq hI ν y x hp,value_eq hI ν y x hw,hf] at hh
  unfold quad at hh
  linarith

theorem potential_prox (hI : I.InClass L μ Δ) (hμ : 0 < μ) (hν : 0 < ν)
    {a : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1) (ha₂ : (ν/2)*a^2=μ/4)
    {u x : E d} (hu : feasible I u) (hx : feasible I x)
    (z : E d) {v p : E d} (hv : feasible I v) (hp : feasible I p)
    (hm : ∀ w, (instanceAt I ν (blend a x z) x).φ v ≤ (instanceAt I ν (blend a x z) x).φ w) :
    let y := blend a x z
    potential I μ u p (momentum a y p z)+a*loss I u x ≤
      potential I μ u x z+(1+ν/μ)*loss (instanceAt I ν y x) v p := by
  intro y
  let z' := momentum a y p z
  let w := blend a x u
  have hw := (phi_blend hI hx hu ha.le ha₁).1
  have H := approx_three_point hI hμ hν y hx hv hp hw hm
  have hwy : w-y = a • (u-z) := by dsimp [w,y,blend]; module
  have hwp : w-p = a • (u-z') := by
    calc
      _ = (w-y)-(p-y) := by abel
      _ = _ := by
        rw [hwy]
        dsimp [z',momentum]
        rw [show u-(z+a⁻¹ • (p-y)) = (u-z)-a⁻¹ • (p-y) by abel, smul_sub a (u-z),
          smul_smul,mul_inv_cancel₀ ha.ne',one_smul]
  change value I p-value I w ≤ _ at H
  rw [hwy,hwp,norm_smul,norm_smul,Real.norm_eq_abs,abs_of_nonneg ha.le,norm_sub_rev u z,
    norm_sub_rev u z'] at H
  have ht : ν/2 * ((a*‖z-u‖)^2-(a*‖z'-u‖)^2-‖p-y‖^2) =
      μ/4*(‖z-u‖^2-‖z'-u‖^2)-ν/2*‖p-y‖^2 := by
    rw [← ha₂]
    ring
  rw [ht] at H
  have hc := loss_blend hI hμ.le hx hu (u := u) ha.le ha₁
  change loss I u w ≤ _ at hc
  change potential I μ u p z'+a*loss I u x ≤ _
  unfold potential loss at *
  linarith [show 0 ≤ ν/2*‖p-y‖^2 by positivity]

theorem initial_reg_loss (hI : I.InClass L μ Δ) (hμ : 0 < μ) (hν : 0 ≤ ν)
    {a : ℝ} (ha : 0 ≤ a) (ha₂ : (ν/2)*a^2=μ/4)
    {u x : E d} (hu : ∀ w, I.φ u ≤ I.φ w) (hx : feasible I x)
    (z : E d) {v : E d} (hv : feasible I v) :
    loss (instanceAt I ν (blend a x z) x) v x ≤ 3*potential I μ u x z := by
  let y := blend a x z
  change value (instanceAt I ν y x) x - value (instanceAt I ν y x) v ≤ _
  rw [value_eq hI ν y x hx,value_eq hI ν y x hv]
  have hval := loss_nonneg hI hu hv
  have hr := radius_bound hI hu hx
  have hwy : x-y = a • (x-z) := by dsimp [y,blend]; module
  have hs : quad ν y x = μ/4*‖x-z‖^2 := by
    unfold quad
    rw [hwy,norm_smul,Real.norm_eq_abs,abs_of_nonneg ha,mul_pow,← mul_assoc,ha₂]
  rw [hs]
  have ht : ‖x-z‖ ≤ ‖x-u‖+‖z-u‖ := by
    have hh := norm_sub_le (x-u) (z-u)
    rwa [show x-u-(z-u)=x-z by abel] at hh
  have hsq : ‖x-z‖^2 ≤ 2*(‖x-u‖^2+‖z-u‖^2) := by
    nlinarith [pow_le_pow_left₀ (norm_nonneg _) ht 2,sq_nonneg (‖x-u‖-‖z-u‖)]
  have hprod := mul_le_mul_of_nonneg_left hsq (by positivity : 0 ≤ μ/4)
  have hq : 0 ≤ quad ν y v := by unfold quad; positivity
  have hmom : 0 ≤ μ/4*‖z-u‖^2 := by positivity
  unfold potential loss at *
  nlinarith

variable {σ : Type}

def evaluate (a : ℝ) (f : E d → E d → σ → E d) :
    (r : ℕ) → Draws σ r → E d → E d → Trace d
  | 0, _, x, z => ⟨x,z,x⟩
  | r+1, s, x, z =>
    let y := blend a x z
    let p := f x y (drawHead s)
    let t := evaluate a f r (drawTail s) p (momentum a y p z)
    ⟨t.x,t.z,blend (fraction r) x t.mean⟩

theorem evaluate_feasible (hI : I.InClass L μ Δ) (a : ℝ) (f : E d → E d → σ → E d)
    (hf : ∀ x, feasible I x → ∀ y s, feasible I (f x y s))
    (r : ℕ) (s : Draws σ r) {x : E d} (hx : feasible I x) (z : E d) :
    feasible I (evaluate a f r s x z).x ∧ feasible I (evaluate a f r s x z).mean := by
  induction r generalizing x z with
  | zero => exact ⟨hx,hx⟩
  | succ r ih =>
    let y := blend a x z
    let p := f x y (drawHead s)
    have hh := ih (drawTail s) (hf x hx y (drawHead s)) (momentum a y p z)
    exact ⟨hh.1,(phi_blend hI hx hh.2 (fraction_nonneg _) (fraction_le _)).1⟩

variable [Fintype σ] [Nonempty σ]

private theorem mean_one (hI : I.InClass L μ Δ) (hμ : 0 < μ) (hν : 0 < ν)
    {a δ : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1) (ha₂ : ν/2*a^2=μ/4) (hδ : 0 ≤ δ)
    (f : E d → E d → σ → E d)
    (hf : ∀ x, feasible I x → ∀ y s, feasible I (f x y s))
    (hg : ∀ x, feasible I x → ∀ y v, feasible I v →
      (∀ w, (instanceAt I ν y x).φ v ≤ (instanceAt I ν y x).φ w) →
      average (fun s => loss (instanceAt I ν y x) v (f x y s)) ≤
        δ*loss (instanceAt I ν y x) v x)
    {u x : E d} (hu : ∀ w, I.φ u ≤ I.φ w) (hx : feasible I x) (z : E d) :
    let y := blend a x z
    average (fun s => potential I μ u (f x y s) (momentum a y (f x y s) z))+a*loss I u x ≤
      (1+3*(1+ν/μ)*δ)*potential I μ u x z := by
  intro y
  obtain ⟨v,hvf,hv⟩ := min_exists hI hν y hx
  let Q := instanceAt I ν y x
  let C := 1+ν/μ
  let P := potential I μ u x z
  have HH := average_mono (fun s => potential_prox hI hμ hν ha ha₁ ha₂
    (optimal_feasible hI hu) hx z hvf (hf x hx y s) hv)
  simp only [average_add,average_const,average_mul] at HH
  refine HH.trans ?_
  have h := (hg x hx y v hvf hv).trans (mul_le_mul_of_nonneg_left
    (initial_reg_loss hI hμ hν.le ha.le ha₂ hu hx z hvf) hδ)
  have hm := mul_le_mul_of_nonneg_left h (show 0 ≤ C by dsimp [C]; positivity)
  change P+C*average (fun s => loss Q v (f x y s)) ≤ _
  change C*average (fun s => loss Q v (f x y s)) ≤ C*(δ*(3*P)) at hm
  change P + C*average (fun s => loss Q v (f x y s)) ≤ (1+3*C*δ)*P
  linarith

private theorem mean_many (hI : I.InClass L μ Δ) (hμ : 0 < μ)
    {a χ : ℝ} (ha : 0 < a) (hχ : 0 ≤ χ)
    (f : E d → E d → σ → E d)
    (hf : ∀ x, feasible I x → ∀ y s, feasible I (f x y s))
    {u : E d} (hu : ∀ w, I.φ u ≤ I.φ w)
    (hs : ∀ x, feasible I x → ∀ z,
      let y := blend a x z
      average (fun s => potential I μ u (f x y s) (momentum a y (f x y s) z))+a*loss I u x ≤
        (1+χ)*potential I μ u x z)
    (r : ℕ) {x : E d} (hx : feasible I x) (z : E d) :
    average (fun s : Draws σ r => let t := evaluate a f r s x z
      potential I μ u t.x t.z+a*(r:ℝ)*loss I u t.mean) ≤ (1+χ)^r * potential I μ u x z := by
  induction r generalizing x z with
  | zero => simp only [evaluate,Nat.cast_zero,mul_zero,zero_mul,add_zero,average_const,pow_zero,one_mul,le_refl]
  | succ r ih =>
    let y := blend a x z
    let pp : σ → E d := fun s => f x y s
    let zz : σ → E d := fun s => momentum a y (pp s) z
    have hfac : 1 ≤ (1+χ)^r := one_le_pow₀ (by linarith)
    have hF (s : σ) : feasible I (pp s) := hf x hx y s
    rw [average_draws_succ]
    calc
      _ ≤ average (fun s : σ => (1+χ)^r * potential I μ u (pp s) (zz s)+a*loss I u x) := by
        apply average_mono
        intro s
        have help (t : Draws σ r) :=
          averaged_loss hI hμ.le (u := u) hx (evaluate_feasible hI a f hf r t (hF s) (zz s)).2 r
        have hh := average_mono (fun t : Draws σ r => by
          dsimp only [evaluate,drawHead,drawTail]
          rw [Nat.cast_succ]
          have H := mul_le_mul_of_nonneg_left (help t) ha.le
          linarith : ∀ t : Draws σ r,
            let e := evaluate a f (r+1) (show Draws σ (r+1) from (s,t)) x z
            potential I μ u e.x e.z + a * ((r+1 : ℕ) : ℝ) * loss I u e.mean ≤
              let e := evaluate a f r t (pp s) (zz s)
              (potential I μ u e.x e.z+a*(r:ℝ)*loss I u e.mean)+a*loss I u x)
        refine hh.trans ?_
        rw [average_add,average_const]
        exact add_le_add_left (ih (hF s) (zz s)) _
      _ ≤ (1+χ)^r *(average (fun s => potential I μ u (pp s) (zz s))+a*loss I u x) := by
        rw [average_add,average_mul,average_const,mul_add]
        exact add_le_add_right
          (le_mul_of_one_le_left (mul_nonneg ha.le (loss_nonneg hI hu hx)) hfac) _
      _ ≤ (1+χ)^r*((1+χ)*potential I μ u x z) :=
        mul_le_mul_of_nonneg_left (hs x hx z) (by positivity)
      _ = _ := by rw [pow_succ,mul_assoc]

theorem epoch_contract (hI : I.InClass L μ Δ) (hμ : 0 < μ) (hν : 0 < ν)
    {a δ : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1) (ha₂ : ν/2*a^2=μ/4) (hδ : 0 ≤ δ)
    (f : E d → E d → σ → E d)
    (hf : ∀ x, feasible I x → ∀ y s, feasible I (f x y s))
    (hg : ∀ x, feasible I x → ∀ y v, feasible I v →
      (∀ w, (instanceAt I ν y x).φ v ≤ (instanceAt I ν y x).φ w) →
      average (fun s => loss (instanceAt I ν y x) v (f x y s)) ≤
        δ*loss (instanceAt I ν y x) v x)
    {u b : E d} (hu : ∀ w, I.φ u ≤ I.φ w) (hb : feasible I b)
    (r : ℕ) (hr : 6 ≤ a*(r:ℝ)) (hnoise : (3*(1+ν/μ)*δ)*(r:ℝ) ≤ 1/4) :
    average (fun s : Draws σ r => loss I u (evaluate a f r s b b).mean) ≤ loss I u b/2 := by
  let χ := 3*(1+ν/μ)*δ
  let G := loss I u b
  have hχ : 0 ≤ χ := by dsimp [χ]; positivity
  have hG : 0 ≤ G := loss_nonneg hI hu hb
  have hstart : potential I μ u b b ≤ 2*G := by
    unfold potential
    have h := radius_bound hI hu hb
    linarith
  have H := mean_many hI hμ ha hχ f hf hu
    (fun x hx z => mean_one hI hμ hν ha ha₁ ha₂ hδ f hf hg hu hx z) r hb b
  have H' : (1+χ)^r * potential I μ u b b ≤ 3*G := by
    calc
      _ ≤ (1+χ)^r * (2*G) := mul_le_mul_of_nonneg_left hstart (by positivity)
      _ ≤ (4/3)*(2*G) := mul_le_mul_of_nonneg_right (growth_bound hχ r hnoise) (by positivity)
      _ ≤ _ := by linarith
  have hlow : 0 ≤ average (fun s : Draws σ r =>
      potential I μ u (evaluate a f r s b b).x (evaluate a f r s b b).z) := by
    calc
      (0:ℝ) = average (fun _ : Draws σ r => 0) := (average_const _).symm
      _ ≤ _ := average_mono (fun s => by
        unfold potential
        exact add_nonneg (loss_nonneg hI hu (evaluate_feasible hI a f hf r s hb b).1) (by positivity))
  have HH := H.trans H'
  simp only [average_add,average_mul] at HH
  have hcost : a*(r:ℝ)*average (fun s : Draws σ r => loss I u (evaluate a f r s b b).mean) ≤ 3*G := by
    linarith
  by_contra! hn
  have hh := mul_lt_mul_of_pos_left hn (show 0 < a*(r:ℝ) by linarith)
  have ht := mul_le_mul_of_nonneg_right hr (show 0 ≤ G/2 by positivity)
  linarith

end Inexact
end ULB63
