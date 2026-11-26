import Theory.ValueTree
import Theory.Recovery
import Theory.Balls

open MeasureTheory Set
open scoped BigOperators Pointwise

namespace Cubature

theorem kink_integrable {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) :
    Integrable (ellipsoidalKink ρ E) (gaussian d) := by
  have hc : Continuous (ellipsoidalKink ρ E) := by
    unfold ellipsoidalKink quadratic
    fun_prop
  apply Integrable.mono' (integrable_const (1:ℝ)) hc.aestronglyMeasurable
  exact ae_of_all _ fun x => by
    have hq := (normSq_nonneg (ρ⁻¹ • x)).trans (E.low _)
    rw [Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
    exact max_le (by linarith) zero_le_one

namespace ValueTree

def trace {X Y : Type*} : ValueTree X Y → (X → ℝ) → List X
  | .ret _, _ => []
  | .ask x k, f => x::(k (f x)).trace f

theorem trace_length {X Y : Type*} (t : ValueTree X Y) (f : X → ℝ) {n : ℕ}
    (hn : t.within n) : (t.trace f).length ≤ n := by
  induction t generalizing n with
  | ret _ => simp [trace]
  | ask x k ih =>
    cases n with
    | zero => exact False.elim hn
    | succ n => exact Nat.succ_le_succ (ih (f x) (hn (f x)))

theorem run_eq_of_agree {X Y : Type*} (t : ValueTree X Y) (f g : X → ℝ)
    (h : ∀ x ∈ t.trace f, g x = f x) :
    t.run g = t.run f := by
  induction t with
  | ret _ => rfl
  | ask x k ih =>
    have hx : g x = f x := h x (by simp [trace])
    simp only [run,hx]
    exact ih (f x) (fun y hy => h y (by simp [trace,hy]))

end ValueTree

def symmetricOf {d : ℕ} (c : RayIndex d → ℝ) : Coefficients d :=
  fun i j ↦ c (rayIndex i j)

theorem symmetricOf_symm {d : ℕ} (c : RayIndex d → ℝ) (i j : Fin d) :
    symmetricOf c i j = symmetricOf c j i := by
  simp [symmetricOf,rayIndex,min_comm,max_comm]

theorem rayIndex_eta {d : ℕ} (i : RayIndex d) : rayIndex i.1.1 i.1.2 = i := by
  obtain ⟨⟨i,j⟩,h⟩ := i
  exact Subtype.ext (by simp [rayIndex,min_eq_left h,max_eq_right h])

def quadraticReadings {d n : ℕ} (z : Fin n → Point d) :
    (RayIndex d → ℝ) →ₗ[ℝ] (Fin n → ℝ) where
  toFun c i := quadratic (symmetricOf c) (z i)
  map_add' c c' := by
    funext i
    simp [quadratic,symmetricOf,Pi.add_apply,add_mul,Finset.sum_add_distrib]
  map_smul' t c := by
    funext i
    simp only [quadratic,symmetricOf,Pi.smul_apply,smul_eq_mul,RingHom.id_apply,Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro a _
    apply Finset.sum_congr rfl
    intro b _
    ring

theorem invisible_symmetric {d n : ℕ} (hn : 2*n < d*(d+1)) (z : Fin n → Point d) :
    ∃ H : Coefficients d, (∀ i j, H i j=H j i) ∧ (∀ i j, |H i j| ≤ 1) ∧
      (∃ i j, |H i j|=1) ∧ (∀ i, quadratic H (z i)=0) := by
  let R := quadraticReadings z
  have hni : ¬Function.Injective R := by
    intro hv
    have h := LinearMap.finrank_le_finrank_of_injective hv
    simp only [Module.finrank_fintype_fun_eq_card,Fintype.card_fin] at h
    have hc := card_rayIndex d
    omega
  obtain ⟨c,e,hce,hne⟩ := Function.not_injective_iff.mp hni
  let v := c-e
  have hv : v ≠ 0 := sub_ne_zero.mpr hne
  have hv0 : R v = 0 := by change R (c-e)=0; rw [map_sub,hce,sub_self]
  have hnonzero : ∃ i, v i ≠ 0 := by
    by_contra! heq
    exact hv (funext heq)
  obtain ⟨j,hj⟩ := hnonzero
  obtain ⟨i,_,hi⟩ := Finset.exists_max_image (Finset.univ : Finset (RayIndex d))
    (fun i ↦ |v i|) ⟨j,Finset.mem_univ _⟩
  let M := |v i|
  have hM : 0 < M := (abs_pos.mpr hj).trans_le (hi j (Finset.mem_univ j))
  let w := M⁻¹ • v
  have hquot (i : RayIndex d) : |w i|=|v i|/M := by
    simp only [w,Pi.smul_apply,smul_eq_mul,abs_mul,abs_of_pos (inv_pos.mpr hM),div_eq_mul_inv]
    ring
  have hw (i : RayIndex d) : |w i|≤1 := by
    rw [hquot,div_le_one₀ hM]
    exact hi i (Finset.mem_univ i)
  have hw0 : |w i|=1 := by rw [hquot]; exact div_self hM.ne'
  refine ⟨symmetricOf w,symmetricOf_symm _,fun a b => hw _,⟨i.1.1,i.1.2,?_⟩,?_⟩
  · simpa [symmetricOf,rayIndex_eta] using hw0
  · have ha : R w=0 := by rw [show w=M⁻¹ • v by rfl,map_smul,hv0,smul_zero]
    intro j
    exact congrFun ha j

theorem abs_quadratic_bound {d : ℕ} (H : Coefficients d) (hH : ∀ i j, |H i j|≤1)
    (x : Point d) : |quadratic H x| ≤ d*normSq x := by
  unfold quadratic
  calc
    _ ≤ ∑ i, ∑ j, |H i j*x i*x j| := by
      apply (Finset.abs_sum_le_sum_abs _ _).trans
      exact Finset.sum_le_sum (fun i _ => Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ i, ∑ j, |x i| * |x j| := by
      apply Finset.sum_le_sum
      intro i _
      apply Finset.sum_le_sum
      intro j _
      simp only [abs_mul,mul_assoc]
      exact mul_le_of_le_one_left (mul_nonneg (abs_nonneg _) (abs_nonneg _)) (hH i j)
    _ ≤ _ := sum_abs_cross x

theorem quadratic_perturbed_value {d : ℕ} (H : Coefficients d) (t : ℝ) (x : Point d) :
    quadratic (fun i j => scalarMatrix (3/2) i j+t*H i j) x =
      (3/2)*normSq x+t*quadratic H x := by
  calc
    _ = quadratic (scalarMatrix (3/2)) x+t*quadratic H x := by
      unfold quadratic
      simp only [add_mul,Finset.sum_add_distrib,Finset.mul_sum,mul_assoc]
    _ = _ := by rw [quadratic_scalarMatrix]

noncomputable def perturbEllipsoid {d : ℕ} (H : Coefficients d)
    (hs : ∀ i j, H i j=H j i) (hb : ∀ i j, |H i j|≤1)
    (t : ℝ) (ht : |t| * d ≤ 1/2) : Ellipsoid d := by
  have hq (x : Point d) : |t*quadratic H x| ≤ normSq x/2 := by
    rw [abs_mul]
    have h := mul_le_mul_of_nonneg_left (abs_quadratic_bound H hb x) (abs_nonneg t)
    have h' := mul_le_mul_of_nonneg_right ht (normSq_nonneg x)
    linarith only [h,h']
  refine ⟨fun i j => scalarMatrix (3/2) i j+t*H i j,0,?_,?_,?_,?_⟩
  · intro i j
    have hh : scalarMatrix (3/2) i j = scalarMatrix (3/2) j i := by simp [scalarMatrix,eq_comm]
    rw [hh,hs i j]
  · intro x
    rw [quadratic_perturbed_value]
    linarith [(abs_le.mp (hq x)).1]
  · intro x
    rw [quadratic_perturbed_value]
    linarith [(abs_le.mp (hq x)).2]
  · intro x
    simp only [linear,Pi.zero_apply,zero_mul,Finset.sum_const_zero,abs_zero]
    linarith [normSq_nonneg x]

noncomputable def centralEllipsoid (d : ℕ) : Ellipsoid d :=
  perturbEllipsoid (fun _ _ => 0) (fun _ _ => rfl) (fun _ _ => by norm_num) 0 (by norm_num)

theorem central_quad {d : ℕ} (x : Point d) :
    quadratic (centralEllipsoid d).A x = (3/2)*normSq x := by
  change quadratic (fun i j => scalarMatrix (3/2) i j + 0*(0:ℝ)) x = _
  rw [quadratic_perturbed_value]
  ring

theorem exact_kink_nonadaptive {d : ℕ} {ρ : ℝ} (hρ : 0 < ρ) (hρ' : ρ ≤ 20) :
    ∃ (z : RayIndex d → Point d) (out : (RayIndex d → ℝ) → ℝ),
      (∀ j, allowed (z j)) ∧
      ∀ E : Ellipsoid d, out (fun j ↦ ellipsoidalKink ρ E (z j)) = kinkIntegral ρ E := by
  let z : RayIndex d → Point d := fun i => (ρ/4) • ray i
  let unpack : (RayIndex d → ℝ) → (Bool × RayIndex d → ℝ) :=
    fun y i => if i.1 then 16*(1-y i.2) else 1
  let out : (RayIndex d → ℝ) → ℝ := fun y =>
    ∫ t, max (1-quadratic (recoveredMatrix (unpack y)) (ρ⁻¹ • t)) 0 ∂(gaussian d)
  refine ⟨z,out,?_,?_⟩
  · intro j i
    simp only [z,Pi.smul_apply,smul_eq_mul,abs_mul,abs_of_pos (by positivity : 0 < ρ/4)]
    have H := mul_le_mul_of_nonneg_left (ray_bound j i) (by positivity : 0 ≤ ρ/4)
    linarith
  · intro E
    let y := fun i => ellipsoidalKink ρ E (z i)
    have hr (i : RayIndex d) : 16*(1-y i)=quadratic E.A (ray i) := by
      have hz : ρ⁻¹*(ρ/4)=1/4 := by field_simp
      have hb : quadratic E.A (ray i)≤4 := by linarith [(ray_norm i).2,E.high (ray i)]
      simp only [y,ellipsoidalKink,z,smul_smul,hz,quadratic_smul,max_eq_left (by nlinarith : 0 ≤ 1-(1/4)^2*quadratic E.A (ray i))]
      ring
    have hval (i : RayIndex d) : recoveredCurvature (unpack y) i = quadratic E.A (ray i) := by
      simp [recoveredCurvature,unpack,hr]
    have heq : recoveredMatrix (unpack y)=E.A := by
      funext i j
      by_cases h : i=j
      · subst j
        rw [recoveredMatrix,if_pos rfl,hval,diagonal_ray,quadratic_axis]
      · rw [recoveredMatrix,if_neg h,hval,hval,hval,mixed_ray h,diagonal_ray,diagonal_ray,
          quadratic_axis,quadratic_axis,quadratic_two_axes]
        ring
    change (∫ t, max (1-quadratic (recoveredMatrix (unpack y)) (ρ⁻¹ • t)) 0 ∂(gaussian d)) =
      kinkIntegral ρ E
    rw [heq]
    rfl

end Cubature
