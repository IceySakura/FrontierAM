import Theory.KinkGeometry

open MeasureTheory Set
open scoped BigOperators Pointwise

namespace Cubature

theorem quadratic_pair_symm {d : ℕ} {H : Coefficients d}
    (h : ∀ i j, H i j=H j i) (i j : Fin d) :
    quadratic H (axis i+axis j) = H i i+H j j+2*H i j := by
  simp only [quadratic,Pi.add_apply,mul_add,add_mul,Finset.sum_add_distrib]
  simp only [axis,mul_ite,mul_one,mul_zero,Finset.sum_ite_eq',Finset.mem_univ,ite_true]
  rw [h j i]
  ring

theorem witness_ray {d : ℕ} {H : Coefficients d} (hs : ∀ i j, H i j=H j i)
    (h : ∃ i j, |H i j|=1) :
    ∃ r : RayIndex d, normSq (ray r)/2 ≤ |quadratic H (ray r)| := by
  obtain ⟨i,j,hij⟩ := h
  by_cases hi : (1/2 : ℝ) ≤ |H i i|
  · exact ⟨rayIndex i i, by simpa [diagonal_ray,quadratic_axis,normSq_axis] using hi⟩
  by_cases hj : (1/2 : ℝ) ≤ |H j j|
  · exact ⟨rayIndex j j, by simpa [diagonal_ray,quadratic_axis,normSq_axis] using hj⟩
  have hne : i≠j := by rintro rfl; linarith
  refine ⟨rayIndex i j, ?_⟩
  rw [mixed_ray hne,normSq_two_axes hne,quadratic_pair_symm hs]
  have hab := abs_add_le (H i i+H j j+2*H i j) (-(H i i+H j j))
  rw [show H i i+H j j+2*H i j+ -(H i i+H j j)=2*H i j by ring,
    abs_mul,abs_neg,abs_of_nonneg (by norm_num : 0≤(2:ℝ)),hij] at hab
  linarith [abs_add_le (H i i) (H j j)]

noncomputable def kinkCenter {d : ℕ} (r : RayIndex d) : Point d :=
  Real.sqrt ((2/3)/normSq (ray r)) • ray r

theorem kinkCenter_norm {d : ℕ} (r : RayIndex d) : normSq (kinkCenter r)=2/3 := by
  have hn : 0 < normSq (ray r) := by linarith [(ray_norm r).1]
  unfold kinkCenter
  rw [normSq_smul,Real.sq_sqrt (by positivity),div_mul_cancel₀ _ hn.ne']

theorem kinkCenter_witness {d : ℕ} {H : Coefficients d} (r : RayIndex d)
    (hr : normSq (ray r)/2 ≤ |quadratic H (ray r)|) :
    1/3 ≤ |quadratic H (kinkCenter r)| := by
  have hn : 0 < normSq (ray r) := by linarith [(ray_norm r).1]
  let a := (2/3:ℝ)/normSq (ray r)
  have ha : 0 ≤ a := by dsimp [a]; positivity
  rw [kinkCenter,quadratic_smul,Real.sq_sqrt ha,abs_mul,abs_of_nonneg ha]
  have hr' := mul_le_mul_of_nonneg_left hr ha
  have hx : a*(normSq (ray r)/2)=1/3 := by dsimp [a]; field_simp
  rwa [hx] at hr'

noncomputable def kinkRadius (d : ℕ) : ℝ := 1/(1000*(d:ℝ)^2)

noncomputable def kinkWindow {d : ℕ} (ρ : ℝ) (r : RayIndex d) : Set (Point d) :=
  (fun z ↦ ρ⁻¹ • z) ⁻¹' Metric.ball (kinkCenter r) (kinkRadius d)

theorem kinkRadius_pos {d : ℕ} (hd : 0 < d) : 0 < kinkRadius d := by
  have h : 0 < (d:ℝ) := by exact_mod_cast hd
  unfold kinkRadius
  positivity

theorem kinkWindow_pos {d : ℕ} (hd : 0 < d) {ρ : ℝ} (hρ : 0 < ρ) (j : RayIndex d) :
    0 < probability (kinkWindow ρ j) := by
  apply probability_pos_of_open (Metric.isOpen_ball.preimage (continuous_const_smul _))
  refine ⟨ρ • kinkCenter j, ?_⟩
  change ρ⁻¹ • (ρ • kinkCenter j) ∈ Metric.ball _ _
  rw [smul_smul,inv_mul_cancel₀ hρ.ne',one_smul]
  exact Metric.mem_ball_self (kinkRadius_pos hd)

theorem near_kinkCenter {d : ℕ} {ρ : ℝ} (j : RayIndex d) {z : Point d}
    (hz : z∈kinkWindow ρ j) :
    ∀ i, |(ρ⁻¹ • z) i-(kinkCenter j) i| ≤ kinkRadius d := by
  let u := ρ⁻¹ • z
  have hh : dist u (kinkCenter j) < kinkRadius d := hz
  have H := (dist_pi_le_iff (dist_nonneg : 0≤dist u (kinkCenter j))).1 le_rfl
  intro i
  change |u i-(kinkCenter j) i| ≤ kinkRadius d
  rw [← Real.dist_eq]
  exact (H i).trans hh.le

theorem local_form_bounds {d : ℕ} (hd : 0<d) {H : Coefficients d} (hH : ∀ i j, |H i j|≤1)
    (j : RayIndex d) (x : Point d)
    (hx : ∀ i, |x i-(kinkCenter j) i|≤kinkRadius d) :
    |quadratic H x-quadratic H (kinkCenter j)| ≤ 3/1000 ∧
      |normSq x-normSq (kinkCenter j)| ≤ 3/(1000*d) := by
  let y := kinkCenter j
  let r := kinkRadius d
  have hd0 : 0<(d:ℝ) := by exact_mod_cast hd
  have hd1 : 1≤(d:ℝ) := by exact_mod_cast hd
  have hr : 0≤r := (kinkRadius_pos hd).le
  have hr' : r≤1 := by
    unfold r kinkRadius
    apply (div_le_one₀ (by positivity)).2
    nlinarith [sq_nonneg ((d:ℝ)-1)]
  have hy (i : Fin d) : |y i| ≤ 1 := by
    nlinarith [sq_abs (y i), abs_nonneg (y i), sq_le_normSq y i, kinkCenter_norm j]
  have hx' (i : Fin d) : |x i| ≤ 2 := by
    have H := abs_add_le (x i-y i) (y i)
    rw [sub_add_cancel] at H
    linarith [hx i,hy i]
  have hprod (i k : Fin d) : |x i*x k-y i*y k| ≤ 3*r := by
    rw [show x i*x k-y i*y k=(x i-y i)*x k+y i*(x k-y k) by ring]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul,abs_mul]
    have ha := mul_le_mul (hx i) (hx' k) (abs_nonneg _) hr
    have hb := mul_le_mul (hy i) (hx k) (abs_nonneg _) (by norm_num : (0:ℝ)≤1)
    linarith only [ha,hb]
  have hR : (d:ℝ)^2*(3*r)=3/1000 := by
    unfold r kinkRadius
    field_simp
  have hR' : (d:ℝ)*(3*r)=3/(1000*d) := by
    unfold r kinkRadius
    field_simp
  constructor
  · have Hq : quadratic H x-quadratic H y =
        ∑ i, ∑ k, H i k*(x i*x k-y i*y k) := by
      simp only [quadratic,← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro k _
      ring
    rw [Hq]
    calc
      _ ≤ ∑ i, ∑ k, |H i k*(x i*x k-y i*y k)| := by
        apply (Finset.abs_sum_le_sum_abs _ _).trans
        exact Finset.sum_le_sum (fun i _ => Finset.abs_sum_le_sum_abs _ _)
      _ ≤ ∑ _ : Fin d, ∑ _ : Fin d, 3*r := by
        refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun k _ => ?_
        rw [abs_mul]
        simpa using mul_le_mul (hH i k) (hprod i k) (abs_nonneg _) zero_le_one
      _ = _ := by simpa [Finset.sum_const,nsmul_eq_mul,pow_two,mul_assoc] using hR
  · unfold normSq
    rw [← Finset.sum_sub_distrib]
    apply (Finset.abs_sum_le_sum_abs _ _).trans
    have H := Finset.sum_le_sum (s := Finset.univ) fun i _ => hprod i i
    simpa [pow_two,Finset.sum_const,nsmul_eq_mul,hR'] using H

theorem hinge_gap_nonneg (u v : ℝ) :
    0 ≤ (max (u+v) 0+max (u-v) 0)/2-max u 0 := by
  by_cases h : 0≤u
  · rw [max_eq_left h]
    linarith [le_max_left (u+v) (0:ℝ),le_max_left (u-v) (0:ℝ)]
  · rw [max_eq_right (le_of_not_ge h)]
    linarith [le_max_right (u+v) (0:ℝ),le_max_right (u-v) (0:ℝ)]

theorem hinge_gap_margin (u v : ℝ) :
    (|v|-|u|)/2 ≤ (max (u+v) 0+max (u-v) 0)/2-max u 0 := by
  rcases le_total 0 u with hu | hu <;> rcases le_total 0 v with hv | hv
  · rw [abs_of_nonneg hu,abs_of_nonneg hv,max_eq_left hu]
    linarith [le_max_left (u+v) (0:ℝ),le_max_right (u-v) (0:ℝ)]
  · rw [abs_of_nonneg hu,abs_of_nonpos hv,max_eq_left hu]
    linarith [le_max_right (u+v) (0:ℝ),le_max_left (u-v) (0:ℝ)]
  · rw [abs_of_nonpos hu,abs_of_nonneg hv,max_eq_right hu]
    linarith [le_max_left (u+v) (0:ℝ),le_max_right (u-v) (0:ℝ)]
  · rw [abs_of_nonpos hu,abs_of_nonpos hv,max_eq_right hu]
    linarith [le_max_right (u+v) (0:ℝ),le_max_left (u-v) (0:ℝ)]

theorem kink_gap_integral {d : ℕ} (hd : 0<d) (ρ : ℝ) (H : Coefficients d)
    (hs : ∀ i j, H i j=H j i) (hb : ∀ i j, |H i j|≤1)
    (j : RayIndex d) (hw : normSq (ray j)/2≤|quadratic H (ray j)|)
    (t : ℝ) (ht : t=1/(4*d)) (hp : |t| * d≤1/2) (hm : |-t| * d≤1/2) :
    (1/(50*d))*probability (kinkWindow ρ j) ≤
      (kinkIntegral ρ (perturbEllipsoid H hs hb t hp)+
        kinkIntegral ρ (perturbEllipsoid H hs hb (-t) hm))/2-kinkIntegral ρ (centralEllipsoid d) := by
  have hd0 : 0<(d:ℝ) := by exact_mod_cast hd
  have ht0 : 0<t := by rw [ht]; positivity
  let P := perturbEllipsoid H hs hb t hp
  let M := perturbEllipsoid H hs hb (-t) hm
  let f := fun z ↦ (ellipsoidalKink ρ P z+ellipsoidalKink ρ M z)/2-ellipsoidalKink ρ (centralEllipsoid d) z
  have HF (z : Point d) :
      f z = let u := 1-(3/2)*normSq (ρ⁻¹ • z)
            let v := t*quadratic H (ρ⁻¹ • z)
            (max (u+v) 0+max (u-v) 0)/2-max u 0 := by
    have hP : quadratic P.A (ρ⁻¹ • z) = (3/2)*normSq (ρ⁻¹ • z)+t*quadratic H (ρ⁻¹ • z) :=
      quadratic_perturbed_value H t _
    have hM : quadratic M.A (ρ⁻¹ • z) = (3/2)*normSq (ρ⁻¹ • z)+(-t)*quadratic H (ρ⁻¹ • z) :=
      quadratic_perturbed_value H (-t) _
    simp only [f,ellipsoidalKink,hP,hM,central_quad]
    rw [show 1-((3/2)*normSq (ρ⁻¹ • z)+(-t)*quadratic H (ρ⁻¹ • z)) =
          1-(3/2)*normSq (ρ⁻¹ • z)+t*quadratic H (ρ⁻¹ • z) by ring,
        show 1-((3/2)*normSq (ρ⁻¹ • z)+t*quadratic H (ρ⁻¹ • z)) =
          1-(3/2)*normSq (ρ⁻¹ • z)-t*quadratic H (ρ⁻¹ • z) by ring]
    rw [add_comm (max _ 0) (max _ 0)]
  have Hf (z : Point d) (hz : z ∈ kinkWindow ρ j) : 1/(50*d) ≤ f z := by
    let x := ρ⁻¹ • z
    have HH := local_form_bounds hd hb j x (near_kinkCenter j hz)
    have hc := kinkCenter_witness j hw
    have ho := abs_sub_le (quadratic H (kinkCenter j)) (quadratic H x) (0:ℝ)
    rw [sub_zero,sub_zero,abs_sub_comm (quadratic H (kinkCenter j)) (quadratic H x)] at ho
    have hql : 1/4 ≤ |quadratic H x| := by linarith [HH.1]
    have hn := HH.2
    rw [kinkCenter_norm] at hn
    have ee : |1-(3/2)*normSq x|=(3/2)*|normSq x-2/3| := by
      rw [show 1-(3/2)*normSq x= -((3/2)*(normSq x-2/3)) by ring,
        abs_neg,abs_mul,abs_of_nonneg (by norm_num : (0:ℝ)≤3/2)]
    have hlow := hinge_gap_margin (1-(3/2)*normSq x) (t*quadratic H x)
    rw [abs_mul,abs_of_pos ht0,ee] at hlow
    rw [HF]
    apply (le_trans ?_ hlow)
    rw [ht]
    have Ha := mul_le_mul_of_nonneg_left hql (by positivity : 0≤1/(4*(d:ℝ)))
    have Hb := mul_le_mul_of_nonneg_left hn (by norm_num : (0:ℝ) ≤ 3/2)
    have hc' : (1/(50*(d:ℝ)))*2 ≤ (1/(4*(d:ℝ)))*(1/4)-(3/2)*(3/(1000*(d:ℝ))) := by
      field_simp
      <;> norm_num
    linarith only [Ha,Hb,hc']
  have hwmeas : MeasurableSet (kinkWindow ρ j) :=
    (Metric.isOpen_ball.preimage (continuous_const_smul _)).measurableSet
  let ind := (kinkWindow ρ j).indicator (fun _ => (1/(50*d):ℝ))
  have hS : Integrable (fun z => (ellipsoidalKink ρ P z+ellipsoidalKink ρ M z)/2)
      (gaussian d) := ((kink_integrable ρ P).add (kink_integrable ρ M)).div_const _
  have hfint : Integrable f (gaussian d) := hS.sub (kink_integrable ρ (centralEllipsoid d))
  have hind : Integrable ind (gaussian d) := (integrable_const _).indicator hwmeas
  have hpw (z : Point d) : ind z ≤ f z := by
    by_cases hz : z ∈ kinkWindow ρ j
    · simpa [ind,hz] using Hf z hz
    · simp only [ind,indicator_of_notMem hz,HF]
      exact hinge_gap_nonneg _ _
  have hI := integral_mono hind hfint hpw
  unfold ind f at hI
  rw [integral_indicator hwmeas,setIntegral_const,smul_eq_mul,
    integral_sub hS
      (kink_integrable ρ (centralEllipsoid d)),
    integral_div,integral_add (kink_integrable ρ P) (kink_integrable ρ M)] at hI
  simpa [probability,kinkIntegral,P,M,mul_comm] using hI

end Cubature
