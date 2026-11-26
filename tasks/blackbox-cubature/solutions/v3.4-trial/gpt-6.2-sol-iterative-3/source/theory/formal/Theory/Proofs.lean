import Theory.Certificate
import Theory.LowerMeasure
import Theory.PrecisionTree
import Theory.FixedTree

open scoped ENNReal
open MeasureTheory

namespace Cubature

theorem ellipsoid_mass_certificate {d m : ℕ} {f : Coordinates d → ℝ}
    {a b : Coordinates d} {A : Coefficients d} {ρ R J δ : ℝ}
    (hρ : 0 < ρ) (hR : ρ ≤ R) (hJ : 0 < J)
    (hSym : ∀ i j, A i j = A j i)
    (hA : ∀ x : Coordinates d, x ≠ 0 → 0 < quadratic A x)
    (hinner : ∀ x, inBox ρ x → quadratic A x + linear b x ≤ 1)
    (houter : ∀ x, quadratic A x + linear b x ≤ 1 → inBox R x)
    (hbox : ∀ i, |a i| + R ≤ 40)
    (hδ : 0 ≤ δ) (hδ' : δ < 1/2) (herr : polynomialError d ρ R m ≤ δ)
    (hf : JumpOracle f J a A b) :
    let T := observe f a ρ R m
    let queries := queryTrace a T
    let truth := J * (gaussian (ellipsoid a A b)).toReal
    let answer := f a * (gaussian (certifiedSet a R δ T)).toReal
    queries.length = 1 + 2*d^2*m ∧
    (∀ z ∈ queries, inBox 40 z) ∧
    0 < truth ∧ massFactor a (1-2*δ) * truth ≤ answer ∧ answer ≤ truth := by
  dsimp only
  let T := observe f a ρ R m
  let S := certifiedSet a R δ T
  let E := ellipsoid a A b
  let t := 1-2*δ
  have ht : 0 < t := by dsimp [t]; linarith
  have ht₁ : t ≤ 1 := by dsimp [t]; linarith
  have h0R := hρ.le.trans hR
  have obs (s : Fin d × Fin d × Bool) :
      (T s).points.length = m ∧ ∀ z ∈ (T s).points, inBox 40 z := by
    rcases s with ⟨i,j,s⟩
    obtain ⟨p,q,_,_,_,_,_,_,_,_,_,hlen,hpts⟩ := observe_ray (m:=m) i j s hρ hR hinner
      houter (hA _ (chord_ne_zero i j s)) hJ hf
    refine ⟨hlen,?_⟩
    intro z hz
    obtain ⟨r,hr,rfl⟩ := hpts _ hz
    intro k
    have hr0 := hρ.le.trans hr.1
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    calc
      |a k+r*orientedChord i j s k| ≤ |a k| + |r*orientedChord i j s k| := abs_add_le _ _
      _ ≤ |a k| + R := by
        rw [abs_mul, abs_of_nonneg hr0]
        have h := mul_le_mul_of_nonneg_left (chord_abs i j k s) hr0
        simpa using (add_le_add_left ((by simpa using h : r*|orientedChord i j s k|≤r).trans hr.2) (|a k|))
      _ ≤ 40 := hbox k
  have hlen : (queryTrace a T).length = 1+2*d^2*m := by
    have hl : ∀ s, (T s).points.length = m := fun s => (obs s).1
    simp [queryTrace, List.length_flatMap, hl, Fintype.card_prod, pow_two]
    ring
  have hcoords : ∀ z ∈ queryTrace a T, inBox 40 z := by
    intro z hz
    rcases List.mem_cons.mp hz with rfl | hh
    · intro i; linarith [hbox i]
    · obtain ⟨s,_,h⟩ := List.mem_flatMap.mp hh
      exact (obs s).2 z h
  have hef := fit_errors (m:=m) hρ hR hinner houter hA hSym hJ hf
  let e := bracketWidth ρ R m
  have he : 0 ≤ e := by dsimp [e,bracketWidth]; positivity
  have hfit x (hx : inBox R x) :
      |(quadratic (fittedMatrix T) x + linear (fittedLinear T) x) -
        (quadratic A x + linear b x)| ≤ δ := by
    have h := evaluation_error A (fittedMatrix T) b (fittedLinear T) x
      (M:=3*e/ρ^3) (N:=2*e/ρ^2) h0R (by positivity) (by positivity) hx hef.1 hef.2
    apply h.trans
    convert herr using 1
    dsimp [polynomialError, e]
    ring
  obtain ⟨hs1,hs2⟩ := contraction_subset hA R δ hδ hδ' houter hfit
  have hE : MeasurableSet E := measurable_ellipsoid A b a
  have hEt : gaussian E ≠ ⊤ := gaussian_box_finite a R hE (fun z hz => houter (z-a) hz)
  have hSt : gaussian S ≠ ⊤ := ne_top_of_le_ne_top hEt (measure_mono hs2)
  have hEpos : 0 < gaussian E := gaussian_ellipsoid_positive a b A hρ hinner
  have htruth : 0 < J * (gaussian E).toReal :=
    mul_pos hJ (ENNReal.toReal_pos hEpos.ne' hEt)
  have hmass := (gaussian_contraction_lower a ht ht₁ hE).trans (measure_mono hs1)
  have H := ENNReal.toReal_mono hSt hmass
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (by unfold massFactor; positivity)] at H
  have K := ENNReal.toReal_mono hEt (measure_mono hs2)
  have height : f a = J := by
    have h : quadratic A (0 : Coordinates d) + linear b 0 < 1 := by
      simp [quadratic,linear]
    simpa using (hf 0).1 h
  change (queryTrace a T).length = 1+2*d^2*m ∧ (∀ z ∈ queryTrace a T, inBox 40 z) ∧
    0 < J*(gaussian E).toReal ∧
    massFactor a t*(J*(gaussian E).toReal) ≤ f a*(gaussian S).toReal ∧
    f a*(gaussian S).toReal ≤ J*(gaussian E).toReal
  rw [height]
  refine ⟨hlen,hcoords,htruth,?_,mul_le_mul_of_nonneg_left K hJ.le⟩
  simpa [mul_comm, mul_left_comm, mul_assoc] using mul_le_mul_of_nonneg_left H hJ.le

theorem centered_ellipsoid_query_lower {d n : ℕ} {V : Type} (P : PointQueryPlan d V n)
    (read : Coordinates d → ℝ → V)
    (hN : n < Nat.choose (d+1) 2) :
    ∃ A : Coefficients d, A.IsHermitian ∧
      (∀ x, (1/2 : ℝ) * squaredLength x ≤ quadratic A x ∧
            quadratic A x ≤ (3/2 : ℝ) * squaredLength x) ∧
      let truth := (gaussian (centeredBody A)).toReal
      0 < truth ∧ truth/50 ≤ |P.run (fun x => read x (quadratic A x)) - truth| := by
  classical
  let f := fun x => read x (quadratic (1 : Coefficients d) x)
  let L := P.points f
  have hL : L.length < Fintype.card (UpperIndices d) := by
    rw [upperIndices_card]
    simpa [L,PointQueryPlan.points_length] using hN
  have Q (x : Coordinates d) :
      (1/2:ℝ)*squaredLength x ≤ quadratic (1 : Coefficients d) x ∧
        quadratic (1 : Coefficients d) x ≤ (3/2:ℝ)*squaredLength x := by
    rw [quadratic_one]
    constructor <;> linarith [square_sum_nonneg x]
  let I := (gaussian (centeredBody (1 : Coefficients d))).toReal
  have hI : 0 < I :=
    ENNReal.toReal_pos centeredBody_identity_positive.ne' (centeredBody_finite (fun x => (Q x).1))
  by_cases hGood : I/50 ≤ |P.run f - I|
  · exact ⟨1,Matrix.isHermitian_one,Q,hI,hGood⟩
  · obtain ⟨B,hB,hB0,hBpts⟩ := invisible_symmetric L.get hL
    obtain ⟨c,U,v,hc,hv,hvdet⟩ := normalized_invisible hB hB0
    let A : Coefficients d := 1 + c • B
    have hA : A.IsHermitian := Matrix.isHermitian_one.add (hB.smul (by simp [IsSelfAdjoint]))
    have hb x :
        (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧
          quadratic A x ≤ (3/2:ℝ)*squaredLength x := by
      change (1/2:ℝ)*squaredLength x ≤ quadratic (1+c • B) x ∧
        quadratic (1+c • B) x ≤ (3/2:ℝ)*squaredLength x
      rw [hc]
      exact quadratic_conjugated_bounds U v hv x
    have hp x (hx : x ∈ L) : quadratic A x = quadratic (1 : Coefficients d) x := by
      obtain ⟨i,rfl⟩ := List.mem_iff_get.mp hx
      change quadratic (1+c • B) (L.get i) = quadratic (1 : Coefficients d) (L.get i)
      rw [quadratic_add,quadratic_smul_matrix,hBpts i,mul_zero,add_zero]
    have oe : P.run f = P.run (fun x => read x (quadratic A x)) := by
      apply PointQueryPlan.run_eq_of_agreement
      intro x hx
      simp [f,hp x hx]
    let I' := (gaussian (centeredBody A)).toReal
    have gap : (21/20:ℝ)*I ≤ I' := by
      dsimp [I,I',A]
      rw [hc]
      exact diagonalized_mass_gap (U:=U) hv hvdet
    have hp : 0 < I' := by linarith
    refine ⟨A,hA,hb,hp,?_⟩
    change I'/50 ≤ |P.run (fun x => read x (quadratic A x)) - I'|
    rw [← oe]
    by_contra! hz
    have h1 := (abs_lt.mp (lt_of_not_ge hGood)).2
    have h2 := (abs_lt.mp hz).1
    linarith

theorem exact_quadratic_reconstruction {d : ℕ} (A : Coefficients d)
    (hA : ∀ i j, A i j = A j i) :
    Fintype.card (UpperIndices d) = Nat.choose (d+1) 2 ∧
    (∀ p, inBox 40 (quadraticProbePoints d p)) ∧
    matrixFromQuadraticValues (fun p => quadratic A (quadraticProbePoints d p)) = A := by
  refine ⟨upperIndices_card d,?_,?_⟩
  · rintro ⟨⟨i,j⟩,h⟩ k
    exact ((by simpa [orientedChord,quadraticProbePoints] using chord_abs i j k true) :
      |quadraticProbePoints d ⟨(i,j),h⟩ k| ≤ 1).trans (by norm_num)
  · ext i j
    by_cases h : i=j
    · subst j
      simp [matrixFromQuadraticValues,quadraticProbePoints,chord,quadratic_coordinate]
    · have he (i : Fin d) : quadratic A (chord i i) = A i i := by
        simp [chord,quadratic_coordinate]
      simp only [matrixFromQuadraticValues,h,↓reduceDIte,quadraticProbePoints,he]
      rcases lt_or_gt_of_ne h with hij | hji
      · rw [min_eq_left hij.le, max_eq_right hij.le, quadratic_chord A hA h]
        ring
      · rw [min_eq_right hji.le, max_eq_left hji.le, quadratic_chord A hA (ne_of_lt hji), hA j i]
        ring

theorem ellipsoid_membership_precision_lower {d m n : ℕ} (hd : 0 < d) (hm : 2 ≤ m)
    (P : PointQueryPlan d Bool n) (hN : 2^n < m^(Nat.choose (d+1) 2))
    (r2 : ℝ) (hr : 0 < r2) (hr' : r2 ≤ (precisionGap d m)^2/100) :
    ∃ A : Coefficients d, A.IsHermitian ∧
      (∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x) ∧
      let truth := (gaussian (bodyAt r2 A)).toReal
      0 < truth ∧ (precisionGap d m)^2/20*truth ≤ |P.run (memberAt r2 A)-truth| := by
  exact grid_precision_lower hd hm P hN r2 hr hr'

theorem fixed_radius_ellipsoid_precision_lower {d m n : ℕ} (hd : 0 < d) (hm : 2 ≤ m)
    (P : PointQueryPlan d Bool n) (hN : 2^n < m^(Nat.choose (d+1) 2))
    (r2 : ℝ) (hr : 0 < r2) (hr' : r2 ≤ 1/100) :
    ∃ A : Coefficients d, A.IsHermitian ∧
      (∀ x, (3/4:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (5/4:ℝ)*squaredLength x) ∧
      let truth := (gaussian (bodyAt r2 A)).toReal
      0 < truth ∧ (precisionGap d m)^2/40*truth ≤ |P.run (memberAt r2 A)-truth| := by
  exact fixed_grid_precision hd hm P hN r2 hr hr'

end Cubature
