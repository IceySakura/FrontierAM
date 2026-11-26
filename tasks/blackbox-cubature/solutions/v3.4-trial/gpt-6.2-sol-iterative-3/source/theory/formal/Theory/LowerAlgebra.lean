import Theory.LowerModel
import Theory.Certificate
import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.Data.Sym.Card

open scoped BigOperators Matrix
open Matrix Set Unitary

namespace Cubature

variable {d n : ℕ}

theorem upperIndices_card (d : ℕ) : Fintype.card (UpperIndices d) = Nat.choose (d+1) 2 := by
  classical
  let f : UpperIndices d → Sym2 (Fin d) := fun p => Sym2.mk p.1.1 p.1.2
  have inj : Function.Injective f := by
    rintro ⟨⟨i,j⟩,hij⟩ ⟨⟨a,b⟩,hab⟩ he
    rcases Sym2.eq_iff.mp he with ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩
    · rfl
    · have h : i = j := le_antisymm hij hab
      cases h
      rfl
  have sur : Function.Surjective f := by
    apply Sym2.ind
    intro i j
    by_cases h : i≤j
    · exact ⟨⟨(i,j),h⟩,rfl⟩
    · exact ⟨⟨(j,i),le_of_not_ge h⟩, Sym2.eq_iff.mpr (Or.inr ⟨rfl,rfl⟩)⟩
  rw [Fintype.card_congr (Equiv.ofBijective f ⟨inj,sur⟩), Sym2.card, Fintype.card_fin]

theorem PointQueryPlan.points_length {V : Type} (f : Coordinates d → V) (P : PointQueryPlan d V n) :
    (P.points f).length = n := by
  induction P with
  | answer => rfl
  | ask x K ih => simp [PointQueryPlan.points, ih]

theorem PointQueryPlan.run_eq_of_agreement {V : Type} (f g : Coordinates d → V)
    (P : PointQueryPlan d V n) (h : ∀ x ∈ P.points f, f x = g x) :
    P.run f = P.run g := by
  induction P with
  | answer => rfl
  | ask x K ih =>
    have h₁ := h x (List.mem_cons_self ..)
    simpa only [PointQueryPlan.run, ← h₁] using ih (f x) (fun y hy => h y (List.mem_cons_of_mem _ hy))

def symmetricEntries (d : ℕ) (v : UpperIndices d → ℝ) : Coefficients d :=
  fun i j => if h : i ≤ j then v ⟨(i,j),h⟩ else v ⟨(j,i), (le_of_not_ge h)⟩

noncomputable def symmetricLift (d : ℕ) : (UpperIndices d → ℝ) →ₗ[ℝ] Coefficients d where
  toFun := symmetricEntries d
  map_add' v w := by
    ext i j
    by_cases h : i ≤ j <;> simp [symmetricEntries, Matrix.add_apply, h]
  map_smul' r v := by
    ext i j
    by_cases h : i ≤ j <;> simp [symmetricEntries, Matrix.smul_apply, h]

theorem symmetricLift_injective : Function.Injective (symmetricLift d) := by
  intro v w h
  funext ⟨⟨i,j⟩,hi⟩
  change symmetricEntries d v = symmetricEntries d w at h
  simpa [symmetricEntries, hi] using congrFun (congrFun h i) j

theorem symmetricLift_hermitian (v : UpperIndices d → ℝ) : (symmetricLift d v).IsHermitian := by
  ext i j
  change star (symmetricEntries d v j i) = symmetricEntries d v i j
  rcases lt_trichotomy i j with h | rfl | h
  · simp [symmetricEntries, h.le, not_le_of_gt h]
  · simp
  · simp [symmetricEntries, h.le, not_le_of_gt h]

theorem quadratic_add (A B : Coefficients d) (x : Coordinates d) :
    quadratic (A+B) x = quadratic A x + quadratic B x := by
  simp [quadratic, Matrix.add_apply, add_mul, Finset.sum_add_distrib]

theorem quadratic_smul_matrix (c : ℝ) (A : Coefficients d) (x : Coordinates d) :
    quadratic (c • A) x = c * quadratic A x := by
  simp only [quadratic, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem quadratic_dot (A : Coefficients d) (x : Coordinates d) :
    quadratic A x = x ⬝ᵥ (A *ᵥ x) := by
  simp only [quadratic, dotProduct, mulVec, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem quadratic_diagonal (v x : Coordinates d) :
    quadratic (diagonal v) x = ∑ i, v i*(x i)^2 := by
  simp only [quadratic_dot, dotProduct, mulVec_diagonal]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem quadratic_one (x : Coordinates d) :
    quadratic (1 : Coefficients d) x = squaredLength x := by
  rw [quadratic_dot, one_mulVec]
  simp [squaredLength, dotProduct, pow_two]

theorem invisible_symmetric (points : Fin n → Coordinates d)
    (h : n < Fintype.card (UpperIndices d)) :
    ∃ B : Coefficients d, B.IsHermitian ∧ B ≠ 0 ∧ ∀ i, quadratic B (points i) = 0 := by
  let L : (UpperIndices d → ℝ) →ₗ[ℝ] (Fin n → ℝ) :=
    { toFun := fun v i => quadratic (symmetricLift d v) (points i)
      map_add' := by intro v w; ext i; simp [quadratic_add]
      map_smul' := by intro c v; ext i; simp [quadratic_smul_matrix] }
  have hd : Module.finrank ℝ (Fin n → ℝ) < Module.finrank ℝ (UpperIndices d → ℝ) := by simpa
  obtain ⟨v,hv,hv0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot (L.ker_ne_bot_of_finrank_lt hd)
  refine ⟨symmetricLift d v, symmetricLift_hermitian v, ?_, ?_⟩
  · intro hz
    exact hv0 (symmetricLift_injective (hz.trans (map_zero _).symm))
  · intro i
    exact congrFun (LinearMap.mem_ker.mp hv) i

theorem unitary_quad (U : Matrix.unitaryGroup (Fin d) ℝ) (v x : Coordinates d) :
    quadratic (conjStarAlgAut ℝ _ U (diagonal v)) x =
      ∑ i, v i*((star (U : Coefficients d) *ᵥ x) i)^2 := by
  rw [conjStarAlgAut_apply, quadratic_dot, Matrix.mul_assoc,
    ← mulVec_mulVec, ← mulVec_mulVec, dotProduct_mulVec]
  have h : x ᵥ* (U : Coefficients d) = star (U : Coefficients d) *ᵥ x := by
    rw [← mulVec_transpose]
    congr 1
  rw [h, ← quadratic_dot, quadratic_diagonal]

theorem unitary_squaredLength (U : Matrix.unitaryGroup (Fin d) ℝ) (x : Coordinates d) :
    squaredLength (star (U : Coefficients d) *ᵥ x) = squaredLength x := by
  have h := unitary_quad U (1 : Coordinates d) x
  have he : diagonal (1 : Coordinates d) = (1 : Coefficients d) := by
    simpa only [Pi.one_def] using (Matrix.diagonal_one : diagonal (fun _ : Fin d => (1:ℝ)) = 1)
  rw [he, map_one, quadratic_one] at h
  simpa [squaredLength] using h.symm

theorem normalized_invisible {B : Coefficients d} (hB : B.IsHermitian) (hB' : B ≠ 0) :
    ∃ (c : ℝ) (U : Matrix.unitaryGroup (Fin d) ℝ) (v : Coordinates d),
      (1 : Coefficients d) + c • B = conjStarAlgAut ℝ _ U (diagonal v) ∧
      (∀ i, (1:ℝ)/2 ≤ v i ∧ v i ≤ 3/2) ∧ (∏ i, v i) ≤ 7/8 := by
  classical
  let eig := hB.eigenvalues
  have hne : eig ≠ 0 := fun h => hB' (hB.eigenvalues_eq_zero_iff.mp h)
  obtain ⟨j,hj⟩ := Function.ne_iff.mp hne
  have hj' : 0 < |eig j| := abs_pos.mpr hj
  obtain ⟨p,_,hp⟩ := Finset.exists_max_image Finset.univ (fun i => |eig i|)
    ⟨j,Finset.mem_univ _⟩
  have hp' : ∀ i, |eig i| ≤ |eig p| := fun i => hp i (Finset.mem_univ _)
  have hn : 0 < |eig p| := hj'.trans_le (hp' j)
  let c := (2*|eig p|)⁻¹
  let z := fun i => c*eig i
  have hb (i) : |z i| ≤ 1/2 := by
    have hc : 0 < c := by dsimp [c]; positivity
    dsimp [z]
    rw [abs_mul, abs_of_pos hc]
    calc
      c*|eig i| ≤ c*|eig p| := mul_le_mul_of_nonneg_left (hp' i) hc.le
      _ = 1/2 := by dsimp [c]; field_simp
  have hbp : |z p| = 1/2 := by
    dsimp [z,c]
    rw [abs_mul, abs_of_pos (by positivity : 0 < (2*|eig p|)⁻¹)]
    field_simp
  have hbounds (i) : -1/2 ≤ z i ∧ z i ≤ 1/2 := by
    obtain ⟨h₁,h₂⟩ := abs_le.mp (hb i)
    exact ⟨by linarith, h₂⟩
  let L := hB.eigenvectorUnitary
  have lift (c : ℝ) : (1 : Coefficients d) + c • B =
      conjStarAlgAut ℝ _ L (diagonal (fun i => 1+c*eig i)) := by
    have hdi : diagonal (fun i => 1+c*eig i) = (1 : Coefficients d) + c • diagonal eig := by
      ext i j
      by_cases h : i=j
      · subst j
        simp
      · simp [h, diagonal_apply_ne, Matrix.one_apply_ne]
    rw [hdi, map_add, map_one, map_smul]
    congr 2
    simpa [eig, L] using hB.spectral_theorem
  let plus := fun i => 1+z i
  let minus := fun i => 1-z i
  have hbplus : ∀ i, (1:ℝ)/2 ≤ plus i ∧ plus i ≤ 3/2 := by
    intro i; dsimp [plus]; constructor <;> linarith [(hbounds i).1,(hbounds i).2]
  have hbminus : ∀ i, (1:ℝ)/2 ≤ minus i ∧ minus i ≤ 3/2 := by
    intro i; dsimp [minus]; constructor <;> linarith [(hbounds i).1,(hbounds i).2]
  have hprod : (∏ i, plus i) * (∏ i, minus i) ≤ 3/4 := by
    rw [← Finset.prod_mul_distrib]
    have hp₀ : plus p * minus p = 3/4 := by
      have h := congrArg (fun x : ℝ => x^2) hbp
      rw [sq_abs] at h
      dsimp [plus,minus]
      nlinarith
    calc
      (∏ i, plus i * minus i) ≤ ∏ i, if i=p then (3/4:ℝ) else 1 := by
        apply Finset.prod_le_prod₀
        · intro i _
          exact mul_nonneg (by linarith [(hbplus i).1]) (by linarith [(hbminus i).1])
        · intro i _
          by_cases h : i=p
          · simpa [h, hp₀]
          · simp only [h, ↓reduceIte, plus, minus]
            nlinarith [sq_nonneg (z i)]
      _ = 3/4 := by simp
  by_cases H : (∏ i, plus i) ≤ 7/8
  · exact ⟨c,L,plus,lift c,hbplus,H⟩
  · refine ⟨-c,L,minus,?_,hbminus,?_⟩
    · convert lift (-c) using 3
      ext i
      dsimp [minus,z]
      ring
    · by_contra! G
      have H' : (7/8:ℝ) < ∏ i, plus i := lt_of_not_ge H
      have HH : (7/8:ℝ)*(7/8) < (∏ i, plus i)*(∏ i, minus i) :=
        (mul_lt_mul_of_pos_right H' (by norm_num : (0:ℝ)<7/8)).trans_le
          (mul_le_mul_of_nonneg_left G.le (by linarith))
      linarith

end Cubature
