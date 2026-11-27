import Theory.EllipsoidForm
import Mathlib.Data.Sym.Card

/-! Relative Gaussian-mass cubature of a centered ellipsoid from Boolean point queries. -/

open Set MeasureTheory Metric WithLp
open scoped Sym2

namespace Cubature

noncomputable section

namespace QueryTree

variable {Q A : Type*}

private def flipAnswers : QueryTree Q A → QueryTree Q A
  | .returnValue a => .returnValue a
  | .ask q k => .ask q (fun b => flipAnswers (k (!b)))

private theorem run_flipAnswers (o : Q → Bool) (t : QueryTree Q A) :
    (flipAnswers t).run o = t.run (fun q => !(o q)) := by
  induction t with
  | returnValue => rfl
  | ask q k ih => exact ih (!(o q))

private theorem depth_flipAnswers (t : QueryTree Q A) :
    (flipAnswers t).depth = t.depth := by
  induction t with
  | returnValue => rfl
  | ask q k ih => simp [flipAnswers,depth,ih,max_comm]

private theorem queriesIn_flipAnswers {t : QueryTree Q A} {S : Set Q} (h : t.queriesIn S) :
    (flipAnswers t).queriesIn S := by
  induction t with
  | returnValue => trivial
  | ask q k ih => exact ⟨h.1, fun b => ih (!b) (h.2 (!b))⟩

end QueryTree

private def inverseRoot (s : ℝ) : ℝ := (Real.sqrt s)⁻¹

private theorem inverseRoot_sq {s : ℝ} (hs : 0 ≤ s) : (inverseRoot s)^2 = s⁻¹ := by
  rw [inverseRoot,inv_pow,Real.sq_sqrt hs]

private def formTest {d : ℕ} (v : GaussianSpace d) (q : Unit × ℝ) : GaussianSpace d :=
  inverseRoot q.2 • v

private theorem formTest_legal {d : ℕ} (v : GaussianSpace d) (hv : ∀ i, |v i| ≤ 1)
    {L s : ℝ} (hL : 0 < L) (hcap : 1 ≤ 1600*L) (hs : L ≤ s) :
    formTest v ((),s) ∈ coordinateBox d 40 := by
  have hp : 0 < s := hL.trans_le hs
  have hroot : 1/40 ≤ Real.sqrt s :=
    Real.le_sqrt_of_sq_le (by nlinarith)
  have hb : inverseRoot s ≤ 40 := by
    have h := (inv_le_inv₀ (Real.sqrt_pos.mpr hp) (by norm_num : (0:ℝ)<1/40)).mpr hroot
    norm_num [inverseRoot] at h ⊢
    exact h
  have hR : 0 ≤ inverseRoot s := inv_nonneg.mpr (Real.sqrt_nonneg _)
  intro i
  rw [formTest,PiLp.smul_apply,smul_eq_mul,abs_mul,abs_of_nonneg hR]
  exact (mul_le_of_le_one_right hR (hv i)).trans hb

private theorem formTest_answer {d : ℕ} (v : GaussianSpace d) (Q : RealMatrix d)
    {s : ℝ} (hs : 0 < s) :
    (!(ellipsoidOracle Q (formTest v ((),s)))) = decide (s ≤ quadEval Q v) := by
  have he : quadEval Q (formTest v ((),s)) = quadEval Q v / s := by
    unfold formTest
    rw [quadEval_smul,inverseRoot_sq hs.le,div_eq_mul_inv,mul_comm]
  have he' : ellipsoidOracle Q (formTest v ((),s)) = decide (quadEval Q v < s) :=
    decide_eq_decide.mpr (by rw [he,div_lt_one hs])
  rw [he']
  by_cases h : s ≤ quadEval Q v
  · simp [h,not_lt_of_ge h]
  · simp [h,lt_of_not_ge h]

private def formBisection {d : ℕ} (v : GaussianSpace d) (k : ℕ) (L U : ℝ) :
    QueryTree (GaussianSpace d) ℝ :=
  ((bisect () k L (2*U)).reindex (formTest v)).flipAnswers

private theorem formBisection_depth {d : ℕ} (v : GaussianSpace d) (k : ℕ) (L U : ℝ) :
    (formBisection v k L U).depth = k := by
  rw [formBisection,QueryTree.depth_flipAnswers,QueryTree.depth_reindex,bisect_depth]

private theorem formBisection_legal {d : ℕ} (v : GaussianSpace d) (hv : ∀ i, |v i| ≤ 1)
    (k : ℕ) {L U : ℝ} (hL : 0 < L) (hLU : L ≤ U) (hcap : 1 ≤ 1600*L) :
    (formBisection v k L U).queriesIn (coordinateBox d 40) := by
  apply QueryTree.queriesIn_flipAnswers
  apply QueryTree.queriesIn_reindex
  have ht := bisect_legal () k (a := L) (b := 2*U) le_rfl (by linarith) le_rfl
  refine QueryTree.queriesIn_mono ?_ ht
  rintro ⟨⟨⟩,s⟩ (hs : s ∈ Icc L (2*U))
  exact formTest_legal v hv hL hcap hs.1

private theorem formBisection_error {d : ℕ} (v : GaussianSpace d) (Q : RealMatrix d)
    (k : ℕ) {L U : ℝ} (hL : 0 < L) (hR : quadEval Q v ∈ Icc L (2*U)) :
    |(formBisection v k L U).run (ellipsoidOracle Q) - quadEval Q v| ≤ formResolution L U k := by
  let o (q : Unit × ℝ) := decide (q.2 ≤ quadEval Q v)
  have he : (formBisection v k L U).run (ellipsoidOracle Q) =
      (bisect () k L (2*U)).run o := by
    rw [formBisection,QueryTree.run_flipAnswers,QueryTree.run_reindex]
    refine QueryTree.run_congr (bisect_legal () k le_rfl (hR.1.trans hR.2) le_rfl) ?_
    rintro ⟨⟨⟩,s⟩ (hs : s ∈ Icc L (2*U))
    exact formTest_answer v Q (hL.trans_le hs.1)
  rw [he]
  have ho (x : Unit) (s : ℝ) :
      (o (x,s) = true → s ≤ (fun _ : Unit => quadEval Q v) x) ∧
      (o (x,s) = false → (fun _ : Unit => quadEval Q v) x ≤ s) := by
    simp only [o,decide_eq_true_eq,decide_eq_false_iff_not]
    exact ⟨id,le_of_lt ∘ lt_of_not_ge⟩
  exact bisect_error o (fun _ : Unit => quadEval Q v) ho () k hR.1 hR.2

private def unorderedProbe {d : ℕ} : Sym2 (Fin d) → GaussianSpace d :=
  Sym2.lift ⟨quadProbe,quadProbe_comm⟩

private theorem unorderedProbe_abs {d : ℕ} (t : Sym2 (Fin d)) (i : Fin d) :
    |unorderedProbe t i| ≤ 1 := by
  obtain ⟨⟨a,b⟩,rfl⟩ := Quot.exists_rep t
  exact quadProbe_abs a b i

private def probeIndex (d : ℕ) : Sym2 (Fin d) ≃ Fin ((d+1).choose 2) :=
  Fintype.equivOfCardEq (by rw [Sym2.card, Fintype.card_fin, Fintype.card_fin])

def ellipsoidRule (d k : ℕ) (L U : ℝ) : QueryTree (GaussianSpace d) ℝ :=
  (QueryTree.batch ((d+1).choose 2) (fun i =>
    formBisection (unorderedProbe ((probeIndex d).symm i)) k L U)).map (fun samples =>
    gaussianSetMass (ellipsoidSet (reconstructForm (fun i j => samples (probeIndex d s(i,j))))))

theorem ellipsoidRule_cost (d k : ℕ) (L U : ℝ) :
    (ellipsoidRule d k L U).depth ≤ (d+1).choose 2 * k := by
  unfold ellipsoidRule
  exact (QueryTree.depth_map_le _ _).trans (QueryTree.depth_batch_le _ _ _
    (fun _ => (formBisection_depth _ _ _ _).le))

theorem ellipsoidRule_legal (d k : ℕ) {L U : ℝ} (hL : 0 < L) (hLU : L ≤ U) (hcap : 1 ≤ 1600*L) :
    (ellipsoidRule d k L U).queriesIn (coordinateBox d 40) :=
  QueryTree.queriesIn_map _ (QueryTree.queriesIn_batch _ _ (fun _ =>
    formBisection_legal _ (unorderedProbe_abs _) _ hL hLU hcap))

private theorem ellipsoidSet_pos {d : ℕ} {Q : RealMatrix d} {U : ℝ} (hU : 0 < U)
    (h : ∀ z, quadEval Q z ≤ U*‖z‖^2) :
    0 < gaussianSetMass (ellipsoidSet Q) := by
  have ht : 0 < inverseRoot U := inv_pos.mpr (Real.sqrt_pos.mpr hU)
  refine (gaussianBallMass_pos (0:GaussianSpace d) ht).trans_le (gaussianSetMass_mono ?_)
  intro z hz
  have hn : ‖z‖ < inverseRoot U := by simpa using (show dist z (0:GaussianSpace d) < _ from hz)
  have hsq := (sq_lt_sq₀ (norm_nonneg z) ht.le).mpr hn
  exact (h z).trans_lt ((mul_lt_mul_of_pos_left hsq hU).trans_eq (by rw [inverseRoot_sq hU.le,mul_inv_cancel₀ hU.ne']))

theorem ellipsoidRule_accuracy {d k : ℕ} (hd : 0 < d) {L U : ℝ}
    (hL : 0 < L) (hLU : L ≤ U)
    (hsmall : (d:ℝ)*relativeFormResolution d k L U ≤ 1/2)
    {Q : RealMatrix d} (hQ : ∀ i j, Q i j = Q j i)
    (hSpec : ∀ z, L*‖z‖^2 ≤ quadEval Q z ∧ quadEval Q z ≤ U*‖z‖^2) :
    0 < gaussianSetMass (ellipsoidSet Q) ∧
      |(ellipsoidRule d k L U).run (ellipsoidOracle Q) - gaussianSetMass (ellipsoidSet Q)| ≤
        (2*(d:ℝ)*relativeFormResolution d k L U)*gaussianSetMass (ellipsoidSet Q) := by
  have hU := hL.trans_le hLU
  refine ⟨ellipsoidSet_pos hU (fun z => (hSpec z).2),?_⟩
  let δ := formResolution L U k
  let samples (i j : Fin d) := (formBisection (quadProbe i j) k L U).run (ellipsoidOracle Q)
  have hδ : 0 ≤ δ := div_nonneg (by linarith) (by positivity)
  have hs (i j : Fin d) : |samples i j - quadEval Q (quadProbe i j)| ≤ δ := by
    apply formBisection_error _ Q _ hL
    have hn := quadProbe_norm i j
    have hb := hSpec (quadProbe i j)
    exact ⟨(le_mul_of_one_le_right hL.le hn.1).trans hb.1,
      hb.2.trans ((mul_le_mul_of_nonneg_left hn.2 hU.le).trans_eq (mul_comm _ _))⟩
  have hO : (ellipsoidRule d k L U).run (ellipsoidOracle Q) =
      gaussianSetMass (ellipsoidSet (reconstructForm samples)) := by
    simp only [ellipsoidRule,QueryTree.run_map,QueryTree.run_batch,Equiv.symm_apply_apply,
      unorderedProbe,Sym2.lift_mk,samples]
  rw [hO]
  let η := relativeFormResolution d k L U
  have hη : 0 ≤ η := div_nonneg (by positivity) hL.le
  refine ellipsoid_relative_from_form hd hη hsmall
    (fun z => (mul_nonneg hL.le (sq_nonneg _)).trans (hSpec z).1) (fun z => ?_)
  calc
    _ ≤ (2*δ*(d:ℝ))*‖z‖^2 := reconstruct_quad_error hQ hδ hs z
    _ = η*(L*‖z‖^2) := by
      change _ = (2*δ*(d:ℝ)/L)*(L*‖z‖^2)
      field_simp [hL.ne']
    _ ≤ _ := mul_le_mul_of_nonneg_left (hSpec z).1 hη

end

end Cubature
