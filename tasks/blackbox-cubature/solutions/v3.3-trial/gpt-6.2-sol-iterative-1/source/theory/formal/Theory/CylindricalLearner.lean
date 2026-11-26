import Theory.CylindricalRays

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

theorem coordinate_hasMax {d : ℕ} (hd : 0<d) (z : Point d) : ∃ k, ∀ j, z j≤z k := by
  obtain ⟨k,_,h⟩ := Finset.exists_max_image (Finset.univ : Finset (Fin d)) z
    ⟨⟨0,hd⟩,Finset.mem_univ _⟩
  exact ⟨k,fun j => h j (Finset.mem_univ _)⟩

noncomputable def coordinateLargest {d : ℕ} (hd : 0<d) (z : Point d) : Fin d :=
  Classical.choose (coordinate_hasMax hd z)

theorem coordinateLargest_isMax {d : ℕ} (hd : 0<d) (z : Point d) :
    ∀ j, z j≤z (coordinateLargest hd z) := Classical.choose_spec (coordinate_hasMax hd z)

noncomputable def signedAxis {d : ℕ} (i : Bool × Fin d) : Point d :=
  if i.1 then axis i.2 else -axis i.2

theorem signedAxis_norm {d : ℕ} (i : Bool × Fin d) : normSq (signedAxis i) = 1 := by
  obtain ⟨b,i⟩ := i
  have H : normSq (-axis i)=normSq (axis i) := by simp [normSq]
  cases b <;> simp [signedAxis,H,normSq_axis]

theorem signedAxis_bound {d : ℕ} (i : Bool × Fin d) (j : Fin d) : |signedAxis i j|≤2 := by
  obtain ⟨b,i⟩ := i
  cases b <;> simpa [signedAxis] using (axis_bound i j).trans (by norm_num : (1:ℝ)≤2)

noncomputable def readShiftAxes {d : ℕ} (ρ : ℝ) (w : Point d) (m : ℕ) :
    QueryTree (Point d) (Bool × Fin d → ℝ) :=
  QueryTree.collect (fun i => cylindricalProbe ρ (rankShift w (signedAxis i)) m)
    (Finset.univ : Finset (Bool × Fin d)).toList

noncomputable def readShiftRow {d : ℕ} (ρ : ℝ) (w : Point d) (m : ℕ) (k : Fin d) :
    QueryTree (Point d) (Point d) :=
  QueryTree.collect (fun i => cylindricalProbe ρ (rankShift w (axis k+axis i)) m)
    (Finset.univ : Finset (Fin d)).toList

noncomputable def fittedDiagonal {d : ℕ} (a : ℝ) (X : Bool × Fin d → ℝ) : Point d :=
  fun j => 128*(X (true,j)+X (false,j)-2*a)

noncomputable def fittedColumn {d : ℕ} (X : Bool × Fin d → ℝ) : Point d :=
  fun j => 4*(X (true,j)-X (false,j))

noncomputable def fittedAnchorMatrix {d : ℕ} (a : ℝ) (X : Bool × Fin d → ℝ) : Coefficients d :=
  rankOnePart (fittedColumn X) a

noncomputable def fittedResidualDiagonal {d : ℕ} (a : ℝ) (X : Bool × Fin d → ℝ) : Point d :=
  fun j => fittedDiagonal a X j-fittedAnchorMatrix a X j j

noncomputable def fittedResidualRow {d : ℕ} (a : ℝ) (X : Bool × Fin d → ℝ)
    (k : Fin d) (m : Point d) : Point d :=
  fun j => 128*(m j-X (true,k)-X (true,j)+a)-fittedAnchorMatrix a X k j

noncomputable def cylinderLearner {d : ℕ} (hd : 0<d) (ρ : ℝ) (w : Point d) (m : ℕ) :
    QueryTree (Point d) (Coefficients d) :=
  (cylindricalProbe ρ w m).bind fun a =>
    (readShiftAxes ρ w m).bind fun X =>
      let D := fittedResidualDiagonal a X
      let k := coordinateLargest hd D
      if 2*(600*cylinderNoise m)<D k then
        (readShiftRow ρ w m k).map fun Z =>
          fittedAnchorMatrix a X+rankOnePart (fittedResidualRow a X k Z) (D k)
      else .ret (fittedAnchorMatrix a X)

theorem readShiftAxes_depth {d : ℕ} (ρ : ℝ) (w : Point d) (m : ℕ) :
    (readShiftAxes ρ w m).depth≤2*d*m := by
  unfold readShiftAxes
  simpa [Fintype.card_prod] using QueryTree.depth_collect
    (fun i : Bool × Fin d => cylindricalProbe ρ (rankShift w (signedAxis i)) m)
    (Finset.univ : Finset (Bool × Fin d)).toList m (fun _ => cylindricalProbe_depth _ _ _)

theorem readShiftRow_depth {d : ℕ} (ρ : ℝ) (w : Point d) (m : ℕ) (k : Fin d) :
    (readShiftRow ρ w m k).depth≤d*m := by
  unfold readShiftRow
  simpa using QueryTree.depth_collect
    (fun i : Fin d => cylindricalProbe ρ (rankShift w (axis k+axis i)) m)
    (Finset.univ : Finset (Fin d)).toList m (fun _ => cylindricalProbe_depth _ _ _)

theorem cylinderLearner_depth {d : ℕ} (hd : 0<d) (ρ : ℝ) (w : Point d) (m : ℕ) :
    (cylinderLearner hd ρ w m).depth≤(3*d+1)*m := by
  have H (a : ℝ) : ∀ X : Bool × Fin d → ℝ,
      (let D := fittedResidualDiagonal a X
       let k := coordinateLargest hd D
       if 2*(600*cylinderNoise m)<D k then
        (readShiftRow ρ w m k).map fun Z =>
          fittedAnchorMatrix a X+rankOnePart (fittedResidualRow a X k Z) (D k)
       else .ret (fittedAnchorMatrix a X)).depth ≤ d*m := by
    intro X
    dsimp only
    split
    · exact (QueryTree.depth_map_le _ _).trans (readShiftRow_depth _ _ _ _)
    · simp [QueryTree.depth]
  unfold cylinderLearner
  refine (QueryTree.depth_bind_le _ _ (2*d*m+d*m) (fun a =>
    (QueryTree.depth_bind_le _ _ (d*m) (H a)).trans
      (Nat.add_le_add_right (readShiftAxes_depth _ _ _) _))).trans ?_
  have h := Nat.add_le_add_right (cylindricalProbe_depth ρ w m) (2*d*m+d*m)
  exact h.trans (by ring_nf; omega)

theorem cylinderLearner_allowed {d : ℕ} (hd : 0<d) {ρ : ℝ} (hρ : 0≤ρ) (hρ' : ρ≤16)
    (w : Point d) (hw : ∀ j, |w j|≤1) (m : ℕ) :
    (cylinderLearner hd ρ w m).admissible allowed := by
  unfold cylinderLearner
  refine QueryTree.admissible_bind _ _ _ (cylindricalProbe_allowed hρ hρ' w
    (fun i => (hw i).trans (by norm_num)) m) ?_
  intro a
  apply QueryTree.admissible_bind
  · unfold readShiftAxes
    apply QueryTree.admissible_collect
    intro i
    exact cylindricalProbe_allowed hρ hρ' _ (rankShift_bound hw (signedAxis_bound i)) m
  · intro X
    dsimp only
    split
    · apply QueryTree.admissible_map
      unfold readShiftRow
      apply QueryTree.admissible_collect
      intro i
      exact cylindricalProbe_allowed hρ hρ' _ (rankShift_bound hw (axis_pair_coord_bound _ i)) m
    · trivial

theorem shift_estimates_diagonal_column {d : ℕ} (G : RankTwoCap d) (w : Point d) {θ a : ℝ}
    (X : Bool × Fin d → ℝ) (ha : |a-quadratic G.matrix w|≤θ)
    (hx : ∀ i, |X i-quadratic G.matrix (rankShift w (signedAxis i))|≤θ) :
    (∀ j, |fittedDiagonal a X j-G.matrix j j|≤512*θ) ∧
    (∀ j, |fittedColumn X j-curvatureProduct G w j|≤8*θ) := by
  have hp (j : Fin d) := abs_le.mp (hx (true,j))
  have hm (j : Fin d) := abs_le.mp (hx (false,j))
  have ha' := abs_le.mp ha
  constructor
  · intro j
    have hp := hp j
    have hm := hm j
    simp only [signedAxis,Prod.fst,Prod.snd,Bool.true_eq,ite_true,
      Bool.false_eq_true,ite_false,rankShift_plus_formula,rankShift_minus_formula] at hp hm
    exact abs_le.mpr ⟨by unfold fittedDiagonal; linarith only [hp.1,hm.1,ha'.2],
      by unfold fittedDiagonal; linarith only [hp.2,hm.2,ha'.1]⟩
  · intro j
    have hp := hp j
    have hm := hm j
    simp only [signedAxis,Prod.fst,Prod.snd,Bool.true_eq,ite_true,
      Bool.false_eq_true,ite_false,rankShift_plus_formula,rankShift_minus_formula] at hp hm
    exact abs_le.mpr ⟨by unfold fittedColumn; linarith only [hp.1,hm.2],
      by unfold fittedColumn; linarith only [hp.2,hm.1]⟩

theorem cylinderLearner_precision {d : ℕ} (hd : 0<d) {ρ : ℝ} (hρ : 0<ρ)
    (G : RankTwoCap d) (w : Point d) (ha₁ : 1≤quadratic G.matrix w) (ha₂ : quadratic G.matrix w≤2)
    (m : ℕ) (hnoise : cylinderNoise m≤1/8) :
    let M := (cylinderLearner hd ρ w m).run (answerBit (cylinderRegion ρ G))
    ∀ i j, |M i j-G.matrix i j|≤cylinderMatrixError m := by
  let f := answerBit (cylinderRegion ρ G)
  let a := (cylindricalProbe ρ w m).run f
  let X := (readShiftAxes ρ w m).run f
  let θ := cylinderNoise m
  let τ := 600*θ
  let A0 := quadratic G.matrix w
  let B := fittedAnchorMatrix a X
  let D := fittedResidualDiagonal a X
  let v := rankRemainderFactor G w
  have hθ : 0≤θ := by unfold θ cylinderNoise; positivity
  have hτ : 0≤τ := by positivity
  have ha0 : 0<A0 := by linarith
  have ha : |a-A0|≤θ := cylindricalProbe_precision hρ G w ⟨by linarith,by linarith⟩ m
  have hx (k : Bool × Fin d) :
      |X k-quadratic G.matrix (rankShift w (signedAxis k))|≤θ := by
    have h : X k = (cylindricalProbe ρ (rankShift w (signedAxis k)) m).run f :=
      QueryTree.run_collect _ _ _ (by simp)
    rw [h]
    exact cylindricalProbe_precision hρ G _ (rankShift_curvature G w _ ha₁ ha₂
      (by rw [signedAxis_norm];norm_num)) m
  obtain ⟨hDiag,hCol⟩ := shift_estimates_diagonal_column G w X ha hx
  have hB (i j : Fin d) :
      |B i j-rankOnePart (curvatureProduct G w) A0 i j|≤88*θ :=
    outer_approximation_bound hθ hnoise ha₁ ha (curvatureProduct_bound G w ha₂ i)
      (curvatureProduct_bound G w ha₂ j) (hCol i) (hCol j)
  have hd0 (i : Fin d) : |D i-(v i)^2|≤τ := by
    have H := rank_remainder G w ha0 i i
    change |fittedDiagonal a X i-B i i-(v i)^2|≤τ
    rw [pow_two,← H]
    have p := abs_le.mp (hDiag i)
    have q := abs_le.mp (hB i i)
    exact abs_le.mpr ⟨by dsimp [τ]; linarith only [p.1,q.2],
      by dsimp [τ]; linarith only [p.2,q.1]⟩
  let k := coordinateLargest hd D
  have hmax := coordinateLargest_isMax hd D
  let Z := (readShiftRow ρ w m k).run f
  have hZ (i : Fin d) :
      |Z i-quadratic G.matrix (rankShift w (axis k+axis i))|≤θ := by
    have he : Z i=(cylindricalProbe ρ (rankShift w (axis k+axis i)) m).run f :=
      QueryTree.run_collect _ _ _ (by simp)
    rw [he]
    exact cylindricalProbe_precision hρ G _ (rankShift_curvature G w _ ha₁ ha₂
      (axis_pair_norm_bound k i)) m
  have hSr (j : Fin d) : |fittedResidualRow a X k Z j-v k*v j|≤τ := by
    have p := abs_le.mp (hZ j)
    have l := abs_le.mp (hx (true,k))
    have r := abs_le.mp (hx (true,j))
    have h0 := abs_le.mp ha
    have b := abs_le.mp (hB k j)
    have hs := rankShift_pair_formula G w k j
    simp only [signedAxis,Prod.fst,Prod.snd,Bool.true_eq,ite_true] at l r
    rw [← rank_remainder G w ha0 k j]
    change |128*(Z j-X (true,k)-X (true,j)+a)-B k j -
      (G.matrix k j-rankOnePart (curvatureProduct G w) A0 k j)|≤τ
    exact abs_le.mpr ⟨by dsimp [τ]; linarith only [p.1,l.2,r.2,h0.1,b.2,hs],
      by dsimp [τ]; linarith only [p.2,l.1,r.1,h0.2,b.1,hs]⟩
  simp only [cylinderLearner,QueryTree.run_bind]
  change ∀ i j, |(if 2*τ<D k then (readShiftRow ρ w m k).map
      (fun Z => B+rankOnePart (fittedResidualRow a X k Z) (D k))
      else QueryTree.ret B).run f i j-G.matrix i j|≤cylinderMatrixError m
  intro i j
  have hre := rank_remainder G w ha0 i j
  have hib := abs_le.mp (hB i j)
  by_cases hp : 2*τ<D k
  · rw [ite_eq_left hp,QueryTree.run_map,Pi.add_apply,Pi.add_apply]
    change |B i j+fittedResidualRow a X k Z i*fittedResidualRow a X k Z j/(D k) -
      G.matrix i j|≤cylinderMatrixError m
    have hf := abs_le.mp (stable_rank_one_pivot v D (fittedResidualRow a X k Z) k hτ hp hmax
      hd0 hSr i j)
    unfold cylinderMatrixError
    change _ ≤ 8000*θ
    exact abs_le.mpr ⟨by dsimp [τ] at *; linarith only [hib.1,hf.1,hre,hθ],
      by dsimp [τ] at *; linarith only [hib.2,hf.2,hre,hθ]⟩
  · rw [ite_eq_right hp,QueryTree.run]
    have hsmall := abs_le.mp (small_residual v D τ hd0 (fun j => (hmax j).trans (le_of_not_gt hp)) i j)
    unfold cylinderMatrixError
    change _ ≤ 8000*θ
    exact abs_le.mpr ⟨by dsimp [τ] at *; linarith only [hib.1,hsmall.2,hre,hθ],
      by dsimp [τ] at *; linarith only [hib.2,hsmall.1,hre,hθ]⟩

end Cubature
