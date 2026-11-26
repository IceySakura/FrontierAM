import Theory.RankAdaptiveModel
import Theory.RankCapRecovery

open MeasureTheory Set Module
open scoped BigOperators Pointwise RealInnerProductSpace

namespace Cubature

namespace ValueTree

theorem trace_bind {X Y Z : Type*} (T : ValueTree X Y) (next : Y → ValueTree X Z)
    (f : X → ℝ) : (T.bind next).trace f = T.trace f ++ (next (T.run f)).trace f := by
  induction T with
  | ret _ => rfl
  | ask x k ih => simp [bind,trace,run,ih]

theorem trace_map {X Y Z : Type*} (T : ValueTree X Y) (g : Y → Z) (f : X → ℝ) :
    (T.map g).trace f = T.trace f := by rw [map,trace_bind]; simp [trace]

end ValueTree

theorem gramProbe_exact {d k : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : GramCap d k)
    (v : Point d) (hv : normSq v≤2) :
    (rankProbe ρ v).run (gramCapPayoff ρ G) = quadratic (gramRows G.rows) v := by
  have hq : quadratic (gramRows G.rows) v≤4 := by linarith [G.upper v]
  have hh : (ρ⁻¹ : ℝ)*(ρ/4)=1/4 := by field_simp
  rw [rankProbe,ValueTree.run_map]
  change 16*(1-max (1-quadratic (gramRows G.rows) (ρ⁻¹ • ((ρ/4) • v))) 0)=_
  rw [smul_smul,hh,quadratic_smul,max_eq_left (by nlinarith)]
  ring

theorem gramDiagonal_exact {d k : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : GramCap d k) :
    (rankDiagonal ρ).run (gramCapPayoff ρ G) = fun i => gramRows G.rows i i := by
  funext i
  rw [rankDiagonal,ValueTree.run_collect _ _ _ (by simp),
    gramProbe_exact hρ G _ (by rw [normSq_axis];norm_num),quadratic_axis]

theorem gramRow_exact {d k : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : GramCap d k) (i : Fin d) :
    (rankRow ρ (fun j => gramRows G.rows j j) i).run (gramCapPayoff ρ G) = gramRows G.rows i := by
  funext j
  rw [rankRow,ValueTree.run_collect _ _ _ (by simp)]
  by_cases he : i=j
  · subst j; simp [ValueTree.run]
  · rw [ite_eq_right he, ValueTree.run_map,gramProbe_exact hρ _ _ (ray_norm _).2,mixed_ray he]
    have hh : quadratic (gramRows G.rows) (axis i+axis j) =
        gramRows G.rows i i+gramRows G.rows j j+2*gramRows G.rows i j := by
      simp only [quadratic,Pi.add_apply,mul_add,add_mul,Finset.sum_add_distrib]
      simp only [axis,mul_ite,mul_one,mul_zero,Finset.sum_ite_eq',Finset.mem_univ,ite_true]
      rw [show gramRows G.rows j i=gramRows G.rows i j from real_inner_comm _ _]
      ring
    rw [hh]
    ring

section Elimination
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable {d : ℕ}

noncomputable def dropPivot (v : Fin d → E) (p : Fin d) : Fin d → E :=
  fun i => v i-(⟪v p,v i⟫/⟪v p,v p⟫ : ℝ) • v p

theorem pivot_orthogonal (v : Fin d → E) (p : Fin d)
    (hp : ⟪v p,v p⟫≠0) (i : Fin d) :
    ⟪v p,dropPivot v p i⟫=0 := by
  simp only [dropPivot,inner_sub_right,real_inner_smul_right]
  rw [div_mul_cancel₀ _ hp,sub_self]

theorem gram_drop (v : Fin d → E) (p : Fin d) (hp : ⟪v p,v p⟫≠0) :
    gramRows v = rankOnePart (gramRows v p) (gramRows v p p)+gramRows (dropPivot v p) := by
  funext i j
  simp only [gramRows,rankOnePart,dropPivot,Pi.add_apply,inner_sub_left,inner_sub_right,
    real_inner_smul_left,real_inner_smul_right,real_inner_comm (v i) (v p)]
  field_simp [hp]
  <;> ring

abbrev rowSpan (v : Fin d → E) : Submodule ℝ E :=
  Submodule.span ℝ (Set.range v)

theorem pivot_rank_decreases (v : Fin d → E) (p : Fin d) (hp : ⟪v p,v p⟫≠0) :
    finrank ℝ (rowSpan (dropPivot v p)) < finrank ℝ (rowSpan v) := by
  let S := rowSpan v
  let U := rowSpan (dropPivot v p)
  have hin (i : Fin d) : v i∈S := Submodule.subset_span (mem_range_self i)
  have sub : U ≤ S := by
    apply Submodule.span_le.mpr
    rintro _ ⟨j,rfl⟩
    exact S.sub_mem (hin j) (S.smul_mem _ (hin p))
  let L := (innerSL ℝ (v p)).toLinearMap
  have orth : U ≤ LinearMap.ker L := by
    apply Submodule.span_le.mpr
    rintro _ ⟨j,rfl⟩
    exact pivot_orthogonal v p hp j
  have hn : U ≠ S := by
    intro he
    have hmem : v p∈U := he.symm ▸ hin p
    exact hp (orth hmem)
  exact Submodule.finrank_lt_finrank_of_lt (lt_of_le_of_ne sub hn)

theorem gram_zero_diag (v : Fin d → E) (hz : ∀ i, gramRows v i i≤0) : gramRows v=0 := by
  have h (i : Fin d) : v i=0 := by
    have he : ⟪v i,v i⟫=0 := le_antisymm (hz i) real_inner_self_nonneg
    exact inner_self_eq_zero.mp he
  funext i j
  simp [gramRows,h i]

end Elimination

noncomputable def pivotTree {d : ℕ} (ρ : ℝ) (D : Point d) :
    ℕ → Coefficients d → ValueTree (Point d) (Coefficients d)
  | 0, B => .ret B
  | m+1, B =>
    if h : ∃ i, 0<D i-B i i then
      let i := Classical.choose h
      (rankRow ρ D i).bind (fun a => pivotTree ρ D m
        (B+rankOnePart (a-B i) (D i-B i i)))
    else .ret B

theorem pivotTree_within {d : ℕ} (ρ : ℝ) (D : Point d) (m : ℕ) (B : Coefficients d) :
    (pivotTree ρ D m B).within (d*m) := by
  induction m generalizing B with
  | zero => trivial
  | succ m ih =>
    rw [pivotTree,Nat.mul_succ,Nat.add_comm]
    split
    · exact ValueTree.within_bind _ _ (rankRow_within _ _ _) (fun _ => ih _)
    · trivial

theorem pivotTree_allowed {d : ℕ} {ρ : ℝ} (hρ : 0≤ρ) (hb : ρ≤20)
    (D : Point d) (m : ℕ) (B : Coefficients d) : (pivotTree ρ D m B).admissible allowed := by
  induction m generalizing B with
  | zero => trivial
  | succ m ih =>
    rw [pivotTree]
    split
    · exact ValueTree.admissible_bind _ _ _ (rankRow_allowed hρ hb _ _) (fun _ => ih _)
    · trivial

theorem pivotTree_correct {d k : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : GramCap d k)
    (m : ℕ) (B : Coefficients d) (v : Fin d → EuclideanSpace ℝ (Fin k))
    (he : gramRows G.rows=B+gramRows v) (hr : finrank ℝ (rowSpan v)≤m) :
    let T := pivotTree ρ (fun j => gramRows G.rows j j) m B
    T.run (gramCapPayoff ρ G)=gramRows G.rows ∧
      (T.trace (gramCapPayoff ρ G)).length≤d*finrank ℝ (rowSpan v) := by
  induction m generalizing B v with
  | zero =>
    have h (i : Fin d) : gramRows v i i≤0 := by
      by_contra! H
      have h := pivot_rank_decreases v i (ne_of_gt H)
      omega
    simp [pivotTree,ValueTree.run,ValueTree.trace,he,gram_zero_diag v h]
  | succ m ih =>
    dsimp only
    rw [pivotTree]
    split
    · rename_i hp
      let p := Classical.choose hp
      have hp' : 0<gramRows G.rows p p-B p p := Classical.choose_spec hp
      have subeq (i j : Fin d) : gramRows G.rows i j-B i j=gramRows v i j := by
        rw [he,Pi.add_apply,Pi.add_apply,add_sub_cancel_left]
      have hvp : ⟪v p,v p⟫≠0 := by
        rw [subeq] at hp'
        exact hp'.ne'
      let W := dropPivot v p
      let R := rankRow ρ (fun j => gramRows G.rows j j) p
      let B' := B+rankOnePart (gramRows G.rows p-B p) (gramRows G.rows p p-B p p)
      have hd : finrank ℝ (rowSpan W)<finrank ℝ (rowSpan v) := pivot_rank_decreases v p hvp
      have hE : gramRows G.rows=B'+gramRows W := by
        have hrow : gramRows G.rows p-B p=gramRows v p := funext (subeq p)
        dsimp [B',W]
        rw [hrow,subeq,he]
        conv_lhs => rw [gram_drop v p hvp]
        rw [add_assoc]
      have hi := ih B' W hE (by omega)
      have hrn : R.run (gramCapPayoff ρ G)=gramRows G.rows p := gramRow_exact hρ G p
      refine ⟨?_,?_⟩
      · rw [ValueTree.run_bind,gramRow_exact hρ G]
        exact hi.1
      · rw [ValueTree.trace_bind,List.length_append,gramRow_exact hρ G]
        have hn := ValueTree.trace_length R (gramCapPayoff ρ G) (rankRow_within _ _ _)
        calc
          _ ≤ d+d*finrank ℝ (rowSpan W) := Nat.add_le_add hn hi.2
          _ = d*(finrank ℝ (rowSpan W)+1) := by rw [Nat.mul_succ,Nat.add_comm]
          _ ≤ d*finrank ℝ (rowSpan v) := Nat.mul_le_mul_left d (Nat.succ_le_of_lt hd)
    · rename_i hz
      have H (j : Fin d) : gramRows v j j≤0 := by
        have h := le_of_not_gt (fun H => hz ⟨j,H⟩)
        rw [he,Pi.add_apply,Pi.add_apply,add_sub_cancel_left] at h
        exact h
      simp [ValueTree.run,ValueTree.trace,he,gram_zero_diag v H]

noncomputable def rankAdaptiveCubature (d : ℕ) (ρ : ℝ) :
    ValueTree (Point d) (Coefficients d × ℝ) :=
  (rankDiagonal ρ).bind (fun D =>
    (pivotTree ρ D d 0).map (fun A => (A,∫ x,matrixCap ρ A x ∂(gaussian d))))

theorem rankAdaptive_within (d : ℕ) (ρ : ℝ) :
    (rankAdaptiveCubature d ρ).within (d*(d+1)) := by
  unfold rankAdaptiveCubature
  rw [Nat.mul_succ,Nat.add_comm]
  exact ValueTree.within_bind _ _ (rankDiagonal_within _) (fun _ =>
    ValueTree.within_map _ _ (pivotTree_within _ _ _ _))

theorem rankAdaptive_allowed (d : ℕ) {ρ : ℝ} (hρ : 0≤ρ) (hb : ρ≤20) :
    (rankAdaptiveCubature d ρ).admissible allowed :=
  ValueTree.admissible_bind _ _ _ (rankDiagonal_allowed hρ hb) (fun _ =>
    ValueTree.admissible_map _ _ _ (pivotTree_allowed hρ hb _ _ _))

theorem rankAdaptive_exact {d k : ℕ} {ρ : ℝ} (hρ : 0<ρ) (G : GramCap d k) :
    (rankAdaptiveCubature d ρ).run (gramCapPayoff ρ G) =
      (gramRows G.rows,gramCapIntegral ρ G) ∧
    ((rankAdaptiveCubature d ρ).trace (gramCapPayoff ρ G)).length≤d*(finrank ℝ (rowSpan G.rows)+1) := by
  have hf : finrank ℝ (rowSpan G.rows)≤d := by
    simpa [Set.finrank] using finrank_range_le_card (R := ℝ) G.rows
  have H := pivotTree_correct hρ G d 0 G.rows (by simp) hf
  constructor
  · rw [rankAdaptiveCubature,ValueTree.run_bind,gramDiagonal_exact hρ G,
      ValueTree.run_map,H.1]
    rfl
  · rw [rankAdaptiveCubature,ValueTree.trace_bind,List.length_append,gramDiagonal_exact hρ G,
      ValueTree.trace_map]
    have h := ValueTree.trace_length (rankDiagonal ρ) (gramCapPayoff ρ G) (rankDiagonal_within ρ)
    calc
      _ ≤ d+d*finrank ℝ (rowSpan G.rows) := Nat.add_le_add h H.2
      _ = d*(finrank ℝ (rowSpan G.rows)+1) := by rw [Nat.mul_succ,Nat.add_comm]

end Cubature
