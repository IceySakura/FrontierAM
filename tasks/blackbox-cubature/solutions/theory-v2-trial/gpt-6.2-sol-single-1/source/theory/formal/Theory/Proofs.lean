import Theory.SpliceInput

open Filter MeasureTheory ProbabilityTheory Set
open scoped ContDiff

namespace Cubature

theorem gaussianMixedOrder_sharp (m q : ℕ) (r : Fin m → ℕ)
    (hq : 1 ≤ q) (hr : q < ∑ i, r i) :
    let φ : (Option (Fin m) → ℝ) → ℝ := sharpPhase q
    let θ : (Option (Fin m) → ℝ) → ℝ := fun _ => 1
    let P := gaussianPreintegral φ θ
    ContDiff ℝ (q : ℕ) φ ∧
    (∀ r' : Fin m → ℕ, GaussianMixedInput (fun i : Option (Fin m) => i.elim q r') φ) ∧
    (∀ s : Option (Fin m) → ℕ, GaussianMixedInput s θ) ∧
    (∀ x, 1 ≤ partialsAlong [none] φ x) ∧
    (∀ y, Tendsto (fun t => φ (fiberPoint t y)) atTop atTop) ∧
    (∃ ψ : (Fin m → ℝ) → ℝ, Continuous ψ ∧
      (∀ t y, φ (fiberPoint t y) = 0 ↔ t = ψ y) ∧
      ∀ (w : List (Option (Fin m))) (ws : List (List (Option (Fin m))))
        (_hw : ∀ u ∈ ws, u ≠ [] ∧ u.count none ≤ q) (b c : ℕ),
        Continuous (boundaryFactor φ θ ψ w ws b c) ∧
        ∃ C : ℝ, 0 ≤ C ∧ ∀ y, |boundaryFactor φ θ ψ w ws b c y| ≤ C) ∧
    ContDiff ℝ (q : ℕ) P ∧
    (∀ w : List (Fin m), w.length = q + 1 →
      ¬ ∃ g, HasPartialsAlong w P g ∧ Continuous g) ∧
    ¬ MixedContinuous r P := by
  refine ⟨contDiff_sharpPhase q, gaussianMixedInput_sharpPhase q, gaussianMixedInput_one,
    ?_, tendsto_sharpPhase_fiber q, ?_, contDiff_preintegrated_sharpPhase hq,
    no_continuous_top_partial_sharpPhase hq, not_mixedContinuous_preintegrated_sharpPhase hq r hr⟩
  · intro x
    rw [partialsAlong_sharpPhase_axis hq]
    exact le_add_of_nonneg_right (truncatedPower_nonneg _ _)
  · exact ⟨sharpRoot q, continuous_sharpRoot q, sharpRoot_eq_iff q, boundaryFactor_sharpPhase hq⟩

theorem gaussianMixedSecond_weakL2_sharp :
    ∃ A : ℝ → ℝ,
      (∀ u, A (inverseSplice u) = u ∧ inverseSplice (A u) = u) ∧
      let φ : (Option (Fin 2) → ℝ) → ℝ := phaseFromAxis A
      let θ : (Option (Fin 2) → ℝ) → ℝ := fun _ => 1
      let P := fun y : ℝ × ℝ => gaussianPreintegral φ θ (pairCoordinates y)
      ContDiff ℝ 1 φ ∧
      (∀ r' : Fin 2 → ℕ, GaussianMixedInput (fun i : Option (Fin 2) => i.elim 1 r') φ) ∧
      (∀ s : Option (Fin 2) → ℕ, GaussianMixedInput s θ) ∧
      (∀ x, 1 / 2 ≤ partialsAlong [none] φ x ∧ partialsAlong [none] φ x ≤ 1) ∧
      (∀ y, Tendsto (fun t => φ (fiberPoint t y)) atTop atTop) ∧
      (∃ ψ : (Fin 2 → ℝ) → ℝ, Continuous ψ ∧
        (∀ t y, φ (fiberPoint t y) = 0 ↔ t = ψ y) ∧
        ∀ (w : List (Option (Fin 2))) (ws : List (List (Option (Fin 2))))
          (_hw : ∀ u ∈ ws, u ≠ [] ∧ u.count none ≤ 1) (b c : ℕ),
          Continuous (boundaryFactor φ θ ψ w ws b c) ∧
          ∃ C : ℝ, 0 ≤ C ∧ ∀ y, |boundaryFactor φ θ ψ w ws b c y| ≤ C) ∧
      ContDiff ℝ 1 P ∧
      ∀ G : ℝ × ℝ → ℝ, LocallyIntegrable G volume → WeakMixedSecond P G →
        (¬ IntegrableOn (fun y => G y ^ 2) (Icc (-1) 1 ×ˢ Icc (-1) 1) volume) ∧
        (¬ MemLp G 2 ((gaussianReal 0 1).prod (gaussianReal 0 1))) := by
  refine ⟨spliceAxis, fun u => ⟨spliceAxis_left u, spliceAxis_right u⟩, contDiff_splicePhase,
    gaussianMixedInput_splicePhase, gaussianMixedInput_one, ?_, tendsto_splicePhase_fiber,
    ?_, ?_, ?_⟩
  · intro x
    rw [partialsAlong_splicePhase_axis]
    exact axisSlope_bounds (x none)
  · exact ⟨spliceRoot, continuous_spliceRoot, spliceRoot_eq_iff, boundaryFactor_splicePhase⟩
  · rw [pair_gaussianPreintegral_splicePhase]
    exact contDiff_spliceTail.comp (contDiff_fst.add contDiff_snd)
  · intro G hG hweak
    rw [pair_gaussianPreintegral_splicePhase] at hweak
    have hbox h := no_square_integrable_weakMixedSecond_splice ⟨G, hG, h, hweak⟩
    exact ⟨hbox, fun h => hbox (square_integrable_box_of_gaussian hG h)⟩

end Cubature
