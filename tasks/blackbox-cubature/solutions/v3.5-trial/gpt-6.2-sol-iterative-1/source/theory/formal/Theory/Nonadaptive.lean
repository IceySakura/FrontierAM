import Theory.BallModel
import Mathlib.MeasureTheory.Integral.Bochner.Set

open Set MeasureTheory
open scoped BigOperators Classical

namespace Cubature

noncomputable section

private def hitIndices {d : ℕ} (r : ℝ) (points : List (Vec d)) :
    Finset (Fin points.length) := by
  classical
  exact Finset.univ.filter fun k => points.get k ∈ roundBody d r

private lemma transcript_eq_of_hits_eq {d : ℕ} {r s : ℝ} (points : List (Vec d))
    (h : hitIndices r points = hitIndices s points) :
    sphereTranscript r points = sphereTranscript s points := by
  unfold sphereTranscript
  apply List.map_congr_left
  intro z hz
  obtain ⟨k, rfl⟩ := List.mem_iff_get.mp hz
  have hk : k ∈ hitIndices r points ↔ k ∈ hitIndices s points := h ▸ Iff.rfl
  have he : points.get k ∈ roundBody d r ↔ points.get k ∈ roundBody d s := by
    simpa [hitIndices] using hk
  simp only [he]

lemma nonadaptive_ball_count {d m N : ℕ} {Ω : Type*} (r : Fin m → ℝ)
    (hpos : ∀ i, 0 ≤ r i) (hmono : Monotone r) (ε : ℝ)
    (hsep : ∀ i j, i < j →
      roundMass d (r i) + ε * (roundMass d (r i) + roundMass d (r j)) < roundMass d (r j))
    (queries : Ω → List (Vec d)) (answer : Ω → List Bool → ℝ)
    (ω : Ω) (hN : (queries ω).length ≤ N) :
    ((Finset.univ : Finset (Fin m)).filter
      (fun i => ω ∈ ballSuccess (r i) ε queries answer)).card ≤ N + 1 := by
  classical
  let points := queries ω
  let S := (Finset.univ : Finset (Fin m)).filter (fun i => ω ∈ ballSuccess (r i) ε queries answer)
  let hits (i : Fin m) := hitIndices (r i) points
  have bound (i : Fin m) : (hits i).card < N+1 := by
    apply Nat.lt_succ_of_le
    exact (Finset.card_le_univ (hits i)).trans (by simpa [points] using hN)
  let f (i : Fin m) : Fin (N+1) := ⟨(hits i).card, bound i⟩
  have monotoneHits {i j : Fin m} (hij : i ≤ j) : hits i ⊆ hits j := by
    intro k hk
    simp only [hits, hitIndices, Finset.mem_filter, Finset.mem_univ, true_and] at hk ⊢
    exact roundBody_mono (hpos i) (hmono hij) hk
  have distinctWithOrder {i j : Fin m} (hi : i ∈ S) (hj : j ∈ S) (hm : i < j)
      (eq : f i = f j) : False := by
    have hfin : hits i = hits j := Finset.eq_of_subset_of_card_le
      (monotoneHits hm.le) (le_of_eq (congrArg Fin.val eq).symm)
    have ht := transcript_eq_of_hits_eq points hfin
    have hs₁ : |answer ω (sphereTranscript (r i) points) - roundMass d (r i)| ≤
        ε * roundMass d (r i) := (Finset.mem_filter.mp hi).2
    have hs₂ : |answer ω (sphereTranscript (r j) points) - roundMass d (r j)| ≤
        ε * roundMass d (r j) := (Finset.mem_filter.mp hj).2
    rw [ht] at hs₁
    linarith only [(abs_le.mp hs₁).2, (abs_le.mp hs₂).1, hsep i j hm]
  have hinj : Set.InjOn f (S : Set (Fin m)) := by
    intro i hi j hj he
    rcases lt_trichotomy i j with h | h | h
    · exact (distinctWithOrder hi hj h he).elim
    · exact h
    · exact (distinctWithOrder hj hi h he.symm).elim
  change S.card ≤ N+1
  simpa using Finset.card_le_card_of_injOn f (t := Finset.univ) (by simp [Set.MapsTo]) hinj

lemma nonadaptive_ball_average {d m N : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (r : Fin m → ℝ)
    (hpos : ∀ i, 0 ≤ r i) (hmono : Monotone r) (ε : ℝ)
    (hsep : ∀ i j, i < j →
      roundMass d (r i) + ε * (roundMass d (r i) + roundMass d (r j)) < roundMass d (r j))
    (queries : Ω → List (Vec d)) (answer : Ω → List Bool → ℝ)
    (hN : ∀ ω, (queries ω).length ≤ N)
    (hMeas : ∀ i, MeasurableSet (ballSuccess (r i) ε queries answer)) :
    ∑ i, P.real (ballSuccess (r i) ε queries answer) ≤ (N : ℝ)+1 := by
  classical
  let G (i : Fin m) := ballSuccess (r i) ε queries answer
  let f (i : Fin m) := (G i).indicator (fun _ : Ω => (1 : ℝ))
  have hi (i : Fin m) : Integrable (f i) P := (integrable_const (1 : ℝ)).indicator (hMeas i)
  have hbound (ω : Ω) : (∑ i : Fin m, f i ω) ≤ (N : ℝ)+1 := by
    have eq : (∑ i : Fin m, f i ω) =
        (((Finset.univ : Finset (Fin m)).filter (fun i => ω ∈ G i)).card : ℝ) := by
      simp [f, Set.indicator_apply, Finset.sum_filter]
    rw [eq]
    exact_mod_cast nonadaptive_ball_count r hpos hmono ε hsep queries answer ω (hN ω)
  calc
    (∑ i, P.real (G i)) = ∑ i : Fin m, ∫ ω, f i ω ∂P := by
      simp [f, integral_indicator_one, hMeas, G]
    _ = ∫ ω, ∑ i : Fin m, f i ω ∂P := (integral_finsetSum _ (fun i _ => hi i)).symm
    _ ≤ ∫ _ : Ω, ((N : ℝ)+1) ∂P := by
      exact integral_mono (integrable_finsetSum _ (fun i _ => hi i)) (integrable_const _) hbound
    _ = _ := by simp

lemma packing_is_separated {d m : ℕ} (hm : 0 < m)
    (r : Fin m → ℝ) (hpack : ∀ i,
      roundMass d (r i) = roundMass d 1 * (1/2 + (i.val : ℝ)/(2*m))) :
    ∀ i j, i < j →
      roundMass d (r i) + (1/(8*(m : ℝ))) * (roundMass d (r i) + roundMass d (r j)) <
        roundMass d (r j) := by
  intro i j hij
  have hm0 : 0 < (m : ℝ) := by exact_mod_cast hm
  have hi : (i.val : ℝ) ≤ m := by exact_mod_cast i.isLt.le
  have hj : (j.val : ℝ) ≤ m := by exact_mod_cast j.isLt.le
  have hd : (i.val : ℝ) + 1 ≤ j.val := by exact_mod_cast Nat.succ_le_of_lt hij
  have hp := roundMass_one_pos d
  rw [hpack i, hpack j]
  have hstep : (1/2 + (i.val : ℝ)/(2*m)) +
      1/(8*(m : ℝ)) * ((1/2 + (i.val : ℝ)/(2*m)) + (1/2 + (j.val : ℝ)/(2*m))) <
      (1/2 + (j.val : ℝ)/(2*m)) := by
    have h1 : (i.val : ℝ)/(2*m) ≤ 1/2 := (div_le_iff₀ (by positivity)).2 (by linarith)
    have h2 : (j.val : ℝ)/(2*m) ≤ 1/2 := (div_le_iff₀ (by positivity)).2 (by linarith)
    have h3 : (1 : ℝ)/(2*m) ≤ (j.val : ℝ)/(2*m)-(i.val : ℝ)/(2*m) := by
      rw [← sub_div]
      exact div_le_div_of_nonneg_right (by linarith only [hd]) (by positivity)
    have hsum : (1/2 + (i.val : ℝ)/(2*m)) + (1/2 + (j.val : ℝ)/(2*m)) ≤ 2 := by
      linarith only [h1,h2]
    have h4 : (1/(8*(m : ℝ))) *
        ((1/2 + (i.val : ℝ)/(2*m)) + (1/2 + (j.val : ℝ)/(2*m))) ≤ 1/(4*(m : ℝ)) :=
      (mul_le_mul_of_nonneg_left hsum (show 0 ≤ 1/(8*(m : ℝ)) by positivity)).trans_eq
        (by field_simp; ring)
    have h5 : (1 : ℝ)/(4*m) < 1/(2*m) :=
      div_lt_div_of_pos_left (by positivity) (by positivity) (by linarith)
    linarith only [h3,h4,h5]
  convert mul_lt_mul_of_pos_left hstep hp using 1 <;> ring

end

end Cubature
