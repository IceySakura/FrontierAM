import Theory.KinkGap

open MeasureTheory Set
open scoped Pointwise

namespace Cubature

theorem uniform_kink_lower (d : ℕ) (hd : 0 < d) (ρ : ℝ) (hρ : 0 < ρ) :
    ∃ ε : ℝ, 0 < ε ∧
      ∀ (n : ℕ), 2*n < d*(d+1) →
        ∀ T : ValueTree (Point d) ℝ, T.within n →
          ∃ E : Ellipsoid d,
            ε ≤ |T.run (ellipsoidalKink ρ E)-kinkIntegral ρ E| := by
  have hd0 : 0<(d:ℝ) := by exact_mod_cast hd
  let i : Fin d := ⟨0,hd⟩
  obtain ⟨j,_,hj⟩ := Finset.exists_min_image (Finset.univ : Finset (RayIndex d))
    (fun k => probability (kinkWindow ρ k)) ⟨rayIndex i i,Finset.mem_univ _⟩
  let η := (1/(100*(d:ℝ)))*probability (kinkWindow ρ j)
  have heta : 0 < η := mul_pos (by positivity) (kinkWindow_pos hd hρ j)
  refine ⟨η,heta,?_⟩
  intro n hn T hT
  let E₀ := centralEllipsoid d
  let f₀ := ellipsoidalKink ρ E₀
  let L := T.trace f₀
  have hL : L.length ≤ n := ValueTree.trace_length T f₀ hT
  have hlen : 2*L.length < d*(d+1) := by omega
  obtain ⟨H,hs,hb,hentry,hzero⟩ := invisible_symmetric hlen (fun i : Fin L.length => ρ⁻¹ • L.get i)
  obtain ⟨k,hk⟩ := witness_ray hs hentry
  let t := 1/(4*(d:ℝ))
  have ht0 : 0 ≤ t := by dsimp [t]; positivity
  have ht : |t| * (d:ℝ)≤1/2 := by
    rw [abs_of_nonneg ht0]
    have h : t*(d:ℝ) = 1/4 := by dsimp [t]; field_simp
    rw [h]
    norm_num
  have hneg : |-t| * (d:ℝ) ≤ 1/2 := by simpa using ht
  let P := perturbEllipsoid H hs hb t ht
  let M := perturbEllipsoid H hs hb (-t) hneg
  have hh (x : Point d) (hx : x ∈ L) : quadratic H (ρ⁻¹ • x) = 0 := by
    obtain ⟨i,hi⟩ := List.get_of_mem hx
    rw [← hi]
    exact hzero i
  have hagree (a : ℝ) (ha : |a| * (d:ℝ)≤1/2) (x : Point d) (hx : x ∈ L) :
      ellipsoidalKink ρ (perturbEllipsoid H hs hb a ha) x = f₀ x := by
    unfold ellipsoidalKink
    have h' : quadratic (perturbEllipsoid H hs hb a ha).A (ρ⁻¹ • x) =
        (3/2)*normSq (ρ⁻¹ • x)+a*quadratic H (ρ⁻¹ • x) := quadratic_perturbed_value H a _
    rw [h',hh x hx,mul_zero,add_zero]
    simp [f₀,ellipsoidalKink,E₀,central_quad]
  have hP : T.run (ellipsoidalKink ρ P) = T.run f₀ :=
    ValueTree.run_eq_of_agree T f₀ _ (hagree t ht)
  have hM : T.run (ellipsoidalKink ρ M) = T.run f₀ :=
    ValueTree.run_eq_of_agree T f₀ _ (hagree (-t) hneg)
  have hg : 2*η ≤ (kinkIntegral ρ P+kinkIntegral ρ M)/2-kinkIntegral ρ E₀ := by
    have he : 2*η = (1/(50*(d:ℝ)))*probability (kinkWindow ρ j) := by
      dsimp [η]
      field_simp
      <;> ring
    rw [he]
    exact (mul_le_mul_of_nonneg_left (hj k (Finset.mem_univ _)) (by positivity)).trans
      (kink_gap_integral hd ρ H hs hb k hk t rfl ht hneg)
  by_contra! hz
  have hp := abs_lt.mp (hz P)
  have hm := abs_lt.mp (hz M)
  have h0 := abs_lt.mp (hz E₀)
  rw [hP] at hp
  rw [hM] at hm
  change -η < T.run f₀-kinkIntegral ρ E₀ ∧ T.run f₀-kinkIntegral ρ E₀ < η at h0
  linarith [hp.1,hm.1,h0.2,hg]

end Cubature
