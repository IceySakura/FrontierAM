import Theory.Balls
import Theory.Tree

open MeasureTheory Set
open scoped BigOperators ENNReal

namespace Cubature

/-- A grid of spherical inputs with integrals equally spaced across the mass gap. -/
theorem massGrid {d M : ℕ} (hd : 0 < d) {ρ : ℝ} (hρ : 0 < ρ) (hM : 0 < M) :
    ∃ r : Fin (M+1) → ℝ,
      (∀ i, r i ∈ Icc (3*ρ/4) ρ) ∧ StrictMono r ∧
      ∀ i, ballMass d (r i) = ballMass d (3*ρ/4) + (i.val : ℝ) * (massGap d ρ / M) := by
  have ha : 0 < 3*ρ/4 := by positivity
  have hab : 3*ρ/4 ≤ ρ := by linarith
  have hgap := massGap_pos hd hρ
  have hM' : 0 < (M : ℝ) := by exact_mod_cast hM
  let w := massGap d ρ / (M : ℝ)
  have hw : 0 < w := div_pos hgap hM'
  have hwM : (M : ℝ)*w = massGap d ρ := by dsimp [w]; field_simp
  have Hex (i : Fin (M+1)) :
      ∃ s ∈ Icc (3*ρ/4) ρ, ballMass d s = ballMass d (3*ρ/4) + (i.val : ℝ)*w := by
    have hi : (i.val : ℝ) ≤ M := by exact_mod_cast (Nat.le_of_lt_succ i.isLt)
    refine ballMass_every_level ha hab ⟨?_,?_⟩
    · exact le_add_of_nonneg_right (mul_nonneg (by positivity) hw.le)
    · have hs := mul_le_mul_of_nonneg_right hi hw.le
      rw [hwM] at hs
      unfold massGap at hs
      exact (le_sub_iff_add_le'.mp hs)
  choose r hr hrm using Hex
  refine ⟨r, hr, ?_, hrm⟩
  intro i j hij
  have hval : (i.val : ℝ) < j.val := by exact_mod_cast hij
  have hlt : ballMass d (r i) < ballMass d (r j) := by
    rw [hrm,hrm]
    gcongr
  exact lt_of_not_ge (fun h ↦ not_lt_of_ge
    (ballMass_mono (ha.le.trans (hr j).1) h) hlt)

theorem same_bits_if_no_cut {d n : ℕ} {r s : ℝ} (hr : 0 ≤ r) (hrs : r ≤ s)
    (nodes : Fin n → Point d) (h : ∀ j, normSq (nodes j) ∉ Ico (r^2) (s^2)) :
    (fun j ↦ answerBit (arithmeticBall r) (nodes j)) =
    (fun j ↦ answerBit (arithmeticBall s) (nodes j)) := by
  funext j
  have hiff : nodes j ∈ arithmeticBall r ↔ nodes j ∈ arithmeticBall s := by
    constructor
    · intro hh
      exact arithmeticBall_mono hr hrs hh
    · intro hj
      by_contra hj'
      exact h j ⟨le_of_not_gt hj',hj⟩
  cases h₁ : answerBit (arithmeticBall r) (nodes j) <;>
    cases h₂ : answerBit (arithmeticBall s) (nodes j)
  · rfl
  · exact False.elim (Bool.false_ne_true (h₁.symm.trans ((answerBit_iff _ _).2
      (hiff.mpr ((answerBit_iff _ _).1 h₂)))))
  · exact False.elim (Bool.false_ne_true (h₂.symm.trans ((answerBit_iff _ _).2
      (hiff.mp ((answerBit_iff _ _).1 h₁)))))
  · rfl

theorem abs_two_targets {a b c w : ℝ} (h : b-a = w) :
    w/2 ≤ |c-a| ∨ w/2 ≤ |c-b| := by
  have H : w ≤ |c-a|+|c-b| := by
    rw [← h]
    calc
      b-a ≤ |b-a| := le_abs_self _
      _ = |(c-a)-(c-b)| := by ring_nf
      _ ≤ _ := by
        simpa only [← sub_eq_add_neg, abs_neg] using abs_add_le (c-a) (-(c-b))
  by_contra! hn
  linarith [hn.1,hn.2]

/-- The lower bound is for arbitrary nodes and arbitrary reconstruction maps. -/
theorem deterministic_nonadaptive_lower {d : ℕ} (hd : 0 < d) {ρ : ℝ} (hρ : 0 < ρ)
    (n : ℕ) (nodes : Fin n → Point d) (output : (Fin n → Bool) → ℝ) :
    ∃ E : Ellipsoid d,
      massGap d ρ / (2 * (n+1 : ℝ)) ≤
        |output (fun j ↦ answerBit (region ρ E) (nodes j))-probability (region ρ E)| := by
  classical
  obtain ⟨r, hr, hmono, hm⟩ := massGrid (M := n+1) hd hρ (by omega)
  have hpos (i : Fin (n+2)) : 0 ≤ r i := (by positivity : 0 ≤ 3*ρ/4).trans (hr i).1
  let cuts (i : Fin (n+1)) := Ico ((r i.castSucc)^2) ((r i.succ)^2)
  have hdisj (i k : Fin (n+1)) (h : i < k) :
      ∀ x : ℝ, x ∈ cuts i → x ∈ cuts k → False := by
    have hik : i.succ ≤ k.castSucc := by
      change i.val+1 ≤ k.val
      omega
    have hsq := pow_le_pow_left₀ (hpos i.succ) (hmono.monotone hik) 2
    intro x h1 h2
    exact not_lt_of_ge (hsq.trans h2.1) h1.2
  have hmissing : ∃ i : Fin (n+1), ∀ j, normSq (nodes j) ∉ cuts i := by
    by_contra! hh
    choose g hg using hh
    have hinj : Function.Injective g := by
      intro i j he
      rcases lt_trichotomy i j with h | h | h
      · exact False.elim (hdisj i j h _ (hg i) (he ▸ hg j))
      · exact h
      · exact False.elim (hdisj j i h _ (hg j) (he ▸ hg i))
    have := Fintype.card_le_of_injective _ hinj
    simp only [Fintype.card_fin] at this
    omega
  obtain ⟨i,hi⟩ := hmissing
  let E₀ := ballEllipsoid d hρ (hr i.castSucc)
  let E₁ := ballEllipsoid d hρ (hr i.succ)
  have h0 : region ρ E₀ = arithmeticBall (r i.castSucc) := region_ballEllipsoid _ hρ _
  have h1 : region ρ E₁ = arithmeticBall (r i.succ) := region_ballEllipsoid _ hρ _
  have hge : r i.castSucc ≤ r i.succ := hmono.monotone (by exact Nat.le_succ _)
  have heq := same_bits_if_no_cut (hpos i.castSucc) hge nodes hi
  have hgap : ballMass d (r i.succ)-ballMass d (r i.castSucc) = massGap d ρ / (n+1 : ℝ) := by
    rw [hm,hm]
    simp only [Fin.val_succ, Fin.val_castSucc, Nat.cast_succ]
    ring
  rcases abs_two_targets (c := output (fun j ↦ answerBit (arithmeticBall (r i.castSucc)) (nodes j)))
    hgap with h | h
  · refine ⟨E₀,?_⟩
    rw [h0]
    have H : massGap d ρ / (2*(n+1 : ℝ)) = (massGap d ρ / (n+1 : ℝ))/2 := by
      rw [div_div, mul_comm 2]
    rw [H]
    exact h
  · refine ⟨E₁,?_⟩
    rw [h1,← heq]
    have H : massGap d ρ / (2*(n+1 : ℝ)) = (massGap d ρ / (n+1 : ℝ))/2 := by
      rw [div_div, mul_comm 2]
    rw [H]
    exact h

end Cubature
