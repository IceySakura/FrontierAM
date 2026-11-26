import Theory.Model

open Set
open scoped BigOperators Pointwise

namespace Cubature

namespace QueryTree

theorem run_bind {X Y Z : Type*} (t : QueryTree X Y) (g : Y → QueryTree X Z) (f : X → Bool) :
    (t.bind g).run f = (g (t.run f)).run f := by
  induction t with
  | ret y => rfl
  | ask x l r hl hr =>
    cases h : f x <;> simp [bind, run, h, hl, hr]

theorem depth_bind_le {X Y Z : Type*} (t : QueryTree X Y) (g : Y → QueryTree X Z) (n : ℕ)
    (hg : ∀ y, (g y).depth ≤ n) :
    (t.bind g).depth ≤ t.depth + n := by
  induction t with
  | ret y => simpa [bind, depth] using hg y
  | ask x l r hl hr =>
    simp only [bind, depth]
    omega

theorem admissible_bind {X Y Z : Type*} (t : QueryTree X Y) (g : Y → QueryTree X Z) (P : X → Prop)
    (ht : t.admissible P) (hg : ∀ y, (g y).admissible P) :
    (t.bind g).admissible P := by
  induction t with
  | ret y => exact hg y
  | ask x l r hl hr => exact ⟨ht.1, hl ht.2.1, hr ht.2.2⟩

theorem run_map {X Y Z : Type*} (t : QueryTree X Y) (g : Y → Z) (f : X → Bool) :
    (t.map g).run f = g (t.run f) := by
  rw [map, run_bind]
  rfl

theorem depth_map_le {X Y Z : Type*} (t : QueryTree X Y) (g : Y → Z) :
    (t.map g).depth ≤ t.depth := by
  simpa [map] using depth_bind_le t (fun y => .ret (g y)) 0 (fun _ => le_refl _)

theorem admissible_map {X Y Z : Type*} (t : QueryTree X Y) (g : Y → Z) (P : X → Prop)
    (ht : t.admissible P) :
    (t.map g).admissible P :=
  admissible_bind t _ P ht fun _ => trivial

/-- Execute the trees in `L` and store their outputs, indexed by `L`. -/
noncomputable def collect {X ι : Type*} [DecidableEq ι]
    (t : ι → QueryTree X ℝ) (L : List ι) : QueryTree X (ι → ℝ) :=
  match L with
  | [] => .ret (fun _ => 0)
  | i :: l => (t i).bind (fun v => (collect t l).map (fun w j => if j = i then v else w j))

theorem run_collect {X ι : Type*} [DecidableEq ι] (t : ι → QueryTree X ℝ)
    (L : List ι) (f : X → Bool) {i : ι} (hi : i ∈ L) :
    (collect t L).run f i = (t i).run f := by
  induction L with
  | nil => simp at hi
  | cons j l ih =>
    rw [collect, run_bind, run_map]
    by_cases h : i = j
    · simp [h]
    · simpa [h] using ih ((List.mem_cons.mp hi).resolve_left h)

theorem depth_collect {X ι : Type*} [DecidableEq ι] (t : ι → QueryTree X ℝ)
    (L : List ι) (n : ℕ) (ht : ∀ i, (t i).depth ≤ n) :
    (collect t L).depth ≤ L.length * n := by
  induction L with
  | nil => simp [collect, depth]
  | cons j l ih =>
    rw [collect]
    refine (depth_bind_le _ _ (l.length * n)
      (fun _ => (depth_map_le _ _).trans ih)).trans ?_
    rw [List.length_cons, Nat.add_mul, Nat.one_mul, Nat.add_comm]
    exact Nat.add_le_add_left (ht j) _

theorem admissible_collect {X ι : Type*} [DecidableEq ι] (t : ι → QueryTree X ℝ)
    (L : List ι) (P : X → Prop) (ht : ∀ i, (t i).admissible P) :
    (collect t L).admissible P := by
  induction L with
  | nil => trivial
  | cons j l ih =>
    exact admissible_bind _ _ P (ht j) (fun _ => admissible_map _ _ P ih)

end QueryTree

noncomputable def bisect {d : ℕ} (ρ : ℝ) (u : Point d) :
    ℕ → ℝ → ℝ → QueryTree (Point d) ℝ
  | 0, l, h => .ret ((l+h)/2)
  | m+1, l, h =>
    let a := (l+h)/2
    .ask ((ρ/a) • u) (bisect ρ u m a h) (bisect ρ u m l a)

theorem bisect_depth {d : ℕ} (ρ : ℝ) (u : Point d) (m : ℕ) (l h : ℝ) :
    (bisect ρ u m l h).depth = m := by
  induction m generalizing l h with
  | zero => rfl
  | succ m ih => simp [bisect, QueryTree.depth, ih, Nat.add_comm]

theorem bisect_admissible {d : ℕ} {ρ : ℝ} {u : Point d}
    (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20) (hu : ∀ i, |u i| ≤ 1)
    (m : ℕ) {l h : ℝ} (hl : 1/2 ≤ l) (hlh : l ≤ h) :
    (bisect ρ u m l h).admissible allowed := by
  induction m generalizing l h with
  | zero => trivial
  | succ m ih =>
    have h1 : l ≤ (l+h)/2 := by linarith
    have h2 : (l+h)/2 ≤ h := by linarith
    refine ⟨?_, ih (hl.trans h1) h2, ih hl h1⟩
    intro i
    have ha : 1/2 ≤ (l+h)/2 := hl.trans h1
    have ha' : 0 < (l+h)/2 := by linarith
    simp only [Pi.smul_apply, smul_eq_mul, abs_mul,
      abs_of_nonneg (div_nonneg hρ ha'.le)]
    calc
      ρ/((l+h)/2)*|u i| ≤ ρ/((l+h)/2)*1 :=
        mul_le_mul_of_nonneg_left (hu i) (div_nonneg hρ ha'.le)
      _ ≤ 40 := by rw [mul_one, div_le_iff₀ ha']; linarith

theorem bisect_correct {d : ℕ} (ρ : ℝ) (u : Point d) (f : Point d → Bool) (r : ℝ)
    (hf : ∀ t > (0 : ℝ), f ((ρ/t) • u) = true ↔ r < t)
    (m : ℕ) {l h : ℝ} (hl : 0 < l) (hlr : l ≤ r) (hrh : r ≤ h) :
    (bisect ρ u m l h).run f ∈ Icc l h ∧
    |(bisect ρ u m l h).run f - r| ≤ (h-l)/(2:ℝ)^(m+1) := by
  induction m generalizing l h with
  | zero =>
    simp only [bisect, QueryTree.run, mem_Icc, zero_add, pow_one]
    refine ⟨⟨by linarith, by linarith⟩, abs_le.mpr ⟨by linarith, by linarith⟩⟩
  | succ m ih =>
    let a := (l+h)/2
    have hal : l ≤ a := by dsimp [a]; linarith
    have hah : a ≤ h := by dsimp [a]; linarith
    have h0 : 0 < a := by dsimp [a]; linarith
    by_cases ha : r < a
    · have hh : f ((ρ/a) • u) = true := (hf a h0).2 ha
      have H := ih hl hlr ha.le
      change (if f ((ρ/a) • u) then (bisect ρ u m l a).run f
        else (bisect ρ u m a h).run f) ∈ Icc l h ∧
        |(if f ((ρ/a) • u) then (bisect ρ u m l a).run f
          else (bisect ρ u m a h).run f) - r| ≤ (h-l)/(2:ℝ)^(m+1+1)
      simp only [hh, ite_true, mem_Icc] at *
      refine ⟨⟨H.1.1, H.1.2.trans hah⟩, H.2.trans_eq ?_⟩
      rw [pow_succ]
      dsimp [a]
      ring
    · have hh : f ((ρ/a) • u) = false := by
        cases h' : f ((ρ/a) • u)
        · rfl
        · exact False.elim (ha ((hf a h0).1 h'))
      have H := ih h0 (le_of_not_gt ha) hrh
      change (if f ((ρ/a) • u) then (bisect ρ u m l a).run f
        else (bisect ρ u m a h).run f) ∈ Icc l h ∧
        |(if f ((ρ/a) • u) then (bisect ρ u m l a).run f
          else (bisect ρ u m a h).run f) - r| ≤ (h-l)/(2:ℝ)^(m+1+1)
      simp only [hh, Bool.false_eq_true, ite_false, mem_Icc] at *
      refine ⟨⟨hal.trans H.1.1, H.1.2⟩, H.2.trans_eq ?_⟩
      rw [pow_succ]
      dsimp [a]
      ring

end Cubature
