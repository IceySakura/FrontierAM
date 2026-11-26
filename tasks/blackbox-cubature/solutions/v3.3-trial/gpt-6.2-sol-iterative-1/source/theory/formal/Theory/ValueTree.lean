import Theory.Model

/-! Real-valued decision trees with an explicit count on every transcript. -/

namespace Cubature

inductive ValueTree (X Y : Type*) where
  | ret (result : Y)
  | ask (point : X) (next : ℝ → ValueTree X Y)

namespace ValueTree

def run {X Y : Type*} : ValueTree X Y → (X → ℝ) → Y
  | .ret a, _ => a
  | .ask x f, g => (f (g x)).run g

def within {X Y : Type*} : ValueTree X Y → ℕ → Prop
  | .ret _, _ => True
  | .ask _ _, 0 => False
  | .ask _ f, n+1 => ∀ v, (f v).within n

def admissible {X Y : Type*} (P : X → Prop) : ValueTree X Y → Prop
  | .ret _ => True
  | .ask x f => P x ∧ ∀ v, (f v).admissible P

def bind {X Y Z : Type*} : ValueTree X Y → (Y → ValueTree X Z) → ValueTree X Z
  | .ret a, g => g a
  | .ask x f, g => .ask x (fun r => (f r).bind g)

def map {X Y Z : Type*} (t : ValueTree X Y) (g : Y → Z) : ValueTree X Z :=
  t.bind (fun r => .ret (g r))

theorem run_bind {X Y Z : Type*} (t : ValueTree X Y) (g : Y → ValueTree X Z) (f : X → ℝ) :
    (t.bind g).run f = (g (t.run f)).run f := by
  induction t with
  | ret _ => rfl
  | ask x k ih => exact ih (f x)

theorem run_map {X Y Z : Type*} (t : ValueTree X Y) (g : Y → Z) (f : X → ℝ) :
    (t.map g).run f = g (t.run f) := by
  rw [map,run_bind]
  rfl

theorem within_mono {X Y : Type*} (t : ValueTree X Y) {n m : ℕ}
    (ht : t.within n) (hn : n ≤ m) : t.within m := by
  induction t generalizing n m with
  | ret _ => trivial
  | ask _ f ih =>
    cases n with
    | zero => exact False.elim ht
    | succ n =>
      cases m with
      | zero => omega
      | succ m => exact fun v => ih v (ht v) (Nat.le_of_succ_le_succ hn)

theorem within_bind {X Y Z : Type*} (t : ValueTree X Y) (g : Y → ValueTree X Z)
    {n m : ℕ} (hn : t.within n) (hm : ∀ y, (g y).within m) :
    (t.bind g).within (n+m) := by
  induction t generalizing n with
  | ret y => exact within_mono _ (hm y) (by omega)
  | ask x f ih =>
    cases n with
    | zero => exact False.elim hn
    | succ n =>
      rw [Nat.succ_add]
      exact fun v => ih v (hn v)

theorem within_map {X Y Z : Type*} (t : ValueTree X Y) (g : Y → Z)
    {n : ℕ} (hn : t.within n) : (t.map g).within n := by
  simpa [map] using within_bind t (fun y => .ret (g y)) hn (m := 0) (fun _ => trivial)

theorem admissible_bind {X Y Z : Type*} (t : ValueTree X Y) (g : Y → ValueTree X Z)
    (P : X → Prop) (ht : t.admissible P) (hg : ∀ y, (g y).admissible P) :
    (t.bind g).admissible P := by
  induction t with
  | ret y => exact hg y
  | ask x f ih => exact ⟨ht.1,fun v => ih v (ht.2 v)⟩

theorem admissible_map {X Y Z : Type*} (t : ValueTree X Y) (g : Y → Z)
    (P : X → Prop) (ht : t.admissible P) :
    (t.map g).admissible P :=
  admissible_bind t _ P ht (fun _ => trivial)

noncomputable def collect {X ι : Type*} [DecidableEq ι]
    (t : ι → ValueTree X ℝ) (l : List ι) : ValueTree X (ι → ℝ) :=
  match l with
  | [] => .ret (fun _ => 0)
  | i::xs => (t i).bind (fun v => (collect t xs).map (fun w j => if j=i then v else w j))

theorem run_collect {X ι : Type*} [DecidableEq ι] (t : ι → ValueTree X ℝ)
    (l : List ι) (f : X → ℝ) {i : ι} (hi : i ∈ l) :
    (collect t l).run f i = (t i).run f := by
  induction l with
  | nil => simp at hi
  | cons j l ih =>
    rw [collect,run_bind,run_map]
    by_cases h : i=j
    · simp [h]
    · simpa [h] using ih ((List.mem_cons.mp hi).resolve_left h)

theorem within_collect {X ι : Type*} [DecidableEq ι] (t : ι → ValueTree X ℝ)
    (l : List ι) (n : ℕ) (ht : ∀ i, (t i).within n) :
    (collect t l).within (l.length*n) := by
  induction l with
  | nil => trivial
  | cons j ls ih =>
    rw [collect,List.length_cons,Nat.add_mul,Nat.one_mul,Nat.add_comm]
    exact within_bind _ _ (ht j) (fun _ => within_map _ _ ih)

theorem admissible_collect {X ι : Type*} [DecidableEq ι] (t : ι → ValueTree X ℝ)
    (l : List ι) (P : X → Prop) (ht : ∀ i, (t i).admissible P) :
    (collect t l).admissible P := by
  induction l with
  | nil => trivial
  | cons j ls ih => exact admissible_bind _ _ P (ht j) (fun _ => admissible_map _ _ P ih)

def observe {X : Type*} (x : X) : ValueTree X ℝ := .ask x .ret

/-- Replace each individual real observation by another value-query tree. -/
def replace {X X' Y : Type*} (reader : X → ValueTree X' ℝ) : ValueTree X Y → ValueTree X' Y
  | .ret y => .ret y
  | .ask z k => (reader z).bind (fun v => replace reader (k v))

theorem run_replace {X X' Y : Type*} (t : ValueTree X Y)
    (reader : X → ValueTree X' ℝ) (f : X' → ℝ) (g : X → ℝ)
    (h : ∀ x, (reader x).run f = g x) :
    (replace reader t).run f = t.run g := by
  induction t with
  | ret _ => rfl
  | ask x k ih =>
    rw [replace,run_bind,h,run,ih]

theorem within_replace {X X' Y : Type*} (t : ValueTree X Y)
    (reader : X → ValueTree X' ℝ) {n c : ℕ}
    (ht : t.within n) (hr : ∀ x, (reader x).within c) :
    (replace reader t).within (c*n) := by
  induction t generalizing n with
  | ret _ => trivial
  | ask x k ih =>
    cases n with
    | zero => exact False.elim ht
    | succ n =>
      rw [replace,Nat.mul_succ,Nat.add_comm]
      exact within_bind _ _ (hr x) (fun v => ih v (ht v))

theorem admissible_replace {X X' Y : Type*} (t : ValueTree X Y)
    (reader : X → ValueTree X' ℝ) (P : X → Prop) (P' : X' → Prop)
    (ht : t.admissible P) (hr : ∀ x, P x → (reader x).admissible P') :
    (replace reader t).admissible P' := by
  induction t with
  | ret _ => trivial
  | ask x k ih =>
    exact admissible_bind _ _ P' (hr x ht.1) (fun v => ih v (ht.2 v))

end ValueTree

/-- Lipschitz continuity in the maximum-coordinate norm. -/
def CoordinateLipschitz {d : ℕ} (L : ℝ) (h : Point d → ℝ) : Prop :=
  ∀ (x y : Point d) (t : ℝ), 0 ≤ t → (∀ i, |x i-y i| ≤ t) → |h x-h y| ≤ L*t

end Cubature
