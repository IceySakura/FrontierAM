import Mathlib.Tactic

/-! Finite Boolean-oracle decision trees and a batched version of bisection. -/

namespace Cubature

/-- A strategy can look at the answer to a point query before choosing its next point. -/
inductive QueryTree (Q : Type*) (A : Type*) where
  | returnValue (a : A)
  | ask (point : Q) (next : Bool → QueryTree Q A)

namespace QueryTree

variable {Q A B : Type*}

def run (oracle : Q → Bool) : QueryTree Q A → A
  | .returnValue a => a
  | .ask q t => run oracle (t (oracle q))

/-- Worst-case number of calls, including calls on branches not taken by a given input. -/
def depth : QueryTree Q A → ℕ
  | .returnValue _ => 0
  | .ask _ t => max (depth (t false)) (depth (t true)) + 1

/-- Every query in every branch belongs to the permitted set. -/
def queriesIn (s : Set Q) : QueryTree Q A → Prop
  | .returnValue _ => True
  | .ask q t => q ∈ s ∧ ∀ b, queriesIn s (t b)

def bind : QueryTree Q A → (A → QueryTree Q B) → QueryTree Q B
  | .returnValue a, k => k a
  | .ask q t, k => .ask q (fun b => bind (t b) k)

def map (f : A → B) (t : QueryTree Q A) : QueryTree Q B :=
  bind t (fun a => .returnValue (f a))

@[simp] theorem run_bind (o : Q → Bool) (t : QueryTree Q A) (k : A → QueryTree Q B) :
    (t.bind k).run o = (k (t.run o)).run o := by
  induction t with
  | returnValue => rfl
  | ask q t ih => exact ih (o q)

@[simp] theorem run_map (o : Q → Bool) (t : QueryTree Q A) (f : A → B) :
    (t.map f).run o = f (t.run o) := by simp [map, run]

theorem depth_bind_le (t : QueryTree Q A) (k : A → QueryTree Q B) (c : ℕ)
    (hc : ∀ a, (k a).depth ≤ c) :
    (t.bind k).depth ≤ t.depth + c := by
  induction t with
  | returnValue a => simpa [bind, depth] using hc a
  | ask q t ih =>
    simp only [bind, depth]
    have h₀ := ih false
    have h₁ := ih true
    omega

theorem depth_map_le (t : QueryTree Q A) (f : A → B) :
    (t.map f).depth ≤ t.depth := by
  simpa [map] using t.depth_bind_le (fun a => .returnValue (f a)) 0 (by simp [depth])

theorem queriesIn_bind {s : Set Q} {t : QueryTree Q A} {k : A → QueryTree Q B}
    (h : t.queriesIn s) (hk : ∀ a, (k a).queriesIn s) :
    (t.bind k).queriesIn s := by
  induction t with
  | returnValue a => exact hk a
  | ask q t ih => exact ⟨h.1, fun b => ih b (h.2 b)⟩

theorem queriesIn_map {s : Set Q} {t : QueryTree Q A} (f : A → B)
    (h : t.queriesIn s) : (t.map f).queriesIn s :=
  queriesIn_bind h (fun _ => trivial)

/-- Evaluate a finite ordered batch, preserving the association with the points. -/
def batch : (n : ℕ) → (Fin n → QueryTree Q A) → QueryTree Q (Fin n → A)
  | 0, _ => .returnValue (fun i => Fin.elim0 i)
  | n + 1, t => (t 0).bind (fun a => (batch n (t ∘ Fin.succ)).map (Fin.cons a))

@[simp] theorem run_batch (o : Q → Bool) (n : ℕ) (t : Fin n → QueryTree Q A) :
    (batch n t).run o = fun i => (t i).run o := by
  induction n with
  | zero =>
    funext i
    exact Fin.elim0 i
  | succ n ih =>
    simp only [batch, run_bind, run_map, ih, Function.comp_def]
    funext i
    refine Fin.cases ?_ (fun _ => ?_) i <;> simp

theorem depth_batch_le (n c : ℕ) (t : Fin n → QueryTree Q A) (ht : ∀ i, (t i).depth ≤ c) :
    (batch n t).depth ≤ n * c := by
  induction n with
  | zero => simp [batch, depth]
  | succ n ih =>
    change ((t 0).bind
        (fun a => (batch n (t ∘ Fin.succ)).map (@Fin.cons n (fun _ => A) a))).depth ≤ _
    calc
      _ ≤ (t 0).depth + n * c := depth_bind_le _ _ _ fun a =>
        ((batch n (t ∘ Fin.succ)).depth_map_le (@Fin.cons n (fun _ => A) a)).trans
          (ih (t ∘ Fin.succ) (fun i => ht i.succ))
      _ ≤ (n + 1) * c := by rw [Nat.add_mul, one_mul, add_comm (n*c)]; gcongr; exact ht 0

theorem queriesIn_batch {s : Set Q} (n : ℕ) (t : Fin n → QueryTree Q A)
    (ht : ∀ i, (t i).queriesIn s) : (batch n t).queriesIn s := by
  induction n with
  | zero => trivial
  | succ n ih =>
    unfold batch
    exact queriesIn_bind (ht 0)
      (fun a => queriesIn_map (@Fin.cons n (fun _ => A) a)
        (ih (t ∘ Fin.succ) (fun i => ht i.succ)))

theorem run_congr {o p : Q → Bool} {s : Set Q} {t : QueryTree Q A}
    (ht : t.queriesIn s) (h : ∀ q ∈ s, o q = p q) : t.run o = t.run p := by
  induction t with
  | returnValue => rfl
  | ask q t ih =>
    simpa only [run, h q ht.1] using ih (p q) (ht.2 (p q))

/-- Implement each abstract query by one query at a mapped point. -/
def reindex {Q' : Type*} (f : Q → Q') : QueryTree Q A → QueryTree Q' A
  | .returnValue a => .returnValue a
  | .ask q t => .ask (f q) (fun b => reindex f (t b))

theorem run_reindex {Q' : Type*} (f : Q → Q') (o : Q' → Bool) (t : QueryTree Q A) :
    (t.reindex f).run o = t.run (o ∘ f) := by
  induction t with
  | returnValue => rfl
  | ask q t ih => exact ih (o (f q))

theorem depth_reindex {Q' : Type*} (f : Q → Q') (t : QueryTree Q A) :
    (t.reindex f).depth = t.depth := by
  induction t with
  | returnValue => rfl
  | ask q t ih => simp only [reindex, depth, ih]

theorem queriesIn_reindex {Q' : Type*} {f : Q → Q'} {s : Set Q'} {t : QueryTree Q A}
    (h : t.queriesIn (f ⁻¹' s)) : (t.reindex f).queriesIn s := by
  induction t with
  | returnValue => trivial
  | ask q t ih => exact ⟨h.1, fun b => ih b (h.2 b)⟩

theorem queriesIn_mono {s t : Set Q} {q : QueryTree Q A} (h : s ⊆ t) (hq : q.queriesIn s) :
    q.queriesIn t := by
  induction q with
  | returnValue => trivial
  | ask q k ih => exact ⟨h hq.1, fun b => ih b (hq.2 b)⟩

end QueryTree

section Bisection

noncomputable section

variable {X : Type*}

def threshold (h : X → ℝ) (q : X × ℝ) : Bool :=
  decide (q.2 < h q.1)

def midpoint (l u : ℝ) : ℝ := (l + u) / 2

/-- Return the midpoint of the retained interval after `k` point queries. -/
def bisect (x : X) : ℕ → ℝ → ℝ → QueryTree (X × ℝ) ℝ
  | 0, l, u => .returnValue (midpoint l u)
  | k + 1, l, u => .ask (x, midpoint l u) (fun b =>
      if b then bisect x k (midpoint l u) u else bisect x k l (midpoint l u))

theorem bisect_depth (x : X) (k : ℕ) (l u : ℝ) : (bisect x k l u).depth = k := by
  induction k generalizing l u with
  | zero => rfl
  | succ k ih => simp [bisect, QueryTree.depth, ih]

theorem bisect_legal (x : X) (k : ℕ) {a b l u : ℝ}
    (ha : a ≤ l) (hu : l ≤ u) (hb : u ≤ b) :
    (bisect x k l u).queriesIn {q | q.2 ∈ Set.Icc a b} := by
  induction k generalizing l u with
  | zero => trivial
  | succ k ih =>
    have hlm : l ≤ midpoint l u := by unfold midpoint; linarith
    have hmu : midpoint l u ≤ u := by unfold midpoint; linarith
    refine ⟨⟨ha.trans hlm, hmu.trans hb⟩, ?_⟩
    intro c
    cases c
    · exact ih ha hlm (hmu.trans hb)
    · exact ih (ha.trans hlm) hmu hb

/-- The endpoint convention does not matter for localization. -/
theorem bisect_error (o : X × ℝ → Bool) (h : X → ℝ)
    (ho : ∀ x t, (o (x, t) = true → t ≤ h x) ∧ (o (x, t) = false → h x ≤ t))
    (x : X) (k : ℕ) {l u : ℝ} (hl : l ≤ h x) (hu : h x ≤ u) :
    |(bisect x k l u).run o - h x| ≤ (u - l) / (2 * (2 : ℝ) ^ k) := by
  induction k generalizing l u with
  | zero =>
    rw [abs_le]
    constructor <;> simp [bisect, QueryTree.run, midpoint] <;> linarith
  | succ k ih =>
    simp only [bisect, QueryTree.run]
    have hpow : (2 : ℝ) ^ k ≠ 0 := by positivity
    cases he : o (x, midpoint l u)
    · simp only [Bool.false_eq_true, ↓reduceIte]
      refine (ih hl ((ho x _).2 he)).trans_eq ?_
      unfold midpoint
      rw [pow_succ]
      field_simp
      ring
    · simp only [↓reduceIte]
      refine (ih ((ho x _).1 he) hu).trans_eq ?_
      unfold midpoint
      rw [pow_succ]
      field_simp
      ring

theorem threshold_spec (h : X → ℝ) (x : X) (t : ℝ) :
    (threshold h (x, t) = true → t ≤ h x) ∧ (threshold h (x, t) = false → h x ≤ t) := by
  simp [threshold]
  exact le_of_lt

end
end Bisection

end Cubature
