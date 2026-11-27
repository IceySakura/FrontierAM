import Mathlib

/-!
# ULB-profile-63 (SUM-SC-NS-SFO-STAT-COMP-PROX): the problem model

Each definition transcribes one clause of the problem statement `/workspace/PROFILE.md`,
§1 "Problem Formulation". `/workspace/MODEL_NOTES.md` maps clauses to definitions and lists the
choices the text leaves open.

This file is part of the task and fixed: the evaluation replaces any edited copy with the
original. State results about the problem with these definitions.
-/

open MeasureTheory
open scoped ENNReal

noncomputable section

namespace ULB63

/-! ## Vectors -/

/-- `ℝ^d` with the Euclidean inner product. -/
abbrev E (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- The coordinates where `x` is nonzero. -/
def supp {d : ℕ} (x : E d) : Set (Fin d) := {k | x k ≠ 0}

/-! ## Function class `𝓕_{L,μ,Δ,m}` and oracle class `𝓞_{IFO,prox}` -/

/-- `F` is `L`-smooth: differentiable with `L`-Lipschitz gradient. -/
def LSmooth {d : ℕ} (L : ℝ) (F : E d → ℝ) : Prop :=
  Differentiable ℝ F ∧ ∀ x y, ‖gradient F x - gradient F y‖ ≤ L * ‖x - y‖

/-- `μ`-strong convexity of a function with values in `ℝ ∪ {±∞}`; `μ = 0` is convexity.
On real-valued functions it agrees with Mathlib's `StrongConvexOn univ μ` (`Theory/Sanity.lean`). -/
def StronglyConvexE {d : ℕ} (μ : ℝ) (φ : E d → EReal) : Prop :=
  ∀ x y : E d, ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
    φ (t • x + (1 - t) • y) + ((μ / 2 * t * (1 - t) * ‖x - y‖ ^ 2 : ℝ) : EReal) ≤
      (t : EReal) * φ x + ((1 - t : ℝ) : EReal) * φ y

/-- `p = prox_{ηh}(v)`: `p` minimizes `u ↦ h u + ‖u - v‖² / (2η)`. -/
def IsProx {d : ℕ} (h : E d → EReal) (η : ℝ) (v p : E d) : Prop :=
  ∀ u, h p + ((‖p - v‖ ^ 2 / (2 * η) : ℝ) : EReal) ≤ h u + ((‖u - v‖ ^ 2 / (2 * η) : ℝ) : EReal)

/-- A problem instance in dimension `d`: the components `F i ξ` with `ξ ∈ D_i ≃ Fin m`, the
regularizer `h : ℝ^d → ℝ ∪ {+∞}`, the common initial point, and the proximal oracle. -/
structure Instance (n m d : ℕ) where
  F : Fin n → Fin m → E d → ℝ
  h : E d → EReal
  x0 : E d
  prox : E d → ℝ → E d

namespace Instance

variable {n m d : ℕ} (I : Instance n m d)

/-- `f = (1/n) ∑ᵢ fᵢ` with `fᵢ = (1/m) ∑_ξ Fᵢ(·; ξ)`. -/
def f (x : E d) : ℝ := (1 / n : ℝ) * ∑ i, (1 / m : ℝ) * ∑ ξ, I.F i ξ x

/-- `φ = f + h`. -/
def φ (x : E d) : EReal := (I.f x : EReal) + I.h x

/-- `φ(x⋆) = inf φ`. -/
def φstar : EReal := ⨅ x, I.φ x

/-- `φ(x) - φ(x⋆)` as an element of `[0, ∞]`; it is `∞` outside `dom h`. -/
def gap (x : E d) : ℝ≥0∞ := (I.φ x - I.φstar).toENNReal

/-- Membership in `𝓕_{L,μ,Δ,m}`, together with the oracle class: the gradient oracle answers
`gradient (F i ξ)`, and `prox` answers the exact proximal map for every `η > 0`. -/
structure InClass (L μ Δ : ℝ) : Prop where
  smooth : ∀ i ξ, LSmooth L (I.F i ξ)
  f_convex : ConvexOn ℝ Set.univ I.f
  h_ne_bot : ∀ x, I.h x ≠ ⊥
  h_proper : ∃ x, I.h x ≠ ⊤
  h_closed : LowerSemicontinuous I.h
  h_convex : StronglyConvexE 0 I.h
  prox_exact : ∀ v η, 0 < η → IsProx I.h η v (I.prox v η)
  φ_strongly_convex : StronglyConvexE μ I.φ
  minimizer : ∃ xs, ∀ y, I.φ xs ≤ I.φ y
  initial_gap : I.φ I.x0 - I.φstar ≤ (Δ : EReal)

end Instance

/-! ## Communication class `𝓦_{n,ρ}` -/

/-- `J = (1/n) 𝟙𝟙ᵀ`. -/
def J (n : ℕ) : Matrix (Fin n) (Fin n) ℝ := fun _ _ => 1 / n

/-- The Euclidean norm on `ℝ^n`. -/
def norm2 {n : ℕ} (v : Fin n → ℝ) : ℝ := Real.sqrt (∑ i, v i ^ 2)

/-- `W ≥ 0`, `W = Wᵀ`, `W𝟙 = 𝟙`, and `‖W - J‖₂ ≤ ρ` for the spectral norm. -/
structure CommClass (n : ℕ) (ρ : ℝ) (W : Matrix (Fin n) (Fin n) ℝ) : Prop where
  nonneg : ∀ i j, 0 ≤ W i j
  symm : W.transpose = W
  row_sum : ∀ i, ∑ j, W i j = 1
  contraction : ∀ v : Fin n → ℝ, norm2 ((W - J n).mulVec v) ≤ ρ * norm2 v

/-! ## Algorithm class `𝓐`

An iteration consists of `microSteps = 5` micro-steps: two oracle slots, the communication
round (position `commSlot = 2`), and two oracle slots. In an oracle slot a node makes one
component-gradient query, one proximal query, or nothing, with at most one of each per
iteration. Every order of the at most three primitive operations of an iteration, chosen
adaptively, is therefore representable. -/

/-- Micro-steps per iteration. -/
def microSteps : ℕ := 5

/-- Position of the communication round within an iteration. -/
def commSlot : ℕ := 2

/-- What a node knows before the run: its index, its row of `W`, `x₀`, and the shared seed.
The parameters `n, m, L, μ, Δ, ρ, ε` and the dimension `d` are fixed before the algorithm acts. -/
structure Info (n d : ℕ) (Ω : Type) where
  node : Fin n
  row : Fin n → ℝ
  x0 : E d
  seed : Ω

/-- What a node observes in one micro-step. -/
inductive Obs (d c : ℕ) where
  | idle
  | grad (g : E d)
  | prox (p : E d)
  | recv (r : Fin c → E d)

/-- A call in an oracle slot: nothing, `∇Fᵢ(x; ξ)`, or `prox_{ηh}(v)`. -/
inductive Call (m d : ℕ) where
  | idle
  | grad (ξ : Fin m) (x : E d)
  | prox (v : E d) (η : ℝ)

namespace Obs

variable {d c : ℕ}

/-- Coordinates where the observation is nonzero. -/
def supp : Obs d c → Set (Fin d)
  | idle => ∅
  | grad g => ULB63.supp g
  | prox p => ULB63.supp p
  | recv r => ⋃ k, ULB63.supp (r k)

def isGrad : Obs d c → Bool
  | grad _ => true
  | _ => false

def isProx : Obs d c → Bool
  | prox _ => true
  | _ => false

end Obs

namespace Call

variable {m d : ℕ}

/-- Coordinates where the query input is nonzero. -/
def supp : Call m d → Set (Fin d)
  | idle => ∅
  | grad _ x => ULB63.supp x
  | prox v _ => ULB63.supp v

def isGrad : Call m d → Bool
  | grad _ _ => true
  | _ => false

def isProx : Call m d → Bool
  | prox _ _ => true
  | _ => false

/-- The call as (kind, component index, input vector, step size), a measurable space. -/
def enc : Call m d → ℕ × ℕ × E d × ℝ
  | idle => (0, 0, 0, 0)
  | grad ξ x => (1, ξ, x, 0)
  | prox v η => (2, 0, v, η)

end Call

/-- Coordinates a node may use: those nonzero in `x₀`, in one of its previous oracle outputs, or
in one of the mixed messages it has received. -/
def allowed {n d c : ℕ} {Ω : Type} (info : Info n d Ω) (H : List (Obs d c)) : Set (Fin d) :=
  supp info.x0 ∪ ⋃ o ∈ H, o.supp

/-- The observations already made in the current iteration (a history of length `s` is the one
held at micro-step `s`). -/
def currentIter {d c : ℕ} (H : List (Obs d c)) : List (Obs d c) :=
  H.drop (microSteps * (H.length / microSteps))

/-- A randomized algorithm: a probability space of shared seeds and, for every dimension `d`,
rules mapping a node's knowledge and observation history to its next oracle call, the `c`
vectors it sends in a communication round, and its current output. -/
structure Algorithm (n m c : ℕ) where
  Ω : Type
  [measurableSpace : MeasurableSpace Ω]
  P : Measure Ω
  [isProbability : IsProbabilityMeasure P]
  call : (d : ℕ) → Info n d Ω → List (Obs d c) → Call m d
  send : (d : ℕ) → Info n d Ω → List (Obs d c) → Fin c → E d
  out : (d : ℕ) → Info n d Ω → List (Obs d c) → E d

attribute [instance] Algorithm.measurableSpace Algorithm.isProbability

namespace Algorithm

section Run

variable {n m c d : ℕ} (A : Algorithm n m c) (I : Instance n m d)
  (W : Matrix (Fin n) (Fin n) ℝ)

/-- Node `i`'s knowledge when `A` runs on `I` over `W` with seed `ω`. -/
def info (ω : A.Ω) (i : Fin n) : Info n d A.Ω := ⟨i, W i, I.x0, ω⟩

/-- The oracle's answer to a call of node `i`. -/
def answer (I : Instance n m d) (i : Fin n) : Call m d → Obs d c
  | .idle => .idle
  | .grad ξ x => .grad (gradient (I.F i ξ) x)
  | .prox v η => .prox (I.prox v η)

/-- The histories of all nodes after `s` micro-steps. In the communication round node `i`
receives `∑ⱼ wᵢⱼ zⱼ` for each of the `c` channels. -/
def hist (ω : A.Ω) : ℕ → Fin n → List (Obs d c)
  | 0 => fun _ => []
  | s + 1 => fun i =>
    if s % microSteps = commSlot then
      hist ω s i ++ [.recv fun k => ∑ j, W i j • A.send d (A.info I W ω j) (hist ω s j) k]
    else
      hist ω s i ++ [answer I i (A.call d (A.info I W ω i) (hist ω s i))]

/-- `x̂_{t,i}`: node `i`'s output after `t` iterations. -/
def output (ω : A.Ω) (t : ℕ) (i : Fin n) : E d :=
  A.out d (A.info I W ω i) (A.hist I W ω (microSteps * t) i)

/-- `x̂_t = (1/n) ∑ᵢ x̂_{t,i}`, formed by the evaluator. -/
def avgOutput (ω : A.Ω) (t : ℕ) : E d := (1 / n : ℝ) • ∑ i, A.output I W ω t i

/-- `𝔼[φ(x̂_t) - φ(x⋆)]` over the algorithm's randomness. -/
def expectedGap (t : ℕ) : ℝ≥0∞ := ∫⁻ ω, I.gap (A.avgOutput I W ω t) ∂A.P

/-- `T_ε = inf {t ∈ ℤ≥0 : 𝔼[φ(x̂_t) - φ(x⋆)] ≤ ε}`, with `inf ∅ = ∞`. -/
def hittingTime (ε : ℝ) : ℕ∞ :=
  ⨅ (t : ℕ) (_ : A.expectedGap I W t ≤ ENNReal.ofReal ε), (t : ℕ∞)

end Run

/-- Membership in `𝓐`.
* zero-respecting: query inputs, sent vectors and outputs vanish outside `allowed`;
* budget: at most one component-gradient query and one proximal query per iteration (one
  communication round per iteration is built into `hist`), and proximal steps `η > 0`;
* a randomized algorithm is a random variable: on every legal instance, every call, sent vector
  and output that the run uses is a measurable function of the seed. -/
structure Admissible {n m c : ℕ} (A : Algorithm n m c) (L μ Δ ρ : ℝ) : Prop where
  call_zero_respecting : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)),
    (A.call d info H).supp ⊆ allowed info H
  send_zero_respecting : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)) k,
    supp (A.send d info H k) ⊆ allowed info H
  out_zero_respecting : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)),
    supp (A.out d info H) ⊆ allowed info H
  prox_step_pos : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)) v η,
    A.call d info H = .prox v η → 0 < η
  one_grad : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)),
    (currentIter H).any Obs.isGrad → (A.call d info H).isGrad = false
  one_prox : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)),
    (currentIter H).any Obs.isProx → (A.call d info H).isProx = false
  measurable_call : ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
    I.InClass L μ Δ → CommClass n ρ W → ∀ s i, s % microSteps ≠ commSlot →
      Measurable fun ω => (A.call d (A.info I W ω i) (A.hist I W ω s i)).enc
  measurable_send : ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
    I.InClass L μ Δ → CommClass n ρ W → ∀ s i, s % microSteps = commSlot →
      Measurable fun ω => A.send d (A.info I W ω i) (A.hist I W ω s i)
  measurable_out : ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
    I.InClass L μ Δ → CommClass n ρ W → ∀ t i,
      Measurable fun ω => A.output I W ω t i

/-- At most one primitive operation per iteration: calls only in the two slots before the
communication round, at most one call per iteration, and zero messages in an iteration that
contains a call. Such an algorithm is legal under every convention for ordering the operations
within an iteration. -/
structure Sequential {n m c : ℕ} (A : Algorithm n m c) : Prop where
  no_late_call : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)),
    commSlot < H.length % microSteps → A.call d info H = .idle
  one_call : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)),
    (currentIter H).any (fun o => o.isGrad || o.isProx) → A.call d info H = .idle
  silent_after_call : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)),
    (currentIter H).any (fun o => o.isGrad || o.isProx) → A.send d info H = 0

end Algorithm

/-! ## The minimax quantity -/

/-- `inf_{A ∈ 𝓐} sup_{(f)† ∈ 𝓕} sup_{(O)† ∈ 𝓞((f)†)} sup_{W ∈ 𝓦} T_ε`. Instances of every
dimension belong to `𝓕`, so the supremum ranges over `d` as well. -/
def minimax (n m c : ℕ) (L μ Δ ρ ε : ℝ) : ℕ∞ :=
  ⨅ (A : Algorithm n m c) (_ : A.Admissible L μ Δ ρ),
    ⨆ (d : ℕ) (I : Instance n m d) (_ : I.InClass L μ Δ)
      (W : Matrix (Fin n) (Fin n) ℝ) (_ : CommClass n ρ W),
      A.hittingTime I W ε

end ULB63
