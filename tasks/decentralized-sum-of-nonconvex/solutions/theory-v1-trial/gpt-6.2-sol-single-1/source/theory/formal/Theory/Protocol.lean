import Theory.Sanity

open MeasureTheory

noncomputable section

namespace ULB63
namespace Protocol

variable {n m d c : ℕ} {α β : Type}

def Supported (S : Set (Fin d)) (x : E d) : Prop := supp x ⊆ S

theorem supported_iff {S : Set (Fin d)} {x : E d} :
    Supported S x ↔ ∀ k ∉ S, x k = 0 := by
  classical
  constructor
  · intro h k hk
    by_contra hn
    exact hk (h hn)
  · intro h k hk
    by_contra hn
    exact hk (h k hn)

theorem supported_zero (S : Set (Fin d)) : Supported S (0 : E d) := by
  simp [Supported, supp]

theorem Supported.mono {S T : Set (Fin d)} {x : E d}
    (h : Supported S x) (hst : S ⊆ T) : Supported T x := h.trans hst

theorem Supported.add {S : Set (Fin d)} {x y : E d}
    (hx : Supported S x) (hy : Supported S y) : Supported S (x + y) := by
  rw [supported_iff] at *
  intro k hk
  simp [hx k hk, hy k hk]

theorem Supported.sub {S : Set (Fin d)} {x y : E d}
    (hx : Supported S x) (hy : Supported S y) : Supported S (x - y) := by
  rw [supported_iff] at *
  intro k hk
  simp [hx k hk, hy k hk]

theorem Supported.smul {S : Set (Fin d)} {x : E d}
    (hx : Supported S x) (a : ℝ) : Supported S (a • x) := by
  rw [supported_iff] at *
  intro k hk
  simp [hx k hk]

theorem Supported.sum {ι : Type*} {s : Finset ι} {v : ι → E d} {S : Set (Fin d)}
    (hv : ∀ i ∈ s, Supported S (v i)) : Supported S (∑ i ∈ s, v i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using supported_zero S
  | insert i s hi ih =>
    rw [Finset.sum_insert hi]
    exact (hv i (Finset.mem_insert_self i s)).add
      (ih fun j hj => hv j (Finset.mem_insert_of_mem hj))

/-- A local continuation after one oracle call or one use of the fixed weighted sum. -/
inductive Plan (m d : ℕ) (α : Type) where
  | done (v : α)
  | grad (j : Fin m) (x : E d) (next : E d → Plan m d α)
  | prox (x : E d) (η : ℝ) (next : E d → Plan m d α)
  | mix (x : E d) (next : E d → Plan m d α)

namespace Plan

def bind (p : Plan m d α) (f : α → Plan m d β) : Plan m d β :=
  match p with
  | .done x => f x
  | .grad j x k => .grad j x (fun y => (k y).bind f)
  | .prox x η k => .prox x η (fun y => (k y).bind f)
  | .mix x k => .mix x (fun y => (k y).bind f)

def Valid (P : Set (Fin d) → α → Prop) (S : Set (Fin d)) : Plan m d α → Prop
  | .done x => P S x
  | .grad _ x k => Supported S x ∧ ∀ y, Valid P (S ∪ supp y) (k y)
  | .prox x η k => (Supported S x ∧ 0 < η) ∧ ∀ y, Valid P (S ∪ supp y) (k y)
  | .mix x k => Supported S x ∧ ∀ y, Valid P (S ∪ supp y) (k y)

theorem valid_bind {P : Set (Fin d) → α → Prop} {Q : Set (Fin d) → β → Prop}
    {p : Plan m d α} {f : α → Plan m d β} {S : Set (Fin d)}
    (hp : p.Valid P S) (hf : ∀ T, S ⊆ T → ∀ x, P T x → (f x).Valid Q T) :
    (p.bind f).Valid Q S := by
  induction p generalizing S with
  | done x => exact hf S (subset_refl _) x hp
  | grad j x k ih =>
    exact ⟨hp.1, fun y => ih y (hp.2 y) fun T h => hf T (Set.subset_union_left.trans h)⟩
  | prox x η k ih =>
    exact ⟨hp.1, fun y => ih y (hp.2 y) fun T h => hf T (Set.subset_union_left.trans h)⟩
  | mix x k ih =>
    exact ⟨hp.1, fun y => ih y (hp.2 y) fun T h => hf T (Set.subset_union_left.trans h)⟩

def active : Plan m d α → Prop
  | .done _ => False
  | _ => True

def message : Plan m d α → E d
  | .mix x _ => x
  | _ => 0

def oracle : Plan m d α → Call m d
  | .grad j x _ => .grad j x
  | .prox x η _ => .prox x η
  | _ => .idle

def result (p : Plan m d α) (default : α) : α :=
  match p with
  | .done x => x
  | _ => default

def advance (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
    (p : Fin n → Plan m d α) (i : Fin n) : Plan m d α :=
  match p i with
  | .done x => .done x
  | .grad j x k => k (gradient (I.F i j) x)
  | .prox x η k => k (I.prox x η)
  | .mix _ k => k (∑ j, W i j • (p j).message)

inductive Executes (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) :
    ℕ → (Fin n → Plan m d α) → (Fin n → α) → Prop where
  | done (v : Fin n → α) : Executes I W 0 (fun i => .done (v i)) v
  | next {r p v} (h : ∀ i, (p i).active) (hr : Executes I W r (advance I W p) v) :
      Executes I W (r + 1) p v

theorem advance_bind (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
    {p : Fin n → Plan m d α} (h : ∀ i, (p i).active)
    (f : Fin n → α → Plan m d β) :
    advance I W (fun i => (p i).bind (f i)) = fun i => (advance I W p i).bind (f i) := by
  have hm : ∀ j, ((p j).bind (f j)).message = (p j).message := by
    intro j
    cases hh : p j with
    | done => exact False.elim (by simpa only [hh, active] using h j)
    | _ => rfl
  funext i
  unfold advance
  cases hh : p i with
  | done => exact False.elim (by simpa only [hh, active] using h i)
  | grad => simp only [hh, bind]
  | prox => simp only [hh, bind]
  | mix => simp only [hh, bind, hm]

theorem Executes.bind {I : Instance n m d} {W : Matrix (Fin n) (Fin n) ℝ}
    {r s : ℕ} {p : Fin n → Plan m d α} {v : Fin n → α}
    {f : Fin n → α → Plan m d β} {z : Fin n → β}
    (hp : Executes I W r p v) (hf : Executes I W s (fun i => f i (v i)) z) :
    Executes I W (r + s) (fun i => (p i).bind (f i)) z := by
  induction hp with
  | done => simpa only [Nat.zero_add, Plan.bind] using hf
  | @next k p v ha hk ih =>
    rw [show k + 1 + s = k + s + 1 by omega]
    refine .next (fun i => ?_) ?_
    · have h := ha i
      cases hh : p i with
      | done => simp only [hh, active] at h
      | grad => trivial
      | prox => trivial
      | mix => trivial
    · rw [advance_bind I W ha f]
      exact ih hf

theorem executes_grad (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
    (j : Fin n → Fin m) (x : Fin n → E d) :
    Executes I W 1 (fun i => .grad (j i) (x i) .done)
      (fun i => gradient (I.F i (j i)) (x i)) := .next (fun _ => trivial) (.done _)

theorem executes_prox (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
    (x : Fin n → E d) (η : ℝ) :
    Executes I W 1 (fun i => .prox (x i) η .done) (fun i => I.prox (x i) η) :=
  .next (fun _ => trivial) (.done _)

theorem executes_mix (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
    (x : Fin n → E d) :
    Executes I W 1 (fun i => .mix (x i) .done) (fun i => ∑ j, W i j • x j) :=
  .next (fun _ => trivial) (.done _)

def gradVec : Obs d 1 → E d
  | .grad x => x
  | _ => 0

def proxVec : Obs d 1 → E d
  | .prox x => x
  | _ => 0

def mixVec : Obs d 1 → E d
  | .recv x => x 0
  | _ => 0

def getObs (H : List (Obs d 1)) (t : ℕ) : Obs d 1 := H[t]?.getD .idle

def replayStep (p : Plan m d α) (H : List (Obs d 1)) (t : ℕ) : Plan m d α :=
  match p with
  | .done v => .done v
  | .grad _ _ k => k (gradVec (getObs H (microSteps * t)))
  | .prox _ _ k => k (proxVec (getObs H (microSteps * t)))
  | .mix _ k => k (mixVec (getObs H (microSteps * t + commSlot)))

def replay (p : Plan m d α) (H : List (Obs d 1)) : ℕ → Plan m d α
  | 0 => p
  | t+1 => replayStep (replay p H t) H t

theorem getObs_append {H T : List (Obs d 1)} {t : ℕ} (ht : t < H.length) :
    getObs (H ++ T) t = getObs H t := by simp [getObs, List.getElem?_append_left, ht]

theorem replay_append {H T : List (Obs d 1)} {t : ℕ} (ht : microSteps * t ≤ H.length)
    (p : Plan m d α) : replay p (H ++ T) t = replay p H t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    have h0 : microSteps * t < H.length := by unfold microSteps at *; omega
    have h2 : microSteps * t + commSlot < H.length := by
      unfold microSteps commSlot at *; omega
    rw [replay, replay, ih h0.le]
    cases replay p H t <;> simp [replayStep, getObs_append h0, getObs_append h2]

theorem supp_getObs (H : List (Obs d 1)) (t : ℕ) (T : Set (Fin d))
    (h : ∀ o ∈ H, o.supp ⊆ T) : (getObs H t).supp ⊆ T := by
  unfold getObs
  cases hh : H[t]? with
  | none => simp [Obs.supp]
  | some o => exact h o (List.mem_of_getElem? hh)

theorem supp_gradVec (o : Obs d 1) : supp (gradVec o) ⊆ o.supp := by
  cases o <;> simp [gradVec, Obs.supp, supp]

theorem supp_proxVec (o : Obs d 1) : supp (proxVec o) ⊆ o.supp := by
  cases o <;> simp [proxVec, Obs.supp, supp]

theorem supp_mixVec (o : Obs d 1) : supp (mixVec o) ⊆ o.supp := by
  cases o with
  | recv x => exact Set.subset_iUnion (fun k => supp (x k)) 0
  | _ => simp [mixVec, Obs.supp, supp]

theorem replay_valid {p : Plan m d α} {P : Set (Fin d) → α → Prop}
    {S T : Set (Fin d)} (hp : p.Valid P S) (hst : S ⊆ T) (H : List (Obs d 1))
    (h : ∀ o ∈ H, o.supp ⊆ T) (t : ℕ) :
    ∃ U ⊆ T, (replay p H t).Valid P U := by
  induction t with
  | zero => exact ⟨S, hst, hp⟩
  | succ t ih =>
    obtain ⟨U, hU, hv⟩ := ih
    have hg := (supp_gradVec (getObs H (microSteps*t))).trans (supp_getObs _ _ T h)
    have hh := (supp_proxVec (getObs H (microSteps*t))).trans (supp_getObs _ _ T h)
    have hm := (supp_mixVec (getObs H (microSteps*t+commSlot))).trans (supp_getObs _ _ T h)
    rw [replay]
    cases hs : replay p H t with
    | done =>
      rw [hs] at hv
      exact ⟨U, hU, hv⟩
    | grad =>
      rw [hs] at hv
      exact ⟨_, Set.union_subset hU hg, hv.2 _⟩
    | prox =>
      rw [hs] at hv
      exact ⟨_, Set.union_subset hU hh, hv.2 _⟩
    | mix =>
      rw [hs] at hv
      exact ⟨_, Set.union_subset hU hm, hv.2 _⟩

theorem Valid.oracle {p : Plan m d α} {P : Set (Fin d) → α → Prop} {S : Set (Fin d)}
    (h : p.Valid P S) : p.oracle.supp ⊆ S := by
  cases p with
  | done => exact Set.empty_subset _
  | grad => exact h.1
  | prox => exact h.1.1
  | mix => exact Set.empty_subset _

theorem Valid.message {p : Plan m d α} {P : Set (Fin d) → α → Prop} {S : Set (Fin d)}
    (h : p.Valid P S) : Supported S p.message := by
  cases p with
  | mix => exact h.1
  | _ => exact supported_zero _

theorem Valid.pos {p : Plan m d α} {P : Set (Fin d) → α → Prop} {S : Set (Fin d)}
    (h : p.Valid P S) {x : E d} {η : ℝ} (he : p.oracle = .prox x η) : 0 < η := by
  cases p with
  | prox v η' k =>
    change Call.prox v η' = Call.prox x η at he
    cases he
    exact h.1.2
  | _ => cases he

end Plan

section Compile

variable {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
  (p : (d : ℕ) → Info n d Ω → Plan m d (E d))

abbrev compile : Algorithm n m 1 where
  Ω := Ω
  P := P
  call d info H :=
    if H.length % microSteps = 0 then (Plan.replay (p d info) H (H.length / microSteps)).oracle
    else .idle
  send d info H _ := (Plan.replay (p d info) H (H.length / microSteps)).message
  out d info H := (Plan.replay (p d info) H (H.length / microSteps)).result info.x0

omit [IsProbabilityMeasure P] in
private theorem currentIter_nil {H : List (Obs d 1)} (h : H.length % microSteps = 0) :
    currentIter H = [] := by
  unfold currentIter
  have hh := Nat.mod_add_div H.length microSteps
  rw [h, zero_add] at hh
  rw [hh, List.drop_length]

theorem compile_admissible [DiscreteMeasurableSpace Ω] {L μ Δ ρ : ℝ}
    (hp : ∀ d (info : Info n d Ω), (p d info).Valid Supported (supp info.x0)) :
    (compile P p).Admissible L μ Δ ρ := by
  have hv (e : ℕ) (info : Info n e Ω) (H : List (Obs e 1)) :
      ∃ U ⊆ allowed info H,
        (Plan.replay (p e info) H (H.length / microSteps)).Valid Supported U :=
    Plan.replay_valid (hp e info) Set.subset_union_left H
      (fun o ho => (Set.subset_iUnion₂ (s := fun u (_ : u ∈ H) => u.supp) o ho).trans
        Set.subset_union_right) _
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro e info H
    obtain ⟨U, hu, hh⟩ := hv e info H
    change (if H.length % microSteps = 0 then
      (Plan.replay (p e info) H (H.length / microSteps)).oracle else Call.idle).supp ⊆ _
    split
    · exact hh.oracle.trans hu
    · simp [Call.supp]
  · intro e info H k
    obtain ⟨U, hu, hh⟩ := hv e info H
    exact (hh.message).mono hu
  · intro e info H
    obtain ⟨U, hu, hh⟩ := hv e info H
    change Supported (allowed info H) ((Plan.replay _ _ _).result info.x0)
    cases h : Plan.replay (p e info) H (H.length / microSteps) with
    | done v =>
      rw [h] at hh
      exact (show Supported U v from hh).mono hu
    | _ => exact Set.subset_union_left
  · intro e info H x η he
    obtain ⟨U, hu, hh⟩ := hv e info H
    change (if H.length % microSteps = 0 then
      (Plan.replay (p e info) H (H.length / microSteps)).oracle else Call.idle) = _ at he
    split at he
    · exact hh.pos he
    · cases he
  · intro e info H h
    change (if H.length % microSteps = 0 then
      (Plan.replay (p e info) H (H.length / microSteps)).oracle else Call.idle).isGrad = false
    split
    · rename_i hl
      simp [currentIter_nil hl] at h
    · rfl
  · intro e info H h
    change (if H.length % microSteps = 0 then
      (Plan.replay (p e info) H (H.length / microSteps)).oracle else Call.idle).isProx = false
    split
    · rename_i hl
      simp [currentIter_nil hl] at h
    · rfl
  · intros; exact Measurable.of_discrete
  · intros; exact Measurable.of_discrete
  · intros; exact Measurable.of_discrete

end Compile

section Execution

variable (A : Algorithm n m 1) (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
  (ω : A.Ω) (i : Fin n)

private def observed (s : ℕ) : Obs d 1 :=
  if s % microSteps = commSlot then
    .recv fun k => ∑ j, W i j • A.send d (A.info I W ω j) (A.hist I W ω s j) k
  else
    Algorithm.answer I i (A.call d (A.info I W ω i) (A.hist I W ω s i))

private theorem hist_succ (s : ℕ) :
    A.hist I W ω (s+1) i = A.hist I W ω s i ++ [observed A I W ω i s] := by
  rw [Algorithm.hist, observed]
  split <;> rfl

theorem hist_length (s : ℕ) : (A.hist I W ω s i).length = s := by
  induction s with
  | zero => rfl
  | succ s ih => simp [hist_succ, ih]

theorem hist_prefix {s t : ℕ} (hst : s ≤ t) :
    ∃ T, A.hist I W ω t i = A.hist I W ω s i ++ T := by
  induction t, hst using Nat.le_induction with
  | base => exact ⟨[], by simp⟩
  | succ t _ ih =>
    obtain ⟨T, hT⟩ := ih
    exact ⟨T ++ [observed A I W ω i t], by rw [hist_succ, hT, List.append_assoc]⟩

private theorem read_hist {s t : ℕ} (hst : s < t) :
    Plan.getObs (A.hist I W ω t i) s = observed A I W ω i s := by
  obtain ⟨T, hT⟩ := hist_prefix A I W ω i (Nat.succ_le_of_lt hst)
  rw [hT, hist_succ, List.append_assoc]
  simp [Plan.getObs, hist_length]

end Execution

section CompiledExecution

variable {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
  (p : (d : ℕ) → Info n d Ω → Plan m d (E d))
  (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (ω : Ω)

def compiledState (t : ℕ) (i : Fin n) : Plan m d (E d) :=
  let A := compile P p
  Plan.replay (p d (A.info I W ω i)) (A.hist I W ω (microSteps * t) i) t

private theorem compiled_prefix (t k : ℕ) (i : Fin n) :
    let A := compile P p
    Plan.replay (p d (A.info I W ω i)) (A.hist I W ω (microSteps * t + k) i) t =
      compiledState P p I W ω t i := by
  let A := compile P p
  obtain ⟨T, hT⟩ := hist_prefix A I W ω i (Nat.le_add_right (microSteps*t) k)
  change Plan.replay (p d (A.info I W ω i)) (A.hist I W ω (microSteps*t+k) i) t = _
  rw [hT, Plan.replay_append (by rw [hist_length])]
  rfl

theorem compiledState_succ (t : ℕ) :
    compiledState P p I W ω (t+1) = Plan.advance I W (compiledState P p I W ω t) := by
  let A := compile P p
  let b := microSteps * t
  have h₀ : b % microSteps = 0 := by dsimp [b, microSteps]; omega
  have h₂ : (b+commSlot) % microSteps = commSlot := by dsimp [b, microSteps, commSlot]; omega
  have hd₀ : b / microSteps = t := by dsimp [b, microSteps]; omega
  have hd₂ : (b+commSlot) / microSteps = t := by dsimp [b, microSteps, commSlot]; omega
  have hk : microSteps * (t+1) = b + microSteps := Nat.mul_succ _ _
  have hn : b < microSteps * (t+1) := by rw [hk]; norm_num [microSteps]
  have hn' : b + commSlot < microSteps * (t+1) := by rw [hk]; norm_num [microSteps, commSlot]
  have hcall (j : Fin n) :
      A.call d (A.info I W ω j) (A.hist I W ω b j) =
        (compiledState P p I W ω t j).oracle := by
    change (if (A.hist I W ω b j).length % microSteps = 0 then
      (Plan.replay _ _ ((A.hist I W ω b j).length / microSteps)).oracle else .idle) = _
    rw [hist_length, ite_eq_left h₀, hd₀]
    rfl
  have hsend (j : Fin n) (k : Fin 1) :
      A.send d (A.info I W ω j) (A.hist I W ω (b+commSlot) j) k =
        (compiledState P p I W ω t j).message := by
    change (Plan.replay _ _ ((A.hist I W ω (b+commSlot) j).length / microSteps)).message = _
    rw [hist_length, hd₂, compiled_prefix]
  funext i
  have hg : Plan.getObs (A.hist I W ω (microSteps*(t+1)) i) b =
      Algorithm.answer I i (compiledState P p I W ω t i).oracle := by
    rw [read_hist A I W ω i hn, observed, ite_eq_right (by rw [h₀]; decide), hcall]
  have hr : Plan.getObs (A.hist I W ω (microSteps*(t+1)) i) (b+commSlot) =
      .recv (fun _ => ∑ j, W i j • (compiledState P p I W ω t j).message) := by
    rw [read_hist A I W ω i hn', observed, ite_eq_left h₂]
    simp only [hsend]
  change Plan.replay _ (A.hist I W ω (microSteps*(t+1)) i) (t+1) = _
  rw [Plan.replay]
  have H : Plan.replay (p d (A.info I W ω i)) (A.hist I W ω (microSteps*(t+1)) i) t =
      compiledState P p I W ω t i := by
    rw [hk]
    exact compiled_prefix P p I W ω t microSteps i
  rw [H, Plan.advance]
  cases hh : compiledState P p I W ω t i <;>
    simp only [Plan.replayStep, ← show b = microSteps*t from rfl, hg, hr, hh, Plan.oracle,
      Algorithm.answer, Plan.gradVec, Plan.proxVec, Plan.mixVec]

theorem compiled_output_of_executes {t : ℕ} {v : Fin n → E d}
    (h : Plan.Executes I W t (fun i => p d ((compile P p).info I W ω i)) v) :
    ∀ i, (compile P p).output I W ω t i = v i := by
  have key {k : ℕ} {q : Fin n → Plan m d (E d)} (hq : Plan.Executes I W k q v) :
      ∀ s, compiledState P p I W ω s = q →
        compiledState P p I W ω (s+k) = fun i => .done (v i) := by
    clear h
    induction hq with
    | done => intro s hs; simpa using hs
    | @next r q v _ _ ih =>
      intro s hs
      rw [show s + (r+1) = (s+1)+r by omega]
      exact ih (s+1) (by rw [compiledState_succ, hs])
  have hh := key h 0 rfl
  rw [Nat.zero_add] at hh
  intro i
  have hd : (microSteps*t) / microSteps = t := by unfold microSteps; omega
  let H := (compile P p).hist I W ω (microSteps*t) i
  have hH : H.length = microSteps*t := hist_length (compile P p) I W ω i _
  change (Plan.replay (p d ((compile P p).info I W ω i))
    H (H.length / microSteps)).result I.x0 = _
  rw [hH, hd]
  change (compiledState P p I W ω t i).result _ = _
  rw [hh]
  rfl

end CompiledExecution

end Protocol
end ULB63
