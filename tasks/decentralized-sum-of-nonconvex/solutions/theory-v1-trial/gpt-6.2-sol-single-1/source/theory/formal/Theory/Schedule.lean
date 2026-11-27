import Theory.Aggregation
import Theory.Epoch

open MeasureTheory hiding average average_const
open scoped ENNReal

noncomputable section

namespace ULB63
namespace Schedule

open Protocol Composite Aggregation FiniteMean Epoch

variable {n m d : ℕ} {L μ Δ ρ : ℝ} {σ : Type}
variable {I : Instance n m d} {W : Matrix (Fin n) (Fin n) ℝ}

def readTable (x : E d) : List (Fin m) → Plan m d (Fin m → E d)
  | [] => .done (fun _ => 0)
  | j :: js => .grad j x (fun g =>
      (readTable x js).bind (fun q => .done (Function.update q j g)))

def TableSupported (S : Set (Fin d)) (q : Fin m → E d) : Prop :=
  ∀ j, Supported S (q j)

theorem read_valid {x : E d} {S : Set (Fin d)} (h : Supported S x) (ls : List (Fin m)) :
    (readTable x ls).Valid TableSupported S := by
  induction ls generalizing S with
  | nil => exact fun _ => supported_zero _
  | cons j js ih =>
    refine ⟨h, fun g => Plan.valid_bind (ih (h.mono Set.subset_union_left)) ?_⟩
    intro T hT q hq k
    by_cases hj : k = j
    · subst k
      simpa using (show Supported T g from Set.subset_union_right.trans hT)
    · simpa [Function.update_of_ne hj] using hq k

theorem read_exec (ls : List (Fin m)) (x : Fin n → E d) :
    Plan.Executes I W ls.length (fun i => readTable (x i) ls)
      (fun i j => if j ∈ ls then gradient (I.F i j) (x i) else 0) := by
  induction ls with
  | nil => exact .done _
  | cons j js ih =>
    let f (i : Fin n) (g : E d) :=
      (readTable (x i) js).bind (fun q => Plan.done (Function.update q j g))
    have h := (Plan.executes_grad I W (fun _ => j) x).bind (f := f)
      (ih.bind (f := fun i q => .done (Function.update q j (gradient (I.F i j) (x i))))
        (.done _))
    rw [show (j::js).length = 1+(js.length+0) by simp; omega]
    convert h using 1
    · funext i
      rfl
    · funext i k
      by_cases hh : k = j
      · subst k
        simp
      · simp [hh, Function.update_of_ne]

theorem table_exec (x : Fin n → E d) :
    Plan.Executes I W m (fun i => readTable (x i) (List.finRange m))
      (fun i j => gradient (I.F i j) (x i)) := by
  simpa only [List.length_finRange, List.mem_finRange, ite_true] using
    (read_exec (I := I) (W := W) (List.finRange m) x)

theorem average_supported {ι : Type*} [Fintype ι] {f : ι → E d} {S : Set (Fin d)}
    (h : ∀ j, Supported S (f j)) : Supported S (average f) :=
  (Supported.sum (fun j _ => h j)).smul _

theorem supported_blend {x z : E d} {S : Set (Fin d)}
    (hx : Supported S x) (hz : Supported S z) (a : ℝ) : Supported S (blend a x z) :=
  (hx.smul _).add (hz.smul _)

theorem supported_momentum {x y z : E d} {S : Set (Fin d)}
    (hx : Supported S x) (hy : Supported S y) (hz : Supported S z) (a : ℝ) :
    Supported S (momentum a y x z) :=
  hz.add ((hx.sub hy).smul _)

def inner (oracle : E d → σ → Plan m d (E d)) (L a : ℝ) :
    (r : ℕ) → Draws σ r → E d → E d → Plan m d (E d)
  | 0, _, x, _ => .done x
  | r+1, s, x, z =>
    let y := blend a x z
    (oracle y (drawHead s)).bind fun g =>
      .prox (y - (4*L)⁻¹ • g) (4*L)⁻¹ fun p =>
      (inner oracle L a r (drawTail s) p (momentum a y p z)).bind fun v =>
      .done (blend (fraction r) x v)

theorem inner_valid (oracle : E d → σ → Plan m d (E d)) (hL : 0 < L) (a : ℝ)
    {B : Set (Fin d)}
    (ho : ∀ S, B ⊆ S → ∀ y, Supported S y → ∀ s, (oracle y s).Valid Supported S)
    (r : ℕ) (s : Draws σ r) {S : Set (Fin d)} {x z : E d}
    (hBS : B ⊆ S) (hx : Supported S x) (hz : Supported S z) :
    (inner oracle L a r s x z).Valid Supported S := by
  induction r generalizing S x z with
  | zero => exact hx
  | succ r ih =>
    have hy := supported_blend hx hz a
    unfold inner
    apply Plan.valid_bind (ho _ hBS _ hy _)
    intro T hST g hg
    refine ⟨⟨(hy.mono hST).sub (hg.smul _), by positivity⟩, fun p => ?_⟩
    have hT : T ⊆ T ∪ supp p := Set.subset_union_left
    have hp : Supported (T ∪ supp p) p := Set.subset_union_right
    apply Plan.valid_bind (ih _ (hBS.trans (hST.trans hT)) hp
      (supported_momentum hp (hy.mono (hST.trans hT)) (hz.mono (hST.trans hT)) a))
    intro U hU v hv
    exact supported_blend (hx.mono (hST.trans (hT.trans hU))) hv _

theorem inner_exec (oracle : Fin n → E d → σ → Plan m d (E d)) (L a : ℝ)
    (g : E d → σ → E d) (cost : ℕ)
    (ho : ∀ y s, Plan.Executes I W cost (fun i => oracle i y s) (fun _ => g y s))
    (r : ℕ) (s : Draws σ r) (x z : E d) :
    Plan.Executes I W (r*(cost+1)) (fun i => inner (oracle i) L a r s x z)
      (fun _ => (evaluate I L a g r s x z).mean) := by
  induction r generalizing x z with
  | zero => simpa only [Nat.zero_mul, inner, evaluate] using
      Plan.Executes.done (I := I) (W := W) (fun _ => x)
  | succ r ih =>
    let y := blend a x z
    let p := step I L y (g y (drawHead s))
    let z' := momentum a y p z
    let rest (i : Fin n) (p : E d) :=
      (inner (oracle i) L a r (drawTail s) p (momentum a y p z)).bind fun v =>
        Plan.done (blend (fraction r) x v)
    have h := (ho y (drawHead s)).bind
      (f := fun i b => .prox (y-(4*L)⁻¹ • b) (4*L)⁻¹ (rest i))
      ((Plan.executes_prox I W (fun _ => y-(4*L)⁻¹ • g y (drawHead s)) (4*L)⁻¹).bind
        (f := rest)
        ((ih (drawTail s) p z').bind (f := fun _ v => .done (blend (fraction r) x v)) (.done _)))
    have hr : (r+1)*(cost+1) = cost+(1+(r*(cost+1)+0)) := by rw [Nat.succ_mul]; omega
    rw [hr]
    exact h

def repeating (f : E d → σ → Plan m d (E d)) :
    (t : ℕ) → Draws σ t → E d → Plan m d (E d)
  | 0, _, x => .done x
  | t+1, s, x => (f x (drawHead s)).bind (fun y => repeating f t (drawTail s) y)

theorem repeating_valid (f : E d → σ → Plan m d (E d))
    (hf : ∀ S x, Supported S x → ∀ s, (f x s).Valid Supported S)
    (t : ℕ) (s : Draws σ t) {S : Set (Fin d)} {x : E d} (hx : Supported S x) :
    (repeating f t s x).Valid Supported S := by
  induction t generalizing S x with
  | zero => exact hx
  | succ t ih =>
    unfold repeating
    exact Plan.valid_bind (hf _ _ hx _) (fun _ _ _ hy => ih _ hy)

theorem repeating_exec (f : Fin n → E d → σ → Plan m d (E d)) (g : E d → σ → E d)
    (cost : ℕ) (hf : ∀ x s, Plan.Executes I W cost (fun i => f i x s) (fun _ => g x s))
    (t : ℕ) (s : Draws σ t) (x : E d) :
    Plan.Executes I W (t*cost) (fun i => repeating (f i) t s x)
      (fun _ => repeatEval g t s x) := by
  induction t generalizing x with
  | zero => simpa only [Nat.zero_mul, repeating, repeatEval] using
      Plan.Executes.done (I := I) (W := W) (fun _ => x)
  | succ t ih =>
    rw [Nat.succ_mul, Nat.add_comm]
    exact (hf x (drawHead s)).bind (ih (drawTail s) (g x (drawHead s)))

def exactOracle (K : ℕ) (r i : Fin n) (x : E d) (_ : Unit) : Plan m d (E d) :=
  (readTable x (List.finRange m)).bind (fun q => collect K r i (average q))

theorem exactOracle_valid (K : ℕ) (r i : Fin n) {S : Set (Fin d)} {x : E d}
    (hx : Supported S x) (s : Unit) : (exactOracle (m := m) K r i x s).Valid Supported S :=
  Plan.valid_bind (read_valid hx _) (fun _ _ _ hq => collect_valid _ _ _ (average_supported hq))

theorem exactOracle_exec (hI : I.InClass L μ Δ) (hW : CommClass n ρ W)
    (K : ℕ) (hp : ∀ i j, 0 < (Network.lazy W ^ K) i j) (r : Fin n) (x : E d) (s : Unit) :
    Plan.Executes I W (m+4*K) (fun i => exactOracle K r i x s)
      (fun _ => gradient I.f x) := by
  rw [fullGrad hI x]
  exact (table_exec (I := I) (W := W) (fun _ => x)).bind
    (collect_exec I W hW K hp r (fun i => localGrad I i x))

abbrev NodeDraw (n m : ℕ) := Fin n → Fin m

def control (I : Instance n m d) (b : E d) (y : E d) (s : NodeDraw n m) : E d :=
  gradient I.f b + average (fun i =>
    gradient (I.F i (s i)) y - gradient (I.F i (s i)) b)

def controlOracle (K : ℕ) (r i : Fin n) (table : Fin m → E d)
    (g : E d) (y : E d) (s : NodeDraw n m) : Plan m d (E d) :=
  .grad (s i) y fun p =>
    (collect K r i (p-table (s i))).bind fun a =>
    .done (g+a)

theorem controlOracle_valid (K : ℕ) (r i : Fin n) {S : Set (Fin d)} {table : Fin m → E d}
    {g y : E d} (ht : TableSupported S table) (hg : Supported S g) (hy : Supported S y)
    (s : NodeDraw n m) : (controlOracle K r i table g y s).Valid Supported S := by
  refine ⟨hy, fun p => ?_⟩
  have hp : Supported (S ∪ supp p) p := Set.subset_union_right
  apply Plan.valid_bind (collect_valid K r i (hp.sub ((ht _).mono Set.subset_union_left)))
  intro T hT a ha
  exact (hg.mono (Set.subset_union_left.trans hT)).add ha

theorem controlOracle_exec (hW : CommClass n ρ W)
    (K : ℕ) (hp : ∀ i j, 0 < (Network.lazy W ^ K) i j) (r : Fin n)
    (b y : E d) (s : NodeDraw n m) :
    Plan.Executes I W (1+4*K) (fun i => controlOracle K r i
      (fun j => gradient (I.F i j) b) (gradient I.f b) y s)
      (fun _ => control I b y s) := by
  have h := (Plan.executes_grad I W s (fun _ => y)).bind
    (f := fun i p => (collect K r i (p - gradient (I.F i (s i)) b)).bind
      (fun a => .done (gradient I.f b + a)))
    ((collect_exec I W hW K hp r (fun i =>
      gradient (I.F i (s i)) y - gradient (I.F i (s i)) b)).bind
      (f := fun _ a => .done (gradient I.f b + a)) (.done _))
  convert h using 1
  · funext i
    rfl
  · rfl

def parallelEpoch (K : ℕ) (r i : Fin n) (L a : ℝ) (R : ℕ)
    (b : E d) (s : Draws (NodeDraw n m) R) : Plan m d (E d) :=
  (readTable b (List.finRange m)).bind fun table =>
  (collect K r i (average table)).bind fun g =>
  inner (controlOracle K r i table g) L a R s b b

theorem parallelEpoch_valid (K : ℕ) (r i : Fin n) (L a : ℝ) (hL : 0 < L) (R : ℕ)
    {S : Set (Fin d)} {b : E d} (hb : Supported S b) (s : Draws (NodeDraw n m) R) :
    (parallelEpoch K r i L a R b s).Valid Supported S := by
  unfold parallelEpoch
  apply Plan.valid_bind (read_valid hb _)
  intro T hST table htable
  apply Plan.valid_bind (collect_valid K r i (average_supported htable))
  intro U hTU g hg
  have hbu := hb.mono (hST.trans hTU)
  exact inner_valid _ hL a (B := U)
    (fun V hUV y hy j => controlOracle_valid K r i
      (fun l => (htable l).mono (hTU.trans hUV)) (hg.mono hUV) hy j)
    R s (subset_refl _) hbu hbu

theorem parallelEpoch_exec (hI : I.InClass L μ Δ) (hW : CommClass n ρ W)
    (K : ℕ) (hp : ∀ i j, 0 < (Network.lazy W ^ K) i j) (r : Fin n)
    (L' a : ℝ) (R : ℕ) (b : E d) (s : Draws (NodeDraw n m) R) :
    Plan.Executes I W (m+4*K+R*(2+4*K)) (fun i => parallelEpoch K r i L' a R b s)
      (fun _ => (evaluate I L' a (control I b) R s b b).mean) := by
  unfold parallelEpoch
  rw [show m+4*K+R*(2+4*K) = m+(4*K+R*((1+4*K)+1)) by
    rw [show 2+4*K = (1+4*K)+1 by omega]; omega]
  apply (table_exec (I := I) (W := W) (fun _ => b)).bind
  apply (collect_exec I W hW K hp r (fun i => localGrad I i b)).bind
  rw [← fullGrad hI b]
  exact inner_exec _ L' a _ _ (fun y s => controlOracle_exec hW K hp r b y s) R s b b

end Schedule
end ULB63
