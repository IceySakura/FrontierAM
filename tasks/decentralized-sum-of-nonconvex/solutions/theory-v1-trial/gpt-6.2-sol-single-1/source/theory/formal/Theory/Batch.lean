import Theory.Upper

open scoped ENNReal

set_option maxHeartbeats 1000000
set_option linter.style.haveILetI false

noncomputable section

namespace ULB63
namespace Batch

open Protocol Composite FiniteMean Aggregation Schedule Epoch Upper

variable {d n m B : ℕ} {L μ Δ ρ ε : ℝ}
variable {I : Instance n m d} {W : Matrix (Fin n) (Fin n) ℝ}

abbrev Choice (n m B : ℕ) := Fin B → NodeDraw n m

def estimate (I : Instance n m d) (b y : E d) (s : Choice n m B) : E d :=
  average (fun j => Schedule.control I b y (s j))

private theorem centered [Nonempty (Fin B)] (b y : E d) (s : Choice n m B) :
    estimate I b y s - gradient I.f y =
      average (fun j => Schedule.control I b y (s j) - gradient I.f y) := by
  rw [average_sub, average_const]
  rfl

theorem mean [Nonempty (Fin B)] [Nonempty (Fin m)] (hI : I.InClass L μ Δ) (b y : E d) :
    average (fun s : Choice n m B => estimate I b y s - gradient I.f y) = 0 := by
  let e (_ : Fin B) (s : NodeDraw n m) := Schedule.control I b y s - gradient I.f y
  have h (i : Fin B) : average (e i) = 0 := control_mean hI b y
  simp only [centered]
  change average (fun s : Choice n m B => (Fintype.card (Fin B) : ℝ)⁻¹ • ∑ i, e i (s i)) = _
  rw [average_smul, indep_sum_centered _ e h, smul_zero]

theorem variance [Nonempty (Fin B)] [Nonempty (Fin n)] [Nonempty (Fin m)]
    (hI : I.InClass L μ Δ) (b y : E d) :
    average (fun s : Choice n m B => ‖estimate I b y s - gradient I.f y‖^2) ≤
      (L^2 / ((n:ℝ)*(B:ℝ))) * ‖y-b‖^2 := by
  let e (_ : Fin B) (s : NodeDraw n m) := Schedule.control I b y s - gradient I.f y
  have hz (i : Fin B) : average (e i) = 0 := control_mean hI b y
  have hb : (0 : ℝ) < B := by
    have h := Fintype.card_pos (α := Fin B)
    simpa using (show (0 : ℝ) < (Fintype.card (Fin B) : ℝ) by exact_mod_cast h)
  simp only [centered]
  change average (fun s : Choice n m B => ‖(Fintype.card (Fin B) : ℝ)⁻¹ • ∑ i, e i (s i)‖^2) ≤ _
  simp only [Fintype.card_fin, norm_smul, Real.norm_eq_abs, abs_of_nonneg (inv_nonneg.mpr hb.le),
    mul_pow, average_mul, indep_sum_square B e hz]
  calc
    _ ≤ ((B : ℝ)⁻¹)^2 * ∑ _ : Fin B, ((L^2/(n:ℝ))*‖y-b‖^2) :=
      mul_le_mul_of_nonneg_left (Finset.sum_le_sum (fun j _ => control_variance hI b y)) (sq_nonneg _)
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      have hn : (0 : ℝ) < n := by
        have h := Fintype.card_pos (α := Fin n)
        simpa using (show (0 : ℝ) < (Fintype.card (Fin n) : ℝ) by exact_mod_cast h)
      field_simp

private theorem average_swap {α β : Type*} [Fintype α] [Fintype β] (f : α → β → E d) :
    (average fun a => average fun b => f a b) = average fun b => average fun a => f a b := by
  simp only [average, Finset.smul_sum, smul_smul]
  rw [Finset.sum_comm]
  simp only [mul_comm]

theorem estimate_eq [Nonempty (Fin B)] (b y : E d) (s : Choice n m B) :
    estimate I b y s =
      gradient I.f b + average (fun i =>
        average (fun j => gradient (I.F i (s j i)) y - gradient (I.F i (s j i)) b)) := by
  simp only [estimate, Schedule.control, average_add, average_const]
  rw [average_swap]

def localSum {α : Type} (pick : α → Fin m) (table : Fin m → E d) (y : E d) :
    List α → Plan m d (E d)
  | [] => .done 0
  | j::js => .grad (pick j) y (fun p =>
      (localSum pick table y js).bind (fun a => .done (p-table (pick j)+a)))

theorem local_valid {α : Type} (pick : α → Fin m) {T : Set (Fin d)}
    {table : Fin m → E d} {y : E d} (ht : TableSupported T table) (hy : Supported T y)
    (ls : List α) :
    (localSum pick table y ls).Valid Supported T := by
  induction ls generalizing T with
  | nil => exact supported_zero _
  | cons j js ih =>
    refine ⟨hy, fun p => ?_⟩
    have hp : Supported (T ∪ supp p) p := Set.subset_union_right
    have hT : T ⊆ T ∪ supp p := Set.subset_union_left
    apply Plan.valid_bind (ih (fun l => (ht l).mono hT) (hy.mono hT))
    intro U hU a ha
    exact ((hp.mono hU).sub ((ht _).mono (hT.trans hU))).add ha

theorem local_exec {α : Type} (pick : Fin n → α → Fin m) (table : Fin n → Fin m → E d)
    (y : E d) (ls : List α) :
    Plan.Executes I W ls.length (fun i => localSum (pick i) (table i) y ls)
      (fun i => (ls.map (fun j => gradient (I.F i (pick i j)) y - table i (pick i j))).sum) := by
  induction ls with
  | nil => exact .done _
  | cons j js ih =>
    let f (i : Fin n) (p : E d) :=
      (localSum (pick i) (table i) y js).bind (fun a => Plan.done (p-table i (pick i j)+a))
    have h := (Plan.executes_grad I W (fun i => pick i j) (fun _ => y)).bind (f := f)
      (ih.bind (f := fun i a => .done (gradient (I.F i (pick i j)) y-table i (pick i j)+a)) (.done _))
    rw [show (j::js).length = 1+(js.length+0) by simp; omega]
    convert h using 1 <;> rfl

def oracle (D : ℕ) (root i : Fin n) (table : Fin m → E d)
    (g : E d) (y : E d) (s : Choice n m B) : Plan m d (E d) :=
  (localSum (fun j => s j i) table y (List.finRange B)).bind fun p =>
  (collect D root i ((B:ℝ)⁻¹ • p)).bind fun diff =>
  .done (g+diff)

theorem oracle_valid (D : ℕ) (root i : Fin n) {S : Set (Fin d)} {table : Fin m → E d}
    {g y : E d} (ht : TableSupported S table) (hg : Supported S g) (hy : Supported S y)
    (s : Choice n m B) : (oracle D root i table g y s).Valid Supported S := by
  unfold oracle
  apply Plan.valid_bind (local_valid _ ht hy _)
  intro T hST p hp
  apply Plan.valid_bind (collect_valid _ _ _ (hp.smul _))
  intro U hTU v hv
  exact (hg.mono (hST.trans hTU)).add hv

theorem oracle_exec [Nonempty (Fin B)] (hW : CommClass n ρ W)
    (D : ℕ) (hp : ∀ i j, 0 < (Network.lazy W ^ D) i j) (root : Fin n)
    (b y : E d) (s : Choice n m B) :
    Plan.Executes I W (B+4*D) (fun i => oracle D root i
      (fun j => gradient (I.F i j) b) (gradient I.f b) y s)
      (fun _ => estimate I b y s) := by
  let a (i : Fin n) (j : Fin B) := gradient (I.F i (s j i)) y - gradient (I.F i (s j i)) b
  have H : Plan.Executes I W B (fun i => localSum (fun j => s j i)
      (fun k => gradient (I.F i k) b) y (List.finRange B)) (fun i => ∑ j, a i j) := by
    simpa [List.length_finRange, List.finRange, List.map_ofFn, List.sum_ofFn, Function.comp_def, a] using
      (local_exec (I := I) (W := W) (fun i j => s j i) (fun i j => gradient (I.F i j) b)
        y (List.finRange B))
  rw [estimate_eq]
  unfold oracle
  apply H.bind
  have H' := (collect_exec I W hW D hp root (fun i => average (a i))).bind
    (f := fun _ v => Plan.done (gradient I.f b+v)) (.done _)
  simpa only [average, Fintype.card_fin, Nat.add_zero, a] using H'

def epoch (D : ℕ) (root i : Fin n) (L a : ℝ) (R : ℕ)
    (b : E d) (s : Draws (Choice n m B) R) : Plan m d (E d) :=
  (readTable b (List.finRange m)).bind fun table =>
  (collect D root i (average table)).bind fun g =>
  inner (oracle D root i table g) L a R s b b

theorem epoch_valid (D : ℕ) (root i : Fin n) (L a : ℝ) (hL : 0 < L) (R : ℕ)
    {S : Set (Fin d)} {b : E d} (hb : Supported S b) (s : Draws (Choice n m B) R) :
    (epoch D root i L a R b s).Valid Supported S := by
  unfold epoch
  apply Plan.valid_bind (read_valid hb _)
  intro T hST table htable
  apply Plan.valid_bind (collect_valid D root i (average_supported htable))
  intro U hTU g hg
  have hbU := hb.mono (hST.trans hTU)
  exact inner_valid _ hL a (B := U) (fun V hUV _ hy j => oracle_valid D root i
    (fun l => (htable l).mono (hTU.trans hUV)) (hg.mono hUV) hy j) R s (subset_refl _) hbU hbU

theorem epoch_exec_from_full [Nonempty (Fin B)]
    (hg : ∀ x, gradient I.f x = average (fun i => localGrad I i x)) (hW : CommClass n ρ W)
    (D : ℕ) (hp : ∀ i j, 0 < (Network.lazy W ^ D) i j) (root : Fin n)
    (L' a : ℝ) (R : ℕ) (b : E d) (s : Draws (Choice n m B) R) :
    Plan.Executes I W (m+4*D+R*(B+1+4*D)) (fun i => epoch D root i L' a R b s)
      (fun _ => (evaluate I L' a (estimate I b) R s b b).mean) := by
  unfold epoch
  rw [show m+4*D+R*(B+1+4*D) = m+(4*D+R*((B+4*D)+1)) by
    rw [show B+1+4*D = (B+4*D)+1 by omega]; omega]
  apply (table_exec (I := I) (W := W) (fun _ => b)).bind
  apply (collect_exec I W hW D hp root (fun i => localGrad I i b)).bind
  rw [← hg b]
  exact inner_exec _ L' a _ _ (fun y s => oracle_exec hW D hp root b y s) R s b b

theorem epoch_exec [Nonempty (Fin B)] (hI : I.InClass L μ Δ) (hW : CommClass n ρ W)
    (D : ℕ) (hp : ∀ i j, 0 < (Network.lazy W ^ D) i j) (root : Fin n)
    (L' a : ℝ) (R : ℕ) (b : E d) (s : Draws (Choice n m B) R) :
    Plan.Executes I W (m+4*D+R*(B+1+4*D)) (fun i => epoch D root i L' a R b s)
      (fun _ => (evaluate I L' a (estimate I b) R s b b).mean) :=
  epoch_exec_from_full (fullGrad hI) hW D hp root L' a R b s

theorem bound (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D R H B : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hB : 1 ≤ B) (hparallel : 8*(L/μ)*(R : ℝ) ≤ (n : ℝ)*(B : ℝ))
    (hH : (1/2 : ℝ)^H * Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤ ((H * (m+4*D+R*(B+1+4*D)) : ℕ) : ℕ∞) := by
  let root : Fin n := ⟨0, by omega⟩
  let a := Real.sqrt (μ/(8*L))
  have hLp := hμ.trans_le hL
  obtain ⟨ha, ha₁, ha₂⟩ := rate_parameters hμ hL
  letI : Nonempty (Fin n) := ⟨root⟩
  letI : Nonempty (Fin m) := ⟨⟨0, by omega⟩⟩
  letI : Nonempty (Fin B) := ⟨⟨0, by omega⟩⟩
  let τ := Draws (Choice n m B) R
  let once (d : ℕ) (I : Instance n m d) (x : E d) (s : τ) :=
    (evaluate I L a (estimate I x) R s x x).mean
  have hnr : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hbr : (0 : ℝ) < B := by exact_mod_cast (show 0 < B by omega)
  let C := L^2/((n:ℝ)*(B:ℝ))
  have hnoise : (2*C/(L*μ))*(R:ℝ) ≤ 1/4 := by
    have he : (2*C/(L*μ))*(R:ℝ) = (8*(L/μ)*(R:ℝ))/((4:ℝ)*n*B) := by
      dsimp [C]
      field_simp
      ring
    rw [he]
    exact (div_le_iff₀ (by positivity)).2 (by convert hparallel using 1; ring)
  apply fromSchedule (by omega : 0 < n) _ (Draws τ H)
    (fun _ info => repeating (epoch D root info.node L a R) H info.seed info.x0)
    (fun d I s => repeatEval (once d I) H s I.x0)
  · intro d info
    exact repeating_valid _ (fun _ _ hx s => epoch_valid D root info.node L a hLp R hx s)
      H info.seed (subset_refl _)
  · intro d I W hI hW s
    have hp := Network.certified_radius (by omega) hρ hρ₁ hW hD
    apply repeating_exec _ (once d I) _
    intro x t
    exact epoch_exec hI hW D hp root L a R x t
  · intro d I hI s
    obtain ⟨u, hu⟩ := hI.minimizer
    exact repeat_feasible (once d I)
      (fun x hx t => (evaluate_feasible hI hLp a (estimate I x) R t hx x).2) H s (initial_feasible hI hu)
  · intro d I hI u hu
    have hf (x : E d) (hx : feasible I x) (s : τ) : feasible I (once d I x s) :=
      (evaluate_feasible hI hLp a (estimate I x) R s hx x).2
    have hg (x : E d) (hx : feasible I x) :
        average (fun s => loss I u (once d I x s)) ≤ loss I u x / 2 :=
      contract hI hLp hμ hu hx ha ha₁ ha₂ (show 0 ≤ C by dsimp [C]; positivity)
        (estimate I x) (mean hI x) (variance hI x) R hR hnoise
    exact (repeat_contract (once d I) hf hg H (initial_feasible hI hu)).trans
      ((mul_le_mul_of_nonneg_left (initial_loss hI hu) (by positivity)).trans hH)

end Batch
end ULB63
