import Theory.Protocol
import Theory.Network
import Theory.Composite

open Matrix

noncomputable section

namespace ULB63
namespace Aggregation

open Protocol Composite

variable {n m d : ℕ}

def applyMatrix (Q : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → E d) (i : Fin n) : E d :=
  WithLp.toLp 2 (fun k => Q.mulVec (fun j => x j k) i)

@[simp] theorem applyMatrix_coord (Q : Matrix (Fin n) (Fin n) ℝ)
    (x : Fin n → E d) (i : Fin n) (k : Fin d) :
    applyMatrix Q x i k = ∑ j, Q i j * x j k := rfl

theorem vec_sum_apply {ι : Type*} (s : Finset ι) (x : ι → E d) (k : Fin d) :
    (∑ i ∈ s, x i) k = ∑ i ∈ s, x i k := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih => simp [Finset.sum_insert ha, ih]

theorem applyMatrix_one (x : Fin n → E d) :
    applyMatrix 1 x = x := by
  funext i
  ext j
  change (1 : Matrix (Fin n) (Fin n) ℝ).mulVec (fun u => x u j) i = _
  rw [Matrix.one_mulVec]

theorem applyMatrix_mul (Q R : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → E d) :
    applyMatrix Q (applyMatrix R x) = applyMatrix (Q*R) x := by
  funext i
  ext j
  change (Q.mulVec (R.mulVec (fun u => x u j))) i = _
  rw [Matrix.mulVec_mulVec]
  rfl

theorem lazy_applyMatrix (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → E d) (i : Fin n) :
    (1/2 : ℝ) • (x i + ∑ j, W i j • x j) = applyMatrix (Network.lazy W) x i := by
  ext k
  change ((1/2 : ℝ) • (x i + ∑ j, W i j • x j) : E d) k =
    (Network.lazy W).mulVec (fun j => x j k) i
  rw [Network.lazy, Matrix.smul_mulVec, Matrix.add_mulVec, Matrix.one_mulVec]
  simp [vec_sum_apply, Matrix.mulVec, dotProduct, mul_add]

def mixRounds : ℕ → E d → Plan m d (E d)
  | 0, x => .done x
  | k+1, x => .mix x (fun y => mixRounds k ((1/2 : ℝ) • (x+y)))

theorem mixRounds_valid {s : Set (Fin d)} {x : E d} (h : Supported s x) (k : ℕ) :
    (mixRounds (m := m) k x).Valid Supported s := by
  induction k generalizing s x with
  | zero => exact h
  | succ k ih =>
    refine ⟨h, fun y => ih ?_⟩
    exact ((h.mono Set.subset_union_left).add Set.subset_union_right).smul _

theorem mixRounds_exec (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (k : ℕ)
    (x : Fin n → E d) :
    Plan.Executes I W k (fun i => mixRounds k (x i)) (applyMatrix (Network.lazy W ^ k) x) := by
  induction k generalizing x with
  | zero => simpa [mixRounds, applyMatrix_one] using Plan.Executes.done (I := I) (W := W) x
  | succ k ih =>
    refine .next (fun _ => trivial) ?_
    have H := ih (applyMatrix (Network.lazy W) x)
    rw [applyMatrix_mul, ← pow_succ] at H
    convert H using 1
    funext i
    change mixRounds k ((1/2 : ℝ) • (x i + ∑ j, W i j • x j)) = _
    rw [lazy_applyMatrix]

def mask (x : E d) : E d := WithLp.toLp 2 (fun k => if x k = 0 then 0 else 1)

def divide (x y : E d) : E d := WithLp.toLp 2 (fun k => x k / y k)

def rootOnly (r i : Fin n) (x : E d) : E d := if i = r then x else 0

theorem supported_mask {s : Set (Fin d)} {x : E d} (h : Supported s x) :
    Supported s (mask x) := by
  rw [supported_iff] at *
  intro k hk
  simp [mask, h k hk]

theorem supported_divide {s : Set (Fin d)} {x : E d} (h : Supported s x) (y : E d) :
    Supported s (divide x y) := by
  rw [supported_iff] at *
  intro k hk
  simp [divide, h k hk]

theorem supported_rootOnly {s : Set (Fin d)} {x : E d} (h : Supported s x) (r i : Fin n) :
    Supported s (rootOnly r i x) := by
  unfold rootOnly
  split
  · exact h
  · exact supported_zero _

/-- Four powers of the lazy matrix, with coordinatewise calibration of the symmetric root
column. `K` is public, but no node needs to know a nonlocal entry of the matrix. -/
def collect (K : ℕ) (r i : Fin n) (x : E d) : Plan m d (E d) :=
  (mixRounds K (mask x)).bind fun u =>
  (mixRounds K (rootOnly r i (mask u))).bind fun b =>
  (mixRounds K (divide x b)).bind fun w =>
  (mixRounds K (rootOnly r i ((n : ℝ)⁻¹ • w))).bind fun s =>
  .done (divide s b)

theorem collect_valid (K : ℕ) (r i : Fin n) {S : Set (Fin d)} {x : E d}
    (h : Supported S x) : (collect (m := m) K r i x).Valid Supported S := by
  unfold collect
  apply Plan.valid_bind (mixRounds_valid (supported_mask h) _)
  intro T hST u hu
  apply Plan.valid_bind (mixRounds_valid (supported_rootOnly (supported_mask hu) r i) _)
  intro U hTU b hb
  apply Plan.valid_bind (mixRounds_valid (supported_divide (h.mono (hST.trans hTU)) b) _)
  intro V hUV w hw
  apply Plan.valid_bind (mixRounds_valid (supported_rootOnly (hw.smul _) r i) _)
  intro X hVX z hz
  exact supported_divide hz b

theorem applyMatrix_rootOnly (Q : Matrix (Fin n) (Fin n) ℝ) (r i : Fin n)
    (x : Fin n → E d) (j : Fin d) :
    applyMatrix Q (fun u => rootOnly r u (x u)) i j = Q i r * x r j := by
  rw [applyMatrix_coord]
  classical
  calc
    _ = ∑ a, if a = r then Q i a * x a j else 0 := by
      apply Finset.sum_congr rfl
      intro a _
      by_cases ha : a = r <;> simp [rootOnly, ha]
    _ = _ := by simp

private theorem calibration (Q : Matrix (Fin n) (Fin n) ℝ)
    (hp : ∀ i j, 0 < Q i j) (hs : Q.transpose = Q)
    (r : Fin n) (x : Fin n → E d) :
    let u := applyMatrix Q (fun i => mask (x i))
    let b := applyMatrix Q (fun i => rootOnly r i (mask (u i)))
    let w := applyMatrix Q (fun i => divide (x i) (b i))
    let s := applyMatrix Q (fun i => rootOnly r i ((n : ℝ)⁻¹ • w i))
    (fun i => divide (s i) (b i)) = fun _ => average x := by
  intro u b w s
  funext i
  ext k
  have hsymm (j : Fin n) : Q r j = Q j r := congrFun (congrFun hs j) r
  change s i k / b i k = average x k
  rw [show s i k = Q i r * ((n : ℝ)⁻¹ * w r k) by
    dsimp only [s]; rw [applyMatrix_rootOnly]; rfl]
  by_cases hz : ∀ j : Fin n, x j k = 0
  · have hb (j : Fin n) : b j k = 0 := by
      have hu : u r k = 0 := by simp [u, mask, hz]
      rw [show b j k = Q j r * mask (u r) k from applyMatrix_rootOnly _ _ _ _ _]
      simp [mask, hu]
    rw [hb]
    simp [average, vec_sum_apply, hz]
  · push Not at hz
    obtain ⟨j, hj⟩ := hz
    have hu : 0 < u r k := by
      rw [show u r k = ∑ t, Q r t * mask (x t) k from applyMatrix_coord _ _ _ _]
      refine Finset.sum_pos' ?_ ⟨j, Finset.mem_univ j, ?_⟩
      · intro a _
        have hmask : 0 ≤ mask (x a) k := by
          change 0 ≤ (if x a k = 0 then 0 else 1)
          split <;> norm_num
        exact mul_nonneg (hp r a).le hmask
      · simpa [mask, hj] using hp r j
    have hb (j : Fin n) : b j k = Q j r := by
      rw [show b j k = Q j r * mask (u r) k from applyMatrix_rootOnly _ _ _ _ _]
      simp [mask, hu.ne']
    have hw : w r k = ∑ j, x j k := by
      rw [show w r k = ∑ j, Q r j * (divide (x j) (b j)) k from applyMatrix_coord _ _ _ _]
      apply Finset.sum_congr rfl
      intro j _
      change Q r j * (x j k / b j k) = _
      rw [hb, hsymm j]
      field_simp [(hp j r).ne']
    rw [hb, hw]
    have he : average x k = (n : ℝ)⁻¹ * ∑ j, x j k := by
      simp [average, vec_sum_apply]
    rw [he]
    field_simp [(hp i r).ne']

theorem collect_exec (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) {ρ : ℝ}
    (hW : CommClass n ρ W) (K : ℕ) (hp : ∀ i j, 0 < (Network.lazy W ^ K) i j)
    (r : Fin n) (x : Fin n → E d) :
    Plan.Executes I W (4*K) (fun i => collect K r i (x i)) (fun _ => average x) := by
  have hs : (Network.lazy W ^ K).transpose = Network.lazy W ^ K := by
    rw [Matrix.transpose_pow, Network.lazy_symm hW]
  have H := calibration _ hp hs r x
  rw [show 4*K = K+(K+(K+(K+0))) by omega]
  unfold collect
  apply (mixRounds_exec I W K _).bind
  apply (mixRounds_exec I W K _).bind
  apply (mixRounds_exec I W K _).bind
  apply (mixRounds_exec I W K _).bind
  rw [← H]
  exact .done _

end Aggregation
end ULB63
