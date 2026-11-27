import Theory.Blocks

open Matrix

set_option maxHeartbeats 1300000

noncomputable section

namespace ULB63
namespace Blocks

structure Packing (n b s : ℕ) : Prop where
  depth : 0 < b
  each : 0 < s
  fits : b*s ≤ n
  fills : n < (b+1)*s

namespace Packing

variable {n b s : ℕ} (P : Packing n b s)

def block (i : Fin n) : Fin b := owner P.depth s i

def members (a : Fin b) : Finset (Fin n) := fiber P.depth s a

theorem mem_members (i : Fin n) (a : Fin b) : i ∈ P.members a ↔ P.block i = a :=
  mem_fiber P.depth s a i

def weight (a : Fin b) : ℝ := (P.members a).card

theorem weight_lo (a : Fin b) : (s:ℝ) ≤ P.weight a := by
  change (s:ℝ) ≤ (size n P.depth s a:ℝ)
  exact_mod_cast (size_range P.depth P.each P.fits P.fills a).1

theorem weight_pos (a : Fin b) : 0 < P.weight a :=
  (show (0:ℝ)<s by exact_mod_cast P.each).trans_le (P.weight_lo a)

theorem weight_sum : ∑ a, P.weight a = n := by
  change ∑ a : Fin b, (size n P.depth s a:ℝ) = n
  exact_mod_cast sum_sizes n P.depth s

def flow (a j : Fin b) : ℝ :=
  (s:ℝ)*Path.matrix b a j + if a=j then P.weight a-s else 0

def matrix : Matrix (Fin n) (Fin n) ℝ := fun i j =>
  P.flow (P.block i) (P.block j)/(P.weight (P.block i)*P.weight (P.block j))

def project (v : Fin n → ℝ) (a : Fin b) : ℝ :=
  (∑ i ∈ P.members a, v i)/P.weight a

def step (u : Fin b → ℝ) (a : Fin b) : ℝ :=
  (s:ℝ)/P.weight a * (Path.matrix b).mulVec u a + (P.weight a-s)/P.weight a * u a

def weighted (u : Fin b → ℝ) : ℝ := ∑ j, P.weight j*(u j)^2

theorem mul_project (v : Fin n → ℝ) (a : Fin b) :
    P.weight a * P.project v a = ∑ i ∈ P.members a, v i := by
  unfold project
  field_simp [(P.weight_pos a).ne']

theorem sum_repeat (f : Fin b → ℝ) :
    ∑ i : Fin n, f (P.block i) = ∑ a, P.weight a*f a := by
  rw [← sum_partition P.depth s]
  apply Finset.sum_congr rfl
  intro a _
  change ∑ i ∈ P.members a, f (P.block i) = _
  have H : ∑ i ∈ P.members a, f (P.block i) = ∑ _ ∈ P.members a, f a := by
    apply Finset.sum_congr rfl
    intro i hi
    rw [(P.mem_members i a).1 hi]
  rw [H]
  simp [weight, Finset.sum_const, nsmul_eq_mul]

theorem apply_matrix (v : Fin n → ℝ) (i : Fin n) :
    P.matrix.mulVec v i = P.step (P.project v) (P.block i) := by
  let a := P.block i
  change ∑ j, P.flow a (P.block j)/(P.weight a*P.weight (P.block j))*v j = _
  rw [← sum_partition P.depth s]
  have H (j : Fin b) : (∑ i ∈ P.members j,
      P.flow a (P.block i)/(P.weight a*P.weight (P.block i))*v i) =
      (1/P.weight a)*P.flow a j*P.project v j := by
    have hx : ∑ i ∈ P.members j, P.flow a (P.block i)/(P.weight a*P.weight (P.block i))*v i =
        ∑ i ∈ P.members j, P.flow a j/(P.weight a*P.weight j)*v i := by
      apply Finset.sum_congr rfl
      intro k hk
      rw [(P.mem_members k j).1 hk]
    rw [hx, ← Finset.mul_sum, ← P.mul_project v j]
    field_simp [(P.weight_pos a).ne', (P.weight_pos j).ne']
  simp only [← show ∀ j, P.members j = fiber P.depth s j from fun _ => rfl, H]
  change (∑ j, (1/P.weight a)*P.flow a j*P.project v j) = P.step (P.project v) a
  have HH (j : Fin b) : (1/P.weight a)*P.flow a j*P.project v j =
      ((s:ℝ)/P.weight a)*(Path.matrix b a j * P.project v j) +
      (if j=a then (P.weight a-s)/P.weight a * P.project v a else 0) := by
    unfold flow
    by_cases hj : j=a
    · subst j
      simp only [ite_true]
      ring
    · have hh : a ≠ j := fun he => hj he.symm
      simp only [ite_eq_right hj, ite_eq_right hh]
      ring
  simp only [HH, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_ite_eq', Finset.mem_univ,
    ite_true, step, Matrix.mulVec, dotProduct]

theorem matrix_nonneg (i j : Fin n) : 0 ≤ P.matrix i j := by
  unfold matrix flow
  apply div_nonneg
  · apply add_nonneg (mul_nonneg (Nat.cast_nonneg _) (Path.nonneg _ _))
    split
    · exact sub_nonneg.mpr (P.weight_lo _)
    · exact le_rfl
  · exact mul_nonneg (P.weight_pos _).le (P.weight_pos _).le

theorem matrix_symm (i j : Fin n) : P.matrix i j = P.matrix j i := by
  unfold matrix
  rw [mul_comm (P.weight (P.block j)) (P.weight (P.block i))]
  congr 1
  unfold flow
  rw [Path.symm_entry (P.block j) (P.block i)]
  by_cases he : P.block i = P.block j
  · rw [he]
  · have he' : P.block j ≠ P.block i := fun h => he h.symm
    rw [ite_eq_right he, ite_eq_right he']

theorem matrix_rows (i : Fin n) : ∑ j, P.matrix i j = 1 := by
  have he (a : Fin b) : P.project (fun _ => 1) a = 1 := by
    simp only [project, Finset.sum_const, nsmul_one]
    change P.weight a / P.weight a = 1
    exact div_self (P.weight_pos a).ne'
  have h := P.apply_matrix (fun _ => 1) i
  simp only [Matrix.mulVec, dotProduct, mul_one] at h
  rw [h]
  unfold step
  simp only [he, Path.apply_vec, mul_one]
  field_simp [(P.weight_pos _).ne']
  ring

theorem support_blocks (i j : Fin n) (h : P.matrix i j ≠ 0) :
    (P.block i).val ≤ (P.block j).val+1 ∧ (P.block j).val ≤ (P.block i).val+1 := by
  by_cases he : P.block i=P.block j
  · rw [he]; omega
  · apply Path.support
    intro hz
    apply h
    simp [matrix, flow, hz, he]

private theorem project_bound (v : Fin n → ℝ) : P.weighted (P.project v) ≤ norm2 v ^ 2 := by
  have H (a : Fin b) : P.weight a*(P.project v a)^2 ≤ ∑ j ∈ P.members a, (v j)^2 := by
    have hh := sq_sum_le_card_mul_sum_sq (s := P.members a) (f := v)
    change (∑ i ∈ P.members a, v i)^2 ≤ P.weight a*∑ j ∈ P.members a, (v j)^2 at hh
    rw [← P.mul_project, mul_pow] at hh
    apply le_of_mul_le_mul_left _ (P.weight_pos a)
    convert hh using 1
    ring
  have hz := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin b))) fun a _ => H a
  rwa [show (∑ j : Fin b, ∑ i ∈ P.members j, (v i)^2) = ∑ i : Fin n, (v i)^2
      from sum_partition P.depth s _, ← Path.norm2_sq] at hz

private theorem wt_poincare (v : Fin b → ℝ) (hv : ∑ j, P.weight j*v j=0) :
    P.weighted v ≤ 2*(s:ℝ)*(b:ℝ)^2*Path.edgeEnergy v := by
  let z := v ⟨0,P.depth⟩
  have H (a : Fin b) := mul_le_mul_of_nonneg_left (Path.point_sq P.depth v a) (P.weight_pos a).le
  have HH := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin b))) fun a _ => H a
  have hs : ∑ a, P.weight a*(v a-z)^2 = P.weighted v+(n:ℝ)*z^2 := by
    calc
      _ = ∑ a, (P.weight a*(v a)^2-2*z*(P.weight a*v a)+P.weight a*z^2) :=
        Finset.sum_congr rfl fun a _ => by ring
      _ = _ := by
        rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hv, mul_zero, sub_zero,
          ← Finset.sum_mul, P.weight_sum]
        rfl
  rw [hs, ← Finset.sum_mul, P.weight_sum] at HH
  have hn : (n:ℝ) ≤ 2*(s:ℝ)*(b:ℝ) := by
    have h1 := P.fills.le.trans (Nat.mul_le_mul_right s (by have := P.depth; omega : b+1 ≤ 2*b))
    exact_mod_cast (show n ≤ 2*s*b from by simpa [Nat.mul_comm, Nat.mul_left_comm] using h1)
  have ht := mul_le_mul_of_nonneg_right hn
    (show 0 ≤ (b:ℝ)*Path.edgeEnergy v by unfold Path.edgeEnergy; positivity)
  nlinarith [show (0:ℝ) ≤ n*z^2 by positivity]

private theorem wt_loss (u : Fin b → ℝ) :
    P.weighted (P.step u) + (s:ℝ)*Path.edgeEnergy u/8 ≤ P.weighted u := by
  have H (a : Fin b) : P.weight a*(P.step u a)^2 ≤
      (s:ℝ)*((Path.matrix b).mulVec u a)^2+(P.weight a-s)*(u a)^2 := by
    let x := (Path.matrix b).mulVec u a
    let y := u a
    let r := P.weight a
    have hr : 0 < r := P.weight_pos _
    have hs : 0 ≤ r-(s:ℝ) := sub_nonneg.mpr (P.weight_lo _)
    change r*((s:ℝ)/r*x+(r-s)/r*y)^2 ≤ _
    have he : r*((s:ℝ)/r*x+(r-s)/r*y)^2 =
        (s:ℝ)*x^2+(r-s)*y^2 - ((s:ℝ)*(r-s)/r)*(x-y)^2 := by
      field_simp
      ring
    rw [he]
    exact sub_le_self _ (by positivity)
  have HH := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin b))) fun a _ => H a
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, sub_mul, Finset.sum_sub_distrib,
    ← Path.norm2_sq] at HH
  have h := mul_le_mul_of_nonneg_left (Path.dissipate u) (show (0:ℝ) ≤ s by positivity)
  unfold weighted
  linarith

private theorem centered_contraction (v : Fin n → ℝ) (hv : ∑ i, v i = 0)
    {ρ : ℝ} (hρ : 1-1/(32*(b:ℝ)^2) ≤ ρ) :
    norm2 (P.matrix.mulVec v) ≤ ρ*norm2 v := by
  let u := P.project v
  have hu : ∑ a, P.weight a*u a = 0 := by
    simp only [u, P.mul_project]
    exact (sum_partition P.depth s v).trans hv
  have hp := P.wt_poincare u hu
  have hl := P.wt_loss u
  have hu2 := P.project_bound v
  have he : norm2 (P.matrix.mulVec v)^2 = P.weighted (P.step u) := by
    rw [Path.norm2_sq]
    simp only [P.apply_matrix]
    exact P.sum_repeat (fun a => (P.step u a)^2)
  have hb : (0:ℝ)<b := by exact_mod_cast P.depth
  have hb₁ : (1:ℝ)≤b := by exact_mod_cast P.depth
  let w := 1/(32*(b:ℝ)^2)
  have hw : 0 ≤ w := by dsimp [w]; positivity
  have hw₁ : w ≤ 1 := (div_le_one (by positivity)).2 (by nlinarith)
  have hdis : 2*w*P.weighted u ≤ (s:ℝ)*Path.edgeEnergy u/8 := by
    have hh : P.weighted u/(16*(b:ℝ)^2) ≤ (s:ℝ)*Path.edgeEnergy u/8 := by
      apply (div_le_iff₀ (by positivity)).2
      convert hp using 1
      ring
    convert hh using 1
    dsimp [w]
    ring
  have hsq : norm2 (P.matrix.mulVec v)^2 ≤ (1-w)^2 * norm2 v^2 := by
    rw [he]
    have h1 : 0 ≤ w^2*P.weighted u :=
      mul_nonneg (sq_nonneg _) (Finset.sum_nonneg fun a _ =>
        mul_nonneg (P.weight_pos a).le (sq_nonneg _))
    have h2 := mul_le_mul_of_nonneg_left hu2 (sq_nonneg (1-w))
    nlinarith
  have hs : norm2 (P.matrix.mulVec v) ≤ (1-w)*norm2 v := by
    apply (sq_le_sq₀ (Path.norm2_nonneg _) (mul_nonneg (sub_nonneg.mpr hw₁) (Path.norm2_nonneg _))).1
    rw [mul_pow]
    exact hsq
  exact hs.trans (mul_le_mul_of_nonneg_right hρ (Path.norm2_nonneg _))

theorem commClass {ρ : ℝ} (hρ : 1-1/(32*(b:ℝ)^2) ≤ ρ) :
    CommClass n ρ P.matrix where
  nonneg := P.matrix_nonneg
  symm := by ext i j; exact P.matrix_symm _ _
  row_sum := P.matrix_rows
  contraction v := by
    have hn : 0 < n := (Nat.mul_pos P.depth P.each).trans_le P.fits
    have hnr : (0:ℝ)<n := by exact_mod_cast hn
    let z := (1/(n:ℝ))*∑ i, v i
    let u : Fin n → ℝ := fun i => v i-z
    have hz : (n:ℝ)*z = ∑ i, v i := by dsimp [z]; field_simp
    have hsu : ∑ i, u i = 0 := by
      simp [u, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, hz]
    have hproj : norm2 u ≤ norm2 v := by
      apply (sq_le_sq₀ (Path.norm2_nonneg _) (Path.norm2_nonneg _)).1
      rw [Path.norm2_sq, Path.norm2_sq]
      calc
        _ = ∑ i, (v i^2-2*z*v i+z^2) := Finset.sum_congr rfl fun _ _ => by dsimp [u]; ring
        _ = (∑ i, v i^2)-(n:ℝ)*z^2 := by
          rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← hz]
          simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring
        _ ≤ _ := sub_le_self _ (by positivity)
    have hcenter : (P.matrix-J n).mulVec v = P.matrix.mulVec u := by
      ext i
      rw [Matrix.sub_mulVec]
      change (∑ j, P.matrix i j*v j)-(∑ j, (1/(n:ℝ))*v j) = ∑ j, P.matrix i j*(v j-z)
      simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.sum_mul, P.matrix_rows,
        one_mul, ← show (1/(n:ℝ))*∑ i, v i = z by rfl]
    rw [hcenter]
    have hb : (1:ℝ) ≤ b := by exact_mod_cast P.depth
    have hp : 0 ≤ ρ := by
      apply le_trans _ hρ
      have hh : (1:ℝ)/(32*(b:ℝ)^2) ≤ 1 := (div_le_one (by positivity)).2 (by nlinarith)
      linarith
    exact (P.centered_contraction u hsu hρ).trans (mul_le_mul_of_nonneg_left hproj hp)

end Packing
end Blocks
end ULB63
