import Theory.Path

noncomputable section

namespace ULB63
namespace Blocks

variable {n b s : ℕ}

/-- All blocks except the last have size `s`. The last absorbs the remainder. -/
def owner (hb : 0 < b) (s : ℕ) (i : Fin n) : Fin b :=
  ⟨min (i.val/s) (b-1), (min_le_right _ _).trans_lt (by omega)⟩

def fiber (hb : 0 < b) (s : ℕ) (a : Fin b) : Finset (Fin n) :=
  Finset.univ.filter (fun i => owner hb s i = a)

def size (n : ℕ) (hb : 0 < b) (s : ℕ) (a : Fin b) : ℕ := (fiber (n := n) hb s a).card

theorem mem_fiber (hb : 0 < b) (s : ℕ) (a : Fin b) (i : Fin n) :
    i ∈ fiber hb s a ↔ owner hb s i = a := by simp [fiber]

theorem sum_partition {V : Type*} [AddCommMonoid V] (hb : 0 < b) (s : ℕ) (v : Fin n → V) :
    ∑ a : Fin b, ∑ i ∈ fiber hb s a, v i = ∑ i : Fin n, v i := by
  simpa only [fiber, Finset.filter_true_of_mem (fun _ _ => Finset.mem_univ _)] using
    (Finset.sum_fiberwise_eq_sum_filter (Finset.univ : Finset (Fin n))
      (Finset.univ : Finset (Fin b)) (owner hb s) v)

theorem sum_sizes (n : ℕ) (hb : 0 < b) (s : ℕ) : ∑ a, size n hb s a = n := by
  simpa only [size, fiber, Finset.card_univ, Fintype.card_fin] using
    (Finset.card_eq_sum_card_fiberwise (s := (Finset.univ : Finset (Fin n)))
      (t := (Finset.univ : Finset (Fin b))) (f := owner hb s)
      (fun _ _ => Finset.mem_univ _)).symm

private theorem interval_card (l u : ℕ) (hu : u ≤ n) :
    (Finset.univ.filter (fun i : Fin n => l ≤ i.val ∧ i.val < u)).card = u-l := by
  let t := Finset.univ.filter (fun i : Fin n => l ≤ i.val ∧ i.val < u)
  have H : t.map Fin.valEmbedding = Finset.Ico l u := by
    ext j
    rw [Finset.mem_map, Finset.mem_Ico]
    constructor
    · rintro ⟨i, hi, rfl⟩
      exact (Finset.mem_filter.mp hi).2
    · intro hj
      exact ⟨⟨j, hj.2.trans_le hu⟩, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩, rfl⟩
  rw [← Finset.card_map (f := Fin.valEmbedding) (s := t), H, Nat.card_Ico]

theorem size_range (hb : 0 < b) (hs : 0 < s) (hpack : b*s ≤ n) (hpack' : n < (b+1)*s)
    (a : Fin b) : s ≤ size n hb s a ∧ size n hb s a ≤ 2*s := by
  have ha := a.isLt
  let u := if a.val=b-1 then n else (a.val+1)*s
  have hu : u ≤ n := by
    unfold u
    split
    · exact le_rfl
    · exact (Nat.mul_le_mul_right s (by omega : a.val+1 ≤ b)).trans hpack
  have H : fiber (n := n) hb s a = Finset.univ.filter (fun i : Fin n =>
      a.val*s ≤ i.val ∧ i.val<u) := by
    ext i
    simp only [mem_fiber, Finset.mem_filter, Finset.mem_univ, true_and]
    rw [Fin.ext_iff]
    change min (i.val/s) (b-1) = a.val ↔ _
    have hlo : a.val ≤ i.val/s ↔ a.val*s ≤ i.val := Nat.le_div_iff_mul_le hs
    have hhi : i.val/s < a.val+1 ↔ i.val < (a.val+1)*s := Nat.div_lt_iff_lt_mul hs
    have hi := i.isLt
    by_cases hlast : a.val=b-1
    · simp only [u, hlast, ite_true]
      have hk : b-1 ≤ i.val/s ↔ (b-1)*s ≤ i.val := Nat.le_div_iff_mul_le hs
      omega
    · simp only [u, hlast, ite_false]
      omega
  unfold size
  rw [H, interval_card _ _ hu]
  dsimp only [u]
  by_cases hlast : a.val=b-1
  · rw [ite_eq_left hlast, hlast]
    have hp : b*s = (b-1)*s+s := by
      conv => lhs; rw [show b = (b-1)+1 by omega, Nat.add_mul, Nat.one_mul]
    rw [Nat.add_mul, Nat.one_mul, hp] at hpack'
    omega
  · rw [ite_eq_right hlast, Nat.add_mul, Nat.one_mul, Nat.add_sub_cancel_left]
    omega

end Blocks
end ULB63
