import Theory.Lifted
import Theory.SlowLower

open scoped ENNReal

noncomputable section

namespace ULB63
namespace Blocks

open SplitChain

variable {n b s m c : ℕ} {L μ Δ ρ ε : ℝ}

def height (P : Packing n b s) (i : Fin n) : ℕ :=
  min (b/4) ((P.block i).val+1-(b/4+1))

private theorem height_left (P : Packing n b s) {i : Fin n} (hi : left i) : height P i = 0 := by
  have h : (P.block i).val ≤ b/4 := by
    by_contra! hb
    have hd : b/4+1 ≤ i.val/s :=
      (Nat.succ_le_of_lt hb).trans (min_le_left _ _)
    have hh := (Nat.le_div_iff_mul_le P.each).1 hd
    have hprod : (b+1)*s ≤ 4*i.val := by
      calc
        _ ≤ (4*(b/4+1))*s := Nat.mul_le_mul_right s (by omega)
        _ = 4*((b/4+1)*s) := by ring
        _ ≤ _ := Nat.mul_le_mul_left 4 hh
    have hcap := P.fills
    change i.val < n/4 at hi
    omega
  unfold height
  rw [Nat.sub_eq_zero_of_le (by omega : (P.block i).val+1 ≤ b/4+1), min_zero]

private theorem height_right (P : Packing n b s) (hb : 4 ≤ b) {i : Fin n} (hi : right i) :
    height P i = b/4 := by
  have hh : 2*(b/4) ≤ i.val/s := by
    apply (Nat.le_div_iff_mul_le P.each).2
    have hprod : 4*((2*(b/4))*s) ≤ 3*n := by
      calc
        _ = (4*(2*(b/4)))*s := by ring
        _ ≤ (3*b)*s := Nat.mul_le_mul_right s (by omega)
        _ = 3*(b*s) := by ring
        _ ≤ _ := Nat.mul_le_mul_left 3 P.fits
    change n-n/4 ≤ i.val at hi
    omega
  have hmin : 2*(b/4) ≤ (P.block i).val :=
    le_min hh (by omega)
  unfold height
  exact min_eq_left (by omega)

def routing (P : Packing n b s) (hb : 4 ≤ b) : Routing P.matrix where
  width := b/4
  height := height P
  height_le _ := min_le_left _ _
  left_zero _ := height_left P
  right_full _ := height_right P hb
  limit_step i j h := by
    obtain ⟨hi,hj⟩ := P.support_blocks i j h
    constructor <;> unfold height <;> omega

theorem minimax_clustered (hn : 4 ≤ n) (hm : 1 ≤ m) (hL : 0 < L) (hΔ : 0 < Δ)
    (hb : 4 ≤ b) (hs : 0 < s) (hpack : b*s ≤ n) (hfill : n < (b+1)*s)
    (hρ : 1-1/(32*(b:ℝ)^2) ≤ ρ) {q : ℝ}
    (hq : 0 ≤ q) (hq₁ : q < 1) (hcond : 32*μ ≤ L*(1-q)^2)
    (k T : ℕ) (hT : 5*T ≤ k*(b/4)) (hε : 0 ≤ ε)
    (htol : ε < Δ*q^(2*k)) :
    ((T+1 : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  let P : Packing n b s := ⟨by omega, hs, hpack, hfill⟩
  exact minimax_routed (routing P hb) (P.commClass hρ) hn hm hL hΔ hq hq₁ hcond k T hT hε htol

theorem minimax_radius (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : 0 < L)
    (hΔ : 0 < Δ) (hρ : 0 ≤ ρ) {p : ℕ} (hp : p ≤ n)
    (hgap : p < 8 ∨ 1-1/(32*(p:ℝ)^2) ≤ ρ)
    {q : ℝ} (hq : 0 ≤ q) (hq₁ : q < 1) (hcond : 32*μ ≤ L*(1-q)^2)
    (k T : ℕ) (hT : 5*T ≤ k*max 1 (p/16)) (hε : 0 ≤ ε)
    (htol : ε < Δ*q^(2*k)) :
    ((T+1 : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  by_cases hsmall : p<8
  · have hp0 : p/16=0 := by omega
    simp [hp0] at hT
    have hh : 2*(microSteps*T) ≤ 2*k := by unfold microSteps; omega
    have hpow : q^(2*k) ≤ q^(2*(microSteps*T)) := by
      obtain ⟨v,hv⟩ := Nat.exists_eq_add_of_le hh
      calc
        q^(2*k) = q^(2*(microSteps*T))*q^v := by rw [hv, pow_add]
        _ ≤ _ := mul_le_of_le_one_right (by positivity) (pow_le_one₀ hq hq₁.le)
    exact Chain.minimax_chain hn hm hμ hL hΔ hρ hq hq₁ (by linarith) T hε
      (htol.trans_le (mul_le_mul_of_nonneg_left hpow hΔ.le))
  · have hp₀ : 0 < p := by omega
    have hdiv : 1 ≤ n/p := (Nat.le_div_iff_mul_le hp₀).2 (by simpa using hp)
    let s := n/p+1
    have hs : 0 < s := Nat.zero_lt_succ _
    let b := n/s
    have hlo : p/2 ≤ b := by
      apply (Nat.le_div_iff_mul_le hs).2
      calc
        (p/2)*s ≤ (p/2)*(2*(n/p)) := Nat.mul_le_mul_left _ (by dsimp [s]; omega)
        _ = (2*(p/2))*(n/p) := by ring
        _ ≤ p*(n/p) := Nat.mul_le_mul_right _ (by omega)
        _ ≤ n := by rw [Nat.mul_comm]; exact Nat.div_mul_le_self _ _
    have hhi : b ≤ p := by
      apply Nat.le_of_lt_succ
      apply (Nat.div_lt_iff_lt_mul hs).2
      have H : n < s*p := (Nat.div_lt_iff_lt_mul hp₀).mp (Nat.lt_succ_self _)
      exact H.trans_le (by rw [Nat.mul_comm s p]; exact Nat.mul_le_mul_right s (Nat.le_succ p))
    have hb : 4 ≤ b := by omega
    have hpack : b*s ≤ n := Nat.div_mul_le_self _ _
    have hfill : n < (b+1)*s := (Nat.div_lt_iff_lt_mul hs).mp (Nat.lt_succ_self _)
    have hdelay : max 1 (p/16) ≤ b/4 := by
      apply max_le <;> omega
    have hgap' : 1-1/(32*(b:ℝ)^2) ≤ ρ := by
      have h := hgap.resolve_left hsmall
      have hb₀ : (0:ℝ) < b := by exact_mod_cast (show 0 < b by omega)
      have hp' : (b:ℝ)≤p := by exact_mod_cast hhi
      have hcomp : 32*(b:ℝ)^2 ≤ 32*(p:ℝ)^2 := by gcongr
      exact (sub_le_sub_left (one_div_le_one_div_of_le (by positivity) hcomp) 1).trans h
    exact minimax_clustered (by omega) hm hL hΔ hb hs hpack hfill hgap' hq hq₁ hcond k T
      (hT.trans (Nat.mul_le_mul_left k hdelay)) hε htol

end Blocks
end ULB63
