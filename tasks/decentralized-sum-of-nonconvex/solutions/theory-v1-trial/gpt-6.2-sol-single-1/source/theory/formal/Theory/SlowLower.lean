import Theory.SplitInstance

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

set_option maxHeartbeats 1000000

noncomputable section

namespace ULB63
namespace SplitChain

open Protocol Lower

variable {d n m c : ℕ} {L μ Δ ρ ε q : ℝ}

def ramp (i : Fin n) : ℕ := min (radius n) (i.val+1-radius n)

theorem ramp_bound (i : Fin n) : ramp i ≤ radius n := min_le_left _ _

theorem ramp_left {i : Fin n} (h : left i) : ramp i = 0 := by
  unfold ramp
  have hh : i.val+1 ≤ radius n := by omega
  rw [Nat.sub_eq_zero_of_le hh, min_zero]

theorem ramp_right {i : Fin n} (h : right i) : ramp i = radius n := by
  unfold ramp
  have hh : n-radius n ≤ i.val := h
  apply min_eq_left
  change n/4 ≤ i.val+1-n/4
  change n-n/4 ≤ i.val at hh
  omega

structure Routing {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) where
  width : ℕ
  height : Fin n → ℕ
  height_le : ∀ i, height i ≤ width
  left_zero : ∀ i, left i → height i = 0
  right_full : ∀ i, right i → height i = width
  limit_step : ∀ i j, W i j ≠ 0 → height i ≤ height j+1 ∧ height j ≤ height i+1

def pathRouting (n : ℕ) : Routing (Path.matrix n) where
  width := radius n
  height := ramp
  height_le := ramp_bound
  left_zero _ := ramp_left
  right_full _ := ramp_right
  limit_step i j h := by
    obtain ⟨hi,hj⟩ := Path.support i j h
    constructor <;> unfold ramp <;> omega

section Routing

variable {W : Matrix (Fin n) (Fin n) ℝ} (R : Routing W)

def stamp (i : Fin n) (k : Fin d) : ℕ :=
  k.val * R.width + (if k.val % 2 = 0 then R.height i else R.width-R.height i)

private theorem stamp_bounds (i : Fin n) (k : Fin d) :
    k.val*R.width ≤ stamp R i k ∧ stamp R i k ≤ k.val*R.width+R.width := by
  have h := R.height_le i
  unfold stamp
  split <;> omega

private theorem stamp_order (i : Fin n) {k l : Fin d} (hl : k.val+1 = l.val) :
    stamp R i k ≤ stamp R i l := by
  have hk := (stamp_bounds R i k).2
  have hm := (stamp_bounds R i l).1
  rw [← hl, Nat.add_mul, Nat.one_mul] at hm
  omega

private theorem stamp_move {i j : Fin n} (hij : W i j ≠ 0) (k : Fin d) :
    stamp R i k ≤ stamp R j k+1 := by
  obtain ⟨hi,hj⟩ := R.limit_step i j hij
  have hri := R.height_le i
  have hrj := R.height_le j
  unfold stamp
  split <;> omega

def atGroup (i : Fin n) (p : ℕ) : Prop := (p=0 ∧ left i) ∨ (p=1 ∧ right i)

private theorem stamp_group {i : Fin n} {p : ℕ} (hp : atGroup i p) {k : Fin d}
    (hk : k.val%2=p) :
    stamp R i k = k.val*R.width := by
  rcases hp with ⟨rfl,h⟩ | ⟨rfl,h⟩
  · simp [stamp, hk, R.left_zero i h]
  · simp [stamp, hk, R.right_full i h]

private theorem stamp_group_prev {i : Fin n} {p : ℕ} (hp : atGroup i p) {k : Fin d}
    (hk : selected p k) :
    stamp R i (Chain.prev k) = stamp R i k := by
  have hh := stamp_group R hp hk.2
  rw [hh]
  have hpos : k.val ≠ 0 := hk.1
  have hsucc : (Chain.prev k).val+1=k.val := by dsimp [Chain.prev]; omega
  have he : ((Chain.prev k).val+1)*R.width =
      (Chain.prev k).val*R.width+R.width := Nat.succ_mul _ _
  rw [hsucc] at he
  rcases hp with ⟨rfl,h⟩ | ⟨rfl,h⟩
  · have hm : (Chain.prev k).val%2 ≠ 0 := by
      have hpar := hk.2
      dsimp [Chain.prev] at *
      omega
    simp [stamp, hm, R.left_zero i h, he]
  · have hm : (Chain.prev k).val%2 = 0 := by
      have hpar := hk.2
      dsimp [Chain.prev] at *
      omega
    simp [stamp, hm, R.right_full i h, he]

def supportAt (t : ℕ) (i : Fin n) (d : ℕ) : Set (Fin d) := {k | stamp R i k < t}

private theorem group_progress {p : ℕ} {i : Fin n} (hp : atGroup i p) (t : ℕ) (x : E (d+1))
    (hx : Supported (supportAt R t i (d+1)) x) :
    Supported (supportAt R (t+1) i (d+1)) (gradient (groupLaw n d p L q Δ) x) := by
  rw [supported_iff] at hx ⊢
  intro k hk
  have htime : t+1 ≤ stamp R i k := Nat.not_lt.mp hk
  let e : E (d+1) := EuclideanSpace.single k 1
  let G := edge (d+1) p
  have hx_zero {j : Fin (d+1)} (h : stamp R i k ≤ stamp R i j) : x j = 0 :=
    hx j (Nat.not_lt.mpr (by omega : t ≤ stamp R i j))
  have hinner : ⟪G x,G e⟫ = 0 := by
    rw [PiLp.inner_apply]
    apply Finset.sum_eq_zero
    intro j _
    by_cases hsel : selected p j
    · by_cases hsame : j=k
      · subst j
        have hz : G x k = 0 := by
          rw [show G x k = x k - x (Chain.prev k) by simp [G,hsel],
            hx_zero le_rfl, hx_zero (by rw [stamp_group_prev R hp hsel]), sub_zero]
        simp [hz]
      · by_cases hpred : Chain.prev j=k
        · have hstep : k.val+1=j.val := by
            have h := congrArg Fin.val hpred
            dsimp [Chain.prev] at h
            have hj := hsel.1
            omega
          have hz : G x j = 0 := by
            rw [show G x j = x j - x (Chain.prev j) by simp [G,hsel],
              hx_zero (stamp_order R i hstep), hpred, hx_zero le_rfl, sub_zero]
          simp [hz]
        · have hz : G e j = 0 := by simp [G, e, hsel, hsame, hpred]
          simp [hz]
    · have hz : G x j = 0 := by simp [G,hsel]
      simp [hz]
  have hquad : G.adjoint (G x) k = 0 := by
    have hh : ⟪G.adjoint (G x), e⟫ = 0 := by rw [ContinuousLinearMap.adjoint_inner_left, hinner]
    simpa [e, EuclideanSpace.inner_single_right, RCLike.conj_to_real] using hh
  have hdr : (gain n • (if p=0 then driver d L q Δ else 0) : E (d+1)) k = 0 := by
    rcases hp with ⟨rfl, hl⟩ | ⟨rfl, hr⟩
    · have hkn : k ≠ 0 := by
        intro h
        subst k
        simp [stamp, R.left_zero i hl] at htime
      simp [driver, hkn]
    · simp
  unfold groupLaw
  rw [rawQuad_grad, PiLp.add_apply, PiLp.smul_apply, hquad, hdr, smul_zero, add_zero]

private theorem node_progress (t : ℕ) (i : Fin n) (x : E (d+1))
    (hx : Supported (supportAt R t i (d+1)) x) :
    Supported (supportAt R (t+1) i (d+1)) (gradient (nodeLaw d L q Δ i) x) := by
  unfold nodeLaw
  split
  · rename_i hl
    exact group_progress R (Or.inl ⟨rfl,hl⟩) t x hx
  · split
    · rename_i hr
      exact group_progress R (Or.inr ⟨rfl,hr⟩) t x hx
    · simpa using supported_zero (supportAt R (t+1) i (d+1))

theorem routing_support {A : Algorithm n m c} (hA : A.Admissible L μ Δ ρ) (ω : A.Ω) :
    ∀ t i, allowed (A.info (hardSplit n m d L q Δ) W ω i)
      (A.hist (hardSplit n m d L q Δ) W ω t i) ⊆ supportAt R t i (d+1) := by
  let I := hardSplit n m d L q Δ
  intro t
  induction t with
  | zero =>
    intro i
    simp [allowed, Algorithm.info, Algorithm.hist, hardSplit, supp]
  | succ t ih =>
    intro i
    have hmono : supportAt R t i (d+1) ⊆ supportAt R (t+1) i (d+1) :=
      fun _ h => Nat.lt_succ_of_lt h
    rw [Algorithm.hist]
    split
    · rw [allowed_append]
      apply Set.union_subset ((ih i).trans hmono)
      apply Set.iUnion_subset
      intro k
      change Supported (supportAt R (t+1) i (d+1))
        (∑ j, W i j • A.send (d+1) (A.info I W ω j) (A.hist I W ω t j) k)
      apply Supported.sum
      intro j _
      by_cases h : W i j = 0
      · rw [h, zero_smul]
        exact supported_zero _
      · have H := (hA.send_zero_respecting (d+1) (A.info I W ω j) (A.hist I W ω t j) k).trans (ih j)
        have H' : Supported (supportAt R (t+1) i (d+1)) (A.send _ (A.info I W ω j) (A.hist I W ω t j) k) :=
          H.trans (fun l ht => by
            have ht' : stamp R j l < t := ht
            have hs := stamp_move R h l
            change stamp R i l < t+1
            omega)
        exact H'.smul _
    · rw [allowed_append]
      apply Set.union_subset ((ih i).trans hmono)
      have H := (hA.call_zero_respecting (d+1) (A.info I W ω i) (A.hist I W ω t i)).trans (ih i)
      cases he : A.call (d+1) (A.info I W ω i) (A.hist I W ω t i) with
      | idle => exact Set.empty_subset _
      | grad j x =>
        rw [he] at H
        exact node_progress R t i x H
      | prox x η =>
        rw [he] at H
        exact (diag_support _ x η).trans (H.trans hmono)

theorem minimax_routed (hW : CommClass n ρ W) (hn : 4 ≤ n) (hm : 1 ≤ m) (hL : 0 < L)
    (hΔ : 0 < Δ) {q : ℝ}
    (hq : 0 ≤ q) (hq₁ : q < 1) (hcond : 32*μ ≤ L*(1-q)^2)
    (k T : ℕ) (hT : 5*T ≤ k*R.width) (hε : 0 ≤ ε)
    (htol : ε < Δ*q^(2*k)) :
    ((T+1 : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  let I := hardSplit n m k L q Δ
  have hs : μ ≤ scale L*(1-q)^2 := by dsimp [scale]; nlinarith
  have hI : I.InClass L μ Δ := split_inClass hn (by omega) hL hq hq₁ hs hΔ.le
  apply minimax_ge
  intro A hA
  refine ⟨k+1,I,W,hI,hW,hitting_gt T ?_⟩
  intro t ht
  let j := Fin.last k
  have hz (ω : A.Ω) : A.avgOutput I W ω t j = 0 := by
    have h (i : Fin n) : allowed (A.info I W ω i) (A.hist I W ω (microSteps*t) i) ⊆
        {l : Fin (k+1) | l ≠ j} := by
      intro l hl hj
      change l=j at hj
      subst l
      have htime : stamp R i j < microSteps*t :=
        (routing_support (q := q) (d := k) R hA ω (microSteps*t) i) hl
      have hs := (stamp_bounds R i j).1
      have htt := Nat.mul_le_mul_left 5 ht
      change k*R.width ≤ stamp R i j at hs
      unfold microSteps at htime
      omega
    exact supported_iff.mp (avg_from_allowed hA I W ω t h) j (by simp)
  have hpoint (ω : A.Ω) : ENNReal.ofReal (Δ*q^(2*k)) ≤ I.gap (A.avgOutput I W ω t) := by
    have H := Chain.F_at_zero (L := L/8) (by positivity) hq hq₁ hΔ.le _ j (hz ω)
    have hφ := split_phi (d := k) (L := L) (q := q) (Δ := Δ) hn (by omega : 0 < m)
      hL hq hq₁ hΔ.le (A.avgOutput I W ω t)
    have hS := split_star (d := k) (L := L) (q := q) (Δ := Δ) hn (by omega : 0 < m) hL hq hq₁ hΔ.le
    change I.φ _ = _ at hφ
    change I.φstar = _ at hS
    rw [Instance.gap, hφ, hS, ← EReal.coe_sub, sub_neg_eq_add, sub_add_cancel]
    exact ENNReal.ofReal_le_ofReal H
  have he := lintegral_mono hpoint (μ := A.P)
  simp only [lintegral_const, measure_univ, mul_one] at he
  exact ((ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε).2 htol).trans_le he

end Routing

theorem minimax_path (hn : 4 ≤ n) (hm : 1 ≤ m) (_hμ : 0 < μ) (hL : 0 < L)
    (hΔ : 0 < Δ) (hρ : 1-1/(16*(n:ℝ)^2) ≤ ρ) {q : ℝ}
    (hq : 0 ≤ q) (hq₁ : q < 1) (hcond : 32*μ ≤ L*(1-q)^2)
    (k T : ℕ) (hT : 5*T ≤ k*(n/4)) (hε : 0 ≤ ε)
    (htol : ε < Δ*q^(2*k)) :
    ((T+1 : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε :=
  minimax_routed (pathRouting n) (Path.commClass (by omega : 0 < n) hρ)
    hn hm hL hΔ hq hq₁ hcond k T hT hε htol

end SplitChain
end ULB63
