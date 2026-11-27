import Theory.Nested

open scoped ENNReal

set_option maxHeartbeats 1800000

noncomputable section

namespace ULB63
namespace Rate

private theorem ceil_bound {x : ℝ} (hx : 0 ≤ x) : (⌈x⌉₊ : ℝ) ≤ x+1 :=
  (Nat.ceil_lt_add_one hx).le

private theorem tune_inner {n m D : ℕ} (hn : 0 < n) {L σ : ℝ}
    (hσ : 0 < σ) (hL : σ ≤ L) :
    ∃ R B : ℕ, 6 ≤ Real.sqrt (σ/(8*L))*(R:ℝ) ∧ 1 ≤ B ∧
      8*(L/σ)*(R:ℝ) ≤ (n:ℝ)*(B:ℝ) ∧
      ((m+4*D+R*(B+1+4*D) : ℕ) : ℝ) ≤
        (m:ℝ)+3200*(L/σ)^2/(n:ℝ)+120*Real.sqrt (L/σ)*((D:ℝ)+1) := by
  have hLp := hσ.trans_le hL
  have hnr : (0:ℝ) < n := by exact_mod_cast hn
  let t := L/σ
  let w := Real.sqrt t
  let a := Real.sqrt (σ/(8*L))
  have ht : 1 ≤ t := (one_le_div hσ).2 hL
  have ht0 : 0 < t := by linarith
  have hw : 1 ≤ w := by simpa [w] using Real.sqrt_le_sqrt ht
  have hwp : 0 ≤ w := by linarith
  have ha : 0 < a := Real.sqrt_pos.2 (by positivity)
  have hs : (a*w)^2 = 1/8 := by
    rw [mul_pow, show a^2=σ/(8*L) from Real.sq_sqrt (by positivity),
      show w^2=t from Real.sq_sqrt ht0.le]
    dsimp [t]
    field_simp
  have haw : 1/3 ≤ a*w := (sq_le_sq₀ (by norm_num) (mul_nonneg ha.le hwp)).1
    (by rw [hs]; norm_num)
  let R := ⌈18*w⌉₊
  have hRc : 18*w ≤ (R:ℝ) := Nat.le_ceil _
  have hR : 6 ≤ a*(R:ℝ) := by
    have H := mul_le_mul_of_nonneg_left hRc ha.le
    linarith
  have hRupper : (R:ℝ) ≤ 20*w := by
    have H : (R:ℝ) ≤ 18*w+1 := ceil_bound (by positivity)
    linarith
  have hRp : (0:ℝ) < R := by
    have : 0 < 18*w := by positivity
    exact this.trans_le hRc
  let X := 8*t*(R:ℝ)/(n:ℝ)
  let B := ⌈X⌉₊
  have hX : 0 < X := by dsimp [X]; positivity
  have hBc : X ≤ (B:ℝ) := Nat.le_ceil _
  have hBup : (B:ℝ) ≤ X+1 := ceil_bound hX.le
  have hB : 1 ≤ B := by
    have H := hX.trans_le hBc
    exact_mod_cast H
  have hN : 8*t*(R:ℝ) ≤ (n:ℝ)*(B:ℝ) := by
    have H := (div_le_iff₀ hnr).1 hBc
    simpa only [mul_comm (B:ℝ)] using H
  refine ⟨R,B,hR,hB,hN,?_⟩
  change ((m+4*D+R*(B+1+4*D) : ℕ) : ℝ) ≤ (m:ℝ)+3200*t^2/(n:ℝ)+120*w*((D:ℝ)+1)
  have hRB : (R:ℝ)*(B:ℝ) ≤ (R:ℝ)+3200*t^2/(n:ℝ) := by
    calc
      _ ≤ (R:ℝ)*(X+1) := mul_le_mul_of_nonneg_left hBup hRp.le
      _ = (R:ℝ)+(8*t/(n:ℝ))*(R:ℝ)^2 := by dsimp [X]; ring
      _ ≤ (R:ℝ)+(8*t/(n:ℝ))*(20*w)^2 := by
        exact add_le_add_right (mul_le_mul_of_nonneg_left
          (pow_le_pow_left₀ hRp.le hRupper 2) (by positivity)) _
      _ = _ := by
        rw [mul_pow, show w^2=t from Real.sq_sqrt ht0.le]
        ring
  have hLin := mul_le_mul_of_nonneg_right hRupper (show 0 ≤ 2+4*(D:ℝ) by positivity)
  have hDist := mul_le_mul_of_nonneg_right hw (show 0 ≤ 4*(D:ℝ) by positivity)
  have hwd : 0 ≤ w*(D:ℝ) := by positivity
  push_cast
  nlinarith only [hRB,hLin,hDist,hwd,hwp]

theorem uniform (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D H K : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hH₀ : 1 ≤ H) (hH : 120*(1+L/μ)*Real.sqrt (L/μ)*(1/2:ℝ)^H ≤ 1)
    (hK : (1/2:ℝ)^K*Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤
          ((K*H*⌈100000*((m:ℝ)+Real.sqrt (L/μ)*
            (1+(D:ℝ)+(m:ℝ)/Real.sqrt (Real.sqrt ((n:ℝ)*(m:ℝ)))))⌉₊ : ℕ) : ℕ∞) := by
  let κ := L/μ
  let w := Real.sqrt κ
  let t := Real.sqrt ((n:ℝ)*(m:ℝ))
  let v := Real.sqrt t
  let G := 100000*((m:ℝ)+w*(1+(D:ℝ)+(m:ℝ)/v))
  let cap := ⌈G⌉₊
  have hnr : (1:ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hmr : (1:ℝ) ≤ m := by exact_mod_cast hm
  have hn₀ : (0:ℝ) < n := by linarith
  have hL₀ := hμ.trans_le hL
  have hk : 1 ≤ κ := (one_le_div hμ).2 hL
  have hk₀ : 0 < κ := by linarith
  have hnm : (1:ℝ) ≤ (n:ℝ)*m := one_le_mul_of_one_le_of_one_le hnr hmr
  have ht : 1 ≤ t := by simpa [t] using Real.sqrt_le_sqrt hnm
  have ht₀ : 0 < t := by linarith
  have hv : 1 ≤ v := by simpa [v] using Real.sqrt_le_sqrt ht
  have hv₀ : 0 < v := by linarith
  have hw₀ : 0 ≤ w := Real.sqrt_nonneg _
  have htsq : t^2 = (n:ℝ)*m := Real.sq_sqrt (by positivity)
  have hwp : 0 ≤ w*((m:ℝ)/v) := by positivity
  have hwd : 0 ≤ w*((D:ℝ)+1) := by positivity
  have hcap : G ≤ (cap:ℝ) := Nat.le_ceil _
  have NatCap (a : ℕ) (h : (a:ℝ) ≤ G) : a ≤ cap := by exact_mod_cast h.trans hcap
  by_cases hlarge : 2*t ≤ κ
  · let σ := L/t
    let ν := σ-μ
    have hσ : 0 < σ := by dsimp [σ]; positivity
    have heq : μ+ν=σ := by dsimp [ν]; ring
    have hν : μ ≤ ν := by
      have hh := (le_div_iff₀ hμ).1 hlarge
      have h₂ : 2*μ ≤ σ := (le_div_iff₀ ht₀).2 (by linarith)
      dsimp [ν]
      linarith
    have hν₀ := hμ.trans_le hν
    have hσL : σ ≤ L := div_le_self hL₀.le ht
    have hνL : ν ≤ L := by dsimp [ν]; linarith
    have hσeq : L/σ=t := by dsimp [σ]; field_simp
    obtain ⟨R,B,hRi,hB,hBatch,hCost⟩ := tune_inner (m := m) (D := D) (by omega : 0 < n) hσ hσL
    have hLowCost : ((m+4*D+R*(B+1+4*D) : ℕ) : ℝ) ≤
        3201*(m:ℝ)+120*v*((D:ℝ)+1) := by
      rw [hσeq,htsq] at hCost
      have hh : 3200*((n:ℝ)*m)/(n:ℝ)=3200*(m:ℝ) := by field_simp
      rw [hh] at hCost
      change _ ≤ (m:ℝ)+3200*(m:ℝ)+120*v*((D:ℝ)+1) at hCost
      linarith
    let o := w/v
    have ho : 1 ≤ o := (one_le_div hv₀).2 (Real.sqrt_le_sqrt (by linarith))
    have ho₀ : 0 < o := by linarith
    have hev : o^2 = κ/t := by
      dsimp [o,w,v]
      rw [div_pow,Real.sq_sqrt hk₀.le,Real.sq_sqrt ht₀.le]
    let a := Real.sqrt (μ/(2*ν))
    have ha : 0 < a := Real.sqrt_pos.2 (by positivity)
    have ham : 2/3 ≤ a*o := by
      apply (sq_le_sq₀ (by norm_num) (by positivity)).1
      rw [mul_pow, show a^2=μ/(2*ν) from Real.sq_sqrt (by positivity),hev]
      have he : μ/(2*ν)*(κ/t) = (μ+ν)/(2*ν) := by
        rw [heq]
        dsimp [κ,σ]
        field_simp
      rw [he]
      exact (le_div_iff₀ (by positivity)).2 (by linarith)
    let S := ⌈9*o⌉₊
    have hSg : 9*o ≤ (S:ℝ) := Nat.le_ceil _
    have hS : 6 ≤ a*(S:ℝ) := by
      have h := mul_le_mul_of_nonneg_left hSg ha.le
      linarith
    have hS' : (S:ℝ) ≤ 10*o := by
      have h := ceil_bound (show 0 ≤ 9*o by positivity)
      change (S:ℝ) ≤ 9*o+1 at h
      linarith
    have hS'' : (S:ℝ) ≤ 10*w := hS'.trans (mul_le_mul_of_nonneg_left (div_le_self hw₀ hv) (by norm_num))
    have hH' : 3*(1+ν/μ)*(1/2:ℝ)^H*(S:ℝ) ≤ 1/4 := by
      have hleft : 3*(1+ν/μ) ≤ 3*(1+κ) := by
        have H := div_le_div_of_nonneg_right hνL hμ.le
        dsimp [κ]; linarith
      have hnon : 0 ≤ 3*(1+κ)*(1/2:ℝ)^H := by positivity
      calc
        _ ≤ 3*(1+κ)*(1/2:ℝ)^H*(S:ℝ) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hleft (by positivity)) (by positivity)
        _ ≤ 3*(1+κ)*(1/2:ℝ)^H*(10*w) := mul_le_mul_of_nonneg_left hS'' hnon
        _ = (120*(1+κ)*w*(1/2:ℝ)^H)/4 := by ring
        _ ≤ _ := by change _ ≤ _ at hH; linarith
    obtain ⟨A,hA,hA'⟩ := Nested.bound hn hm hμ hν (heq.symm ▸ hσL) hρ hρ₁
      hD (heq.symm ▸ hRi) hB (heq.symm ▸ hBatch) hS hH' hK
    have hC : ((S*(m+4*D+R*(B+1+4*D)) : ℕ) : ℝ) ≤ G := by
      rw [Nat.cast_mul]
      calc
        _ ≤ (10*o)*(3201*(m:ℝ)+120*v*((D:ℝ)+1)) :=
          mul_le_mul hS' hLowCost (by positivity) (by positivity)
        _ = 32010*w*((m:ℝ)/v)+1200*w*((D:ℝ)+1) := by
          dsimp [o]
          field_simp
          ring
        _ ≤ G := by
          change _ ≤ 100000*((m:ℝ)+w*(1+(D:ℝ)+(m:ℝ)/v))
          nlinarith
    refine ⟨A,hA,fun d I W hI hW => (hA' d I W hI hW).trans ?_⟩
    have hbound := Nat.mul_le_mul_left (K*H) (NatCap _ hC)
    have he : K*(S*(H*(m+4*D+R*(B+1+4*D)))) = K*H*(S*(m+4*D+R*(B+1+4*D))) := by ring
    rw [he]
    change ((K*H*(S*(m+4*D+R*(B+1+4*D))):ℕ):ℕ∞) ≤ ((K*H*cap:ℕ):ℕ∞)
    exact_mod_cast hbound
  · obtain ⟨R,B,hRi,hB,hBatch,hCost⟩ := tune_inner (m := m) (D := D) (by omega : 0 < n) hμ hL
    have ht2 : κ^2 ≤ 4*((n:ℝ)*m) := by
      have H := pow_le_pow_left₀ hk₀.le (le_of_not_ge hlarge) 2
      rwa [mul_pow, htsq, show (2:ℝ)^2=4 by norm_num] at H
    have ht3 : 3200*κ^2/(n:ℝ) ≤ 12800*(m:ℝ) := by
      apply (div_le_iff₀ hn₀).2
      linarith
    obtain ⟨A,hA,hA'⟩ := Batch.bound hn hm hμ hL hρ hρ₁ hD hRi hB hBatch hK
    have hC : ((m+4*D+R*(B+1+4*D) : ℕ) : ℝ) ≤ G := by
      change _ ≤ (m:ℝ)+3200*κ^2/(n:ℝ)+120*w*((D:ℝ)+1) at hCost
      dsimp [G]
      nlinarith
    refine ⟨A,hA,fun d I W hI hW => (hA' d I W hI hW).trans ?_⟩
    have Hbound := Nat.mul_le_mul_left K (NatCap _ hC)
    have HK : K ≤ K*H := by simpa using Nat.mul_le_mul_left K hH₀
    change ((K*(m+4*D+R*(B+1+4*D)):ℕ):ℕ∞) ≤ ((K*H*cap:ℕ):ℕ∞)
    exact_mod_cast Hbound.trans (Nat.mul_le_mul_right cap HK)

end Rate
end ULB63
