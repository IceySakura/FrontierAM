import Theory.Batch
import Theory.ClusterLower
import Theory.Hidden
import Theory.Rate

namespace ULB63

/-- R1: Deterministic restarted acceleration using exact static weighted-sum averaging. -/
theorem finite_network_exact {n m : ℕ} {L μ Δ ρ ε : ℝ}
    (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D R H : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hH : (1/2 : ℝ)^H * Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤ ((H * (R * (m+4*D+1)) : ℕ) : ℕ∞) :=
  Upper.exact_bound hn hm hμ hL hρ hρ₁ hD hR hH

/-- R2: A batch of `B` queries per node is charged `B` unit iterations. All `n B` samples
contribute to the variance reduction, and each batch uses just one exact average. -/
theorem finite_network_parallel {n m : ℕ} {L μ Δ ρ ε : ℝ}
    (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D R H B : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hB : 1 ≤ B) (hparallel : 8*(L/μ)*(R : ℝ) ≤ (n : ℝ)*(B : ℝ))
    (hH : (1/2 : ℝ)^H * Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤ ((H * (m+4*D+R*(B+1+4*D)) : ℕ) : ℕ∞) :=
  Batch.bound hn hm hμ hL hρ hρ₁ hD hR hB hparallel hH

/-- R3: Both sides of the minimax bound in the sufficiently parallel regime.
The lower bound continues to hold with any message width `c`. -/
theorem fast_network_minimax_bracket {n m : ℕ} {L μ Δ ρ ε : ℝ}
    (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hΔ : 0 < Δ) (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    {D R H B T : ℕ} {q : ℝ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hB : 1 ≤ B) (hparallel : 8*(L/μ)*(R : ℝ) ≤ (n : ℝ)*(B : ℝ))
    (hwork : R*B ≤ m+R) (hH : (1/2 : ℝ)^H * Δ ≤ ε)
    (hq : 0 ≤ q) (hq₁ : q < 1) (hcond : 4*μ ≤ L*(1-q)^2)
    (hε : 0 ≤ ε) (hsmall : ε < Δ/4) (htail : ε < Δ*q^(10*T)) :
    (∀ c : ℕ, ((max (m/10+1) (T+1) : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε) ∧
      minimax n m 1 L μ Δ ρ ε ≤ ((H*(2*m+4*D+R*(2+4*D)) : ℕ) : ℕ∞) := by
  refine ⟨?_,?_⟩
  · intro c
    have hd := Hidden.minimax_hidden (c := c) hn hm hμ hL hΔ hρ (m/10) (by
      unfold microSteps
      omega) hε hsmall
    have hc := Chain.minimax_chain (c := c) hn hm hμ (hμ.trans_le hL) hΔ hρ hq hq₁ hcond T hε
      (by simpa only [microSteps, ← Nat.mul_assoc] using htail)
    rw [max_def]
    split <;> assumption
  · obtain ⟨A, hA, hguarantee⟩ := finite_network_parallel (ε := ε) hn hm hμ hL hρ hρ₁
      hD hR hB hparallel hH
    refine (iInf₂_le A hA).trans ?_
    apply iSup_le
    intro d
    apply iSup_le
    intro I
    apply iSup_le
    intro hI
    apply iSup_le
    intro W
    apply iSup_le
    intro hW
    refine (hguarantee d I W hI hW).trans ?_
    have hb : m+4*D+R*(B+1+4*D) ≤ 2*m+4*D+R*(2+4*D) := by nlinarith
    exact_mod_cast Nat.mul_le_mul_left H hb

/-- R4: Both sides of the minimax bound on the finite-size network plateau. The hard
communication matrix has exactly the prescribed `n` nodes. -/
theorem slow_network_minimax_bracket {n m : ℕ} {L μ Δ ρ ε : ℝ}
    (hn : 4 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hΔ : 0 < Δ) (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) (hslow : 1-1/(16*(n:ℝ)^2) ≤ ρ)
    {R H B k T : ℕ} {q : ℝ}
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hB : 1 ≤ B) (hparallel : 8*(L/μ)*(R : ℝ) ≤ (n : ℝ)*(B : ℝ))
    (hwork : R*B ≤ m+n*R) (hH : (1/2 : ℝ)^H * Δ ≤ ε)
    (hq : 0 ≤ q) (hq₁ : q < 1) (hcond : 32*μ ≤ L*(1-q)^2)
    (hT : 5*T ≤ k*(n/4)) (hε : 0 ≤ ε) (hsmall : ε < Δ/4) (htail : ε < Δ*q^(2*k)) :
    (∀ c : ℕ, ((max (m/10+1) (T+1) : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε) ∧
      minimax n m 1 L μ Δ ρ ε ≤ ((H*(2*m+4*n+R*(5*n+1)) : ℕ) : ℕ∞) := by
  refine ⟨?_,?_⟩
  · intro c
    have hd := Hidden.minimax_hidden (c := c) (by omega : 2 ≤ n) hm hμ hL hΔ hρ (m/10)
      (by unfold microSteps; omega) hε hsmall
    have hc := SplitChain.minimax_path (c := c) hn hm hμ (hμ.trans_le hL) hΔ hslow hq hq₁
      hcond k T hT hε htail
    rw [max_def]
    split <;> assumption
  · obtain ⟨A, hA, hguarantee⟩ := finite_network_parallel (ε := ε) (by omega : 2 ≤ n) hm hμ hL hρ hρ₁
      (D := n-1) (Or.inl le_rfl) hR hB hparallel hH
    refine (iInf₂_le A hA).trans ?_
    apply iSup_le
    intro d
    apply iSup_le
    intro I
    apply iSup_le
    intro hI
    apply iSup_le
    intro W
    apply iSup_le
    intro hW
    refine (hguarantee d I W hI hW).trans ?_
    have hb : m+4*(n-1)+R*(B+1+4*(n-1)) ≤ 2*m+4*n+R*(5*n+1) := by
      have hl : 4*(n-1) ≤ 4*n := Nat.mul_le_mul_left 4 (Nat.sub_le _ _)
      calc
        _ ≤ m+4*n+R*(B+1+4*n) :=
          Nat.add_le_add (Nat.add_le_add_left hl m) (Nat.mul_le_mul_left R (Nat.add_le_add_left hl _))
        _ ≤ _ := by
          rw [show m+4*n+R*(B+1+4*n) = m+4*n+R*B+(R+4*(n*R)) by ring,
            show 2*m+4*n+R*(5*n+1) = 2*m+4*n+(R+5*(n*R)) by ring]
          omega
    exact_mod_cast Nat.mul_le_mul_left H hb

/-- R5: A two-sided minimax bound across network sizes and spectral gaps. Here `p` certifies
the lower-bound radius, while `D` certifies the upper-bound reachability radius. -/
theorem parallel_minimax_bracket {n m : ℕ} {L μ Δ ρ ε : ℝ}
    (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hΔ : 0 < Δ) (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    {p D R H B k T : ℕ} {q : ℝ} (hp : p ≤ n)
    (hgap : p < 8 ∨ 1-1/(32*(p:ℝ)^2) ≤ ρ)
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt (μ/(8*L)) * (R : ℝ))
    (hB : 1 ≤ B) (hparallel : 8*(L/μ)*(R : ℝ) ≤ (n : ℝ)*(B : ℝ))
    (hH : (1/2 : ℝ)^H * Δ ≤ ε)
    (hq : 0 ≤ q) (hq₁ : q < 1) (hcond : 32*μ ≤ L*(1-q)^2)
    (hT : 5*T ≤ k*max 1 (p/16)) (hε : 0 ≤ ε)
    (hsmall : ε < Δ/4) (htail : ε < Δ*q^(2*k)) :
    (∀ c : ℕ, ((max (m/10+1) (T+1) : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε) ∧
      minimax n m 1 L μ Δ ρ ε ≤ ((H*(m+4*D+R*(B+1+4*D)) : ℕ) : ℕ∞) := by
  refine ⟨?_,?_⟩
  · intro c
    have hd := Hidden.minimax_hidden (c := c) hn hm hμ hL hΔ hρ (m/10)
      (by unfold microSteps; omega) hε hsmall
    have hc := Blocks.minimax_radius (c := c) hn hm hμ (hμ.trans_le hL) hΔ hρ hp hgap
      hq hq₁ hcond k T hT hε htail
    rw [max_def]
    split <;> assumption
  · obtain ⟨A,hA,hguarantee⟩ := finite_network_parallel hn hm hμ hL hρ hρ₁ hD hR hB hparallel hH
    refine (iInf₂_le A hA).trans ?_
    exact iSup_le fun d => iSup_le fun I => iSup_le fun hI => iSup_le fun W => iSup_le fun hW =>
      hguarantee d I W hI hW

/-- R6: A nested accelerated proximal upper bound. `S` is the number of outer steps per
restart, and each outer step uses `H` inner epochs of `R` steps with local batch `B`. -/
theorem accelerated_parallel_bound {n m : ℕ} {L μ Δ ρ ν ε : ℝ}
    (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hν : μ ≤ ν) (hL : μ+ν ≤ L)
    (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D R H B S K : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hR : 6 ≤ Real.sqrt ((μ+ν)/(8*L)) * (R : ℝ))
    (hB : 1 ≤ B) (hbatch : 8*(L/(μ+ν))*(R : ℝ) ≤ (n : ℝ)*(B : ℝ))
    (hS : 6 ≤ Real.sqrt (μ/(2*ν))*(S:ℝ))
    (hH : 3*(1+ν/μ)*(1/2:ℝ)^H*(S:ℝ) ≤ 1/4)
    (hK : (1/2 : ℝ)^K * Δ ≤ ε) :
    ∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤ ((K*(S*(H*(m+4*D+R*(B+1+4*D)))) : ℕ) : ℕ∞) :=
  Nested.bound hn hm hμ hν hL hρ hρ₁ hD hR hB hbatch hS hH hK

/-- R7: The tuned upper bound for all condition numbers, together with the fixed-size
lower bound. The only remaining upper-bound budgets are the mixing radius `D` and the
logarithmic factors `H` and `K`. -/
theorem optimized_parallel_minimax {n m : ℕ} {L μ Δ ρ ε : ℝ}
    (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hΔ : 0 < Δ) (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) {D H K : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D))
    (hH₀ : 1 ≤ H) (hH : 120*(1+L/μ)*Real.sqrt (L/μ)*(1/2:ℝ)^H ≤ 1)
    (hK : (1/2:ℝ)^K*Δ ≤ ε) :
    (∀ (p k T : ℕ) (q : ℝ), p ≤ n →
      (p < 8 ∨ 1-1/(32*(p:ℝ)^2) ≤ ρ) →
      0 ≤ q → q < 1 → 32*μ ≤ L*(1-q)^2 →
      5*T ≤ k*max 1 (p/16) → 0 ≤ ε → ε < Δ/4 → ε < Δ*q^(2*k) →
      ∀ c : ℕ, ((max (m/10+1) (T+1) : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε) ∧
    (∃ A : Algorithm n m 1, A.Admissible L μ Δ ρ ∧
      ∀ d (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ → CommClass n ρ W →
        A.hittingTime I W ε ≤
          ((K*H*⌈100000*((m:ℝ)+Real.sqrt (L/μ)*
            (1+(D:ℝ)+(m:ℝ)/Real.sqrt (Real.sqrt ((n:ℝ)*(m:ℝ)))))⌉₊ : ℕ) : ℕ∞)) := by
  refine ⟨?_, Rate.uniform hn hm hμ hL hρ hρ₁ hD hH₀ hH hK⟩
  intro p k T q hp hgap hq hq₁ hcond hT hε hsmall htail c
  have hd := Hidden.minimax_hidden (c := c) hn hm hμ hL hΔ hρ (m/10)
    (by unfold microSteps; omega) hε hsmall
  have hc := Blocks.minimax_radius (c := c) hn hm hμ (hμ.trans_le hL) hΔ hρ hp hgap
    hq hq₁ hcond k T hT hε htail
  rw [max_def]
  split <;> assumption

end ULB63
