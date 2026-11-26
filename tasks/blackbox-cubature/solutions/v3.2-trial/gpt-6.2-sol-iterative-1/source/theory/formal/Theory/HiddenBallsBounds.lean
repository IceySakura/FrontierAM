import Theory.HiddenBalls

open MeasureTheory Set
open scoped BigOperators ENNReal

namespace Cubature

variable {d : ℕ}

lemma signBall_isOpen (s : Signs d) : IsOpen (signBall s) := by
  unfold signBall
  apply isOpen_lt <;> fun_prop

lemma signBall_nonempty (s : Signs d) : (signBall s).Nonempty := by
  refine ⟨signCenter s,?_⟩
  simp [signBall]

lemma signBall_mass_pos (s : Signs d) : 0 < (gamma d).real (signBall s) := by
  have hO := signBall_isOpen s
  have hS := signBall_nonempty s
  have h : 0 < gamma d (signBall s) := by
    rw [gamma_eq_density, withDensity_apply _ hO.measurableSet]
    rw [lintegral_pos_iff_support (measurable_density d).ennreal_ofReal]
    have hs :
        Function.support (fun x : Point d ↦ ENNReal.ofReal (density d x)) = univ := by
      ext x
      have hP : 0 < Real.sqrt (2*Real.pi) := Real.sqrt_pos.mpr (by positivity)
      have hh : 0 < density d x := by unfold density; positivity
      simp [ne_of_gt (ENNReal.ofReal_pos.mpr hh)]
    rw [hs, Measure.restrict_apply_univ]
    exact hO.measure_pos volume hS
  exact ENNReal.toReal_pos h.ne' (measure_ne_top _ _)

lemma signBall_eq_of_mem (s t : Signs d) (x : Point d)
    (hs : x ∈ signBall s) (ht : x ∈ signBall t) : s=t := by
  ext i
  have hs' : (x i-signCenter s i)^2 < 1/4 := by
    exact (Finset.single_le_sum
      (f := fun j ↦ (x j-signCenter s j)^2) (fun j _ ↦ sq_nonneg _) (Finset.mem_univ i)).trans_lt hs
  have ht' : (x i-signCenter t i)^2 < 1/4 := by
    exact (Finset.single_le_sum
      (f := fun j ↦ (x j-signCenter t j)^2) (fun j _ ↦ sq_nonneg _) (Finset.mem_univ i)).trans_lt ht
  cases eqs : s i <;> cases eqt : t i <;> first
    | rfl
    | simp [signCenter, eqs, eqt] at hs' ht' <;>
        nlinarith only [hs',ht',sq_nonneg (x i)]

lemma hiddenBall_mass (s : Signs d) (a : ℝ) :
    (∫ x, hiddenBallPayoff s a x ∂(gamma d)) = a*(gamma d).real (signBall s) := by
  unfold hiddenBallPayoff
  rw [integral_indicator_const a (signBall_isOpen s).measurableSet, smul_eq_mul, mul_comm]

lemma avoid_signBall (query : List ℝ → Point d) (n : ℕ) (hn : n < 2^d) :
    ∃ s : Signs d, ∀ j < n, query (List.replicate j 0) ∉ signBall s := by
  classical
  by_contra! H
  have H' (s : Signs d) : ∃ j : Fin n, query (List.replicate j.val 0) ∈ signBall s := by
    obtain ⟨j,hj,hs⟩ := H s
    exact ⟨⟨j,hj⟩,hs⟩
  choose i hi using H'
  have hI : Function.Injective i := by
    intro s t he
    have Ht := hi t
    rw [← he] at Ht
    exact signBall_eq_of_mem s t _ (hi s) Ht
  have HN := Fintype.card_le_of_injective i hI
  change Fintype.card (Fin d → Bool) ≤ Fintype.card (Fin n) at HN
  simp at HN
  omega

lemma transcript_hidden_ball (query : List ℝ → Point d) (s : Signs d) (a : ℝ) (n : ℕ)
    (hs : ∀ j < n, query (List.replicate j 0) ∉ signBall s) :
    transcript query (hiddenBallPayoff s a) n = List.replicate n 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [transcript, ih (fun j hj ↦ hs j (Nat.lt_succ_of_lt hj)), List.replicate_succ']
    rw [hiddenBallPayoff, indicator_of_notMem (hs n (Nat.lt_succ_self n))]

lemma one_third_either {p : ℝ} (a : ℝ) :
    p/3 ≤ |a-p| ∨ (2*p)/3 ≤ |a-2*p| := by
  by_contra! h
  have g1 := (abs_lt.mp h.1).2
  have g2 := (abs_lt.mp h.2).1
  linarith only [g1,g2]

lemma many_bad_balls (n : ℕ) (query : List ℝ → Point d) (decode : List ℝ → ℝ) :
    letI : DecidablePred (badBallEstimate n query decode) := Classical.decPred _
    2^d-n ≤ (Finset.univ.filter (badBallEstimate n query decode)).card := by
  classical
  let hit (s : Signs d) := ∃ j : Fin n, query (List.replicate j.val 0) ∈ signBall s
  have hH : Fintype.card {s : Signs d // hit s} ≤ n := by
    have h (s : {s : Signs d // hit s}) := s.property
    choose j hj using h
    have hJ : Function.Injective j := by
      intro x y hxy
      apply Subtype.ext
      have q := hj y
      rw [← hxy] at q
      exact signBall_eq_of_mem x.val y.val _ (hj x) q
    simpa using Fintype.card_le_of_injective j hJ
  have hMiss : 2^d-n ≤ Fintype.card {s : Signs d // ¬ hit s} := by
    rw [Fintype.card_subtype_compl]
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
    omega
  have hFound (s : {s : Signs d // ¬ hit s}) :
      ∃ b : Bool, badBallEstimate n query decode (s.val,b) := by
    have hs : ∀ j < n, query (List.replicate j 0) ∉ signBall s.val := by
      intro j hj h
      exact s.property ⟨⟨j,hj⟩,h⟩
    let p := (gamma d).real (signBall s.val)
    rcases one_third_either (p := p) (decode (List.replicate n 0)) with h1|h2
    · refine ⟨false,?_⟩
      simpa [badBallEstimate, packedPayoff, hiddenBall_mass, transcript_hidden_ball _ _ _ n hs, p] using h1
    · refine ⟨true,?_⟩
      simpa [badBallEstimate, packedPayoff, hiddenBall_mass, transcript_hidden_ball _ _ _ n hs, p] using h2
  choose height hheight using hFound
  let map (s : {s : Signs d // ¬ hit s}) :
      {p : Signs d × Bool // badBallEstimate n query decode p} := ⟨(s.val,height s),hheight s⟩
  have iMap : Function.Injective map := by
    intro s t he
    apply Subtype.ext
    exact congrArg (fun x ↦ x.val.1) he
  exact hMiss.trans (by simpa only [Fintype.card_subtype] using Fintype.card_le_of_injective map iMap)

lemma hidden_ball_bad_seed_count (d n m : ℕ) (_hm : 0 < m)
    (query : Fin m → List ℝ → Point d) (decode : Fin m → List ℝ → ℝ) :
    ∃ input : Signs d × Bool, m*(2^d-n) ≤ (2*2^d)*badSeedCount n query decode input := by
  classical
  have hsum :
      m*(2^d-n) ≤ ∑ j : Fin m, (Finset.univ.filter (badBallEstimate n (query j) (decode j))).card := by
    simpa using Finset.sum_le_sum (s := (Finset.univ : Finset (Fin m))) (fun j _ ↦
      many_bad_balls n (query j) (decode j))
  have hs :
      (∑ j : Fin m, (Finset.univ.filter (badBallEstimate n (query j) (decode j))).card) =
      ∑ p : Signs d × Bool, badSeedCount n query decode p := by
    simp only [badSeedCount, Finset.card_filter]
    rw [Finset.sum_comm]
  rw [hs] at hsum
  by_contra! h
  have hl := Finset.sum_lt_sum_of_nonempty (s := (Finset.univ : Finset (Signs d × Bool)))
    Finset.univ_nonempty (fun p _ ↦ h p)
  rw [← Finset.mul_sum] at hl
  have he : (∑ _ : Signs d × Bool, m*(2^d-n)) = (2*2^d)*(m*(2^d-n)) := by
    simp [Finset.sum_const, Fintype.card_prod, Nat.mul_comm (2^d)]
  rw [he] at hl
  exact (not_lt_of_ge (Nat.mul_le_mul_left (2*2^d) hsum)) hl

end Cubature
