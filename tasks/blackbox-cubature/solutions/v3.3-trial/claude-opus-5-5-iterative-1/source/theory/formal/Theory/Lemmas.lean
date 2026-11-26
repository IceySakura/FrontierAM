import Theory.Defs
import Mathlib.Probability.CDF

/-!
# Lemmas: the results for an arbitrary atomless probability measure on `ℝ`

Everything here is proved for a probability measure `μ` on `ℝ` without atoms (`NullSingletonClass μ`);
`Theory/Proofs.lean` specializes to the standard Gaussian measure `gaussianReal 0 1`.

The adversary arguments work with pairs `h ≤ g` of monotone `0/1` step functions (`StepFam`): every function `f`
with `h ≤ f ≤ g` gives the same answers to the queries made so far, and the integrals of `g` and `h` differ by the
mass of the "uncertainty interval" `{g ≠ h}`.
-/

open MeasureTheory Set Filter Topology ProbabilityTheory

namespace Cubature

/-- Monotone `0/1` step functions: the constants `0`, `1` and the indicators of `(x, ∞)` and `[x, ∞)`. -/
def StepFam : Set (ℝ → ℝ) :=
  {f | f = 0 ∨ f = 1 ∨ ∃ x : ℝ, f = (Ioi x).indicator 1 ∨ f = (Ici x).indicator 1}

lemma zero_mem_stepFam : (0 : ℝ → ℝ) ∈ StepFam := Or.inl rfl

lemma one_mem_stepFam : (1 : ℝ → ℝ) ∈ StepFam := Or.inr (Or.inl rfl)

lemma ioi_mem_stepFam (x : ℝ) : (Ioi x).indicator (1 : ℝ → ℝ) ∈ StepFam :=
  Or.inr (Or.inr ⟨x, Or.inl rfl⟩)

lemma ici_mem_stepFam (x : ℝ) : (Ici x).indicator (1 : ℝ → ℝ) ∈ StepFam :=
  Or.inr (Or.inr ⟨x, Or.inr rfl⟩)

lemma ioi_ind_apply (x t : ℝ) : (Ioi x).indicator (1 : ℝ → ℝ) t = if x < t then 1 else 0 := by
  simp [indicator_apply]

lemma ici_ind_apply (x t : ℝ) : (Ici x).indicator (1 : ℝ → ℝ) t = if x ≤ t then 1 else 0 := by
  simp [indicator_apply]

namespace StepFam

variable {f g h : ℝ → ℝ}

lemma zero_or_one (hf : f ∈ StepFam) (t : ℝ) : f t = 0 ∨ f t = 1 := by
  rcases hf with rfl | rfl | ⟨x, rfl | rfl⟩
  · exact Or.inl rfl
  · exact Or.inr rfl
  · rw [ioi_ind_apply]; split_ifs <;> simp
  · rw [ici_ind_apply]; split_ifs <;> simp

lemma mono (hf : f ∈ StepFam) : Monotone f := by
  rcases hf with rfl | rfl | ⟨x, rfl | rfl⟩
  · exact monotone_const
  · exact monotone_const
  · intro a b hab
    simp only [ioi_ind_apply]
    split_ifs with h1 h2 <;> first | exact absurd (h1.trans_le hab) h2 | norm_num
  · intro a b hab
    simp only [ici_ind_apply]
    split_ifs with h1 h2 <;> first | exact absurd (h1.trans hab) h2 | norm_num

lemma nonneg (hf : f ∈ StepFam) (t : ℝ) : 0 ≤ f t := by
  rcases (StepFam.zero_or_one hf) t with h | h <;> rw [h] <;> norm_num

lemma le_one (hf : f ∈ StepFam) (t : ℝ) : f t ≤ 1 := by
  rcases (StepFam.zero_or_one hf) t with h | h <;> rw [h] <;> norm_num

lemma mem_stepClass (hf : f ∈ StepFam) : f ∈ StepClass := by
  rcases hf with rfl | rfl | ⟨x, rfl | rfl⟩
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)
  · refine Or.inr (Or.inr ⟨x, fun t ht => ?_, fun t ht => ?_⟩)
    · rw [ioi_ind_apply]; simp [not_lt.mpr ht.le]
    · rw [ioi_ind_apply]; simp [ht]
  · refine Or.inr (Or.inr ⟨x, fun t ht => ?_, fun t ht => ?_⟩)
    · rw [ici_ind_apply]; simp [not_le.mpr ht]
    · rw [ici_ind_apply]; simp [ht.le]

lemma measurable (hf : f ∈ StepFam) : Measurable f := (StepFam.mono hf).measurable

lemma integrable (hf : f ∈ StepFam) (μ : Measure ℝ) [IsFiniteMeasure μ] : Integrable f μ := by
  refine Integrable.of_bound (C := 1) (StepFam.measurable hf).aestronglyMeasurable (ae_of_all _ fun t => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg ((StepFam.nonneg hf) t)]
  exact (StepFam.le_one hf) t

/-- If two functions `h ≤ g` of the family differ at `x`, then `g x = 1` and `h x = 0`. -/
lemma pair_at (hg : g ∈ StepFam) (hh : h ∈ StepFam) (hle : h ≤ g) {x : ℝ} (hx : g x ≠ h x) :
    g x = 1 ∧ h x = 0 := by
  have := hle x
  rcases (StepFam.zero_or_one hg) x with h1 | h1 <;> rcases (StepFam.zero_or_one hh) x with h2 | h2 <;>
    simp_all <;> linarith

lemma le_ioi (hh : h ∈ StepFam) {x : ℝ} (hx : h x = 0) : h ≤ (Ioi x).indicator 1 := by
  intro t
  rw [ioi_ind_apply]
  split_ifs with ht
  · exact (StepFam.le_one hh) t
  · have := (StepFam.mono hh) (not_lt.mp ht)
    linarith

lemma ioi_le_ici (x : ℝ) : (Ioi x).indicator (1 : ℝ → ℝ) ≤ (Ici x).indicator 1 := by
  intro t
  rw [ioi_ind_apply, ici_ind_apply]
  split_ifs with h1 h2 <;> first | exact absurd h1.le h2 | norm_num

lemma ici_le (hg : g ∈ StepFam) {x : ℝ} (hx : g x = 1) : (Ici x).indicator 1 ≤ g := by
  intro t
  rw [ici_ind_apply]
  split_ifs with ht
  · have := (StepFam.mono hg) ht
    linarith
  · exact (StepFam.nonneg hg) t

lemma le_ici (hh : h ∈ StepFam) {x : ℝ} (hx : h x = 0) : h ≤ (Ici x).indicator 1 :=
  (le_ioi hh hx).trans (ioi_le_ici x)

lemma ioi_le (hg : g ∈ StepFam) {x : ℝ} (hx : g x = 1) : (Ioi x).indicator 1 ≤ g :=
  (ioi_le_ici x).trans (ici_le hg hx)

end StepFam

variable {μ : Measure ℝ} [IsProbabilityMeasure μ] [NullSingletonClass μ]

lemma integral_ici_eq_ioi (x : ℝ) :
    ∫ t, (Ici x).indicator (1 : ℝ → ℝ) t ∂μ = ∫ t, (Ioi x).indicator (1 : ℝ → ℝ) t ∂μ := by
  rw [integral_indicator_one measurableSet_Ici, integral_indicator_one measurableSet_Ioi]
  exact measureReal_congr Ioi_ae_eq_Ici.symm

lemma integral_mono_stepFam {g h : ℝ → ℝ} (hg : g ∈ StepFam) (hh : h ∈ StepFam) (hle : h ≤ g) :
    ∫ t, h t ∂μ ≤ ∫ t, g t ∂μ :=
  integral_mono ((StepFam.integrable hh) μ) ((StepFam.integrable hg) μ) hle

/-- Two functions with the same information and integrals at distance `≥ 2ε`: one has error `≥ ε`. -/
lemma two_point {o a b ε : ℝ} (hab : 2 * ε ≤ a - b) : ε ≤ |o - a| ∨ ε ≤ |o - b| := by
  by_contra hc
  rw [not_or, not_le, not_le] at hc
  obtain ⟨h1, h2⟩ := hc
  rw [abs_lt] at h1 h2
  linarith [h1.1, h1.2, h2.1, h2.2]

lemma observe_succ {n : ℕ} (A : AdaptiveAlg n) (f : ℝ → ℝ) (k : ℕ) :
    A.observe f (k + 1) = Fin.snoc (α := fun _ => ℝ) (A.observe f k) (f (A.node k (A.observe f k))) := rfl

/-- If `f` and `g` agree at all points queried on `g` before step `k`, then the first `k` observed values agree. -/
lemma observe_eq_of_eq {n : ℕ} (A : AdaptiveAlg n) (f g : ℝ → ℝ) (k : ℕ)
    (hfg : ∀ j < k, f (A.node j (A.observe g j)) = g (A.node j (A.observe g j))) :
    A.observe f k = A.observe g k := by
  induction k with
  | zero => funext j; exact j.elim0
  | succ k ih =>
    have ih' := ih (fun j hj => hfg j (by omega))
    rw [observe_succ, observe_succ, ih', hfg k (by omega)]

/-! ### R1: the adversary for adaptive algorithms on `J` -/

/-- The adversary: after `k` queries of an adaptive algorithm there is a pair `h ≤ g` of monotone `0/1` steps
whose integrals differ by at least `2^{-k}` and such that every `f` with `h ≤ f ≤ g` gives the same `k` values. -/
theorem adversary {n : ℕ} (A : AdaptiveAlg n) (k : ℕ) :
    ∃ g ∈ StepFam, ∃ h ∈ StepFam, h ≤ g ∧ 1 / (2 : ℝ) ^ k ≤ (∫ t, g t ∂μ) - ∫ t, h t ∂μ ∧
      ∀ f : ℝ → ℝ, h ≤ f → f ≤ g → A.observe f k = A.observe g k := by
  induction k with
  | zero =>
    refine ⟨1, one_mem_stepFam, 0, zero_mem_stepFam, fun t => by norm_num, by simp, ?_⟩
    intro f _ _
    funext j
    exact j.elim0
  | succ k ih =>
    obtain ⟨g, hg, h, hh, hle, hgap, hobs⟩ := ih
    set x := A.node k (A.observe g k) with hx_def
    have obs_succ : ∀ f, h ≤ f → f ≤ g →
        A.observe f (k + 1) = Fin.snoc (α := fun _ => ℝ) (A.observe g k) (f x) := by
      intro f h1 h2
      rw [observe_succ, hobs f h1 h2]
    have hpow : 1 / (2 : ℝ) ^ (k + 1) = 1 / 2 ^ k / 2 := by rw [pow_succ]; ring
    have hpos : 0 < 1 / (2 : ℝ) ^ k := by positivity
    by_cases hx : g x = h x
    · refine ⟨g, hg, h, hh, hle, by rw [hpow]; linarith, ?_⟩
      intro f h1 h2
      rw [obs_succ f h1 h2, obs_succ g hle le_rfl]
      congr 1
      exact le_antisymm (h2 x) (hx ▸ h1 x)
    · obtain ⟨hgx, hhx⟩ := StepFam.pair_at hg hh hle hx
      have hI := integral_ici_eq_ioi (μ := μ) x
      by_cases hc : (∫ t, (Ioi x).indicator 1 t ∂μ) - ∫ t, h t ∂μ ≤
          (∫ t, g t ∂μ) - ∫ t, (Ici x).indicator 1 t ∂μ
      · refine ⟨g, hg, (Ici x).indicator 1, ici_mem_stepFam x, StepFam.ici_le hg hgx,
          by rw [hpow]; linarith, ?_⟩
        intro f h1 h2
        have h1' : h ≤ f := (StepFam.le_ici hh hhx).trans h1
        rw [obs_succ f h1' h2, obs_succ g hle le_rfl]
        congr 1
        have := h1 x
        rw [ici_ind_apply] at this; simp only [le_refl, ite_true] at this
        exact le_antisymm (h2 x) (hgx ▸ this)
      · rw [not_le] at hc
        refine ⟨(Ioi x).indicator 1, ioi_mem_stepFam x, h, hh, StepFam.le_ioi hh hhx,
          by rw [hpow]; linarith, ?_⟩
        intro f h1 h2
        have hg' := StepFam.ioi_le hg hgx
        rw [obs_succ f h1 (h2.trans hg'), obs_succ _ (StepFam.le_ioi hh hhx) hg']
        congr 1
        have := h2 x
        rw [ioi_ind_apply] at this ⊢; simp only [lt_irrefl, ite_false] at this ⊢
        exact le_antisymm this (hhx ▸ h1 x)

/-- **R1** for an atomless probability measure. -/
theorem adaptive_lower_bound_general {n : ℕ} (A : AdaptiveAlg n) :
    ∃ f ∈ StepClass, 1 / (2 : ℝ) ^ (n + 1) ≤ |A.output (A.info f) - ∫ t, f t ∂μ| := by
  obtain ⟨g, hg, h, hh, hle, hgap, hobs⟩ := adversary (μ := μ) A n
  have hinfo : A.info h = A.info g := hobs h le_rfl hle
  have hε : 2 * (1 / (2 : ℝ) ^ (n + 1)) ≤ (∫ t, g t ∂μ) - ∫ t, h t ∂μ := by
    rw [pow_succ]
    calc 2 * (1 / ((2 : ℝ) ^ n * 2)) = 1 / 2 ^ n := by field_simp
      _ ≤ _ := hgap
  rcases two_point (o := A.output (A.info g)) hε with h1 | h1
  · exact ⟨g, (StepFam.mem_stepClass hg), h1⟩
  · exact ⟨h, (StepFam.mem_stepClass hh), by rw [hinfo]; exact h1⟩

/-! ### The gap lemma (pigeonhole) for R3 and R4 -/

/-- Pigeonhole in the form needed for R3 and R4: if at most `m` points of a finite set `S` lie in the uncertainty
region `{g ≠ h}` of a pair `h ≤ g`, then some pair `h' ≤ g'` agrees on all of `S` and still has integral gap at
least a `1/(m+1)` fraction of the original one. Proof: induction on `m`, splitting at the smallest point of `S` in
the uncertainty region. -/
theorem gap_pairs (m : ℕ) : ∀ g ∈ StepFam, ∀ h ∈ StepFam, h ≤ g → ∀ S : Finset ℝ,
    (S.filter (fun t => g t ≠ h t)).card ≤ m →
    ∃ g' ∈ StepFam, ∃ h' ∈ StepFam, h' ≤ g' ∧ (∀ t ∈ S, g' t = h' t) ∧
      ((∫ t, g t ∂μ) - ∫ t, h t ∂μ) / ((m : ℝ) + 1) ≤ (∫ t, g' t ∂μ) - ∫ t, h' t ∂μ := by
  induction m with
  | zero =>
    intro g hg h hh hle S hS
    refine ⟨g, hg, h, hh, hle, ?_, by simp⟩
    intro t ht
    by_contra hne
    have hmem : t ∈ S.filter (fun t => g t ≠ h t) := Finset.mem_filter.mpr ⟨ht, hne⟩
    have := Finset.card_pos.mpr ⟨t, hmem⟩
    omega
  | succ m ih =>
    intro g hg h hh hle S hS
    have hD : 0 ≤ (∫ t, g t ∂μ) - ∫ t, h t ∂μ := sub_nonneg.mpr (integral_mono_stepFam hg hh hle)
    rcases (S.filter (fun t => g t ≠ h t)).eq_empty_or_nonempty with hTe | hTne
    · refine ⟨g, hg, h, hh, hle, ?_, ?_⟩
      · intro t ht
        by_contra hne
        have hmem : t ∈ S.filter (fun t => g t ≠ h t) := Finset.mem_filter.mpr ⟨ht, hne⟩
        rw [hTe] at hmem
        simp at hmem
      · exact div_le_self hD (by push_cast; linarith)
    · set T := S.filter (fun t => g t ≠ h t) with hT
      set x := T.min' hTne with hx_def
      have hxT : x ∈ T := T.min'_mem hTne
      have hxS : x ∈ S := (Finset.mem_filter.mp hxT).1
      have hx : g x ≠ h x := (Finset.mem_filter.mp hxT).2
      obtain ⟨hgx, hhx⟩ := StepFam.pair_at hg hh hle hx
      have hI := integral_ici_eq_ioi (μ := μ) x
      have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity
      by_cases hc : ((∫ t, g t ∂μ) - ∫ t, h t ∂μ) / (((m + 1 : ℕ) : ℝ) + 1) ≤
          (∫ t, g t ∂μ) - ∫ t, (Ici x).indicator 1 t ∂μ
      · refine ⟨g, hg, (Ici x).indicator 1, ici_mem_stepFam x, StepFam.ici_le hg hgx, ?_, hc⟩
        intro t ht
        rw [ici_ind_apply]
        by_cases htx : x ≤ t
        · simp only [htx, ite_true]
          have := StepFam.mono hg htx
          have := StepFam.le_one hg t
          linarith
        · simp only [htx, ite_false]
          have htT : t ∉ T := fun htT => htx (T.min'_le t htT)
          have hgh : g t = h t := by
            by_contra hne
            exact htT (Finset.mem_filter.mpr ⟨ht, hne⟩)
          have h1 := StepFam.mono hh (not_le.mp htx).le
          have h2 := StepFam.nonneg hh t
          rw [hgh]
          linarith
      · rw [not_le] at hc
        have hT' : (S.filter (fun t => (Ioi x).indicator (1 : ℝ → ℝ) t ≠ h t)).card ≤ m := by
          have hsub : S.filter (fun t => (Ioi x).indicator (1 : ℝ → ℝ) t ≠ h t) ⊆ T.erase x := by
            intro t ht
            obtain ⟨htS, hne⟩ := Finset.mem_filter.mp ht
            rw [ioi_ind_apply] at hne
            by_cases hxt : x < t
            · simp only [hxt, ite_true] at hne
              refine Finset.mem_erase.mpr ⟨hxt.ne', Finset.mem_filter.mpr ⟨htS, ?_⟩⟩
              have hh0 : h t = 0 := by
                rcases StepFam.zero_or_one hh t with h0 | h0
                · exact h0
                · exact absurd h0.symm hne
              have := StepFam.mono hg hxt.le
              have := StepFam.le_one hg t
              rw [hh0]
              intro hg0
              linarith
            · simp only [hxt, ite_false] at hne
              exfalso
              have h1 := StepFam.mono hh (not_lt.mp hxt)
              have h2 := StepFam.nonneg hh t
              exact hne (by linarith)
          have := Finset.card_le_card hsub
          rw [Finset.card_erase_of_mem hxT] at this
          omega
        obtain ⟨g', hg', h', hh', hle', hS', hbound⟩ :=
          ih ((Ioi x).indicator 1) (ioi_mem_stepFam x) h hh (StepFam.le_ioi hh hhx) S hT'
        refine ⟨g', hg', h', hh', hle', hS', le_trans ?_ hbound⟩
        push_cast at hc ⊢
        rw [div_le_div_iff₀ (by linarith) hm]
        rw [lt_div_iff₀ (by linarith), hI] at hc
        nlinarith [hc]

/-- **R3** for an atomless probability measure. -/
theorem nonadaptive_lower_bound_general {n : ℕ} (A : NonadaptiveAlg n) :
    ∃ f ∈ StepClass, 1 / (2 * ((n : ℝ) + 1)) ≤ |A.output (A.info f) - ∫ t, f t ∂μ| := by
  classical
  set S : Finset ℝ := Finset.univ.image A.nodes
  have hS : (S.filter (fun t => (1 : ℝ → ℝ) t ≠ (0 : ℝ → ℝ) t)).card ≤ n :=
    (Finset.card_filter_le _ _).trans (Finset.card_image_le.trans (by simp))
  obtain ⟨g, hg, h, hh, hle, hagree, hbound⟩ :=
    gap_pairs (μ := μ) n 1 one_mem_stepFam 0 zero_mem_stepFam (fun t => by norm_num) S hS
  have hinfo : A.info h = A.info g :=
    funext fun i => (hagree _ (Finset.mem_image_of_mem _ (Finset.mem_univ i))).symm
  have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hε : 2 * (1 / (2 * ((n : ℝ) + 1))) ≤ (∫ t, g t ∂μ) - ∫ t, h t ∂μ := by
    have e : 2 * (1 / (2 * ((n : ℝ) + 1))) = 1 / ((n : ℝ) + 1) := by field_simp
    rw [e]
    simpa using hbound
  rcases two_point (o := A.output (A.info g)) hε with h1 | h1
  · exact ⟨g, StepFam.mem_stepClass hg, h1⟩
  · exact ⟨h, StepFam.mem_stepClass hh, by rw [hinfo]; exact h1⟩

/-! ### R4: two crossings -/

/-- The uncertainty region `{g ≠ h}` of a pair `h ≤ g` of monotone `0/1` steps is order-connected. -/
lemma gap_ordConnected {g h : ℝ → ℝ} (hg : g ∈ StepFam) (hh : h ∈ StepFam) (hle : h ≤ g) :
    {t | g t ≠ h t}.OrdConnected := by
  constructor
  intro a ha b hb t ⟨hat, htb⟩
  obtain ⟨hga, -⟩ := StepFam.pair_at hg hh hle ha
  obtain ⟨-, hhb⟩ := StepFam.pair_at hg hh hle hb
  have h1 := StepFam.mono hg hat
  have h2 := StepFam.mono hh htb
  have h3 := StepFam.le_one hg t
  have h4 := StepFam.nonneg hh t
  show g t ≠ h t
  intro he
  linarith

/-- The integral gap of a pair equals the mass of its uncertainty region. -/
lemma integral_gap_eq {g h : ℝ → ℝ} (hg : g ∈ StepFam) (hh : h ∈ StepFam) (hle : h ≤ g) :
    (∫ t, g t ∂μ) - ∫ t, h t ∂μ = μ.real {t | g t ≠ h t} := by
  have hmeas : MeasurableSet {t | g t ≠ h t} :=
    (measurableSet_eq_fun (StepFam.measurable hg) (StepFam.measurable hh)).compl
  rw [← integral_sub (StepFam.integrable hg μ) (StepFam.integrable hh μ), ← integral_indicator_one hmeas]
  congr 1
  funext t
  by_cases ht : g t = h t
  · rw [indicator_of_notMem (show t ∉ {t | g t ≠ h t} from fun h' => h' ht), ht, sub_self]
  · obtain ⟨h1, h2⟩ := StepFam.pair_at hg hh hle ht
    rw [indicator_of_mem (show t ∈ {t | g t ≠ h t} from ht), h1, h2]
    simp

/-- A bounded order-connected set has an indicator function in the class `P`. -/
lemma indicator_mem_intervalClass {B : Set ℝ} (hB : B.OrdConnected) (hbb : BddBelow B) (hba : BddAbove B) :
    B.indicator 1 ∈ IntervalClass := by
  rcases B.eq_empty_or_nonempty with rfl | hne
  · exact ⟨0, 0, le_rfl, fun t h1 h2 => absurd (h1.trans h2) (lt_irrefl _), fun t _ => by simp⟩
  · refine ⟨sInf B, sSup B, csInf_le_csSup hne hbb hba, ?_, ?_⟩
    · intro t h1 h2
      obtain ⟨a, haB, hat⟩ := exists_lt_of_csInf_lt hne h1
      obtain ⟨b, hbB, htb⟩ := exists_lt_of_lt_csSup hne h2
      have htB : t ∈ B := hB.out haB hbB ⟨hat.le, htb.le⟩
      rw [indicator_of_mem htB]
      rfl
    · intro t ht
      have htB : t ∉ B := by
        intro htB
        have := subset_Icc_csInf_csSup hbb hba htB
        rcases ht with ht | ht
        · exact absurd this.1 (not_le.mpr ht)
        · exact absurd this.2 (not_le.mpr ht)
      rw [indicator_of_notMem htB]

omit [NullSingletonClass μ] in
/-- Continuity from below: bounded pieces `G ∩ (-R, R)` exhaust the mass of `G`. -/
lemma exists_bounded_part {G : Set ℝ} {τ : ℝ} (hτ : τ < μ.real G) :
    ∃ R : ℕ, τ < μ.real (G ∩ Ioo (-(R : ℝ)) R) := by
  have hmono : Monotone (fun R : ℕ => G ∩ Ioo (-(R : ℝ)) R) := by
    intro a b hab t ⟨htG, ht1, ht2⟩
    have : (a : ℝ) ≤ b := Nat.cast_le.mpr hab
    exact ⟨htG, by linarith, by linarith⟩
  have hU : (⋃ R : ℕ, G ∩ Ioo (-(R : ℝ)) R) = G := by
    ext t
    simp only [mem_iUnion, mem_inter_iff, mem_Ioo]
    constructor
    · rintro ⟨R, h, -⟩
      exact h
    · intro h
      obtain ⟨R, hR⟩ := exists_nat_gt |t|
      exact ⟨R, h, by linarith [neg_abs_le t], by linarith [le_abs_self t]⟩
  have ht := tendsto_measure_iUnion_atTop (μ := μ) hmono
  rw [hU] at ht
  have ht' : Tendsto (fun R : ℕ => μ.real (G ∩ Ioo (-(R : ℝ)) R)) atTop (𝓝 (μ.real G)) :=
    (ENNReal.tendsto_toReal (measure_ne_top μ G)).comp ht
  exact (ht'.eventually (lt_mem_nhds hτ)).exists

/-- **R4** for an atomless probability measure. -/
theorem adaptive_lower_bound_interval_general {n : ℕ} (A : AdaptiveAlg n) :
    ∃ f ∈ IntervalClass, 1 / (2 * ((n : ℝ) + 1)) ≤ |A.output (A.info f) - ∫ t, f t ∂μ| := by
  classical
  set o := A.output (A.info 0) with ho_def
  have h0mem : (0 : ℝ → ℝ) ∈ IntervalClass :=
    ⟨0, 0, le_rfl, fun t h1 h2 => absurd (h1.trans h2) (lt_irrefl _), fun _ _ => rfl⟩
  have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  by_cases ho : 1 / (2 * ((n : ℝ) + 1)) ≤ |o - 0|
  · refine ⟨0, h0mem, ?_⟩
    simpa using ho
  · rw [not_le, sub_zero, abs_lt] at ho
    -- the points queried when the algorithm is run on `f = 0`
    set S : Finset ℝ := Finset.univ.image (fun j : Fin n => A.node j (A.observe 0 j)) with hS_def
    have hS : (S.filter (fun t => (1 : ℝ → ℝ) t ≠ (0 : ℝ → ℝ) t)).card ≤ n :=
      (Finset.card_filter_le _ _).trans (Finset.card_image_le.trans (by simp))
    obtain ⟨g, hg, h, hh, hle, hagree, hbound⟩ :=
      gap_pairs (μ := μ) n 1 one_mem_stepFam 0 zero_mem_stepFam (fun t => by norm_num) S hS
    set G := {t | g t ≠ h t} with hG_def
    have hGmeas : MeasurableSet G :=
      (measurableSet_eq_fun (StepFam.measurable hg) (StepFam.measurable hh)).compl
    have hGmass : 1 / ((n : ℝ) + 1) ≤ μ.real G := by
      rw [← integral_gap_eq hg hh hle]
      simpa using hbound
    have hτ : o + 1 / (2 * ((n : ℝ) + 1)) < μ.real G := by
      have e : 1 / ((n : ℝ) + 1) = 2 * (1 / (2 * ((n : ℝ) + 1))) := by field_simp
      rw [e] at hGmass
      linarith [ho.2]
    obtain ⟨R, hR⟩ := exists_bounded_part hτ
    set B := G ∩ Ioo (-(R : ℝ)) R with hB_def
    have hBord : B.OrdConnected := (gap_ordConnected hg hh hle).inter ordConnected_Ioo
    refine ⟨B.indicator 1, indicator_mem_intervalClass hBord (bddBelow_Ioo.mono inter_subset_right)
      (bddAbove_Ioo.mono inter_subset_right), ?_⟩
    have hinfo : A.info (B.indicator 1) = A.info 0 := by
      apply observe_eq_of_eq
      intro j hj
      have hmem : A.node j (A.observe 0 j) ∈ S :=
        Finset.mem_image.mpr ⟨⟨j, hj⟩, Finset.mem_univ _, rfl⟩
      have hnot : A.node j (A.observe 0 j) ∉ B := fun hB => hB.1 (hagree _ hmem)
      rw [indicator_of_notMem hnot]
      rfl
    rw [hinfo, integral_indicator_one (hGmeas.inter measurableSet_Ioo), ← ho_def, abs_sub_comm]
    exact le_trans (by linarith) (le_abs_self _)

/-! ### R2: bisection in quantile coordinates -/

/-- The distribution function of an atomless probability measure on `ℝ` is continuous. -/
lemma continuous_cdf_of_nullSingleton : Continuous (cdf μ) := by
  rw [continuous_iff_continuousAt]
  intro x
  rw [continuousAt_iff_continuous_left'_right']
  constructor
  · rw [(cdf μ).mono.continuousWithinAt_Iio_iff_leftLim_eq]
    have h := (cdf μ).measure_singleton x
    rw [measure_cdf, measure_singleton] at h
    have h1 : cdf μ x - Function.leftLim (cdf μ) x ≤ 0 := ENNReal.ofReal_eq_zero.mp h.symm
    have h2 : Function.leftLim (cdf μ) x ≤ cdf μ x := (cdf μ).mono.leftLim_le le_rfl
    linarith
  · exact ((cdf μ).right_continuous x).mono Ioi_subset_Ici_self

omit [NullSingletonClass μ] in
lemma exists_cdf_eq_of_continuous (hc : Continuous (cdf μ)) {p : ℝ} (h0 : 0 < p) (h1 : p < 1) :
    ∃ x, cdf μ x = p := by
  obtain ⟨a, ha⟩ := ((tendsto_cdf_atBot μ).eventually (gt_mem_nhds h0)).exists
  obtain ⟨b, hb⟩ := ((tendsto_cdf_atTop μ).eventually (lt_mem_nhds h1)).exists
  exact intermediate_value_univ a b hc ⟨ha.le, hb.le⟩

open Classical in
/-- A quantile function: `cdf μ (quantile μ p) = p` whenever such a point exists. -/
noncomputable def quantile (ν : Measure ℝ) (p : ℝ) : ℝ :=
  if h : ∃ x, cdf ν x = p then h.choose else 0

lemma cdf_quantile {p : ℝ} (h0 : 0 < p) (h1 : p < 1) : cdf μ (quantile μ p) = p := by
  have h := exists_cdf_eq_of_continuous (μ := μ) continuous_cdf_of_nullSingleton h0 h1
  rw [quantile, dite_eq_left h]
  exact h.choose_spec

/-- The state of the bisection after observing the values `y`: an interval `[lo, hi]` of probability levels
containing the level `cdf μ s` of the jump. Each value `0` moves `lo` up to the midpoint, any other value moves `hi`
down to the midpoint. -/
noncomputable def bisect : (k : ℕ) → (Fin k → ℝ) → ℝ × ℝ
  | 0, _ => (0, 1)
  | k + 1, y =>
    if y (Fin.last k) = 0 then
      (((bisect k (Fin.init y)).1 + (bisect k (Fin.init y)).2) / 2, (bisect k (Fin.init y)).2)
    else
      ((bisect k (Fin.init y)).1, ((bisect k (Fin.init y)).1 + (bisect k (Fin.init y)).2) / 2)

lemma bisect_snoc (k : ℕ) (v : Fin k → ℝ) (a : ℝ) :
    bisect (k + 1) (Fin.snoc (α := fun _ => ℝ) v a) =
      if a = 0 then (((bisect k v).1 + (bisect k v).2) / 2, (bisect k v).2)
      else ((bisect k v).1, ((bisect k v).1 + (bisect k v).2) / 2) := by
  simp only [bisect, Fin.init_snoc, Fin.snoc_last]

/-- Bisection in quantile coordinates: query the point whose level is the midpoint of the current interval of
levels; output `1 -` the midpoint of the final interval. -/
noncomputable def bisectionAlg (ν : Measure ℝ) (n : ℕ) : AdaptiveAlg n where
  node k y := quantile ν (((bisect k y).1 + (bisect k y).2) / 2)
  output y := 1 - ((bisect n y).1 + (bisect n y).2) / 2

omit [NullSingletonClass μ] in
lemma integral_step_eq {f : ℝ → ℝ} {s : ℝ} (hμ : μ {s} = 0) (h0 : ∀ t, t < s → f t = 0)
    (h1 : ∀ t, s < t → f t = 1) : ∫ t, f t ∂μ = 1 - cdf μ s := by
  have hae : f =ᵐ[μ] (Ioi s).indicator 1 := by
    have hne : ∀ᵐ t ∂μ, t ≠ s := by
      rw [ae_iff]
      simpa using hμ
    filter_upwards [hne] with t ht
    rw [ioi_ind_apply]
    rcases lt_or_gt_of_ne ht with h | h
    · rw [h0 t h, ite_eq_right (not_lt.mpr h.le)]
    · rw [h1 t h, ite_eq_left h]
  rw [integral_congr_ae hae, integral_indicator_one measurableSet_Ioi, cdf_eq_real, ← compl_Iic,
    measureReal_compl measurableSet_Iic]
  simp

/-- What one value of `f ∈ J` at `x` reveals about the level `1 - ∫ f` of the jump. -/
lemma stepClass_query {f : ℝ → ℝ} (hf : f ∈ StepClass) (x : ℝ) :
    (f x = 0 → cdf μ x ≤ 1 - ∫ t, f t ∂μ) ∧ (f x ≠ 0 → 1 - ∫ t, f t ∂μ ≤ cdf μ x) := by
  rcases hf with rfl | rfl | ⟨s, h0, h1⟩
  · refine ⟨fun _ => ?_, fun h => absurd rfl h⟩
    simpa using cdf_le_one μ x
  · refine ⟨fun h => absurd h one_ne_zero, fun _ => ?_⟩
    simpa using cdf_nonneg μ x
  · rw [integral_step_eq (measure_singleton s) h0 h1]
    refine ⟨fun hx => ?_, fun hx => ?_⟩
    · have hxs : x ≤ s := by
        by_contra hlt
        rw [h1 x (not_le.mp hlt)] at hx
        exact one_ne_zero hx
      have := (cdf μ).mono hxs
      linarith
    · have hsx : s ≤ x := by
        by_contra hlt
        exact hx (h0 x (not_le.mp hlt))
      have := (cdf μ).mono hsx
      linarith

lemma stepClass_integral_bounds {f : ℝ → ℝ} (hf : f ∈ StepClass) :
    0 ≤ ∫ t, f t ∂μ ∧ ∫ t, f t ∂μ ≤ 1 := by
  rcases hf with rfl | rfl | ⟨s, h0, h1⟩
  · simp
  · simp
  · rw [integral_step_eq (measure_singleton s) h0 h1]
    constructor <;> linarith [cdf_nonneg μ s, cdf_le_one μ s]

/-- The bisection invariant: after `k` steps the interval of levels has length `2^{-k}`, lies in `[0, 1]`, and
contains the level `1 - ∫ f` of every `f ∈ J`. -/
lemma bisect_invariant {f : ℝ → ℝ} (hf : f ∈ StepClass) (n k : ℕ) :
    0 ≤ (bisect k ((bisectionAlg μ n).observe f k)).1 ∧ (bisect k ((bisectionAlg μ n).observe f k)).2 ≤ 1 ∧
      (bisect k ((bisectionAlg μ n).observe f k)).2 - (bisect k ((bisectionAlg μ n).observe f k)).1 =
        1 / (2 : ℝ) ^ k ∧
      (bisect k ((bisectionAlg μ n).observe f k)).1 ≤ 1 - ∫ t, f t ∂μ ∧
      1 - ∫ t, f t ∂μ ≤ (bisect k ((bisectionAlg μ n).observe f k)).2 := by
  induction k with
  | zero =>
    obtain ⟨h0, h1⟩ := stepClass_integral_bounds (μ := μ) hf
    simp only [bisect, pow_zero]
    refine ⟨le_rfl, le_rfl, by norm_num, by linarith, by linarith⟩
  | succ k ih =>
    obtain ⟨hlo, hhi, hlen, hl, hr⟩ := ih
    set v := (bisectionAlg μ n).observe f k with hv
    set lo := (bisect k v).1
    set hi := (bisect k v).2
    have hmid0 : 0 < (lo + hi) / 2 := by
      have : 0 < 1 / (2 : ℝ) ^ k := by positivity
      linarith
    have hmid1 : (lo + hi) / 2 < 1 := by
      have : 0 < 1 / (2 : ℝ) ^ k := by positivity
      linarith
    have hq : cdf μ ((bisectionAlg μ n).node k v) = (lo + hi) / 2 := cdf_quantile hmid0 hmid1
    have hpow : 1 / (2 : ℝ) ^ (k + 1) = 1 / 2 ^ k / 2 := by rw [pow_succ]; ring
    rw [observe_succ, bisect_snoc, ← hv]
    obtain ⟨hq0, hq1⟩ := stepClass_query (μ := μ) hf ((bisectionAlg μ n).node k v)
    rw [hq] at hq0 hq1
    split_ifs with hx
    · refine ⟨by linarith, hhi, by rw [hpow]; linarith, hq0 hx, hr⟩
    · refine ⟨hlo, by linarith, by rw [hpow]; linarith, hl, hq1 hx⟩

/-- **R2** for an atomless probability measure: bisection in quantile coordinates has error `≤ 2^{-(n+1)}`. -/
theorem bisection_upper_bound_general (n : ℕ) :
    ∃ A : AdaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂μ| ≤ 1 / (2 : ℝ) ^ (n + 1) := by
  refine ⟨bisectionAlg μ n, fun f hf => ?_⟩
  obtain ⟨-, -, hlen, hl, hr⟩ := bisect_invariant (μ := μ) hf n n
  have hpow : 1 / (2 : ℝ) ^ (n + 1) = 1 / 2 ^ n / 2 := by rw [pow_succ]; ring
  show |1 - ((bisect n ((bisectionAlg μ n).observe f n)).1 + (bisect n ((bisectionAlg μ n).observe f n)).2) / 2
    - ∫ t, f t ∂μ| ≤ _
  rw [abs_le, hpow]
  constructor <;> linarith

/-! ### Matching nonadaptive upper bound on `J` (quantile nodes) -/

lemma cdf_quantile_node (n : ℕ) (i : Fin n) :
    cdf μ (quantile μ (((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1))) = ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) := by
  have hi : ((i : ℕ) : ℝ) + 1 ≤ n := by exact_mod_cast i.isLt
  apply cdf_quantile
  · positivity
  · rw [div_lt_one (by positivity)]
    linarith

/-- Nonadaptive algorithm with nodes at the quantiles of levels `1/(n+1), …, n/(n+1)`: if `k` of the `n` values
are `0`, the jump level lies in `[k/(n+1), (k+1)/(n+1)]`; the estimate is `1 - (k + 1/2)/(n+1)`. -/
noncomputable def quantileRule (ν : Measure ℝ) (n : ℕ) : NonadaptiveAlg n where
  nodes i := quantile ν (((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1))
  output y := 1 - (((Finset.univ.filter (fun i => y i = 0)).card : ℝ) + 1 / 2) / ((n : ℝ) + 1)

/-- **R3, matching upper bound** for an atomless probability measure. -/
theorem nonadaptive_upper_bound_general (n : ℕ) :
    ∃ A : NonadaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂μ| ≤ 1 / (2 * ((n : ℝ) + 1)) := by
  classical
  refine ⟨quantileRule μ n, fun f hf => ?_⟩
  set p := 1 - ∫ t, f t ∂μ with hp_def
  obtain ⟨hI0, hI1⟩ := stepClass_integral_bounds (μ := μ) hf
  set Z := Finset.univ.filter (fun i : Fin n => f ((quantileRule μ n).nodes i) = 0) with hZ
  set k := Z.card with hk
  have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hin : ∀ i : Fin n, i ∈ Z → ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) ≤ p := by
    intro i hi
    have := (stepClass_query (μ := μ) hf ((quantileRule μ n).nodes i)).1 (Finset.mem_filter.mp hi).2
    rwa [show (quantileRule μ n).nodes i = quantile μ (((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1)) from rfl,
      cdf_quantile_node] at this
  have hout : ∀ i : Fin n, i ∉ Z → p ≤ ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) := by
    intro i hi
    have hne : f ((quantileRule μ n).nodes i) ≠ 0 := fun h => hi (Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩)
    have := (stepClass_query (μ := μ) hf ((quantileRule μ n).nodes i)).2 hne
    rwa [show (quantileRule μ n).nodes i = quantile μ (((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1)) from rfl,
      cdf_quantile_node] at this
  -- lower bound `k/(n+1) ≤ p`
  have hA : (k : ℝ) / ((n : ℝ) + 1) ≤ p := by
    rcases Nat.eq_zero_or_pos k with h0 | hpos
    · rw [h0]; simp; linarith
    · have : ∃ i ∈ Z, k ≤ (i : ℕ) + 1 := by
        by_contra hcon
        push Not at hcon
        have hsub : Z.map Fin.valEmbedding ⊆ Finset.range (k - 1) := by
          intro j hj
          obtain ⟨i, hiZ, rfl⟩ := Finset.mem_map.mp hj
          have := hcon i hiZ
          simp only [Fin.valEmbedding_apply, Finset.mem_range]
          omega
        have := Finset.card_le_card hsub
        simp only [Finset.card_map, Finset.card_range] at this
        omega
      obtain ⟨i, hiZ, hki⟩ := this
      calc (k : ℝ) / ((n : ℝ) + 1) ≤ ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) := by
            gcongr; exact_mod_cast hki
        _ ≤ p := hin i hiZ
  -- upper bound `p ≤ (k+1)/(n+1)`
  have hB : p ≤ ((k : ℝ) + 1) / ((n : ℝ) + 1) := by
    have hkn : k ≤ n := (Finset.card_le_univ Z).trans (by simp)
    rcases Nat.lt_or_ge k n with hlt | hge
    · have : ∃ i : Fin n, i ∉ Z ∧ (i : ℕ) ≤ k := by
        by_contra hcon
        push Not at hcon
        have hsub : Finset.range (k + 1) ⊆ Z.map Fin.valEmbedding := by
          intro j hj
          rw [Finset.mem_range] at hj
          refine Finset.mem_map.mpr ⟨⟨j, by omega⟩, ?_, rfl⟩
          by_contra hjZ
          have := hcon ⟨j, by omega⟩ hjZ
          simp at this
          omega
        have := Finset.card_le_card hsub
        simp only [Finset.card_map, Finset.card_range] at this
        omega
      obtain ⟨i, hiZ, hik⟩ := this
      calc p ≤ ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) := hout i hiZ
        _ ≤ ((k : ℝ) + 1) / ((n : ℝ) + 1) := by
            gcongr
    · have : k = n := le_antisymm hkn hge
      rw [this, div_self hn.ne']
      linarith
  show |1 - ((k : ℝ) + 1 / 2) / ((n : ℝ) + 1) - ∫ t, f t ∂μ| ≤ 1 / (2 * ((n : ℝ) + 1))
  have e1 : ((k : ℝ) + 1) / ((n : ℝ) + 1) = (k : ℝ) / ((n : ℝ) + 1) + 2 * (1 / (2 * ((n : ℝ) + 1))) := by
    field_simp
  have e2 : ((k : ℝ) + 1 / 2) / ((n : ℝ) + 1) = (k : ℝ) / ((n : ℝ) + 1) + 1 / (2 * ((n : ℝ) + 1)) := by
    field_simp
  rw [e2, abs_le]
  constructor <;> linarith

/-! ### Nonadaptive upper bound on `P` (quantile nodes, counting) -/

lemma intervalClass_integral {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) (h1 : ∀ t, a < t → t < b → f t = 1)
    (h0 : ∀ t, t < a ∨ b < t → f t = 0) : ∫ t, f t ∂μ = cdf μ b - cdf μ a := by
  have hae : f =ᵐ[μ] (Ioo a b).indicator 1 := by
    filter_upwards [Measure.ae_ne μ a, Measure.ae_ne μ b] with t hta htb
    by_cases hin : a < t ∧ t < b
    · rw [indicator_of_mem (show t ∈ Ioo a b from hin), h1 t hin.1 hin.2]
      rfl
    · rw [indicator_of_notMem (show t ∉ Ioo a b from hin)]
      apply h0
      by_contra hc
      push Not at hc
      exact hin ⟨lt_of_le_of_ne hc.1 (Ne.symm hta), lt_of_le_of_ne hc.2 htb⟩
  have hIoc := (cdf μ).measure_Ioc a b
  rw [measure_cdf] at hIoc
  rw [integral_congr_ae hae, integral_indicator_one measurableSet_Ioo, measureReal_congr Ioo_ae_eq_Ioc,
    measureReal_def, hIoc, ENNReal.toReal_ofReal (sub_nonneg.mpr ((cdf μ).mono hab))]

/-- Nonadaptive algorithm for `P` with nodes at the quantiles of levels `1/(n+1), …, n/(n+1)`: if `k` of the `n`
values equal `1`, the estimate is `k/(n+1)`. -/
noncomputable def countingRule (ν : Measure ℝ) (n : ℕ) : NonadaptiveAlg n where
  nodes i := quantile ν (((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1))
  output y := ((Finset.univ.filter (fun i => y i = 1)).card : ℝ) / ((n : ℝ) + 1)

/-- **Upper bound on `P`** for an atomless probability measure: error at most `1/(n+1)`. -/
theorem nonadaptive_upper_bound_interval_general (n : ℕ) :
    ∃ A : NonadaptiveAlg n, ∀ f ∈ IntervalClass,
      |A.output (A.info f) - ∫ t, f t ∂μ| ≤ 1 / ((n : ℝ) + 1) := by
  classical
  refine ⟨countingRule μ n, fun f hf => ?_⟩
  obtain ⟨a, b, hab, h1, h0⟩ := hf
  rw [intervalClass_integral hab h1 h0]
  set α := cdf μ a
  set β := cdf μ b
  have hαβ : α ≤ β := (cdf μ).mono hab
  have hα0 : 0 ≤ α := cdf_nonneg μ a
  have hβ1 : β ≤ 1 := cdf_le_one μ b
  set O := Finset.univ.filter (fun i : Fin n => f ((countingRule μ n).nodes i) = 1) with hO
  set k := O.card with hk
  have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hnode : ∀ i : Fin n, cdf μ ((countingRule μ n).nodes i) = ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) :=
    fun i => cdf_quantile_node n i
  -- nodes with value `1` lie in `[a, b]`; nodes with another value lie outside `(a, b)`
  have hin : ∀ i : Fin n, i ∈ O →
      α ≤ ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) ∧ ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) ≤ β := by
    intro i hi
    have hv := (Finset.mem_filter.mp hi).2
    rw [← hnode i]
    constructor
    · apply (cdf μ).mono
      by_contra hc
      rw [h0 _ (Or.inl (not_le.mp hc))] at hv
      exact zero_ne_one hv
    · apply (cdf μ).mono
      by_contra hc
      rw [h0 _ (Or.inr (not_le.mp hc))] at hv
      exact zero_ne_one hv
  have hout : ∀ i : Fin n, i ∉ O →
      ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) ≤ α ∨ β ≤ ((i : ℕ) + 1 : ℝ) / ((n : ℝ) + 1) := by
    intro i hi
    have hv : f ((countingRule μ n).nodes i) ≠ 1 :=
      fun h => hi (Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩)
    rw [← hnode i]
    by_cases ha : (countingRule μ n).nodes i ≤ a
    · exact Or.inl ((cdf μ).mono ha)
    · by_cases hb : b ≤ (countingRule μ n).nodes i
      · exact Or.inr ((cdf μ).mono hb)
      · exact absurd (h1 _ (not_le.mp ha) (not_le.mp hb)) hv
  -- upper bound: `β - α ≤ (k+1)/(n+1)`
  have hU : (β - α) * ((n : ℝ) + 1) ≤ (k : ℝ) + 1 := by
    by_contra hcon
    push Not at hcon
    set x := α * ((n : ℝ) + 1)
    have hx0 : 0 ≤ x := by positivity
    set j₀ := ⌊x⌋₊ + 1 with hj₀
    have hj₀x : x < (j₀ : ℝ) := by push_cast [hj₀]; exact Nat.lt_floor_add_one x
    have hj₀x' : (j₀ : ℝ) ≤ x + 1 := by push_cast [hj₀]; linarith [Nat.floor_le hx0]
    have hsub : Finset.Ico (j₀ - 1) (j₀ + k) ⊆ O.map Fin.valEmbedding := by
      intro i hi
      rw [Finset.mem_Ico] at hi
      have hj1 : 1 ≤ j₀ := by omega
      have hi1 : (j₀ : ℝ) ≤ (i : ℝ) + 1 := by exact_mod_cast (show j₀ ≤ i + 1 by omega)
      have hi2 : (i : ℝ) + 1 ≤ (j₀ : ℝ) + k := by exact_mod_cast (show i + 1 ≤ j₀ + k by omega)
      have hlow : α < ((i : ℝ) + 1) / ((n : ℝ) + 1) := by
        rw [lt_div_iff₀ hn]; linarith
      have hhigh : ((i : ℝ) + 1) / ((n : ℝ) + 1) < β := by
        rw [div_lt_iff₀ hn]; nlinarith
      have hin_n : i < n := by
        have : ((i : ℝ) + 1) / ((n : ℝ) + 1) < 1 := lt_of_lt_of_le hhigh hβ1
        rw [div_lt_one hn] at this
        exact_mod_cast (show (i : ℝ) < n by linarith)
      refine Finset.mem_map.mpr ⟨⟨i, hin_n⟩, ?_, rfl⟩
      by_contra hiO
      rcases hout ⟨i, hin_n⟩ hiO with h | h
      · exact absurd h (not_le.mpr hlow)
      · exact absurd h (not_le.mpr hhigh)
    have := Finset.card_le_card hsub
    rw [Nat.card_Ico, Finset.card_map] at this
    omega
  -- lower bound: `(k-1)/(n+1) ≤ β - α`
  have hL : (k : ℝ) - 1 ≤ (β - α) * ((n : ℝ) + 1) := by
    rcases O.eq_empty_or_nonempty with hO0 | hOne
    · have : k = 0 := by rw [hk, hO0]; rfl
      rw [this]
      push_cast
      nlinarith
    · set O' := O.map Fin.valEmbedding
      have hO'ne : O'.Nonempty := hOne.map
      set lo := O'.min' hO'ne
      set hi := O'.max' hO'ne
      have hsub : O' ⊆ Finset.Icc lo hi := fun i hi' =>
        Finset.mem_Icc.mpr ⟨O'.min'_le i hi', O'.le_max' i hi'⟩
      have hcard := Finset.card_le_card hsub
      rw [Nat.card_Icc, Finset.card_map] at hcard
      have hloO : lo ∈ O' := O'.min'_mem hO'ne
      have hhiO : hi ∈ O' := O'.max'_mem hO'ne
      obtain ⟨ilo, hiloO, hilo⟩ := Finset.mem_map.mp hloO
      obtain ⟨ihi, hihiO, hihi⟩ := Finset.mem_map.mp hhiO
      have h1' := (hin ilo hiloO).1
      have h2' := (hin ihi hihiO).2
      simp only [Fin.valEmbedding_apply] at hilo hihi
      rw [hilo] at h1'
      rw [hihi] at h2'
      rw [le_div_iff₀ hn] at h1'
      rw [div_le_iff₀ hn] at h2'
      have hlohi : lo ≤ hi := O'.min'_le hi hhiO
      have : (k : ℝ) ≤ (hi : ℝ) - lo + 1 := by
        have : k ≤ hi + 1 - lo := hcard
        have : (k : ℝ) ≤ ((hi + 1 - lo : ℕ) : ℝ) := by exact_mod_cast this
        rw [Nat.cast_sub (by omega)] at this
        push_cast at this
        linarith
      nlinarith
  show |(k : ℝ) / ((n : ℝ) + 1) - (β - α)| ≤ 1 / ((n : ℝ) + 1)
  rw [abs_le]
  constructor
  · rw [← sub_nonneg]
    have : (k : ℝ) / ((n : ℝ) + 1) - (β - α) - -(1 / ((n : ℝ) + 1)) =
        ((k : ℝ) + 1 - (β - α) * ((n : ℝ) + 1)) / ((n : ℝ) + 1) := by field_simp; ring
    rw [this]
    apply div_nonneg _ hn.le
    linarith
  · rw [← sub_nonneg]
    have : 1 / ((n : ℝ) + 1) - ((k : ℝ) / ((n : ℝ) + 1) - (β - α)) =
        ((β - α) * ((n : ℝ) + 1) - ((k : ℝ) - 1)) / ((n : ℝ) + 1) := by field_simp; ring
    rw [this]
    apply div_nonneg _ hn.le
    linarith

/-! ### R5: randomized lower bound on `J` -/

/-- Test functions for the randomized lower bound: for `j ≤ M` a monotone `0/1` step whose integral is
`1 - j/M` (the constant `1` for `j = 0`, the constant `0` for `j = M`). -/
noncomputable def testFun (ν : Measure ℝ) (M j : ℕ) : ℝ → ℝ :=
  if j = 0 then 1 else if M ≤ j then 0 else (Ioi (quantile ν ((j : ℝ) / M))).indicator 1

lemma testFun_mem (ν : Measure ℝ) (M j : ℕ) : testFun ν M j ∈ StepFam := by
  unfold testFun
  split_ifs
  · exact one_mem_stepFam
  · exact zero_mem_stepFam
  · exact ioi_mem_stepFam _

lemma integral_testFun {M j : ℕ} (hj : j ≤ M) (hM : 1 ≤ M) :
    ∫ t, testFun μ M j t ∂μ = 1 - (j : ℝ) / M := by
  unfold testFun
  split_ifs with h0 h1
  · subst h0
    simp
  · have : j = M := le_antisymm hj h1
    subst this
    have hM' : (j : ℝ) ≠ 0 := by
      have : (1 : ℝ) ≤ j := by exact_mod_cast hM
      linarith
    simp [div_self hM']
  · have hpos : 0 < (j : ℝ) / M := by
      have : (0 : ℝ) < j := by exact_mod_cast Nat.pos_of_ne_zero h0
      have : (0 : ℝ) < M := by exact_mod_cast hM
      positivity
    have hlt : (j : ℝ) / M < 1 := by
      rw [div_lt_one (by exact_mod_cast hM)]
      exact_mod_cast (not_le.mp h1)
    rw [integral_step_eq (measure_singleton _) (fun t ht => ?_) (fun t ht => ?_), cdf_quantile hpos hlt]
    · rw [ioi_ind_apply, ite_eq_right (not_lt.mpr ht.le)]
    · rw [ioi_ind_apply, ite_eq_left ht]

/-- On a `0/1`-valued function, the information of an adaptive algorithm lies in `{0,1}^n`. -/
lemma info_mem_piFinset {n : ℕ} (A : AdaptiveAlg n) {f : ℝ → ℝ} (hf : ∀ t, f t = 0 ∨ f t = 1) :
    A.info f ∈ Fintype.piFinset (fun _ : Fin n => ({0, 1} : Finset ℝ)) := by
  rw [Fintype.mem_piFinset]
  intro i
  rw [AdaptiveAlg.info_apply]
  rcases hf (A.node i (fun j : Fin i => A.info f (Fin.castLE i.isLt.le j))) with h | h <;> simp [h]

/-- If one estimate `o` is used for all levels `j ∈ F`, the summed error is at least `(#F - 1)/M`
(already the smallest and the largest level of `F` force this). -/
lemma fiber_bound {M : ℕ} (hM : 0 < M) (F : Finset ℕ) (o : ℝ) :
    ((F.card : ℝ) - 1) / M ≤ ∑ j ∈ F, |o - (1 - (j : ℝ) / M)| := by
  have hM' : (0 : ℝ) < M := by exact_mod_cast hM
  rcases Nat.lt_or_ge F.card 2 with hc | hc
  · have h1 : (F.card : ℝ) ≤ 1 := by exact_mod_cast (show F.card ≤ 1 by omega)
    exact le_trans (div_nonpos_of_nonpos_of_nonneg (by linarith) hM'.le)
      (Finset.sum_nonneg fun j _ => abs_nonneg _)
  · have hne : F.Nonempty := Finset.card_pos.mp (by omega)
    set a := F.min' hne with ha_def
    set b := F.max' hne with hb_def
    have haF : a ∈ F := F.min'_mem hne
    have hbF : b ∈ F := F.max'_mem hne
    have hsub : F ⊆ Finset.Icc a b := fun j hj =>
      Finset.mem_Icc.mpr ⟨F.min'_le j hj, F.le_max' j hj⟩
    have hcard : F.card ≤ b + 1 - a := by
      have := Finset.card_le_card hsub
      rwa [Nat.card_Icc] at this
    have hab : a < b := by
      by_contra h
      push Not at h
      omega
    have h2 : ∑ j ∈ ({a, b} : Finset ℕ), |o - (1 - (j : ℝ) / M)| ≤ ∑ j ∈ F, |o - (1 - (j : ℝ) / M)| := by
      apply Finset.sum_le_sum_of_subset_of_nonneg
      · intro j hj
        rw [Finset.mem_insert, Finset.mem_singleton] at hj
        rcases hj with rfl | rfl
        · exact haF
        · exact hbF
      · exact fun j _ _ => abs_nonneg _
    rw [Finset.sum_pair hab.ne] at h2
    have h3 : ((b : ℝ) - a) / M ≤ |o - (1 - (a : ℝ) / M)| + |o - (1 - (b : ℝ) / M)| := by
      have e : ((b : ℝ) - a) / M = (o - (1 - (b : ℝ) / M)) - (o - (1 - (a : ℝ) / M)) := by
        field_simp
        ring
      rw [e]
      have h1 := le_abs_self (o - (1 - (b : ℝ) / M))
      have h2 := neg_abs_le (o - (1 - (a : ℝ) / M))
      linarith
    have h4 : (F.card : ℝ) ≤ (b : ℝ) + 1 - a := by
      have h' : ((b + 1 - a : ℕ) : ℝ) = (b : ℝ) + 1 - a := by
        rw [Nat.cast_sub (by omega)]
        push_cast
        ring
      rw [← h']
      exact_mod_cast hcard
    have h5 : ((F.card : ℝ) - 1) / M ≤ ((b : ℝ) - a) / M :=
      div_le_div_of_nonneg_right (by linarith) hM'.le
    linarith

/-- Average-case lower bound for deterministic adaptive algorithms: summed over the `2^{n+1}` test functions
(levels `j/(2^{n+1}-1)`), the error is at least `2^n / (2^{n+1} - 1)`. -/
theorem sum_error_lower_bound {n : ℕ} (A : AdaptiveAlg n) :
    (2 : ℝ) ^ n / ((2 : ℝ) ^ (n + 1) - 1) ≤ ∑ j ∈ Finset.range (2 ^ (n + 1)),
      |A.output (A.info (testFun μ (2 ^ (n + 1) - 1) j)) - ∫ t, testFun μ (2 ^ (n + 1) - 1) j t ∂μ| := by
  classical
  set N := 2 ^ (n + 1) with hN
  set M := N - 1 with hM
  have hN2 : 2 ≤ N := by
    rw [hN, pow_succ]
    have := Nat.one_le_two_pow (n := n)
    omega
  have hM1 : 1 ≤ M := by omega
  have hMr : (M : ℝ) = (2 : ℝ) ^ (n + 1) - 1 := by
    rw [hM, Nat.cast_sub (by omega), hN]
    push_cast
    ring
  set V := Fintype.piFinset (fun _ : Fin n => ({0, 1} : Finset ℝ)) with hV_def
  have hV : V.card = 2 ^ n := by
    rw [hV_def, Fintype.card_piFinset]
    simp
  set φ := fun j => A.info (testFun μ M j) with hφ
  have hmaps : ∀ j ∈ Finset.range N, φ j ∈ V := fun j _ =>
    info_mem_piFinset A (StepFam.zero_or_one (testFun_mem μ M j))
  have hint : ∀ j ∈ Finset.range N, |A.output (A.info (testFun μ M j)) - ∫ t, testFun μ M j t ∂μ| =
      |A.output (φ j) - (1 - (j : ℝ) / M)| := by
    intro j hj
    rw [integral_testFun (by rw [Finset.mem_range] at hj; omega) hM1]
  rw [Finset.sum_congr rfl hint, ← Finset.sum_fiberwise_of_maps_to hmaps]
  have hfib : ∀ y ∈ V, ((((Finset.range N).filter (fun j => φ j = y)).card : ℝ) - 1) / M ≤
      ∑ j ∈ (Finset.range N).filter (fun j => φ j = y), |A.output (φ j) - (1 - (j : ℝ) / M)| := by
    intro y _
    have h := fiber_bound (by omega : 0 < M) ((Finset.range N).filter (fun j => φ j = y)) (A.output y)
    refine h.trans (le_of_eq (Finset.sum_congr rfl fun j hj => ?_))
    rw [(Finset.mem_filter.mp hj).2]
  refine le_trans (le_of_eq ?_) (Finset.sum_le_sum hfib)
  have hcount : ∑ y ∈ V, (((Finset.range N).filter (fun j => φ j = y)).card : ℝ) = N := by
    rw [← Nat.cast_sum, ← Finset.card_eq_sum_card_fiberwise (fun j hj => hmaps j hj), Finset.card_range]
  rw [← Finset.sum_div, Finset.sum_sub_distrib, hcount, Finset.sum_const, hV, hMr]
  simp only [nsmul_eq_mul, mul_one]
  rw [hN]
  push_cast
  rw [pow_succ]
  ring

/-- **R5** for an atomless probability measure: randomized algorithms (a probability distribution over
deterministic adaptive algorithms) have expected error at least `1 / (2 (2^{n+1} - 1))` on some `f ∈ J`. -/
theorem randomized_lower_bound_general {n : ℕ} {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → AdaptiveAlg n)
    (hA : ∀ f ∈ StepClass, Measurable fun ω => (A ω).output ((A ω).info f)) :
    ∃ f ∈ StepClass, ENNReal.ofReal (1 / (2 * ((2 : ℝ) ^ (n + 1) - 1))) ≤
      ∫⁻ ω, ENNReal.ofReal |(A ω).output ((A ω).info f) - ∫ t, f t ∂μ| ∂P := by
  classical
  set N := 2 ^ (n + 1) with hN
  set M := N - 1 with hM
  set c : ℝ := 1 / (2 * ((2 : ℝ) ^ (n + 1) - 1)) with hc
  by_contra hcon
  push Not at hcon
  have hlt : ∀ j ∈ Finset.range N, ∫⁻ ω, ENNReal.ofReal
      |(A ω).output ((A ω).info (testFun μ M j)) - ∫ t, testFun μ M j t ∂μ| ∂P < ENNReal.ofReal c :=
    fun j _ => hcon _ (StepFam.mem_stepClass (testFun_mem μ M j))
  have hNpos : 0 < N := by positivity
  have hsum_lt := ENNReal.sum_lt_sum_of_nonempty (Finset.nonempty_range_iff.mpr hNpos.ne') hlt
  have hmeas : ∀ j ∈ Finset.range N, Measurable fun ω => ENNReal.ofReal
      |(A ω).output ((A ω).info (testFun μ M j)) - ∫ t, testFun μ M j t ∂μ| := fun j _ =>
    (continuous_abs.measurable.comp
      ((hA _ (StepFam.mem_stepClass (testFun_mem μ M j))).sub_const _)).ennreal_ofReal
  rw [← lintegral_finsetSum _ hmeas] at hsum_lt
  have hpow : (1 : ℝ) < (2 : ℝ) ^ (n + 1) := one_lt_pow₀ (by norm_num) (by omega)
  have hrhs : ∑ j ∈ Finset.range N, ENNReal.ofReal c =
      ENNReal.ofReal ((2 : ℝ) ^ n / ((2 : ℝ) ^ (n + 1) - 1)) := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
      ← ENNReal.ofReal_mul (by positivity)]
    congr 1
    rw [hN, hc]
    push_cast
    have : (2 : ℝ) ^ (n + 1) - 1 ≠ 0 := by linarith
    field_simp
    rw [pow_succ]
    ring
  have hlhs : ENNReal.ofReal ((2 : ℝ) ^ n / ((2 : ℝ) ^ (n + 1) - 1)) ≤ ∫⁻ ω, ∑ j ∈ Finset.range N,
      ENNReal.ofReal |(A ω).output ((A ω).info (testFun μ M j)) - ∫ t, testFun μ M j t ∂μ| ∂P := by
    calc ENNReal.ofReal ((2 : ℝ) ^ n / ((2 : ℝ) ^ (n + 1) - 1))
        = ∫⁻ ω, ENNReal.ofReal ((2 : ℝ) ^ n / ((2 : ℝ) ^ (n + 1) - 1)) ∂P := by
          rw [lintegral_const, measure_univ, mul_one]
      _ ≤ _ := lintegral_mono fun ω => by
          rw [← ENNReal.ofReal_sum_of_nonneg (fun j _ => abs_nonneg _)]
          exact ENNReal.ofReal_le_ofReal (sum_error_lower_bound (μ := μ) (A ω))
  rw [hrhs] at hsum_lt
  exact absurd (lt_of_le_of_lt hlhs hsum_lt) (lt_irrefl _)

end Cubature
