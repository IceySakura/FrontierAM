import Theory.RankCapRecovery
import Theory.RankCapLower

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

theorem linear_add_coord {d : ℕ} (b x y : Point d) : linear b (x+y)=linear b x+linear b y := by
  simp [linear,Pi.add_apply,mul_add,Finset.sum_add_distrib]

theorem abs_div_products_difference {x y p u v q : ℝ} (hp : p≠0) (hq : q≠0) :
    |u*v/q-x*y/p| ≤ |u-x| * |v/q| + |v-y| * |x/q| + |p-q| * |x/p| * |y/q| := by
  have he : u*v/q-x*y/p =
      (u-x)*(v/q)+(v-y)*(x/q)+(p-q)*(x/p)*(y/q) := by field_simp; ring
  rw [he]
  refine (abs_add_le _ _).trans ?_
  simp only [abs_mul]
  have h := abs_add_le ((u-x)*(v/q)) ((v-y)*(x/q))
  simpa only [abs_mul,add_comm] using add_le_add_right h (|p-q| * |x/p| * |y/q|)

theorem ratio_le_of_abs {x p c : ℝ} (hp : 0<p) (h : |x|≤c*p) : |x/p|≤c := by
  rw [abs_div,abs_of_pos hp,div_le_iff₀ hp]
  exact h

theorem outer_approximation_bound {θ a ah x y u v : ℝ} (hθ : 0≤θ) (hθ' : θ≤1/8)
    (ha : 1≤a) (hah : |ah-a|≤θ)
    (hx : |x|≤2) (hy : |y|≤2) (hu : |u-x|≤8*θ) (hv : |v-y|≤8*θ) :
    |u*v/ah-x*y/a|≤88*θ := by
  have ha0 : 0<a := by linarith
  have h1 : 1/2≤ah := by linarith [(abs_le.mp hah).1]
  have h0 : 0<ah := by linarith
  have hv0 : |v|≤3 := by
    have h := abs_add_le (v-y) y
    rw [sub_add_cancel] at h
    linarith
  have hq1 : |v/ah|≤6 := ratio_le_of_abs h0 (by linarith)
  have hq2 : |x/ah|≤4 := ratio_le_of_abs h0 (by linarith)
  have hq3 : |x/a|≤2 := ratio_le_of_abs ha0 (by linarith)
  have hq4 : |y/ah|≤4 := ratio_le_of_abs h0 (by linarith)
  have hh := abs_div_products_difference ha0.ne' h0.ne' (x:=x) (y:=y) (u:=u) (v:=v)
  have H1 := mul_le_mul hu hq1 (abs_nonneg _) (by positivity : 0≤8*θ)
  have H2 := mul_le_mul hv hq2 (abs_nonneg _) (by positivity : 0≤8*θ)
  have H3 := mul_le_mul
    (mul_le_mul ((abs_sub_comm a ah ▸ hah) : |a-ah|≤θ) hq3 (abs_nonneg _) hθ)
    hq4 (abs_nonneg _) (by positivity : 0≤θ*2)
  exact hh.trans (by linarith only [H1,H2,H3])

theorem abs_mul_le_diag {a b L : ℝ} (ha : a^2≤L) (hb : b^2≤L) : |a*b|≤L := by
  rw [abs_mul]
  nlinarith [sq_nonneg (|a|-|b|),sq_abs a,sq_abs b]

theorem small_residual {d : ℕ} (v D : Point d) (τ : ℝ)
    (h : ∀ i, |D i-(v i)^2|≤τ) (hn : ∀ i, D i≤2*τ) :
    ∀ i j, |v i*v j|≤3*τ :=
  fun i j => abs_mul_le_diag (by linarith [(abs_le.mp (h i)).1,hn i])
    (by linarith [(abs_le.mp (h j)).1,hn j])

theorem stable_rank_one_pivot {d : ℕ} (v D R : Point d) (k : Fin d) {τ : ℝ}
    (hτ : 0≤τ) (hp : 2*τ<D k) (hmax : ∀ i, D i≤D k)
    (hd : ∀ i, |D i-(v i)^2|≤τ)
    (hr : ∀ i, |R i-v k*v i|≤τ) :
    ∀ i j, |R i*R j/(D k)-v i*v j|≤13*τ := by
  let p := (v k)^2
  let q := D k
  have hd0 := abs_le.mp (hd k)
  have hp0 : 0<p := by dsimp [p]; linarith
  have hq0 : 0<q := by dsimp [q]; linarith
  have hcomp₁ : q/2≤p := by dsimp [q,p]; linarith 
  have hcomp₂ : p≤(3/2)*q := by dsimp [q,p]; linarith
  have hv (i : Fin d) : |v k*v i|≤2*p := by
    have hx : (v i)^2≤4*p := by dsimp [p,q] at *; linarith [(abs_le.mp (hd i)).1,hmax i]
    have H := mul_le_mul_of_nonneg_left hx (sq_nonneg (v k))
    rw [← mul_pow] at H
    have HH : |v k*v i|^2≤(2*p)^2 := by
      rw [sq_abs]
      nlinarith only [H]
    exact (sq_le_sq₀ (abs_nonneg _) (by positivity)).1 HH
  have hu (i : Fin d) : |R i|≤(7/2)*q := by
    have a := abs_add_le (R i-v k*v i) (v k*v i)
    rw [sub_add_cancel] at a
    have hv0 := hv i
    have hri := hr i
    change 2*τ<q at hp
    linarith only [a,hv0,hri,hcomp₂,hp]
  intro i j
  have htrue : (v k*v i)*(v k*v j)/p=v i*v j := by
    unfold p
    have h0 : v k≠0 := by intro h; simp [p,h] at hp0
    field_simp [h0] <;> ring
  have H := abs_div_products_difference hp0.ne' hq0.ne'
    (x:=v k*v i) (y:=v k*v j) (u:=R i) (v:=R j)
  rw [htrue] at H
  have H1 := mul_le_mul (hr i) (ratio_le_of_abs hq0 (hu j)) (abs_nonneg _) hτ
  have H2 := mul_le_mul (hr j) (ratio_le_of_abs hq0 (show |v k*v i|≤3*q from by linarith [hv i]))
    (abs_nonneg _) hτ
  have hpk : |p-q|≤τ := by
    simpa [p,q,abs_sub_comm] using hd k
  have H3 := mul_le_mul
    (mul_le_mul hpk (ratio_le_of_abs hp0 (hv i)) (abs_nonneg _) hτ)
    (ratio_le_of_abs hq0 (show |v k*v j|≤3*q from by linarith [hv j]))
    (abs_nonneg _) (by positivity : 0≤τ*2)
  exact H.trans (by linarith only [H1,H2,H3,hτ])

def curvatureProduct {d : ℕ} (G : RankTwoCap d) (w : Point d) : Point d :=
  fun i => linear G.u w*G.u i+linear G.v w*G.v i

noncomputable def rankRemainderFactor {d : ℕ} (G : RankTwoCap d) (w : Point d) : Point d :=
  fun i => (linear G.u w*G.v i-linear G.v w*G.u i)/Real.sqrt (quadratic G.matrix w)

theorem rank_remainder {d : ℕ} (G : RankTwoCap d) (w : Point d) (ha : 0<quadratic G.matrix w)
    (i j : Fin d) :
    G.matrix i j-rankOnePart (curvatureProduct G w) (quadratic G.matrix w) i j =
      rankRemainderFactor G w i*rankRemainderFactor G w j := by
  have hs : Real.sqrt (quadratic G.matrix w)≠0 := (Real.sqrt_pos.mpr ha).ne'
  have hsq := Real.sq_sqrt ha.le
  unfold rankOnePart curvatureProduct rankRemainderFactor
  rw [← mul_div_mul_comm,← pow_two,hsq]
  field_simp [ha.ne']
  rw [quadratic_gram]
  unfold RankTwoCap.matrix
  ring

theorem curvatureProduct_bound {d : ℕ} (G : RankTwoCap d) (w : Point d)
    (h : quadratic G.matrix w≤2) (i : Fin d) : |curvatureProduct G w i|≤2 := by
  have hv : G.u i ^ 2+G.v i ^ 2≤2 := by
    have H := G.upper (axis i)
    simpa [normSq_axis,linear_axis] using H
  have hq : (linear G.u w)^2+(linear G.v w)^2≤2 := by rwa [quadratic_gram] at h
  have hq0 : 0≤(linear G.u w)^2+(linear G.v w)^2 := by positivity
  have hprod := mul_le_mul hq hv (by positivity : 0≤G.u i^2+G.v i^2) (by norm_num : (0:ℝ)≤2)
  have hh := sq_nonneg (linear G.u w*G.v i-linear G.v w*G.u i)
  have hC : |curvatureProduct G w i|^2≤(2:ℝ)^2 := by
    rw [sq_abs,curvatureProduct]
    nlinarith only [hprod,hh]
  exact (sq_le_sq₀ (abs_nonneg _) (by norm_num)).1 hC

end Cubature
