#pragma once

// Small subset of the installed FFTW3 double-precision C ABI.
extern "C" {
struct fftw_plan_s;
using fftw_plan = fftw_plan_s*;
fftw_plan fftw_plan_r2r_2d(int, int, double*, double*, int, int, unsigned);
void fftw_execute_r2r(const fftw_plan, double*, double*);
void fftw_destroy_plan(fftw_plan);

// System libmvec's AVX2 vector exp.
__m256d _ZGVdN4v_exp(__m256d);
}
