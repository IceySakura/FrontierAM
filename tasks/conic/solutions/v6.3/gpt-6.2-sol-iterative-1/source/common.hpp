#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace conic {

using Vec = std::vector<double>;
using Int = std::vector<int>;
constexpr double root2 = 1.41421356237309504880;
constexpr double invroot2 = 0.70710678118654752440;
constexpr double eps = 1e-8;
constexpr double inf = std::numeric_limits<double>::infinity();

inline double clock_time() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double norm(const double* x, int n) {
    double a = 0;
    for (int i = 0; i < n; ++i) a = std::max(a, std::abs(x[i]));
    return a;
}
inline double norm(const Vec& x) { return norm(x.data(), int(x.size())); }
inline double dot(const double* x, const double* y, int n) {
    double t = 0;
    for (int i = 0; i < n; ++i) t += x[i] * y[i];
    return t;
}
inline double dot(const Vec& x, const Vec& y) {
    return dot(x.data(), y.data(), int(x.size()));
}
inline double dot_precise(const double* x, const double* y, int n) {
    long double t = 0;
    for (int i = 0; i < n; ++i) t += (long double)x[i] * y[i];
    return double(t);
}
inline bool finite(const Vec& x) {
    return std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
}

struct CSC {
    int rows = 0, cols = 0;
    const int *p = nullptr, *i = nullptr;
    const double *v = nullptr;
    int nz() const { return p[cols]; }
    void add(const double* x, double* y, double alpha = 1) const {
        for (int j = 0; j < cols; ++j) {
            double a = alpha * x[j];
            for (int k = p[j]; k < p[j+1]; ++k) y[i[k]] += a * v[k];
        }
    }
    void add_transpose(const double* x, double* y, double alpha = 1) const {
        for (int j = 0; j < cols; ++j) {
            double a = 0;
            for (int k = p[j]; k < p[j+1]; ++k) a += v[k] * x[i[k]];
            y[j] += alpha * a;
        }
    }
};

enum Kind { Zero, Linear, Soc, PSD, Exp };
enum Status { Unresolved, Solved, PrimalInfeasible, DualInfeasible };

struct ConeInfo {
    int kind, dim, off, width;
};

struct Problem {
    CSC P, A;
    const double *q, *b;
    std::vector<ConeInfo> cones;
    int n() const { return A.cols; }
    int m() const { return A.rows; }
};

struct Result {
    Vec x, s, z;
    int status = Unresolved;
};

}  // namespace conic
