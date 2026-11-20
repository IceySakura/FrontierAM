#include <cstddef>
#include <cstring>

namespace {

template <int D>
void forward_filter(int n, int q, const double* __restrict__ coefficients,
                    const double* __restrict__ input, double* __restrict__ output) {
    for (int t = 0; t < n; ++t) {
        double* now = output + static_cast<std::size_t>(t) * D;
        double result[D];
        for (int j = 0; j < D; ++j) result[j] = input[t*D+j];
        for (int l = 1; l <= q && l <= t; ++l) {
            const double* row = coefficients + (l-1) * D * D;
            const double* prev = now - l*D;
            for (int i = 0; i < D; ++i) {
                const double v = prev[i];
                for (int j = 0; j < D; ++j) result[j] -= row[i*D+j] * v;
            }
        }
        for (int j = 0; j < D; ++j) now[j] = result[j];
    }
}

template <int D>
void backward_filter(int n, int q, const double* __restrict__ coefficients,
                     double* __restrict__ score) {
    for (int t = n-1; t >= 0; --t) {
        double* now = score + static_cast<std::size_t>(t) * D;
        double result[D];
        for (int j = 0; j < D; ++j) result[j] = now[j];
        for (int l = 1; l <= q && l+t < n; ++l) {
            const double* row = coefficients + (l-1) * D * D;
            const double* next = now + l*D;
            for (int i = 0; i < D; ++i) {
                double correction = 0.;
                for (int j = 0; j < D; ++j) correction += row[i*D+j] * next[j];
                result[i] -= correction;
            }
        }
        for (int j = 0; j < D; ++j) now[j] = result[j];
    }
}

}

extern "C" {

// Apply the causal inverse of a matrix moving-average filter.
void innovations(int n, int d, int q, const double* coefficients,
                 const double* input, double* output) {
    switch (d) {
        case 1: return forward_filter<1>(n,q,coefficients,input,output);
        case 2: return forward_filter<2>(n,q,coefficients,input,output);
        case 3: return forward_filter<3>(n,q,coefficients,input,output);
        case 4: return forward_filter<4>(n,q,coefficients,input,output);
        case 5: return forward_filter<5>(n,q,coefficients,input,output);
        case 6: return forward_filter<6>(n,q,coefficients,input,output);
    }
    for (int t = 0; t < n; ++t) {
        double* now = output + static_cast<std::size_t>(t) * d;
        std::memcpy(now, input + static_cast<std::size_t>(t) * d, d * sizeof(double));
        for (int l = 1; l <= q && l <= t; ++l) {
            const double* row = coefficients + (l-1) * d * d;
            const double* prev = now - l*d;
            for (int i = 0; i < d; ++i) {
                const double z = prev[i];
                for (int j = 0; j < d; ++j)
                    now[j] -= row[i*d+j] * z;
            }
        }
    }
}

// Transpose of the causal inverse. The score array is overwritten.
void innovation_adjoint(int n, int d, int q, const double* coefficients, double* score) {
    switch (d) {
        case 1: return backward_filter<1>(n,q,coefficients,score);
        case 2: return backward_filter<2>(n,q,coefficients,score);
        case 3: return backward_filter<3>(n,q,coefficients,score);
        case 4: return backward_filter<4>(n,q,coefficients,score);
        case 5: return backward_filter<5>(n,q,coefficients,score);
        case 6: return backward_filter<6>(n,q,coefficients,score);
    }
    for (int t = n-1; t >= 0; --t) {
        double* now = score + static_cast<std::size_t>(t) * d;
        for (int l = 1; l <= q && l+t < n; ++l) {
            const double* row = coefficients + (l-1) * d * d;
            const double* next = now + l*d;
            for (int i = 0; i < d; ++i) {
                double correction = 0.0;
                for (int j = 0; j < d; ++j)
                    correction += row[i*d+j] * next[j];
                now[i] -= correction;
            }
        }
    }
}

}
