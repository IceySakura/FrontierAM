#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

struct SmallBlocks {
    static constexpr int max_size = 8;
    struct Block {
        int first, size, vectors;
        long double orthogonal_error;
    };
    struct Mode { long double e, f, defect; };
    bool ready = false, rejected = false;
    std::vector<Block> blocks;
    std::vector<int> index;
    std::vector<Mode> modes;
    std::vector<long double> eigenvectors;

    static constexpr long double unit = .5L*std::numeric_limits<long double>::epsilon();

    static long double integral(long double x) {
        return x == 0 ? 1 : -std::expm1(-x)/x;
    }

    static void diagonalize(int k, long double (&m)[max_size][max_size],
                            long double (&u)[max_size][max_size]) {
        for (int i = 0; i < k; ++i) u[i][i] = 1;
        for (int sweep = 0; sweep < 22; ++sweep) {
            bool changed = false;
            for (int i = 0; i < k; ++i) {
                for (int j = i+1; j < k; ++j) {
                    const long double off = m[i][j];
                    if (std::abs(off) <= 2*unit*(std::abs(m[i][i])+std::abs(m[j][j]))) continue;
                    const long double r = (m[j][j]-m[i][i])/(2*off);
                    const long double t = std::copysign(1.L,r)/(std::abs(r)+std::sqrt(1+r*r));
                    const long double c = 1/std::sqrt(1+t*t), s = t*c;
                    m[i][i] -= t*off;
                    m[j][j] += t*off;
                    m[i][j] = m[j][i] = 0;
                    for (int p = 0; p < k; ++p) {
                        if (p != i && p != j) {
                            const long double a = m[p][i], b = m[p][j];
                            m[p][i] = m[i][p] = c*a-s*b;
                            m[p][j] = m[j][p] = s*a+c*b;
                        }
                        const long double x = u[p][i], y = u[p][j];
                        u[p][i] = c*x-s*y;
                        u[p][j] = s*x+c*y;
                    }
                    changed = true;
                }
            }
            if (!changed) break;
        }
    }

    bool prepare(int n, int root, const int* row, const uint32_t* col, const double* a,
                 double time) {
        std::vector<int> position(n,-1);
        blocks.clear();
        index.clear();
        index.reserve(n);
        const auto discover = [&](int first) {
            if (position[first] >= 0) return true;
            const int begin = static_cast<int>(index.size());
            int end = begin+1;
            position[first] = begin;
            index.push_back(first);
            for (int p = begin; p < end; ++p) {
                const int i = index[p];
                for (int j = row[i]; j < row[i+1]; ++j) {
                    const int v = static_cast<int>(col[j]);
                    if (position[v] >= 0 || a[j] == 0) continue;
                    if (end-begin >= max_size) return false;
                    position[v] = end++;
                    index.push_back(v);
                }
            }
            blocks.push_back({begin,end-begin,0,0});
            return true;
        };
        if (!discover(root)) return false;
        for (int i = 0; i < n; ++i)
            if (!discover(i)) return false;
        int total = 0;
        for (auto& b : blocks) {
            b.vectors = total;
            total += b.size*b.size;
        }
        eigenvectors.resize(total);
        modes.resize(n);
        for (auto& b : blocks) {
            const int k = b.size;
            long double original[max_size][max_size] = {};
            long double work[max_size][max_size] = {};
            long double vectors[max_size][max_size] = {};
            for (int p = 0; p < k; ++p) {
                const int i = index[b.first+p];
                for (int j = row[i]; j < row[i+1]; ++j) {
                    if (!a[j]) continue;
                    const int m = position[col[j]] - b.first;
                    original[p][m] = work[p][m] = static_cast<long double>(time)*a[j];
                }
            }
            diagonalize(k,work,vectors);
            long double orthogonal2 = 0;
            for (int j = 0; j < k; ++j) {
                const long double lambda = work[j][j];
                long double defect2 = 0;
                for (int p = 0; p < k; ++p) {
                    long double s = 0, magnitude = 0;
                    for (int i = 0; i < k; ++i) {
                        const long double t = original[p][i]*vectors[i][j];
                        s += t;
                        magnitude += std::abs(t);
                    }
                    const long double v = lambda*vectors[p][j];
                    const long double r = std::abs(s-v)+(k+5)*unit*(magnitude+std::abs(v));
                    defect2 += r*r;
                    eigenvectors[b.vectors+j*k+p] = vectors[p][j];
                }
                const long double f = integral(lambda);
                modes[b.first+j] = {std::exp(-lambda),time*f,std::sqrt(defect2)*f};
                for (int l = 0; l < k; ++l) {
                    long double s = 0;
                    for (int p = 0; p < k; ++p) s += vectors[p][l]*vectors[p][j];
                    s -= (l == j);
                    orthogonal2 += s*s;
                }
            }
            b.orthogonal_error = std::sqrt(orthogonal2)+4*k*k*unit;
        }
        ready = true;
        return true;
    }

    double evaluate(const double* __restrict x, const double* __restrict q,
                    double* __restrict y, double t) const {
        long double worst = 0;
        for (const auto& b : blocks) {
            const int k = b.size;
            const int* ids = index.data()+b.first;
            const long double* basis = eigenvectors.data()+b.vectors;
            long double out[max_size] = {}, sz = 0, magnitude = 0, error = 0;
            for (int i = 0; i < k; ++i)
                sz = std::max(sz,std::abs(static_cast<long double>(x[ids[i]]))+
                                 static_cast<long double>(t)*std::abs(q[ids[i]]));
            for (int j = 0; j < k; ++j) {
                long double a = 0, b1 = 0, sum_x = 0, sum_q = 0;
                const long double* v = basis+j*k;
                for (int i = 0; i < k; ++i) {
                    const long double p = v[i]*x[ids[i]], r = v[i]*q[ids[i]];
                    a += p; b1 += r;
                    sum_x += std::abs(p); sum_q += std::abs(r);
                }
                const auto& c = modes[b.first+j];
                error += c.defect*(std::abs(a)+t*std::abs(b1));
                magnitude += c.e*sum_x+c.f*sum_q;
                const long double d = c.e*a+c.f*b1;
                for (int i = 0; i < k; ++i) out[i] += d*v[i];
            }
            error += b.orthogonal_error*std::sqrt(2.L*k)*sz+
                     (4*k+12)*unit*magnitude;
            for (int i = 0; i < k; ++i) y[ids[i]] = static_cast<double>(out[i]);
            worst = std::max(worst,error);
        }
        return static_cast<double>(worst);
    }
};
