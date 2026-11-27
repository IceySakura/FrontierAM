// Minimal declarations for the installed FFTW double-precision C ABI.
extern "C" {
    typedef struct fftw_plan_s* fftw_plan;
    typedef enum {
        FFTW_R2HC = 0, FFTW_HC2R = 1, FFTW_DHT = 2, FFTW_REDFT00 = 3,
        FFTW_REDFT01 = 4, FFTW_REDFT10 = 5, FFTW_REDFT11 = 6,
        FFTW_RODFT00 = 7, FFTW_RODFT01 = 8, FFTW_RODFT10 = 9, FFTW_RODFT11 = 10
    } fftw_r2r_kind;
    void* fftw_malloc(size_t);
    void fftw_free(void*);
    fftw_plan fftw_plan_r2r(int, const int*, double*, double*, const fftw_r2r_kind*, unsigned);
    void fftw_execute_r2r(const fftw_plan, double*, double*);
    void fftw_destroy_plan(fftw_plan);
}

class Separable {
    const Matrix& A;
    const Header& h;
    std::vector<int> size, stride;
    std::vector<int8_t> sign;
    Vec weight, e, f;
    double discrepancy = 0, smallest = 0, normalization = 1;
    double *a = nullptr, *b = nullptr, *c = nullptr;
    fftw_plan forward = nullptr, backward = nullptr;
    bool debug;

    Separable(const Matrix& matrix, const Header& header) : A(matrix), h(header),
        debug(std::getenv("FAMEXP_DEBUG") != nullptr) {}

    bool prepare() {
        std::vector<int> offsets;
        for (int i = 0; i < A.n; ++i) {
            for (uint32_t j = A.row[i]; j < A.row[i + 1]; ++j) {
                int d = std::abs(static_cast<int>(A.col[j]) - i);
                if (std::find(offsets.begin(), offsets.end(), d) == offsets.end()) {
                    offsets.push_back(d);
                    if (offsets.size() > 3) return false;
                }
            }
        }
        std::sort(offsets.begin(), offsets.end());
        if (offsets.empty() || offsets[0] != 1) return false;
        stride = offsets;
        int rank = static_cast<int>(stride.size());
        size.resize(rank);
        for (int k = 0; k < rank; ++k) {
            int next = k + 1 < rank ? stride[k + 1] : A.n;
            if (next % stride[k]) return false;
            size[k] = next / stride[k];
            if (size[k] < 2) return false;
        }
        weight.assign(rank, -1);
        for (int i = 0; i < A.n; ++i) {
            int expected_count = 0;
            for (int k = 0; k < rank; ++k) {
                int coord = (i / stride[k]) % size[k];
                expected_count += (coord > 0) + (coord + 1 < size[k]);
            }
            if (A.row[i + 1] - A.row[i] != static_cast<uint32_t>(expected_count)) return false;
            for (uint32_t j = A.row[i]; j < A.row[i + 1]; ++j) {
                int delta = static_cast<int>(A.col[j]) - i;
                int k = static_cast<int>(std::lower_bound(stride.begin(), stride.end(),
                                                         std::abs(delta)) - stride.begin());
                int coord = (i / stride[k]) % size[k];
                if ((delta < 0 && coord == 0) || (delta > 0 && coord + 1 == size[k])) return false;
                double value = std::abs(A.val[j]);
                if (weight[k] < 0) weight[k] = value;
                else if (weight[k] != value) return false;
            }
        }
        sign.assign(A.n, 0);
        std::vector<int> queue;
        queue.reserve(A.n);
        queue.push_back(0);
        sign[0] = 1;
        for (size_t q = 0; q < queue.size(); ++q) {
            int i = queue[q];
            for (uint32_t j = A.row[i]; j < A.row[i + 1]; ++j) {
                int v = A.col[j];
                int wanted = A.val[j] < 0 ? sign[i] : -sign[i];
                if (sign[v] == 0) { sign[v] = wanted; queue.push_back(v); }
                else if (sign[v] != wanted) return false;
            }
        }
        if (queue.size() != static_cast<size_t>(A.n)) return false;
        int best_mask = -1;
        long double best_potential = 0;
        double best_error = std::numeric_limits<double>::infinity();
        for (int mask = 0; mask < (1 << rank); ++mask) {
            long double potential = A.diagonal[0];
            for (int k = 0; k < rank; ++k)
                potential -= static_cast<long double>(weight[k]) * ((mask >> k & 1) ? 1 : 2);
            double error = 0;
            for (int i = 0; i < A.n; ++i) {
                long double diagonal = potential;
                for (int k = 0; k < rank; ++k) {
                    int coord = (i / stride[k]) % size[k];
                    int degree = (mask >> k & 1) ?
                        (coord > 0) + (coord + 1 < size[k]) : 2;
                    diagonal += static_cast<long double>(weight[k]) * degree;
                }
                error = std::max(error, static_cast<double>(
                    std::abs(diagonal - static_cast<long double>(A.diagonal[i]))));
            }
            if (error < best_error) {
                best_error = error; best_mask = mask; best_potential = potential;
            }
        }
        if (best_mask < 0 || best_error > 1e-13 * A.upper) return false;
        discrepancy = best_error;
        e.resize(A.n);
        f.resize(A.n);
        smallest = std::numeric_limits<double>::infinity();
        for (int i = 0; i < A.n; ++i) {
            long double lambda = best_potential;
            for (int k = 0; k < rank; ++k) {
                int coord = (i / stride[k]) % size[k];
                bool neumann = best_mask >> k & 1;
                double angle = 3.14159265358979323846 * (coord + (neumann ? 0 : 1)) /
                               (2 * (size[k] + (neumann ? 0 : 1)));
                double s = std::sin(angle);
                lambda += 4 * static_cast<long double>(weight[k]) * s * s;
            }
            double value = static_cast<double>(lambda);
            smallest = std::min(smallest, value);
            e[i] = std::exp(-h.t * value);
            f[i] = phi_scalar(value, h.t);
        }
        std::vector<int> dimensions(rank);
        std::vector<fftw_r2r_kind> kinds(rank), inverse(rank);
        for (int k = 0; k < rank; ++k) {
            bool neumann = best_mask >> k & 1;
            dimensions[rank - k - 1] = size[k];
            kinds[rank - k - 1] = neumann ? FFTW_REDFT10 : FFTW_RODFT00;
            inverse[rank - k - 1] = neumann ? FFTW_REDFT01 : FFTW_RODFT00;
            normalization *= 2 * (size[k] + (neumann ? 0 : 1));
        }
        a = static_cast<double*>(fftw_malloc(8ULL * A.n));
        b = static_cast<double*>(fftw_malloc(8ULL * A.n));
        c = static_cast<double*>(fftw_malloc(8ULL * A.n));
        if (!a || !b || !c) throw std::runtime_error("FFT allocation failed");
        forward = fftw_plan_r2r(rank, dimensions.data(), a, b, kinds.data(), 1U << 6);
        backward = fftw_plan_r2r(rank, dimensions.data(), c, a, inverse.data(), 1U << 6);
        return forward && backward;
    }

public:
    static std::unique_ptr<Separable> detect(const Matrix& A, const Header& h) {
        std::unique_ptr<Separable> result(new Separable(A, h));
        if (!result->prepare()) return nullptr;
        return result;
    }
    ~Separable() {
        if (forward) fftw_destroy_plan(forward);
        if (backward) fftw_destroy_plan(backward);
        if (a) fftw_free(a);
        if (b) fftw_free(b);
        if (c) fftw_free(c);
    }
    bool solve(const Vec& b0, const Vec& b1, Vec& y) {
        for (int i = 0; i < A.n; ++i) a[i] = sign[i] * b0[i];
        fftw_execute_r2r(forward, a, b);
        for (int i = 0; i < A.n; ++i) a[i] = sign[i] * b1[i];
        fftw_execute_r2r(forward, a, c);
        for (int i = 0; i < A.n; ++i) c[i] = (e[i] * b[i] + f[i] * c[i]) / normalization;
        fftw_execute_r2r(backward, c, a);
        for (int i = 0; i < A.n; ++i) y[i] = sign[i] * a[i];
        if (discrepancy) {
            double lower = std::max(0.0, smallest - discrepancy);
            double exp_bound = h.t * std::exp(-h.t * lower);
            double phi_bound = h.t * h.t / 2;
            if (lower > 0) phi_bound = std::min(phi_bound, 1 / (lower * lower));
            double bound = discrepancy * (exp_bound * norm_two(b0) + phi_bound * norm_two(b1));
            if (bound > 0.05 * (h.atol + h.rtol * norm_inf(y))) return false;
        }
        if (debug) std::fprintf(stderr, "separable n=%d rank=%zu norm=%.6g\n",
                                A.n, size.size(), norm_inf(y));
        return true;
    }
};
