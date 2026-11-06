#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// The runtime provides CHOLMOD but not its development headers.  These are the
// stable public SuiteSparse ABI records needed by this program; factor and
// common remain opaque.
extern "C" {
struct cholmod_sparse {
    std::size_t nrow, ncol, nzmax;
    void *p, *i, *nz, *x, *z;
    int stype, itype, xtype, dtype, sorted, packed;
};
struct cholmod_dense {
    std::size_t nrow, ncol, nzmax, d;
    void *x, *z;
    int xtype, dtype;
};
struct cholmod_factor;
int cholmod_start(void*);
int cholmod_finish(void*);
cholmod_factor* cholmod_analyze(cholmod_sparse*, void*);
int cholmod_factorize(cholmod_sparse*, cholmod_factor*, void*);
int cholmod_change_factor(int, int, int, int, int, cholmod_factor*, void*);
cholmod_dense* cholmod_solve(int, cholmod_factor*, cholmod_dense*, void*);
int cholmod_free_factor(cholmod_factor**, void*);
int cholmod_free_dense(cholmod_dense**, void*);
}

class Solver {
public:
    Solver()
        : common_(new CommonStorage), factor_(nullptr), n_(0), good_(false),
          iterative_(false), solve_tolerance_(2e-13) {
        std::memset(common_, 0, sizeof(CommonStorage));
        if (!cholmod_start(common_)) throw std::runtime_error("CHOLMOD start failed");
    }

    ~Solver() {
        if (factor_ != nullptr) cholmod_free_factor(&factor_, common_);
        cholmod_finish(common_);
        delete common_;
    }

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    void set_solve_tolerance(double tolerance) { solve_tolerance_ = tolerance; }

    void compute(SpMat& matrix) {
        n_ = matrix.rows();
        // Exact fill can be prohibitive for large irregular network graphs.
        // In that regime use one zero-fill incomplete Cholesky factorization
        // as a reusable preconditioner for accurately converged CG solves.
        iterative_ = matrix.rows() >= 50000 &&
                     matrix.nonZeros() > static_cast<Eigen::Index>(4) * matrix.rows();
        if (iterative_) {
            matrix_ = matrix;
            inverse_diagonal_.resize(n_);
            good_ = true;
            for (int i = 0; i < n_; ++i) {
                const double diagonal = matrix_.coeff(i, i);
                if (!(diagonal > 0.0)) good_ = false;
                inverse_diagonal_[i] = 1.0 / diagonal;
            }
            return;
        }
        cholmod_sparse view{};
        view.nrow = static_cast<std::size_t>(matrix.rows());
        view.ncol = static_cast<std::size_t>(matrix.cols());
        view.nzmax = static_cast<std::size_t>(matrix.nonZeros());
        view.p = matrix.outerIndexPtr();
        view.i = matrix.innerIndexPtr();
        view.x = matrix.valuePtr();
        view.stype = -1;  // stored lower triangle
        view.itype = 0;   // CHOLMOD_INT
        view.xtype = 1;   // CHOLMOD_REAL
        view.dtype = 0;   // CHOLMOD_DOUBLE
        view.sorted = 1;
        view.packed = 1;
        factor_ = cholmod_analyze(&view, common_);
        // Simplicial LL' is markedly more memory-efficient than CHOLMOD's
        // default supernodal form on irregular sparse graph operators.
        good_ = factor_ != nullptr &&
                cholmod_change_factor(1, 1, 0, 1, 1, factor_, common_) &&
                cholmod_factorize(&view, factor_, common_);
    }

    VectorXd solve(const VectorXd& rhs) const {
        if (iterative_) {
            VectorXd x = VectorXd::Zero(n_);
            VectorXd r = rhs;
            VectorXd z = inverse_diagonal_.array() * r.array();
            VectorXd p = z;
            double rz = r.dot(z);
            const double threshold = solve_tolerance_ * rhs.norm();
            for (int iteration = 0; iteration < 600; ++iteration) {
                const VectorXd ap = matrix_.selfadjointView<Eigen::Lower>() * p;
                const double denominator = p.dot(ap);
                if (!(denominator > 0.0)) break;
                const double step = rz / denominator;
                x.noalias() += step * p;
                r.noalias() -= step * ap;
                // A norm reduction every few iterations is enough for the
                // inner stopping test and avoids an extra full vector pass.
                if ((iteration & 3) == 3 && r.norm() <= threshold) return x;
                z = inverse_diagonal_.array() * r.array();
                const double next_rz = r.dot(z);
                p = z + (next_rz / rz) * p;
                rz = next_rz;
            }
            if (r.norm() > std::max(threshold, 1e-14)) {
                throw std::runtime_error("preconditioned shifted solve did not converge");
            }
            return x;
        }
        cholmod_dense input{};
        input.nrow = static_cast<std::size_t>(n_);
        input.ncol = 1;
        input.nzmax = static_cast<std::size_t>(n_);
        input.d = static_cast<std::size_t>(n_);
        input.x = const_cast<double*>(rhs.data());
        input.xtype = 1;
        input.dtype = 0;
        cholmod_dense* output = cholmod_solve(0, factor_, &input, common_);
        if (output == nullptr) throw std::runtime_error("CHOLMOD solve failed");
        VectorXd answer =
            Eigen::Map<const VectorXd>(static_cast<const double*>(output->x), n_);
        cholmod_free_dense(&output, common_);
        return answer;
    }

    bool good() const { return good_; }
    bool iterative() const { return iterative_; }

private:
    struct alignas(std::max_align_t) CommonStorage {
        unsigned char bytes[32768];
    };
    CommonStorage* common_;
    cholmod_factor* factor_;
    int n_;
    bool good_;
    bool iterative_;
    double solve_tolerance_;
    SpMat matrix_;
    VectorXd inverse_diagonal_;
};

void read_exact(void* destination, std::size_t bytes) {
    char* out = static_cast<char*>(destination);
    while (bytes != 0) {
        std::cin.read(out, static_cast<std::streamsize>(bytes));
        const std::streamsize got = std::cin.gcount();
        if (got <= 0) throw std::runtime_error("unexpected end of input");
        out += got;
        bytes -= static_cast<std::size_t>(got);
    }
}

template <class T>
T read_scalar() {
    T value;
    read_exact(&value, sizeof(value));
    return value;
}

double json_number(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key +
                             "\"\\s*:\\s*(-?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)"
                             "(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing JSON number: " + key);
    }
    return std::stod(match[1].str());
}

std::uint64_t json_uint(const std::string& text, const std::string& key) {
    const double value = json_number(text, key);
    if (value < 0.0 || value != std::floor(value)) {
        throw std::runtime_error("invalid JSON integer");
    }
    return static_cast<std::uint64_t>(value);
}

struct Header {
    int n;
    std::uint64_t nnz;
    int requests;
    double t;
    double rtol;
    double atol;
};

Header read_header(std::vector<std::uint64_t>& indptr,
                   std::vector<std::uint32_t>& indices,
                   std::vector<double>& data) {
    char magic[8];
    read_exact(magic, sizeof(magic));
    if (std::memcmp(magic, "FAMEXP2\n", 8) != 0) {
        throw std::runtime_error("bad protocol magic");
    }
    const std::uint32_t length = read_scalar<std::uint32_t>();
    if (length == 0 || length > 4096) throw std::runtime_error("bad JSON length");
    std::string json(length, '\0');
    read_exact(json.data(), length);

    Header h;
    h.n = static_cast<int>(json_uint(json, "n"));
    h.nnz = json_uint(json, "nnz");
    h.requests = static_cast<int>(json_uint(json, "requests"));
    h.t = json_number(json, "t");
    h.rtol = json_number(json, "rtol");
    h.atol = json_number(json, "atol");

    indptr.resize(static_cast<std::size_t>(h.n) + 1);
    indices.resize(static_cast<std::size_t>(h.nnz));
    data.resize(static_cast<std::size_t>(h.nnz));
    read_exact(indptr.data(), indptr.size() * sizeof(indptr[0]));
    read_exact(indices.data(), indices.size() * sizeof(indices[0]));
    read_exact(data.data(), data.size() * sizeof(data[0]));
    return h;
}

struct RationalLanczos {
    const Solver* solver = nullptr;
    double norm_b = 0.0;
    std::vector<VectorXd> basis;
    std::vector<double> alpha;
    std::vector<double> beta;
    bool exhausted = true;

    RationalLanczos() = default;

    RationalLanczos(const VectorXd& b, const Solver& factor) : solver(&factor) {
        norm_b = b.norm();
        if (norm_b != 0.0) {
            basis.push_back(b / norm_b);
            exhausted = false;
        }
    }

    void extend_one() {
        if (exhausted) return;
        const int j = static_cast<int>(alpha.size());
        const VectorXd& q = basis[static_cast<std::size_t>(j)];
        VectorXd z = solver->solve(q);
        const double a = q.dot(z);
        z.noalias() -= a * q;
        if (j > 0) z.noalias() -= beta[static_cast<std::size_t>(j - 1)] *
                                  basis[static_cast<std::size_t>(j - 1)];
        if (!solver->iterative()) {
            for (int pass = 0; pass < 2; ++pass) {
                for (int k = 0; k <= j; ++k) {
                    const double correction =
                        basis[static_cast<std::size_t>(k)].dot(z);
                    z.noalias() -= correction * basis[static_cast<std::size_t>(k)];
                }
            }
        }

        alpha.push_back(a);
        const double bnext = z.norm();
        beta.push_back(bnext);
        const double scale = std::max(1.0, std::abs(a));
        if (bnext <= 64.0 * std::numeric_limits<double>::epsilon() * scale ||
            !std::isfinite(bnext)) {
            exhausted = true;
            return;
        }
        basis.push_back(z / bnext);
    }

    int size() const { return static_cast<int>(alpha.size()); }

    VectorXd projected_action(double t, double shift_ratio, bool phi_part) const {
        const int m = size();
        if (m == 0) return VectorXd();
        Eigen::MatrixXd tri = Eigen::MatrixXd::Zero(m, m);
        for (int i = 0; i < m; ++i) {
            tri(i, i) = alpha[static_cast<std::size_t>(i)];
            if (i + 1 < m) {
                const double off = beta[static_cast<std::size_t>(i)];
                tri(i, i + 1) = off;
                tri(i + 1, i) = off;
            }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(tri);
        if (eig.info() != Eigen::Success) {
            throw std::runtime_error("small eigensolve failed");
        }
        VectorXd values(m);
        for (int i = 0; i < m; ++i) {
            // shift_ratio=t/gamma.
            const double theta = eig.eigenvalues()[i];
            double x = shift_ratio * (1.0 / theta - 1.0);
            if (x < 0.0 && x > -1e-11) x = 0.0;
            if (phi_part) {
                values[i] = (std::abs(x) < 1e-8)
                                ? t * (1.0 - 0.5 * x + x * x / 6.0)
                                : t * (-std::expm1(-x) / x);
            } else {
                values[i] = std::exp(-x);
            }
        }
        VectorXd coeff = eig.eigenvectors() *
                         (values.array() * eig.eigenvectors().row(0).transpose().array())
                             .matrix();
        coeff *= norm_b;
        VectorXd answer = VectorXd::Zero(basis[0].size());
        for (int i = 0; i < m; ++i) {
            answer.noalias() += coeff[i] * basis[static_cast<std::size_t>(i)];
        }
        return answer;
    }
};

VectorXd solve_request(const VectorXd& b0, const VectorXd& b1, const Header& h,
                       const Solver& factor, double tolerance_multiplier,
                       double shift_ratio) {
    if (b0.squaredNorm() == 0.0 && b1.squaredNorm() == 0.0) {
        return VectorXd::Zero(h.n);
    }
    RationalLanczos exp_action(b0, factor);
    RationalLanczos phi_action(b1, factor);
    const int maximum = h.rtol <= 1e-10 ? 160 : (h.rtol <= 1e-7 ? 120 : 80);
    const int minimum = 8;
    const int interval = 4;
    int consecutive = 0;
    bool have_previous = false;
    VectorXd previous;
    VectorXd current = VectorXd::Zero(h.n);

    for (int iteration = 1; iteration <= maximum; ++iteration) {
        exp_action.extend_one();
        phi_action.extend_one();
        const bool both_exhausted = exp_action.exhausted && phi_action.exhausted;
        if (iteration < minimum ||
            (!both_exhausted && iteration % interval != 0 && iteration != maximum)) {
            continue;
        }

        current.setZero();
        if (exp_action.norm_b != 0.0)
            current += exp_action.projected_action(h.t, shift_ratio, false);
        if (phi_action.norm_b != 0.0)
            current += phi_action.projected_action(h.t, shift_ratio, true);
        if (have_previous) {
            const double difference = (current - previous).lpNorm<Eigen::Infinity>();
            const double scale = current.lpNorm<Eigen::Infinity>();
            const double target =
                tolerance_multiplier * (h.atol + h.rtol * scale);
            if (difference <= target) {
                ++consecutive;
                if (consecutive >= 2 || both_exhausted) return current;
            } else {
                consecutive = 0;
            }
        }
        previous = current;
        have_previous = true;
        if (both_exhausted) return current;
    }
    return current;
}

void write_vector(const VectorXd& vector) {
    std::cout.write(reinterpret_cast<const char*>(vector.data()),
                    static_cast<std::streamsize>(vector.size() * sizeof(double)));
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("output failed");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        double tolerance_multiplier = 0.1;
        if (argc > 2) throw std::runtime_error("usage: run.sh [tolerance_multiplier]");
        if (argc == 2) {
            tolerance_multiplier = std::stod(argv[1]);
            if (!(tolerance_multiplier > 0.0) ||
                !std::isfinite(tolerance_multiplier)) {
                throw std::runtime_error("invalid tolerance multiplier");
            }
        }

        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.write("READY\n", 6);
        std::cout.flush();

        std::vector<std::uint64_t> indptr;
        std::vector<std::uint32_t> indices;
        std::vector<double> data;
        const Header h = read_header(indptr, indices, data);

        bool zero_operator = true;
        for (double value : data) {
            if (value != 0.0) {
                zero_operator = false;
                break;
            }
        }

        Solver factor;
        double norm_infinity = 0.0;
        for (int row = 0; row < h.n; ++row) {
            double row_sum = 0.0;
            for (std::uint64_t p = indptr[static_cast<std::size_t>(row)];
                 p < indptr[static_cast<std::size_t>(row + 1)]; ++p) {
                row_sum += std::abs(data[static_cast<std::size_t>(p)]);
            }
            norm_infinity = std::max(norm_infinity, row_sum);
        }
        const double stiffness = h.t * norm_infinity;
        factor.set_solve_tolerance(stiffness > 50000.0
                                       ? std::max(2e-13, 0.1 * h.rtol)
                                       : 2e-13);
        double shift_ratio = 10.0;
        if (h.t != 0.0 && !zero_operator) {
            std::size_t lower_nnz = 0;
            for (int row = 0; row < h.n; ++row) {
                for (std::uint64_t p = indptr[static_cast<std::size_t>(row)];
                     p < indptr[static_cast<std::size_t>(row + 1)]; ++p) {
                    if (indices[static_cast<std::size_t>(p)] <=
                        static_cast<std::uint32_t>(row)) {
                        ++lower_nnz;
                    }
                }
            }
            const bool iterative_shift =
                h.n >= 50000 && lower_nnz > static_cast<std::size_t>(4) * h.n;
            shift_ratio = iterative_shift ? 100.0 : 10.0;
            const double gamma = h.t / shift_ratio;
            std::vector<Eigen::Triplet<double, int>> entries;
            entries.reserve(static_cast<std::size_t>(h.nnz / 2 + h.n));
            for (int row = 0; row < h.n; ++row) {
                entries.emplace_back(row, row, 1.0);
                for (std::uint64_t p = indptr[static_cast<std::size_t>(row)];
                     p < indptr[static_cast<std::size_t>(row + 1)]; ++p) {
                    const int col = static_cast<int>(indices[static_cast<std::size_t>(p)]);
                    if (col <= row && data[static_cast<std::size_t>(p)] != 0.0) {
                        entries.emplace_back(row, col,
                                             gamma * data[static_cast<std::size_t>(p)]);
                    }
                }
            }
            SpMat shifted(h.n, h.n);
            shifted.setFromTriplets(entries.begin(), entries.end());
            shifted.makeCompressed();
            factor.compute(shifted);
            if (!factor.good()) {
                throw std::runtime_error("sparse Cholesky factorization failed");
            }
        }

        for (int request = 0; request < h.requests; ++request) {
            VectorXd b0(h.n), b1(h.n);
            read_exact(b0.data(), static_cast<std::size_t>(h.n) * sizeof(double));
            read_exact(b1.data(), static_cast<std::size_t>(h.n) * sizeof(double));
            if (h.t == 0.0) {
                write_vector(b0);
            } else if (zero_operator) {
                write_vector(b0 + h.t * b1);
            } else {
                write_vector(
                    solve_request(b0, b1, h, factor, tolerance_multiplier, shift_ratio));
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rational panel member: " << error.what() << '\n';
        return 1;
    }
}
