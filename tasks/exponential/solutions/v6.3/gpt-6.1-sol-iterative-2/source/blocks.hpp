#include <Eigen/Eigenvalues>

class SmallBlocks {
    using Real = long double;
    using Dense = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    const Matrix& A;
    const Header& h;
    std::vector<int> nodes, start;
    std::vector<size_t> spectral_start;
    std::vector<Real> eigenvectors, exponent, forcing;
    bool debug;

    SmallBlocks(const Matrix& matrix, const Header& header) : A(matrix), h(header),
        debug(std::getenv("FAMEXP_DEBUG") != nullptr) {}

    bool prepare() {
        std::vector<int> parent(A.n), count(A.n, 1);
        for (int i = 0; i < A.n; ++i) parent[i] = i;
        auto root = [&](int v) {
            while (parent[v] != v) {
                parent[v] = parent[parent[v]];
                v = parent[v];
            }
            return v;
        };
        for (int i = 0; i < A.n; ++i) {
            for (uint32_t k = A.row[i]; k < A.row[i + 1]; ++k) {
                int j = A.col[k];
                if (j < i) continue;
                int a = root(i), b = root(j);
                if (a != b) {
                    if (count[a] < count[b]) std::swap(a, b);
                    parent[b] = a;
                    count[a] += count[b];
                    if (count[a] > 32) return false;
                }
            }
        }
        std::vector<int> block(A.n, -1), block_id(A.n), position(A.n), sizes;
        for (int i = 0; i < A.n; ++i) {
            int r = root(i);
            if (block[r] < 0) { block[r] = static_cast<int>(sizes.size()); sizes.push_back(0); }
            block_id[i] = block[r];
            ++sizes[block[r]];
        }
        start.resize(sizes.size() + 1);
        spectral_start.resize(sizes.size() + 1);
        for (size_t k = 0; k < sizes.size(); ++k) {
            start[k + 1] = start[k] + sizes[k];
            spectral_start[k + 1] = spectral_start[k] + static_cast<size_t>(sizes[k]) * sizes[k];
        }
        nodes.resize(A.n);
        std::fill(sizes.begin(), sizes.end(), 0);
        for (int i = 0; i < A.n; ++i) {
            int k = block_id[i], p = sizes[k]++;
            nodes[start[k] + p] = i;
            position[i] = p;
        }
        eigenvectors.resize(spectral_start.back());
        exponent.resize(A.n);
        forcing.resize(A.n);
        for (size_t k = 0; k < sizes.size(); ++k) {
            int m = sizes[k], begin = start[k];
            Dense matrix = Dense::Zero(m, m);
            for (int r = 0; r < m; ++r) {
                int i = nodes[begin + r];
                matrix(r, r) = A.diagonal[i];
                for (uint32_t j = A.row[i]; j < A.row[i + 1]; ++j)
                    matrix(r, position[A.col[j]]) = A.val[j];
            }
            Eigen::SelfAdjointEigenSolver<Dense> eig(matrix);
            if (eig.info() != Eigen::Success) throw std::runtime_error("block eigensolver failed");
            const Dense& vectors = eig.eigenvectors();
            for (int j = 0; j < m; ++j) {
                Real lambda = std::max(Real(0), eig.eigenvalues()[j]);
                Real x = static_cast<Real>(h.t) * lambda;
                exponent[begin + j] = std::exp(-x);
                forcing[begin + j] = x == 0 ? static_cast<Real>(h.t) : -std::expm1(-x) / lambda;
                for (int r = 0; r < m; ++r)
                    eigenvectors[spectral_start[k] + r + j * m] = vectors(r, j);
            }
        }
        return true;
    }

public:
    static std::unique_ptr<SmallBlocks> detect(const Matrix& A, const Header& h) {
        std::unique_ptr<SmallBlocks> result(new SmallBlocks(A, h));
        if (!result->prepare()) return nullptr;
        return result;
    }
    void solve(const Vec& b0, const Vec& b1, Vec& y) const {
        Real coefficients[32], output[32];
        for (size_t k = 0; k + 1 < start.size(); ++k) {
            int begin = start[k], m = start[k + 1] - begin;
            const Real* V = eigenvectors.data() + spectral_start[k];
            for (int j = 0; j < m; ++j) {
                Real c0 = 0, c1 = 0;
                for (int r = 0; r < m; ++r) {
                    int i = nodes[begin + r];
                    c0 += V[r + j * m] * static_cast<Real>(b0[i]);
                    c1 += V[r + j * m] * static_cast<Real>(b1[i]);
                }
                coefficients[j] = exponent[begin + j] * c0 + forcing[begin + j] * c1;
                output[j] = 0;
            }
            for (int j = 0; j < m; ++j)
                for (int r = 0; r < m; ++r) output[r] += V[r + j * m] * coefficients[j];
            for (int r = 0; r < m; ++r) y[nodes[begin + r]] = static_cast<double>(output[r]);
        }
        if (debug) std::fprintf(stderr, "small_blocks n=%d blocks=%zu norm=%.6g\n",
                                A.n, start.size() - 1, norm_inf(y));
    }
};
