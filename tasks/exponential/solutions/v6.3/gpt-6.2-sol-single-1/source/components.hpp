#pragma once

// Exact connectivity permits independent small symmetric eigensystems.
// Each spectral action is checked using the computed eigenvector residuals.
class TinyComponents {
    static constexpr int cap = 6;
    struct Component {
        int size;
        std::array<int,cap> rows;
    };
    std::vector<Component> parts;
    struct Node {
        std::array<Real,cap> column;
        Real exp, integ, err_e, err_g;
    };
    std::vector<Node> modes;
    std::vector<Real> orth;
    int n = 0;
    double t = 0, rtol = 0, atol = 0;

    void factor(const Matrix& mat, const Component& local) {
        using Square = std::array<std::array<Real,cap>,cap>;
        const int k = local.size;
        Square a{}, q{}, initial{};
        Real amplitude = 0;
        for (int i = 0; i < k; ++i) {
            const int p = local.rows[i];
            q[i][i] = 1;
            for (int j = mat.row[p]; j < mat.row[p+1]; ++j) {
                if (!mat.val[j]) continue;
                const auto it = std::find(local.rows.begin(),local.rows.begin()+k,mat.col[j]);
                a[i][it-local.rows.begin()] = Real(t)*mat.val[j];
                amplitude = std::max(amplitude, std::abs(a[i][it-local.rows.begin()]));
            }
        }
        initial = a;
        for (int sweep = 0; sweep < 16; ++sweep) {
            Real off = 0;
            for (int i = 0; i < k; ++i) {
                for (int j = i+1; j < k; ++j) {
                    const Real b = a[i][j];
                    off = std::max(off,std::abs(b));
                    if (std::abs(b) <= 5.5e-20L*amplitude) continue;
                    const Real d = (a[j][j]-a[i][i])/2;
                    const Real r = std::sqrt(d*d+b*b);
                    const Real u = (d < 0 ? -b : b)/(std::abs(d)+r);
                    const Real c = 1/std::sqrt(1+u*u), s=c*u;
                    a[i][i] -= u*b;
                    a[j][j] += u*b;
                    a[i][j] = a[j][i] = 0;
                    for (int z = 0; z < k; ++z) {
                        if (z != i && z != j) {
                            const Real x = a[z][i], y = a[z][j];
                            a[z][i] = a[i][z] = c*x-s*y;
                            a[z][j] = a[j][z] = s*x+c*y;
                        }
                        const Real x = q[z][i], y = q[z][j];
                        q[z][i] = c*x-s*y;
                        q[z][j] = s*x+c*y;
                    }
                }
            }
            if (off <= 5.5e-20L*amplitude) break;
        }
        for (int i = 0; i < k; ++i) {
            Node f{};
            const Real mu = std::max(0.L,a[i][i]);
            f.exp = mu > 80 ? 0 : std::exp(-mu);
            const Real fm = mu > 80 ? 1/mu : phi(mu);
            f.integ = t*fm;
            Real r = 0;
            for (int j = 0; j < k; ++j) {
                f.column[j] = q[j][i];
                Real v = -q[j][i]*mu, m = std::abs(v);
                for (int l = 0; l < k; ++l) {
                    v += initial[j][l]*q[l][i];
                    m += std::abs(initial[j][l]*q[l][i]);
                }
                v = std::abs(v)+8e-19L*m;
                r += v*v;
            }
            r = std::sqrt(r);
            f.err_e = fm*r + 4e-19L*f.exp;
            f.err_g = (mu < 1e-3 ? .5L : (1-fm)/mu)*r*t+4e-19L*f.integ;
            modes.push_back(f);
        }
        Real e = 0;
        for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) {
            Real z = i == j ? -1 : 0;
            for (int l = 0; l < k; ++l) z += q[i][l]*q[j][l];
            e += z*z;
        }
        orth.push_back(std::sqrt(e)+1e-18L*k);
    }
public:
    bool setup(const Matrix& a, const Header& h) {
        const double eta = h.t*a.norm;
        if (eta < 45 || a.val.size() > 7u*a.n) return false;
        const int maxsize = eta > 1300 ? cap : 4;
        n = a.n; t = h.t; rtol = h.rtol; atol = h.atol;
        std::vector<unsigned char> seen(n,0);
        for (int i = 0; i < n; ++i) {
            if (seen[i]) continue;
            Component c{1,{}};
            c.rows[0] = i;
            seen[i] = 1;
            for (int head = 0; head < c.size; ++head) {
                const int p = c.rows[head];
                for (int j = a.row[p]; j < a.row[p+1]; ++j) {
                    const int q = a.col[j];
                    if (seen[q] || a.val[j] == 0.) continue;
                    if (c.size >= maxsize) {
                        parts.clear();
                        return false;
                    }
                    seen[q] = 1;
                    c.rows[c.size++] = q;
                }
            }
            parts.push_back(c);
        }
        modes.reserve(n);
        orth.reserve(parts.size());
        for (const auto& c : parts) factor(a,c);
#ifdef FAM_PROFILE
        std::fprintf(stderr,"components %zu\n",parts.size());
#endif
        return true;
    }
    bool solve(const double* b0, const double* b1, double* out) const {
        int base = 0;
        size_t part = 0;
        Real yn = 0, err = 0;
        for (const auto& c : parts) {
            const int m = c.size;
            Real xx[cap], zz[cap], y[cap] = {};
            Real n0 = 0, n1 = 0, res = 0;
            for (int j = 0; j < m; ++j) {
                xx[j] = b0[c.rows[j]]; zz[j] = b1[c.rows[j]];
                n0 += xx[j]*xx[j];
                n1 += zz[j]*zz[j];
            }
            Real weighted = 0;
            for (int i = 0; i < m; ++i) {
                const auto& f = modes[base+i];
                Real u = 0, v = 0;
                for (int j = 0; j < m; ++j) {
                    u += f.column[j]*xx[j];
                    v += f.column[j]*zz[j];
                }
                const Real z = u*f.exp+v*f.integ;
                weighted += std::abs(u)*f.exp+std::abs(v)*f.integ;
                for (int j = 0; j < m; ++j) y[j] += f.column[j]*z;
                res += std::abs(u)*f.err_e+std::abs(v)*f.err_g;
            }
            for (int j = 0; j < m; ++j) {
                out[c.rows[j]] = double(y[j]);
                yn = std::max(yn,std::abs(y[j]));
            }
            res += orth[part++]*(std::sqrt(n0)+t*std::sqrt(n1));
            err = std::max(err, res+6e-18L*weighted);
            base += m;
        }
        return err <= .42*(atol+rtol*yn);
    }
};
