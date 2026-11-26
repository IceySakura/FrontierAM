#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

static constexpr double EPS2 = 0.0001;
static constexpr int PMAX = 2;
static constexpr int DMAX = 2 * PMAX;
static constexpr int LEAF = 16;
static double THETA = 0.5;

struct Idx {
    int offset[DMAX + 2];
    int ncoef[DMAX + 2];
    int id[DMAX + 2][DMAX + 2][DMAX + 2];
    int i[DMAX + 2][DMAX + 2][DMAX + 2];
} idx;

static int MC, DC;
static std::vector<std::array<int,3>> midx;
static std::vector<double> fact;
static std::vector<double> invfact;

static void init_index() {
    fact.assign(DMAX + 2, 1.0);
    for (int i = 1; i <= DMAX + 1; ++i) fact[i] = fact[i-1] * i;
    int cur = 0;
    for (int n = 0; n <= DMAX + 1; ++n) {
        idx.offset[n] = cur;
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n - i; ++j) {
                int k = n - i - j;
                idx.id[i][j][k] = cur;
                idx.i[i][j][k] = cur;
                midx.push_back({i,j,k});
                ++cur;
            }
        idx.ncoef[n] = cur - idx.offset[n];
    }
    MC = idx.offset[PMAX + 1];
    DC = idx.offset[DMAX + 1];
    invfact.resize(MC);
    for (int a = 0; a <= PMAX; ++a)
        for (int b = 0; a + b <= PMAX; ++b)
            for (int c = 0; a + b + c <= PMAX; ++c)
                invfact[idx.id[a][b][c]] = 1.0 / (fact[a] * fact[b] * fact[c]);
}

struct Node {
    int start, end;
    int child[8];
    double cx, cy, cz, half;
    double mass, comx, comy, comz;
    int depth;
};

struct Solver {
    int N;
    std::vector<double> m, x, y, z;
    std::vector<int> order;
    std::vector<int> tmp_order;
    std::vector<Node> nodes;
    std::vector<double> mom;
    std::vector<double> ax, ay, az;
    double root_cx, root_cy, root_cz, root_half;

    void read(const char* path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) std::exit(2);
        std::streamsize bytes = f.tellg();
        f.seekg(0);
        N = int(bytes / 32);
        m.resize(N); x.resize(N); y.resize(N); z.resize(N);
        std::vector<double> buf(4 * N);
        f.read(reinterpret_cast<char*>(buf.data()), bytes);
        for (int i = 0; i < N; ++i) {
            m[i] = buf[4*i]; x[i] = buf[4*i+1]; y[i] = buf[4*i+2]; z[i] = buf[4*i+3];
        }
    }

    uint64_t morton(double px, double py, double pz) const {
        auto q = [&](double v, double c) {
            double t = (v - (c - root_half)) / (2 * root_half);
            int64_t u = int64_t(t * 2097152.0);
            if (u < 0) u = 0; if (u > 2097151) u = 2097151;
            return uint64_t(u);
        };
        uint64_t xx = q(px, root_cx), yy = q(py, root_cy), zz = q(pz, root_cz);
        auto part = [](uint64_t v) {
            v &= 0x1fffff;
            v = (v | v << 32) & 0x1f00000000ffffULL;
            v = (v | v << 16) & 0x1f0000ff0000ffULL;
            v = (v | v << 8)  & 0x100f00f00f00f00fULL;
            v = (v | v << 4)  & 0x10c30c30c30c30c3ULL;
            v = (v | v << 2)  & 0x1249249249249249ULL;
            return v;
        };
        return part(xx) | (part(yy) << 1) | (part(zz) << 2);
    }

    void build() {
        double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0], zmin = z[0], zmax = z[0];
        for (int i = 1; i < N; ++i) {
            xmin = std::min(xmin, x[i]); xmax = std::max(xmax, x[i]);
            ymin = std::min(ymin, y[i]); ymax = std::max(ymax, y[i]);
            zmin = std::min(zmin, z[i]); zmax = std::max(zmax, z[i]);
        }
        root_cx = 0.5 * (xmin + xmax); root_cy = 0.5 * (ymin + ymax); root_cz = 0.5 * (zmin + zmax);
        root_half = 0.5 * std::max({xmax - xmin, ymax - ymin, zmax - zmin});
        if (root_half <= 0) root_half = 1.0;
        root_half *= 1.0000000001;
        order.resize(N);
        tmp_order.resize(N);
        for (int i = 0; i < N; ++i) order[i] = i;
        std::vector<uint64_t> code(N);
        for (int i = 0; i < N; ++i) code[i] = morton(x[i], y[i], z[i]);
        std::sort(order.begin(), order.end(), [&](int a, int b){ return code[a] < code[b]; });
        nodes.reserve(2 * N / LEAF + 16);
        mom.reserve((2 * N / LEAF + 16) * MC);
        build_rec(0, N, root_cx, root_cy, root_cz, root_half, 0);
    }

    int new_node(int s, int e, double cx, double cy, double cz, double h, int depth) {
        Node n; n.start = s; n.end = e; n.cx = cx; n.cy = cy; n.cz = cz; n.half = h; n.depth = depth;
        n.mass = 0; n.comx = cx; n.comy = cy; n.comz = cz;
        for (int k = 0; k < 8; ++k) n.child[k] = -1;
        nodes.push_back(n);
        mom.resize(nodes.size() * MC, 0.0);
        return int(nodes.size() - 1);
    }

    void build_rec(int s, int e, double cx, double cy, double cz, double h, int depth) {
        int ni = new_node(s, e, cx, cy, cz, h, depth);
        if (e - s <= LEAF || depth >= 20 || h < 1e-12) {
            leaf_moments(ni);
            return;
        }
        int count[8] = {0,0,0,0,0,0,0,0};
        for (int t = s; t < e; ++t) {
            int i = order[t];
            int oct = (x[i] >= cx ? 1 : 0) | (y[i] >= cy ? 2 : 0) | (z[i] >= cz ? 4 : 0);
            ++count[oct];
        }
        int start[8];
        int pos = s;
        for (int oct = 0; oct < 8; ++oct) { start[oct] = pos; pos += count[oct]; }
        int next[8]; for (int oct = 0; oct < 8; ++oct) next[oct] = start[oct];
        for (int t = s; t < e; ++t) {
            int i = order[t];
            int oct = (x[i] >= cx ? 1 : 0) | (y[i] >= cy ? 2 : 0) | (z[i] >= cz ? 4 : 0);
            tmp_order[next[oct]++] = i;
        }
        for (int t = s; t < e; ++t) order[t] = tmp_order[t];
        int child_ids[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
        double hh = 0.5 * h;
        for (int oct = 0; oct < 8; ++oct) {
            if (count[oct] == 0) continue;
            double ccx = cx + ((oct & 1) ? hh : -hh);
            double ccy = cy + ((oct & 2) ? hh : -hh);
            double ccz = cz + ((oct & 4) ? hh : -hh);
            child_ids[oct] = int(nodes.size());
            build_rec(start[oct], start[oct] + count[oct], ccx, ccy, ccz, hh, depth + 1);
        }
        for (int oct = 0; oct < 8; ++oct) nodes[ni].child[oct] = child_ids[oct];
        shift_moments(ni);
    }

    void leaf_moments(int ni) {
        const Node& n = nodes[ni];
        double* M = mom.data() + size_t(ni) * MC;
        double mass = 0, sx = 0, sy = 0, sz = 0;
        for (int t = n.start; t < n.end; ++t) {
            int i = order[t];
            mass += m[i]; sx += m[i] * x[i]; sy += m[i] * y[i]; sz += m[i] * z[i];
        }
        nodes[ni].mass = mass;
        if (mass > 0) {
            nodes[ni].comx = sx / mass; nodes[ni].comy = sy / mass; nodes[ni].comz = sz / mass;
        }
        double cx = nodes[ni].comx, cy = nodes[ni].comy, cz = nodes[ni].comz;
        for (int t = n.start; t < n.end; ++t) {
            int i = order[t];
            double dx = x[i] - cx, dy = y[i] - cy, dz = z[i] - cz;
            double px[PMAX + 1], py[PMAX + 1], pz[PMAX + 1];
            px[0] = py[0] = pz[0] = 1.0;
            for (int a = 1; a <= PMAX; ++a) { px[a] = px[a-1] * dx; py[a] = py[a-1] * dy; pz[a] = pz[a-1] * dz; }
            double mm = m[i];
            for (int a = 0; a <= PMAX; ++a) {
                double sx = (a & 1) ? -mm : mm;
                for (int b = 0; a + b <= PMAX; ++b) {
                    double sy = (b & 1) ? -sx : sx;
                    for (int c = 0; a + b + c <= PMAX; ++c) {
                        double sz = (c & 1) ? -sy : sy;
                        M[idx.id[a][b][c]] += sz * px[a] * py[b] * pz[c] / (fact[a] * fact[b] * fact[c]);
                    }
                }
            }
        }
    }

    void shift_moments(int ni) {
        const Node& n = nodes[ni];
        double* Mp = mom.data() + size_t(ni) * MC;
        double mass = 0, sx = 0, sy = 0, sz = 0;
        for (int oct = 0; oct < 8; ++oct) {
            int ci = n.child[oct];
            if (ci < 0) continue;
            const Node& c = nodes[ci];
            mass += c.mass; sx += c.mass * c.comx; sy += c.mass * c.comy; sz += c.mass * c.comz;
        }
        nodes[ni].mass = mass;
        if (mass > 0) {
            nodes[ni].comx = sx / mass; nodes[ni].comy = sy / mass; nodes[ni].comz = sz / mass;
        }
        for (int oct = 0; oct < 8; ++oct) {
            int ci = n.child[oct];
            if (ci < 0) continue;
            const Node& c = nodes[ci];
            const double* Mc = mom.data() + size_t(ci) * MC;
            double dx = nodes[ni].comx - c.comx, dy = nodes[ni].comy - c.comy, dz = nodes[ni].comz - c.comz;
            double px[PMAX + 1], py[PMAX + 1], pz[PMAX + 1];
            px[0] = py[0] = pz[0] = 1.0;
            for (int a = 1; a <= PMAX; ++a) { px[a] = px[a-1] * dx / a; py[a] = py[a-1] * dy / a; pz[a] = pz[a-1] * dz / a; }
            for (int a = 0; a <= PMAX; ++a)
                for (int b = 0; a + b <= PMAX; ++b)
                    for (int c = 0; a + b + c <= PMAX; ++c) {
                        int ia = idx.id[a][b][c];
                        double sum = 0;
                        for (int u = 0; u <= a; ++u)
                            for (int v = 0; v <= b; ++v)
                                for (int w = 0; w <= c; ++w)
                                    sum += Mc[idx.id[u][v][w]] * px[a-u] * py[b-v] * pz[c-w];
                        Mp[ia] += sum;
                    }
        }
    }

    void derivatives(const double R[3], double* D, int maxorder) const {
        double g = R[0]*R[0] + R[1]*R[1] + R[2]*R[2] + EPS2;
        D[0] = 1.0 / std::sqrt(g);
        for (int n = 1; n <= maxorder; ++n) {
            for (int t = idx.offset[n]; t < idx.offset[n+1]; ++t) {
                int bi = midx[t][0], bj = midx[t][1], bk = midx[t][2];
                int axis = bi > 0 ? 0 : (bj > 0 ? 1 : 2);
                int ai = bi, aj = bj, ak = bk;
                if (axis == 0) --ai; else if (axis == 1) --aj; else --ak;
                int alpha = idx.id[ai][aj][ak];
                double Ri = R[axis];
                double val = Ri * D[alpha];
                int aa[3] = {ai, aj, ak};
                if (aa[axis] > 0) {
                    int b[3] = {ai, aj, ak}; --b[axis];
                    val += aa[axis] * D[idx.id[b[0]][b[1]][b[2]]];
                }
                for (int j = 0; j < 3; ++j) {
                    if (aa[j] == 0) continue;
                    int b[3] = {ai, aj, ak}; --b[j]; b[axis] += 1;
                    val += 2.0 * aa[j] * R[j] * D[idx.id[b[0]][b[1]][b[2]]];
                    if (aa[j] >= 2) {
                        int c[3] = {ai, aj, ak}; c[j] -= 2; c[axis] += 1;
                        val += aa[j] * (aa[j] - 1) * D[idx.id[c[0]][c[1]][c[2]]];
                    }
                }
                D[t] = -val / g;
            }
        }
    }

    void m2l(int si, double tcx, double tcy, double tcz, double* L, int p) const {
        const Node& s = nodes[si];
        double R[3] = {tcx - s.comx, tcy - s.comy, tcz - s.comz};
        std::array<double, 2048> D;
        derivatives(R, D.data(), 2 * p);
        const double* S = mom.data() + size_t(si) * MC;
        for (int a = 0; a <= p; ++a)
            for (int b = 0; a + b <= p; ++b)
                for (int c = 0; a + b + c <= p; ++c) {
                    int ia = idx.id[a][b][c];
                    double sv = S[ia];
                    if (sv == 0.0) continue;
                    for (int u = 0; u <= p; ++u)
                        for (int v = 0; u + v <= p; ++v)
                            for (int w = 0; u + v + w <= p; ++w) {
                                int ib = idx.id[u][v][w];
                                L[ib] += sv * D[idx.id[a+u][b+v][c+w]] * invfact[ib];
                            }
                }
    }

    void l2p(int t, double tcx, double tcy, double tcz, const double* L, double& axv, double& ayv, double& azv) const {
        int i = order[t];
        double ux = x[i] - tcx, uy = y[i] - tcy, uz = z[i] - tcz;
        double px[PMAX + 1], py[PMAX + 1], pz[PMAX + 1];
        px[0] = py[0] = pz[0] = 1.0;
        for (int a = 1; a <= PMAX; ++a) { px[a] = px[a-1] * ux; py[a] = py[a-1] * uy; pz[a] = pz[a-1] * uz; }
        for (int a = 0; a <= PMAX; ++a)
            for (int b = 0; a + b <= PMAX; ++b)
                for (int c = 0; a + b + c <= PMAX; ++c) {
                    double lv = L[idx.id[a][b][c]];
                    if (a) axv += a * lv * px[a-1] * py[b] * pz[c];
                    if (b) ayv += b * lv * px[a] * py[b-1] * pz[c];
                    if (c) azv += c * lv * px[a] * py[b] * pz[c-1];
                }
    }

    void direct_leaf(int ni, int self_t, double tx, double ty, double tz, double& axv, double& ayv, double& azv) const {
        const Node& n = nodes[ni];
        for (int t = n.start; t < n.end; ++t) {
            if (t == self_t) continue;
            int j = order[t];
            double dx = x[j] - tx, dy = y[j] - ty, dz = z[j] - tz;
            double r2 = dx*dx + dy*dy + dz*dz + EPS2;
            double inv = 1.0 / (r2 * std::sqrt(r2));
            double w = m[j] * inv;
            axv += w * dx; ayv += w * dy; azv += w * dz;
        }
    }

    void eval_node(int ni, double tx, double ty, double tz, double& axv, double& ayv, double& azv) const {
        const Node& n = nodes[ni];
        double rx = tx - n.comx, ry = ty - n.comy, rz = tz - n.comz;
        double g = rx*rx + ry*ry + rz*rz + EPS2;
        double invg = 1.0 / g;
        double A = invg / std::sqrt(g);
        if (n.half * n.half < 0.04 * g) {
            double w = n.mass * A;
            axv -= w * rx; ayv -= w * ry; azv -= w * rz;
            return;
        }
        double B = A * invg;
        double rx2 = rx*rx, ry2 = ry*ry, rz2 = rz*rz;
        std::array<double, 64> D;
        D[idx.id[0][0][0]] = 1.0 / std::sqrt(g);
        D[idx.id[1][0][0]] = -rx * A;
        D[idx.id[0][1][0]] = -ry * A;
        D[idx.id[0][0][1]] = -rz * A;
        D[idx.id[2][0][0]] = (3*rx2*invg - 1) * A;
        D[idx.id[0][2][0]] = (3*ry2*invg - 1) * A;
        D[idx.id[0][0][2]] = (3*rz2*invg - 1) * A;
        D[idx.id[1][1][0]] = 3*rx*ry*invg * A;
        D[idx.id[1][0][1]] = 3*rx*rz*invg * A;
        D[idx.id[0][1][1]] = 3*ry*rz*invg * A;
        D[idx.id[3][0][0]] = (9*rx*invg - 15*rx*rx2*invg*invg) * A;
        D[idx.id[0][3][0]] = (9*ry*invg - 15*ry*ry2*invg*invg) * A;
        D[idx.id[0][0][3]] = (9*rz*invg - 15*rz*rz2*invg*invg) * A;
        D[idx.id[2][1][0]] = (3*ry*invg - 15*rx2*ry*invg*invg) * A;
        D[idx.id[2][0][1]] = (3*rz*invg - 15*rx2*rz*invg*invg) * A;
        D[idx.id[1][2][0]] = (3*rx*invg - 15*rx*ry2*invg*invg) * A;
        D[idx.id[0][2][1]] = (3*rz*invg - 15*ry2*rz*invg*invg) * A;
        D[idx.id[1][0][2]] = (3*rx*invg - 15*rx*rz2*invg*invg) * A;
        D[idx.id[0][1][2]] = (3*ry*invg - 15*ry*rz2*invg*invg) * A;
        D[idx.id[1][1][1]] = -15*rx*ry*rz*invg*invg * A;
        const double* M = mom.data() + size_t(ni) * MC;
        for (int a = 0; a <= PMAX; ++a)
            for (int b = 0; a + b <= PMAX; ++b)
                for (int c = 0; a + b + c <= PMAX; ++c) {
                    double mv = M[idx.id[a][b][c]];
                    axv += mv * D[idx.id[a+1][b][c]];
                    ayv += mv * D[idx.id[a][b+1][c]];
                    azv += mv * D[idx.id[a][b][c+1]];
                }
    }

    void solve() {
        ax.assign(N, 0.0); ay.assign(N, 0.0); az.assign(N, 0.0);
        std::vector<int> stack; stack.reserve(4096);
        double theta2 = THETA * THETA;
        for (int t = 0; t < N; ++t) {
            int i = order[t];
            double tx = x[i], ty = y[i], tz = z[i];
            double axv = 0, ayv = 0, azv = 0;
            stack.clear(); stack.push_back(0);
            while (!stack.empty()) {
                int ni = stack.back(); stack.pop_back();
                const Node& n = nodes[ni];
                bool leaf = true;
                for (int k = 0; k < 8; ++k) if (n.child[k] >= 0) { leaf = false; break; }
                if (leaf) {
                    direct_leaf(ni, t, tx, ty, tz, axv, ayv, azv);
                    continue;
                }
                if (t >= n.start && t < n.end) {
                    for (int k = 0; k < 8; ++k) if (n.child[k] >= 0) stack.push_back(n.child[k]);
                    continue;
                }
                double dx = tx - n.comx, dy = ty - n.comy, dz = tz - n.comz;
                double d2 = dx*dx + dy*dy + dz*dz;
                if (d2 * theta2 > n.half * n.half) {
                    eval_node(ni, tx, ty, tz, axv, ayv, azv);
                } else {
                    for (int k = 0; k < 8; ++k) if (n.child[k] >= 0) stack.push_back(n.child[k]);
                }
            }
            ax[i] = axv; ay[i] = ayv; az[i] = azv;
        }
    }

    void write(const char* path) const {
        std::ofstream f(path, std::ios::binary);
        std::vector<double> buf(3 * N);
        for (int i = 0; i < N; ++i) { buf[3*i] = ax[i]; buf[3*i+1] = ay[i]; buf[3*i+2] = az[i]; }
        f.write(reinterpret_cast<const char*>(buf.data()), 24 * N);
    }
};

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    init_index();
    Solver s;
    s.read(argv[1]);
    if (const char* th = std::getenv("GALAXY_THETA")) {
        THETA = std::atof(th);
    } else {
        double budget = 1e9;
        if (const char* b = std::getenv("GALAXY_CPU_LIMIT_S")) budget = std::atof(b);
        double scale = s.N / 65536.0;
        double extx = 0, exty = 0, extz = 0;
        // Root extents are available after build; use a cheap pre-scan here.
        double xmin = s.x[0], xmax = s.x[0], ymin = s.y[0], ymax = s.y[0], zmin = s.z[0], zmax = s.z[0];
        for (int i = 1; i < s.N; ++i) {
            xmin = std::min(xmin, s.x[i]); xmax = std::max(xmax, s.x[i]);
            ymin = std::min(ymin, s.y[i]); ymax = std::max(ymax, s.y[i]);
            zmin = std::min(zmin, s.z[i]); zmax = std::max(zmax, s.z[i]);
        }
        extx = xmax - xmin; exty = ymax - ymin; extz = zmax - zmin;
        double emax = std::max({extx, exty, extz});
        double emin = std::min({extx, exty, extz});
        double t05 = (emax > 0 && emin / emax < 0.6 ? 0.7 : 2.8) * scale;
        double theta = 0.5 * std::cbrt(t05 / (0.55 * budget));
        if (theta < 0.5) theta = 0.5;
        if (theta > 0.8) theta = 0.8;
        THETA = theta;
    }
    s.build();
    s.solve();
    s.write(argv[2]);
    return 0;
}
