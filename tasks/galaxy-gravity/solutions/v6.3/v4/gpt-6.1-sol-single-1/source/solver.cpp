#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <cstdint>
#include <sys/stat.h>
#include <immintrin.h>

static double cpu_time() {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return double(t.tv_sec) + 1.e-9 * double(t.tv_nsec);
}

constexpr int PMAX = 20;
constexpr int CMAX = (PMAX + 1) * (PMAX + 2) * (PMAX + 3) / 6;
constexpr double EPS2 = 0.0001;

#include "kernels.inc"

struct Particle {
    double m, x, y, z;
    uint32_t id;
};

struct Node {
    double x, y, z, r, mass;
    int begin, end, left, right, leaf;
};

struct Term {
    int x, y, z, degree;
    int prev[3], prev2[3];
    double factorial;
};

struct Pair { int a, b; uint8_t p; bool mutual, fullsoft; };

class Solver;
struct Reduction { int i; double weight; };
struct Shift { int i, degree; double weight; };
class Harmonic {
public:
    int p, nc, stride;
    std::vector<int> full, compact;
    std::vector<int> gradient_x, gradient_y, gradient_z;
    std::vector<double> gradient_factor;
    std::vector<std::vector<Reduction>> reduction;
    std::vector<std::vector<Shift>> upshift[3], downshift[3];
    std::vector<std::vector<int>> gamma;
    double *moments = nullptr, *locals = nullptr;
    Harmonic(int degree, Solver &s);
    ~Harmonic() { std::free(moments); std::free(locals); }
    void shift(double *a, double dx, double dy, double dz, bool upward);
    void upward(Solver &s);
    void far_field(Solver &s);
    void downward(Solver &s);
    template<bool Both> void translate4(Solver &s, const Pair *pairs, int valid, int degree = 0);
};

class Solver {
public:
    int order, moment_order, derivative_order, leaf_size, nc, mnc, stride, mstride;
    double theta, budget, started, tolerance, soft_cutoff, soft_theta;
    bool debug;
    bool hybrid;
    size_t n;
    std::vector<Particle> particles;
    std::vector<Node> nodes;
    std::vector<int> leaves;
    std::vector<Pair> far_pairs;
    std::vector<std::vector<int>> near;
    std::vector<double> x, y, z, mass, ax, ay, az;
    std::vector<uint32_t> ids;
    std::vector<unsigned char> active;
    std::vector<int> active_count, correction_targets;
    std::vector<double> field;
    std::vector<Term> terms;
    std::vector<std::vector<int>> gamma;
    std::array<int, PMAX + 1> count{};
    int index[PMAX + 1][PMAX + 1][PMAX + 1];
    double *moments = nullptr, *locals = nullptr;
    uint64_t direct_work = 0;
    double finish_reserve = 0, direct_deadline = 1.e100;
    double write_margin = 0.02, point_opening = 0.35;
    double measured_direct = 1.4e-9, measured_base[2][2]{}, measured_harmonic[2]{};

    Solver(int p, int ls, double th, double lim)
        : order(p), leaf_size(ls), theta(th), budget(lim),
          started(cpu_time()), debug(std::getenv("GALAXY_DEBUG") != nullptr) {
        hybrid = false;
        if (const char *s = std::getenv("GALAXY_HORDER")) hybrid = std::atoi(s) > 0;
        soft_cutoff = 0.2;
        soft_theta = 0.45;
        if (const char *s = std::getenv("GALAXY_SOFT_CUTOFF")) soft_cutoff = std::atof(s);
        if (const char *s = std::getenv("GALAXY_SOFT_THETA")) soft_theta = std::atof(s);
        tolerance = 0;
        if (const char *s = std::getenv("GALAXY_TOL")) tolerance = std::atof(s);
        moment_order = order;
        if (const char *s = std::getenv("GALAXY_MORDER"))
            moment_order = std::max(2, std::min(order, std::atoi(s)));
        derivative_order = std::min(PMAX, 2*order);
        if (const char *s = std::getenv("GALAXY_DMAX"))
            derivative_order = std::max(order, std::min(PMAX, std::atoi(s)));
        std::memset(index, -1, sizeof(index));
        double fact[PMAX + 1];
        fact[0] = 1;
        for (int i = 1; i <= PMAX; ++i) fact[i] = fact[i-1] * i;
        for (int d = 0; d <= PMAX; ++d) {
            for (int i = 0; i <= d; ++i) {
                for (int j = 0; j <= d-i; ++j) {
                    int k = d-i-j;
                    Term t{};
                    t.x = i; t.y = j; t.z = k; t.degree = d;
                    t.factorial = fact[i] * fact[j] * fact[k];
                    t.prev[0] = i ? index[i-1][j][k] : -1;
                    t.prev[1] = j ? index[i][j-1][k] : -1;
                    t.prev[2] = k ? index[i][j][k-1] : -1;
                    t.prev2[0] = i > 1 ? index[i-2][j][k] : -1;
                    t.prev2[1] = j > 1 ? index[i][j-2][k] : -1;
                    t.prev2[2] = k > 1 ? index[i][j][k-2] : -1;
                    index[i][j][k] = int(terms.size());
                    terms.push_back(t);
                }
            }
            count[d] = int(terms.size());
        }
        nc = count[order];
        mnc = count[moment_order];
        stride = (nc + 7) & ~7;
        mstride = (mnc + 7) & ~7;
        gamma.resize(nc);
        for (int a = 1; a < nc; ++a) {
            const auto &ta = terms[a];
            gamma[a].resize(count[std::min(moment_order, derivative_order-ta.degree)]);
            for (int b = 0; b < int(gamma[a].size()); ++b) {
                const auto &tb = terms[b];
                gamma[a][b] = index[ta.x+tb.x][ta.y+tb.y][ta.z+tb.z];
            }
        }
    }

    ~Solver() {
        std::free(moments);
        std::free(locals);
    }

    void log(const char *s) const {
        if (debug) std::fprintf(stderr, "%s %.6f\n", s, cpu_time()-started);
    }

    bool read(const char *path) {
        struct stat st;
        if (stat(path, &st) || st.st_size % 32) return false;
        n = st.st_size / 32;
        FILE *f = std::fopen(path, "rb");
        if (!f) return false;
        particles.resize(n);
        constexpr size_t block = 16384;
        std::vector<double> buf(4*block);
        for (size_t off = 0; off < n; off += block) {
            size_t len = std::min(block, n-off);
            if (std::fread(buf.data(), 32, len, f) != len) {
                std::fclose(f); return false;
            }
            for (size_t i = 0; i < len; ++i) {
                particles[off+i] = {buf[4*i], buf[4*i+1], buf[4*i+2],
                                    buf[4*i+3], uint32_t(off+i)};
            }
        }
        std::fclose(f);
        return true;
    }

    int build_node(int begin, int end) {
        int ni = int(nodes.size());
        nodes.push_back({});
        double lo[3] = {1.e300, 1.e300, 1.e300};
        double hi[3] = {-1.e300, -1.e300, -1.e300};
        double sx = 0, sy = 0, sz = 0, sm = 0;
        for (int i = begin; i < end; ++i) {
            const auto &p = particles[i];
            lo[0] = std::min(lo[0], p.x); hi[0] = std::max(hi[0], p.x);
            lo[1] = std::min(lo[1], p.y); hi[1] = std::max(hi[1], p.y);
            lo[2] = std::min(lo[2], p.z); hi[2] = std::max(hi[2], p.z);
            sm += p.m; sx += p.m*p.x; sy += p.m*p.y; sz += p.m*p.z;
        }
        double cx = sx/sm, cy = sy/sm, cz = sz/sm;
        double r2 = 0;
        for (int i = begin; i < end; ++i) {
            double dx = particles[i].x-cx, dy = particles[i].y-cy, dz = particles[i].z-cz;
            r2 = std::max(r2, dx*dx+dy*dy+dz*dz);
        }
        Node nd{cx, cy, cz, std::max(std::sqrt(r2), 1.e-20), sm,
                begin, end, -1, -1, -1};
        if (end-begin <= leaf_size) {
            nd.leaf = int(leaves.size());
            leaves.push_back(ni);
        } else {
            int axis = 0;
            if (hi[1]-lo[1] > hi[axis]-lo[axis]) axis = 1;
            if (hi[2]-lo[2] > hi[axis]-lo[axis]) axis = 2;
            double split = (lo[axis]+hi[axis])*0.5;
            auto cmp = [axis, split](const Particle &a) {
                return (axis == 0 ? a.x : axis == 1 ? a.y : a.z) < split;
            };
            int mid = int(std::partition(particles.begin()+begin, particles.begin()+end, cmp)
                          - particles.begin());
            if (mid == begin || mid == end) mid = begin+(end-begin)/2;
            nd.left = build_node(begin, mid);
            nd.right = build_node(mid, end);
        }
        nodes[ni] = nd;
        return ni;
    }

    void build() {
        nodes.reserve(4*n/leaf_size+16);
        build_node(0, int(n));
        x.resize(n); y.resize(n); z.resize(n); mass.resize(n); ids.resize(n);
        ax.assign(n, 0); ay.assign(n, 0); az.assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            x[i] = particles[i].x; y[i] = particles[i].y; z[i] = particles[i].z;
            mass[i] = particles[i].m; ids[i] = particles[i].id;
        }
        std::vector<Particle>().swap(particles);
        select_targets();
        near.resize(leaves.size());
        far_pairs.reserve(nodes.size()*200);
        size_t bytes = size_t(nodes.size()) * stride * sizeof(double);
        size_t mbytes = size_t(nodes.size()) * mstride * sizeof(double);
        if (posix_memalign(reinterpret_cast<void **>(&moments), 64, mbytes) ||
            posix_memalign(reinterpret_cast<void **>(&locals), 64, bytes)) {
            std::fprintf(stderr, "Unable to allocate expansions\n");
            std::exit(2);
        }
        std::memset(moments, 0, mbytes);
        std::memset(locals, 0, bytes);
        log("tree");
    }

    void select_targets() {
        active.assign(n, 1);
        const char *s = std::getenv("GALAXY_CORRECT");
        double fraction = s ? std::atof(s) : 0;
        if (fraction > 0) {
            std::vector<std::pair<double, int>> ranked;
            ranked.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                double dx = x[i]-nodes[0].x, dy = y[i]-nodes[0].y, dz = z[i]-nodes[0].z;
                ranked.emplace_back(dx*dx+dy*dy+dz*dz, int(i));
            }
            size_t num = std::min(n, size_t(n*fraction));
            std::nth_element(ranked.begin(), ranked.begin()+num, ranked.end(),
                             [](auto a, auto b) { return a.first > b.first; });
            correction_targets.reserve(num);
            for (size_t k = 0; k < num; ++k) {
                int i = ranked[k].second;
                active[i] = 0;
                correction_targets.push_back(i);
            }
            std::sort(correction_targets.begin(), correction_targets.end());
        }
        active_count.resize(nodes.size());
        for (int k = int(nodes.size())-1; k >= 0; --k) {
            const auto &nd = nodes[k];
            if (nd.left >= 0) active_count[k] = active_count[nd.left]+active_count[nd.right];
            else {
                int num = 0;
                for (int i = nd.begin; i < nd.end; ++i) num += active[i] != 0;
                active_count[k] = num;
            }
        }
    }

    void shift_moment(double *a, double dx, double dy, double dz) {
        double powers[3][PMAX+1];
        double delta[3] = {dx, dy, dz};
        for (int axis = 0; axis < 3; ++axis) {
            powers[axis][0] = 1;
            for (int k = 1; k <= moment_order; ++k)
                powers[axis][k] = powers[axis][k-1]*delta[axis]/k;
        }
        for (int axis = 0; axis < 3; ++axis) {
            for (int i = mnc-1; i >= 1; --i) {
                int xx = terms[i].x, yy = terms[i].y, zz = terms[i].z;
                int lim = axis == 0 ? xx : axis == 1 ? yy : zz;
                double s = a[i];
                for (int k = 1; k <= lim; ++k) {
                    int j = index[xx-(axis==0?k:0)][yy-(axis==1?k:0)][zz-(axis==2?k:0)];
                    s += a[j]*powers[axis][k];
                }
                a[i] = s;
            }
        }
    }

    void upward() {
        for (int ni = int(nodes.size())-1; ni >= 0; --ni) {
            const Node &nd = nodes[ni];
            double *out = moments + size_t(ni)*mstride;
            if (nd.left < 0) {
                for (int j = nd.begin; j < nd.end; ++j) {
                    double xp[PMAX+1], yp[PMAX+1], zp[PMAX+1];
                    xp[0] = yp[0] = zp[0] = 1;
                    double dx = (x[j]-nd.x)/nd.r, dy = (y[j]-nd.y)/nd.r,
                           dz = (z[j]-nd.z)/nd.r;
                    for (int d = 1; d <= moment_order; ++d) {
                        xp[d] = xp[d-1]*dx/d;
                        yp[d] = yp[d-1]*dy/d;
                        zp[d] = zp[d-1]*dz/d;
                    }
                    for (int k = 0; k < mnc; ++k) {
                        const auto &t = terms[k];
                        out[k] += mass[j]*xp[t.x]*yp[t.y]*zp[t.z];
                    }
                }
            } else {
                for (int child : {nd.left, nd.right}) {
                    const auto &ch = nodes[child];
                    const double *in = moments + size_t(child)*mstride;
                    alignas(64) double tmp[CMAX];
                    double scale[PMAX+1];
                    scale[0] = 1;
                    for (int k = 1; k <= moment_order; ++k)
                        scale[k] = scale[k-1]*(ch.r/nd.r);
                    for (int k = 0; k < mnc; ++k)
                        tmp[k] = in[k]*scale[terms[k].degree];
                    shift_moment(tmp, (ch.x-nd.x)/nd.r, (ch.y-nd.y)/nd.r, (ch.z-nd.z)/nd.r);
                    for (int k = 0; k < mnc; ++k) out[k] += tmp[k];
                }
            }
            out[0] = nd.mass;
            out[1] = out[2] = out[3] = 0;
        }
        log("moments");
    }

    void traverse(int ia, int ib) {
        if (!active_count[ia] && !active_count[ib]) return;
        const auto &a = nodes[ia];
        const auto &b = nodes[ib];
        if (ia != ib) {
            double dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
            double r = a.r+b.r;
            double dist2 = dx*dx+dy*dy+dz*dz;
            bool fullsoft = hybrid && dist2 < soft_cutoff*soft_cutoff;
            double q = dist2+EPS2;
            double opening = fullsoft ? std::min(theta, soft_theta) : theta;
            double separation = hybrid && !fullsoft ? dist2 : q;
            if (r*r < opening*opening*separation) {
                int p = derivative_order;
                if (tolerance > 0) {
                    double sigma = r/std::sqrt(q);
                    double strength = std::max(b.mass/field[ia], a.mass/field[ib])/q;
                    strength /= (1-sigma)*(1-sigma);
                    double power = sigma*sigma;
                    p = 2;
                    while (p <= derivative_order && strength*power*(p+1) > tolerance) {
                        power *= sigma; ++p;
                    }
                }
                if (p <= derivative_order) {
                    if (active_count[ia])
                        far_pairs.push_back({ia, ib, uint8_t(p), active_count[ib] != 0, fullsoft});
                    else
                        far_pairs.push_back({ib, ia, uint8_t(p), false, fullsoft});
                    return;
                }
            }
        }
        if (a.left < 0 && b.left < 0) {
            if (active_count[ia]) near[a.leaf].push_back(ib);
            direct_work += uint64_t(active_count[ia])*(b.end-b.begin);
            if (ia != ib && active_count[ib]) {
                near[b.leaf].push_back(ia);
                direct_work += uint64_t(a.end-a.begin)*active_count[ib];
            }
        } else if (ia == ib) {
            traverse(a.left, a.left);
            traverse(a.left, a.right);
            traverse(a.right, a.right);
        } else if (b.left < 0 || (a.left >= 0 && a.r >= b.r)) {
            traverse(a.left, ib);
            traverse(a.right, ib);
        } else {
            traverse(ia, b.left);
            traverse(ia, b.right);
        }
    }

    void make_pairs() {
        if (tolerance > 0) estimate_fields();
        traverse(0, 0);
        std::stable_sort(far_pairs.begin(), far_pairs.end(),
                         [](const Pair &a, const Pair &b) {
                             if (a.mutual != b.mutual) return a.mutual < b.mutual;
                             if (a.fullsoft != b.fullsoft) return a.fullsoft < b.fullsoft;
                             return a.p < b.p;
                         });
        if (debug) std::fprintf(stderr, "nodes %zu leaves %zu far %zu direct %.3f M\n",
                               nodes.size(), leaves.size(), far_pairs.size(), direct_work*1.e-6);
        if (debug) {
            for (int p = 2; p <= derivative_order; ++p) {
                size_t num = 0;
                for (const auto &pair : far_pairs) num += pair.p == p;
                std::fprintf(stderr, "%d:%zu ", p, num);
            }
            std::fprintf(stderr, "\n");
        }
        log("pairs");
    }

    void reset_pairs() {
        far_pairs.clear();
        for (auto &v : near) v.clear();
        direct_work = 0;
        make_pairs();
    }

    void calibrate(Harmonic *h, double upward_seconds) {
        double begin = cpu_time();
        uint64_t work = direct(true);
        measured_direct = work ? 0.92*(cpu_time()-begin)/work : 1.4e-9;
        measured_direct = std::max(measured_direct, 0.7e-9);
        for (int both = 0; both < 2; ++both) {
            for (int soft = 0; soft < 2; ++soft) {
                size_t candidates = 0;
                for (auto pair : far_pairs)
                    if (pair.mutual == bool(both) && pair.fullsoft == bool(soft))
                        ++candidates;
                std::vector<Pair> sample;
                size_t amount = std::min(size_t(64), candidates);
                amount -= amount%4;
                size_t matching = 0;
                for (auto pair : far_pairs) {
                    if (sample.size() == amount) break;
                    if (pair.mutual != bool(both) || pair.fullsoft != bool(soft)) continue;
                    if (matching == sample.size()*candidates/amount) sample.push_back(pair);
                    ++matching;
                }
                double t = cpu_time();
                for (size_t k = 0; k < sample.size(); k += 4) {
                    if (both) translate4<true>(sample.data()+k, derivative_order, hybrid && !soft);
                    else translate4<false>(sample.data()+k, derivative_order, hybrid && !soft);
                }
                double elapsed = cpu_time()-t;
                measured_base[both][soft] = amount ? 0.92*elapsed/amount : (both?3.e-6:2.e-6);
                for (auto pair : sample) {
                    std::memset(locals+size_t(pair.a)*stride, 0, nc*sizeof(double));
                    std::memset(locals+size_t(pair.b)*stride, 0, nc*sizeof(double));
                }
                if (h && !soft) {
                    t = cpu_time();
                    for (size_t k = 0; k < sample.size(); k += 4) {
                        if (both) h->translate4<true>(*this, sample.data()+k, 4);
                        else h->translate4<false>(*this, sample.data()+k, 4);
                    }
                    elapsed = cpu_time()-t;
                    measured_harmonic[both] = amount ? 0.92*elapsed/amount : (both?6.e-6:4.e-6);
                    for (auto pair : sample) {
                        std::memset(h->locals+size_t(pair.a)*h->stride, 0, h->nc*sizeof(double));
                        std::memset(h->locals+size_t(pair.b)*h->stride, 0, h->nc*sizeof(double));
                    }
                }
            }
        }
        double tc = 0;
        if (!correction_targets.empty()) {
            size_t groups = std::min(size_t(32), correction_targets.size()/4);
            begin = cpu_time();
            for (size_t k = 0; k < groups; ++k) {
                size_t offset = (k*(correction_targets.size()/4)/groups)*4;
                point_force4(correction_targets.data()+offset, point_opening, std::min(8, moment_order+1));
            }
            tc = groups ? (cpu_time()-begin)*correction_targets.size()/(4*groups) : 0;
        }
        write_margin = std::max(0.015*budget, 0.01) + n*4.e-8;
        finish_reserve = 1.15*upward_seconds + 1.10*tc + write_margin;
        if (debug) std::fprintf(stderr, "calibration direct %.3g base %.3g/%.3g harmonic %.3g/%.3g finish %.4f\n",
            measured_direct, measured_base[0][0], measured_base[1][0],
            measured_harmonic[0], measured_harmonic[1], finish_reserve);
    }

    double predicted_compute(double *far_time = nullptr) const {
        uint64_t counts[2][2]{};
        for (auto pair : far_pairs) ++counts[pair.mutual][pair.fullsoft];
        double far = 0;
        for (int both = 0; both < 2; ++both)
            for (int soft = 0; soft < 2; ++soft)
                far += counts[both][soft]*(measured_base[both][soft]+
                          (hybrid && !soft ? measured_harmonic[both] : 0));
        if (far_time) *far_time = far;
        return direct_work*measured_direct+far;
    }

    void tune_geometry() {
        for (int trial = 0; trial < 4; ++trial) {
            double available = budget-cpu_time()-finish_reserve;
            available = std::max(0.02*budget, available);
            double predicted = predicted_compute();
            double ratio = predicted/(0.90*available);
            if (debug) std::fprintf(stderr, "tune theta %.4f soft %.4f compute %.4f available %.4f\n",
                                   theta, soft_theta, predicted, available);
            if (ratio > 0.97 && ratio < 1.03) break;
            double factor = std::pow(ratio, 1.0/3.0);
            factor = std::max(0.75, std::min(1.5, factor));
            double next = std::max(0.3, std::min(0.96, theta*factor));
            double nextsoft = std::max(0.22, std::min(0.85, soft_theta*factor));
            if (std::abs(next-theta)+std::abs(nextsoft-soft_theta) < 0.005) break;
            theta = next; soft_theta = nextsoft;
            reset_pairs();
        }
        double far = 0;
        predicted_compute(&far);
        direct_deadline = budget-finish_reserve-0.65*far+0.03*budget;
    }

    void estimate_fields() {
        field.resize(nodes.size());
        auto force_at = [&](int ti) {
            double fx = 0, fy = 0, fz = 0;
            int stack[256], sp = 0;
            stack[sp++] = 0;
            while (sp) {
                const auto &s = nodes[stack[--sp]];
                double dx = s.x-x[ti], dy = s.y-y[ti], dz = s.z-z[ti];
                double q = dx*dx+dy*dy+dz*dz+EPS2;
                if (s.r*s.r < 0.49*q) {
                    double f = s.mass/(q*std::sqrt(q));
                    fx += dx*f; fy += dy*f; fz += dz*f;
                } else if (s.left >= 0) {
                    stack[sp++] = s.left; stack[sp++] = s.right;
                } else {
                    for (int i = s.begin; i < s.end; ++i) {
                        double rx = x[i]-x[ti], ry = y[i]-y[ti], rz = z[i]-z[ti];
                        double q = rx*rx+ry*ry+rz*rz+EPS2;
                        double f = mass[i]/(q*std::sqrt(q));
                        fx += rx*f; fy += ry*f; fz += rz*f;
                    }
                }
            }
            return std::max(std::sqrt(fx*fx+fy*fy+fz*fz), 1.e-20);
        };
        for (int k = int(nodes.size())-1; k >= 0; --k) {
            const auto &t = nodes[k];
            if (t.left >= 0) {
                field[k] = std::min(field[t.left], field[t.right]);
            } else if (active_count[k] == 0) {
                field[k] = 1.e100;
            } else {
                int furthest = t.begin;
                double rmax = -1;
                for (int i = t.begin; i < t.end; ++i) {
                    if (!active[i]) continue;
                    double dx = x[i]-nodes[0].x, dy = y[i]-nodes[0].y, dz = z[i]-nodes[0].z;
                    double r2 = dx*dx+dy*dy+dz*dz;
                    if (r2 > rmax) { rmax = r2; furthest = i; }
                }
                double f = force_at(furthest);
                for (int j = 0; j < 4; ++j) {
                    int ti = t.begin+(t.end-t.begin-1)*j/3;
                    if (active[ti]) f = std::min(f, force_at(ti));
                }
                field[k] = f;
            }
        }
        log("field estimates");
    }

    static inline double horizontal(__m256d a) {
        __m128d s = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
        s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
        return _mm_cvtsd_f64(s);
    }

    uint64_t direct(bool calibration = false) {
        const __m256d soft = _mm256_set1_pd(EPS2);
        const __m256d one = _mm256_set1_pd(1.0);
        size_t step = calibration ? std::max(size_t(1), leaves.size()/12) : 1;
        bool fast = false;
        uint64_t work = 0;
        for (size_t li = 0; li < leaves.size(); li += step) {
            if (!calibration && (li%16 == 0) && cpu_time() > direct_deadline) fast = true;
            const auto &a = nodes[leaves[li]];
            for (int nb : near[li])
                work += uint64_t(active_count[leaves[li]])*(nodes[nb].end-nodes[nb].begin);
            for (int i = a.begin; i < a.end; ++i) {
                if (!active[i]) continue;
                if (fast) {
                    double fx = 0, fy = 0, fz = 0;
                    for (int nb : near[li]) {
                        const auto &b = nodes[nb];
                        double dx = b.x-x[i], dy = b.y-y[i], dz = b.z-z[i];
                        double q = dx*dx+dy*dy+dz*dz+EPS2;
                        double f = b.mass/(q*std::sqrt(q));
                        fx += dx*f; fy += dy*f; fz += dz*f;
                    }
                    ax[i] = fx; ay[i] = fy; az[i] = fz;
                    continue;
                }
                __m256d tx = _mm256_set1_pd(x[i]), ty = _mm256_set1_pd(y[i]), tz = _mm256_set1_pd(z[i]);
                __m256d fx = _mm256_setzero_pd(), fy = fx, fz = fx;
                double sx = 0, sy = 0, sz = 0;
                for (int nb : near[li]) {
                    const auto &b = nodes[nb];
                    int j = b.begin;
                    for (; j+3 < b.end; j += 4) {
                        __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(x.data()+j), tx);
                        __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(y.data()+j), ty);
                        __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(z.data()+j), tz);
                        __m256d q = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, soft)));
                        __m256d inv = _mm256_div_pd(one, _mm256_sqrt_pd(q));
                        __m256d f = _mm256_mul_pd(_mm256_loadu_pd(mass.data()+j),
                                                 _mm256_mul_pd(inv, _mm256_mul_pd(inv, inv)));
                        fx = _mm256_fmadd_pd(dx, f, fx);
                        fy = _mm256_fmadd_pd(dy, f, fy);
                        fz = _mm256_fmadd_pd(dz, f, fz);
                    }
                    for (; j < b.end; ++j) {
                        double dx = x[j]-x[i], dy = y[j]-y[i], dz = z[j]-z[i];
                        double q = dx*dx+dy*dy+dz*dz+EPS2;
                        double f = mass[j]/(q*std::sqrt(q));
                        sx += dx*f; sy += dy*f; sz += dz*f;
                    }
                }
                ax[i] = sx+horizontal(fx);
                ay[i] = sy+horizontal(fy);
                az[i] = sz+horizontal(fz);
            }
        }
        if (!calibration) log(fast ? "direct (budget fallback)" : "direct");
        return work;
    }

    void translate(int ia, int ib, int p) {
        const auto &a = nodes[ia];
        const auto &b = nodes[ib];
        double dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
        double inv = 1/std::sqrt(dx*dx+dy*dy+dz*dz+EPS2);
        double u[3] = {dx*inv, dy*inv, dz*inv};
        alignas(64) double c[CMAX], deriv[CMAX], ma[CMAX], mb[CMAX];
        double sa[PMAX+1], sb[PMAX+1];
        sa[0] = sb[0] = 1;
        for (int k = 1; k <= p; ++k) {
            sa[k] = sa[k-1]*(a.r*inv);
            sb[k] = sb[k-1]*(-b.r*inv);
        }
        c[0] = deriv[0] = 1;
        int len = count[std::min(order, p)];
        for (int i = 1; i < count[p]; ++i) {
            const auto &t = terms[i];
            double s = 0, s2 = 0;
            for (int k = 0; k < 3; ++k) {
                if (t.prev[k] >= 0) s += u[k]*c[t.prev[k]];
                if (t.prev2[k] >= 0) s2 += c[t.prev2[k]];
            }
            c[i] = -(2*t.degree-1)*s/t.degree - (t.degree-1)*s2/t.degree;
            deriv[i] = c[i]*t.factorial;
        }
        const double *am = moments + size_t(ia)*mstride;
        const double *bm = moments + size_t(ib)*mstride;
        for (int i = 0; i < count[std::min(moment_order, p-1)]; ++i) {
            int d = terms[i].degree;
            ma[i] = am[i]*sa[d];
            mb[i] = bm[i]*sb[d];
        }
        double *al = locals + size_t(ia)*stride;
        double *bl = locals + size_t(ib)*stride;
        for (int i = 1; i < len; ++i) {
            int deg = terms[i].degree;
            int bmax = count[std::min(moment_order, p-deg)];
            const int *g = gamma[i].data();
            double v1 = deriv[i]*bm[0], v2 = deriv[i]*am[0];
            for (int j = 4; j < bmax; ++j) {
                double d = deriv[g[j]];
                v1 += d*mb[j];
                v2 += d*ma[j];
            }
            al[i] += v1 * inv * sa[deg];
            bl[i] += v2 * inv * sb[deg];
        }
    }

    void far_field() {
        size_t i = 0;
        bool fast = false;
        while (i < far_pairs.size()) {
            int p = far_pairs[i].p;
            bool mutual = far_pairs[i].mutual;
            bool fullsoft = far_pairs[i].fullsoft;
            size_t end = i+1;
            while (end < far_pairs.size() && far_pairs[end].p == p &&
                   far_pairs[end].mutual == mutual && far_pairs[end].fullsoft == fullsoft) ++end;
            for (; i+3 < end; i += 4) {
                if ((i&255) < 4 && cpu_time() > budget-finish_reserve) fast = true;
                int usep = fast ? std::max(4, p-2) : p;
                if (fast && cpu_time() > budget-write_margin-0.025*budget) usep = 1;
                if (mutual) translate4<true>(far_pairs.data()+i, usep, hybrid && !fullsoft);
                else translate4<false>(far_pairs.data()+i, usep, hybrid && !fullsoft);
            }
            for (; i < end; ++i) {
                Pair padding[4] = {far_pairs[i], far_pairs[i], far_pairs[i], far_pairs[i]};
                if (mutual) translate4<true>(padding, p, hybrid && !fullsoft, 1);
                else translate4<false>(padding, p, hybrid && !fullsoft, 1);
            }
        }
        log(fast ? "m2l (budget fallback)" : "m2l");
    }

    template<bool Both> void translate4(const Pair *pairs, int p, bool subtract_laplace, int valid = 4) {
        const Node *a[4], *b[4];
        const double *am[4], *bm[4];
        double *al[4], *bl[4];
        for (int k = 0; k < 4; ++k) {
            a[k] = &nodes[pairs[k].a]; b[k] = &nodes[pairs[k].b];
            am[k] = moments+size_t(pairs[k].a)*mstride;
            bm[k] = moments+size_t(pairs[k].b)*mstride;
            al[k] = locals+size_t(pairs[k].a)*stride;
            bl[k] = locals+size_t(pairs[k].b)*stride;
        }
        __m256d dx = _mm256_set_pd(a[3]->x-b[3]->x, a[2]->x-b[2]->x,
                                  a[1]->x-b[1]->x, a[0]->x-b[0]->x);
        __m256d dy = _mm256_set_pd(a[3]->y-b[3]->y, a[2]->y-b[2]->y,
                                  a[1]->y-b[1]->y, a[0]->y-b[0]->y);
        __m256d dz = _mm256_set_pd(a[3]->z-b[3]->z, a[2]->z-b[2]->z,
                                  a[1]->z-b[1]->z, a[0]->z-b[0]->z);
        __m256d one = _mm256_set1_pd(1.0);
        __m256d inv = _mm256_div_pd(one, _mm256_sqrt_pd(
            _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy,
                _mm256_fmadd_pd(dz, dz, _mm256_set1_pd(EPS2))))));
        __m256d u[3] = {_mm256_mul_pd(dx, inv), _mm256_mul_pd(dy, inv),
                         _mm256_mul_pd(dz, inv)};
        alignas(64) __m256d deriv[CMAX], ma[CMAX], mb[CMAX];
        __m256d sa[PMAX+1], sb[PMAX+1];
        sa[0] = sb[0] = one;
        __m256d ratioa = _mm256_mul_pd(inv, _mm256_set_pd(a[3]->r, a[2]->r, a[1]->r, a[0]->r));
        __m256d ratiob = _mm256_mul_pd(inv, _mm256_set_pd(-b[3]->r, -b[2]->r, -b[1]->r, -b[0]->r));
        for (int k = 1; k <= p; ++k) {
            sa[k] = _mm256_mul_pd(sa[k-1], ratioa);
            sb[k] = _mm256_mul_pd(sb[k-1], ratiob);
        }
        int len = count[std::min(order, p)];
        kernel4(p, u[0], u[1], u[2], deriv);
        if (subtract_laplace) {
            alignas(64) __m256d laplace[CMAX];
            __m256d qlap = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
            __m256d ilap = _mm256_div_pd(one, _mm256_sqrt_pd(qlap));
            kernel4(p, _mm256_mul_pd(dx, ilap), _mm256_mul_pd(dy, ilap),
                    _mm256_mul_pd(dz, ilap), laplace);
            __m256d ratio = _mm256_div_pd(ilap, inv), power[PMAX+2];
            power[0] = ratio;
            for (int k = 1; k <= p; ++k) power[k] = _mm256_mul_pd(power[k-1], ratio);
            for (int k = 0; k < count[p]; ++k)
                deriv[k] = _mm256_fnmadd_pd(laplace[k], power[terms[k].degree], deriv[k]);
        }
        __m256d am0 = _mm256_set_pd(am[3][0], am[2][0], am[1][0], am[0][0]);
        __m256d bm0 = _mm256_set_pd(bm[3][0], bm[2][0], bm[1][0], bm[0][0]);
        for (int i = 4; i < count[std::min(moment_order, p-1)]; ++i) {
            int d = terms[i].degree;
            if constexpr (Both)
                ma[i] = _mm256_mul_pd(sa[d], _mm256_set_pd(am[3][i], am[2][i], am[1][i], am[0][i]));
            mb[i] = _mm256_mul_pd(sb[d], _mm256_set_pd(bm[3][i], bm[2][i], bm[1][i], bm[0][i]));
        }
        for (int i = 1; i < len; ++i) {
            int deg = terms[i].degree;
            int bmax = count[std::min(moment_order, p-deg)];
            const int *g = gamma[i].data();
            __m256d v1 = _mm256_mul_pd(deriv[i], bm0);
            __m256d v2 = _mm256_mul_pd(deriv[i], am0);
            __m256d w1 = _mm256_setzero_pd(), w2 = w1;
            __m256d t1 = w1, t2 = w1, s1 = w1, s2 = w1;
            int j = 4;
            for (; j+3 < bmax; j += 4) {
                __m256d d0 = deriv[g[j]], d1 = deriv[g[j+1]],
                         d2 = deriv[g[j+2]], d3 = deriv[g[j+3]];
                v1 = _mm256_fmadd_pd(d0, mb[j], v1);
                if constexpr (Both) v2 = _mm256_fmadd_pd(d0, ma[j], v2);
                w1 = _mm256_fmadd_pd(d1, mb[j+1], w1);
                if constexpr (Both) w2 = _mm256_fmadd_pd(d1, ma[j+1], w2);
                t1 = _mm256_fmadd_pd(d2, mb[j+2], t1);
                if constexpr (Both) t2 = _mm256_fmadd_pd(d2, ma[j+2], t2);
                s1 = _mm256_fmadd_pd(d3, mb[j+3], s1);
                if constexpr (Both) s2 = _mm256_fmadd_pd(d3, ma[j+3], s2);
            }
            v1 = _mm256_add_pd(_mm256_add_pd(v1, w1), _mm256_add_pd(t1, s1));
            v2 = _mm256_add_pd(_mm256_add_pd(v2, w2), _mm256_add_pd(t2, s2));
            for (; j < bmax; ++j) {
                __m256d d = deriv[g[j]];
                v1 = _mm256_fmadd_pd(d, mb[j], v1);
                if constexpr (Both) v2 = _mm256_fmadd_pd(d, ma[j], v2);
            }
            v1 = _mm256_mul_pd(v1, _mm256_mul_pd(inv, sa[deg]));
            v2 = _mm256_mul_pd(v2, _mm256_mul_pd(inv, sb[deg]));
            alignas(32) double da[4], db[4];
            _mm256_store_pd(da, v1); _mm256_store_pd(db, v2);
            for (int k = 0; k < valid; ++k) {
                al[k][i] += da[k];
                if constexpr (Both) bl[k][i] += db[k];
            }
        }
    }

    void shift_local(double *a, double dx, double dy, double dz) {
        double powers[3][PMAX+1];
        double delta[3] = {dx, dy, dz};
        for (int axis = 0; axis < 3; ++axis) {
            powers[axis][0] = 1;
            for (int k = 1; k <= order; ++k)
                powers[axis][k] = powers[axis][k-1]*delta[axis]/k;
        }
        for (int axis = 0; axis < 3; ++axis) {
            for (int i = 1; i < nc; ++i) {
                int xx = terms[i].x, yy = terms[i].y, zz = terms[i].z;
                int lim = order-terms[i].degree;
                double s = a[i];
                for (int k = 1; k <= lim; ++k) {
                    int j = index[xx+(axis==0?k:0)][yy+(axis==1?k:0)][zz+(axis==2?k:0)];
                    s += a[j]*powers[axis][k];
                }
                a[i] = s;
            }
        }
    }

    void downward() {
        bool fast = false;
        for (size_t ni = 0; ni < nodes.size(); ++ni) {
            if ((ni&127) == 0 && cpu_time() > budget-write_margin-0.005*budget) fast = true;
            if (!active_count[ni]) continue;
            const auto &nd = nodes[ni];
            double *in = locals + ni*stride;
            if (fast) {
                if (nd.left >= 0) {
                    for (int child : {nd.left, nd.right}) {
                        double ratio = nodes[child].r/nd.r;
                        double *out = locals+size_t(child)*stride;
                        for (int k = 1; k <= 3; ++k) out[k] += in[k]*ratio;
                    }
                } else {
                    for (int j = nd.begin; j < nd.end; ++j) if (active[j]) {
                        ax[j] += in[index[1][0][0]]/nd.r;
                        ay[j] += in[index[0][1][0]]/nd.r;
                        az[j] += in[index[0][0][1]]/nd.r;
                    }
                }
                continue;
            }
            if (nd.left >= 0) {
                for (int child : {nd.left, nd.right}) {
                    if (!active_count[child]) continue;
                    const auto &ch = nodes[child];
                    alignas(64) double tmp[CMAX];
                    std::memcpy(tmp, in, nc*sizeof(double));
                    shift_local(tmp, (ch.x-nd.x)/nd.r, (ch.y-nd.y)/nd.r, (ch.z-nd.z)/nd.r);
                    double scale[PMAX+1];
                    scale[0] = 1;
                    for (int k = 1; k <= order; ++k)
                        scale[k] = scale[k-1]*(ch.r/nd.r);
                    double *out = locals + size_t(child)*stride;
                    for (int k = 1; k < nc; ++k)
                        out[k] += tmp[k]*scale[terms[k].degree];
                }
            } else {
                for (int j = nd.begin; j < nd.end; ++j) {
                    if (!active[j]) continue;
                    double xp[PMAX+1], yp[PMAX+1], zp[PMAX+1];
                    xp[0] = yp[0] = zp[0] = 1;
                    double dx = (x[j]-nd.x)/nd.r, dy = (y[j]-nd.y)/nd.r,
                           dz = (z[j]-nd.z)/nd.r;
                    for (int k = 1; k <= order; ++k) {
                        xp[k] = xp[k-1]*dx/k;
                        yp[k] = yp[k-1]*dy/k;
                        zp[k] = zp[k-1]*dz/k;
                    }
                    double fx = 0, fy = 0, fz = 0;
                    for (int k = 1; k < nc; ++k) {
                        const auto &t = terms[k];
                        if (t.x) fx += in[k]*xp[t.x-1]*yp[t.y]*zp[t.z];
                        if (t.y) fy += in[k]*xp[t.x]*yp[t.y-1]*zp[t.z];
                        if (t.z) fz += in[k]*xp[t.x]*yp[t.y]*zp[t.z-1];
                    }
                    ax[j] += fx/nd.r; ay[j] += fy/nd.r; az[j] += fz/nd.r;
                }
            }
        }
        log(fast ? "downward (budget fallback)" : "downward");
    }

    void point_force(int ti, double opening, int p, double &fx, double &fy, double &fz) {
        fx = fy = fz = 0;
        int stack[256], sp = 0;
        stack[sp++] = 0;
        alignas(64) double deriv[CMAX];
        while (sp) {
            int ni = stack[--sp];
            const auto &s = nodes[ni];
            double dx = x[ti]-s.x, dy = y[ti]-s.y, dz = z[ti]-s.z;
            double q = dx*dx+dy*dy+dz*dz+EPS2;
            if (s.r*s.r < opening*opening*q) {
                double inv = 1/std::sqrt(q);
                double u[3] = {dx*inv, dy*inv, dz*inv};
                kernel_scalar(p, u[0], u[1], u[2], deriv);
                double scale[PMAX+1];
                scale[0] = 1;
                for (int k = 1; k < p; ++k) scale[k] = scale[k-1]*(-s.r*inv);
                const double *m = moments+size_t(ni)*mstride;
                double sx = m[0]*deriv[index[1][0][0]];
                double sy = m[0]*deriv[index[0][1][0]];
                double sz = m[0]*deriv[index[0][0][1]];
                for (int k = 4; k < count[p-1]; ++k) {
                    const auto &t = terms[k];
                    double f = m[k]*scale[t.degree];
                    sx += f*deriv[index[t.x+1][t.y][t.z]];
                    sy += f*deriv[index[t.x][t.y+1][t.z]];
                    sz += f*deriv[index[t.x][t.y][t.z+1]];
                }
                fx += sx*inv*inv; fy += sy*inv*inv; fz += sz*inv*inv;
            } else if (s.left >= 0) {
                stack[sp++] = s.left; stack[sp++] = s.right;
            } else {
                __m256d tx = _mm256_set1_pd(x[ti]), ty = _mm256_set1_pd(y[ti]),
                        tz = _mm256_set1_pd(z[ti]);
                __m256d sx = _mm256_setzero_pd(), sy = sx, sz = sx;
                int j = s.begin;
                for (; j+3 < s.end; j += 4) {
                    __m256d rx = _mm256_sub_pd(_mm256_loadu_pd(x.data()+j), tx);
                    __m256d ry = _mm256_sub_pd(_mm256_loadu_pd(y.data()+j), ty);
                    __m256d rz = _mm256_sub_pd(_mm256_loadu_pd(z.data()+j), tz);
                    __m256d q = _mm256_fmadd_pd(rx, rx, _mm256_fmadd_pd(ry, ry,
                        _mm256_fmadd_pd(rz, rz, _mm256_set1_pd(EPS2))));
                    __m256d inv = _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(q));
                    __m256d f = _mm256_mul_pd(_mm256_loadu_pd(mass.data()+j),
                        _mm256_mul_pd(inv, _mm256_mul_pd(inv, inv)));
                    sx = _mm256_fmadd_pd(rx, f, sx);
                    sy = _mm256_fmadd_pd(ry, f, sy);
                    sz = _mm256_fmadd_pd(rz, f, sz);
                }
                fx += horizontal(sx); fy += horizontal(sy); fz += horizontal(sz);
                for (; j < s.end; ++j) {
                    double rx = x[j]-x[ti], ry = y[j]-y[ti], rz = z[j]-z[ti];
                    double q = rx*rx+ry*ry+rz*rz+EPS2;
                    double f = mass[j]/(q*std::sqrt(q));
                    fx += rx*f; fy += ry*f; fz += rz*f;
                }
            }
        }
    }

    void point_force4(const int *ti, double opening, int p) {
        __m256d tx = _mm256_set_pd(x[ti[3]], x[ti[2]], x[ti[1]], x[ti[0]]);
        __m256d ty = _mm256_set_pd(y[ti[3]], y[ti[2]], y[ti[1]], y[ti[0]]);
        __m256d tz = _mm256_set_pd(z[ti[3]], z[ti[2]], z[ti[1]], z[ti[0]]);
        __m256d fx = _mm256_setzero_pd(), fy = fx, fz = fx;
        __m256d one = _mm256_set1_pd(1), soft = _mm256_set1_pd(EPS2);
        int stack[256], sp = 0;
        stack[sp++] = 0;
        alignas(64) __m256d deriv[CMAX];
        while (sp) {
            int ni = stack[--sp];
            const auto &s = nodes[ni];
            __m256d dx = _mm256_sub_pd(tx, _mm256_set1_pd(s.x));
            __m256d dy = _mm256_sub_pd(ty, _mm256_set1_pd(s.y));
            __m256d dz = _mm256_sub_pd(tz, _mm256_set1_pd(s.z));
            __m256d q = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, soft)));
            __m256d cmp = _mm256_cmp_pd(q, _mm256_set1_pd(s.r*s.r/(opening*opening)), _CMP_GT_OQ);
            if (_mm256_movemask_pd(cmp) == 15) {
                __m256d inv = _mm256_div_pd(one, _mm256_sqrt_pd(q));
                __m256d ux = _mm256_mul_pd(dx, inv), uy = _mm256_mul_pd(dy, inv),
                        uz = _mm256_mul_pd(dz, inv);
                kernel4(p, ux, uy, uz, deriv);
                __m256d scale[PMAX+1];
                scale[0] = one;
                __m256d ratio = _mm256_mul_pd(_mm256_set1_pd(-s.r), inv);
                for (int k = 1; k < p; ++k) scale[k] = _mm256_mul_pd(scale[k-1], ratio);
                const double *m = moments+size_t(ni)*mstride;
                __m256d sx = _mm256_mul_pd(_mm256_set1_pd(m[0]), deriv[index[1][0][0]]);
                __m256d sy = _mm256_mul_pd(_mm256_set1_pd(m[0]), deriv[index[0][1][0]]);
                __m256d sz = _mm256_mul_pd(_mm256_set1_pd(m[0]), deriv[index[0][0][1]]);
                for (int k = 4; k < count[p-1]; ++k) {
                    const auto &t = terms[k];
                    __m256d f = _mm256_mul_pd(_mm256_set1_pd(m[k]), scale[t.degree]);
                    sx = _mm256_fmadd_pd(f, deriv[index[t.x+1][t.y][t.z]], sx);
                    sy = _mm256_fmadd_pd(f, deriv[index[t.x][t.y+1][t.z]], sy);
                    sz = _mm256_fmadd_pd(f, deriv[index[t.x][t.y][t.z+1]], sz);
                }
                __m256d inv2 = _mm256_mul_pd(inv, inv);
                fx = _mm256_fmadd_pd(sx, inv2, fx);
                fy = _mm256_fmadd_pd(sy, inv2, fy);
                fz = _mm256_fmadd_pd(sz, inv2, fz);
            } else if (s.left >= 0) {
                stack[sp++] = s.left; stack[sp++] = s.right;
            } else {
                for (int j = s.begin; j < s.end; ++j) {
                    __m256d rx = _mm256_sub_pd(_mm256_set1_pd(x[j]), tx);
                    __m256d ry = _mm256_sub_pd(_mm256_set1_pd(y[j]), ty);
                    __m256d rz = _mm256_sub_pd(_mm256_set1_pd(z[j]), tz);
                    __m256d q = _mm256_fmadd_pd(rx, rx, _mm256_fmadd_pd(ry, ry,
                        _mm256_fmadd_pd(rz, rz, soft)));
                    __m256d inv = _mm256_div_pd(one, _mm256_sqrt_pd(q));
                    __m256d f = _mm256_mul_pd(_mm256_set1_pd(mass[j]),
                        _mm256_mul_pd(inv, _mm256_mul_pd(inv, inv)));
                    fx = _mm256_fmadd_pd(rx, f, fx);
                    fy = _mm256_fmadd_pd(ry, f, fy);
                    fz = _mm256_fmadd_pd(rz, f, fz);
                }
            }
        }
        alignas(32) double outx[4], outy[4], outz[4];
        _mm256_store_pd(outx, fx); _mm256_store_pd(outy, fy); _mm256_store_pd(outz, fz);
        for (int k = 0; k < 4; ++k) {
            ax[ti[k]] = outx[k]; ay[ti[k]] = outy[k]; az[ti[k]] = outz[k];
        }
    }

    void correct() {
        if (correction_targets.empty()) return;
        double opening = point_opening;
        if (const char *s = std::getenv("GALAXY_TC_THETA")) opening = std::atof(s);
        int p = std::min(moment_order+1, 8);
        if (const char *s = std::getenv("GALAXY_TC_ORDER")) p = std::min(moment_order+1, std::atoi(s));
        for (size_t j = 0; j < correction_targets.size(); j += 4) {
            int ti[4];
            for (int k = 0; k < 4; ++k)
                ti[k] = correction_targets[std::min(j+k, correction_targets.size()-1)];
            double remaining = budget-cpu_time()-write_margin;
            if (remaining < 0.005*budget) {
                const auto &root = nodes[0];
                for (int k = 0; k < 4; ++k) {
                    int i = ti[k];
                    double dx = root.x-x[i], dy = root.y-y[i], dz = root.z-z[i];
                    double q = dx*dx+dy*dy+dz*dz+EPS2;
                    double f = root.mass/(q*std::sqrt(q));
                    ax[i] = dx*f; ay[i] = dy*f; az[i] = dz*f;
                }
            } else {
                point_force4(ti, remaining < 0.03*budget ? 0.85 : opening,
                             remaining < 0.03*budget ? 2 : p);
            }
        }
        log("correct");
    }

    bool write(const char *path) {
        std::vector<double> out(3*n);
        for (size_t i = 0; i < n; ++i) {
            size_t j = ids[i];
            out[3*j] = ax[i]; out[3*j+1] = ay[i]; out[3*j+2] = az[i];
        }
        FILE *f = std::fopen(path, "wb");
        if (!f) return false;
        bool ok = std::fwrite(out.data(), 24, n, f) == n;
        ok = std::fclose(f) == 0 && ok;
        log("write");
        return ok;
    }
};

#include "harmonic.inc"

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    double limit = 2.0;
    if (const char *s = std::getenv("GALAXY_CPU_LIMIT_S")) limit = std::atof(s);
    struct stat input_stat;
    if (stat(argv[1], &input_stat) || input_stat.st_size%32) return 2;
    size_t input_n = input_stat.st_size/32;
    double scaled_budget = limit*65536.0/std::max(size_t(1), input_n);
    int p = input_n >= 1048576 ? 12 : 10;
    int leaf = input_n >= 2097152 ? 128 : 64;
    int default_hp = scaled_budget < 1.2 ? 16 : scaled_budget > 3.0 ? 20 : 18;
    bool autotune = !std::getenv("GALAXY_THETA") || std::getenv("GALAXY_AUTOTUNE");
    auto set_default = [](const char *key, const char *value) {
        if (!std::getenv(key)) setenv(key, value, 0);
    };
    set_default("GALAXY_HORDER", std::to_string(default_hp).c_str());
    set_default("GALAXY_DMAX", std::to_string(p).c_str());
    double correction_fraction = 0.1*std::min(1.0, std::sqrt(65536.0/std::max(size_t(1), input_n)));
    set_default("GALAXY_CORRECT", std::to_string(correction_fraction).c_str());
    set_default("GALAXY_SOFT_CUTOFF", input_n >= 1048576 ? "0.2" : "0.5");
    set_default("GALAXY_SOFT_THETA", input_n >= 1048576 ? "0.45" : "0.3");
    if (input_n <= 1024) {
        p = 2; leaf = std::max(1, int(input_n));
        setenv("GALAXY_HORDER", "0", 1);
        setenv("GALAXY_DMAX", "2", 1);
        setenv("GALAXY_CORRECT", "0", 1);
        autotune = false;
    }
    double theta = 0.65;
    if (const char *s = std::getenv("GALAXY_ORDER")) p = std::atoi(s);
    if (const char *s = std::getenv("GALAXY_LEAF")) leaf = std::atoi(s);
    if (const char *s = std::getenv("GALAXY_THETA")) theta = std::atof(s);
    p = std::max(2, std::min(PMAX, p));
    Solver solver(p, leaf, theta, limit);
    if (!solver.read(argv[1])) return 2;
    if (!solver.n) return solver.write(argv[2]) ? 0 : 3;
    solver.build();
    double upward_start = cpu_time();
    solver.upward();
    int hp = 0;
    if (const char *s = std::getenv("GALAXY_HORDER")) hp = std::max(0, std::min(PMAX, std::atoi(s)));
    while (hp > 10) {
        size_t hstride = (((hp+1)*(hp+1))+7)&~size_t(7);
        double memory = double(solver.nodes.size())*(solver.stride+solver.mstride+2*hstride)*8
                      + double(solver.n)*125;
        if (memory < 1.75*1024*1024*1024) break;
        hp -= 2;
    }
    Harmonic *harmonic = hp ? new Harmonic(hp, solver) : nullptr;
    if (harmonic) harmonic->upward(solver);
    double upward_seconds = cpu_time()-upward_start;
    solver.make_pairs();
    if (autotune) {
        solver.calibrate(harmonic, upward_seconds);
        solver.tune_geometry();
    } else {
        solver.write_margin = std::max(0.015*limit, .01)+solver.n*4.e-8;
        solver.finish_reserve = 1.5*upward_seconds+solver.write_margin;
    }
    solver.direct();
    solver.far_field();
    if (harmonic) harmonic->far_field(solver);
    std::vector<Pair>().swap(solver.far_pairs);
    std::vector<std::vector<int>>().swap(solver.near);
    solver.downward();
    if (harmonic) harmonic->downward(solver);
    delete harmonic;
    harmonic = nullptr;
    std::free(solver.locals);
    solver.locals = nullptr;
    solver.correct();
    bool ok = solver.write(argv[2]);
    return ok ? 0 : 3;
}
