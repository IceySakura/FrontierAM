#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

extern "C" {
typedef struct z_stream_s {
    const unsigned char* next_in;
    unsigned int avail_in;
    unsigned long total_in;
    unsigned char* next_out;
    unsigned int avail_out;
    unsigned long total_out;
    const char* msg;
    void* state;
    void* (*zalloc)(void*, unsigned int, unsigned int);
    void (*zfree)(void*, void*);
    void* opaque;
    int data_type;
    unsigned long adler;
    unsigned long reserved;
} z_stream;
int inflateInit2_(z_stream*, int, const char*, int);
int inflate(z_stream*, int);
int inflateEnd(z_stream*);
unsigned long crc32(unsigned long, const unsigned char*, unsigned int);

void scipy_openblas_set_num_threads(int);
void scipy_dpotrf_(const char*, const int*, double*, const int*, int*);
void scipy_dpotrs_(const char*, const int*, const int*, double*, const int*, double*, const int*, int*);
void scipy_dgemm_(const char*, const char*, const int*, const int*, const int*, const double*,
                  const double*, const int*, const double*, const int*, const double*, double*, const int*);
void scipy_dgemv_(const char*, const int*, const int*, const double*, const double*, const int*,
                  const double*, const int*, const double*, double*, const int*);
void scipy_dsyev_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*);
void scipy_dsygv_(const int*, const char*, const char*, const int*, double*, const int*, double*, const int*,
                  double*, double*, const int*, int*);
}

static double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static inline double inf_norm(const double* v, int n) {
    double r = 0;
    for (int i = 0; i < n; ++i) r = std::max(r, std::fabs(v[i]));
    return r;
}
static inline double dot(const double* a, const double* b, int n) {
    double r = 0;
    for (int i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
}
static inline uint16_t rd16(const unsigned char* p) { return p[0] | (uint16_t(p[1]) << 8); }
static inline uint32_t rd32(const unsigned char* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static inline void wr16(std::vector<unsigned char>& v, uint16_t x) { v.push_back(x & 255); v.push_back(x >> 8); }
static inline void wr32(std::vector<unsigned char>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back((x >> (8 * i)) & 255);
}

struct NpyArray {
    std::string descr;
    std::vector<int64_t> shape;
    bool fortran = false;
    const unsigned char* data = nullptr;
    size_t nbytes = 0;
};

static std::map<std::string, std::vector<unsigned char>> read_npz(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input");
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 22) throw std::runtime_error("bad npz");
    size_t eocd = buf.size();
    for (size_t i = buf.size() - 22; i + 22 <= buf.size(); --i) {
        if (rd32(buf.data() + i) == 0x06054b50) { eocd = i; break; }
        if (i == 0) break;
    }
    if (eocd == buf.size()) throw std::runtime_error("npz eocd missing");
    uint16_t count = rd16(buf.data() + eocd + 10);
    uint32_t cdoff = rd32(buf.data() + eocd + 16);
    size_t p = cdoff;
    std::map<std::string, std::vector<unsigned char>> out;
    for (int e = 0; e < count; ++e) {
        if (p + 46 > buf.size() || rd32(buf.data() + p) != 0x02014b50) throw std::runtime_error("bad central dir");
        uint16_t method = rd16(buf.data() + p + 10);
        uint32_t csize = rd32(buf.data() + p + 20);
        uint32_t usize = rd32(buf.data() + p + 24);
        uint16_t flen = rd16(buf.data() + p + 28);
        uint16_t xlen = rd16(buf.data() + p + 30);
        uint16_t clen = rd16(buf.data() + p + 32);
        uint32_t loff = rd32(buf.data() + p + 42);
        std::string name((const char*)buf.data() + p + 46, flen);
        p += 46 + flen + xlen + clen;
        if (loff + 30 > buf.size()) throw std::runtime_error("bad local header");
        uint16_t lflen = rd16(buf.data() + loff + 26);
        uint16_t lxlen = rd16(buf.data() + loff + 28);
        size_t doff = loff + 30 + lflen + lxlen;
        if (doff + csize > buf.size()) throw std::runtime_error("bad compressed data");
        std::vector<unsigned char> data(usize);
        if (method == 0) {
            if (csize != usize) throw std::runtime_error("stored size mismatch");
            memcpy(data.data(), buf.data() + doff, usize);
        } else if (method == 8) {
            z_stream zs{};
            if (inflateInit2_(&zs, -15, "1.2.13", sizeof(zs)) != 0) throw std::runtime_error("inflate init");
            zs.next_in = buf.data() + doff;
            zs.avail_in = csize;
            zs.next_out = data.data();
            zs.avail_out = usize;
            int rc = inflate(&zs, 4);
            inflateEnd(&zs);
            if (rc != 1) throw std::runtime_error("inflate failed");
        } else throw std::runtime_error("unsupported zip method");
        out[name] = std::move(data);
    }
    return out;
}

static NpyArray parse_npy(const std::vector<unsigned char>& v) {
    if (v.size() < 10 || memcmp(v.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error("bad npy");
    int major = v[6];
    size_t hlen, doff;
    if (major == 1) { hlen = rd16(v.data() + 8); doff = 10 + hlen; }
    else { hlen = rd32(v.data() + 8); doff = 12 + hlen; }
    if (doff > v.size()) throw std::runtime_error("bad npy header");
    std::string h((const char*)v.data() + (major == 1 ? 10 : 12), hlen);
    NpyArray a;
    size_t d = h.find("'descr'");
    if (d == std::string::npos) throw std::runtime_error("descr missing");
    d = h.find('\'', d + 7);
    size_t e = h.find('\'', d + 1);
    a.descr = h.substr(d + 1, e - d - 1);
    a.fortran = h.find("'fortran_order': True") != std::string::npos;
    size_t s = h.find("'shape'");
    s = h.find('(', s);
    size_t se = h.find(')', s);
    std::string sh = h.substr(s + 1, se - s - 1);
    size_t pos = 0;
    while (pos < sh.size()) {
        while (pos < sh.size() && (sh[pos] == ' ' || sh[pos] == ',')) ++pos;
        if (pos >= sh.size()) break;
        size_t end;
        long long x = std::stoll(sh.substr(pos), &end);
        a.shape.push_back(x);
        pos += end;
    }
    a.data = v.data() + doff;
    a.nbytes = v.size() - doff;
    return a;
}

static std::vector<double> get_f64(const std::map<std::string, std::vector<unsigned char>>& npz, const std::string& key) {
    auto it = npz.find(key + ".npy");
    if (it == npz.end()) throw std::runtime_error("missing " + key);
    NpyArray a = parse_npy(it->second);
    if (a.descr != "<f8") throw std::runtime_error("expected float64 " + key);
    size_t n = 1;
    for (auto x : a.shape) n *= x;
    std::vector<double> r(n);
    memcpy(r.data(), a.data, n * 8);
    return r;
}
static std::vector<int64_t> get_int(const std::map<std::string, std::vector<unsigned char>>& npz, const std::string& key) {
    auto it = npz.find(key + ".npy");
    if (it == npz.end()) throw std::runtime_error("missing " + key);
    NpyArray a = parse_npy(it->second);
    size_t n = 1;
    for (auto x : a.shape) n *= x;
    std::vector<int64_t> r(n);
    if (a.descr == "<i4") {
        const int32_t* p = (const int32_t*)a.data;
        for (size_t i = 0; i < n; ++i) r[i] = p[i];
    } else if (a.descr == "<i8") {
        memcpy(r.data(), a.data, n * 8);
    } else if (a.descr == "<u4") {
        const uint32_t* p = (const uint32_t*)a.data;
        for (size_t i = 0; i < n; ++i) r[i] = p[i];
    } else throw std::runtime_error("expected int " + key);
    return r;
}
static std::string get_meta(const std::map<std::string, std::vector<unsigned char>>& npz) {
    auto it = npz.find("metadata_json.npy");
    if (it == npz.end()) throw std::runtime_error("missing metadata");
    NpyArray a = parse_npy(it->second);
    if (a.descr.rfind("<U", 0) != 0) throw std::runtime_error("metadata not unicode");
    size_t n = a.nbytes / 4;
    std::string r;
    const unsigned char* p = a.data;
    for (size_t i = 0; i < n; ++i) {
        uint32_t c = rd32(p + 4 * i);
        if (c < 0x80) r.push_back(char(c));
        else if (c < 0x800) { r.push_back(char(0xC0 | (c >> 6))); r.push_back(char(0x80 | (c & 63))); }
        else if (c < 0x10000) { r.push_back(char(0xE0 | (c >> 12))); r.push_back(char(0x80 | ((c >> 6) & 63))); r.push_back(char(0x80 | (c & 63))); }
        else { r.push_back(char(0xF0 | (c >> 18))); r.push_back(char(0x80 | ((c >> 12) & 63))); r.push_back(char(0x80 | ((c >> 6) & 63))); r.push_back(char(0x80 | (c & 63))); }
    }
    return r;
}

enum ConeType { NONNEG = 0, SOC = 1, PSD = 2, EXP = 3, ZERO = 4 };
struct Cone { ConeType type; int dim; int start; int width; };
static std::vector<Cone> parse_cones(const std::string& meta) {
    size_t c = meta.find("\"cones\"");
    if (c == std::string::npos) throw std::runtime_error("cones missing");
    c = meta.find('[', c);
    size_t end = meta.find(']', c);
    std::vector<Cone> r;
    int start = 0;
    size_t p = c + 1;
    while (p < end) {
        size_t ob = meta.find('{', p);
        if (ob == std::string::npos || ob >= end) break;
        size_t oe = meta.find('}', ob);
        std::string obj = meta.substr(ob, oe - ob + 1);
        auto getstr = [&](const std::string& key) {
            size_t k = obj.find("\"" + key + "\"");
            k = obj.find(':', k);
            k = obj.find('"', k);
            size_t e = obj.find('"', k + 1);
            return obj.substr(k + 1, e - k - 1);
        };
        auto getint = [&](const std::string& key) {
            size_t k = obj.find("\"" + key + "\"");
            k = obj.find(':', k);
            return std::atoi(obj.c_str() + k + 1);
        };
        std::string t = getstr("type");
        int dim = getint("dim");
        ConeType ct = t == "zero" ? ZERO : t == "nonnegative" ? NONNEG : t == "soc" ? SOC : t == "psd" ? PSD : EXP;
        int width = ct == PSD ? dim * (dim + 1) / 2 : dim;
        r.push_back({ct, dim, start, width});
        start += width;
        p = oe + 1;
    }
    return r;
}
static double get_offset(const std::string& meta) {
    size_t k = meta.find("\"objective_offset\"");
    if (k == std::string::npos) return 0;
    k = meta.find(':', k);
    return std::strtod(meta.c_str() + k + 1, nullptr);
}

struct CSC {
    int rows = 0, cols = 0;
    std::vector<int> indptr, indices;
    std::vector<double> data;
    void build(int r, int c, const std::vector<int64_t>& ip, const std::vector<int64_t>& ix, const std::vector<double>& v) {
        rows = r; cols = c; indptr.assign(c + 1, 0);
        std::vector<std::vector<std::pair<int,double>>> col(c);
        for (int j = 0; j < c; ++j) {
            for (int64_t k = ip[j]; k < ip[j + 1]; ++k) col[j].push_back({(int)ix[k], v[k]});
            std::sort(col[j].begin(), col[j].end());
            for (auto& pr : col[j]) {
                if (!indices.empty() && indptr[j] < (int)indices.size() && indices.back() == pr.first && (int)indices.size() > indptr[j])
                    data.back() += pr.second;
                else { indices.push_back(pr.first); data.push_back(pr.second); }
            }
            indptr[j + 1] = indices.size();
        }
    }
    void matvec(const double* x, double* y) const {
        std::fill(y, y + rows, 0.0);
        for (int j = 0; j < cols; ++j) {
            double xj = x[j];
            for (int k = indptr[j]; k < indptr[j + 1]; ++k) y[indices[k]] += data[k] * xj;
        }
    }
    void tmatvec(const double* x, double* y) const {
        std::fill(y, y + cols, 0.0);
        for (int j = 0; j < cols; ++j) {
            double s = 0;
            for (int k = indptr[j]; k < indptr[j + 1]; ++k) s += data[k] * x[indices[k]];
            y[j] = s;
        }
    }
};
struct CSR {
    int rows = 0, cols = 0;
    std::vector<int> indptr, indices;
    std::vector<double> data;
    void from_csc(const CSC& a) {
        rows = a.rows; cols = a.cols;
        std::vector<std::vector<std::pair<int,double>>> row(rows);
        for (int j = 0; j < a.cols; ++j)
            for (int k = a.indptr[j]; k < a.indptr[j + 1]; ++k) row[a.indices[k]].push_back({j, a.data[k]});
        indptr.assign(rows + 1, 0);
        for (int i = 0; i < rows; ++i) {
            std::sort(row[i].begin(), row[i].end());
            for (auto& pr : row[i]) { indices.push_back(pr.first); data.push_back(pr.second); }
            indptr[i + 1] = indices.size();
        }
    }
    void matvec(const double* x, double* y) const {
        for (int i = 0; i < rows; ++i) {
            double s = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k) s += data[k] * x[indices[k]];
            y[i] = s;
        }
    }
};

static void smat(const double* v, int d, double* S) {
    const double r2 = std::sqrt(2.0);
    int pos = 0;
    for (int j = 0; j < d; ++j) {
        for (int i = 0; i <= j; ++i) {
            double x = v[pos++];
            if (i != j) x /= r2;
            S[i + j * d] = x;
            S[j + i * d] = x;
        }
    }
}
static void svec(const double* S, int d, double* v) {
    const double r2 = std::sqrt(2.0);
    int pos = 0;
    for (int j = 0; j < d; ++j)
        for (int i = 0; i <= j; ++i) v[pos++] = S[i + j * d] * (i == j ? 1.0 : r2);
}
static double min_eig(const double* S, int d) {
    if (d == 1) return S[0];
    std::vector<double> A(S, S + d * d), w(d), work(1);
    int lwork = -1, info;
    scipy_dsyev_("N", "L", &d, A.data(), &d, w.data(), work.data(), &lwork, &info);
    lwork = (int)work[0];
    work.resize(lwork);
    scipy_dsyev_("N", "L", &d, A.data(), &d, w.data(), work.data(), &lwork, &info);
    if (info != 0) return -1e100;
    return w[0];
}

static double exp_defect(double a, double b, double c) {
    double B = std::max(std::fabs(b), std::max(std::max(a, 0.0), std::max(-c, 0.0)));
    if (b > 0 && c > 0) {
        double interior = std::max(0.0, a - b * (std::log(c) - std::log(b)));
        return std::min(B, interior);
    }
    return B;
}
static double cone_defect(const double* v, const std::vector<Cone>& cones, bool dual) {
    double worst = 0;
    for (auto& c : cones) {
        const double* b = v + c.start;
        double sc = std::max(1.0, inf_norm(b, c.width));
        std::vector<double> u(c.width);
        for (int i = 0; i < c.width; ++i) u[i] = b[i] / sc;
        double def = 0;
        if (c.type == ZERO) def = dual ? 0 : inf_norm(u.data(), c.width);
        else if (c.type == NONNEG) { double m = 0; for (double x : u) m = std::min(m, x); def = std::max(0.0, -m); }
        else if (c.type == SOC) { double nrm = 0; for (int i = 1; i < c.width; ++i) nrm += u[i] * u[i]; def = std::max(0.0, std::sqrt(nrm) - u[0]); }
        else if (c.type == EXP) {
            if (dual) def = exp_defect(-u[1], -u[0], std::exp(1.0) * u[2]);
            else def = exp_defect(u[0], u[1], u[2]);
        } else {
            std::vector<double> S(c.dim * c.dim);
            smat(u.data(), c.dim, S.data());
            def = std::max(0.0, -min_eig(S.data(), c.dim));
        }
        worst = std::max(worst, def);
        if (def > 1e-3 && getenv("CONIC_DEBUG")) fprintf(stderr, "cone defect type %d start %d dim %d def %.3e val0 %.3e\n", c.type, c.start, c.dim, def, b[0]);
    }
    return worst;
}

struct Result { std::string status; std::vector<double> x, s, z; };

struct EqPresolve {
    bool active = false;
    int n = 0, k = 0, e = 0, rank = 0;
    std::vector<double> T, t;
    std::vector<int> pivot_col, zero_rows;
    std::vector<double> Lbasis;
    CSR AC;
    std::vector<int> crows;
    std::vector<Cone> conesC;
    CSC P, A;
    std::vector<double> q, b;
    std::vector<Cone> cones;
};

static bool eliminate_equalities(const CSC& P, const CSR& Acsr, const std::vector<double>& q,
                                 const std::vector<double>& b, const std::vector<Cone>& cones,
                                 EqPresolve& ep, Result& cert) {
    int n = q.size();
    std::vector<int> erows;
    for (auto& c : cones) if (c.type == ZERO)
        for (int r = c.start; r < c.start + c.width; ++r) erows.push_back(r);
    int e = erows.size();
    if (e == 0) return false;
    if ((int64_t)e * n > 3000000) return false;
    ep.e = e; ep.n = n; ep.zero_rows = erows;
    double norm = 1;
    for (int r : erows) for (int k = Acsr.indptr[r]; k < Acsr.indptr[r + 1]; ++k) norm = std::max(norm, std::fabs(Acsr.data[k]));
    double tol = 1e-9 * norm;
    std::vector<double> Aug((size_t)e * (n + 1), 0.0), L((size_t)e * e, 0.0);
    for (int i = 0; i < e; ++i) {
        int r = erows[i];
        for (int k = Acsr.indptr[r]; k < Acsr.indptr[r + 1]; ++k) Aug[(size_t)i * (n + 1) + Acsr.indices[k]] = Acsr.data[k];
        Aug[(size_t)i * (n + 1) + n] = b[r];
        L[(size_t)i * e + i] = 1;
    }
    int rank = 0;
    std::vector<int> pivots;
    for (int j = 0; j < n && rank < e; ++j) {
        int best = rank;
        double mx = std::fabs(Aug[(size_t)rank * (n + 1) + j]);
        for (int i = rank + 1; i < e; ++i) {
            double v = std::fabs(Aug[(size_t)i * (n + 1) + j]);
            if (v > mx) { mx = v; best = i; }
        }
        if (mx <= tol) continue;
        if (best != rank) {
            for (int c = 0; c <= n; ++c) std::swap(Aug[(size_t)rank * (n + 1) + c], Aug[(size_t)best * (n + 1) + c]);
            for (int c = 0; c < e; ++c) std::swap(L[(size_t)rank * e + c], L[(size_t)best * e + c]);
        }
        double pv = Aug[(size_t)rank * (n + 1) + j];
        for (int c = 0; c <= n; ++c) Aug[(size_t)rank * (n + 1) + c] /= pv;
        for (int c = 0; c < e; ++c) L[(size_t)rank * e + c] /= pv;
        for (int i = 0; i < e; ++i) if (i != rank) {
            double f = Aug[(size_t)i * (n + 1) + j];
            if (f == 0) continue;
            for (int c = 0; c <= n; ++c) Aug[(size_t)i * (n + 1) + c] -= f * Aug[(size_t)rank * (n + 1) + c];
            for (int c = 0; c < e; ++c) L[(size_t)i * e + c] -= f * L[(size_t)rank * e + c];
        }
        pivots.push_back(j);
        ++rank;
    }
    for (int i = rank; i < e; ++i) {
        double rhs = Aug[(size_t)i * (n + 1) + n];
        if (std::fabs(rhs) > 1e-7 * norm) {
            std::vector<double> z(b.size(), 0.0);
            double sign = rhs < 0 ? 1 : -1;
            for (int j = 0; j < e; ++j) z[erows[j]] = sign * L[(size_t)i * e + j];
            cert.status = "primal_infeasible"; cert.z = z; cert.x.assign(n, 0); cert.s.assign(b.size(), 0);
            return true;
        }
    }
    ep.rank = rank;
    std::vector<int> freecols;
    std::vector<bool> ispivot(n, false);
    for (int p : pivots) ispivot[p] = true;
    for (int j = 0; j < n; ++j) if (!ispivot[j]) freecols.push_back(j);
    int k = freecols.size();
    ep.k = k;
    ep.T.assign((size_t)n * k, 0.0); ep.t.assign(n, 0.0);
    for (int a = 0; a < k; ++a) ep.T[(size_t)freecols[a] * k + a] = 1;
    for (int i = 0; i < rank; ++i) {
        int p = pivots[i];
        ep.t[p] = Aug[(size_t)i * (n + 1) + n];
        for (int a = 0; a < k; ++a) ep.T[(size_t)p * k + a] = -Aug[(size_t)i * (n + 1) + freecols[a]];
    }
    ep.pivot_col = pivots;
    ep.Lbasis.assign((size_t)rank * e, 0.0);
    for (int i = 0; i < rank; ++i)
        for (int j = 0; j < e; ++j) ep.Lbasis[(size_t)i * e + j] = L[(size_t)i * e + j];
    std::vector<int> crows;
    int start = 0;
    for (auto& c : cones) if (c.type != ZERO) {
        ep.conesC.push_back({c.type, c.dim, start, c.width});
        start += c.width;
        for (int r = c.start; r < c.start + c.width; ++r) crows.push_back(r);
    }
    ep.cones = ep.conesC;
    int mc = crows.size();
    ep.crows = crows;
    ep.AC.rows = mc; ep.AC.cols = n; ep.AC.indptr.assign(mc + 1, 0);
    for (int i = 0; i < mc; ++i) {
        int r = crows[i];
        for (int k2 = Acsr.indptr[r]; k2 < Acsr.indptr[r + 1]; ++k2) {
            ep.AC.indices.push_back(Acsr.indices[k2]); ep.AC.data.push_back(Acsr.data[k2]);
        }
        ep.AC.indptr[i + 1] = ep.AC.indices.size();
    }
    std::vector<double> AR((size_t)mc * k, 0.0), br(mc), qr(k, 0.0), PR((size_t)k * k, 0.0);
    for (int i = 0; i < mc; ++i) {
        int r = crows[i];
        br[i] = b[r];
        for (int k2 = Acsr.indptr[r]; k2 < Acsr.indptr[r + 1]; ++k2) {
            int j = Acsr.indices[k2]; double v = Acsr.data[k2];
            br[i] -= v * ep.t[j];
            for (int a = 0; a < k; ++a) AR[(size_t)i * k + a] += v * ep.T[(size_t)j * k + a];
        }
    }
    for (int j = 0; j < n; ++j)
        for (int k2 = P.indptr[j]; k2 < P.indptr[j + 1]; ++k2) {
            int i = P.indices[k2]; double v = P.data[k2];
            for (int a = 0; a < k; ++a) {
                double tia = ep.T[(size_t)i * k + a];
                if (tia == 0) continue;
                for (int c = 0; c < k; ++c) PR[(size_t)a * k + c] += tia * v * ep.T[(size_t)j * k + c];
            }
        }
    std::vector<double> Pt(n, 0.0);
    P.matvec(ep.t.data(), Pt.data());
    for (int a = 0; a < k; ++a)
        for (int j = 0; j < n; ++j) qr[a] += ep.T[(size_t)j * k + a] * (q[j] + Pt[j]);
    ep.q = qr; ep.b = br;
    ep.A.rows = mc; ep.A.cols = k; ep.A.indptr.assign(k + 1, 0);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < mc; ++i) if (AR[(size_t)i * k + j] != 0) {
            ep.A.indices.push_back(i); ep.A.data.push_back(AR[(size_t)i * k + j]);
        }
        ep.A.indptr[j + 1] = ep.A.indices.size();
    }
    ep.P.rows = k; ep.P.cols = k; ep.P.indptr.assign(k + 1, 0);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) if (PR[(size_t)i * k + j] != 0) {
            ep.P.indices.push_back(i); ep.P.data.push_back(PR[(size_t)i * k + j]);
        }
        ep.P.indptr[j + 1] = ep.P.indices.size();
    }
    ep.active = true;
    return false;
}

static std::vector<double> lift_z(const EqPresolve& ep, const std::vector<double>& zc, int morig) {
    std::vector<double> z(morig, 0.0);
    for (size_t i = 0; i < ep.crows.size(); ++i) z[ep.crows[i]] = zc[i];
    std::vector<double> v(ep.n, 0.0);
    for (int i = 0; i < ep.AC.rows; ++i)
        for (int k = ep.AC.indptr[i]; k < ep.AC.indptr[i + 1]; ++k)
            v[ep.AC.indices[k]] -= ep.AC.data[k] * zc[i];
    std::vector<double> zb(ep.rank, 0.0);
    for (int i = 0; i < ep.rank; ++i) zb[i] = v[ep.pivot_col[i]];
    for (int j = 0; j < ep.e; ++j) {
        double s = 0;
        for (int i = 0; i < ep.rank; ++i) s += ep.Lbasis[(size_t)i * ep.e + j] * zb[i];
        z[ep.zero_rows[j]] = s;
    }
    return z;
}
static std::vector<double> lift_z_solved(const EqPresolve& ep, const CSC& P, const std::vector<double>& q,
                                         const std::vector<double>& x, const std::vector<double>& zc, int morig) {
    std::vector<double> z(morig, 0.0);
    for (size_t i = 0; i < ep.crows.size(); ++i) z[ep.crows[i]] = zc[i];
    std::vector<double> v(ep.n, 0.0), Px(ep.n);
    P.matvec(x.data(), Px.data());
    for (int i = 0; i < ep.n; ++i) v[i] = -(Px[i] + q[i]);
    for (int i = 0; i < ep.AC.rows; ++i)
        for (int k = ep.AC.indptr[i]; k < ep.AC.indptr[i + 1]; ++k)
            v[ep.AC.indices[k]] -= ep.AC.data[k] * zc[i];
    std::vector<double> zb(ep.rank, 0.0);
    for (int i = 0; i < ep.rank; ++i) zb[i] = v[ep.pivot_col[i]];
    for (int j = 0; j < ep.e; ++j) {
        double s = 0;
        for (int i = 0; i < ep.rank; ++i) s += ep.Lbasis[(size_t)i * ep.e + j] * zb[i];
        z[ep.zero_rows[j]] = s;
    }
    return z;
}

static bool check_solved(const CSC& P, const CSC& A, const std::vector<double>& q, const std::vector<double>& b,
                         const std::vector<Cone>& cones, const std::vector<double>& x, const std::vector<double>& z,
                         std::vector<double>& s_out, double tol) {
    int n = x.size(), m = b.size();
    std::vector<double> Ax(m), Px(n), Atz(n);
    A.matvec(x.data(), Ax.data());
    P.matvec(x.data(), Px.data());
    A.tmatvec(z.data(), Atz.data());
    s_out.resize(m);
    for (int i = 0; i < m; ++i) s_out[i] = b[i] - Ax[i];
    double rp = 0, rd = 0;
    for (int i = 0; i < m; ++i) rp = std::max(rp, std::fabs(Ax[i] + s_out[i] - b[i]));
    for (int i = 0; i < n; ++i) rd = std::max(rd, std::fabs(Px[i] + q[i] + Atz[i]));
    rp /= 1 + std::max({inf_norm(Ax.data(), m), inf_norm(s_out.data(), m), inf_norm(b.data(), m)});
    rd /= 1 + std::max({inf_norm(Px.data(), n), inf_norm(q.data(), n), inf_norm(Atz.data(), n)});
    double xPx = dot(x.data(), Px.data(), n), qx = dot(q.data(), x.data(), n), bz = dot(b.data(), z.data(), m);
    double p0 = 0.5 * xPx + qx, d0 = -0.5 * xPx - bz;
    double gap = std::fabs(p0 - d0), scale = std::max(std::fabs(p0), std::fabs(d0));
    bool gapok = gap <= tol || (scale > 0 && gap / scale <= tol);
    double cp = cone_defect(s_out.data(), cones, false), cd = cone_defect(z.data(), cones, true);
    if (getenv("CONIC_DEBUG")) fprintf(stderr, "check rp %.3e rd %.3e gap %.3e cp %.3e cd %.3e x0 %.6e z0 %.6e\n", rp, rd, gap, cp, cd, x.empty()?0:x[0], z.empty()?0:z[0]);
    return rp <= tol && rd <= tol && gapok && cp <= tol && cd <= tol;
}
static bool check_primal_cert(const CSC& A, const std::vector<double>& b, const std::vector<Cone>& cones,
                              const std::vector<double>& z, double tol) {
    int m = b.size();
    double nz = inf_norm(z.data(), m), nb = inf_norm(b.data(), m);
    if (nz <= 0 || nb <= 0) return false;
    double amax = 0; for (double v : A.data) amax = std::max(amax, std::fabs(v)); if (amax == 0) amax = 1;
    std::vector<double> zh(m), Atz(A.cols);
    for (int i = 0; i < m; ++i) zh[i] = z[i] / nz;
    double beta = 0; for (int i = 0; i < m; ++i) beta -= (b[i] / nb) * zh[i];
    if (beta <= tol) return false;
    A.tmatvec(zh.data(), Atz.data());
    double lin = inf_norm(Atz.data(), A.cols) / amax / beta;
    double cd = cone_defect(zh.data(), cones, true) / beta;
    return lin <= tol && cd <= tol;
}
static bool check_dual_cert(const CSC& P, const CSC& A, const std::vector<double>& q, const std::vector<Cone>& cones,
                            const std::vector<double>& x, std::vector<double>& s_out, double tol) {
    int n = x.size(), m = A.rows;
    double nx = inf_norm(x.data(), n), nq = inf_norm(q.data(), n);
    if (nx <= 0 || nq <= 0) return false;
    double amax = 0; for (double v : A.data) amax = std::max(amax, std::fabs(v)); if (amax == 0) amax = 1;
    double pmax = 0; for (double v : P.data) pmax = std::max(pmax, std::fabs(v)); if (pmax == 0) pmax = 1;
    std::vector<double> xh(n), Ax(m), Px(n);
    for (int i = 0; i < n; ++i) xh[i] = x[i] / nx;
    A.matvec(xh.data(), Ax.data());
    P.matvec(xh.data(), Px.data());
    double alpha = 0; for (int i = 0; i < n; ++i) alpha -= (q[i] / nq) * xh[i];
    if (alpha <= tol) return false;
    s_out.resize(m);
    for (int i = 0; i < m; ++i) s_out[i] = -Ax[i] * nx;
    double lin = 0;
    for (int i = 0; i < m; ++i) lin = std::max(lin, std::fabs(Ax[i] / amax + s_out[i] / nx / amax));
    lin /= alpha;
    double qp = inf_norm(Px.data(), n) / pmax / alpha;
    std::vector<double> ss(m);
    for (int i = 0; i < m; ++i) ss[i] = s_out[i] / nx / amax;
    double cd = cone_defect(ss.data(), cones, false) / alpha;
    return qp <= tol && lin <= tol && cd <= tol;
}

struct Block {
    ConeType type; int start; int dim; int width;
    std::vector<double> S, Sinv, Winv;
    std::vector<double> expH, expG;
};
static double psd_inv(int d, double* S, double* Sinv) {
    std::vector<double> A(S, S + d * d);
    int info;
    scipy_dpotrf_("L", &d, A.data(), &d, &info);
    if (info != 0) return false;
    std::vector<double> I(d * d, 0.0);
    for (int i = 0; i < d; ++i) I[i + i * d] = 1;
    scipy_dpotrs_("L", &d, &d, A.data(), &d, I.data(), &d, &info);
    if (info != 0) return false;
    memcpy(Sinv, I.data(), d * d * sizeof(double));
    return true;
}
static void apply_psd_congruence(const double* in, double* out, int d, const double* L, const double* R, double scale) {
    std::vector<double> V(d * d), T(d * d);
    smat(in, d, V.data());
    for (int j = 0; j < d; ++j)
        for (int i = 0; i < d; ++i) {
            double s = 0;
            for (int k = 0; k < d; ++k) s += L[i + k * d] * V[k + j * d];
            T[i + j * d] = s;
        }
    for (int j = 0; j < d; ++j)
        for (int i = 0; i < d; ++i) {
            double s = 0;
            for (int k = 0; k < d; ++k) s += T[i + k * d] * R[k + j * d];
            V[i + j * d] = s * scale;
        }
    svec(V.data(), d, out);
}
static void soc_hinv_matrix(const double* s, const double* z, int k, double* B) {
    double s0 = s[0], z0 = z[0], ss = 0, zz = 0, sz = 0;
    for (int i = 1; i < k; ++i) { ss += s[i] * s[i]; zz += z[i] * z[i]; sz += s[i] * z[i]; }
    double ds = s0 * s0 - ss, dz = z0 * z0 - zz;
    if (ds <= 1e-30) ds = 1e-30;
    if (dz <= 1e-30) dz = 1e-30;
    double gam2 = ds / dz, gam = std::sqrt(gam2);
    double r0 = (s0 * z0 - sz) / dz;
    double den = r0 + gam;
    if (den <= 1e-30) den = 1e-30;
    std::vector<double> rv(k - 1);
    for (int i = 1; i < k; ++i) rv[i - 1] = (-s0 * z[i] + z0 * s[i]) / dz;
    double c = 1 / gam2;
    B[0] = c * r0;
    for (int i = 1; i < k; ++i) {
        B[i] = -c * rv[i - 1];
        B[i * k] = B[i];
    }
    for (int i = 1; i < k; ++i)
        for (int j = 1; j < k; ++j)
            B[i + j * k] = (i == j ? 1 / gam : 0.0) + c * rv[i - 1] * rv[j - 1] / den;
}
static void exp_hess(const double* s, double mu, double* B) {
    double x = s[0], y = s[1], z = s[2];
    if (y <= 1e-30) y = 1e-30;
    if (z <= 1e-30) z = 1e-30;
    double L = std::log(z / y), D = y * L - x;
    if (D <= 1e-30) D = 1e-30;
    double a0 = -1, a1 = L - 1, a2 = y / z;
    double H[9];
    H[0] = 1 / (D * D);
    H[1] = a0 * a1 / (D * D);
    H[2] = a0 * a2 / (D * D);
    H[3] = H[1];
    H[4] = a1 * a1 / (D * D) + 1 / (D * y) + 1 / (y * y);
    H[5] = a1 * a2 / (D * D) - 1 / (D * z);
    H[6] = H[2];
    H[7] = H[5];
    H[8] = a2 * a2 / (D * D) + y / (D * z * z) + 1 / (z * z);
    for (int i = 0; i < 9; ++i) B[i] = mu * H[i];
}
static void inv3(const double* A, double* Inv) {
    double a = A[0], b = A[3], c = A[6], d = A[1], e = A[4], f = A[7], g = A[2], h = A[5], i = A[8];
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::fabs(det) < 1e-300) det = det >= 0 ? 1e-300 : -1e-300;
    double id = 1 / det;
    Inv[0] = (e * i - f * h) * id;
    Inv[1] = (c * h - b * i) * id;
    Inv[2] = (b * f - c * e) * id;
    Inv[3] = (f * g - d * i) * id;
    Inv[4] = (a * i - c * g) * id;
    Inv[5] = (c * d - a * f) * id;
    Inv[6] = (d * h - e * g) * id;
    Inv[7] = (b * g - a * h) * id;
    Inv[8] = (a * e - b * d) * id;
}
static bool exp_primal_interior(const double* w) {
    return w[1] > 1e-14 && w[2] > 1e-14 && w[1] * std::log(w[2] / w[1]) - w[0] > 1e-14;
}
static void exp_dual_grad(const double* z, double* g) {
    double a = z[0], b = z[1], c = z[2];
    if (a >= -1e-14) a = -1e-14;
    if (c <= 1e-14) c = 1e-14;
    double E = std::exp(b / a);
    double D = std::exp(1.0) * c + a * E;
    if (D <= 1e-14) D = 1e-14;
    g[0] = E * (1 - b / a) / D + 1 / a;
    g[1] = E / D;
    g[2] = std::exp(1.0) / D + 1 / c;
}
static void exp_nt(const double* s, const double* z, double* H, double* g) {
    double w[3] = {s[0], s[1], s[2]};
    if (!exp_primal_interior(w)) { w[0] = -1; w[1] = 1; w[2] = 1; }
    auto resid = [&](const double* ww, double* R) {
        double F[9]; exp_hess(ww, 1.0, F);
        for (int i = 0; i < 3; ++i) R[i] = F[i] * s[0] + F[i + 3] * s[1] + F[i + 6] * s[2] - z[i];
    };
    double R[3];
    for (int it = 0; it < 30; ++it) {
        resid(w, R);
        double nr = std::max({std::fabs(R[0]), std::fabs(R[1]), std::fabs(R[2])});
        double nz = std::max({1.0, std::fabs(z[0]), std::fabs(z[1]), std::fabs(z[2])});
        if (nr <= 1e-10 * nz) break;
        double J[9];
        for (int j = 0; j < 3; ++j) {
            double h = 1e-6 * std::max(1.0, std::fabs(w[j]));
            double wp[3] = {w[0], w[1], w[2]}, Rp[3], Rm[3];
            wp[j] += h; resid(wp, Rp);
            wp[j] -= 2 * h; resid(wp, Rm);
            for (int i = 0; i < 3; ++i) J[i + j * 3] = (Rp[i] - Rm[i]) / (2 * h);
        }
        double Ji[9], dw[3]; inv3(J, Ji);
        for (int i = 0; i < 3; ++i) dw[i] = -(Ji[i] * R[0] + Ji[i + 3] * R[1] + Ji[i + 6] * R[2]);
        double alpha = 1;
        bool ok = false;
        for (int bt = 0; bt < 30; ++bt) {
            double wn[3] = {w[0] + alpha * dw[0], w[1] + alpha * dw[1], w[2] + alpha * dw[2]};
            if (exp_primal_interior(wn)) {
                double Rn[3]; resid(wn, Rn);
                double nn = std::max({std::fabs(Rn[0]), std::fabs(Rn[1]), std::fabs(Rn[2])});
                if (nn < nr || alpha < 1e-8) { w[0] = wn[0]; w[1] = wn[1]; w[2] = wn[2]; ok = true; break; }
            }
            alpha *= 0.5;
        }
        if (!ok) break;
    }
    exp_hess(w, 1.0, H);
    exp_dual_grad(z, g);
    if (getenv("CONIC_DEBUG")) {
        double R[3]; resid(w, R);
        fprintf(stderr, "expnt res %.3e %.3e %.3e w %.3e %.3e %.3e\n", R[0], R[1], R[2], w[0], w[1], w[2]);
    }
}

struct Solver {
    int n, m;
    CSC P, A;
    std::vector<double> q, b;
    std::vector<Block> blocks;
    std::vector<double> E, D;
    std::vector<int> map_row, aug_to_orig, zero_plus, zero_minus;
    std::vector<Cone> orig_cones;
    CSC origA, origP;
    std::vector<double> origq, origb;
    double nu = 1;
    std::vector<double> Adense, Pdense, HA;
    double budget = 0.45;
    double start_time;

    void apply_hinv_vec(const double* in, double* out, double mu, const std::vector<double>& s, const std::vector<double>& z) {
        for (auto& bl : blocks) {
            const double* si = s.data() + bl.start;
            const double* zi = z.data() + bl.start;
            const double* vi = in + bl.start;
            double* vo = out + bl.start;
            if (bl.type == NONNEG) {
                for (int i = 0; i < bl.width; ++i) vo[i] = (zi[i] / si[i]) * vi[i];
            } else if (bl.type == SOC) {
                std::vector<double> B(bl.width * bl.width);
                soc_hinv_matrix(si, zi, bl.width, B.data());
                int k = bl.width;
                double one = 1, zero = 0;
                int inc = 1;
                scipy_dgemv_("N", &k, &k, &one, B.data(), &k, vi, &inc, &zero, vo, &inc);
            } else if (bl.type == EXP) {
                double B[9]; exp_hess(si, mu, B);
                for (int i = 0; i < 3; ++i) vo[i] = B[i] * vi[0] + B[i + 3] * vi[1] + B[i + 6] * vi[2];
            } else {
                apply_psd_congruence(vi, vo, bl.dim, bl.Winv.data(), bl.Winv.data(), 1.0);
            }
        }
    }
    void apply_hinv_mat(double mu, const std::vector<double>& s, const std::vector<double>& z) {
        HA.assign((size_t)m * n, 0.0);
        for (auto& bl : blocks) {
            int w = bl.width;
            const double* si = s.data() + bl.start;
            const double* zi = z.data() + bl.start;
            if (bl.type == NONNEG) {
                for (int i = 0; i < w; ++i) {
                    double wt = zi[i] / si[i];
                    for (int j = 0; j < n; ++j) HA[bl.start + i + (size_t)j * m] = wt * Adense[bl.start + i + (size_t)j * m];
                }
            } else if (bl.type == SOC || bl.type == EXP) {
                std::vector<double> B(w * w);
                if (bl.type == SOC) soc_hinv_matrix(si, zi, w, B.data());
                else exp_hess(si, mu, B.data());
                double one = 1, zero = 0;
                int nn = n;
                scipy_dgemm_("N", "N", &w, &nn, &w, &one, B.data(), &w, Adense.data() + bl.start, &m, &zero, HA.data() + bl.start, &m);
            } else {
                for (int j = 0; j < n; ++j)
                    apply_psd_congruence(Adense.data() + bl.start + (size_t)j * m, HA.data() + bl.start + (size_t)j * m,
                                         bl.dim, bl.Winv.data(), bl.Winv.data(), 1.0);
            }
        }
    }
    void compute_rhs_c(double mu, double mu_target, const std::vector<double>& s, const std::vector<double>& z, std::vector<double>& rhs,
                       const std::vector<double>* dsaff = nullptr, const std::vector<double>* dzaff = nullptr) {
        rhs.assign(m, 0.0);
        double sigma = mu_target / mu;
        for (auto& bl : blocks) {
            const double* si = s.data() + bl.start;
            const double* zi = z.data() + bl.start;
            double* r = rhs.data() + bl.start;
            if (bl.type == NONNEG) {
                for (int i = 0; i < bl.width; ++i) {
                    r[i] = -si[i] + mu_target / zi[i];
                    if (dsaff && dzaff) r[i] -= (*dsaff)[bl.start + i] * (*dzaff)[bl.start + i] / zi[i];
                }
            } else if (bl.type == SOC) {
                double z0 = zi[0], zz = 0;
                for (int i = 1; i < bl.width; ++i) zz += zi[i] * zi[i];
                double dz = z0 * z0 - zz;
                if (dz <= 1e-30) dz = 1e-30;
                r[0] = -si[0] + mu_target * z0 / dz;
                for (int i = 1; i < bl.width; ++i) r[i] = -si[i] - mu_target * zi[i] / dz;
            } else if (bl.type == EXP) {
                double B[9], Inv[9]; exp_hess(si, 1.0, B); inv3(B, Inv);
                for (int i = 0; i < 3; ++i) {
                    double hz = (Inv[i] * zi[0] + Inv[i + 3] * zi[1] + Inv[i + 6] * zi[2]) / mu;
                    r[i] = -hz + (mu_target / mu) * si[i];
                }
            } else {
                std::vector<double> Z(bl.dim * bl.dim), Zinv(bl.dim * bl.dim), rz(bl.width);
                smat(zi, bl.dim, Z.data());
                psd_inv(bl.dim, Z.data(), Zinv.data());
                svec(Zinv.data(), bl.dim, rz.data());
                for (int i = 0; i < bl.width; ++i) r[i] = -si[i] + mu_target * rz[i];
            }
        }
    }
    void refresh_psd(const std::vector<double>& s) {
        for (auto& bl : blocks) if (bl.type == PSD) {
            bl.S.assign(bl.dim * bl.dim, 0);
            bl.Sinv.assign(bl.dim * bl.dim, 0);
            bl.Winv.assign(bl.dim * bl.dim, 0);
            smat(s.data() + bl.start, bl.dim, bl.S.data());
            if (!psd_inv(bl.dim, bl.S.data(), bl.Sinv.data())) {
                for (int i = 0; i < bl.dim; ++i) bl.S[i + i * bl.dim] += 1e-8;
                psd_inv(bl.dim, bl.S.data(), bl.Sinv.data());
            }
        }
    }
    void refresh_psd_nt(const std::vector<double>& s, const std::vector<double>& z) {
        for (auto& bl : blocks) if (bl.type == PSD) {
            int d = bl.dim;
            bl.S.assign(d * d, 0); bl.Winv.assign(d * d, 0);
            std::vector<double> Z(d * d);
            smat(s.data() + bl.start, d, bl.S.data());
            smat(z.data() + bl.start, d, Z.data());
            auto sqrt_mat = [&](const double* A, double* Root, double* InvRoot) {
                std::vector<double> V(A, A + d * d), w(d), work(1);
                int lwork = -1, info;
                scipy_dsyev_("V", "L", &d, V.data(), &d, w.data(), work.data(), &lwork, &info);
                lwork = (int)work[0]; work.resize(lwork);
                scipy_dsyev_("V", "L", &d, V.data(), &d, w.data(), work.data(), &lwork, &info);
                for (int i = 0; i < d; ++i) if (w[i] < 1e-14) w[i] = 1e-14;
                std::fill(Root, Root + d * d, 0); std::fill(InvRoot, InvRoot + d * d, 0);
                for (int j = 0; j < d; ++j) {
                    double rw = std::sqrt(w[j]), iw = 1 / rw;
                    for (int i = 0; i < d; ++i) {
                        Root[i + j * d] = V[i + j * d] * rw;
                        InvRoot[i + j * d] = V[i + j * d] * iw;
                    }
                }
                // Root = V diag(rw) V^T
                std::vector<double> T(d * d);
                for (int j = 0; j < d; ++j)
                    for (int i = 0; i < d; ++i) {
                        double v = 0;
                        for (int k = 0; k < d; ++k) v += Root[i + k * d] * V[j + k * d];
                        T[i + j * d] = v;
                    }
                memcpy(Root, T.data(), d * d * sizeof(double));
                for (int j = 0; j < d; ++j)
                    for (int i = 0; i < d; ++i) {
                        double v = 0;
                        for (int k = 0; k < d; ++k) v += InvRoot[i + k * d] * V[j + k * d];
                        T[i + j * d] = v;
                    }
                memcpy(InvRoot, T.data(), d * d * sizeof(double));
            };
            std::vector<double> Sqrt(d * d), iSqrt(d * d), M(d * d), Msqrt(d * d), iMsqrt(d * d), T(d * d);
            sqrt_mat(bl.S.data(), Sqrt.data(), iSqrt.data());
            for (int j = 0; j < d; ++j)
                for (int i = 0; i < d; ++i) {
                    double v = 0;
                    for (int k = 0; k < d; ++k) v += Sqrt[i + k * d] * Z[k + j * d];
                    T[i + j * d] = v;
                }
            for (int j = 0; j < d; ++j)
                for (int i = 0; i < d; ++i) {
                    double v = 0;
                    for (int k = 0; k < d; ++k) v += T[i + k * d] * Sqrt[k + j * d];
                    M[i + j * d] = v;
                }
            sqrt_mat(M.data(), Msqrt.data(), iMsqrt.data());
            for (int j = 0; j < d; ++j)
                for (int i = 0; i < d; ++i) {
                    double v = 0;
                    for (int k = 0; k < d; ++k) v += iSqrt[i + k * d] * Msqrt[k + j * d];
                    T[i + j * d] = v;
                }
            for (int j = 0; j < d; ++j)
                for (int i = 0; i < d; ++i) {
                    double v = 0;
                    for (int k = 0; k < d; ++k) v += T[i + k * d] * iSqrt[k + j * d];
                    bl.Winv[i + j * d] = v;
                }
        }
    }
    void refresh_exp_nt(const std::vector<double>& s, const std::vector<double>& z) {
        for (auto& bl : blocks) if (bl.type == EXP) {
            bl.expH.assign(9, 0); bl.expG.assign(3, 0);
            exp_nt(s.data() + bl.start, z.data() + bl.start, bl.expH.data(), bl.expG.data());
        }
    }
    double max_step(const std::vector<double>& s, const std::vector<double>& ds, const std::vector<double>& z, const std::vector<double>& dz) {
        double alpha = 1e100;
        auto upd = [&](const double* v, const double* dv, const Block& bl) {
            if (bl.type == NONNEG) {
                for (int i = 0; i < bl.width; ++i) if (dv[i] < 0) alpha = std::min(alpha, -v[i] / dv[i]);
            } else if (bl.type == SOC) {
                int k = bl.width;
                double t = v[0], dt = dv[0], r = 0, dr = 0, tdr = 0;
                for (int i = 1; i < k; ++i) { r += v[i] * v[i]; dr += dv[i] * dv[i]; tdr += v[i] * dv[i]; }
                double a = dt * dt - dr, bb = 2 * (t * dt - tdr), c = t * t - r;
                if (dt < 0) alpha = std::min(alpha, -t / dt);
                if (std::fabs(a) < 1e-14) { if (bb < 0) alpha = std::min(alpha, -c / bb); }
                else {
                    double disc = bb * bb - 4 * a * c;
                    if (disc > 0) {
                        double sq = std::sqrt(disc);
                        double r1 = (-bb - sq) / (2 * a), r2 = (-bb + sq) / (2 * a);
                        if (r1 > 1e-12) alpha = std::min(alpha, r1);
                        if (r2 > 1e-12) alpha = std::min(alpha, r2);
                    }
                }
            } else if (bl.type == PSD) {
                int d = bl.dim;
                std::vector<double> Sv(d * d), Dv(d * d), w(d), work(1);
                smat(v, d, Sv.data()); smat(dv, d, Dv.data());
                int itype = 1, lwork = -1, info;
                scipy_dsygv_(&itype, "N", "L", &d, Dv.data(), &d, Sv.data(), &d, w.data(), work.data(), &lwork, &info);
                lwork = (int)work[0]; work.resize(lwork);
                scipy_dsygv_(&itype, "N", "L", &d, Dv.data(), &d, Sv.data(), &d, w.data(), work.data(), &lwork, &info);
                if (info == 0 && w[0] < -1e-12) alpha = std::min(alpha, -1 / w[0]);
            }
        };
        for (auto& bl : blocks) {
            upd(s.data() + bl.start, ds.data() + bl.start, bl);
            upd(z.data() + bl.start, dz.data() + bl.start, bl);
        }
        return alpha;
    }
    bool exp_interior(const std::vector<double>& s, const std::vector<double>& z) {
        for (auto& bl : blocks) if (bl.type == EXP) {
            const double* p = s.data() + bl.start;
            const double* d = z.data() + bl.start;
            if (p[1] <= 1e-16 || p[2] <= 1e-16 || p[1] * std::log(p[2] / p[1]) - p[0] <= 1e-16) return false;
            if (d[0] >= -1e-16 || d[2] <= 1e-16) return false;
            if (-d[0] * std::exp(d[1] / d[0]) >= std::exp(1.0) * d[2] * (1 - 1e-14)) return false;
        }
        return true;
    }
    CSR Acsr;
    std::vector<int> row_block;
    void prepare_sparse() {
        Acsr.from_csc(A);
        row_block.assign(m, -1);
        for (int bi = 0; bi < (int)blocks.size(); ++bi)
            for (int r = blocks[bi].start; r < blocks[bi].start + blocks[bi].width; ++r) row_block[r] = bi;
    }
    void compute_diag(double mu, const std::vector<double>& s, const std::vector<double>& z, std::vector<double>& diag) {
        diag.assign(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int k = P.indptr[j]; k < P.indptr[j + 1]; ++k)
                if (P.indices[k] == j) diag[j] += P.data[k];
        for (auto& bl : blocks) if (bl.type == NONNEG) {
            for (int r = bl.start; r < bl.start + bl.width; ++r) {
                double w = z[r] / s[r];
                for (int k = Acsr.indptr[r]; k < Acsr.indptr[r + 1]; ++k)
                    diag[Acsr.indices[k]] += w * Acsr.data[k] * Acsr.data[k];
            }
        }
        for (auto& bl : blocks) if (bl.type == SOC || bl.type == EXP) {
            std::vector<double> B(bl.width * bl.width);
            if (bl.type == SOC) soc_hinv_matrix(s.data() + bl.start, z.data() + bl.start, bl.width, B.data());
            else exp_hess(s.data() + bl.start, mu, B.data());
            for (int j = 0; j < n; ++j)
                for (int k = A.indptr[j]; k < A.indptr[j + 1]; ++k) {
                    int r = A.indices[k];
                    if (r >= bl.start && r < bl.start + bl.width) {
                        int l = r - bl.start;
                        diag[j] += B[l + l * bl.width] * A.data[k] * A.data[k];
                    }
                }
        }
        for (int j = 0; j < n; ++j) if (diag[j] < 1e-12) diag[j] = 1e-12;
    }
    void matvec_M(const double* v, double* out, double sigma, double mu, const std::vector<double>& s, const std::vector<double>& z) {
        std::vector<double> Av(m), HAv(m), AtHAv(n), Pv(n);
        A.matvec(v, Av.data());
        apply_hinv_vec(Av.data(), HAv.data(), mu, s, z);
        A.tmatvec(HAv.data(), AtHAv.data());
        P.matvec(v, Pv.data());
        for (int i = 0; i < n; ++i) out[i] = Pv[i] + AtHAv[i] + sigma * v[i];
    }
    void cg_solve(const std::vector<double>& rhs, std::vector<double>& out, double sigma, double mu,
                  const std::vector<double>& s, const std::vector<double>& z, const std::vector<double>& diag) {
        out.assign(n, 0.0);
        std::vector<double> r(rhs), p(n), p2(n), Ap(n);
        double rz = 0;
        for (int i = 0; i < n; ++i) { p[i] = r[i] / diag[i]; rz += r[i] * p[i]; }
        double nrhs = inf_norm(rhs.data(), n);
        if (nrhs == 0) return;
        int maxit = std::min(1000, 2 * n + 50);
        for (int it = 0; it < maxit; ++it) {
            matvec_M(p.data(), Ap.data(), sigma, mu, s, z);
            double pAp = dot(p.data(), Ap.data(), n);
            if (!(pAp > 0)) break;
            double alpha = rz / pAp;
            for (int i = 0; i < n; ++i) { out[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
            if (inf_norm(r.data(), n) <= 1e-10 * (1 + nrhs)) break;
            double rz2 = 0;
            for (int i = 0; i < n; ++i) { p2[i] = r[i] / diag[i]; rz2 += r[i] * p2[i]; }
            double beta = rz2 / rz;
            for (int i = 0; i < n; ++i) p[i] = p2[i] + beta * p[i];
            rz = rz2;
        }
    }
    Result run() {
        if (P.data.empty() && !getenv("CONIC_ORIG")) return run_hsd();
        return run_orig();
    }
    Result run_hsd() {
        start_time = now_sec();
        bool debug = getenv("CONIC_DEBUG") != nullptr;
        bool dense = n <= 1000 && (int64_t)m * n <= 6000000;
        if (!dense) prepare_sparse();
        std::vector<double> x(n, 0.0), s(m), z(m);
        double tau = 1, kappa = 1;
        for (auto& bl : blocks) {
            if (bl.type == NONNEG) for (int i = 0; i < bl.width; ++i) { s[bl.start + i] = 1; z[bl.start + i] = 1; }
            else if (bl.type == SOC) { s[bl.start] = z[bl.start] = 1; }
            else if (bl.type == EXP) { s[bl.start] = -1; s[bl.start + 1] = 1; s[bl.start + 2] = 1; z[bl.start] = -1; z[bl.start + 2] = 1; }
            else { for (int i = 0; i < bl.dim; ++i) { int idx = bl.start + i * (i + 1) / 2 + i; s[idx] = z[idx] = 1; } }
        }
        if (dense) {
            Adense.assign((size_t)m * n, 0.0);
            for (int j = 0; j < n; ++j)
                for (int k = A.indptr[j]; k < A.indptr[j + 1]; ++k) Adense[A.indices[k] + (size_t)j * m] = A.data[k];
        }
        std::vector<double> M, rhs(n * 2), dxu(n), dx0(n), Hb(m), Ht(m), y(m), t_s(m), rhs_c(m), diag;
        if (dense) M.resize((size_t)n * n);
        std::vector<double> rx(n), rs(m), Atz(n), w(n), v(n);
        double sigma_reg = 1e-8;
        int mo = origb.size();
        Result res; res.status = "unresolved"; res.x.assign(n, 0); res.s.assign(mo, 0); res.z.assign(mo, 0);
        for (int iter = 0; iter < 200; ++iter) {
            if (now_sec() - start_time > budget) break;
            double mu = (dot(s.data(), z.data(), m) + tau * kappa) / nu;
            if (!(mu > 1e-30) || !std::isfinite(mu)) break;
            refresh_psd_nt(s, z);
            double one = 1;
            int nn = n, mm = m;
            if (dense) {
                apply_hinv_mat(mu, s, z);
                std::fill(M.begin(), M.end(), 0.0);
                scipy_dgemm_("T", "N", &nn, &nn, &mm, &one, Adense.data(), &mm, HA.data(), &mm, &one, M.data(), &nn);
                bool factored = false;
                for (int tries = 0; tries < 6; ++tries) {
                    for (int i = 0; i < n; ++i) M[i + (size_t)i * n] += sigma_reg;
                    int info;
                    scipy_dpotrf_("L", &nn, M.data(), &nn, &info);
                    if (info == 0) { factored = true; break; }
                    sigma_reg *= 10;
                    std::fill(M.begin(), M.end(), 0.0);
                    scipy_dgemm_("T", "N", &nn, &nn, &mm, &one, Adense.data(), &mm, HA.data(), &mm, &one, M.data(), &nn);
                }
                if (!factored) break;
            } else {
                compute_diag(mu, s, z, diag);
            }
            apply_hinv_vec(b.data(), Hb.data(), mu, s, z);
            A.tmatvec(Hb.data(), w.data());
            double bHb = dot(b.data(), Hb.data(), m), h = kappa / tau;
            A.tmatvec(z.data(), Atz.data());
            for (int i = 0; i < n; ++i) rx[i] = Atz[i] + q[i] * tau;
            A.matvec(x.data(), rs.data());
            for (int i = 0; i < m; ++i) rs[i] += s[i] - b[i] * tau;
            double rtau = dot(q.data(), x.data(), n) + dot(b.data(), z.data(), m) + kappa;
            if (debug && !std::isfinite(rtau)) fprintf(stderr, "rtau parts %.3e %.3e %.3e n %d m %d z0 %.3e b0 %.3e\n",
                dot(q.data(), x.data(), n), dot(b.data(), z.data(), m), kappa, n, m, z.empty()?0:z[0], b.empty()?0:b[0]);
            for (int i = 0; i < n; ++i) { v[i] = q[i] + w[i]; rhs[i + (size_t)n] = q[i] - w[i]; }
            auto compute_aff = [&](double mu_target, const std::vector<double>* dsaffp = nullptr, const std::vector<double>* dzaffp = nullptr) {
                compute_rhs_c(mu, mu_target, s, z, rhs_c, dsaffp, dzaffp);
                for (int i = 0; i < m; ++i) t_s[i] = rs[i] + rhs_c[i];
                apply_hinv_vec(t_s.data(), Ht.data(), mu, s, z);
                std::vector<double> hat(n);
                if (dense) {
                    int inc = 1;
                    scipy_dgemv_("T", &mm, &nn, &one, HA.data(), &mm, t_s.data(), &inc, &one, hat.data(), &inc);
                } else A.tmatvec(Ht.data(), hat.data());
                for (int i = 0; i < n; ++i) rhs[i] = -rx[i] - hat[i];
            };
            compute_aff(0.0);
            if (dense) {
                int nrhs = 2, info;
                scipy_dpotrs_("L", &nn, &nrhs, M.data(), &nn, rhs.data(), &nn, &info);
                for (int i = 0; i < n; ++i) { dx0[i] = rhs[i]; dxu[i] = rhs[i + (size_t)n]; }
            } else {
                std::vector<double> rhsu(rhs.begin() + n, rhs.end());
                cg_solve(rhs, dx0, sigma_reg, mu, s, z, diag);
                cg_solve(rhsu, dxu, sigma_reg, mu, s, z, diag);
            }
            double rhs_k = -kappa;
            double rhs3 = -rtau - rhs_k - dot(b.data(), Ht.data(), m);
            double denom = -(bHb + h) - dot(v.data(), dxu.data(), n);
            if (!std::isfinite(denom) || std::fabs(denom) < 1e-12) denom = -1e-12;
            double dtau_aff = (rhs3 - dot(v.data(), dx0.data(), n)) / denom;
            if (!std::isfinite(dtau_aff) || std::fabs(dtau_aff) > 1e6) dtau_aff = 0;
            std::vector<double> dxaff(n), dsaff(m), dzaff(m);
            for (int i = 0; i < n; ++i) dxaff[i] = dx0[i] - dxu[i] * dtau_aff;
            A.matvec(dxaff.data(), y.data());
            for (int i = 0; i < m; ++i) y[i] += -b[i] * dtau_aff + t_s[i];
            apply_hinv_vec(y.data(), dzaff.data(), mu, s, z);
            for (int i = 0; i < m; ++i) dsaff[i] = rhs_c[i] - y[i];
            double dkappa_aff = rhs_k - h * dtau_aff;
            double alpha_aff = std::min(1.0, 0.995 * max_step(s, dsaff, z, dzaff));
            if (dtau_aff < 0) alpha_aff = std::min(alpha_aff, 0.995 * -tau / dtau_aff);
            if (dkappa_aff < 0) alpha_aff = std::min(alpha_aff, 0.995 * -kappa / dkappa_aff);
            if (debug && alpha_aff < 1e-2) {
                double mn = 1e100; int typ = -1, st = -1;
                for (auto& bl : blocks) if (bl.type == NONNEG) {
                    for (int i = 0; i < bl.width; ++i) {
                        if (dsaff[bl.start + i] < 0) { double a = -s[bl.start + i] / dsaff[bl.start + i]; if (a < mn) { mn = a; typ = 0; st = bl.start + i; } }
                        if (dzaff[bl.start + i] < 0) { double a = -z[bl.start + i] / dzaff[bl.start + i]; if (a < mn) { mn = a; typ = 1; st = bl.start + i; } }
                    }
                }
                for (auto& bl : blocks) if (bl.type == SOC) {
                    for (int pass = 0; pass < 2; ++pass) {
                        const double* v = pass ? z.data() + bl.start : s.data() + bl.start;
                        const double* dv = pass ? dzaff.data() + bl.start : dsaff.data() + bl.start;
                        int k = bl.width;
                        double t = v[0], dt = dv[0], r = 0, dr = 0, tdr = 0;
                        for (int i = 1; i < k; ++i) { r += v[i] * v[i]; dr += dv[i] * dv[i]; tdr += v[i] * dv[i]; }
                        double a = dt * dt - dr, bb = 2 * (t * dt - tdr), c = t * t - r, al = 1e100;
                        if (dt < 0) al = -t / dt;
                        if (std::fabs(a) < 1e-14) { if (bb < 0) al = std::min(al, -c / bb); }
                        else {
                            double disc = bb * bb - 4 * a * c;
                            if (disc > 0) {
                                double sq = std::sqrt(disc);
                                double r1 = (-bb - sq) / (2 * a), r2 = (-bb + sq) / (2 * a);
                                if (r1 > 1e-12) al = std::min(al, r1);
                                if (r2 > 1e-12) al = std::min(al, r2);
                            }
                        }
                        if (al < 1e-2) fprintf(stderr, "soc limit pass %d start %d al %.3e t %.3e dt %.3e r %.3e dr %.3e\n", pass, bl.start, al, t, dt, r, dr);
                    }
                }
                fprintf(stderr, "limit nn %.3e type %d idx %d dtau %.3e dk %.3e tau %.3e kap %.3e\n", mn, typ, st, dtau_aff, dkappa_aff, tau, kappa);
            }
            double mu_aff = 0;
            for (int i = 0; i < m; ++i) mu_aff += (s[i] + alpha_aff * dsaff[i]) * (z[i] + alpha_aff * dzaff[i]);
            mu_aff += (tau + alpha_aff * dtau_aff) * (kappa + alpha_aff * dkappa_aff);
            mu_aff /= nu;
            double sigma = mu_aff > 0 ? std::pow(std::min(1.0, mu_aff / mu), 3.0) : 0.0;
            sigma = std::min(sigma, 0.99);
            compute_aff(sigma * mu, &dsaff, &dzaff);
            if (dense) {
                int nrhs = 1, info;
                scipy_dpotrs_("L", &nn, &nrhs, M.data(), &nn, rhs.data(), &nn, &info);
                for (int i = 0; i < n; ++i) dx0[i] = rhs[i];
            } else cg_solve(rhs, dx0, sigma_reg, mu, s, z, diag);
            rhs_k = -kappa + sigma * mu / tau;
            rhs3 = -rtau - rhs_k - dot(b.data(), Ht.data(), m);
            double dtau_cc = (rhs3 - dot(v.data(), dx0.data(), n)) / denom;
            if (!std::isfinite(dtau_cc) || std::fabs(dtau_cc) > 1e6) dtau_cc = 0;
            std::vector<double> dxcc(n), dscc(m), dzcc(m);
            for (int i = 0; i < n; ++i) dxcc[i] = dx0[i] - dxu[i] * dtau_cc;
            A.matvec(dxcc.data(), y.data());
            for (int i = 0; i < m; ++i) y[i] += -b[i] * dtau_cc + t_s[i];
            apply_hinv_vec(y.data(), dzcc.data(), mu, s, z);
            for (int i = 0; i < m; ++i) dscc[i] = rhs_c[i] - y[i];
            double dkappa_cc = rhs_k - h * dtau_cc;
            auto try_step = [&](std::vector<double>& dx, std::vector<double>& ds, std::vector<double>& dz, double dtau, double dkappa) {
                double alpha = std::min(1.0, 0.995 * max_step(s, ds, z, dz));
                if (dtau < 0) alpha = std::min(alpha, 0.995 * -tau / dtau);
                if (dkappa < 0) alpha = std::min(alpha, 0.995 * -kappa / dkappa);
                if (!(alpha > 1e-12)) return false;
                std::vector<double> sn(m), zn(m);
                double res_old = std::max({inf_norm(rx.data(), n), inf_norm(rs.data(), m), std::fabs(rtau)});
                for (int bt = 0; bt < 30; ++bt) {
                    for (int i = 0; i < m; ++i) { sn[i] = s[i] + alpha * ds[i]; zn[i] = z[i] + alpha * dz[i]; }
                    double taun = tau + alpha * dtau, kappan = kappa + alpha * dkappa;
                    bool ok = taun > 0 && kappan > 0 && exp_interior(sn, zn);
                    double mun = (dot(sn.data(), zn.data(), m) + taun * kappan) / nu;
                    double res_new = 1e100;
                    if (ok) {
                        std::vector<double> xn(n), rsn(m), rxn(n), tmp(n);
                        for (int i = 0; i < n; ++i) xn[i] = x[i] + alpha * dx[i];
                        A.matvec(xn.data(), rsn.data());
                        for (int i = 0; i < m; ++i) rsn[i] += sn[i] - b[i] * taun;
                        A.tmatvec(zn.data(), tmp.data());
                        for (int i = 0; i < n; ++i) rxn[i] = tmp[i] + q[i] * taun;
                        double rtn = dot(q.data(), xn.data(), n) + dot(b.data(), zn.data(), m) + kappan;
                        res_new = std::max({inf_norm(rxn.data(), n), inf_norm(rsn.data(), m), std::fabs(rtn)});
                    }
                    bool reduce = mun > 0 && (mun < mu * (1 - 1e-5 * alpha) || (res_new < res_old * 0.999 && mun < mu * 10) ||
                                             (mun < mu * 1.001 && res_new <= res_old * 1.001));
                    if (ok && reduce) {
                        for (int i = 0; i < n; ++i) x[i] += alpha * dx[i];
                        s.swap(sn); z.swap(zn); tau = taun; kappa = kappan;
                        return true;
                    }
                    alpha *= 0.5;
                    if (alpha < 1e-12) break;
                }
                return false;
            };
            bool ok = try_step(dxcc, dscc, dzcc, dtau_cc, dkappa_cc);
            if (!ok) ok = try_step(dxaff, dsaff, dzaff, dtau_aff, dkappa_aff);
            if (debug) fprintf(stderr, "hsd %d mu %.3e tau %.3e kap %.3e sig %.3e aaff %.3e ok %d rx %.3e rs %.3e rt %.3e\n",
                               iter, mu, tau, kappa, sigma, alpha_aff, ok, inf_norm(rx.data(), n), inf_norm(rs.data(), m), fabs(rtau));
            if (!ok) break;
            if (iter % 2 == 1 || mu < 1e-6) {
                double tol = 9e-9;
                if (tau > 1e-30) {
                    std::vector<double> xo(n), zo(mo), so;
                    for (int i = 0; i < n; ++i) xo[i] = E[i] * x[i] / tau;
                    for (int i = 0; i < m; ++i) if (aug_to_orig[i] >= 0) zo[aug_to_orig[i]] = D[i] * z[i] / tau;
                    for (int i = 0; i < (int)zero_plus.size(); ++i) if (zero_plus[i] >= 0) {
                        int p = zero_plus[i], mm2 = zero_minus[i];
                        zo[i] = D[p] * z[p] / tau - D[mm2] * z[mm2] / tau;
                    }
                    if (check_solved(origP, origA, origq, origb, orig_cones, xo, zo, so, tol)) {
                        res.status = "solved"; res.x = xo; res.s = so; res.z = zo; return res;
                    }
                }
                std::vector<double> zc(mo, 0.0), xc(n), sc;
                for (int i = 0; i < m; ++i) if (aug_to_orig[i] >= 0) zc[aug_to_orig[i]] = D[i] * z[i];
                for (int i = 0; i < (int)zero_plus.size(); ++i) if (zero_plus[i] >= 0) {
                    int p = zero_plus[i], mm2 = zero_minus[i];
                    zc[i] = D[p] * z[p] - D[mm2] * z[mm2];
                }
                if (check_primal_cert(origA, origb, orig_cones, zc, tol)) {
                    res.status = "primal_infeasible"; res.z = zc; return res;
                }
                for (int i = 0; i < n; ++i) xc[i] = E[i] * x[i];
                if (check_dual_cert(origP, origA, origq, orig_cones, xc, sc, tol)) {
                    res.status = "dual_infeasible"; res.x = xc; res.s = sc; return res;
                }
            }
        }
        return res;
    }
    Result run_orig() {
        start_time = now_sec();
        bool debug = getenv("CONIC_DEBUG") != nullptr;
        bool dense = n <= 1000 && (int64_t)m * n <= 6000000;
        if (!dense) prepare_sparse();
        std::vector<double> x(n, 0.0), s(m), z(m);
        for (auto& bl : blocks) {
            if (bl.type == NONNEG) for (int i = 0; i < bl.width; ++i) { s[bl.start + i] = 1; z[bl.start + i] = 1; }
            else if (bl.type == SOC) { s[bl.start] = z[bl.start] = 1; }
            else if (bl.type == EXP) { s[bl.start] = -1; s[bl.start + 1] = 1; s[bl.start + 2] = 1; z[bl.start] = -1; z[bl.start + 2] = 1; }
            else { for (int i = 0; i < bl.dim; ++i) { int idx = bl.start + i * (i + 1) / 2 + i; s[idx] = z[idx] = 1; } }
        }
        if (dense) {
            Adense.assign((size_t)m * n, 0.0);
            for (int j = 0; j < n; ++j)
                for (int k = A.indptr[j]; k < A.indptr[j + 1]; ++k) Adense[A.indices[k] + (size_t)j * m] = A.data[k];
            Pdense.assign((size_t)n * n, 0.0);
            for (int j = 0; j < n; ++j)
                for (int k = P.indptr[j]; k < P.indptr[j + 1]; ++k) Pdense[P.indices[k] + (size_t)j * n] = P.data[k];
        }
        std::vector<double> M, rhs(n), Hb(m), Ht(m), y(m), t_s(m), rhs_c(m), diag;
        if (dense) M.resize((size_t)n * n);
        std::vector<double> rx(n), rs(m), Atz(n), Px(n);
        double sigma_reg = 1e-7;
        int mo = origb.size();
        Result res; res.status = "unresolved"; res.x.assign(n, 0); res.s.assign(mo, 0); res.z.assign(mo, 0);
        double nuc = nu - 1;
        for (int iter = 0; iter < 200; ++iter) {
            if (now_sec() - start_time > budget) break;
            double mu = dot(s.data(), z.data(), m) / nuc;
            if (!(mu > 1e-30) || !std::isfinite(mu)) break;
            refresh_psd_nt(s, z);
            double one = 1;
            int nn = n, mm = m;
            if (dense) {
                apply_hinv_mat(mu, s, z);
                M = Pdense;
                scipy_dgemm_("T", "N", &nn, &nn, &mm, &one, Adense.data(), &mm, HA.data(), &mm, &one, M.data(), &nn);
                bool factored = false;
                for (int tries = 0; tries < 6; ++tries) {
                    for (int i = 0; i < n; ++i) M[i + (size_t)i * n] += sigma_reg;
                    int info;
                    scipy_dpotrf_("L", &nn, M.data(), &nn, &info);
                    if (info == 0) { factored = true; break; }
                    sigma_reg *= 10;
                    M = Pdense;
                    scipy_dgemm_("T", "N", &nn, &nn, &mm, &one, Adense.data(), &mm, HA.data(), &mm, &one, M.data(), &nn);
                }
                if (!factored) break;
            } else {
                compute_diag(mu, s, z, diag);
            }
            P.matvec(x.data(), Px.data());
            A.tmatvec(z.data(), Atz.data());
            for (int i = 0; i < n; ++i) rx[i] = Px[i] + Atz[i] + q[i];
            A.matvec(x.data(), rs.data());
            for (int i = 0; i < m; ++i) rs[i] += s[i] - b[i];
            auto solve_dir = [&](double mu_target, std::vector<double>& dxout, std::vector<double>& dsout, std::vector<double>& dzout,
                                 const std::vector<double>* dsaffp = nullptr, const std::vector<double>* dzaffp = nullptr) {
                compute_rhs_c(mu, mu_target, s, z, rhs_c, dsaffp, dzaffp);
                for (int i = 0; i < m; ++i) t_s[i] = rs[i] + rhs_c[i];
                apply_hinv_vec(t_s.data(), Ht.data(), mu, s, z);
                std::vector<double> hat(n);
                if (dense) {
                    int inc = 1;
                    scipy_dgemv_("T", &mm, &nn, &one, HA.data(), &mm, t_s.data(), &inc, &one, hat.data(), &inc);
                } else {
                    A.tmatvec(Ht.data(), hat.data());
                }
                for (int i = 0; i < n; ++i) rhs[i] = -rx[i] - hat[i];
                if (dense) {
                    int nrhs = 1, info;
                    scipy_dpotrs_("L", &nn, &nrhs, M.data(), &nn, rhs.data(), &nn, &info);
                    for (int i = 0; i < n; ++i) dxout[i] = rhs[i];
                } else {
                    cg_solve(rhs, dxout, sigma_reg, mu, s, z, diag);
                }
                A.matvec(dxout.data(), y.data());
                for (int i = 0; i < m; ++i) y[i] += t_s[i];
                apply_hinv_vec(y.data(), dzout.data(), mu, s, z);
                for (int i = 0; i < m; ++i) dsout[i] = rhs_c[i] - y[i];
            };
            std::vector<double> dxaff(n), dsaff(m), dzaff(m);
            solve_dir(0.0, dxaff, dsaff, dzaff);
            double alpha_aff = std::min(1.0, 0.995 * max_step(s, dsaff, z, dzaff));
            double mu_aff = 0;
            for (int i = 0; i < m; ++i) mu_aff += (s[i] + alpha_aff * dsaff[i]) * (z[i] + alpha_aff * dzaff[i]);
            mu_aff /= nuc;
            double sigma = mu_aff > 0 ? std::pow(std::min(1.0, mu_aff / mu), 3.0) : 0.0;
            sigma = std::min(sigma, 0.99);
            std::vector<double> dxcc(n), dscc(m), dzcc(m);
            solve_dir(sigma * mu, dxcc, dscc, dzcc, &dsaff, &dzaff);
            auto try_step = [&](std::vector<double>& dx, std::vector<double>& ds, std::vector<double>& dz) -> bool {
                double alpha = std::min(1.0, 0.995 * max_step(s, ds, z, dz));
                if (!(alpha > 1e-12)) return false;
                std::vector<double> sn(m), zn(m);
                double res_old = std::max(inf_norm(rx.data(), n), inf_norm(rs.data(), m));
                for (int bt = 0; bt < 30; ++bt) {
                    for (int i = 0; i < m; ++i) { sn[i] = s[i] + alpha * ds[i]; zn[i] = z[i] + alpha * dz[i]; }
                    bool ok = exp_interior(sn, zn);
                    double mun = dot(sn.data(), zn.data(), m) / nuc;
                    double res_new = 1e100;
                    if (ok) {
                        std::vector<double> xn(n), rsn(m), rxn(n), tmp(n);
                        for (int i = 0; i < n; ++i) xn[i] = x[i] + alpha * dx[i];
                        A.matvec(xn.data(), rsn.data());
                        for (int i = 0; i < m; ++i) rsn[i] += sn[i] - b[i];
                        P.matvec(xn.data(), rxn.data());
                        A.tmatvec(zn.data(), tmp.data());
                        for (int i = 0; i < n; ++i) rxn[i] += q[i] + tmp[i];
                        res_new = std::max(inf_norm(rxn.data(), n), inf_norm(rsn.data(), m));
                    }
                    bool reduce = mun > 0 && (mun < mu * (1 - 1e-5 * alpha) || (res_new < res_old * 0.999 && mun < mu * 10) ||
                                             (mun < mu * 1.001 && res_new <= res_old * 1.001));
                    if (ok && reduce) {
                        for (int i = 0; i < n; ++i) x[i] += alpha * dx[i];
                        s.swap(sn); z.swap(zn);
                        return true;
                    }
                    alpha *= 0.5;
                    if (alpha < 1e-12) break;
                }
                return false;
            };
            bool ok = try_step(dxcc, dscc, dzcc);
            if (!ok) ok = try_step(dxaff, dsaff, dzaff);
            if (debug) fprintf(stderr, "iter %d mu %.3e sig %.3e reg %.3e aaff %.3e ok %d rx %.3e rs %.3e\n",
                               iter, mu, sigma, sigma_reg, alpha_aff, ok, inf_norm(rx.data(), n), inf_norm(rs.data(), m));
            if (!ok) break;
            if (iter % 2 == 1 || mu < 1e-6) {
                double tol = 9e-9;
                std::vector<double> xo(n), zo(mo), so;
                for (int i = 0; i < n; ++i) xo[i] = E[i] * x[i];
                for (int i = 0; i < m; ++i) {
                    if (aug_to_orig[i] >= 0) zo[aug_to_orig[i]] = D[i] * z[i];
                }
                for (int i = 0; i < (int)zero_plus.size(); ++i) if (zero_plus[i] >= 0) {
                    int p = zero_plus[i], mm2 = zero_minus[i];
                    zo[i] = D[p] * z[p] - D[mm2] * z[mm2];
                }
                if (check_solved(origP, origA, origq, origb, orig_cones, xo, zo, so, tol)) {
                    res.status = "solved"; res.x = xo; res.s = so; res.z = zo; return res;
                }
                std::vector<double> zc(mo, 0.0), xc(n), sc;
                for (int i = 0; i < m; ++i) if (aug_to_orig[i] >= 0) zc[aug_to_orig[i]] = D[i] * z[i];
                for (int i = 0; i < (int)zero_plus.size(); ++i) if (zero_plus[i] >= 0) {
                    int p = zero_plus[i], mm2 = zero_minus[i];
                    zc[i] = D[p] * z[p] - D[mm2] * z[mm2];
                }
                if (check_primal_cert(origA, origb, orig_cones, zc, tol)) {
                    res.status = "primal_infeasible"; res.z = zc; return res;
                }
                for (int i = 0; i < n; ++i) xc[i] = E[i] * x[i];
                if (check_dual_cert(origP, origA, origq, orig_cones, xc, sc, tol)) {
                    res.status = "dual_infeasible"; res.x = xc; res.s = sc; return res;
                }
            }
        }
        return res;
    }
};

static Result solve_with_split(const CSC& P, const CSC& A, const std::vector<double>& q,
                               const std::vector<double>& b, const std::vector<Cone>& cones) {
    int n = q.size(), m = b.size();
    CSR Acsr; Acsr.from_csc(A);
    Solver sol;
    sol.origA = A; sol.origP = P; sol.origq = q; sol.origb = b; sol.orig_cones = cones;
    sol.n = n;
    sol.map_row.assign(m, -1);
    sol.aug_to_orig.clear();
    sol.zero_plus.assign(m, -1); sol.zero_minus.assign(m, -1);
    std::vector<std::vector<std::pair<int,double>>> cols(n);
    std::vector<double> baug;
    std::vector<Block> blocks;
    int row = 0;
    for (auto& c : cones) {
        if (c.type == ZERO) {
            int pstart = row;
            for (int r = c.start; r < c.start + c.width; ++r) {
                sol.zero_plus[r] = row;
                sol.aug_to_orig.push_back(-1);
                baug.push_back(b[r]);
                for (int k = Acsr.indptr[r]; k < Acsr.indptr[r + 1]; ++k) cols[Acsr.indices[k]].push_back({row, Acsr.data[k]});
                ++row;
            }
            blocks.push_back({NONNEG, pstart, c.width, c.width, {}, {}});
            int mstart = row;
            for (int r = c.start; r < c.start + c.width; ++r) {
                sol.zero_minus[r] = row;
                sol.aug_to_orig.push_back(-1);
                baug.push_back(-b[r]);
                for (int k = Acsr.indptr[r]; k < Acsr.indptr[r + 1]; ++k) cols[Acsr.indices[k]].push_back({row, -Acsr.data[k]});
                ++row;
            }
            blocks.push_back({NONNEG, mstart, c.width, c.width, {}, {}});
        } else {
            int start = row;
            for (int r = c.start; r < c.start + c.width; ++r) {
                sol.map_row[r] = row;
                sol.aug_to_orig.push_back(r);
                baug.push_back(b[r]);
                for (int k = Acsr.indptr[r]; k < Acsr.indptr[r + 1]; ++k) cols[Acsr.indices[k]].push_back({row, Acsr.data[k]});
                ++row;
            }
            blocks.push_back({c.type, start, c.dim, c.width, {}, {}});
        }
    }
    sol.m = row;
    sol.blocks = blocks;
    sol.A.rows = row; sol.A.cols = n; sol.A.indptr.assign(n + 1, 0);
    for (int j = 0; j < n; ++j) {
        std::sort(cols[j].begin(), cols[j].end());
        for (auto& pr : cols[j]) { sol.A.indices.push_back(pr.first); sol.A.data.push_back(pr.second); }
        sol.A.indptr[j + 1] = sol.A.indices.size();
    }
    sol.b = baug;
    sol.q = q;
    sol.P = P;
    sol.E.assign(n, 1.0); sol.D.assign(row, 1.0);
    CSR Aaugcsr; Aaugcsr.from_csc(sol.A);
    int ruiz_iters = 5;
    for (auto& bl : sol.blocks) if (bl.type == PSD || bl.type == SOC || bl.type == EXP) ruiz_iters = 1;
    for (int it = 0; it < ruiz_iters; ++it) {
        std::vector<double> cn(n, 1.0), rn(row, 1.0);
        for (int j = 0; j < n; ++j) {
            for (int k = sol.A.indptr[j]; k < sol.A.indptr[j + 1]; ++k)
                cn[j] = std::max(cn[j], std::fabs(sol.A.data[k] * sol.D[sol.A.indices[k]] * sol.E[j]));
            for (int k = P.indptr[j]; k < P.indptr[j + 1]; ++k)
                cn[j] = std::max(cn[j], std::fabs(P.data[k] * sol.E[P.indices[k]] * sol.E[j]));
        }
        for (int r = 0; r < row; ++r) {
            rn[r] = std::max(1.0, std::fabs(baug[r] * sol.D[r]));
            for (int k = Aaugcsr.indptr[r]; k < Aaugcsr.indptr[r + 1]; ++k)
                rn[r] = std::max(rn[r], std::fabs(Aaugcsr.data[k] * sol.D[r] * sol.E[Aaugcsr.indices[k]]));
        }
        for (auto& bl : sol.blocks) if (bl.type != NONNEG) {
            double mx = 1.0;
            for (int r = bl.start; r < bl.start + bl.width; ++r) mx = std::max(mx, rn[r]);
            for (int r = bl.start; r < bl.start + bl.width; ++r) rn[r] = mx;
        }
        for (int j = 0; j < n; ++j) sol.E[j] /= std::sqrt(cn[j]);
        for (int r = 0; r < row; ++r) sol.D[r] /= std::sqrt(rn[r]);
        if (getenv("CONIC_DEBUG") && row > 1) fprintf(stderr, "ruiz %d D1 %.3e rn1 %.3e cn0 %.3e\n", it, sol.D[1], rn[1], cn[0]);
    }
    for (auto& bl : sol.blocks) if (bl.type != NONNEG) {
        double mx = 0;
        for (int r = bl.start; r < bl.start + bl.width; ++r) mx = std::max(mx, sol.D[r]);
        for (int r = bl.start; r < bl.start + bl.width; ++r) sol.D[r] = mx;
    }
    for (int j = 0; j < n; ++j) {
        for (int k = sol.A.indptr[j]; k < sol.A.indptr[j + 1]; ++k) sol.A.data[k] *= sol.D[sol.A.indices[k]] * sol.E[j];
        for (int k = sol.P.indptr[j]; k < sol.P.indptr[j + 1]; ++k) sol.P.data[k] *= sol.E[sol.P.indices[k]] * sol.E[j];
        sol.q[j] *= sol.E[j];
    }
    for (int i = 0; i < row; ++i) sol.b[i] *= sol.D[i];
    if (getenv("CONIC_DEBUG")) {
        double emin=1e100,emax=0,dmin=1e100,dmax=0,bmax=0,qmax=0;
        for (double v: sol.E){emin=std::min(emin,v);emax=std::max(emax,v);}
        for (double v: sol.D){dmin=std::min(dmin,v);dmax=std::max(dmax,v);}
        for (double v: sol.b)bmax=std::max(bmax,std::fabs(v));
        for (double v: sol.q)qmax=std::max(qmax,std::fabs(v));
        fprintf(stderr,"scale E %.3e %.3e D %.3e %.3e b %.3e q %.3e\n",emin,emax,dmin,dmax,bmax,qmax);
        for (int i=0;i<row;++i) if (!std::isfinite(sol.b[i])||!std::isfinite(sol.D[i])) { fprintf(stderr,"bad row %d b %.3e D %.3e\n",i,sol.b[i],sol.D[i]); break; }
        for (int i=0;i<n;++i) if (!std::isfinite(sol.E[i])||!std::isfinite(sol.q[i])) { fprintf(stderr,"bad col %d E %.3e q %.3e\n",i,sol.E[i],sol.q[i]); break; }
    }
    sol.nu = 1;
    bool has_large_psd = false;
    bool has_exp = false, has_soc = false;
    for (auto& bl : sol.blocks) {
        if (bl.type == NONNEG) sol.nu += bl.width;
        else if (bl.type == SOC) { sol.nu += 1; has_soc = true; }
        else if (bl.type == EXP) { sol.nu += 3; has_exp = true; }
        else { sol.nu += bl.dim; if (bl.dim >= 20) has_large_psd = true; }
    }
    if (has_large_psd) sol.budget = 15.0;
    else if (has_exp) sol.budget = 0.8;
    else if (has_soc) sol.budget = 0.6;
    Result result;
    if (n == 0 || row == 0) {
        result.status = "unresolved";
        result.x.assign(n, 0); result.s.assign(m, 0); result.z.assign(m, 0);
    } else result = sol.run();
    if (result.status == "unresolved") {
        result.x.assign(n, 0); result.s.assign(m, 0); result.z.assign(m, 0);
    }
    if (result.status == "primal_infeasible") { result.x.assign(n, 0); result.s.assign(m, 0); }
    if (result.status == "dual_infeasible") result.z.assign(m, 0);
    return result;
}

static std::vector<unsigned char> npy_double(const std::vector<double>& v) {
    std::vector<unsigned char> out;
    std::string shape = "(" + std::to_string(v.size()) + ",)";
    std::string dict = "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape + ", }";
    while ((10 + dict.size() + 1) % 64 != 0) dict.push_back(' ');
    dict.push_back('\n');
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.insert(out.end(), magic, magic + 8);
    wr16(out, dict.size());
    out.insert(out.end(), dict.begin(), dict.end());
    const unsigned char* p = (const unsigned char*)v.data();
    out.insert(out.end(), p, p + v.size() * 8);
    return out;
}
static std::vector<unsigned char> npy_unicode(const std::string& s) {
    std::vector<unsigned char> out;
    std::string dict = "{'descr': '<U" + std::to_string(s.size()) + "', 'fortran_order': False, 'shape': (), }";
    while ((10 + dict.size() + 1) % 64 != 0) dict.push_back(' ');
    dict.push_back('\n');
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.insert(out.end(), magic, magic + 8);
    wr16(out, dict.size());
    out.insert(out.end(), dict.begin(), dict.end());
    for (char c : s) wr32(out, (unsigned char)c);
    return out;
}
static void write_npz(const std::string& path, const std::vector<std::pair<std::string, std::vector<unsigned char>>>& entries) {
    std::vector<unsigned char> out;
    std::vector<uint32_t> offsets, crcs, sizes;
    for (auto& e : entries) {
        offsets.push_back(out.size());
        uint32_t crc = crc32(0, e.second.data(), e.second.size());
        crcs.push_back(crc); sizes.push_back(e.second.size());
        wr32(out, 0x04034b50); wr16(out, 20); wr16(out, 0); wr16(out, 0); wr16(out, 0); wr16(out, 0);
        wr32(out, crc); wr32(out, e.second.size()); wr32(out, e.second.size());
        wr16(out, e.first.size()); wr16(out, 0);
        out.insert(out.end(), e.first.begin(), e.first.end());
        out.insert(out.end(), e.second.begin(), e.second.end());
    }
    uint32_t cd = out.size();
    for (size_t i = 0; i < entries.size(); ++i) {
        wr32(out, 0x02014b50); wr16(out, 20); wr16(out, 20); wr16(out, 0); wr16(out, 0); wr16(out, 0); wr16(out, 0);
        wr32(out, crcs[i]); wr32(out, sizes[i]); wr32(out, sizes[i]);
        wr16(out, entries[i].first.size()); wr16(out, 0); wr16(out, 0); wr16(out, 0); wr16(out, 0); wr32(out, 0); wr32(out, offsets[i]);
        out.insert(out.end(), entries[i].first.begin(), entries[i].first.end());
    }
    uint32_t cdsize = out.size() - cd;
    wr32(out, 0x06054b50); wr16(out, 0); wr16(out, 0); wr16(out, entries.size()); wr16(out, entries.size());
    wr32(out, cdsize); wr32(out, cd); wr16(out, 0);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output");
    f.write((const char*)out.data(), out.size());
}

int main(int argc, char** argv) {
    scipy_openblas_set_num_threads(1);
    std::string inpath, outpath;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--input") && i + 1 < argc) inpath = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) outpath = argv[++i];
    }
    if (inpath.empty() || outpath.empty()) return 1;
    Result result;
    try {
        auto npz = read_npz(inpath);
        std::string meta = get_meta(npz);
        auto cones = parse_cones(meta);
        auto q = get_f64(npz, "q"), b = get_f64(npz, "b");
        int n = q.size(), m = b.size();
        auto Pshape = get_int(npz, "P_shape"), Ashape = get_int(npz, "A_shape");
        CSC P, A;
        P.build(n, n, get_int(npz, "P_indptr"), get_int(npz, "P_indices"), get_f64(npz, "P_data"));
        A.build(m, n, get_int(npz, "A_indptr"), get_int(npz, "A_indices"), get_f64(npz, "A_data"));
        CSR Acsr; Acsr.from_csc(A);
        EqPresolve ep;
        Result eqcert;
        bool inconsistent = eliminate_equalities(P, Acsr, q, b, cones, ep, eqcert);
        if (inconsistent) {
            if (check_primal_cert(A, b, cones, eqcert.z, 1e-8)) result = eqcert;
            else result = solve_with_split(P, A, q, b, cones);
        } else if (ep.active) {
            Result rr = solve_with_split(ep.P, ep.A, ep.q, ep.b, ep.cones);
            if (rr.status == "solved") {
                std::vector<double> xo(n), zo, so;
                for (int i = 0; i < n; ++i) {
                    double v = ep.t[i];
                    for (int a = 0; a < ep.k; ++a) v += ep.T[(size_t)i * ep.k + a] * rr.x[a];
                    xo[i] = v;
                }
                zo = lift_z_solved(ep, P, q, xo, rr.z, m);
                if (check_solved(P, A, q, b, cones, xo, zo, so, 1e-8)) {
                    result.status = "solved"; result.x = xo; result.s = so; result.z = zo;
                }
            } else if (rr.status == "primal_infeasible") {
                std::vector<double> zo = lift_z(ep, rr.z, m);
                if (check_primal_cert(A, b, cones, zo, 1e-8)) {
                    result.status = "primal_infeasible"; result.z = zo; result.x.assign(n, 0); result.s.assign(m, 0);
                }
            } else if (rr.status == "dual_infeasible") {
                std::vector<double> xo(n, 0.0), so;
                for (int i = 0; i < n; ++i)
                    for (int a = 0; a < ep.k; ++a) xo[i] += ep.T[(size_t)i * ep.k + a] * rr.x[a];
                if (check_dual_cert(P, A, q, cones, xo, so, 1e-8)) {
                    result.status = "dual_infeasible"; result.x = xo; result.s = so; result.z.assign(m, 0);
                }
            }
            if (result.status.empty()) {
                result.status = "unresolved"; result.x.assign(n, 0); result.s.assign(m, 0); result.z.assign(m, 0);
            }
        } else {
            result = solve_with_split(P, A, q, b, cones);
        }
    } catch (...) {
        result.status = "unresolved";
    }
    std::string mj = "{\"schema_version\":1,\"status\":\"" + result.status + "\"}";
    std::vector<std::pair<std::string, std::vector<unsigned char>>> entries;
    entries.push_back({"metadata_json.npy", npy_unicode(mj)});
    entries.push_back({"x.npy", npy_double(result.x)});
    entries.push_back({"s.npy", npy_double(result.s)});
    entries.push_back({"z.npy", npy_double(result.z)});
    try { write_npz(outpath, entries); } catch (...) { return 1; }
    return 0;
}
