#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

extern "C" {
typedef unsigned char Bytef;
typedef unsigned int uInt;
typedef unsigned long uLong;
typedef void *(*alloc_func)(void *, uInt, uInt);
typedef void (*free_func)(void *, void *);
struct z_stream {
    Bytef *next_in;
    uInt avail_in;
    uLong total_in;
    Bytef *next_out;
    uInt avail_out;
    uLong total_out;
    char *msg;
    void *state;
    alloc_func zalloc;
    free_func zfree;
    void *opaque;
    int data_type;
    uLong adler;
    uLong reserved;
};
const char *zlibVersion();
int inflateInit2_(z_stream *, int, const char *, int);
int inflate(z_stream *, int);
int inflateEnd(z_stream *);
uLong crc32(uLong, const Bytef *, uInt);
}

using Bytes = std::vector<unsigned char>;
static constexpr double inf = std::numeric_limits<double>::infinity();

static uint16_t u16(const unsigned char *p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
static uint32_t u32(const unsigned char *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16)
           | (uint32_t(p[3]) << 24);
}
static uint64_t u64(const unsigned char *p) {
    return uint64_t(u32(p)) | (uint64_t(u32(p + 4)) << 32);
}
static void utf8(std::string &s, uint32_t c) {
    if (c < 128) s.push_back(char(c));
    else if (c < 2048) {
        s.push_back(char(192 | (c >> 6)));
        s.push_back(char(128 | (c & 63)));
    } else if (c < 65536) {
        s.push_back(char(224 | (c >> 12)));
        s.push_back(char(128 | ((c >> 6) & 63)));
        s.push_back(char(128 | (c & 63)));
    } else {
        s.push_back(char(240 | (c >> 18)));
        s.push_back(char(128 | ((c >> 12) & 63)));
        s.push_back(char(128 | ((c >> 6) & 63)));
        s.push_back(char(128 | (c & 63)));
    }
}

struct Value {
    std::string text;
    double number = 0;
    std::vector<Value> array;
    std::map<std::string, Value> object;
};

class Parser {
    const std::string &s;
    size_t pos = 0;
    bool python;
    void space() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }
    char take() {
        if (pos == s.size()) throw std::runtime_error("Truncated structured text");
        return s[pos++];
    }
    uint32_t hex() {
        uint32_t result = 0;
        for (int i = 0; i < 4; ++i) {
            char c = take();
            int v = c >= '0' && c <= '9' ? c - '0'
                    : c >= 'a' && c <= 'f' ? c - 'a' + 10
                    : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (v < 0) throw std::runtime_error("Bad Unicode escape");
            result = (result << 4) | uint32_t(v);
        }
        return result;
    }
    std::string string() {
        const char quote = take();
        std::string result;
        while (true) {
            char c = take();
            if (c == quote) return result;
            if (c != '\\') {
                result.push_back(c);
                continue;
            }
            c = take();
            if (c == 'u') {
                uint32_t cp = hex();
                if (cp >= 0xd800 && cp <= 0xdbff && pos + 6 <= s.size()
                    && s[pos] == '\\' && s[pos + 1] == 'u') {
                    pos += 2;
                    uint32_t low = hex();
                    if (low >= 0xdc00 && low <= 0xdfff)
                        cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                }
                utf8(result, cp);
            } else {
                switch (c) {
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                default: result.push_back(c); break;
                }
            }
        }
    }
    Value value(int depth) {
        if (depth > 1000) throw std::runtime_error("Excessive nesting");
        space();
        Value result;
        char c = pos < s.size() ? s[pos] : '\0';
        if (c == '"' || (python && c == '\'')) {
            result.text = string();
        } else if (c == '{') {
            ++pos;
            space();
            while (pos < s.size() && s[pos] != '}') {
                std::string key = string();
                space();
                if (take() != ':') throw std::runtime_error("Bad object");
                result.object.emplace(std::move(key), value(depth + 1));
                space();
                if (s[pos] == '}') break;
                if (take() != ',') throw std::runtime_error("Bad object delimiter");
                space();
            }
            if (take() != '}') throw std::runtime_error("Bad object end");
        } else if (c == '[' || (python && c == '(')) {
            ++pos;
            char end = c == '[' ? ']' : ')';
            space();
            while (pos < s.size() && s[pos] != end) {
                result.array.push_back(value(depth + 1));
                space();
                if (s[pos] == end) break;
                if (take() != ',') throw std::runtime_error("Bad array delimiter");
                space();
            }
            if (take() != end) throw std::runtime_error("Bad array end");
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            const char *begin = s.c_str() + pos;
            char *end = nullptr;
            result.number = std::strtod(begin, &end);
            if (end == begin) throw std::runtime_error("Bad number");
            pos = size_t(end - s.c_str());
        } else {
            size_t begin = pos;
            while (pos < s.size() && std::isalpha(static_cast<unsigned char>(s[pos]))) ++pos;
            std::string word = s.substr(begin, pos - begin);
            if (word == "true" || word == "True") result.number = 1;
            else if (word != "false" && word != "False" && word != "null"
                     && word != "None") throw std::runtime_error("Bad literal");
        }
        return result;
    }
public:
    Parser(const std::string &text, bool is_python = false): s(text), python(is_python) {}
    Value parse() {
        Value result = value(0);
        space();
        if (pos != s.size()) throw std::runtime_error("Trailing text");
        return result;
    }
};

struct Entry {
    uint64_t compressed = 0, size = 0, offset = 0;
    uint16_t method = 0;
};

class Archive {
    std::ifstream file;
    std::map<std::string, Entry> entries;
    Bytes read(uint64_t offset, size_t size) {
        Bytes result(size);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(size));
        if (file.gcount() != static_cast<std::streamsize>(size))
            throw std::runtime_error("Truncated ZIP");
        return result;
    }
public:
    explicit Archive(const std::string &path): file(path, std::ios::binary) {
        if (!file) throw std::runtime_error("Cannot read input");
        file.seekg(0, std::ios::end);
        uint64_t size = uint64_t(file.tellg());
        size_t tail_size = size_t(std::min<uint64_t>(size, 65557));
        Bytes tail = read(size - tail_size, tail_size);
        size_t end = tail.size();
        for (size_t i = tail.size() >= 22 ? tail.size() - 22 : 0; ; --i) {
            if (i + 22 <= tail.size() && u32(tail.data() + i) == 0x06054b50) {
                end = i;
                break;
            }
            if (i == 0) break;
        }
        if (end == tail.size()) throw std::runtime_error("Missing ZIP directory");
        uint32_t directory_size = u32(tail.data() + end + 12);
        uint32_t directory_offset = u32(tail.data() + end + 16);
        Bytes directory = read(directory_offset, directory_size);
        size_t pos = 0;
        while (pos + 46 <= directory.size()) {
            const unsigned char *p = directory.data() + pos;
            if (u32(p) != 0x02014b50) break;
            uint16_t name_size = u16(p + 28), extra_size = u16(p + 30);
            uint16_t comment_size = u16(p + 32);
            if (pos + 46 + name_size + extra_size + comment_size > directory.size())
                throw std::runtime_error("Truncated directory entry");
            Entry e;
            e.method = u16(p + 10);
            e.compressed = u32(p + 20);
            e.size = u32(p + 24);
            e.offset = u32(p + 42);
            const unsigned char *extra = p + 46 + name_size;
            for (size_t at = 0; at + 4 <= extra_size; ) {
                uint16_t id = u16(extra + at), length = u16(extra + at + 2);
                if (at + 4 + length > extra_size) throw std::runtime_error("Bad ZIP extra");
                if (id == 1) {
                    size_t q = at + 4;
                    if (e.size == 0xffffffff && q + 8 <= at + 4 + length) {
                        e.size = u64(extra + q); q += 8;
                    }
                    if (e.compressed == 0xffffffff && q + 8 <= at + 4 + length) {
                        e.compressed = u64(extra + q); q += 8;
                    }
                    if (e.offset == 0xffffffff && q + 8 <= at + 4 + length)
                        e.offset = u64(extra + q);
                }
                at += 4 + length;
            }
            std::string name(reinterpret_cast<const char *>(p + 46), name_size);
            entries.emplace(std::move(name), e);
            pos += 46 + name_size + extra_size + comment_size;
        }
    }
    Bytes get(const std::string &key, bool prefix = false) {
        const Entry &e = entries.at(key + ".npy");
        if (e.size > (uint64_t(1) << 30) || e.compressed > (uint64_t(1) << 30))
            throw std::runtime_error("Oversized entry");
        Bytes local = read(e.offset, 30);
        if (u32(local.data()) != 0x04034b50) throw std::runtime_error("Bad local header");
        uint64_t offset = e.offset + 30 + u16(local.data() + 26) + u16(local.data() + 28);
        size_t csize = size_t(prefix ? std::min<uint64_t>(e.compressed, 8192) : e.compressed);
        size_t usize = size_t(prefix ? std::min<uint64_t>(e.size, 4096) : e.size);
        Bytes compressed = read(offset, csize);
        if (e.method == 0) {
            if (prefix && compressed.size() > usize) compressed.resize(usize);
            return compressed;
        }
        if (e.method != 8) throw std::runtime_error("Unsupported ZIP compression");
        Bytes result(usize);
        z_stream stream{};
        stream.next_in = compressed.data();
        stream.avail_in = static_cast<uInt>(compressed.size());
        stream.next_out = result.data();
        stream.avail_out = static_cast<uInt>(result.size());
        int status = inflateInit2_(&stream, -15, zlibVersion(), sizeof(z_stream));
        if (status != 0) throw std::runtime_error("Cannot initialize inflater");
        status = inflate(&stream, 4);
        size_t actual = size_t(stream.total_out);
        inflateEnd(&stream);
        if ((!prefix && (status != 1 || actual != usize)) || (prefix && actual < 10))
            throw std::runtime_error("Cannot inflate NPZ entry");
        result.resize(actual);
        return result;
    }
};

struct Array {
    Bytes storage;
    std::string dtype;
    std::vector<size_t> shape;
    size_t offset = 0, count = 1, width = 0;
    explicit Array(Bytes bytes, bool prefix = false): storage(std::move(bytes)) {
        if (storage.size() < 10 || std::memcmp(storage.data(), "\x93NUMPY", 6) != 0)
            throw std::runtime_error("Bad NPY header");
        unsigned major = storage[6];
        size_t start = major == 1 ? 10 : 12;
        size_t length = major == 1 ? u16(storage.data() + 8) : u32(storage.data() + 8);
        if (start + length > storage.size()) throw std::runtime_error("Truncated NPY header");
        std::string text(reinterpret_cast<const char *>(storage.data() + start), length);
        Value header = Parser(text, true).parse();
        dtype = header.object.at("descr").text;
        if (dtype.size() < 3) throw std::runtime_error("Bad dtype");
        width = size_t(std::stoul(dtype.substr(2)));
        if (dtype[1] == 'U') width *= 4;
        for (const Value &v : header.object.at("shape").array) {
            size_t d = size_t(v.number);
            shape.push_back(d);
            if (d != 0 && count > (uint64_t(1) << 30) / d)
                throw std::runtime_error("Oversized shape");
            count *= d;
        }
        offset = start + length;
        if (!prefix && offset + count * width != storage.size())
            throw std::runtime_error("Wrong array payload size");
    }
    size_t item_size() const {
        return width;
    }
    uint64_t bits(size_t i) const {
        const unsigned char *p = storage.data() + offset + i * width;
        if (dtype[0] != '>' && width == 4) return u32(p);
        if (dtype[0] != '>' && width == 8) return u64(p);
        uint64_t result = 0;
        if (dtype[0] == '>') {
            for (size_t j = 0; j < width; ++j) result = (result << 8) | p[j];
        } else {
            for (size_t j = 0; j < width; ++j) result |= uint64_t(p[j]) << (8 * j);
        }
        return result;
    }
    int64_t integer(size_t i) const {
        if (dtype[1] != 'i' && dtype[1] != 'u') throw std::runtime_error("Not integral");
        uint64_t value = bits(i);
        if (dtype[1] == 'i' && item_size() == 4) return int32_t(value);
        return int64_t(value);
    }
    std::vector<double> doubles() const {
        if (dtype[1] != 'f' || item_size() != 8) throw std::runtime_error("Not float64");
        std::vector<double> result(count);
        if (dtype[0] != '>') {
            std::memcpy(result.data(), storage.data() + offset, count * sizeof(double));
        } else {
            for (size_t i = 0; i < count; ++i) {
                uint64_t value = bits(i);
                std::memcpy(&result[i], &value, sizeof(double));
            }
        }
        return result;
    }
    std::string unicode() const {
        if (dtype[1] != 'U' || !shape.empty()) throw std::runtime_error("Not Unicode scalar");
        std::string result;
        const unsigned char *p = storage.data() + offset;
        for (size_t i = 0; i < item_size() / 4; ++i) {
            uint32_t cp;
            if (dtype[0] == '>') cp = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16)
                                      | (uint32_t(p[4*i+2]) << 8) | p[4*i+3];
            else cp = u32(p + 4*i);
            if (!cp) break;
            utf8(result, cp);
        }
        return result;
    }
};

struct Cone { bool equal; size_t start, dim; };
struct Result {
    std::string status;
    std::vector<double> x, s, z;
};
static double norm(const std::vector<double> &v) {
    double result = 0;
    for (double x : v) result = std::max(result, std::abs(x));
    return result;
}
static long double dot(const std::vector<double> &a, const std::vector<double> &b) {
    long double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += (long double)a[i] * b[i];
    return sum;
}
static double defect(const std::vector<double> &v, const std::vector<Cone> &cones, bool dual) {
    double worst = 0;
    for (const Cone &cone : cones) {
        if (cone.equal && dual) continue;
        double scale = 1, error = 0;
        for (size_t i = cone.start; i < cone.start + cone.dim; ++i) {
            scale = std::max(scale, std::abs(v[i]));
            error = std::max(error, cone.equal ? std::abs(v[i]) : -v[i]);
        }
        worst = std::max(worst, error / scale);
    }
    return worst;
}

static bool separable(Archive &archive, Result &result) {
    Array meta_header(archive.get("metadata_json", true), true);
    if (meta_header.item_size() > 65536) return false;
    Value meta = Parser(Array(archive.get("metadata_json")).unicode()).parse();
    if (meta.object.at("schema_version").number != 1) return false;
    std::vector<Cone> cones;
    size_t row_count = 0;
    for (const Value &v : meta.object.at("cones").array) {
        std::string kind = v.object.at("type").text;
        if (kind != "zero" && kind != "nonnegative") return false;
        size_t dim = size_t(v.object.at("dim").number);
        if (!dim) return false;
        cones.push_back({kind == "zero", row_count, dim});
        row_count += dim;
    }
    Array shape(archive.get("A_shape"));
    if (shape.count != 2) return false;
    size_t m = size_t(shape.integer(0)), n = size_t(shape.integer(1));
    if (m != row_count) return false;
    Array a_header(archive.get("A_data", true), true);
    Array p_header(archive.get("P_data", true), true);
    if (a_header.count > m || p_header.count > n) return false;
    std::vector<double> q = Array(archive.get("q")).doubles();
    std::vector<double> b = Array(archive.get("b")).doubles();
    if (q.size() != n || b.size() != m) return false;
    std::vector<double> pd = Array(archive.get("P_data")).doubles();
    Array pi(archive.get("P_indices")), pp(archive.get("P_indptr"));
    std::vector<double> p(n, 0);
    if (pp.count != n + 1 || pi.count != pd.size()) return false;
    for (size_t j = 0; j < n; ++j) {
        for (int64_t at = pp.integer(j); at < pp.integer(j + 1); ++at) {
            if (pd[size_t(at)] == 0) continue;
            if (pi.integer(size_t(at)) != int64_t(j)) return false;
            p[j] += pd[size_t(at)];
        }
        if (p[j] < 0) return false;
    }
    std::vector<double> ad = Array(archive.get("A_data")).doubles();
    Array ai(archive.get("A_indices")), ap(archive.get("A_indptr"));
    if (ap.count != n + 1 || ai.count != ad.size()) return false;
    std::vector<int64_t> owner(m, -1);
    std::vector<double> a(m, 0);
    for (size_t j = 0; j < n; ++j) {
        for (int64_t at = ap.integer(j); at < ap.integer(j + 1); ++at) {
            double value = ad[size_t(at)];
            if (value == 0) continue;
            size_t row = size_t(ai.integer(size_t(at)));
            if (row >= m) return false;
            if (owner[row] != -1 && owner[row] != int64_t(j)) return false;
            owner[row] = int64_t(j);
            a[row] += value;
        }
    }
    std::vector<unsigned char> equal(m, 0);
    for (const Cone &cone : cones)
        if (cone.equal) std::fill(equal.begin() + cone.start, equal.begin() + cone.start + cone.dim, 1);
    auto multiply = [&](const std::vector<double> &x) {
        std::vector<double> out(m);
        for (size_t i = 0; i < m; ++i) if (a[i] != 0) out[i] = a[i] * x[size_t(owner[i])];
        return out;
    };
    auto transpose = [&](const std::vector<double> &z, double scale) {
        std::vector<double> out(n, 0);
        for (size_t i = 0; i < m; ++i)
            if (a[i] != 0) out[size_t(owner[i])] += (a[i] / scale) * z[i];
        return out;
    };
    auto finite = [](const std::vector<double> &v) {
        for (double x : v) if (!std::isfinite(x)) return false;
        return true;
    };
    auto verified = [&](const Result &r) {
        constexpr double tol = 7e-9;
        if (!finite(r.x) || !finite(r.s) || !finite(r.z)) return false;
        double ascale = norm(a), pscale = norm(p);
        if (!ascale) ascale = 1;
        if (!pscale) pscale = 1;
        if (r.status == "solved") {
            std::vector<double> Ax = multiply(r.x), Atz = transpose(r.z, 1), Px(n);
            double rp = 0, rd = 0;
            for (size_t i = 0; i < m; ++i)
                rp = std::max(rp, std::abs(Ax[i] + r.s[i] - b[i]));
            for (size_t j = 0; j < n; ++j) {
                Px[j] = p[j] * r.x[j];
                rd = std::max(rd, std::abs(Px[j] + q[j] + Atz[j]));
            }
            rp /= 1 + std::max({norm(Ax), norm(r.s), norm(b)});
            rd /= 1 + std::max({norm(Px), norm(q), norm(Atz)});
            long double xPx = dot(r.x, Px), qx = dot(q, r.x), bz = dot(b, r.z);
            long double primal = xPx / 2 + qx, dual = -xPx / 2 - bz;
            long double gap = std::abs(primal - dual);
            return finite(Px) && finite(Ax) && finite(Atz) && std::isfinite(primal)
                && std::isfinite(dual) && std::isfinite(dot(r.s, r.z))
                && std::isfinite(double(xPx)) && std::isfinite(double(qx))
                && std::isfinite(double(bz)) && std::isfinite(double(primal))
                && std::isfinite(double(dual)) && std::isfinite(double(dot(r.s, r.z)))
                && rp <= tol && rd <= tol
                && (gap <= tol || gap <= tol * std::max(std::abs(primal), std::abs(dual)))
                && defect(r.s, cones, false) <= tol && defect(r.z, cones, true) <= tol;
        }
        if (r.status == "primal_infeasible") {
            double zn = norm(r.z), bn = norm(b);
            if (!zn || !bn) return false;
            std::vector<double> z(r.z), bs(b);
            for (double &v : z) v /= zn;
            for (double &v : bs) v /= bn;
            long double margin = -dot(bs, z);
            return margin > 1e-8 && norm(transpose(z, ascale)) <= tol * margin
                   && defect(z, cones, true) <= tol * margin;
        }
        double xn = norm(r.x), qn = norm(q);
        if (!xn || !qn) return false;
        std::vector<double> x(r.x), s(r.s), qs(q), Px(n);
        for (double &v : x) v /= xn;
        for (double &v : s) v /= xn * ascale;
        for (double &v : qs) v /= qn;
        std::vector<double> Ax = multiply(x);
        for (size_t i = 0; i < m; ++i) Ax[i] = Ax[i] / ascale + s[i];
        for (size_t j = 0; j < n; ++j) Px[j] = (p[j] / pscale) * x[j];
        long double margin = -dot(qs, x);
        return margin > 1e-8 && norm(Px) <= tol * margin && norm(Ax) <= tol * margin
               && defect(s, cones, false) <= tol * margin;
    };
    Result cert;
    cert.status = "primal_infeasible";
    cert.z.assign(m, 0);
    for (size_t i = 0; i < m; ++i) if (a[i] == 0) {
        if (equal[i] && b[i] != 0) cert.z[i] = b[i] > 0 ? -1 : 1;
        else if (!equal[i] && b[i] < 0) cert.z[i] = 1;
    }
    if (verified(cert)) { result = std::move(cert); return true; }
    std::vector<double> lo(n, -inf), hi(n, inf);
    std::vector<int64_t> lr(n, -1), ur(n, -1);
    for (size_t i = 0; i < m; ++i) {
        if (a[i] == 0) continue;
        size_t j = size_t(owner[i]);
        double bound = b[i] / a[i];
        if ((equal[i] || a[i] < 0) && bound >= lo[j]) { lo[j] = bound; lr[j] = int64_t(i); }
        if ((equal[i] || a[i] > 0) && bound <= hi[j]) { hi[j] = bound; ur[j] = int64_t(i); }
    }
    size_t conflict = n;
    double separation = 0;
    for (size_t j = 0; j < n; ++j)
        if (lo[j] > hi[j] && lo[j] - hi[j] > separation) {
            conflict = j; separation = lo[j] - hi[j];
        }
    if (conflict != n) {
        size_t l = size_t(lr[conflict]), u = size_t(ur[conflict]);
        double scale = std::max(std::abs(a[l]), std::abs(a[u]));
        cert.z.assign(m, 0);
        cert.z[l] += -(a[l] > 0 ? 1 : -1) * std::abs(a[u]) / scale;
        cert.z[u] += (a[u] > 0 ? 1 : -1) * std::abs(a[l]) / scale;
        if (verified(cert)) { result = std::move(cert); return true; }
    }
    Result direction;
    direction.status = "dual_infeasible";
    direction.x.assign(n, 0);
    for (size_t j = 0; j < n; ++j) if (p[j] == 0) {
        if (q[j] > 0 && lo[j] == -inf) direction.x[j] = -1;
        if (q[j] < 0 && hi[j] == inf) direction.x[j] = 1;
    }
    direction.s = multiply(direction.x);
    for (double &v : direction.s) v = -v;
    if (verified(direction)) { result = std::move(direction); return true; }
    result.status = "solved";
    result.x.assign(n, 0);
    result.z.assign(m, 0);
    for (size_t j = 0; j < n; ++j) {
        double x = p[j] > 0 ? -q[j] / p[j] : q[j] > 0 ? lo[j] : q[j] < 0 ? hi[j] : 0;
        x = std::min(std::max(x, lo[j]), hi[j]);
        if (lo[j] > hi[j]) x = lo[j] / 2 + hi[j] / 2;
        result.x[j] = x;
        double gradient = p[j] * x + q[j];
        if (x == lo[j] && gradient > 0 && lr[j] >= 0)
            result.z[size_t(lr[j])] = -gradient / a[size_t(lr[j])];
        else if (x == hi[j] && gradient < 0 && ur[j] >= 0)
            result.z[size_t(ur[j])] = -gradient / a[size_t(ur[j])];
    }
    result.s = multiply(result.x);
    for (size_t i = 0; i < m; ++i)
        result.s[i] = equal[i] ? 0.0 : std::max(b[i] - result.s[i], 0.0);
    return verified(result);
}

static void put16(std::ostream &out, uint16_t v) {
    char bytes[2] = {char(v), char(v >> 8)};
    out.write(bytes, 2);
}
static void put32(std::ostream &out, uint32_t v) {
    char bytes[4] = {char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
    out.write(bytes, 4);
}
static Bytes npy_header(const std::string &dtype, const std::string &shape) {
    std::string h = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': " + shape + ", }";
    size_t padding = (16 - ((10 + h.size() + 1) % 16)) % 16;
    h.append(padding, ' ');
    h.push_back('\n');
    Bytes result(10 + h.size());
    std::memcpy(result.data(), "\x93NUMPY\x01\x00", 8);
    result[8] = static_cast<unsigned char>(h.size());
    result[9] = static_cast<unsigned char>(h.size() >> 8);
    std::memcpy(result.data() + 10, h.data(), h.size());
    return result;
}
static void save(const std::string &path, const Result &result) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write output");
    struct Record { std::string name; uint32_t crc, size, offset; };
    std::vector<Record> records;
    auto add = [&](const std::string &name, const Bytes &header, const void *data, size_t bytes) {
        uint32_t checksum = uint32_t(crc32(0, header.data(), static_cast<uInt>(header.size())));
        if (bytes) checksum = uint32_t(crc32(checksum, static_cast<const Bytef *>(data), static_cast<uInt>(bytes)));
        Record r{name + ".npy", checksum, uint32_t(header.size() + bytes), uint32_t(out.tellp())};
        put32(out, 0x04034b50); put16(out, 20); put16(out, 0); put16(out, 0);
        put16(out, 0); put16(out, 0); put32(out, r.crc); put32(out, r.size); put32(out, r.size);
        put16(out, uint16_t(r.name.size())); put16(out, 0);
        out.write(r.name.data(), static_cast<std::streamsize>(r.name.size()));
        out.write(reinterpret_cast<const char *>(header.data()), static_cast<std::streamsize>(header.size()));
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        records.push_back(std::move(r));
    };
    auto vector = [&](const std::string &name, const std::vector<double> &v) {
        Bytes header = npy_header("<f8", "(" + std::to_string(v.size()) + ",)");
        add(name, header, v.data(), v.size() * sizeof(double));
    };
    if (result.status == "solved" || result.status == "dual_infeasible") {
        vector("x", result.x); vector("s", result.s);
    }
    if (result.status == "solved" || result.status == "primal_infeasible") vector("z", result.z);
    std::string text = "{\"schema_version\":1,\"status\":\"" + result.status + "\"}";
    std::vector<uint32_t> unicode;
    for (unsigned char c : text) unicode.push_back(c);
    add("metadata_json", npy_header("<U" + std::to_string(text.size()), "()"),
        unicode.data(), unicode.size() * sizeof(uint32_t));
    uint32_t directory = uint32_t(out.tellp());
    for (const Record &r : records) {
        put32(out, 0x02014b50); put16(out, 20); put16(out, 20); put16(out, 0);
        put16(out, 0); put16(out, 0); put16(out, 0); put32(out, r.crc);
        put32(out, r.size); put32(out, r.size); put16(out, uint16_t(r.name.size()));
        put16(out, 0); put16(out, 0); put16(out, 0); put16(out, 0); put32(out, 0); put32(out, r.offset);
        out.write(r.name.data(), static_cast<std::streamsize>(r.name.size()));
    }
    uint32_t end = uint32_t(out.tellp());
    put32(out, 0x06054b50); put16(out, 0); put16(out, 0);
    put16(out, uint16_t(records.size())); put16(out, uint16_t(records.size()));
    put32(out, end - directory); put32(out, directory); put16(out, 0);
    if (!out) throw std::runtime_error("Incomplete output");
}

int main(int argc, char **argv) {
    std::string input, output;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--input") input = argv[++i];
        else if (std::string(argv[i]) == "--output") output = argv[++i];
    }
    try {
        if (!input.empty() && !output.empty()) {
            Archive archive(input);
            Result result;
            if (separable(archive, result)) {
                save(output, result);
                return 0;
            }
        }
    } catch (const std::exception &) {
        // Unsupported encodings and numerical difficulties use the general path.
    }
    std::string own = argv[0];
    size_t slash = own.find_last_of('/');
    std::string script = (slash == std::string::npos ? "." : own.substr(0, slash)) + "/main.py";
    std::vector<char *> args;
    args.push_back(const_cast<char *>("python"));
    args.push_back(script.data());
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
    args.push_back(nullptr);
    execvp("python", args.data());
    return 1;
}
