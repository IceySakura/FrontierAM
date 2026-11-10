// NPZ (zip + .npy) reading/writing and a minimal JSON parser.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

typedef int64_t I;

extern "C" {
struct zs_t {
  const unsigned char *next_in; unsigned avail_in; unsigned long total_in;
  unsigned char *next_out; unsigned avail_out; unsigned long total_out;
  const char *msg; void *state; void *zalloc; void *zfree; void *opaque;
  int data_type; unsigned long adler; unsigned long reserved;
};
int inflateInit2_(zs_t *, int, const char *, int);
int inflate(zs_t *, int);
int inflateEnd(zs_t *);
}

struct NpyArr {
  std::string descr;
  std::vector<I> shape;
  std::vector<unsigned char> raw;  // whole .npy file
  size_t data_off = 0;
  I count() const { I c = 1; for (I s : shape) c *= s; return c; }
};

static inline uint16_t rd16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t rd32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t rd64(const unsigned char *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

static void parse_npy(NpyArr &a) {
  const unsigned char *r = a.raw.data();
  if (a.raw.size() < 10 || r[0] != 0x93 || memcmp(r + 1, "NUMPY", 5) != 0) throw std::runtime_error("bad npy");
  int major = r[6];
  size_t hl, ho;
  if (major == 1) { hl = rd16(r + 8); ho = 10; } else { hl = rd32(r + 8); ho = 12; }
  std::string h((const char *)r + ho, hl);
  size_t p = h.find("descr");
  if (p == std::string::npos) throw std::runtime_error("npy descr");
  p = h.find(':', p);
  size_t q1 = h.find_first_of("'\"", p);
  size_t q2 = h.find_first_of("'\"", q1 + 1);
  a.descr = h.substr(q1 + 1, q2 - q1 - 1);
  p = h.find("shape");
  p = h.find('(', p);
  size_t e = h.find(')', p);
  std::string sh = h.substr(p + 1, e - p - 1);
  a.shape.clear();
  size_t i = 0;
  while (i < sh.size()) {
    while (i < sh.size() && (sh[i] < '0' || sh[i] > '9')) i++;
    if (i >= sh.size()) break;
    I v = 0;
    while (i < sh.size() && sh[i] >= '0' && sh[i] <= '9') { v = v * 10 + (sh[i] - '0'); i++; }
    a.shape.push_back(v);
  }
  a.data_off = ho + hl;
}

static void read_npz(const char *path, std::map<std::string, NpyArr> &out) {
  FILE *f = fopen(path, "rb");
  if (!f) throw std::runtime_error("cannot open input");
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> buf(sz);
  if (sz > 0 && fread(buf.data(), 1, sz, f) != (size_t)sz) { fclose(f); throw std::runtime_error("read"); }
  fclose(f);
  long e = -1;
  for (long i = sz - 22; i >= 0 && i >= sz - 22 - 70000; i--)
    if (rd32(&buf[i]) == 0x06054b50) { e = i; break; }
  if (e < 0) throw std::runtime_error("no zip eocd");
  uint64_t cnt = rd16(&buf[e + 10]);
  uint64_t cdoff = rd32(&buf[e + 16]);
  if ((cdoff == 0xFFFFFFFFu || cnt == 0xFFFF) && e >= 20 && rd32(&buf[e - 20]) == 0x07064b50) {
    uint64_t z64 = rd64(&buf[e - 20 + 8]);
    cnt = rd64(&buf[z64 + 32]);
    cdoff = rd64(&buf[z64 + 48]);
  }
  size_t p = cdoff;
  for (uint64_t k = 0; k < cnt; k++) {
    if (rd32(&buf[p]) != 0x02014b50) throw std::runtime_error("bad central dir");
    int method = rd16(&buf[p + 10]);
    uint64_t csize = rd32(&buf[p + 20]), usize = rd32(&buf[p + 24]);
    int nl = rd16(&buf[p + 28]), xl = rd16(&buf[p + 30]), cl = rd16(&buf[p + 32]);
    uint64_t lho = rd32(&buf[p + 42]);
    std::string name((const char *)&buf[p + 46], nl);
    if (csize == 0xFFFFFFFFu || usize == 0xFFFFFFFFu || lho == 0xFFFFFFFFu) {
      size_t x = p + 46 + nl, xe = x + xl;
      while (x + 4 <= xe) {
        int id = rd16(&buf[x]), len = rd16(&buf[x + 2]);
        if (id == 1) {
          size_t y = x + 4;
          if (usize == 0xFFFFFFFFu) { usize = rd64(&buf[y]); y += 8; }
          if (csize == 0xFFFFFFFFu) { csize = rd64(&buf[y]); y += 8; }
          if (lho == 0xFFFFFFFFu) { lho = rd64(&buf[y]); y += 8; }
        }
        x += 4 + len;
      }
    }
    size_t lp = lho;
    size_t dstart = lp + 30 + rd16(&buf[lp + 26]) + rd16(&buf[lp + 28]);
    NpyArr a;
    if (method == 0) {
      a.raw.assign(buf.begin() + dstart, buf.begin() + dstart + usize);
    } else if (method == 8) {
      a.raw.resize(usize);
      zs_t st;
      memset(&st, 0, sizeof st);
      if (inflateInit2_(&st, -15, "1.2.13", (int)sizeof(zs_t)) != 0) throw std::runtime_error("zlib init");
      st.next_in = &buf[dstart];
      st.avail_in = (unsigned)csize;
      st.next_out = a.raw.data();
      st.avail_out = (unsigned)usize;
      int r = inflate(&st, 4);
      inflateEnd(&st);
      if (r != 1) throw std::runtime_error("inflate failed");
    } else throw std::runtime_error("unsupported zip method");
    parse_npy(a);
    if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") name = name.substr(0, name.size() - 4);
    out[name] = std::move(a);
    p += 46 + nl + xl + cl;
  }
}

static std::vector<double> npy_f64(const NpyArr &a) {
  I c = a.count();
  std::vector<double> v(c);
  const unsigned char *d = a.raw.data() + a.data_off;
  const std::string &t = a.descr;
  char kind = t.size() > 1 ? t[1] : 'f';
  int sz = atoi(t.c_str() + 2);
  for (I i = 0; i < c; i++) {
    const unsigned char *e = d + (size_t)i * sz;
    if (kind == 'f' && sz == 8) { double x; memcpy(&x, e, 8); v[i] = x; }
    else if (kind == 'f' && sz == 4) { float x; memcpy(&x, e, 4); v[i] = x; }
    else if (kind == 'i' && sz == 8) { int64_t x; memcpy(&x, e, 8); v[i] = (double)x; }
    else if (kind == 'i' && sz == 4) { int32_t x; memcpy(&x, e, 4); v[i] = x; }
    else throw std::runtime_error("unsupported float dtype " + t);
  }
  return v;
}

static std::vector<I> npy_int(const NpyArr &a) {
  I c = a.count();
  std::vector<I> v(c);
  const unsigned char *d = a.raw.data() + a.data_off;
  const std::string &t = a.descr;
  char kind = t.size() > 1 ? t[1] : 'i';
  int sz = atoi(t.c_str() + 2);
  for (I i = 0; i < c; i++) {
    const unsigned char *e = d + (size_t)i * sz;
    if (sz == 8) { int64_t x; memcpy(&x, e, 8); v[i] = x; }
    else if (sz == 4) { if (kind == 'u') { uint32_t x; memcpy(&x, e, 4); v[i] = x; } else { int32_t x; memcpy(&x, e, 4); v[i] = x; } }
    else if (sz == 2) { if (kind == 'u') { uint16_t x; memcpy(&x, e, 2); v[i] = x; } else { int16_t x; memcpy(&x, e, 2); v[i] = x; } }
    else if (sz == 1) { if (kind == 'u') v[i] = e[0]; else v[i] = (int8_t)e[0]; }
    else throw std::runtime_error("unsupported int dtype " + t);
  }
  return v;
}

static std::string npy_unicode(const NpyArr &a) {
  const unsigned char *d = a.raw.data() + a.data_off;
  size_t nbytes = a.raw.size() - a.data_off;
  std::string s;
  if (a.descr.size() > 1 && a.descr[1] == 'U') {
    for (size_t i = 0; i + 4 <= nbytes; i += 4) {
      uint32_t c = rd32(d + i);
      if (c == 0) break;
      if (c < 0x80) s += (char)c;
      else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
      else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
      else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
    }
  } else {
    for (size_t i = 0; i < nbytes && d[i]; i++) s += (char)d[i];
  }
  return s;
}

// ---------------- minimal JSON ----------------
struct JV {
  int t = 0;  // 0 null 1 bool 2 num 3 str 4 arr 5 obj
  double num = 0;
  bool b = false;
  std::string str;
  std::vector<JV> arr;
  std::vector<std::pair<std::string, JV>> obj;
  const JV *get(const std::string &k) const {
    for (auto &p : obj) if (p.first == k) return &p.second;
    return nullptr;
  }
};

struct JParser {
  const std::string &s;
  size_t i = 0;
  explicit JParser(const std::string &x) : s(x) {}
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) i++; }
  std::string pstr() {
    std::string r;
    i++;  // opening quote
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        i++;
        char c = s[i];
        if (c == 'n') r += '\n'; else if (c == 't') r += '\t'; else if (c == 'u') { r += '?'; i += 4; }
        else r += c;
        i++;
      } else r += s[i++];
    }
    i++;
    return r;
  }
  JV parse() {
    ws();
    JV v;
    if (i >= s.size()) return v;
    char c = s[i];
    if (c == '{') {
      v.t = 5; i++; ws();
      if (s[i] == '}') { i++; return v; }
      while (true) {
        ws();
        std::string k = pstr();
        ws(); i++;  // ':'
        JV val = parse();
        v.obj.emplace_back(k, std::move(val));
        ws();
        if (s[i] == ',') { i++; continue; }
        i++; break;
      }
    } else if (c == '[') {
      v.t = 4; i++; ws();
      if (s[i] == ']') { i++; return v; }
      while (true) {
        v.arr.push_back(parse());
        ws();
        if (s[i] == ',') { i++; continue; }
        i++; break;
      }
    } else if (c == '"') {
      v.t = 3; v.str = pstr();
    } else if (c == 't') { v.t = 1; v.b = true; i += 4; }
    else if (c == 'f') { v.t = 1; v.b = false; i += 5; }
    else if (c == 'n') { v.t = 0; i += 4; }
    else {
      v.t = 2;
      size_t st = i;
      while (i < s.size() && (strchr("+-0123456789.eE", s[i]) != nullptr)) i++;
      v.num = strtod(s.substr(st, i - st).c_str(), nullptr);
    }
    return v;
  }
};

// ---------------- NPZ writer (stored zip) ----------------
static uint32_t crc_table[256];
static void crc_init() {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    crc_table[i] = c;
  }
}
static uint32_t crc32_buf(const unsigned char *d, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; i++) c = crc_table[(c ^ d[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

static std::vector<unsigned char> make_npy(const std::string &descr, const std::string &shape, const void *data, size_t nbytes) {
  std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  size_t tot = 10 + h.size() + 1;
  size_t pad = (64 - tot % 64) % 64;
  h += std::string(pad, ' ');
  h += '\n';
  std::vector<unsigned char> r(10 + h.size() + nbytes);
  r[0] = 0x93; memcpy(&r[1], "NUMPY", 5); r[6] = 1; r[7] = 0;
  r[8] = h.size() & 0xFF; r[9] = (h.size() >> 8) & 0xFF;
  memcpy(&r[10], h.data(), h.size());
  if (nbytes) memcpy(&r[10 + h.size()], data, nbytes);
  return r;
}

static void w16(std::vector<unsigned char> &o, uint32_t v) { o.push_back(v & 0xFF); o.push_back((v >> 8) & 0xFF); }
static void w32(std::vector<unsigned char> &o, uint32_t v) { for (int k = 0; k < 4; k++) o.push_back((v >> (8 * k)) & 0xFF); }

static bool write_npz(const char *path, const std::vector<std::pair<std::string, std::vector<unsigned char>>> &files) {
  crc_init();
  std::vector<unsigned char> out, cd;
  for (auto &f : files) {
    std::string name = f.first + ".npy";
    uint32_t crc = crc32_buf(f.second.data(), f.second.size());
    uint32_t off = (uint32_t)out.size();
    w32(out, 0x04034b50); w16(out, 20); w16(out, 0); w16(out, 0); w16(out, 0); w16(out, 0x21);
    w32(out, crc); w32(out, (uint32_t)f.second.size()); w32(out, (uint32_t)f.second.size());
    w16(out, (uint32_t)name.size()); w16(out, 0);
    out.insert(out.end(), name.begin(), name.end());
    out.insert(out.end(), f.second.begin(), f.second.end());
    w32(cd, 0x02014b50); w16(cd, 20); w16(cd, 20); w16(cd, 0); w16(cd, 0); w16(cd, 0); w16(cd, 0x21);
    w32(cd, crc); w32(cd, (uint32_t)f.second.size()); w32(cd, (uint32_t)f.second.size());
    w16(cd, (uint32_t)name.size()); w16(cd, 0); w16(cd, 0); w16(cd, 0); w16(cd, 0); w32(cd, 0); w32(cd, off);
    cd.insert(cd.end(), name.begin(), name.end());
  }
  uint32_t cdoff = (uint32_t)out.size();
  out.insert(out.end(), cd.begin(), cd.end());
  w32(out, 0x06054b50); w16(out, 0); w16(out, 0); w16(out, (uint32_t)files.size()); w16(out, (uint32_t)files.size());
  w32(out, (uint32_t)cd.size()); w32(out, cdoff); w16(out, 0);
  FILE *fo = fopen(path, "wb");
  if (!fo) return false;
  size_t wr = fwrite(out.data(), 1, out.size(), fo);
  fclose(fo);
  return wr == out.size();
}

static std::vector<unsigned char> npy_vec(const std::vector<double> &v) {
  return make_npy("<f8", "(" + std::to_string(v.size()) + ",)", v.data(), v.size() * 8);
}
static std::vector<unsigned char> npy_str(const std::string &s) {
  std::vector<uint32_t> u;
  for (unsigned char c : s) u.push_back(c);  // ASCII-only JSON we generate
  if (u.empty()) u.push_back(0);
  return make_npy("<U" + std::to_string(u.size()), "()", u.data(), u.size() * 4);
}
