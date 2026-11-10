// NPZ (zip + npy) reading/writing and a minimal JSON parser.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

extern "C" {
struct z_stream_s {
  const unsigned char* next_in; unsigned avail_in; unsigned long total_in;
  unsigned char* next_out; unsigned avail_out; unsigned long total_out;
  const char* msg; void* state; void* zalloc; void* zfree; void* opaque;
  int data_type; unsigned long adler; unsigned long reserved;
};
int inflateInit2_(z_stream_s* strm, int windowBits, const char* version, int stream_size);
int inflate(z_stream_s* strm, int flush);
int inflateEnd(z_stream_s* strm);
unsigned long crc32(unsigned long crc, const unsigned char* buf, unsigned len);
}

struct NpyArr {
  std::string descr;
  std::vector<int64_t> shape;
  std::vector<char> data;
  int64_t count() const { int64_t c = 1; for (auto s : shape) c *= s; return c; }
};

static inline uint64_t rd16(const unsigned char* p) { return p[0] | (uint64_t(p[1]) << 8); }
static inline uint64_t rd32(const unsigned char* p) { return rd16(p) | (rd16(p + 2) << 16); }
static inline uint64_t rd64(const unsigned char* p) { return rd32(p) | (rd32(p + 4) << 32); }

static void parse_npy(const std::vector<char>& raw, NpyArr& out) {
  const unsigned char* p = (const unsigned char*)raw.data();
  if (raw.size() < 10 || p[0] != 0x93 || memcmp(p + 1, "NUMPY", 5) != 0) throw std::runtime_error("bad npy");
  int major = p[6];
  size_t hlen, hoff;
  if (major == 1) { hlen = rd16(p + 8); hoff = 10; } else { hlen = rd32(p + 8); hoff = 12; }
  std::string h(raw.data() + hoff, hlen);
  size_t k = h.find("'descr'");
  if (k == std::string::npos) throw std::runtime_error("npy descr");
  size_t q1 = h.find('\'', k + 7); size_t q2 = h.find('\'', q1 + 1);
  out.descr = h.substr(q1 + 1, q2 - q1 - 1);
  k = h.find("'fortran_order'");
  if (k != std::string::npos && h.find("True", k) == h.find_first_not_of(" :", k + 15)) {
    // fortran order only matters for >1-D arrays; we only use 1-D
  }
  k = h.find("'shape'");
  size_t a = h.find('(', k), b = h.find(')', a);
  std::string sh = h.substr(a + 1, b - a - 1);
  out.shape.clear();
  size_t i = 0;
  while (i < sh.size()) {
    while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) i++;
    if (i >= sh.size()) break;
    size_t j = i;
    while (j < sh.size() && sh[j] != ',' && sh[j] != ' ') j++;
    out.shape.push_back(atoll(sh.substr(i, j - i).c_str()));
    i = j;
  }
  size_t off = hoff + hlen;
  out.data.assign(raw.begin() + off, raw.end());
}

static std::map<std::string, NpyArr> read_npz(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) throw std::runtime_error("cannot open input");
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> buf(sz);
  if (sz > 0 && fread(buf.data(), 1, sz, f) != (size_t)sz) { fclose(f); throw std::runtime_error("read"); }
  fclose(f);
  // find EOCD
  long eocd = -1;
  for (long i = sz - 22; i >= 0 && i >= sz - 65600; i--)
    if (rd32(&buf[i]) == 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw std::runtime_error("no eocd");
  uint64_t nent = rd16(&buf[eocd + 10]);
  uint64_t cdoff = rd32(&buf[eocd + 16]);
  if (cdoff == 0xFFFFFFFFull || nent == 0xFFFF) {
    long loc = eocd - 20;
    if (loc >= 0 && rd32(&buf[loc]) == 0x07064b50) {
      uint64_t z64 = rd64(&buf[loc + 8]);
      nent = rd64(&buf[z64 + 32]);
      cdoff = rd64(&buf[z64 + 48]);
    }
  }
  std::map<std::string, NpyArr> res;
  uint64_t p = cdoff;
  for (uint64_t e = 0; e < nent; e++) {
    if (rd32(&buf[p]) != 0x02014b50) throw std::runtime_error("bad cd");
    int method = rd16(&buf[p + 10]);
    uint64_t csize = rd32(&buf[p + 20]), usize = rd32(&buf[p + 24]);
    int nl = rd16(&buf[p + 28]), el = rd16(&buf[p + 30]), cl = rd16(&buf[p + 32]);
    uint64_t lho = rd32(&buf[p + 42]);
    std::string name((const char*)&buf[p + 46], nl);
    // zip64 extra
    uint64_t ep = p + 46 + nl, eend = ep + el;
    while (ep + 4 <= eend) {
      int hid = rd16(&buf[ep]), hsz = rd16(&buf[ep + 2]);
      if (hid == 1) {
        uint64_t q = ep + 4;
        if (usize == 0xFFFFFFFFull) { usize = rd64(&buf[q]); q += 8; }
        if (csize == 0xFFFFFFFFull) { csize = rd64(&buf[q]); q += 8; }
        if (lho == 0xFFFFFFFFull) { lho = rd64(&buf[q]); q += 8; }
      }
      ep += 4 + hsz;
    }
    p += 46 + nl + el + cl;
    if (rd32(&buf[lho]) != 0x04034b50) throw std::runtime_error("bad local header");
    uint64_t dataoff = lho + 30 + rd16(&buf[lho + 26]) + rd16(&buf[lho + 28]);
    std::vector<char> raw(usize);
    if (method == 0) {
      memcpy(raw.data(), &buf[dataoff], usize);
    } else if (method == 8) {
      z_stream_s zs; memset(&zs, 0, sizeof(zs));
      if (inflateInit2_(&zs, -15, "1.2.13", (int)sizeof(z_stream_s)) != 0) throw std::runtime_error("inflateInit");
      zs.next_in = &buf[dataoff]; zs.avail_in = (unsigned)csize;
      zs.next_out = (unsigned char*)raw.data(); zs.avail_out = (unsigned)usize;
      int r = inflate(&zs, 4 /*Z_FINISH*/);
      inflateEnd(&zs);
      if (r != 1 /*Z_STREAM_END*/ && zs.total_out != usize) throw std::runtime_error("inflate");
    } else throw std::runtime_error("unsupported zip method");
    if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") name = name.substr(0, name.size() - 4);
    parse_npy(raw, res[name]);
  }
  return res;
}

static std::vector<double> npy_to_double(const NpyArr& a) {
  int64_t c = a.count();
  std::vector<double> v(c);
  const std::string& d = a.descr;
  if (d == "<f8" || d == "=f8" || d == "f8") { memcpy(v.data(), a.data.data(), c * 8); }
  else if (d == "<f4") { const float* p = (const float*)a.data.data(); for (int64_t i = 0; i < c; i++) v[i] = p[i]; }
  else if (d == "<i8" || d == "<u8") { const int64_t* p = (const int64_t*)a.data.data(); for (int64_t i = 0; i < c; i++) v[i] = (double)p[i]; }
  else if (d == "<i4" || d == "<u4") { const int32_t* p = (const int32_t*)a.data.data(); for (int64_t i = 0; i < c; i++) v[i] = p[i]; }
  else throw std::runtime_error("unsupported dtype " + d);
  return v;
}
static std::vector<int64_t> npy_to_int(const NpyArr& a) {
  int64_t c = a.count();
  std::vector<int64_t> v(c);
  const std::string& d = a.descr;
  if (d == "<i8" || d == "<u8") memcpy(v.data(), a.data.data(), c * 8);
  else if (d == "<i4") { const int32_t* p = (const int32_t*)a.data.data(); for (int64_t i = 0; i < c; i++) v[i] = p[i]; }
  else if (d == "<u4") { const uint32_t* p = (const uint32_t*)a.data.data(); for (int64_t i = 0; i < c; i++) v[i] = p[i]; }
  else if (d == "<i2") { const int16_t* p = (const int16_t*)a.data.data(); for (int64_t i = 0; i < c; i++) v[i] = p[i]; }
  else if (d == "|u1" || d == "|i1") { for (int64_t i = 0; i < c; i++) v[i] = (unsigned char)a.data[i]; }
  else if (d == "<f8") { const double* p = (const double*)a.data.data(); for (int64_t i = 0; i < c; i++) v[i] = (int64_t)p[i]; }
  else throw std::runtime_error("unsupported int dtype " + d);
  return v;
}
static std::string npy_to_string(const NpyArr& a) {
  std::string s;
  if (a.descr.size() >= 2 && a.descr[1] == 'U') {
    const uint32_t* p = (const uint32_t*)a.data.data();
    size_t n = a.data.size() / 4;
    for (size_t i = 0; i < n; i++) {
      uint32_t c = p[i];
      if (c == 0) break;
      if (c < 0x80) s.push_back((char)c);
      else if (c < 0x800) { s.push_back((char)(0xC0 | (c >> 6))); s.push_back((char)(0x80 | (c & 0x3F))); }
      else if (c < 0x10000) { s.push_back((char)(0xE0 | (c >> 12))); s.push_back((char)(0x80 | ((c >> 6) & 0x3F))); s.push_back((char)(0x80 | (c & 0x3F))); }
      else { s.push_back((char)(0xF0 | (c >> 18))); s.push_back((char)(0x80 | ((c >> 12) & 0x3F))); s.push_back((char)(0x80 | ((c >> 6) & 0x3F))); s.push_back((char)(0x80 | (c & 0x3F))); }
    }
  } else {
    s.assign(a.data.begin(), a.data.end());
    while (!s.empty() && s.back() == 0) s.pop_back();
  }
  return s;
}

// ---------------- minimal JSON ----------------
struct JVal {
  int t = 0;  // 0 null,1 bool,2 num,3 str,4 arr,5 obj
  double num = 0; bool b = false; std::string str;
  std::vector<JVal> arr; std::vector<std::pair<std::string, JVal>> obj;
  const JVal* get(const std::string& k) const { for (auto& kv : obj) if (kv.first == k) return &kv.second; return nullptr; }
};
struct JParser {
  const std::string& s; size_t i = 0;
  JParser(const std::string& s_) : s(s_) {}
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) i++; }
  std::string pstr() {
    std::string r; i++;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        i++;
        char c = s[i];
        if (c == 'u') { r.push_back('?'); i += 4; }
        else if (c == 'n') r.push_back('\n'); else if (c == 't') r.push_back('\t');
        else r.push_back(c);
        i++;
      } else r.push_back(s[i++]);
    }
    i++;
    return r;
  }
  JVal parse() {
    ws(); JVal v;
    if (i >= s.size()) return v;
    char c = s[i];
    if (c == '{') {
      v.t = 5; i++; ws();
      if (s[i] == '}') { i++; return v; }
      while (true) {
        ws(); std::string k = pstr(); ws(); i++;  // ':'
        JVal x = parse(); v.obj.push_back({k, x}); ws();
        if (s[i] == ',') { i++; continue; }
        i++; break;
      }
    } else if (c == '[') {
      v.t = 4; i++; ws();
      if (s[i] == ']') { i++; return v; }
      while (true) {
        v.arr.push_back(parse()); ws();
        if (s[i] == ',') { i++; continue; }
        i++; break;
      }
    } else if (c == '"') { v.t = 3; v.str = pstr(); }
    else if (c == 't') { v.t = 1; v.b = true; i += 4; }
    else if (c == 'f') { v.t = 1; v.b = false; i += 5; }
    else if (c == 'n') { v.t = 0; i += 4; }
    else {
      size_t j = i;
      while (j < s.size() && (isdigit((unsigned char)s[j]) || s[j] == '-' || s[j] == '+' || s[j] == '.' || s[j] == 'e' || s[j] == 'E')) j++;
      v.t = 2; v.num = strtod(s.substr(i, j - i).c_str(), nullptr); i = j;
    }
    return v;
  }
};

// ---------------- NPZ writer (stored) ----------------
struct NpzWriter {
  struct Ent { std::string name; std::vector<char> data; };
  std::vector<Ent> ents;
  static std::string header(const std::string& descr, const std::string& shape) {
    std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    size_t tot = 10 + h.size() + 1;
    size_t pad = (64 - tot % 64) % 64;
    h += std::string(pad, ' ');
    h += "\n";
    std::string out = "\x93NUMPY";
    out.push_back(1); out.push_back(0);
    out.push_back((char)(h.size() & 0xFF)); out.push_back((char)((h.size() >> 8) & 0xFF));
    return out + h;
  }
  void add_double(const std::string& name, const std::vector<double>& v) {
    std::string h = header("<f8", "(" + std::to_string(v.size()) + ",)");
    Ent e; e.name = name + ".npy";
    e.data.assign(h.begin(), h.end());
    const char* p = (const char*)v.data();
    e.data.insert(e.data.end(), p, p + v.size() * 8);
    ents.push_back(std::move(e));
  }
  void add_unicode_scalar(const std::string& name, const std::string& s) {
    size_t L = s.size(); if (L == 0) L = 1;
    std::string h = header("<U" + std::to_string(L), "()");
    Ent e; e.name = name + ".npy";
    e.data.assign(h.begin(), h.end());
    for (size_t i = 0; i < L; i++) {
      uint32_t c = i < s.size() ? (unsigned char)s[i] : 0;
      e.data.push_back((char)(c & 0xFF)); e.data.push_back(0); e.data.push_back(0); e.data.push_back(0);
    }
    ents.push_back(std::move(e));
  }
  static void w16(std::vector<unsigned char>& o, uint32_t v) { o.push_back(v & 0xFF); o.push_back((v >> 8) & 0xFF); }
  static void w32(std::vector<unsigned char>& o, uint32_t v) { w16(o, v & 0xFFFF); w16(o, v >> 16); }
  bool write(const char* path) {
    std::vector<unsigned char> out, cd;
    for (auto& e : ents) {
      uint32_t crc = (uint32_t)crc32(0, (const unsigned char*)e.data.data(), (unsigned)e.data.size());
      uint32_t off = (uint32_t)out.size();
      w32(out, 0x04034b50); w16(out, 20); w16(out, 0); w16(out, 0); w16(out, 0); w16(out, 0x21);
      w32(out, crc); w32(out, (uint32_t)e.data.size()); w32(out, (uint32_t)e.data.size());
      w16(out, (uint32_t)e.name.size()); w16(out, 0);
      out.insert(out.end(), e.name.begin(), e.name.end());
      out.insert(out.end(), e.data.begin(), e.data.end());
      w32(cd, 0x02014b50); w16(cd, 20); w16(cd, 20); w16(cd, 0); w16(cd, 0); w16(cd, 0); w16(cd, 0x21);
      w32(cd, crc); w32(cd, (uint32_t)e.data.size()); w32(cd, (uint32_t)e.data.size());
      w16(cd, (uint32_t)e.name.size()); w16(cd, 0); w16(cd, 0); w16(cd, 0); w16(cd, 0); w32(cd, 0); w32(cd, off);
      cd.insert(cd.end(), e.name.begin(), e.name.end());
    }
    uint32_t cdoff = (uint32_t)out.size();
    out.insert(out.end(), cd.begin(), cd.end());
    w32(out, 0x06054b50); w16(out, 0); w16(out, 0); w16(out, (uint32_t)ents.size()); w16(out, (uint32_t)ents.size());
    w32(out, (uint32_t)cd.size()); w32(out, cdoff); w16(out, 0);
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return true;
  }
};
