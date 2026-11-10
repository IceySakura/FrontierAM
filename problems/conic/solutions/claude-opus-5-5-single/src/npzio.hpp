// npzio.hpp -- header-only reader/writer for NumPy .npz archives (C++17).
//
// Reading : zip archives whose entries are stored (method 0) or deflated
//           (method 8), including zip64 archives; .npy format versions
//           1.0, 2.0 and 3.0.  Entries are inflated in one pass straight into
//           the destination buffer (size taken from the central directory).
// Writing : uncompressed ("stored") .npz with a correct CRC-32 per entry.
//
// No zlib.h is needed: the few zlib entry points used are declared below.
// Link the program against the zlib shared library, e.g.
//   g++ -O2 -std=c++17 prog.cpp /usr/lib/x86_64-linux-gnu/libz.so.1
//   (arm64: /usr/lib/aarch64-linux-gnu/libz.so.1)
#ifndef NPZIO_HPP_INCLUDED
#define NPZIO_HPP_INCLUDED

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace npz {

// ---------------------------------------------------------------------------
// Minimal zlib ABI (zlib 1.2.x, LP64: x86_64 / aarch64).
// ---------------------------------------------------------------------------
namespace zlib_detail {
struct ZStream {
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
};
static_assert(sizeof(void*) != 8 || sizeof(ZStream) == 112, "z_stream layout mismatch");

extern "C" {
int inflateInit2_(ZStream* strm, int windowBits, const char* version, int stream_size);
int inflate(ZStream* strm, int flush);
int inflateEnd(ZStream* strm);
unsigned long crc32(unsigned long crc, const unsigned char* buf, unsigned int len);
}

constexpr int kOk = 0;
constexpr int kStreamEnd = 1;
constexpr int kBufError = -5;
constexpr int kNoFlush = 0;
}  // namespace zlib_detail

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------
struct Array {
  std::string dtype;            // descr from the npy header, e.g. "<f8", "<i4", "<U167", "|b1"
  std::vector<int64_t> shape;   // empty for a 0-d (scalar) array
  bool fortran_order = false;
  std::vector<char> data;       // raw payload bytes (after the npy header), storage order
  int64_t size() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------
namespace detail {

inline uint16_t rd16(const unsigned char* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t rd32(const unsigned char* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline uint64_t rd64(const unsigned char* p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

inline bool host_little_endian() {
  const uint16_t x = 1;
  unsigned char c;
  std::memcpy(&c, &x, 1);
  return c == 1;
}

// Whole-file read-only view: mmap, falling back to read().
struct FileView {
  const unsigned char* data = nullptr;
  size_t size = 0;
  void* map = nullptr;
  std::vector<unsigned char> buf;

  FileView() = default;
  FileView(const FileView&) = delete;
  FileView& operator=(const FileView&) = delete;
  ~FileView() {
    if (map) ::munmap(map, size);
  }

  bool open(const std::string& path, std::string& err) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      err = "cannot open '" + path + "': " + std::strerror(errno);
      return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      err = "cannot stat '" + path + "': " + std::strerror(errno);
      ::close(fd);
      return false;
    }
    size = (size_t)st.st_size;
    if (size == 0) {
      ::close(fd);
      err = "'" + path + "' is empty";
      return false;
    }
    int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
    flags |= MAP_POPULATE;
#endif
    void* m = ::mmap(nullptr, size, PROT_READ, flags, fd, 0);
    if (m != MAP_FAILED) {
      map = m;
      data = static_cast<const unsigned char*>(m);
      ::close(fd);
      return true;
    }
    buf.resize(size);
    size_t got = 0;
    while (got < size) {
      ssize_t r = ::read(fd, buf.data() + got, size - got);
      if (r < 0) {
        if (errno == EINTR) continue;
        err = "read error on '" + path + "': " + std::strerror(errno);
        ::close(fd);
        return false;
      }
      if (r == 0) break;
      got += (size_t)r;
    }
    ::close(fd);
    if (got != size) {
      err = "short read on '" + path + "'";
      return false;
    }
    data = buf.data();
    return true;
  }
};

// Sequential reader over one zip entry (stored or raw-deflate).
class EntryReader {
 public:
  EntryReader(const unsigned char* in, uint64_t in_len, bool deflated)
      : in_(in), in_len_(in_len), deflated_(deflated) {
    std::memset(&zs_, 0, sizeof zs_);
  }
  ~EntryReader() {
    if (zinit_) zlib_detail::inflateEnd(&zs_);
  }
  EntryReader(const EntryReader&) = delete;
  EntryReader& operator=(const EntryReader&) = delete;

  bool init(std::string& err) {
    if (!deflated_) return true;
    int r = zlib_detail::inflateInit2_(&zs_, -15, "1.2.13", (int)sizeof(zlib_detail::ZStream));
    if (r != zlib_detail::kOk) {
      err = "inflateInit2 failed (code " + std::to_string(r) + ")";
      return false;
    }
    zinit_ = true;
    return true;
  }

  // Produce exactly len bytes into dst.
  bool read(unsigned char* dst, uint64_t len, std::string& err) {
    if (!deflated_) {
      if (len > in_len_ - pos_) {
        err = "stored entry is truncated";
        return false;
      }
      if (len) std::memcpy(dst, in_ + pos_, (size_t)len);
      pos_ += len;
      return true;
    }
    while (len > 0) {
      if (zs_.avail_in == 0 && pos_ < in_len_) {
        const uint64_t c = std::min<uint64_t>(in_len_ - pos_, 1u << 30);
        zs_.next_in = in_ + pos_;
        zs_.avail_in = (unsigned int)c;
        pos_ += c;
      }
      const unsigned int chunk = (unsigned int)std::min<uint64_t>(len, 1u << 30);
      zs_.next_out = dst;
      zs_.avail_out = chunk;
      const int r = zlib_detail::inflate(&zs_, zlib_detail::kNoFlush);
      const uint64_t produced = chunk - zs_.avail_out;
      dst += produced;
      len -= produced;
      if (r == zlib_detail::kStreamEnd) {
        if (len > 0) {
          err = "deflate stream ended before the declared uncompressed size";
          return false;
        }
        break;
      }
      if (r != zlib_detail::kOk) {
        if (r == zlib_detail::kBufError)
          err = "compressed data is truncated";
        else
          err = "inflate error " + std::to_string(r) + (zs_.msg ? std::string(": ") + zs_.msg : std::string());
        return false;
      }
    }
    return true;
  }

 private:
  const unsigned char* in_;
  uint64_t in_len_;
  uint64_t pos_ = 0;
  bool deflated_;
  bool zinit_ = false;
  zlib_detail::ZStream zs_;
};

struct DTypeInfo {
  char order = '|';      // '<', '>', '=', '|'
  char kind = 0;         // 'f', 'i', 'u', 'b', 'U', 'S', ...
  int64_t itemsize = -1; // bytes per element, -1 if unknown
};

inline DTypeInfo parse_dtype(const std::string& d) {
  DTypeInfo t;
  size_t i = 0;
  if (i < d.size() && (d[i] == '<' || d[i] == '>' || d[i] == '=' || d[i] == '|')) t.order = d[i++];
  if (i >= d.size()) return t;
  t.kind = d[i++];
  if (t.kind == '?') {
    t.kind = 'b';
    t.itemsize = 1;
    return t;
  }
  if (i < d.size() && d[i] >= '0' && d[i] <= '9') {
    int64_t n = 0;
    while (i < d.size() && d[i] >= '0' && d[i] <= '9' && n < (int64_t(1) << 40)) n = n * 10 + (d[i++] - '0');
    t.itemsize = (t.kind == 'U') ? 4 * n : n;
  }
  return t;
}

inline bool needs_swap(char order, int64_t unit) {
  if (unit <= 1) return false;
  if (order == '<') return !host_little_endian();
  if (order == '>') return host_little_endian();
  return false;
}

inline void skip_ws(const std::string& s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

inline bool parse_pystr(const std::string& s, size_t& i, std::string& out) {
  if (i >= s.size() || (s[i] != '\'' && s[i] != '"')) return false;
  const char q = s[i++];
  out.clear();
  while (i < s.size() && s[i] != q) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      out += s[i + 1];
      i += 2;
      continue;
    }
    out += s[i++];
  }
  if (i >= s.size()) return false;
  ++i;
  return true;
}

// Captures a (possibly nested) value up to the next top-level ',' or closing '}'.
inline bool capture_raw_value(const std::string& s, size_t& i, std::string& raw) {
  const size_t start = i;
  int depth = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (c == '\'' || c == '"') {
      std::string tmp;
      if (!parse_pystr(s, i, tmp)) return false;
      continue;
    }
    if (c == '(' || c == '[' || c == '{') {
      ++depth;
    } else if (c == ')' || c == ']' || c == '}') {
      if (depth == 0) break;
      --depth;
    } else if (c == ',' && depth == 0) {
      break;
    }
    ++i;
  }
  raw = s.substr(start, i - start);
  while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\n' || raw.back() == '\t')) raw.pop_back();
  return depth == 0;
}

inline bool parse_shape(const std::string& s, size_t& i, std::vector<int64_t>& shape) {
  if (i >= s.size() || (s[i] != '(' && s[i] != '[')) return false;
  const char close = (s[i] == '(') ? ')' : ']';
  ++i;
  shape.clear();
  for (;;) {
    skip_ws(s, i);
    if (i >= s.size()) return false;
    if (s[i] == close) {
      ++i;
      return true;
    }
    int64_t v = 0;
    bool any = false;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      if (v > (INT64_MAX - 9) / 10) return false;
      v = v * 10 + (s[i] - '0');
      ++i;
      any = true;
    }
    if (!any) return false;
    if (i < s.size() && (s[i] == 'L' || s[i] == 'l')) ++i;  // python2 long suffix
    shape.push_back(v);
    skip_ws(s, i);
    if (i < s.size() && s[i] == ',') {
      ++i;
      continue;
    }
    if (i < s.size() && s[i] == close) {
      ++i;
      return true;
    }
    return false;
  }
}

inline bool parse_header_dict(const std::string& h, Array& a, std::string& err) {
  size_t i = 0;
  skip_ws(h, i);
  if (i >= h.size() || h[i] != '{') {
    err = "npy header is not a dict";
    return false;
  }
  ++i;
  bool have_descr = false, have_shape = false;
  for (;;) {
    skip_ws(h, i);
    if (i >= h.size()) {
      err = "unterminated npy header dict";
      return false;
    }
    if (h[i] == '}') break;
    std::string key;
    if (!parse_pystr(h, i, key)) {
      err = "bad key in npy header";
      return false;
    }
    skip_ws(h, i);
    if (i >= h.size() || h[i] != ':') {
      err = "missing ':' in npy header";
      return false;
    }
    ++i;
    skip_ws(h, i);
    if (key == "descr") {
      const bool ok = (i < h.size() && (h[i] == '\'' || h[i] == '"')) ? parse_pystr(h, i, a.dtype)
                                                                      : capture_raw_value(h, i, a.dtype);
      if (!ok) {
        err = "bad descr in npy header";
        return false;
      }
      have_descr = true;
    } else if (key == "fortran_order") {
      if (h.compare(i, 4, "True") == 0) {
        a.fortran_order = true;
        i += 4;
      } else if (h.compare(i, 5, "False") == 0) {
        a.fortran_order = false;
        i += 5;
      } else {
        err = "bad fortran_order in npy header";
        return false;
      }
    } else if (key == "shape") {
      if (!parse_shape(h, i, a.shape)) {
        err = "bad shape in npy header";
        return false;
      }
      have_shape = true;
    } else {
      std::string raw;
      if (!capture_raw_value(h, i, raw)) {
        err = "malformed value in npy header";
        return false;
      }
    }
    skip_ws(h, i);
    if (i < h.size() && h[i] == ',') {
      ++i;
      continue;
    }
    if (i < h.size() && h[i] == '}') break;
    err = "malformed npy header dict";
    return false;
  }
  if (!have_descr || !have_shape) {
    err = "npy header lacks 'descr' or 'shape'";
    return false;
  }
  return true;
}

// Parses one .npy stream (stored or deflated) of total size usize into a.
inline bool read_npy_entry(const unsigned char* in, uint64_t in_len, uint64_t usize, bool deflated, Array& a,
                           std::string& err) {
  EntryReader r(in, in_len, deflated);
  if (!r.init(err)) return false;
  unsigned char pre[12];
  if (usize < 10) {
    err = "entry too small to be an npy file";
    return false;
  }
  if (!r.read(pre, 8, err)) return false;
  if (std::memcmp(pre, "\x93NUMPY", 6) != 0) {
    err = "not an npy file (bad magic)";
    return false;
  }
  const unsigned major = pre[6];
  uint64_t hlen = 0, pre_len = 0;
  if (major == 1) {
    if (!r.read(pre + 8, 2, err)) return false;
    hlen = rd16(pre + 8);
    pre_len = 10;
  } else if (major == 2 || major == 3) {
    if (usize < 12) {
      err = "entry too small to be an npy file";
      return false;
    }
    if (!r.read(pre + 8, 4, err)) return false;
    hlen = rd32(pre + 8);
    pre_len = 12;
  } else {
    err = "unsupported npy format version " + std::to_string(major) + "." + std::to_string((unsigned)pre[7]);
    return false;
  }
  if (hlen > usize - pre_len) {
    err = "npy header longer than the entry";
    return false;
  }
  std::string h((size_t)hlen, '\0');
  if (hlen && !r.read(reinterpret_cast<unsigned char*>(&h[0]), hlen, err)) return false;
  if (!parse_header_dict(h, a, err)) return false;

  uint64_t payload = usize - pre_len - hlen;
  const DTypeInfo t = parse_dtype(a.dtype);
  if (t.itemsize >= 0 && t.kind != 'O' && t.kind != 0) {
    uint64_t cnt = 1;
    for (int64_t d : a.shape) {
      if (d != 0 && cnt > UINT64_MAX / (uint64_t)d) {
        err = "npy shape overflows";
        return false;
      }
      cnt *= (uint64_t)d;
    }
    const uint64_t isz = (uint64_t)t.itemsize;
    if (isz != 0 && cnt > payload / isz) {
      err = "npy payload smaller than shape*itemsize";
      return false;
    }
    payload = cnt * isz;  // ignore any trailing bytes
  }
  a.data.resize((size_t)payload);
  if (payload && !r.read(reinterpret_cast<unsigned char*>(a.data.data()), payload, err)) return false;
  return true;
}

inline float half_to_float(uint16_t h) {
  const int e = (h >> 10) & 0x1f;
  const int m = h & 0x3ff;
  float v;
  if (e == 0)
    v = std::ldexp((float)m, -24);
  else if (e == 31)
    v = m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
  else
    v = std::ldexp((float)(m | 0x400), e - 25);
  return (h & 0x8000) ? -v : v;
}

template <class T, bool Swap>
inline T load_elem(const char* p) {
  T v;
  if (Swap) {
    char b[sizeof(T)];
    for (size_t k = 0; k < sizeof(T); ++k) b[k] = p[sizeof(T) - 1 - k];
    std::memcpy(&v, b, sizeof(T));
  } else {
    std::memcpy(&v, p, sizeof(T));
  }
  return v;
}

template <class Src, bool Swap, class Dst, class Conv>
inline void convert_loop(const char* p, size_t n, Dst* out, Conv conv) {
  for (size_t i = 0; i < n; ++i) out[i] = conv(load_elem<Src, Swap>(p + i * sizeof(Src)));
}

template <class Src, class Dst, class Conv>
inline void convert_typed(const char* p, size_t n, bool swap, Dst* out, Conv conv) {
  if (swap)
    convert_loop<Src, true>(p, n, out, conv);
  else
    convert_loop<Src, false>(p, n, out, conv);
}

template <class Dst, class Conv>
inline void convert_numeric(const Array& a, std::vector<Dst>& out, Conv conv, const char* fn) {
  const DTypeInfo t = parse_dtype(a.dtype);
  auto unsupported = [&]() {
    return std::runtime_error(std::string("npz::") + fn + ": unsupported dtype '" + a.dtype + "'");
  };
  if (t.itemsize <= 0) throw unsupported();
  const int64_t n64 = a.size();
  if (n64 < 0 || (uint64_t)n64 > (uint64_t)a.data.size() / (uint64_t)t.itemsize)
    throw std::runtime_error(std::string("npz::") + fn + ": data buffer smaller than shape implies");
  const size_t n = (size_t)n64;
  out.resize(n);
  const char* p = a.data.data();
  Dst* o = out.data();
  const bool sw = needs_swap(t.order, t.itemsize);
  switch (t.kind) {
    case 'f':
      if (t.itemsize == 8) return convert_typed<double>(p, n, sw, o, conv);
      if (t.itemsize == 4) return convert_typed<float>(p, n, sw, o, conv);
      if (t.itemsize == 2) {
        for (size_t i = 0; i < n; ++i)
          o[i] = conv(half_to_float(sw ? load_elem<uint16_t, true>(p + 2 * i) : load_elem<uint16_t, false>(p + 2 * i)));
        return;
      }
      break;
    case 'i':
      if (t.itemsize == 1) return convert_typed<int8_t>(p, n, sw, o, conv);
      if (t.itemsize == 2) return convert_typed<int16_t>(p, n, sw, o, conv);
      if (t.itemsize == 4) return convert_typed<int32_t>(p, n, sw, o, conv);
      if (t.itemsize == 8) return convert_typed<int64_t>(p, n, sw, o, conv);
      break;
    case 'u':
      if (t.itemsize == 1) return convert_typed<uint8_t>(p, n, sw, o, conv);
      if (t.itemsize == 2) return convert_typed<uint16_t>(p, n, sw, o, conv);
      if (t.itemsize == 4) return convert_typed<uint32_t>(p, n, sw, o, conv);
      if (t.itemsize == 8) return convert_typed<uint64_t>(p, n, sw, o, conv);
      break;
    case 'b':
      if (t.itemsize == 1) {
        for (size_t i = 0; i < n; ++i) o[i] = conv(p[i] != 0);
        return;
      }
      break;
    default:
      break;
  }
  throw unsupported();
}

inline int64_t round_to_i64(double x) {
  if (std::isnan(x)) return 0;
  if (x >= 9223372036854775807.0) return INT64_MAX;
  if (x <= -9223372036854775808.0) return INT64_MIN;
  return (int64_t)std::llround(x);
}

inline void append_utf8(std::string& s, uint32_t c) {
  if (c >= 0xD800 && c <= 0xDFFF) c = 0xFFFD;
  if (c > 0x10FFFF) c = 0xFFFD;
  if (c < 0x80) {
    s += (char)c;
  } else if (c < 0x800) {
    s += (char)(0xC0 | (c >> 6));
    s += (char)(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    s += (char)(0xE0 | (c >> 12));
    s += (char)(0x80 | ((c >> 6) & 0x3F));
    s += (char)(0x80 | (c & 0x3F));
  } else {
    s += (char)(0xF0 | (c >> 18));
    s += (char)(0x80 | ((c >> 12) & 0x3F));
    s += (char)(0x80 | ((c >> 6) & 0x3F));
    s += (char)(0x80 | (c & 0x3F));
  }
}

// UTF-8 -> code points; invalid sequences become U+FFFD.
inline std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
  std::vector<uint32_t> out;
  out.reserve(s.size());
  const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
  const size_t n = s.size();
  size_t i = 0;
  static const uint32_t kMin[4] = {0, 0x80, 0x800, 0x10000};
  while (i < n) {
    const unsigned char b = p[i];
    if (b < 0x80) {
      out.push_back(b);
      ++i;
      continue;
    }
    uint32_t c;
    int extra;
    if ((b & 0xE0) == 0xC0) {
      c = b & 0x1F;
      extra = 1;
    } else if ((b & 0xF0) == 0xE0) {
      c = b & 0x0F;
      extra = 2;
    } else if ((b & 0xF8) == 0xF0) {
      c = b & 0x07;
      extra = 3;
    } else {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    bool ok = i + (size_t)extra < n;
    for (int k = 1; ok && k <= extra; ++k) {
      if ((p[i + k] & 0xC0) != 0x80)
        ok = false;
      else
        c = (c << 6) | (p[i + k] & 0x3F);
    }
    if (ok && (c < kMin[extra] || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF))) ok = false;
    if (!ok) {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    out.push_back(c);
    i += (size_t)extra + 1;
  }
  return out;
}

inline void put16(std::string& s, uint32_t v) {
  s += (char)(v & 0xFF);
  s += (char)((v >> 8) & 0xFF);
}
inline void put32(std::string& s, uint32_t v) {
  for (int k = 0; k < 4; ++k) s += (char)((v >> (8 * k)) & 0xFF);
}
inline void put64(std::string& s, uint64_t v) {
  for (int k = 0; k < 8; ++k) s += (char)((v >> (8 * k)) & 0xFF);
}

// .npy v1.0 preamble (magic, version, header length, padded header dict).
inline std::string npy_preamble(const std::string& descr, const std::string& shape) {
  std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  const size_t unpadded = 10 + dict.size() + 1;
  dict.append((64 - unpadded % 64) % 64, ' ');
  dict += '\n';
  std::string out;
  out += (char)0x93;
  out += "NUMPY";
  out += (char)1;
  out += (char)0;
  put16(out, (uint32_t)dict.size());
  out += dict;
  return out;
}

inline uint32_t crc32_update(uint32_t crc, const unsigned char* p, uint64_t n) {
  unsigned long c = crc;
  while (n > 0) {
    const unsigned int chunk = (unsigned int)std::min<uint64_t>(n, 1u << 30);
    c = zlib_detail::crc32(c, p, chunk);
    p += chunk;
    n -= chunk;
  }
  return (uint32_t)c;
}

struct WriteEntry {
  std::string name;
  std::string pre;
  const unsigned char* data = nullptr;
  uint64_t len = 0;
  uint32_t crc = 0;
  uint64_t size = 0;
  uint64_t offset = 0;
};

inline bool write_npz_impl(const std::string& path,
                           const std::vector<std::pair<std::string, const std::vector<double>*>>& arrays,
                           const std::string& metadata_json, std::string& err, bool force_zip64) {
  const uint16_t kDosTime = 0;       // 00:00:00
  const uint16_t kDosDate = 0x0021;  // 1980-01-01
  const std::string f8 = host_little_endian() ? "<f8" : ">f8";

  std::vector<WriteEntry> ents;
  ents.reserve(arrays.size() + 1);
  for (const auto& pr : arrays) {
    WriteEntry e;
    e.name = pr.first + ".npy";
    const std::vector<double>* v = pr.second;  // nullptr is treated as an empty array
    const uint64_t cnt = v ? (uint64_t)v->size() : 0;
    e.pre = npy_preamble(f8, "(" + std::to_string(cnt) + ",)");
    e.data = (v && cnt) ? reinterpret_cast<const unsigned char*>(v->data()) : nullptr;
    e.len = cnt * sizeof(double);
    ents.push_back(std::move(e));
  }
  const std::vector<uint32_t> cps = utf8_to_codepoints(metadata_json);
  const size_t L = std::max<size_t>(1, cps.size());
  std::string meta(4 * L, '\0');
  for (size_t i = 0; i < cps.size(); ++i)
    for (int k = 0; k < 4; ++k) meta[4 * i + k] = (char)((cps[i] >> (8 * k)) & 0xFF);
  {
    WriteEntry e;
    e.name = "metadata_json.npy";
    e.pre = npy_preamble("<U" + std::to_string(L), "()");
    e.data = reinterpret_cast<const unsigned char*>(meta.data());
    e.len = meta.size();
    ents.push_back(std::move(e));
  }
  {
    std::set<std::string> seen;
    for (const auto& e : ents) {
      if (e.name.size() <= 4) {
        err = "write_npz: empty array name";
        return false;
      }
      if (e.name.size() > 0xFFFF) {
        err = "write_npz: array name too long";
        return false;
      }
      if (!seen.insert(e.name).second) {
        err = "write_npz: duplicate array name '" + e.name.substr(0, e.name.size() - 4) + "'";
        return false;
      }
    }
  }

  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    err = "cannot create '" + path + "': " + std::strerror(errno);
    return false;
  }
  std::setvbuf(f, nullptr, _IOFBF, 1 << 20);
  uint64_t off = 0;
  bool ok = true;
  auto W = [&](const void* d, uint64_t len) {
    if (ok && len && std::fwrite(d, 1, (size_t)len, f) != (size_t)len) ok = false;
    off += len;
  };
  auto name_flags = [](const std::string& nm) -> uint16_t {
    for (unsigned char ch : nm)
      if (ch >= 0x80) return 0x0800;  // UTF-8 file name
    return 0;
  };

  for (auto& e : ents) {
    e.size = e.pre.size() + e.len;
    e.crc = crc32_update(crc32_update(0, reinterpret_cast<const unsigned char*>(e.pre.data()), e.pre.size()),
                         e.data, e.len);
    e.offset = off;
    const bool z64 = force_zip64 || e.size >= 0xFFFFFFFFull;
    std::string lh;
    put32(lh, 0x04034b50u);
    put16(lh, z64 ? 45 : 20);
    put16(lh, name_flags(e.name));
    put16(lh, 0);  // stored
    put16(lh, kDosTime);
    put16(lh, kDosDate);
    put32(lh, e.crc);
    put32(lh, z64 ? 0xFFFFFFFFu : (uint32_t)e.size);
    put32(lh, z64 ? 0xFFFFFFFFu : (uint32_t)e.size);
    put16(lh, (uint32_t)e.name.size());
    put16(lh, z64 ? 20 : 0);
    lh += e.name;
    if (z64) {
      put16(lh, 0x0001);
      put16(lh, 16);
      put64(lh, e.size);
      put64(lh, e.size);
    }
    W(lh.data(), lh.size());
    W(e.pre.data(), e.pre.size());
    W(e.data, e.len);
  }

  const uint64_t cd_off = off;
  std::string cd;
  for (const auto& e : ents) {
    const bool big_size = force_zip64 || e.size >= 0xFFFFFFFFull;
    const bool big_off = force_zip64 || e.offset >= 0xFFFFFFFFull;
    std::string extra;
    if (big_size || big_off) {
      std::string body;
      if (big_size) {
        put64(body, e.size);
        put64(body, e.size);
      }
      if (big_off) put64(body, e.offset);
      put16(extra, 0x0001);
      put16(extra, (uint32_t)body.size());
      extra += body;
    }
    const bool z64 = !extra.empty();
    put32(cd, 0x02014b50u);
    put16(cd, (3u << 8) | (z64 ? 45u : 20u));  // made by: Unix
    put16(cd, z64 ? 45 : 20);
    put16(cd, name_flags(e.name));
    put16(cd, 0);  // stored
    put16(cd, kDosTime);
    put16(cd, kDosDate);
    put32(cd, e.crc);
    put32(cd, big_size ? 0xFFFFFFFFu : (uint32_t)e.size);
    put32(cd, big_size ? 0xFFFFFFFFu : (uint32_t)e.size);
    put16(cd, (uint32_t)e.name.size());
    put16(cd, (uint32_t)extra.size());
    put16(cd, 0);  // comment length
    put16(cd, 0);  // disk number start
    put16(cd, 0);  // internal attributes
    put32(cd, 0x81A40000u);  // external attributes: regular file, rw-r--r--
    put32(cd, big_off ? 0xFFFFFFFFu : (uint32_t)e.offset);
    cd += e.name;
    cd += extra;
  }
  W(cd.data(), cd.size());

  const uint64_t cd_size = cd.size();
  const uint64_t n_ent = ents.size();
  const bool z64_end = force_zip64 || n_ent >= 0xFFFF || cd_size >= 0xFFFFFFFFull || cd_off >= 0xFFFFFFFFull;
  std::string tail;
  if (z64_end) {
    const uint64_t rec_off = off;
    put32(tail, 0x06064b50u);  // zip64 end of central directory record
    put64(tail, 44);
    put16(tail, (3u << 8) | 45u);
    put16(tail, 45);
    put32(tail, 0);
    put32(tail, 0);
    put64(tail, n_ent);
    put64(tail, n_ent);
    put64(tail, cd_size);
    put64(tail, cd_off);
    put32(tail, 0x07064b50u);  // zip64 end of central directory locator
    put32(tail, 0);
    put64(tail, rec_off);
    put32(tail, 1);
  }
  put32(tail, 0x06054b50u);
  put16(tail, 0);
  put16(tail, 0);
  put16(tail, (uint32_t)std::min<uint64_t>(n_ent, 0xFFFF));
  put16(tail, (uint32_t)std::min<uint64_t>(n_ent, 0xFFFF));
  put32(tail, (uint32_t)std::min<uint64_t>(cd_size, 0xFFFFFFFFull));
  put32(tail, (uint32_t)std::min<uint64_t>(cd_off, 0xFFFFFFFFull));
  put16(tail, 0);
  W(tail.data(), tail.size());

  if (std::fflush(f) != 0) ok = false;
  const int saved_errno = errno;
  if (std::fclose(f) != 0) ok = false;
  if (!ok) {
    err = "write error on '" + path + "': " + std::strerror(saved_errno ? saved_errno : errno);
    std::remove(path.c_str());
    return false;
  }
  return true;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Reads every "*.npy" entry of an .npz archive into `out` (cleared first);
// keys are entry names without the ".npy" suffix. Other entries are skipped.
// Returns false and sets `err` on failure (out may then be partially filled).
inline bool read_npz(const std::string& path, std::map<std::string, Array>& out, std::string& err) {
  using detail::rd16;
  using detail::rd32;
  using detail::rd64;
  out.clear();
  err.clear();
  detail::FileView fv;
  if (!fv.open(path, err)) return false;
  const unsigned char* p = fv.data;
  const uint64_t n = fv.size;
  auto fail = [&](const std::string& msg) {
    err = "'" + path + "': " + msg;
    return false;
  };
  if (n < 22) return fail("too small to be a zip/npz file");

  // End of central directory: scan backwards (comment may be up to 64 KiB).
  uint64_t eocd = UINT64_MAX;
  const uint64_t lo = (n - 22 > 65535) ? n - 22 - 65535 : 0;
  for (uint64_t i = n - 22 + 1; i-- > lo;) {
    if (p[i] == 0x50 && rd32(p + i) == 0x06054b50u && i + 22 + rd16(p + i + 20) <= n) {
      eocd = i;
      break;
    }
  }
  if (eocd == UINT64_MAX) return fail("end of central directory not found (not a zip/npz file?)");
  uint64_t count = rd16(p + eocd + 10);
  uint64_t cd_size = rd32(p + eocd + 12);
  uint64_t cd_off = rd32(p + eocd + 16);
  if (eocd >= 20 && rd32(p + eocd - 20) == 0x07064b50u) {
    const uint64_t z = rd64(p + eocd - 20 + 8);
    if (n < 56 || z > n - 56 || rd32(p + z) != 0x06064b50u) return fail("bad zip64 end of central directory");
    count = rd64(p + z + 32);
    cd_size = rd64(p + z + 40);
    cd_off = rd64(p + z + 48);
  }
  if (cd_off > n || cd_size > n - cd_off) return fail("central directory out of bounds");

  uint64_t pos = cd_off;
  for (uint64_t k = 0; k < count; ++k) {
    if (pos > n || n - pos < 46 || rd32(p + pos) != 0x02014b50u) return fail("corrupt central directory");
    const unsigned char* c = p + pos;
    const uint16_t flags = rd16(c + 8);
    const uint16_t method = rd16(c + 10);
    uint64_t csize = rd32(c + 20);
    uint64_t usize = rd32(c + 24);
    const uint64_t nlen = rd16(c + 28), xlen = rd16(c + 30), clen = rd16(c + 32);
    uint64_t lho = rd32(c + 42);
    if (n - pos < 46 + nlen + xlen + clen) return fail("corrupt central directory");
    const std::string name(reinterpret_cast<const char*>(c + 46), (size_t)nlen);
    if (csize == 0xFFFFFFFFu || usize == 0xFFFFFFFFu || lho == 0xFFFFFFFFu) {
      const unsigned char* x = c + 46 + nlen;
      const unsigned char* xe = x + xlen;
      while (xe - x >= 4) {
        const uint16_t id = rd16(x), sz = rd16(x + 2);
        const unsigned char* d = x + 4;
        if (xe - d < sz) break;
        if (id == 0x0001) {
          const unsigned char* q = d;
          const unsigned char* qe = d + sz;
          if (usize == 0xFFFFFFFFu && qe - q >= 8) { usize = rd64(q); q += 8; }
          if (csize == 0xFFFFFFFFu && qe - q >= 8) { csize = rd64(q); q += 8; }
          if (lho == 0xFFFFFFFFu && qe - q >= 8) { lho = rd64(q); q += 8; }
          break;
        }
        x = d + sz;
      }
    }
    pos += 46 + nlen + xlen + clen;

    if (name.size() < 4 || name.compare(name.size() - 4, 4, ".npy") != 0) continue;  // not an array
    const std::string key = name.substr(0, name.size() - 4);
    if (flags & 1) return fail("entry '" + name + "' is encrypted");
    if (method != 0 && method != 8)
      return fail("entry '" + name + "' uses unsupported compression method " + std::to_string(method));
    if (lho > n || n - lho < 30 || rd32(p + lho) != 0x04034b50u) return fail("bad local header for '" + name + "'");
    const uint64_t data_off = lho + 30 + rd16(p + lho + 26) + rd16(p + lho + 28);
    if (data_off > n || csize > n - data_off) return fail("data of entry '" + name + "' is out of bounds");
    // Deflate cannot expand more than ~1032:1; reject corrupt sizes before allocating.
    if ((method == 0 && usize > csize) || (method == 8 && usize / 1040 > csize + 1))
      return fail("entry '" + name + "' has an implausible uncompressed size");

    Array a;
    std::string e;
    if (!detail::read_npy_entry(p + data_off, csize, usize, method == 8, a, e))
      return fail("entry '" + name + "': " + e);
    out[key] = std::move(a);
  }
  return true;
}

// Numeric conversion (elements in storage order). Supports f2/f4/f8,
// i1/i2/i4/i8, u1/u2/u4/u8, b1/? with any byte-order marker.
// Throws std::runtime_error for unsupported dtypes or inconsistent buffers.
inline std::vector<double> to_double(const Array& a) {
  std::vector<double> out;
  detail::convert_numeric(a, out, [](auto v) { return static_cast<double>(v); }, "to_double");
  return out;
}

// Same dtypes as to_double; floating values are rounded to nearest
// (half away from zero), NaN -> 0, out-of-range values are clamped.
inline std::vector<int64_t> to_int64(const Array& a) {
  std::vector<int64_t> out;
  detail::convert_numeric(
      a, out,
      [](auto v) -> int64_t {
        if constexpr (std::is_floating_point<decltype(v)>::value)
          return detail::round_to_i64((double)v);
        else
          return static_cast<int64_t>(v);
      },
      "to_int64");
  return out;
}

// First element of a "U" (UTF-32) or "S" (bytes) array as a UTF-8 string,
// trailing NULs stripped. Returns "" for empty arrays; throws for other dtypes.
inline std::string to_utf8(const Array& a) {
  const detail::DTypeInfo t = detail::parse_dtype(a.dtype);
  if ((t.kind != 'U' && t.kind != 'S' && t.kind != 'a') || t.itemsize < 0)
    throw std::runtime_error("npz::to_utf8: unsupported dtype '" + a.dtype + "'");
  if (a.size() <= 0 || t.itemsize == 0) return std::string();
  if ((uint64_t)a.data.size() < (uint64_t)t.itemsize)
    throw std::runtime_error("npz::to_utf8: data buffer smaller than itemsize");
  const unsigned char* p = reinterpret_cast<const unsigned char*>(a.data.data());
  if (t.kind == 'U') {
    const bool be = (t.order == '>') || (t.order == '=' && !detail::host_little_endian());
    auto cp = [&](size_t i) -> uint32_t {
      const unsigned char* q = p + 4 * i;
      return be ? ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) | ((uint32_t)q[2] << 8) | (uint32_t)q[3]
                : detail::rd32(q);
    };
    size_t len = (size_t)t.itemsize / 4;
    while (len > 0 && cp(len - 1) == 0) --len;
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i) detail::append_utf8(s, cp(i));
    return s;
  }
  size_t len = (size_t)t.itemsize;
  while (len > 0 && p[len - 1] == 0) --len;
  return std::string(reinterpret_cast<const char*>(p), len);
}

// Writes an uncompressed .npz: each (name, values) becomes "name.npy" holding a
// 1-D '<f8' array (nullptr is written as an empty array), and metadata_json
// becomes a 0-d '<U{L}' array "metadata_json.npy". Returns false + err on failure.
inline bool write_npz(const std::string& path,
                      const std::vector<std::pair<std::string, const std::vector<double>*>>& arrays,
                      const std::string& metadata_json, std::string& err) {
  err.clear();
  return detail::write_npz_impl(path, arrays, metadata_json, err, false);
}

}  // namespace npz

#endif  // NPZIO_HPP_INCLUDED
