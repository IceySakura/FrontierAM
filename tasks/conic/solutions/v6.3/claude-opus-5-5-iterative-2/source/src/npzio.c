/* npzio.c - minimal, self-contained NumPy .npz reader/writer (C99, Linux, little-endian host).
   Reading: zip (stored / deflate, ZIP64 aware) via the central directory, built-in raw-DEFLATE
   decoder and CRC-32, .npy v1/v2/v3 headers, metadata JSON. Writing: stored zip members. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "npzio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif

#define DIG(c) ((unsigned)((c) - '0') < 10u)

static int fail(char *err, int errlen, const char *fmt, ...) {
  if (err && errlen > 0) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, (size_t)errlen, fmt, ap);
    va_end(ap);
  }
  return -1;
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32; }
static void wr16(uint8_t *p, unsigned v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void wr32(uint8_t *p, uint32_t v) { wr16(p, v & 0xFFFFu); wr16(p + 2, v >> 16); }
static void *xmalloc(size_t n) { return malloc(n ? n : 1); }

/* ------------------------------------------------------------------ CRC-32 (slicing-by-8) */
static uint32_t crc_tab[8][256];
static volatile int crc_ready;

static uint32_t crc32_upd(uint32_t crc, const void *buf, size_t n) {
  const uint8_t *p = (const uint8_t *)buf;
  if (!crc_ready) {
    uint32_t i, c;
    int k;
    for (i = 0; i < 256; i++) {
      for (c = i, k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      crc_tab[0][i] = c;
    }
    for (i = 0; i < 256; i++)
      for (k = 1; k < 8; k++) crc_tab[k][i] = (crc_tab[k - 1][i] >> 8) ^ crc_tab[0][crc_tab[k - 1][i] & 0xFF];
    crc_ready = 1;
  }
  crc = ~crc;
  for (; n >= 8; p += 8, n -= 8) {
    uint32_t a = rd32(p) ^ crc, b = rd32(p + 4);
    crc = crc_tab[7][a & 0xFF] ^ crc_tab[6][(a >> 8) & 0xFF] ^ crc_tab[5][(a >> 16) & 0xFF] ^ crc_tab[4][a >> 24] ^
          crc_tab[3][b & 0xFF] ^ crc_tab[2][(b >> 8) & 0xFF] ^ crc_tab[1][(b >> 16) & 0xFF] ^ crc_tab[0][b >> 24];
  }
  while (n--) crc = crc_tab[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

/* ------------------------------------------------------------------ raw DEFLATE decoder */
#define FB 11 /* bits of the fast Huffman lookup table */
typedef struct { uint16_t fast[1 << FB], count[16], sym[288]; } huff;

static int huff_build(huff *h, const uint8_t *len, int n) {
  uint16_t offs[16];
  int i, l, left = 1, k = 0;
  unsigned code = 0;
  memset(h->count, 0, sizeof h->count);
  for (i = 0; i < n; i++) h->count[len[i]]++;
  for (l = 1; l < 16; l++) {
    left = (left << 1) - h->count[l];
    if (left < 0) return -1; /* over-subscribed */
  }
  offs[1] = 0;
  for (l = 1; l < 15; l++) offs[l + 1] = (uint16_t)(offs[l] + h->count[l]);
  for (i = 0; i < n; i++)
    if (len[i]) h->sym[offs[len[i]]++] = (uint16_t)i;
  memset(h->fast, 0, sizeof h->fast);
  for (l = 1; l <= FB; l++, code <<= 1)
    for (i = 0; i < h->count[l]; i++, k++, code++) {
      unsigned r = 0, c = code, j;
      int b;
      for (b = 0; b < l; b++, c >>= 1) r = (r << 1) | (c & 1);
      for (j = r; j < (1u << FB); j += 1u << l) h->fast[j] = (uint16_t)(l << 9 | h->sym[k]);
    }
  return 0;
}

/* canonical decode for codes longer than FB bits; returns len<<9|sym or -1 */
static int huff_slow(const huff *h, uint64_t b) {
  int code = 0, first = 0, index = 0, l;
  for (l = 1; l < 16; l++, b >>= 1) {
    int count = h->count[l];
    code |= (int)(b & 1);
    if (code - count < first) return l << 9 | h->sym[index + (code - first)];
    index += count;
    first = (first + count) << 1;
    code <<= 1;
  }
  return -1;
}

static const uint16_t LBASE[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                   31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const uint8_t LEXT[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const uint16_t DBASE[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,    65,    97,    129,
                                   193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const uint8_t DEXT[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
static const uint8_t CLORD[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

/* bit buffer: bits above bc are either zero or consistent copies of upcoming input bits */
#define REFILL()                                                                      \
  do {                                                                                \
    if (ie - in >= 8) {                                                               \
      uint64_t w_;                                                                    \
      memcpy(&w_, in, 8);                                                             \
      bb |= w_ << bc;                                                                 \
      in += (63 - bc) >> 3;                                                           \
      bc |= 56;                                                                       \
    } else                                                                            \
      while (bc <= 56) {                                                              \
        if (in < ie) bb |= (uint64_t)*in++ << bc; else virt++; /* zero padding */     \
        bc += 8;                                                                      \
      }                                                                               \
  } while (0)
#define BITS(n) ((unsigned)(bb & ((1ull << (n)) - 1)))
#define DROP(n) do { bb >>= (n); bc -= (n); } while (0)
#define DECODE(h, res)                                                                \
  do {                                                                                \
    int e_ = (h)->fast[bb & ((1u << FB) - 1)];                                        \
    if (!e_ && (e_ = huff_slow((h), bb)) < 0) goto bad;                               \
    DROP(e_ >> 9);                                                                    \
    (res) = e_ & 511;                                                                 \
  } while (0)

/* returns 0 ok, -1 corrupt stream, -2 output size mismatch */
static int inflate_raw(const uint8_t *in, size_t inlen, uint8_t *out, size_t outlen) {
  const uint8_t *ie = in + inlen;
  uint8_t *op = out, *oe = out + outlen;
  uint64_t bb = 0;
  int bc = 0, virt = 0, final, type, i, sym;
  huff hl, hd;
  uint8_t lens[320];
  do {
    REFILL();
    final = (int)BITS(1);
    type = (int)(bb >> 1) & 3;
    DROP(3);
    if (type == 0) { /* stored block: rewind whole buffered bytes */
      unsigned len, nb, v;
      DROP(bc & 7);
      nb = (unsigned)bc >> 3;
      v = nb < (unsigned)virt ? nb : (unsigned)virt;
      virt -= (int)v;
      in -= nb - v;
      bb = 0;
      bc = 0;
      if (virt || ie - in < 4) goto bad;
      len = rd16(in);
      if (len != (~(unsigned)rd16(in + 2) & 0xFFFFu)) goto bad;
      in += 4;
      if ((size_t)(ie - in) < len || (size_t)(oe - op) < len) goto bad;
      memcpy(op, in, len);
      op += len;
      in += len;
      continue;
    }
    if (type == 1) {
      for (i = 0; i < 144; i++) lens[i] = 8;
      for (; i < 256; i++) lens[i] = 9;
      for (; i < 280; i++) lens[i] = 7;
      for (; i < 288; i++) lens[i] = 8;
      for (; i < 318; i++) lens[i] = 5;
      huff_build(&hl, lens, 288);
      huff_build(&hd, lens + 288, 30);
    } else if (type == 2) {
      int hlit, hdist, hclen;
      REFILL();
      hlit = (int)BITS(5) + 257;
      DROP(5);
      hdist = (int)BITS(5) + 1;
      DROP(5);
      hclen = (int)BITS(4) + 4;
      DROP(4);
      if (hlit > 286 || hdist > 30) goto bad;
      memset(lens, 0, 19);
      for (i = 0; i < hclen; i++) {
        REFILL();
        lens[CLORD[i]] = (uint8_t)BITS(3);
        DROP(3);
      }
      if (huff_build(&hl, lens, 19)) goto bad;
      for (i = 0; i < hlit + hdist;) {
        int rep, val = 0;
        REFILL();
        DECODE(&hl, sym);
        if (sym < 16) { lens[i++] = (uint8_t)sym; continue; }
        if (sym == 16) {
          if (!i) goto bad;
          val = lens[i - 1];
          rep = 3 + (int)BITS(2);
          DROP(2);
        } else if (sym == 17) {
          rep = 3 + (int)BITS(3);
          DROP(3);
        } else {
          rep = 11 + (int)BITS(7);
          DROP(7);
        }
        if (i + rep > hlit + hdist) goto bad;
        while (rep--) lens[i++] = (uint8_t)val;
      }
      if (!lens[256] || huff_build(&hl, lens, hlit) || huff_build(&hd, lens + hlit, hdist)) goto bad;
    } else
      goto bad;
    for (;;) { /* decode symbols; at most 48 bits are consumed per refill */
      unsigned len, dist;
      int ds;
      REFILL();
      DECODE(&hl, sym);
      if (sym < 256) {
        if (op == oe) goto bad;
        *op++ = (uint8_t)sym;
        continue;
      }
      if (sym == 256) break;
      if ((sym -= 257) >= 29) goto bad;
      len = LBASE[sym] + BITS(LEXT[sym]);
      DROP(LEXT[sym]);
      DECODE(&hd, ds);
      if (ds >= 30) goto bad;
      dist = DBASE[ds] + BITS(DEXT[ds]);
      DROP(DEXT[ds]);
      if (dist > (size_t)(op - out) || len > (size_t)(oe - op)) goto bad;
      {
        const uint8_t *src = op - dist;
        uint8_t *end = op + len;
        if (dist >= len) { memcpy(op, src, len); op = end; }
        else while (op < end) *op++ = *src++;
      }
    }
  } while (!final);
  if (virt * 8 > bc) goto bad; /* consumed bits past the end of input */
  return op == oe ? 0 : -2;
bad:
  return -1;
}

/* ------------------------------------------------------------------ zip archive */
typedef struct { const char *name; int nlen, method, flags; uint32_t crc; uint64_t off, csize, usize; } zent;
typedef struct { const uint8_t *d; size_t n; int mapped, nent; zent *ent; } zfile;

static void zip_close(zfile *z) {
  if (z->d) {
    if (z->mapped) munmap((void *)z->d, z->n);
    else free((void *)z->d);
  }
  free(z->ent);
  memset(z, 0, sizeof *z);
}

static int zip_open(zfile *z, const char *path, char *err, int errlen) {
  struct stat st;
  const uint8_t *d, *p, *pe;
  uint64_t nent, cdsize, cdoff, i;
  size_t e = (size_t)-1, lo, k;
  int fd;
  memset(z, 0, sizeof *z);
  if ((fd = open(path, O_RDONLY)) < 0) return fail(err, errlen, "cannot open '%s': %s", path, strerror(errno));
  if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 22) {
    close(fd);
    return fail(err, errlen, "'%s' is not a zip file (too small or not a regular file)", path);
  }
  z->n = (size_t)st.st_size;
  {
    void *m = mmap(NULL, z->n, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (m != MAP_FAILED) {
      z->d = (const uint8_t *)m;
      z->mapped = 1;
    } else {
      uint8_t *b = (uint8_t *)malloc(z->n);
      size_t got = 0;
      while (b && got < z->n) {
        ssize_t r = read(fd, b + got, z->n - got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        got += (size_t)r;
      }
      if (!b || got != z->n) {
        free(b);
        close(fd);
        return fail(err, errlen, "cannot read '%s'", path);
      }
      z->d = b;
    }
  }
  close(fd);
  d = z->d;
  lo = z->n > 22 + 65535 ? z->n - 22 - 65535 : 0;
  for (k = z->n - 22 + 1; k-- > lo;)
    if (rd32(d + k) == 0x06054b50u && k + 22 + rd16(d + k + 20) <= z->n) { e = k; break; }
  if (e == (size_t)-1) { zip_close(z); return fail(err, errlen, "'%s': zip end-of-central-directory not found", path); }
  nent = rd16(d + e + 10);
  cdsize = rd32(d + e + 12);
  cdoff = rd32(d + e + 16);
  if (e >= 20 && rd32(d + e - 20) == 0x07064b50u) { /* ZIP64 locator */
    uint64_t z64 = rd64(d + e - 20 + 8);
    if (z64 > e || e - z64 < 56 || rd32(d + z64) != 0x06064b50u) {
      zip_close(z);
      return fail(err, errlen, "'%s': bad ZIP64 end-of-central-directory", path);
    }
    nent = rd64(d + z64 + 32);
    cdsize = rd64(d + z64 + 40);
    cdoff = rd64(d + z64 + 48);
  }
  if (cdoff > z->n || cdsize > z->n - cdoff || nent > cdsize / 46) {
    zip_close(z);
    return fail(err, errlen, "'%s': bad zip central directory", path);
  }
  z->ent = (zent *)calloc(nent ? (size_t)nent : 1, sizeof(zent));
  if (!z->ent) { zip_close(z); return fail(err, errlen, "out of memory"); }
  p = d + cdoff;
  pe = p + cdsize;
  for (i = 0; i < nent; i++) {
    zent *t = &z->ent[i];
    const uint8_t *x, *xe;
    int nl, xl, cl;
    if (pe - p < 46 || rd32(p) != 0x02014b50u) break;
    nl = rd16(p + 28);
    xl = rd16(p + 30);
    cl = rd16(p + 32);
    if (pe - p < 46 + nl + xl + cl) break;
    t->flags = rd16(p + 8);
    t->method = rd16(p + 10);
    t->crc = rd32(p + 16);
    t->csize = rd32(p + 20);
    t->usize = rd32(p + 24);
    t->off = rd32(p + 42);
    t->name = (const char *)p + 46;
    t->nlen = nl;
    for (x = p + 46 + nl, xe = x + xl; xe - x >= 4; x += 4 + rd16(x + 2)) {
      const uint8_t *q = x + 4, *qe = q + rd16(x + 2);
      if (qe > xe) break;
      if (rd16(x) != 1) continue; /* ZIP64 extended information */
      if (t->usize == 0xFFFFFFFFu && qe - q >= 8) { t->usize = rd64(q); q += 8; }
      if (t->csize == 0xFFFFFFFFu && qe - q >= 8) { t->csize = rd64(q); q += 8; }
      if (t->off == 0xFFFFFFFFu && qe - q >= 8) { t->off = rd64(q); q += 8; }
    }
    p += 46 + nl + xl + cl;
  }
  if (i != nent) { zip_close(z); return fail(err, errlen, "'%s': corrupt zip central directory", path); }
  z->nent = (int)nent;
  return 0;
}

/* Get the uncompressed bytes of a member: either a pointer into the file or a malloc'd buffer (*own). */
static int zip_get(zfile *z, const char *name, uint8_t **own, const uint8_t **ptr, size_t *len, char *err, int errlen) {
  int i, nl = (int)strlen(name);
  const zent *t = NULL;
  uint64_t start;
  *own = NULL;
  for (i = 0; i < z->nent; i++)
    if (z->ent[i].nlen == nl && !memcmp(z->ent[i].name, name, (size_t)nl)) t = &z->ent[i];
  if (!t) return fail(err, errlen, "missing member '%s'", name);
  if (t->flags & 1) return fail(err, errlen, "member '%s' is encrypted", name);
  if (t->off > z->n || z->n - t->off < 30 || rd32(z->d + t->off) != 0x04034b50u)
    return fail(err, errlen, "member '%s': bad local header", name);
  start = t->off + 30 + rd16(z->d + t->off + 26) + rd16(z->d + t->off + 28);
  if (start > z->n || t->csize > z->n - start) return fail(err, errlen, "member '%s': data out of range", name);
  if (t->usize > ((uint64_t)1 << 46)) return fail(err, errlen, "member '%s': too large", name);
  if (t->method == 0) {
    if (t->csize != t->usize) return fail(err, errlen, "member '%s': size mismatch", name);
    *ptr = z->d + start;
  } else if (t->method == 8) {
    int r;
    if (!(*own = (uint8_t *)xmalloc((size_t)t->usize))) return fail(err, errlen, "out of memory (%s)", name);
    r = inflate_raw(z->d + start, (size_t)t->csize, *own, (size_t)t->usize);
    if (r) {
      free(*own);
      *own = NULL;
      return fail(err, errlen, "member '%s': %s", name, r == -2 ? "uncompressed size mismatch" : "corrupt deflate data");
    }
    *ptr = *own;
  } else
    return fail(err, errlen, "member '%s': unsupported compression method %d", name, t->method);
  *len = (size_t)t->usize;
  if (crc32_upd(0, *ptr, *len) != t->crc) {
    free(*own);
    *own = NULL;
    return fail(err, errlen, "member '%s': CRC mismatch", name);
  }
  return 0;
}

/* ------------------------------------------------------------------ .npy */
typedef struct { int kind, isize; int64_t count; const uint8_t *data; size_t off; } npyarr;

static const char *dict_val(const char *h, const char *he, const char *key) {
  size_t kl = strlen(key);
  const char *p;
  for (p = h; (size_t)(he - p) >= kl + 2; p++)
    if ((*p == '\'' || *p == '"') && p[kl + 1] == *p && !memcmp(p + 1, key, kl)) {
      for (p += kl + 2; p < he && *p == ' '; p++) {}
      if (p >= he || *p != ':') return NULL;
      for (p++; p < he && *p == ' '; p++) {}
      return p < he ? p : NULL;
    }
  return NULL;
}

static int npy_parse(const uint8_t *b, size_t len, npyarr *a, const char *nm, char *err, int errlen) {
  size_t hs, hl;
  const char *h, *he, *v, *e;
  int nd = 0, bo = '|';
  int64_t cnt = 1, sz = 0;
  if (len < 10 || memcmp(b, "\x93NUMPY", 6)) return fail(err, errlen, "%s: not a .npy array", nm);
  if (b[6] == 1) { hs = 10; hl = rd16(b + 8); }
  else if ((b[6] == 2 || b[6] == 3) && len >= 12) { hs = 12; hl = rd32(b + 8); }
  else return fail(err, errlen, "%s: unsupported .npy version %d", nm, b[6]);
  if (hl > len - hs) return fail(err, errlen, "%s: truncated .npy header", nm);
  h = (const char *)b + hs;
  he = h + hl;
  if (!(v = dict_val(h, he, "descr")) || (*v != '\'' && *v != '"')) return fail(err, errlen, "%s: bad dtype descr", nm);
  for (e = v + 1; e < he && *e != *v; e++) {}
  if (e >= he) return fail(err, errlen, "%s: bad dtype descr", nm);
  v++;
  if (v < e && (*v == '<' || *v == '>' || *v == '|' || *v == '=')) bo = *v++;
  if (v >= e) return fail(err, errlen, "%s: bad dtype descr", nm);
  a->kind = *v++;
  if (v >= e) return fail(err, errlen, "%s: bad dtype descr", nm);
  for (; v < e && DIG(*v) && sz < (1 << 28); v++) sz = sz * 10 + (*v - '0');
  if (v != e) return fail(err, errlen, "%s: unsupported dtype", nm);
  a->isize = (int)(a->kind == 'U' ? 4 * sz : sz);
  if (bo == '>' && (a->kind == 'U' || sz > 1)) return fail(err, errlen, "%s: big-endian arrays not supported", nm);
  if (!(v = dict_val(h, he, "shape")) || *v != '(') return fail(err, errlen, "%s: bad shape", nm);
  for (v++;;) {
    int64_t dim = 0;
    while (v < he && *v == ' ') v++;
    if (v >= he) return fail(err, errlen, "%s: bad shape", nm);
    if (*v == ')') break;
    if (!DIG(*v)) return fail(err, errlen, "%s: bad shape", nm);
    for (; v < he && DIG(*v); v++)
      if ((dim = dim * 10 + (*v - '0')) > ((int64_t)1 << 48)) return fail(err, errlen, "%s: bad shape", nm);
    if (v < he && *v == 'L') v++;
    if (dim && cnt > ((int64_t)1 << 48) / dim) return fail(err, errlen, "%s: array too large", nm);
    cnt *= dim;
    nd++;
    while (v < he && *v == ' ') v++;
    if (v < he && *v == ',') v++;
  }
  if (nd > 1 && (v = dict_val(h, he, "fortran_order")) && he - v >= 4 && !memcmp(v, "True", 4))
    return fail(err, errlen, "%s: fortran-ordered arrays not supported", nm);
  a->off = hs + hl;
  if (a->isize && (uint64_t)cnt > (len - a->off) / (uint64_t)a->isize) return fail(err, errlen, "%s: truncated array data", nm);
  a->count = cnt;
  a->data = b + a->off;
  return 0;
}

#define CVT(T) for (i = 0; i < n; i++) { T v_; memcpy(&v_, s + (size_t)i * sizeof(T), sizeof(T)); d[i] = v_; }

/* convert to a new float64 ('f') or int64 ('i') array; may steal *own */
static int npy_convert(const npyarr *a, int want, uint8_t **own, void **out, const char *nm, char *err, int errlen) {
  int64_t i, n = a->count;
  const uint8_t *s = a->data;
  int k = a->kind == 'b' ? 'u' : a->kind, sz = a->isize;
  if (k == 'f' ? (sz != 4 && sz != 8) : (k == 'i' || k == 'u') ? (sz != 1 && sz != 2 && sz != 4 && sz != 8) : 1)
    return fail(err, errlen, "%s: unsupported dtype", nm);
  if (want == 'i' && k == 'f') return fail(err, errlen, "%s: expected an integer array", nm);
  if (sz == 8 && k == (want == 'f' ? 'f' : 'i')) { /* same type: move in place */
    if (*own) {
      memmove(*own, s, (size_t)n * 8);
      *out = realloc(*own, n ? (size_t)n * 8 : 1);
      if (!*out) *out = *own;
      *own = NULL;
    } else if ((*out = xmalloc((size_t)n * 8)))
      memcpy(*out, s, (size_t)n * 8);
    return *out ? 0 : fail(err, errlen, "out of memory (%s)", nm);
  }
  if (!(*out = xmalloc((size_t)n * 8))) return fail(err, errlen, "out of memory (%s)", nm);
  if (want == 'f') {
    double *d = (double *)*out;
    if (k == 'f') { CVT(float) }
    else if (k == 'i') { if (sz == 1) { CVT(int8_t) } else if (sz == 2) { CVT(int16_t) } else if (sz == 4) { CVT(int32_t) } else { CVT(int64_t) } }
    else { if (sz == 1) { CVT(uint8_t) } else if (sz == 2) { CVT(uint16_t) } else if (sz == 4) { CVT(uint32_t) } else { CVT(uint64_t) } }
  } else {
    int64_t *d = (int64_t *)*out;
    if (k == 'i') { if (sz == 1) { CVT(int8_t) } else if (sz == 2) { CVT(int16_t) } else { CVT(int32_t) } }
    else if (sz == 1) { CVT(uint8_t) } else if (sz == 2) { CVT(uint16_t) } else if (sz == 4) { CVT(uint32_t) }
    else {
      for (i = 0; i < n; i++) {
        uint64_t v_;
        memcpy(&v_, s + (size_t)i * 8, 8);
        if (v_ > (uint64_t)INT64_MAX) { free(*out); *out = NULL; return fail(err, errlen, "%s: index out of range", nm); }
        d[i] = (int64_t)v_;
      }
    }
  }
  return 0;
}

static int load_arr(zfile *z, const char *name, int want, void *outp, int64_t *cnt, char *err, int errlen) {
  uint8_t *own = NULL;
  const uint8_t *ptr;
  size_t len;
  npyarr a;
  int rc;
  if (zip_get(z, name, &own, &ptr, &len, err, errlen)) return -1;
  rc = npy_parse(ptr, len, &a, name, err, errlen);
  if (!rc) rc = npy_convert(&a, want, &own, (void **)outp, name, err, errlen);
  if (!rc) *cnt = a.count;
  free(own);
  return rc;
}

/* ------------------------------------------------------------------ JSON */
typedef struct { const char *p, *e; int depth; } js;

static void jws(js *j) {
  while (j->p < j->e && (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t')) j->p++;
}

/* parse a string at j->p; decoded (non-ASCII -> '?') into out[cap] if out; *olen = decoded length */
static int jstr(js *j, char *out, int cap, int *olen) {
  int n = 0, k;
  if (j->p >= j->e || *j->p != '"') return -1;
  for (j->p++; j->p < j->e;) {
    unsigned c = (unsigned char)*j->p++;
    if (c == '"') {
      if (out) out[n < cap ? n : cap - 1] = 0;
      *olen = n;
      return 0;
    }
    if (c < 0x20) return -1;
    if (c == '\\') {
      if (j->p >= j->e) return -1;
      switch (c = (unsigned char)*j->p++) {
        case '"': case '\\': case '/': break;
        case 'b': c = 8; break;
        case 'f': c = 12; break;
        case 'n': c = 10; break;
        case 'r': c = 13; break;
        case 't': c = 9; break;
        case 'u':
          if (j->e - j->p < 4) return -1;
          for (c = 0, k = 0; k < 4; k++) {
            unsigned h = (unsigned char)*j->p++;
            if (DIG(h)) h -= '0';
            else if ((h | 32) >= 'a' && (h | 32) <= 'f') h = (h | 32) - 'a' + 10;
            else return -1;
            c = c << 4 | h;
          }
          if (c >= 0x80) c = '?';
          break;
        default: return -1;
      }
    }
    if (out && n < cap - 1) out[n] = (char)c;
    n++;
  }
  return -1;
}

/* JSON number (also NaN/Infinity/-Infinity as Python emits); dv may be NULL */
static int jnum(js *j, double *dv, int64_t *iv, int *isint) {
  const char *s = j->p, *p = s, *e = j->e;
  int neg = 0, isi = 1, ovf = 0;
  uint64_t acc = 0;
  *isint = 0;
  if (e - p >= 3 && !memcmp(p, "NaN", 3)) { j->p = p + 3; if (dv) *dv = NAN; return 0; }
  if (p < e && *p == '-') { neg = 1; p++; }
  if (e - p >= 8 && !memcmp(p, "Infinity", 8)) { j->p = p + 8; if (dv) *dv = neg ? -INFINITY : INFINITY; return 0; }
  if (p >= e || !DIG(*p)) return -1;
  if (*p == '0') p++;
  else
    for (; p < e && DIG(*p); p++) {
      if (acc > (UINT64_MAX - 9) / 10) ovf = 1;
      else acc = acc * 10 + (uint64_t)(*p - '0');
    }
  if (p < e && *p == '.') {
    isi = 0;
    if (++p >= e || !DIG(*p)) return -1;
    while (p < e && DIG(*p)) p++;
  }
  if (p < e && (*p == 'e' || *p == 'E')) {
    isi = 0;
    if (++p < e && (*p == '+' || *p == '-')) p++;
    if (p >= e || !DIG(*p)) return -1;
    while (p < e && DIG(*p)) p++;
  }
  if (dv) {
    size_t L = (size_t)(p - s);
    char tmp[80], *t = L < sizeof tmp ? tmp : (char *)malloc(L + 1);
    if (!t) return -1;
    memcpy(t, s, L);
    t[L] = 0;
    *dv = strtod(t, NULL);
    if (t != tmp) free(t);
  }
  *isint = isi && !ovf && acc <= (uint64_t)INT64_MAX;
  if (iv) *iv = neg ? -(int64_t)acc : (int64_t)acc;
  j->p = p;
  return 0;
}

static int jskip(js *j) {
  int dummy, isi;
  char c, close;
  jws(j);
  if (j->p >= j->e) return -1;
  c = *j->p;
  if (c == '"') return jstr(j, NULL, 0, &dummy);
  if (c == '{' || c == '[') {
    if (++j->depth > 10000) return -1;
    close = c == '{' ? '}' : ']';
    j->p++;
    jws(j);
    if (j->p < j->e && *j->p == close) { j->p++; j->depth--; return 0; }
    for (;;) {
      if (c == '{') {
        jws(j);
        if (jstr(j, NULL, 0, &dummy)) return -1;
        jws(j);
        if (j->p >= j->e || *j->p != ':') return -1;
        j->p++;
      }
      if (jskip(j)) return -1;
      jws(j);
      if (j->p >= j->e) return -1;
      if (*j->p == ',') { j->p++; continue; }
      if (*j->p != close) return -1;
      j->p++;
      j->depth--;
      return 0;
    }
  }
  if (j->e - j->p >= 4 && (!memcmp(j->p, "true", 4) || !memcmp(j->p, "null", 4))) { j->p += 4; return 0; }
  if (j->e - j->p >= 5 && !memcmp(j->p, "false", 5)) { j->p += 5; return 0; }
  return jnum(j, NULL, NULL, &isi);
}

#define JFAIL(msg) do { rc = fail(err, errlen, "metadata_json: %s", msg); goto out; } while (0)

static int parse_meta(const char *txt, size_t len, npz_problem *p, char *err, int errlen) {
  static const char *const names[5] = {"zero", "nonnegative", "soc", "exponential", "psd"};
  js j;
  int rc = 0, kl, have_cones = 0, cap = 0, isi;
  char key[32];
  j.p = txt;
  j.e = txt + len;
  j.depth = 0;
  jws(&j);
  if (j.p >= j.e || *j.p != '{') JFAIL("expected a JSON object");
  j.p++;
  jws(&j);
  if (j.p < j.e && *j.p == '}') j.p++;
  else
    for (;;) {
      jws(&j);
      if (jstr(&j, key, sizeof key, &kl)) JFAIL("bad object key");
      jws(&j);
      if (j.p >= j.e || *j.p != ':') JFAIL("expected ':'");
      j.p++;
      jws(&j);
      if (kl == 5 && !memcmp(key, "cones", 5)) {
        have_cones = 1;
        p->ncones = 0;
        if (j.p >= j.e || *j.p != '[') JFAIL("'cones' must be a list");
        j.p++;
        jws(&j);
        if (j.p < j.e && *j.p == ']') j.p++;
        else
          for (;;) {
            int type = -1, have_dim = 0, t;
            int64_t dim = 0;
            jws(&j);
            if (j.p >= j.e || *j.p != '{') JFAIL("cone entries must be objects");
            j.p++;
            jws(&j);
            if (j.p < j.e && *j.p == '}') j.p++;
            else
              for (;;) {
                jws(&j);
                if (jstr(&j, key, sizeof key, &kl)) JFAIL("bad cone key");
                jws(&j);
                if (j.p >= j.e || *j.p != ':') JFAIL("expected ':'");
                j.p++;
                jws(&j);
                if (kl == 4 && !memcmp(key, "type", 4)) {
                  char tv[16];
                  int tl;
                  if (j.p >= j.e || *j.p != '"' || jstr(&j, tv, sizeof tv, &tl)) JFAIL("cone 'type' must be a string");
                  for (type = -1, t = 0; t < 5; t++)
                    if ((size_t)tl == strlen(names[t]) && !memcmp(tv, names[t], (size_t)tl)) type = t;
                  if (type < 0) {
                    rc = fail(err, errlen, "metadata_json: unsupported cone type '%s'", tv);
                    goto out;
                  }
                } else if (kl == 3 && !memcmp(key, "dim", 3)) {
                  if (jnum(&j, NULL, &dim, &isi) || !isi) JFAIL("cone 'dim' must be an integer");
                  have_dim = 1;
                } else if (jskip(&j))
                  JFAIL("syntax error");
                jws(&j);
                if (j.p < j.e && *j.p == ',') { j.p++; continue; }
                if (j.p < j.e && *j.p == '}') { j.p++; break; }
                JFAIL("syntax error in cone object");
              }
            if (type < 0 || !have_dim) JFAIL("cone without 'type' or 'dim'");
            if (p->ncones == cap) {
              int nc = cap ? 2 * cap : 64;
              int *nt;
              int64_t *ndm;
              if (cap > (1 << 29)) JFAIL("too many cones");
              nt = (int *)realloc(p->cone_type, (size_t)nc * sizeof(int));
              if (nt) p->cone_type = nt;
              ndm = (int64_t *)realloc(p->cone_dim, (size_t)nc * sizeof(int64_t));
              if (ndm) p->cone_dim = ndm;
              if (!nt || !ndm) JFAIL("out of memory");
              cap = nc;
            }
            p->cone_type[p->ncones] = type;
            p->cone_dim[p->ncones++] = dim;
            jws(&j);
            if (j.p < j.e && *j.p == ',') { j.p++; continue; }
            if (j.p < j.e && *j.p == ']') { j.p++; break; }
            JFAIL("syntax error in 'cones'");
          }
      } else if (kl == 16 && !memcmp(key, "objective_offset", 16)) {
        int64_t iv;
        if (jnum(&j, &p->objective_offset, &iv, &isi)) JFAIL("'objective_offset' must be a number");
        if (!isfinite(p->objective_offset)) JFAIL("'objective_offset' is not finite");
      } else if (kl == 14 && !memcmp(key, "schema_version", 14)) {
        int64_t iv;
        if (jnum(&j, NULL, &iv, &isi) || !isi || iv != 1) JFAIL("unsupported schema_version");
      } else if (jskip(&j))
        JFAIL("syntax error");
      jws(&j);
      if (j.p < j.e && *j.p == ',') { j.p++; continue; }
      if (j.p < j.e && *j.p == '}') { j.p++; break; }
      JFAIL("syntax error in top-level object");
    }
  jws(&j);
  if (j.p != j.e) JFAIL("trailing characters");
  if (!have_cones) JFAIL("missing 'cones'");
  if (!p->cone_type) { /* zero cones: keep pointers valid */
    p->cone_type = (int *)xmalloc(sizeof(int));
    p->cone_dim = (int64_t *)xmalloc(sizeof(int64_t));
    if (!p->cone_type || !p->cone_dim) JFAIL("out of memory");
  }
out:
  return rc;
}

static int load_meta(zfile *z, npz_problem *p, char *err, int errlen) {
  uint8_t *own = NULL;
  const uint8_t *ptr;
  size_t len, k, i;
  npyarr a;
  char *txt = NULL;
  int rc = -1;
  if (zip_get(z, "metadata_json.npy", &own, &ptr, &len, err, errlen)) return -1;
  if (npy_parse(ptr, len, &a, "metadata_json", err, errlen)) goto out;
  if (a.kind != 'U' || a.count != 1) { fail(err, errlen, "metadata_json: expected a scalar unicode array"); goto out; }
  for (k = (size_t)a.isize / 4; k > 0 && !rd32(a.data + 4 * (k - 1)); k--) {}
  if (!(txt = (char *)xmalloc(k))) { fail(err, errlen, "out of memory"); goto out; }
  for (i = 0; i < k; i++) {
    uint32_t c = rd32(a.data + 4 * i);
    txt[i] = c < 0x80 ? (char)c : '?';
  }
  rc = parse_meta(txt, k, p, err, errlen);
out:
  free(txt);
  free(own);
  return rc;
}

/* ------------------------------------------------------------------ problem */
static int check_csc(const char *nm, int64_t rows, int64_t cols, const int64_t *shp, int64_t nshp, const int64_t *cp,
                     int64_t ncp, const int64_t *ri, int64_t nri, int64_t nval, int64_t *nnz, char *err, int errlen) {
  int64_t j, k;
  if (nshp != 2 || shp[0] != rows || shp[1] != cols)
    return fail(err, errlen, "%s_shape does not match (%lld, %lld)", nm, (long long)rows, (long long)cols);
  if (ncp != cols + 1) return fail(err, errlen, "%s_indptr has length %lld, expected %lld", nm, (long long)ncp, (long long)cols + 1);
  if (nri != nval) return fail(err, errlen, "%s_indices and %s_data lengths differ", nm, nm);
  if (cp[0] != 0) return fail(err, errlen, "%s_indptr[0] != 0", nm);
  for (j = 0; j < cols; j++)
    if (cp[j + 1] < cp[j]) return fail(err, errlen, "%s_indptr is not nondecreasing", nm);
  if (cp[cols] > nri) return fail(err, errlen, "%s_indptr[-1] exceeds number of stored entries", nm);
  for (k = 0; k < cp[cols]; k++)
    if ((uint64_t)ri[k] >= (uint64_t)rows) return fail(err, errlen, "%s_indices out of range", nm);
  *nnz = cp[cols];
  return 0;
}

static int read_problem(zfile *z, npz_problem *p, char *err, int errlen) {
  int64_t n, m, nps, nas, c1, c2, c3, tot = 0, d;
  int64_t *ps = NULL, *as = NULL;
  int rc = -1, i, t;
  if (load_arr(z, "q.npy", 'f', &p->q, &n, err, errlen) || load_arr(z, "b.npy", 'f', &p->b, &m, err, errlen) ||
      load_arr(z, "P_shape.npy", 'i', &ps, &nps, err, errlen) || load_arr(z, "A_shape.npy", 'i', &as, &nas, err, errlen))
    goto out;
  p->n = n;
  p->m = m;
  if (load_arr(z, "P_indptr.npy", 'i', &p->P_colptr, &c1, err, errlen) ||
      load_arr(z, "P_indices.npy", 'i', &p->P_rowind, &c2, err, errlen) ||
      load_arr(z, "P_data.npy", 'f', &p->P_val, &c3, err, errlen) ||
      check_csc("P", n, n, ps, nps, p->P_colptr, c1, p->P_rowind, c2, c3, &p->P_nnz, err, errlen))
    goto out;
  if (load_arr(z, "A_indptr.npy", 'i', &p->A_colptr, &c1, err, errlen) ||
      load_arr(z, "A_indices.npy", 'i', &p->A_rowind, &c2, err, errlen) ||
      load_arr(z, "A_data.npy", 'f', &p->A_val, &c3, err, errlen) ||
      check_csc("A", m, n, as, nas, p->A_colptr, c1, p->A_rowind, c2, c3, &p->A_nnz, err, errlen))
    goto out;
  if (load_meta(z, p, err, errlen)) goto out;
  for (i = 0; i < p->ncones; i++) {
    t = p->cone_type[i];
    d = p->cone_dim[i];
    if (d <= 0 || d > ((int64_t)1 << 31) || (t == NPZ_CONE_EXP && d != 3) || (t == NPZ_CONE_SOC && d < 2)) {
      fail(err, errlen, "cone %d (type %d) has invalid dim %lld", i, t, (long long)d);
      goto out;
    }
    tot += t == NPZ_CONE_PSD ? d * (d + 1) / 2 : d;
    if (tot > ((int64_t)1 << 62)) break;
  }
  if (tot != m) {
    fail(err, errlen, "cone dimensions sum to %lld but len(b) = %lld", (long long)tot, (long long)m);
    goto out;
  }
  rc = 0;
out:
  free(ps);
  free(as);
  return rc;
}

void npz_free_problem(npz_problem *p) {
  if (!p) return;
  free(p->P_colptr); free(p->P_rowind); free(p->P_val);
  free(p->A_colptr); free(p->A_rowind); free(p->A_val);
  free(p->q); free(p->b); free(p->cone_type); free(p->cone_dim);
  memset(p, 0, sizeof *p);
}

int npz_read_problem(const char *path, npz_problem *p, char *err, int errlen) {
  zfile z;
  int rc;
  if (err && errlen > 0) err[0] = 0;
  if (!p) return fail(err, errlen, "null problem pointer");
  memset(p, 0, sizeof *p);
  if (!path) return fail(err, errlen, "null path");
  if (zip_open(&z, path, err, errlen)) return -1;
  rc = read_problem(&z, p, err, errlen);
  zip_close(&z);
  if (rc) npz_free_problem(p);
  return rc ? -1 : 0;
}

/* ------------------------------------------------------------------ writer */
typedef struct { FILE *f; uint64_t off; uint8_t cd[512]; size_t cdlen; int cnt; } zw;

static int zw_add(zw *w, const char *name, const char *descr, int64_t len /* -1: 0-d */, const void *data, size_t dlen) {
  char hdr[256];
  uint8_t lh[30], *c = w->cd + w->cdlen;
  size_t nl = strlen(name), hl;
  uint64_t tot;
  uint32_t crc;
  int dl;
  if (len < 0) dl = snprintf(hdr + 10, 200, "{'descr': '%s', 'fortran_order': False, 'shape': (), }", descr);
  else dl = snprintf(hdr + 10, 200, "{'descr': '%s', 'fortran_order': False, 'shape': (%lld,), }", descr, (long long)len);
  hl = 10 + (size_t)dl + 1;
  hl += (64 - hl % 64) % 64;
  memset(hdr + 10 + dl, ' ', hl - 10 - (size_t)dl);
  hdr[hl - 1] = '\n';
  memcpy(hdr, "\x93NUMPY\x01\x00", 8);
  wr16((uint8_t *)hdr + 8, (unsigned)(hl - 10));
  tot = hl + (uint64_t)dlen;
  if (tot >= 0xFFFFFFFFu || w->off + 30 + nl + tot >= 0xFFFFFFFFu || w->cdlen + 46 + nl > sizeof w->cd) return -1;
  crc = crc32_upd(crc32_upd(0, hdr, hl), data, dlen);
  memset(lh, 0, sizeof lh);
  wr32(lh, 0x04034b50u);
  wr16(lh + 4, 20);
  wr16(lh + 12, 0x21); /* 1980-01-01 */
  wr32(lh + 14, crc);
  wr32(lh + 18, (uint32_t)tot);
  wr32(lh + 22, (uint32_t)tot);
  wr16(lh + 26, (unsigned)nl);
  if (fwrite(lh, 1, 30, w->f) != 30 || fwrite(name, 1, nl, w->f) != nl || fwrite(hdr, 1, hl, w->f) != hl ||
      (dlen && fwrite(data, 1, dlen, w->f) != dlen))
    return -1;
  memset(c, 0, 46);
  wr32(c, 0x02014b50u);
  wr16(c + 4, 20);
  wr16(c + 6, 20);
  wr16(c + 14, 0x21);
  wr32(c + 16, crc);
  wr32(c + 20, (uint32_t)tot);
  wr32(c + 24, (uint32_t)tot);
  wr16(c + 28, (unsigned)nl);
  wr32(c + 42, (uint32_t)w->off);
  memcpy(c + 46, name, nl);
  w->cdlen += 46 + nl;
  w->cnt++;
  w->off += 30 + nl + tot;
  return 0;
}

static size_t fmt_num(char *o, const char *key, double v) {
  size_t k = (size_t)sprintf(o, ", \"%s\": %.17g", key, v), i;
  for (i = 0; i < k; i++)
    if (o[i] == ',' && i > 0) o[i] = '.'; /* guard against non-C LC_NUMERIC */
  return k;
}

int npz_write_result(const char *path, const char *status, const double *x, int64_t nx, const double *s, int64_t ns,
                     const double *z, int64_t nz, int has_obj, double obj_primal, double obj_dual) {
  zw w;
  char *json = NULL, descr[32];
  uint8_t *u = NULL, eo[22];
  size_t k = 0, i, sl;
  int ok = 0;
  if (!path || !status || nx < 0 || ns < 0 || nz < 0) return -1;
  sl = strlen(status);
  if (!(json = (char *)malloc(6 * sl + 200))) return -1;
  k = (size_t)sprintf(json, "{\"schema_version\": 1, \"status\": \"");
  for (i = 0; i < sl; i++) {
    unsigned char c = (unsigned char)status[i];
    if (c == '"' || c == '\\') { json[k++] = '\\'; json[k++] = (char)c; }
    else if (c < 0x20 || c >= 0x7F) k += (size_t)sprintf(json + k, "\\u%04x", c);
    else json[k++] = (char)c;
  }
  json[k++] = '"';
  if (has_obj && isfinite(obj_primal) && isfinite(obj_dual)) {
    k += fmt_num(json + k, "objective_primal", obj_primal);
    k += fmt_num(json + k, "objective_dual", obj_dual);
  }
  json[k++] = '}';
  if (!(u = (uint8_t *)malloc(4 * k))) { free(json); return -1; }
  for (i = 0; i < k; i++) wr32(u + 4 * i, (unsigned char)json[i]);
  snprintf(descr, sizeof descr, "<U%lu", (unsigned long)k);
  memset(&w, 0, sizeof w);
  if (!(w.f = fopen(path, "wb"))) goto out;
  if (zw_add(&w, "metadata_json.npy", descr, -1, u, 4 * k)) goto out;
  if (x && zw_add(&w, "x.npy", "<f8", nx, x, (size_t)nx * 8)) goto out;
  if (s && zw_add(&w, "s.npy", "<f8", ns, s, (size_t)ns * 8)) goto out;
  if (z && zw_add(&w, "z.npy", "<f8", nz, z, (size_t)nz * 8)) goto out;
  memset(eo, 0, sizeof eo);
  wr32(eo, 0x06054b50u);
  wr16(eo + 8, (unsigned)w.cnt);
  wr16(eo + 10, (unsigned)w.cnt);
  wr32(eo + 12, (uint32_t)w.cdlen);
  wr32(eo + 16, (uint32_t)w.off);
  if (fwrite(w.cd, 1, w.cdlen, w.f) != w.cdlen || fwrite(eo, 1, 22, w.f) != 22) goto out;
  ok = 1;
out:
  if (w.f && fclose(w.f)) ok = 0;
  if (!ok && w.f) remove(path);
  free(json);
  free(u);
  return ok ? 0 : -1;
}
