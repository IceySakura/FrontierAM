/* npz.c - minimal reader/writer for NumPy .npz files (ZIP container of .npy).
 *
 * C99, no global state, links against zlib (only the raw-inflate and crc32
 * entry points are used; the zlib ABI is declared here so that zlib.h is not
 * required at build time).
 *
 * Build example:
 *   gcc -O2 -Wall -o prog prog.c npz.c /usr/lib/x86_64-linux-gnu/libz.so.1
 */
#define _POSIX_C_SOURCE 200809L

#include "npz.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ===================================================================== */
/* minimal zlib ABI (zlib 1.2.x, LP64).  sizeof(npz_z_stream) is passed    */
/* to inflateInit2_, which rejects it if it disagrees with the library.    */
/* ===================================================================== */
typedef struct npz_z_stream_s {
    unsigned char *next_in;
    unsigned int   avail_in;
    unsigned long  total_in;
    unsigned char *next_out;
    unsigned int   avail_out;
    unsigned long  total_out;
    const char    *msg;
    void          *state;
    void *(*zalloc)(void *, unsigned, unsigned);
    void  (*zfree)(void *, void *);
    void  *opaque;
    int           data_type;
    unsigned long adler;
    unsigned long reserved;
} npz_z_stream;

extern int inflateInit2_(npz_z_stream *strm, int windowBits,
                         const char *version, int stream_size);
extern int inflate(npz_z_stream *strm, int flush);
extern int inflateEnd(npz_z_stream *strm);
extern unsigned long crc32(unsigned long crc, const unsigned char *buf,
                           unsigned int len);

#define NPZ_Z_OK         0
#define NPZ_Z_STREAM_END 1
#define NPZ_Z_NO_FLUSH   0
/* zlib only compares the first character of this string, then sizeof(). */
#define NPZ_ZLIB_VERSION "1.2.13"

#define NPZ_CHUNK      65536u          /* inflate input chunk            */
#define NPZ_MAX_HDRL   (1u << 20)      /* max .npy header length (1 MiB) */
#define NPZ_MAX_NAME   1024            /* max zip entry name we accept   */
#define NPZ_MAX_DIM    8
#define NPZ_MAX_USIZE  ((uint64_t)8 << 30)   /* 8 GiB sanity cap        */

/* ===================================================================== */
/* small utilities                                                        */
/* ===================================================================== */
static void npz_err(const char *fmt, ...)
{
    va_list ap;
    fputs("npz: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)((unsigned)p[0] | ((unsigned)p[1] << 8));
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static void wr16(unsigned char *p, uint16_t v)
{
    p[0] = (unsigned char)(v & 0xff);
    p[1] = (unsigned char)((v >> 8) & 0xff);
}

static void wr32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xff);
    p[1] = (unsigned char)((v >> 8) & 0xff);
    p[2] = (unsigned char)((v >> 16) & 0xff);
    p[3] = (unsigned char)((v >> 24) & 0xff);
}

static int host_is_little(void)
{
    const uint16_t one = 1;
    unsigned char b[2];
    memcpy(b, &one, 2);
    return b[0] == 1;
}

/* Read exactly `n` bytes at `off`.  Returns 0 on success. */
static int read_at(FILE *fp, uint64_t off, void *buf, size_t n)
{
    if (n == 0) return 0;
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) return -1;
    if (fread(buf, 1, n, fp) != n) return -1;
    return 0;
}

static uint64_t file_size(FILE *fp, const char *path)
{
    if (fseeko(fp, 0, SEEK_END) != 0) {
        npz_err("%s: cannot seek", path);
        return (uint64_t)-1;
    }
    off_t sz = ftello(fp);
    if (sz < 0) {
        npz_err("%s: cannot tell size", path);
        return (uint64_t)-1;
    }
    rewind(fp);
    return (uint64_t)sz;
}

/* ===================================================================== */
/* ZIP reading                                                            */
/* ===================================================================== */
#define ZIP_SIG_LOCAL 0x04034b50u
#define ZIP_SIG_CDIR  0x02014b50u
#define ZIP_SIG_EOCD  0x06054b50u
#define ZIP_SIG_EOCD6 0x06064b50u
#define ZIP_SIG_LOC64 0x07064b50u
#define ZIP64_LIM32   0xffffffffu
#define ZIP64_LIM16   0xffffu

typedef struct {
    const char *name;   /* points into the central-directory block */
    uint16_t namelen;
    uint16_t method;
    uint16_t flags;
    uint32_t crc;
    uint64_t csize;
    uint64_t usize;
    uint64_t lho;
} ZipEnt;

/* Fill in usize / csize / lho from a ZIP64 extended-information extra
 * field.  Only values still sitting at their 32-bit sentinel are consumed,
 * in the order the spec prescribes (usize, csize, lho, disk). */
static void zip64_extra(const unsigned char *e, uint64_t elen,
                        uint64_t *usize, uint64_t *csize, uint64_t *lho)
{
    uint64_t p = 0;
    while (p + 4 <= elen) {
        uint16_t id = rd16(e + p), sz = rd16(e + p + 2);
        if ((uint64_t)sz > elen - p - 4) break;
        if (id == 0x0001) {
            uint64_t q = p + 4, end = p + 4 + sz;
            if (usize && *usize == ZIP64_LIM32 && q + 8 <= end) { *usize = rd64(e + q); q += 8; }
            if (csize && *csize == ZIP64_LIM32 && q + 8 <= end) { *csize = rd64(e + q); q += 8; }
            if (lho   && *lho   == ZIP64_LIM32 && q + 8 <= end) { *lho   = rd64(e + q); }
        }
        p += 4 + sz;
    }
}

/* Locate the end-of-central-directory record inside `tail` (the last
 * tail_len bytes of the file, tail_base = file offset of tail[0]) and return
 * the central directory offset/size and the entry count. */
static int zip_find_eocd(const unsigned char *tail, uint64_t tail_len,
                         uint64_t tail_base, FILE *fp,
                         uint64_t *cd_off, uint64_t *cd_size, uint64_t *nent)
{
    if (tail_len < 22) return -1;
    uint64_t i = tail_len - 22, found = (uint64_t)-1;
    for (;;) {
        if (rd32(tail + i) == ZIP_SIG_EOCD) {
            uint16_t clen = rd16(tail + i + 20);
            if (i + 22 + clen == tail_len) { found = i; break; }  /* comment runs to EOF */
        }
        if (i == 0) break;
        i--;
    }
    if (found == (uint64_t)-1) return -1;

    const unsigned char *e = tail + found;
    uint64_t n    = rd16(e + 10);
    uint64_t csz  = rd32(e + 12);
    uint64_t coff = rd32(e + 16);

    if (n == ZIP64_LIM16 || csz == ZIP64_LIM32 || coff == ZIP64_LIM32) {
        /* ZIP64: the locator sits immediately before the EOCD */
        if (found < 20 || rd32(tail + found - 20) != ZIP_SIG_LOC64) return -1;
        uint64_t z64off = rd64(tail + found - 12);
        uint64_t fsz = tail_base + tail_len;
        unsigned char z[56];
        if (fsz < 56 || z64off > fsz - 56) return -1;
        if (read_at(fp, z64off, z, sizeof z) != 0) return -1;
        if (rd32(z) != ZIP_SIG_EOCD6) return -1;
        n    = rd64(z + 32);
        csz  = rd64(z + 40);
        coff = rd64(z + 48);
    }
    *nent = n; *cd_size = csz; *cd_off = coff;
    return 0;
}

/* Parse a central-directory block that has already been read into memory. */
static int zip_parse_cd(const unsigned char *cd, uint64_t cd_len,
                        uint64_t nent, ZipEnt **out, uint64_t *out_n)
{
    if (nent > ((uint64_t)1 << 22)) {
        npz_err("absurd entry count %llu", (unsigned long long)nent);
        return -1;
    }
    ZipEnt *e = calloc((size_t)nent ? (size_t)nent : 1, sizeof *e);
    if (!e) { npz_err("out of memory"); return -1; }
    uint64_t p = 0, k = 0;
    while (k < nent && p + 46 <= cd_len) {
        if (rd32(cd + p) != ZIP_SIG_CDIR) break;
        ZipEnt *z = &e[k];
        z->flags   = rd16(cd + p + 8);
        z->method  = rd16(cd + p + 10);
        z->crc     = rd32(cd + p + 16);
        z->csize   = rd32(cd + p + 20);
        z->usize   = rd32(cd + p + 24);
        z->namelen = rd16(cd + p + 28);
        uint16_t el = rd16(cd + p + 30);
        uint16_t cl = rd16(cd + p + 32);
        z->lho     = rd32(cd + p + 42);
        if (p + 46 + (uint64_t)z->namelen + el + cl > cd_len) break;
        z->name = (const char *)cd + p + 46;
        if (z->csize == ZIP64_LIM32 || z->usize == ZIP64_LIM32 || z->lho == ZIP64_LIM32)
            zip64_extra(cd + p + 46 + z->namelen, el, &z->usize, &z->csize, &z->lho);
        p += 46 + (uint64_t)z->namelen + el + cl;
        k++;
    }
    if (k != nent) {
        npz_err("central directory truncated (%llu of %llu entries parsed)",
                (unsigned long long)k, (unsigned long long)nent);
        free(e);
        return -1;
    }
    *out = e; *out_n = k;
    return 0;
}

/* ===================================================================== */
/* .npy header parsing                                                    */
/* ===================================================================== */
enum { TC_BAD = -1, TC_F8, TC_F4, TC_I8, TC_I4, TC_I2, TC_I1,
       TC_U8, TC_U4, TC_U2, TC_U1, TC_BOOL, TC_UCS4, TC_BYTES };

typedef struct {
    int      tc;        /* type code from the enum above */
    uint32_t esz;       /* bytes per element */
    int      swap;      /* element bytes need reversing */
    uint32_t nchars;    /* TC_UCS4: code points per element */
    uint64_t count;     /* total number of elements */
    uint64_t data_off;  /* offset of the payload inside the .npy blob */
    int      ndim;
    uint64_t dims[NPZ_MAX_DIM];
} NpyHdr;

/* Find a header key (a bare word such as descr) followed by an optional
 * quote and ':'.  Returns the offset just past ':' and any spaces. */
static uint64_t hdr_find(const char *h, size_t hlen, const char *key)
{
    const uint64_t no = (uint64_t)-1;
    size_t klen = strlen(key), i;
    for (i = 0; i + klen <= hlen; i++) {
        if (memcmp(h + i, key, klen) != 0) continue;
        size_t j = i + klen;
        while (j < hlen && (h[j] == ' ' || h[j] == '\t')) j++;
        if (j < hlen && (h[j] == '\'' || h[j] == '"')) { j++; }
        while (j < hlen && (h[j] == ' ' || h[j] == '\t')) j++;
        if (j < hlen && h[j] == ':') {
            j++;
            while (j < hlen && (h[j] == ' ' || h[j] == '\t')) j++;
            return j;
        }
    }
    return no;
}

static int npy_parse(const unsigned char *buf, uint64_t len, NpyHdr *h,
                     const char *what)
{
    static const unsigned char magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
    const uint64_t no = (uint64_t)-1;
    memset(h, 0, sizeof *h);
    h->tc = TC_BAD;

    if (len < 12 || memcmp(buf, magic, 6) != 0) {
        npz_err("%s: not a .npy file", what);
        return -1;
    }
    unsigned major = buf[6];
    uint64_t hlen, hoff;
    if (major == 1)                     { hlen = rd16(buf + 8); hoff = 10; }
    else if (major == 2 || major == 3)  { hlen = rd32(buf + 8); hoff = 12; }
    else { npz_err("%s: unsupported .npy version %u", what, major); return -1; }
    if (hlen == 0 || hlen > NPZ_MAX_HDRL || hoff + hlen > len) {
        npz_err("%s: bad .npy header length %llu", what, (unsigned long long)hlen);
        return -1;
    }

    const char *hs = (const char *)buf + hoff;
    size_t hsl = (size_t)hlen;
    uint64_t pos;

    /* ---- descr ---- */
    pos = hdr_find(hs, hsl, "descr");
    if (pos == no || pos >= hsl || (hs[pos] != '\'' && hs[pos] != '"')) {
        npz_err("%s: cannot find 'descr' in the .npy header", what);
        return -1;
    }
    char qc = hs[pos++];
    uint64_t e2 = pos;
    while (e2 < hsl && hs[e2] != qc) e2++;
    if (e2 >= hsl || e2 - pos > 32) {
        npz_err("%s: malformed descr in the .npy header", what);
        return -1;
    }
    char descr[33];
    size_t dl = (size_t)(e2 - pos);
    memcpy(descr, hs + pos, dl);
    descr[dl] = '\0';

    /* ---- fortran_order ---- */
    pos = hdr_find(hs, hsl, "fortran_order");
    if (pos == no || pos >= hsl) {
        npz_err("%s: cannot find 'fortran_order'", what);
        return -1;
    }
    int fortran;
    if (pos + 4 <= hsl && memcmp(hs + pos, "True", 4) == 0)       fortran = 1;
    else if (pos + 5 <= hsl && memcmp(hs + pos, "False", 5) == 0) fortran = 0;
    else { npz_err("%s: malformed fortran_order value", what); return -1; }
    if (fortran) {
        npz_err("%s: fortran_order=True is not supported", what);
        return -1;
    }

    /* ---- shape ---- */
    pos = hdr_find(hs, hsl, "shape");
    if (pos == no || pos >= hsl || (hs[pos] != '(' && hs[pos] != '[')) {
        npz_err("%s: cannot find 'shape'", what);
        return -1;
    }
    char close = (hs[pos] == '(') ? ')' : ']';
    pos++;
    h->count = 1;
    for (;;) {
        while (pos < hsl && (hs[pos] == ' ' || hs[pos] == ',')) pos++;
        if (pos >= hsl) { npz_err("%s: unterminated shape", what); return -1; }
        if (hs[pos] == close) { pos++; break; }
        if (hs[pos] < '0' || hs[pos] > '9') {
            npz_err("%s: bad shape entry", what);
            return -1;
        }
        uint64_t v = 0;
        while (pos < hsl && hs[pos] >= '0' && hs[pos] <= '9') {
            if (v > (UINT64_MAX - 9) / 10) { npz_err("%s: shape overflow", what); return -1; }
            v = v * 10 + (uint64_t)(hs[pos] - '0');
            pos++;
        }
        if (h->ndim >= NPZ_MAX_DIM) {
            npz_err("%s: more than %d dimensions", what, NPZ_MAX_DIM);
            return -1;
        }
        h->dims[h->ndim++] = v;
        if (v != 0 && h->count > UINT64_MAX / v) {
            npz_err("%s: element count overflow", what);
            return -1;
        }
        h->count *= v;
    }
    if (h->ndim == 0) h->count = 1;   /* scalar: exactly one element */

    /* ---- dtype from descr, e.g. "<f8", "|i1", "<U157", "|S5" ---- */
    if (dl < 2) { npz_err("%s: short descr '%s'", what, descr); return -1; }
    char endian = descr[0], kind = descr[1];
    long asz = strtol(descr + 2, NULL, 10);
    /* asz < 0 marks a malformed size; string dtypes legitimately allow 0
     * (numpy writes '<U0' for an empty scalar string). */
    if (asz < 0 || asz > (1L << 28)) asz = -1;
    int swap = 0;
    if (endian == '<')      swap = !host_is_little();
    else if (endian == '>') swap = host_is_little();
    else if (endian != '|' && endian != '=' && endian != '!') {
        npz_err("%s: unknown byte order '%c' in descr '%s'", what, endian, descr);
        return -1;
    }
    h->esz = (uint32_t)asz;
    h->swap = swap;
    switch (kind) {
    case 'f': h->tc = (asz == 8) ? TC_F8 : (asz == 4) ? TC_F4 : TC_BAD; break;
    case 'i': h->tc = (asz == 8) ? TC_I8 : (asz == 4) ? TC_I4 :
                      (asz == 2) ? TC_I2 : (asz == 1) ? TC_I1 : TC_BAD; break;
    case 'u': h->tc = (asz == 8) ? TC_U8 : (asz == 4) ? TC_U4 :
                      (asz == 2) ? TC_U2 : (asz == 1) ? TC_U1 : TC_BAD; break;
    case 'b': h->tc = (asz == 1) ? TC_BOOL : TC_BAD; break;
    case 'U':
        if (asz > (1L << 28) / 4) h->tc = TC_BAD;
        else { h->tc = TC_UCS4; h->nchars = (uint32_t)asz; h->esz = (uint32_t)asz * 4u; }
        break;
    case 'S': case 'a': h->tc = (asz >= 0) ? TC_BYTES : TC_BAD; break;
    default: h->tc = TC_BAD; break;
    }
    if (h->tc == TC_BAD ||
        (h->esz == 0 && h->tc != TC_UCS4 && h->tc != TC_BYTES)) {
        npz_err("%s: unsupported dtype '%s'", what, descr);
        return -1;
    }
    if (h->esz && h->count > UINT64_MAX / h->esz) {
        npz_err("%s: payload size overflow", what);
        return -1;
    }
    h->data_off = hoff + hlen;
    if (h->count * (uint64_t)h->esz > len - h->data_off) {
        npz_err("%s: truncated payload (needs %llu bytes, have %llu)", what,
                (unsigned long long)(h->data_off + h->count * h->esz),
                (unsigned long long)len);
        return -1;
    }
    return 0;
}

/* ===================================================================== */
/* dtype conversion                                                       */
/* ===================================================================== */
static uint64_t load_u(const unsigned char *p, int esz, int swap)
{
    uint64_t v = 0;
    int i;
    if (!swap) for (i = 0; i < esz; i++) v |= (uint64_t)p[i] << (8 * i);
    else       for (i = 0; i < esz; i++) v |= (uint64_t)p[i] << (8 * (esz - 1 - i));
    return v;
}

static int64_t sign_ext(uint64_t v, int bits)
{
    if (bits < 64 && ((v >> (bits - 1)) & 1)) v |= ~((1ULL << bits) - 1ULL);
    return (int64_t)v;
}

static double u64_as_f8(uint64_t u)
{
    double d;
    memcpy(&d, &u, 8);
    return d;
}

static double u32_as_f4(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return (double)f;
}

#define NPZ_CONV_LOOP(EXPR)                                     \
    do {                                                        \
        uint64_t i_;                                            \
        for (i_ = 0; i_ < n; i_++) {                            \
            const unsigned char *p_ = src + i_ * (uint64_t)esz; \
            dst[i_] = (EXPR);                                   \
        }                                                       \
    } while (0)

/* Convert `n` elements to double.  The switch keeps the per-element work
 * branch-free inside the hot loop. */
static int conv_to_double(const unsigned char *src, double *dst, uint64_t n,
                          const NpyHdr *h, const char *what)
{
    const int esz = (int)h->esz, sw = h->swap;
    switch (h->tc) {
    case TC_F8:   NPZ_CONV_LOOP(u64_as_f8(load_u(p_, 8, sw)));                  break;
    case TC_F4:   NPZ_CONV_LOOP(u32_as_f4((uint32_t)load_u(p_, 4, sw)));        break;
    case TC_I8:   NPZ_CONV_LOOP((double)sign_ext(load_u(p_, 8, sw), 64));       break;
    case TC_I4:   NPZ_CONV_LOOP((double)sign_ext(load_u(p_, 4, sw), 32));       break;
    case TC_I2:   NPZ_CONV_LOOP((double)sign_ext(load_u(p_, 2, sw), 16));       break;
    case TC_I1:   NPZ_CONV_LOOP((double)(int8_t)p_[0]);                         break;
    case TC_U8:   NPZ_CONV_LOOP((double)load_u(p_, 8, sw));                     break;
    case TC_U4:   NPZ_CONV_LOOP((double)(uint32_t)load_u(p_, 4, sw));           break;
    case TC_U2:   NPZ_CONV_LOOP((double)(uint16_t)load_u(p_, 2, sw));           break;
    case TC_U1:   NPZ_CONV_LOOP((double)p_[0]);                                 break;
    case TC_BOOL: NPZ_CONV_LOOP(p_[0] ? 1.0 : 0.0);                             break;
    default:
        npz_err("%s: cannot convert this dtype to float64", what);
        return -1;
    }
    return 0;
}

/* Convert `n` elements to long (int64).  Floats are rounded to nearest. */
static int conv_to_long(const unsigned char *src, long *dst, uint64_t n,
                        const NpyHdr *h, const char *what)
{
    const int esz = (int)h->esz, sw = h->swap;
    switch (h->tc) {
    case TC_I8:   NPZ_CONV_LOOP((long)sign_ext(load_u(p_, 8, sw), 64));         break;
    case TC_I4:   NPZ_CONV_LOOP((long)sign_ext(load_u(p_, 4, sw), 32));         break;
    case TC_I2:   NPZ_CONV_LOOP((long)sign_ext(load_u(p_, 2, sw), 16));         break;
    case TC_I1:   NPZ_CONV_LOOP((long)(int8_t)p_[0]);                           break;
    case TC_U8:   NPZ_CONV_LOOP((long)load_u(p_, 8, sw));                       break;
    case TC_U4:   NPZ_CONV_LOOP((long)(uint32_t)load_u(p_, 4, sw));             break;
    case TC_U2:   NPZ_CONV_LOOP((long)(uint16_t)load_u(p_, 2, sw));             break;
    case TC_U1:   NPZ_CONV_LOOP((long)p_[0]);                                   break;
    case TC_BOOL: NPZ_CONV_LOOP((long)(p_[0] ? 1 : 0));                         break;
    case TC_F8:   NPZ_CONV_LOOP((long)(u64_as_f8(load_u(p_, 8, sw)) +
                     (u64_as_f8(load_u(p_, 8, sw)) >= 0.0 ? 0.5 : -0.5)));      break;
    case TC_F4:   NPZ_CONV_LOOP((long)(u32_as_f4((uint32_t)load_u(p_, 4, sw)) +
                     (u32_as_f4((uint32_t)load_u(p_, 4, sw)) >= 0.0 ? 0.5 : -0.5))); break;
    default:
        npz_err("%s: cannot convert this dtype to int64", what);
        return -1;
    }
    return 0;
}

/* Convert one .npy blob into a freshly malloc'd double array. */
static int npy_to_double(const unsigned char *buf, uint64_t len,
                         double **out, long *out_n, const char *what)
{
    NpyHdr h;
    *out = NULL; *out_n = 0;
    if (npy_parse(buf, len, &h, what) != 0) return -1;
    if (h.count > (uint64_t)PTRDIFF_MAX / sizeof(double)) {
        npz_err("%s: too many elements", what);
        return -1;
    }
    double *d = NULL;
    if (h.count) {
        d = malloc((size_t)h.count * sizeof *d);
        if (!d) { npz_err("%s: out of memory", what); return -1; }
        if (conv_to_double(buf + h.data_off, d, h.count, &h, what) != 0) {
            free(d);
            return -1;
        }
    }
    *out = d;
    *out_n = (long)h.count;
    return 0;
}

static int npy_to_long(const unsigned char *buf, uint64_t len,
                       long **out, long *out_n, const char *what)
{
    NpyHdr h;
    *out = NULL; *out_n = 0;
    if (npy_parse(buf, len, &h, what) != 0) return -1;
    if (h.count > (uint64_t)PTRDIFF_MAX / sizeof(long)) {
        npz_err("%s: too many elements", what);
        return -1;
    }
    long *d = NULL;
    if (h.count) {
        d = malloc((size_t)h.count * sizeof *d);
        if (!d) { npz_err("%s: out of memory", what); return -1; }
        if (conv_to_long(buf + h.data_off, d, h.count, &h, what) != 0) {
            free(d);
            return -1;
        }
    }
    *out = d;
    *out_n = (long)h.count;
    return 0;
}

/* Convert a unicode / byte-string .npy blob into a NUL-terminated UTF-8
 * string (all elements concatenated; for numpy's 0-d scalars that is one). */
static int npy_to_string(const unsigned char *buf, uint64_t len, char **out,
                         const char *what)
{
    NpyHdr h;
    *out = NULL;
    if (npy_parse(buf, len, &h, what) != 0) return -1;

    if (h.tc == TC_BYTES) {
        uint64_t total = h.count * (uint64_t)h.esz;
        char *s = malloc((size_t)total + 1);
        if (!s) { npz_err("%s: out of memory", what); return -1; }
        memcpy(s, buf + h.data_off, (size_t)total);
        s[total] = '\0';
        *out = s;
        return 0;
    }
    if (h.tc != TC_UCS4) {
        npz_err("%s: expected a unicode or byte-string array", what);
        return -1;
    }
    if (h.nchars && h.count > SIZE_MAX / ((size_t)h.nchars * 4 + 1)) {
        npz_err("%s: string too long", what);
        return -1;
    }
    uint64_t nch = h.count * (uint64_t)h.nchars;
    char *s = malloc((size_t)nch * 4 + 1);
    if (!s) { npz_err("%s: out of memory", what); return -1; }
    const unsigned char *src = buf + h.data_off;
    size_t o = 0;
    uint64_t i;
    for (i = 0; i < nch; i++) {
        uint32_t c = (uint32_t)load_u(src + i * 4, 4, h.swap);
        if (c > 0x10FFFFu || (c >= 0xD800u && c <= 0xDFFFu)) {
            s[o++] = '?';
        } else if (c < 0x80u) {
            s[o++] = (char)c;
        } else if (c < 0x800u) {
            s[o++] = (char)(0xC0u | (c >> 6));
            s[o++] = (char)(0x80u | (c & 0x3Fu));
        } else if (c < 0x10000u) {
            s[o++] = (char)(0xE0u | (c >> 12));
            s[o++] = (char)(0x80u | ((c >> 6) & 0x3Fu));
            s[o++] = (char)(0x80u | (c & 0x3Fu));
        } else {
            s[o++] = (char)(0xF0u | (c >> 18));
            s[o++] = (char)(0x80u | ((c >> 12) & 0x3Fu));
            s[o++] = (char)(0x80u | ((c >> 6) & 0x3Fu));
            s[o++] = (char)(0x80u | (c & 0x3Fu));
        }
    }
    s[o] = '\0';
    *out = s;
    return 0;
}

/* ===================================================================== */
/* entry extraction                                                       */
/* ===================================================================== */
static uint32_t crc_buf(const unsigned char *buf, uint64_t len)
{
    unsigned long c = 0;
    uint64_t off = 0;
    while (off < len) {
        uint64_t chunk = len - off;
        if (chunk > 0x40000000ULL) chunk = 0x40000000ULL;
        c = crc32(c, buf + off, (unsigned int)chunk);
        off += chunk;
    }
    return (uint32_t)c;
}

/* Read and decompress one zip entry into a fresh buffer.  Compressed bytes
 * are pulled in NPZ_CHUNK-sized pieces, so peak memory stays at the size of
 * the decompressed entry plus 64 KiB. */
static int zip_extract(FILE *fp, const ZipEnt *e, uint64_t fsz,
                       unsigned char **out, uint64_t *out_len)
{
    unsigned char lh[30];
    *out = NULL; *out_len = 0;

    if (e->lho > fsz || fsz - e->lho < 30) {
        npz_err("local header offset out of range");
        return -1;
    }
    if (read_at(fp, e->lho, lh, 30) != 0) {
        npz_err("cannot read local header");
        return -1;
    }
    if (rd32(lh) != ZIP_SIG_LOCAL) {
        npz_err("bad local header signature");
        return -1;
    }
    uint16_t lflags  = rd16(lh + 6);
    uint16_t lmethod = rd16(lh + 8);
    uint64_t lcsize  = rd32(lh + 18);
    uint64_t lusize  = rd32(lh + 22);
    uint16_t nl      = rd16(lh + 26);
    uint16_t el      = rd16(lh + 28);   /* trust the LOCAL extra length */

    if (lmethod != e->method) {
        npz_err("local/central method mismatch (%u vs %u)", lmethod, e->method);
        return -1;
    }

    /* The central directory is authoritative.  numpy >= 1.25 writes local
     * headers with 0xFFFFFFFF placeholders plus a zip64 extra field holding
     * the real sizes, so fall back to that field, then to the local 32-bit
     * values (which are meaningless when a data descriptor is in use). */
    uint64_t usize = e->usize, csize = e->csize;
    if ((usize == ZIP64_LIM32 || csize == ZIP64_LIM32) && el) {
        unsigned char *ex = malloc(el);
        if (!ex) { npz_err("out of memory"); return -1; }
        if (read_at(fp, e->lho + 30 + nl, ex, el) == 0)
            zip64_extra(ex, el, &usize, &csize, NULL);
        free(ex);
    }
    if (!(lflags & 0x08)) {             /* no data descriptor */
        if (usize == ZIP64_LIM32 && lusize != ZIP64_LIM32) usize = lusize;
        if (csize == ZIP64_LIM32 && lcsize != ZIP64_LIM32) csize = lcsize;
    }

    uint64_t data_off = e->lho + 30 + (uint64_t)nl + (uint64_t)el;
    if (data_off > fsz) { npz_err("entry data offset out of range"); return -1; }
    if (usize == ZIP64_LIM32) {
        npz_err("uncompressed size is unknown for this entry");
        return -1;
    }
    if (usize > NPZ_MAX_USIZE || usize > (uint64_t)SIZE_MAX) {
        npz_err("implausible uncompressed size %llu", (unsigned long long)usize);
        return -1;
    }

    unsigned char *buf = malloc((size_t)usize ? (size_t)usize : 1);
    if (!buf) {
        npz_err("out of memory (%llu bytes)", (unsigned long long)usize);
        return -1;
    }

    if (e->method == 0) {                                  /* STORED */
        if (usize > fsz - data_off) {
            free(buf); npz_err("truncated stored entry"); return -1;
        }
        if (read_at(fp, data_off, buf, (size_t)usize) != 0) {
            free(buf); npz_err("cannot read stored entry"); return -1;
        }
    } else if (e->method == 8) {                           /* DEFLATE */
        int have_csize = (csize != ZIP64_LIM32);
        uint64_t remaining = have_csize ? csize : (fsz - data_off);
        npz_z_stream s;
        memset(&s, 0, sizeof s);
        if (inflateInit2_(&s, -15, NPZ_ZLIB_VERSION,
                          (int)sizeof(npz_z_stream)) != NPZ_Z_OK) {
            free(buf);
            npz_err("inflateInit2_ rejected our z_stream (sizeof=%lu)",
                    (unsigned long)sizeof(npz_z_stream));
            return -1;
        }
        unsigned char ibuf[NPZ_CHUNK];
        uint64_t got = 0;
        int failed = 0;
        for (;;) {
            if (s.avail_in == 0) {
                if (remaining == 0) { failed = 1; npz_err("truncated deflate stream"); break; }
                size_t want = NPZ_CHUNK;
                if (have_csize && remaining < want) want = (size_t)remaining;
                if (fseeko(fp, (off_t)(data_off + s.total_in), SEEK_SET) != 0) {
                    failed = 1; npz_err("seek failed"); break;
                }
                size_t rd = fread(ibuf, 1, want, fp);
                if (rd == 0) { failed = 1; npz_err("truncated deflate stream"); break; }
                remaining -= rd;
                s.next_in = ibuf;
                s.avail_in = (unsigned int)rd;
            }
            if (got == usize) {
                failed = 1;
                npz_err("deflate output exceeds the declared size %llu",
                        (unsigned long long)usize);
                break;
            }
            uint64_t space = usize - got;
            if (space > 0xFFFFFFFFULL) space = 0xFFFFFFFFULL;
            s.next_out = buf + got;
            s.avail_out = (unsigned int)space;
            int rc = inflate(&s, NPZ_Z_NO_FLUSH);
            got = (uint64_t)(s.next_out - buf);
            if (rc == NPZ_Z_STREAM_END) break;
            if (rc != NPZ_Z_OK) {
                failed = 1;
                npz_err("inflate error %d (%s)", rc, s.msg ? s.msg : "?");
                break;
            }
        }
        inflateEnd(&s);
        if (failed) { free(buf); return -1; }
        if (got != usize) {
            free(buf);
            npz_err("inflated %llu bytes, expected %llu",
                    (unsigned long long)got, (unsigned long long)usize);
            return -1;
        }
    } else {
        free(buf);
        npz_err("unsupported zip compression method %u", e->method);
        return -1;
    }

    if (usize > 0 && e->crc != 0 && crc_buf(buf, usize) != e->crc) {
        free(buf);
        npz_err("CRC32 mismatch (expected %08x)", e->crc);
        return -1;
    }
    *out = buf;
    *out_len = usize;
    return 0;
}

/* ===================================================================== */
/* npz_read                                                               */
/* ===================================================================== */
enum { F_Q = 0, F_B, F_PDATA, F_PIDX, F_PIP, F_PSHAPE,
       F_ADATA, F_AIDX, F_AIP, F_ASHAPE, F_META, F_N };

static const char *const npz_field_names[F_N] = {
    "q", "b", "P_data", "P_indices", "P_indptr", "P_shape",
    "A_data", "A_indices", "A_indptr", "A_shape", "metadata_json"
};

/* _own[] slot that keeps each field's memory alive (-1 = temporary) */
static const int npz_field_slot[F_N] = {
    0, 1, 2, 3, 4, -1, 5, 6, 7, -1, 8
};

static int name_match(const char *name, uint16_t namelen, int *field)
{
    size_t bl = namelen;
    if (bl > 4 && memcmp(name + bl - 4, ".npy", 4) == 0) bl -= 4;
    if (bl == 0 || bl > 64) return 0;
    int i;
    for (i = 0; i < F_N; i++) {
        size_t kl = strlen(npz_field_names[i]);
        if (kl == bl && memcmp(name, npz_field_names[i], bl) == 0) {
            *field = i;
            return 1;
        }
    }
    return 0;
}

int npz_read(const char *path, NpzProblem *out)
{
    if (!path || !out) { npz_err("null argument"); return 1; }
    memset(out, 0, sizeof *out);

    FILE *fp = fopen(path, "rb");
    if (!fp) { npz_err("%s: cannot open", path); return 1; }
    uint64_t fsz = file_size(fp, path);
    if (fsz == (uint64_t)-1) { fclose(fp); return 1; }
    if (fsz < 22) {
        fclose(fp);
        npz_err("%s: too small to be a zip archive", path);
        return 1;
    }

    int ret = 1;
    unsigned char *cd = NULL, *blob = NULL;
    ZipEnt *ents = NULL;
    long pshape0 = -1, pshape1 = -1, ashape0 = -1, ashape1 = -1;
    /* lengths of the data / indptr arrays when present, for the structural
     * consistency check performed after the walk */
    long pdlen = -1, adlen = -1, piplen = -1, aiplen = -1;

    /* ---- tail scan for the EOCD ---- */
    uint64_t tail_len  = fsz < 66000 ? fsz : 66000;
    uint64_t tail_base = fsz - tail_len;
    unsigned char *tail = malloc((size_t)tail_len);
    if (!tail) { npz_err("out of memory"); fclose(fp); return 1; }
    if (read_at(fp, tail_base, tail, (size_t)tail_len) != 0) {
        npz_err("%s: cannot read the end of the file", path);
        free(tail); fclose(fp); return 1;
    }
    uint64_t cd_off = 0, cd_size = 0, nent = 0;
    int eok = zip_find_eocd(tail, tail_len, tail_base, fp, &cd_off, &cd_size, &nent);
    free(tail);
    if (eok != 0) {
        npz_err("%s: no end-of-central-directory record (not a zip?)", path);
        fclose(fp);
        return 1;
    }
    if (cd_size > fsz || cd_off > fsz || cd_off + cd_size > fsz) {
        npz_err("%s: central directory out of range", path);
        fclose(fp);
        return 1;
    }

    cd = malloc((size_t)cd_size ? (size_t)cd_size : 1);
    if (!cd) { npz_err("out of memory"); fclose(fp); return 1; }
    if (read_at(fp, cd_off, cd, (size_t)cd_size) != 0) {
        npz_err("%s: cannot read the central directory", path);
        goto done;
    }
    uint64_t ne = 0;
    if (zip_parse_cd(cd, cd_size, nent, &ents, &ne) != 0) goto done;

    /* ---- walk the entries ---- */
    int seen[F_N], found_any = 0;
    memset(seen, 0, sizeof seen);
    uint64_t i;
    for (i = 0; i < ne; i++) {
        const ZipEnt *e = &ents[i];
        int field = -1;
        if (e->namelen == 0 || e->namelen > NPZ_MAX_NAME) continue;
        if (e->name[e->namelen - 1] == '/') continue;         /* directory */
        if (!name_match(e->name, e->namelen, &field)) continue;
        if (seen[field]) continue;                            /* first wins */

        uint64_t blen = 0;
        free(blob); blob = NULL;
        if (zip_extract(fp, e, fsz, &blob, &blen) != 0) {
            npz_err("%s: entry '%.*s' could not be read",
                    path, (int)e->namelen, e->name);
            goto done;
        }
        char what[NPZ_MAX_NAME + 8];
        memcpy(what, e->name, e->namelen);
        what[e->namelen] = '\0';

        if (field == F_META) {
            char *s = NULL;
            if (npy_to_string(blob, blen, &s, what) != 0) goto done;
            out->metadata_json = s;
            out->_own[npz_field_slot[F_META]] = s;
        } else if (field == F_PSHAPE || field == F_ASHAPE) {
            long *v = NULL, vn = 0;
            if (npy_to_long(blob, blen, &v, &vn, what) != 0) goto done;
            if (field == F_PSHAPE) {
                if (vn >= 1) pshape0 = v[0];
                if (vn >= 2) pshape1 = v[1];
            } else {
                if (vn >= 1) ashape0 = v[0];
                if (vn >= 2) ashape1 = v[1];
            }
            free(v);
        } else if (field == F_PIDX || field == F_PIP ||
                   field == F_AIDX || field == F_AIP) {
            long *v = NULL, vn = 0;
            if (npy_to_long(blob, blen, &v, &vn, what) != 0) goto done;
            out->_own[npz_field_slot[field]] = v;
            switch (field) {
            case F_PIDX: out->P_indices = v; break;
            case F_PIP:  out->P_indptr  = v; piplen = vn; break;
            case F_AIDX: out->A_indices = v; break;
            default:     out->A_indptr  = v; aiplen = vn; break;
            }
        } else {
            /* float64 arrays */
            double *v = NULL; long vn = 0;
            if (npy_to_double(blob, blen, &v, &vn, what) != 0) goto done;
            out->_own[npz_field_slot[field]] = v;
            switch (field) {
            case F_Q:     out->q = v; out->n = vn; break;
            case F_B:     out->b = v; out->m = vn; break;
            case F_PDATA: out->P_data = v; pdlen = vn; break;
            default:      out->A_data = v; adlen = vn; break;
            }
        }
        free(blob); blob = NULL;
        seen[field] = 1;
        found_any = 1;
    }

    if (!found_any) {
        npz_err("%s: no recognized arrays (expected q/b/P_*/A_*/metadata_json)",
                path);
        goto done;
    }

    /* ---- structural consistency: a CSC matrix with `ncols` columns has
     * exactly ncols+1 indptr entries and indptr[ncols] == len(data).
     * For A the column count is A_shape[1] (= Am); for P it is P_shape[1]. ---- */
    if (pshape1 >= 0 && piplen >= 0) {
        if (piplen != pshape1 + 1) {
            npz_err("%s: P_indptr has %ld entries but P_shape[1]=%ld",
                    path, piplen, pshape1);
            goto done;
        }
        if (pdlen >= 0 && out->P_indptr[pshape1] != pdlen) {
            npz_err("%s: P_indptr[%ld]=%ld disagrees with len(P_data)=%ld",
                    path, pshape1, out->P_indptr[pshape1], pdlen);
            goto done;
        }
    }
    if (ashape1 >= 0 && aiplen >= 0) {
        if (aiplen != ashape1 + 1) {
            npz_err("%s: A_indptr has %ld entries but A_shape[1]=%ld",
                    path, aiplen, ashape1);
            goto done;
        }
        if (adlen >= 0 && out->A_indptr[ashape1] != adlen) {
            npz_err("%s: A_indptr[%ld]=%ld disagrees with len(A_data)=%ld",
                    path, ashape1, out->A_indptr[ashape1], adlen);
            goto done;
        }
    }

    /* ---- dimensions ---- */
    out->Pn = (pshape0 >= 0) ? pshape0 : out->n;
    out->An = (ashape0 >= 0) ? ashape0 : out->m;
    out->Am = (ashape1 >= 0) ? ashape1 : out->n;
    if (out->Pn < 0) out->Pn = 0;
    if (out->An < 0) out->An = 0;
    if (out->Am < 0) out->Am = 0;

    ret = 0;

done:
    free(blob);
    free(ents);
    free(cd);
    fclose(fp);
    if (ret != 0) {
        int k;
        for (k = 0; k < 16; k++) { free(out->_own[k]); out->_own[k] = NULL; }
        memset(out, 0, sizeof *out);   /* never expose half-built state */
    }
    return ret;
}

void npz_free(NpzProblem *p)
{
    int i;
    if (!p) return;
    for (i = 0; i < 16; i++) { free(p->_own[i]); p->_own[i] = NULL; }
    memset(p, 0, sizeof *p);
}

/* ===================================================================== */
/* npz_write                                                              */
/* ===================================================================== */
typedef struct {
    char    *zname;      /* entry name including ".npy" */
    size_t   zlen;
    uint32_t crc;
    uint64_t size;       /* .npy header + payload */
    uint64_t lho;
} WEnt;

#define NPZ_DOS_TIME 0      /* 00:00:00   */
#define NPZ_DOS_DATE 0x21   /* 1980-01-01 */

/* Build a numpy v1.0 .npy header.  Returns its length, or 0 on failure.
 * (10 + length) % 64 == 0 and the header ends with '\n'. */
static size_t npy_make_header(char *out, size_t outsz, const char *descr,
                              int ndim, const long *dims)
{
    char body[256];
    int n = snprintf(body, sizeof body,
                     "{'descr': '%s', 'fortran_order': False, 'shape': (", descr);
    if (n < 0 || (size_t)n >= sizeof body) return 0;
    size_t bl = (size_t)n;
    int i;
    for (i = 0; i < ndim; i++) {
        n = snprintf(body + bl, sizeof body - bl, "%ld,", dims[i]);
        if (n < 0 || bl + (size_t)n >= sizeof body) return 0;
        bl += (size_t)n;
    }
    n = snprintf(body + bl, sizeof body - bl, ") }");
    if (n < 0 || bl + (size_t)n >= sizeof body) return 0;
    bl += (size_t)n;

    size_t pad = (64u - ((10u + bl + 1u) % 64u)) % 64u;
    if (10u + bl + pad + 1u > outsz) return 0;
    if (bl + pad + 1u > 0xffffu) return 0;
    memcpy(out, "\x93NUMPY", 6);
    out[6] = 1; out[7] = 0;                      /* version 1.0 */
    wr16((unsigned char *)out + 8, (uint16_t)(bl + pad + 1));
    memcpy(out + 10, body, bl);
    memset(out + 10 + bl, ' ', pad);
    out[10 + bl + pad] = '\n';
    return 10u + bl + pad + 1u;
}

static int wr_all(FILE *fp, const void *p, size_t n)
{
    return (n == 0 || fwrite(p, 1, n, fp) == n) ? 0 : -1;
}

/* Encode a UTF-8 string as UCS4-LE.  Returns the code point count, or
 * (uint32_t)-1 on failure; *buf receives 4*n malloc'd bytes. */
static uint32_t utf8_to_ucs4(const char *s, unsigned char **buf)
{
    *buf = NULL;
    size_t slen = strlen(s), i;
    uint32_t n = 0;
    for (i = 0; i < slen; ) {                    /* count code points */
        unsigned char c = (unsigned char)s[i];
        size_t adv = (c < 0x80) ? 1 : (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        if (adv > slen - i) adv = slen - i;
        i += adv;
        n++;
    }
    unsigned char *b = malloc((size_t)n * 4 + 4);
    if (!b) { npz_err("out of memory"); return (uint32_t)-1; }
    uint32_t k = 0;
    for (i = 0; i < slen; ) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        size_t adv;
        if (c < 0x80) { cp = c; adv = 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < slen) {
            cp  = (uint32_t)(c & 0x1F) << 6;
            cp |= (uint32_t)((unsigned char)s[i + 1] & 0x3F);
            adv = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < slen) {
            cp  = (uint32_t)(c & 0x0F) << 12;
            cp |= (uint32_t)((unsigned char)s[i + 1] & 0x3F) << 6;
            cp |= (uint32_t)((unsigned char)s[i + 2] & 0x3F);
            adv = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < slen) {
            cp  = (uint32_t)(c & 0x07) << 18;
            cp |= (uint32_t)((unsigned char)s[i + 1] & 0x3F) << 12;
            cp |= (uint32_t)((unsigned char)s[i + 2] & 0x3F) << 6;
            cp |= (uint32_t)((unsigned char)s[i + 3] & 0x3F);
            adv = 4;
        } else { cp = '?'; adv = 1; }            /* malformed byte */
        wr32(b + (size_t)k * 4, cp);
        k++;
        i += adv;
    }
    *buf = b;
    return n;
}

int npz_write(const char *path, const NpzArrayOut *arrays, int narrays,
              const char *metadata_json, const double *x, long nx,
              const double *s, long ns, const double *z, long nz)
{
    if (!path) { npz_err("null path"); return 1; }
    if (narrays < 0 || (narrays > 0 && !arrays)) { npz_err("bad array list"); return 1; }
    if (!host_is_little()) {
        npz_err("the writer needs a little-endian host");
        return 1;
    }

    long total = (long)narrays + 4;               /* + x, s, z, metadata_json */
    WEnt *w = calloc((size_t)total, sizeof *w);
    if (!w) { npz_err("out of memory"); return 1; }
    long nw = 0;
    int ret = 1, i;
    FILE *fp = NULL;
    unsigned char *ucs = NULL;
    uint32_t ncp;

    fp = fopen(path, "wb");
    if (!fp) { npz_err("%s: cannot create", path); free(w); return 1; }

    ncp = utf8_to_ucs4(metadata_json ? metadata_json : "", &ucs);
    if (ncp == (uint32_t)-1) goto wdone;

    for (i = 0; i < total; i++) {
        const char *base;
        const double *dvec = NULL;
        long dlen = 0;
        int is_str = 0;

        if (i < narrays) {
            if (!arrays[i].name) { npz_err("array %d has no name", i); goto wdone; }
            base = arrays[i].name;
            dvec = arrays[i].data;
            dlen = arrays[i].len;
        } else if (i == narrays + 0) { base = "x"; dvec = x; dlen = nx; if (nx < 0) continue; }
        else if (i == narrays + 1)   { base = "s"; dvec = s; dlen = ns; if (ns < 0) continue; }
        else if (i == narrays + 2)   { base = "z"; dvec = z; dlen = nz; if (nz < 0) continue; }
        else                         { base = "metadata_json"; is_str = 1; }

        if (dlen < 0) { npz_err("array '%s' has a negative length", base); goto wdone; }
        if (!is_str && dlen > 0 && !dvec) {
            npz_err("array '%s' is NULL with a nonzero length", base);
            goto wdone;
        }

        size_t bl = strlen(base);
        if (bl == 0 || bl > 4000) { npz_err("bad array name '%s'", base); goto wdone; }
        int addsfx = !(bl > 4 && memcmp(base + bl - 4, ".npy", 4) == 0);
        char *zn = malloc(bl + 5);
        if (!zn) { npz_err("out of memory"); goto wdone; }
        memcpy(zn, base, bl);
        if (addsfx) { memcpy(zn + bl, ".npy", 4); bl += 4; }
        zn[bl] = '\0';

        char descr[32];
        long dims[1];
        int ndim;
        if (is_str) { snprintf(descr, sizeof descr, "<U%u", ncp); ndim = 0; }
        else        { snprintf(descr, sizeof descr, "<f8"); dims[0] = dlen; ndim = 1; }

        char hdr[512];
        size_t hl = npy_make_header(hdr, sizeof hdr, descr, ndim, dims);
        if (hl == 0) {
            npz_err("cannot build a .npy header for '%s'", zn);
            free(zn);
            goto wdone;
        }

        uint64_t payload = is_str ? (uint64_t)ncp * 4u : (uint64_t)dlen * 8u;
        uint64_t size = (uint64_t)hl + payload;
        if (size > ZIP64_LIM32) {
            npz_err("entry '%s' is too large for this writer", zn);
            free(zn);
            goto wdone;
        }

        uint32_t crc = (uint32_t)crc32(0, (const unsigned char *)hdr, (unsigned int)hl);
        if (is_str) {
            if (payload) crc = (uint32_t)crc32(crc, ucs, (unsigned int)payload);
        } else {
            long off = 0;
            while (off < dlen) {
                long chunk = dlen - off;
                if (chunk > 0x10000000L) chunk = 0x10000000L;
                crc = (uint32_t)crc32(crc, (const unsigned char *)(dvec + off),
                                      (unsigned int)((size_t)chunk * sizeof(double)));
                off += chunk;
            }
        }

        off_t lho = ftello(fp);
        if (lho < 0) { npz_err("ftello failed"); free(zn); goto wdone; }

        WEnt *we = &w[nw++];
        we->zname = zn; we->zlen = bl; we->crc = crc;
        we->size = size; we->lho = (uint64_t)lho;

        unsigned char lh[30];
        wr32(lh + 0,  ZIP_SIG_LOCAL);
        wr16(lh + 4,  20);                       /* version needed */
        wr16(lh + 6,  0);                        /* flags */
        wr16(lh + 8,  0);                        /* method 0 = stored */
        wr16(lh + 10, NPZ_DOS_TIME);
        wr16(lh + 12, NPZ_DOS_DATE);
        wr32(lh + 14, crc);
        wr32(lh + 18, (uint32_t)size);
        wr32(lh + 22, (uint32_t)size);
        wr16(lh + 26, (uint16_t)bl);
        wr16(lh + 28, 0);
        if (wr_all(fp, lh, 30) || wr_all(fp, zn, bl) || wr_all(fp, hdr, hl)) {
            npz_err("%s: write failed", path);
            goto wdone;
        }
        if (is_str) {
            if (payload && wr_all(fp, ucs, (size_t)payload)) {
                npz_err("%s: write failed", path);
                goto wdone;
            }
        } else {
            long off = 0;
            while (off < dlen) {
                long chunk = dlen - off;
                if (chunk > 0x10000000L) chunk = 0x10000000L;
                if (wr_all(fp, dvec + off, (size_t)chunk * sizeof(double))) {
                    npz_err("%s: write failed", path);
                    goto wdone;
                }
                off += chunk;
            }
        }
    }

    /* ---- central directory + EOCD ---- */
    {
        off_t co = ftello(fp);
        if (co < 0) { npz_err("ftello failed"); goto wdone; }
        for (i = 0; i < nw; i++) {
            unsigned char ch[46];
            const WEnt *we = &w[i];
            wr32(ch + 0,  ZIP_SIG_CDIR);
            wr16(ch + 4,  (uint16_t)((3 << 8) | 20));  /* made by unix, v2.0 */
            wr16(ch + 6,  20);                         /* version needed */
            wr16(ch + 8,  0);                          /* flags */
            wr16(ch + 10, 0);                          /* method: stored */
            wr16(ch + 12, NPZ_DOS_TIME);
            wr16(ch + 14, NPZ_DOS_DATE);
            wr32(ch + 16, we->crc);
            wr32(ch + 20, (uint32_t)we->size);
            wr32(ch + 24, (uint32_t)we->size);
            wr16(ch + 28, (uint16_t)we->zlen);
            wr16(ch + 30, 0);                          /* extra len */
            wr16(ch + 32, 0);                          /* comment len */
            wr16(ch + 34, 0);                          /* disk number */
            wr16(ch + 36, 0);                          /* internal attrs */
            wr32(ch + 38, 0x81A40000u);                /* regular file 0644 */
            wr32(ch + 42, (uint32_t)we->lho);
            if (wr_all(fp, ch, 46) || wr_all(fp, we->zname, we->zlen)) {
                npz_err("%s: write failed", path);
                goto wdone;
            }
        }
        off_t ce = ftello(fp);
        if (ce < 0) { npz_err("ftello failed"); goto wdone; }
        uint64_t cd_size = (uint64_t)(ce - co);
        if (cd_size > ZIP64_LIM32 || (uint64_t)co > ZIP64_LIM32 || nw > ZIP64_LIM16) {
            npz_err("archive too large for this writer");
            goto wdone;
        }
        unsigned char eocd[22];
        wr32(eocd + 0,  ZIP_SIG_EOCD);
        wr16(eocd + 4,  0);
        wr16(eocd + 6,  0);
        wr16(eocd + 8,  (uint16_t)nw);
        wr16(eocd + 10, (uint16_t)nw);
        wr32(eocd + 12, (uint32_t)cd_size);
        wr32(eocd + 16, (uint32_t)co);
        wr16(eocd + 20, 0);
        if (wr_all(fp, eocd, 22)) { npz_err("%s: write failed", path); goto wdone; }
    }

    if (fflush(fp) != 0) { npz_err("%s: flush failed", path); goto wdone; }
    ret = 0;

wdone:
    free(ucs);
    for (i = 0; i < nw; i++) free(w[i].zname);
    free(w);
    if (fp && fclose(fp) != 0 && ret == 0) { npz_err("%s: close failed", path); ret = 1; }
    if (ret != 0) remove(path);
    return ret;
}
