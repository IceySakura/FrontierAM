/* npzio.c - NPZ problem reader / result writer without Python.
 *
 * Reading: the file is mmap'ed (fallback: read into memory), the zip central
 * directory is parsed (ZIP64 aware), and every needed .npy entry is inflated
 * (raw DEFLATE via the system zlib) or copied (STORED) directly into its final
 * array whenever the stored dtype matches the target type; other dtypes are
 * converted in chunks.  Matrices are validated and canonicalized (sorted rows,
 * duplicates summed, explicit zeros dropped).
 *
 * Writing: STORED zip with NPY v1.0 entries and correct CRC-32 values.
 *
 * Requires a little-endian host (x86-64, ARM64).  Link with -l:libz.so.1.
 */
#define _GNU_SOURCE
#include "npzio.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "npzio.c requires a little-endian host"
#endif

/* ------------------------------------------------------------------------ */
/* Minimal zlib ABI (no zlib.h on the build machine).                        */
/* ------------------------------------------------------------------------ */
typedef struct npz_z_stream_s {
    const unsigned char *next_in;
    unsigned int avail_in;
    unsigned long total_in;
    unsigned char *next_out;
    unsigned int avail_out;
    unsigned long total_out;
    const char *msg;
    void *state;
    void *(*zalloc)(void *, unsigned int, unsigned int);
    void (*zfree)(void *, void *);
    void *opaque;
    int data_type;
    unsigned long adler;
    unsigned long reserved;
} npz_z_stream;

extern int inflateInit2_(npz_z_stream *strm, int windowBits, const char *version,
                         int stream_size);
extern int inflate(npz_z_stream *strm, int flush);
extern int inflateEnd(npz_z_stream *strm);
extern unsigned long crc32(unsigned long crc, const unsigned char *buf, unsigned int len);

#define NPZ_Z_OK 0
#define NPZ_Z_STREAM_END 1
#define NPZ_Z_NO_FLUSH 0

#define IO_CHUNK (1u << 30) /* max bytes per zlib call (avail_* are 32-bit) */
#define CONV_CHUNK (1u << 20)

/* ------------------------------------------------------------------------ */
/* helpers                                                                    */
/* ------------------------------------------------------------------------ */
static void seterr(char *err, size_t errlen, const char *fmt, ...)
{
    va_list ap;
    if (!err || errlen == 0) return;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static uint16_t rd16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const unsigned char *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

static void *xmalloc(size_t n) { return malloc(n ? n : 1); }

/* ------------------------------------------------------------------------ */
/* file loading                                                               */
/* ------------------------------------------------------------------------ */
typedef struct {
    const unsigned char *data;
    size_t size;
    void *map;   /* mmap base or NULL */
    void *heap;  /* malloc'ed copy or NULL */
} filebuf_t;

static int file_load(const char *path, filebuf_t *fb, char *err, size_t errlen)
{
    struct stat st;
    int fd;
    memset(fb, 0, sizeof *fb);
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        seterr(err, errlen, "cannot open '%s': %s", path, strerror(errno));
        return -1;
    }
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
        size_t size = (size_t)st.st_size;
        int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
        flags |= MAP_POPULATE;
#endif
        void *p = mmap(NULL, size, PROT_READ, flags, fd, 0);
        if (p != MAP_FAILED) {
            fb->map = p;
            fb->data = (const unsigned char *)p;
            fb->size = size;
            close(fd);
            return 0;
        }
    }
    /* fallback: read everything */
    {
        size_t cap = 1 << 20, len = 0;
        unsigned char *buf = (unsigned char *)malloc(cap);
        if (!buf) { close(fd); seterr(err, errlen, "out of memory"); return -1; }
        for (;;) {
            ssize_t r;
            if (len == cap) {
                unsigned char *nb = (unsigned char *)realloc(buf, cap * 2);
                if (!nb) { free(buf); close(fd); seterr(err, errlen, "out of memory"); return -1; }
                buf = nb; cap *= 2;
            }
            r = read(fd, buf + len, cap - len);
            if (r < 0) {
                if (errno == EINTR) continue;
                seterr(err, errlen, "read error on '%s': %s", path, strerror(errno));
                free(buf); close(fd); return -1;
            }
            if (r == 0) break;
            len += (size_t)r;
        }
        close(fd);
        fb->heap = buf;
        fb->data = buf;
        fb->size = len;
    }
    return 0;
}

static void file_unload(filebuf_t *fb)
{
    if (fb->map) munmap(fb->map, fb->size);
    free(fb->heap);
    memset(fb, 0, sizeof *fb);
}

/* ------------------------------------------------------------------------ */
/* zip directory                                                              */
/* ------------------------------------------------------------------------ */
enum { E_Q, E_B, E_PDATA, E_PIND, E_PPTR, E_PSHAPE, E_ADATA, E_AIND, E_APTR, E_ASHAPE, E_META, E_COUNT };
static const char *const entry_names[E_COUNT] = {
    "q", "b", "P_data", "P_indices", "P_indptr", "P_shape",
    "A_data", "A_indices", "A_indptr", "A_shape", "metadata_json"};

typedef struct {
    int present;
    int method;
    unsigned flags;
    uint64_t csize, usize, lho;
} zentry_t;

typedef struct {
    const unsigned char *d;
    size_t size;
    zentry_t ent[E_COUNT];
} zip_t;

static int zip_scan(zip_t *z, char *err, size_t errlen)
{
    const unsigned char *d = z->d;
    size_t size = z->size, lo, i, pos = (size_t)-1, cd_end_limit;
    uint64_t cdsize, cdoff, cdstart, cdend, p;
    int64_t concat;
    zentry_t exact[E_COUNT], npy[E_COUNT];
    int k;

    memset(exact, 0, sizeof exact);
    memset(npy, 0, sizeof npy);
    if (size < 22) { seterr(err, errlen, "not a zip file (too small)"); return -1; }
    lo = size > 22 + 65535 ? size - 22 - 65535 : 0;
    for (i = size - 22 + 1; i-- > lo;) {
        if (d[i] == 0x50 && rd32(d + i) == 0x06054b50) { pos = i; break; }
    }
    if (pos == (size_t)-1) { seterr(err, errlen, "not a zip file (no end of central directory)"); return -1; }
    cdsize = rd32(d + pos + 12);
    cdoff = rd32(d + pos + 16);
    cd_end_limit = pos;
    if (pos >= 20 && rd32(d + pos - 20) == 0x07064b50) {
        /* ZIP64 end of central directory locator; like Python's zipfile, fall
         * back to the 32-bit record if the ZIP64 record cannot be found */
        size_t recpos = (size_t)-1;
        if (pos >= 20 + 56 && rd32(d + pos - 20 - 56) == 0x06064b50) {
            recpos = pos - 20 - 56;
        } else {
            uint64_t off = rd64(d + pos - 20 + 8);
            if (pos >= 20 + 56 && off <= (uint64_t)(pos - 20 - 56) && rd32(d + off) == 0x06064b50)
                recpos = (size_t)off;
        }
        if (recpos != (size_t)-1) {
            const unsigned char *rec = d + recpos;
            cdsize = rd64(rec + 40);
            cdoff = rd64(rec + 48);
            cd_end_limit = recpos;
        }
    }
    if (cdsize > cd_end_limit || cdoff > cd_end_limit - cdsize) {
        seterr(err, errlen, "bad central directory size/offset");
        return -1;
    }
    concat = (int64_t)(cd_end_limit - cdsize - cdoff);
    cdstart = cdoff + (uint64_t)concat;
    cdend = cdstart + cdsize;
    p = cdstart;
    while (p + 46 <= cdend) {
        const unsigned char *h = d + p, *name, *ex, *exend;
        unsigned nlen, elen, clen;
        uint64_t csize, usize, lho;
        zentry_t e;
        if (rd32(h) != 0x02014b50) { seterr(err, errlen, "bad central directory entry"); return -1; }
        nlen = rd16(h + 28); elen = rd16(h + 30); clen = rd16(h + 32);
        if (p + 46 + nlen + elen + clen > cdend) { seterr(err, errlen, "truncated central directory"); return -1; }
        csize = rd32(h + 20); usize = rd32(h + 24); lho = rd32(h + 42);
        name = h + 46;
        ex = name + nlen;
        exend = ex + elen;
        while (ex + 4 <= exend) {
            unsigned id = rd16(ex), sz = rd16(ex + 2);
            const unsigned char *q = ex + 4, *qe = ex + 4 + sz;
            if (qe > exend) break;
            if (id == 0x0001) {
                if (usize == 0xFFFFFFFFu) { if (q + 8 > qe) goto bad64; usize = rd64(q); q += 8; }
                if (csize == 0xFFFFFFFFu) { if (q + 8 > qe) goto bad64; csize = rd64(q); q += 8; }
                if (lho == 0xFFFFFFFFu) { if (q + 8 > qe) goto bad64; lho = rd64(q); q += 8; }
            }
            ex = qe;
        }
        memset(&e, 0, sizeof e);
        e.present = 1;
        e.flags = rd16(h + 8);
        e.method = rd16(h + 10);
        e.csize = csize;
        e.usize = usize;
        e.lho = lho + (uint64_t)concat;
        for (k = 0; k < E_COUNT; k++) {
            size_t L = strlen(entry_names[k]);
            if (nlen == L && memcmp(name, entry_names[k], L) == 0) exact[k] = e;
            else if (nlen == L + 4 && memcmp(name, entry_names[k], L) == 0 && memcmp(name + L, ".npy", 4) == 0)
                npy[k] = e;
        }
        p += 46 + nlen + elen + clen;
    }
    for (k = 0; k < E_COUNT; k++) z->ent[k] = exact[k].present ? exact[k] : npy[k];
    return 0;
bad64:
    seterr(err, errlen, "corrupt ZIP64 extra field");
    return -1;
}

/* ------------------------------------------------------------------------ */
/* entry reader (STORED or raw DEFLATE)                                       */
/* ------------------------------------------------------------------------ */
typedef struct {
    const char *name;
    const unsigned char *src;
    uint64_t csize, usize;
    uint64_t src_pos, out_pos;
    int method, zinit;
    npz_z_stream zs;
} ereader_t;

static int er_open(ereader_t *r, const zip_t *z, int k, char *err, size_t errlen)
{
    const zentry_t *e = &z->ent[k];
    const unsigned char *lh;
    uint64_t start;
    memset(r, 0, sizeof *r);
    r->name = entry_names[k];
    if (e->lho > z->size || z->size - e->lho < 30 || rd32(z->d + e->lho) != 0x04034b50) {
        seterr(err, errlen, "%s: bad local file header", r->name);
        return -1;
    }
    lh = z->d + e->lho;
    start = e->lho + 30 + (uint64_t)rd16(lh + 26) + (uint64_t)rd16(lh + 28);
    if (start > z->size || z->size - start < e->csize) {
        seterr(err, errlen, "%s: entry data extends past end of file", r->name);
        return -1;
    }
    if (e->flags & 1u) { seterr(err, errlen, "%s: encrypted zip entries are not supported", r->name); return -1; }
    if (e->method != 0 && e->method != 8) {
        seterr(err, errlen, "%s: unsupported zip compression method %d", r->name, e->method);
        return -1;
    }
    if (e->method == 0 && e->csize < e->usize) { seterr(err, errlen, "%s: inconsistent stored sizes", r->name); return -1; }
    r->src = z->d + start;
    r->csize = e->csize;
    r->usize = e->usize;
    r->method = e->method;
    if (r->method == 8) {
        if (inflateInit2_(&r->zs, -15, "1.2.13", (int)sizeof(npz_z_stream)) != NPZ_Z_OK) {
            seterr(err, errlen, "%s: inflateInit failed", r->name);
            return -1;
        }
        r->zinit = 1;
    }
    return 0;
}

static void er_close(ereader_t *r)
{
    if (r->zinit) inflateEnd(&r->zs);
    r->zinit = 0;
}

/* reads exactly n bytes of uncompressed entry data into dst */
static int er_read(ereader_t *r, void *dst, uint64_t n, char *err, size_t errlen)
{
    unsigned char *out = (unsigned char *)dst;
    if (n > r->usize - r->out_pos) { seterr(err, errlen, "%s: entry is truncated", r->name); return -1; }
    if (n == 0) return 0;
    if (r->method == 0) {
        memcpy(dst, r->src + r->out_pos, (size_t)n);
        r->out_pos += n;
        return 0;
    }
    while (n > 0) {
        unsigned int chunk = n > IO_CHUNK ? IO_CHUNK : (unsigned int)n;
        unsigned int produced;
        r->zs.next_out = out;
        r->zs.avail_out = chunk;
        while (r->zs.avail_out > 0) {
            int rc;
            if (r->zs.avail_in == 0 && r->src_pos < r->csize) {
                uint64_t left = r->csize - r->src_pos;
                unsigned int c = left > IO_CHUNK ? IO_CHUNK : (unsigned int)left;
                r->zs.next_in = r->src + r->src_pos;
                r->zs.avail_in = c;
                r->src_pos += c;
            }
            rc = inflate(&r->zs, NPZ_Z_NO_FLUSH);
            if (rc == NPZ_Z_STREAM_END) {
                if (r->zs.avail_out > 0) { seterr(err, errlen, "%s: compressed data ends early", r->name); return -1; }
                break;
            }
            if (rc != NPZ_Z_OK) {
                seterr(err, errlen, "%s: inflate error %d (%s)", r->name, rc,
                       r->zs.msg ? r->zs.msg : (r->src_pos >= r->csize ? "truncated data" : "corrupt data"));
                return -1;
            }
        }
        produced = chunk - r->zs.avail_out;
        out += produced;
        n -= produced;
        r->out_pos += produced;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* NPY header                                                                 */
/* ------------------------------------------------------------------------ */
enum { T_F4, T_F8, T_I1, T_I2, T_I4, T_I8, T_U1, T_U2, T_U4, T_U8, T_STR };

typedef struct {
    char kind;      /* f i u b U S */
    int code;       /* T_* */
    int swap;       /* big-endian data */
    int fortran;
    int ndim;
    uint64_t itemsize;
    uint64_t count;
    uint64_t nontrivial_dims; /* number of dims with extent > 1 */
} npyinfo_t;

static const char *skip_pyws(const char *p, const char *e)
{
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* parses a python quoted string literal (no escapes needed for npy headers) */
static const char *parse_pystr(const char *p, const char *e, char *buf, size_t cap)
{
    char q;
    size_t n = 0;
    if (p >= e || (*p != '\'' && *p != '"')) return NULL;
    q = *p++;
    while (p < e && *p != q) {
        if (*p == '\\') return NULL;
        if (n + 1 >= cap) return NULL;
        buf[n++] = *p++;
    }
    if (p >= e) return NULL;
    buf[n] = 0;
    return p + 1;
}

static int npy_parse_header(const char *h, size_t len, npyinfo_t *info, const char *name, char *err,
                            size_t errlen)
{
    const char *p = h, *e = h + len;
    char key[32], descr[64];
    int got_descr = 0, got_fo = 0, got_shape = 0;
    memset(info, 0, sizeof *info);
    info->count = 1;
    p = skip_pyws(p, e);
    if (p >= e || *p != '{') goto bad;
    p++;
    for (;;) {
        p = skip_pyws(p, e);
        if (p < e && *p == '}') break;
        p = parse_pystr(p, e, key, sizeof key);
        if (!p) goto bad;
        p = skip_pyws(p, e);
        if (p >= e || *p != ':') goto bad;
        p = skip_pyws(p + 1, e);
        if (strcmp(key, "descr") == 0) {
            p = parse_pystr(p, e, descr, sizeof descr);
            if (!p) {
                seterr(err, errlen, "%s: unsupported dtype (structured or malformed descr)", name);
                return -1;
            }
            got_descr = 1;
        } else if (strcmp(key, "fortran_order") == 0) {
            if (e - p >= 4 && memcmp(p, "True", 4) == 0) { info->fortran = 1; p += 4; }
            else if (e - p >= 5 && memcmp(p, "False", 5) == 0) { info->fortran = 0; p += 5; }
            else goto bad;
            got_fo = 1;
        } else if (strcmp(key, "shape") == 0) {
            if (p >= e || *p != '(') goto bad;
            p++;
            info->ndim = 0;
            info->count = 1;
            info->nontrivial_dims = 0;
            for (;;) {
                uint64_t v = 0;
                p = skip_pyws(p, e);
                if (p < e && *p == ')') { p++; break; }
                if (p >= e || *p < '0' || *p > '9') goto bad;
                while (p < e && *p >= '0' && *p <= '9') {
                    v = v * 10 + (uint64_t)(*p - '0');
                    if (v > ((uint64_t)1 << 48)) { seterr(err, errlen, "%s: array too large", name); return -1; }
                    p++;
                }
                if (p < e && *p == 'L') p++;
                if (info->ndim >= 64) goto bad;
                info->ndim++;
                if (v > 1) info->nontrivial_dims++;
                if (v != 0 && info->count > ((uint64_t)1 << 48) / v) {
                    seterr(err, errlen, "%s: array too large", name);
                    return -1;
                }
                info->count *= v;
                p = skip_pyws(p, e);
                if (p < e && *p == ',') { p++; continue; }
                if (p < e && *p == ')') { p++; break; }
                goto bad;
            }
            got_shape = 1;
        } else {
            goto bad;
        }
        p = skip_pyws(p, e);
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == '}') break;
        goto bad;
    }
    if (!got_descr || !got_fo || !got_shape) goto bad;
    {
        const char *d = descr;
        char bo = '=';
        uint64_t sz = 0;
        if (*d == '<' || *d == '>' || *d == '|' || *d == '=') bo = *d++;
        info->kind = *d ? *d++ : 0;
        if (*d < '0' || *d > '9') {
            seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr);
            return -1;
        }
        while (*d >= '0' && *d <= '9') {
            sz = sz * 10 + (uint64_t)(*d - '0');
            if (sz > (1u << 28)) break;
            d++;
        }
        if (*d) { seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr); return -1; }
        info->swap = (bo == '>');
        switch (info->kind) {
        case 'f':
            if (sz == 8) info->code = T_F8;
            else if (sz == 4) info->code = T_F4;
            else { seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr); return -1; }
            break;
        case 'i':
            if (sz == 1) info->code = T_I1;
            else if (sz == 2) info->code = T_I2;
            else if (sz == 4) info->code = T_I4;
            else if (sz == 8) info->code = T_I8;
            else { seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr); return -1; }
            break;
        case 'u': case 'b':
            if (sz == 1) info->code = T_U1;
            else if (sz == 2 && info->kind == 'u') info->code = T_U2;
            else if (sz == 4 && info->kind == 'u') info->code = T_U4;
            else if (sz == 8 && info->kind == 'u') info->code = T_U8;
            else { seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr); return -1; }
            break;
        case 'U':
            info->code = T_STR;
            if (sz == 0 || sz > (1u << 28)) { seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr); return -1; }
            sz *= 4;
            break;
        case 'S':
            info->code = T_STR;
            if (sz == 0 || sz > (1u << 30)) { seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr); return -1; }
            break;
        default:
            seterr(err, errlen, "%s: unsupported dtype '%s'", name, descr);
            return -1;
        }
        info->itemsize = sz;
        if (info->count > ((uint64_t)1 << 48) / sz) { seterr(err, errlen, "%s: array too large", name); return -1; }
    }
    return 0;
bad:
    seterr(err, errlen, "%s: malformed NPY header", name);
    return -1;
}

/* reads the NPY preamble+header; leaves the reader positioned at the data */
static int npy_open(ereader_t *r, npyinfo_t *info, char *err, size_t errlen)
{
    unsigned char pre[12];
    uint64_t hl;
    char *h;
    int rc;
    if (r->usize < 10) { seterr(err, errlen, "%s: not an NPY array", r->name); return -1; }
    if (er_read(r, pre, 10, err, errlen)) return -1;
    if (memcmp(pre, "\x93NUMPY", 6) != 0) { seterr(err, errlen, "%s: not an NPY array (bad magic)", r->name); return -1; }
    if (pre[6] == 1) {
        hl = rd16(pre + 8);
    } else if (pre[6] == 2 || pre[6] == 3) {
        if (er_read(r, pre + 10, 2, err, errlen)) return -1;
        hl = rd32(pre + 8);
    } else {
        seterr(err, errlen, "%s: unsupported NPY version %d.%d", r->name, pre[6], pre[7]);
        return -1;
    }
    if (hl > (1u << 20)) { seterr(err, errlen, "%s: NPY header too large", r->name); return -1; }
    h = (char *)xmalloc((size_t)hl + 1);
    if (!h) { seterr(err, errlen, "out of memory"); return -1; }
    if (er_read(r, h, hl, err, errlen)) { free(h); return -1; }
    h[hl] = 0;
    rc = npy_parse_header(h, (size_t)hl, info, r->name, err, errlen);
    free(h);
    if (rc) return -1;
    if (info->count * info->itemsize > r->usize - r->out_pos) {
        seterr(err, errlen, "%s: array data is truncated", r->name);
        return -1;
    }
    return 0;
}

/* element loaders (unaligned-safe) */
static inline uint64_t load_bits(const unsigned char *p, unsigned sz, int swap)
{
    uint64_t v = 0;
    switch (sz) {
    case 1: v = p[0]; break;
    case 2: { uint16_t t; memcpy(&t, p, 2); v = swap ? __builtin_bswap16(t) : t; } break;
    case 4: { uint32_t t; memcpy(&t, p, 4); v = swap ? __builtin_bswap32(t) : t; } break;
    default: { uint64_t t; memcpy(&t, p, 8); v = swap ? __builtin_bswap64(t) : t; } break;
    }
    return v;
}

static const unsigned code_size[] = {4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 0};

static void conv_double(const unsigned char *src, size_t cnt, int code, int swap, double *dst)
{
    size_t i;
    unsigned sz = code_size[code];
    for (i = 0; i < cnt; i++, src += sz) {
        uint64_t v = load_bits(src, sz, swap);
        switch (code) {
        case T_F8: { double d; memcpy(&d, &v, 8); dst[i] = d; } break;
        case T_F4: { uint32_t t = (uint32_t)v; float f; memcpy(&f, &t, 4); dst[i] = f; } break;
        case T_I1: dst[i] = (double)(int8_t)v; break;
        case T_I2: dst[i] = (double)(int16_t)v; break;
        case T_I4: dst[i] = (double)(int32_t)v; break;
        case T_I8: dst[i] = (double)(int64_t)v; break;
        default: dst[i] = (double)v; break; /* unsigned / bool */
        }
    }
}

/* converts to int, requiring 0 <= value <= INT_MAX; returns index of first bad value or cnt */
static size_t conv_int(const unsigned char *src, size_t cnt, int code, int swap, int *dst)
{
    size_t i;
    unsigned sz = code_size[code];
    for (i = 0; i < cnt; i++, src += sz) {
        uint64_t v = load_bits(src, sz, swap);
        int64_t s;
        switch (code) {
        case T_I1: s = (int8_t)v; break;
        case T_I2: s = (int16_t)v; break;
        case T_I4: s = (int32_t)v; break;
        case T_I8: s = (int64_t)v; break;
        default:
            if (v > (uint64_t)INT_MAX) return i;
            s = (int64_t)v;
            break;
        }
        if (s < 0 || s > INT_MAX) return i;
        dst[i] = (int)s;
    }
    return cnt;
}

/* reads a numeric array as double (want_int=0) or int (want_int=1) */
static int read_numeric(const zip_t *z, int k, int want_int, void **out, int64_t *count, char *err, size_t errlen)
{
    ereader_t r;
    npyinfo_t info;
    void *arr = NULL;
    unsigned char *buf = NULL;
    size_t tsz = want_int ? sizeof(int) : sizeof(double);
    *out = NULL;
    *count = 0;
    if (er_open(&r, z, k, err, errlen)) return -1;
    if (npy_open(&r, &info, err, errlen)) goto fail;
    if (info.code == T_STR) { seterr(err, errlen, "%s: expected a numeric array", r.name); goto fail; }
    if (want_int && (info.code == T_F4 || info.code == T_F8)) {
        seterr(err, errlen, "%s: expected an integer array", r.name);
        goto fail;
    }
    if (info.fortran && info.nontrivial_dims > 1) {
        seterr(err, errlen, "%s: multi-dimensional Fortran-order arrays are not supported", r.name);
        goto fail;
    }
    if (info.count > (uint64_t)INT_MAX - 1) { seterr(err, errlen, "%s: array too large", r.name); goto fail; }
    arr = xmalloc((size_t)info.count * tsz);
    if (!arr) { seterr(err, errlen, "%s: out of memory", r.name); goto fail; }
    if (!info.swap && ((!want_int && info.code == T_F8) || (want_int && info.code == T_I4 && sizeof(int) == 4))) {
        if (er_read(&r, arr, info.count * tsz, err, errlen)) goto fail;
        if (want_int) {
            const int *a = (const int *)arr;
            size_t i;
            for (i = 0; i < info.count; i++)
                if (a[i] < 0) { seterr(err, errlen, "%s: negative value %d at position %zu", r.name, a[i], i); goto fail; }
        }
    } else {
        uint64_t done = 0;
        size_t per = CONV_CHUNK / info.itemsize;
        buf = (unsigned char *)xmalloc(CONV_CHUNK);
        if (!buf) { seterr(err, errlen, "%s: out of memory", r.name); goto fail; }
        while (done < info.count) {
            size_t c = (size_t)(info.count - done < per ? info.count - done : per);
            if (er_read(&r, buf, (uint64_t)c * info.itemsize, err, errlen)) goto fail;
            if (want_int) {
                size_t bad = conv_int(buf, c, info.code, info.swap, (int *)arr + done);
                if (bad != c) {
                    seterr(err, errlen, "%s: value out of range at position %llu", r.name,
                           (unsigned long long)(done + bad));
                    goto fail;
                }
            } else {
                conv_double(buf, c, info.code, info.swap, (double *)arr + done);
            }
            done += c;
        }
        free(buf);
        buf = NULL;
    }
    er_close(&r);
    *out = arr;
    *count = (int64_t)info.count;
    return 0;
fail:
    er_close(&r);
    free(buf);
    free(arr);
    return -1;
}

/* reads metadata_json and returns a NUL-terminated UTF-8 string */
static int read_meta_text(const zip_t *z, char **text, size_t *textlen, char *err, size_t errlen)
{
    ereader_t r;
    npyinfo_t info;
    unsigned char *buf = NULL;
    uint64_t nbytes;
    size_t w = 0;
    *text = NULL;
    if (er_open(&r, z, E_META, err, errlen)) return -1;
    if (npy_open(&r, &info, err, errlen)) goto fail;
    if (info.code != T_STR || info.count != 1) {
        seterr(err, errlen, "metadata_json: expected a scalar unicode array");
        goto fail;
    }
    nbytes = info.itemsize;
    buf = (unsigned char *)xmalloc((size_t)nbytes + 1);
    if (!buf) { seterr(err, errlen, "metadata_json: out of memory"); goto fail; }
    if (er_read(&r, buf, nbytes, err, errlen)) goto fail;
    er_close(&r);
    if (info.kind == 'S') {
        while (w < nbytes && buf[w]) w++;
    } else {
        /* UTF-32 -> UTF-8 in place (output never overtakes input) */
        size_t ncp = (size_t)(nbytes / 4), i;
        for (i = 0; i < ncp; i++) {
            uint32_t cp = (uint32_t)load_bits(buf + 4 * i, 4, info.swap);
            if (cp == 0) break;
            if (cp < 0x80) {
                buf[w++] = (unsigned char)cp;
            } else if (cp < 0x800) {
                buf[w++] = (unsigned char)(0xC0 | (cp >> 6));
                buf[w++] = (unsigned char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                buf[w++] = (unsigned char)(0xE0 | (cp >> 12));
                buf[w++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
                buf[w++] = (unsigned char)(0x80 | (cp & 0x3F));
            } else if (cp <= 0x10FFFF) {
                buf[w++] = (unsigned char)(0xF0 | (cp >> 18));
                buf[w++] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
                buf[w++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
                buf[w++] = (unsigned char)(0x80 | (cp & 0x3F));
            } else {
                seterr(err, errlen, "metadata_json: invalid code point 0x%x", (unsigned)cp);
                free(buf);
                return -1;
            }
        }
    }
    buf[w] = 0;
    *text = (char *)buf;
    *textlen = w;
    return 0;
fail:
    er_close(&r);
    free(buf);
    return -1;
}

/* ------------------------------------------------------------------------ */
/* JSON (recursive descent, specialized for the metadata schema)             */
/* ------------------------------------------------------------------------ */
typedef struct {
    const char *base, *p, *end;
    char *err;
    size_t errlen;
} jp_t;

static int jerr(jp_t *j, const char *msg)
{
    seterr(j->err, j->errlen, "metadata_json: %s (at offset %ld)", msg, (long)(j->p - j->base));
    return -1;
}

static void jws(jp_t *j)
{
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

static int jpeek(jp_t *j, char c) { return j->p < j->end && *j->p == c; }

static int jlit(jp_t *j, const char *lit)
{
    size_t L = strlen(lit);
    if ((size_t)(j->end - j->p) >= L && memcmp(j->p, lit, L) == 0) { j->p += L; return 1; }
    return 0;
}

static int hex4(jp_t *j, uint32_t *out)
{
    uint32_t v = 0;
    int i;
    if (j->end - j->p < 4) return jerr(j, "bad \\u escape");
    for (i = 0; i < 4; i++) {
        char c = j->p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return jerr(j, "bad \\u escape");
    }
    j->p += 4;
    *out = v;
    return 0;
}

static void jput(char *buf, size_t cap, size_t *len, int *trunc, const char *s, size_t n)
{
    if (!buf) return;
    if (*len + n + 1 > cap) { *trunc = 1; return; }
    memcpy(buf + *len, s, n);
    *len += n;
}

/* parses a string; decoded UTF-8 into buf (may be NULL); *trunc set if it did not fit */
static int jstring(jp_t *j, char *buf, size_t cap, int *trunc)
{
    size_t len = 0;
    int tr = 0;
    if (!jpeek(j, '"')) return jerr(j, "expected string");
    j->p++;
    for (;;) {
        unsigned char c;
        uint32_t cp;
        char tmp[4];
        size_t tn;
        if (j->p >= j->end) return jerr(j, "unterminated string");
        c = (unsigned char)*j->p++;
        if (c == '"') break;
        if (c < 0x20) return jerr(j, "control character in string");
        if (c != '\\') { char cc = (char)c; jput(buf, cap, &len, &tr, &cc, 1); continue; }
        if (j->p >= j->end) return jerr(j, "unterminated string");
        c = (unsigned char)*j->p++;
        switch (c) {
        case '"': cp = '"'; break;
        case '\\': cp = '\\'; break;
        case '/': cp = '/'; break;
        case 'b': cp = 8; break;
        case 'f': cp = 12; break;
        case 'n': cp = 10; break;
        case 'r': cp = 13; break;
        case 't': cp = 9; break;
        case 'u':
            if (hex4(j, &cp)) return -1;
            if (cp >= 0xD800 && cp < 0xDC00 && j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                const char *save = j->p;
                uint32_t lo;
                j->p += 2;
                if (hex4(j, &lo)) return -1;
                if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                else j->p = save;
            }
            break;
        default:
            return jerr(j, "invalid escape in string");
        }
        if (cp < 0x80) { tmp[0] = (char)cp; tn = 1; }
        else if (cp < 0x800) { tmp[0] = (char)(0xC0 | (cp >> 6)); tmp[1] = (char)(0x80 | (cp & 0x3F)); tn = 2; }
        else if (cp < 0x10000) {
            tmp[0] = (char)(0xE0 | (cp >> 12)); tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            tmp[2] = (char)(0x80 | (cp & 0x3F)); tn = 3;
        } else {
            tmp[0] = (char)(0xF0 | (cp >> 18)); tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
            tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); tmp[3] = (char)(0x80 | (cp & 0x3F)); tn = 4;
        }
        jput(buf, cap, &len, &tr, tmp, tn);
    }
    if (buf && cap) buf[len] = 0;
    if (trunc) *trunc = tr;
    return 0;
}

/* parses a JSON number. is_int: literal has no fraction/exponent;
 * ival_ok: integer literal fits in int64 (value in *ival) */
static int jnumber(jp_t *j, double *val, int *is_int, int64_t *ival, int *ival_ok)
{
    const char *s = j->p, *p = s, *e = j->end;
    int isint = 1, neg = 0, ok = 1;
    uint64_t acc = 0;
    char tmp[128], *b;
    size_t L;
    double v;
    char *endp;
    if (p < e && *p == '-') { neg = 1; p++; }
    if (p < e && *p == '0') {
        p++;
    } else if (p < e && *p >= '1' && *p <= '9') {
        while (p < e && *p >= '0' && *p <= '9') {
            unsigned dgt = (unsigned)(*p - '0');
            if (acc > (UINT64_MAX - dgt) / 10) ok = 0;
            else acc = acc * 10 + dgt;
            p++;
        }
    } else {
        return jerr(j, "invalid number");
    }
    if (p < e && *p == '.') {
        isint = 0;
        p++;
        if (!(p < e && *p >= '0' && *p <= '9')) return jerr(j, "invalid number");
        while (p < e && *p >= '0' && *p <= '9') p++;
    }
    if (p < e && (*p == 'e' || *p == 'E')) {
        isint = 0;
        p++;
        if (p < e && (*p == '+' || *p == '-')) p++;
        if (!(p < e && *p >= '0' && *p <= '9')) return jerr(j, "invalid number");
        while (p < e && *p >= '0' && *p <= '9') p++;
    }
    L = (size_t)(p - s);
    b = L < sizeof tmp ? tmp : (char *)malloc(L + 1);
    if (!b) return jerr(j, "out of memory");
    memcpy(b, s, L);
    b[L] = 0;
    v = strtod(b, &endp);
    if (endp != b + L) {
        /* locale with a non-'.' decimal point: retry with ',' */
        char *dot = strchr(b, '.');
        if (dot) { *dot = ','; v = strtod(b, &endp); }
    }
    if (b != tmp) free(b);
    if (val) *val = v;
    if (is_int) *is_int = isint;
    if (ival_ok) {
        *ival_ok = isint && ok && acc <= (uint64_t)INT64_MAX;
        if (*ival_ok && ival) *ival = neg ? -(int64_t)acc : (int64_t)acc;
    }
    j->p = p;
    return 0;
}

static int jskip(jp_t *j, int depth)
{
    jws(j);
    if (depth > 1000) return jerr(j, "nesting too deep");
    if (j->p >= j->end) return jerr(j, "unexpected end of input");
    if (*j->p == '{') {
        j->p++;
        jws(j);
        if (jpeek(j, '}')) { j->p++; return 0; }
        for (;;) {
            jws(j);
            if (jstring(j, NULL, 0, NULL)) return -1;
            jws(j);
            if (!jpeek(j, ':')) return jerr(j, "expected ':'");
            j->p++;
            if (jskip(j, depth + 1)) return -1;
            jws(j);
            if (jpeek(j, ',')) { j->p++; continue; }
            if (jpeek(j, '}')) { j->p++; return 0; }
            return jerr(j, "expected ',' or '}'");
        }
    }
    if (*j->p == '[') {
        j->p++;
        jws(j);
        if (jpeek(j, ']')) { j->p++; return 0; }
        for (;;) {
            if (jskip(j, depth + 1)) return -1;
            jws(j);
            if (jpeek(j, ',')) { j->p++; continue; }
            if (jpeek(j, ']')) { j->p++; return 0; }
            return jerr(j, "expected ',' or ']'");
        }
    }
    if (*j->p == '"') return jstring(j, NULL, 0, NULL);
    if (jlit(j, "true") || jlit(j, "false") || jlit(j, "null") || jlit(j, "NaN") || jlit(j, "Infinity") ||
        jlit(j, "-Infinity"))
        return 0;
    return jnumber(j, NULL, NULL, NULL, NULL);
}

/* parses a numeric-ish value (number, true/false, NaN/Infinity) */
static int jnumeric_value(jp_t *j, double *v, int *is_int, int64_t *ival, int *ival_ok, const char *what)
{
    jws(j);
    if (is_int) *is_int = 0;
    if (ival_ok) *ival_ok = 0;
    if (jlit(j, "true")) { *v = 1.0; return 0; }
    if (jlit(j, "false")) { *v = 0.0; return 0; }
    if (jlit(j, "NaN")) { *v = NAN; return 0; }
    if (jlit(j, "Infinity")) { *v = INFINITY; return 0; }
    if (jlit(j, "-Infinity")) { *v = -INFINITY; return 0; }
    if (j->p < j->end && (*j->p == '-' || (*j->p >= '0' && *j->p <= '9')))
        return jnumber(j, v, is_int, ival, ival_ok);
    {
        char msg[96];
        snprintf(msg, sizeof msg, "%s must be a number", what);
        return jerr(j, msg);
    }
}

typedef struct {
    int ncones, cap;
    int *type, *dim;
    double offset;
    int have_cones, have_schema;
} meta_t;

static int parse_cones(jp_t *j, meta_t *mt)
{
    free(mt->type);
    free(mt->dim);
    mt->type = mt->dim = NULL;
    mt->ncones = mt->cap = 0;
    jws(j);
    if (!jpeek(j, '[')) return jerr(j, "cones must be an array");
    j->p++;
    jws(j);
    if (jpeek(j, ']')) { j->p++; return 0; }
    for (;;) {
        int ctype = -1, have_type = 0, have_dim = 0;
        int64_t dim = 0;
        jws(j);
        if (!jpeek(j, '{')) return jerr(j, "cone entries must be objects");
        j->p++;
        jws(j);
        if (jpeek(j, '}')) {
            j->p++;
        } else {
            for (;;) {
                char key[16];
                int tr = 0;
                jws(j);
                if (jstring(j, key, sizeof key, &tr)) return -1;
                jws(j);
                if (!jpeek(j, ':')) return jerr(j, "expected ':'");
                j->p++;
                jws(j);
                if (!tr && strcmp(key, "type") == 0) {
                    char tv[16];
                    int ttr = 0;
                    if (!jpeek(j, '"')) return jerr(j, "cone type must be a string");
                    if (jstring(j, tv, sizeof tv, &ttr)) return -1;
                    have_type = 1;
                    ctype = -1;
                    if (!ttr) {
                        if (strcmp(tv, "zero") == 0) ctype = CONE_ZERO;
                        else if (strcmp(tv, "nonnegative") == 0) ctype = CONE_NONNEG;
                        else if (strcmp(tv, "soc") == 0) ctype = CONE_SOC;
                        else if (strcmp(tv, "exponential") == 0) ctype = CONE_EXP;
                        else if (strcmp(tv, "psd") == 0) ctype = CONE_PSD;
                    }
                    if (ctype < 0) return jerr(j, "unsupported cone type");
                } else if (!tr && strcmp(key, "dim") == 0) {
                    int is_int = 0, ok = 0;
                    if (!(j->p < j->end && (*j->p == '-' || (*j->p >= '0' && *j->p <= '9'))))
                        return jerr(j, "cone dim must be a positive integer");
                    if (jnumber(j, NULL, &is_int, &dim, &ok)) return -1;
                    if (!is_int || !ok || dim <= 0 || dim > INT_MAX)
                        return jerr(j, "cone dim must be a positive integer");
                    have_dim = 1;
                } else {
                    if (jskip(j, 2)) return -1;
                }
                jws(j);
                if (jpeek(j, ',')) { j->p++; continue; }
                if (jpeek(j, '}')) { j->p++; break; }
                return jerr(j, "expected ',' or '}'");
            }
        }
        if (!have_type || !have_dim) return jerr(j, "cone entry needs 'type' and 'dim'");
        if (ctype == CONE_EXP && dim != 3) return jerr(j, "exponential cone dimension must be 3");
        if (ctype == CONE_SOC && dim < 2) return jerr(j, "SOC dimension must be at least 2");
        if (mt->ncones == mt->cap) {
            int ncap = mt->cap ? mt->cap * 2 : 16;
            int *nt, *nd;
            if (mt->cap > INT_MAX / 4) return jerr(j, "too many cones");
            nt = (int *)realloc(mt->type, (size_t)ncap * sizeof(int));
            if (!nt) return jerr(j, "out of memory");
            mt->type = nt;
            nd = (int *)realloc(mt->dim, (size_t)ncap * sizeof(int));
            if (!nd) return jerr(j, "out of memory");
            mt->dim = nd;
            mt->cap = ncap;
        }
        mt->type[mt->ncones] = ctype;
        mt->dim[mt->ncones] = (int)dim;
        mt->ncones++;
        jws(j);
        if (jpeek(j, ',')) { j->p++; continue; }
        if (jpeek(j, ']')) { j->p++; return 0; }
        return jerr(j, "expected ',' or ']'");
    }
}

static int parse_meta(const char *text, size_t len, meta_t *mt, char *err, size_t errlen)
{
    jp_t j;
    j.base = j.p = text;
    j.end = text + len;
    j.err = err;
    j.errlen = errlen;
    memset(mt, 0, sizeof *mt);
    jws(&j);
    if (!jpeek(&j, '{')) return jerr(&j, "top level must be an object");
    j.p++;
    jws(&j);
    if (jpeek(&j, '}')) {
        j.p++;
    } else {
        for (;;) {
            char key[32];
            int tr = 0;
            jws(&j);
            if (jstring(&j, key, sizeof key, &tr)) return -1;
            jws(&j);
            if (!jpeek(&j, ':')) return jerr(&j, "expected ':'");
            j.p++;
            jws(&j);
            if (!tr && strcmp(key, "cones") == 0) {
                if (parse_cones(&j, mt)) return -1;
                mt->have_cones = 1;
            } else if (!tr && strcmp(key, "objective_offset") == 0) {
                double v = 0;
                if (jnumeric_value(&j, &v, NULL, NULL, NULL, "objective_offset")) return -1;
                if (!isfinite(v)) return jerr(&j, "objective_offset is not finite");
                mt->offset = v;
            } else if (!tr && strcmp(key, "schema_version") == 0) {
                double v = 0;
                if (jnumeric_value(&j, &v, NULL, NULL, NULL, "schema_version")) return -1;
                if (v != 1.0) return jerr(&j, "unsupported schema_version");
                mt->have_schema = 1;
            } else {
                if (jskip(&j, 1)) return -1;
            }
            jws(&j);
            if (jpeek(&j, ',')) { j.p++; continue; }
            if (jpeek(&j, '}')) { j.p++; break; }
            return jerr(&j, "expected ',' or '}'");
        }
    }
    jws(&j);
    if (j.p != j.end) return jerr(&j, "extra data after JSON object");
    if (!mt->have_schema) return jerr(&j, "missing schema_version");
    if (!mt->have_cones) return jerr(&j, "missing cones");
    return 0;
}

/* ------------------------------------------------------------------------ */
/* CSC validation + canonicalization                                          */
/* ------------------------------------------------------------------------ */
static int csc_canonicalize(const char *nm, int nrows, int ncols, int *Ap, int **pAi, double **pAx,
                            char *err, size_t errlen)
{
    int *Ai = *pAi;
    double *Ax = *pAx;
    int j, nnz = Ap[ncols];
    int need_sort = 0, nzero = 0;
    for (j = 0; j < ncols; j++) {
        int k, prev = -1, end = Ap[j + 1];
        for (k = Ap[j]; k < end; k++) {
            int r = Ai[k];
            double v = Ax[k];
            if ((unsigned)r >= (unsigned)nrows) {
                seterr(err, errlen, "%s: row index %d out of range [0,%d) in column %d", nm, r, nrows, j);
                return -1;
            }
            if (r <= prev) need_sort = 1;
            prev = r;
            if (!isfinite(v)) { seterr(err, errlen, "%s: nonfinite coefficient", nm); return -1; }
            if (v == 0.0) nzero = 1;
        }
    }
    if (need_sort) {
        /* two counting-sort transposes: stable, O(nnz + nrows + ncols) */
        int *Rp = (int *)calloc((size_t)nrows + 1, sizeof(int));
        int *next = (int *)xmalloc(((size_t)(nrows > ncols ? nrows : ncols) + 1) * sizeof(int));
        int *Rj = (int *)xmalloc((size_t)nnz * sizeof(int));
        double *Rx = (double *)xmalloc((size_t)nnz * sizeof(double));
        int k, r;
        if (!Rp || !next || !Rj || !Rx) {
            free(Rp); free(next); free(Rj); free(Rx);
            seterr(err, errlen, "%s: out of memory", nm);
            return -1;
        }
        for (k = 0; k < nnz; k++) Rp[Ai[k] + 1]++;
        for (r = 0; r < nrows; r++) Rp[r + 1] += Rp[r];
        memcpy(next, Rp, (size_t)nrows * sizeof(int));
        for (j = 0; j < ncols; j++) {
            for (k = Ap[j]; k < Ap[j + 1]; k++) {
                int dst = next[Ai[k]]++;
                Rj[dst] = j;
                Rx[dst] = Ax[k];
            }
        }
        memcpy(next, Ap, (size_t)ncols * sizeof(int));
        for (r = 0; r < nrows; r++) {
            for (k = Rp[r]; k < Rp[r + 1]; k++) {
                int dst = next[Rj[k]]++;
                Ai[dst] = r;
                Ax[dst] = Rx[k];
            }
        }
        free(Rp); free(next); free(Rj); free(Rx);
    }
    if (need_sort || nzero) {
        int w = 0, start = 0;
        for (j = 0; j < ncols; j++) {
            int end = Ap[j + 1], k = start;
            Ap[j] = w;
            while (k < end) {
                int r = Ai[k];
                double v = Ax[k++];
                while (k < end && Ai[k] == r) v += Ax[k++];
                if (v != 0.0) {
                    if (!isfinite(v)) { seterr(err, errlen, "%s: nonfinite coefficient after summing duplicates", nm); return -1; }
                    Ai[w] = r;
                    Ax[w] = v;
                    w++;
                }
            }
            start = end;
        }
        Ap[ncols] = w;
        nnz = w;
    }
    /* shrink storage (also drops entries past indptr[-1]) */
    {
        int *ni = (int *)realloc(Ai, (size_t)(nnz ? nnz : 1) * sizeof(int));
        double *nx = (double *)realloc(Ax, (size_t)(nnz ? nnz : 1) * sizeof(double));
        if (ni) *pAi = ni;
        if (nx) *pAx = nx;
    }
    return 0;
}

static int load_csc(const zip_t *z, int kdata, int kind, int kptr, const char *nm, int nrows, int ncols,
                    int **pp, int **pi, double **px, char *err, size_t errlen)
{
    void *vx = NULL, *vi = NULL, *vp = NULL;
    int64_t nx, ni, np;
    int *ptr, j;
    if (read_numeric(z, kdata, 0, &vx, &nx, err, errlen)) goto fail;
    if (read_numeric(z, kind, 1, &vi, &ni, err, errlen)) goto fail;
    if (read_numeric(z, kptr, 1, &vp, &np, err, errlen)) goto fail;
    ptr = (int *)vp;
    if (np != (int64_t)ncols + 1) {
        seterr(err, errlen, "%s_indptr has length %lld, expected %d", nm, (long long)np, ncols + 1);
        goto fail;
    }
    if (ni != nx) {
        seterr(err, errlen, "%s_indices and %s_data have different lengths (%lld vs %lld)", nm, nm,
               (long long)ni, (long long)nx);
        goto fail;
    }
    if (ptr[0] != 0) { seterr(err, errlen, "%s_indptr must start with 0", nm); goto fail; }
    for (j = 0; j < ncols; j++)
        if (ptr[j + 1] < ptr[j]) { seterr(err, errlen, "%s_indptr is not nondecreasing", nm); goto fail; }
    if ((int64_t)ptr[ncols] > ni) {
        seterr(err, errlen, "%s_indptr[-1]=%d exceeds number of stored entries %lld", nm, ptr[ncols], (long long)ni);
        goto fail;
    }
    {
        int *ii = (int *)vi;
        double *xx = (double *)vx;
        if (csc_canonicalize(nm, nrows, ncols, ptr, &ii, &xx, err, errlen)) { vi = ii; vx = xx; goto fail; }
        vi = ii;
        vx = xx;
    }
    *pp = ptr;
    *pi = (int *)vi;
    *px = (double *)vx;
    return 0;
fail:
    free(vx); free(vi); free(vp);
    return -1;
}

static int read_shape(const zip_t *z, int k, int *r, int *c, char *err, size_t errlen)
{
    void *v = NULL;
    int64_t cnt;
    if (read_numeric(z, k, 1, &v, &cnt, err, errlen)) return -1;
    if (cnt != 2) { free(v); seterr(err, errlen, "%s must have 2 entries", entry_names[k]); return -1; }
    *r = ((int *)v)[0];
    *c = ((int *)v)[1];
    free(v);
    return 0;
}

void problem_free(problem_t *prob)
{
    if (!prob) return;
    free(prob->Pp); free(prob->Pi); free(prob->Px);
    free(prob->Ap); free(prob->Ai); free(prob->Ax);
    free(prob->q); free(prob->b);
    free(prob->cone_type); free(prob->cone_dim);
    memset(prob, 0, sizeof *prob);
}

int npz_read_problem(const char *path, problem_t *prob, char *err, size_t errlen)
{
    filebuf_t fb;
    zip_t z;
    meta_t mt;
    char *text = NULL;
    size_t textlen = 0;
    void *vq = NULL, *vb = NULL;
    int64_t nq = 0, nb = 0;
    int pr, pc, ar, ac, k, rc = -1;
    if (!prob) return -1;
    memset(prob, 0, sizeof *prob);
    memset(&mt, 0, sizeof mt);
    if (err && errlen) err[0] = 0;
    if (!path) { seterr(err, errlen, "no input path"); return -1; }
    if (file_load(path, &fb, err, errlen)) return -1;
    memset(&z, 0, sizeof z);
    z.d = fb.data;
    z.size = fb.size;
    if (zip_scan(&z, err, errlen)) goto done;
    for (k = 0; k < E_COUNT; k++)
        if (!z.ent[k].present) { seterr(err, errlen, "missing array '%s'", entry_names[k]); goto done; }

    /* metadata */
    if (read_meta_text(&z, &text, &textlen, err, errlen)) goto done;
    if (parse_meta(text, textlen, &mt, err, errlen)) goto done;
    free(text);
    text = NULL;

    /* vectors and shapes */
    if (read_numeric(&z, E_Q, 0, &vq, &nq, err, errlen)) goto done;
    prob->q = (double *)vq;
    vq = NULL;
    if (read_numeric(&z, E_B, 0, &vb, &nb, err, errlen)) goto done;
    prob->b = (double *)vb;
    vb = NULL;
    prob->n = (int)nq;
    prob->m = (int)nb;
    for (k = 0; k < prob->n; k++)
        if (!isfinite(prob->q[k])) { seterr(err, errlen, "q: nonfinite coefficient"); goto done; }
    for (k = 0; k < prob->m; k++)
        if (!isfinite(prob->b[k])) { seterr(err, errlen, "b: nonfinite coefficient"); goto done; }
    if (read_shape(&z, E_PSHAPE, &pr, &pc, err, errlen)) goto done;
    if (read_shape(&z, E_ASHAPE, &ar, &ac, err, errlen)) goto done;
    if (pr != prob->n || pc != prob->n || ar != prob->m || ac != prob->n) {
        seterr(err, errlen, "dimension mismatch: P_shape=(%d,%d), A_shape=(%d,%d), len(q)=%d, len(b)=%d", pr, pc,
               ar, ac, prob->n, prob->m);
        goto done;
    }

    /* cones */
    {
        int64_t rows = 0;
        for (k = 0; k < mt.ncones; k++) {
            int64_t d = mt.dim[k];
            rows += mt.type[k] == CONE_PSD ? d * (d + 1) / 2 : d;
            if (rows > (int64_t)prob->m) break;
        }
        if (rows != (int64_t)prob->m) {
            seterr(err, errlen, "cone blocks do not partition the %d constraint rows", prob->m);
            goto done;
        }
    }

    /* matrices */
    if (load_csc(&z, E_PDATA, E_PIND, E_PPTR, "P", prob->n, prob->n, &prob->Pp, &prob->Pi, &prob->Px, err, errlen))
        goto done;
    if (load_csc(&z, E_ADATA, E_AIND, E_APTR, "A", prob->m, prob->n, &prob->Ap, &prob->Ai, &prob->Ax, err, errlen))
        goto done;

    prob->ncones = mt.ncones;
    prob->cone_type = mt.type ? mt.type : (int *)xmalloc(sizeof(int));
    prob->cone_dim = mt.dim ? mt.dim : (int *)xmalloc(sizeof(int));
    mt.type = mt.dim = NULL;
    prob->objective_offset = mt.offset;
    if (!prob->cone_type || !prob->cone_dim) { seterr(err, errlen, "out of memory"); goto done; }
    rc = 0;
done:
    free(text);
    free(vq);
    free(vb);
    free(mt.type);
    free(mt.dim);
    file_unload(&fb);
    if (rc) problem_free(prob);
    return rc;
}

/* ------------------------------------------------------------------------ */
/* writer                                                                     */
/* ------------------------------------------------------------------------ */
typedef struct {
    const char *name;
    unsigned char hdr[256];
    size_t hdrlen;
    const unsigned char *data;
    uint64_t datalen;
    uint32_t crc;
    uint64_t offset;
} wentry_t;

static void npy_make_header(wentry_t *e, const char *descr, const char *shape)
{
    char dict[192];
    int L = snprintf(dict, sizeof dict, "{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shape);
    size_t total = ((size_t)10 + (size_t)L + 1 + 63) / 64 * 64;
    size_t hl = total - 10;
    memcpy(e->hdr, "\x93NUMPY\x01\x00", 8);
    e->hdr[8] = (unsigned char)(hl & 0xFF);
    e->hdr[9] = (unsigned char)(hl >> 8);
    memcpy(e->hdr + 10, dict, (size_t)L);
    memset(e->hdr + 10 + L, ' ', hl - (size_t)L - 1);
    e->hdr[total - 1] = '\n';
    e->hdrlen = total;
}

static uint32_t crc_update(uint32_t crc, const unsigned char *p, uint64_t n)
{
    unsigned long c = crc;
    while (n > 0) {
        unsigned int chunk = n > IO_CHUNK ? IO_CHUNK : (unsigned int)n;
        c = crc32(c, p, chunk);
        p += chunk;
        n -= chunk;
    }
    return (uint32_t)c;
}

static void put16(unsigned char *p, unsigned v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }
static void put32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

static void fmt_double(char *buf, size_t cap, double v)
{
    char *c;
    snprintf(buf, cap, "%.17g", v);
    for (c = buf; *c; c++)
        if (*c == ',') *c = '.';
    if (!strpbrk(buf, ".eE")) strncat(buf, ".0", cap - strlen(buf) - 1);
}

int npz_write_result(const char *path, const char *status, const double *x, int n, const double *s,
                     const double *z, int m, int has_obj, double obj_p, double obj_d)
{
    wentry_t ent[4];
    int ne = 0, i;
    char *json = NULL;
    unsigned char *u32 = NULL;
    size_t jlen = 0, cap;
    const char *st = status ? status : "unresolved";
    FILE *f = NULL;
    uint64_t off = 0, cdsize = 0;
    unsigned char cd[4 * (46 + 32)], *cp = cd;
    char descr[32], shape[32];
    int rc = -1;

    if (!path || n < 0 || m < 0) return -1;

    /* metadata JSON (ASCII only) */
    cap = strlen(st) * 6 + 256;
    json = (char *)malloc(cap);
    if (!json) return -1;
    jlen = (size_t)snprintf(json, cap, "{\"schema_version\": 1, \"status\": \"");
    for (; *st; st++) {
        unsigned char ch = (unsigned char)*st;
        if (ch == '"' || ch == '\\') { json[jlen++] = '\\'; json[jlen++] = (char)ch; }
        else if (ch < 0x20 || ch >= 0x7F) jlen += (size_t)snprintf(json + jlen, cap - jlen, "\\u%04x", ch);
        else json[jlen++] = (char)ch;
    }
    json[jlen++] = '"';
    json[jlen] = 0;
    if (has_obj && isfinite(obj_p) && isfinite(obj_d) && status && strcmp(status, "solved") == 0) {
        char bp[40], bd[40];
        fmt_double(bp, sizeof bp, obj_p);
        fmt_double(bd, sizeof bd, obj_d);
        jlen += (size_t)snprintf(json + jlen, cap - jlen, ", \"objective_primal\": %s, \"objective_dual\": %s", bp, bd);
    }
    json[jlen++] = '}';
    json[jlen] = 0;
    u32 = (unsigned char *)malloc(jlen * 4);
    if (!u32) goto done;
    for (i = 0; i < (int)jlen; i++) {
        u32[4 * i] = (unsigned char)json[i];
        u32[4 * i + 1] = u32[4 * i + 2] = u32[4 * i + 3] = 0;
    }

    memset(ent, 0, sizeof ent);
    ent[ne].name = "metadata_json.npy";
    snprintf(descr, sizeof descr, "<U%zu", jlen);
    npy_make_header(&ent[ne], descr, "()");
    ent[ne].data = u32;
    ent[ne].datalen = (uint64_t)jlen * 4;
    ne++;
    if (x) {
        ent[ne].name = "x.npy";
        snprintf(shape, sizeof shape, "(%d,)", n);
        npy_make_header(&ent[ne], "<f8", shape);
        ent[ne].data = (const unsigned char *)x;
        ent[ne].datalen = (uint64_t)n * 8;
        ne++;
    }
    if (s) {
        ent[ne].name = "s.npy";
        snprintf(shape, sizeof shape, "(%d,)", m);
        npy_make_header(&ent[ne], "<f8", shape);
        ent[ne].data = (const unsigned char *)s;
        ent[ne].datalen = (uint64_t)m * 8;
        ne++;
    }
    if (z) {
        ent[ne].name = "z.npy";
        snprintf(shape, sizeof shape, "(%d,)", m);
        npy_make_header(&ent[ne], "<f8", shape);
        ent[ne].data = (const unsigned char *)z;
        ent[ne].datalen = (uint64_t)m * 8;
        ne++;
    }
    for (i = 0; i < ne; i++) {
        uint32_t c = crc_update(0, ent[i].hdr, ent[i].hdrlen);
        ent[i].crc = crc_update(c, ent[i].data, ent[i].datalen);
        if (ent[i].hdrlen + ent[i].datalen >= 0xFFFFFFFFull) goto done; /* would need ZIP64 */
    }

    f = fopen(path, "wb");
    if (!f) goto done;
    for (i = 0; i < ne; i++) {
        unsigned char lh[30];
        size_t nl = strlen(ent[i].name);
        uint32_t usz = (uint32_t)(ent[i].hdrlen + ent[i].datalen);
        if (off >= 0xFFFFFFFFull) goto done;
        ent[i].offset = off;
        put32(lh, 0x04034b50);
        put16(lh + 4, 20);
        put16(lh + 6, 0);
        put16(lh + 8, 0);
        put16(lh + 10, 0);
        put16(lh + 12, 0x0021);
        put32(lh + 14, ent[i].crc);
        put32(lh + 18, usz);
        put32(lh + 22, usz);
        put16(lh + 26, (unsigned)nl);
        put16(lh + 28, 0);
        if (fwrite(lh, 1, 30, f) != 30 || fwrite(ent[i].name, 1, nl, f) != nl ||
            fwrite(ent[i].hdr, 1, ent[i].hdrlen, f) != ent[i].hdrlen)
            goto done;
        if (ent[i].datalen && fwrite(ent[i].data, 1, (size_t)ent[i].datalen, f) != (size_t)ent[i].datalen) goto done;
        off += 30 + nl + usz;

        put32(cp, 0x02014b50);
        put16(cp + 4, 20);
        put16(cp + 6, 20);
        put16(cp + 8, 0);
        put16(cp + 10, 0);
        put16(cp + 12, 0);
        put16(cp + 14, 0x0021);
        put32(cp + 16, ent[i].crc);
        put32(cp + 20, usz);
        put32(cp + 24, usz);
        put16(cp + 28, (unsigned)nl);
        put16(cp + 30, 0);
        put16(cp + 32, 0);
        put16(cp + 34, 0);
        put16(cp + 36, 0);
        put32(cp + 38, 0);
        put32(cp + 42, (uint32_t)ent[i].offset);
        memcpy(cp + 46, ent[i].name, nl);
        cp += 46 + nl;
    }
    if (off >= 0xFFFFFFFFull) goto done;
    cdsize = (uint64_t)(cp - cd);
    {
        unsigned char eocd[22];
        put32(eocd, 0x06054b50);
        put16(eocd + 4, 0);
        put16(eocd + 6, 0);
        put16(eocd + 8, (unsigned)ne);
        put16(eocd + 10, (unsigned)ne);
        put32(eocd + 12, (uint32_t)cdsize);
        put32(eocd + 16, (uint32_t)off);
        put16(eocd + 20, 0);
        if (fwrite(cd, 1, (size_t)cdsize, f) != (size_t)cdsize || fwrite(eocd, 1, 22, f) != 22) goto done;
    }
    rc = 0;
done:
    if (f) {
        if (fclose(f) != 0) rc = -1;
    }
    free(json);
    free(u32);
    return rc;
}
