#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "npz.h"

/* ---------------- zlib minimal declarations ---------------- */
typedef void *(*zaf)(void *, unsigned, unsigned);
typedef void (*zff)(void *, void *);
typedef struct z_stream_s {
    unsigned char *next_in; unsigned int avail_in; unsigned long total_in;
    unsigned char *next_out; unsigned int avail_out; unsigned long total_out;
    char *msg; void *state;
    zaf zalloc; zff zfree; void *opaque;
    int data_type; unsigned long adler; unsigned long reserved;
} z_stream;
extern int inflateInit2_(z_stream *, int, const char *, int);
extern int inflate(z_stream *, int);
extern int inflateEnd(z_stream *);
#define ZLIB_VER "1.2.13"
#define Z_OK_ 0
#define Z_STREAM_END 1
#define Z_FINISH 4
#define Z_NO_FLUSH 0

static unsigned char *rd_u8(const unsigned char *p) { return (unsigned char *)p; }
static uint16_t g16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t g32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void p16(unsigned char *p, uint16_t v) { p[0] = v & 255; p[1] = (v >> 8) & 255; }
static void p32(unsigned char *p, uint32_t v) {
    p[0] = v & 255; p[1] = (v >> 8) & 255; p[2] = (v >> 16) & 255; p[3] = (v >> 24) & 255;
}

static int inflate_raw(const unsigned char *src, long srclen, unsigned char *dst, long dstcap, long *out) {
    z_stream s;
    memset(&s, 0, sizeof s);
    s.next_in = (unsigned char *)src; s.avail_in = (unsigned int)srclen;
    s.next_out = dst; s.avail_out = (unsigned int)dstcap;
    int r = inflateInit2_(&s, -15, ZLIB_VER, (int)sizeof(z_stream));
    if (r != Z_OK_) return -1;
    r = inflate(&s, Z_NO_FLUSH);
    *out = (long)s.total_out;
    inflateEnd(&s);
    return (r == Z_STREAM_END || r == Z_OK_) ? 0 : -2;
}

/* ---------------- npy header parse ---------------- */
static int npy_parse(const unsigned char *buf, long len, NpyArr *a, long *dataoff) {
    if (len < 10 || memcmp(buf, "\x93NUMPY", 6) != 0) return -1;
    long hlen;
    const char *hdr;
    if (buf[6] == 1) { hlen = g16(buf + 8); hdr = (const char *)buf + 10; }
    else { hlen = (long)g32(buf + 8); hdr = (const char *)buf + 12; }
    if (10 + hlen > len && 12 + hlen > len) return -1;
    *dataoff = (buf[6] == 1 ? 10 : 12) + hlen;
    char *h = (char *)malloc(hlen + 1);
    memcpy(h, hdr, hlen); h[hlen] = 0;
    char *p;
    /* descr */
    p = strstr(h, "'descr':"); if (!p) p = strstr(h, "\"descr\":");
    if (!p) { free(h); return -2; }
    p = strchr(p, ':'); if (!p) { free(h); return -2; }
    p = strchr(p, '\''); if (!p) { free(h); return -2; }
    p++; char *e = strchr(p, '\''); if (!e) { free(h); return -2; }
    long dl = e - p; if (dl > 23) dl = 23;
    memcpy(a->descr, p, dl); a->descr[dl] = 0;
    /* fortran */
    a->fortran = (strstr(h, "'fortran_order': True") != NULL || strstr(h, "\"fortran_order\": True") != NULL);
    /* shape */
    p = strstr(h, "'shape':"); if (!p) p = strstr(h, "\"shape\":");
    a->ndim = 0; a->shape[0] = 1;
    if (p) {
        p = strchr(p, '(');
        if (p) {
            p++;
            int nd = 0;
            while (*p && *p != ')') {
                while (*p == ' ' || *p == ',') p++;
                if (*p == ')') break;
                long v = strtol(p, &p, 10);
                if (nd < 8) a->shape[nd++] = v;
            }
            a->ndim = nd;
            if (nd == 0) { a->shape[0] = 1; }
        }
    }
    free(h);
    /* itemsize & count */
    long cnt = 1;
    for (int i = 0; i < (a->ndim ? a->ndim : 0); i++) cnt *= a->shape[i];
    a->count = cnt;
    const char *d = a->descr;
    int isz = 0;
    if (d[0] == '<' || d[0] == '>' || d[0] == '|' || d[0] == '=') d++;
    if (d[0] == 'f') isz = atoi(d + 1);
    else if (d[0] == 'i' || d[0] == 'u') isz = atoi(d + 1);
    else if (d[0] == 'U') isz = 4 * atoi(d + 1);
    else if (d[0] == 'b') isz = 1;
    else isz = 8;
    a->itemsize = isz;
    return 0;
}

Npz *npz_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(fsz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, fsz, f) != (size_t)fsz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    /* find EOCD */
    long eocd = -1;
    for (long i = fsz - 22; i >= 0 && i >= fsz - 22 - 65536; i--) {
        if (g32(buf + i) == 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) { free(buf); return NULL; }
    int nent = g16(buf + eocd + 10);
    long cd = g32(buf + eocd + 16);
    Npz *z = (Npz *)calloc(1, sizeof(Npz));
    z->arr = (NpyArr *)calloc(nent > 0 ? nent : 1, sizeof(NpyArr));
    long p = cd;
    for (int k = 0; k < nent; k++) {
        if (g32(buf + p) != 0x02014b50) break;
        int method = g16(buf + p + 10);
        uint32_t csz = g32(buf + p + 20), usz = g32(buf + p + 24);
        int nl = g16(buf + p + 28), el = g16(buf + p + 30), cl = g16(buf + p + 32);
        uint32_t lho = g32(buf + p + 42);
        char name[512]; int nlen = nl < 511 ? nl : 511;
        memcpy(name, buf + p + 46, nlen); name[nlen] = 0;
        p += 46 + nl + el + cl;
        /* local header */
        if (g32(buf + lho) != 0x04034b50) continue;
        int lnl = g16(buf + lho + 26), lel = g16(buf + lho + 28);
        long doff = lho + 30 + lnl + lel;
        unsigned char *raw = NULL;
        if (method == 0) { raw = buf + doff; }
        else if (method == 8) {
            raw = (unsigned char *)malloc((size_t)usz + 1);
            long got = 0;
            if (inflate_raw(buf + doff, csz, raw, usz, &got) != 0) { free(raw); continue; }
        } else continue;
        NpyArr a; memset(&a, 0, sizeof a);
        long dataoff = 0;
        if (npy_parse(raw, usz, &a, &dataoff) != 0) { if (method == 8) free(raw); continue; }
        long dlen = (long)a.count * a.itemsize;
        a.data = malloc(dlen > 0 ? dlen : 1);
        if (method == 8) {
            memcpy(a.data, raw + dataoff, dlen);
            free(raw);
        } else {
            memcpy(a.data, raw + dataoff, dlen);
        }
        /* strip .npy */
        char *nm = strdup(name);
        int L = strlen(nm);
        if (L > 4 && strcmp(nm + L - 4, ".npy") == 0) nm[L - 4] = 0;
        a.name = nm;
        z->arr[z->n++] = a;
    }
    free(buf);
    return z;
}

NpyArr *npz_get(Npz *z, const char *name) {
    if (!z) return NULL;
    for (int i = 0; i < z->n; i++) if (strcmp(z->arr[i].name, name) == 0) return &z->arr[i];
    return NULL;
}
void npz_free(Npz *z) {
    if (!z) return;
    for (int i = 0; i < z->n; i++) { free(z->arr[i].data); free(z->arr[i].name); }
    free(z->arr); free(z);
}

/* ---------------- writer ---------------- */
static uint32_t crc32_tab[256];
static int crc_init = 0;
static uint32_t crc32buf(const unsigned char *b, long n) {
    if (!crc_init) {
        for (uint32_t i = 0; i < 256; i++) { uint32_t c = i; for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; crc32_tab[i] = c; }
        crc_init = 1;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (long i = 0; i < n; i++) c = crc32_tab[(c ^ b[i]) & 255] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void npzw_init(NpzW *w) { w->e = NULL; w->n = 0; w->cap = 0; }
static WEnt *npzw_new(NpzW *w) {
    if (w->n == w->cap) { w->cap = w->cap ? w->cap * 2 : 8; w->e = (WEnt *)realloc(w->e, w->cap * sizeof(WEnt)); }
    WEnt *e = &w->e[w->n++];
    memset(e, 0, sizeof *e);
    return e;
}
static char *mk_npy(const char *descr, int ndim, const long *shape, const void *data, long nbytes, long *outlen) {
    char hdr[256];
    char shp[64]; int sl = 0;
    shp[sl++] = '(';
    for (int i = 0; i < ndim; i++) { sl += snprintf(shp + sl, sizeof shp - sl, "%ld,", shape[i]); }
    if (ndim == 1) { /* keep trailing comma */ }
    shp[sl++] = ')'; shp[sl] = 0;
    int hl = snprintf(hdr, sizeof hdr, "{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shp);
    /* pad: 10 + hl + pad must be multiple of 64, and end with \n */
    int total = 10 + hl + 1;
    int pad = (64 - (total % 64)) % 64;
    long hlen = hl + pad + 1;
    long n = hlen + 10 + nbytes;
    unsigned char *b = (unsigned char *)malloc(n);
    memcpy(b, "\x93NUMPY\x01\x00", 8);
    p16(b + 8, (uint16_t)hlen);
    memcpy(b + 10, hdr, hl);
    for (int i = 0; i < pad; i++) b[10 + hl + i] = ' ';
    b[10 + hl + pad] = '\n';
    if (nbytes) memcpy(b + 10 + hlen, data, nbytes);
    *outlen = n;
    return (char *)b;
}
void npzw_add_f8(NpzW *w, const char *name, const double *data, long n) {
    WEnt *e = npzw_new(w);
    e->name = strdup(name); e->descr = strdup("<f8"); e->ndim = 1; e->shape[0] = n;
    long L; char *b = mk_npy("<f8", n == 0 ? 1 : 1, e->shape, data, n * 8, &L);
    e->data = b; e->nbytes = L;
}
void npzw_add_empty_f8(NpzW *w, const char *name) { npzw_add_f8(w, name, NULL, 0); }
void npzw_add_ustr(NpzW *w, const char *name, const char *utf8) {
    /* convert utf8 -> ucs4 little endian; assume ASCII-ish but handle general UTF-8 */
    long len = 0; const unsigned char *s = (const unsigned char *)utf8;
    while (*s) { unsigned char c = *s; int nb = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4; s += nb; len++; }
    if (len == 0) len = 1;
    unsigned char *buf = (unsigned char *)calloc(len, 4);
    s = (const unsigned char *)utf8; long i = 0;
    while (*s) {
        unsigned char c = *s; uint32_t cp;
        if (c < 0x80) { cp = c; s += 1; }
        else if (c < 0xE0) { cp = (c & 0x1F) << 6; cp |= (s[1] & 0x3F); s += 2; }
        else if (c < 0xF0) { cp = (c & 0xF) << 12; cp |= (s[1] & 0x3F) << 6; cp |= (s[2] & 0x3F); s += 3; }
        else { cp = (c & 0x7) << 18; cp |= (s[1] & 0x3F) << 12; cp |= (s[2] & 0x3F) << 6; cp |= (s[3] & 0x3F); s += 4; }
        buf[4 * i + 0] = cp & 255; buf[4 * i + 1] = (cp >> 8) & 255; buf[4 * i + 2] = (cp >> 16) & 255; buf[4 * i + 3] = (cp >> 24) & 255;
        i++;
    }
    WEnt *e = npzw_new(w);
    e->name = strdup(name);
    char dd[32]; snprintf(dd, sizeof dd, "<U%ld", len);
    e->descr = strdup(dd);
    e->ndim = 0; e->shape[0] = 0;
    long L; char *b = mk_npy(dd, 0, e->shape, buf, len * 4, &L);
    e->data = b; e->nbytes = L;
    free(buf);
}
int npzw_write(NpzW *w, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    long *offs = (long *)malloc(w->n * sizeof(long));
    for (int i = 0; i < w->n; i++) {
        WEnt *e = &w->e[i];
        char nm[512]; snprintf(nm, sizeof nm, "%s.npy", e->name);
        int nl = strlen(nm);
        unsigned char lh[30];
        offs[i] = ftell(f);
        p32(lh, 0x04034b50); p16(lh + 4, 20); p16(lh + 6, 0); p16(lh + 8, 0);
        p16(lh + 10, 0); p16(lh + 12, 0);
        p32(lh + 14, crc32buf((unsigned char *)e->data, e->nbytes));
        p32(lh + 18, (uint32_t)e->nbytes); p32(lh + 22, (uint32_t)e->nbytes);
        p16(lh + 26, nl); p16(lh + 28, 0);
        fwrite(lh, 1, 30, f); fwrite(nm, 1, nl, f);
        fwrite(e->data, 1, e->nbytes, f);
    }
    long cd = ftell(f);
    for (int i = 0; i < w->n; i++) {
        WEnt *e = &w->e[i];
        char nm[512]; snprintf(nm, sizeof nm, "%s.npy", e->name);
        int nl = strlen(nm);
        unsigned char ch[46];
        p32(ch, 0x02014b50); p16(ch + 4, 20); p16(ch + 6, 20); p16(ch + 8, 0); p16(ch + 10, 0);
        p16(ch + 12, 0); p16(ch + 14, 0);
        p32(ch + 16, crc32buf((unsigned char *)e->data, e->nbytes));
        p32(ch + 20, (uint32_t)e->nbytes); p32(ch + 24, (uint32_t)e->nbytes);
        p16(ch + 28, nl); p16(ch + 30, 0); p16(ch + 32, 0); p16(ch + 34, 0);
        p16(ch + 36, 0); p32(ch + 38, 0); p32(ch + 42, (uint32_t)offs[i]);
        fwrite(ch, 1, 46, f); fwrite(nm, 1, nl, f);
    }
    long cde = ftell(f);
    unsigned char eo[22];
    p32(eo, 0x06054b50); p16(eo + 4, 0); p16(eo + 6, 0);
    p16(eo + 8, w->n); p16(eo + 10, w->n);
    p32(eo + 12, (uint32_t)(cde - cd)); p32(eo + 16, (uint32_t)cd); p16(eo + 20, 0);
    fwrite(eo, 1, 22, f);
    free(offs);
    fclose(f);
    return 0;
}
void npzw_free(NpzW *w) {
    for (int i = 0; i < w->n; i++) { free(w->e[i].name); free(w->e[i].descr); free(w->e[i].data); }
    free(w->e); w->e = NULL; w->n = 0;
}
