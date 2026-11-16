#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static const double G = 1.0;
static const double EPS = 0.01;
static const double EPS2 = EPS * EPS;
static const int BUCKET = 4;

typedef struct { double x, y, z, m; } Part;
typedef struct {
    int c[8], first, n, ov;
    double cx, cy, cz, m, sz, ox, oy, oz;
} Node;

static Node *nodes = NULL;
static size_t ncap = 0, nnodes = 0;
static Part *parts = NULL;
static size_t pcap = 0, nparts = 0;
static int *part_ov = NULL;
static double theta = 0.5;

static void *xrealloc(void *p, size_t n, size_t sz) {
    void *q = realloc(p, n * sz);
    if (!q) { fputs("out of memory\n", stderr); exit(1); }
    return q;
}

static int new_node(double ox, double oy, double oz, double sz) {
    if (nnodes == ncap) {
        ncap = ncap ? ncap * 2 : 1024;
        nodes = xrealloc(nodes, ncap, sizeof(Node));
    }
    int id = (int)nnodes++;
    Node *d = &nodes[id];
    for (int k = 0; k < 8; k++) d->c[k] = -1;
    d->cx = d->cy = d->cz = d->m = 0.0;
    d->sz = sz; d->ox = ox; d->oy = oy; d->oz = oz;
    d->first = -1; d->n = 0; d->ov = -1;
    return id;
}

static void grow_parts(void) {
    if (nparts == pcap) {
        pcap = pcap ? pcap * 2 : 1024;
        parts = xrealloc(parts, pcap, sizeof(Part));
        part_ov = xrealloc(part_ov, pcap, sizeof(int));
    }
}

static inline int octant(const Part *p, const Node *d) {
    int o = 0;
    if (p->x >= d->ox + d->sz) o |= 1;
    if (p->y >= d->oy + d->sz) o |= 2;
    if (p->z >= d->oz + d->sz) o |= 4;
    return o;
}

static void insert_child(int nid, int pi);
static void insert(int nid, int pi);

static void insert_child(int nid, int pi) {
    if (nid < 0 || (size_t)nid >= nnodes) { fprintf(stderr, "insert_child BAD nid=%d nnodes=%zu pi=%d\n", nid, nnodes, pi); abort(); }
    if (pi < 0 || (size_t)pi >= nparts) { fprintf(stderr, "insert_child BAD pi=%d nparts=%zu nid=%d\n", pi, nparts, nid); abort(); }
    int o = octant(&parts[pi], &nodes[nid]);
    int ch = nodes[nid].c[o];
    if (ch < 0) {
        double h = nodes[nid].sz * 0.5;
        int nid2 = new_node(nodes[nid].ox + (o & 1 ? h : 0.0),
                            nodes[nid].oy + (o & 2 ? h : 0.0),
                            nodes[nid].oz + (o & 4 ? h : 0.0), h);
        Node *cd = &nodes[nid2];
        const Part *p = &parts[pi];
        cd->cx = p->x; cd->cy = p->y; cd->cz = p->z; cd->m = p->m;
        cd->first = pi; cd->n = 1; cd->ov = -1;
        nodes[nid].c[o] = nid2;
    } else {
        insert(ch, pi);
    }
}

static void insert(int nid, int pi) {
    if (nid < 0 || (size_t)nid >= nnodes) { fprintf(stderr, "insert BAD nid=%d nnodes=%zu pi=%d\n", nid, nnodes, pi); abort(); }
    if (pi < 0 || (size_t)pi >= nparts) { fprintf(stderr, "insert BAD pi=%d nparts=%zu nid=%d\n", pi, nparts, nid); abort(); }
    if (nodes[nid].n >= 0) {
        if (nodes[nid].n < BUCKET) {
            const Part *p = &parts[pi];
            double nm = nodes[nid].m + p->m;
            nodes[nid].cx = (nodes[nid].cx * nodes[nid].m + p->x * p->m) / nm;
            nodes[nid].cy = (nodes[nid].cy * nodes[nid].m + p->y * p->m) / nm;
            nodes[nid].cz = (nodes[nid].cz * nodes[nid].m + p->z * p->m) / nm;
            nodes[nid].m = nm;
            part_ov[pi] = nodes[nid].first; nodes[nid].first = pi;
            nodes[nid].n++;
            return;
        }
        int of = nodes[nid].first, on = nodes[nid].n;
        double ox = nodes[nid].ox, oy = nodes[nid].oy, oz = nodes[nid].oz, sz = nodes[nid].sz;
        double h = sz * 0.5;
        nodes[nid].n = -1; nodes[nid].first = -1; nodes[nid].ov = -1;
        for (int k = 0; k < 8; k++)
            nodes[nid].c[k] = new_node(ox + (k & 1 ? h : 0.0),
                                       oy + (k & 2 ? h : 0.0),
                                       oz + (k & 4 ? h : 0.0), h);
        for (int j = 0; j < on; j++) {
            int q = of;
            of = part_ov[of];
            insert_child(nid, q);
        }
        insert_child(nid, pi);
        return;
    }
    const Part *p = &parts[pi];
    double nm = nodes[nid].m + p->m;
    nodes[nid].cx = (nodes[nid].cx * nodes[nid].m + p->x * p->m) / nm;
    nodes[nid].cy = (nodes[nid].cy * nodes[nid].m + p->y * p->m) / nm;
    nodes[nid].cz = (nodes[nid].cz * nodes[nid].m + p->z * p->m) / nm;
    nodes[nid].m = nm;
    insert_child(nid, pi);
}

static inline void contrib(double qx, double qy, double qz, double qm,
                           const Part *p, double *ax, double *ay, double *az) {
    double dx = qx - p->x, dy = qy - p->y, dz = qz - p->z;
    double d2 = dx*dx + dy*dy + dz*dz + EPS2;
    double f = G * qm / (d2 * sqrt(d2));
    *ax += f * dx; *ay += f * dy; *az += f * dz;
}

static void eval(int nid, int pi, double *ax, double *ay, double *az) {
    if (nid < 0 || (size_t)nid >= nnodes) { fprintf(stderr, "eval BAD nid=%d nnodes=%zu pi=%d\n", nid, nnodes, pi); abort(); }
    if (pi < 0 || (size_t)pi >= nparts) { fprintf(stderr, "eval BAD pi=%d nparts=%zu nid=%d\n", pi, nparts, nid); abort(); }
    const Node *d = &nodes[nid];
    const Part *p = &parts[pi];
    if (d->n >= 0) {
        for (int q = d->first; q >= 0; q = part_ov[q])
            if (q != pi) contrib(parts[q].x, parts[q].y, parts[q].z, parts[q].m, p, ax, ay, az);
        return;
    }
    for (int k = 0; k < 8; k++) {
        int ch = d->c[k];
        if (ch < 0) continue;
        const Node *c = &nodes[ch];
        double dx = c->cx - p->x, dy = c->cy - p->y, dz = c->cz - p->z;
        double d2 = dx*dx + dy*dy + dz*dz + EPS2;
        double dd = sqrt(d2);
        if (2.0 * c->sz / dd > theta) {
            eval(ch, pi, ax, ay, az);
        } else {
            double f = G * c->m / (d2 * dd);
            *ax += f * dx; *ay += f * dy; *az += f * dz;
        }
    }
}

int main(int argc, char **argv) {
    const char *te = getenv("GALAXY_THETA");
    if (te && *te) theta = atof(te);
    if (theta <= 0.0) theta = 0.5;

    if (argc != 3) { fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]); return 2; }

    FILE *in = fopen(argv[1], "rb");
    if (!in) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fseek(in, 0, SEEK_END);
    long fsz = ftell(in);
    rewind(in);
    if (fsz < 32 || fsz % 32 != 0) { fprintf(stderr, "bad input size\n"); fclose(in); return 1; }
    size_t N = (size_t)fsz / 32;

    double *buf = malloc(N * 4 * sizeof(double));
    if (!buf) { fputs("out of memory\n", stderr); return 1; }
    if (fread(buf, 32, N, in) != N) { fprintf(stderr, "short read\n"); free(buf); fclose(in); return 1; }
    fclose(in);

    for (size_t i = 0; i < N; i++) {
        grow_parts();
        Part p;
        p.x = buf[4*i+1]; p.y = buf[4*i+2]; p.z = buf[4*i+3]; p.m = buf[4*i];
        parts[nparts] = p; part_ov[nparts] = -1;
        nparts++;
    }
    free(buf);

    double minx = 1e308, miny = 1e308, minz = 1e308;
    double maxx = -1e308, maxy = -1e308, maxz = -1e308;
    for (size_t i = 0; i < N; i++) {
        if (parts[i].x < minx) minx = parts[i].x;
        if (parts[i].y < miny) miny = parts[i].y;
        if (parts[i].z < minz) minz = parts[i].z;
        if (parts[i].x > maxx) maxx = parts[i].x;
        if (parts[i].y > maxy) maxy = parts[i].y;
        if (parts[i].z > maxz) maxz = parts[i].z;
    }
    double range = maxx - minx;
    if (maxy - miny > range) range = maxy - miny;
    if (maxz - minz > range) range = maxz - minz;
    if (!isfinite(range) || range == 0.0) range = 1.0;
    double h = range * 0.5 * 1.0001;
    double cx = (minx + maxx) * 0.5;
    double cy = (miny + maxy) * 0.5;
    double cz = (minz + maxz) * 0.5;

    int root = new_node(cx - h, cy - h, cz - h, h);
    for (size_t i = 0; i < N; i++) insert(root, (int)i);

    double *out = malloc(N * 3 * sizeof(double));
    if (!out) { fputs("out of memory\n", stderr); return 1; }
    for (size_t i = 0; i < N; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        eval(root, (int)i, &ax, &ay, &az);
        out[3*i] = ax; out[3*i+1] = ay; out[3*i+2] = az;
    }

    FILE *fo = fopen(argv[2], "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    if (fwrite(out, 24, N, fo) != N) { fprintf(stderr, "short write\n"); fclose(fo); return 1; }
    if (fclose(fo) != 0) { fprintf(stderr, "write error\n"); return 1; }
    return 0;
}
