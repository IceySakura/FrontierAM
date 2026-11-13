Harmonic::Harmonic(int degree, Solver &s) : p(degree) {
    compact.assign(s.count[p], -1);
    for (int i = 0; i < s.count[p]; ++i) {
        if (s.terms[i].z <= 1) {
            compact[i] = int(full.size());
            full.push_back(i);
        }
    }
    nc = int(full.size());
    stride = (nc+7)&~7;
    for (int ix = p-1; ix >= 0; --ix) {
        for (int iy = p-1-ix; iy >= 0; --iy) {
            for (int iz = p-1-ix-iy; iz >= 0; --iz) {
                gradient_x.push_back(s.index[ix+1][iy][iz]);
                gradient_y.push_back(s.index[ix][iy+1][iz]);
                gradient_z.push_back(s.index[ix][iy][iz+1]);
                gradient_factor.push_back(1/s.terms[s.index[ix][iy][iz]].factorial);
            }
        }
    }
    reduction.resize(s.count[p]);
    for (int i = 0; i < s.count[p]; ++i) {
        const auto &t = s.terms[i];
        int k = t.z/2;
        double binomial = 1;
        for (int j = 0; j <= k; ++j) {
            int target = s.index[t.x+2*j][t.y+2*(k-j)][t.z%2];
            reduction[i].push_back({compact[target], (k%2 ? -1 : 1)*binomial});
            binomial *= double(k-j)/(j+1);
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        upshift[axis].resize(nc);
        downshift[axis].resize(nc);
    }
    for (int a = 0; a < nc; ++a) {
        const auto &t = s.terms[full[a]];
        for (int axis = 0; axis < 2; ++axis) {
            int lim = axis == 0 ? t.x : t.y;
            for (int k = 1; k <= lim; ++k) {
                int b = compact[s.index[t.x-(axis==0?k:0)][t.y-(axis==1?k:0)][t.z]];
                upshift[axis][a].push_back({b, k, 1});
                downshift[axis][b].push_back({a, k, 1});
            }
        }
        for (int k = 1; k <= p-t.degree; ++k) {
            int f = s.index[t.x][t.y][t.z+k];
            for (auto r : reduction[f]) {
                upshift[2][r.i].push_back({a, k, r.weight});
                downshift[2][a].push_back({r.i, k, r.weight});
            }
        }
    }
    gamma.resize(nc);
    for (int a = 1; a < nc; ++a) {
        const auto &ta = s.terms[full[a]];
        int limit = (p-ta.degree+1)*(p-ta.degree+1);
        gamma[a].resize(limit);
        for (int b = 0; b < limit; ++b) {
            const auto &tb = s.terms[full[b]];
            gamma[a][b] = s.index[ta.x+tb.x][ta.y+tb.y][ta.z+tb.z];
        }
    }
    size_t bytes = size_t(s.nodes.size())*stride*sizeof(double);
    if (posix_memalign(reinterpret_cast<void **>(&moments), 64, bytes) ||
        posix_memalign(reinterpret_cast<void **>(&locals), 64, bytes)) std::exit(2);
    std::memset(moments, 0, bytes);
    std::memset(locals, 0, bytes);
}

void Harmonic::shift(double *a, double dx, double dy, double dz, bool upward) {
    double delta[3] = {dx, dy, dz};
    for (int axis = 0; axis < 3; ++axis) {
        double powers[PMAX+1];
        powers[0] = 1;
        for (int k = 1; k <= p; ++k) powers[k] = powers[k-1]*delta[axis]/k;
        if (upward) {
            for (int i = nc-1; i >= 1; --i) {
                double sum = a[i];
                for (auto sh : upshift[axis][i])
                    sum += a[sh.i]*powers[sh.degree]*sh.weight;
                a[i] = sum;
            }
        } else {
            for (int i = 1; i < nc; ++i) {
                double sum = a[i];
                for (auto sh : downshift[axis][i])
                    sum += a[sh.i]*powers[sh.degree]*sh.weight;
                a[i] = sum;
            }
        }
    }
}

void Harmonic::upward(Solver &s) {
    for (int ni = int(s.nodes.size())-1; ni >= 0; --ni) {
        const auto &nd = s.nodes[ni];
        double *out = moments+size_t(ni)*stride;
        if (nd.left < 0) {
            alignas(64) double raw[CMAX] = {};
            alignas(64) __m256d raw4[CMAX];
            for (int k = 0; k < s.count[p]; ++k) raw4[k] = _mm256_setzero_pd();
            int j = nd.begin;
            for (; j+3 < nd.end; j += 4) {
                __m256d xp[PMAX+1], yp[PMAX+1], zp[PMAX+1];
                xp[0] = yp[0] = zp[0] = _mm256_set1_pd(1);
                __m256d ir = _mm256_set1_pd(1/nd.r);
                __m256d dx = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(s.x.data()+j), _mm256_set1_pd(nd.x)), ir);
                __m256d dy = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(s.y.data()+j), _mm256_set1_pd(nd.y)), ir);
                __m256d dz = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(s.z.data()+j), _mm256_set1_pd(nd.z)), ir);
                for (int d = 1; d <= p; ++d) {
                    __m256d id = _mm256_set1_pd(1.0/d);
                    xp[d] = _mm256_mul_pd(_mm256_mul_pd(xp[d-1], dx), id);
                    yp[d] = _mm256_mul_pd(_mm256_mul_pd(yp[d-1], dy), id);
                    zp[d] = _mm256_mul_pd(_mm256_mul_pd(zp[d-1], dz), id);
                }
                __m256d m = _mm256_loadu_pd(s.mass.data()+j);
                for (int k = 0; k < s.count[p]; ++k) {
                    const auto &t = s.terms[k];
                    __m256d v = _mm256_mul_pd(_mm256_mul_pd(m, xp[t.x]), yp[t.y]);
                    raw4[k] = _mm256_fmadd_pd(v, zp[t.z], raw4[k]);
                }
            }
            for (; j < nd.end; ++j) {
                double xp[PMAX+1], yp[PMAX+1], zp[PMAX+1];
                xp[0] = yp[0] = zp[0] = 1;
                double dx = (s.x[j]-nd.x)/nd.r, dy = (s.y[j]-nd.y)/nd.r,
                       dz = (s.z[j]-nd.z)/nd.r;
                for (int d = 1; d <= p; ++d) {
                    xp[d] = xp[d-1]*dx/d; yp[d] = yp[d-1]*dy/d; zp[d] = zp[d-1]*dz/d;
                }
                for (int k = 0; k < s.count[p]; ++k) {
                    const auto &t = s.terms[k];
                    raw[k] += s.mass[j]*xp[t.x]*yp[t.y]*zp[t.z];
                }
            }
            for (int k = 0; k < s.count[p]; ++k) raw[k] += Solver::horizontal(raw4[k]);
            for (int k = 0; k < s.count[p]; ++k)
                for (auto r : reduction[k]) out[r.i] += raw[k]*r.weight;
        } else {
            for (int child : {nd.left, nd.right}) {
                const auto &ch = s.nodes[child];
                const double *in = moments+size_t(child)*stride;
                alignas(64) double tmp[CMAX], scale[PMAX+1];
                scale[0] = 1;
                for (int k = 1; k <= p; ++k) scale[k] = scale[k-1]*(ch.r/nd.r);
                for (int k = 0; k < nc; ++k) tmp[k] = in[k]*scale[s.terms[full[k]].degree];
                shift(tmp, (ch.x-nd.x)/nd.r, (ch.y-nd.y)/nd.r, (ch.z-nd.z)/nd.r, true);
                for (int k = 0; k < nc; ++k) out[k] += tmp[k];
            }
        }
        out[0] = nd.mass;
        out[1] = out[2] = out[3] = 0;
    }
    s.log("harmonic moments");
}

template<bool Both> void Harmonic::translate4(Solver &s, const Pair *pairs, int valid, int degree) {
    int q = degree ? degree : p;
    const Node *a[4], *b[4];
    const double *am[4], *bm[4];
    double *al[4], *bl[4];
    for (int k = 0; k < 4; ++k) {
        a[k] = &s.nodes[pairs[k].a]; b[k] = &s.nodes[pairs[k].b];
        am[k] = moments+size_t(pairs[k].a)*stride;
        bm[k] = moments+size_t(pairs[k].b)*stride;
        al[k] = locals+size_t(pairs[k].a)*stride;
        bl[k] = locals+size_t(pairs[k].b)*stride;
    }
    __m256d dx = _mm256_set_pd(a[3]->x-b[3]->x, a[2]->x-b[2]->x, a[1]->x-b[1]->x, a[0]->x-b[0]->x);
    __m256d dy = _mm256_set_pd(a[3]->y-b[3]->y, a[2]->y-b[2]->y, a[1]->y-b[1]->y, a[0]->y-b[0]->y);
    __m256d dz = _mm256_set_pd(a[3]->z-b[3]->z, a[2]->z-b[2]->z, a[1]->z-b[1]->z, a[0]->z-b[0]->z);
    __m256d one = _mm256_set1_pd(1);
    __m256d inv = _mm256_div_pd(one, _mm256_sqrt_pd(_mm256_fmadd_pd(dx, dx,
                    _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)))));
    alignas(64) __m256d deriv[CMAX], ma[CMAX], mb[CMAX];
    kernel_harmonic4(q, _mm256_mul_pd(dx, inv), _mm256_mul_pd(dy, inv), _mm256_mul_pd(dz, inv), deriv);
    for (int i = 0; i < s.count[q]; ++i) {
        const auto &t = s.terms[i];
        if (t.z == 2)
            deriv[i] = _mm256_sub_pd(_mm256_setzero_pd(),
                _mm256_add_pd(deriv[s.index[t.x+2][t.y][0]], deriv[s.index[t.x][t.y+2][0]]));
    }
    __m256d sa[PMAX+1], sb[PMAX+1];
    sa[0] = sb[0] = one;
    __m256d ratioa = _mm256_mul_pd(inv, _mm256_set_pd(a[3]->r, a[2]->r, a[1]->r, a[0]->r));
    __m256d ratiob = _mm256_mul_pd(inv, _mm256_set_pd(-b[3]->r, -b[2]->r, -b[1]->r, -b[0]->r));
    for (int k = 1; k <= q; ++k) {
        sa[k] = _mm256_mul_pd(sa[k-1], ratioa); sb[k] = _mm256_mul_pd(sb[k-1], ratiob);
    }
    __m256d am0 = _mm256_set_pd(am[3][0], am[2][0], am[1][0], am[0][0]);
    __m256d bm0 = _mm256_set_pd(bm[3][0], bm[2][0], bm[1][0], bm[0][0]);
    for (int k = 4; k < q*q; ++k) {
        int d = s.terms[full[k]].degree;
        if constexpr (Both)
            ma[k] = _mm256_mul_pd(sa[d], _mm256_set_pd(am[3][k], am[2][k], am[1][k], am[0][k]));
        mb[k] = _mm256_mul_pd(sb[d], _mm256_set_pd(bm[3][k], bm[2][k], bm[1][k], bm[0][k]));
    }
    for (int i = 1; i < (q+1)*(q+1); ++i) {
        int fi = full[i], deg = s.terms[fi].degree;
        int bmax = (q-deg+1)*(q-deg+1);
        const int *g = gamma[i].data();
        __m256d v1 = _mm256_mul_pd(deriv[fi], bm0), v2 = _mm256_mul_pd(deriv[fi], am0);
        __m256d w1 = _mm256_setzero_pd(), w2 = w1, t1 = w1, t2 = w1, z1 = w1, z2 = w1;
        int j = 4;
        for (; j+3 < bmax; j += 4) {
            __m256d d0 = deriv[g[j]], d1 = deriv[g[j+1]], d2 = deriv[g[j+2]], d3 = deriv[g[j+3]];
            v1 = _mm256_fmadd_pd(d0, mb[j], v1); w1 = _mm256_fmadd_pd(d1, mb[j+1], w1);
            t1 = _mm256_fmadd_pd(d2, mb[j+2], t1); z1 = _mm256_fmadd_pd(d3, mb[j+3], z1);
            if constexpr (Both) {
                v2 = _mm256_fmadd_pd(d0, ma[j], v2); w2 = _mm256_fmadd_pd(d1, ma[j+1], w2);
                t2 = _mm256_fmadd_pd(d2, ma[j+2], t2); z2 = _mm256_fmadd_pd(d3, ma[j+3], z2);
            }
        }
        v1 = _mm256_add_pd(_mm256_add_pd(v1, w1), _mm256_add_pd(t1, z1));
        if constexpr (Both) v2 = _mm256_add_pd(_mm256_add_pd(v2, w2), _mm256_add_pd(t2, z2));
        for (; j < bmax; ++j) {
            __m256d d = deriv[g[j]];
            v1 = _mm256_fmadd_pd(d, mb[j], v1);
            if constexpr (Both) v2 = _mm256_fmadd_pd(d, ma[j], v2);
        }
        alignas(32) double da[4], db[4];
        _mm256_store_pd(da, _mm256_mul_pd(v1, _mm256_mul_pd(inv, sa[deg])));
        if constexpr (Both) _mm256_store_pd(db, _mm256_mul_pd(v2, _mm256_mul_pd(inv, sb[deg])));
        for (int k = 0; k < valid; ++k) {
            al[k][i] += da[k];
            if constexpr (Both) bl[k][i] += db[k];
        }
    }
}

void Harmonic::far_field(Solver &s) {
    const auto &pairs = s.far_pairs;
    size_t i = 0;
    bool fast = false;
    while (i < pairs.size()) {
        if (pairs[i].fullsoft) { ++i; continue; }
        bool both = pairs[i].mutual;
        size_t end = i+1;
        while (end < pairs.size() && pairs[end].mutual == both && !pairs[end].fullsoft) ++end;
        for (; i+3 < end; i += 4) {
            if ((i&255) < 4 && cpu_time() > s.budget-s.finish_reserve) fast = true;
            int usep = fast ? std::max(8, p-4) : p;
            if (fast && cpu_time() > s.budget-s.write_margin-0.025*s.budget) usep = 1;
            if (both) translate4<true>(s, pairs.data()+i, 4, usep);
            else translate4<false>(s, pairs.data()+i, 4, usep);
        }
        if (i < end) {
            Pair padded[4];
            int valid = int(end-i);
            for (int k = 0; k < 4; ++k) padded[k] = pairs[std::min(i+k, end-1)];
            if (both) translate4<true>(s, padded, valid, fast ? std::max(8, p-4) : p);
            else translate4<false>(s, padded, valid, fast ? std::max(8, p-4) : p);
            i = end;
        }
    }
    s.log(fast ? "harmonic m2l (budget fallback)" : "harmonic m2l");
}

void Harmonic::downward(Solver &s) {
    bool fast = false;
    for (size_t ni = 0; ni < s.nodes.size(); ++ni) {
        if ((ni&127) == 0 && cpu_time() > s.budget-s.write_margin-0.005*s.budget) fast = true;
        if (!s.active_count[ni]) continue;
        const auto &nd = s.nodes[ni];
        double *in = locals+ni*stride;
        if (fast) {
            if (nd.left >= 0) {
                for (int child : {nd.left, nd.right}) {
                    double ratio = s.nodes[child].r/nd.r;
                    double *out = locals+size_t(child)*stride;
                    for (int k = 1; k <= 3; ++k) out[k] += in[k]*ratio;
                }
            } else {
                for (int j = nd.begin; j < nd.end; ++j) if (s.active[j]) {
                    s.ax[j] += in[compact[s.index[1][0][0]]]/nd.r;
                    s.ay[j] += in[compact[s.index[0][1][0]]]/nd.r;
                    s.az[j] += in[compact[s.index[0][0][1]]]/nd.r;
                }
            }
            continue;
        }
        if (nd.left >= 0) {
            for (int child : {nd.left, nd.right}) {
                if (!s.active_count[child]) continue;
                const auto &ch = s.nodes[child];
                alignas(64) double tmp[CMAX], scale[PMAX+1];
                std::memcpy(tmp, in, nc*sizeof(double));
                shift(tmp, (ch.x-nd.x)/nd.r, (ch.y-nd.y)/nd.r, (ch.z-nd.z)/nd.r, false);
                scale[0] = 1;
                for (int k = 1; k <= p; ++k) scale[k] = scale[k-1]*(ch.r/nd.r);
                double *out = locals+size_t(child)*stride;
                for (int k = 1; k < nc; ++k) out[k] += tmp[k]*scale[s.terms[full[k]].degree];
            }
        } else {
            alignas(64) double raw[CMAX];
            for (int k = 0; k < s.count[p]; ++k) {
                double sum = 0;
                for (auto r : reduction[k]) sum += in[r.i]*r.weight;
                raw[k] = sum;
            }
            alignas(64) double cx[CMAX], cy[CMAX], cz[CMAX];
            for (size_t k = 0; k < gradient_factor.size(); ++k) {
                cx[k] = raw[gradient_x[k]]*gradient_factor[k]/nd.r;
                cy[k] = raw[gradient_y[k]]*gradient_factor[k]/nd.r;
                cz[k] = raw[gradient_z[k]]*gradient_factor[k]/nd.r;
            }
            for (int j = nd.begin; j < nd.end; j += 4) {
                int j1 = std::min(j+1, nd.end-1), j2 = std::min(j+2, nd.end-1), j3 = std::min(j+3, nd.end-1);
                __m256d ir = _mm256_set1_pd(1/nd.r);
                __m256d dx = _mm256_mul_pd(_mm256_sub_pd(_mm256_set_pd(s.x[j3], s.x[j2], s.x[j1], s.x[j]), _mm256_set1_pd(nd.x)), ir);
                __m256d dy = _mm256_mul_pd(_mm256_sub_pd(_mm256_set_pd(s.y[j3], s.y[j2], s.y[j1], s.y[j]), _mm256_set1_pd(nd.y)), ir);
                __m256d dz = _mm256_mul_pd(_mm256_sub_pd(_mm256_set_pd(s.z[j3], s.z[j2], s.z[j1], s.z[j]), _mm256_set1_pd(nd.z)), ir);
                __m256d fx = _mm256_setzero_pd(), fy = fx, fz = fx;
                size_t off = 0;
                for (int ix = p-1; ix >= 0; --ix) {
                    __m256d yx = _mm256_setzero_pd(), yy = yx, yz = yx;
                    for (int iy = p-1-ix; iy >= 0; --iy) {
                        __m256d zx = _mm256_setzero_pd(), zy = zx, zz = zx;
                        for (int iz = p-1-ix-iy; iz >= 0; --iz, ++off) {
                            zx = _mm256_fmadd_pd(zx, dz, _mm256_set1_pd(cx[off]));
                            zy = _mm256_fmadd_pd(zy, dz, _mm256_set1_pd(cy[off]));
                            zz = _mm256_fmadd_pd(zz, dz, _mm256_set1_pd(cz[off]));
                        }
                        yx = _mm256_fmadd_pd(yx, dy, zx); yy = _mm256_fmadd_pd(yy, dy, zy);
                        yz = _mm256_fmadd_pd(yz, dy, zz);
                    }
                    fx = _mm256_fmadd_pd(fx, dx, yx); fy = _mm256_fmadd_pd(fy, dx, yy);
                    fz = _mm256_fmadd_pd(fz, dx, yz);
                }
                alignas(32) double ox[4], oy[4], oz[4];
                _mm256_store_pd(ox, fx); _mm256_store_pd(oy, fy); _mm256_store_pd(oz, fz);
                for (int k = 0; k < 4 && j+k < nd.end; ++k) {
                    if (s.active[j+k]) { s.ax[j+k] += ox[k]; s.ay[j+k] += oy[k]; s.az[j+k] += oz[k]; }
                }
            }
        }
    }
    s.log(fast ? "harmonic downward (budget fallback)" : "harmonic downward");
}
