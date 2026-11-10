#pragma once

void Solver::precise_product(const LongVec& x, LongVec& out, long double scale) {
    if (difference && rowsum_long.empty()) {
        rowsum_long.assign(n, 0);
        for (int i = 0; i < n; ++i)
            for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k) rowsum_long[i] += original[k];
    }
    for (int i = 0; i < n; ++i) {
        long double sum = 0;
        if (difference) {
            long double xi = x[i];
            sum = rowsum_long[i] * xi;
            for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k)
                sum += static_cast<long double>(original[k]) * (x[col[k]] - xi);
        } else {
            for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k)
                sum += static_cast<long double>(original[k]) * x[col[k]];
        }
        out[i] = sum * scale;
    }
}

void Solver::precise_direct(const Vec& b0, const Vec& b1) {
    for (int i = 0; i < n; ++i) {
        int j = mate[i];
        if (j >= 0 && j < i) continue;
        long double aa = 0, bb = 0;
        for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k) {
            if (col[k] == i) aa = static_cast<long double>(original[k]) * t;
            else if (col[k] == j) bb = static_cast<long double>(original[k]) * t;
        }
        if (j < 0) {
            y[i] = static_cast<double>(std::exp(-aa) * b0[i] + phi(aa) * t * b1[i]);
            continue;
        }
        long double cc = 0;
        for (int k = ptr[j]; k < static_cast<int>(ptr[j + 1]); ++k)
            if (col[k] == j) cc = static_cast<long double>(original[k]) * t;
        long double gap = std::hypot(aa - cc, 2 * bb);
        long double hi = (aa + cc + gap) / 2;
        long double lo = hi == 0 ? 0 : (aa * cc - bb * bb) / hi;
        long double c, s;
        if (aa >= cc) {
            c = std::sqrt((1 + (aa - cc) / gap) / 2);
            s = bb / (gap * c);
        } else {
            s = std::copysign(std::sqrt((1 - (aa - cc) / gap) / 2), bb);
            c = bb / (gap * s);
        }
        long double h0 = c * b0[i] + s * b0[j];
        long double l0 = -s * b0[i] + c * b0[j];
        long double h1 = static_cast<long double>(t) * (c * b1[i] + s * b1[j]);
        long double l1 = static_cast<long double>(t) * (-s * b1[i] + c * b1[j]);
        long double yh = std::exp(-hi) * h0 + phi(hi) * h1;
        long double yl = std::exp(-lo) * l0 + phi(lo) * l1;
        y[i] = static_cast<double>(c * yh - s * yl);
        y[j] = static_cast<double>(s * yh + c * yl);
    }
}

void Solver::precise_taylor(const Vec& b0, const Vec& b1) {
    q_long.resize(n);
    delta_long.resize(n);
    work_long.resize(n);
    for (int i = 0; i < n; ++i) q_long[i] = b0[i];
    precise_product(q_long, work_long, t);
    for (int i = 0; i < n; ++i) {
        q_long[i] = static_cast<long double>(t) * b1[i] - work_long[i];
        delta_long[i] = b0[i];
    }
    for (int k = 0; k < 128; ++k) {
        long double norm2 = 0, ym = 0;
        for (int i = 0; i < n; ++i) {
            norm2 += q_long[i] * q_long[i];
            ym = std::max(ym, std::abs(delta_long[i]));
        }
        if (std::sqrt(norm2) <= 0.005L * (atol + rtol * ym)) break;
        for (int i = 0; i < n; ++i) delta_long[i] += q_long[i];
        precise_product(q_long, work_long, t);
        long double scale = -1.0L / (k + 2);
        for (int i = 0; i < n; ++i) q_long[i] = scale * work_long[i];
    }
    for (int i = 0; i < n; ++i) y[i] = static_cast<double>(delta_long[i]);
}

void Solver::precise_chebyshev(const Vec& b0, const Vec& b1, int d) {
    q_long.assign(n, 0);
    delta_long.assign(n, 0);
    work_long.resize(n);
    long double scale = 4 * static_cast<long double>(t) / upper;
    for (int k = d; k >= 1; --k) {
        if (k != d) precise_product(q_long, work_long, scale);
        else std::fill(work_long.begin(), work_long.end(), 0);
        long double e = ecoef_long[k], p = pcoef_long[k] * t;
        for (int i = 0; i < n; ++i) {
            long double dd = (delta_long[i] - work_long[i]) + (e * b0[i] + p * b1[i]);
            delta_long[i] = dd;
            q_long[i] += dd;
        }
    }
    if (d) precise_product(q_long, work_long, scale);
    else std::fill(work_long.begin(), work_long.end(), 0);
    long double e = ecoef_long[0], p = pcoef_long[0] * t;
    for (int i = 0; i < n; ++i)
        y[i] = static_cast<double>((delta_long[i] - 0.5L * work_long[i])
                                   + (e * b0[i] + p * b1[i]));
}

void Solver::refine_if_needed(const Vec& b0, const Vec& b1) {
    if (rtol > 1e-7 || t == 0 || upper == 0) return;
    double bmax = 0, cmax = 0, ym = 0;
    for (int i = 0; i < n; ++i) {
        bmax = std::max(bmax, std::abs(b0[i]));
        cmax = std::max(cmax, t * std::abs(b1[i]));
        ym = std::max(ym, std::abs(y[i]));
    }
    double scale = bmax + cmax;
    double threshold = atol + rtol * ym;
    double indicator = 4 * std::numeric_limits<double>::epsilon() * upper * scale;
    if (ym >= scale * 1e-4 || indicator <= threshold) return;
    used_extended = true;
    if (matrix_norm <= 1.5) {
        precise_taylor(b0, b1);
        return;
    }
    if (!prepared) prepare();
    if (direct) {
        precise_direct(b0, b1);
        return;
    }
    long double bn2 = 0, cn2 = 0;
    for (int i = 0; i < n; ++i) {
        bn2 += static_cast<long double>(b0[i]) * b0[i];
        long double c = static_cast<long double>(t) * b1[i];
        cn2 += c * c;
    }
    int d = degree(static_cast<double>(std::sqrt(bn2)), static_cast<double>(std::sqrt(cn2)),
                   0.005 * threshold);
    d = std::min(d + 2, static_cast<int>(ecoef.size()) - 1);
    precise_chebyshev(b0, b1, d);
}
