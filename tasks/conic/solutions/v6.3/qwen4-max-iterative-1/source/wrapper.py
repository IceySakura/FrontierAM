import json
import os
import sys
import time
import zipfile
import array
import struct

def read_npy_raw(path, name):
    with zipfile.ZipFile(path) as z:
        b = z.read(name + ".npy")
    if b[:6] != b"\x93NUMPY": raise ValueError("npy")
    major = b[6]
    if major == 1:
        hlen = int.from_bytes(b[8:10], "little"); off = 10 + hlen
    else:
        hlen = int.from_bytes(b[8:12], "little"); off = 12 + hlen
    h = b[10 if major == 1 else 12:off].decode("latin1")
    d = h.find("'descr'"); d = h.find("'", d + 7); e = h.find("'", d + 1); descr = h[d+1:e]
    s = h.find("'shape'"); s = h.find("(", s); e = h.find(")", s)
    shape = tuple(int(x) for x in h[s+1:e].replace(",", " ").split())
    data = b[off:]
    return descr, shape, data

def read_f64_list(path, name):
    descr, shape, data = read_npy_raw(path, name)
    a = array.array("d"); a.frombytes(data)
    if sys.byteorder != "little": a.byteswap()
    return a.tolist()

def read_i_list(path, name):
    descr, shape, data = read_npy_raw(path, name)
    code = "i" if descr in ("<i4", "|i4") else "q"
    a = array.array(code); a.frombytes(data)
    if sys.byteorder != "little": a.byteswap()
    return a.tolist()

def write_npz_lists(path, status, x=None, s=None, z=None):
    def npy_f64(v):
        if v is None: v = []
        a = array.array("d", v)
        if sys.byteorder != "little": a.byteswap()
        hdr = "{'descr': '<f8', 'fortran_order': False, 'shape': (%d,), }" % len(v)
        hdr = hdr.ljust(64 - 10 - 1) + "\n"
        return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode("latin1") + a.tobytes()
    def npy_u(s):
        data = s.encode("utf-32-le")
        hdr = "{'descr': '<U%d', 'fortran_order': False, 'shape': (), }" % len(s)
        hdr = hdr.ljust(64 - 10 - 1) + "\n"
        return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode("latin1") + data
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata_json.npy", npy_u(json.dumps({"schema_version": 1, "status": status})))
        if x is not None: zf.writestr("x.npy", npy_f64(x))
        if s is not None: zf.writestr("s.npy", npy_f64(s))
        if z is not None: zf.writestr("z.npy", npy_f64(z))

def solve_lp_fast(path, cones, outpath):
    Pd = read_f64_list(path, "P_data")
    if Pd: return False
    q = read_f64_list(path, "q"); b = read_f64_list(path, "b")
    n = len(q); m = len(b)
    Ap = read_i_list(path, "A_indptr"); Ai = read_i_list(path, "A_indices"); Av = read_f64_list(path, "A_data")
    starts = [0]; inds = []; vals = []
    for j in range(n):
        col = {}
        for k in range(Ap[j], Ap[j+1]):
            i = Ai[k]; col[i] = col.get(i, 0.0) + Av[k]
        for i in sorted(col):
            if col[i] != 0.0: inds.append(i); vals.append(col[i])
        starts.append(len(inds))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
    import _core_shim
    _core = _core_shim.core
    h = _core._Highs()
    h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "on"); h.setOptionValue("time_limit", 0.85)
    for opt in ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "ipm_optimality_tolerance"):
        try: h.setOptionValue(opt, 1e-10)
        except Exception: pass
    lp = _core.HighsLp()
    lp.num_col_ = n; lp.num_row_ = m; lp.col_cost_ = q
    lp.col_lower_ = [-1e30] * n; lp.col_upper_ = [1e30] * n
    rl = [-1e30] * m; ru = [1e30] * m; st = 0
    for c in cones:
        w = c["dim"]
        for r in range(st, st + w):
            if c["type"] == "zero": rl[r] = ru[r] = b[r]
            else: ru[r] = b[r]
        st += w
    lp.row_lower_ = rl; lp.row_upper_ = ru
    lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = m
    lp.a_matrix_.format_ = _core.MatrixFormat.kColwise
    lp.a_matrix_.start_ = starts; lp.a_matrix_.index_ = inds; lp.a_matrix_.value_ = vals
    if h.passModel(lp) != _core.HighsStatus.kOk: return False
    if h.run() != _core.HighsStatus.kOk: return False
    status = h.getModelStatus()
    sol = h.getSolution()
    ok = status == _core.HighsModelStatus.kOptimal or (status in (_core.HighsModelStatus.kTimeLimit, _core.HighsModelStatus.kIterationLimit) and sol.value_valid and sol.dual_valid)
    if not ok:
        if status == _core.HighsModelStatus.kInfeasible:
            tstarts = [0]; tinds = []; tvals = []
            rowentries = [[] for _ in range(m)]
            for j in range(n):
                for k in range(starts[j], starts[j+1]): rowentries[inds[k]].append((j, vals[k]))
            for i in range(m):
                for j, v in rowentries[i]:
                    if v != 0: tinds.append(j); tvals.append(v)
                tstarts.append(len(tinds))
            h2 = _core._Highs(); h2.setOptionValue("output_flag", False); h2.setOptionValue("threads", 1); h2.setOptionValue("time_limit", 0.15)
            lp2 = _core.HighsLp(); lp2.num_col_ = m; lp2.num_row_ = n; lp2.col_cost_ = b
            lb = [0.0] * m; ub = [1.0] * m; st = 0
            for c in cones:
                w = c["dim"]
                if c["type"] == "zero":
                    for r in range(st, st + w): lb[r] = -1.0
                st += w
            lp2.col_lower_ = lb; lp2.col_upper_ = ub; lp2.row_lower_ = [0.0] * n; lp2.row_upper_ = [0.0] * n
            lp2.a_matrix_.num_col_ = m; lp2.a_matrix_.num_row_ = n; lp2.a_matrix_.format_ = _core.MatrixFormat.kColwise
            lp2.a_matrix_.start_ = tstarts; lp2.a_matrix_.index_ = tinds; lp2.a_matrix_.value_ = tvals
            if h2.passModel(lp2) == _core.HighsStatus.kOk and h2.run() == _core.HighsStatus.kOk:
                if h2.getObjectiveValue() < -1e-8:
                    z = list(h2.getSolution().col_value)
                    write_npz_lists(outpath, "primal_infeasible", None, None, z); return True
        if status == _core.HighsModelStatus.kUnbounded:
            rowentries = [[] for _ in range(m)]
            for j in range(n):
                for k in range(starts[j], starts[j+1]): rowentries[inds[k]].append((j, vals[k]))
            h2 = _core._Highs(); h2.setOptionValue("output_flag", False); h2.setOptionValue("threads", 1); h2.setOptionValue("time_limit", 0.15)
            lp2 = _core.HighsLp(); lp2.num_col_ = n; lp2.num_row_ = m; lp2.col_cost_ = q
            lp2.col_lower_ = [-1.0] * n; lp2.col_upper_ = [1.0] * n
            rl = [-1e30] * m; ru = [0.0] * m; st = 0
            for c in cones:
                w = c["dim"]
                if c["type"] == "zero":
                    for r in range(st, st + w): rl[r] = 0.0
                st += w
            lp2.row_lower_ = rl; lp2.row_upper_ = ru
            cstarts = [0]; cinds = []; cvals = []
            colentries = [[] for _ in range(n)]
            for i in range(m):
                for j, v in rowentries[i]: colentries[j].append((i, v))
            for j in range(n):
                col = {}
                for i, v in colentries[j]: col[i] = col.get(i, 0.0) + v
                for i in sorted(col):
                    if col[i] != 0: cinds.append(i); cvals.append(col[i])
                cstarts.append(len(cinds))
            lp2.a_matrix_.num_col_ = n; lp2.a_matrix_.num_row_ = m; lp2.a_matrix_.format_ = _core.MatrixFormat.kColwise
            lp2.a_matrix_.start_ = cstarts; lp2.a_matrix_.index_ = cinds; lp2.a_matrix_.value_ = cvals
            if h2.passModel(lp2) == _core.HighsStatus.kOk and h2.run() == _core.HighsStatus.kOk:
                if h2.getObjectiveValue() < -1e-8:
                    x = list(h2.getSolution().col_value)
                    Ax = [0.0] * m
                    for j in range(n):
                        for k in range(starts[j], starts[j+1]): Ax[inds[k]] += vals[k] * x[j]
                    s = [-v for v in Ax]
                    write_npz_lists(outpath, "dual_infeasible", x, s, None); return True
        return False
    x = list(sol.col_value); z = [-v for v in sol.row_dual]
    Ax = [0.0] * m
    for j in range(n):
        xj = x[j]
        for k in range(starts[j], starts[j+1]): Ax[inds[k]] += vals[k] * xj
    s = [b[i] - Ax[i] for i in range(m)]
    write_npz_lists(outpath, "solved", x, s, z)
    return True

def solve_qp_fast(path, cones, outpath):
    q = read_f64_list(path, "q"); b = read_f64_list(path, "b")
    n = len(q); m = len(b)
    Ap = read_i_list(path, "A_indptr"); Ai = read_i_list(path, "A_indices"); Av = read_f64_list(path, "A_data")
    Pp = read_i_list(path, "P_indptr"); Pi = read_i_list(path, "P_indices"); Pv = read_f64_list(path, "P_data")
    if not Pv: return False
    starts = [0]; inds = []; vals = []
    for j in range(n):
        col = {}
        for k in range(Ap[j], Ap[j+1]):
            i = Ai[k]; col[i] = col.get(i, 0.0) + Av[k]
        for i in sorted(col):
            if col[i] != 0.0: inds.append(i); vals.append(col[i])
        starts.append(len(inds))
    hstarts = [0]; hinds = []; hvals = []
    for j in range(n):
        col = {}
        for k in range(Pp[j], Pp[j+1]):
            i = Pi[k]
            if i >= j: col[i] = col.get(i, 0.0) + Pv[k]
        for i in sorted(col):
            if col[i] != 0.0: hinds.append(i); hvals.append(col[i])
        hstarts.append(len(hinds))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
    import _core_shim
    _core = _core_shim.core
    h = _core._Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "on"); h.setOptionValue("time_limit", 0.85)
    for opt in ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "ipm_optimality_tolerance"):
        try: h.setOptionValue(opt, 1e-10)
        except Exception: pass
    lp = _core.HighsLp(); lp.num_col_ = n; lp.num_row_ = m; lp.col_cost_ = q
    lp.col_lower_ = [-1e30] * n; lp.col_upper_ = [1e30] * n
    rl = [-1e30] * m; ru = [1e30] * m; st = 0
    for c in cones:
        w = c["dim"]
        for r in range(st, st + w):
            if c["type"] == "zero": rl[r] = ru[r] = b[r]
            else: ru[r] = b[r]
        st += w
    lp.row_lower_ = rl; lp.row_upper_ = ru
    lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = m; lp.a_matrix_.format_ = _core.MatrixFormat.kColwise
    lp.a_matrix_.start_ = starts; lp.a_matrix_.index_ = inds; lp.a_matrix_.value_ = vals
    if h.passModel(lp) != _core.HighsStatus.kOk: return False
    hes = _core.HighsHessian(); hes.dim_ = n; hes.format_ = _core.HessianFormat.kTriangular
    hes.start_ = hstarts; hes.index_ = hinds; hes.value_ = hvals
    if h.passHessian(hes) != _core.HighsStatus.kOk: return False
    if h.run() != _core.HighsStatus.kOk: return False
    if h.getModelStatus() != _core.HighsModelStatus.kOptimal: return False
    sol = h.getSolution()
    if not (sol.value_valid and sol.dual_valid): return False
    x = list(sol.col_value); z = [-v for v in sol.row_dual]
    Px = [0.0] * n; Ax = [0.0] * m; Atz = [0.0] * n
    for j in range(n):
        xj = x[j]
        for k in range(Pp[j], Pp[j+1]): Px[Pi[k]] += Pv[k] * xj
        for k in range(starts[j], starts[j+1]): Ax[inds[k]] += vals[k] * xj
    for j in range(n):
        ss = 0.0
        for k in range(starts[j], starts[j+1]): ss += vals[k] * z[inds[k]]
        Atz[j] = ss
    s = [b[i] - Ax[i] for i in range(m)]
    def infn(v): return max((abs(t) for t in v), default=0.0)
    rp = infn([Ax[i] + s[i] - b[i] for i in range(m)]) / (1 + max(infn(Ax), infn(s), infn(b)))
    rd = infn([Px[i] + q[i] + Atz[i] for i in range(n)]) / (1 + max(infn(Px), infn(q), infn(Atz)))
    xPx = sum(x[i] * Px[i] for i in range(n)); qx = sum(q[i] * x[i] for i in range(n)); bz = sum(b[i] * z[i] for i in range(m))
    p0 = 0.5 * xPx + qx; d0 = -0.5 * xPx - bz; gap = abs(p0 - d0); scale = max(abs(p0), abs(d0))
    gapok = gap <= 8e-9 or (scale > 0 and gap / scale <= 8e-9)
    cp = cd = 0.0; st = 0
    for c in cones:
        w = c["dim"]; sb = s[st:st+w]; zb = z[st:st+w]
        sc = max(1.0, infn(sb)); zc = max(1.0, infn(zb))
        if c["type"] == "zero": cp = max(cp, infn(sb) / sc)
        else:
            cp = max(cp, max(0.0, -min(sb) / sc)); cd = max(cd, max(0.0, -min(zb) / zc))
        st += w
    if rp <= 8e-9 and rd <= 8e-9 and gapok and cp <= 8e-9 and cd <= 8e-9:
        write_npz_lists(outpath, "solved", x, s, z); return True
    return False

OUT_PATH_GLOBAL = None

def read_meta(path):
    with zipfile.ZipFile(path) as z:
        b = z.read("metadata_json.npy")
    if b[:6] != b"\x93NUMPY":
        raise ValueError("bad npy")
    major = b[6]
    if major == 1:
        hlen = int.from_bytes(b[8:10], "little"); off = 10 + hlen
    else:
        hlen = int.from_bytes(b[8:12], "little"); off = 12 + hlen
    data = b[off:]
    return json.loads(data.decode("utf-32-le"))

def csc_mv(data, indices, indptr, x, nrows):
    if data.size == 0:
        return np.zeros(nrows)
    cols = np.repeat(np.arange(x.size, dtype=np.int64), np.diff(indptr))
    return np.bincount(indices, weights=data * x[cols], minlength=nrows)

def csc_tmv(data, indices, indptr, x, ncols):
    if data.size == 0:
        return np.zeros(ncols)
    cols = np.repeat(np.arange(ncols, dtype=np.int64), np.diff(indptr))
    return np.bincount(cols, weights=data * x[indices], minlength=ncols)

def inf_norm(x):
    return float(np.max(np.abs(x))) if x.size else 0.0

def sum_csc(data, indices, indptr, nrows):
    if data.size == 0:
        return data, indices, indptr
    cols = np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(indptr))
    order = np.lexsort((indices, cols))
    cs = cols[order]; ix = indices[order]; vv = data[order]
    key = cs * np.int64(nrows) + ix
    uniq = np.ones(key.size, dtype=bool)
    uniq[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(uniq)
    vals = np.add.reduceat(vv, starts)
    rows = ix[starts]
    ccol = cs[starts]
    cnt = np.bincount(ccol, minlength=indptr.size - 1)
    ip = np.zeros(indptr.size, dtype=np.int32)
    ip[1:] = np.cumsum(cnt)
    return vals, rows.astype(np.int32), ip

def check_linear_qp(P, q, A, b, cones, x, z, tol=8e-9):
    n, m = q.size, b.size
    Pd, Pi, Pp = P
    Ad, Ai, Ap = A
    Px = csc_mv(Pd, Pi, Pp, x, n)
    Ax = csc_mv(Ad, Ai, Ap, x, m)
    Atz = csc_tmv(Ad, Ai, Ap, z, n)
    s = b - Ax
    rp = inf_norm(Ax + s - b) / (1 + max(inf_norm(Ax), inf_norm(s), inf_norm(b)))
    rd = inf_norm(Px + q + Atz) / (1 + max(inf_norm(Px), inf_norm(q), inf_norm(Atz)))
    xPx = float(x @ Px); qx = float(q @ x); bz = float(b @ z)
    p0 = 0.5 * xPx + qx; d0 = -0.5 * xPx - bz
    gap = abs(p0 - d0); scale = max(abs(p0), abs(d0))
    gapok = gap <= tol or (scale > 0 and gap / scale <= tol)
    cp = 0.0; cd = 0.0; start = 0
    for c in cones:
        w = c['dim']; sb = s[start:start+w]; zb = z[start:start+w]
        sc = max(1.0, inf_norm(sb)); zc = max(1.0, inf_norm(zb))
        if c['type'] == 'zero':
            cp = max(cp, inf_norm(sb) / sc)
        else:
            cp = max(cp, max(0.0, -float(np.min(sb)) / sc))
            cd = max(cd, max(0.0, -float(np.min(zb)) / zc))
        start += w
    return rp <= tol and rd <= tol and gapok and cp <= tol and cd <= tol, s

def check_soc2(P, q, A, b, cones, x, z, tol=8e-9):
    n, m = q.size, b.size
    Pd, Pi, Pp = P; Ad, Ai, Ap = A
    Px = csc_mv(Pd, Pi, Pp, x, n); Ax = csc_mv(Ad, Ai, Ap, x, m); Atz = csc_tmv(Ad, Ai, Ap, z, n)
    s = b - Ax
    rp = inf_norm(Ax + s - b) / (1 + max(inf_norm(Ax), inf_norm(s), inf_norm(b)))
    rd = inf_norm(Px + q + Atz) / (1 + max(inf_norm(Px), inf_norm(q), inf_norm(Atz)))
    xPx = float(x @ Px); qx = float(q @ x); bz = float(b @ z)
    p0 = 0.5*xPx+qx; d0 = -0.5*xPx-bz; gap = abs(p0-d0); scale = max(abs(p0),abs(d0))
    gapok = gap <= tol or (scale > 0 and gap/scale <= tol)
    cp = cd = 0.0; start = 0
    for c in cones:
        w = c["dim"]; sb = s[start:start+w]; zb = z[start:start+w]
        sc = max(1.0, inf_norm(sb)); zc = max(1.0, inf_norm(zb))
        if c["type"] == "zero": cp = max(cp, inf_norm(sb)/sc)
        elif c["type"] == "nonnegative":
            cp = max(cp, max(0.0, -float(np.min(sb))/sc)); cd = max(cd, max(0.0, -float(np.min(zb))/zc))
        else:
            cp = max(cp, max(0.0, abs(float(sb[1])) - float(sb[0]))/sc)
            cd = max(cd, max(0.0, abs(float(zb[1])) - float(zb[0]))/zc)
        start += w
    return rp <= tol and rd <= tol and gapok and cp <= tol and cd <= tol, s

def check_general(P, q, A, b, cones, x, z, tol=8e-9):
    n, m = q.size, b.size
    Pd, Pi, Pp = P; Ad, Ai, Ap = A
    Px = csc_mv(Pd, Pi, Pp, x, n); Ax = csc_mv(Ad, Ai, Ap, x, m); Atz = csc_tmv(Ad, Ai, Ap, z, n)
    s = b - Ax
    rp = inf_norm(Ax + s - b) / (1 + max(inf_norm(Ax), inf_norm(s), inf_norm(b)))
    rd = inf_norm(Px + q + Atz) / (1 + max(inf_norm(Px), inf_norm(q), inf_norm(Atz)))
    xPx = float(x @ Px); qx = float(q @ x); bz = float(b @ z)
    p0 = 0.5*xPx+qx; d0 = -0.5*xPx-bz; gap = abs(p0-d0); scale = max(abs(p0),abs(d0))
    gapok = gap <= tol or (scale > 0 and gap/scale <= tol)
    cp = cd = 0.0; start = 0
    for c in cones:
        w = c["dim"]; sb = s[start:start+w]; zb = z[start:start+w]
        sc = max(1.0, inf_norm(sb)); zc = max(1.0, inf_norm(zb))
        if c["type"] == "zero": cp = max(cp, inf_norm(sb)/sc)
        elif c["type"] == "nonnegative":
            cp = max(cp, max(0.0, -float(np.min(sb))/sc)); cd = max(cd, max(0.0, -float(np.min(zb))/zc))
        elif c["type"] == "soc":
            cp = max(cp, max(0.0, float(np.linalg.norm(sb[1:])) - float(sb[0]))/sc)
            cd = max(cd, max(0.0, float(np.linalg.norm(zb[1:])) - float(zb[0]))/zc)
        elif c["type"] == "exponential":
            def ed(a, bb, cc):
                B = max(abs(bb), max(a, 0.0), max(-cc, 0.0))
                if bb > 0 and cc > 0:
                    return min(B, max(0.0, a - bb * (np.log(cc) - np.log(bb))))
                return B
            u = sb / sc
            cp = max(cp, ed(float(u[0]), float(u[1]), float(u[2])))
            zz = zb / zc
            cd = max(cd, ed(-float(zz[1]), -float(zz[0]), np.e * float(zz[2])))
        elif c["type"] == "psd":
            dd = c["dim"]; S = np.zeros((dd, dd)); pos = 0
            for j in range(dd):
                for i in range(j + 1):
                    val = sb[pos] / sc; pos += 1
                    if i != j: val /= np.sqrt(2)
                    S[i, j] = S[j, i] = val
            cp = max(cp, max(0.0, -float(np.linalg.eigvalsh(S)[0])))
            Z = np.zeros((dd, dd)); pos = 0
            for j in range(dd):
                for i in range(j + 1):
                    val = zb[pos] / zc; pos += 1
                    if i != j: val /= np.sqrt(2)
                    Z[i, j] = Z[j, i] = val
            cd = max(cd, max(0.0, -float(np.linalg.eigvalsh(Z)[0])))
        start += w
    return rp <= tol and rd <= tol and gapok and cp <= tol and cd <= tol, s

def transform_soc_cuts(d, cones, extra=None):
    q = d["q"]; b = d["b"]; n = q.size; m = b.size
    Ad, Ai, Ap = d["A_data"], d["A_indices"], d["A_indptr"]
    orig_rows = [[] for _ in range(m)]
    for j in range(n):
        for k in range(int(Ap[j]), int(Ap[j+1])): orig_rows[int(Ai[k])].append((j, float(Ad[k])))
    rows = [[] for _ in range(n)]; bb = []; newcones = []; cuts = []
    start = 0; row = 0
    for c in cones:
        w = c["dim"]
        if c["type"] == "soc":
            dvec = w - 1
            dirs = []
            for i in range(dvec):
                a = np.zeros(dvec); a[i] = 1; dirs.append(a)
                a = np.zeros(dvec); a[i] = -1; dirs.append(a)
            if dvec <= 4:
                for mask in range(1 << dvec):
                    a = np.array([1 if (mask >> i) & 1 else -1 for i in range(dvec)], dtype=float) / np.sqrt(dvec)
                    dirs.append(a)
            if extra and start in extra:
                dirs.extend(extra[start])
            r0 = start
            for a in dirs:
                e = {j: v for j, v in orig_rows[r0]}
                for i in range(dvec):
                    for j, v in orig_rows[r0 + 1 + i]: e[j] = e.get(j, 0.0) - a[i] * v
                for j, v in e.items():
                    if v != 0: rows[j].append((row, v))
                val = b[r0] - float(a @ b[r0+1:r0+w])
                bb.append(val); newcones.append({"type": "nonnegative", "dim": 1})
                cuts.append((start, a, row)); row += 1
        else:
            for r in range(start, start + w):
                for j, v in orig_rows[r]: rows[j].append((row, v))
                bb.append(b[r]); row += 1
            newcones.append(c)
        start += w
    starts = [0]; inds = []; vals = []
    for j in range(n):
        col = {}
        for i, v in rows[j]: col[i] = col.get(i, 0.0) + v
        for i in sorted(col):
            if col[i] != 0: inds.append(i); vals.append(col[i])
        starts.append(len(inds))
    nd = {k: d[k] for k in d.files}
    nd["b"] = np.array(bb); nd["A_data"] = np.array(vals); nd["A_indices"] = np.array(inds, dtype=np.int32); nd["A_indptr"] = np.array(starts, dtype=np.int32)
    return nd, newcones, cuts

def transform_exp_cuts(d, cones, extra=None):
    q = d["q"]; b = d["b"]; n = q.size; m = b.size
    Ad, Ai, Ap = d["A_data"], d["A_indices"], d["A_indptr"]
    orig_rows = [[] for _ in range(m)]
    for j in range(n):
        for k in range(int(Ap[j]), int(Ap[j+1])): orig_rows[int(Ai[k])].append((j, float(Ad[k])))
    rows = [[] for _ in range(n)]; bb = []; newcones = []; cuts = []
    start = 0; row = 0
    for c in cones:
        w = c["dim"]
        if c["type"] == "exponential":
            rs = list(np.linspace(-8, 8, 33))
            if extra and start in extra: rs.extend(extra[start])
            e0 = {j: v for j, v in orig_rows[start]}
            e1 = {j: v for j, v in orig_rows[start+1]}
            e2 = {j: v for j, v in orig_rows[start+2]}
            for r in rs:
                er = float(np.exp(r)); a = er; bv = er * (1 - r)
                e = {j: v for j, v in e2.items()}
                for j, v in e0.items(): e[j] = e.get(j, 0.0) - a * v
                for j, v in e1.items(): e[j] = e.get(j, 0.0) - bv * v
                for j, v in e.items():
                    if v != 0: rows[j].append((row, v))
                bb.append(b[start+2] - a * b[start] - bv * b[start+1])
                newcones.append({"type": "nonnegative", "dim": 1})
                cuts.append((start, r, row)); row += 1
        else:
            for r in range(start, start + w):
                for j, v in orig_rows[r]: rows[j].append((row, v))
                bb.append(b[r]); row += 1
            newcones.append(c)
        start += w
    starts = [0]; inds = []; vals = []
    for j in range(n):
        col = {}
        for i, v in rows[j]: col[i] = col.get(i, 0.0) + v
        for i in sorted(col):
            if col[i] != 0: inds.append(i); vals.append(col[i])
        starts.append(len(inds))
    nd = {k: d[k] for k in d.files}
    nd["b"] = np.array(bb); nd["A_data"] = np.array(vals); nd["A_indices"] = np.array(inds, dtype=np.int32); nd["A_indptr"] = np.array(starts, dtype=np.int32)
    return nd, newcones, cuts

def transform_psd_cuts(d, cones, extra=None):
    q = d["q"]; b = d["b"]; n = q.size; m = b.size
    Ad, Ai, Ap = d["A_data"], d["A_indices"], d["A_indptr"]
    orig_rows = [[] for _ in range(m)]
    for j in range(n):
        for k in range(int(Ap[j]), int(Ap[j+1])): orig_rows[int(Ai[k])].append((j, float(Ad[k])))
    rows = [[] for _ in range(n)]; bb = []; newcones = []; cuts = []
    start = 0; row = 0
    for c in cones:
        w = c["dim"]
        if c["type"] == "psd":
            dd = w; vecs = []
            for i in range(dd):
                v = np.zeros(dd); v[i] = 1; vecs.append(v)
            v = np.ones(dd) / np.sqrt(dd); vecs.append(v)
            if extra and start in extra: vecs.extend(extra[start])
            for v in vecs:
                coeff = {}
                pos = 0
                for j in range(dd):
                    for i in range(j + 1):
                        cf = v[i] * v[j] * (1 if i == j else np.sqrt(2))
                        if cf != 0: coeff[start + pos] = cf
                        pos += 1
                e = {}
                val = 0.0
                for r, cf in coeff.items():
                    val += cf * b[r]
                    for j, vv in orig_rows[r]: e[j] = e.get(j, 0.0) + cf * vv
                for j, vv in e.items():
                    if vv != 0: rows[j].append((row, vv))
                bb.append(val); newcones.append({"type": "nonnegative", "dim": 1})
                cuts.append((start, v, row)); row += 1
        else:
            for r in range(start, start + w):
                for j, vv in orig_rows[r]: rows[j].append((row, vv))
                bb.append(b[r]); row += 1
            newcones.append(c)
        start += w
    starts = [0]; inds = []; vals = []
    for j in range(n):
        col = {}
        for i, v in rows[j]: col[i] = col.get(i, 0.0) + v
        for i in sorted(col):
            if col[i] != 0: inds.append(i); vals.append(col[i])
        starts.append(len(inds))
    nd = {k: d[k] for k in d.files}
    nd["b"] = np.array(bb); nd["A_data"] = np.array(vals); nd["A_indices"] = np.array(inds, dtype=np.int32); nd["A_indptr"] = np.array(starts, dtype=np.int32)
    return nd, newcones, cuts

def solve_scs_soc(d, cones):
    q = d["q"]; b = d["b"]; n = q.size; m = b.size
    if n > 300: return None
    A = np.zeros((m, n))
    cols = np.repeat(np.arange(n), np.diff(d["A_indptr"]))
    np.add.at(A, (d["A_indices"], cols), d["A_data"])
    erows = []; crows = []; conesC = []; start = 0
    for c in cones:
        w = c["dim"]
        if c["type"] == "zero": erows.extend(range(start, start + w))
        else:
            crows.extend(range(start, start + w)); conesC.append(c)
        start += w
    mc = len(crows)
    if erows:
        E = A[erows]; be = b[erows]
        U, s, Vh = np.linalg.svd(E, full_matrices=True)
        rank = int(np.sum(s > 1e-9 * max(1.0, s[0] if s.size else 1.0)))
        if rank < E.shape[0]:
            if np.max(np.abs(U[:, rank:].T @ be)) > 1e-7: return None
        xp = np.linalg.lstsq(E, be, rcond=None)[0]
        N = Vh[rank:].T
    else:
        xp = np.zeros(n); N = np.eye(n)
    nr = N.shape[1]
    if nr == 0 or nr > 300: return None
    Ac = A[crows]; bc = b[crows]
    Ar = Ac @ N; br = bc - Ac @ xp; qr = N.T @ q
    M = np.eye(nr) + Ar.T @ Ar
    K = np.zeros((nr + 1, nr + 1))
    K[:nr, :nr] = M; K[:nr, nr] = qr - Ar.T @ br; K[nr, :nr] = -(qr + Ar.T @ br); K[nr, nr] = 1 + br @ br
    try: Ki = np.linalg.inv(K)
    except Exception: return None
    z = np.zeros(mc); s = np.zeros(mc); st = 0
    for c in conesC:
        w = c["dim"]
        if c["type"] == "soc": z[st] = s[st] = 1
        elif c["type"] == "nonnegative": z[st:st+w] = s[st:st+w] = 1
        st += w
    u = np.concatenate([np.zeros(nr), z, [1.0]])
    v = np.concatenate([np.zeros(nr), s, [1.0]])
    def proj(zz):
        zp = zz.copy(); st = 0
        for c in conesC:
            w = c["dim"]; t = zp[st:st+w]
            if c["type"] == "nonnegative": t[:] = np.maximum(t, 0)
            elif c["type"] == "soc":
                tt = t[0]; xx = t[1:]; nx = np.linalg.norm(xx)
                if tt <= -nx: t[:] = 0
                elif tt < nx:
                    a = (tt + nx) / 2; t[0] = a; t[1:] = a * xx / nx if nx > 0 else 0
            st += w
        return zp
    P0 = (d["P_data"], d["P_indices"], d["P_indptr"])
    A0 = (d["A_data"], d["A_indices"], d["A_indptr"])
    for it in range(6000):
        u_old = u
        rhs = u + v
        rx = rhs[:nr]; rz = rhs[nr:nr+mc]; rt = rhs[-1]
        rr = np.concatenate([rx - Ar.T @ rz, [rt + br @ rz]])
        sol = Ki @ rr
        xt = sol[:nr]; tt = sol[-1]; zt = rz + Ar @ xt - br * tt
        tilde = np.concatenate([xt, zt, [tt]])
        un = tilde - v
        zn = proj(un[nr:nr+mc]); tn = max(un[-1], 0.0)
        un = np.concatenate([un[:nr], zn, [tn]])
        v = v - tilde + un; u = 1.6 * un - 0.6 * u_old
        if it % 500 == 499:
            tau = u[-1]
            if tau > 1e-8:
                y = u[:nr] / tau; x = xp + N @ y
                zc = u[nr:nr+mc] / tau
                z = np.zeros(m); z[crows] = zc
                if erows:
                    rhsz = -(q + A[crows].T @ zc)
                    ze = np.linalg.lstsq(A[erows].T, rhsz, rcond=None)[0]
                    z[erows] = ze
                ok, s = check_general(P0, q, A0, b, cones, x, z, tol=8e-9)
                if ok: return x, s, z
    return None

def transform_soc2(d, cones):
    q = d["q"]; b = d["b"]; n = q.size
    Ad, Ai, Ap = d["A_data"], d["A_indices"], d["A_indptr"]
    m = b.size
    orig_rows = [[] for _ in range(m)]
    for j in range(n):
        for k in range(int(Ap[j]), int(Ap[j+1])):
            orig_rows[int(Ai[k])].append((j, float(Ad[k])))
    rows = [[] for _ in range(n)]; bb = []; newcones = []; mapping = []
    start = 0; row = 0
    for c in cones:
        w = c["dim"]
        if c["type"] == "soc" and w == 2:
            r0 = start; r1 = start + 1
            p = row; mrow = row + 1
            e0 = {j: v for j, v in orig_rows[r0]}
            e1 = {j: v for j, v in orig_rows[r1]}
            for j in set(e0) | set(e1):
                v = e0.get(j, 0.0) - e1.get(j, 0.0)
                if v != 0: rows[j].append((p, v))
                v = e0.get(j, 0.0) + e1.get(j, 0.0)
                if v != 0: rows[j].append((mrow, v))
            bb += [b[r0] - b[r1], b[r0] + b[r1]]
            newcones.append({"type": "nonnegative", "dim": 2})
            mapping.append(("soc2", p, mrow))
            row += 2
        else:
            for r in range(start, start + w):
                for j, v in orig_rows[r]: rows[j].append((row, v))
                bb.append(b[r]); mapping.append(("direct", row))
                row += 1
            newcones.append(c)
        start += w
    starts = [0]; inds = []; vals = []
    for j in range(n):
        col = {}
        for i, v in rows[j]: col[i] = col.get(i, 0.0) + v
        for i in sorted(col):
            if col[i] != 0: inds.append(i); vals.append(col[i])
        starts.append(len(inds))
    nd = {k: d[k] for k in d.files}
    nd["b"] = np.array(bb); nd["A_data"] = np.array(vals); nd["A_indices"] = np.array(inds, dtype=np.int32); nd["A_indptr"] = np.array(starts, dtype=np.int32)
    return nd, newcones, mapping

def recover_soc2(znew, mapping, m):
    z = np.zeros(m)
    for item in mapping:
        if item[0] == "direct": z[len([x for x in mapping if x[0]=='direct' and mapping.index(x)<mapping.index(item)])] = znew[item[1]]
    return z

def write_result(path, status, x=None, s=None, z=None):
    meta = {"schema_version": 1, "status": status}
    arrays = {"metadata_json": np.array(json.dumps(meta))}
    if x is not None: arrays["x"] = np.asarray(x, dtype=np.float64)
    if s is not None: arrays["s"] = np.asarray(s, dtype=np.float64)
    if z is not None: arrays["z"] = np.asarray(z, dtype=np.float64)
    np.savez(path, **arrays)

def solve_highs(d, cones, tl=None):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
    import _core_shim
    _core = _core_shim.core
    q = d["q"]; b = d["b"]; n = q.size; m = b.size
    Ad, Ai, Ap = sum_csc(d["A_data"], d["A_indices"], d["A_indptr"], m)
    Pd, Pi, Pp = sum_csc(d["P_data"], d["P_indices"], d["P_indptr"], n)
    h = _core._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "on")
    h.setOptionValue("time_limit", tl if tl is not None else 0.85)
    for opt in ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "ipm_optimality_tolerance"):
        try: h.setOptionValue(opt, 1e-10)
        except Exception: pass
    lp = _core.HighsLp()
    lp.num_col_ = n; lp.num_row_ = m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -1e30); lp.col_upper_ = np.full(n, 1e30)
    rl = np.full(m, -1e30); ru = np.full(m, 1e30)
    start = 0
    for c in cones:
        w = c["dim"]
        for r in range(start, start + w):
            if c["type"] == "zero":
                rl[r] = ru[r] = b[r]
            else:
                ru[r] = b[r]
        start += w
    lp.row_lower_ = rl; lp.row_upper_ = ru
    lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = m
    lp.a_matrix_.format_ = _core.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Ap; lp.a_matrix_.index_ = Ai; lp.a_matrix_.value_ = Ad
    if h.passModel(lp) != _core.HighsStatus.kOk:
        return None
    if Pd.size:
        cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Pp))
        mask = Pi >= cols
        lc = cols[mask]; li = Pi[mask]; lv = Pd[mask]
        order = np.lexsort((li, lc))
        lc = lc[order]; li = li[order]; lv = lv[order]
        key = lc * np.int64(n) + li
        uniq = np.ones(key.size, dtype=bool); uniq[1:] = key[1:] != key[:-1]
        st = np.flatnonzero(uniq)
        vals = np.add.reduceat(lv, st); inds = li[st]; ccol = lc[st]
        cnt = np.bincount(ccol, minlength=n)
        starts = np.zeros(n + 1, dtype=np.int32); starts[1:] = np.cumsum(cnt)
        hes = _core.HighsHessian()
        hes.dim_ = n; hes.format_ = _core.HessianFormat.kTriangular
        hes.start_ = starts; hes.index_ = inds.astype(np.int32); hes.value_ = vals
        if h.passHessian(hes) != _core.HighsStatus.kOk:
            return None
    if h.run() != _core.HighsStatus.kOk:
        return None
    status = h.getModelStatus()
    sol = h.getSolution()
    acceptable = status == _core.HighsModelStatus.kOptimal
    if status in (_core.HighsModelStatus.kTimeLimit, _core.HighsModelStatus.kIterationLimit, _core.HighsModelStatus.kObjectiveBound):
        acceptable = bool(sol.value_valid and sol.dual_valid)
    if not acceptable:
        return None
    x = np.array(sol.col_value, dtype=np.float64)
    z = -np.array(sol.row_dual, dtype=np.float64)
    ok, s = check_linear_qp((Pd, Pi, Pp), q, (Ad, Ai, Ap), b, cones, x, z)
    if not ok:
        if n <= 1000:
            pol = polish_linear_qp(d, cones, x)
            if pol is not None:
                return pol
        return None
    return x, s, z

def polish_linear_qp(d, cones, x0):
    q = d["q"]; b = d["b"]; n = q.size; m = b.size
    Ad, Ai, Ap = sum_csc(d["A_data"], d["A_indices"], d["A_indptr"], m)
    Pd, Pi, Pp = sum_csc(d["P_data"], d["P_indices"], d["P_indptr"], n)
    Ax = csc_mv(Ad, Ai, Ap, x0, m)
    s0 = b - Ax
    bs = max(1.0, inf_norm(b))
    rowtype = np.zeros(m, dtype=int)
    start = 0
    for c in cones:
        w = c["dim"]
        rowtype[start:start+w] = 0 if c["type"] == "zero" else 1
        start += w
    act = [r for r in range(m) if rowtype[r] == 0 or s0[r] <= 1e-4 * bs]
    for attempt in range(6):
        ma = len(act)
        if n + ma <= 2500:
            P = np.zeros((n, n))
            cols = np.repeat(np.arange(n), np.diff(Pp))
            np.add.at(P, (Pi, cols), Pd)
            P = 0.5 * (P + P.T)
            Aa = np.zeros((ma, n))
            rowmap = {r: i for i, r in enumerate(act)}
            for j in range(n):
                for k in range(int(Ap[j]), int(Ap[j+1])):
                    r = int(Ai[k])
                    if r in rowmap: Aa[rowmap[r], j] += Ad[k]
            K = np.zeros((n + ma, n + ma))
            K[:n, :n] = P; K[:n, n:] = Aa.T; K[n:, :n] = Aa
            rhs = np.concatenate([-q, b[act]])
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                try: sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
                except Exception: return None
        else:
            if n > 10000: return None
            try:
                import scipy.sparse as sp
                from scipy.sparse.linalg import splu
                Pm = sp.csc_matrix((Pd, Pi, Pp), shape=(n, n))
                rr = []; cc = []; vv = []
                rowmap = {r: i for i, r in enumerate(act)}
                for j in range(n):
                    for k in range(int(Ap[j]), int(Ap[j+1])):
                        r = int(Ai[k])
                        if r in rowmap:
                            rr.append(rowmap[r]); cc.append(j); vv.append(Ad[k])
                Aa = sp.csc_matrix((vv, (rr, cc)), shape=(ma, n))
                K = sp.bmat([[Pm, Aa.T], [Aa, None]], format="csc")
                rhs = np.concatenate([-q, b[act]])
                lu = splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
                sol = lu.solve(rhs)
            except Exception:
                return None
        x = sol[:n]; z = np.zeros(m); z[act] = sol[n:]
        ok, s = check_linear_qp((Pd, Pi, Pp), q, (Ad, Ai, Ap), b, cones, x, z)
        if ok: return x, s, z
        neg = [r for r in act if rowtype[r] == 1 and z[r] < -1e-7]
        if neg:
            act = [r for r in act if r not in set(neg)]
            continue
        actset = set(act)
        pos = [r for r in range(m) if rowtype[r] == 1 and r not in actset and s[r] < -1e-7]
        if pos:
            pos.sort(key=lambda r: s[r])
            act.extend(pos[:max(1, len(pos)//2)])
            continue
        break
    return None

def check_primal_cert(A, b, cones, z, tol=8e-9):
    Ad, Ai, Ap = A
    m = b.size; n = Ap.size - 1
    nz = inf_norm(z); nb = inf_norm(b)
    if nz <= 0 or nb <= 0: return False
    amax = float(np.max(np.abs(Ad))) if Ad.size else 1.0
    if amax == 0: amax = 1.0
    zh = z / nz
    beta = -float((b / nb) @ zh)
    if beta <= tol: return False
    Atz = csc_tmv(Ad, Ai, Ap, zh, n)
    if inf_norm(Atz) / amax / beta > tol: return False
    start = 0
    for c in cones:
        w = c["dim"]; zb = zh[start:start+w]
        if c["type"] == "nonnegative" and np.min(zb) < -tol: return False
        start += w
    return True

def check_dual_cert(P, A, q, cones, x, tol=8e-9):
    Pd, Pi, Pp = P; Ad, Ai, Ap = A
    n = q.size; m = Ap.size and (Ai.max() + 1 if Ai.size else 0)
    m = b_size_global
    nx = inf_norm(x); nq = inf_norm(q)
    if nx <= 0 or nq <= 0: return False
    amax = float(np.max(np.abs(Ad))) if Ad.size else 1.0
    pmax = float(np.max(np.abs(Pd))) if Pd.size else 1.0
    if amax == 0: amax = 1.0
    if pmax == 0: pmax = 1.0
    xh = x / nx
    alpha = -float((q / nq) @ xh)
    if alpha <= tol: return False
    Px = csc_mv(Pd, Pi, Pp, xh, n)
    if inf_norm(Px) / pmax / alpha > tol: return False
    Ax = csc_mv(Ad, Ai, Ap, xh, m)
    s = -Ax * nx
    ss = s / nx / amax
    start = 0
    for c in cones:
        w = c["dim"]; sb = ss[start:start+w]
        if c["type"] == "zero":
            if inf_norm(sb) > tol: return False
        else:
            if np.min(sb) < -tol: return False
        start += w
    return True

b_size_global = 0

def solve_cert_highs(d, cones, primal):
    global b_size_global
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
    import _core_shim
    _core = _core_shim.core
    q = d["q"]; b = d["b"]; n = q.size; m = b.size
    b_size_global = m
    Ad, Ai, Ap = sum_csc(d["A_data"], d["A_indices"], d["A_indptr"], m)
    Pd, Pi, Pp = sum_csc(d["P_data"], d["P_indices"], d["P_indptr"], n)
    h = _core._Highs()
    h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "on"); h.setOptionValue("time_limit", 0.15)
    for opt in ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "ipm_optimality_tolerance"):
        try: h.setOptionValue(opt, 1e-10)
        except Exception: pass
    lp = _core.HighsLp()
    if primal:
        lp.num_col_ = m; lp.num_row_ = n
        lp.col_cost_ = b
        lb = np.empty(m); ub = np.empty(m); start = 0
        for c in cones:
            w = c["dim"]
            if c["type"] == "zero": lb[start:start+w] = -1; ub[start:start+w] = 1
            else: lb[start:start+w] = 0; ub[start:start+w] = 1
            start += w
        lp.col_lower_ = lb; lp.col_upper_ = ub
        lp.row_lower_ = np.zeros(n); lp.row_upper_ = np.zeros(n)
        cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Ap))
        order = np.argsort(Ai, kind="stable")
        tAi = cols[order]; tAv = Ad[order]
        tAp = np.zeros(m + 1, dtype=np.int32)
        cnt = np.bincount(Ai, minlength=m)
        tAp[1:] = np.cumsum(cnt)
        lp.a_matrix_.num_col_ = m; lp.a_matrix_.num_row_ = n
        lp.a_matrix_.format_ = _core.MatrixFormat.kColwise
        lp.a_matrix_.start_ = tAp; lp.a_matrix_.index_ = tAi.astype(np.int32); lp.a_matrix_.value_ = tAv
    else:
        rows = []
        rl = []; ru = []
        start = 0
        for c in cones:
            w = c["dim"]
            for r in range(start, start + w):
                rows.append(r)
                if c["type"] == "zero": rl.append(0.0); ru.append(0.0)
                else: rl.append(-1e30); ru.append(0.0)
            start += w
        mc = len(rows)
        lp.num_col_ = n; lp.num_row_ = mc + n
        lp.col_cost_ = q
        lp.col_lower_ = np.full(n, -1.0); lp.col_upper_ = np.full(n, 1.0)
        lp.row_lower_ = np.array(rl + [0.0] * n); lp.row_upper_ = np.array(ru + [0.0] * n)
        starts = [0]; inds = []; vals = []
        rowmap = {r: i for i, r in enumerate(rows)}
        for j in range(n):
            col = {}
            for k in range(int(Ap[j]), int(Ap[j+1])):
                r = int(Ai[k])
                if r in rowmap:
                    i = rowmap[r]; col[i] = col.get(i, 0.0) + float(Ad[k])
            for k in range(int(Pp[j]), int(Pp[j+1])):
                i = mc + int(Pi[k]); col[i] = col.get(i, 0.0) + float(Pd[k])
            for i in sorted(col):
                if col[i] != 0.0: inds.append(i); vals.append(col[i])
            starts.append(len(inds))
        lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = mc + n
        lp.a_matrix_.format_ = _core.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(starts, dtype=np.int32)
        lp.a_matrix_.index_ = np.array(inds, dtype=np.int32)
        lp.a_matrix_.value_ = np.array(vals, dtype=np.float64)
    if h.passModel(lp) != _core.HighsStatus.kOk: return None
    if h.run() != _core.HighsStatus.kOk: return None
    sol = h.getSolution()
    st = h.getModelStatus()
    okst = st == _core.HighsModelStatus.kOptimal or (st in (_core.HighsModelStatus.kTimeLimit, _core.HighsModelStatus.kIterationLimit) and sol.value_valid)
    if not okst or h.getObjectiveValue() >= -1e-7: return None
    if primal:
        z = np.array(sol.col_value, dtype=np.float64)
        if check_primal_cert((Ad, Ai, Ap), b, cones, z): return "primal_infeasible", None, None, z
    else:
        x = np.array(sol.col_value, dtype=np.float64)
        if check_dual_cert((Pd, Pi, Pp), (Ad, Ai, Ap), q, cones, x):
            Ax = csc_mv(Ad, Ai, Ap, x, m)
            return "dual_infeasible", x, -Ax, None
    return None

def main():
    inpath = outpath = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--input" and i + 1 < len(args): inpath = args[i+1]
        elif a == "--output" and i + 1 < len(args): outpath = args[i+1]
    if not inpath or not outpath:
        sys.exit(1)
    global OUT_PATH_GLOBAL
    OUT_PATH_GLOBAL = outpath
    try:
        meta = read_meta(inpath)
        cones = meta["cones"]
        use_highs = all(c["type"] in ("zero", "nonnegative") for c in cones)
        if use_highs:
            try:
                if solve_lp_fast(inpath, cones, outpath): return
            except Exception:
                pass
            if False:
                try:
                    if solve_qp_fast(inpath, cones, outpath): return
                except Exception:
                    pass
        soc2 = all(c["type"] in ("zero", "nonnegative", "soc") and (c["type"] != "soc" or c["dim"] == 2) for c in cones)
        if use_highs or soc2:
            global np
            import numpy as np
            d = np.load(inpath, allow_pickle=False)
            q = d["q"]; b = d["b"]
        if soc2 and not use_highs:
            dt, ct, mapping = transform_soc2(d, cones)
            res = solve_highs(dt, ct)
            if res is not None:
                x, _, znew = res
                z = np.zeros(b.size); oi = 0
                for item in mapping:
                    if item[0] == "direct":
                        z[oi] = znew[item[1]]; oi += 1
                    else:
                        zp = znew[item[1]]; zm = znew[item[2]]
                        z[oi] = zp + zm; z[oi+1] = -zp + zm; oi += 2
                ok, s = check_soc2((d["P_data"], d["P_indices"], d["P_indptr"]), q, (d["A_data"], d["A_indices"], d["A_indptr"]), b, cones, x, z)
                if ok:
                    write_result(outpath, "solved", x, s, z); return
        if False and not use_highs and all(c["type"] in ("zero", "nonnegative", "soc") for c in cones):
            extra = {}
            for it in range(3):
                dt, ct, cuts = transform_soc_cuts(d, cones, extra)
                res = solve_highs(dt, ct, tl=0.08)
                if res is None: break
                x, _, znew = res
                z = np.zeros(b.size)
                for st, a, cr in cuts:
                    z[st] += znew[cr]
                    z[st+1:st+1+len(a)] -= a * znew[cr]
                ok, s = check_general((d["P_data"], d["P_indices"], d["P_indptr"]), q, (d["A_data"], d["A_indices"], d["A_indptr"]), b, cones, x, z)
                if ok:
                    write_result(outpath, "solved", x, s, z); return
                start = 0; added = False
                for c in cones:
                    if c["type"] == "soc":
                        sb = s[start:start+c["dim"]]
                        nv = float(np.linalg.norm(sb[1:]))
                        if nv > sb[0] + 1e-7 and nv > 0:
                            extra.setdefault(start, []).append(sb[1:] / nv); added = True
                    start += c["dim"]
                if not added: break
        if False and not use_highs and all(c["type"] in ("zero", "nonnegative", "exponential") for c in cones):
            extra = {}
            for it in range(5):
                dt, ct, cuts = transform_exp_cuts(d, cones, extra)
                res = solve_highs(dt, ct, tl=0.08)
                if res is None: break
                x, _, znew = res
                z = np.zeros(b.size)
                for st, r, cr in cuts:
                    er = float(np.exp(r)); lam = znew[cr]
                    z[st] -= er * lam; z[st+1] -= er * (1 - r) * lam; z[st+2] += lam
                ok, s = check_general((d["P_data"], d["P_indices"], d["P_indptr"]), q, (d["A_data"], d["A_indices"], d["A_indptr"]), b, cones, x, z)
                if ok:
                    write_result(outpath, "solved", x, s, z); return
                start = 0; added = False
                for c in cones:
                    if c["type"] == "exponential":
                        sb = s[start:start+3]
                        if sb[1] > 1e-9 and sb[2] > 1e-9:
                            r = float(sb[0] / sb[1])
                            if sb[1] * np.exp(r) > sb[2] + 1e-7:
                                extra.setdefault(start, []).append(r); added = True
                    start += c["dim"]
                if not added: break
        if False and not use_highs and all(c["type"] in ("zero", "nonnegative", "psd") for c in cones):
            extra = {}
            for it in range(3):
                dt, ct, cuts = transform_psd_cuts(d, cones, extra)
                res = solve_highs(dt, ct, tl=0.1)
                if res is None: break
                x, _, znew = res
                z = np.zeros(b.size)
                for st, v, cr in cuts:
                    dd = len(v); pos = 0; lam = znew[cr]
                    for j in range(dd):
                        for i in range(j + 1):
                            z[st + pos] += lam * v[i] * v[j] * (1 if i == j else np.sqrt(2)); pos += 1
                ok, s = check_general((d["P_data"], d["P_indices"], d["P_indptr"]), q, (d["A_data"], d["A_indices"], d["A_indptr"]), b, cones, x, z)
                if ok:
                    write_result(outpath, "solved", x, s, z); return
                start = 0; added = False
                for c in cones:
                    if c["type"] == "psd":
                        dd = c["dim"]; S = np.zeros((dd, dd)); pos = 0
                        for j in range(dd):
                            for i in range(j + 1):
                                val = s[start + pos]; pos += 1
                                if i != j: val /= np.sqrt(2)
                                S[i, j] = S[j, i] = val
                        w, V = np.linalg.eigh(S)
                        if w[0] < -1e-7:
                            extra.setdefault(start, []).append(V[:, 0]); added = True
                    start += c["dim"] * (c["dim"] + 1) // 2 if c["type"] == "psd" else c["dim"]
                if not added: break
        if False and not use_highs and all(c["type"] in ("zero", "nonnegative", "soc") for c in cones) and d["P_data"].size == 0:
            res = solve_scs_soc(d, cones)
            if res is not None:
                x, s, z = res
                write_result(outpath, "solved", x, s, z); return
        if use_highs:
            res = solve_highs(d, cones)
            if res is not None:
                x, s, z = res
                write_result(outpath, "solved", x, s, z)
                return
            for primal in (True, False):
                cert = solve_cert_highs(d, cones, primal)
                if cert is not None:
                    status, x, s, z = cert
                    write_result(outpath, status, x, s, z)
                    return
    except Exception:
        pass
    solver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver")
    os.execv(solver, [solver, "--input", inpath, "--output", outpath])

if __name__ == "__main__":
    main()
