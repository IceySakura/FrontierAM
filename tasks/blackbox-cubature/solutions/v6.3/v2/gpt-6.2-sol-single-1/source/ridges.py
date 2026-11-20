"""Inference and integration of low-dimensional ridge structure."""
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import CubicSpline
from scipy.linalg import svd
from scipy.optimize import least_squares, minimize_scalar

from common import AxisQuad, Model, NoBudget, legmat, multiindices, residual, solve_ls
from periods import peak_moment, periodic_peak_profile
from projection import Projection
from structure import kink_lin, line_model, _steps_profile


def make_line(orc, kind=0, endpoints=None, size=65):
    d = orc.d
    if endpoints is not None:
        a, b = np.asarray(endpoints)
        center, direction = (a+b)*.5, b-a
    else:
        center = .5 + orc.rng.uniform(-.047,.047,d)
        sg = np.ones(d) if kind == 0 else orc.rng.choice([-1.,1.],d)
        if kind == 0:
            sg *= orc.rng.choice([-1.,1.])
        direction = orc.rng.uniform(.73,.83,d)*sg
    t = np.linspace(-.5,.5,size)
    p = center + t[:,None]*direction
    return center, direction, t, orc.sample(p)


def gradients(orc, x, scale, order=4, step=0.00018):
    x = np.asarray(x)
    d = orc.d
    if order == 4:
        mult = np.array([-2,-1,1,2])
        weights = np.array([1.,-8.,8.,-1.])/12.
    else:
        mult = np.array([-1,1])
        weights = np.array([-.5,.5])
    h = np.minimum(np.minimum(x,1.-x)*.45, np.broadcast_to(step,x.shape))
    h = np.maximum(h,1e-9)
    cloud = x[:,None,None,:] + (h[:,:,None,None] * mult[None,None,:,None] *
                               np.eye(d)[None,:,None,:])
    vals = orc.sample(cloud.reshape(-1,d)).reshape(len(x),d,len(mult))/scale
    return (vals @ weights)/h


def _periodic_profile(orc, project, fn, grid, values):
    span = np.ptp(values)
    if span < 1e-14 or orc.left < 600:
        return None
    flips = int(np.sum(np.diff(np.sign(np.diff(values))) != 0))
    if flips < 5:
        return None
    spline = CubicSpline(grid,values)
    ds = grid[1]-grid[0]
    s = np.linspace(0,1,83)
    def loss(a):
        u = s*(1-a)
        return np.mean(((spline(u+a)-spline(u))/span)**2)
    lag = np.linspace(2.7*ds,.409,156)
    obj = np.array([loss(a) for a in lag])
    q = np.where((obj[1:-1]<obj[:-2]) & (obj[1:-1]<obj[2:]) & (obj[1:-1]<.030))[0]+1
    if not len(q):
        return None
    qq = q[np.argsort(obj[q])[:4]]
    use = sorted([p for p in qq if obj[p] <= max(.00012,6.*min(obj[qq]))])[:2]
    for j in use:
        if orc.left < 310:
            break
        ans = minimize_scalar(loss,bounds=(lag[j-1],lag[j+1]),method="bounded",
                              options={"xatol":2e-8})
        period = float(ans.x)
        nodes = grid[(grid>.006)&(grid<1.-period-2*ds)]
        if len(nodes) < 16:
            continue
        slopes = np.abs(spline(nodes,1))
        rank = np.argsort(slopes*orc.rng.uniform(.63,1.,len(slopes)))
        t = nodes[rank[-12:]]
        train = fn(t)
        err = np.inf
        for _ in range(5):
            h = max(2e-7,period*2e-5)
            vals = fn(np.r_[t+period,t+period+h,t+period-h]).reshape(3,-1)
            delta = vals[0]-train
            err = np.sqrt(np.mean(delta**2))
            if err < 2e-12*span:
                break
            der = (vals[1]-vals[2])/(2*h)
            nu = np.dot(der,der)
            if nu < 1e-20:
                break
            shift = -np.dot(der,delta)/nu
            if abs(shift) < 1e-13 or abs(shift) > 1.5*ds:
                break
            period += shift
            if not 2*ds < period < .45:
                break
        if err > 1.2e-8*span:
            continue
        ps = (np.arange(26)+orc.rng.random(26))/26 * (1.-period)
        vs = fn(np.r_[ps,ps+period]).reshape(2,-1)
        if np.max(abs(vs[0]-vs[1])) > 2e-8*span:
            continue
        start = .5-period*.5
        n = 64
        while n > orc.left and n>=16:
            n//=2
        if n < 16:
            break
        nodes = np.arange(n)*period/n
        p = fn(start+nodes)
        prev = None
        while True:
            omega = 2*np.pi*np.fft.fftfreq(n,d=period/n)
            weights = np.exp(1j*omega*(.5-start))*np.prod(
                np.sinc(omega[:,None]*project.v[None,:]/(2*np.pi)),axis=1)
            est = float(np.real((np.fft.fft(p)/n)@weights))
            if prev is not None and abs(est-prev) < 2e-12*max(abs(est),span*1e-5):
                return est
            if n>orc.left or n>=1024:
                return est
            prev = est
            q = fn(start+nodes+period/(2*n))
            nodes = np.column_stack((nodes,nodes+period/(2*n))).ravel()
            p = np.column_stack((p,q)).ravel()
            n *= 2
    return None


def rank_one(orc, x, y, scale):
    """Check an arbitrary scalar function of a single linear projection."""
    d = orc.d
    if d > 10 or orc.left < 4*d*3+95 or np.ptp(y) < 1e-13:
        return None
    mask = np.flatnonzero(np.all((x > .025) & (x < .975),axis=1))
    if len(mask) < 8:
        return None
    ys = y[mask]
    ids = mask[np.argsort(ys)[np.rint(np.array([.13, .48, .79, .93])*(len(mask)-1)).astype(int)]]
    g = gradients(orc, x[ids], scale, step=.000065)
    _, sig, v = svd(g, full_matrices=False, check_finite=False)
    if sig[0]>0 and 2e-8*sig[0] < sig[1] < 1e-4*sig[0] and orc.left>330:
        g = gradients(orc,x[ids],scale,step=.0000105)
        _,sig,v=svd(g,full_matrices=False,check_finite=False)
    if sig[0] < 5e-8*np.ptp(y) or sig[1] > sig[0]*2e-8:
        return None
    w = v[0].copy()
    w[np.abs(w) < 3e-11] = 0.
    project = Projection(w)
    order = np.argsort(y)
    choose = np.unique(np.r_[order[np.linspace(0,len(y)-1,32).astype(int)],
                             np.linspace(0,len(y)-1,25).astype(int)])
    t = project.param(x[choose])
    val = orc.sample(project.points(t))/scale
    delta = y[choose]-val
    mx = np.max(np.abs(delta))
    if 2e-9 < mx < 1e-5 and orc.left>=310:
        h = 8e-7
        l,r = np.minimum(1.,t+h),np.maximum(0.,t-h)
        uv = orc.sample(project.points(np.r_[l,r]))/scale
        lp,rp=uv[:len(t)],uv[len(t):]
        g=(lp-rp)/(l-r)
        a=(x[choose]-.5)*g[:,None]
        s=np.sign(project.v)
        shift=solve_ls(np.vstack((a,max(1.,np.linalg.norm(a))*s)),np.r_[delta,0.])
        v1=project.v+shift
        if np.max(abs(shift))<5e-7 and np.all(np.sign(v1[s!=0])==s[s!=0]):
            project=Projection(v1)
            t=project.param(x[choose])
            val=orc.sample(project.points(t))/scale
    if np.max(np.abs(val-y[choose])) > 2e-9:
        return None
    # A second check uses a slice with no common vertices with the first.
    audit = orc.new_sobol(24)
    pp = project.param(audit)
    yy = orc.sample(np.vstack((audit,project.points(pp))))/scale
    if np.max(np.abs(yy[:len(audit)]-yy[len(audit):])) > 3e-9:
        return None
    def fn(s):
        return orc.sample(project.points(s))/scale
    n = min(97,max(35,orc.left//6))
    if orc.left < n + 45:
        return None
    grid = np.linspace(0,1,n)
    vals = fn(grid)
    if orc.left>=90 and np.count_nonzero(np.diff(np.sign(np.diff(vals))))>=5:
        mdl = periodic_peak_profile(grid,vals)
        if mdl is not None and mdl[0] < 1e-7:
            _,b,A,k,omega,phase=mdl
            r=orc.rng.random(31)
            roots=(np.arange(-1,2+int(omega/(2*np.pi)))*2*np.pi-phase)/omega
            roots=roots[(roots>0)&(roots<1)][:27]
            r=np.r_[r,roots]
            e=np.max(np.abs(fn(r)-(b+A*np.exp(k*(np.cos(omega*r+phase)-1)))))
            if e < 2.5e-8*max(1e-10,np.ptp(vals)):
                return b+A*peak_moment(k,omega*project.v,phase+omega*.5)
    br = [a[0] for a in kink_lin(grid,vals)]
    if br:
        feature = np.column_stack([np.ones(n), grid, np.maximum(grid[:,None]-br,0.)])
        coeff = solve_ls(feature,vals)
        if np.max(np.abs(feature@coeff-vals)) < 2e-10:
            # The orientation came from gradients. Independently verify
            # the piecewise linear description, including the ends.
            vt = project.param(x)
            pf = coeff[0]+coeff[1]*vt+np.maximum(vt[:,None]-br,0.)@coeff[2:]
            if np.max(np.abs(pf-y)) < 2e-9:
                m = [project.moment(1.-b,1) for b in br]
                return float(np.dot(np.r_[1.,.5,m],coeff))
    periodic = _periodic_profile(orc,project,fn,grid,vals)
    if periodic is not None:
        return periodic
    if not br and orc.left >= 160:
        steps, _ = _steps_profile(orc,grid,vals,fn,min(150,orc.left//4))
        br += steps
    if len(br) < 6:
        for i in range(1,n-1):
            if (abs(vals[i]) > 8.*np.median(np.abs(vals)) and
                    abs(vals[i]) >= max(abs(vals[i-1]),abs(vals[i+1]))):
                br.extend([grid[max(0,i-1)],grid[i],grid[min(n-1,i+1)]])
                if len(br) >= 6:
                    break
    bumps = int(np.sum(np.diff(np.sign(np.diff(vals))) != 0))
    sections = min(18,orc.left//44,1+bumps//4)
    if sections >= 2:
        br += list(np.linspace(0.,1.,sections+1)[1:-1])
    maxpts = min(orc.left,1450)
    estimate, err = AxisQuad(project).adaptive(fn,lambda:orc.left,br,maxpts=maxpts,tol=1e-12)
    return estimate


def _spectral_candidates(t, y):
    n = len(y)
    channels = y.reshape(n,-1)
    channels = channels - channels.mean(axis=0)
    out = []
    widths = (n//2,) if channels.shape[1] == 1 else (n//3,n//2,max(6,n//5))
    for width in widths:
        h = np.concatenate([sliding_window_view(c,width) for c in channels.T],axis=1)
        u,s,_ = svd(h, full_matrices=False, check_finite=False)
        if s[0] == 0.:
            continue
        counts = [int((s > p*s[0]).sum()) for p in (1e-12,1e-10,7e-9,1e-7,3e-6,1e-4)]
        ranks = list(dict.fromkeys(r for r in counts if 1 <= r <= 23))
        for k in ranks:
            v = u[:,:k]
            shifted = solve_ls(v[:-1],v[1:])
            eig = np.linalg.eigvals(shifted)
            roots = []
            for z in eig:
                if z.imag < -1e-9 or abs(z) < .08:
                    continue
                if z.imag < 1e-9 and z.real <= 0.:
                    continue
                a,b = np.log(abs(z))/(t[1]-t[0]), np.angle(z)/(t[1]-t[0])
                if abs(a) > 85. or b > .994*np.pi/(t[1]-t[0]) or max(abs(a),b) < .04:
                    continue
                if abs(a) < .0015 and b < .02:
                    continue
                if abs(a) < 3e-6:
                    a = 0.
                if abs(b) < 1e-6:
                    b = 0.
                roots.append(complex(a,b))
            roots = np.array(sorted(roots,key=lambda p: (p.imag,p.real)))
            if 1 <= len(roots) <= 10:
                out.append(roots)
    return out


def _exp_basis(t, roots, deg=2):
    n = len(t)
    m = 1+deg+len(roots)+int(np.sum(np.imag(roots) > 0.))
    ret = np.empty((n,m))
    ret[:,0] = 1.
    for j in range(1,deg+1):
        ret[:,j] = t**j
    j = deg+1
    for s in roots:
        ra = np.exp(np.clip(t*s.real,-200.,200.))
        if s.imag:
            ret[:,j] = ra*np.cos(t*s.imag)
            ret[:,j+1] = ra*np.sin(t*s.imag)
            j += 2
        else:
            ret[:,j] = ra
            j += 1
    return ret


def spectral_quality(t,y,roots):
    ref = max(float(np.std(y)),1e-290)
    best = np.inf
    for w in roots:
        if not np.any(np.imag(w)>.35):
            continue
        a = _exp_basis(t,w,2)
        c = solve_ls(a,y)
        e = np.std(a@c-y)/ref
        best = min(best,e)
    return best


def spectral(orc, line, x, y, scale, seeds=None, ensemble=False):
    """Recover multiple independent frequencies from translated line slices."""
    center, direction, t, raw = line
    yl = raw/scale
    d = x.shape[1]
    delta = np.minimum(.044 if ensemble else .015,.55*(1.-(center + np.abs(direction)*.5)))
    if np.min(delta) <= 0.:
        return None
    if ensemble:
        if orc.left < len(t)*d+110:
            return None
        pts = center + t[None,:,None]*direction+(np.eye(d)*delta)[:,None,:]
        vx = orc.sample(pts.reshape(-1,d)).reshape(d,len(t))/scale
        seeds = _spectral_candidates(t,np.column_stack((yl,vx.T)))
    cand = []
    for roots in _spectral_candidates(t,yl) if seeds is None else seeds:
        if roots.size > 4 and orc.left < 1000:
            continue
        for deg in (0,1,2):
            a = _exp_basis(t,roots,deg)
            if len(a) < a.shape[1] + 10:
                continue
            c = solve_ls(a,yl)
            e = np.max(np.abs(yl-a@c))
            cond = np.linalg.cond(a)
            if e < 1e-4*max(np.ptp(yl),1e-20) and cond < 2e9:
                if ensemble:
                    allv = np.column_stack((yl,vx.T))
                    av = solve_ls(a,allv)
                    e = np.max(np.abs(allv-a@av))
                cand.append((max(e,1e-13*np.max(np.abs(yl)))*max(1,cond/1e5)**.25,
                             roots, deg))
    if not cand:
        return None
    _, roots, deg = min(cand,key=lambda z:z[0])
    num = max(19,3*(len(roots)*2+deg+1))
    if ensemble:
        tt,old=t,yl
    else:
        per = min(num,39)
        if orc.left < per*d+95:
            return None
        ids = np.rint(np.linspace(0,len(t)-1,per)).astype(int)
        tt = t[ids]
        pts = center + tt[None,:,None]*direction + (np.eye(d)*delta)[:,None,:]
        old = yl[ids]
        vx = orc.sample(pts.reshape(-1,d)).reshape(d,per)/scale
    aa = _exp_basis(tt,roots,deg)
    all_coef = solve_ls(aa,np.column_stack((old,vx.T)))
    if np.max(np.abs(aa@all_coef-np.column_stack((old,vx.T)))) > .008*max(np.ptp(y),1e-20):
        return None
    w = []
    j = deg+1
    for root in roots:
        coef = all_coef[j].astype(complex)
        if root.imag:
            coef -= 1j*all_coef[j+1]
            j += 1
        j += 1
        if np.min(np.abs(coef)) < 1e-10*max(np.ptp(yl),1e-20):
            return None
        ratio = coef[1:]/coef[0]
        lg = (np.log(np.abs(ratio)) + 1j*np.angle(ratio))/delta
        if not root.imag:
            if np.any(np.real(ratio)<0):
                return None
            lg = np.real(lg).astype(complex)
        if not root.real:
            lg = 1j*np.imag(lg)
        err_ax = root-lg@direction
        if root.imag:
            alias = 2*np.pi/(t[1]-t[0])
            err_ax -= 1j*round(err_ax.imag/alias)*alias
        if abs(err_ax) > 1.5+.045*abs(root):
            return None
        lg += err_ax*direction/np.dot(direction,direction)
        w.append(lg)
    w = np.asarray(w)
    m = len(w)
    complex_masks = np.imag(roots) != 0.
    real_masks = np.real(roots) != 0.
    n = int(.82*len(x))
    # Additional sampled slices improve conditioning without taking points
    # from the independent global holdout.
    u = np.vstack([x[:n],center+t[:,None]*direction,pts.reshape(-1,d)])-.5
    values = np.r_[y[:n],yl,vx.ravel()]
    degree_options = (min(deg,1),deg) if deg==2 else (deg,)
    best = None
    for basdeg in degree_options:
        inds = multiindices(d,basdeg,max(basdeg,1))
        pol = legmat(u+.5,inds)
        p0 = pol.shape[1]
        def design(z, ww, poly=None):
            if poly is None:
                poly = legmat(z+.5,inds)
            zz = z @ ww.T
            e = np.exp(np.clip(zz.real,-160.,160.))
            pieces = [poly]
            for k in range(m):
                if complex_masks[k]:
                    pieces.extend([(e[:,k]*np.cos(zz[:,k].imag))[:,None],
                                   (e[:,k]*np.sin(zz[:,k].imag))[:,None]])
                else:
                    pieces.append(e[:,k,None])
            return np.hstack(pieces)
        a = design(u,w,pol)
        c = solve_ls(a,values)
        first_err = np.max(np.abs(a@c-values))
        if first_err > .13*max(np.ptp(values),1e-20):
            continue
        ww = w.copy()
        if first_err > 2e-13:
            def unpack(theta):
                p = p0+sum(1+cc for cc in complex_masks)
                vv = np.zeros((m,d),dtype=complex)
                n_real = int(real_masks.sum())*d
                vv[real_masks] = theta[p:p+n_real].reshape(-1,d)
                vv[complex_masks] += 1j*theta[p+n_real:].reshape(-1,d)
                return theta[:p],vv
            def fun(theta):
                co, ww = unpack(theta)
                return design(u,ww,pol)@co-values
            def jac(theta):
                co, ww = unpack(theta)
                a = design(u,ww,pol)
                j = p0
                active_r, active_i = [], []
                deriv = u @ ww.T
                e = np.exp(np.clip(deriv.real,-160.,160.))
                for k in range(m):
                    if complex_masks[k]:
                        if real_masks[k]:
                            active_r.append((co[j]*a[:,j]+co[j+1]*a[:,j+1])[:,None]*u)
                        active_i.append((co[j+1]*a[:,j]-co[j]*a[:,j+1])[:,None]*u)
                        j += 2
                    else:
                        if real_masks[k]:
                            active_r.append((co[j]*e[:,k])[:,None]*u)
                        j += 1
                return np.hstack([a]+active_r+active_i)
            param = np.r_[c,w.real[real_masks].ravel(),w.imag[complex_masks].ravel()]
            sol = least_squares(fun,param,jac=jac,max_nfev=45,method="trf",
                                ftol=2e-12,xtol=2e-12,gtol=4e-13,x_scale="jac")
            c,ww = unpack(sol.x)
        err = np.max(np.abs(design(x[n:]-.5,ww)@c-y[n:]))
        if err < 1e-10:
            c = solve_ls(design(x-.5,ww),y)
        if err > .035*max(np.ptp(y),1e-20):
            continue
        integ = float(c[0])
        j = p0
        for k,s in enumerate(ww):
            zz = s/2.
            moment = np.ones(d,dtype=complex)
            np.divide(np.sinh(zz),zz,out=moment,where=zz != 0)
            p = np.prod(moment)
            integ += c[j]*p.real
            j += 1
            if complex_masks[k]:
                integ += c[j]*p.imag
                j += 1
        def predict(p, c=c, ww=ww, design=design, inds=inds):
            return design(p-.5,ww,legmat(p,inds))@c
        mdl = Model(predict, integ, "ridge spectrum", float(err))
        if best is None or err < best.error:
            best = mdl
        if err < 1e-11:
            break
    return best


def _diff_groups(v, order, abs_tol):
    diffs = np.diff(v,order)
    nz = np.flatnonzero(np.abs(diffs)>abs_tol)
    g = np.split(nz, np.where(np.diff(nz)>1)[0]+1)
    return g if len(nz) else [], len(nz)


def _poly_breaks(t, v, allow_approx=False):
    """Find jumps in the first or second derivative on an equally spaced line."""
    span = np.ptp(v)
    if span <= 1e-13:
        return [], False
    step = t[1]-t[0]
    one = kink_lin(t,v,2e-10)
    options = [(a,1,abs(slope)) for a,i,j,slope in one]
    tol = max(4e-12*np.max(np.abs(v)),2e-11*span)
    col = False
    for order in (2,3,4,5):
        groups,nz = _diff_groups(v,order,tol*(2**(order-2)))
        if not len(groups):
            continue
        if nz > .47*(len(v)-order):
            continue
        col = True
        if order==2:
            if all(len(g)<=2 for g in groups) and len(options)>=len(groups)-1:
                break
            continue
        k = order-1
        for ix,g in enumerate(groups):
            if len(g) > order+1:
                continue
            i,j = int(g[0]),int(g[-1])
            il,ir = min(j+1,i+order-1),j+1
            if ir-il > 2:
                continue
            mid = (t[il]+t[ir])*.5
            nb = k+1
            if il+1 < nb or ir+nb > len(t):
                continue
            if ix > 0 and groups[ix-1][-1]+1 > il-nb+1:
                continue
            if ix+1 < len(groups) and groups[ix+1][0]+order-1 < ir+nb-1:
                continue
            def coef(indices):
                z = (t[indices]-mid)/step
                return np.polynomial.polynomial.polyfit(z,v[indices],k)
            c = coef(slice(ir,ir+nb))-coef(slice(il-nb+1,il+1))
            # The two pieces coincide at a continuous kink. Use the
            # derivative polynomial to locate double roots.
            rc = np.polynomial.polynomial.polyroots(c)
            dr = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(c))
            found = None
            for r in dr:
                if abs(r.imag) < .02 and abs(r.real) < 1.45:
                    if abs(np.polynomial.polynomial.polyval(r.real,c)) < 60*tol:
                        cp = np.polynomial.polynomial.polyval(r.real,np.polynomial.polynomial.polyder(c,2))
                        if abs(cp) > 30*tol:
                            found = (mid+step*r.real,2,abs(cp)/step**2)
                            break
            if found is None:
                for r in rc:
                    if abs(r.imag) < 1e-4 and abs(r.real) < 1.1:
                        cp = np.polynomial.polynomial.polyval(r.real,np.polynomial.polynomial.polyder(c))
                        if abs(cp)>40*tol:
                            found = (mid+step*r.real,1,abs(cp)/step)
                            break
            if found and all(abs(found[0]-b[0])>step*.85 for b in options):
                options.append(found)
        if len(options) >= .93*len(groups):
            break
    if not col and allow_approx:
        # Detect sharp changes against a background of slowly varying curvature.
        dd = np.diff(v,2)
        for i in np.argsort(np.abs(dd))[::-1][:12]:
            if i < 3 or i+4 >= len(v):
                continue
            nearby = np.r_[dd[i-3:i-1],dd[i+2:i+4]]
            if abs(dd[i]) < 3.8*np.max(np.abs(nearby))+span*2e-5:
                continue
            il,ir = i,i+2
            if abs(dd[i-1]) > abs(dd[i+1]) and abs(dd[i-1]) > .1*abs(dd[i]):
                il,ir = i-1,i+1
            sl = (v[il]-v[il-1])/step
            sr = (v[ir+1]-v[ir])/step
            rt = t[il]+(v[ir]-v[il]-sr*(t[ir]-t[il]))/(sl-sr)
            if t[il]-.3*step <= rt <= t[ir]+.3*step and all(abs(rt-p[0])>.9*step for p in options):
                options.append((rt,1,abs(sr-sl)))
            if len(options) >= 6:
                break
    return sorted(options),col


def _line_smooth_frequency(t,v,near):
    dt = t[1]-t[0]
    amp = max(1e-100,np.max(np.abs(v)))
    for order in (3,4):
        dd = np.diff(v,order)
        mid = (t[:-order]+t[order:])*.5
        for k in (3,2,1):
            l, q, r = dd[:-2*k],dd[k:-k],dd[2*k:]
            z = .5*(l+r)
            keep = np.abs(q)>6e-12*amp
            for a,_,_ in near:
                keep &= np.abs(mid[k:-k]-a)>(k+.6*order+1.1)*dt
            if keep.sum()<8:
                continue
            q,z = q[keep],z[keep]
            u = z/q
            u = u[(u>-1.000001)&(u<1.000001)]
            if len(u)<8:
                continue
            a = np.median(u)
            tol = np.maximum(np.abs(q)*.00006,2.0e-12*amp)
            ok = np.abs(z-q*a)<tol
            if ok.sum()<max(7,len(q)*.43):
                continue
            a = np.dot(z[ok],q[ok])/np.dot(q[ok],q[ok])
            if -1+1e-8<a<1-1e-6:
                return np.arccos(a)/(dt*k)
    return None


def _local_power(t,v,a,bounds):
    delta=t[1]-t[0]
    h=min(5.1*delta,.78*min(bounds))
    if h<3.7*delta:
        return None
    ok=np.abs(t-a)<h
    span=np.ptp(v[ok]) if ok.sum()>=8 else 0.
    if span<1e-10:
        return None
    z=(t[ok]-a)/h
    ys=(v[ok]-np.mean(v[ok]))/span
    pol=np.column_stack([np.ones(len(z)),z,z*z,z**3])
    opts=[]
    def basis(shift,power,mode):
        u=z-shift
        phi=np.abs(u) if mode==0 else np.maximum(0.,u*mode)
        return np.column_stack((pol,phi**power))
    for mode in (0,1,-1):
        for p in (.52,1.,1.6,2.):
            A=basis(0.,p,mode)
            c=solve_ls(A,ys)
            e=np.std(A@c-ys)
            opts.append((e,p,mode))
    opts.sort()
    fits=[]
    for e,p,mode in opts[:4]:
        if e>.048:
            break
        def fun(params):
            aa=basis(*params,mode)
            cc=solve_ls(aa,ys)
            return aa@cc-ys
        sol=least_squares(fun,[0.,p],bounds=([-.42,.20],[.42,2.8]),max_nfev=24,
                          ftol=1e-8,xtol=1e-8,gtol=1e-10)
        err=np.max(np.abs(sol.fun))
        q0,q1=sol.x
        if abs(q1-round(q1)) > .065 and err<min(.006, e*.28+1e-5) and abs(q0)<.415:
            aa=basis(q0,q1,mode)
            amp=abs(solve_ls(aa,ys)[-1])
            if amp> max(.035,err*20):
                fits.append((err,a+h*q0,q1,mode))
    if not fits:
        return None
    _,root,power,mode=min(fits)
    return root,power,mode


def _plane_probes(orc, line, scale, exact=False, ignore=(), known=None, smooth=None):
    center,direction,t,raw = line
    v = raw/scale
    candidates,col = _poly_breaks(t,v,allow_approx=not exact)
    if known is not None:
        rem = v - known.predict(center+t[:,None]*direction)
        second,_ = _poly_breaks(t,rem,allow_approx=not exact)
        candidates = candidates + [z for z in second if
                                   all(abs(z[0]-p[0]) > .6*(t[1]-t[0]) for p in candidates)]
    if col and orc.left>360:
        tol=max(4e-12*np.max(np.abs(v)),2e-11*np.ptp(v))
        for order in (2,3,4,5):
            groups,nz=_diff_groups(v,order,tol*(2**(order-2)))
            if not 0<nz<.47*(len(t)-order):
                continue
            bad=[]
            for g in groups:
                a,b=t[max(0,g[0]-1)],t[min(len(t)-1,g[-1]+order+1)]
                if len(g)>order or not any(a<q[0]<b for q in candidates):
                    bad.append((len(g),a,b))
            for _,a,b in sorted(bad,reverse=True)[:2]:
                if orc.left<320:
                    break
                tt=np.linspace(a,b,23)
                vv=orc.sample(center+tt[:,None]*direction)/scale
                new,_=_poly_breaks(tt,vv)
                if new:
                    prev=[q for q in candidates if a+.002<q[0]<b-.002]
                    if len(new)>len(prev):
                        candidates=[q for q in candidates if not a+.002<q[0]<b-.002]+new
            break
    if len(ignore):
        candidates = [c for c in candidates if all(abs(c[0]-p[0]) > p[1] for p in ignore)]
    candidates = sorted(candidates, key=lambda q:q[2], reverse=True)[:9]
    candidates.sort()
    out = []
    harmon = _line_smooth_frequency(t,v,candidates) if smooth is not None and not col and len(candidates) else None
    abs_tol = 3e-11 * max(np.max(np.abs(v)),1e-8)
    affine = np.count_nonzero(np.abs(np.diff(v,2))>abs_tol) < .32*len(t)
    delta = t[1]-t[0]
    for j,(a,k,amp) in enumerate(candidates):
        bounds = [t[-1]-a,a-t[0]]
        if j:
            bounds.append(a-candidates[j-1][0])
        if j+1<len(candidates):
            bounds.append(candidates[j+1][0]-a)
        pow_shape = _local_power(t,v,a,bounds) if not col and harmon is None else None
        if pow_shape is not None:
            a_frac,p_frac,s_frac = pow_shape
            a = a_frac
        dt = min(.012 if affine else .0082, .21*min(bounds))
        step = min(.00015, dt*np.linalg.norm(direction)*.035)
        order = 2
        if dt < 1e-6 or orc.left < (4 if k==1 and affine else 12)*orc.d+65:
            continue
        # When curvature is present, get a more accurate intersection by
        # matching the two one-sided polynomial extensions on the line.
        if (not affine or not col) and pow_shape is None:
            width = min(.62*delta, .72*min(bounds))
            if width > 2e-6 and orc.left >= 80+12*orc.d:
                ss = np.array([-1.,-.84,-.69,-.55,.55,.69,.84,1.])
                pp = center+(a+width*ss)[:,None]*direction
                vv = orc.sample(pp)/scale
                cl = np.polynomial.polynomial.polyfit(ss[:4],vv[:4],3)
                cr = np.polynomial.polynomial.polyfit(ss[4:],vv[4:],3)
                dc = cr-cl
                cc = dc if k==1 else np.polynomial.polynomial.polyder(dc)
                roots = np.polynomial.polynomial.polyroots(cc)
                cand = [r.real for r in roots if abs(r.imag) < 2e-5 and abs(r.real) < .53]
                if cand:
                    a += width*min(cand,key=abs)
                    dt = min(dt, .17*min(t[-1]-a,a-t[0]))
        if k==1 and affine and pow_shape is None:
            pts = center+(a+dt*np.array([-1.,1.]))[:,None]*direction
            g = gradients(orc,pts,scale,order=order,step=step)
            jumps = g[1]-g[0]
        else:
            pts = center+(a+dt*np.array([-3.,-2.,-1.,1.,2.,3.]))[:,None]*direction
            g = gradients(orc,pts,scale,order=order,step=step)
            if k == 1:
                jumps = 3*(g[3]-g[2])-3*(g[4]-g[1])+(g[5]-g[0])
            else:
                jumps = 4*(g[4]+g[1])-2.5*(g[3]+g[2])-1.5*(g[5]+g[0])
            if pow_shape is not None:
                # The smooth part contributes a linear change to the
                # difference across the cusp. Cancel it with two distances.
                jumps = 2*(g[3]-g[2])-(g[4]-g[1])
            if harmon is not None:
                gl,gr = g[2]-2*g[1]+g[0],g[5]-2*g[4]+g[3]
                nl,nr = np.linalg.norm(gl),np.linalg.norm(gr)
                if min(nl,nr)>2.5e-9 and abs(np.dot(gl,gr))/(nl*nr) > .999985:
                    w = gl/nl if nl>nr else gr/nr
                    wd = float(w@direction)
                    if abs(wd)>.075 and .65 < harmon/abs(wd) < 100:
                        ww = w*harmon/wd
                        smooth.append((ww,min(nl,nr)))
        norm = np.linalg.norm(jumps)
        if norm < 2e-10 or abs(jumps@direction) < .025*norm:
            continue
        w = jumps/norm
        imax = np.argmax(np.abs(w))
        if pow_shape is None or s_frac == 0:
            w *= np.sign(w[imax])
        else:
            w *= s_frac*np.sign(w@direction)
        if pow_shape is not None:
            k=-p_frac if s_frac==0 else p_frac
        b = (center + a*direction-.5)@w
        out.append((w,b,k,norm/((dt if k==2 else 1.))))
    return out,col


def _combine_planes(planes, nearby):
    for w,b,k,q in nearby:
        def close(w0,b0,k0):
            if np.sign(k)!=np.sign(k0) or abs(k-k0)>.14:
                return False
            dist=np.linalg.norm(w-w0)+abs(b-b0)
            if k in (1,2) or k<0:
                dist=min(dist,np.linalg.norm(w+w0)+abs(b+b0))
            return dist < (.012 if k not in (1,2) else .0035)
        if not any(close(w0,b0,k0) for w0,b0,k0,q0 in planes):
            planes.append((w,b,k,q))
    return planes


def _hinge_fit(planes, xx, yy, valid, valid_y, refit=True,frequency=None):
    if not planes:
        return None
    d = xx.shape[1]
    planes = sorted(planes,key=lambda p:p[3],reverse=True)[:20]
    n = len(xx)
    v = xx-.5
    axes = np.array([np.argmax(np.abs(p[0])) for p in planes])
    w = np.array([p[0] for p in planes])
    b = np.array([p[1] for p in planes])
    powers = np.array([p[2] for p in planes],float)
    norm = np.abs(w[np.arange(len(w)),axes])
    w /= norm[:,None]
    b /= norm
    d0 = v@w.T - b
    ok = (d0>0).sum(axis=0) >= 2
    ok &= (d0<0).sum(axis=0) >= 2
    w,b,powers,axes = w[ok],b[ok],powers[ok],axes[ok]
    if not len(w):
        return None
    z0 = v@w.T-b
    z = np.where(powers<0,np.abs(z0),np.maximum(z0,0.))**abs(powers)
    best = None
    for degree in (1,2,3):
        inds = multiindices(d,degree,2)
        if inds.shape[1] > n*.36:
            break
        original = legmat(xx,inds)
        def feat(q, freq, base):
            if freq is None:
                return base
            s = q@freq
            return np.column_stack([base,np.cos(s),np.sin(s)])
        ww = frequency.copy() if frequency is not None else None
        base = feat(v,ww,original)
        a = np.column_stack([base,z])
        c = solve_ls(a,yy)
        p = base.shape[1]
        kee = np.abs(c[p:])*np.std(z,axis=0) > 8e-12
        wi,bi,ki,ax = w[kee].copy(),b[kee].copy(),powers[kee],axes[kee]
        if not len(wi) or len(wi)>15:
            continue
        a = np.column_stack([base,z[:,kee]])
        c = solve_ls(a,yy)
        err = float(np.max(np.abs(a@c-yy)))
        absolute = ki<0
        movable = (abs(ki)!=1.)&(abs(ki)!=2.)
        variable = int(movable.sum())
        pp = abs(ki)
        anchor = wi[np.arange(len(wi)),ax].copy()
        def primitive(s):
            return np.where(absolute,np.abs(s),np.maximum(s,0.))
        if refit and 1.5e-12 < err < .13*max(np.ptp(yy),1e-20):
            m = len(wi)
            fm = np.ones((m,d),dtype=bool)
            fm[np.arange(m),ax] = False
            def unpk(qq):
                wi = np.ones((m,d))
                wi[fm] = qq[p+m:p+m*d]
                wi[np.arange(m),ax]=anchor
                mid=p+m*d+m
                pp0=pp.copy()
                pp0[movable]=qq[mid:mid+variable]
                return (qq[:p+m],wi,qq[p+m*d:mid],pp0,
                        None if ww is None else qq[-d:])
            def fun(qq):
                c,wi,bi,pow0,ww0 = unpk(qq)
                feat = primitive(v@wi.T-bi)**pow0
                ss = original
                if ww0 is not None:
                    a0 = v@ww0
                    ss = np.column_stack([ss,np.cos(a0),np.sin(a0)])
                return ss@c[:p]+feat@c[p:]-yy
            def jac(qq):
                c,w,b,pow0,ww0 = unpk(qq)
                z = v@w.T-b
                pos = primitive(z)
                low = np.maximum(pos,1e-13)
                feats=pos**pow0
                sign=np.where(absolute,np.sign(z),(z>0))
                de = sign*pow0*low**(pow0-1.)*c[p:]
                jo = [(de[:,k,None]*v[:,fm[k]]) for k in range(m)]
                a = feat(v,ww0,original)
                pb=(c[p:]*feats*np.log(low))[:,movable]
                if ww0 is None:
                    return np.column_stack([a,feats,*jo,-de,pb])
                jr = (c[p-1]*a[:,p-2]-c[p-2]*a[:,p-1])[:,None]*v
                return np.column_stack([a,feats,*jo,-de,pb,jr])
            qq = np.r_[c,wi[fm],bi,pp[movable]]
            if ww is not None:
                qq = np.r_[qq,ww]
            lo,hi=np.full(len(qq),-np.inf),np.full(len(qq),np.inf)
            if variable:
                mid=p+m*d+m
                lo[mid:mid+variable],hi[mid:mid+variable]=.19,2.95
            sol = least_squares(fun,qq,jac=jac,max_nfev=43 if variable else 32,
                                bounds=(lo,hi),ftol=9e-12,gtol=6e-14,
                                xtol=5e-12,x_scale="jac", method="trf")
            c,wi,bi,pp,ww = unpk(sol.x)
        def pred(x, wi=wi, bi=bi, pp=pp, absolute=absolute, inds=inds, c=c, p=p,ww=ww):
            s=(x-.5)@wi.T-bi
            return feat(x-.5,ww,legmat(x,inds))@c[:p]+np.where(absolute,np.abs(s),np.maximum(s,0.))**pp@c[p:]
        er = float(np.max(np.abs(pred(valid)-valid_y)))
        if er > .25*max(np.ptp(yy),1e-20) or (best is not None and best.error <= er):
            continue
        ints = []
        for wi0,bi0,ki0,ai in zip(wi,bi,pp,absolute):
            proj = Projection(wi0)
            moment = proj.moment(.5-bi0/proj.span,ki0)
            if ai:
                moment += proj.moment(.5+bi0/proj.span,ki0)
            ints.append(moment * proj.span**ki0)
        i0=c[0]+np.dot(c[p:],ints)
        if ww is not None:
            i0 += c[p-2]*np.prod(np.sinc(ww/(2*np.pi)))
        model = Model(pred, float(i0), "piecewise ridge", er)
        model.planes = (wi,bi,np.where(absolute,-pp,pp))
        model.weights = c[p:]
        model.frequency = ww
        model.train_error = float(np.max(np.abs(pred(xx)-yy)))
        best = model
        if er < 1e-12:
            break
    return best


def _near_planes(orc, model, n=18):
    w,b,_ = model.planes
    pts = []
    dirs = []
    for wi,bi,ci in zip(w,b,model.weights):
        if abs(ci)*np.linalg.norm(wi)<1e-8:
            continue
        # Form plane points in the box by mixing a low and a high
        # point on the projection; use random coordinates when possible.
        axis = np.argmax(np.abs(wi))
        q = orc.rng.uniform(.005,.995,(max(96,n*4),orc.d))
        q[:,axis] += (bi-(q-.5)@wi)/wi[axis]
        q = q[(q[:,axis]>.008)&(q[:,axis]<.992)]
        if not len(q):
            span = np.sum(np.abs(wi))
            ss = .5+bi/span
            if not 0 < ss < 1:
                continue
            q = np.tile(np.where(wi>0,ss,1.-ss),(n,1))
        v = wi/np.max(np.abs(wi))
        q = q[:n]
        h = np.resize(np.array([0.,0.001,-0.001,0.018,-0.018,0.,.004,-.004]),len(q))
        pts.extend(q+h[:,None]*v)
        dirs.extend(np.broadcast_to(v,q.shape))
    return np.clip(np.asarray(pts).reshape(-1,orc.d),0.,1.),np.asarray(dirs).reshape(-1,orc.d)


def _known_on_line(model, center, direction, spacing):
    if model is None:
        return []
    w,b,_ = model.planes
    speed = w@direction
    use = np.abs(speed) > 1e-10
    root = (b[use] - w[use]@(center-.5))/speed[use]
    return [(t,.20*spacing) for t in root if -.5 < t < .5]


def hinges(orc, line, x, y, scale):
    """Fit and integrate a sum of truncated-power ridges."""
    d = orc.d
    if d > 10 or orc.left < 180:
        return None
    probes,col = _poly_breaks(line[2],line[3]/scale,allow_approx=True)
    if not probes and not col and np.ptp(line[3]/scale) < 1e-12:
        return None
    n0 = int(len(x)*.8)
    smooth = []
    planes,_ = _plane_probes(orc,line,scale,smooth=smooth)
    planes = _combine_planes([],planes)
    # Boundary data finds corner caps that scarcely occur in volume samples.
    use_corners = (d <= 8 and orc.left >= 2**d+480)
    if use_corners:
        cs = ((np.arange(2**d)[:,None] >> np.arange(d)) & 1).astype(float)
        cy = orc.sample(cs)/scale
    else:
        cs,cy = np.empty((0,d)),np.empty(0)
    baseline_inds = multiindices(d,2,2)
    bx = legmat(x,baseline_inds)
    bc = solve_ls(bx,y)
    for _ in range(4):
        er = np.abs(bx@bc-y)
        subset = er < max(np.quantile(er,.72),1e-12)
        if subset.sum() <= 2*len(bc):
            break
        bc = solve_ls(bx[subset],y[subset])
    def base_predict(p):
        return legmat(p,baseline_inds)@bc
    val_x = x[n0:]
    val_y = y[n0:]
    def fit(refit=True):
        tx,ty = orc.data()
        xx,yy = np.vstack((x[:n0],tx[len(x):])),np.r_[y[:n0],ty[len(x):]/scale]
        warm = _hinge_fit(planes,xx,yy,val_x,val_y,refit=refit)
        if smooth and (warm is None or warm.error > 2e-11):
            ff = max(smooth,key=lambda pair:pair[1])[0]
            v = _hinge_fit(planes,xx,yy,val_x,val_y,frequency=ff)
            if v is not None and (warm is None or v.error < warm.error*.92):
                return v
        return warm
    model = fit(refit=not col) if planes else None
    best = model
    known_x = np.vstack((x,cs))
    known_y = np.r_[y,cy]
    attempts = set()
    for rep in range(9):
        if orc.left < 128 or orc.elapsed > 11.:
            break
        if rep>=4 and orc.left<510 and not col and not smooth:
            if model is None or np.std(model.predict(x)-y)>.13*np.std(y):
                break
        pred = model.predict if model is not None else base_predict
        diff = np.abs(pred(known_x)-known_y)
        maxerr = diff.max()
        # Asking where the extrapolated planes are is more discriminating
        # than another volume sample once the global residual is small.
        if model is not None and maxerr < 1e-8 and orc.left > 110:
            n = min(20,orc.left//max(1,4*len(model.weights)))
            nx,ndir = _near_planes(orc,model,n)
            if len(nx):
                ny = orc.sample(nx)/scale
                ne = np.abs(pred(nx)-ny)
                if ne.max() < 5e-10 and maxerr < 5e-10 and rep>=1:
                    model.error = max(model.error,ne.max(),maxerr)
                    return model
                known_x=np.vstack((known_x,nx))
                known_y=np.r_[known_y,ny]
                diff=np.r_[diff,ne]
        chosen = None
        if diff.max()>5e-10 and (rep != 1 or model is None):
            candidates = np.argsort(diff)[::-1][:50]
            for j in candidates:
                if tuple(known_x[j]) not in attempts:
                    chosen=j
                    break
        if chosen is None or rep in (1,5) or diff[chosen] < maxerr*.08:
            sl = make_line(orc,rep+2,size=65)
        else:
            a = known_x[chosen]
            attempts.add(tuple(a))
            inds = np.flatnonzero(diff <= max(np.quantile(diff,.47),2e-12))
            dd = known_x[inds]
            vv = dd-a
            # Long directions reach a different linear region. Include
            # randomness to avoid repeatedly crossing the same set of planes.
            weight = np.linalg.norm(vv,axis=1) * orc.rng.uniform(.62,1.,len(vv))
            far = dd[np.argmax(weight)]
            span = np.max(np.abs(a-far))
            if span < .03:
                sl = make_line(orc,rep+3,size=65)
            else:
                sl = make_line(orc,endpoints=(a,far),size=65)
        near = _known_on_line(model,sl[0],sl[1],sl[2][1]-sl[2][0]) if model is not None else []
        kfrom,_ = _plane_probes(orc,sl,scale,ignore=near if rep%3 != 2 else (),known=model,smooth=smooth)
        prev = len(planes)
        _combine_planes(planes,kfrom)
        if len(planes) > prev or (model is not None and rep%2==1):
            trial = fit(refit=(not col or rep>=1))
            if trial is not None:
                this = np.max(np.abs(trial.predict(known_x)-known_y))
                if (model is None or this < np.max(np.abs(model.predict(known_x)-known_y))*1.03
                        or trial.error < model.error*.18):
                    model = trial
                if best is None or (this < np.max(np.abs(best.predict(known_x)-known_y))*.997):
                    best = trial
        if not planes and rep>=3:
            break
    return best if model is None else model
