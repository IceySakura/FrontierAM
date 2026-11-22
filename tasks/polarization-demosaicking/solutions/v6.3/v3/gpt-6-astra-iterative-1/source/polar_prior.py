"""A translation-aware, constant-preserving local interpolation prior."""
import numpy as np

_CHANNEL = np.tile(np.arange(4), 25)
_POSITION = np.array([[2*y+c//2, 2*x+c%2]
                      for y in range(-2, 3) for x in range(-2, 3)
                      for c in range(4)], dtype=np.float64)
_PAIRS = [(c, p) for c in range(4) for p in range(4) if c != p]
_TARGET = np.array([c for c, p in _PAIRS] + list(range(4)))
_QUERY = np.array([[p//2, p%2] for c, p in _PAIRS]
                  + [[c//2, c%2] for c in range(4)], dtype=np.float64)
_INDICATOR = (_CHANNEL[:, None] == np.arange(4)).astype(np.float64)
_SAME_CHANNEL = _CHANNEL[:, None] == _CHANNEL[None, :]


def contrast_scales(q):
    variance = np.zeros(4, np.float32)
    # Consecutive lags avoid mistaking periodic texture for a constant plane.
    for lag in (3, 4):
        difference = q[lag:]-q[:-lag]
        variance += np.mean(difference*difference, axis=(0, 1))
        difference = q[:, lag:]-q[:, :-lag]
        variance += np.mean(difference*difference, axis=(0, 1))
    constant = variance == 0
    average = float(variance.mean())
    if average < 1e-6:
        return None, constant
    gain = np.sqrt(variance/average).clip(.05, 2)
    log_gain = np.log(gain)
    strength = np.clip((float(np.max(np.abs(log_gain)))-.12)/.12, 0, 1)
    if strength == 0:
        return None, constant
    return np.exp(log_gain*strength).astype(np.float32), constant


def estimate_shifts(q, reference=0):
    low = (q[0::2, 0::2] + q[0::2, 1::2]
           + q[1::2, 0::2] + q[1::2, 1::2]) * 0.25
    h, w, _ = low.shape
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    spectrum = np.fft.rfft2((low-low.mean((0, 1))) * window[:, :, None], axes=(0, 1))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    r2 = fy*fy+fx*fx
    band = np.exp(-r2/0.04) * (1-np.exp(-r2/0.0004))
    shifts = np.zeros((4, 2), np.float64)
    radius = 8
    for c in range(4):
        if c == reference:
            continue
        cross = spectrum[:, :, reference] * spectrum[:, :, c].conj()
        magnitude = np.abs(cross)
        peak = magnitude.max()
        if peak < 1e-10:
            continue
        cross *= band / (np.sqrt(magnitude)+np.sqrt(peak)*0.001+1e-12)
        corr = np.fft.irfft2(cross, s=(h, w))
        local = np.roll(corr, (radius, radius), axis=(0, 1))[:17, :17]
        py, px = np.unravel_index(np.argmax(local), local.shape)
        iy, ix = (py-radius) % h, (px-radius) % w
        center = corr[iy, ix]
        up, down = corr[(iy-1) % h, ix], corr[(iy+1) % h, ix]
        left, right = corr[iy, (ix-1) % w], corr[iy, (ix+1) % w]
        curvature_y = 2*(up+down-2*center)
        curvature_x = 2*(left+right-2*center)
        epsilon = 1e-12*max(1., abs(center))
        dy = 0. if abs(curvature_y) < epsilon else np.clip((up-down)/curvature_y, -.75, .75)
        dx = 0. if abs(curvature_x) < epsilon else np.clip((left-right)/curvature_x, -.75, .75)
        shifts[c] = [-(py-radius+dy)*4+c//2-reference//2,
                     -(px-radius+dx)*4+c%2-reference%2]
    return shifts


def make_kernel(shifts, observed=False):
    count = 16 if observed else 12
    target = _TARGET[:count]
    coordinates = _POSITION-shifts[_CHANNEL]
    query = _QUERY[:count]-shifts[target]
    distance = ((coordinates[:, None]-coordinates[None])**2).sum(2)
    covariance = 36*np.exp(-distance/4.5) + 900*np.exp(-distance/72)
    covariance += _SAME_CHANNEL * (0.25*np.exp(-distance/18))
    covariance.flat[::101] += 3.24
    distance = ((coordinates[:, None]-query[None])**2).sum(2)
    cross = 36*np.exp(-distance/4.5) + 900*np.exp(-distance/72)
    cross += (_CHANNEL[:, None] == target[None]) * (0.25*np.exp(-distance/18))
    # Unknown channel-specific constants are estimated without shrinking them.
    system = np.empty((104, 104), np.float64)
    system[:100, :100] = covariance
    system[:100, 100:] = _INDICATOR
    system[100:, :100] = _INDICATOR.T
    system[100:, 100:] = 0
    rhs = np.concatenate([cross, (np.arange(4)[:, None] == target[None]).astype(float)])
    weights = np.linalg.solve(system, rhs)[:100].astype(np.float32)
    if observed:
        for c in range(4):
            center = weights[48+c, 12+c]
            weights[48+c, 12+c] = 0
            weights[:, 12+c] /= 1-center
    return weights.reshape(5, 5, 4, count)
