# Polarization-mosaic reconstruction (single-channel DoFP demosaicking)

## Interface

`solution.py` exposes `reconstruct(raw)` which takes the `(1024, 1024)` `uint8`
mosaic and returns a `(4, 1024, 1024)` `float32` array holding the 0°, 45°,
90° and 135° images. No build step is required (no `build.sh`); the only
runtime dependency is NumPy.

## Method

The filter arrangement is 2×2 periodic:

| phase p | pixels      | channel |
|---------|-------------|---------|
| 0       | (odd, odd)  | 0°      |
| 1       | (even, odd) | 45°     |
| 2       | (even,even) | 90°     |
| 3       | (odd, even) | 135°    |

### 1. Modulation (polyphase) decomposition

Write the mosaic as

```
M(y,x) = E0(y,x) + u·E1(y,x) + v·E2(y,x) + u·v·E3(y,x),  u = (-1)^y, v = (-1)^x
```

with smooth envelopes. Solving the 2×2 sign system gives

```
E1 = (I_90 + I_45 - I_135 - I_0)/4      E2 = (I_90 - I_45 + I_135 - I_0)/4
E3 = (I_90 - I_45 - I_135 + I_0)/4      E0 = (I_90 + I_45 + I_135 + I_0)/4
```

and conversely each channel is a sign combination of the envelopes:

```
I_0  = E0 - E1 - E2 + E3      I_45 = E0 + E1 - E2 - E3
I_90 = E0 + E1 + E2 + E3      I_135= E0 - E1 + E2 - E3
```

The four images are almost identical, so the envelopes E1, E2, E3 are small
and smooth, while E0 carries the intensity. Crucially, **E0 never has to be
estimated**: at every pixel the mosaic already observes one channel exactly,
so

```
I_k = M + Σ_m (s_{k,m} − s_{p(y,x),m}) · E_m ,     m = 1,2,3
```

and the term in parentheses vanishes when `k = p(y,x)`. The observed samples
are therefore reproduced bit-exactly, and all the remaining work is the
estimation of the three smooth difference envelopes.

### 2. Envelope estimation

Each envelope is recovered from the sign-demodulated mosaic
(`u·M`, `v·M`, `uv·M`) by a 7×7 FIR convolution. The three kernels were fitted
by ordinary least squares on the five public development scenes, minimising
the full-image squared error of the demodulated envelopes against the
reference planes. The normal equations were built from FFT-based
autocorrelations/cross-correlations (reflect padding) and solved directly,
with a small ridge term for conditioning.

The filters automatically realise the bias/variance trade-off that a plain
low-pass demodulation cannot: they pass the low-frequency envelope content
while rejecting the aliased image content that leaks into the demodulated
signal, and they do so with the per-frequency weighting that is optimal for
this data.

### 3. Reconstruction

The three estimated envelopes are combined with the fixed sign table to form
the offsets `Σ_m (s_{k,m} − s_{p,m})·Ê_m`, which are added to the mosaic on
each of the four phase sub-lattices.

## Results on the public development scenes

| scene | 0° | 45° | 90° | 135° | mean |
|------:|---:|----:|----:|-----:|-----:|
| 43 | 2.292 | 2.123 | 2.008 | 2.129 | 2.138 |
| 55 | 1.538 | 1.491 | 1.450 | 1.493 | 1.493 |
| 72 | 1.327 | 1.232 | 1.216 | 1.233 | 1.252 |
| 91 | 1.634 | 1.504 | 1.485 | 1.525 | 1.537 |
| 99 | 1.462 | 1.419 | 1.371 | 1.389 | 1.410 |

`public_mean_rmse = 1.566` (vs. 3.63 for the trivial "copy the mosaic"
estimate and ≈2.6 for a wide low-pass demodulation).

## Performance

Single-threaded, one 1024×1024 scene: ≈0.15 s of compute plus ≈0.09 s of
Python/NumPy start-up, i.e. ≈0.24 s of wall clock in total — well inside the
1 s per-scene budget. Peak memory is a few tens of MB. The work is a fixed
number of element-wise array operations, so the cost scales linearly with the
number of pixels and contains no data-dependent branches.

## Notes

* The estimator is a single global linear filter; it is scene independent,
  uses no per-scene look-up, stores no state between calls, and involves no
  randomness.
* Because the reconstruction is anchored on the observed mosaic value at every
  pixel, any degradation of the envelope estimates affects only the 3/4 of the
  samples that are not observed for a given channel, which keeps the method
  robust on scenes whose statistics differ from the development set.
