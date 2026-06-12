"""RT5 -- GPU parity + benchmark.  Run on SkyNet (RTX 3090):

    uv add cupy-cuda12x        # or: pip install cupy-cuda12x
    python3 gpu_parity.py

Compares the SAME code (pbr.py is backend-agnostic) on both backends:

  (a) deterministic paths exactly (float32 tolerance):
      ray-capsule intersection, direct emission, display transform, LUT;
  (b) the stochastic scatter estimator statistically: NumPy and CuPy
      RNGs produce different sequences for the same seed, so pixel-exact
      equality is impossible by construction -- instead the two MC means
      must agree within 5 standard errors (both estimate the same
      integral, validated against closed forms in RT2);
  (c) benchmark: 1080p frame at production sample counts.
"""

import time
import pickle
import sys
import numpy as np
sys.path.insert(0, ".")
import pbr
from pbr import (capsule_intersect, equiangular_scatter, display_transform,
                 apply_lut, bake_lut, render_pbr, spectral_bands, SIGMA_M)

try:
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() >= 1
except Exception as e:
    print("No CUDA device / CuPy:", e)
    sys.exit(1)

print("RT5 -- GPU parity (RTX:", cp.cuda.runtime.getDeviceProperties(0)
      ["name"].decode(), ")")
ok = True
rng = np.random.default_rng(5)

# (a) deterministic: intersection
S = 200
A = rng.uniform(-5, 5, (S, 3)).astype(np.float32)
B = (A + rng.uniform(-3, 3, (S, 3))).astype(np.float32)
R = rng.uniform(0.1, 0.6, S).astype(np.float32)
ro = rng.uniform(-12, 12, (5000, 3)).astype(np.float32)
rd = rng.normal(size=(5000, 3)).astype(np.float32)
rd /= np.linalg.norm(rd, axis=1, keepdims=True)
t_c, i_c = capsule_intersect(ro, rd, A, B, R, np)
t_g, i_g = capsule_intersect(cp.asarray(ro), cp.asarray(rd),
                             cp.asarray(A), cp.asarray(B),
                             cp.asarray(R), cp)
t_g, i_g = cp.asnumpy(t_g), cp.asnumpy(i_g)
hit = np.isfinite(t_c) & np.isfinite(t_g)
dev = np.abs(t_c[hit] - t_g[hit]).max() if hit.any() else 0.0
same_idx = (i_c == i_g).mean()
p1 = dev < 1e-3 and same_idx > 0.999 \
    and np.array_equal(np.isfinite(t_c), np.isfinite(t_g))
print(f"  intersection parity: max |dt| {dev:.2e}, "
      f"index agreement {100*same_idx:.2f}%  {'PASS' if p1 else 'FAIL'}")
ok &= p1

# (a) deterministic: display transform + LUT
test = (2.0 ** (rng.uniform(0, 1, (100000, 3)) * 16.5 - 10)) \
    .astype(np.float32)
d_c = display_transform(test, np)
d_g = cp.asnumpy(display_transform(cp.asarray(test), cp))
lut = bake_lut(33)
l_c = apply_lut(test, lut, np)
l_g = cp.asnumpy(apply_lut(cp.asarray(test), lut, cp))
e1, e2 = np.abs(d_c - d_g).max(), np.abs(l_c - l_g).max()
p2 = e1 < 1e-5 and e2 < 1e-5
print(f"  display/LUT parity: max dev {max(e1,e2):.2e}  "
      f"{'PASS' if p2 else 'FAIL'}")
ok &= p2

# (b) stochastic: scatter estimator, statistical agreement
lam, Sb, XYZw, sigR = spectral_bands()
sigT = (sigR + SIGMA_M).astype(np.float32)
ro1 = np.zeros((200000, 3), np.float32)
rd1 = np.tile(np.array([[0.6, 0.4, 0.69282]], np.float32), (200000, 1))
A1 = np.array([[30, 60, 5]], np.float32)
B1 = np.array([[32, 40, -4]], np.float32)
tm = np.full(200000, 900.0, np.float32)
kw = dict(A=A1, B=B1, w_len_cdf=np.array([1.0], np.float32), W_total=1.0,
          S=Sb.astype(np.float32), sigR=sigR.astype(np.float32),
          sigM=np.float32(SIGMA_M), sigT=sigT, spp=1)
v_c = equiangular_scatter(ro1, rd1, tm, rng=np.random.default_rng(0),
                          xp=np, **kw)[:, 4]
kw_g = {k: (cp.asarray(v) if isinstance(v, np.ndarray) else v)
        for k, v in kw.items()}
v_g = cp.asnumpy(equiangular_scatter(
    cp.asarray(ro1), cp.asarray(rd1), cp.asarray(tm),
    rng=cp.random.default_rng(0), xp=cp, **kw_g)[:, 4])
se = np.sqrt(v_c.var() / len(v_c) + v_g.var() / len(v_g))
z = abs(v_c.mean() - v_g.mean()) / se
p3 = z < 5.0
print(f"  scatter estimator: |mean diff| = {z:.2f} standard errors "
      f"(CPU {v_c.mean():.4e}, GPU {v_g.mean():.4e})  "
      f"{'PASS' if p3 else 'FAIL'}")
ok &= p3

# (c) benchmark
res = pickle.load(open("leader3d.pkl", "rb"))
cur = pickle.load(open("currents3d.pkl", "rb"))
for (w, h, spp, aa) in ((960, 540, 64, 4), (1920, 1080, 256, 8)):
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    img, _ = render_pbr(res, cur, out_w=w, out_h=h, spp=spp, aa=aa,
                        xp=cp, lut=lut)
    cp.cuda.Stream.null.synchronize()
    dt = time.time() - t0
    print(f"  bench {w}x{h} @ {spp} spp, aa={aa}: {dt:.2f} s")
    if (w, h) == (1920, 1080):
        import cv2
        cv2.imwrite("lightning_pbr_hd.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("  saved lightning_pbr_hd.png")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
