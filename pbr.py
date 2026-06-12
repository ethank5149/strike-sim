"""
pbr.py -- physically based spectral ray tracer for the 3-D strike.
Backend-agnostic: every routine takes `xp` (numpy or cupy) and runs
identically on CPU (validation) and an RTX 3090 (production).

Physics
  * Geometry: channel segments are emissive capsules; exact closed-form
    ray-capsule intersection (infinite-cylinder quadratic + spherical
    caps), nearest-hit over all segments.
  * Media: two-component homogeneous clear-storm atmosphere --
    molecular Rayleigh scattering, sigma_R(lambda) = sigma_R(550) *
    (550/lambda)^4 with the Rayleigh phase 3/(16 pi)(1 + cos^2),
    plus aerosol scattering with Henyey-Greenstein phase (g = 0.76).
    Transmittance is Beer-Lambert (closed form in homogeneous media).
  * Single scattering: Monte Carlo with equiangular distance sampling
    (Kulla & Fajardo 2012) and power-proportional light selection.
    With P(light) ~ w_c * len_c and uniform point-on-segment sampling,
    the estimator reduces to
        W * S(lambda) * [sigma_R p_R + sigma_M p_HG]
          * exp(-sigma_t (t + r)) * dTheta / (4 pi D)
    -- the 1/r^2 cancels against the pdf (zero variance in the
    isotropic, no-extinction limit, which Stage RT2 exploits).
  * Emission: 30,000 K Planck spectrum in N_BANDS bands; channel
    radiance L_e = w_c S(lambda) / (2 pi^2 r_core) from power-per-length
    -> exitance -> radiance, so core and halo share one energy scale.
    r_core is the resolved luminous envelope (the cm-scale conducting
    core is sub-pixel at 12.5 m/cell; documented stylization).
  * Ground: Lambertian albedo 0.07 with MC direct lighting from the
    segment set + Fresnel (F0 = 0.02, wet) mirror reflection.
  * Display: scene-referred spectral -> XYZ (validated CMFs) -> linear
    sRGB -> AgX-family transform (log2 shaper, inset/outset gamut
    matrices, tanh sigmoid) baked into a 33^3 LUT, applied trilinearly.

Scale: 1 cell = 12.5 m  (160-cell channel ~= 2 km cloud base).
"""

import numpy as np
import sys
sys.path.insert(0, "/home/claude/lightning")
from colorimetry import cie_xyz_bar, planck, _XYZ_TO_SRGB
from render3d import camera, world_points

CELL_M = 12.5
SIGMA_R550 = 1.2e-5 * CELL_M      # Rayleigh @550nm, per cell  (sea level)
SIGMA_M = 2.5e-4 * CELL_M         # aerosol (light storm haze), per cell
HG_G = 0.76
T_CHANNEL = 30000.0
N_BANDS = 8
ALBEDO_GROUND = 0.07
F0_WATER = 0.02
R_CORE0 = 0.30                    # cells; luminous envelope radius scale


# ----------------------------------------------------------------------
# spectral setup
# ----------------------------------------------------------------------

def spectral_bands(n=N_BANDS):
    edges = np.linspace(380.0, 740.0, n + 1)
    lam = 0.5 * (edges[:-1] + edges[1:])
    dlam = np.diff(edges)
    S = planck(lam * 1e-9, T_CHANNEL)
    xb, yb, zb = cie_xyz_bar(lam)
    XYZw = np.stack([xb * dlam, yb * dlam, zb * dlam], axis=0)  # 3 x n
    S = S / (S * XYZw[1]).sum()         # normalize to unit luminance
    sigR = SIGMA_R550 * (550.0 / lam) ** 4
    return lam, S, XYZw, sigR


def phase_rayleigh(cos_t, xp):
    return 3.0 / (16.0 * np.pi) * (1.0 + cos_t ** 2)


def phase_hg(cos_t, xp, g=HG_G):
    return (1.0 - g * g) / (4.0 * np.pi
                            * (1.0 + g * g - 2.0 * g * cos_t) ** 1.5)


# ----------------------------------------------------------------------
# Stage RT1: exact ray-capsule intersection (vectorized, agnostic)
# ----------------------------------------------------------------------

def capsule_intersect(ro, rd, A, B, R, xp, block=512, pix_chunk=8192):
    """Nearest positive hit distance per ray over all capsules.

    Memory-bounded: outer loop over pixel chunks (pix_chunk), inner loop
    over segment blocks (block); peak temporary is pix_chunk*block*3
    floats.  Raise pix_chunk on GPU (e.g. 1<<18) for throughput.
    """
    P = ro.shape[0]
    t_best = xp.full(P, xp.inf, dtype=ro.dtype)
    i_best = xp.full(P, -1, dtype=xp.int64)
    for p0 in range(0, P, pix_chunk):
        sl = slice(p0, min(p0 + pix_chunk, P))
        t_b, i_b = _capsule_intersect_chunk(ro[sl], rd[sl], A, B, R,
                                            xp, block)
        t_best[sl] = t_b
        i_best[sl] = i_b
    return t_best, i_best


def _capsule_intersect_chunk(ro, rd, A, B, R, xp, block):
    """Nearest positive hit distance per ray over all capsules.

    ro: (P,3) origins, rd: (P,3) unit dirs, A,B: (S,3) endpoints,
    R: (S,) radii.  Returns (t_hit (P,), seg_index (P,)) with inf / -1
    for misses.  Blocked over segments to bound memory.

    Precondition: ray origins lie OUTSIDE all capsules (the near-root
    branch is taken).  The renderer guarantees this -- the camera is
    ~2.2H from the channel and mirror rays originate on the ground
    plane; origins inside a capsule are out of contract (validated as
    such in RT1).
    """
    P = ro.shape[0]
    t_best = xp.full(P, xp.inf, dtype=ro.dtype)
    i_best = xp.full(P, -1, dtype=xp.int64)
    for s0 in range(0, A.shape[0], block):
        a = A[s0:s0 + block][None, :, :]        # 1,S,3
        b = B[s0:s0 + block][None, :, :]
        r = R[s0:s0 + block][None, :]
        ba = b - a
        oa = ro[:, None, :] - a
        baba = (ba * ba).sum(-1)
        bard = (ba * rd[:, None, :]).sum(-1)
        baoa = (ba * oa).sum(-1)
        rdoa = (rd[:, None, :] * oa).sum(-1)
        oaoa = (oa * oa).sum(-1)
        a2 = baba - bard ** 2
        b2 = baba * rdoa - baoa * bard
        c2 = baba * oaoa - baoa ** 2 - r * r * baba
        h = b2 * b2 - a2 * c2
        sq = xp.sqrt(xp.maximum(h, 0.0))
        t = (-b2 - sq) / xp.where(xp.abs(a2) > 1e-12, a2, 1e-12)
        y = baoa + t * bard
        body = (h > 0) & (t > 1e-4) & (y > 0) & (y < baba)
        t_body = xp.where(body, t, xp.inf)
        # caps
        oc = xp.where((y <= 0)[..., None], oa, ro[:, None, :] - b)
        b3 = (rd[:, None, :] * oc).sum(-1)
        c3 = (oc * oc).sum(-1) - r * r
        h2 = b3 * b3 - c3
        tc = -b3 - xp.sqrt(xp.maximum(h2, 0.0))
        cap = (h2 > 0) & (tc > 1e-4)
        t_cap = xp.where(cap, tc, xp.inf)
        t_blk = xp.minimum(t_body, t_cap)
        i_blk = xp.argmin(t_blk, axis=1)
        t_min = xp.take_along_axis(t_blk, i_blk[:, None], 1)[:, 0]
        upd = t_min < t_best
        t_best = xp.where(upd, t_min, t_best)
        i_best = xp.where(upd, i_blk + s0, i_best)
    return t_best, i_best


def capsule_sdf(p, A, B, R, xp):
    """Signed distance to nearest capsule (brute reference for RT1)."""
    pa = p[:, None, :] - A[None]
    ba = (B - A)[None]
    h = xp.clip((pa * ba).sum(-1) / (ba * ba).sum(-1), 0.0, 1.0)
    d = xp.sqrt(((pa - ba * h[..., None]) ** 2).sum(-1)) - R[None]
    return d.min(axis=1)


# ----------------------------------------------------------------------
# Stage RT2: equiangular single-scattering estimator
# ----------------------------------------------------------------------

def equiangular_scatter(ro, rd, t_max, A, B, w_len_cdf, W_total,
                        S, sigR, sigM, sigT, spp, rng, xp):
    """MC single-scattered spectral radiance per ray.

    ro,rd: (P,3); t_max: (P,); A,B: segment endpoints; w_len_cdf:
    cumulative power distribution over segments; S: (n_bands,) emission;
    sigR: (n_bands,) Rayleigh sigma_s; sigM scalar aerosol sigma_s;
    sigT: (n_bands,) extinction.  Returns (P, n_bands).
    """
    P = ro.shape[0]
    nb = S.shape[0]
    acc = xp.zeros((P, nb), dtype=ro.dtype)
    for _ in range(spp):
        # light selection ~ power, point uniform on segment
        u = rng.random(P, dtype=ro.dtype) if xp is not np \
            else rng.random(P).astype(ro.dtype)
        ci = xp.searchsorted(w_len_cdf, u * w_len_cdf[-1])
        ci = xp.clip(ci, 0, A.shape[0] - 1)
        ul = rng.random(P, dtype=ro.dtype) if xp is not np \
            else rng.random(P).astype(ro.dtype)
        y = A[ci] + (B[ci] - A[ci]) * ul[:, None]
        # equiangular sample along the ray
        t0 = ((y - ro) * rd).sum(-1)
        Dv = y - (ro + t0[:, None] * rd)
        D = xp.sqrt((Dv * Dv).sum(-1)) + 1e-9
        tha = xp.arctan(-t0 / D)
        thb = xp.arctan((t_max - t0) / D)
        ut = rng.random(P, dtype=ro.dtype) if xp is not np \
            else rng.random(P).astype(ro.dtype)
        th = tha + (thb - tha) * ut
        t = t0 + D * xp.tan(th)
        r = xp.sqrt(D * D + (t - t0) ** 2)
        # phase angle between ray dir and direction scatter-point -> light
        x = ro + t[:, None] * rd
        toL = (y - x) / r[:, None]
        cos_t = (rd * toL).sum(-1)
        pR = phase_rayleigh(cos_t, xp)[:, None]      # P,1
        pM = phase_hg(cos_t, xp)[:, None]
        dth = (thb - tha)[:, None]
        Tatt = xp.exp(-sigT[None, :] * (t + r)[:, None])
        contrib = (W_total * S[None, :]
                   * (sigR[None, :] * pR + sigM * pM)
                   * Tatt * dth / (4.0 * np.pi * D[:, None]))
        acc += contrib
    return acc / spp


# ----------------------------------------------------------------------
# ground shading
# ----------------------------------------------------------------------

def shade_ground(xg, A, B, mid_w, w_len_cdf, W_total, S, sigT,
                 k_samples, rng, xp):
    """Lambertian direct lighting at ground points xg (G,3)."""
    G = xg.shape[0]
    nb = S.shape[0]
    acc = xp.zeros((G, nb), dtype=xg.dtype)
    for _ in range(k_samples):
        u = rng.random(G, dtype=xg.dtype) if xp is not np \
            else rng.random(G).astype(xg.dtype)
        ci = xp.clip(xp.searchsorted(w_len_cdf, u * w_len_cdf[-1]),
                     0, A.shape[0] - 1)
        ul = rng.random(G, dtype=xg.dtype) if xp is not np \
            else rng.random(G).astype(xg.dtype)
        y = A[ci] + (B[ci] - A[ci]) * ul[:, None]
        d = y - xg
        r2 = (d * d).sum(-1) + 1e-9
        r = xp.sqrt(r2)
        cos_i = xp.clip(d[:, 1] / r, 0.0, 1.0)
        Tatt = xp.exp(-sigT[None, :] * r[:, None])
        acc += (W_total * S[None, :] * Tatt
                * (cos_i / (4.0 * np.pi * r2))[:, None])
    E = acc / k_samples
    return (ALBEDO_GROUND / np.pi) * E


# ----------------------------------------------------------------------
# Stage RT4: AgX-family display transform + 3-D LUT
# ----------------------------------------------------------------------

AGX_ALPHA = 0.12
AGX_S, AGX_X0 = 6.5, 0.57
LOG2_MIN, LOG2_RANGE = -10.0, 16.5


def _sigmoid(x, xp):
    f = 0.5 * (1.0 + xp.tanh(AGX_S * (x - AGX_X0)))
    f0 = 0.5 * (1.0 + np.tanh(AGX_S * (0.0 - AGX_X0)))
    f1 = 0.5 * (1.0 + np.tanh(AGX_S * (1.0 - AGX_X0)))
    return (f - f0) / (f1 - f0)


def _srgb_encode(v, xp):
    return xp.where(v <= 0.0031308, 12.92 * v,
                    1.055 * xp.maximum(v, 1e-10) ** (1 / 2.4) - 0.055)


def display_transform(rgb_lin, xp):
    """Scene-linear sRGB -> display sRGB (AgX-family), shape (...,3)."""
    a = AGX_ALPHA
    m = rgb_lin.mean(axis=-1, keepdims=True)
    inset = (1.0 - a) * rgb_lin + a * m
    x = xp.clip((xp.log2(xp.maximum(inset, 1e-10)) - LOG2_MIN)
                / LOG2_RANGE, 0.0, 1.0)
    y = _sigmoid(x, xp)
    my = y.mean(axis=-1, keepdims=True)
    outset = (y - a * my) / (1.0 - a)
    return xp.clip(_srgb_encode(xp.clip(outset, 0.0, 1.0), xp), 0.0, 1.0)


def bake_lut(n=33):
    """Bake the display transform into an n^3 LUT over shaper space."""
    g = np.linspace(0.0, 1.0, n)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    shaper = np.stack([R, G, B], axis=-1)
    rgb_lin = 2.0 ** (shaper * LOG2_RANGE + LOG2_MIN)
    return display_transform(rgb_lin, np).astype(np.float32)


def apply_lut(rgb_lin, lut, xp):
    """Trilinear LUT application; input scene-linear, output display."""
    n = lut.shape[0]
    s = xp.clip((xp.log2(xp.maximum(rgb_lin, 1e-10)) - LOG2_MIN)
                / LOG2_RANGE, 0.0, 1.0) * (n - 1)
    i0 = xp.clip(xp.floor(s).astype(xp.int64), 0, n - 2)
    f = s - i0
    lutx = xp.asarray(lut)
    out = 0
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                wgt = ((f[..., 0] if dr else 1 - f[..., 0])
                       * (f[..., 1] if dg else 1 - f[..., 1])
                       * (f[..., 2] if db else 1 - f[..., 2]))
                out = out + wgt[..., None] * lutx[i0[..., 0] + dr,
                                                  i0[..., 1] + dg,
                                                  i0[..., 2] + db]
    return out


# ----------------------------------------------------------------------
# full frame
# ----------------------------------------------------------------------

def render_pbr(res, cur, weights=None, azimuth_deg=0.0, out_w=480,
               out_h=270, spp=48, ground_k=12, exposure=220.0,
               f_hd=2100.0, xp=np, seed=1, lut=None, dtype=None, aa=4):
    """Spectral path-traced frame.  Pass xp=cupy on the 3090.

    aa: jittered sub-pixel primary samples per pixel (the capsule core
    is sub-pixel, so single center rays under-sample it); spp scatter
    samples are split across the aa passes; averaging happens in
    scene-linear radiance before the display transform, as a physical
    sensor integrates."""
    dtype = dtype or (xp.float32 if xp is not np else np.float64)
    cells, parent = res["cells"], res["parent"]
    H, W, D = res["H"], res["W"], res["D"]
    I = cur["I"].copy(); I[0] = 1.0
    rel = I / I.max()
    if weights is None:
        weights = rel ** 0.75

    lam, S, XYZw, sigR = spectral_bands()
    sigT = sigR + SIGMA_M
    S = xp.asarray(S, dtype=dtype)
    sigR = xp.asarray(sigR, dtype=dtype)
    sigTx = xp.asarray(sigT, dtype=dtype)

    Pw = world_points(cells, H, W, D)
    A = xp.asarray(Pw[parent[1:]], dtype=dtype)
    B = xp.asarray(Pw[1:], dtype=dtype)
    wseg = xp.asarray(weights[1:], dtype=dtype)
    seg_len = xp.sqrt(((B - A) ** 2).sum(-1))
    Rcap = xp.asarray(R_CORE0 * (0.4 + 0.6 * rel[1:] ** 0.4), dtype=dtype)
    w_len = wseg * seg_len
    w_len_cdf = xp.cumsum(w_len)
    W_total = float(w_len.sum())

    az = np.deg2rad(azimuth_deg)
    Rorb = 2.2 * H
    eye_np, basis_np = camera(np.array([Rorb * np.sin(az), 0.45 * H,
                                        Rorb * np.cos(az)]),
                              (0.0, 0.52 * H, 0.0))
    eye = xp.asarray(eye_np, dtype=dtype)
    basis = xp.asarray(basis_np, dtype=dtype)
    k = out_h / 1080.0
    f = f_hd * k
    rng = xp.random.default_rng(seed)
    npix = out_w * out_h
    L_acc = xp.zeros((npix, N_BANDS), dtype=dtype)
    spp_pass = max(1, spp // aa)

    for a_i in range(aa):
        ju = float(rng.random()) - 0.5 if aa > 1 else 0.0
        jv = float(rng.random()) - 0.5 if aa > 1 else 0.0
        u = (xp.arange(out_w, dtype=dtype) - out_w / 2.0 + 0.5 + ju) / f
        v = (out_h / 2.0 - xp.arange(out_h, dtype=dtype) - 0.5 + jv) / f
        UU, VV = xp.meshgrid(u, v, indexing="xy")
        rd = (UU[..., None] * basis[0] + VV[..., None] * basis[1]
              + basis[2]).reshape(-1, 3)
        rd = rd / xp.sqrt((rd * rd).sum(-1, keepdims=True))
        ro = xp.broadcast_to(eye, (npix, 3))
        L_acc += _render_pass(ro, rd, eye, A, B, Rcap, wseg, w_len_cdf,
                              W_total, S, sigR, sigTx, spp_pass,
                              ground_k, Rorb, rng, xp, dtype)
    L = L_acc / aa

    XYZwx = xp.asarray(XYZw, dtype=dtype)
    XYZ = L @ XYZwx.T
    M = xp.asarray(_XYZ_TO_SRGB, dtype=dtype)
    rgb = xp.maximum(XYZ @ M.T, 0.0) * exposure
    if lut is not None:
        disp = apply_lut(rgb, lut, xp)
    else:
        disp = display_transform(rgb, xp)
    img = (xp.clip(disp, 0, 1) * 255.0 + 0.5).astype(xp.uint8)
    img = img.reshape(out_h, out_w, 3)
    return (np.asarray(img.get()) if xp is not np else img), rgb


def _render_pass(ro, rd, eye, A, B, Rcap, wseg, w_len_cdf, W_total,
                 S, sigR, sigTx, spp, ground_k, Rorb, rng, xp, dtype):
    P = ro.shape[0]
    t_hit, i_hit = capsule_intersect(ro, rd, A, B, Rcap, xp)
    t_ground = xp.where(rd[:, 1] < -1e-6, -eye[1] / rd[:, 1], xp.inf)
    t_max = xp.minimum(xp.minimum(t_hit, t_ground), 6.0 * Rorb)

    L = xp.zeros((P, N_BANDS), dtype=dtype)

    hit = t_hit < xp.minimum(t_ground, 6.0 * Rorb)
    ih = xp.where(hit, i_hit, 0)
    L_e = (wseg[ih] / (2.0 * np.pi ** 2 * Rcap[ih]))[:, None] * S[None, :]
    L += xp.where(hit[:, None],
                  L_e * xp.exp(-sigTx[None, :] * t_hit[:, None]), 0.0)

    g = (t_ground < xp.minimum(t_hit, 6.0 * Rorb))
    if bool(g.any()):
        gi = xp.where(g)[0]
        xg = ro[gi] + t_ground[gi, None] * rd[gi]
        Lg = shade_ground(xg, A, B, wseg, w_len_cdf, W_total, S, sigTx,
                          ground_k, rng, xp)
        rr = rd[gi].copy()
        rr[:, 1] = -rr[:, 1]
        t_m, i_m = capsule_intersect(xg, rr, A, B, Rcap, xp)
        mhit = xp.isfinite(t_m)
        im = xp.where(mhit, i_m, 0)
        cos_i = xp.clip(-rd[gi, 1], 0.0, 1.0)
        F = F0_WATER + (1.0 - F0_WATER) * (1.0 - cos_i) ** 5
        L_m = (wseg[im] / (2.0 * np.pi ** 2 * Rcap[im]))[:, None] \
            * S[None, :] * xp.exp(-sigTx[None, :]
                                  * xp.where(mhit, t_m, 0.0)[:, None])
        Lg = Lg + xp.where(mhit[:, None], F[:, None] * L_m, 0.0)
        Tcam = xp.exp(-sigTx[None, :] * t_ground[gi, None])
        Lfull = xp.zeros_like(L)
        Lfull[gi] = Lg * Tcam
        L += Lfull

    L += equiangular_scatter(ro, rd, t_max, A, B, w_len_cdf, W_total,
                             S, sigR, SIGMA_M, sigTx, spp, rng, xp)
    return L
