"""
temporal.py -- time-resolved luminosity model for the 3-D strike.

Physical sequence of a negative cloud-to-ground flash:
  1. Stepped leader: the channel extends in steps toward ground at
     ~2e5 m/s, faintly luminous (~1e-2 of return-stroke radiance) with a
     bright advancing tip.  Our DBM growth order IS this clock: cell c
     becomes visible when the animation reaches its growth index.
  2. Attachment & return stroke: a luminous front propagates UP the
     channel from the ground contact at ~1e8 m/s (~c/3).  A cell
     activates when the front has covered its tree-path distance from
     the attachment point (exact tree metric, validated Stage 3 tree).
  3. Decay: channel luminosity falls exponentially with constant tau.

Animation time t in [0, 1] is a documented nonuniform warp of physical
time (the leader lasts ~20 ms, the return stroke ~70 us; rendered 1:1
the stroke would occupy a single frame).

  t in [0,   T1): leader phase, growth index g(t) = n*(t/T1)^2
                  (quadratic ramp mirrors the observed acceleration of
                  the tip as it approaches ground)
  t in [T1, T2): return stroke, front distance s(t) linear in t
  t in [T2, 1 ]: decay, exp(-(t-T2)/tau)
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, "/home/claude/lightning")
from colorimetry import blackbody_rgb
from render import value_noise, aces
from render3d import camera, project, world_points

T1, T2, TAU = 0.55, 0.63, 0.12
LEADER_LEVEL = 0.012          # leader radiance relative to return stroke
TIP_BOOST = 0.30              # advancing-tip brightness
TIP_FRACTION = 0.02           # trailing fraction of cells forming the tip

_EDGE_LEN = {1: 1.0, 2: np.sqrt(2.0), 3: np.sqrt(3.0)}


def activation_distances(cells, parent, main, attach):
    """Tree-path distance from the attachment cell to every cell.

    dist_root via parent chains; the attachment's path to the root is the
    main channel, so LCA(attach, c) is c's junction onto the main path:
        d(c) = (dist_root[attach] - dist_root[junc])
             + (dist_root[c]      - dist_root[junc]).
    """
    n = len(parent)
    dist_root = np.zeros(n)
    for c in range(1, n):
        step = int((np.abs(cells[c] - cells[parent[c]]) ** 2).sum())
        dist_root[c] = dist_root[parent[c]] + _EDGE_LEN[step]

    junc = np.full(n, -1, dtype=np.int64)
    for c in range(n):
        a = c
        while not main[a]:
            a = parent[a]
        junc[c] = a

    d = (dist_root[attach] - dist_root[junc]) + (dist_root - dist_root[junc])
    return d, dist_root


def cell_weights(t, n, rel, d_act, d_max):
    """Full per-cell luminosity factor at animation time t.

    Contract: the rendered line radiance is weights[c] * (camera depth
    terms).  At full return-stroke illumination weights = rel^0.75,
    which reproduces the validated HD still exactly.
    """
    idx = np.arange(n)
    w = np.zeros(n)
    leader_glow = LEADER_LEVEL * rel ** 0.3
    if t < T1:
        g = n * (t / T1) ** 2
        born = idx <= g
        w[born] = leader_glow[born]
        tip = born & (idx >= g - TIP_FRACTION * n)
        w[tip] += TIP_BOOST
    else:
        s = d_max * min((t - T1) / (T2 - T1), 1.0)
        lit = d_act <= s
        w = np.where(lit, rel ** 0.75, leader_glow)
        if t > T2:
            w = w * np.exp(-(t - T2) / TAU)
    return w


def render_frame(res, cur, t, azimuth_deg=0.0, out_w=960, out_h=540,
                 exposure=15.0, f_hd=2100.0, seed=4, weights=None):
    """Render one animation frame at arbitrary resolution.

    All length-scale parameters (focal length, line widths, PSF sigmas)
    scale with `k = out_h / 1080` so a frame is resolution-consistent
    with the validated HD still.
    """
    cells, parent = res["cells"], res["parent"]
    H, W, D = res["H"], res["W"], res["D"]
    I = cur["I"].copy()
    I[0] = 1.0
    rel = I / I.max()
    n = len(cells)

    if weights is None:
        d_act, _ = activation_distances(cells, parent, cur["main"],
                                        res["attach"])
        weights = cell_weights(t, n, rel, d_act, d_act.max())

    k = out_h / 1080.0
    f = f_hd * k
    cx, cy = out_w / 2.0, out_h / 2.0

    P = world_points(cells, H, W, D)
    az = np.deg2rad(azimuth_deg)
    R = 2.2 * H
    eye = np.array([R * np.sin(az), 0.45 * H, R * np.cos(az)])
    eye, basis = camera(eye, (0.0, 0.52 * H, 0.0))
    u, v, z = project(P, eye, basis, f, cx, cy)
    z_ref = R
    L_ext = 4.0 * R

    lum = np.zeros((out_h, out_w), np.float32)
    order = np.argsort(weights[1:]) + 1
    for c in order:
        wc = weights[c]
        if wc <= 1e-4:
            continue
        p = parent[c]
        zm = 0.5 * (z[c] + z[p])
        L = wc * (z_ref / zm) * np.exp(-(zm - z_ref) / L_ext)
        wpx = max(1, int(round((1 + 3.0 * rel[c] ** 0.4) * z_ref / zm * k)))
        cv2.line(lum, (int(round(u[p])), int(round(v[p]))),
                 (int(round(u[c])), int(round(v[c]))),
                 float(L), thickness=wpx, lineType=cv2.LINE_AA)

    glow = np.zeros_like(lum)
    for sigma, wgt in ((3, 0.45), (9, 0.25), (27, 0.18), (80, 0.12)):
        glow += wgt * cv2.GaussianBlur(lum, (0, 0), max(sigma * k, 0.8))

    rgb_hot, _ = blackbody_rgb(30000.0)
    rgb_halo, _ = blackbody_rgb(30000.0, rayleigh=True)

    img = np.zeros((out_h, out_w, 3), np.float32)
    yy = np.linspace(0, 1, out_h, dtype=np.float32)[:, None]
    img += (0.0035 + 0.004 * (1 - yy) ** 2)[..., None] \
        * np.array([0.45, 0.55, 1.0], np.float32)

    clouds = value_noise(out_h, out_w,
                         octaves=tuple(max(int(s * k), 4)
                                       for s in (160, 80, 40, 20)),
                         seed=seed)
    cloud_band = np.clip(1.0 - yy / 0.40, 0, 1) ** 1.6
    cloud_density = np.clip(clouds - 0.42, 0, 1) * cloud_band
    cloud_light = cv2.GaussianBlur(lum, (0, 0), 120 * k) * 70.0 + 0.05
    img += (cloud_density * cloud_light)[..., None] \
        * np.array([0.55, 0.62, 0.95], np.float32) * 0.45
    img += float(lum.mean()) * 1.5 * np.array([0.5, 0.58, 1.0], np.float32)

    _, v_g, _ = project(np.array([[0.0, 0.0, 0.0]]), eye, basis, f, cx, cy)
    horizon = int(np.clip(v_g[0] + 4 * k, 0, out_h - 2))
    gm = np.zeros((out_h, out_w), np.float32)
    gm[horizon:, :] = 1.0
    gm = cv2.GaussianBlur(gm, (0, 0), max(3 * k, 0.8))
    gglow = cv2.GaussianBlur(lum, (0, 0), 60 * k)[max(horizon - 1, 0), :] * 4
    img *= (1.0 - gm[..., None] * 0.985)
    img[horizon:, :, :] += (gglow[None, :, None]
                            * np.array([0.5, 0.6, 1.0]) * 0.02)

    img += exposure * lum[..., None] * rgb_hot[None, None, :]
    img += 0.9 * exposure * glow[..., None] * rgb_halo[None, None, :]
    img = np.power(aces(img), 1.0 / 2.2)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8), lum


def write_video(res, cur, path, n_frames=150, fps=30, out_w=960,
                out_h=540, azimuth_fn=None, log=None):
    """Render the full timeline to MP4 (OpenCV VideoWriter)."""
    d_act, _ = activation_distances(res["cells"], res["parent"],
                                    cur["main"], res["attach"])
    I = cur["I"].copy()
    I[0] = 1.0
    rel = I / I.max()
    n = len(res["cells"])

    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (out_w, out_h))
    for fi in range(n_frames):
        t = fi / (n_frames - 1)
        az = azimuth_fn(t) if azimuth_fn else 0.0
        w = cell_weights(t, n, rel, d_act, d_act.max())
        frame, _ = render_frame(res, cur, t, azimuth_deg=az,
                                out_w=out_w, out_h=out_h, weights=w)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if log and fi % 25 == 0:
            log(f"  frame {fi}/{n_frames}")
    vw.release()
