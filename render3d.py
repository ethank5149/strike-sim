"""
render3d.py -- pinhole-camera HD composite of the 3-D strike.

Geometry: world units = grid cells.  y is height (y = H-1-i), x,z lateral
and centered.  Camera orbits the channel at radius ~2.2 H, slightly below
mid-height (typical storm-photography geometry).

Radiometry vs the 2-D renderer:
  * the channel core is narrower than a pixel, so the flux collected per
    unit projected length scales as 1/z (width compression), not 1/z^2;
  * projected core width w ~ f * w_phys(I) / z;
  * aerial perspective: extinction exp(-(z - z_ref)/L_ext) on top of the
    Rayleigh-blue halo (validated colorimetry reused unchanged).
"""

import numpy as np
import cv2
import pickle
import sys
sys.path.insert(0, "/home/claude/lightning")
from colorimetry import blackbody_rgb
from render import atmospheric_psf, value_noise, aces

HD_W, HD_H = 1920, 1080


def camera(eye, target, up=(0, 1, 0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    upv = np.cross(right, fwd)
    return eye, np.stack([right, upv, fwd])


def project(P, eye, basis, f, cx, cy):
    d = P - eye
    xc, yc, zc = (d @ basis.T).T
    u = cx + f * xc / zc
    v = cy - f * yc / zc
    return u, v, zc


def world_points(cells, H, W, D):
    i, j, k = cells[:, 0], cells[:, 1], cells[:, 2]
    return np.stack([j - W / 2.0, (H - 1.0) - i, k - D / 2.0], axis=1)


def render_view(res, cur, azimuth_deg=0.0, exposure=15.0, f=2100.0,
                seed=4):
    cells, parent = res["cells"], res["parent"]
    H, W, D = res["H"], res["W"], res["D"]
    I = cur["I"].copy()
    I[0] = 1.0
    Imax = I.max()

    P = world_points(cells, H, W, D)
    az = np.deg2rad(azimuth_deg)
    R = 2.2 * H
    eye = np.array([R * np.sin(az), 0.45 * H, R * np.cos(az)])
    eye, basis = camera(eye, (0.0, 0.52 * H, 0.0))
    cx, cy = HD_W / 2.0, HD_H / 2.0
    u, v, z = project(P, eye, basis, f, cx, cy)
    z_ref = R

    lum = np.zeros((HD_H, HD_W), np.float32)
    order = np.argsort(I[1:]) + 1
    L_ext = 4.0 * R
    for c in order:
        p = parent[c]
        rel = I[c] / Imax
        zm = 0.5 * (z[c] + z[p])
        depth = (z_ref / zm)                      # 1/z flux compression
        ext = np.exp(-(zm - z_ref) / L_ext)       # aerial extinction
        L = rel ** 0.75 * depth * ext
        w = max(1, int(round((1 + 3.0 * rel ** 0.4) * z_ref / zm)))
        cv2.line(lum, (int(round(u[p])), int(round(v[p]))),
                 (int(round(u[c])), int(round(v[c]))),
                 float(L), thickness=w, lineType=cv2.LINE_AA)

    glow = atmospheric_psf(lum)
    rgb_hot, _ = blackbody_rgb(30000.0)
    rgb_halo, _ = blackbody_rgb(30000.0, rayleigh=True)

    img = np.zeros((HD_H, HD_W, 3), np.float32)
    yy = np.linspace(0, 1, HD_H, dtype=np.float32)[:, None]
    img += (0.0035 + 0.004 * (1 - yy) ** 2)[..., None] \
        * np.array([0.45, 0.55, 1.0], np.float32)

    clouds = value_noise(HD_H, HD_W, seed=seed)
    cloud_band = np.clip(1.0 - yy / 0.40, 0, 1) ** 1.6
    cloud_density = np.clip(clouds - 0.42, 0, 1) * cloud_band
    cloud_light = cv2.GaussianBlur(lum, (0, 0), 120) * 70.0 + 0.05
    img += (cloud_density * cloud_light)[..., None] \
        * np.array([0.55, 0.62, 0.95], np.float32) * 0.45
    img += float(lum.mean()) * 1.5 * np.array([0.5, 0.58, 1.0], np.float32)

    # ground: horizon at the projected base of the channel
    _, v_g, _ = project(np.array([[0.0, 0.0, 0.0]]), eye, basis, f, cx, cy)
    horizon = int(np.clip(v_g[0] + 4, 0, HD_H - 2))
    gm = np.zeros((HD_H, HD_W), np.float32)
    gm[horizon:, :] = 1.0
    gm = cv2.GaussianBlur(gm, (0, 0), 3)
    gglow = cv2.GaussianBlur(lum, (0, 0), 60)[max(horizon - 1, 0), :] * 4.0
    img *= (1.0 - gm[..., None] * 0.985)
    img[horizon:, :, :] += (gglow[None, :, None]
                            * np.array([0.5, 0.6, 1.0]) * 0.02)

    img += exposure * lum[..., None] * rgb_hot[None, None, :]
    img += 0.9 * exposure * glow[..., None] * rgb_halo[None, None, :]

    img = np.power(aces(img), 1.0 / 2.2)
    out = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return out, lum


if __name__ == "__main__":
    with open("/home/claude/lightning/leader3d.pkl", "rb") as f:
        res = pickle.load(f)
    with open("/home/claude/lightning/currents3d.pkl", "rb") as f:
        cur = pickle.load(f)
    for az, name in ((0.0, "lightning3d_hd.png"),
                     (50.0, "lightning3d_az50.png")):
        out, lum = render_view(res, cur, azimuth_deg=az)
        cv2.imwrite(f"/home/claude/lightning/{name}",
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        np.save(f"/home/claude/lightning/lum_az{int(az)}.npy", lum)
        print("rendered", name, out.shape)
