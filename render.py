"""
render.py -- HD (1920x1080) physically based composite of the strike.

Radiometric model
  * Segment radiance  L ~ I^0.75  (channel luminosity grows sublinearly
    with return-stroke current; the 3 orders of magnitude of current
    contrast between main channel and corona twigs produces the
    photographic appearance -- faint streamers fade out by radiometry,
    not by pruning).
  * Channel core width grows weakly with current (w ~ I^0.4).
  * Atmospheric point-spread function: the long-tailed glow from
    molecular + droplet scattering is approximated by a sum of Gaussians
    of geometrically increasing width (a standard approximation of a
    power-law-winged PSF).
  * Direct channel light: 30,000 K blackbody.  Halo: same spectrum
    weighted by the Rayleigh lambda^-4 cross-section (bluer).
  * Camera: exposure such that the core saturates (as in real photos),
    ACES filmic tonemap, sRGB gamma.
"""

import numpy as np
import cv2
import pickle
import sys
sys.path.insert(0, "/home/claude/lightning")
from colorimetry import blackbody_rgb

HD_W, HD_H = 1920, 1080


def build_luminance(cells, parent, I, H, W):
    """Rasterize the channel tree into a float32 HD luminance map."""
    scale = HD_H / H
    x_off = (HD_W - W * scale) / 2.0

    def to_px(c):
        i, j = c
        return (int(round(x_off + (j + 0.5) * scale)),
                int(round((i + 0.5) * scale)))

    Imax = I.max()
    lum = np.zeros((HD_H, HD_W), np.float32)
    order = np.argsort(I[1:]) + 1          # draw faint first, bright on top
    for c in order:
        rel = I[c] / Imax
        L = rel ** 0.75
        w = max(1, int(round(1 + 3.0 * rel ** 0.4)))
        cv2.line(lum, to_px(cells[parent[c]]), to_px(cells[c]),
                 float(L), thickness=w, lineType=cv2.LINE_AA)
    return lum


def atmospheric_psf(lum):
    """Sum-of-Gaussians approximation of the scattering PSF."""
    glow = np.zeros_like(lum)
    for sigma, wgt in ((3, 0.45), (9, 0.25), (27, 0.18), (80, 0.12)):
        glow += wgt * cv2.GaussianBlur(lum, (0, 0), sigma)
    return glow


def value_noise(h, w, octaves=(160, 80, 40, 20), seed=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for s in octaves:
        n = rng.random(((h // s) + 2, (w // s) + 2)).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * n
        tot += amp
        amp *= 0.55
    return out / tot


def aces(x):
    """Narkowicz ACES filmic approximation (per linear channel)."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip(x * (a * x + b) / (x * (c * x + d) + e), 0.0, 1.0)


def render(leader_pkl, currents_pkl, out_png, exposure=14.0, seed=4):
    with open(leader_pkl, "rb") as f:
        res = pickle.load(f)
    with open(currents_pkl, "rb") as f:
        cur = pickle.load(f)

    cells, parent = res["cells"], res["parent"]
    I = cur["I"].copy()
    I[0] = 1.0

    lum = build_luminance(cells, parent, I, res["H"], res["W"])
    glow = atmospheric_psf(lum)

    rgb_hot, _ = blackbody_rgb(30000.0)               # direct channel
    rgb_halo, _ = blackbody_rgb(30000.0, rayleigh=True)  # scattered halo

    img = np.zeros((HD_H, HD_W, 3), np.float32)

    # --- background: night sky gradient + cloud deck lit by the strike
    yy = np.linspace(0, 1, HD_H, dtype=np.float32)[:, None]
    sky = (0.0035 + 0.004 * (1 - yy) ** 2)            # darker toward ground
    sky_rgb = np.array([0.45, 0.55, 1.0], np.float32)  # cold night blue
    img += sky[..., None] * sky_rgb

    clouds = value_noise(HD_H, HD_W, seed=seed)
    cloud_band = np.clip(1.0 - yy / 0.42, 0, 1) ** 1.6   # top ~40%
    cloud_density = np.clip(clouds - 0.42, 0, 1) * cloud_band
    cloud_light = cv2.GaussianBlur(lum, (0, 0), 120) * 70.0 + 0.05
    img += (cloud_density * cloud_light)[..., None] \
        * np.array([0.55, 0.62, 0.95], np.float32) * 0.45
    # whole-scene flash: a fraction of total channel flux scattered back
    flash = float(lum.mean()) * 1.5
    img += flash * np.array([0.5, 0.58, 1.0], np.float32)

    # --- ground silhouette with horizon glow near the attachment
    horizon = int(HD_H * 0.94)
    ground_mask = np.zeros((HD_H, HD_W), np.float32)
    ground_mask[horizon:, :] = 1.0
    ground_mask = cv2.GaussianBlur(ground_mask, (0, 0), 3)
    ground_glow = cv2.GaussianBlur(lum, (0, 0), 60)[horizon - 1, :] * 4.0
    img *= (1.0 - ground_mask[..., None] * 0.985)
    img[horizon:, :, :] += (ground_glow[None, :, None]
                            * np.array([0.5, 0.6, 1.0]) * 0.02)

    # --- the strike itself (linear HDR addition)
    img += exposure * lum[..., None] * rgb_hot[None, None, :]
    img += 0.9 * exposure * glow[..., None] * rgb_halo[None, None, :]

    # --- camera: ACES tonemap + sRGB gamma
    img = aces(img)
    img = np.power(img, 1.0 / 2.2)
    out = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(out_png, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out, lum, glow


if __name__ == "__main__":
    out, lum, glow = render("/home/claude/lightning/leader.pkl",
                            "/home/claude/lightning/currents.pkl",
                            "/home/claude/lightning/lightning_hd.png")
    print("rendered", out.shape, out.dtype)
