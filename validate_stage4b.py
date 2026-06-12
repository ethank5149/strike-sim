"""Stage 4b validation: rendered HD image.

  (1) Format: 1920x1080, 3-channel, 8-bit.
  (2) Dynamic range: full [0, 255] used; clipped (255-white) pixels are a
      small fraction (core only), background genuinely dark.
  (3) Core saturation physics: brightest pixels are achromatic white
      (sensor saturation), as in real lightning photographs.
  (4) Halo physics: glow pixels around the channel are blue-dominant
      (Rayleigh weighting), B > G > R in the mean.
"""

import numpy as np
import cv2
import sys

if __name__ == "__main__":
    img_bgr = cv2.imread("/home/claude/lightning/lightning_hd.png",
                         cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print("Stage 4b validation -- HD render")
    ok = True

    # (1) format
    fmt = img.shape == (1080, 1920, 3) and img.dtype == np.uint8
    print(f"  format {img.shape} {img.dtype}: {'PASS' if fmt else 'FAIL'}")
    ok &= fmt

    # (2) dynamic range
    lum = img.astype(np.float32).mean(axis=2)
    clipped = float((lum >= 254.5).mean())
    dark = float((lum < 16).mean())
    rng_ok = (img.max() == 255 and img.min() <= 5
              and 0 < clipped < 0.02 and dark > 0.5)
    print(f"  min={img.min()} max={img.max()}  "
          f"saturated: {100*clipped:.2f}% of frame  "
          f"dark (<16): {100*dark:.1f}%  {'PASS' if rng_ok else 'FAIL'}")
    ok &= rng_ok

    # (3) core is achromatic white at saturation
    core = img[lum >= 254.5].astype(np.float32)
    spread = float(np.abs(core - core.mean(axis=1, keepdims=True)).max())
    core_ok = spread <= 2.0
    print(f"  core achromaticity: max channel spread {spread:.1f}/255 "
          f"({'PASS' if core_ok else 'FAIL'})")
    ok &= core_ok

    # (4) halo is blue-dominant
    halo_mask = (lum > 40) & (lum < 180)
    halo = img[halo_mask].astype(np.float32)
    r, g, b = halo[:, 0].mean(), halo[:, 1].mean(), halo[:, 2].mean()
    halo_ok = b > g > r
    print(f"  halo mean RGB = ({r:.1f}, {g:.1f}, {b:.1f}); "
          f"B > G > R (Rayleigh): {'PASS' if halo_ok else 'FAIL'}")
    ok &= halo_ok

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
