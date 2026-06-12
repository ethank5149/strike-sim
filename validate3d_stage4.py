"""Stage 4 (3-D) validation: rendered views.

  (1)-(4): same image-physics checks as the 2-D pipeline (format,
           dynamic range, achromatic saturated core, Rayleigh-blue halo).
  (5) Parallax: the SAME channel rendered from azimuths 0 deg and 50 deg
      must produce decorrelated luminance maps (Pearson r well below 1).
      A flat (2-D) object would be invariant up to horizontal scaling;
      genuine depth structure breaks that.  Require r < 0.80, while both
      views keep > 95% of their luminous energy inside the frame.
"""

import numpy as np
import cv2
import sys

def image_checks(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED),
                       cv2.COLOR_BGR2RGB)
    ok = True
    fmt = img.shape == (1080, 1920, 3) and img.dtype == np.uint8
    lum = img.astype(np.float32).mean(axis=2)
    clipped = float((lum >= 254.5).mean())
    dark = float((lum < 16).mean())
    # background criterion: corner patches (far from the channel) must be
    # night-dark; global dark fraction depends on the object's projected
    # aspect, so it only gets a loose floor.
    cs = 120
    # bottom corners only: top corners contain the lit cloud deck, which
    # is scene content; bottom corners are empty night background.
    corners = [lum[-cs:, :cs], lum[-cs:, -cs:]]
    corner_mean = float(np.mean([c.mean() for c in corners]))
    rng_ok = (img.max() == 255 and img.min() <= 5
              and 0 < clipped < 0.02 and dark > 0.3 and corner_mean < 16)
    core = img[lum >= 254.5].astype(np.float32)
    spread = float(np.abs(core - core.mean(axis=1, keepdims=True)).max()) \
        if len(core) else 99.0
    halo = img[(lum > 40) & (lum < 180)].astype(np.float32)
    r, g, b = halo.mean(axis=0)
    halo_ok = b > g > r
    print(f"  {path.split('/')[-1]}: format {'PASS' if fmt else 'FAIL'}; "
          f"range (sat {100*clipped:.2f}%, dark {100*dark:.0f}%, "
          f"corner bg {corner_mean:.1f}/255) "
          f"{'PASS' if rng_ok else 'FAIL'}; "
          f"core spread {spread:.1f}/255 {'PASS' if spread <= 2 else 'FAIL'}; "
          f"halo RGB ({r:.0f},{g:.0f},{b:.0f}) "
          f"{'PASS' if halo_ok else 'FAIL'}")
    return ok and fmt and rng_ok and spread <= 2 and halo_ok


if __name__ == "__main__":
    print("Stage 4 (3-D) validation -- HD renders")
    ok = True
    ok &= image_checks("/home/claude/lightning/lightning3d_hd.png")
    ok &= image_checks("/home/claude/lightning/lightning3d_az50.png")

    l0 = np.load("/home/claude/lightning/lum_az0.npy").ravel()
    l5 = np.load("/home/claude/lightning/lum_az50.npy").ravel()
    r = float(np.corrcoef(l0, l5)[0, 1])
    par_ok = r < 0.80
    print(f"  parallax decorrelation az 0 vs 50 deg: r = {r:.3f} "
          f"({'PASS' if par_ok else 'FAIL'}, threshold < 0.80)")
    ok &= par_ok

    for name in ("lum_az0.npy", "lum_az50.npy"):
        lum = np.load(f"/home/claude/lightning/{name}")
        inner = lum[:, 100:-100].sum() / max(lum.sum(), 1e-9)
        good = inner > 0.95
        print(f"  {name}: {100*inner:.1f}% of flux in frame "
              f"({'PASS' if good else 'FAIL'})")
        ok &= good

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
