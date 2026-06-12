"""Stage 4a validation: spectral colorimetry pipeline.

Reference values (CIE 1931 Planckian locus, standard tables):
    T = 3000 K  -> (x, y) ~= (0.4369, 0.4041)
    T = 6500 K  -> (x, y) ~= (0.3135, 0.3237)
    T = 10000 K -> (x, y) ~= (0.2807, 0.2884)
Tolerance 0.005 in each coordinate (the CMF fits are ~1% accurate).

Physics sanity:
    - hotter blackbody => bluer (monotonically decreasing x);
    - Rayleigh-weighted spectrum is bluer than the direct spectrum
      at the same temperature.
"""

import numpy as np
import sys
sys.path.insert(0, "/home/claude/lightning")
from colorimetry import blackbody_rgb, chromaticity

REF = {3000.0: (0.4369, 0.4041),
       6500.0: (0.3135, 0.3237),
       10000.0: (0.2807, 0.2884)}

if __name__ == "__main__":
    print("Stage 4a validation -- Planck -> CIE -> sRGB colorimetry")
    ok = True
    xs = {}
    for T, (xr, yr) in REF.items():
        _, XYZ = blackbody_rgb(T)
        x, y = chromaticity(XYZ)
        good = abs(x - xr) < 0.005 and abs(y - yr) < 0.005
        ok &= good
        xs[T] = x
        print(f"  T={T:7.0f} K: (x,y)=({x:.4f},{y:.4f})  "
              f"ref=({xr:.4f},{yr:.4f})  {'PASS' if good else 'FAIL'}")

    # monotone blue shift with temperature
    Ts = [3000, 6500, 10000, 20000, 30000]
    xvals = []
    for T in Ts:
        _, XYZ = blackbody_rgb(float(T))
        xvals.append(chromaticity(XYZ)[0])
    mono = all(a > b for a, b in zip(xvals, xvals[1:]))
    print(f"  chromaticity x monotone decreasing with T: "
          f"{'PASS' if mono else 'FAIL'}")
    ok &= mono

    # Rayleigh halo bluer than direct light at 30 kK
    rgb_d, XYZ_d = blackbody_rgb(30000.0)
    rgb_r, XYZ_r = blackbody_rgb(30000.0, rayleigh=True)
    xd, _ = chromaticity(XYZ_d)
    xr_, _ = chromaticity(XYZ_r)
    blue = xr_ < xd
    print(f"  Rayleigh-scattered halo bluer than channel "
          f"(x: {xr_:.4f} < {xd:.4f}): {'PASS' if blue else 'FAIL'}")
    ok &= blue

    print(f"  channel color (30 kK, linear sRGB, max-normalized): "
          f"({rgb_d[0]:.3f}, {rgb_d[1]:.3f}, {rgb_d[2]:.3f})")
    print(f"  halo color    (30 kK + Rayleigh):                   "
          f"({rgb_r[0]:.3f}, {rgb_r[1]:.3f}, {rgb_r[2]:.3f})")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
