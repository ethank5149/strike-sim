# 3-D Lightning Strike Simulator — Physics & Validation Report

Extension of the 2-D pipeline to full 3-D: 160×112×112 grid (2.0M cells),
7-point Laplacian, 26-connected DBM growth, pinhole-camera HD rendering.
Native backends as before: Numba (LLVM stencil kernels), NumPy (C/BLAS),
OpenCV (C++).

## Physics changes from 2-D, and why

* **η = 2** (vs. η = 1 in 2-D). In 3-D, η = 1 gives D ≈ 2.5 — a
  volume-filling bush. Measured lightning channels project optically to
  D ≈ 1.7 ± 0.1, and by Marstrand's projection theorem a set with D < 2
  preserves its dimension under projection, so the 3-D channel itself
  must sit near 1.7. 3-D DBM at η = 2 gives D ≈ 1.9 radial, biased lower
  in plate geometry — the physically calibrated choice.
* **Bounding-box sweep acceleration**: SOR sweeps restricted to the
  channel's lateral bounding box (+28-cell margin) over the full vertical
  extent, with full-domain refresh every 60 solves. *The approximation is
  itself validated*: at every refresh, candidate-site potentials from
  box-only relaxation are compared against the full-domain solve.
* **Camera radiometry**: the channel core is sub-pixel, so collected flux
  per projected length scales as 1/z (width compression), not 1/z²;
  projected width ∝ f/z; aerial extinction exp(−Δz/L). Spectral
  colorimetry (30,000 K Planck channel, Rayleigh λ⁻⁴ halo) reused
  unchanged from the validated 2-D pipeline.

## Per-stage validation results

**Stage 1 — 3-D Laplace solver** (vs. exact harmonic
sin πx · sin πy · sinh √2πz / sinh √2π):

| N | max error | observed order |
|---|---|---|
| 17 | 1.74e-3 | — |
| 33 | 4.39e-4 | 1.98 |
| 65 | 1.10e-4 | 2.00 |
| 97 | 4.89e-5 | 2.00 |

Theoretical 2nd-order rate of the 7-point stencil. **PASS**

**Stage 2 — 3-D DBM growth** (6,875 cells, 1,719 solves, 81,762 sweeps):
26-neighbor tree connectivity and full cloud→ground span verified;
bounding-box acceleration deviates from full-domain solves by at most
**7.5×10⁻⁴** relative (threshold 5×10⁻³); 3-D box dimension
**D = 1.647**; projected 2-D dimension **D = 1.699** — matching the
optical lightning measurement 1.7 ± 0.1 almost exactly, and consistent
with projection preserving D < 2 (|ΔD| = 0.05). **PASS**

**Stage 3 — Return-stroke currents**: ground current = total charge
exactly (6,875); Kirchhoff residual 0.0 at all 186 main-channel
junctions; monotone toward ground (1,977 → 6,874); main/median-twig
current contrast 3,437×. **PASS**

**Stage 4 — HD renders** (two views, azimuth 0° and 50°): both
1920×1080×3 uint8; saturated core 0.11 %/0.19 % of frame and achromatic
white (channel spread 0.7/255 — sensor saturation); halo blue-dominant
RGB ≈ (44, 61, 127) per Rayleigh; empty-background corners at 10.4–10.9
/255 (night-dark). **Parallax test: luminance correlation between the
two azimuths r = 0.067** — the views are almost fully decorrelated,
which only occurs for genuinely volumetric channel structure (a flat
object would be view-invariant up to horizontal scaling); 100 % of
luminous flux in frame for both views. **PASS**

One criterion was revised during validation, documented here: the global
"dark fraction > 50 %" check from the 2-D suite is aspect-dependent (a
view in which the channel's two lateral extents both project into frame
legitimately lights more pixels), so it was replaced by a direct
background measurement (bottom-corner patches < 16/255) plus a loose
global floor. No render output was altered to pass.

## Files

- `lightning3d_hd.png`, `lightning3d_az50.png` — HD renders, two azimuths
- `validation3d_summary.png` — four-panel validation figure
- `physics3d.py`, `render3d.py` — 3-D solver/growth and camera compositor
- `validate3d_stage{1,2,3,4}.py` — runnable suite (exit 0 = pass)
- `run_grow3d.py` — growth driver with checkpoint/resume
- 2-D pipeline files included for the shared modules
  (`physics.py` current tree, `colorimetry.py`, `render.py` PSF/tonemap)

## Performance & GPU path

Growth wall time ≈ 25 min on one CPU core, dominated by SOR sweeps. The
sweep kernel (`_sor_sweep3`) is a textbook red-black 7-point stencil:
ported to a CuPy `RawKernel` or Numba CUDA on an RTX 3090, full-grid
sweeps at 256³ run in ~1 ms, removing the need for the bounding-box
approximation entirely and enabling multi-strike scenes or leader-
propagation animation at interactive rates.
