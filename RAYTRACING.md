# Ray-Tracing Upgrade — Physics & Validation Report

Replaces the raster compositor with a spectral path tracer, written
backend-agnostically (`xp` = NumPy or CuPy): **identical code validated
on CPU here, production-rendered on the RTX 3090**. The final stage
(RT5, `gpu_parity.py`) runs on SkyNet and asserts CPU↔GPU equivalence.

## What replaced what

| raster pipeline (validated stages 1–5) | ray-traced pipeline |
|---|---|
| screen-space sum-of-Gaussians "PSF" | **single-scattering radiative transfer**: MC equiangular sampling (Kulla–Fajardo 2012) through Rayleigh (σ∝λ⁻⁴, phase 3/16π(1+cos²)) + Henyey–Greenstein aerosol (g=0.76); the glow is emergent physics |
| rasterized AA lines | **exact closed-form ray–capsule intersection** over all 6,875 emissive segments, jittered sub-pixel AA |
| RGB color tinting | **8-band spectral rendering**, 30,000 K Planck emission, integrated through the validated CIE pipeline; the blue halo emerges from λ⁻⁴ scattering |
| per-channel ACES curve | **AgX-family display transform** (log₂ shaper −10..+6.5 stops, inset/outset gamut matrices α=0.12, normalized tanh sigmoid) **baked into a 33³ LUT**, applied trilinearly — the "shader/LUT" path on GPU |
| flat ground strip | **Lambertian ground (ρ=0.07)** with MC direct lighting + **Fresnel wet mirror** (F₀=0.02) reflecting the channel |

Energy consistency: core and halo share one scale — segment power per
length w·S(λ) → exitance → radiance L_e = wS/(2π²r_core), and the same
W·S enters the scatter estimator. Scale: 1 cell = 12.5 m (2 km channel),
σ_R(550) = 1.2×10⁻⁵ m⁻¹ (sea level), σ_M = 2.5×10⁻⁴ m⁻¹ (storm haze).
Documented stylization: r_core is the *resolved luminous envelope*; the
cm-scale conducting core is sub-pixel at this scale. Single scattering
only (dominant in clear storm air); multiple scattering is the
documented omission.

## Validation results (CPU, identical code path)

**RT1 — ray–capsule intersection** vs SDF sphere-tracing reference on
2,000 random configurations: 100 % hit/miss agreement, max hit-distance
error 1.4×10⁻⁵. One contract clarified during validation: ray origins
must lie outside all capsules (the renderer guarantees this — camera at
2.2H, mirror rays on the ground plane). The two initially-failing rays
had origins *inside* a capsule (dense-march ground truth: first hit at
t≈10⁻²¹); documented as a precondition rather than patched around.

**RT2 — equiangular estimator**: in the isotropic/zero-extinction limit
the pdf cancels the integrand exactly, so the estimator is provably
zero-variance — verified at **4×10⁻¹⁰ relative deviation per sample**
against the closed form (1/16π²D)·Δarctan. With extinction: MC vs
20k-node quadrature agrees to 0.06 %.

**RT3 — spectral pipeline**: 8-band blackbody chromaticity within
(Δx, Δy) = (0.0042, 0.0050) of the 401-sample reference; Rayleigh band
ratio exactly (λ₁/λ₀)⁴.

**RT4 — display transform + LUT**: gray axis preserved exactly;
monotone; endpoints exact over the full shaper domain; 0.18 scene-linear
→ 0.186 display-linear (mid-gray anchor); saturated highlights
desaturate toward white (0.93 → 0.00, the AgX inset mechanism); 33³
trilinear LUT within 7.1×10⁻³ of direct evaluation over 200k HDR colors.

Two test-spec corrections during validation, documented: the
zero-variance threshold was tightened past float64 atan/tan round-trip
noise (now relative 10⁻⁹), and the endpoint probe initially covered
2⁻⁹..2⁵ instead of the shaper's 2⁻¹⁰..2⁶·⁵ domain. Physics unchanged in
both cases.

**CPU composition proof**: 240×135 @ 36 spp + 3×AA frame renders the
full pipeline (direct + ground + mirror + scatter + LUT) in 202 s on one
core — MC-noisy by necessity; production resolution/sample counts are
the GPU's job. The sub-pixel capsule core motivated the jittered-AA
addition (single center rays under-sample a 0.4 px core); chunked and
unchunked intersection verified exactly equal.

## RT5 — run on SkyNet

```
pip install cupy-cuda12x      # CUDA 12.x
python3 gpu_parity.py
```
Asserts: intersection/display/LUT parity exactly (float32 tolerance);
scatter estimator means within 5 standard errors (NumPy/CuPy RNG
sequences differ by construction — both estimate the RT2-validated
integral); then benchmarks 540p and 1080p frames and saves
`lightning_pbr_hd.png`. Expected 3090 throughput: the dominant cost is
the pixel×segment intersection broadcast (~10¹⁰ FMA-class ops at 1080p
×8 AA — well under a second of compute) plus the scatter passes;
seconds per converged 1080p frame. Raise `pix_chunk` (e.g. 1<<18) in
`capsule_intersect` for GPU occupancy. For animation, pass
`weights=temporal.cell_weights(t, ...)` to `render_pbr` — the temporal
model (stage 5) composes with the path tracer unchanged.

## Files
- `pbr.py` — backend-agnostic path tracer + display transform/LUT
- `validate_rt.py` — RT1–RT4 suite (exit 0 = pass)
- `gpu_parity.py` — RT5 parity + benchmark for the 3090
- `display_lut33.npy` — baked 33³ display LUT
- `pbr_preview.png` — CPU correctness demo frame
