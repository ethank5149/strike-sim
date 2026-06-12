# Stage 5 Addendum — Interactive Animation & Temporal Model

## Architecture

The animation physics lives in `temporal.py` (importable, headlessly
testable); `lightning_interactive.ipynb` is a thin ipywidgets layer over
it. This keeps the stage-1–4 modules untouched and gives the temporal
model the same validation treatment as the rest of the pipeline.

## Temporal model

Animation time t ∈ [0, 1] is a documented nonuniform warp of physical
time (the ~20 ms leader and ~70 µs return stroke cannot share a linear
clock at 30 fps):

* **t < 0.55 — stepped leader.** The channel appears in *exact DBM
  growth order* (the simulation's own clock), at 1.2×10⁻² of
  return-stroke radiance, with a +0.30 boost on the trailing 2 % of
  cells — the multiple simultaneously advancing tips with corona bursts
  seen in high-speed leader photography. The quadratic index ramp
  mirrors the tip acceleration observed in the growth log.
* **0.55 ≤ t < 0.63 — return stroke.** A luminous front sweeps up the
  tree from the ground attachment at constant speed in the exact
  tree-path metric d(c) = d_root(attach) + d_root(c) − 2 d_root(junction)
  computed on the Stage-3-validated tree.
* **t ≥ 0.63 — decay.** Exponential, τ = 0.12.

At full illumination the per-cell weights equal rel^0.75 **exactly**, so
the animation passes through the Stage-4-validated radiometry.

## Stage 5 validation results

* Leader causality: visible set monotone, parent always precedes child
  across 25 sampled times. **PASS**
* Return-stroke causality: lit set equals the causal set
  {c : d(c) ≤ s(t)} exactly at all 30 sampled times. **PASS**
* Front speed: measured 6395.2 vs specified 6391.6 cells/unit-t —
  0.06 % error. **PASS**
* Decay: fitted τ = 0.1200 vs spec 0.12. **PASS**
* Still-frame equivalence at t = T2: max weight deviation 0.0. **PASS**
* Video: 150 frames @ 960×540, decodable. **PASS**
* Notebook: executed headlessly end-to-end via nbclient (load → widget
  construction → HD export → suite cell). **PASS**

One test fix during validation, documented: the initial lit-cell
classifier used a global brightness threshold, which misclassified
faint-but-lit twigs (rel^0.75 of a 1-cell twig sits below any global
glow threshold); replaced with exact per-cell two-value comparison.
The model itself was unchanged.

## Usage

```
jupyter lab lightning_interactive.ipynb
```
Scrub time through leader → attachment → return stroke → decay; orbit
azimuth ±90° (live parallax of the 3-D channel); exposure slider;
480×270 fast preview or 960×540. HD stills and MP4 export cells
included; the final cell re-runs validation stages 1–5.

On the RTX 3090: port `_sor_sweep3` to a CuPy RawKernel and regrowth
drops from ~25 min to seconds — the scrubber can then drive live
simulation. All validation stages run unchanged against a GPU backend.
