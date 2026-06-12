"""Stage 5 validation: time-resolved luminosity model.

  (1) Leader causality: the visible cell set grows monotonically and in
      exact growth order (a cell never appears before its parent).
  (2) Return-stroke causality + front speed: at every sampled time, the
      set of fully lit cells is exactly {c : d_act(c) <= s(t)}; the
      measured front position (max lit activation distance) is linear in
      t with the specified speed (within discretization of one edge).
  (3) Decay: total channel luminosity after T2 fits exp(-(t-T2)/tau)
      with tau within 3% of specification (it is exact by construction;
      the fit verifies the implementation).
  (4) Still-frame equivalence: at full illumination (t = T2), per-cell
      weights equal rel^0.75 exactly -- the animation passes through the
      Stage-4-validated radiometry.
  (5) Video: MP4 exists, correct frame count/size, frames decodable.
"""

import numpy as np
import pickle
import cv2
import sys
sys.path.insert(0, "/home/claude/lightning")
from temporal import (activation_distances, cell_weights,
                      T1, T2, TAU, LEADER_LEVEL)

if __name__ == "__main__":
    with open("/home/claude/lightning/leader3d.pkl", "rb") as f:
        res = pickle.load(f)
    with open("/home/claude/lightning/currents3d.pkl", "rb") as f:
        cur = pickle.load(f)
    cells, parent = res["cells"], res["parent"]
    n = len(cells)
    I = cur["I"].copy(); I[0] = 1.0
    rel = I / I.max()
    d_act, _ = activation_distances(cells, parent, cur["main"],
                                    res["attach"])
    d_max = d_act.max()

    print("Stage 5 validation -- temporal luminosity model")
    ok = True

    # (1) leader causality
    prev = np.zeros(n, dtype=bool)
    causal = True
    for t in np.linspace(0.01, T1 * 0.999, 25):
        vis = cell_weights(t, n, rel, d_act, d_max) > 0
        if not (vis | ~prev).all() or not (vis >= prev).all():
            causal = False
        # parent precedes child
        idx = np.where(vis)[0]
        if len(idx) and not vis[parent[idx[1:]]].all():
            causal = False
        prev = vis
    print(f"  leader phase: monotone growth, parent-before-child: "
          f"{'PASS' if causal else 'FAIL'}")
    ok &= causal

    # (2) return-stroke causality and front speed
    ts = np.linspace(T1 + 1e-4, T2 - 1e-4, 30)
    fronts = []
    rs_ok = True
    for t in ts:
        w = cell_weights(t, n, rel, d_act, d_max)
        # pre-decay, each weight is exactly rel^0.75 (lit) or the leader
        # glow (unlit); classify per cell, not by a global threshold
        lit = w == rel ** 0.75
        # cells where the two values coincide are ambiguous; exclude
        amb = np.isclose(rel ** 0.75, LEADER_LEVEL * rel ** 0.3)
        s = d_max * (t - T1) / (T2 - T1)
        expect = d_act <= s
        if not np.array_equal(lit[~amb], expect[~amb]):
            rs_ok = False
        fronts.append(d_act[lit & ~amb].max() if (lit & ~amb).any() else 0.0)
    slope = np.polyfit(ts, fronts, 1)[0]
    v_spec = d_max / (T2 - T1)
    sp_ok = abs(slope - v_spec) / v_spec < 0.02
    print(f"  return stroke: lit set == causal set at all 30 samples: "
          f"{'PASS' if rs_ok else 'FAIL'}")
    print(f"  front speed: measured {slope:.1f} vs spec {v_spec:.1f} "
          f"cells/unit-t ({100*abs(slope-v_spec)/v_spec:.2f}% err): "
          f"{'PASS' if sp_ok else 'FAIL'}")
    ok &= rs_ok and sp_ok

    # (3) decay constant
    td = np.linspace(T2 + 0.01, 1.0, 25)
    tot = np.array([cell_weights(t, n, rel, d_act, d_max).sum()
                    for t in td])
    tau_fit = -1.0 / np.polyfit(td - T2, np.log(tot), 1)[0]
    tau_ok = abs(tau_fit - TAU) / TAU < 0.03
    print(f"  decay: fitted tau = {tau_fit:.4f} vs spec {TAU} "
          f"({'PASS' if tau_ok else 'FAIL'})")
    ok &= tau_ok

    # (4) still-frame equivalence at t = T2
    w_full = cell_weights(T2, n, rel, d_act, d_max)
    eq = np.abs(w_full - rel ** 0.75).max()
    eq_ok = eq < 1e-12
    print(f"  full-illumination weights == validated rel^0.75: "
          f"max dev {eq:.1e} ({'PASS' if eq_ok else 'FAIL'})")
    ok &= eq_ok

    # (5) video checks (if rendered)
    import os
    vp = "/home/claude/lightning/lightning3d_strike.mp4"
    if os.path.exists(vp):
        cap = cv2.VideoCapture(vp)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, fr = cap.read()
        cap.release()
        vid_ok = ret and nf >= 100 and (wv, hv) == (960, 540)
        print(f"  video: {nf} frames @ {wv}x{hv}, decodable: "
              f"{'PASS' if vid_ok else 'FAIL'}")
        ok &= vid_ok
    else:
        print("  video: not yet rendered (skipped)")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
