"""Stage 2 (3-D) validation: DBM growth.

  (a) tree connectivity (26-neighbor, parent added earlier), span.
  (b) solver-acceleration audit: max box-vs-full potential deviation.
  (c) 3-D box-counting dimension.  References: 3-D DBM at eta = 2 gives
      D ~ 1.9 in radial geometry (Pietronero & Wiesmann, J. Stat. Phys.
      36, 909 (1984); Sanudo et al. 1995); plate geometry biases lower
      (our 2-D run: 1.54 vs radial 1.75).  Accept 1.4-2.1.
  (d) 2-D projected dimension: for a set with D < 2, orthogonal
      projection preserves D (Marstrand's projection theorem), so the
      projected channel must match optical lightning measurements,
      D ~ 1.7 +/- 0.1; accept 1.35-1.95 for a single plate-geometry
      realization, and require |D_proj - D_3d| small.
"""

import numpy as np
import pickle
import sys
sys.path.insert(0, "/home/claude/lightning")
from physics3d import _NBRS3


def box_dim(occ_coords, shape):
    occ = np.zeros(shape, dtype=np.uint8)
    occ[tuple(occ_coords.T)] = 1
    sizes, counts = [], []
    for k in range(1, 20):
        b = 2 ** k
        if b > min(shape) // 4:
            break
        padded_shape = tuple(((s + b - 1) // b) * b for s in shape)
        pad = np.zeros(padded_shape, dtype=np.uint8)
        pad[tuple(slice(0, s) for s in shape)] = occ
        view = pad
        for ax in range(len(shape)):
            ns = view.shape[:ax * 2] if False else None
        # reshape trick per axis
        r = pad
        newshape = []
        for s in padded_shape:
            newshape += [s // b, b]
        r = pad.reshape(newshape)
        axes = tuple(range(1, 2 * len(shape), 2))
        n = r.max(axis=axes).sum()
        sizes.append(b)
        counts.append(n)
    sizes, counts = np.array(sizes, float), np.array(counts, float)
    D = np.polyfit(np.log(1 / sizes), np.log(counts), 1)[0]
    return D, sizes, counts


if __name__ == "__main__":
    with open("/home/claude/lightning/leader3d.pkl", "rb") as f:
        res = pickle.load(f)
    cells, parent = res["cells"], res["parent"]
    H, W, D_ = res["H"], res["W"], res["D"]
    n = len(cells)
    print("Stage 2 (3-D) validation -- DBM growth (eta = 2)")
    print(f"  {n} cells, {res['solves']} solves, {res['sweeps']} sweeps")
    ok = True

    # (a) connectivity + span
    nbr = set(map(tuple, _NBRS3))
    conn = all(0 <= parent[c] < c
               and tuple(cells[c] - cells[parent[c]]) in nbr
               for c in range(1, n))
    print(f"  tree connectivity (26-neighbor): {'PASS' if conn else 'FAIL'}")
    ok &= conn
    span = cells[:, 0].min() == 0 and cells[:, 0].max() == H - 2
    print(f"  cloud->ground span rows {cells[:,0].min()}..{cells[:,0].max()}"
          f": {'PASS' if span else 'FAIL'}")
    ok &= span

    # (b) bounding-box acceleration audit
    bd = res["box_dev"]
    bd_ok = bd < 5e-3
    print(f"  box-sweep vs full-domain max rel. deviation: {bd:.2e} "
          f"({'PASS' if bd_ok else 'FAIL'}, threshold 5e-3)")
    ok &= bd_ok

    # (c) 3-D fractal dimension
    D3, s3, c3 = box_dim(cells, (H, W, D_))
    d3_ok = 1.4 <= D3 <= 2.1
    print(f"  3-D box dimension D = {D3:.3f} "
          f"(3-D DBM eta=2 ~ 1.9 radial; plate biases lower) "
          f"{'PASS' if d3_ok else 'FAIL'}")
    ok &= d3_ok

    # (d) projected dimension vs optical lightning measurements
    proj = np.unique(cells[:, [0, 1]], axis=0)   # project along depth axis
    Dp, sp, cp = box_dim(proj, (H, W))
    dp_ok = 1.35 <= Dp <= 1.95 and abs(Dp - D3) < 0.35
    print(f"  projected 2-D dimension D = {Dp:.3f} "
          f"(optical lightning: 1.7 +/- 0.1; projection preserves D<2) "
          f"{'PASS' if dp_ok else 'FAIL'}")
    ok &= dp_ok

    with open("/home/claude/lightning/leader3d.pkl", "wb") as f:
        res.update(D3=D3, Dproj=Dp, bc3=(s3, c3), bcp=(sp, cp))
        pickle.dump(res, f)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
