"""Stage 2 validation: DBM leader growth.

Checks
  (a) structural sanity: tree is connected (every cell's parent is an
      8-neighbor added earlier), channel spans cloud->ground;
  (b) box-counting fractal dimension of the discharge pattern.

Reference values: NPW (1984) report D ~= 1.75 for eta = 1; optical
measurements of real lightning channels give D ~= 1.7 +/- 0.1 (e.g.
Sanudo et al., Nonlin. Proc. Geophys. 2, 101 (1995)).  Accept 1.45-1.85
for a single realization in plate (cloud-ground) geometry, which biases
D slightly below the radial NPW value.
"""

import numpy as np
import pickle
import sys
sys.path.insert(0, "/home/claude/lightning")
from physics import grow_leader, _NBRS


def box_count_dimension(cells, H, W):
    occ = np.zeros((H, W), dtype=np.uint8)
    occ[cells[:, 0], cells[:, 1]] = 1
    sizes, counts = [], []
    for k in range(1, 20):
        b = 2 ** k
        if b > min(H, W) // 4:
            break
        Hc, Wc = (H + b - 1) // b, (W + b - 1) // b
        pad = np.zeros((Hc * b, Wc * b), dtype=np.uint8)
        pad[:H, :W] = occ
        n = pad.reshape(Hc, b, Wc, b).max(axis=(1, 3)).sum()
        sizes.append(b)
        counts.append(n)
    sizes = np.array(sizes, float)
    counts = np.array(counts, float)
    slope, intercept = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return slope, sizes, counts


if __name__ == "__main__":
    print("Stage 2 validation -- DBM stepped-leader growth (eta = 1)")
    res = grow_leader(H=480, W=320, eta=1.0, rng_seed=11, verbose=True)
    cells, parent = res["cells"], res["parent"]
    n = len(cells)
    print(f"  channel cells: {n}, Laplace solves: {res['solves']}, "
          f"total SOR sweeps: {res['sweeps']}")

    ok = True

    # (a) tree structure: each parent is an 8-neighbor with smaller index
    nbr = set(map(tuple, _NBRS))
    for c in range(1, n):
        p = parent[c]
        if not (0 <= p < c):
            ok = False
            print(f"  FAIL: cell {c} parent index {p}")
            break
        d = tuple(cells[c] - cells[p])
        if d not in nbr:
            ok = False
            print(f"  FAIL: cell {c} not adjacent to its parent")
            break
    else:
        print("  tree connectivity: PASS (every parent is an earlier 8-neighbor)")

    # vertical span: seed row 0 to one cell above ground
    top, bot = cells[:, 0].min(), cells[:, 0].max()
    span_ok = (top == 0 and bot == res["H"] - 2)
    print(f"  cloud->ground span: rows {top}..{bot} "
          f"({'PASS' if span_ok else 'FAIL'})")
    ok &= span_ok

    # (b) fractal dimension
    D, sizes, counts = box_count_dimension(cells, res["H"], res["W"])
    print(f"  box-counting fractal dimension D = {D:.3f} "
          f"(lightning: 1.7 +/- 0.1; NPW eta=1: 1.75)")
    dim_ok = 1.45 <= D <= 1.85
    print(f"  fractal dimension: {'PASS' if dim_ok else 'FAIL'}")
    ok &= dim_ok

    with open("/home/claude/lightning/leader.pkl", "wb") as f:
        pickle.dump({**res, "D": D, "bc_sizes": sizes, "bc_counts": counts}, f)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
