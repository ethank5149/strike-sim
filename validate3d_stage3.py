"""Stage 3 (3-D) validation: return-stroke currents on the 3-D tree.
Same conservation checks as 2-D (the tree algebra is dimension-free)."""

import numpy as np
import pickle
import sys
sys.path.insert(0, "/home/claude/lightning")
from physics import segment_currents

if __name__ == "__main__":
    with open("/home/claude/lightning/leader3d.pkl", "rb") as f:
        res = pickle.load(f)
    parent, attach = res["parent"], res["attach"]
    n = len(parent)
    I, subtree, main = segment_currents(parent, attach)

    print("Stage 3 (3-D) validation -- return-stroke current tree")
    ok = True
    path = []
    c = attach
    while c >= 0:
        path.append(c)
        c = parent[c]
    path = path[::-1]
    print(f"  main channel: {len(path)} cells of {n}")

    ground = I[attach] + 1
    print(f"  current into ground = {ground:.0f} vs charge {n}: "
          f"{'PASS' if ground == n else 'FAIL'}")
    ok &= (ground == n)

    children = [[] for _ in range(n)]
    for c in range(1, n):
        children[parent[c]].append(c)
    worst = 0.0
    for k in range(len(path) - 1):
        node, down = path[k], path[k + 1]
        inflow = I[node] if k > 0 else 0.0
        branch = sum(I[ch] for ch in children[node] if ch != down)
        worst = max(worst, abs(I[down] - (inflow + branch + 1.0)))
    print(f"  Kirchhoff max residual ({len(path)-1} junctions): {worst:.1e} "
          f"{'PASS' if worst == 0 else 'FAIL'}")
    ok &= (worst == 0)

    Im = np.array([I[c] for c in path[1:]])
    mono = bool(np.all(np.diff(Im) >= 0))
    print(f"  monotone toward ground ({Im[0]:.0f} -> {Im[-1]:.0f}): "
          f"{'PASS' if mono else 'FAIL'}")
    ok &= mono

    side = I[~main & (np.arange(n) > 0)]
    print(f"  current contrast main/median-twig: "
          f"{Im[-1]/np.median(side):.0f}x")

    with open("/home/claude/lightning/currents3d.pkl", "wb") as f:
        pickle.dump({"I": I, "main": main, "path": np.array(path)}, f)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
