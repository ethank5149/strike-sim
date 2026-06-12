"""Stage 3 validation: return-stroke charge transport on the channel tree.

Physical checks (unit leader charge per cell):
  (1) Conservation at the ground attachment: total current delivered to
      ground = total deposited charge N.
  (2) Kirchhoff's current law at EVERY main-channel node:
      I_out(toward ground) = I_in(from cloud side) + sum(branch inflows)
      + the node's own charge.
  (3) Monotonicity: current is non-decreasing along the main channel
      from cloud to ground (charge only funnels IN, never out).
"""

import numpy as np
import pickle
import sys
sys.path.insert(0, "/home/claude/lightning")
from physics import segment_currents

if __name__ == "__main__":
    with open("/home/claude/lightning/leader.pkl", "rb") as f:
        res = pickle.load(f)
    parent, attach = res["parent"], res["attach"]
    n = len(parent)

    I, subtree, main = segment_currents(parent, attach)

    print("Stage 3 validation -- return-stroke current tree")
    ok = True

    # main-channel path, cloud -> ground order
    path = []
    c = attach
    while c >= 0:
        path.append(c)
        c = parent[c]
    path = path[::-1]
    print(f"  main channel length: {len(path)} cells "
          f"(of {n} total; {n - len(path)} in side branches)")

    # (1) conservation at ground
    ground_current = I[attach] + 1   # + attachment cell's own charge
    print(f"  current into ground = {ground_current:.0f}, "
          f"total charge = {n} "
          f"({'PASS' if ground_current == n else 'FAIL'})")
    ok &= (ground_current == n)

    # (2) Kirchhoff at every interior main-channel node
    children = [[] for _ in range(n)]
    for c in range(1, n):
        children[parent[c]].append(c)

    worst = 0.0
    for k in range(len(path) - 1):
        node, down = path[k], path[k + 1]
        inflow_cloud = I[node] if k > 0 else 0.0
        branch_in = sum(I[ch] for ch in children[node] if ch != down)
        resid = abs(I[down] - (inflow_cloud + branch_in + 1.0))
        worst = max(worst, resid)
    print(f"  Kirchhoff max residual over {len(path)-1} main-channel "
          f"nodes: {worst:.1e} ({'PASS' if worst == 0 else 'FAIL'})")
    ok &= (worst == 0)

    # (3) monotonicity along main channel
    I_main = np.array([I[c] for c in path[1:]])
    mono = bool(np.all(np.diff(I_main) >= 0))
    print(f"  current monotone non-decreasing toward ground: "
          f"{'PASS' if mono else 'FAIL'}  "
          f"(I: {I_main[0]:.0f} at cloud -> {I_main[-1]:.0f} at ground)")
    ok &= mono

    # current contrast that drives the rendering
    side = I[~main & (np.arange(n) > 0)]
    print(f"  median side-branch current: {np.median(side):.0f} "
          f"vs main-channel ground current {I_main[-1]:.0f} "
          f"(contrast {I_main[-1]/np.median(side):.0f}x -> faint corona "
          f"falls below exposure threshold, as in photographs)")

    with open("/home/claude/lightning/currents.pkl", "wb") as f:
        pickle.dump({"I": I, "subtree": subtree, "main": main,
                     "path": np.array(path)}, f)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
