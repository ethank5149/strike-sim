"""Stage 1 validation: Laplace solver correctness.

Test problem: u(x,y) = sin(pi x) * sinh(pi y) / sinh(pi) on the unit
square, which satisfies del^2 u = 0 exactly.  All four sides Dirichlet
(taken from the analytic solution).  The 5-point stencil is 2nd-order
accurate, so the max-norm error must shrink ~4x per grid refinement.
"""

import numpy as np
import sys
sys.path.insert(0, "/home/claude/lightning")
from physics import solve_laplace, optimal_omega


def analytic(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)


def run_case(N):
    u_exact = analytic(N)
    phi = np.zeros((N, N))
    fixed = np.zeros((N, N), dtype=np.bool_)
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    phi[fixed] = u_exact[fixed]

    sweeps = solve_laplace(phi, fixed, optimal_omega(N, N),
                           tol=1e-10, max_sweeps=200000)
    err = np.abs(phi - u_exact).max()
    return err, sweeps


if __name__ == "__main__":
    print("Stage 1 validation -- Laplace solver vs analytic harmonic u(x,y)")
    print(f"{'N':>6} {'max error':>12} {'ratio':>8} {'order':>7} {'sweeps':>8}")
    prev = None
    results = []
    ok = True
    for N in (33, 65, 129, 257):
        err, sweeps = run_case(N)
        if prev is None:
            print(f"{N:>6} {err:>12.3e} {'-':>8} {'-':>7} {sweeps:>8}")
        else:
            ratio = prev / err
            order = np.log2(ratio)
            print(f"{N:>6} {err:>12.3e} {ratio:>8.2f} {order:>7.2f} {sweeps:>8}")
            if not (1.7 < order < 2.3):
                ok = False
        results.append((N, err))
        prev = err
    # absolute accuracy check at finest grid
    if results[-1][1] > 1e-3:
        ok = False
    print("PASS" if ok else "FAIL",
          "- second-order convergence and small absolute error" if ok else "")
    np.save("/home/claude/lightning/val_stage1.npy",
            np.array(results))
    sys.exit(0 if ok else 1)
