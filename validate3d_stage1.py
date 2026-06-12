"""Stage 1 (3-D) validation: Laplace solver correctness.

Test problem on the unit cube:
    u(x,y,z) = sin(pi x) sin(pi y) sinh(sqrt(2) pi z) / sinh(sqrt(2) pi)
which is exactly harmonic (-pi^2 - pi^2 + 2 pi^2 = 0).  All faces
Dirichlet from the analytic solution.  The 7-point stencil is 2nd-order:
max error must shrink ~4x per refinement.
"""

import numpy as np
import sys
sys.path.insert(0, "/home/claude/lightning")
from physics3d import solve_laplace3, optimal_omega3


def analytic(N):
    t = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
    return (np.sin(np.pi * X) * np.sin(np.pi * Y)
            * np.sinh(np.sqrt(2) * np.pi * Z) / np.sinh(np.sqrt(2) * np.pi))


def run_case(N):
    u = analytic(N)
    phi = np.zeros((N, N, N))
    fixed = np.zeros((N, N, N), dtype=np.bool_)
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax], sl1[ax] = 0, N - 1
        fixed[tuple(sl0)] = fixed[tuple(sl1)] = True
    phi[fixed] = u[fixed]
    sweeps = solve_laplace3(phi, fixed, optimal_omega3(N, N, N),
                            1e-10, 200000, 1, N - 1, 1, N - 1, 1, N - 1)
    return np.abs(phi - u).max(), sweeps


if __name__ == "__main__":
    print("Stage 1 (3-D) validation -- 7-point SOR vs analytic harmonic")
    print(f"{'N':>5} {'max error':>12} {'ratio':>7} {'order':>7} {'sweeps':>7}")
    prev, ok = None, True
    results = []
    for N in (17, 33, 65, 97):
        err, sweeps = run_case(N)
        if prev is None:
            print(f"{N:>5} {err:>12.3e} {'-':>7} {'-':>7} {sweeps:>7}")
        else:
            ratio = prev / err
            order = np.log(ratio) / np.log((N - 1) / (Nprev - 1))
            print(f"{N:>5} {err:>12.3e} {ratio:>7.2f} {order:>7.2f} {sweeps:>7}")
            if not (1.7 < order < 2.3):
                ok = False
        results.append((N, err))
        prev, Nprev = err, N
    if results[-1][1] > 2e-3:
        ok = False
    np.save("/home/claude/lightning/val3d_stage1.npy", np.array(results))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
