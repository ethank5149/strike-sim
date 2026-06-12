"""
physics.py -- Electrostatics + Dielectric Breakdown Model (DBM) lightning growth.

Stage 1: Laplace solver
    Red-black successive over-relaxation (SOR) for the electric potential
    phi, with del^2 phi = 0, compiled to native code via Numba (LLVM).
    Dirichlet cells are encoded in a `fixed` mask; lateral/top boundaries
    are homogeneous Neumann (mirror ghost cells).

Stage 2: DBM growth (Niemeyer, Pietronero & Wiesmann, PRL 52:1033, 1984)
    The leader channel is a grounded (phi=0) equipotential growing toward
    the charged ground plane (phi=1).  Candidate sites adjacent to the
    channel are added with probability p_i ~ (phi_i)^eta.  eta = 1
    reproduces the fractal dimension D ~ 1.7 measured for real lightning.

Stage 3: Return-stroke current tree
    Each channel cell deposits unit leader charge.  At attachment, all
    charge drains through the tree to the ground contact point; the
    current through every segment follows from charge conservation
    (Kirchhoff) on the tree.
"""

import numpy as np
from numba import njit


# ----------------------------------------------------------------------
# Stage 1: Laplace solver (red-black SOR, Numba -> native code)
# ----------------------------------------------------------------------

@njit(cache=True)
def _sor_sweep(phi, fixed, omega):
    """One red-black SOR sweep on the interior. Returns max |update|.

    Boundary convention (handled by caller via ghost rows/cols):
        row 0      : Neumann (mirror) unless fixed
        row H-1    : typically fixed (ground plane)
        cols 0,W-1 : Neumann (mirror)
    """
    H, W = phi.shape
    max_du = 0.0
    for parity in range(2):
        for i in range(1, H - 1):
            jstart = 1 + ((i + parity) & 1)
            for j in range(jstart, W - 1, 2):
                if not fixed[i, j]:
                    new = 0.25 * (phi[i - 1, j] + phi[i + 1, j]
                                  + phi[i, j - 1] + phi[i, j + 1])
                    du = omega * (new - phi[i, j])
                    phi[i, j] += du
                    if du < 0.0:
                        du = -du
                    if du > max_du:
                        max_du = du
    return max_du


@njit(cache=True)
def _apply_neumann(phi, fixed):
    """Mirror ghost cells for homogeneous Neumann sides/top."""
    H, W = phi.shape
    for i in range(H):
        if not fixed[i, 0]:
            phi[i, 0] = phi[i, 1]
        if not fixed[i, W - 1]:
            phi[i, W - 1] = phi[i, W - 2]
    for j in range(W):
        if not fixed[0, j]:
            phi[0, j] = phi[1, j]


@njit(cache=True)
def solve_laplace(phi, fixed, omega, tol, max_sweeps):
    """Relax phi in place until max update < tol. Returns sweep count."""
    for s in range(max_sweeps):
        _apply_neumann(phi, fixed)
        du = _sor_sweep(phi, fixed, omega)
        if du < tol:
            _apply_neumann(phi, fixed)
            return s + 1
    _apply_neumann(phi, fixed)
    return max_sweeps


def optimal_omega(H, W):
    """Asymptotically optimal SOR factor for the 5-point Laplacian."""
    n = min(H, W)
    return 2.0 / (1.0 + np.sin(np.pi / n))


# ----------------------------------------------------------------------
# Stage 2: DBM stepped-leader growth
# ----------------------------------------------------------------------

_NBRS = np.array([(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)], dtype=np.int64)


def grow_leader(H=480, W=320, eta=1.0, seed_col=None, rng_seed=11,
                cells_per_solve=3, tol=1e-4, verbose=False):
    """Grow a DBM discharge from a seed at the top toward the ground plane.

    Returns dict with channel cell list (in growth order), parent indices,
    final potential field, and the index of the ground-attachment cell.
    """
    rng = np.random.default_rng(rng_seed)
    if seed_col is None:
        seed_col = W // 2

    phi = np.linspace(0.0, 1.0, H)[:, None] * np.ones((1, W))  # warm start
    phi = np.ascontiguousarray(phi)
    fixed = np.zeros((H, W), dtype=np.bool_)
    fixed[H - 1, :] = True          # ground plane, phi = 1
    phi[H - 1, :] = 1.0

    omega = optimal_omega(H, W)

    cells = [(0, seed_col)]          # channel cells in growth order
    parent = [-1]
    cell_index = {(0, seed_col): 0}
    fixed[0, seed_col] = True
    phi[0, seed_col] = 0.0

    # candidate -> index of adjacent channel cell that first exposed it
    candidates = {}

    def expose(ci):
        i, j = cells[ci]
        for di, dj in _NBRS:
            ii, jj = i + di, j + dj
            if 0 <= ii < H - 1 and 0 <= jj < W:
                if (ii, jj) not in cell_index and (ii, jj) not in candidates:
                    candidates[(ii, jj)] = ci

    expose(0)
    attached = -1
    solves = 0
    total_sweeps = 0

    while attached < 0:
        total_sweeps += solve_laplace(phi, fixed, omega, tol, 20000)
        solves += 1

        for _ in range(cells_per_solve):
            keys = list(candidates.keys())
            w = np.array([max(phi[k], 0.0) for k in keys]) ** eta
            s = w.sum()
            if s <= 0.0:
                raise RuntimeError("degenerate growth weights")
            pick = keys[rng.choice(len(keys), p=w / s)]

            ci = len(cells)
            cells.append(pick)
            parent.append(candidates.pop(pick))
            cell_index[pick] = ci
            fixed[pick] = True
            phi[pick] = 0.0
            expose(ci)

            if pick[0] == H - 2:            # one cell above ground plane
                attached = ci
                break

        if verbose and solves % 200 == 0:
            print(f"  solve {solves}: {len(cells)} cells, "
                  f"tip depth {max(c[0] for c in cells)}/{H-1}")

    return dict(cells=np.array(cells), parent=np.array(parent),
                phi=phi, fixed=fixed, attach=attached,
                H=H, W=W, eta=eta,
                solves=solves, sweeps=total_sweeps)


# ----------------------------------------------------------------------
# Stage 3: return-stroke current tree
# ----------------------------------------------------------------------

def segment_currents(parent, attach):
    """Current through edge (parent[c] -> c) for every channel cell c > 0.

    Tree is rooted at the seed (cell 0).  Each cell carries unit leader
    charge; during the return stroke all charge drains to ground through
    the attachment cell `attach`.  For an edge NOT on the seed->attach
    main path, the draining charge is the subtree hanging off it.  For an
    edge ON the main path, it is everything outside the lower subtree.
    """
    n = len(parent)
    subtree = np.ones(n, dtype=np.int64)
    for c in range(n - 1, 0, -1):          # children have larger indices
        subtree[parent[c]] += subtree[c]

    main = np.zeros(n, dtype=np.bool_)
    c = attach
    while c >= 0:
        main[c] = True
        c = parent[c]

    I = np.zeros(n, dtype=np.float64)      # I[c] = current parent->c edge
    for c in range(1, n):
        I[c] = (n - subtree[c]) if main[c] else subtree[c]
    return I, subtree, main
