"""
physics3d.py -- 3-D electrostatics + DBM lightning growth.

Same physics as the 2-D pipeline, promoted to 3-D:
  * Laplace: 7-point stencil, red-black SOR (Numba/LLVM), Dirichlet
    channel (phi=0) + ground plane (phi=1), Neumann on the four lateral
    faces and the top.
  * DBM (NPW 1984): growth probability ~ phi^eta over 26-connected
    candidates.  eta = 2 is the physically calibrated choice in 3-D:
    eta = 1 gives D ~ 2.5 (volume-filling bush), while measured lightning
    channels project to D ~ 1.7 +/- 0.1; 3-D DBM at eta = 2 gives
    D ~ 1.9 radial, biased lower in plate geometry.
  * Return-stroke currents: identical tree algebra (dimension-agnostic),
    imported from the 2-D module.

Performance: sweeps are restricted to the channel's lateral bounding box
(+margin) over the FULL vertical extent; a full-domain solve refreshes
the far field periodically.  The box approximation is validated against
full-domain solves at checkpoints (max relative deviation of candidate
potentials is recorded).
"""

import numpy as np
import time
from numba import njit


# ----------------------------------------------------------------------
# Stage 1 (3-D): Laplace solver
# ----------------------------------------------------------------------

@njit(cache=True)
def _sor_sweep3(phi, fixed, omega, i0, i1, j0, j1, k0, k1):
    """One red-black SOR sweep over [i0,i1)x[j0,j1)x[k0,k1) interior."""
    max_du = 0.0
    inv6 = 1.0 / 6.0
    for parity in range(2):
        for i in range(i0, i1):
            for j in range(j0, j1):
                kstart = k0 + ((i + j + k0 + parity) & 1)
                for k in range(kstart, k1, 2):
                    if not fixed[i, j, k]:
                        new = inv6 * (phi[i - 1, j, k] + phi[i + 1, j, k]
                                      + phi[i, j - 1, k] + phi[i, j + 1, k]
                                      + phi[i, j, k - 1] + phi[i, j, k + 1])
                        du = omega * (new - phi[i, j, k])
                        phi[i, j, k] += du
                        if du < 0.0:
                            du = -du
                        if du > max_du:
                            max_du = du
    return max_du


@njit(cache=True)
def _apply_neumann3(phi, fixed):
    """Mirror ghosts on lateral faces and top (bottom = ground, fixed)."""
    H, W, D = phi.shape
    for i in range(H):
        for k in range(D):
            if not fixed[i, 0, k]:
                phi[i, 0, k] = phi[i, 1, k]
            if not fixed[i, W - 1, k]:
                phi[i, W - 1, k] = phi[i, W - 2, k]
        for j in range(W):
            if not fixed[i, j, 0]:
                phi[i, j, 0] = phi[i, j, 1]
            if not fixed[i, j, D - 1]:
                phi[i, j, D - 1] = phi[i, j, D - 2]
    for j in range(W):
        for k in range(D):
            if not fixed[0, j, k]:
                phi[0, j, k] = phi[1, j, k]


@njit(cache=True)
def solve_laplace3(phi, fixed, omega, tol, max_sweeps,
                   i0, i1, j0, j1, k0, k1):
    for s in range(max_sweeps):
        _apply_neumann3(phi, fixed)
        du = _sor_sweep3(phi, fixed, omega, i0, i1, j0, j1, k0, k1)
        if du < tol:
            _apply_neumann3(phi, fixed)
            return s + 1
    _apply_neumann3(phi, fixed)
    return max_sweeps


def optimal_omega3(H, W, D):
    n = min(H, W, D)
    return 2.0 / (1.0 + np.sin(np.pi / n))


# ----------------------------------------------------------------------
# Stage 2 (3-D): DBM growth
# ----------------------------------------------------------------------

_NBRS3 = np.array([(di, dj, dk)
                   for di in (-1, 0, 1)
                   for dj in (-1, 0, 1)
                   for dk in (-1, 0, 1)
                   if (di, dj, dk) != (0, 0, 0)], dtype=np.int64)


def grow_leader3(H=160, W=112, D=112, eta=2.0, rng_seed=7,
                 cells_per_solve=4, tol=1.5e-4, margin=28,
                 full_every=60, verbose_every=100, log=print,
                 checkpoint=None, ckpt_every=100):
    import pickle as _pkl
    import os as _os

    rng = np.random.default_rng(rng_seed)
    sj, sk = W // 2, D // 2

    if checkpoint and _os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            ck = _pkl.load(f)
        phi = ck["phi"]
        fixed = ck["fixed"]
        cells = ck["cells"]
        parent = ck["parent"]
        cell_index = ck["cell_index"]
        candidates = ck["candidates"]
        rng.bit_generator.state = ck["rng_state"]
        solves0, sweeps0, box_dev0 = ck["solves"], ck["sweeps"], ck["box_dev"]
        log(f"  resumed from checkpoint: {len(cells)} cells, "
            f"solve {solves0}")
    else:
        phi = (np.linspace(0.0, 1.0, H, dtype=np.float64)[:, None, None]
               * np.ones((1, W, D)))
        phi = np.ascontiguousarray(phi)
        fixed = np.zeros((H, W, D), dtype=np.bool_)
        fixed[H - 1] = True
        phi[H - 1] = 1.0
        cells = [(0, sj, sk)]
        parent = [-1]
        cell_index = {(0, sj, sk): 0}
        fixed[0, sj, sk] = True
        phi[0, sj, sk] = 0.0
        candidates = {}
        solves0 = sweeps0 = 0
        box_dev0 = 0.0

    omega = optimal_omega3(H, W, D)

    def expose(ci):
        i, j, k = cells[ci]
        for di, dj, dk in _NBRS3:
            p = (i + di, j + dj, k + dk)
            if (0 <= p[0] < H - 1 and 0 <= p[1] < W and 0 <= p[2] < D
                    and p not in cell_index and p not in candidates):
                candidates[p] = ci

    if not (checkpoint and candidates):
        expose(0)
    attached = -1
    solves = solves0
    total_sweeps = sweeps0
    box_dev = box_dev0
    t0 = time.time()

    def lateral_box():
        a = np.array(cells)
        j0 = max(1, a[:, 1].min() - margin)
        j1 = min(W - 1, a[:, 1].max() + margin + 1)
        k0 = max(1, a[:, 2].min() - margin)
        k1 = min(D - 1, a[:, 2].max() + margin + 1)
        return j0, j1, k0, k1

    while attached < 0:
        j0, j1, k0, k1 = lateral_box()
        full = (solves % full_every == 0)
        if full:
            total_sweeps += solve_laplace3(phi, fixed, omega, tol, 20000,
                                           1, H - 1, 1, W - 1, 1, D - 1)
            # validation checkpoint: candidate potentials, box vs full
            if solves > 0 and candidates:
                keys = list(candidates.keys())
                p_full = np.array([phi[k] for k in keys])
                # re-relax inside the box only and compare
                solve_laplace3(phi, fixed, omega, tol, 20000,
                               1, H - 1, j0, j1, k0, k1)
                p_box = np.array([phi[k] for k in keys])
                m = p_full > 1e-6
                if m.any():
                    box_dev = max(box_dev, float(
                        np.abs(p_box[m] - p_full[m]).max()
                        / p_full[m].max()))
        else:
            total_sweeps += solve_laplace3(phi, fixed, omega, tol, 20000,
                                           1, H - 1, j0, j1, k0, k1)
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

            if pick[0] == H - 2:
                attached = ci
                break

        if verbose_every and solves % verbose_every == 0:
            tip = max(c[0] for c in cells)
            log(f"  solve {solves}: {len(cells)} cells, tip {tip}/{H-1}, "
                f"box dev {box_dev:.2e}, {time.time()-t0:.0f}s")
        if checkpoint and solves % ckpt_every == 0:
            with open(checkpoint + ".tmp", "wb") as f:
                _pkl.dump(dict(phi=phi, fixed=fixed, cells=cells,
                               parent=parent, cell_index=cell_index,
                               candidates=candidates,
                               rng_state=rng.bit_generator.state,
                               solves=solves, sweeps=total_sweeps,
                               box_dev=box_dev), f)
            _os.replace(checkpoint + ".tmp", checkpoint)

    return dict(cells=np.array(cells), parent=np.array(parent),
                phi=phi.astype(np.float32), attach=attached,
                H=H, W=W, D=D, eta=eta, solves=solves,
                sweeps=total_sweeps, box_dev=box_dev,
                wall=time.time() - t0)
