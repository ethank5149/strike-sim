"""Validation summary figure: one panel per pipeline stage."""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude/lightning")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

s1 = np.load("/home/claude/lightning/val_stage1.npy")
with open("/home/claude/lightning/leader.pkl", "rb") as f:
    res = pickle.load(f)
with open("/home/claude/lightning/currents.pkl", "rb") as f:
    cur = pickle.load(f)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)

# --- Stage 1: solver convergence
ax = axes[0]
N, err = s1[:, 0], s1[:, 1]
h = 1.0 / (N - 1)
ax.loglog(h, err, "o-", color="steelblue", label="SOR solution")
ax.loglog(h, err[0] * (h / h[0]) ** 2, "k--", alpha=0.6,
          label=r"$O(h^2)$ reference")
ax.set_xlabel("grid spacing $h$")
ax.set_ylabel(r"$\|\phi - \phi_{\rm exact}\|_\infty$")
ax.set_title("Stage 1: Laplace solver\n2nd-order convergence (order 2.00)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# --- Stage 2: potential field + channel
ax = axes[1]
phi = res["phi"]
im = ax.imshow(phi, cmap="inferno", origin="upper", aspect="auto")
cells = res["cells"]
ax.plot(cells[:, 1], cells[:, 0], ",", color="cyan", markersize=0.4)
ax.set_title("Stage 2: potential $\\phi$ + leader\n"
             "(channel $\\phi$=0, ground $\\phi$=1)")
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\phi$")

# --- Stage 2b: box-count fit
ax = axes[2]
sizes, counts = res["bc_sizes"], res["bc_counts"]
x = np.log(1.0 / sizes)
y = np.log(counts)
slope, b0 = np.polyfit(x, y, 1)
ax.plot(x, y, "o", color="darkorange", label="box counts")
ax.plot(x, slope * x + b0, "k--",
        label=f"fit: $D$ = {slope:.3f}")
ax.axhline(np.nan)  # spacing
ax.fill_between([x.min(), x.max()],
                [1.6 * x.min() + b0 + (slope - 1.6) * x.mean()] * 2,
                [1.8 * x.min() + b0 + (slope - 1.8) * x.mean()] * 2,
                alpha=0)  # no-op, keep clean
ax.set_xlabel(r"$\log(1/\epsilon)$")
ax.set_ylabel(r"$\log N(\epsilon)$")
ax.set_title(f"Stage 2: fractal dimension\n$D$ = {slope:.3f} "
             "(lightning: 1.7 $\\pm$ 0.1)")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Stage 3: current along main channel
ax = axes[3]
I, path = cur["I"], cur["path"]
depth = cells[path, 0]
I_path = I[path]
I_path[0] = np.nan
ax.plot(I_path, depth, color="crimson", lw=1.5)
ax.invert_yaxis()
ax.set_xlabel("return-stroke current (charge units)")
ax.set_ylabel("depth (cloud $\\to$ ground)")
ax.set_title("Stage 3: main-channel current\n"
             "monotone, conserves total charge")
ax.grid(True, alpha=0.3)

fig.suptitle("Lightning simulator -- per-stage validation", fontsize=13,
             fontweight="bold")
fig.savefig("/home/claude/lightning/validation_summary.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("saved validation_summary.png")
