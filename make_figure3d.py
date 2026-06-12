"""3-D validation summary figure."""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude/lightning")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

s1 = np.load("/home/claude/lightning/val3d_stage1.npy")
with open("/home/claude/lightning/leader3d.pkl", "rb") as f:
    res = pickle.load(f)
with open("/home/claude/lightning/currents3d.pkl", "rb") as f:
    cur = pickle.load(f)
cells = res["cells"]

fig = plt.figure(figsize=(17, 4.6), constrained_layout=True)

# Stage 1: convergence
ax = fig.add_subplot(1, 4, 1)
N, err = s1[:, 0], s1[:, 1]
h = 1.0 / (N - 1)
ax.loglog(h, err, "o-", color="steelblue", label="SOR solution")
ax.loglog(h, err[0] * (h / h[0]) ** 2, "k--", alpha=0.6,
          label=r"$O(h^2)$ reference")
ax.set_xlabel("grid spacing $h$")
ax.set_ylabel(r"$\|\phi-\phi_{\rm exact}\|_\infty$")
ax.set_title("Stage 1: 3-D Laplace solver\n(7-point stencil, order 2.00)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

# Stage 2: 3-D channel, colored by depth (parallax axis)
ax = fig.add_subplot(1, 4, 2, projection="3d")
I = cur["I"].copy(); I[0] = 1
sub = np.arange(len(cells))[::2]
sc = ax.scatter(cells[sub, 1], cells[sub, 2], res["H"] - cells[sub, 0],
                c=cells[sub, 2], cmap="coolwarm", s=0.7, alpha=0.8)
ax.set_title(f"Stage 2: 3-D channel\n{len(cells)} cells, "
             f"$\\eta$ = {res['eta']:.0f}")
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=8, azim=-65)

# Stage 2: box-count fits, 3-D and projected
ax = fig.add_subplot(1, 4, 3)
for (s, c), lab, col in ((res["bc3"], f"3-D: $D$ = {res['D3']:.3f}",
                          "darkorange"),
                         (res["bcp"], f"proj.: $D$ = {res['Dproj']:.3f}",
                          "seagreen")):
    x, y = np.log(1 / s), np.log(c)
    sl, b0 = np.polyfit(x, y, 1)
    ax.plot(x, y, "o", color=col)
    ax.plot(x, sl * x + b0, "--", color=col, label=lab)
ax.set_xlabel(r"$\log(1/\epsilon)$"); ax.set_ylabel(r"$\log N(\epsilon)$")
ax.set_title("Stage 2: fractal dimensions\n"
             "projected $D$ vs lightning 1.7 $\\pm$ 0.1")
ax.legend(); ax.grid(True, alpha=0.3)

# Stage 3: main-channel current
ax = fig.add_subplot(1, 4, 4)
path = cur["path"]
Ip = I[path].astype(float); Ip[0] = np.nan
ax.plot(Ip, cells[path, 0], color="crimson", lw=1.5)
ax.invert_yaxis()
ax.set_xlabel("return-stroke current (charge units)")
ax.set_ylabel("depth (cloud $\\to$ ground)")
ax.set_title("Stage 3: main-channel current\nmonotone, exact conservation")
ax.grid(True, alpha=0.3)

fig.suptitle("3-D lightning simulator -- per-stage validation",
             fontsize=13, fontweight="bold")
fig.savefig("/home/claude/lightning/validation3d_summary.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("saved")
