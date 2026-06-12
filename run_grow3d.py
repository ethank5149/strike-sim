import pickle
import sys
sys.path.insert(0, "/home/claude/lightning")
from physics3d import grow_leader3

res = grow_leader3(H=160, W=112, D=112, eta=2.0, rng_seed=7,
                   cells_per_solve=4, verbose_every=50,
                   checkpoint="/home/claude/lightning/grow3d.ckpt")
print(f"done: {len(res['cells'])} cells, {res['solves']} solves, "
      f"{res['sweeps']} sweeps, box dev {res['box_dev']:.2e}, "
      f"{res['wall']:.0f}s wall")
with open("/home/claude/lightning/leader3d.pkl", "wb") as f:
    pickle.dump(res, f)
