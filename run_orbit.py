import pickle, sys, numpy as np
sys.path.insert(0, "/home/claude/lightning")
from temporal import write_video

res = pickle.load(open("/home/claude/lightning/leader3d.pkl", "rb"))
cur = pickle.load(open("/home/claude/lightning/currents3d.pkl", "rb"))

# Gentle orbit about the vertical axis through the channel root.
# smoothstep easing over the whole clip: motion is slowest at the start
# (leader) and end (decay tail) and passes evenly through the return
# stroke -- a calm, continuous parallax reveal rather than a spin.
AZ0, AZ1 = -20.0, 20.0
def orbit(t):
    e = t * t * (3.0 - 2.0 * t)        # smoothstep
    return AZ0 + (AZ1 - AZ0) * e

write_video(res, cur, "/home/claude/lightning/lightning3d_orbit.mp4",
            n_frames=180, fps=30, out_w=960, out_h=540,
            azimuth_fn=orbit, log=print)
print("orbit video done")
