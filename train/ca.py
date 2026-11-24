"""Hand-written cellular automata as nets in the runtime's format (no training):
uv run ca.py <out folder> [mu] [width] [seed]   e.g. uv run ca.py ../ca/lenia, then WEIGHTS=ca/lenia ./dev.sh

Lenia-like rule on hidden channel 0 (the activation a, kept in 0..1), riding on the splat
reprojection like a trained net, so its patterns stay attached to the 3D world:
  u      = mean of a over the taps of mip levels 1..2 (a ring, 2-4 px), plus the local
           luminance contrast, plus seed * the pixel's own luminance (the scene seeds it)
  growth = triangle bump around mu (half width `width`), mapped to -1..1
  da     = growth, pushed back into 0..1
  d rgb  = a pull toward a palette colour, cool where a = 0, warm where a = 1
Depth and the other hidden channels stay as they are.
"""
import json, sys
from pathlib import Path

import numpy as np

from sunk.model import tap_offsets

out = Path(sys.argv[1])
mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
width = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
seed = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
levels, near = 5, 10.0
taps = tap_offsets(levels, 3)
inputs = 4 * (2 * (1 + len(taps)) + 1)
units = 12


def col(tap, field):
    """Input index: tap -1 = self; field 0..2 rgb, 3 inv / depth difference, 4..7 hidden."""
    return 8 * (tap + 1) + field


ring = [i for i, (level, _, _) in enumerate(taps) if level in (1, 2)]
W1, b1 = np.zeros((units, inputs)), np.zeros(units)
u = np.zeros(inputs)
for i in ring:
    u[col(i, 4)] += 1 / len(ring)  # mean activation
    for c in range(3):  # luminance contrast: taps minus self
        u[col(i, c)] += 0.5 / 3 / len(ring)
        u[col(-1, c)] -= 0.5 / 3 / len(ring)
for c in range(3):  # the scene's own brightness seeds (and keeps feeding) the rule
    u[col(-1, c)] += seed / 3
for k, centre in enumerate([mu - width, mu, mu + width]):  # relu(u - x) for the triangle
    W1[k], b1[k] = u, -centre
W1[3, col(-1, 4)], b1[3] = 1, -1     # relu(a - 1)
W1[4, col(-1, 4)] = -1               # relu(-a)
W1[5, col(-1, 4)] = 1                # relu(a)
b1[6] = 1                            # constant 1
for c in range(3):
    W1[8 + c, col(-1, c)] = 1        # relu(rgb) = rgb

W2, b2 = np.zeros((8, units)), np.zeros(8)
tri = np.array([1, -2, 1]) / width   # 0..1 bump
W2[4, :3] = 2 * tri                  # da = 2 * bump - 1 ...
W2[4, 6] = -1
W2[4, 3], W2[4, 4] = -8, 8           # ... pushed back into 0..1
warm, cool = np.array([1.0, 0.45, 0.15]), np.array([0.1, 0.3, 0.8])
pull = 0.3                           # d rgb = pull * (palette(a) - rgb), palette = cool .. warm
W2[0:3, 5] = pull * (warm - cool)
W2[0:3, 6] = pull * cool
for c in range(3):
    W2[c, 8 + c] = -pull
out.mkdir(parents=True, exist_ok=True)
np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype("<f4").tofile(out / "weights.bin")
(out / "net.json").write_text(json.dumps({
    "inputs": inputs, "outputs": 8, "layers": [[inputs, units], [units, 8]],
    "taps": [list(t) for t in taps], "levels": levels, "hidden_state": 4, "near": near,
    "max_change": 0.25, "rate": 24.0, "max_tau": 2.0, "background": 0.0, "gate": 0.0, "copy": 0.0,
    "down": [], "cnn": [], "ca": {"rule": "lenia", "mu": mu, "width": width, "seed": seed},
}, indent=1))
print(f"{out}: lenia-like CA, mu {mu}, width {width}, seed {seed}, {W1.size + b1.size + W2.size + b2.size} weights")
