"""Check that the GPU net computes what the torch net computes.

  uv run parity.py <net dir> <dump dir>

The app with WEIGHTS=<net dir> DUMP=<dump dir> [DUMP_TAU=1] (wild 0) writes eye 0's
resolved state (the net's input) and new state (its output) after 60 frames, one net
update of DUMP_TAU ticks. This
runs the torch net on the same input and compares channel by channel.
reproject-v2 (dump.json net_layout 2): the resolved images are premultiplied by the
splat weight (D = (motion, 1) * w); the torch net gets them unpremultiplied, the weight in
the certainty channel and D's motion directly (screen_motion is checked separately).
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

from sunk.model import CERT, HID, INV, RGB, VEL, load_export

net_dir, dump_dir = map(Path, sys.argv[1:3])
meta = json.loads((dump_dir / "dump.json").read_text())
n = meta["size"]
assert meta["wild"] == 0, "dump with wild 0"
tau = meta["tau"]
raw = np.fromfile(dump_dir / "dump.bin", dtype="<f2")
images, at = [], 0
v2 = meta.get("net_layout") == 2
for channels in (4, 4, 4, 4, 4, 4, 4):
    images.append(raw[at:at + n * n * channels].reshape(n, n, channels).astype(np.float32))
    at += n * n * channels
mid_a, mid_b, mid_c, mid_d, out_a, out_b, out_c = (torch.tensor(i).permute(2, 0, 1)[None] for i in images)

net = load_export(net_dir)
assert (net.layout == "reproject-v2") == v2, "dumped with a net of another layout"
state = torch.zeros(1, net.channels, n, n)
motion = None
if v2:  # unpremultiply (the net premultiplies again: same mips as the GPU's)
    w = mid_d[:, 3:4]
    k = torch.where(w > 0, 1 / w.clamp_min(1e-12), torch.zeros_like(w))
    mid_a, mid_b, mid_c, motion = mid_a * k, mid_b * k, mid_c * k, mid_d[:, :3] * k
    cert = w
else:
    cert = mid_d[:, :1]
state[:, RGB], state[:, INV] = mid_a[:, :3], mid_a[:, 3]
state[:, VEL], state[:, CERT] = mid_b[:, :3], cert[:, 0]
state[:, HID] = mid_c
with torch.no_grad():
    ref = net(state, tau, motion)
if v2:
    ref[:, CERT] = state[:, CERT]  # the GPU passes the weight on in B.w (for the display)
gpu = torch.cat([out_a[:, :3], out_a[:, 3:], out_b[:, :3], out_b[:, 3:], out_c], 1)

valid = (state[:, CERT] > 0.01).float() if v2 else (state[:, INV] > 0).float()
print(f"{n}x{n}, tau {tau}, holes {1 - valid.mean():.1%}, input |rgb| {state[:, RGB].abs().mean():.3f} |h| {state[:, HID].abs().mean():.3f}")
names = ["r", "g", "b", "inv", "vx", "vy", "vz", "cert", "h0", "h1", "h2", "h3"]
change = (ref - state).abs()
worst = 0.0
for c, name in enumerate(names):
    diff = (gpu[:, c] - ref[:, c]).abs()
    scale = ref[:, c].abs().mean().item() + 1e-6
    # 99.9th percentile: a few hole pixels may flip the coverage threshold of a coarse
    # mip read (f16 storage, hardware bilinear weights) and fill differently.
    worst = max(worst, diff.flatten().quantile(0.999).item() / max(ref[:, c].abs().max().item(), 1e-3))
    print(f"{name:5} gpu-torch mean {diff.mean():.2e} max {diff.max():.2e}  (|value| {scale:.3f}, net change {change[:, c].mean():.2e})")
print("PASS" if worst < 0.02 else "FAIL", f"(worst 99.9th percentile diff relative to channel range: {worst:.3%})")
