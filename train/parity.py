"""Check that the GPU net computes what the torch net computes.

  uv run parity.py <net dir> <dump dir>

The app with WEIGHTS=<net dir> DUMP=<dump dir> (wild 0, rate 1) writes eye 0's
resolved state (the net's input) and new state (its output) after 60 frames. This
runs the torch net on the same input and compares channel by channel.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

from sunk.model import CERT, HID, INV, RGB, VEL, load_export

net_dir, dump_dir = map(Path, sys.argv[1:3])
meta = json.loads((dump_dir / "dump.json").read_text())
n = meta["size"]
assert meta["rate"] == 1 and meta["wild"] == 0, "dump with rate 1 and wild 0"
raw = np.fromfile(dump_dir / "dump.bin", dtype="<f2")
images, at = [], 0
for channels in (4, 4, 4, 1, 4, 4, 4):
    images.append(raw[at:at + n * n * channels].reshape(n, n, channels).astype(np.float32))
    at += n * n * channels
mid_a, mid_b, mid_c, cert, out_a, out_b, out_c = (torch.tensor(i).permute(2, 0, 1)[None] for i in images)

net = load_export(net_dir)
state = torch.zeros(1, net.channels, n, n)
state[:, RGB], state[:, INV] = mid_a[:, :3], mid_a[:, 3]
state[:, VEL], state[:, CERT] = mid_b[:, :3], cert[:, 0]
state[:, HID] = mid_c
with torch.no_grad():
    ref = net(state)
gpu = torch.cat([out_a[:, :3], out_a[:, 3:], out_b[:, :3], out_b[:, 3:], out_c], 1)

valid = (state[:, INV] > 0).float()
print(f"{n}x{n}, holes {1 - valid.mean():.1%}, input |rgb| {state[:, RGB].abs().mean():.3f} |h| {state[:, HID].abs().mean():.3f}")
names = ["r", "g", "b", "inv", "vx", "vy", "vz", "cert", "h0", "h1", "h2", "h3"]
change = (ref - state).abs()
worst = 0.0
for c, name in enumerate(names):
    diff = (gpu[:, c] - ref[:, c]).abs()
    scale = ref[:, c].abs().mean().item() + 1e-6
    worst = max(worst, diff.max().item() / max(ref[:, c].abs().max().item(), 1e-3))
    print(f"{name:5} gpu-torch mean {diff.mean():.2e} max {diff.max():.2e}  (|value| {scale:.3f}, net change {change[:, c].mean():.2e})")
print("PASS" if worst < 0.02 else "FAIL", f"(worst max diff relative to channel range: {worst:.3%})")
