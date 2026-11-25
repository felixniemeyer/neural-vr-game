"""Check that the GPU net computes what the torch net computes (all layouts; cube-v1 per face).

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


def cube():
    """cube-v1 (net_layout 3): the dump holds the 3 x 2 face atlas (F x F faces, each in a
    cell with a gutter). Each face
    runs through the torch net on its own (forward_v2 with the scale input); taps that
    leave a face read the neighbour face on the GPU and the face's border in torch, so
    only texels at least MARGIN from the face edges count. The GPU also gives texels with
    a weight below 1 some of the scene's depth (net.frag), which torch doesn't: reported
    over all interior texels and over those with weight >= 0.999."""
    import math
    from sunk.model import scale_code
    MARGIN = 40
    h, w = meta["height"], meta["width"]
    f, g = meta["face"], meta["gutter"]  # faces sit in cells of f + 2 g (a gutter around each)
    images, at = [], 0
    for _ in range(7):
        images.append(torch.tensor(raw[at:at + h * w * 4].reshape(h, w, 4).astype(np.float32)).permute(2, 0, 1)[None])
        at += h * w * 4
    mid_a, mid_b, mid_c, mid_d, out_a, out_b, out_c = images
    net = load_export(net_dir)
    assert net.layout == "cube-v1", "dumped with a cube-v1 net"
    wt = mid_d[:, 3:4]
    k = torch.where(wt > 0, 1 / wt.clamp_min(1e-12), torch.zeros_like(wt))
    state = torch.zeros(1, net.channels, h, w)
    state[:, RGB], state[:, INV] = (mid_a[:, :3] * k), (mid_a[:, 3] * k[:, 0])
    state[:, VEL], state[:, CERT] = mid_b[:, :3] * k, wt[:, 0]
    state[:, HID] = mid_c * k
    motion = mid_d[:, :3] * k
    scale = scale_code(math.pi / 2 / f)
    gpu = torch.cat([out_a, out_b[:, :3], out_c], 1)
    names = ["r", "g", "b", "inv", "vx", "vy", "vz", "h0", "h1", "h2", "h3"]
    diffs, refs, fulls = [], [], []
    with torch.no_grad():
        for face in range(6):
            y, x = (face // 3) * (f + 2 * g) + g, (face % 3) * (f + 2 * g) + g
            cut = lambda t: t[..., y:y + f, x:x + f]
            ref = net(cut(state), tau, cut(motion), scale)
            ref = torch.cat([ref[:, RGB], ref[:, INV:INV + 1], ref[:, VEL], ref[:, HID]], 1)
            inner = lambda t: t[..., MARGIN:-MARGIN, MARGIN:-MARGIN]
            diffs.append(inner((cut(gpu) - ref).abs()))
            refs.append(inner(ref))
            fulls.append(inner(cut(wt)) >= 0.999)
    diff, ref, full = torch.cat(diffs, -1), torch.cat(refs, -1), torch.cat(fulls, -1)
    print(f"6 faces of {f}x{f}, tau {tau}, scale input {float(scale):.3f}, weight >= 0.999 on {full.float().mean():.1%} of the interior")
    worst = 0.0
    for c, name in enumerate(names):
        d = diff[:, c]
        span = max(ref[:, c].abs().max().item(), 1e-3)
        sure = d[full[:, 0]] if full.any() else d.flatten()
        worst = max(worst, sure.flatten().quantile(0.999).item() / span)
        print(f"{name:5} gpu-torch mean {d.mean():.2e} max {d.max():.2e}, weight >= 0.999: mean {sure.mean():.2e} (|value| {ref[:, c].abs().mean():.3f})")
    print("PASS" if worst < 0.02 else "FAIL", f"(worst 99.9th percentile diff where weight >= 0.999, relative to channel range: {worst:.3%})")
    sys.exit(0)


if meta.get("net_layout") == 3:
    cube()
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
