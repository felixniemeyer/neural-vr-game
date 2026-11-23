"""Torch copy of the runtime loop: splat -> resolve -> net. The shaders implement the
same math (see native/shaders); export() writes what they load.

State [B, C, H, W], channels:
  0:3  rgb
  3    inverse distance (1/m, radial like the GPU), 0 = no data (hole)
  4:7  velocity (m per frame, current camera coords: x right, y up, z ahead)
  7    certainty
  8:   hidden state (net.hidden_state channels)

Net input per pixel, as vec4s (all mat4-friendly):
  self      (rgb, inv * near), (h)            exact, native resolution
  level L   points at radius 2^L px, angle L * 60 deg + k * 120 deg (+ the mip centre
            for points = 4), read bilinearly from mip level L of the hole-aware
            pyramid: (rgb, (inv - inv_self) * near), (h); 0 where the pyramid is empty
  extra     (certainty, valid, 1, tau)
Output (4 + hidden_state): d rgb, d log inv, d h; residual update, tanh-bounded.
Holes are first filled from the coarsest-needed pyramid level (push-pull).

Time: a step covers tau ticks (tau = dt * net.rate). The update is an Euler step,
max_change * tau * tanh(out), and the net sees tau, so it can run at any step size.
Certainty decays toward CERT_REST by 0.965 per tick. tau = 0: fill only, no change.
Nets exported before tau existed (no "rate" in net.json) see 0 instead of tau.
"""
import json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RGB, INV, VEL, CERT, HID = slice(0, 3), 3, slice(4, 7), 7, slice(8, None)
CERT_REST, CERT_KEEP = 0.03 / 0.035, 0.965  # certainty decay per tick


def tap_offsets(levels, points):
    """[(level, dx, dy)] in pixels of level 0; y down (image rows)."""
    taps = []
    for level in range(levels):
        radius = 2.0 ** level
        for k in range(3):
            angle = math.radians(90 + 60 * level + 120 * k)
            taps.append((level, radius * math.cos(angle), -radius * math.sin(angle)))
        if points == 4 and level > 0:
            taps.append((level, 0.0, 0.0))
    return taps


def pyramid(state, levels, down=None, near=1.0, weight=None):
    """Hole-aware mips: average of valid pixels only. Returns [(values, coverage)].

    down: optional learned downsampling, one 1x1 conv per level 1.. over the 2x2 children
    (pixel-unshuffled; per child rgb, inv * near, hidden, coverage, normalised). Its output
    is added to the averaged hidden channels, so the coarse levels carry learned features
    (partial-conv style: empty children contribute zeros, their coverage says so).
    weight: optional per-pixel weight [B, 1, H, W] of valid pixels (coverage becomes a sum
    of weights, so only use it where values are read normalised, as the fill does).
    """
    valid = (state[:, INV:INV + 1] > 0).float()
    if weight is not None:
        valid = valid * weight
    level = (state * valid, valid)
    out = []
    for index in range(levels):
        out.append(level)
        if index == levels - 1:
            break
        values, cover = F.avg_pool2d(level[0], 2), F.avg_pool2d(level[1], 2)
        if down is not None:
            v = level[0] / level[1].clamp_min(1e-4)
            child = torch.cat([v[:, RGB], v[:, INV:INV + 1] * near, v[:, HID], (level[1] > 1e-3).float()], 1)
            extra = down[index](F.pixel_unshuffle(child, 2))
            values = values.clone()
            values[:, HID] = values[:, HID] + extra * cover
        level = (values, cover)
    return out


def read(level, grid):
    """Bilinear read of a mip (premultiplied values, coverage) at a grid; normalised."""
    v = F.grid_sample(level[0], grid, mode="bilinear", padding_mode="border", align_corners=False)
    c = F.grid_sample(level[1], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return v / c.clamp_min(1e-4), (c > 1e-3).float()


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.net
        assert n.hidden_state % 4 == 0 and all(h % 4 == 0 for h in n.hidden)
        self.near, self.levels, self.max_change, self.rate = n.near, n.levels, n.max_change, n.rate
        self.background = n.background
        self.gate = n.gate
        self.tau_input = True
        self.taps = tap_offsets(n.levels, n.points)
        self.channels = 8 + n.hidden_state
        self.inputs = 4 * (2 * (1 + len(self.taps)) + 1)
        self.outputs = 4 + n.hidden_state
        sizes = [self.inputs] + list(n.hidden) + [self.outputs]
        self.layers = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(sizes[:-1], sizes[1:]))
        # Learned downsampling (net.down): starts at zero = the plain averaged pyramid.
        child = 4 + 1 + n.hidden_state
        self.down = nn.ModuleList(nn.Conv2d(4 * child, n.hidden_state, 1) for _ in range(n.levels - 1)) \
            if n.down else None
        if self.down is not None:
            with torch.no_grad():
                for conv in self.down:
                    conv.weight.zero_()
                    conv.bias.zero_()
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.1)
            self.layers[-1].bias.zero_()
        self.grids = {}

    def grid(self, h, w, dx, dy, device):
        key = (h, w, dx, dy)
        if key not in self.grids:
            y, x = torch.meshgrid(torch.arange(h, device=device) + 0.5, torch.arange(w, device=device) + 0.5, indexing="ij")
            self.grids[key] = torch.stack([(x + dx) / w * 2 - 1, (y + dy) / h * 2 - 1], -1)[None]
        return self.grids[key]

    def forward(self, state, tau=1.0):
        """tau: ticks this step covers, a number or [B] (per sample)."""
        b, _, h, w = state.shape
        tau = torch.as_tensor(tau, dtype=state.dtype).reshape(-1, 1, 1, 1).expand(b, 1, 1, 1)
        mips = pyramid(state, self.levels, self.down, self.near)
        valid = (state[:, INV:INV + 1] > 0).float()

        # Push-pull hole fill: the finest pyramid level that has data here. With
        # net.background > 0 the fill prefers far pixels (disocclusions reveal background):
        # weight (inv * near)^-background (read normalised, so the scale does not matter).
        fill_mips = mips
        if self.background > 0:
            far = (state[:, INV:INV + 1] * self.near).clamp_min(1e-3) ** -self.background
            fill_mips = pyramid(state, self.levels, weight=far)
        filled = state
        need = 1 - valid
        for level in fill_mips[1:]:
            value, covered = read(level, self.grid(h, w, 0, 0, state.device).expand(b, -1, -1, -1))
            take = need * covered
            filled = filled * (1 - take) + value * take
            need = need * (1 - covered)
        filled = filled.clone()
        filled[:, CERT:CERT + 1] = state[:, CERT:CERT + 1] * valid

        inv_self = filled[:, INV:INV + 1]
        features = [filled[:, RGB], inv_self * self.near, filled[:, HID]]
        for level, dx, dy in self.taps:
            grid = self.grid(h, w, dx, dy, state.device).expand(b, -1, -1, -1)
            value, covered = read(mips[level], grid)
            features += [value[:, RGB] * covered, (value[:, INV:INV + 1] - inv_self) * self.near * covered,
                         value[:, HID] * covered]
        one = torch.ones_like(valid)
        features += [filled[:, CERT:CERT + 1], valid, one, one * tau if self.tau_input else 0 * one]
        x = torch.cat(features, 1)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        delta = self.max_change * tau * torch.tanh(self.layers[-1](x))
        if self.gate:  # net.gate: confident (well-observed) pixels change less
            delta = delta * (1 - self.gate * filled[:, CERT:CERT + 1].clamp(0, 1))

        out = filled.clone()
        out[:, RGB] = (filled[:, RGB] + delta[:, 0:3]).clamp(0, 1.5)
        out[:, INV:INV + 1] = inv_self * torch.exp(delta[:, 3:4])
        out[:, HID] = (filled[:, HID] + delta[:, 4:]).clamp(-4, 4)
        cert = filled[:, CERT:CERT + 1]
        out[:, CERT:CERT + 1] = CERT_REST + (cert - CERT_REST) * CERT_KEEP ** tau
        return out

    def export(self, folder: Path, config, max_tau=1.0):
        """net.json (layout) + weights.bin (float32: per layer W [out][in] row-major, then b;
        then the same for each learned downsampling conv, if any, level 1 first)."""
        folder.mkdir(parents=True, exist_ok=True)
        blobs = []
        for layer in list(self.layers) + list(self.down or []):
            blobs += [layer.weight[:, :, 0, 0].detach().float().numpy().ravel(), layer.bias.detach().float().numpy()]
        np.concatenate(blobs).astype("<f4").tofile(folder / "weights.bin")
        (folder / "net.json").write_text(json.dumps({
            "inputs": self.inputs, "outputs": self.outputs,
            "layers": [[l.in_channels, l.out_channels] for l in self.layers],
            "taps": [list(t) for t in self.taps], "levels": self.levels,
            "hidden_state": config.net.hidden_state, "near": self.near, "max_change": self.max_change,
            "rate": self.rate, "max_tau": max_tau, "background": self.background, "gate": self.gate,
            "down": [[c.in_channels, c.out_channels] for c in self.down] if self.down is not None else [],
        }, indent=1))


def accumulate(state, r, t, rays, origin, focal, full, dt, keep=None):
    """2x2 tent splat of a state (camera c) into camera c' with p' = R (p + v dt) + t.

    rays: [B, H, W, 3] of the crop's pixels (full-frame rays); origin [B, 2]: crop corner
    in the full frame (size `full`); points landing outside the crop are lost. dt [B]
    seconds (velocity is m/s); keep [B, 1, H, W]: which source pixels splat (default
    all). Returns the weighted sums [B, C + 1, H, W] (last channel: the weight).
    """
    b, c, h, w = state.shape
    inv = state[:, INV]
    alive = (inv > 0).float() if keep is None else (inv > 0).float() * keep[:, 0]
    p = F.normalize(rays, dim=-1) / inv.clamp_min(1e-6)[..., None]  # [B, H, W, 3], radial
    p = p + state[:, VEL].permute(0, 2, 3, 1) * torch.as_tensor(dt, dtype=p.dtype).reshape(-1, 1, 1, 1)
    q = torch.einsum("bij,bhwj->bhwi", r, p) + t[:, None, None]
    z = q[..., 2]
    alive = alive * (z > 1e-3).float()
    zs = z.clamp_min(1e-3)
    cx = focal * q[..., 0] / zs + full / 2 - 0.5 - origin[:, 0, None, None]
    cy = -focal * q[..., 1] / zs + full / 2 - 0.5 - origin[:, 1, None, None]

    values = state.clone()
    values[:, INV] = 1 / q.norm(dim=-1).clamp_min(0.05)  # as the GPU: 1 / max(distance, 0.05)
    values[:, VEL] = torch.einsum("bij,bjhw->bihw", r, state[:, VEL])
    values = torch.cat([values, torch.ones_like(inv)[:, None]], 1).flatten(2)  # + weight channel
    acc = torch.zeros(b, c + 1, h * w)
    x0, y0 = cx.floor(), cy.floor()
    for ox in (0, 1):
        for oy in (0, 1):
            x, y = x0 + ox, y0 + oy
            weight = (1 - (cx - x).abs()) * (1 - (cy - y).abs()) * alive
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            index = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).long().flatten(1)
            wv = (weight * inside).flatten(1)[:, None]
            acc.scatter_add_(2, index[:, None].expand(-1, c + 1, -1), values * wv)
    return acc.view(b, c + 1, h, w)


def resolve(acc):
    """Normalise splat sums: the resolved state and the hole mask."""
    weight = acc[:, -1:]
    hole = (weight < 0.01).float()
    out = acc[:, :-1] / weight.clamp_min(0.01) * (1 - hole)
    out[:, CERT:CERT + 1] = acc[:, CERT:CERT + 1].clamp(0, 1) * (1 - hole)  # partial coverage lowers it
    return out, hole


def splat(state, r, t, rays, origin, focal, full, dt=0.0):
    """accumulate + resolve: the state seen from camera c', and its holes."""
    return resolve(accumulate(state, r, t, rays, origin, focal, full, dt))


def load_export(folder: Path) -> Net:
    """A Net from an export (net.json + weights.bin), as the runtime loads it."""
    from sunk.config import Config
    spec = json.loads((Path(folder) / "net.json").read_text())
    config = Config()
    config.net.levels = spec["levels"]
    taps_per_level = sum(1 for t in spec["taps"] if t[0] == 1)
    config.net.points = taps_per_level
    config.net.hidden = [o for _, o in spec["layers"][:-1]]
    config.net.hidden_state = spec["hidden_state"]
    config.net.near = spec["near"]
    config.net.max_change = spec["max_change"]
    config.net.rate = spec.get("rate", 10.0)  # older nets: one tick per 10 fps video frame
    config.net.down = bool(spec.get("down"))
    config.net.background = spec.get("background", 0.0)
    config.net.gate = spec.get("gate", 0.0)
    net = Net(config)
    net.tau_input = "rate" in spec
    assert [list(t) for t in net.taps] == [list(t) for t in spec["taps"]], "tap layout differs"
    flat = torch.tensor(np.fromfile(Path(folder) / "weights.bin", dtype="<f4"))
    at = 0
    with torch.no_grad():
        for layer in list(net.layers) + list(net.down or []):
            n = layer.weight.numel()
            layer.weight.copy_(flat[at:at + n].view_as(layer.weight))
            at += n
            layer.bias.copy_(flat[at:at + layer.bias.numel()])
            at += layer.bias.numel()
    assert at == len(flat)
    return net
