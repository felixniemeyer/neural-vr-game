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

That is layout reproject-v1 (net.layout). reproject-v2 (forward_v2) differs:
  - the certainty channel holds the raw splat weight w of this step (resolve: values =
    sums / (w + EPS), no hole threshold): 0 = nothing landed, 1 = a clean 1:1
    reprojection, > 1 = compression or a mix. Depth is only depth.
  - the pyramid is premultiplied by w (mips average x w and w, reads divide), and the
    push-pull fill is continuous: need = 1 - min(w, 1).
  - inputs: self (rgb, inv * near), (h), (motion, w); per tap (rgb, (inv - inv_self) *
    near * min(w_t, 1)), (h), (motion, w_t); extra (1, tau, 0, 0). motion =
    screen_motion(): the apparent motion without camera rotation, px and log distance
    per tick.
  - no certainty decay, no gate.

cube-v1 is v2's net on a world-fixed EAC sphere (sunk/sphere.py, train_cube.py): the
state is splatted (or pulled) on the sphere, motion is each point's move (texels of its
gravity frame, sphere.frame_motion, and log distance per tick), the taps sit in each
texel's gravity frame (grids=, sphere.tap_grids: upright everywhere, no face seams), and
the extra input is (1, tau, scale, 0) with scale = log2(texel angle / 1 arcmin) / 4.
forward() can build the pyramid over a canvas plus a context ring but evaluate only the
inner rectangle (inner=), and tau may be per texel.
"""
import json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RGB, INV, VEL, CERT, HID = slice(0, 3), 3, slice(4, 7), 7, slice(8, None)
CERT_REST, CERT_KEEP = 0.03 / 0.035, 0.965  # certainty decay per tick
V1, V2, CUBE = "reproject-v1", "reproject-v2", "cube-v1"
EPS = 1e-3  # v2: values = premultiplied sums / (weight + EPS)


def screen_motion(state, cam_vel, rays, focal, rate):
    """v2's motion input [B, 3, H, W]: how each point moves on screen without the camera's
    rotation (that only shifts the image, it reveals nothing): (dx, dy) in px and the
    change of log distance, per tick. cam_vel [B, 3]: the camera translation per second
    in camera coords (+ for a static point: p' = R p + t, so t / dt); rays [B, H, W, 3];
    pinhole projection x = focal X / Z, y = -focal Y / Z (rows down). 0 without depth."""
    inv = state[:, INV]
    p = F.normalize(rays, dim=-1) / inv.clamp_min(1e-6)[..., None]
    q = p + (state[:, VEL].permute(0, 2, 3, 1) + cam_vel[:, None, None]) / rate

    def pixel(x):
        z = x[..., 2:3].clamp_min(1e-3)
        return focal * x[..., :2] / z * torch.tensor([1.0, -1.0])

    out = torch.cat([pixel(q) - pixel(p), (q.norm(dim=-1, keepdim=True) / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_min(1e-6).log()], -1)
    alive = ((inv > 1e-6) & (p[..., 2] > 1e-3) & (q[..., 2] > 1e-3)).float()[..., None]
    return (out * alive).permute(0, 3, 1, 2)


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


def encode(state, mips, convs, near):
    """CNN pyramid (net.cnn): per level 1.. a 1x1 conv over the 2x2 children (pixel-unshuffled;
    per child the normalised rgb, inv * near, hidden, the child's features, coverage) gives
    that level's features, ReLU, widening with depth. Premultiplied by coverage like the
    mips, so read() normalises them. Returns [None, (features, coverage), ...]."""
    out = [None]
    feats = None
    for index, conv in enumerate(convs):
        values, cover = mips[index]
        v = values / cover.clamp_min(1e-4)
        parts = [v[:, RGB], v[:, INV:INV + 1] * near, v[:, HID], (cover > 1e-3).float()]
        if feats is not None:
            parts.append(feats)
        f = F.relu(conv(F.pixel_unshuffle(torch.cat(parts, 1), 2)))
        coarse = mips[index + 1][1]
        out.append((f * coarse, coarse))
        feats = f
    return out


def read_v2(level, grid):
    """v2: bilinear read of a premultiplied mip at a grid: (sums / (w + EPS), w)."""
    v = F.grid_sample(level[0], grid, mode="bilinear", padding_mode="border", align_corners=False)
    c = F.grid_sample(level[1], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return v / (c + EPS), c


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
        self.copy = n.copy
        self.layout = n.layout
        self.reproject, self.nearness = n.reproject, n.nearness
        assert self.layout in (V1, V2, CUBE), f"net.layout: {V1}, {V2} or {CUBE}"
        if self.layout in (V2, CUBE):
            assert not (n.gate or n.background or n.copy or n.down or n.cnn), \
                "reproject-v2 has no gate, background, copy, down or cnn"
        self.tau_input = True
        self.taps = tap_offsets(n.levels, n.points)
        self.channels = 8 + n.hidden_state
        groups = 2 if self.layout == V1 else 3  # vec4s per position: A, C (+ motion, w)
        self.inputs = 4 * (groups * (1 + len(self.taps)) + 1)
        self.outputs = 4 + n.hidden_state
        sizes = [self.inputs] + list(n.hidden) + [self.outputs]
        self.layers = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(sizes[:-1], sizes[1:]))
        # Learned downsampling (net.down): starts at zero = the plain averaged pyramid.
        child = 4 + 1 + n.hidden_state
        self.down = nn.ModuleList(nn.Conv2d(4 * child, n.hidden_state, 1) for _ in range(n.levels - 1)) \
            if n.down else None
        # Infilling vectors (net.copy px): a zero-initialised head predicts, for hole pixels,
        # where to copy the filled state from (offset within +-copy px) instead of keeping
        # push-pull's blur. Starts as the plain net.
        self.copy_head = nn.Conv2d(sizes[-2], 2, 1) if n.copy else None
        if self.copy_head is not None:
            with torch.no_grad():
                self.copy_head.weight.zero_()
                self.copy_head.bias.zero_()
        # CNN pyramid (net.cnn): an encoder with widening features per level, read by the
        # same taps through a zero-initialised extra input layer (starts as the plain net).
        self.cnn = list(n.cnn)
        assert not self.cnn or len(self.cnn) == n.levels - 1, "net.cnn: one width per level 1.."
        widths = [0] + self.cnn
        self.encoder = nn.ModuleList(nn.Conv2d(4 * (child + widths[i]), widths[i + 1], 1)
                                     for i in range(len(self.cnn))) if self.cnn else None
        self.wide_in = nn.Conv2d(sum(widths[level] for level, _, _ in self.taps), sizes[1], 1, bias=False) \
            if self.cnn else None
        if self.wide_in is not None:
            with torch.no_grad():
                self.wide_in.weight.zero_()
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

    def forward(self, state, tau=1.0, motion=None, scale=None, inner=None, grids=None):
        """tau: ticks this step covers, a number or [B] (per sample), cube-v1 also
        [B, 1, h, w] (per texel of the inner rectangle). motion: v2's screen_motion() of
        the state [B, 3, H, W] (None: zero). cube-v1: scale [B], inner (y0, y1, x0, x1),
        grids: per tap its grid_sample grid [B, hi, wi, 2] over the inner rectangle
        (sphere.tap_grids: the taps in each texel's gravity frame; None: pixel offsets)."""
        if self.layout != V1:
            return self.forward_v2(state, tau, motion, scale, inner, grids)
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

        wide = None
        if self.encoder is not None:
            coded, reads = encode(state, mips, self.encoder, self.near), []
            for level, dx, dy in self.taps:
                if level > 0:
                    value, covered = read(coded[level], self.grid(h, w, dx, dy, state.device).expand(b, -1, -1, -1))
                    reads.append(value * covered)
            wide = self.wide_in(torch.cat(reads, 1))
        for index, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if index == 0 and wide is not None:
                x = x + wide
            x = F.relu(x)
        if self.copy_head is not None:
            offset = self.copy * torch.tanh(self.copy_head(x)).permute(0, 2, 3, 1)  # [B, H, W, 2] px
            scale = torch.tensor([2.0 / w, 2.0 / h], device=state.device)
            grid = self.grid(h, w, 0, 0, state.device).expand(b, -1, -1, -1) + offset * scale
            copied = F.grid_sample(filled, grid, mode="bilinear", padding_mode="border", align_corners=False)
            hole = 1 - valid
            filled = filled.clone()
            for channels in (RGB, slice(INV, INV + 1), HID):
                filled[:, channels] = filled[:, channels] * valid + copied[:, channels] * hole
            inv_self = filled[:, INV:INV + 1]
        delta = self.max_change * tau * torch.tanh(self.layers[-1](x))
        if self.gate:  # net.gate: confident (well-observed) pixels change less
            delta = delta * (1 - self.gate * filled[:, CERT:CERT + 1].clamp(0, 1))

        out = filled.clone()
        self.raw = filled[:, RGB] + delta[:, 0:3]  # before the clamp (train.range)
        out[:, RGB] = self.raw.clamp(0, 1.5)
        out[:, INV:INV + 1] = inv_self * torch.exp(delta[:, 3:4])
        out[:, HID] = (filled[:, HID] + delta[:, 4:]).clamp(-4, 4)
        cert = filled[:, CERT:CERT + 1]
        out[:, CERT:CERT + 1] = CERT_REST + (cert - CERT_REST) * CERT_KEEP ** tau
        return out

    def forward_v2(self, state, tau, motion, scale=None, inner=None, grids=None):
        """reproject-v2 (see the module docstring); net.frag's LAYOUT 2 branch. cube-v1:
        the same with the scale input and an inner rectangle (pyramid over all of state,
        output only for state[..., y0:y1, x0:x1])."""
        b, _, h, w = state.shape
        y0, y1, x0, x1 = inner or (0, h, 0, w)
        tau = torch.as_tensor(tau, dtype=state.dtype)
        tau = tau if tau.dim() == 4 else tau.reshape(-1, 1, 1, 1).expand(b, 1, 1, 1)
        if motion is None:
            motion = torch.zeros(b, 3, h, w, dtype=state.dtype)
        weight_all = state[:, CERT:CERT + 1].clamp_min(0)
        # Premultiplied pyramid of (state, motion) by the weight, as the GPU's mips of the
        # resolved images; level 0 read back the same way (sums / (w + EPS)).
        full = torch.cat([state, motion], 1)
        mips = [(full * weight_all, weight_all)]
        for _ in range(self.levels - 1):
            mips.append((F.avg_pool2d(mips[-1][0], 2), F.avg_pool2d(mips[-1][1], 2)))
        cut = lambda g: g[:, y0:y1, x0:x1].expand(b, -1, -1, -1)
        centre = cut(self.grid(h, w, 0, 0, state.device))
        weight = weight_all[..., y0:y1, x0:x1]
        # Continuous push-pull: the pixel's own value counts min(w, 1), each level what is
        # still missing times its coverage; normalised by what was gathered (a hole is
        # not pulled toward 0 where even the coarsest level is only partly covered).
        own = weight.clamp(max=1)
        gathered = mips[0][0][..., y0:y1, x0:x1] / (weight + EPS) * own
        total, need = own, 1 - own
        for level in mips[1:]:
            value, covered = read_v2(level, centre)
            take = need * covered.clamp(max=1)
            gathered = gathered + value * take
            total = total + take
            need = need * (1 - covered.clamp(max=1))
        filled = gathered / total.clamp_min(1e-6)
        m = filled[:, -3:]
        filled = filled[:, :-3]

        inv_self = filled[:, INV:INV + 1]
        features = [filled[:, RGB], inv_self * self.near, filled[:, HID], m, weight]
        for k, (level, dx, dy) in enumerate(self.taps):
            value, covered = read_v2(mips[level], grids[k] if grids is not None else cut(self.grid(h, w, dx, dy, state.device)))
            features += [value[:, RGB], (value[:, INV:INV + 1] - inv_self) * self.near * covered.clamp(max=1),
                         value[:, HID.start:self.channels], value[:, -3:], covered]
        one = torch.ones_like(weight)
        code = 0 * one if scale is None else one * torch.as_tensor(scale, dtype=state.dtype).reshape(-1, 1, 1, 1)
        features += [one, one * tau, code, 0 * one]
        x = torch.cat(features, 1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        delta = self.max_change * tau * torch.tanh(self.layers[-1](x))

        out = filled.clone()
        self.raw = filled[:, RGB] + delta[:, 0:3]  # before the clamp (train.range)
        out[:, RGB] = self.raw.clamp(0, 1.5)
        out[:, INV:INV + 1] = inv_self * torch.exp(delta[:, 3:4])
        out[:, HID] = (filled[:, HID] + delta[:, 4:]).clamp(-4, 4)
        out[:, CERT:CERT + 1] = 1  # the next resolve replaces it with that step's weight
        return out

    def export(self, folder: Path, config, max_tau=1.0):
        """net.json (layout) + weights.bin (float32: per layer W [out][in] row-major, then b;
        then the same for each learned downsampling conv, if any, level 1 first, then the copy head)."""
        folder.mkdir(parents=True, exist_ok=True)
        blobs = []
        for layer in list(self.layers) + list(self.down or []) + ([self.copy_head] if self.copy_head is not None else []):
            blobs += [layer.weight[:, :, 0, 0].detach().float().numpy().ravel(), layer.bias.detach().float().numpy()]
        np.concatenate(blobs).astype("<f4").tofile(folder / "weights.bin")
        (folder / "net.json").write_text(json.dumps({
            "inputs": self.inputs, "outputs": self.outputs,
            "layers": [[l.in_channels, l.out_channels] for l in self.layers],
            "taps": [list(t) for t in self.taps], "levels": self.levels,
            "hidden_state": config.net.hidden_state, "near": self.near, "max_change": self.max_change,
            "layout": self.layout, "rate": self.rate, "max_tau": max_tau, "background": self.background, "gate": self.gate, "copy": self.copy,
            "reproject": self.reproject, "nearness": self.nearness,
            "down": [[c.in_channels, c.out_channels] for c in self.down] if self.down is not None else [],
            "cnn": self.cnn,  # torch only: weights.bin leaves the encoder out, the app cannot run it
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


def resolve(acc, layout=V1):
    """Normalise splat sums: the resolved state and the hole mask. v2: sums / (w + EPS),
    the weight in the certainty channel, no threshold (the mask is for the losses)."""
    weight = acc[:, -1:]
    hole = (weight < 0.01).float()
    if layout == V2:
        out = acc[:, :-1] / (weight + EPS)
        out[:, CERT:CERT + 1] = weight
        return out, hole
    out = acc[:, :-1] / weight.clamp_min(0.01) * (1 - hole)
    out[:, CERT:CERT + 1] = acc[:, CERT:CERT + 1].clamp(0, 1) * (1 - hole)  # partial coverage lowers it
    return out, hole


def splat(state, r, t, rays, origin, focal, full, dt=0.0, layout=V1):
    """accumulate + resolve: the state seen from camera c', and its holes."""
    return resolve(accumulate(state, r, t, rays, origin, focal, full, dt), layout)


def load_export(folder: Path) -> Net:
    """A Net from an export (net.json + weights.bin), as the runtime loads it."""
    from sunk.config import Config
    spec = json.loads((Path(folder) / "net.json").read_text())
    config = Config()
    config.net.layout = spec.get("layout", V1)
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
    config.net.copy = spec.get("copy", 0.0)
    config.net.reproject = spec.get("reproject", "pull")
    config.net.nearness = spec.get("nearness", 2.0)
    net = Net(config)
    net.tau_input = "rate" in spec
    assert [list(t) for t in net.taps] == [list(t) for t in spec["taps"]], "tap layout differs"
    flat = torch.tensor(np.fromfile(Path(folder) / "weights.bin", dtype="<f4"))
    at = 0
    with torch.no_grad():
        for layer in list(net.layers) + list(net.down or []) + ([net.copy_head] if net.copy_head is not None else []):
            n = layer.weight.numel()
            layer.weight.copy_(flat[at:at + n].view_as(layer.weight))
            at += n
            layer.bias.copy_(flat[at:at + layer.bias.numel()])
            at += layer.bias.numel()
    assert at == len(flat)
    return net


def scale_code(angle):
    """cube-v1's scale input from the texel angle (rad): log2(angle / 1 arcmin) / 4."""
    return torch.log2(torch.as_tensor(angle) / math.radians(1 / 60)) / 4


def init_from(net, parent):
    """Start `net` from an export of another layout: v1 -> v2 (init_from_v1), v2 ->
    cube-v1 (same shapes; the scale column, unused 0 in v2, starts at zero)."""
    if parent.layout == V1 and net.layout in (V2, CUBE):
        return init_from_v1(net, parent)
    assert parent.layout in (V2, CUBE) and net.layout == CUBE, f"no init from {parent.layout} to {net.layout}"
    taps, old = len(net.taps), len(parent.taps)
    if taps == old:
        net.load_state_dict(parent.state_dict())
    else:  # more tap levels: the parent's taps are a prefix, the new columns start at zero
        assert net.taps[:old] == parent.taps and taps > old, "tap layout differs"
        assert [l.weight.shape for l in net.layers[1:]] == [l.weight.shape for l in parent.layers[1:]]
        with torch.no_grad():
            src, dst = parent.layers[0].weight, net.layers[0].weight
            dst.zero_()
            dst[:, :12 * (1 + old)] = src[:, :12 * (1 + old)]
            dst[:, 12 * (1 + taps):] = src[:, 12 * (1 + old):]
            net.layers[0].bias.copy_(parent.layers[0].bias)
            for a, b in zip(net.layers[1:], parent.layers[1:]):
                a.weight.copy_(b.weight)
                a.bias.copy_(b.bias)
    if parent.layout == V2:
        with torch.no_grad():
            net.layers[0].weight[:, 4 * 3 * (1 + len(net.taps)) + 2] = 0


def init_from_v1(net, v1):
    """Start a reproject-v2 net from a v1 one (same taps and widths): layer 1's columns
    move to v2's input order (self and tap A, C), the new (motion, w) columns start at
    zero, and v1's (certainty, valid, 1, tau) fold into v2's (1, tau): certainty ~0.93
    and valid 1 on covered pixels, so it starts close to the v1 net."""
    assert net.layout in (V2, CUBE) and v1.layout == V1 and net.taps == v1.taps
    assert [l.weight.shape for l in net.layers[1:]] == [l.weight.shape for l in v1.layers[1:]]
    taps = len(v1.taps)
    with torch.no_grad():
        src, dst = v1.layers[0].weight[:, :, 0, 0], net.layers[0].weight[:, :, 0, 0]
        assert src.shape[0] == dst.shape[0], "same first layer width"
        dst.zero_()
        def move(a, b):  # vec4 a of v1 -> vec4 b of v2
            dst[:, 4 * b:4 * b + 4] = src[:, 4 * a:4 * a + 4]
        move(0, 0), move(1, 1)
        for t in range(taps):
            move(2 + 2 * t, 3 + 3 * t), move(3 + 2 * t, 4 + 3 * t)
        extra_v1, extra_v2 = 4 * (2 + 2 * taps), 4 * (3 + 3 * taps)
        dst[:, extra_v2] = src[:, extra_v1] * 0.93 + src[:, extra_v1 + 1] + src[:, extra_v1 + 2]
        dst[:, extra_v2 + 1] = src[:, extra_v1 + 3]
        net.layers[0].bias.copy_(v1.layers[0].bias)
        for a, b in zip(net.layers[1:], v1.layers[1:]):
            a.weight.copy_(b.weight)
            a.bias.copy_(b.bias)
