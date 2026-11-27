"""Train a cube-v1 project (net.layout = "cube-v1"; train.py hands over).

The runtime's state will be one head-centred sphere (an equi-angular cubemap) fixed to
the world: turning the head moves nothing, only translation (parallax) and objects move
content, by pull or splat reprojection (sunk/sphere.py, net.reproject). Training puts a patch of that sphere,
a canvas, into ordinary video:

  clips (sunk/clips.py)  footage at a random scale; the canvas is a rectangle of a
                         virtual EAC face, world-fixed, its texel angle the frame's
                         centre pixel at that scale (the net sees it as `scale`)
  per step               around the carried state (the canvas and a band, train.band)
                         a context ring of footage (of the step the state represents,
                         never the target); all of it moves by the head's translation and
                         the objects' velocity (splat), so the coarse taps see moved
                         surroundings with their motion, as on the sphere; the net runs on
                         the canvas and the band (what moves in from the far ring, hidden
                         0, gets the net's hidden state in the band before it arrives),
                         its taps in each texel's gravity frame (up: the video's y axis);
                         then the observation (train.observe) blends toward the footage,
                         after the net as at runtime
  view                   a canvas-sized view sweeps over the canvas and back
                         (train.turn): texels out of view keep their state, and when
                         they come back their pixel losses weigh train.recall more
  periods                per 8x8 tile an update period P (train.periods): every P ticks
                         the net runs with tau * P and the tile then moves toward that
                         target by 1/P, 1/(P-1), .. 1/1 (the runtime's slow periphery)

Losses as train.py's v2 (pixels near the stream's real start, statistics far, texture,
damage at the start state), on the canvas's inner crop where the view sees the footage.
Mono: one store, the eyes are a display matter.

Writes the same files as train.py: log.txt (zoo.py parses the validation lines),
preview/, net/ + nets/ (layout cube-v1), checkpoint.pt.
"""
import os, signal, time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sunk.clips import ClipPool, Clips, focal_of
from sunk.config import load
from sunk.model import CERT, INV, RGB, VEL, Net, init_from, load_export, scale_code
from sunk.sphere import directions, pull, splat_resolve, splat_sums, tap_grids
from train import Footage, damage, gradients, on_mains, pixel_weight, preview, style, tone

FLAGS = 2  # carried after the net's channels: seen (ever known), missed (out of view since);
# then the period target (the net's channels again), all moved by the pull


class Stream:
    """One batch sample's position in its clip and its carried state."""

    def __init__(self):
        self.clip, self.f, self.state, self.age, self.tick = None, 0, None, 0, 0
        self.period = self.phase = None


def tiles(rng, n, periods):
    """Per 8x8 tile a period from `periods` and a random phase, as [1, n, n] maps."""
    t = max(n // 8, 1)
    p = torch.tensor(rng.choice(periods, size=(t, t)), dtype=torch.float32)
    phase = torch.floor(torch.rand(t, t, generator=torch.Generator().manual_seed(int(rng.integers(1 << 31)))) * p)
    up = lambda x: F.interpolate(x[None, None], size=(n, n), mode="nearest")[0]
    return up(p), up(phase)


def truth(clip, chunk, frames, config, big_origin, big):
    """Footage on the big canvas for clip frames `frames` [K]: rgb [K, 3, M, M], inv
    [K, M, M] (radial 1/m), vel [K, 3, M, M] (canvas frame, m/s), inside [K, M, M]."""
    near, size0 = config.net.near, config.data.size
    dirs = directions(torch.tensor([clip.angle]), big_origin[None], big, big)[0]
    index = torch.tensor(frames)
    cams = clip.cams[index]
    c = torch.einsum("kij,hwj->khwi", cams, dirs)
    z = c[..., 2].clamp_min(1e-6)
    front = c[..., 2] > 1e-3

    def pixels(focal, size):
        return focal * c[..., 0] / z + size / 2 - 0.5, -focal * c[..., 1] / z + size / 2 - 0.5

    rows = clip.t0 + index
    px, py = pixels(clip.focal[index].view(-1, 1, 1), clip.scale)
    inside = (front & (px >= 0) & (px <= clip.scale - 1) & (py >= 0) & (py <= clip.scale - 1)).float()
    wh, ww = clip.window.shape[1:3]
    off = clip.offset[index].float()
    grid = torch.stack([(px - off[:, None, None, 0] + 0.5) / ww * 2 - 1, (py - off[:, None, None, 1] + 0.5) / wh * 2 - 1], -1)
    window = torch.tensor(clip.window[frames]).permute(0, 3, 1, 2).float() / 255
    rgb = F.grid_sample(window, grid, mode="bilinear", padding_mode="border", align_corners=False)

    qx, qy = pixels(torch.tensor([focal_of(size0, float(f)) for f in chunk["fov"][rows.numpy()]]).view(-1, 1, 1), size0)
    grid = torch.stack([(qx + 0.5) / size0 * 2 - 1, (qy + 0.5) / size0 * 2 - 1], -1)
    disp = torch.tensor(chunk["disp"][rows.numpy()].astype(np.float32))[:, None]
    inv = F.grid_sample(disp, grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0] * z / near
    last = len(chunk["vel"]) - 1
    vel_raw = torch.tensor(chunk["vel"][rows.clamp(max=last).numpy()].astype(np.float32)).permute(0, 3, 1, 2)
    vel = F.grid_sample(vel_raw, grid, mode="bilinear", padding_mode="border", align_corners=False)
    has_next = (index + 1 < clip.frames).float()
    after = clip.cams[(index + 1).clamp(max=clip.frames - 1)]
    vel = torch.einsum("kji,kjhw->kihw", after, vel) * near * chunk["fps"] * has_next[:, None, None, None]
    return rgb * inside[:, None], inv * inside, vel * inside[:, None], inside


def make_batch(footage, streams, config, steps, rng, strides, sweep=True):
    """Per stream (clip, frame f, stride): K = steps net steps. Returns a dict of
    [B, K+1, ...] truth (rgb, inv, vel, inside), landed [B, K+1], delta [B, K, 3] (head
    translation per step, canvas frame, m), dt [B], angle [B], origin [B, 2] (inner
    canvas), up [B, 3] (the video's up, its y axis, in canvas coords), view [B, K+1, nb, nb]
    (canvas + band: where the sweeping view is on the canvas, the band always)."""
    t = config.train
    n = t.crop + 2 * t.margin
    band, nb, _ = band_of(config)
    big = n + 2 * t.context
    out = {key: [] for key in ("rgb", "inv", "vel", "inside", "landed", "delta", "dt", "angle", "origin", "up", "view")}
    for s, stride in zip(streams, strides):
        clip = s.clip
        chunk = footage.chunks[clip.c]
        fps, near = chunk["fps"], config.net.near
        moves = torch.tensor(chunk["t"][clip.t0:clip.t0 + clip.frames - 1]) * near  # frame g -> g+1, camera g+1 coords
        shift = [-(clip.cams[g + 1].T @ moves[g]) for g in range(clip.frames - 1)]  # canvas frame
        delta, lands = [], []
        if stride >= 1:
            st = int(stride)
            for j in range(steps):
                delta.append(sum(shift[g] for g in range(s.f + j * st, s.f + (j + 1) * st)))
            lands = [s.f + j * st for j in range(steps + 1)]
        else:
            sub = stride.denominator
            for j in range(steps):
                delta.append(shift[s.f + j // sub] / sub)
            lands = [s.f + j // sub if j % sub == 0 else None for j in range(steps + 1)]
        frames = []
        for g in lands:
            frames.append(g if g is not None else frames[-1])
        rgb, inv, vel, inside = truth(clip, chunk, frames, config, clip.origin, big)
        landed = torch.tensor([g is not None for g in lands], dtype=torch.float32)
        out["rgb"].append(rgb)
        out["inv"].append(inv)
        out["vel"].append(vel)
        out["inside"].append(inside)
        out["landed"].append(landed)
        out["delta"].append(torch.stack(delta))
        out["dt"].append(torch.tensor(float(stride) / fps))
        out["angle"].append(torch.tensor(clip.angle, dtype=torch.float32))
        out["origin"].append(clip.origin + t.context)
        out["up"].append(clip.cams[0][1].float())  # canvas -> camera 0 maps it to camera 0's y
        # The view: canvas-sized, its centre sweeping out to `amplitude` and back.
        amplitude = t.turn * n * rng.random() if sweep else 0.0
        heading = rng.normal(size=2)
        heading = heading / np.linalg.norm(heading)
        j, i = torch.meshgrid(torch.arange(nb) + 0.5 - band, torch.arange(nb) + 0.5 - band, indexing="ij")
        canvas = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        views = []
        for k in range(steps + 1):
            cx, cy = n / 2 + amplitude * np.sin(np.pi * k / steps) * heading
            views.append((((i - cx).abs() <= n / 2) & ((j - cy).abs() <= n / 2) | ~canvas).float())
        out["view"].append(torch.stack(views))
    return {key: torch.stack(v) for key, v in out.items()}


def band_of(config):
    """(W, nb, e): texels of the ring the net evolves (train.band, at most the context),
    the carried state's side (canvas + band), its offset in the big canvas."""
    t = config.train
    n = t.crop + 2 * t.margin
    w = min(t.band, t.context)
    return w, n + 2 * w, t.context - w


def start_state(batch, channels, config):
    """The canvas + band at step 0 from the footage, plus the flags."""
    _, nb, e = band_of(config)
    b = len(batch["dt"])
    rect = slice(e, e + nb)
    state = torch.zeros(b, 2 * channels + FLAGS, nb, nb)
    known = batch["inside"][:, 0, rect, rect] * batch["view"][:, 0]
    state[:, RGB] = batch["rgb"][:, 0, :, rect, rect]
    state[:, INV] = batch["inv"][:, 0, rect, rect]
    state[:, VEL] = batch["vel"][:, 0, :, rect, rect]
    state[:, CERT] = 1
    state[:, :channels] *= known[:, None]  # outside the frame: holes
    state[:, channels] = known  # seen
    state[:, channels + FLAGS:] = state[:, :channels]  # period target
    return state


def ring(batch, k, state_band, weight, e, channels):
    """The big canvas: the footage of step k (hidden 0, weight train.context_weight where
    the frame has it) around the carried canvas + band (at offset e)."""
    b, _, m, _ = batch["rgb"][:, k].shape
    big = torch.zeros(b, channels, m, m)
    inside = batch["inside"][:, k]
    big[:, RGB] = batch["rgb"][:, k]
    big[:, INV] = batch["inv"][:, k]
    big[:, VEL] = batch["vel"][:, k]
    big[:, CERT] = inside * weight
    big = big.clone()
    nb = state_band.shape[-1]
    big[:, :, e:e + nb, e:e + nb] = state_band
    return big


def paste(whole, part, rect):
    """whole [..., M, M] with part in [..., rect, rect]."""
    out = whole.clone()
    out[..., rect, rect] = part
    return out


def rollout(net, config, batch, observe, start=None, ages=None, periods=None, ticks=None, record=False):
    """Loss over the batch's steps (see the module docstring). The carried state is the
    canvas and a band of the ring around it (train.band): the net evolves both, the
    losses look at the canvas only, so what moves into the canvas arrives with a hidden
    state the net made (the far ring is footage, hidden 0). start: carried states
    [C + FLAGS, nb, nb] per sample or None (fresh from the footage); ages [B] seconds since
    each stream's real start (training; None: validation, all steps alike); periods:
    (period, phase) maps [B, 1, nb, nb] or None (every tick); ticks [B]: each stream's tick
    count at step 0 (for the phases).
    Returns (loss, recorded frames, final state, recall error sum, recall count)."""
    t, near = config.train, config.net.near
    channels, ctx = net.channels, t.context
    band, nb, e = band_of(config)
    n = nb - 2 * band
    steps = batch["delta"].shape[1]
    b = len(batch["dt"])
    fresh = start_state(batch, channels, config)
    keep = channels + FLAGS if periods is None else 2 * channels + FLAGS  # the period target only when on
    state = torch.stack([(s if s is not None else fresh[i])[:keep] for i, s in enumerate(start)]) if start else fresh[:, :keep]
    if t.damage and ages is not None:
        body, _ = damage(state[:, :channels], t.damage, t.damage_area, t.damage_kinds)
        state = torch.cat([body, state[:, channels:]], 1)
    tau_step = batch["dt"] * net.rate
    scale = scale_code(batch["angle"])
    crop = torch.zeros(1, 1, n, n)
    crop[..., t.margin:n - t.margin, t.margin:n - t.margin] = 1
    m = batch["rgb"].shape[-1]
    rect = slice(e, e + nb)
    region = (e, e + nb, e, e + nb)  # the net's outputs: canvas + band, in the big canvas
    io = slice(band, band + n)  # the canvas within the carried state
    ci = slice(ctx, ctx + n)  # ... and within the big canvas
    origin = batch["origin"] - ctx  # the big canvas's
    grids = tap_grids(net.taps, batch["angle"], origin, batch["up"], m, m, region)
    loss, total_known, frames = 0.0, 0.0, []
    recall_sum, recall_count = 0.0, 0.0
    for k in range(steps):
        # Sources: the whole big canvas (the carried state, the footage of step k around
        # it, as the runtime's sphere has what it saw), all moved forward: the coarse taps
        # see moved surroundings with their motion, as on the sphere.
        source = ring(batch, k, state[:, :channels], t.context_weight, e, channels)
        parts = [source, F.pad(state[:, channels:channels + FLAGS], (e, e, e, e))]
        if periods is not None:
            parts.append(ring(batch, k, state[:, channels + FLAGS:], t.context_weight, e, channels))
        source = torch.cat(parts, 1)
        if net.reproject == "splat":
            # Only the carried region has gradients (the far ring is footage), so the sums
            # of the whole canvas come without them and the region's own points add theirs
            # (same values: sums - sums.detach()); a big splat with gradients would keep 4
            # tent copies of every channel per step.
            step = (batch["delta"][:, k], batch["dt"], batch["up"])
            with torch.no_grad():
                acc, cover = splat_sums(source, batch["angle"], origin, *step, nearness=net.nearness)
            part, _ = splat_sums(source[..., rect, rect], batch["angle"], origin + e, *step, nearness=net.nearness)
            acc = paste(acc, acc[..., rect, rect] + part - part.detach(), rect)
            moved, displacement, log_change = splat_resolve(acc, cover, source.shape[1])
        else:  # pull: only the region (its search is costly), the far ring stays as it is
            pulled, disp, logc = pull(source, batch["angle"], origin, batch["delta"][:, k], batch["dt"],
                                      levels=net.levels, inner=region)
            moved = paste(source, pulled, rect)
            displacement = F.pad(disp, (e, e, e, e))
            log_change = F.pad(logc, (e, e, e, e))
        weight = moved[:, CERT:CERT + 1, ci, ci]
        hole = (weight < 0.01).float()
        flags = (moved[:, channels:channels + FLAGS, rect, rect] > 0.5).float()
        body = moved[:, :channels, rect, rect]
        target = moved[:, channels + FLAGS:, rect, rect]
        per_tick = tau_step.view(-1, 1, 1, 1).clamp_min(1e-6)
        motion = torch.cat([displacement, log_change], 1) / per_tick
        # The net's activations are recomputed in the backward pass (checkpoint): on the
        # canvas + band they would be most of the memory, 8 steps of them.
        run = lambda tau: checkpoint(net, moved[:, :channels], tau, motion, scale, region, grids, use_reentrant=False)
        if periods is None:
            pred = run(tau_step)
            new = pred
        else:
            period, phase = periods
            beat = (ticks.view(-1, 1, 1, 1) + k - phase) % period
            due = (beat == 0).float()
            remaining = period - beat
            pred = run(tau_step.view(-1, 1, 1, 1) * period)
            target = pred * due + target * (1 - due)
            new = body + (target - body) / remaining
            new[:, CERT:CERT + 1] = 1
        # Truth of step k + 1, where the view sees the footage; the losses on the canvas.
        landed = batch["landed"][:, k + 1].view(-1, 1, 1, 1)
        seen_now = batch["inside"][:, k + 1, None, rect, rect] * batch["view"][:, k + 1, None] * landed
        known = seen_now[..., io, io]
        recall = known * flags[:, 1:2, io, io]
        score = known * crop
        truth_rgb = batch["rgb"][:, k + 1, :, rect, rect]
        truth_inv = batch["inv"][:, k + 1, None, rect, rect]
        image = torch.cat([new[:, RGB, io, io], new[:, INV:INV + 1, io, io] * near], 1)
        goal = torch.cat([truth_rgb[..., io, io], truth_inv[..., io, io] * near], 1)
        weights = (1 + t.hole_weight * hole + (t.recall - 1) * recall) * score
        if ages is None:
            pixels = torch.ones(b)
        else:
            pixels = pixel_weight(ages + (k + 1) * batch["dt"], t.pixel_seconds)
        error = (image - goal).abs()
        pixel_loss = (error * weights).mean((1, 2, 3))
        pairs = (score[..., 1:, :] * score[..., :-1, :], score[..., :, 1:] * score[..., :, :-1])
        for gp, gt, both in zip(gradients(image), gradients(goal), pairs):
            pixel_loss = pixel_loss + t.edge_weight * ((gp - gt).abs() * both).mean((1, 2, 3))
        step_loss = (pixel_loss * pixels).mean()
        if t.texture:
            def energy(x):
                gy, gx = gradients(x)
                return F.avg_pool2d(gy[..., :, :-1].abs() + gx[..., :-1, :].abs(), 8, stride=4)
            window = F.avg_pool2d(score[..., :-1, :-1], 8, stride=4) > 0.99
            step_loss = step_loss + t.texture * ((energy(image) - energy(goal)).abs() * window).mean()
        style_weight = t.style + t.style_far * (1 - pixels)
        if style_weight.any():
            gap = (style(image[:, RGB], score) - style(goal[:, RGB], score)).abs().mean((1, 2, 3))
            step_loss = step_loss + (style_weight * gap).mean()
        if t.tone:
            step_loss = step_loss + t.tone * (tone(image[:, RGB], score) - tone(goal[:, RGB], score)).abs().mean()
        if t.range:  # every texel, seen or not: the state's colour stays displayable
            step_loss = step_loss + t.range * (F.relu(net.raw - 1) + F.relu(-net.raw)).mean()
        loss = loss + step_loss
        total_known = total_known + landed.mean()
        recall_sum = recall_sum + (error.mean(1, keepdim=True) * recall * crop).sum()
        recall_count = recall_count + (recall * crop).sum()
        if record:
            shown = goal[:, RGB] * (0.35 + 0.65 * known)
            frames.append((body[..., io, io], hole, new[..., io, io], torch.cat([shown, truth_inv[..., io, io], torch.zeros_like(new[:, 4:, io, io])], 1)))
        # Flags: seen where known; missed where seen but not known now, cleared on return.
        seen = torch.maximum(flags[:, :1], seen_now)
        missed = ((flags[:, 1:2] + flags[:, :1] * (1 - seen_now)) > 0).float() * (1 - seen_now)
        # Observe (after the net, as the runtime's net.frag): blend toward the footage where
        # the view sees it (not late in a stream).
        alpha = observe * torch.rand(b, 1, 1, 1) * seen_now
        if ages is not None and t.observe_seconds > 0:
            alpha = alpha * ((ages + (k + 1) * batch["dt"]) <= t.observe_seconds).float().view(-1, 1, 1, 1)
        nxt = new.clone()
        nxt[:, RGB] = new[:, RGB] * (1 - alpha) + truth_rgb * alpha
        nxt[:, INV:INV + 1] = new[:, INV:INV + 1] * (1 - alpha) + truth_inv * alpha
        nxt[:, VEL] = new[:, VEL] * (1 - alpha) + batch["vel"][:, k + 1, :, rect, rect] * alpha
        state = torch.cat([nxt, seen, missed] + ([target] if periods is not None else []), 1)
    return loss / max(float(total_known), 1.0), frames, state, recall_sum, recall_count


class Streams:
    """train.batch streams, each following a clip for up to train.horizon frames."""

    def __init__(self, footage, pool, config, n):
        self.footage, self.pool, self.config = footage, pool, config
        self.streams = [Stream() for _ in range(n)]

    def batch(self, rng):
        cfg = self.config
        _, nb, _ = band_of(cfg)
        strides = []
        for s in self.streams:
            stride = self.footage.stride(rng)
            frames = int(cfg.train.steps * stride)
            if s.clip is None or s.f + frames >= s.clip.frames or s.age + frames > cfg.train.horizon:
                s.clip, s.f, s.state, s.age, s.tick = self.pool.take(rng), 0, None, 0, 0
                s.period, s.phase = tiles(rng, nb, cfg.train.periods)
                while frames >= s.clip.frames:  # a short clip: the smallest stride that fits
                    stride = min(self.footage.strides)
                    frames = int(cfg.train.steps * stride)
                    if frames >= s.clip.frames:
                        s.clip = self.pool.take(rng)
            strides.append(stride)
        batch = make_batch(self.footage, self.streams, cfg, cfg.train.steps, rng, strides)
        ages = torch.tensor([s.age / self.footage.chunks[s.clip.c]["fps"] for s in self.streams], dtype=torch.float32)
        periods = (torch.stack([s.period for s in self.streams]), torch.stack([s.phase for s in self.streams]))
        ticks = torch.tensor([float(s.tick) for s in self.streams])
        return batch, [s.state for s in self.streams], ages, periods, ticks, strides

    def advance(self, final, strides):
        for i, s in enumerate(self.streams):
            frames = int(self.config.train.steps * strides[i])
            s.f += frames
            s.age += frames
            s.tick += self.config.train.steps
            s.state = final[i].detach()


def validation_set(footage, clips, config, steps, seed, count=8):
    """Fixed streams at three scales (data.size, between, native), stride 1."""
    rng = np.random.default_rng(seed)
    lo, hi = config.train.scales[0], config.train.scales[-1]
    scales = [lo, int(round((lo * hi) ** 0.5)), hi]
    streams = []
    for i in range(count):
        s = Stream()
        s.clip = clips.make(rng, steps + 1, scale=scales[i % 3])
        if s.clip.frames < steps + 1:
            for _ in range(20):
                s.clip = clips.make(rng, steps + 1, scale=scales[i % 3])
                if s.clip.frames >= steps + 1:
                    break
        streams.append(s)
    usable = [s for s in streams if s.clip.frames >= steps + 1]
    one = Fraction(1)
    return make_batch(footage, usable, config, steps, np.random.default_rng(seed), [one] * len(usable))


def validate(net, config, batch):
    """(val, push-pull only, recall, recall push-pull only) of a free-running rollout."""
    with torch.no_grad():
        val, frames, _, rs, rc = rollout(net, config, batch, 0.0, record=True)
        change, net.max_change = net.max_change, 0.0
        base, _, _, bs, bc = rollout(net, config, batch, 0.0)
        net.max_change = change
    return val.item(), base.item(), (rs / max(float(rc), 1)).item() if rc else 0.0, \
        (bs / max(float(bc), 1)).item() if bc else 0.0, frames


def evaluate(project, samples=24, steps=None, chunks=None):
    """evaluate.py for cube-v1: free-running rollouts on fixed clips at three scales;
    the net vs pull + fill only: mean error, recall error, and at the last step the
    error, sharpness and contrast (relative to the footage's, 1 = like it)."""
    config = load(project)
    if chunks:
        config.train.chunks = chunks
    torch.set_num_threads(2)
    footage = Footage(config)
    clips = Clips(footage, config)
    net = load_export(Path(project) / "net")
    steps = steps or config.train.steps
    batch = validation_set(footage, clips, config, steps, 7, count=samples)
    ctx, n = config.train.context, config.train.crop + 2 * config.train.margin
    band = band_of(config)[0]
    io = slice(band, band + n)
    truth = batch["rgb"][:, -1, :, ctx:ctx + n, ctx:ctx + n]
    known = batch["inside"][:, -1, None, ctx:ctx + n, ctx:ctx + n] * batch["view"][:, -1, None, io, io]

    def grad(x):
        return ((x[..., 1:, :-1] - x[..., :-1, :-1]).abs() + (x[..., :-1, 1:] - x[..., :-1, :-1]).abs()).mean((1, 2, 3))

    def clipped(x):  # share of texels with a channel at black or white
        return ((x <= 0.01) | (x >= 0.99)).any(1).float().mean().item()

    rows = {}
    for name, change in (("net", net.max_change), ("pull", 0.0)):
        net.max_change = change
        with torch.no_grad():
            loss, _, final, rs, rc = rollout(net, config, batch, 0.0)
        final = final[..., io, io]
        image = final[:, RGB].clamp(0, 1)
        last = ((image - truth).abs().mean(1, keepdim=True) * known).sum() / known.sum().clamp_min(1)
        sharp = (grad(image) / grad(truth).clamp_min(1e-4)).mean().item()
        contrast = (image.mean(1).flatten(1).std(1) / truth.mean(1).flatten(1).std(1).clamp_min(1e-4)).mean().item()
        rows[name] = (loss.item(), (rs / max(float(rc), 1)).item(), last.item(), sharp, contrast, clipped(final[:, RGB]))
    print(f"{project}: {len(batch['dt'])} free-running rollouts of {steps} steps (scales {config.train.scales[0]}..{config.train.scales[-1]})")
    print(f"            loss     recall   last L1   sharpness  contrast  clipped (footage {clipped(truth):.1%})")
    for name, r in rows.items():
        print(f"  {name:<8} {r[0]:.4f}   {r[1]:.4f}   {r[2]:.4f}    {r[3]:.2f}       {r[4]:.2f}      {r[5]:.1%}")
    gain = [1 - a / b if b else 0 for a, b in zip(rows["net"][:3], rows["pull"][:3])]
    print(f"net vs pull only: {gain[0]:+.1%} loss, {gain[1]:+.1%} recall, {gain[2]:+.1%} last step (positive = better)")


def main(project):
    project = Path(project)
    config = load(project)
    assert config.net.layout == "cube-v1" and not config.train.stereo, "cube-v1 is mono (train.stereo = 0)"
    if config.train.threads:
        torch.set_num_threads(config.train.threads)
    os.nice(19)
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (stopping.append(1), print("stopping after this iteration", flush=True)))
    footage = Footage(config)
    clips = Clips(footage, config)
    max_frames = config.train.horizon + int(config.train.steps * max(footage.strides)) + 1
    pool = ClipPool(clips, config, max_frames, seed=int(time.time()))
    net = Net(config)
    opt = torch.optim.Adam(net.parameters(), lr=config.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.train.iters)
    start = 0
    checkpoint = project / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint)
        net.load_state_dict(saved["net"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        start = saved["iter"]
        print(f"resuming at {start}")
    elif config.train.init:
        source = (project / config.train.init).resolve()
        parent = load_export(source)
        if parent.layout != net.layout or parent.taps != net.taps:
            init_from(net, parent)
        else:
            net.load_state_dict(parent.state_dict())
        print(f"starting from {source}")
    print(f"net: {net.inputs} inputs, layers {[l.out_channels for l in net.layers]}, "
          f"{sum(p.numel() for p in net.parameters())} weights, {len(net.taps) + 1} taps")
    max_tau = max(float(s) for s in footage.strides) / min(c["fps"] for c in footage.chunks) * net.rate * max(config.train.periods)
    rng = np.random.default_rng(start)
    fixed = validation_set(footage, clips, config, config.train.steps, 1234)
    long = config.train.horizon
    fixed_long = validation_set(footage, clips, config, long, 4321) if long > config.train.steps else None
    streams = Streams(footage, pool, config, config.train.batch)
    (project / "preview").mkdir(exist_ok=True)
    log = open(project / "log.txt", "a")
    began, total = time.time(), 0.0

    def save(it):
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it}, checkpoint)

    periods_on = config.train.periods != [1]
    duty = config.train.duty
    for it in range(start + 1, config.train.iters + 1):
        work = time.time()
        batch, initial, ages, periods, ticks, strides = streams.batch(rng)
        loss, _, final, _, _ = rollout(net, config, batch, config.train.observe, start=initial, ages=ages,
                                       periods=periods if periods_on else None, ticks=ticks)
        streams.advance(final, strides)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.item()
        if it % 25 == 0:
            line = f"{it} loss {total / 25:.4f} {(time.time() - began) / 25:.2f}s/it"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            began, total = time.time(), 0.0
        if it % config.train.preview == 0 or it == config.train.iters:
            val, base, rec, rec_base, frames = validate(net, config, fixed)
            line = f"{it} free-running val {val:.4f} (push-pull only {base:.4f})"
            if fixed_long is not None:
                lval, lbase, lrec, lrec_base, _ = validate(net, config, fixed_long)
                line += f", over {long} frames {lval:.4f} (push-pull only {lbase:.4f})"
                rec, rec_base = lrec, lrec_base
            line += f", recall {rec:.4f} (push-pull only {rec_base:.4f})"
            print(line, flush=True)
            log.write(line + "\n")
            preview(project / "preview" / f"{it:06d}.png", frames, config.net.near)
            net.export(project / "net", config, max_tau=max_tau)
            net.export(project / "nets" / f"{it:06d}", config, max_tau=max_tau)
            save(it)
        elif it % config.train.checkpoint == 0:
            save(it)
            try:
                duty = load(project).train.duty
            except Exception as e:
                print(f"config.toml not reloaded: {e}", flush=True)
        if stopping:
            save(it)
            pool.stop = True
            print(f"saved at {it}; run again to resume", flush=True)
            return
        if config.train.on_battery != "run" and not on_mains():
            save(it)
            if config.train.on_battery == "stop":
                print(f"on battery: saved at {it}, stopping", flush=True)
                return
            print(f"on battery: saved at {it}, waiting for AC", flush=True)
            while not on_mains() and not stopping:
                time.sleep(30)
            if stopping:
                return
            print("on AC again, continuing", flush=True)
            began = time.time()
        elif duty < 1:
            time.sleep((time.time() - work) * (1 / max(duty, 0.05) - 1))
    pool.stop = True
