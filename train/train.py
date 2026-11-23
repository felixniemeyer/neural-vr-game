"""Train a project: uv run train.py runs/<name>

Rollouts on real footage: the state starts as frame t (colour, depth, object velocity),
then for each of train.steps net steps it is splatted with the estimated camera motion
(+ object velocity) into the next camera, resolved, and the net runs. Where a step
lands on a real frame, the result is compared with it (colour, depth, edges; holes
weigh more), and the state is blended toward it by a random 0..train.observe, like the
rendered blocks at runtime.

Time: a net step covers `stride` video frames (train.strides, drawn per sample and
iteration): below 1 the frame's camera motion is split into exact substeps (truth only
at whole frames), above 1 frames are skipped. The net sees the step as
tau = stride / fps * net.rate ticks, so it learns to run at any step size.

Streams (train.horizon): every sample follows its clip, continuing from the net's own
(detached) state of the previous iteration, so it learns to keep its own old states
good. A stream restarts from a real frame at a cut, the clip's end, or the horizon.

Stereo (train.stereo, train.baseline): each sample is two eyes, the second one shifted
by a random baseline to the right. Its truth is the real frame splatted over (unknown
where uncovered). As in the app, each eye's resolve also takes half of the other eye's
state (checkerboard). The consistency loss warps each eye's result into the other
and penalises differences.

Writes into the project: log.txt, preview/<iter>.png (per sample: resolved with holes
magenta | prediction | truth | predicted depth | true depth, at 3 steps of a
free-running rollout), net/ (latest export for the app), checkpoint.pt (resumes).
"""
import sys, time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from prepare import rays as make_rays
from sunk.config import load
from sunk.model import CERT, HID, INV, RGB, VEL, Net, accumulate, load_export, resolve, splat


def rotation(axis, angle):
    """Rodrigues: rotation by `angle` (rad) about the unit vector `axis`."""
    x, y, z = axis.tolist()
    k = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float32)
    return torch.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def root(r, t, n):
    """The motion (S, u) that, applied n times, is (R, t): S = R^(1/n) about R's axis,
    u from t = (S^(n-1) + .. + S + I) u. Exact, so substeps compose to the frame."""
    angle = float(np.arccos(np.clip((torch.trace(r).item() - 1) / 2, -1, 1)))
    if angle < 1e-7:
        return torch.eye(3), t / n
    axis = torch.stack([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(angle))
    s = rotation(axis, angle / n)
    total, power = torch.zeros(3, 3), torch.eye(3)
    for _ in range(n):
        total, power = total + power, power @ s
    return s, torch.linalg.solve(total, t)


class Footage:
    def __init__(self, config):
        self.config = config
        self.paths, self.chunks = [], []
        self.starts = []  # (chunk, t0) with a few usable pairs ahead
        self.ahead = []   # per chunk: usable pairs from each frame on (until a cut)
        size = config.data.size
        self.focal = size / 2 / np.tan(np.radians(config.data.fov) / 2)
        self.rays = make_rays(size, size, config.data.fov)
        self.strides = [Fraction(s).limit_denominator(16) for s in config.train.strides]
        self.refresh()

    def refresh(self):
        """Load chunks that appeared since (prepare.py may still be running)."""
        for path in sorted(self.config.prep_dir().glob("chunk_[0-9][0-9][0-9].npz")):
            if path in self.paths:
                continue
            z = np.load(path)
            chunk = {k: z[k] for k in ("rgb", "disp", "vel", "R", "t", "ok")}
            chunk["fps"] = float(z["fps"]) if "fps" in z else self.config.data.fps or 10.0
            ahead = np.zeros(len(chunk["ok"]) + 1, int)
            for t in range(len(chunk["ok"]) - 1, -1, -1):
                ahead[t] = (ahead[t + 1] + 1) * bool(chunk["ok"][t])
            self.starts += [(len(self.chunks), t) for t in np.nonzero(ahead >= 2)[0]]
            self.ahead.append(ahead)
            self.paths.append(path)
            self.chunks.append(chunk)
            print(f"{len(self.chunks)} chunks, {len(self.starts)} rollout starts", flush=True)

    def frames(self, stride, steps=None):
        """Video frames a rollout of `steps` net steps covers at `stride`."""
        covered = (steps or self.config.train.steps) * stride
        assert covered.denominator == 1, f"train.steps * {stride} must be whole frames"
        return int(covered)

    def usable(self, c, t0, frames):
        return t0 < len(self.ahead[c]) and self.ahead[c][t0] >= frames

    def stride(self, rng):
        return self.strides[rng.integers(len(self.strides))]

    def pick(self, rng, steps=None, stride=None):
        """A random (chunk, t0, crop x, crop y, stride, baseline) with the frames ahead."""
        cfg = self.config.train
        stride = stride or self.stride(rng)
        frames = self.frames(stride, steps)
        while True:
            c, t0 = self.starts[rng.integers(len(self.starts))]
            if self.usable(c, t0, frames):
                ox, oy = rng.integers(self.config.data.size - cfg.crop - 2 * cfg.margin + 1, size=2)
                baseline = cfg.baseline * rng.random() if cfg.stereo else 0.0
                return (c, t0, ox, oy, stride, baseline)

    def batch(self, n, rng, steps=None, picks=None):
        """Per sample (picks: see pick(), default random), over K = steps net steps:
        truth frames as seen by the (virtual) cameras [B, K+1, ...] (where a step lands
        on a real frame, else masked), the camera motion per step [B, K], crop rays
        [B, crop, crop, 3], origins, where the truth is known [B, K+1], step seconds dt
        [B] and baselines [B].

        Head-turn augmentation (train.turn): the virtual camera at step k is the real one
        turned by D_k = rotation(theta * sin(pi k / K)) about a random axis in the image
        plane, theta up to a shift of turn * crop px. It turns away and back, so the crop
        loses content at one border and has to regenerate it. The truth is the real frame
        seen through the turned camera (rotation only: no depth needed, radial distance
        unchanged), unknown outside the real frame (mask).
        """
        cfg = self.config
        k, size, near = steps or cfg.train.steps, cfg.data.size, cfg.net.near
        margin = cfg.train.margin
        crop = cfg.train.crop + 2 * margin  # the simulated state; losses see the inner crop
        picks = picks or [self.pick(rng, k) for _ in range(n)]
        out = {key: [] for key in ("rgb", "inv", "vel", "R", "t", "rays", "origin", "mask", "dt", "baseline", "inner")}
        inner = torch.zeros(1, crop, crop)
        inner[:, margin:crop - margin, margin:crop - margin] = 1
        radial = self.rays.norm(dim=-1)
        for c, t0, ox, oy, stride, baseline in picks:
            chunk, fps = self.chunks[c], self.chunks[c]["fps"]
            frames = self.frames(stride, k)
            rgb = torch.tensor(chunk["rgb"][t0:t0 + frames + 1]).float() / 255
            # disp is planar (1 / z in near units); the state holds radial 1 / distance.
            inv = torch.tensor(chunk["disp"][t0:t0 + frames + 1].astype(np.float32)) / near / radial
            r = torch.tensor(chunk["R"][t0:t0 + frames])
            move = torch.tensor(chunk["t"][t0:t0 + frames]) * near
            # Object velocity of pair t -> t+1 is in camera t+1 coords (near units per
            # frame); the state carries it in camera t coords in m/s: R^T v * near * fps.
            vel = torch.einsum("kji,khwj->khwi", r, torch.tensor(chunk["vel"][t0:t0 + frames].astype(np.float32)))
            vel = torch.cat([vel * near * fps, torch.zeros_like(vel[:1])])

            # Camera motion per net step, and the frame each step lands on (or None).
            if stride >= 1:
                s = int(stride)
                steps_r, steps_t = [], []
                for j in range(k):
                    rj, tj = torch.eye(3), torch.zeros(3)
                    for f in range(j * s, j * s + s):
                        rj, tj = r[f] @ rj, r[f] @ tj + move[f]
                    steps_r.append(rj)
                    steps_t.append(tj)
                lands = [j * s for j in range(k + 1)]
            else:
                n_sub = stride.denominator
                roots = [root(r[f], move[f], n_sub) for f in range(frames)]
                steps_r = [roots[j // n_sub][0] for j in range(k)]
                steps_t = [roots[j // n_sub][1] for j in range(k)]
                lands = [j // n_sub if j % n_sub == 0 else None for j in range(k + 1)]

            angle = np.arctan(cfg.train.turn * cfg.train.crop * rng.random() / self.focal)
            axis = rng.normal(size=2)
            axis = torch.tensor([*axis / np.linalg.norm(axis), 0.0], dtype=torch.float32)
            turns = torch.stack([rotation(axis, angle * np.sin(np.pi * j / k)) for j in range(k + 1)])

            # Sample the real frames through the virtual cameras (zero where no frame).
            rays = self.rays[oy:oy + crop, ox:ox + crop]
            real = torch.einsum("kji,hwj->khwi", turns, rays)  # D^T d
            z = real[..., 2].clamp_min(1e-6)
            px = self.focal * real[..., 0] / z + size / 2 - 0.5
            py = -self.focal * real[..., 1] / z + size / 2 - 0.5
            mask = ((real[..., 2] > 0) & (px >= 0) & (px <= size - 1) & (py >= 0) & (py <= size - 1)).float()
            landed = torch.tensor([f is not None for f in lands], dtype=torch.float32)
            mask = mask * landed[:, None, None]
            index = torch.tensor([f if f is not None else 0 for f in lands])
            grid = torch.stack([(px + 0.5) / size * 2 - 1, (py + 0.5) / size * 2 - 1], -1)
            frames_t = torch.cat([rgb, inv[..., None], vel], -1)[index].permute(0, 3, 1, 2)
            seen = F.grid_sample(frames_t, grid, mode="bilinear", padding_mode="border", align_corners=False)
            seen = seen.permute(0, 2, 3, 1) * mask[..., None]
            out["rgb"].append(seen[..., :3])
            out["inv"].append(seen[..., 3])
            out["vel"].append(torch.einsum("kij,khwj->khwi", turns, seen[..., 4:]))
            # Virtual camera motion: p_virtual' = D' (R (D^T p) + t).
            sr, st = torch.stack(steps_r), torch.stack(steps_t)
            out["R"].append(turns[1:] @ sr @ turns[:-1].transpose(1, 2))
            out["t"].append(torch.einsum("kij,kj->ki", turns[1:], st))
            out["rays"].append(rays)
            out["origin"].append(torch.tensor([ox, oy], dtype=torch.float32))
            out["mask"].append(mask)
            out["dt"].append(torch.tensor(float(stride) / fps))
            out["baseline"].append(torch.tensor(float(baseline)))
            out["inner"].append(inner)
        return {key: torch.stack(v) for key, v in out.items()}


def truth_state(batch, step, channels):
    b, _, h, w = batch["rgb"].shape[:4]
    state = torch.zeros(b, channels, h, w)
    state[:, RGB] = batch["rgb"][:, step].permute(0, 3, 1, 2)
    state[:, INV] = batch["inv"][:, step]
    state[:, VEL] = batch["vel"][:, step].permute(0, 3, 1, 2)
    state[:, CERT] = 1
    return state


def gradients(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def shift(baseline):
    """[B, 3]: the second eye's offset (x right) times the baseline."""
    return torch.stack([baseline, torch.zeros_like(baseline), torch.zeros_like(baseline)], -1)


class Eyes:
    """Both eyes of a stereo batch stacked along the batch dim (left B, then right B),
    or the left eye only. The right camera is the left one shifted by b along x:
    p_right = p_left - b x."""

    def __init__(self, batch, focal, size, stereo):
        self.batch, self.focal, self.size, self.stereo = batch, focal, size, stereo
        self.b = shift(batch["baseline"])
        self.eye = torch.eye(3).expand(len(self.b), 3, 3)

    def both(self, x):
        return torch.cat([x, x]) if self.stereo else x

    def to_right(self, state):
        """Splat a left-eye state into the right eye (same moment)."""
        return splat(state, self.eye, -self.b, self.batch["rays"], self.batch["origin"], self.focal, self.size)

    def to_left(self, state):
        return splat(state, self.eye, self.b, self.batch["rays"], self.batch["origin"], self.focal, self.size)

    def truth(self, step, channels):
        """Truth state and where it is known, for each eye."""
        left = truth_state(self.batch, step, channels)
        known = self.batch["mask"][:, step, None]
        if not self.stereo:
            return left, known
        right, hole = self.to_right(left * known)
        return torch.cat([left, right]), torch.cat([known, (1 - hole) * known.amax((2, 3), keepdim=True)])

    def step(self, state, k):
        """Splat (both eyes) for net step k; each eye also takes half of the other's state
        (checkerboard, alternating per step), as splat.vert does."""
        r, t = self.batch["R"][:, k], self.batch["t"][:, k]
        args = (self.batch["rays"], self.batch["origin"], self.focal, self.size, self.batch["dt"])
        if not self.stereo:
            return resolve(accumulate(state, r, t, *args))
        n = len(r)
        rb = torch.einsum("bij,bj->bi", r, self.b)
        left, right = state[:n], state[n:]
        h, w = state.shape[2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        keep = (((x + y + k) % 2) == 0).float()[None, None].expand(n, 1, h, w)
        acc_left = accumulate(left, r, t, *args) + accumulate(right, r, t + rb, *args, keep=keep)
        acc_right = accumulate(right, r, t + rb - self.b, *args) + accumulate(left, r, t - self.b, *args, keep=keep)
        return resolve(torch.cat([acc_left, acc_right]))

    def disagreement(self, pred, near):
        """Each eye's result warped into the other, L1 of colour and depth where covered."""
        n = len(self.b)
        loss = 0.0
        for warped, other in (self.to_right(pred[:n]), pred[n:]), (self.to_left(pred[n:]), pred[:n]):
            state, hole = warped
            seen = 1 - hole
            diff = torch.cat([(state[:, RGB] - other[:, RGB]).abs(),
                              (state[:, INV:INV + 1] - other[:, INV:INV + 1]).abs() * near], 1)
            loss = loss + (diff * seen).mean()
        return loss / 2


def corridor(pred, truth, known, slack=0.05, contrast=0.8):
    """Global stats in a corridor (train.corridor): per sample, the known pixels' mean
    luminance may differ from the truth's by `slack`, and their spread (std) may not fall
    below `contrast` times the truth's. Zero inside, linear outside: no pull toward the
    exact frame, only against drifting dark/bright or fading to fog."""
    def stats(x):
        y = x.mean(1, keepdim=True)
        weight = known / known.sum((1, 2, 3), keepdim=True).clamp_min(1)
        mean = (y * weight).sum((1, 2, 3))
        std = (((y - mean.view(-1, 1, 1, 1)) ** 2 * weight).sum((1, 2, 3)) + 1e-6).sqrt()
        return mean, std
    (mp, sp), (mt, st) = stats(pred), stats(truth)
    return (F.relu((mp - mt).abs() - slack) + F.relu(contrast * st - sp)).mean()


def knock_out(resolved, hole, most):
    """Dropout (train.dropout): blobs of about 8 px (smooth noise, thresholded) covering
    up to `most` of each sample become holes (all channels 0, as the splat leaves them),
    so the net learns to regenerate them; they weigh as holes in the loss."""
    b, _, h, w = resolved.shape
    noise = F.interpolate(torch.rand(b, 1, max(h // 8, 1), max(w // 8, 1)), size=(h, w), mode="bilinear", align_corners=False)
    out = (noise < most * torch.rand(b, 1, 1, 1)).float()
    return resolved * (1 - out), torch.maximum(hole, out)


def rollout(net, data, batch, observe, record=False, start=None, dropout=0.0):
    """Loss over the batch's steps; start: initial states (per sample [eyes * C, H, W]
    as [C, H, W] per eye stacked, None where the rollout starts from the real frame).
    Returns (loss, recorded frames (left eye), final state [eyes * B, C, H, W])."""
    cfg = data.config
    near = cfg.net.near
    steps = batch["R"].shape[1]
    eyes = Eyes(batch, data.focal, cfg.data.size, cfg.train.stereo > 0)
    n = len(batch["dt"])
    state, _ = eyes.truth(0, net.channels)
    if start is not None:
        fresh = state.view(-1, n, *state.shape[1:])
        state = torch.stack([s if s is not None else fresh[:, i] for i, s in enumerate(start)], 1)
        state = state.flatten(0, 1)
    tau = eyes.both(batch["dt"] * net.rate)
    loss, total_known, frames = 0.0, 0.0, []
    for k in range(steps):
        resolved, hole = eyes.step(state, k)
        if dropout:  # training only (train.dropout); validation sees the real holes
            resolved, hole = knock_out(resolved, hole, dropout)
        pred = net(resolved, tau)
        truth, observed = eyes.truth(k + 1, net.channels)
        known = observed * eyes.both(batch["inner"])  # losses: the crop without the margin
        weight = (1 + cfg.train.hole_weight * hole) * known
        image = torch.cat([pred[:, RGB], pred[:, INV:INV + 1] * near], 1)
        target = torch.cat([truth[:, RGB], truth[:, INV:INV + 1] * near], 1)
        step_loss = ((image - target).abs() * weight).mean()
        pairs = (known[..., 1:, :] * known[..., :-1, :], known[..., :, 1:] * known[..., :, :-1])
        for gp, gt, both in zip(gradients(image), gradients(target), pairs):
            step_loss = step_loss + cfg.train.edge_weight * ((gp - gt).abs() * both).mean()
        if cfg.train.texture:
            # Local edge energy (mean gradient magnitude per 8x8 window), compared without
            # requiring the edges to line up: rewards plausible structure over blur.
            def energy(x):
                gy, gx = gradients(x)
                g = gy[..., :, :-1].abs() + gx[..., :-1, :].abs()
                return F.avg_pool2d(g, 8, stride=4)
            window = F.avg_pool2d(known[..., :-1, :-1], 8, stride=4) > 0.99
            step_loss = step_loss + cfg.train.texture * ((energy(image) - energy(target)).abs() * window).mean()
        if cfg.train.corridor:
            step_loss = step_loss + cfg.train.corridor * corridor(pred[:, RGB], truth[:, RGB], known)
        if cfg.train.semigroup:
            # Step-size consistency (ac841219: a 10 Hz periphery steps 100 ms at once): one
            # step of tau should match two of tau / 2 on the same view, no truth needed.
            halves = net(net(resolved, tau / 2), tau / 2)
            both = [torch.cat([p[:, RGB], p[:, INV:INV + 1] * near], 1) for p in (pred, halves)]
            step_loss = step_loss + cfg.train.semigroup * (both[0] - both[1]).abs().mean()
        if eyes.stereo:
            step_loss = step_loss + cfg.train.stereo * eyes.disagreement(pred, near)
        loss = loss + step_loss
        total_known = total_known + batch["mask"][:, k + 1].amax((1, 2)).mean()  # steps on a real frame
        if record:
            frames.append((resolved[:n], hole[:n], pred[:n], truth[:n]))
        alpha = observe * torch.rand(len(pred), 1, 1, 1) * observed
        state = pred.clone()
        for channels in (RGB, slice(INV, INV + 1), VEL):
            state[:, channels] = pred[:, channels] * (1 - alpha) + truth[:, channels] * alpha
        state[:, CERT:CERT + 1] = pred[:, CERT:CERT + 1] * (1 - alpha) + alpha
    return loss / total_known.clamp_min(1), frames, state


class Streams:
    """Long rollouts with short backprop (truncated BPTT): each batch sample follows its
    clip for up to train.horizon net steps' worth of video, continuing every iteration
    from the net's own (detached) state, same crop and baseline, a new stride each time.
    It restarts from a real frame at a cut, the end of its clip, or the horizon."""

    def __init__(self, data, n):
        self.data, self.n = data, n
        self.picks, self.states, self.ages = [None] * n, [None] * n, [0] * n

    def batch(self, rng):
        data, horizon = self.data, self.data.config.train.horizon
        for i in range(self.n):
            p, stride = self.picks[i], data.stride(rng)
            frames = data.frames(stride)
            if p is None or self.ages[i] + frames > horizon or not data.usable(p[0], p[1], frames):
                self.picks[i], self.states[i], self.ages[i] = data.pick(rng, stride=stride), None, 0
            else:
                self.picks[i] = (*p[:4], stride, p[5])
        return data.batch(self.n, rng, picks=self.picks), self.states

    def advance(self, final):
        eyes = final.view(-1, self.n, *final.shape[1:])
        for i in range(self.n):
            c, t0, ox, oy, stride, baseline = self.picks[i]
            frames = self.data.frames(stride)
            self.picks[i] = (c, t0 + frames, ox, oy, stride, baseline)
            self.states[i] = eyes[:, i].detach()
            self.ages[i] += frames


def preview(path, frames, near, scale=4):
    def rgb(x):
        return (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

    def depth(inv):
        return np.repeat(rgb((inv * near).clamp(0, 1)), 3, -1)

    picks = [0, len(frames) // 2, len(frames) - 1]
    rows = []
    for i in range(len(frames[0][0])):
        row = []
        for k in picks:
            resolved, hole, pred, truth = (f[i] for f in frames[k])
            shown = resolved[RGB] * (1 - hole) + torch.tensor([1.0, 0, 1])[:, None, None] * hole
            row += [rgb(shown), rgb(pred[RGB]), rgb(truth[RGB]), depth(pred[INV:INV + 1]), depth(truth[INV:INV + 1])]
            row.append(np.zeros((row[0].shape[0], 2, 3), np.uint8))
        rows.append(np.concatenate(row, 1))
    image = Image.fromarray(np.concatenate(rows, 0))
    image.resize((image.width * scale, image.height * scale), Image.NEAREST).save(path)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: uv run train.py runs/<name>")
    project = Path(sys.argv[1])
    config = load(project)
    if config.train.threads:
        torch.set_num_threads(config.train.threads)
    data = Footage(config)
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
        # Fresh optimiser and schedule, the weights of an export (same layout).
        source = (project / config.train.init).resolve()
        missing, _ = net.load_state_dict(load_export(source).state_dict(), strict=False)
        assert all(k.startswith(("down.", "copy_head.")) for k in missing), missing  # new parts start at zero
        print(f"starting from {source}")
    print(f"net: {net.inputs} inputs, layers {[l.out_channels for l in net.layers]}, "
          f"{sum(p.numel() for p in net.parameters())} weights, {len(net.taps) + 1} taps")
    # The largest step it trains on (ticks), for the app's clamp.
    max_tau = max(float(s) for s in data.strides) / min(c["fps"] for c in data.chunks) * net.rate
    rng = np.random.default_rng(start)
    one = Fraction(1)
    fixed = data.batch(8, np.random.default_rng(1234), picks=[(*data.pick(np.random.default_rng(1234 + i), stride=one)[:4], one, 0.0)
                                       for i in range(8)])
    long = config.train.horizon
    fixed_long = None
    if long > config.train.steps:
        g = np.random.default_rng(1234)
        fixed_long = data.batch(8, g, steps=long, picks=[(*data.pick(g, steps=long, stride=one)[:4], one, 0.0)
                                                            for _ in range(8)])
    streams = Streams(data, config.train.batch)
    (project / "preview").mkdir(exist_ok=True)
    log = open(project / "log.txt", "a")
    began, total = time.time(), 0.0
    stereo = config.train.stereo
    for it in range(start + 1, config.train.iters + 1):
        batch, initial = streams.batch(rng)
        loss, _, final = rollout(net, data, batch, config.train.observe, start=initial, dropout=config.train.dropout)
        streams.advance(final)
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
            data.refresh()
            config.train.stereo = 0  # validation: one eye, whole frames, as before
            with torch.no_grad():
                val, frames, _ = rollout(net, data, fixed, 0.0, record=True)
                change, net.max_change = net.max_change, 0.0
                base, _, _ = rollout(net, data, fixed, 0.0)  # push-pull fill only
                long_base = rollout(net, data, fixed_long, 0.0)[0].item() if fixed_long else 0
                net.max_change = change
                long_val = rollout(net, data, fixed_long, 0.0)[0].item() if fixed_long else 0
            config.train.stereo = stereo
            line = f"{it} free-running val {val.item():.4f} (push-pull only {base.item():.4f})"
            if fixed_long:
                line += f", over {long} frames {long_val:.4f} (push-pull only {long_base:.4f})"
            print(line, flush=True)
            log.write(line + "\n")
            preview(project / "preview" / f"{it:06d}.png", frames, config.net.near)
            net.export(project / "net", config, max_tau=max_tau)
            net.export(project / "nets" / f"{it:06d}", config, max_tau=max_tau)  # every version (~20 KB)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it},
                       checkpoint)


if __name__ == "__main__":
    main()
