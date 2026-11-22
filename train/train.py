"""Train a project: uv run train.py runs/<name>

Rollouts on real footage: the state starts as frame t (colour, depth, object velocity),
then for each of train.steps frames it is splatted with the estimated camera motion
(+ object velocity) into the next camera, resolved, and the net runs. The result is
compared with the real next frame (colour, depth, edges; holes weigh more). Then the
state is blended toward the real frame by a random 0..train.observe, like the
rendered blocks at runtime.

Writes into the project: log.txt, preview/<iter>.png (per sample: resolved with holes
magenta | prediction | truth | predicted depth | true depth, at 3 steps of a
free-running rollout), net/ (latest export for the app), checkpoint.pt (resumes).
"""
import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from prepare import rays as make_rays
from sunk.config import load
from sunk.model import CERT, HID, INV, RGB, VEL, Net, splat


class Footage:
    def __init__(self, config):
        self.config = config
        self.paths, self.chunks = [], []
        self.starts = []  # (chunk, t0) with train.steps usable pairs ahead
        self.ahead = []   # per chunk: usable pairs from each frame on (until a cut)
        size = config.data.size
        self.focal = size / 2 / np.tan(np.radians(config.data.fov) / 2)
        self.rays = make_rays(size, size, config.data.fov)
        self.refresh()

    def refresh(self):
        """Load chunks that appeared since (prepare.py may still be running)."""
        steps = self.config.train.steps
        for path in sorted(self.config.prep_dir().glob("chunk_[0-9][0-9][0-9].npz")):
            if path in self.paths:
                continue
            z = np.load(path)
            chunk = {k: z[k] for k in ("rgb", "disp", "vel", "R", "t", "ok")}
            ahead = np.zeros(len(chunk["ok"]) + 1, int)
            for t in range(len(chunk["ok"]) - 1, -1, -1):
                ahead[t] = (ahead[t + 1] + 1) * bool(chunk["ok"][t])
            self.starts += [(len(self.chunks), t) for t in np.nonzero(ahead >= steps)[0]]
            self.ahead.append(ahead)
            self.paths.append(path)
            self.chunks.append(chunk)
            print(f"{len(self.chunks)} chunks, {len(self.starts)} rollout starts", flush=True)

    def pick(self, rng, steps):
        """A random (chunk, t0, crop x, crop y) with `steps` usable pairs ahead."""
        crop, size = self.config.train.crop, self.config.data.size
        while True:
            c, t0 = self.starts[rng.integers(len(self.starts))]
            if self.ahead[c][t0] >= steps:
                return (c, t0, *rng.integers(size - crop + 1, size=2))

    def usable(self, c, t0, steps):
        return self.ahead[c][t0] >= steps

    def batch(self, n, rng, steps=None, picks=None):
        """Frames [B, K+1, ...] as seen by the (virtual) cameras, their motion [B, K],
        crop rays [B, crop, crop, 3], origins, and where the truth is known [B, K+1].
        K = steps (default train.steps); picks: (chunk, t0, crop x, crop y) per sample,
        default random.

        Head-turn augmentation (train.turn): the virtual camera k is the real one turned
        by D_k = rotation(theta * sin(pi k / K)) about a random axis in the image plane,
        theta up to a shift of turn * crop px. It turns away and back, so the crop loses
        content at one border and has to regenerate it. The truth is the real frame
        seen through the turned camera (rotation only: no depth needed, radial distance
        unchanged), unknown outside the real frame (mask).
        """
        cfg = self.config
        k, crop, size, near = steps or cfg.train.steps, cfg.train.crop, cfg.data.size, cfg.net.near
        picks = picks or [self.pick(rng, k) for _ in range(n)]
        out = {key: [] for key in ("rgb", "inv", "vel", "R", "t", "rays", "origin", "mask")}
        radial = self.rays.norm(dim=-1)
        for c, t0, ox, oy in picks:
            chunk = self.chunks[c]
            rgb = torch.tensor(chunk["rgb"][t0:t0 + k + 1]).float() / 255
            # disp is planar (1 / z in near units); the state holds radial 1 / distance.
            inv = torch.tensor(chunk["disp"][t0:t0 + k + 1].astype(np.float32)) / near / radial
            r = torch.tensor(chunk["R"][t0:t0 + k])
            # Object velocity of pair t -> t+1 is in camera t+1 coords (near units);
            # the state carries it in camera t coords (m): R^T v * near.
            vel = torch.einsum("kji,khwj->khwi", r, torch.tensor(chunk["vel"][t0:t0 + k].astype(np.float32))) * near
            vel = torch.cat([vel, torch.zeros_like(vel[:1])])

            angle = np.arctan(cfg.train.turn * crop * rng.random() / self.focal)
            axis = rng.normal(size=2)
            axis = torch.tensor([*axis / np.linalg.norm(axis), 0.0], dtype=torch.float32)
            turns = torch.stack([rotation(axis, angle * np.sin(np.pi * j / k)) for j in range(k + 1)])

            # Sample the real frames through the virtual cameras.
            rays = self.rays[oy:oy + crop, ox:ox + crop]
            real = torch.einsum("kji,hwj->khwi", turns, rays)  # D^T d
            z = real[..., 2].clamp_min(1e-6)
            px = self.focal * real[..., 0] / z + size / 2 - 0.5
            py = -self.focal * real[..., 1] / z + size / 2 - 0.5
            mask = ((real[..., 2] > 0) & (px >= 0) & (px <= size - 1) & (py >= 0) & (py <= size - 1)).float()
            grid = torch.stack([(px + 0.5) / size * 2 - 1, (py + 0.5) / size * 2 - 1], -1)
            frames = torch.cat([rgb, inv[..., None], vel], -1).permute(0, 3, 1, 2)
            seen = F.grid_sample(frames, grid, mode="bilinear", padding_mode="border", align_corners=False)
            seen = seen.permute(0, 2, 3, 1) * mask[..., None]
            out["rgb"].append(seen[..., :3])
            out["inv"].append(seen[..., 3])
            out["vel"].append(torch.einsum("kij,khwj->khwi", turns, seen[..., 4:]))
            # Virtual camera motion: p_virtual' = D' (R (D^T p) + t).
            out["R"].append(turns[1:] @ r @ turns[:-1].transpose(1, 2))
            out["t"].append(torch.einsum("kij,kj->ki", turns[1:], torch.tensor(chunk["t"][t0:t0 + k]) * near))
            out["rays"].append(rays)
            out["origin"].append(torch.tensor([ox, oy], dtype=torch.float32))
            out["mask"].append(mask)
        return {key: torch.stack(v) for key, v in out.items()}


def rotation(axis, angle):
    """Rodrigues: rotation by `angle` (rad) about the unit vector `axis`."""
    x, y, z = axis.tolist()
    k = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float32)
    return torch.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


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


def rollout(net, data, batch, observe, record=False, start=None):
    """Loss over the batch's steps; start: initial states [B, C, H, W], None where the
    rollout starts from the real frame. Returns (loss, recorded frames, final state)."""
    cfg = data.config
    near = cfg.net.near
    steps = batch["R"].shape[1]
    state = truth_state(batch, 0, net.channels)
    if start is not None:
        state = torch.stack([s if s is not None else f for s, f in zip(start, state)])
    loss, frames = 0.0, []
    for k in range(steps):
        resolved, hole = splat(state, batch["R"][:, k], batch["t"][:, k], batch["rays"], batch["origin"],
                               data.focal, cfg.data.size)
        pred = net(resolved)
        truth = truth_state(batch, k + 1, net.channels)
        known = batch["mask"][:, k + 1, None]
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
        loss = loss + step_loss / steps
        if record:
            frames.append((resolved, hole, pred, truth))
        alpha = observe * torch.rand(len(pred), 1, 1, 1) * known
        state = pred.clone()
        for channels in (RGB, slice(INV, INV + 1), VEL):
            state[:, channels] = pred[:, channels] * (1 - alpha) + truth[:, channels] * alpha
        state[:, CERT:CERT + 1] = pred[:, CERT:CERT + 1] * (1 - alpha) + alpha
    return loss, frames, state


class Streams:
    """Long rollouts with short backprop (truncated BPTT): each batch sample follows its
    clip for up to train.horizon frames. Every iteration continues where the previous
    one ended, from the net's own (detached) state, same crop, so the net learns to
    keep its own old states good, not only states fresh from a real frame. A stream
    restarts from a real frame at a cut, the end of its clip, or after horizon frames."""

    def __init__(self, data, n):
        self.data, self.n = data, n
        self.picks, self.states, self.ages = [None] * n, [None] * n, [0] * n

    def batch(self, rng):
        steps, horizon = self.data.config.train.steps, self.data.config.train.horizon
        for i in range(self.n):
            p = self.picks[i]
            if p is None or self.ages[i] + steps > horizon or not self.data.usable(p[0], p[1], steps):
                self.picks[i], self.states[i], self.ages[i] = self.data.pick(rng, steps), None, 0
        return self.data.batch(self.n, rng, picks=self.picks), self.states

    def advance(self, final):
        steps = self.data.config.train.steps
        for i in range(self.n):
            c, t0, ox, oy = self.picks[i]
            self.picks[i] = (c, t0 + steps, ox, oy)
            self.states[i] = final[i].detach()
            self.ages[i] += steps


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
    print(f"net: {net.inputs} inputs, layers {[l.out_channels for l in net.layers]}, "
          f"{sum(p.numel() for p in net.parameters())} weights, {len(net.taps) + 1} taps")
    rng = np.random.default_rng(start)
    fixed = data.batch(8, np.random.default_rng(1234))
    long = config.train.horizon
    fixed_long = data.batch(8, np.random.default_rng(1234), steps=long) if long > config.train.steps else None
    streams = Streams(data, config.train.batch)
    (project / "preview").mkdir(exist_ok=True)
    log = open(project / "log.txt", "a")
    began, total = time.time(), 0.0
    for it in range(start + 1, config.train.iters + 1):
        batch, start = streams.batch(rng)
        loss, _, final = rollout(net, data, batch, config.train.observe, start=start)
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
            with torch.no_grad():
                val, frames, _ = rollout(net, data, fixed, 0.0, record=True)
                change, net.max_change = net.max_change, 0.0
                base, _, _ = rollout(net, data, fixed, 0.0)  # push-pull fill only
                long_base = rollout(net, data, fixed_long, 0.0)[0].item() if fixed_long else 0
                net.max_change = change
                long_val = rollout(net, data, fixed_long, 0.0)[0].item() if fixed_long else 0
            line = f"{it} free-running val {val.item():.4f} (push-pull only {base.item():.4f})"
            if fixed_long:
                line += f", over {long} frames {long_val:.4f} (push-pull only {long_base:.4f})"
            print(line, flush=True)
            log.write(line + "\n")
            preview(project / "preview" / f"{it:06d}.png", frames, config.net.near)
            net.export(project / "net", config)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it},
                       checkpoint)


if __name__ == "__main__":
    main()
