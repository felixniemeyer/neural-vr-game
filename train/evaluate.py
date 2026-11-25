"""Compare a trained net with the push-pull fill alone: uv run evaluate.py runs/<name> [samples] [steps] [chunks] [--damage[=colour|depth|both]]
(cube-v1 projects: train_cube.evaluate, at most 24 samples, no --damage)

Free-running rollouts (no blend toward the real frames) on fixed samples; mean L1 of
colour and depth per step, over all known pixels and over pixels that were holes
before the net ran. Lower than "push-pull" = the net adds something.
L1 rewards blur, so also: sharpness (mean |image gradient| relative to the truth's) and
contrast (luminance std relative to the truth's), 1 = like the footage, < 1 = washed out.
--damage: every start state is damaged (train.damage's blobs, fixed seed); the "holes"
column becomes the error where the damage is (tracked along the rollout): does it regrow?
"""
import sys

from fractions import Fraction

import numpy as np
import torch

from sunk.config import load
from sunk.model import INV, RGB, V2, load_export, screen_motion, splat
from train import Footage, damage, truth_state

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--damage")]
damaged = flags[0].partition("=")[2] or "colour" if flags else None
project = args[0]
samples = int(args[1]) if len(args) > 1 else 64
if load(project).net.layout == "cube-v1":  # the sphere pipeline: train_cube.evaluate
    import train_cube
    train_cube.evaluate(project, min(samples, 24), int(args[2]) if len(args) > 2 else None,
                        int(args[3]) if len(args) > 3 else None)
    sys.exit()
torch.set_num_threads(2)
config = load(project)
if len(args) > 3:
    config.train.chunks = int(args[3])  # same data subset for comparing runs (memory)
data = Footage(config)
net = load_export(f"{project}/net")
steps = int(args[2]) if len(args) > 2 else config.train.steps
g = np.random.default_rng(7)
one = Fraction(1)  # whole video frames, one eye
batch = data.batch(samples, g, steps=steps, picks=[(*data.pick(g, steps, one)[:4], one, 0.0) for _ in range(samples)])


def grad(x):
    return ((x[..., 1:, :-1] - x[..., :-1, :-1]).abs() + (x[..., :-1, 1:] - x[..., :-1, :-1]).abs()).mean((1, 2, 3))


def ratio(a, b):
    return (a / b.clamp_min(1e-4)).mean().item()


def run(change):
    net.max_change = change
    near = config.net.near
    state = truth_state(batch, 0, net.channels)
    if damaged:  # the damage rides along in a tracker state's red channel
        state, mask = damage(state, 1.0, 0.5, (damaged,), torch.Generator().manual_seed(3))
        tracker = truth_state(batch, 0, net.channels)
        tracker[:, 0] = mask[:, 0]
    totals = np.zeros((steps, 4))
    with torch.no_grad():
        for k in range(steps):
            move = (batch["R"][:, k], batch["t"][:, k], batch["rays"], batch["origin"], data.focal, config.data.size, batch["dt"])
            resolved, hole = splat(state, *move, layout=net.layout)
            if damaged:
                tracker, _ = splat(tracker, *move)
                hole = (tracker[:, :1] > 0.5).float()
            motion = screen_motion(resolved, batch["t"][:, k] / batch["dt"][:, None], batch["rays"], data.focal,
                                   net.rate) if net.layout == V2 else None
            state = net(resolved, batch["dt"] * net.rate, motion)
            truth = truth_state(batch, k + 1, net.channels)
            known = batch["mask"][:, k + 1, None] * batch["inner"]
            error = torch.cat([(state[:, RGB] - truth[:, RGB]).abs(),
                               (state[:, INV:INV + 1] - truth[:, INV:INV + 1]).abs() * near], 1).mean(1, keepdim=True)
            totals[k] = [(error * known).sum() / known.sum(),
                         (error * known * hole).sum() / (known * hole).sum().clamp_min(1),
                         ratio(grad(state[:, RGB].clamp(0, 1)), grad(truth[:, RGB])),
                         ratio(state[:, RGB].clamp(0, 1).mean(1).flatten(1).std(1), truth[:, RGB].mean(1).flatten(1).std(1))]
    return totals


trained, base = run(net.max_change), run(0.0)
print(f"{project}: {samples} free-running rollouts of {steps} steps")
print(f"step   all: net / push-pull    {damaged + ' damage' if damaged else 'holes'}: net / push-pull    sharpness: net / pp    contrast: net / pp")
for k in range(steps):
    print(f"{k + 1:4}   {trained[k, 0]:.4f} / {base[k, 0]:.4f}      {trained[k, 1]:.4f} / {base[k, 1]:.4f}"
          f"         {trained[k, 2]:.2f} / {base[k, 2]:.2f}          {trained[k, 3]:.2f} / {base[k, 3]:.2f}")
gain = 1 - trained.mean(0) / base.mean(0)
print(f"net vs push-pull: {gain[0]:+.1%} all pixels, {gain[1]:+.1%} {'damage' if damaged else 'holes'} (positive = better); "
      f"mean sharpness {trained[:, 2].mean():.2f} vs {base[:, 2].mean():.2f}, contrast {trained[:, 3].mean():.2f} vs {base[:, 3].mean():.2f} (1 = like the footage)")
