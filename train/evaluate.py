"""Compare a trained net with the push-pull fill alone: uv run evaluate.py runs/<name> [samples] [steps]

Free-running rollouts (no blend toward the real frames) on fixed samples; mean L1 of
colour and depth per step, over all known pixels and over pixels that were holes
before the net ran. Lower than "push-pull" = the net adds something.
"""
import sys

import numpy as np
import torch

from sunk.config import load
from sunk.model import INV, RGB, load_export, splat
from train import Footage, truth_state

project = sys.argv[1]
samples = int(sys.argv[2]) if len(sys.argv) > 2 else 64
torch.set_num_threads(2)
config = load(project)
data = Footage(config)
net = load_export(f"{project}/net")
steps = int(sys.argv[3]) if len(sys.argv) > 3 else config.train.steps
batch = data.batch(samples, np.random.default_rng(7), steps=steps)


def run(change):
    net.max_change = change
    near = config.net.near
    state = truth_state(batch, 0, net.channels)
    totals = np.zeros((steps, 2))
    with torch.no_grad():
        for k in range(steps):
            resolved, hole = splat(state, batch["R"][:, k], batch["t"][:, k], batch["rays"], batch["origin"],
                                   data.focal, config.data.size)
            state = net(resolved)
            truth = truth_state(batch, k + 1, net.channels)
            known = batch["mask"][:, k + 1, None]
            error = torch.cat([(state[:, RGB] - truth[:, RGB]).abs(),
                               (state[:, INV:INV + 1] - truth[:, INV:INV + 1]).abs() * near], 1).mean(1, keepdim=True)
            totals[k] = [(error * known).sum() / known.sum(),
                         (error * known * hole).sum() / (known * hole).sum().clamp_min(1)]
    return totals


trained, base = run(net.max_change), run(0.0)
print(f"{project}: {samples} free-running rollouts of {steps} steps")
print("step   all: net / push-pull    holes: net / push-pull")
for k in range(steps):
    print(f"{k + 1:4}   {trained[k, 0]:.4f} / {base[k, 0]:.4f}      {trained[k, 1]:.4f} / {base[k, 1]:.4f}")
gain = 1 - trained.mean(0) / base.mean(0)
print(f"net vs push-pull: {gain[0]:+.1%} all pixels, {gain[1]:+.1%} holes (positive = better)")
