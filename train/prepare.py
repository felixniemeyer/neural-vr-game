"""Preprocess video for a project: uv run prepare.py runs/<name>

Samples [data].minutes of footage in chunks spread over all videos, then per frame:
  disp   disparity (Depth Anything V2 Small, affine-invariant), per frame its 2nd..98th
         percentile mapped to [1/ratio, 1], i.e. depth 1 (near) .. ratio (far)
per consecutive pair (t -> t+1):
  R, t   rigid camera motion, p' = R p + t (camera t coords -> camera t+1 coords),
         robust reprojection fit: frame t's depth + RAFT flow (x right, y up, z ahead)
  vel    motion the camera doesn't explain (moving objects): the residual flow lifted
         to 3D at the pixel's depth, lateral only, camera t+1 coords; 0 elsewhere
  flow   RAFT optical flow (px, x right, y down)
  valid  flow target inside and photometrically consistent
  ok     pair usable (no cut, enough valid pixels)
Depth and flow run at [data].infer, results are area-downsampled to [data].size.
Output: data/prepared/<name>/chunk_NNN.npz (+ .png preview), shared by projects with the
same [data] section.
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from transformers import AutoModelForDepthEstimation

from sunk.config import ROOT, load
from sunk.camera import fit_pair, sample
from sunk.video import chunks, probe, read

BATCH = 16


class Models:
    def __init__(self, config):
        self.depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").eval()
        self.flow = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()
        self.size = config.data.depth_size
        self.iters = config.data.flow_iters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def disparity(self, x):
        out = []
        for b in x.split(BATCH):
            inp = F.interpolate((b - self.mean) / self.std, size=(self.size, self.size), mode="bilinear", antialias=True)
            d = self.depth(pixel_values=inp).predicted_depth[:, None]
            out.append(F.interpolate(d, size=x.shape[-2:], mode="bilinear")[:, 0])  # at infer size
        return torch.cat(out).clamp_min(1e-3)

    @torch.no_grad()
    def flows(self, x):
        a, b = x[:-1] * 2 - 1, x[1:] * 2 - 1
        return torch.cat([self.flow(p, q, num_flow_updates=self.iters)[-1]
                          for p, q in zip(a.split(BATCH), b.split(BATCH))])


def process(models, config, frames):
    """Depth and flow at the inference size, everything stored at data.size."""
    big = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255
    h = w = config.data.size
    down = big.shape[-1] // w
    x = F.avg_pool2d(big, down)
    frames = (x * 255).round().byte().permute(0, 2, 3, 1).numpy()
    n = len(x)
    raw = F.interpolate(models.disparity(big)[:, None], size=(h, w), mode="area")[:, 0]
    lo, hi = torch.quantile(raw.flatten(1), torch.tensor([0.02, 0.98]), dim=1)
    unit = ((raw - lo[:, None, None]) / (hi - lo)[:, None, None]).clamp(0, 1)
    disp = 1 / config.data.ratio + (1 - 1 / config.data.ratio) * unit
    flow = F.avg_pool2d(models.flows(big), down) / down
    pix = torch.stack(torch.meshgrid(torch.arange(w) + 0., torch.arange(h) + 0., indexing="xy"), -1)
    R = np.tile(np.eye(3, dtype=np.float32), (n - 1, 1, 1))
    T = np.zeros((n - 1, 3), np.float32)
    vel = np.zeros((n - 1, h, w, 3), np.float16)
    valid = np.zeros((n - 1, h, w), np.uint8)
    ok = np.zeros(n - 1, bool)
    for i in range(n - 1):
        target = pix + flow[i].permute(1, 2, 0)
        inside = (target[..., 0] >= 0) & (target[..., 0] <= w - 1) & (target[..., 1] >= 0) & (target[..., 1] <= h - 1)
        error = (x[i] - sample(x[i + 1], target)).abs().mean(0)
        good = inside & (error < 0.1)
        if error.mean() > config.data.cut or good.float().mean() < 0.5:
            continue
        fov = config.data.fov
        r, t, v = fit_pair(disp[i], flow[i], good, fov, fov, config.data.still, config.data.moving)
        R[i], T[i], vel[i], valid[i], ok[i] = r.numpy(), t.numpy(), v.numpy(), good.numpy(), True
    return dict(rgb=frames, flow=flow.permute(0, 2, 3, 1).numpy().astype(np.float16), disp=disp.numpy().astype(np.float16), R=R, t=T, vel=vel, valid=valid, ok=ok)


def preview(path, out):
    rows = []
    for i in np.linspace(0, len(out["ok"]) - 1, 6).astype(int):
        d = out["disp"][i].astype(np.float32)
        d = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        v = np.linalg.norm(out["vel"][i].astype(np.float32), axis=-1)
        v = (np.clip(v / 0.5, 0, 1) * 255).astype(np.uint8)
        m = out["valid"][i] * (255 if out["ok"][i] else 80)
        rows.append(np.concatenate([out["rgb"][i]] + [np.repeat(c[..., None], 3, -1) for c in (d, v, m)], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: uv run prepare.py runs/<name>")
    config = load(sys.argv[1])
    out_dir = config.prep_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = json.dumps(config.data.__dict__, indent=2)
    if (out_dir / "data.json").exists() and (out_dir / "data.json").read_text() != settings:
        sys.exit(f"{out_dir} holds data made with other [data] settings; set data.name")
    (out_dir / "data.json").write_text(settings)
    torch.set_num_threads(config.train.threads or torch.get_num_threads())
    models = Models(config)
    jobs = list(chunks(config))
    for k, (video, start) in enumerate(jobs):
        path = out_dir / f"chunk_{k:03d}.npz"
        if path.exists():
            continue
        began = time.time()
        _, fps, w, h = probe(video)
        out = process(models, config, read(video, start, config.data.chunk, w, h))
        out["fps"] = np.float32(fps)
        np.savez_compressed(path.with_suffix(".tmp.npz"), **out)
        path.with_suffix(".tmp.npz").rename(path)
        preview(path.with_suffix(".png"), out)
        print(f"{k + 1}/{len(jobs)} {video.name[:24]} @{start:.0f}s: {out['ok'].mean():.0%} pairs ok, "
              f"{time.time() - began:.0f}s", flush=True)
    print(f"done: {out_dir}\nper-frame FOV + cameras for train.cameras = measured: uv run fov.py {sys.argv[1]} [--measure]")


if __name__ == "__main__":
    main()
