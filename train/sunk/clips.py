"""cube-v1 training footage: clips at a random scale, cut around a world-fixed canvas.

A clip is one stream's footage: chunk c from frame t0 on, `frames` frames, the square
frame seen at S px (log-uniform in train.scales; data.size = the prepared chunks, up to
the native square crop), and a canvas: a rectangle of a virtual EAC face (sunk/sphere)
with texel angle 1 / focal_S (the frame's centre pixel), fixed to the world. Per frame
it keeps only a uint8 RGB window around where the canvas (with its context ring)
lands, plus the window's offset. Depth and velocity come from the chunk's prepared
arrays (data.size, already in RAM), sampled by ray: bilinear upsampling.

Frames: S = data.size from the chunk itself; S <= 256 from the prepared square videos
(data/frames/<infer>/<rate>/); larger from train/videos/ (the originals: the same centre
crop, at train.trim + chunk start + t0 / fps). The decoded frames are aligned to the
chunk by comparing them at data.size (best of -2..2 frames).

ClipPool keeps train.clips clips; a background thread replaces the oldest, one ffmpeg
at a time.
"""
import math, subprocess, threading, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sunk.camera import focal_of
from sunk.config import ROOT
from sunk.sphere import QUARTER, directions, look_at
from sunk.video import chunks as chunk_schedule

SLACK = 2  # frames decoded before t0 (alignment search -SLACK..SLACK)


class Clip:
    """See the module docstring. Fields: c, t0, frames, scale, focal, angle, origin [2]
    (face texel of the big canvas's (0, 0)), big (canvas + ring, texels), cams [frames, 3, 3]
    (canvas -> camera f), focal [frames] (px at scale), window [frames, h, w, 3] uint8, offset [frames, 2] (x, y px),
    error (alignment: mean abs error at data.size, 0..1)."""


def place(rng, big, angle, fov):
    """A canvas of big x big texels on a virtual face and its rotation into camera 0:
    face position random where it fits, its centre looking at a random point of the
    middle of camera 0's view. Returns (origin [2], canvas -> camera 0 [3, 3])."""
    half = big * angle / 2
    room = max(QUARTER - half, 0.0)
    u0, v0 = rng.uniform(-room, room, size=2)
    origin = torch.tensor([(u0 + QUARTER) / angle - 0.5 - (big - 1) / 2,
                           (v0 + QUARTER) / angle - 0.5 - (big - 1) / 2], dtype=torch.float32)
    centre = torch.tensor([math.tan(u0), -math.tan(v0), 1.0])
    reach = math.tan(math.radians(fov) / 2) * 0.5
    look = torch.tensor([rng.uniform(-reach, reach), rng.uniform(-reach, reach), 1.0])
    return origin, look_at(look).T @ look_at(centre)


def cameras(chunk, t0, frames, start):
    """canvas -> camera f for f in 0..frames-1: C_{f+1} = R_f C_f."""
    cams = [start]
    for f in range(t0, t0 + frames - 1):
        cams.append(torch.tensor(chunk["R"][f]) @ cams[-1])
    return torch.stack(cams)


def project(dirs, cams, focal, size):
    """Pixel coords [F, ..., 2] (x right, y down; centres at +0.5 - 0.5) and in-front mask;
    focal: one per frame [F]."""
    c = torch.einsum("fij,...j->f...i", cams, dirs)
    z = c[..., 2].clamp_min(1e-6)
    focal = torch.as_tensor(focal, dtype=c.dtype).reshape((-1,) + (1,) * (c.dim() - 2))
    px = focal * c[..., 0] / z + size / 2 - 0.5
    py = -focal * c[..., 1] / z + size / 2 - 0.5
    return torch.stack([px, py], -1), c[..., 2] > 1e-3


def decode(path, start, count, size, crop):
    """count RGB frames from `start` s, centre-cropped square (crop) and scaled to size;
    yields them one by one (no full buffer)."""
    vf = ("crop=ih:ih," if crop else "") + f"scale={size}:{size}:flags=area"
    proc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-ss", f"{max(start, 0):.4f}", "-i", str(path),
                             "-frames:v", str(count), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    n = size * size * 3
    try:
        for _ in range(count):
            raw = proc.stdout.read(n)
            if len(raw) < n:
                break
            yield np.frombuffer(raw, np.uint8).reshape(size, size, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def small(frame, size):
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.BOX), np.float32) / 255


class Clips:
    """Makes clips from a train.Footage (its chunks, starts, usable())."""

    def __init__(self, footage, config):
        self.footage, self.config = footage, config
        self.schedule = list(chunk_schedule(config))  # chunk index -> (prepared video, start s)

    def big(self):
        t = self.config.train
        return t.crop + 2 * t.margin + 2 * t.context

    def source(self, c, scale):
        """(path, start offset s, crop?) of chunk c's footage at this scale."""
        index = int(self.footage.paths[c].stem.split("_")[1])
        video, start = self.schedule[index]
        if scale <= self.config.data.infer:
            return video, start, False
        return ROOT / "videos" / video.name, start + self.config.train.trim, True

    def make(self, rng, frames, scale=None, pick=None):
        cfg, footage = self.config, self.footage
        lo, hi = cfg.train.scales[0], cfg.train.scales[-1]
        scale = int(round(scale or math.exp(rng.uniform(math.log(lo), math.log(hi)))))
        if pick is None:  # prefer starts with the whole clip ahead (no cut), else the longest seen
            best = None
            for _ in range(64):
                c, t0 = footage.starts[rng.integers(len(footage.starts))]
                if best is None or footage.ahead[c][t0] > footage.ahead[best[0]][best[1]]:
                    best = (c, t0)
                if footage.usable(c, t0, frames - 1):
                    break
            c, t0 = best
        else:
            c, t0 = pick
        chunk = footage.chunks[c]
        frames = int(min(frames, footage.ahead[c][t0] + 1))  # pairs ahead + 1 frames
        clip = Clip()
        clip.c, clip.t0, clip.frames, clip.scale = c, t0, frames, scale
        fov = chunk["fov"][t0:t0 + frames]  # per frame (deg): zooms change the focal length
        clip.focal = torch.tensor([focal_of(scale, float(f)) for f in fov])
        clip.angle = 1 / float(clip.focal[0])  # the canvas keeps frame 0's texel angle
        big = self.big()
        clip.big = big
        clip.origin, start = place(rng, big, clip.angle, float(fov[0]))
        clip.cams = cameras(chunk, t0, frames, start)

        # Where the big canvas lands in each frame: a coarse grid of its texels is enough.
        dirs = directions(torch.tensor([clip.angle]), clip.origin[None], big, big)[0]
        edge = torch.cat([dirs[::4, ::4].reshape(-1, 3), dirs[-1, :].reshape(-1, 3), dirs[:, -1].reshape(-1, 3)])
        px, front = project(edge, clip.cams, clip.focal, scale)
        inside = front & (px >= -1).all(-1) & (px <= scale).all(-1)
        boxes = []
        for f in range(frames):
            if inside[f].any():
                p = px[f][inside[f]]
                lo_ = (p.amin(0).floor() - 2).clamp(0, scale - 1)
                hi_ = (p.amax(0).ceil() + 3).clamp(1, scale)
                boxes.append((lo_, hi_))
            else:
                boxes.append(None)
        size = [1, 1]
        for box in boxes:
            if box is not None:
                size = [max(size[i], int(box[1][i] - box[0][i])) for i in (0, 1)]
        ww, wh = size
        clip.offset = torch.zeros(frames, 2, dtype=torch.long)
        for f, box in enumerate(boxes):
            if box is not None:
                clip.offset[f, 0] = int(min(box[0][0], scale - ww))
                clip.offset[f, 1] = int(min(box[0][1], scale - wh))
        clip.window = np.zeros((frames, wh, ww, 3), np.uint8)

        def cut(f, frame):
            ox, oy = clip.offset[f].tolist()
            clip.window[f] = frame[oy:oy + wh, ox:ox + ww]

        size0 = cfg.data.size
        if scale == size0:
            for f in range(frames):
                cut(f, chunk["rgb"][t0 + f])
            clip.error = 0.0
            return clip
        path, start_s, crop = self.source(c, scale)
        fps = chunk["fps"]
        begin = start_s + (t0 - SLACK) / fps
        stream = decode(path, begin, frames + 2 * SLACK, scale, crop)
        head = [next(stream, None) for _ in range(2 * SLACK + min(frames, 8))]
        head = [h for h in head if h is not None]
        # Alignment: decoded frame SLACK + k + shift should be chunk frame t0 + k.
        best, best_err = 0, float("inf")
        reference = chunk["rgb"][t0:t0 + min(frames, 8)].astype(np.float32) / 255
        smalls = [small(h, size0) for h in head]
        for shift in range(-SLACK, SLACK + 1):
            errs = [np.abs(smalls[SLACK + k + shift] - reference[k]).mean()
                    for k in range(len(reference)) if 0 <= SLACK + k + shift < len(smalls)]
            if errs and np.mean(errs) < best_err:
                best, best_err = shift, float(np.mean(errs))
        clip.error = best_err
        index = 0
        for frame in head:
            f = index - SLACK - best
            if 0 <= f < frames:
                cut(f, frame)
            index += 1
        for frame in stream:
            f = index - SLACK - best
            if 0 <= f < frames:
                cut(f, frame)
            index += 1
        return clip


class ClipPool:
    """train.clips clips; a background thread keeps replacing the oldest."""

    def __init__(self, clips, config, frames, seed=0):
        self.clips, self.frames = clips, frames
        self.rng = np.random.default_rng(seed)
        self.pool, self.lock, self.stop = [], threading.Lock(), False
        began = time.time()
        while len(self.pool) < config.train.clips:
            self.pool.append(clips.make(self.rng, frames))
        print(f"{len(self.pool)} clips in {time.time() - began:.0f}s", flush=True)
        self.next = 0
        self.thread = threading.Thread(target=self.refill, daemon=True)
        self.thread.start()

    def refill(self):
        rng = np.random.default_rng(self.rng.integers(1 << 31))
        while not self.stop:
            try:
                clip = self.clips.make(rng, self.frames)
            except Exception as e:  # a bad decode: skip it, keep the old clip
                print(f"clip failed: {e}", flush=True)
                time.sleep(5)
                continue
            with self.lock:
                self.pool[self.next] = clip
                self.next = (self.next + 1) % len(self.pool)
            time.sleep(0.5)

    def take(self, rng):
        with self.lock:
            return self.pool[rng.integers(len(self.pool))]
