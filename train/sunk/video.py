"""Light video helpers shared by prepare.py and training (no model imports):
probe, read (ffmpeg), the chunk schedule, pinhole rays."""
import json, subprocess

import numpy as np
import torch


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    return float(info["format"]["duration"]), num / den, stream["width"], stream["height"]


def read(path, start, seconds, width, height):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-i", path, "-t", str(seconds),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def chunks(config):
    """Stratified: split the total length into equal segments, one random chunk each."""
    videos = sorted(config.frames_dir().glob("*.mp4"))
    lengths = [probe(v)[0] - config.data.chunk for v in videos]
    total = sum(lengths)
    count = max(1, round(config.data.minutes * 60 / config.data.chunk))
    rng = np.random.default_rng(config.data.seed)
    for k in range(count):
        at = (k + rng.random()) * total / count
        for video, length in zip(videos, lengths):
            if at < length:
                yield video, at
                break
            at -= length


def rays(h, w, fov):
    focal = w / 2 / np.tan(np.radians(fov) / 2)
    y, x = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    return torch.stack([(x - w / 2) / focal, -(y - h / 2) / focal, torch.ones_like(x)], -1)  # y up, looks +z
