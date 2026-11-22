"""Side-by-side video of a prepared chunk: colour | depth (bright = near) | flow | object velocity.
Flow: standard colour wheel (hue = direction, saturation = speed, per frame normalised).
Velocity: red = right, green = up, 0.3 near-units = full.

uv run view.py data/prepared/<name>/chunk_000.npz   -> chunk_000.mp4 next to it (4x upscaled)
"""
import subprocess, sys
import numpy as np
import torch
from torchvision.utils import flow_to_image

path = sys.argv[1]
z = np.load(path)
rgb, disp, vel, ok = z["rgb"], z["disp"].astype(np.float32), z["vel"].astype(np.float32), z["ok"]
n, h, w, _ = rgb.shape
gray = lambda a: np.repeat((np.clip(a, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
frames = []
for i in range(n):
    v = vel[min(i, n - 2)]
    v = np.clip(np.concatenate([v[..., :2] / 0.3 * 0.5 + 0.5, np.zeros((h, w, 1))], -1), 0, 1)
    v = (v * 255).astype(np.uint8) * (np.abs(vel[min(i, n - 2)]).sum(-1, keepdims=True) > 0)
    f = flow_to_image(torch.tensor(z["flow"][min(i, n - 2)].astype(np.float32)).permute(2, 0, 1))
    f = f.permute(1, 2, 0).numpy()
    row = np.concatenate([rgb[i], gray(disp[i]), f, v], 1)
    if i < n - 1 and not ok[i]:
        row[:4] = (255, 0, 0)  # red bar: cut / dropped pair
    frames.append(row)
out = path.replace(".npz", ".mp4")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{4 * w}x{h}",
                "-r", "10", "-i", "-", "-vf", "scale=iw*4:ih*4:flags=neighbor", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "18", out], input=np.stack(frames).tobytes(), check=True)
print(out)
