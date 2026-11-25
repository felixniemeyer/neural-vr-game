"""Per-frame field of view, and the camera motion refit with it: uv run fov.py runs/<name> [--measure]

FOV = the vertical field of view of the full video frame (deg), which is that of the
square centre crop (frames.sh crops ih x ih). Per frame, best source first:
  measured  --measure: GeoCalib (single-image calibration) on frames of the original video
            (train/videos/, at train.trim + chunk start), every MEASURE_EVERY s of each shot
            (at least 2). Per shot the estimates are smoothed in time (Kalman filter + RTS
            smoother; the FOV is a random walk of DRIFT deg/sqrt(s), so zooms are followed
            and lens noise is averaged), starting from the prior below, restarting at cuts.
            Needs the optional extra: uv add --optional fov "geocalib @ git+https://github.com/cvg/GeoCalib"
            and uv run --extra fov fov.py runs/<name> --measure
  sidecar   videos/<video>.json {"vfov": deg} (next to videos/<video>.mp4)
  assumed   data.fov, with a warning
Shots: a cut is a pair prepare.py marked not ok whose frames differ by more than data.cut.

Output: data/prepared/<name>/chunk_NNN.cam.npz per chunk: fov [frames] (deg), source
[frames] (0 assumed, 1 sidecar, 2 measured), and R, t, vel refit from the chunk's stored
disp / flow / valid with each frame's FOV (ok unchanged); fov.json: the estimates per
shot. Training reads them with train.cameras = "measured" (cube-v1).
"""
import json, subprocess, sys, time

import numpy as np
import torch

from sunk.camera import fit_pair
from sunk.config import ROOT, load
from sunk.video import chunks

MEASURE_EVERY = 2.0  # s between measured frames within a shot
PRIOR_SIGMA = 8.0    # deg: how far a measurement may pull away from the sidecar / data.fov
DRIFT = 1.0          # deg / sqrt(s): how fast the FOV may change within a shot (zooms)
FLOOR = 1.0          # deg: minimum uncertainty of one measurement
ASSUMED, SIDECAR, MEASURED = 0, 1, 2


def sidecar(video, config):
    """(prior FOV deg, source) of a prepared video: its original's videos/<name>.json."""
    path = ROOT / "videos" / (video.stem + ".json")
    if path.exists():
        return float(json.loads(path.read_text())["vfov"]), SIDECAR
    print(f"warning: no {path.relative_to(ROOT)}, assuming data.fov = {config.data.fov} deg", flush=True)
    return float(config.data.fov), ASSUMED


def shots(chunk, config):
    """[(first, last)] frame ranges between cuts."""
    rgb = chunk["rgb"]
    out, first = [], 0
    for i, ok in enumerate(chunk["ok"]):
        if not ok and np.abs(rgb[i + 1].astype(np.float32) - rgb[i]).mean() / 255 > config.data.cut:
            out.append((first, i))
            first = i + 1
    out.append((first, len(rgb) - 1))
    return out


def smooth(times, values, sigmas, prior, prior_sigma, drift):
    """1-D Kalman filter + RTS smoother of a random walk starting at prior; returns the
    smoothed values at `times`."""
    n = len(times)
    xp, pp, xf, pf = (np.zeros(n) for _ in range(4))
    x, p, last = prior, prior_sigma ** 2, times[0]
    for k in range(n):
        p += drift ** 2 * (times[k] - last)
        xp[k], pp[k] = x, p
        gain = p / (p + sigmas[k] ** 2)
        x, p = x + gain * (values[k] - x), (1 - gain) * p
        xf[k], pf[k], last = x, p, times[k]
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        xs[k] = xf[k] + pf[k] / pp[k + 1] * (xs[k + 1] - xp[k + 1])
    return xs


class Meter:
    """GeoCalib on single frames of the original videos."""

    def __init__(self):
        try:
            from geocalib import GeoCalib
        except ImportError:
            sys.exit('--measure needs GeoCalib: uv add --optional fov "geocalib @ git+https://github.com/cvg/GeoCalib",\n'
                     "then uv run --extra fov fov.py runs/<name> --measure")
        self.model = GeoCalib().eval()

    def frame(self, path, at):
        """The full frame at `at` s, 480 px high, as a [3, H, W] tensor in 0..1."""
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{at:.4f}", "-i", str(path), "-frames:v", "1",
                              "-vf", "scale=-2:480", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        w = len(raw) // (480 * 3)
        return torch.tensor(np.frombuffer(raw, np.uint8).reshape(480, w, 3)).permute(2, 0, 1).float() / 255

    @torch.no_grad()
    def __call__(self, path, at):
        """(vfov deg, uncertainty deg) of the frame at `at` s."""
        result = self.model.calibrate(self.frame(path, at))
        vfov = float(np.degrees(result["camera"].vfov.reshape(-1)[0]))
        sigma = result.get("vfov_uncertainty")
        sigma = float(np.degrees(sigma.reshape(-1)[0])) if sigma is not None else 2.0
        return vfov, max(sigma, FLOOR)


def per_frame(chunk, config, prior, source, meter, original, start):
    """fov [frames], source [frames], and the per-shot record for fov.json."""
    n, fps = len(chunk["rgb"]), float(chunk["fps"])
    fov = np.full(n, prior, np.float32)
    kind = np.full(n, source, np.uint8)
    record = []
    for first, last in shots(chunk, config):
        entry = {"frames": [first, last]}
        if meter is not None and last > first:
            length = (last - first + 1) / fps
            count = int(np.clip(round(length / MEASURE_EVERY), 2, 8))
            frames = [first + (k + 0.5) / count * (last - first) for k in range(count)]
            values, sigmas = zip(*(meter(original, start + f / fps) for f in frames))
            times = np.array(frames) / fps
            smoothed = smooth(times, np.array(values), np.array(sigmas), prior, PRIOR_SIGMA, DRIFT)
            fov[first:last + 1] = np.interp(np.arange(first, last + 1) / fps, times, smoothed)
            kind[first:last + 1] = MEASURED
            entry.update(measured=[round(v, 2) for v in values], sigma=[round(s, 2) for s in sigmas],
                         smoothed=[round(float(v), 2) for v in smoothed])
        record.append(entry)
    return fov, kind, record


def refit(z, fov, config):
    """R, t, vel of every ok pair, fitted with frames t and t+1's FOV."""
    R, T, vel = z["R"].copy(), z["t"].copy(), z["vel"].copy()
    for i in np.nonzero(z["ok"])[0]:
        good = torch.tensor(z["valid"][i] > 0)
        disp = torch.tensor(z["disp"][i].astype(np.float32))
        flow = torch.tensor(z["flow"][i].astype(np.float32)).permute(2, 0, 1)
        r, t, v = fit_pair(disp, flow, good, float(fov[i]), float(fov[i + 1]), config.data.still, config.data.moving)
        R[i], T[i], vel[i] = r.numpy(), t.numpy(), v.numpy()
    return R, T, vel


def main():
    args = sys.argv[1:]
    measure = "--measure" in args
    args = [a for a in args if a != "--measure"]
    if len(args) != 1:
        sys.exit("usage: uv run fov.py runs/<name> [--measure]")
    config = load(args[0])
    torch.set_num_threads(config.train.threads or torch.get_num_threads())
    out_dir = config.prep_dir()
    meter = Meter() if measure else None
    schedule = list(chunks(config))
    summary = json.loads((out_dir / "fov.json").read_text()) if (out_dir / "fov.json").exists() else {}
    for path in sorted(out_dir.glob("chunk_[0-9][0-9][0-9].npz")):
        k = int(path.stem.split("_")[1])
        video, start = schedule[k]
        prior, source = sidecar(video, config)
        key = f"{'measured' if measure else 'prior'}:{prior}"
        cam = path.with_suffix(".cam.npz")
        if cam.exists() and str(np.load(cam)["key"]) == key:
            continue
        began = time.time()
        z = np.load(path)
        fov, kind, record = per_frame(z, config, prior, source, meter, ROOT / "videos" / video.name, start + config.train.trim)
        R, T, vel = refit(z, fov, config)
        np.savez_compressed(cam.with_suffix(".tmp.npz"), fov=fov, source=kind, R=R, t=T, vel=vel, ok=z["ok"], key=key)
        cam.with_suffix(".tmp.npz").rename(cam)
        summary[path.stem] = {"video": video.name, "prior": prior, "shots": record}
        (out_dir / "fov.json").write_text(json.dumps(summary, indent=1))
        print(f"{path.stem}: {len(record)} shots, fov {fov.min():.1f}..{fov.max():.1f} deg, "
              f"{time.time() - began:.0f}s", flush=True)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
