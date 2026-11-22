"""One training project = one folder runs/<name>/ with a config.toml.

`uv run new.py <name> [--section.key value ...]` writes the defaults below (plus
overrides) into runs/<name>/config.toml; edit it freely before training. Every other
script takes the project folder. Preprocessed video data is shared between projects
with the same [data] section (data/prepared/<name>/, see Data.name).
"""
import json, tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Data:
    name: str = ""              # data/prepared/<name>/; empty = "<infer>to<size>-<minutes>min-<rate>"
    fps: float = 0              # video frame rate, 0 = each video's own (FPS=<fps> ./frames.sh)
    infer: int = 256            # depth + flow run on data/frames/<infer>/<rate>/ (SIZE=<infer> ./frames.sh)
    size: int = 128             # stored resolution (downsampled from infer)
    minutes: float = 15.0       # total footage sampled from all videos
    chunk: float = 20.0         # seconds per contiguous chunk
    fov: float = 60.0           # assumed horizontal field of view of the footage (deg)
    ratio: float = 8.0          # far / near of the footage's depth range
    depth_size: int = 252       # Depth Anything input (multiple of 14); 126 is too blurry
    flow_iters: int = 12        # RAFT refinement iterations
    still: float = 1.0          # camera translation penalty (prefer pans over parallax)
    moving: float = 0.3         # unexplained flow (px at data.size) from which a pixel moves
    cut: float = 0.12           # mean photometric error above which a pair is a cut
    seed: int = 0


@dataclass
class Net:
    levels: int = 5             # mip levels with a triangle of taps (radius 1, 2, 4, .. px)
    points: int = 3             # taps per level: 3 = triangle, 4 = triangle + mip centre
    hidden: list = field(default_factory=lambda: [16])  # hidden layer widths (multiples of 4)
    hidden_state: int = 4       # hidden channels in the state (multiple of 4)
    near: float = 10.0          # depth range (m): the footage's near .. near * data.ratio
    max_change: float = 0.5     # tanh-bounded update per tick
    rate: float = 24.0          # net ticks per second: a step of dt seconds is tau = dt * rate ticks


@dataclass
class Train:
    crop: int = 32              # training crop (px at data.size); the net is per pixel
    steps: int = 8              # net steps per iteration (BPTT)
    strides: list = field(default_factory=lambda: [1.0])  # video frames per net step, drawn per sample
    horizon: int = 8            # video frames a sample follows its clip, continuing from its own state
    stereo: float = 0.0         # weight of the two eyes' disagreement; 0 = one eye
    baseline: float = 1.0       # max distance of the second eye (m, uniform 0..baseline)
    batch: int = 16
    lr: float = 2e-3
    iters: int = 20000
    turn: float = 0.0           # head-turn augmentation: max border shift (fraction of crop)
    observe: float = 0.3        # max blend toward the real frame per step (like the blocks)
    hole_weight: float = 4.0    # extra loss weight on pixels that were holes
    edge_weight: float = 1.0    # loss on image gradients
    texture: float = 0.0        # loss on local edge energy (8x8 windows, position-free)
    preview: int = 250          # write preview + export every n iterations
    threads: int = 0            # torch threads, 0 = all
    init: str = ""              # start from these weights (an export folder, e.g. ../stereo24/nets/001250); "" = random


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    net: Net = field(default_factory=Net)
    train: Train = field(default_factory=Train)

    def prep_dir(self) -> Path:
        d = self.data
        return ROOT / "data" / "prepared" / (d.name or f"{d.infer}to{d.size}-{d.minutes:g}min-{self.rate()}")

    def rate(self) -> str:
        """The frame rate as a folder name: 'native' or e.g. '10fps'."""
        return f"{self.data.fps:g}fps" if self.data.fps else "native"

    def frames_dir(self) -> Path:
        return ROOT / "data" / "frames" / str(self.data.infer) / self.rate()


def _toml(value) -> str:
    return json.dumps(value) if isinstance(value, str) else repr(value).lower() if isinstance(value, bool) else repr(value)


def _cast(default, value):
    if isinstance(default, list):
        kind = type(default[0]) if default else int
        return [kind(v) for v in (value.split(",") if isinstance(value, str) else value)]
    return type(default)(value)


def save(config: Config, path: Path):
    lines = []
    for section in fields(config):
        part = getattr(config, section.name)
        lines.append(f"[{section.name}]")
        lines += [f"{f.name} = {_toml(getattr(part, f.name))}" for f in fields(part)]
        lines.append("")
    path.write_text("\n".join(lines))


def load(project: str | Path) -> Config:
    project = Path(project)
    raw = tomllib.loads((project / "config.toml").read_text())
    config = Config()
    for section, values in raw.items():
        part = getattr(config, section)
        for key, value in values.items():
            if not hasattr(part, key):
                raise KeyError(f"unknown setting {section}.{key}")
            setattr(part, key, _cast(getattr(part, key), value))
    return config


def override(config: Config, args: list[str]):
    """Apply `--section.key value` pairs (lists comma-separated: --net.hidden 32,16)."""
    for flag, value in zip(args[::2], args[1::2]):
        section, key = flag.removeprefix("--").split(".")
        part = getattr(config, section)
        if not hasattr(part, key):
            raise KeyError(f"unknown setting {section}.{key}")
        setattr(part, key, _cast(getattr(part, key), value))
