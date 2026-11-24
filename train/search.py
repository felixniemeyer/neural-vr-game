"""Hyperparameter search with Optuna: uv run search.py runs/<base> [trials] [iters] [--arch]

Each trial copies the base run's config.toml into runs/search/<base>/<trial>/ with sampled
settings, trains it (train.py, one trial at a time: memory), then scores it with
evaluate.py on free-running rollouts. The study lives in runs/search/<base>/study.db, so
the search resumes where it stopped (also after Ctrl+C or a pause on battery).

Score (lower = better), for the art track: the L1 error relative to push-pull, plus how
far the sharpness and contrast are from the footage's (|log ratio|), so blur and noise
both cost. Default: loss weights and training settings, from the base's init (same net
layout). --arch: also the layout (hidden width, mip levels), from scratch.
"""
import re, subprocess, sys
from pathlib import Path

import optuna

base = Path(sys.argv[1])
args = [a for a in sys.argv[2:] if not a.startswith("--")]
trials = int(args[0]) if args else 20
iters = int(args[1]) if len(args) > 1 else 250
arch = "--arch" in sys.argv
root = Path("runs/search") / (base.name + ("-arch" if arch else ""))
root.mkdir(parents=True, exist_ok=True)
template = (base / "config.toml").read_text()


def setting(text, section, key, value):
    """Set key = value in [section] of a config.toml text (added if missing)."""
    value = repr(value).lower() if isinstance(value, bool) else repr(value)
    head = text.index(f"[{section}]")
    end = text.find("\n[", head + 1)
    end = len(text) if end < 0 else end
    block = text[head:end]
    line = re.compile(rf"^{key} = .*$", re.M)
    block = line.sub(f"{key} = {value}", block) if line.search(block) else block.rstrip("\n") + f"\n{key} = {value}\n"
    return text[:head] + block + text[end:]


def objective(trial):
    project = root / f"{trial.number:03d}"
    project.mkdir(exist_ok=True)
    text = template
    values = {
        ("train", "lr"): trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        ("train", "texture"): trial.suggest_float("texture", 0.0, 3.0),
        ("train", "style"): trial.suggest_float("style", 0.0, 30.0),
        ("train", "edge_weight"): trial.suggest_float("edge_weight", 0.0, 2.0),
        ("train", "hole_weight"): trial.suggest_float("hole_weight", 0.0, 8.0),
        ("train", "observe"): trial.suggest_float("observe", 0.0, 0.6),
        ("train", "steps"): trial.suggest_int("steps", 4, 16),
        ("train", "iters"): iters,
        ("train", "preview"): iters,
        ("train", "on_battery"): "pause",
    }
    if arch:
        width = trial.suggest_categorical("hidden", [4, 8, 16, 32])
        values[("net", "hidden")] = [width] if width <= 8 else [width, 16]
        values[("net", "levels")] = trial.suggest_int("levels", 2, 5)
        values[("train", "init")] = ""
    elif '\ninit = "' in template:
        # The base's init is relative to the base's folder.
        init = re.search(r'\ninit = "([^"]+)"', template).group(1)
        values[("train", "init")] = str((base / init).resolve())
    for (section, key), value in values.items():
        text = setting(text, section, key, value)
    (project / "config.toml").write_text(text)
    with open(project / "train.out", "a") as log:
        subprocess.run(["uv", "run", "train.py", str(project)], stdout=log, stderr=subprocess.STDOUT, check=True)
    out = subprocess.run(["uv", "run", "evaluate.py", str(project), "16", "48"], capture_output=True, text=True, check=True).stdout
    (project / "eval48.txt").write_text(out)
    summary = re.search(r"net vs push-pull: ([+-][\d.]+)% all pixels.*mean sharpness ([\d.]+) vs [\d.]+, contrast ([\d.]+) vs", out)
    gain, sharp, contrast = float(summary[1]) / 100, float(summary[2]), float(summary[3])
    trial.set_user_attr("gain", gain)
    trial.set_user_attr("sharpness", sharp)
    trial.set_user_attr("contrast", contrast)
    import math
    return (1 - gain) + abs(math.log(max(sharp, 1e-3))) + abs(math.log(max(contrast, 1e-3)))


study = optuna.create_study(study_name=root.name, storage=f"sqlite:///{root / 'study.db'}", load_if_exists=True)
study.optimize(objective, n_trials=trials)
print(study.best_trial.number, study.best_value, study.best_params, study.best_trial.user_attrs)
