"""Create a training project: uv run new.py <name> [--section.key value ...]

e.g. uv run new.py big --train.size 384 --net.taps 16 --net.hidden 64
"""
import sys
from sunk.config import ROOT, Config, override, save

if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
    sys.exit(__doc__)
project = ROOT / "runs" / sys.argv[1]
if (project / "config.toml").exists():
    sys.exit(f"{project} exists")
config = Config()
override(config, sys.argv[2:])
project.mkdir(parents=True)
save(config, project / "config.toml")
print(f"{project / 'config.toml'}\nnext: uv run prepare.py runs/{sys.argv[1]}  (data: {config.prep_dir().relative_to(ROOT)})")
