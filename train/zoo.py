"""The model zoo: every trained net, grouped by layout. Stdlib only (dev.sh runs it with python3).

  python3 zoo.py              list (layouts, runs, metrics)
  python3 zoo.py pick <file>  interactive picker (arrows/j/k, Enter, q); writes the choice
                              to <file>: "newest", "follow <run dir>" or "pin <net dir>"
                              (dev.sh: at start, and live with p in its terminal)
  python3 zoo.py runnable <net dir>   exit 0 if the app can run it, else print why

Nets: train/runs/<run>/net (followed: the app restarts on new exports of that run) and
loose exports such as nice-net/ at the repo root (pinned). Metrics from the run's
log.txt, last validation: the free-running error over train.steps steps and over the
long horizon, each next to push-pull alone (lower = better); cube-v1 also the error on
texels back in view after being out of it (recall).
"""
import json, re, sys, time
from pathlib import Path

TRAIN = Path(__file__).resolve().parent
ROOT = TRAIN.parent
ORDER = ["cube-v1", "reproject-v2", "reproject-v1"]  # newest layout first
NO_RUNTIME = {"cube-v1"}  # torch only so far


def problem(net_dir: Path):
    """Why the app cannot run this export (native/src/rule.rs), or None."""
    try:
        spec = json.loads((net_dir / "net.json").read_text())
    except (OSError, ValueError) as e:
        return f"unreadable net.json ({e.__class__.__name__})"
    if spec.get("layout", "reproject-v1") not in ORDER:
        return f"layout {spec['layout']}"
    if spec.get("layout") in NO_RUNTIME:
        return "no runtime yet"
    for feature in ("cnn", "down"):
        if spec.get(feature):
            return f"net.{feature} (torch only)"
    for feature in ("copy", "background"):
        if spec.get(feature):
            return f"net.{feature} (not in the shaders)"
    if spec.get("hidden_state", 4) != 4:
        return "hidden_state != 4"
    return None


def metrics(run: Path):
    """(steps val, push-pull, frames, long val, push-pull, (recall, push-pull) or None)
    from the last validation line."""
    try:
        lines = (run / "log.txt").read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        m = re.search(r"free-running val ([\d.]+) \(push-pull only ([\d.]+)\)(?:, over (\d+) frames ([\d.]+) \(push-pull only ([\d.]+)\))?", line)
        if m:
            val, pp, frames, long, long_pp = m.groups()
            r = re.search(r"recall ([\d.]+) \(push-pull only ([\d.]+)\)", line)
            return (float(val), float(pp), frames and int(frames), long and float(long), long_pp and float(long_pp),
                    r and (float(r.group(1)), float(r.group(2))))
    return None


def entries():
    """[(layout, name, kind, path, net dir, problem, metrics, exported, training)]"""
    found = []
    nets = [(p.parent.parent.name, "follow", p.parent.parent, p.parent) for p in TRAIN.glob("runs/*/net/net.json")]
    nets += [(p.parent.name, "pin", p.parent, p.parent) for p in ROOT.glob("*/net.json")]
    for name, kind, path, net in nets:
        try:
            layout = json.loads((net / "net.json").read_text()).get("layout", "reproject-v1")
            exported = (net / "weights.bin").stat().st_mtime
        except (OSError, ValueError):
            continue
        log = path / "log.txt"
        training = log.exists() and time.time() - log.stat().st_mtime < 120
        found.append((layout, name, kind, path, net, problem(net), metrics(path), exported, training))
    rank = {layout: i for i, layout in enumerate(ORDER)}
    found.sort(key=lambda e: (rank.get(e[0], len(ORDER)), -e[7]))
    return found


HEADER = f"  {'net':<16} {'exported':<12}  {'short: net / push-pull':<22}  {'long: net / push-pull':<27}  {'recall':<15}  notes"


def describe(e):
    """One table row (columns as HEADER): name, export time, the last validation's error
    over train.steps steps and over the long horizon (frames), each next to push-pull."""
    layout, name, kind, path, net, why, m, exported, training = e
    when = time.strftime("%d.%m. %H:%M", time.localtime(exported))
    short = long = recall = ""
    if m:
        val, pp, frames, long_val, long_pp, rec = m
        if rec:
            recall = f"{rec[0]:.4f} / {rec[1]:.4f}"
        short = f"{val:.4f} / {pp:.4f}"
        if frames:
            long = f"{long_val:.4f} / {long_pp:.4f} ({frames} fr)"
    notes = ["training" if training else "", f"can't run: {why}" if why else "", "pinned" if kind == "pin" else ""]
    return f"  {name:<16} {when:<12}  {short:<22}  {long:<27}  {recall:<15}  {', '.join(n for n in notes if n)}"


def rows():
    """Display rows: (text, entry or None for headers, "newest" for the first)."""
    out = [("follow the newest export of any run", "newest")]
    layout = None
    for e in entries():
        if e[0] != layout:
            layout = e[0]
            out.append(("", None))
            out.append((layout, None))
            out.append((HEADER, None))
        out.append((describe(e), e))
    return out


def pick(result: Path):
    import curses

    table = rows()
    selectable = [i for i, (_, e) in enumerate(table) if e == "newest" or (e and not e[5])]

    def ui(screen):
        curses.curs_set(0)
        curses.use_default_colors()
        at = 0
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            screen.addnstr(0, 0, "sunk-city: which net?  (arrows/j/k, Enter: start, q: quit)", width - 1, curses.A_BOLD)
            current = selectable[at]
            top = max(0, current - (height - 4))
            for y, (text, e) in enumerate(table[top:top + height - 2]):
                index = top + y
                style = curses.A_NORMAL
                if e is None and text == HEADER:
                    style = curses.A_DIM
                elif e is None and text:
                    style = curses.A_BOLD | curses.A_UNDERLINE
                elif e not in (None, "newest") and e[5]:
                    style = curses.A_DIM
                if index == current:
                    style |= curses.A_REVERSE
                screen.addnstr(y + 2, 0, text, width - 1, style)
            key = screen.getch()
            if key in (curses.KEY_UP, ord("k")):
                at = max(0, at - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                at = min(len(selectable) - 1, at + 1)
            elif key in (curses.KEY_ENTER, 10, 13):
                return table[current][1]
            elif key in (ord("q"), 27):
                return None

    choice = curses.wrapper(ui)
    if choice is None:
        return 1
    if choice == "newest":
        result.write_text("newest")
    else:
        result.write_text(f"{choice[2]} {choice[3]}")
    return 0


def main():
    args = sys.argv[1:]
    if args[:1] == ["pick"] and len(args) == 2:
        return pick(Path(args[1]))
    if args[:1] == ["runnable"] and len(args) == 2:
        why = problem(Path(args[1]))
        if why:
            print(why)
            return 1
        return 0
    if args:
        sys.exit(__doc__)
    for text, _ in rows()[1:]:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
