# Sunk City

An artsy VR experiment for the Steam Frame. Every frame, the previous frame is reprojected
into the new head pose, and a tiny per-pixel neural net, trained on video, fills the holes
and keeps the image alive. The only geometry is a field of simple skyscraper blocks. They are
blended into the image before the net runs, so the raw render is never shown; everything
else is the image's own memory.

Native Rust + Vulkan (ash) + OpenXR, no engine.

## Quick start (desktop, no headset)

Needs Rust, Node.js, [Monado](https://monado.freedesktop.org/) and `glslangValidator`
(`pacman -S glslang` / `apt install glslang-tools`).

```sh
git clone --recursive https://github.com/felixniemeyer/sunk-city.git
cd sunk-city
./dev.sh
```

`dev.sh` starts everything, and Ctrl+C stops it all:
- an [av-controls](https://github.com/avonx2/av-controls) broker and the controller UI, both from npm
- Monado with a keyboard-steered headset and its debug UI: focus a Monado window, arrows to
  look, WASD/QE to move (`HEADSET=simulated` auto-rotates instead)
- the app

Open the printed controller URL and pick the panel **sunk-city** to get live faders.
Shader edits reload live; Rust edits rebuild and restart the app. The broker, controller
and Monado logs go to `build/dev/`.

## Steam Frame

```sh
# one-time setup
rustup target add aarch64-unknown-linux-gnu
cargo install cargo-zigbuild --version 0.21.8 --locked   # newer versions need rustc >= 1.88
uv tool install ziglang                                   # or any `zig` on PATH

native/build-frame.sh    # → build/frame/sunk-city/ (+ .tar.gz)
```

To run it on the headset, upload `build/frame/sunk-city/` with the SteamOS Devkit Client:
- start command `./start.sh`
- runtime **Steam Linux Runtime 4.0 ARM64**

The log is at `~/sunk-city.log` on the headset; it prints the GPU time every 120 frames.
For a command-line alternative to the Devkit Client there is
[sdev](https://github.com/neguse/sdev) (unofficial). This is untested on a real Frame so far.

## Settings

Environment variables (on the Frame, set them in `native/frame/start.sh`):

| Variable   | Default | Meaning |
|------------|---------|---------|
| `EYE_SIZE` | 256 (Frame: 384) | Per-eye resolution; at 90 Hz the GPU time must stay below ~11 ms |
| `MOVE`     | 0 (Frame: 1, `dev.sh`: 4) | Head translation scale: 0 rotation only, 1 real, more = giant steps |
| `VIEW`     | rgb     | Debug view: `rgb`, `certainty`, `depth`, `velocity`, `hidden`, `scene` |
| `CONTROLS` | wss://localhost:8080 (`dev.sh`: ws://localhost:8090) | av-controls broker; on the Frame, use your PC's IP |
| `WEIGHTS`  | unset   | A trained net, e.g. `train/runs/tiny/net`; unset = random weights |
| `SNAPSHOT` | unset   | Write the left eye's colour to this `.ppm` every 300 frames |
| `DUMP`, `DUMP_FRAME` | unset, 60 | Write eye 0's net input and output once (for `train/parity.py`) |
| `OVERSCAN` | 1.15    | Field-of-view margin for the compositor's reprojection |
| `VALIDATE` | unset   | 1 = Vulkan validation layer |

Controllers: point at a block and hold the trigger to zip toward it (it buzzes when it
hits); release to glide to a stop. Both hands work at once. Bindings are suggested for the
simple, Index and Touch profiles (Monado's qwerty device emulates simple controllers).

Live faders (av-controls panel `sunk-city`). Without a broker these defaults apply:

| Fader      | Default | Meaning |
|------------|---------|---------|
| `color`, `depth`, `identity` | 0.2, 0.1, 0.1 | How strongly the rendered blocks set colour, depth (+ velocity) and the hidden state (from the block id) per frame |
| `wild`, `rate` | 0, 1 | Trained → randomly morphing weights (the distortion knob); speed of the net's changes (1 = per second as in training, whose steps were 10 fps video frames) |
| `blocks`   | 24      | Blocks per side around the player |
| `cell`, `height` | 4, 7 | Grid cell size and maximum block height (m) |
| `radius`   | 20      | Gaussian falloff (m): blocks rise as you approach and sink as you leave |
| `altitude` | 7.5     | Player height (m); changing it moves you up or down |
| `pull`, `drag` | 25, 2.5 | Zip: acceleration toward the anchored block (m/s²) and velocity damping (1/s); top speed ≈ pull / drag |
| `mid`, `clip`, `black`, `tint` | 0.5, 0.03, 0.05, 0.3 | Display auto-exposure, per channel: 80th percentile → `mid`, brightest `clip` fraction clips, darkest `black` fraction is 0; `tint` pulls the channels' mappings together (keeps some colour cast) |

## Layout

- `dev.sh`: desktop dev loop; `run.sh`: build and run once (start Monado yourself)
- `native/src/`: `main.rs` (OpenXR/Vulkan, frame loop), `world.rs` (passes), `rule.rs` (net weights, spiral taps)
- `native/shaders/`:
  - `scene.*` renders the blocks
  - `splat.*` reprojects (2×2 tent splat)
  - `resolve.frag` normalises the splat and blends in the blocks; its mips are the
    hole-aware pyramid
  - `net.frag` is the trained net: holes filled from the pyramid, then triangle taps on
    5 mip levels, layers as in `net.json`
- `train/`: training on video (below)
- `native/build.sh`, `native/build-frame.sh`: desktop and Steam Frame builds
- `experiments/`: older experiments (submodule), not needed

## Training

The net is trained on video with a torch copy of the splat, resolve and net
(`train/sunk/model.py`); `train/parity.py` checks the GPU against it. Needs
[uv](https://docs.astral.sh/uv/) and ffmpeg; runs on the CPU.

```sh
cd train
ln -s /path/to/your/videos/*.mp4 videos/      # any footage
SIZE=256 ./frames.sh                           # crop + scale to data/frames/256/
uv run new.py tiny                             # runs/tiny/config.toml: all settings
uv run prepare.py runs/tiny                    # depth, flow, camera motion -> data/prepared/
uv run view.py data/prepared/<name>/chunk_000.npz   # colour | depth | flow | object motion
uv run train.py runs/tiny                      # previews in runs/tiny/preview/, net in runs/tiny/net/
WEIGHTS=train/runs/tiny/net ./dev.sh           # (from the repo root)
```

Each project folder holds its `config.toml` (data, net and training settings, e.g.
`uv run new.py big --net.hidden 32,16 --train.turn 0.5`); projects with the same
`[data]` settings share their preprocessing.

To work on the av-controls Rust receiver locally, point cargo at your checkout in
`native/.cargo/config.toml` (git-ignored):

```toml
[patch."https://github.com/avonx2/av-controls"]
av-controls = { path = "/path/to/av-controls/rust" }
```
