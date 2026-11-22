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

NET=nice-net native/build-frame.sh    # → build/frame/sunk-city/ (+ .tar.gz); NET packs a trained net
```

To run it on the headset, upload `build/frame/sunk-city/` with the SteamOS Devkit Client:
- start command `./start.sh`
- runtime **Steam Linux Runtime 4.0 ARM64**

The log is at `~/sunk-city.log` on the headset; it prints the GPU time every 120 frames
(the net's time on frames where it runs, and fill-only frames) and, where the driver offers
them, each shader's statistics (`stats ...`: registers, instructions) at start and reload.
For a command-line alternative to the Devkit Client there is
[sdev](https://github.com/neguse/sdev) (unofficial). This is untested on a real Frame so far.

## Windows PC VR (e.g. Oculus Rift), untested

```sh
rustup target add x86_64-pc-windows-gnu
NET=nice-net native/build-windows.sh   # → build/windows/sunk-city/ (+ .zip), NET packs a trained net
```

On the PC: put `openxr_loader.dll` (x64, from a [Khronos OpenXR SDK release](https://github.com/KhronosGroup/OpenXR-SDK-Source/releases))
next to `sunk-city.exe`, set the Meta/Oculus app as the active OpenXR runtime and run
`start.bat`; the log goes to `sunk-city.log`. Needs a GPU with Vulkan and `shaderFloat16`.

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
| `DUMP`, `DUMP_FRAME`, `DUMP_TAU` | unset, 60, 1 | Write eye 0's net input and output once, one update of `DUMP_TAU` ticks (for `train/parity.py`) |
| `ZIP_ANYWHERE` | 1 | 1: the trigger also zips when the ray misses the blocks (toward a point 40 m along it); 0: only toward blocks |
| `OVERSCAN` | 1.15    | Field-of-view margin for the compositor's reprojection |
| `VALIDATE` | unset   | 1 = Vulkan validation layer |

Controllers: point at a block and hold the trigger to zip toward it (it buzzes when it
hits); release to glide to a stop. Both hands work at once. Bindings are suggested for the
simple, Index and Touch profiles (Monado's qwerty device emulates simple controllers).

Live controls (av-controls panel `sunk-city`, in tabs: net, scene, city, motion, display). Without a broker these defaults apply:

| Fader      | Default | Meaning |
|------------|---------|---------|
| `color`, `depth` | 0.01, 0.05 | How much of the rendered scene's colour, and depth (with velocity and certainty), goes into the state per frame at 90 Hz (applied per second, so other frame rates look the same). Only where the state has data; fading out with distance |
| `sky` | 150 | Distance (m) of the sky: where no block or ground is rendered, depth is blended toward it like `depth` (no colour), so the net's sky doesn't come close; 0 = off |
| `focus` | 0 | Looking uncovers detail: toward the view centre the scene blends in up to 1 + `focus` times faster (gaussian, ~a quarter of the view), so detail gathers where you look and stays; 0 = off |
| `wild` | 0 | Trained → randomly morphing weights (the distortion knob) |
| `timescale`, `netfps` | 1, 24 | How fast the net's world runs (1 = real time, as in the footage); net updates per second. Between updates frames only reproject and fill holes; each update covers the (scaled) time since the last one |
| `blocks`   | 24      | Blocks per side around the player |
| `cell`, `height` | 4, 7 | Grid cell size and maximum block height (m). Blocks grow out of the ground (from 0 at the field's edge, `blocks`/2 cells away) as you approach, turning from the ground colour to their own, and sink as you leave |
| `auto` | off | Autopilot: wander through the city at about 0.9 of the block height (± 0.9 m), looking around (also `FADERS=auto=1`); `speed` its speed (m/s, default 3) |
| `pull`, `drag` | 25, 2.5 | Zip: acceleration toward the anchored block (m/s²) and velocity damping (1/s); top speed ≈ pull / drag |
| `swing`, `reel` | off, 2 | Swing instead of zip: the trigger holds a rope as long as the distance to the anchor, reeling in at `reel` m/s (down to 1.5 m); pulling the controller back (away from the anchor) pulls you in at twice that; gravity on, so you swing like a pendulum and fly on when you let go (lower `drag` for longer flights) |
| `body` | 0.3 | The player's collision sphere around the head (radius, m) |
| `brake` | 1.5 | Braking distance (m) to the net's own geometry: each frame a 32 px wide mip of both eyes' depth is read back as points, and motion toward a point within `body` + `brake` fades out (0 = off) |
| `exposure` | off | Off: the state's colour is shown 1:1. On: automatic per-channel exposure with the faders below |
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
    5 mip levels, layers as in `net.json`, in float16 (needs `shaderFloat16`; weights in a
    uniform buffer, packed by `rule::pack`)
  - the fullscreen passes are compiled per eye (`TARGET`), so no per-texel eye switches
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
SIZE=256 ./frames.sh                           # crop + scale to data/frames/256/native/ (FPS=10: .../10fps/)
uv run new.py tiny                             # runs/tiny/config.toml: all settings
uv run prepare.py runs/tiny                    # depth, flow, camera motion -> data/prepared/
uv run view.py data/prepared/<name>/chunk_000.npz   # colour | depth | flow | object motion
uv run train.py runs/tiny                      # previews in runs/tiny/preview/, net in runs/tiny/net/ (every version in nets/<iteration>/)
WEIGHTS=train/runs/tiny/net ./dev.sh           # (from the repo root)
```

Each project folder holds its `config.toml` (data, net and training settings, e.g.
`uv run new.py big --net.hidden 32,16 --train.turn 0.5`); projects with the same
`[data]` settings share their preprocessing.

Training settings worth knowing (`[train]`):
- `strides`: video frames per net step, drawn per sample (e.g. `0.25,0.5,1,2`); the net sees
  its step size (tau ticks, `net.rate` per second), so the app can update it at any rate
- `horizon`: video frames a sample follows its clip, continuing from the net's own state
- `steps`: net steps per iteration that gradients flow through (longer = slower per iteration, fights slow drift)
- `corridor`: weight of a loss that only acts when a prediction's mean luminance leaves the real frame's ±0.05 or its contrast falls below 0.8× (against drifting dark or fading to fog)
- `dropout`: up to this fraction of each step's resolved state is knocked out in ~8 px blobs, which the net learns to regenerate (counted as holes in the loss)
- `init`: start from an exported net (e.g. `../stereo24/nets/001250`) with a fresh optimiser
- `stereo`, `baseline`: a second eye up to `baseline` m to the right, with the app's
  cross-eye splat and a loss on the eyes' disagreement

To work on the av-controls Rust receiver locally, point cargo at your checkout in
`native/.cargo/config.toml` (git-ignored):

```toml
[patch."https://github.com/avonx2/av-controls"]
av-controls = { path = "/path/to/av-controls/rust" }
```
