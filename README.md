# Sunk City

An artsy VR experiment for the Steam Frame. Every frame, the previous frame is reprojected
into the new head pose, and a tiny per-pixel neural net fills the holes and keeps the
patterns moving. The only geometry is a field of simple skyscraper blocks. They are
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
| `SNAPSHOT` | unset   | Write the left eye's colour to this `.ppm` every 300 frames |
| `OVERSCAN` | 1.15    | Field-of-view margin for the compositor's reprojection |
| `VALIDATE` | unset   | 1 = Vulkan validation layer |

Live faders (av-controls panel `sunk-city`). Without a broker these defaults apply:

| Fader      | Default | Meaning |
|------------|---------|---------|
| `color`, `depth`, `identity` | 0, 0.1, 0.1 | How strongly the rendered blocks set colour, depth (+ velocity) and the hidden state (from the block id) per frame |
| `noise`, `seeds` | 1, 1 | Strength of the (smooth) hidden-state noise, frequency of seed kicks |
| `reach`, `alive` | 16, 0.3 | Net: outermost of its 12 spiral taps (px); update rate kept on settled pixels (the world never settles) |
| `blocks`   | 24      | Blocks per side around the player |
| `cell`, `height` | 4, 7 | Grid cell size and maximum block height (m) |
| `radius`   | 20      | Gaussian falloff (m): blocks rise as you approach and sink as you leave |
| `altitude` | 7.5     | Player height (m) |
| `near`, `far` | 10, 80 | Depth range (m) of the net's own geometry; closer than `near` is pushed away, so you can't fly into it |
| `mid`, `clip`, `black`, `tint` | 0.5, 0.03, 0.05, 0.3 | Display auto-exposure, per channel: 80th percentile → `mid`, brightest `clip` fraction clips, darkest `black` fraction is 0; `tint` pulls the channels' mappings together (keeps some colour cast) |

## Layout

- `dev.sh`: desktop dev loop; `run.sh`: build and run once (start Monado yourself)
- `native/src/`: `main.rs` (OpenXR/Vulkan, frame loop), `world.rs` (passes), `rule.rs` (net weights, spiral taps)
- `native/shaders/`:
  - `scene.*` renders the blocks
  - `splat.*` reprojects (2×2 tent splat)
  - `resolve.frag` normalises the splat and blends in the blocks
  - `net.frag` is the tiny net (12 spiral taps, 16 hidden, all `mat4`)
- `native/build.sh`, `native/build-frame.sh`: desktop and Steam Frame builds
- `experiments/`: older experiments (submodule), not needed

To work on the av-controls Rust receiver locally, point cargo at your checkout in
`native/.cargo/config.toml` (git-ignored):

```toml
[patch."https://github.com/avonx2/av-controls"]
av-controls = { path = "/path/to/av-controls/rust" }
```
