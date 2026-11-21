# Twisted Crystals

An artsy VR experiment for the Steam Frame. Every frame, the previous frame is reprojected
into the new head pose; a small neural cellular automaton fills the holes and lets
patterns grow. There is no geometry: the world is the image's own memory.

Native Rust + Vulkan (ash) + OpenXR, no engine.

## Layout

- `native/`: the app
  - `src/main.rs`: OpenXR/Vulkan setup and the frame loop
  - `src/world.rs`: state images and passes (splat → CA → display)
  - `src/rule.rs`: random, slowly morphing CA weights
  - `shaders/`: GLSL; `splat.*` reprojects (2×2 tent, weight = certainty), `ca.frag` is the neural CA
  - `build.sh`: desktop build (x86_64), `build-frame.sh`: Steam Frame build (ARM64)
  - `frame/start.sh`: launcher on the headset
- `run.sh`: run on the desktop against Monado (simulated headset)
- `experiments/`: older experiments (submodule), not needed

## Build for the Steam Frame

On a Linux PC (cross-compiles to aarch64, glibc pinned to 2.31 for the Steam Linux Runtime):

```sh
git clone --recursive https://github.com/felixniemeyer/neural-vr-game.git
cd neural-vr-game

# one-time setup
rustup target add aarch64-unknown-linux-gnu
cargo install cargo-zigbuild --version 0.21.8 --locked   # newer versions need rustc >= 1.88
uv tool install ziglang                                   # or any `zig` on PATH
# glslangValidator from your distro: pacman -S glslang / apt install glslang-tools

native/build-frame.sh
```

Output: `build/frame/twisted-crystals/` (binary, `.spv` shaders, `start.sh`) and a `.tar.gz` of it.

## Run on the headset

1. Upload `build/frame/twisted-crystals/` with the **SteamOS Devkit Client**.
2. Start command: `./start.sh`, runtime: **Steam Linux Runtime 4.0 ARM64**.
3. Log: `~/twisted-crystals.log` on the headset. Every 120 frames it prints the GPU time
   per frame (both eyes).

The app loads SteamVR's OpenXR loader from `/opt/steamvr/bin/linuxarm64/libopenxr_loader.so`
(override with `OPENXR_LOADER=/path/to/libopenxr_loader.so`).

Untested on a real Frame so far, so the first run is the real test. If it fails, the log is
the first place to look.

## Settings (environment variables, set in `start.sh`)

| Variable   | Default | Meaning |
|------------|---------|---------|
| `EYE_SIZE` | 384     | Per-eye render resolution (square, the headset upscales). At 90 Hz the GPU time must stay below ~11 ms: raise it if there is headroom, lower it if not. |
| `MOVE`     | 1       | 1 = head translation moves the view, 0 = rotation only |
| `VIEW`     | rgb     | Debug view: `rgb`, `certainty`, `depth`, `velocity`, `hidden`, `hidden2` |
| `OVERSCAN` | 1.15    | Rendered FOV margin for the compositor's reprojection |
| `VALIDATE` | unset   | 1 = enable the Vulkan validation layer (if installed) |

Shaders are reloaded live when a `.spv` file changes, so uploading new `.spv` files only
(compile with `native/build.sh <dir> --shaders`) updates a running app.

## Run on the desktop (no headset)

Needs [Monado](https://monado.freedesktop.org/) (OpenXR runtime) with a simulated headset:

```sh
SIMULATED_ENABLE=1 SIMULATED_ROTATE=1 sh -c 'sleep infinity | monado-service' &
EYE_SIZE=256 ./run.sh
```

(`monado-service` needs a pollable stdin, hence the `sleep infinity |`.) `run.sh` builds
into `build/native/` and stops a running instance first.
