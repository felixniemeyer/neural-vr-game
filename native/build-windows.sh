#!/usr/bin/env bash
# Cross-build for Windows x64 (PC VR, e.g. Oculus Rift) into build/windows/sunk-city/ (+ .zip).
# Needs: rustup target x86_64-pc-windows-gnu, cargo-zigbuild, zig (uv tool install ziglang).
# NET=<export folder> (e.g. ../nice-net) packs a trained net (start.bat uses it).
# Then copy openxr_loader.dll (x64, from a Khronos OpenXR SDK release) into the folder.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$source_dir/../build/windows/sunk-city"}
zig=$(ls "$HOME"/.local/share/uv/tools/ziglang/lib/python*/site-packages/ziglang/zig 2>/dev/null | head -1 || true)
export CARGO_ZIGBUILD_ZIG_PATH=${CARGO_ZIGBUILD_ZIG_PATH:-${zig:-zig}}
target=x86_64-pc-windows-gnu
rm -rf "$output_dir"
mkdir -p "$output_dir"
for shader in "$source_dir"/shaders/*.vert "$source_dir"/shaders/*.frag; do
    glslangValidator -V --target-env vulkan1.1 -o "$output_dir/$(basename "$shader").spv" "$shader" >/dev/null
done
(cd "$source_dir" && cargo zigbuild --release --target "$target")
install -m 755 "$source_dir/target/$target/release/sunk-city.exe" "$output_dir/sunk-city.exe"
install -m 644 "$source_dir/windows/start.bat" "$output_dir/start.bat"
if [[ -n ${NET:-} ]]; then mkdir -p "$output_dir/net" && cp "$NET"/net.json "$NET"/weights.bin "$output_dir/net/"; fi
(cd "$(dirname "$output_dir")" && rm -f "$(basename "$output_dir").zip" && python3 -m zipfile -c "$(basename "$output_dir").zip" "$(basename "$output_dir")")
echo "Packaged $output_dir.zip (add openxr_loader.dll before running)"
