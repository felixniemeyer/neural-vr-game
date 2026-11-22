#!/usr/bin/env bash
# Cross-compile for the Steam Frame (Linux ARM64) and package it.
# Needs: rustup target aarch64-unknown-linux-gnu, cargo-zigbuild, zig (uv tool install ziglang).
# glibc is pinned to 2.31 so the binary runs in the Steam Linux Runtime.
# NET=<export folder> (e.g. ../nice-net) packs a trained net (start.sh uses it).
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$source_dir/../build/frame/sunk-city"}
zig=$(ls "$HOME"/.local/share/uv/tools/ziglang/lib/python*/site-packages/ziglang/zig 2>/dev/null | head -1 || true)
export CARGO_ZIGBUILD_ZIG_PATH=${CARGO_ZIGBUILD_ZIG_PATH:-${zig:-zig}}
target=aarch64-unknown-linux-gnu
rm -rf "$output_dir"
mkdir -p "$output_dir"
for shader in "$source_dir"/shaders/*.vert "$source_dir"/shaders/*.frag; do
    glslangValidator -V --target-env vulkan1.1 -o "$output_dir/$(basename "$shader").spv" "$shader" >/dev/null
done
(cd "$source_dir" && cargo zigbuild --release --target "$target.2.31")
install -m 755 "$source_dir/target/$target/release/sunk-city" "$output_dir/sunk-city"
install -m 755 "$source_dir/frame/start.sh" "$output_dir/start.sh"
if [[ -n ${NET:-} ]]; then mkdir -p "$output_dir/net" && cp "$NET"/net.json "$NET"/weights.bin "$output_dir/net/"; fi
tar -C "$(dirname "$output_dir")" -czf "$output_dir.tar.gz" "$(basename "$output_dir")"
echo "Packaged $output_dir.tar.gz"
