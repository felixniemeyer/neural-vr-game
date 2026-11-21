#!/usr/bin/env bash
# Cross-compile for the Steam Frame (Linux ARM64) and package it.
# Needs: rustup target aarch64-unknown-linux-gnu, cargo-zigbuild, zig (uv tool install ziglang).
# glibc is pinned to 2.31 so the binary runs in the Steam Linux Runtime.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$source_dir/../build/frame/twisted-crystals"}
zig=$(ls "$HOME"/.local/share/uv/tools/ziglang/lib/python*/site-packages/ziglang/zig 2>/dev/null | head -1 || true)
export CARGO_ZIGBUILD_ZIG_PATH=${CARGO_ZIGBUILD_ZIG_PATH:-${zig:-zig}}
target=aarch64-unknown-linux-gnu
rm -rf "$output_dir"
mkdir -p "$output_dir"
for shader in "$source_dir"/shaders/*.vert "$source_dir"/shaders/*.frag; do
    glslangValidator -V --target-env vulkan1.1 -o "$output_dir/$(basename "$shader").spv" "$shader" >/dev/null
done
cargo zigbuild --locked --release --target "$target.2.31" --manifest-path "$source_dir/Cargo.toml"
install -m 755 "$source_dir/target/$target/release/twisted-crystals" "$output_dir/twisted-crystals"
install -m 755 "$source_dir/frame/start.sh" "$output_dir/start.sh"
tar -C "$(dirname "$output_dir")" -czf "$output_dir.tar.gz" "$(basename "$output_dir")"
echo "Packaged $output_dir.tar.gz"
