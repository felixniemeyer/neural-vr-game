#!/usr/bin/env bash
# Compile shaders to SPIR-V and build the release binary into an output directory.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$source_dir/../build/native"}
mkdir -p "$output_dir"
for shader in "$source_dir"/shaders/*.vert "$source_dir"/shaders/*.frag; do
    glslangValidator -V --target-env vulkan1.1 -o "$output_dir/$(basename "$shader").spv" "$shader" >/dev/null
done
cargo build --locked --release --manifest-path "$source_dir/Cargo.toml"
install -m 755 "$source_dir/target/release/twisted-crystals" "$output_dir/twisted-crystals"
