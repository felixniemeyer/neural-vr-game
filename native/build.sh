#!/usr/bin/env bash
# Compile shaders to SPIR-V and build the release binary into an output directory.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$source_dir/../build/native"}
[[ $output_dir == --shaders ]] && set -- "$source_dir/../build/native" --shaders && output_dir=$1
mkdir -p "$output_dir"
for shader in "$source_dir"/shaders/*.vert "$source_dir"/shaders/*.frag; do
    if ! log=$(glslangValidator -V --target-env vulkan1.1 -o "$output_dir/$(basename "$shader").spv" "$shader" 2>&1); then
        echo "$log" >&2
        exit 1
    fi
done
# --shaders: only recompile shaders (a running app reloads them live).
if [[ ${2:-} == --shaders ]]; then exit 0; fi
# Run from native/ so a local .cargo/config.toml (av-controls override) applies.
(cd "$source_dir" && cargo build --release)
install -m 755 "$source_dir/target/release/sunk-city" "$output_dir/sunk-city"
