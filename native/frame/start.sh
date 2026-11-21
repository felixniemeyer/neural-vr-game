#!/bin/sh
# Steam Frame launcher (Steam Linux Runtime 4.0 ARM64). Logs to ~/twisted-crystals.log.
#   EYE_SIZE: per-eye render resolution (the headset upscales). The log prints the
#   GPU time per frame; at 90 Hz it must stay below ~11 ms.
set -eu
cd "$(dirname "$0")"
export EYE_SIZE="${EYE_SIZE:-384}"
export MOVE="${MOVE:-1}"
exec ./twisted-crystals > "$HOME/twisted-crystals.log" 2>&1
