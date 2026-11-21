#!/usr/bin/env bash
# Build and run against a local Monado (start monado-service first).
#   EYE_SIZE=128 (default)   VIEW=rgb|certainty|depth|hidden   VALIDATE=1
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
native/build.sh "$root/build/native" >/dev/null
export XR_RUNTIME_JSON=${XR_RUNTIME_JSON:-/usr/share/openxr/1/openxr_monado.json}
if [[ ${VALIDATE:-0} == 1 ]]; then export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation; fi
exec "$root/build/native/twisted-crystals" "$@"
