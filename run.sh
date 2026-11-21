#!/usr/bin/env bash
# Build and run against a local Monado (start monado-service first).
#   EYE_SIZE=256 (default)   VIEW=rgb|certainty|depth|velocity|hidden|hidden2|scene   MOVE=1
#   HIDDEN=3|7   OVERSCAN=1.15   CONTROLS=wss://localhost:8080   SNAPSHOT=file.ppm   VALIDATE=1
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
native/build.sh "$root/build/native" >/dev/null
# Stop a running instance gracefully (it ends its OpenXR session), so Monado keeps running.
if pkill -INT -x sunk-city; then
    for _ in $(seq 50); do pgrep -x sunk-city >/dev/null || break; sleep 0.1; done
    # Monado tears the old session down asynchronously; connecting again right away
    # can deadlock the new client, so give it a moment.
    sleep 1.5
fi
export XR_RUNTIME_JSON=${XR_RUNTIME_JSON:-/usr/share/openxr/1/openxr_monado.json}
if [[ ${VALIDATE:-0} == 1 ]]; then export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation; fi
exec "$root/build/native/sunk-city" "$@"
