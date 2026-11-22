#!/usr/bin/env bash
# Render a trained net in the real app: ./snapshot.sh runs/<name> [label=FADERS ...]
# Starts a simulated, auto-rotating Monado (restarts one that is running), runs the app
# 20 s per setting with WEIGHTS=runs/<name>/net, saves runs/<name>/app/<label>.png
# (the left eye, exposed like the display). Default settings: as trained, slower
# update (rate 0.11 ~ 10 fps training / 90 Hz), and a bit wild.
set -euo pipefail
cd "$(dirname "$0")"
run=$(cd "$1" && pwd); shift
settings=("$@")
[[ ${#settings[@]} -gt 0 ]] || settings=("trained=" "slow=rate=0.11" "wild=wild=0.3")
root=$(cd .. && pwd)
out=$run/app && mkdir -p "$out"

# A fresh Monado per run: it stalls when an app reconnects.
monado=
restart_monado() {
    [[ -n $monado ]] && kill -- -"$monado" 2>/dev/null || true
    for p in $(pgrep -x monado-service); do kill -INT "$p"; done
    for _ in $(seq 20); do pgrep -x monado-service >/dev/null || break; sleep 0.2; done
    for p in $(pgrep -x monado-service); do kill -9 "$p"; done
    SIMULATED_ENABLE=1 SIMULATED_ROTATE=1 setsid sh -c 'sleep infinity | monado-service' >"$out/monado.log" 2>&1 &
    monado=$!
    sleep 3
}
trap '[[ -n $monado ]] && kill -- -$monado 2>/dev/null; pkill -INT -x monado-service || true' EXIT

for setting in "${settings[@]}"; do
    label=${setting%%=*} faders=${setting#*=}
    restart_monado
    (cd "$root/build/native" && XR_RUNTIME_JSON=/usr/share/openxr/1/openxr_monado.json CONTROLS=ws://localhost:1 \
        EYE_SIZE=${EYE_SIZE:-256} WEIGHTS="$run/net" FADERS="$faders" SNAPSHOT="$out/$label.ppm" \
        timeout -s INT 20 ./sunk-city >"$out/$label.log" 2>&1) || true
    uv run python -c "from PIL import Image; Image.open('$out/$label.ppm').resize((512, 512), Image.NEAREST).save('$out/$label.png')"
    rm -f "$out/$label.ppm"
    echo "$out/$label.png ($(grep -o 'GPU [0-9.]* ms' "$out/$label.log" | tail -1))"
done
