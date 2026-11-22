#!/usr/bin/env bash
# One command for the desktop dev loop. Starts:
#   - an av-controls broker and the controller UI (from npm via npx, needs Node.js)
#   - Monado with a headset you steer with the keyboard (qwerty driver + debug GUI:
#     focus a Monado window, arrows to look, WASD/QE to move)
#   - the app, connected to the broker
# Shader edits are compiled and reloaded live; Rust edits rebuild and restart the app.
# Monado keeps running (and its windows stay put) unless it stalls on the reconnect;
# then it is restarted too. Ctrl+C stops everything.
#
#   BROKER_PORT=8090  CONTROLLER_PORT=4173  HEADSET=qwerty|simulated (auto-rotating)
#   MONADO_WORKSPACE (sway): where Monado's windows go, default 5
#   plus the app's variables: EYE_SIZE=256 VIEW=... MOVE=4 ...
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
out=$root/build/native
logs=$root/build/dev
mkdir -p "$logs"
broker_port=${BROKER_PORT:-8090}
controller_port=${CONTROLLER_PORT:-4173}
export EYE_SIZE=${EYE_SIZE:-1024} #384
export MOVE=${MOVE:-4} # qwerty WASD is slow at city scale
headset=${HEADSET:-qwerty}
# sway: put Monado's windows on this workspace: the compositor (Wayland app_id
# "openxr") and the debug GUI (XWayland class "monado-service"; its title comes late).
if [[ -n ${SWAYSOCK:-} ]] && command -v swaymsg >/dev/null; then
    workspace=${MONADO_WORKSPACE:-5}
    for criteria in 'app_id="openxr"' 'class="monado-service"' 'title="^Monado"'; do
        swaymsg "for_window [$criteria] move container to workspace $workspace" >/dev/null
    done
fi
export CONTROLS=${CONTROLS:-ws://localhost:$broker_port}
export XR_RUNTIME_JSON=${XR_RUNTIME_JSON:-/usr/share/openxr/1/openxr_monado.json}
if [[ ${VALIDATE:-0} == 1 ]]; then export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation; fi
for tool in npx monado-service glslangValidator cargo; do
    command -v "$tool" >/dev/null || { echo "dev.sh: '$tool' not found (see README)" >&2; exit 1; }
done

groups=() # process groups started here (each service runs in its own session)
app=""
monado=""
listening() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
service() { # name, command...: start in its own process group, log to build/dev/<name>.log
    local name=$1; shift
    setsid "$@" >"$logs/$name.log" 2>&1 &
    groups+=($!)
}
stop_app() {
    [[ -n $app ]] && kill -0 "$app" 2>/dev/null || { app=""; return; }
    kill -INT "$app" # the app ends its OpenXR session cleanly
    for _ in $(seq 50); do kill -0 "$app" 2>/dev/null || break; sleep 0.1; done
    kill -9 "$app" 2>/dev/null || true
    app=""
}
stop_monado() { # Monado can ignore SIGTERM: ask with SIGINT, then force
    [[ -n $monado ]] && kill -- -"$monado" 2>/dev/null || true
    pkill -INT -x monado-service 2>/dev/null || return 0
    for _ in $(seq 20); do pgrep -x monado-service >/dev/null || return 0; sleep 0.1; done
    pkill -9 -x monado-service 2>/dev/null || true
    sleep 0.5
}
start_monado() {
    stop_monado
    if [[ $headset == simulated ]]; then
        export SIMULATED_ENABLE=1 SIMULATED_ROTATE=1
    else
        # Debug GUI: focus it to steer (arrows/WASD/QE; mouse look may not work in sway).
        export QWERTY_ENABLE=1 XRT_DEBUG_GUI=1
    fi
    setsid sh -c 'sleep infinity | monado-service' >"$logs/monado.log" 2>&1 &
    monado=$!
    sleep 2
}
launch() { # the app prints "... frames, GPU ..." every 120 frames once it renders
    "$out/sunk-city" "$out" > >(tee "$logs/app.log") 2>&1 &
    app=$!
    for _ in $(seq 100); do
        grep -q "frames, GPU" "$logs/app.log" 2>/dev/null && return 0
        kill -0 "$app" 2>/dev/null || return 1
        sleep 0.1
    done
    return 1
}
start_app() {
    stop_app
    pgrep -x monado-service >/dev/null || start_monado
    launch && return
    echo "dev.sh: no frames, restarting Monado"
    stop_app
    start_monado
    launch || echo "dev.sh: the app does not render, see build/dev/app.log"
}
cleanup() {
    trap - EXIT INT TERM
    stop_app
    for group in "${groups[@]}"; do kill -- -"$group" 2>/dev/null || true; done
    stop_monado
    echo "dev.sh: stopped"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

if listening "$broker_port"; then
    echo "dev.sh: broker port $broker_port in use, assuming an av-controls broker is running"
else
    WS_PORT=$broker_port service broker npx -y @av-controls/server
fi
if listening "$controller_port"; then
    echo "dev.sh: controller port $controller_port in use, assuming the controller is running"
else
    service controller npx -y -p @av-controls/controller av-controls-controller --port "$controller_port"
fi

"$root/native/build.sh" "$out"
start_app
echo "dev.sh: controller UI: http://localhost:$controller_port/?net=$CONTROLS  (panel: sunk-city)"
echo "dev.sh: logs of broker, controller and Monado in build/dev/; Ctrl+C stops everything"

# Watch the sources (polling, no extra tools needed).
stamp=$logs/.stamp
touch "$stamp"
while true; do
    sleep 1
    touch "$stamp.new"
    changed=$(find "$root/native/shaders" "$root/native/src" "$root/native/Cargo.toml" -newer "$stamp" -type f)
    mv "$stamp.new" "$stamp"
    if grep -q "/src/\|Cargo.toml" <<<"$changed"; then
        echo "dev.sh: Rust changed, rebuilding"
        if "$root/native/build.sh" "$out"; then start_app; else echo "dev.sh: build failed, keeping the running app"; fi
    elif [[ -n $changed ]]; then
        echo "dev.sh: shaders changed, compiling (the app reloads them)"
        "$root/native/build.sh" "$out" --shaders || echo "dev.sh: shader compile failed, keeping the old ones"
    fi
    if [[ -n $app ]] && ! kill -0 "$app" 2>/dev/null; then
        echo "dev.sh: the app exited; save a Rust file to restart it"
        app=""
    fi
done
