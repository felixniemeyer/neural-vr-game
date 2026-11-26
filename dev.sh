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
#   WEIGHTS=dir pins a net (e.g. WEIGHTS=nice-net, WEIGHTS=random for random weights).
#   Unset, on a terminal: a picker (train/zoo.py) lists the nets by layout (cube-v1,
#   reproject-v2, reproject-v1) with their metrics. Picking a run follows its exports (train/runs/<run>/net,
#   rewritten every train.preview iterations); a loose export (nice-net) is pinned; "newest"
#   follows the newest export of any run (also without a terminal). p in this terminal
#   opens the picker again. Nets switch live: dev.sh writes the net's path to
#   build/dev/net (NET_FILE) and the running app swaps it in, no restart.
#   OUTPUT=stereo|dome|equirect (a cube-v1 net; the picker asks after the net): stereo is
#   VR through Monado; dome (domemaster) and equirect run without VR (DESKTOP=1,
#   native/src/desktop.rs) in the app's own window (Tab switches dome/equirect), the
#   camera flown from the controller's camera tab; new nets restart the app there.
#   DOME_FOV, PANORAMA_WIDTH pass through.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
out=$root/build/native
logs=$root/build/dev
mkdir -p "$logs"
broker_port=${BROKER_PORT:-8090}
controller_port=${CONTROLLER_PORT:-4173}
export EYE_SIZE=${EYE_SIZE:-512} #384
export MOVE=${MOVE:-4} # qwerty WASD is slow at city scale
headset=${HEADSET:-qwerty}
output=${OUTPUT:-stereo}
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

export NET_FILE=$logs/net # the running app swaps in the net named here (live)
follow=0
runs=("$root"/train/runs/*) # the runs followed
use_net() { # the app's net: WEIGHTS for (re)starts, NET_FILE for the running app
    export WEIGHTS=$1
    echo "$1" >"$NET_FILE"
}
choose() { # the picker (train/zoo.py); returns 1 if nothing was picked
    local file choice
    file=$(mktemp)
    if ! python3 "$root/train/zoo.py" pick "$file"; then rm -f "$file"; return 1; fi
    choice=$(sed -n 1p "$file")
    output=$(sed -n 2p "$file")
    output=${output:-stereo}
    rm -f "$file"
    net_stamp="" # announce (and send) the followed net even if it is the same
    case $choice in
        newest) follow=1; runs=("$root"/train/runs/*) ;;
        follow\ *) follow=1; runs=("${choice#follow }") ;;
        pin\ *) follow=0; use_net "${choice#pin }"; echo "dev.sh: net ${WEIGHTS#"$root"/} (pinned)" ;;
    esac
}
if [[ -z ${WEIGHTS:-} ]]; then
    follow=1
    if [[ -t 0 && -t 1 ]]; then
        choose || { echo "dev.sh: nothing picked"; exit 0; }
    fi
elif [[ $WEIGHTS == random ]]; then unset WEIGHTS
else use_net "$WEIGHTS"
fi
latest_net() { # newest net/ (by weights.bin) of the followed runs the app can run, empty if none
    local file
    for file in $(ls -t "${runs[@]/%//net/weights.bin}" 2>/dev/null); do
        python3 "$root/train/zoo.py" runnable "${file%/*}" >/dev/null || continue
        dirname "$file"; return
    done
}
pick_net() { # the newest export of the followed runs to the app (use_net); returns 1 if unchanged
    [[ $follow == 1 ]] || return 1
    local net; net=$(latest_net)
    [[ -n $net ]] || return 1
    local stamp; stamp="$net $(stat -c %Y "$net/weights.bin")"
    [[ $stamp != "${net_stamp:-}" ]] || return 1
    net_stamp=$stamp
    use_net "$net"
    echo "dev.sh: net ${net#"$root"/}, exported $(date -d @"${stamp##* }" +%H:%M)"
}
pick_net || true

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
desktop() { [[ $output != stereo ]]; }
launch() { # the app prints "... frames ... GPU ..." every 90-120 frames once it renders
    local mode=()
    if desktop; then
        mode=(env DESKTOP=1)
        [[ $output == dome ]] && mode+=(DOME=1)
    fi
    "${mode[@]}" "$out/sunk-city" "$out" > >(tee "$logs/app.log") 2>&1 &
    app=$!
    local tries=100 # desktop: the first line after 90 simulated frames (slow while training)
    desktop && tries=900
    for _ in $(seq $tries); do
        grep -q "frames.*GPU" "$logs/app.log" 2>/dev/null && return 0
        kill -0 "$app" 2>/dev/null || return 1
        sleep 0.1
    done
    return 1
}
start_app() {
    stop_app
    if desktop; then
        echo "dev.sh: $output output without VR (DESKTOP=1); fly the camera in the controller's camera tab"
        launch && return
        echo "dev.sh: the desktop app does not render, see build/dev/app.log"
        return
    fi
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
[[ -t 0 ]] && echo "dev.sh: p picks another net (switched live)"
while true; do
    if [[ -t 0 ]]; then
        key=""
        read -rsn1 -t 1 key || true
        if [[ $key == p ]]; then
            before=$output
            if choose; then
                pick_net || true
                # desktop outputs read the net at start only; switching output restarts too
                if desktop || [[ $output != "$before" ]]; then start_app; fi
            else
                echo "dev.sh: nothing picked, keeping the net"
            fi
        fi
    else
        sleep 1
    fi
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
    # a newer export goes to the running app (NET_FILE), no restart; desktop: a restart
    if pick_net && desktop; then start_app; fi
    if [[ -n $app ]] && ! kill -0 "$app" 2>/dev/null; then
        echo "dev.sh: the app exited; save a Rust file to restart it"
        app=""
    fi
done
