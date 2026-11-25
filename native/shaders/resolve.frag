#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Resolve: normalise the splat accumulators once per pixel and composite the
// rendered scene (skyscraper G-buffer), so the raw render is never shown. Same math
// as train/sunk/model.py splat(): value = acc / w, holes (w < 0.01) all 0, certainty
// = clamp(sum certainty * w) (partial coverage lowers it).
//
// Scene blend where the scene has geometry and the state has data (the runtime
// analogue of training's "observe" blend): colour with u.mixing.x, depth, velocity and
// certainty with u.mixing.z, both fading to 0 at GROUND_REACH field radii. In holes
// nothing happens here: net.frag fills them by push-pull (colour too), then gives them
// the scene's depth and velocity. Where the scene has no geometry it is sky: depth at
// the `sky` distance, no colour, so the image's own guesses there don't come close.
//
// Output B.w is the coverage (inverse depth > 0), so that the mip chain of A, B, C
// is the hole-aware pyramid the net reads; certainty goes to D.x.
//
// LAYOUT 2 (reproject-v2, train/sunk/model.py resolve and forward_v2): no hole
// threshold: values = sums / (w + 1e-3); the scene blends in by min(w, 1); everything
// is written premultiplied by the raw weight w (B.w = D.w = w), so the mips are the
// weighted pyramid; D = (motion w, w) with motion = screenMotion().

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 outA; // rgb, inverse depth (0 = no data)
layout(location = 1) out vec4 outB; // world velocity, coverage (0 or 1)
layout(location = 2) out vec4 outC; // h0..h3
layout(location = 3) out vec4 outD; // v1: (certainty, 0, 0, 0); v2: (motion, 1) * w

// Where a world point moving at `velocity` (m/s) goes on screen without the camera's
// rotation (rotation only shifts the image): (dx, dy) px and the change of log
// distance, per net tick (u.netRun.z ticks per second). As model.py screen_motion():
// the camera's translation counts, its orientation stays this frame's.
vec3 screenMotion(vec3 world, vec3 velocity) {
    vec3 eye = u.eye[TARGET].xyz;
    vec3 eyeVelocity = u.step.x > 0.0 ? (eye - u.prevEye[TARGET].xyz) / u.step.x : vec3(0.0);
    vec3 moved = world + (velocity - eyeVelocity) / max(u.netRun.z, 1e-3);
    vec4 from = u.viewProj[TARGET] * vec4(world, 1.0);
    vec4 to = u.viewProj[TARGET] * vec4(moved, 1.0);
    if (from.w <= 1e-3 || to.w <= 1e-3) return vec3(0.0);
    vec2 size = u.info.xy;
    return vec3(pixelFromClip(to, size) - pixelFromClip(from, size),
                log(max(length(moved - eye), 1e-6) / max(length(world - eye), 1e-6)));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a0 = accFetch(TARGET, 0, p);
    vec4 a1 = accFetch(TARGET, 1, p);
    vec4 a2 = accFetch(TARGET, 2, p);
    float w = accFetch(TARGET, 3, p).x;
    bool broken = isnan(w) || any(isnan(a0)) || any(isnan(a1)) || any(isnan(a2));
    bool hole = w < 0.01 || broken;
    float k = LAYOUT == 2 ? 1.0 / (w + 1e-3) : 1.0 / max(w, 0.01);
    if (LAYOUT == 2) {
        if (broken) w = 0.0;
        hole = broken;
    }
    vec4 a = hole ? vec4(0.0) : a0 * k;
    vec3 velocity = hole ? vec3(0.0) : a1.xyz * k;
    float certainty = hole ? 0.0 : clamp(a1.w, 0.0, 1.0);
    vec4 h = hole ? vec4(0.0) : a2 * k;
    // How much is here to blend the scene into: v1 0 or 1, v2 min(w, 1).
    float present = hole ? 0.0 : (LAYOUT == 2 ? min(w, 1.0) : 1.0);

    // Holes stay holes here (push-pull fills them in net.frag, which then takes the
    // scene's depth): seeding them here would mark black pixels as covered.
    float fade;
    vec4 g = sceneAt(TARGET, p, fade);
    vec3 motion = sceneFetch(TARGET, 1, p).xyz;
    float sky = sceneFetch(TARGET, 0, p).w > 0.0 ? 0.0 : 1.0;
    // Focus: where you look (the gaze where tracked, else the view centre), the scene blends in up to 1 + focus times
    // as fast (a gaussian of about a quarter of the view), so detail gathers there.
    vec2 centred = (vec2(p) + 0.5) / u.info.xy * 2.0 - 1.0 - (TARGET == 0 ? u.gaze.xy : u.gaze.zw);
    float focus = 1.0 + u.mixing.w * exp(-dot(centred, centred) / 0.18);
    float alpha = present * min(fade * u.mixing.x * (1.0 - sky) * focus, 1.0);
    float depthAlpha = present * min(fade * u.mixing.z * focus, 1.0);
    a.rgb = mix(a.rgb, g.rgb, alpha);
    a.w = mix(a.w, g.w, depthAlpha);
    velocity = mix(velocity, motion, depthAlpha);
    certainty = mix(certainty, 1.0, depthAlpha);

    if (LAYOUT == 2) {
        vec3 screen = vec3(0.0);
        if (a.w > 1e-6) {
            vec4 q = u.invViewProj[TARGET] * vec4(ndcFromPixel(vec2(p), u.info.xy), 0.0, 1.0);
            vec3 eye = u.eye[TARGET].xyz;
            screen = screenMotion(eye + normalize(q.xyz / q.w - eye) / a.w, velocity);
        }
        outA = a * w;
        outB = vec4(velocity * w, w);
        outC = h * w;
        outD = vec4(screen * w, w);
        return;
    }
    float covered = a.w > 0.0 ? 1.0 : 0.0;
    // Keep holes exactly 0 (the pyramid relies on it).
    outA = a * covered;
    outB = vec4(velocity * covered, covered);
    outC = h * covered;
    outD = vec4(certainty * covered, 0.0, 0.0, 0.0);
}
