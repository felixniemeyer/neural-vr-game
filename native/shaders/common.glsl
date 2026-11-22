// Shared uniforms: both eyes' current view-projection, previous inverse
// view-projection and previous eye position (all OpenGL-style clip space, y up).
layout(set = 0, binding = 0) uniform Frame {
    mat4 viewProj[2];
    mat4 prevInvViewProj[2];
    mat4 invViewProj[2];
    vec4 eye[2];
    vec4 prevEye[2];
    vec4 info; // width, height, time, frame index
    vec4 step; // dt seconds, tau (net ticks this frame, 0 = fill only), 0, 0
    // Net (see net.frag; layout from the trained net.json, rule.rs):
    vec4 net;        // taps, layers, near (m; depth scale the net was trained with), max change
    vec4 netRun;     // tau input (1, 0 for nets trained before it), 0, 0, 0
    vec4 netLayers[4]; // per layer: inputs, outputs, offset into weights (vec4s), 0
    vec4 netTaps[32];  // per tap: mip level, dx, dy (px, y down), 0
    vec4 city;       // cell size m, max height m, 0, blocks per side
    vec4 mixing;     // this frame's blend toward the scene: colour, sky inverse depth, depth + velocity + certainty, focus
    vec4 player;     // ground position x, z now (xy) and in the previous frame (zw)
    vec4 tone[3];    // display exposure per channel: pow(max(rgb * tone[0] + tone[1], 0), tone[2])
} u;

// Pixel convention: integer pixel centres, origin top-left, y down.
vec2 pixelFromClip(vec4 clip, vec2 size) {
    vec2 ndc = clip.xy / clip.w;
    ndc.y = -ndc.y;
    return (ndc * 0.5 + 0.5) * size - 0.5;
}

vec2 ndcFromPixel(vec2 pixel, vec2 size) {
    vec2 ndc = (pixel + 0.5) / size * 2.0 - 1.0;
    ndc.y = -ndc.y;
    return ndc;
}

float hash(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

// The block field: blocks stand on cells within this distance of the player (m); the
// ground plate and the scene's influence reach to GROUND_REACH times it.
const float GROUND_REACH = 8.0;
// The rise's gaussian width, as a fraction of the field radius (city.rs SPREAD).
const float SPREAD = 0.45;

float fieldRadius() {
    return max(u.city.w * 0.5 - 1.0, 1.0) * u.city.x;
}

vec3 hash3(vec3 p) {
    return vec3(hash(p), hash(p + 19.19), hash(p + 47.47));
}
