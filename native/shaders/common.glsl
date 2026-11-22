// Shared uniforms: both eyes' current view-projection, previous inverse
// view-projection and previous eye position (all OpenGL-style clip space, y up).
layout(set = 0, binding = 0) uniform Frame {
    mat4 viewProj[2];
    mat4 prevInvViewProj[2];
    mat4 invViewProj[2];
    vec4 eye[2];
    vec4 prevEye[2];
    vec4 info; // width, height, time, frame index
    vec4 step; // dt seconds, CA depth range near and far (m), 0
    // Tiny net (see net.frag), weights from the CPU (rule.rs):
    vec4 taps[6];    // 12 spiral tap offsets in pixels, two per vec4 (xy, zw)
    mat4 w1[56];     // layer 1: input k (0-11 taps, 12-13 self) to hidden quad q: w1[k * 4 + q]
    mat4 w2[8];      // layer 2: hidden quad q to output j (0-1): w2[j * 4 + q]
    vec4 nca;        // update rate, noise strength, alive (rate floor where certain), seed probability
    vec4 city;       // cell size m, max height m, gaussian radius m, blocks per side
    vec4 mixing;     // scene alpha for colour, for identity (hidden), for depth + velocity; noise strength
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

vec3 hash3(vec3 p) {
    return vec3(hash(p), hash(p + 19.19), hash(p + 47.47));
}
