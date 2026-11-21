// Shared uniforms: both eyes' current view-projection, previous inverse
// view-projection and previous eye position (all OpenGL-style clip space, y up).
layout(set = 0, binding = 0) uniform Frame {
    mat4 viewProj[2];
    mat4 prevInvViewProj[2];
    vec4 eye[2];
    vec4 prevEye[2];
    vec4 info; // width, height, time, frame index
    vec4 step; // dt seconds, 0, 0, 0
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
