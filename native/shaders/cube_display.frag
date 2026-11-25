#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"
#include "cube.glsl"

// cube-v1 display of one eye (TARGET): each pixel's ray looks up the head-centred sphere.
// The store is seen from the head (u.head), the eye sits beside it: a few fixed-point
// steps find where the eye's ray meets the stored depth (unknown depth: far away).
// pc.source: 1 = read the state this frame's update wrote (next), 0 = the latest from
// before (prev; no update this frame). Modes as display.frag.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;
layout(location = 0) out vec4 color;

vec4 fetch(int image, ivec2 p) {
    if (pc.source == 1) {
        if (image == 0) return texelFetch(cubeNextA, p, 0);
        if (image == 1) return texelFetch(cubeNextB, p, 0);
        return texelFetch(cubeNextC, p, 0);
    }
    if (image == 0) return texelFetch(cubePrevA, p, 0);
    if (image == 1) return texelFetch(cubePrevB, p, 0);
    return texelFetch(cubePrevC, p, 0);
}

// Bilinear in direction d's face; texels past its edge come from the neighbour face (the
// state has no gutter).
vec4 latest(int image, vec3 d) {
    int f = faceOf(d);
    vec2 x = faceTexel(f, d);
    vec2 base = floor(x), t = x - base;
    ivec2 c = ivec2(base);
    return mix(mix(fetch(image, cubeTexel(f, c)), fetch(image, cubeTexel(f, c + ivec2(1, 0))), t.x),
               mix(fetch(image, cubeTexel(f, c + ivec2(0, 1))), fetch(image, cubeTexel(f, c + ivec2(1, 1))), t.x), t.y);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 q = u.invViewProj[TARGET] * vec4(ndcFromPixel(vec2(p), u.info.xy), 0.0, 1.0);
    vec3 from = u.eye[TARGET].xyz;
    vec3 ray = normalize(q.xyz / q.w - from);
    vec3 head = u.head.xyz;
    vec3 d = ray;
    for (int i = 0; i < 3; i++) {
        float inv = latest(0, d).w;
        float reach = inv > 1e-3 ? 1.0 / inv : 1000.0;
        d = normalize(from + ray * reach - head);
    }
    vec4 a = latest(0, d);
    vec4 b = latest(1, d);
    vec4 h = latest(2, d);
    vec3 c = clamp(pow(max(a.rgb * u.tone[0].rgb + u.tone[1].rgb, 0.0), u.tone[2].rgb), 0.0, 1.0);
    if (pc.mode == 1) c = vec3(b.w);
    if (pc.mode == 2) c = vec3(clamp(a.w * u.net.z, 0.0, 1.0)); // near = white
    if (pc.mode == 3) c = b.xyz * 0.5 + 0.5;
    if (pc.mode == 4) c = h.xyz * 0.5 + 0.5;
    if (pc.mode == 6) c = sceneFetch(TARGET, 0, p).rgb;
    if (any(isnan(a)) || any(isnan(h)) || any(isinf(a)) || any(isinf(h))) c = vec3(1.0, 0.0, 1.0); // NaN: magenta
    color = vec4(c, 1.0);
}
