#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Resolve: normalise the splat accumulators once per pixel and composite the
// rendered scene (skyscraper G-buffer), so the raw render is never shown. Same math
// as train/sunk/model.py splat(): value = acc / w, holes (w < 0.01) all 0, certainty
// = clamp(sum certainty * w) (partial coverage lowers it).
//
// Scene composite where there is a building (the runtime analogue of training's
// "observe" blend): colour with alpha u.mixing.x, depth, velocity and certainty with
// u.mixing.z, the building's identity into the hidden state with u.mixing.y.
//
// Output B.w is the coverage (inverse depth > 0), so that the mip chain of A, B, C
// is the hole-aware pyramid the net reads; certainty goes to its own image.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 outA; // rgb, inverse depth (0 = no data)
layout(location = 1) out vec4 outB; // world velocity, coverage (0 or 1)
layout(location = 2) out vec4 outC; // h0..h3
layout(location = 3) out float outCertainty;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a0 = accFetch(pc.target, 0, p);
    vec4 a1 = accFetch(pc.target, 1, p);
    vec4 a2 = accFetch(pc.target, 2, p);
    float w = accFetch(pc.target, 3, p).x;
    bool hole = w < 0.01 || any(isnan(a0)) || any(isnan(a1)) || any(isnan(a2));
    vec4 a = hole ? vec4(0.0) : a0 / w;
    vec3 velocity = hole ? vec3(0.0) : a1.xyz / w;
    float certainty = hole ? 0.0 : clamp(a1.w, 0.0, 1.0);
    vec4 h = hole ? vec4(0.0) : a2 / w;

    vec4 g = sceneFetch(pc.target, p);
    float building = g.w > 0.0 ? 1.0 : 0.0;
    float alpha = building * u.mixing.x;
    float depthAlpha = building * u.mixing.z;
    float identity = building * u.mixing.y;
    vec3 palette = 0.6 + 0.4 * cos(6.28318 * (3.0 * g.w + vec3(0.0, 0.33, 0.67)));
    a.rgb = mix(a.rgb, palette * g.x, alpha);
    a.w = building > 0.0 && a.w <= 0.0 ? g.y : mix(a.w, g.y, depthAlpha);
    velocity = mix(velocity, vec3(0.0, g.z, 0.0), depthAlpha);
    certainty = mix(certainty, 1.0, depthAlpha);
    vec4 id = vec4(hash3(vec3(g.w * 997.0, 1.0, 2.0)), hash(vec3(g.w * 997.0, 3.0, 4.0))) * 2.0 - 1.0;
    h = mix(h, id, identity);

    float covered = a.w > 0.0 ? 1.0 : 0.0;
    // Keep holes exactly 0 (the pyramid relies on it).
    outA = a * covered;
    outB = vec4(velocity * covered, covered);
    outC = h * covered;
    outCertainty = certainty * covered;
}
