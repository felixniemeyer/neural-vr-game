#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Resolve: normalise the splat accumulators once per pixel and composite the
// rendered scene (skyscraper G-buffer), so the raw render is never shown. The net
// then reads this resolved state, one texel per tap.
//
// Holes (nothing splatted here) get inverse depth 0 = "no data"; the net fills them
// from its taps. Scene composite where there is a building: colour with alpha
// u.mixing.x, depth and velocity with u.mixing.z, the building's identity into the
// hidden state with u.mixing.y. Where there is no building nothing changes.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 outA; // rgb, certainty
layout(location = 1) out vec4 outB; // world velocity, building (0 or 1)
layout(location = 2) out vec4 outC; // h0, h1, h2, inverse depth (0 = no data)

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a0 = accFetch(pc.target, 0, p);
    vec4 a1 = accFetch(pc.target, 1, p);
    vec4 a2 = accFetch(pc.target, 2, p);
    float w = a0.a;
    bool hole = w < 0.01 || any(isnan(a0)) || any(isnan(a1)) || any(isnan(a2));
    vec3 rgb = hole ? vec3(0.0) : a0.rgb / w;
    vec3 velocity = hole ? vec3(0.0) : a1.xyz / w;
    float certainty = hole ? 0.0 : clamp(a1.w, 0.0, 1.0); // partial coverage lowers it
    vec4 c = hole ? vec4(0.0) : a2 / w;

    vec4 g = sceneFetch(pc.target, p);
    float building = g.w > 0.0 ? 1.0 : 0.0;
    float alpha = building * u.mixing.x;
    float depthAlpha = building * u.mixing.z;
    float identity = building * u.mixing.y;
    vec3 palette = 0.6 + 0.4 * cos(6.28318 * (3.0 * g.w + vec3(0.0, 0.33, 0.67)));
    rgb = mix(rgb, palette * g.x, alpha);
    velocity = mix(velocity, vec3(0.0, g.z, 0.0), depthAlpha);
    certainty = mix(certainty, 1.0, depthAlpha);
    vec3 id = hash3(vec3(g.w * 997.0, 1.0, 2.0)) * 2.0 - 1.0;
    c.xyz = mix(c.xyz, id, identity);
    c.w = building > 0.0 && c.w <= 0.0 ? g.y : mix(c.w, g.y, depthAlpha);

    outA = vec4(rgb, certainty);
    outB = vec4(velocity, building);
    outC = c;
}
