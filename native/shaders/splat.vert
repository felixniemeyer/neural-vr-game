#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Previous state of the source eye:
// A = (rgb, inverse depth), B = (world velocity, certainty), C/D = hidden.
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) flat out vec4 vA;
layout(location = 1) flat out vec4 vB;
layout(location = 2) flat out vec4 vC;
layout(location = 3) flat out vec4 vD;
layout(location = 4) flat out vec2 vCenter;

// Each previous pixel becomes a quad covering every pixel centre within 1px of
// its projected position: the support of the 2x2 tent.
const vec2 corners[6] = vec2[](
    vec2(-1, -1), vec2(1, -1), vec2(1, 1),
    vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    vec2 size = u.info.xy;
    ivec2 px = ivec2(gl_InstanceIndex % int(size.x), gl_InstanceIndex / int(size.x));
    vec4 a = prevFetch(pc.source, 0, px);
    vB = prevFetch(pc.source, 1, px);
    vC = prevFetch(pc.source, 2, px);
    vD = prevFetch(pc.source, 3, px);
    vA = a;
    vCenter = vec2(0.0);
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); // degenerate: dropped
    if (a.a <= 1e-5) return; // no depth yet

    vec4 p = u.prevInvViewProj[pc.source] * vec4(ndcFromPixel(vec2(px), size), 0.0, 1.0);
    vec3 eye = u.prevEye[pc.source].xyz;
    vec3 world = eye + normalize(p.xyz / p.w - eye) / a.a;
    world += vB.xyz * u.step.x; // the point's own motion; camera motion comes on top
    vec4 clip = u.viewProj[pc.target] * vec4(world, 1.0);
    if (clip.w <= 1e-4) return;

    vec2 center = pixelFromClip(clip, size);
    vec2 corner = center + 0.5 + corners[gl_VertexIndex];
    gl_Position = vec4(corner / size * 2.0 - 1.0, 0.0, 1.0);
    vA.a = 1.0 / max(length(world - u.eye[pc.target].xyz), 0.05); // depth as seen from the target eye
    vCenter = center;
}
