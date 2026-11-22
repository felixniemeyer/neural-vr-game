#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Previous state of the source eye: A = (rgb, inverse depth), B = (world velocity,
// certainty), C = (h0, h1, h2, h3).
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) flat out vec4 vA;
layout(location = 1) flat out vec4 vB;
layout(location = 2) flat out vec4 vC;
layout(location = 3) flat out vec2 vCenter;

// Each previous pixel becomes a 2x2 point sprite covering every pixel centre within
// 1 px of its projected position: the support of the 2x2 tent.

void main() {
    vec2 size = u.info.xy;
    ivec2 px = ivec2(gl_VertexIndex % int(size.x), gl_VertexIndex / int(size.x));
    vA = prevFetch(SOURCE, 0, px);
    vB = prevFetch(SOURCE, 1, px);
    vC = prevFetch(SOURCE, 2, px);
    vCenter = vec2(0.0);
    gl_PointSize = 2.0;
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); // outside: dropped
    if (vA.w <= 1e-5) return; // no depth yet
    // The other eye's state mostly lands where our own already is; splat only half of
    // it (checkerboard, alternating per frame): half the blending cost, and the own
    // eye dominates where both cover.
    if (SOURCE != TARGET && ((px.x + px.y + int(u.info.w)) & 1) == 1) return;

    vec4 p = u.prevInvViewProj[SOURCE] * vec4(ndcFromPixel(vec2(px), size), 0.0, 1.0);
    vec3 eye = u.prevEye[SOURCE].xyz;
    vec3 world = eye + normalize(p.xyz / p.w - eye) / vA.w;
    world += vB.xyz * u.step.x; // the point's own motion; camera motion comes on top
    vec4 clip = u.viewProj[TARGET] * vec4(world, 1.0);
    if (clip.w <= 1e-4) return;

    vec2 center = pixelFromClip(clip, size);
    gl_Position = vec4((center + 0.5) / size * 2.0 - 1.0, 0.0, 1.0);
    vA.w = 1.0 / max(length(world - u.eye[TARGET].xyz), 0.05); // depth as seen from the target eye
    vCenter = center;
}
