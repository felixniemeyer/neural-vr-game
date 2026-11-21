#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Scene G-buffer: (sun shade, inverse distance, vertical velocity, building id).
// Composited into the state by the CA pass, before the CA step, so the raw render
// is never shown.
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) in vec3 vWorld;
layout(location = 1) flat in vec3 vNormal;
layout(location = 2) flat in float vRise;
layout(location = 3) flat in float vId;
layout(location = 0) out vec4 gbuffer;

void main() {
    const vec3 sun = normalize(vec3(0.45, 0.8, 0.3));
    float shade = 0.15 + 0.85 * max(dot(vNormal, sun), 0.0);
    float inv = 1.0 / max(length(vWorld - u.eye[pc.target].xyz), 0.05);
    gbuffer = vec4(shade, inv, vWorld.y * vRise, vId);
}
