#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Scene: (sun-shaded colour, inverse distance) and (world velocity, 0). Blended into
// the state by resolve, gently, so the net's image dominates.
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) in vec3 vWorld;
layout(location = 1) flat in vec3 vNormal;
layout(location = 2) flat in float vRise;
layout(location = 3) flat in vec3 vColor;
layout(location = 0) out vec4 colorDepth;
layout(location = 1) out vec4 velocity;

void main() {
    const vec3 sun = normalize(vec3(0.45, 0.8, 0.3));
    float shade = 0.15 + 0.85 * max(dot(vNormal, sun), 0.0);
    float inv = 1.0 / max(length(vWorld - u.eye[pc.target].xyz), 0.05);
    colorDepth = vec4(vColor * shade, inv);
    velocity = vec4(0.0, vWorld.y * vRise, 0.0, 0.0); // a block grows: its points move up
}
