#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// A particle: a small lit sphere (round point), its colour and world velocity into the
// scene G-buffer like the blocks (scene.frag).
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vVelocity;
layout(location = 0) out vec4 colorDepth;
layout(location = 1) out vec4 velocity;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r = dot(d, d);
    if (r > 1.0) discard;
    float shade = 0.35 + 0.65 * sqrt(1.0 - r);
    // Fast particles glow a little: the motion is the picture.
    float glow = clamp(length(vVelocity) * 0.05, 0.0, 0.6);
    vec3 color = mix(vColor * shade, vec3(1.0, 0.95, 0.85), glow);
    colorDepth = vec4(color, 1.0 / max(length(vWorld - u.eye[pc.target].xyz), 0.05));
    velocity = vec4(vVelocity, 0.0);
}
