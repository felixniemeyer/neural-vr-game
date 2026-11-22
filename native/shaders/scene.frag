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

// Night (the day fader): the same blocks as another world: complementary, dim colours and
// warm windows on the walls, lit by a cool moon.
vec3 night(vec3 color) {
    vec3 moon = normalize(vec3(-0.4, 0.7, -0.5));
    vec3 base = (vec3(1.0) - color.bgr) * vec3(0.25, 0.3, 0.5) * (0.3 + 0.7 * max(dot(vNormal, moon), 0.0));
    if (abs(vNormal.y) > 0.5) return base;
    vec2 wall = abs(vNormal.x) > 0.5 ? vWorld.zy : vWorld.xy;
    vec2 cell = fract(wall * vec2(1.2, 0.9));
    float window = step(0.35, cell.x) * step(cell.x, 0.75) * step(0.3, cell.y) * step(cell.y, 0.8);
    float lit = step(0.45, fract(sin(dot(floor(wall * vec2(1.2, 0.9)), vec2(12.9898, 78.233))) * 43758.5453));
    return base + window * lit * vRise * vec3(1.0, 0.7, 0.35);
}

void main() {
    float phase = u.step.z;
    float angle = 6.2831853 * phase;
    vec3 sun = phase < 0.0 ? normalize(vec3(0.45, 0.8, 0.3)) : normalize(vec3(cos(angle), sin(angle), 0.35));
    float shade = 0.15 + 0.85 * max(dot(vNormal, sun), 0.0);
    float inv = 1.0 / max(length(vWorld - u.eye[pc.target].xyz), 0.05);
    vec3 day = vColor * shade;
    float daylight = phase < 0.0 ? 1.0 : smoothstep(-0.2, 0.15, sun.y);
    colorDepth = vec4(mix(night(vColor), day, daylight), inv);
    velocity = vec4(0.0, vWorld.y * vRise, 0.0, 0.0); // a block grows: its points move up
}
