#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Skyscraper blocks: one instance per cell of a (blocks x blocks) grid around the
// player, 5 faces (no bottom) = 30 vertices each. Each block's id and base height
// come from its grid cell; the height is multiplied by a gaussian of the distance
// to the player's ground position, so blocks rise out of the ground as you come
// closer and sink back as you leave. That growth is the block's velocity.
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec3 vWorld;
layout(location = 1) flat out vec3 vNormal;
layout(location = 2) flat out float vRise; // (d height / dt) / height
layout(location = 3) flat out float vId;   // in (0, 1]; 0 means no building

const vec3 normals[5] = vec3[](
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 0, 1), vec3(0, 0, -1), vec3(0, 1, 0));
const vec3 corners[20] = vec3[](
    vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1), vec3(1, 0, 1),
    vec3(0, 0, 1), vec3(0, 1, 1), vec3(0, 1, 0), vec3(0, 0, 0),
    vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1), vec3(0, 0, 1),
    vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0), vec3(1, 0, 0),
    vec3(0, 1, 0), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0));
const int quad[6] = int[](0, 1, 2, 0, 2, 3);

float gaussian(vec2 center, vec2 player) {
    vec2 d = center - player;
    float radius = u.city.z;
    return exp(-dot(d, d) / (2.0 * radius * radius));
}

void main() {
    int n = int(u.city.w);
    float cellSize = u.city.x;
    ivec2 cell = ivec2(floor(u.player.xy / cellSize))
               + ivec2(gl_InstanceIndex % n, gl_InstanceIndex / n) - n / 2;
    vec2 center = (vec2(cell) + 0.5) * cellSize;
    float base = u.city.y * (0.25 + 0.75 * hash(vec3(cell, 3.7)));
    float height = base * gaussian(center, u.player.xy);
    float previous = base * gaussian(center, u.player.zw);
    float inset = cellSize * (0.12 + 0.12 * hash(vec3(cell, 9.1))); // half street width

    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // degenerate: dropped
    vNormal = vec3(0.0);
    vRise = 0.0;
    vId = 0.0;
    vWorld = vec3(0.0);
    if (height < 0.5) return;

    int face = gl_VertexIndex / 6;
    vec3 corner = corners[face * 4 + quad[gl_VertexIndex % 6]];
    vec2 low = vec2(cell) * cellSize + inset;
    vec2 span = vec2(cellSize - 2.0 * inset);
    vWorld = vec3(low.x + corner.x * span.x, corner.y * height, low.y + corner.z * span.y);
    vNormal = normals[face];
    vRise = (height - previous) / max(u.step.x, 1e-3) / height;
    vId = 0.02 + 0.98 * hash(vec3(cell, 5.3));
    gl_Position = u.viewProj[pc.target] * vec4(vWorld, 1.0);
    gl_Position.y = -gl_Position.y; // state pixels are y down (see pixelFromClip)
}
