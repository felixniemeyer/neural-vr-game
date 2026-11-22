#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Skyscraper blocks: one instance per cell of a (blocks x blocks) grid around the
// player, 5 faces (no bottom) = 30 vertices each, plus one last instance: the ground
// plate (6 vertices). Each block's base height comes from its grid cell; it is scaled
// by its rise, a gaussian of the distance to the player shifted so it is exactly 0 at
// the field radius: blocks grow out of the ground as you come closer (no popping) and
// sink back as you leave. Growing, they turn from the ground colour to their own. The
// growth is their velocity. city.rs mirrors this (raycast, collision).
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec3 vWorld;
layout(location = 1) flat out vec3 vNormal;
layout(location = 2) flat out float vRise; // (d height / dt) / height
layout(location = 3) flat out vec3 vColor;

const vec3 GROUND = vec3(0.30, 0.28, 0.26);
const vec3 normals[5] = vec3[](
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 0, 1), vec3(0, 0, -1), vec3(0, 1, 0));
const vec3 corners[20] = vec3[](
    vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1), vec3(1, 0, 1),
    vec3(0, 0, 1), vec3(0, 1, 1), vec3(0, 1, 0), vec3(0, 0, 0),
    vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1), vec3(0, 0, 1),
    vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0), vec3(1, 0, 0),
    vec3(0, 1, 0), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0));
const int quad[6] = int[](0, 1, 2, 0, 2, 3);

// 0 at and beyond the field radius, 1 at the player.
float rise(vec2 center, vec2 player) {
    vec2 d = center - player;
    float field = fieldRadius(), radius = SPREAD * field;
    float edge = exp(-field * field / (2.0 * radius * radius));
    float g = exp(-dot(d, d) / (2.0 * radius * radius));
    return max(g - edge, 0.0) / (1.0 - edge);
}

void main() {
    int n = int(u.city.w);
    float cellSize = u.city.x;
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // degenerate: dropped
    vNormal = vec3(0.0, 1.0, 0.0);
    vRise = 0.0;
    vColor = GROUND;
    vWorld = vec3(0.0);

    if (gl_InstanceIndex == n * n) {
        // The ground plate around the player.
        if (gl_VertexIndex >= 6) return;
        vec3 corner = corners[16 + quad[gl_VertexIndex]]; // the top face's corners
        float reach = GROUND_REACH * fieldRadius();
        vWorld = vec3(u.player.x + (corner.x * 2.0 - 1.0) * reach, 0.0, u.player.y + (corner.z * 2.0 - 1.0) * reach);
    } else {
        ivec2 cell = ivec2(floor(u.player.xy / cellSize))
                   + ivec2(gl_InstanceIndex % n, gl_InstanceIndex / n) - n / 2;
        vec2 center = (vec2(cell) + 0.5) * cellSize;
        float base = u.city.y * (0.25 + 0.75 * hash(vec3(cell, 3.7)));
        float grown = rise(center, u.player.xy);
        float height = base * grown;
        float previous = base * rise(center, u.player.zw);
        if (height < 1e-3) return;
        float inset = cellSize * (0.12 + 0.12 * hash(vec3(cell, 9.1))); // half street width
        int face = gl_VertexIndex / 6;
        vec3 corner = corners[face * 4 + quad[gl_VertexIndex % 6]];
        vec2 low = vec2(cell) * cellSize + inset;
        vec2 span = vec2(cellSize - 2.0 * inset);
        vWorld = vec3(low.x + corner.x * span.x, corner.y * height, low.y + corner.z * span.y);
        vNormal = normals[face];
        vRise = (height - previous) / max(u.step.x, 1e-3) / height;
        float id = hash(vec3(cell, 5.3));
        vec3 own = 0.6 + 0.4 * cos(6.28318 * (3.0 * id + vec3(0.0, 0.33, 0.67)));
        vColor = mix(GROUND, own, grown);
    }
    gl_Position = u.viewProj[pc.target] * vec4(vWorld, 1.0);
    gl_Position.y = -gl_Position.y; // state pixels are y down (see pixelFromClip)
}
