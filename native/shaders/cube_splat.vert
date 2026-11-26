#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"
#include "cube.glsl"

// cube-v1 splat reprojection, part 1: train/sunk/sphere.py splat() on the sphere. One point
// per face texel of the previous state (6 F^2 vertices): its point moves by its velocity
// over this update (u.cube.y seconds) and by the head's move, and lands as a 2x2 tent
// (cube_splat.frag) on the face that now holds its direction, in that face's texel
// coords. A tent within a texel of the edge spills into the face's gutter; cube_fold.frag
// adds the spill to the neighbour face it belongs to. Unseen texels (inv 0) splat nothing.
// Each point weighs its certainty c and its nearness n = (inverse distance after the move,
// clamped, * NEAR_SCALE) ^ u.step.w, so nearer points win overlaps.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) flat out vec4 vA; // rgb, inverse distance after the move
layout(location = 1) flat out vec4 vB; // velocity, certainty
layout(location = 2) flat out vec4 vC; // h0..h3
layout(location = 3) flat out vec4 vD; // motion (texels moved xy, change of log distance), nearness
layout(location = 4) flat out vec2 vCenter;

const float NEAR_SCALE = 2.2360680; // sqrt(5), sphere.py

void main() {
    int n = int(faceSize());
    int f = gl_VertexIndex / (n * n);
    int i = gl_VertexIndex % (n * n);
    ivec2 x = ivec2(i % n, i / n);
    ivec2 at = ivec2(faceOrigin(f)) + x;
    vec4 a = texelFetch(cubePrevA, at, 0);
    vec4 b = texelFetch(cubePrevB, at, 0);
    vC = texelFetch(cubePrevC, at, 0);
    gl_PointSize = 2.0;
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); // outside: dropped
    vA = vec4(0.0);
    vB = vec4(0.0);
    vD = vec4(0.0);
    vCenter = vec2(0.0);
    if (a.w <= 1e-6) return; // unseen

    vec3 source = faceDir(f, vec2(x));
    vec3 q = source / a.w + b.xyz * u.cube.y - u.headMove.xyz;
    float distance = max(length(q), 0.05);
    int g = faceOf(q);
    vec2 t = faceTexel(g, q);
    // Texels moved, in the landing face's coords (the source's direction there; a face
    // across the sphere: no meaningful displacement).
    vec2 moved = dot(source, FACE_FORWARD[g]) > 0.05 ? t - faceTexel(g, source) : vec2(0.0);
    float nearness = pow(clamp(1.0 / distance, 0.01, 20.0) * NEAR_SCALE, u.step.w);

    vA = vec4(a.rgb, 1.0 / distance);
    vB = vec4(b.xyz, max(b.w, 0.0));
    vD = vec4(moved, log(distance * a.w), nearness);
    vec2 center = faceOrigin(g) + t; // atlas texel coords (centres at integers)
    vCenter = center;
    gl_Position = vec4((center + 0.5) / atlasSize() * 2.0 - 1.0, 0.0, 1.0);
}
