#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_control_flow_attributes : require
#include "common.glsl"
#include "state.glsl"
#include "cube.glsl"

// cube-v1 splat reprojection, part 2 (in place of cube_pull): the accumulators of
// cube_splat into mid A..D, as sphere.py splat() normalises them:
//     values = sum(w n c v) / sum(w n c),  certainty = min(sum w, 1) * sum(w n c) / sum(w n)
// Fold: a face texel's sums are its own plus the spill that neighbour faces' points left
// in their gutters on it (tents within a texel of an edge; adjacent EAC grids meet exactly,
// so the spill texel just past a neighbour's edge is this texel). Gutter texels take the
// folded sums of the face texel that owns their direction (nearest), for the mips.
// Then the outputs as cube_pull.frag's (the observation is net.frag's, after the net).

layout(location = 0) out vec4 outA; // (rgb, inv) w
layout(location = 1) out vec4 outB; // (velocity w, w)
layout(location = 2) out vec4 outC; // h w
layout(location = 3) out vec4 outD; // (motion per tick w, w)

const float SPLAT_EPS = 1e-6; // sphere.py
const ivec2 AROUND[8] = ivec2[](ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0), ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));

void add(ivec2 p, inout vec4 a, inout vec4 b, inout vec4 c, inout vec4 d, inout float w) {
    a += texelFetch(cubeAccA, p, 0);
    b += texelFetch(cubeAccB, p, 0);
    c += texelFetch(cubeAccC, p, 0);
    d += texelFetch(cubeAccD, p, 0);
    w += texelFetch(cubeAccW, p, 0).r;
}

void main() {
    int f;
    vec2 here;
    atlasTexel(gl_FragCoord.xy, f, here);
    int n = int(faceSize());
    // The face texel whose sums this texel shows.
    int owner = f;
    ivec2 c = ivec2(here);
    if (!inFace(here)) {
        vec3 dir = faceDir(f, here);
        owner = faceOf(dir);
        c = clamp(ivec2(floor(faceTexel(owner, dir) + 0.5)), ivec2(0), ivec2(n - 1));
    }
    vec4 A = vec4(0.0), B = vec4(0.0), C = vec4(0.0), D = vec4(0.0);
    float W = 0.0;
    add(ivec2(faceOrigin(owner)) + c, A, B, C, D, W);
    // Edge texels: each neighbour face (up to 3 at a corner) once, its spill texel on c.
    if (any(equal(c, ivec2(0))) || any(equal(c, ivec2(n - 1)))) {
        vec3 dir = faceDir(owner, vec2(c));
        int seen = 1 << owner;
        [[unroll]] for (int k = 0; k < 8; k++) {
            ivec2 out_ = c + AROUND[k];
            if (all(greaterThanEqual(out_, ivec2(0))) && all(lessThan(out_, ivec2(n)))) continue;
            int g = faceOf(faceDir(owner, vec2(out_)));
            if (((seen >> g) & 1) != 0 || dot(dir, FACE_FORWARD[g]) <= 0.0) continue;
            seen |= 1 << g;
            ivec2 s = ivec2(floor(faceTexel(g, dir) + 0.5));
            // Only the first gutter ring holds spill (a tent reaches one texel past the edge).
            if (any(lessThan(s, ivec2(-1))) || any(greaterThan(s, ivec2(n)))) continue;
            if (all(greaterThanEqual(s, ivec2(0))) && all(lessThan(s, ivec2(n)))) continue;
            add(ivec2(faceOrigin(g)) + s, A, B, C, D, W);
        }
    }

    float k = 1.0 / (B.w + SPLAT_EPS);
    vec3 rgb = A.rgb * k;
    float inv = A.w * k;
    vec3 velocity = B.xyz * k;
    vec4 h = C * k;
    vec3 motion = D.xyz * k;
    float w = min(W, 1.0) * B.w / (D.w + SPLAT_EPS);

    bool broken = isnan(w) || isinf(w) || any(isnan(vec4(rgb, inv))) || any(isnan(h)) || any(isnan(motion));
    if (broken) w = 0.0;
    float perTick = max(u.step.y, 1e-6);
    outA = broken ? vec4(0.0) : vec4(rgb, inv) * w;
    outB = broken ? vec4(0.0) : vec4(velocity * w, w);
    outC = broken ? vec4(0.0) : h * w;
    outD = broken ? vec4(0.0) : vec4(motion / perTick * w, w);
}
