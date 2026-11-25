#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_control_flow_attributes : require
#include "common.glsl"
#include "state.glsl"
#include "cube.glsl"

// cube-v1 pull, part 2: train/sunk/sphere.py pull() per target texel, in its face's
// extended texel coords (cube.glsl). The search for the source x with x + m(x) = here
// (m: cube_forward.frag's motion, read from its premultiplied pyramid): coarse to fine,
// plus the motion of the 3 x 3 coarse cells around the target at levels 2.., each refined
// 3 times at level 0; where several converge the nearest wins (it occludes). Then the
// previous state, premultiplied by its weight, read bilinearly at x; weight = coverage *
// exp(-(error / SIGMA)^2); the moved point's distance and the motion input (texels moved,
// change of log distance, per tick).
//
// Then the scene blend of resolve.frag's v2 branch (the runtime's "observe"): where an eye
// sees the scene in this direction, colour, depth and velocity move toward it by min(w, 1)
// times the frame's mixing. Written premultiplied by w (cube.glsl mid A..D).

layout(location = 0) out vec4 outA; // (rgb, inv) w
layout(location = 1) out vec4 outB; // (velocity w, w)
layout(location = 2) out vec4 outC; // h w
layout(location = 3) out vec4 outD; // (motion per tick w, w)

layout(constant_id = 1) const int LEVELS = 5;
const float SIGMA = 0.5;
const float SMOOTH = 0.25; // texels: finest vs coarsest motion here, below = no candidate search

// m at x (face f's coords) from mip `level`: texels.
vec2 motionAt(int f, vec2 x, int level) {
    vec4 v = textureLod(cubeM, cubeUV(f, x, float(level)), float(level));
    vec3 moved = faceDir(f, x) + v.xyz / (v.w + CUBE_EPS);
    return faceTexel(f, moved) - x;
}

vec2 refine(int f, vec2 here, vec2 x) {
    [[unroll]] for (int i = 0; i < 3; i++) x = here - motionAt(f, x, 0);
    return x;
}

// Bilinear read of the previous state at x, premultiplied by the weight: sums of A w,
// (velocity w, w), C w.
void readPrev(int f, vec2 x, out vec4 A, out vec4 B, out vec4 C) {
    vec2 base = floor(x), t = x - base;
    A = vec4(0.0);
    B = vec4(0.0);
    C = vec4(0.0);
    [[unroll]] for (int j = 0; j < 2; j++) {
        [[unroll]] for (int i = 0; i < 2; i++) {
            float bilinear = (i == 0 ? 1.0 - t.x : t.x) * (j == 0 ? 1.0 - t.y : t.y);
            ivec2 p = cubeTexel(f, ivec2(base) + ivec2(i, j));
            vec4 b = texelFetch(cubePrevB, p, 0);
            float w = max(b.w, 0.0) * bilinear;
            A += texelFetch(cubePrevA, p, 0) * w;
            B += vec4(b.xyz * w, w);
            C += texelFetch(cubePrevC, p, 0) * w;
        }
    }
}

// (error in texels, inverse distance) of a candidate source x.
vec2 judge(int f, vec2 here, vec2 x) {
    vec2 base = floor(x), t = x - base;
    float sum = 0.0, weight = 0.0;
    [[unroll]] for (int j = 0; j < 2; j++) {
        [[unroll]] for (int i = 0; i < 2; i++) {
            float bilinear = (i == 0 ? 1.0 - t.x : t.x) * (j == 0 ? 1.0 - t.y : t.y);
            ivec2 p = cubeTexel(f, ivec2(base) + ivec2(i, j));
            float w = max(texelFetch(cubePrevB, p, 0).w, 0.0) * bilinear;
            sum += texelFetch(cubePrevA, p, 0).w * w;
            weight += w;
        }
    }
    return vec2(length(x + motionAt(f, x, 0) - here), sum / (weight + CUBE_EPS));
}

void main() {
    int f;
    vec2 here;
    atlasTexel(gl_FragCoord.xy, f, here);

    vec2 x = here;
    [[unroll]] for (int level = LEVELS - 1; level >= 0; level--) x = here - motionAt(f, x, level);
    vec2 best = refine(f, here, x);
    vec2 bestJudged = judge(f, here, best);
    // The candidates only matter where the motion field is not smooth (a near object's
    // edge): where the plain search converged and the finest and coarsest motion here
    // agree, they all converge to the same source (training runs them everywhere).
    bool steady = bestJudged.x < SIGMA && distance(motionAt(f, here, 0), motionAt(f, here, LEVELS - 1)) < SMOOTH;
    for (int level = 2; level < (steady ? 2 : LEVELS); level++) {
        float cell = exp2(float(level));
        for (int oy = -1; oy <= 1; oy++) {
            for (int ox = -1; ox <= 1; ox++) {
                vec2 candidate = refine(f, here, here - motionAt(f, here + vec2(ox, oy) * cell, level));
                vec2 judged = judge(f, here, candidate);
                bool ok = judged.x < SIGMA, bestOk = bestJudged.x < SIGMA;
                bool take = (ok && bestOk && judged.y > bestJudged.y * 1.05) || (ok && !bestOk)
                         || (!ok && !bestOk && judged.x < bestJudged.x);
                if (take) {
                    best = candidate;
                    bestJudged = judged;
                }
            }
        }
    }

    vec4 A, B, C;
    readPrev(f, best, A, B, C);
    float got = B.w;
    float k = 1.0 / (got + CUBE_EPS);
    vec3 rgb = A.rgb * k;
    float inv = A.w * k;
    vec3 velocity = B.xyz * k;
    vec4 h = C * k;
    float w = got * exp(-(bestJudged.x / SIGMA) * (bestJudged.x / SIGMA));

    // The moved point, from the source values: distance and velocity.
    vec3 q = faceDir(f, best) / max(inv, 1e-6) + velocity * u.cube.y - u.headMove.xyz;
    float distance = max(length(q), 0.05);
    float seen = inv > 1e-6 ? 1.0 : 0.0;
    float logChange = log(distance * max(inv, 1e-6)) * seen;
    inv = seen / distance;
    vec2 moved = here - best;

    // Scene blend (resolve.frag, v2 branch).
    vec4 g;
    vec3 sceneVelocity;
    float fade, sky;
    vec2 centred;
    if (sceneDir(faceDir(f, here), inv > 1e-6 ? 1.0 / inv : 20.0, g, sceneVelocity, fade, sky, centred)) {
        float focus = 1.0 + u.mixing.w * exp(-dot(centred, centred) / 0.18);
        float present = min(w, 1.0);
        float alpha = present * min(fade * u.mixing.x * (1.0 - sky) * focus, 1.0);
        float depthAlpha = present * min(fade * u.mixing.z * focus, 1.0);
        rgb = mix(rgb, g.rgb, alpha);
        inv = mix(inv, g.w, depthAlpha);
        velocity = mix(velocity, sceneVelocity, depthAlpha);
    }

    bool broken = isnan(w) || isinf(w) || any(isnan(vec4(rgb, inv))) || any(isnan(h)) || isnan(logChange);
    if (broken) w = 0.0;
    float perTick = max(u.step.y, 1e-6);
    outA = broken ? vec4(0.0) : vec4(rgb, inv) * w;
    outB = broken ? vec4(0.0) : vec4(velocity * w, w);
    outC = broken ? vec4(0.0) : h * w;
    outD = broken ? vec4(0.0) : vec4(vec3(moved, logChange) / perTick * w, w);
}
