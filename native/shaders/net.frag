#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// The trained per-pixel net, the same math as train/sunk/model.py Net.forward:
//
//   fill     holes take the finest pyramid level that has data here (push-pull)
//   inputs   self (rgb, inv * near), (h); per tap (mip level L, offset dx, dy):
//            (rgb, (inv - inv_self) * near), (h), bilinear from the hole-aware
//            pyramid, 0 where it is empty; extra (certainty, valid, 1, tau)
//   layers   1x1 "convolutions" from the weights buffer, relu between them
//   update   d = max_change * tau * tanh(out) (an Euler step of tau ticks):
//            rgb += d.rgb, inv *= exp(d.w), h += d_h; certainty decays by 0.965 per tick
//
// tau = u.step.y: the net runs at its own rate (main.rs); on the frames in between
// tau = 0 and only the holes are filled.
//
// Weights (rule.rs): per layer W [outputs][inputs] row-major, then the bias, as vec4s.
// The layout is fixed per net and comes in as specialization constants, so every loop
// below has a constant trip count (unrolled, accumulators in registers, weight reads
// at constant addresses). Layer 1 is accumulated while its inputs are read.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 stateA; // rgb, inverse depth
layout(location = 1) out vec4 stateB; // world velocity, certainty
layout(location = 2) out vec4 stateC; // h0..h3

layout(constant_id = 0) const int TAPS = 15;
layout(constant_id = 1) const int LEVELS = 5;
layout(constant_id = 2) const int LAYERS = 2;  // 2..4
layout(constant_id = 3) const int UNITS0 = 16; // outputs of each layer; the last is 8
layout(constant_id = 4) const int UNITS1 = 8;
layout(constant_id = 5) const int UNITS2 = 8;
layout(constant_id = 6) const int UNITS3 = 8;

const int IN0 = 4 * (2 * (1 + TAPS) + 1);
const int OFF0 = 0;
const int OFF1 = OFF0 + (IN0 * UNITS0 + UNITS0) / 4;
const int OFF2 = OFF1 + (UNITS0 * UNITS1 + UNITS1) / 4;
const int OFF3 = OFF2 + (UNITS1 * UNITS2 + UNITS2) / 4;
const int MAX_QUADS = 16;

vec4 hidden[MAX_QUADS];

// Layer 1: add input vec4 number k to every unit.
void feed(int k, vec4 x) {
    for (int q = 0; q < UNITS0 / 4; q++) {
        int row = OFF0 + 4 * q * (IN0 / 4) + k;
        hidden[q] += vec4(dot(weights[row], x), dot(weights[row + IN0 / 4], x),
                          dot(weights[row + 2 * (IN0 / 4)], x), dot(weights[row + 3 * (IN0 / 4)], x));
    }
}

// A further layer: inputs relu(hidden[0 .. IN/4]), outputs into hidden[0 .. OUT/4].
void dense(const int offset, const int inputs, const int outputs) {
    vec4 x[MAX_QUADS];
    for (int q = 0; q < inputs / 4; q++) x[q] = max(hidden[q], 0.0);
    for (int q = 0; q < outputs / 4; q++) {
        vec4 sum = weights[offset + outputs * (inputs / 4) + q];
        for (int k = 0; k < inputs / 4; k++) {
            int row = offset + 4 * q * (inputs / 4) + k;
            sum += vec4(dot(weights[row], x[k]), dot(weights[row + inputs / 4], x[k]),
                        dot(weights[row + 2 * (inputs / 4)], x[k]), dot(weights[row + 3 * (inputs / 4)], x[k]));
        }
        hidden[q] = sum;
    }
}

struct Tap {
    vec4 a; // rgb, inv
    vec4 b; // velocity, coverage
    vec4 c; // h
    float covered;
};

// Bilinear read of mip `level` at `uv`, normalised by its coverage.
Tap read(vec2 uv, float level) {
    vec4 a = midLod(pc.target, 0, uv, level);
    vec4 b = midLod(pc.target, 1, uv, level);
    vec4 c = midLod(pc.target, 2, uv, level);
    float k = 1.0 / max(b.w, 1e-4);
    return Tap(a * k, vec4(b.xyz * k, b.w), c * k, b.w > 1e-3 ? 1.0 : 0.0);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 size = u.info.xy;
    float near = u.net.z;

    vec4 A = midFetch(pc.target, 0, p);
    vec4 B = midFetch(pc.target, 1, p);
    vec4 C = midFetch(pc.target, 2, p);
    float valid = A.w > 0.0 ? 1.0 : 0.0;
    float certainty = certFetch(pc.target, p) * valid;

    // Push-pull hole fill from the pyramid.
    float need = 1.0 - valid;
    for (int level = 1; level < LEVELS; level++) {
        Tap t = read(gl_FragCoord.xy / size, float(level));
        float take = need * t.covered;
        A = mix(A, t.a, take);
        B = mix(B, t.b, take);
        C = mix(C, t.c, take);
        need *= 1.0 - t.covered;
    }
    // Former holes take the scene's depth and velocity (colour stays push-pull's).
    if (valid == 0.0) {
        float fade;
        vec4 g = sceneAt(pc.target, p, fade);
        if (fade > 0.0) {
            A.w = g.w;
            B.xyz = sceneFetch(pc.target, 1, p).xyz;
        }
    }
    float inv = A.w;
    float tau = u.step.y;
    if (tau == 0.0) {
        stateA = vec4(A.rgb, inv);
        stateB = vec4(B.xyz, certainty);
        stateC = C;
        return;
    }

    // Layer 1, fed input by input.
    for (int q = 0; q < UNITS0 / 4; q++) hidden[q] = weights[OFF0 + UNITS0 * (IN0 / 4) + q];
    feed(0, vec4(A.rgb, inv * near));
    feed(1, C);
    for (int t = 0; t < TAPS; t++) {
        vec4 tap = u.netTaps[t];
        Tap v = read((gl_FragCoord.xy + tap.yz) / size, tap.x);
        feed(2 + 2 * t, vec4(v.a.rgb, (v.a.w - inv) * near) * v.covered);
        feed(3 + 2 * t, v.c * v.covered);
    }
    feed(2 + 2 * TAPS, vec4(certainty, valid, 1.0, tau * u.netRun.x));

    // Further layers (relu between).
    if (LAYERS > 1) dense(OFF1, UNITS0, UNITS1);
    if (LAYERS > 2) dense(OFF2, UNITS1, UNITS2);
    if (LAYERS > 3) dense(OFF3, UNITS2, UNITS3);

    // Residual update (outputs: d rgb, d log inv, d h), tau ticks.
    float change = u.net.w * tau;
    vec4 d0 = change * tanh(hidden[0]);
    vec4 d1 = change * tanh(hidden[1]);
    stateA = vec4(clamp(A.rgb + d0.rgb, 0.0, 1.5), inv * exp(d0.w));
    const float rest = 0.03 / 0.035;
    stateB = vec4(B.xyz, rest + (certainty - rest) * pow(0.965, tau));
    stateC = clamp(C + d1, -4.0, 4.0);
}
