#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// The trained per-pixel net, the same math as train/sunk/model.py Net.forward:
//
//   fill     holes take the finest pyramid level that has data here (push-pull)
//   inputs   self (rgb, inv * near), (h); per tap (mip level L, offset dx, dy):
//            (rgb, (inv - inv_self) * near), (h), bilinear from the hole-aware
//            pyramid, 0 where it is empty; extra (certainty, valid, 1, 0)
//   layers   1x1 "convolutions" from the weights buffer, relu between them
//   update   d = max_change * tanh(out) * rate:
//            rgb += d.rgb, inv *= exp(d.w), h += d_h; certainty decays toward 0.03 / 1
//
// Weights (rule.rs): per layer W [outputs][inputs] row-major, then the bias; as vec4s
// from u.netLayers[l].z. Layer 1 is accumulated while its inputs are read (no input
// array). Hidden layers up to 64 units, up to 4 layers.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 stateA; // rgb, inverse depth
layout(location = 1) out vec4 stateB; // world velocity, certainty
layout(location = 2) out vec4 stateC; // h0..h3

const int MAX_QUADS = 16;
vec4 hidden[MAX_QUADS];
int units, inputQuads, offset;

// Layer 1: add input vec4 number k to every unit.
void feed(int k, vec4 x) {
    for (int j = 0; j < units; j++) {
        hidden[j >> 2][j & 3] += dot(weights[offset + j * inputQuads + k], x);
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
    int taps = int(u.net.x), layers = int(u.net.y);

    vec4 A = midFetch(pc.target, 0, p);
    vec4 B = midFetch(pc.target, 1, p);
    vec4 C = midFetch(pc.target, 2, p);
    float valid = A.w > 0.0 ? 1.0 : 0.0;
    float certainty = certFetch(pc.target, p) * valid;

    // Push-pull hole fill from the pyramid.
    int levels = 1;
    for (int t = 0; t < taps; t++) levels = max(levels, int(u.netTaps[t].x) + 1);
    float need = 1.0 - valid;
    for (int level = 1; level < levels; level++) {
        Tap t = read(gl_FragCoord.xy / size, float(level));
        float take = need * t.covered;
        A = mix(A, t.a, take);
        B = mix(B, t.b, take);
        C = mix(C, t.c, take);
        need *= 1.0 - t.covered;
    }
    float inv = A.w;

    // Layer 1, fed input by input.
    vec4 layer = u.netLayers[0];
    units = int(layer.y);
    inputQuads = int(layer.x) / 4;
    offset = int(layer.z);
    int bias = offset + units * inputQuads;
    for (int q = 0; q < units / 4; q++) hidden[q] = weights[bias + q];
    feed(0, vec4(A.rgb, inv * near));
    feed(1, C);
    for (int t = 0; t < taps; t++) {
        vec4 tap = u.netTaps[t];
        Tap v = read((gl_FragCoord.xy + tap.yz) / size, tap.x);
        feed(2 + 2 * t, vec4(v.a.rgb, (v.a.w - inv) * near) * v.covered);
        feed(3 + 2 * t, v.c * v.covered);
    }
    feed(2 + 2 * taps, vec4(certainty, valid, 1.0, 0.0));

    // Further layers.
    vec4 x[MAX_QUADS];
    for (int l = 1; l < layers; l++) {
        for (int q = 0; q < units / 4; q++) x[q] = max(hidden[q], 0.0);
        layer = u.netLayers[l];
        inputQuads = int(layer.x) / 4;
        units = int(layer.y);
        offset = int(layer.z);
        bias = offset + units * inputQuads;
        for (int j = 0; j < units; j++) {
            float sum = weights[bias + (j >> 2)][j & 3];
            for (int k = 0; k < inputQuads; k++) sum += dot(weights[offset + j * inputQuads + k], x[k]);
            hidden[j >> 2][j & 3] = sum;
        }
    }

    // Residual update (outputs: d rgb, d log inv, d h).
    float change = u.net.w * u.netRun.x;
    vec4 d0 = change * tanh(hidden[0]);
    vec4 d1 = change * tanh(hidden[1]);
    stateA = vec4(clamp(A.rgb + d0.rgb, 0.0, 1.5), inv * exp(d0.w));
    stateB = vec4(B.xyz, certainty * 0.995 + (1.0 - certainty) * 0.03);
    stateC = clamp(C + d1, -4.0, 4.0);
}
