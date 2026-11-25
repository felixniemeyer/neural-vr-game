#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#include "common.glsl"
#include "state.glsl"
#include "cube.glsl"

// The trained per-pixel net, the same math as train/sunk/model.py Net.forward:
//
//   fill     holes take the finest pyramid level that has data here (push-pull)
//   inputs   self (rgb, inv * near), (h); per tap (mip level L, offset dx, dy):
//            (rgb, (inv - inv_self) * near), (h), bilinear from the hole-aware
//            pyramid, 0 where it is empty; extra (certainty, valid, 1, tau)
//   layers   1x1 "convolutions" from the weights buffer, relu between them; float16
//            (packed math: per input one fma for 4 outputs)
//   update   d = max_change * tau * (1 - gate * certainty) * tanh(out) (an Euler step of tau ticks):
//            rgb += d.rgb, inv *= exp(d.w), h += d_h; certainty decays by 0.965 per tick
//
// tau = u.step.y: the net runs at its own rate (main.rs); on the frames in between
// tau = 0 and only the holes are filled.
//
// LAYOUT 2 (reproject-v2, model.py forward_v2): the resolved images are premultiplied by
// the splat weight w (state.glsl); the fill is continuous (need = 1 - min(w, 1)); inputs
// self (rgb, inv * near), (h), (motion, w), per tap (rgb, (inv - inv_self) * near *
// min(w_t, 1)), (h), (motion, w_t), extra (1, tau, 0, 0); no gate, no certainty decay
// (B.w passes w on, for the display).
//
// LAYOUT 3 (cube-v1, model.py forward_v2 with scale; cube.glsl): the same as LAYOUT 2 on
// the cube atlas (cube mid A..D, written by cube_pull.frag): taps at face texel offsets,
// mapped across face edges; extra (1, tau, scale, 0); the pulled weight is replaced by 1
// (as in training: the next pull's coverage and consistency make the new weight).
//
// Weights (rule::pack): float16 in slots of 4; per layer 4x4 blocks (q outputs, k inputs)
// of 4 slots, one per input holding its 4 outputs' weights, then the bias.
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

const int IN0 = 4 * ((LAYOUT == 1 ? 2 : 3) * (1 + TAPS) + 1); // vec4s per position: 2 (v1) or 3 (v2, cube)
const int OFF0 = 0;
const int OFF1 = OFF0 + (IN0 * UNITS0 + UNITS0) / 4;
const int OFF2 = OFF1 + (UNITS0 * UNITS1 + UNITS1) / 4;
const int OFF3 = OFF2 + (UNITS1 * UNITS2 + UNITS2) / 4;
const int MAX_QUADS = 16;

f16vec4 hidden[MAX_QUADS];

// Weight slot j: 4 halves (bit casts, free).
f16vec4 slot(int j) {
    uvec4 e = weights[j >> 1];
    uvec2 h = (j & 1) == 0 ? e.xy : e.zw;
    return f16vec4(unpackFloat2x16(h.x), unpackFloat2x16(h.y));
}

// Outputs 4q.. of a layer from its inputs 4k.. (x): block (q, k).
f16vec4 block(const int offset, const int inputs, int q, int k, f16vec4 x) {
    int b = offset + 4 * (q * (inputs / 4) + k);
    return fma(slot(b), x.xxxx, fma(slot(b + 1), x.yyyy, fma(slot(b + 2), x.zzzz, slot(b + 3) * x.w)));
}

// Layer 1: add input vec4 number k to every unit.
void feed(int k, vec4 x) {
    f16vec4 h = f16vec4(x);
    [[unroll]] for (int q = 0; q < UNITS0 / 4; q++) hidden[q] += block(OFF0, IN0, q, k, h);
}

// A further layer: inputs relu(hidden[0 .. IN/4]), outputs into hidden[0 .. OUT/4].
void dense(const int offset, const int inputs, const int outputs) {
    f16vec4 x[MAX_QUADS];
    [[unroll]] for (int q = 0; q < inputs / 4; q++) x[q] = max(hidden[q], f16vec4(0.0));
    [[unroll]] for (int q = 0; q < outputs / 4; q++) {
        f16vec4 sum = slot(offset + outputs * (inputs / 4) + q);
        [[unroll]] for (int k = 0; k < inputs / 4; k++) sum += block(offset, inputs, q, k, x[k]);
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
    vec4 a = midLod(TARGET, 0, uv, level);
    vec4 b = midLod(TARGET, 1, uv, level);
    vec4 c = midLod(TARGET, 2, uv, level);
    float k = 1.0 / max(b.w, 1e-4);
    return Tap(a * k, vec4(b.xyz * k, b.w), c * k, b.w > 1e-3 ? 1.0 : 0.0);
}

struct Tap2 {
    vec4 a; // rgb, inv
    vec3 v; // world velocity
    vec4 c; // h
    vec3 m; // screen motion
    float w;
};

// v2: a premultiplied read (level 0 exact at p, else bilinear at uv on `level`) divided
// by its weight + 1e-3.
Tap2 unpremultiply(vec4 a, vec4 b, vec4 c, vec4 d) {
    float k = 1.0 / (d.w + 1e-3);
    return Tap2(a * k, b.xyz * k, c * k, d.xyz * k, d.w);
}

Tap2 read2(vec2 uv, float level) {
    return unpremultiply(midLod(TARGET, 0, uv, level), midLod(TARGET, 1, uv, level),
                         midLod(TARGET, 2, uv, level), midLod(TARGET, 3, uv, level));
}

void mainV2() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 size = u.info.xy;
    float near = u.net.z;
    Tap2 s = unpremultiply(midFetch(TARGET, 0, p), midFetch(TARGET, 1, p), midFetch(TARGET, 2, p), midFetch(TARGET, 3, p));
    float w = s.w;

    // Continuous push-pull fill: the own value counts min(w, 1), each level what is still
    // missing times its coverage; normalised by what was gathered (model.py forward_v2).
    float own = min(w, 1.0);
    float total = own, need = 1.0 - own;
    Tap2 g = Tap2(s.a * own, s.v * own, s.c * own, s.m * own, w);
    [[unroll]] for (int level = 1; level < LEVELS; level++) {
        Tap2 t = read2(gl_FragCoord.xy / size, float(level));
        float covered = min(t.w, 1.0);
        float take = need * covered;
        g.a += t.a * take;
        g.v += t.v * take;
        g.c += t.c * take;
        g.m += t.m * take;
        total += take;
        need *= 1.0 - covered;
    }
    float k = 1.0 / max(total, 1e-6);
    s = Tap2(g.a * k, g.v * k, g.c * k, g.m * k, w);
    // What was missing takes the scene's depth and velocity (colour stays push-pull's).
    float missing = 1.0 - min(w, 1.0);
    if (missing > 0.0) {
        float fade;
        vec4 g = sceneAt(TARGET, p, fade);
        if (fade > 0.0) {
            s.a.w = mix(s.a.w, g.w, missing);
            s.v = mix(s.v, sceneFetch(TARGET, 1, p).xyz, missing);
        }
    }
    float inv = s.a.w;
    float tau = u.step.y;
    if (tau == 0.0) {
        stateA = s.a;
        stateB = vec4(s.v, w);
        stateC = s.c;
        return;
    }

    [[unroll]] for (int q = 0; q < UNITS0 / 4; q++) hidden[q] = slot(OFF0 + UNITS0 * (IN0 / 4) + q);
    feed(0, vec4(s.a.rgb, inv * near));
    feed(1, s.c);
    feed(2, vec4(s.m, w));
    [[unroll]] for (int t = 0; t < TAPS; t++) {
        vec4 tap = u.netTaps[t];
        Tap2 v = read2((gl_FragCoord.xy + tap.yz) / size, tap.x);
        feed(3 + 3 * t, vec4(v.a.rgb, (v.a.w - inv) * near * min(v.w, 1.0)));
        feed(4 + 3 * t, v.c);
        feed(5 + 3 * t, vec4(v.m, v.w));
    }
    feed(3 + 3 * TAPS, vec4(1.0, tau * u.netRun.x, 0.0, 0.0));
    if (LAYERS > 1) dense(OFF1, UNITS0, UNITS1);
    if (LAYERS > 2) dense(OFF2, UNITS1, UNITS2);
    if (LAYERS > 3) dense(OFF3, UNITS2, UNITS3);

    float change = u.net.w * tau;
    vec4 d0 = change * tanh(vec4(hidden[0]));
    vec4 d1 = change * tanh(vec4(hidden[1]));
    stateA = vec4(clamp(s.a.rgb + d0.rgb, 0.0, 1.5), inv * exp(d0.w));
    stateB = vec4(s.v, w);
    stateC = clamp(s.c + d1, -4.0, 4.0);
}

Tap2 readCube(vec2 uv, float level) {
    return unpremultiply(textureLod(cubeMidA, uv, level), textureLod(cubeMidB, uv, level),
                         textureLod(cubeMidC, uv, level), textureLod(cubeMidD, uv, level));
}

void mainCube() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int f;
    vec2 here;
    atlasTexel(gl_FragCoord.xy, f, here);
    vec2 size = atlasSize();
    float near = u.net.z;
    Tap2 s = unpremultiply(texelFetch(cubeMidA, p, 0), texelFetch(cubeMidB, p, 0), texelFetch(cubeMidC, p, 0), texelFetch(cubeMidD, p, 0));
    float w = s.w;

    // Continuous push-pull fill (as mainV2; the mip texels stay within the face).
    float own = min(w, 1.0);
    float total = own, need = 1.0 - own;
    Tap2 g = Tap2(s.a * own, s.v * own, s.c * own, s.m * own, w);
    [[unroll]] for (int level = 1; level < LEVELS; level++) {
        Tap2 t = readCube(gl_FragCoord.xy / size, float(level));
        float covered = min(t.w, 1.0);
        float take = need * covered;
        g.a += t.a * take;
        g.v += t.v * take;
        g.c += t.c * take;
        g.m += t.m * take;
        total += take;
        need *= 1.0 - covered;
    }
    float k = 1.0 / max(total, 1e-6);
    s = Tap2(g.a * k, g.v * k, g.c * k, g.m * k, w);
    // What was missing takes the scene's depth and velocity where an eye sees it.
    float missing = 1.0 - min(w, 1.0);
    if (missing > 0.0) {
        vec4 scene;
        vec3 sceneVelocity;
        float fade, sky;
        vec2 centred;
        if (sceneDir(faceDir(f, here), s.a.w > 1e-6 ? 1.0 / s.a.w : 20.0, scene, sceneVelocity, fade, sky, centred) && fade > 0.0) {
            s.a.w = mix(s.a.w, scene.w, missing);
            s.v = mix(s.v, sceneVelocity, missing);
        }
    }
    float inv = s.a.w;
    float tau = u.step.y;

    [[unroll]] for (int q = 0; q < UNITS0 / 4; q++) hidden[q] = slot(OFF0 + UNITS0 * (IN0 / 4) + q);
    feed(0, vec4(s.a.rgb, inv * near));
    feed(1, s.c);
    feed(2, vec4(s.m, w));
    [[unroll]] for (int t = 0; t < TAPS; t++) {
        vec4 tap = u.netTaps[t];
        Tap2 v = readCube(cubeUV(f, here + tap.yz, tap.x), tap.x);
        feed(3 + 3 * t, vec4(v.a.rgb, (v.a.w - inv) * near * min(v.w, 1.0)));
        feed(4 + 3 * t, v.c);
        feed(5 + 3 * t, vec4(v.m, v.w));
    }
    feed(3 + 3 * TAPS, vec4(1.0, tau * u.netRun.x, u.cube.z, 0.0));
    if (LAYERS > 1) dense(OFF1, UNITS0, UNITS1);
    if (LAYERS > 2) dense(OFF2, UNITS1, UNITS2);
    if (LAYERS > 3) dense(OFF3, UNITS2, UNITS3);

    float change = u.net.w * tau;
    vec4 d0 = change * tanh(vec4(hidden[0]));
    vec4 d1 = change * tanh(vec4(hidden[1]));
    stateA = vec4(clamp(s.a.rgb + d0.rgb, 0.0, 1.5), inv * exp(d0.w));
    stateB = vec4(s.v, 1.0);
    stateC = clamp(s.c + d1, -4.0, 4.0);
}

void main() {
    if (LAYOUT == 3) {
        mainCube();
        return;
    }
    if (LAYOUT == 2) {
        mainV2();
        return;
    }
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 size = u.info.xy;
    float near = u.net.z;

    vec4 A = midFetch(TARGET, 0, p);
    vec4 B = midFetch(TARGET, 1, p);
    vec4 C = midFetch(TARGET, 2, p);
    float valid = A.w > 0.0 ? 1.0 : 0.0;
    float certainty = certFetch(TARGET, p) * valid;

    // Push-pull hole fill from the pyramid.
    float need = 1.0 - valid;
    [[unroll]] for (int level = 1; level < LEVELS; level++) {
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
        vec4 g = sceneAt(TARGET, p, fade);
        if (fade > 0.0) {
            A.w = g.w;
            B.xyz = sceneFetch(TARGET, 1, p).xyz;
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
    [[unroll]] for (int q = 0; q < UNITS0 / 4; q++) hidden[q] = slot(OFF0 + UNITS0 * (IN0 / 4) + q);
    feed(0, vec4(A.rgb, inv * near));
    feed(1, C);
    [[unroll]] for (int t = 0; t < TAPS; t++) {
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
    // Certainty gate (model.py net.gate): well-observed pixels change less.
    float change = u.net.w * tau * (1.0 - u.netRun.y * clamp(certainty, 0.0, 1.0));
    vec4 d0 = change * tanh(vec4(hidden[0]));
    vec4 d1 = change * tanh(vec4(hidden[1]));
    stateA = vec4(clamp(A.rgb + d0.rgb, 0.0, 1.5), inv * exp(d0.w));
    const float rest = 0.03 / 0.035;
    stateB = vec4(B.xyz, rest + (certainty - rest) * pow(0.965, tau));
    stateC = clamp(C + d1, -4.0, 4.0);
}
