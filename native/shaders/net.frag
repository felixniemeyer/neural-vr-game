#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Tiny per-pixel net on the resolved state (weights from the CPU, rule.rs).
//
//   taps:    12 offsets on a golden-angle spiral, radius growing exponentially
//            (1 px .. reach), from u.taps; each tap is (h0, h1, h2, relative depth)
//   self:    (h0, h1, h2, certainty), (region0, region1, near / distance, 1)
//   hidden:  16 = relu(sum_k w1[k] * input_k)              14 inputs x 4 quads of mat4
//   out:     (dh0, dh1, dh2, speed), (r, g, b, -) = w2 * hidden
//   h        <- mix(h, tanh(dh), rate), rate = pace * mix(1, alive, certainty)
//   rgb      <- towards the hue-rotated readout (fully in holes)
//
// Each tap has its own weights, so the kernel sees direction (flow, cells, edges)
// and several scales at once (no single stripe wavelength). Regions are inputs of
// the one network (world-anchored noise), not separate rules. Holes take the
// weighted mean of the valid taps first; with nothing in reach they are seeded.
// The speed moves the point along its view ray within near..far (the splat applies
// the velocity next frame); closer than near is pushed away.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 stateA; // rgb, certainty
layout(location = 1) out vec4 stateB; // world velocity, 0
layout(location = 2) out vec4 stateC; // h0, h1, h2, inverse depth

// Smooth value noise in [0, 1].
float noise(vec2 q, float seed) {
    vec2 i = floor(q), f = q - i;
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(vec3(i, seed)), b = hash(vec3(i + vec2(1, 0), seed));
    float c = hash(vec3(i + vec2(0, 1), seed)), d = hash(vec3(i + vec2(1, 1), seed));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Rotate a colour around the grey axis.
vec3 hueRotate(vec3 c, float angle) {
    const vec3 k = vec3(0.57735);
    return c * cos(angle) + cross(k, c) * sin(angle) + k * dot(k, c) * (1.0 - cos(angle));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 last = ivec2(u.info.xy) - 1;
    float frame = u.info.w;
    float near = u.step.y, far = u.step.z;
    vec4 A = midFetch(pc.target, 0, p);
    vec4 B = midFetch(pc.target, 1, p);
    vec4 C = midFetch(pc.target, 2, p);
    float building = B.w;
    bool hole = C.w <= 0.0;

    // Taps (one texel each) and the hole fill.
    vec4 taps[12];
    for (int k = 0; k < 12; k++) {
        vec4 o = u.taps[k / 2];
        vec2 offset = (k & 1) == 0 ? o.xy : o.zw;
        taps[k] = midFetch(pc.target, 2, clamp(p + ivec2(offset), ivec2(0), last));
    }
    if (hole) {
        vec3 r = hash3(vec3(p, frame));
        float weight = 0.0;
        for (int k = 0; k < 12; k++) {
            float wk = taps[k].w > 0.0 ? 1.0 / float(k + 1) : 0.0;
            C += taps[k] * wk;
            weight += wk;
        }
        C = weight > 0.0 ? C / weight
                         : vec4(r * 2.0 - 1.0, mix(1.0 / near, 1.0 / far, hash(vec3(p, frame + 7.0))));
    }

    // World position of this pixel (regions, radial motion).
    vec4 onRay = u.invViewProj[pc.target] * vec4(ndcFromPixel(vec2(p), u.info.xy), 0.0, 1.0);
    vec3 eyePos = u.eye[pc.target].xyz;
    vec3 ray = normalize(onRay.xyz / onRay.w - eyePos);
    float distance = 1.0 / max(C.w, 1e-3);
    vec3 world = eyePos + ray * distance;

    // Regions: world-anchored noise on the scale of the city grid, slowly drifting:
    // two network inputs, a pace and a hue per neighbourhood.
    float t = u.info.z;
    vec2 q = world.xz / (1.5 * u.city.x) + vec2(0.011, 0.007) * t;
    vec2 region = vec2(noise(q, 11.0), noise(q * 0.7, 13.0)) * 2.0 - 1.0;
    float pace = mix(0.4, 1.6, noise(q * 0.8, 23.0));
    float hue = 6.28318 * noise(q * 0.6, 37.0);

    // Layer 1: 14 input vec4s -> 16 hidden (4 quads), all mat4 * vec4.
    vec4 h1[4] = vec4[](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    for (int k = 0; k < 12; k++) {
        vec4 x = taps[k].w > 0.0 ? vec4(taps[k].xyz, (taps[k].w - C.w) * near) : vec4(0.0);
        for (int j = 0; j < 4; j++) h1[j] += u.w1[k * 4 + j] * x;
    }
    vec4 self0 = vec4(C.xyz, A.a);
    vec4 self1 = vec4(region, near * C.w, 1.0);
    for (int j = 0; j < 4; j++) {
        h1[j] = max(h1[j] + u.w1[48 + j] * self0 + u.w1[52 + j] * self1, 0.0);
    }
    // Layer 2: 16 hidden -> 2 output vec4s.
    vec4 o0 = vec4(0.0), o1 = vec4(0.0);
    for (int j = 0; j < 4; j++) {
        o0 += u.w2[j] * h1[j];
        o1 += u.w2[4 + j] * h1[j];
    }

    // Update: never settles; certain pixels keep a fraction `alive` of the rate.
    float rate = min(u.nca.x * pace * mix(1.0, u.nca.z, A.a), 1.0);
    vec3 h = mix(C.xyz, tanh(o0.xyz), rate);
    // Smooth drifting noise on the hidden state (strength u.nca.y * u.mixing.w).
    vec2 nq = vec2(p) / 5.0 + vec2(0.37, 0.23) * (0.05 * frame);
    h += (vec3(noise(nq, 51.0), noise(nq, 52.0), noise(nq, 53.0)) - 0.5) * (3.0 * u.nca.y * u.mixing.w);
    // Seeds: frequent tiny kicks (+-0.1) to single pixels (probability u.nca.w).
    vec3 cell = vec3(vec2(p), frame);
    if (hash(cell + vec3(0.0, 0.0, 41.0)) < u.nca.w) h += (hash3(cell + 43.0) - 0.5) * 0.2;

    vec3 rgb = mix(A.rgb, hueRotate(o1.rgb * 0.5, hue), hole ? 1.0 : 0.3);

    // Radial speed within soft bounds near..far; buildings keep their own velocity.
    float speed = tanh(o0.w) * 0.05 * far;                  // m/s, away (+) or closer (-)
    speed += max(0.0, near - distance) * 2.0;               // too close: push away
    speed -= max(0.0, distance - far) * 0.5;                // too far: pull in
    vec3 velocity = mix(B.xyz, ray * speed, 0.1 * (1.0 - building));
    // Hard push-away: what you fly into recedes (half the way back to `near` per frame).
    float inv = mix(C.w, min(C.w, 1.0 / near), 0.5 * (1.0 - building));

    float certainty = clamp(A.a * 0.995 + (1.0 - A.a) * 0.03, 0.0, 1.0);

    stateA = vec4(clamp(rgb, -3.0, 4.0), certainty);
    stateB = vec4(velocity, 0.0);
    stateC = vec4(clamp(h, -4.0, 4.0), inv);
}
