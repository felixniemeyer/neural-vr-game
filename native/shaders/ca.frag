#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Neural cellular automaton with (for now) random, slowly morphing weights.
//
//   perception = [self, ring1, ring2, ring4, ring8], each 7 hidden channels + 1 extra
//                (sparse rings: 8, 6, 4, 3 taps, randomly rotated; the extra is the
//                bias for self and the relative inverse depth for each ring)
//   h          <- mix(h, tanh(W * perception), rate)        W: 7 x 40 (u.rule)
//   rgb, speed <- sigmoid(R * [h, 1])                       R: 4 x 8  (u.readout)
//
// Pixels fire asynchronously (probability u.nca.z) and rare seeds kick the hidden
// state, so structure appears locally rather than everywhere at once. The speed
// moves the point along its view ray (towards or away from the eye); the splat
// carries that motion, so depth emerges from movement. Holes revealed by
// reprojection first take a random covered neighbour.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 stateA; // rgb, inverse depth
layout(location = 1) out vec4 stateB; // world velocity xyz, certainty
layout(location = 2) out vec4 stateC; // h0..h3
layout(location = 3) out vec4 stateD; // h4..h6, spare

struct Cell {
    vec3 rgb;
    float inv;
    vec3 velocity;
    float certainty;
    vec4 h0; // h0..h3
    vec4 h1; // h4..h6, extra
    float coverage;
};

Cell fetch(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(u.info.xy) - 1);
    vec4 a0 = accFetch(pc.target, 0, p);
    vec4 a1 = accFetch(pc.target, 1, p);
    vec4 a2 = accFetch(pc.target, 2, p);
    vec4 a3 = accFetch(pc.target, 3, p);
    Cell c;
    float w = max(a0.a, 1e-4);
    c.rgb = a0.rgb / w;
    c.inv = a1.x / w;
    c.velocity = a1.yzw / w;
    c.certainty = clamp(a2.x, 0.0, 1.0);
    c.h0 = vec4(a2.yzw, a3.x) / w;
    c.h1 = vec4(a3.yzw / w, 0.0);
    c.coverage = clamp(a0.a, 0.0, 1.0);
    if (any(isnan(a0)) || any(isnan(a1)) || any(isnan(a2)) || any(isnan(a3))) {
        c.coverage = 0.0; // a NaN would spread: treat it as a hole
    }
    return c;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float frame = u.info.w;
    vec3 r = hash3(vec3(p, frame));
    Cell self = fetch(p);

    // Perception: self plus the mean hidden state on 4 sparse rings.
    const float radii[4] = float[](1.0, 2.0, 4.0, 8.0);
    const int counts[4] = int[](8, 6, 4, 3);
    vec4 perception[10];
    Cell pick = self;
    bool found = false;
    for (int ring = 0; ring < 4; ring++) {
        vec4 sumA = vec4(0.0), sumB = vec4(0.0);
        float sumInv = 0.0;
        float weight = 0.0;
        for (int k = 0; k < counts[ring]; k++) {
            float angle = 6.28318 * (float(k) + r.x) / float(counts[ring]) + float(ring);
            Cell n = fetch(p + ivec2(round(vec2(cos(angle), sin(angle)) * radii[ring])));
            if (n.coverage < 0.25) continue;
            sumA += n.h0;
            sumB += n.h1;
            sumInv += n.inv;
            weight += 1.0;
            if (!found && ring == 0 && hash(vec3(p, frame + float(k) + 5.0)) < 0.4) {
                pick = n;
                found = true;
            }
        }
        perception[2 + ring * 2] = weight > 0.0 ? sumA / weight : vec4(0.0);
        perception[3 + ring * 2] = weight > 0.0 ? sumB / weight : vec4(0.0);
        // extra slot: relative inverse depth of the ring (relief), filled in below
        perception[3 + ring * 2].w = weight > 0.0 ? sumInv / weight : -1.0;
    }

    // Holes (newly revealed) take a random covered neighbour; nothing near: seed.
    float cov = self.coverage;
    Cell c = self;
    c.rgb = mix(pick.rgb, self.rgb, cov);
    c.inv = mix(pick.inv, self.inv, cov);
    c.velocity = mix(pick.velocity, self.velocity, cov) * 0.98;
    c.h0 = mix(pick.h0, self.h0, cov);
    c.h1 = mix(pick.h1, self.h1, cov);
    if (!found && cov < 0.01) {
        c.rgb = vec3(0.5);
        c.inv = 1.0 / (1.5 + 4.0 * r.z);
        c.velocity = vec3(0.0);
        c.h0 = vec4(hash3(vec3(p, frame + 31.0)), r.y) * 2.0 - 1.0;
        c.h1 = vec4(hash3(vec3(p, frame + 37.0)) * 2.0 - 1.0, 0.0);
    }
    perception[0] = c.h0;
    perception[1] = vec4(c.h1.xyz, 1.0); // bias
    for (int ring = 0; ring < 4; ring++) {
        float ringInv = perception[3 + ring * 2].w;
        perception[3 + ring * 2].w = ringInv < 0.0 ? 0.0 : (ringInv - c.inv) * 4.0;
    }

    // World position of this pixel (for the region field and the radial motion).
    vec2 ndc = ((vec2(p) + 0.5) / u.info.xy * 2.0 - 1.0) * vec2(1.0, -1.0);
    vec4 onRay = u.invViewProj[pc.target] * vec4(ndc, 0.0, 1.0);
    vec3 eyePos = u.eye[pc.target].xyz;
    vec3 ray = normalize(onRay.xyz / onRay.w - eyePos);
    float distance = 1.0 / max(c.inv, 1e-3);
    vec3 world = eyePos + ray * distance;

    // Region field: low-frequency, world-anchored, slowly drifting. Areas of a few
    // metres follow rule A or rule B; the boundaries are where they compete.
    float t = u.info.z;
    float field = sin(dot(world, vec3(0.61, 0.23, -0.47)) + 0.05 * t)
                * sin(dot(world, vec3(-0.19, 0.71, 0.35)) - 0.03 * t)
                + 0.5 * sin(dot(world, vec3(0.9, -0.4, 0.8)) * 1.7 + 0.07 * t);
    float region = smoothstep(-0.25, 0.25, field);

    // h <- mix(h, tanh(W * perception), rate) with W = mix(A, B, region).
    float hidden[7];
    for (int o = 0; o < 7; o++) {
        float sumA = 0.0, sumB = 0.0;
        for (int j = 0; j < 10; j++) {
            sumA += dot(u.rule[o * 10 + j], perception[j]);
            sumB += dot(u.ruleB[o * 10 + j], perception[j]);
        }
        hidden[o] = tanh(mix(sumA, sumB, region));
    }
    // Asynchronous firing: only some pixels update this frame.
    vec3 chance = hash3(vec3(p, frame + 23.0));
    float rate = chance.x < u.nca.z ? u.nca.x : 0.0;
    vec3 noise = (hash3(vec3(p, frame + 17.0)) - 0.5) * u.nca.y;
    c.h0 = mix(c.h0, vec4(hidden[0], hidden[1], hidden[2], hidden[3]), rate) + vec4(noise, -noise.x);
    c.h1 = vec4(mix(c.h1.xyz, vec3(hidden[4], hidden[5], hidden[6]), rate) + noise.zxy, 0.0);
    // Rare seeds: a strong local kick that structure can grow from.
    if (chance.y < u.nca.w) {
        c.h0 = vec4(hash3(vec3(p, frame + 41.0)), chance.z) * 4.0 - 2.0;
        c.h1 = vec4(hash3(vec3(p, frame + 43.0)) * 4.0 - 2.0, 0.0);
    }

    // Readout: colour and radial speed from the hidden state.
    vec4 inA = c.h0, inB = vec4(c.h1.xyz, 1.0);
    vec4 out4;
    for (int o = 0; o < 4; o++) {
        out4[o] = sigmoid(2.0 * (dot(u.readout[o * 2], inA) + dot(u.readout[o * 2 + 1], inB)));
    }
    c.rgb = mix(c.rgb, out4.rgb, 0.3);

    // Radial speed: move along the view ray (towards or away from the eye), within
    // soft bounds of about 1..12 m. The splat applies the velocity next frame.
    float speed = (out4.a * 2.0 - 1.0) * 0.6;              // m/s, away (+) or closer (-)
    speed += max(0.0, 1.0 - distance) * 2.0;                // too close: push away
    speed -= max(0.0, distance - 12.0) * 0.5;               // too far: pull in
    c.velocity = mix(c.velocity, ray * speed, 0.1);

    float certainty = clamp(self.certainty * 0.995 + (1.0 - self.certainty) * 0.03, 0.0, 1.0);

    stateA = vec4(clamp(c.rgb, 0.0, 1.0), c.inv);
    stateB = vec4(c.velocity, certainty);
    stateC = c.h0;
    stateD = c.h1;
}
