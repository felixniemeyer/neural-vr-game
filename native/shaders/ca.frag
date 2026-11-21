#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Neural cellular automaton with (for now) random, slowly morphing weights.
//
//   perception = [self, ring1, ring2, ring4, ring8] (the first u.lookup.x rings),
//                each group = hidden channels + 1 extra (the bias for self, the
//                relative inverse depth for a ring); sparse rings of 8, 6, 4, 3 taps
//                scaled by u.lookup.y, randomly rotated
//   h          <- mix(h, tanh(W * perception), rate)        W: hidden x groups (u.rule)
//   rgb <- R * [h, 1], speed <- sigmoid(R * [h, 1])         R: 4 rows (u.readout)
//   (colour lives in [-3, 4]; the display maps it with an automatic exposure)
//
// Hidden size (u.lookup.z): 1 image = 3 channels, a group is one vec4 (h0, h1, h2,
// extra); 2 images = 7 channels, a group is two vec4s (h0..h3), (h4, h5, h6, extra).
//
// Before the CA step the rendered scene (skyscraper G-buffer) is composited into the
// cell: colour with alpha u.mixing.x, depth and velocity with u.mixing.z, the
// building's identity into the hidden state with u.mixing.y. Where there is no
// building nothing changes.
//
// Regions: smooth noise on the city grid picks, per neighbourhood, the mix of rule A
// and B, the update rate and a hue rotation of the readout colour.
//
// Pixels fire asynchronously (probability u.nca.z) and rare seeds kick the hidden
// state, so structure appears locally rather than everywhere at once. The speed
// moves the point along its view ray (towards or away from the eye); the splat
// carries that motion, so depth emerges from movement. Holes revealed by
// reprojection first take a random covered neighbour.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 stateA; // rgb, inverse depth
layout(location = 1) out vec4 stateB; // world velocity xyz, certainty
layout(location = 2) out vec4 stateC; // h0..h3 (1 image: h0..h2, 0)
layout(location = 3) out vec4 stateD; // h4..h6, 0 (only with 2 hidden images)

struct Cell {
    vec3 rgb;
    float inv;
    vec3 velocity;
    float certainty;
    vec4 h0; // h0..h3 (1 image: h0..h2, extra)
    vec4 h1; // h4..h6, extra (1 image: unused)
    float coverage;
};

bool wide() {
    return u.lookup.z > 1.5;
}

Cell fetch(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(u.info.xy) - 1);
    vec4 a0 = accFetch(pc.target, 0, p);
    vec4 a1 = accFetch(pc.target, 1, p);
    vec4 a2 = accFetch(pc.target, 2, p);
    vec4 a3 = wide() ? accFetch(pc.target, 3, p) : vec4(0.0);
    Cell c;
    float w = max(a0.a, 1e-4);
    c.rgb = a0.rgb / w;
    c.inv = a1.x / w;
    c.velocity = a1.yzw / w;
    c.certainty = clamp(a2.x, 0.0, 1.0);
    c.h0 = wide() ? vec4(a2.yzw, a3.x) / w : vec4(a2.yzw / w, 0.0);
    c.h1 = vec4(a3.yzw / w, 0.0);
    c.coverage = clamp(a0.a, 0.0, 1.0);
    if (any(isnan(a0)) || any(isnan(a1)) || any(isnan(a2)) || any(isnan(a3))) {
        c.coverage = 0.0; // a NaN would spread: treat it as a hole
    }
    return c;
}

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

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float frame = u.info.w;
    vec3 r = hash3(vec3(p, frame));
    Cell self = fetch(p);
    int rings = int(u.lookup.x);

    // Perception: self plus the mean hidden state on sparse rings.
    const float radii[4] = float[](1.0, 2.0, 4.0, 8.0);
    const int counts[4] = int[](8, 6, 4, 3);
    vec4 perception[10];
    for (int j = 0; j < 10; j++) perception[j] = vec4(0.0);
    Cell pick = self;
    bool found = false;
    for (int ring = 0; ring < rings; ring++) {
        int taps = max(1, int(round(float(counts[ring]) * u.lookup.y)));
        vec4 sumA = vec4(0.0), sumB = vec4(0.0);
        float sumInv = 0.0;
        float weight = 0.0;
        for (int k = 0; k < taps; k++) {
            float angle = 6.28318 * (float(k) + r.x) / float(taps) + float(ring);
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
        perception[wide() ? 3 + ring * 2 : 2 + ring * 2].w = weight > 0.0 ? sumInv / weight : -1.0;
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
        c.inv = mix(1.0 / u.step.y, 1.0 / u.step.z, r.z);
        c.velocity = vec3(0.0);
        c.h0 = vec4(hash3(vec3(p, frame + 31.0)), r.y) * 2.0 - 1.0;
        c.h1 = vec4(hash3(vec3(p, frame + 37.0)) * 2.0 - 1.0, 0.0);
    }

    // Scene: composite the rendered building (if any) before the CA step.
    vec4 g = sceneFetch(pc.target, p);
    float building = g.w > 0.0 ? 1.0 : 0.0;
    float alpha = building * u.mixing.x;
    float depthAlpha = building * u.mixing.z;
    float identity = building * u.mixing.y;
    vec3 palette = 0.6 + 0.4 * cos(6.28318 * (3.0 * g.w + vec3(0.0, 0.33, 0.67)));
    c.rgb = mix(c.rgb, palette * g.x, alpha);
    c.inv = mix(c.inv, g.y, depthAlpha);
    c.velocity = mix(c.velocity, vec3(0.0, g.z, 0.0), depthAlpha);
    c.certainty = mix(self.certainty, 1.0, depthAlpha);
    c.h0 = mix(c.h0, vec4(hash3(vec3(g.w * 997.0, 1.0, 2.0)), hash(vec3(g.w * 997.0, 3.0, 4.0))) * 2.0 - 1.0, identity);
    c.h1 = mix(c.h1, vec4(hash3(vec3(g.w * 997.0, 5.0, 6.0)) * 2.0 - 1.0, 0.0), identity);

    perception[0] = c.h0;
    perception[1] = vec4(c.h1.xyz, 0.0);
    perception[wide() ? 1 : 0].w = 1.0; // bias
    for (int ring = 0; ring < rings; ring++) {
        int slot = wide() ? 3 + ring * 2 : 2 + ring * 2;
        float ringInv = perception[slot].w;
        perception[slot].w = ringInv < 0.0 ? 0.0 : (ringInv - c.inv) * u.step.y; // relative to the near depth
    }

    // World position of this pixel (for the region field and the radial motion).
    vec2 ndc = ((vec2(p) + 0.5) / u.info.xy * 2.0 - 1.0) * vec2(1.0, -1.0);
    vec4 onRay = u.invViewProj[pc.target] * vec4(ndc, 0.0, 1.0);
    vec3 eyePos = u.eye[pc.target].xyz;
    vec3 ray = normalize(onRay.xyz / onRay.w - eyePos);
    float distance = 1.0 / max(c.inv, 1e-3);
    vec3 world = eyePos + ray * distance;

    // Regions: world-anchored noise on the scale of the city grid, slowly drifting.
    // Each neighbourhood leans to rule A or B, has its own pace and its own hues; the
    // boundaries are where they compete.
    float t = u.info.z;
    vec2 q = world.xz / (1.5 * u.city.x) + vec2(0.011, 0.007) * t;
    float region = smoothstep(0.3, 0.7, noise(q, 11.0));
    float pace = mix(0.4, 1.6, noise(q * 0.8, 23.0));
    float hue = 6.28318 * noise(q * 0.6, 37.0);

    // h <- mix(h, tanh(W * perception), rate) with W = mix(A, B, region).
    int outputs = wide() ? 7 : 3;
    int groups = 1 + rings;
    float hidden[7];
    for (int o = 0; o < 7; o++) {
        hidden[o] = 0.0;
        if (o >= outputs) continue;
        float sumA = 0.0, sumB = 0.0;
        for (int group = 0; group < groups; group++) {
            int j = group * 2;
            sumA += dot(u.rule[o * 10 + j], perception[j]);
            sumB += dot(u.ruleB[o * 10 + j], perception[j]);
            if (wide()) {
                sumA += dot(u.rule[o * 10 + j + 1], perception[j + 1]);
                sumB += dot(u.ruleB[o * 10 + j + 1], perception[j + 1]);
            }
        }
        hidden[o] = tanh(mix(sumA, sumB, region));
    }
    // Asynchronous firing: only some pixels update this frame.
    vec3 chance = hash3(vec3(p, frame + 23.0));
    float rate = chance.x < u.nca.z ? min(u.nca.x * pace, 1.0) : 0.0;
    // Noise on the hidden state only, smooth over a few pixels and drifting, so it
    // perturbs the patterns without visible speckle (strength: u.mixing.w).
    vec2 nq = vec2(p) / 5.0 + vec2(0.37, 0.23) * (0.05 * frame);
    vec3 jitter = (vec3(noise(nq, 51.0), noise(nq, 52.0), noise(nq, 53.0)) - 0.5)
               * (3.0 * u.nca.y * u.mixing.w);
    vec4 next0 = wide() ? vec4(hidden[0], hidden[1], hidden[2], hidden[3])
                        : vec4(hidden[0], hidden[1], hidden[2], c.h0.w);
    c.h0 = mix(c.h0, next0, rate) + vec4(jitter, -jitter.x);
    c.h1 = vec4(mix(c.h1.xyz, vec3(hidden[4], hidden[5], hidden[6]), rate) + jitter.zxy, 0.0);
    // Seeds: frequent tiny kicks (+-0.1) to single pixels' hidden state, which
    // structure can grow from or collapse with (frequency: u.lookup.w).
    vec3 cell = vec3(vec2(p), frame);
    if (hash(cell + vec3(0.0, 0.0, 41.0)) < u.nca.w * 16.0 * u.lookup.w) {
        c.h0 += (vec4(hash3(cell + 43.0), hash(cell + 47.0)) - 0.5) * 0.2;
        c.h1 += vec4(hash3(cell + 53.0) - 0.5, 0.0) * 0.2;
    }
    if (!wide()) {
        c.h0.w = 0.0;
        c.h1 = vec4(0.0);
    }

    // Readout: colour and radial speed from the hidden state.
    vec4 inA = wide() ? c.h0 : vec4(c.h0.xyz, 1.0);
    vec4 inB = wide() ? vec4(c.h1.xyz, 1.0) : vec4(0.0);
    vec4 raw;
    for (int o = 0; o < 4; o++) {
        raw[o] = 2.0 * (dot(u.readout[o * 2], inA) + dot(u.readout[o * 2 + 1], inB));
    }
    // raw has a standard deviation of a few units; scale the colour to about 1 so the
    // [-3, 4] range is rarely hit (the display's auto exposure does the rest).
    c.rgb = mix(c.rgb, hueRotate(raw.rgb * 0.3, hue), 0.3);
    float out4a = sigmoid(raw.a);

    // Radial speed: move along the view ray (towards or away from the eye), within
    // soft bounds near..far. The splat applies the velocity next frame.
    // Buildings keep their own (vertical) velocity.
    float near = u.step.y, far = u.step.z;
    float speed = (out4a * 2.0 - 1.0) * 0.05 * far;         // m/s, away (+) or closer (-)
    speed += max(0.0, near - distance) * 2.0;               // too close: push away
    speed -= max(0.0, distance - far) * 0.5;                // too far: pull in
    c.velocity = mix(c.velocity, ray * speed, 0.1 * (1.0 - building));
    // Hard push-away: what you fly into recedes instead of reaching your face
    // (half of the way back to `near` per frame).
    c.inv = mix(c.inv, min(c.inv, 1.0 / near), 0.5 * (1.0 - building));

    float certainty = clamp(c.certainty * 0.995 + (1.0 - c.certainty) * 0.03, 0.0, 1.0);

    stateA = vec4(clamp(c.rgb, -3.0, 4.0), c.inv);
    stateB = vec4(c.velocity, certainty);
    stateC = c.h0;
    stateD = c.h1;
}
