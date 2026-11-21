#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Resolve the splat accumulation, then one step of a trivial stochastic CA:
// uncertain pixels copy a random covered neighbour (grainy, no streaks), drift
// in colour, depth and velocity, and occasionally spark a new colour. Certain
// pixels keep their reprojected value.
layout(set = 0, binding = 5) uniform sampler2D acc0Eye0;
layout(set = 0, binding = 6) uniform sampler2D acc1Eye0;
layout(set = 0, binding = 7) uniform sampler2D acc2Eye0;
layout(set = 0, binding = 8) uniform sampler2D acc0Eye1;
layout(set = 0, binding = 9) uniform sampler2D acc1Eye1;
layout(set = 0, binding = 10) uniform sampler2D acc2Eye1;

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 stateA; // rgb, inverse depth
layout(location = 1) out vec4 stateB; // world velocity xyz, certainty

struct Cell {
    vec3 rgb;
    float inv;
    vec3 velocity;
    float coverage;
    float certainty;
};

Cell fetch(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(u.info.xy) - 1);
    vec4 a0, a1, a2;
    if (pc.target == 0) {
        a0 = texelFetch(acc0Eye0, p, 0);
        a1 = texelFetch(acc1Eye0, p, 0);
        a2 = texelFetch(acc2Eye0, p, 0);
    } else {
        a0 = texelFetch(acc0Eye1, p, 0);
        a1 = texelFetch(acc1Eye1, p, 0);
        a2 = texelFetch(acc2Eye1, p, 0);
    }
    Cell c;
    float w = max(a0.a, 1e-4);
    c.rgb = a0.rgb / w;
    c.inv = a1.x / w;
    c.velocity = a1.yzw / w;
    c.coverage = clamp(a0.a, 0.0, 1.0);
    c.certainty = clamp(a2.x, 0.0, 1.0);
    // A single NaN/Inf would spread through neighbour copies: treat it as a hole.
    bvec4 bad = bvec4(any(isnan(a0)) || any(isinf(a0)), any(isnan(a1)) || any(isinf(a1)),
                      isnan(a2.x) || isinf(a2.x), false);
    if (any(bad)) {
        c.rgb = vec3(0.0);
        c.inv = 0.0;
        c.velocity = vec3(0.0);
        c.coverage = 0.0;
        c.certainty = 0.0;
    }
    return c;
}

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0, 0.9, 0.7) * t + vec3(0.0, 0.25, 0.55)));
}

// Slowly drifting CA parameters: sines with unrelated periods (seconds).
float wave(float period, float phase) {
    return 0.5 + 0.5 * sin(6.28318 * u.info.z / period + phase);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float frame = u.info.w;
    float drift = mix(0.03, 0.2, wave(23.0, 0.0));      // colour noise
    float sparks = mix(0.0, 0.012, wave(37.0, 1.3));    // new-colour rate
    float hue = u.info.z / 61.0;                         // palette rotation
    float turbulence = mix(0.05, 0.8, wave(17.0, 2.1)); // velocity jitter
    int reach = 1 + int(wave(29.0, 4.2) * 3.0);         // neighbour distance 1..3
    vec3 r0 = hash3(vec3(p, frame));
    vec3 r1 = hash3(vec3(p, frame + 101.0));
    Cell self = fetch(p);

    // Random covered neighbour (up to a few tries) instead of an average: keeps grain.
    Cell pick = self;
    bool found = false;
    for (int i = 0; i < 4 && !found; i++) {
        vec2 r = hash3(vec3(p, frame + 7.0 * float(i) + 3.0)).xy;
        ivec2 o = ivec2(floor(r * 3.0)) - 1;
        Cell n = fetch(p + o * (1 + int(r1.z * float(reach))));
        if (n.coverage > 0.25) {
            pick = n;
            found = true;
        }
    }

    float cov = self.coverage;
    vec3 rgb = mix(pick.rgb, self.rgb, cov);
    float inv = mix(pick.inv, self.inv, cov);
    vec3 velocity = mix(pick.velocity, self.velocity, cov);
    float certainty = self.certainty;

    // Nothing anywhere near: seed.
    if (!found && cov < 0.01) {
        rgb = palette(r0.x * 0.4 + hue) * (0.3 + 0.7 * r0.y);
        inv = 1.0 / (1.5 + 6.0 * r0.z);
        velocity = vec3(0.0);
    }

    // Randomness where uncertain: colour drift, sparks, depth and velocity jitter.
    float wild = 1.0 - certainty;
    rgb += (r0 - 0.5) * drift * wild;
    if (r1.x < sparks * wild) rgb = palette(r1.y * 0.4 + hue) * (0.4 + 0.6 * r1.z);
    inv = clamp(inv * (1.0 + (r1.y - 0.5) * 0.06 * wild), 0.1, 1.0);
    velocity += (hash3(vec3(p, frame + 211.0)) - 0.5) * turbulence * wild;
    velocity *= 0.995;
    if (length(velocity) > 1.0) velocity = normalize(velocity);

    // Brightness floor (a first, local version of the brightness corridor):
    // dark pixels slowly get re-sparked, even where certain.
    float luminance = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    if (luminance < 0.12 && r0.z < 0.01) rgb = palette(r1.y * 0.4 + hue) * (0.5 + 0.5 * r1.z);

    // Uncertain pixels settle over ~1s; certain ones decay very slowly.
    certainty = clamp(certainty * 0.999 + (1.0 - certainty) * 0.02, 0.0, 1.0);

    stateA = vec4(clamp(rgb, 0.0, 1.0), inv);
    stateB = vec4(velocity, certainty);
}
