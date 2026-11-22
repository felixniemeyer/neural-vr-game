// Descriptor bindings for the per-eye images, with accessors (eye-major groups):
//   1-6    previous state (2 eyes x A, B, C)
//   7-14   splat accumulators (2 eyes x acc0..acc3)
//   15-20  resolved state (2 eyes x A, B, C), with the hole-aware mip pyramid
//   21-22  resolved certainty (per eye)
//   23-28  new state (2 eyes x A, B, C)
//   29-32  scene (2 eyes x colour + inverse distance, world velocity)
//   33     net weights (storage buffer)
//
// State (prev, next):  A = (rgb, inverse depth; 0 = no data), B = (velocity, certainty),
//                      C = (h0, h1, h2, h3)
// Accumulators:        acc0 = (rgb, inv)*w, acc1 = (velocity, certainty)*w, acc2 = h*w, acc3 = w
// Resolved (mid):      A = (rgb, inv), B = (velocity, coverage), C = h; holes are all 0, so
//                      level 0 is "premultiplied by coverage" and its mips (2x2 averages)
//                      are the hole-aware pyramid: value = mip / coverage.

layout(set = 0, binding = 1) uniform sampler2D prev00;
layout(set = 0, binding = 2) uniform sampler2D prev01;
layout(set = 0, binding = 3) uniform sampler2D prev02;
layout(set = 0, binding = 4) uniform sampler2D prev10;
layout(set = 0, binding = 5) uniform sampler2D prev11;
layout(set = 0, binding = 6) uniform sampler2D prev12;
layout(set = 0, binding = 7) uniform sampler2D acc00;
layout(set = 0, binding = 8) uniform sampler2D acc01;
layout(set = 0, binding = 9) uniform sampler2D acc02;
layout(set = 0, binding = 10) uniform sampler2D acc03;
layout(set = 0, binding = 11) uniform sampler2D acc10;
layout(set = 0, binding = 12) uniform sampler2D acc11;
layout(set = 0, binding = 13) uniform sampler2D acc12;
layout(set = 0, binding = 14) uniform sampler2D acc13;
layout(set = 0, binding = 15) uniform sampler2D mid00;
layout(set = 0, binding = 16) uniform sampler2D mid01;
layout(set = 0, binding = 17) uniform sampler2D mid02;
layout(set = 0, binding = 18) uniform sampler2D mid10;
layout(set = 0, binding = 19) uniform sampler2D mid11;
layout(set = 0, binding = 20) uniform sampler2D mid12;
layout(set = 0, binding = 21) uniform sampler2D cert0;
layout(set = 0, binding = 22) uniform sampler2D cert1;
layout(set = 0, binding = 23) uniform sampler2D next00;
layout(set = 0, binding = 24) uniform sampler2D next01;
layout(set = 0, binding = 25) uniform sampler2D next02;
layout(set = 0, binding = 26) uniform sampler2D next10;
layout(set = 0, binding = 27) uniform sampler2D next11;
layout(set = 0, binding = 28) uniform sampler2D next12;
layout(set = 0, binding = 29) uniform sampler2D scene00;
layout(set = 0, binding = 30) uniform sampler2D scene01;
layout(set = 0, binding = 31) uniform sampler2D scene10;
layout(set = 0, binding = 32) uniform sampler2D scene11;
layout(std430, set = 0, binding = 33) readonly buffer Weights { vec4 weights[]; };

// Scene: image 0 = (colour, inverse distance; 0 = nothing rendered), image 1 = (world
// velocity, 0).
vec4 sceneFetch(int eye, int image, ivec2 p) {
    switch (eye * 2 + image) {
    case 0: return texelFetch(scene00, p, 0);
    case 1: return texelFetch(scene01, p, 0);
    case 2: return texelFetch(scene10, p, 0);
    case 3: return texelFetch(scene11, p, 0);
    }
    return vec4(0.0);
}

// previous state (ping-pong source)
vec4 prevFetch(int eye, int image, ivec2 p) {
    switch (eye * 3 + image) {
    case 0: return texelFetch(prev00, p, 0);
    case 1: return texelFetch(prev01, p, 0);
    case 2: return texelFetch(prev02, p, 0);
    case 3: return texelFetch(prev10, p, 0);
    case 4: return texelFetch(prev11, p, 0);
    case 5: return texelFetch(prev12, p, 0);
    }
    return vec4(0.0);
}

// splat accumulators
vec4 accFetch(int eye, int image, ivec2 p) {
    switch (eye * 4 + image) {
    case 0: return texelFetch(acc00, p, 0);
    case 1: return texelFetch(acc01, p, 0);
    case 2: return texelFetch(acc02, p, 0);
    case 3: return texelFetch(acc03, p, 0);
    case 4: return texelFetch(acc10, p, 0);
    case 5: return texelFetch(acc11, p, 0);
    case 6: return texelFetch(acc12, p, 0);
    case 7: return texelFetch(acc13, p, 0);
    }
    return vec4(0.0);
}

// resolved state, exact (level 0)
vec4 midFetch(int eye, int image, ivec2 p) {
    switch (eye * 3 + image) {
    case 0: return texelFetch(mid00, p, 0);
    case 1: return texelFetch(mid01, p, 0);
    case 2: return texelFetch(mid02, p, 0);
    case 3: return texelFetch(mid10, p, 0);
    case 4: return texelFetch(mid11, p, 0);
    case 5: return texelFetch(mid12, p, 0);
    }
    return vec4(0.0);
}

// resolved state, bilinear at uv on a mip level (premultiplied by coverage)
vec4 midLod(int eye, int image, vec2 uv, float level) {
    switch (eye * 3 + image) {
    case 0: return textureLod(mid00, uv, level);
    case 1: return textureLod(mid01, uv, level);
    case 2: return textureLod(mid02, uv, level);
    case 3: return textureLod(mid10, uv, level);
    case 4: return textureLod(mid11, uv, level);
    case 5: return textureLod(mid12, uv, level);
    }
    return vec4(0.0);
}

float certFetch(int eye, ivec2 p) {
    return eye == 0 ? texelFetch(cert0, p, 0).x : texelFetch(cert1, p, 0).x;
}

// new state (net output)
vec4 nextFetch(int eye, int image, ivec2 p) {
    switch (eye * 3 + image) {
    case 0: return texelFetch(next00, p, 0);
    case 1: return texelFetch(next01, p, 0);
    case 2: return texelFetch(next02, p, 0);
    case 3: return texelFetch(next10, p, 0);
    case 4: return texelFetch(next11, p, 0);
    case 5: return texelFetch(next12, p, 0);
    }
    return vec4(0.0);
}

// The scene at p as (colour, inverse distance) with the sky filled in (no geometry:
// no colour, inverse distance u.mixing.y), and its weight: 1 near, fading to 0 at
// GROUND_REACH field radii; the sky counts fully.
vec4 sceneAt(int eye, ivec2 p, out float fade) {
    vec4 g = sceneFetch(eye, 0, p);
    if (g.w <= 0.0) {
        fade = 1.0;
        return vec4(0.0, 0.0, 0.0, u.mixing.y);
    }
    fade = clamp(1.0 - 1.0 / g.w / (GROUND_REACH * fieldRadius()), 0.0, 1.0);
    return g;
}
