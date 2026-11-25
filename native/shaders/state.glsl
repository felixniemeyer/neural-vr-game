// Descriptor bindings for the per-eye images, with accessors (eye-major groups):
//   1-6    previous state (2 eyes x A, B, C)
//   7-14   splat accumulators (2 eyes x acc0..acc3)
//   15-20  resolved state (2 eyes x A, B, C), with the hole-aware mip pyramid
//   21-22  resolved D (per eye), with mips like A, B, C
//   23-28  new state (2 eyes x A, B, C)
//   29-32  scene (2 eyes x colour + inverse distance, world velocity)
//   33     net weights (uniform buffer)
//
// The passes reading these are compiled per eye: TARGET is the eye drawn, SOURCE the
// splat's source eye (world.rs pipelines), so the switches below fold to one image.
//
// State (prev, next):  A = (rgb, inverse depth; 0 = no data), B = (velocity, certainty),
//                      C = (h0, h1, h2, h3)
// Accumulators:        acc0 = (rgb, inv)*w, acc1 = (velocity, certainty)*w, acc2 = h*w, acc3 = w
// Resolved (mid):      A = (rgb, inv), B = (velocity, coverage), C = h; holes are all 0, so
//                      level 0 is "premultiplied by coverage" and its mips (2x2 averages)
//                      are the hole-aware pyramid: value = mip / coverage. D = (certainty).
//   LAYOUT 2 (reproject-v2): everything premultiplied by the raw splat weight w instead:
//                      A = (rgb, inv) w, B = (velocity w, w), C = h w, D = (motion w, w),
//                      value = mip / (mip weight + 1e-3) (see resolve.frag).

layout(constant_id = 7) const int LAYOUT = 1; // the net's layout: 1 = reproject-v1, 2 = reproject-v2
layout(constant_id = 8) const int TARGET = 0;
layout(constant_id = 9) const int SOURCE = 0; // splat: the eye whose state it reprojects

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
layout(set = 0, binding = 21) uniform sampler2D mid03;
layout(set = 0, binding = 22) uniform sampler2D mid13;
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
// float16 weights (rule::pack), 8 per uvec4; 64 KiB = world.rs WEIGHT_HALVES
layout(std140, set = 0, binding = 33) uniform Weights { uvec4 weights[4096]; };

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

// resolved state, exact (level 0); image 3 = D
vec4 midFetch(int eye, int image, ivec2 p) {
    if (image == 3) return eye == 0 ? texelFetch(mid03, p, 0) : texelFetch(mid13, p, 0);
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

// resolved state, bilinear at uv on a mip level (premultiplied by coverage); image 3 = D
vec4 midLod(int eye, int image, vec2 uv, float level) {
    if (image == 3) return eye == 0 ? textureLod(mid03, uv, level) : textureLod(mid13, uv, level);
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
    return midFetch(eye, 3, p).x;
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
// no colour, inverse distance u.mixing.y, 0 = no sky), and its weight: 1 near, fading to 0 at
// GROUND_REACH field radii; the sky counts fully.
vec4 sceneAt(int eye, ivec2 p, out float fade) {
    vec4 g = sceneFetch(eye, 0, p);
    if (g.w <= 0.0) {
        fade = u.mixing.y > 0.0 ? 1.0 : 0.0; // sky 0: off
        return vec4(0.0, 0.0, 0.0, u.mixing.y);
    }
    fade = clamp(1.0 - 1.0 / g.w / (GROUND_REACH * fieldRadius()), 0.0, 1.0);
    return g;
}
