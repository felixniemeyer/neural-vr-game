// Descriptor bindings for the per-eye images, with accessors. Each group has 2 eyes
// x 3 images (eye-major): bindings 1-6 previous state, 7-12 splat accumulators,
// 13-18 resolved state, 19-24 new state, 25-26 scene G-buffer per eye.
//
// State (prev, mid, next):  A = (rgb, certainty), B = (world velocity, spare),
//                           C = (h0, h1, h2, inverse depth; 0 = no data)
// Accumulators:             acc0 = (rgb, 1)*w, acc1 = (velocity, certainty)*w,
//                           acc2 = (h0, h1, h2, inverse depth)*w

layout(set = 0, binding = 1) uniform sampler2D prev00;
layout(set = 0, binding = 2) uniform sampler2D prev01;
layout(set = 0, binding = 3) uniform sampler2D prev02;
layout(set = 0, binding = 4) uniform sampler2D prev10;
layout(set = 0, binding = 5) uniform sampler2D prev11;
layout(set = 0, binding = 6) uniform sampler2D prev12;
layout(set = 0, binding = 7) uniform sampler2D acc00;
layout(set = 0, binding = 8) uniform sampler2D acc01;
layout(set = 0, binding = 9) uniform sampler2D acc02;
layout(set = 0, binding = 10) uniform sampler2D acc10;
layout(set = 0, binding = 11) uniform sampler2D acc11;
layout(set = 0, binding = 12) uniform sampler2D acc12;
layout(set = 0, binding = 13) uniform sampler2D mid00;
layout(set = 0, binding = 14) uniform sampler2D mid01;
layout(set = 0, binding = 15) uniform sampler2D mid02;
layout(set = 0, binding = 16) uniform sampler2D mid10;
layout(set = 0, binding = 17) uniform sampler2D mid11;
layout(set = 0, binding = 18) uniform sampler2D mid12;
layout(set = 0, binding = 19) uniform sampler2D next00;
layout(set = 0, binding = 20) uniform sampler2D next01;
layout(set = 0, binding = 21) uniform sampler2D next02;
layout(set = 0, binding = 22) uniform sampler2D next10;
layout(set = 0, binding = 23) uniform sampler2D next11;
layout(set = 0, binding = 24) uniform sampler2D next12;
layout(set = 0, binding = 25) uniform sampler2D scene0;
layout(set = 0, binding = 26) uniform sampler2D scene1;

// Scene G-buffer: (sun shade, inverse distance, vertical velocity, building id);
// id 0 = no building.
vec4 sceneFetch(int eye, ivec2 p) {
    return eye == 0 ? texelFetch(scene0, p, 0) : texelFetch(scene1, p, 0);
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
    switch (eye * 3 + image) {
    case 0: return texelFetch(acc00, p, 0);
    case 1: return texelFetch(acc01, p, 0);
    case 2: return texelFetch(acc02, p, 0);
    case 3: return texelFetch(acc10, p, 0);
    case 4: return texelFetch(acc11, p, 0);
    case 5: return texelFetch(acc12, p, 0);
    }
    return vec4(0.0);
}

// resolved state (resolve pass output, net input)
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
