// Descriptor bindings for the per-eye state (A, B, C, D) and splat accumulators,
// with accessors. Bindings: 1-8 previous state, 9-16 accumulators, 17-24 new state
// (eye-major, 4 images each).

layout(set = 0, binding = 1) uniform sampler2D prev00;
layout(set = 0, binding = 2) uniform sampler2D prev01;
layout(set = 0, binding = 3) uniform sampler2D prev02;
layout(set = 0, binding = 4) uniform sampler2D prev03;
layout(set = 0, binding = 5) uniform sampler2D prev10;
layout(set = 0, binding = 6) uniform sampler2D prev11;
layout(set = 0, binding = 7) uniform sampler2D prev12;
layout(set = 0, binding = 8) uniform sampler2D prev13;
layout(set = 0, binding = 9) uniform sampler2D acc00;
layout(set = 0, binding = 10) uniform sampler2D acc01;
layout(set = 0, binding = 11) uniform sampler2D acc02;
layout(set = 0, binding = 12) uniform sampler2D acc03;
layout(set = 0, binding = 13) uniform sampler2D acc10;
layout(set = 0, binding = 14) uniform sampler2D acc11;
layout(set = 0, binding = 15) uniform sampler2D acc12;
layout(set = 0, binding = 16) uniform sampler2D acc13;
layout(set = 0, binding = 17) uniform sampler2D next00;
layout(set = 0, binding = 18) uniform sampler2D next01;
layout(set = 0, binding = 19) uniform sampler2D next02;
layout(set = 0, binding = 20) uniform sampler2D next03;
layout(set = 0, binding = 21) uniform sampler2D next10;
layout(set = 0, binding = 22) uniform sampler2D next11;
layout(set = 0, binding = 23) uniform sampler2D next12;
layout(set = 0, binding = 24) uniform sampler2D next13;

vec4 prevFetch(int eye, int image, ivec2 p) {
    switch (eye * 4 + image) {
    case 0: return texelFetch(prev00, p, 0);
    case 1: return texelFetch(prev01, p, 0);
    case 2: return texelFetch(prev02, p, 0);
    case 3: return texelFetch(prev03, p, 0);
    case 4: return texelFetch(prev10, p, 0);
    case 5: return texelFetch(prev11, p, 0);
    case 6: return texelFetch(prev12, p, 0);
    case 7: return texelFetch(prev13, p, 0);
    }
    return vec4(0.0);
}

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

vec4 nextFetch(int eye, int image, ivec2 p) {
    switch (eye * 4 + image) {
    case 0: return texelFetch(next00, p, 0);
    case 1: return texelFetch(next01, p, 0);
    case 2: return texelFetch(next02, p, 0);
    case 3: return texelFetch(next03, p, 0);
    case 4: return texelFetch(next10, p, 0);
    case 5: return texelFetch(next11, p, 0);
    case 6: return texelFetch(next12, p, 0);
    case 7: return texelFetch(next13, p, 0);
    }
    return vec4(0.0);
}
