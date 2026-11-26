#version 450
// cube-v1 splat reprojection: the 2x2 tent of one point (splat.frag's), into the sums
// sphere.py splat() normalises: values * w n c, w n c, w n, and the coverage w.
layout(location = 0) flat in vec4 vA; // rgb, inverse distance
layout(location = 1) flat in vec4 vB; // velocity, certainty
layout(location = 2) flat in vec4 vC; // h
layout(location = 3) flat in vec4 vD; // motion xyz, nearness
layout(location = 4) flat in vec2 vCenter;
layout(location = 0) out vec4 accA; // (rgb, inv) w n c
layout(location = 1) out vec4 accB; // (velocity w n c, w n c)
layout(location = 2) out vec4 accC; // h w n c
layout(location = 3) out vec4 accD; // (motion w n c, w n)
layout(location = 4) out float accW; // w

void main() {
    vec2 delta = abs(gl_FragCoord.xy - 0.5 - vCenter);
    vec2 axis = max(vec2(0.0), vec2(1.0) - delta);
    float w = axis.x * axis.y;
    if (w <= 0.0) discard;
    float wn = w * vD.w;
    float wnc = wn * vB.w;
    accA = vA * wnc;
    accB = vec4(vB.xyz * wnc, wnc);
    accC = vC * wnc;
    accD = vec4(vD.xyz * wnc, wn);
    accW = w;
}
