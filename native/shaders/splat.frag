#version 450
// 2x2 bilinear tent splat (as fabric-artwork reproject-points.fs). The summed
// weight is the coverage; everything is accumulated pre-multiplied by it.
layout(location = 0) flat in vec4 vA; // rgb, inverse depth (as seen from the target eye)
layout(location = 1) flat in vec4 vB; // velocity xyz, certainty
layout(location = 2) flat in vec4 vC; // h0..h3
layout(location = 3) flat in vec2 vCenter;
layout(location = 0) out vec4 acc0; // (rgb, inv)*w
layout(location = 1) out vec4 acc1; // (velocity, certainty)*w
layout(location = 2) out vec4 acc2; // h*w
layout(location = 3) out float acc3; // w

void main() {
    vec2 delta = abs(gl_FragCoord.xy - 0.5 - vCenter);
    vec2 axisWeight = max(vec2(0.0), vec2(1.0) - delta);
    float w = axisWeight.x * axisWeight.y;
    if (w <= 0.0) discard;
    acc0 = vA * w;
    acc1 = vB * w;
    acc2 = vC * w;
    acc3 = w;
}
