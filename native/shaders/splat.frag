#version 450
// 2x2 bilinear tent splat (as fabric-artwork reproject-points.fs). The summed
// weight is the coverage; everything else is accumulated pre-multiplied by it.
// Certainty accumulates as weight * previous certainty.
layout(location = 0) flat in vec4 vA; // rgb, inverse depth
layout(location = 1) flat in vec4 vB; // velocity xyz, certainty
layout(location = 2) flat in vec4 vC; // h0..h3
layout(location = 3) flat in vec4 vD; // h4..h6, spare
layout(location = 4) flat in vec2 vCenter;
layout(location = 0) out vec4 acc0; // rgb*w, w
layout(location = 1) out vec4 acc1; // (invDepth, velocity)*w
layout(location = 2) out vec4 acc2; // (certainty, h0, h1, h2)*w
layout(location = 3) out vec4 acc3; // (h3, h4, h5, h6)*w

void main() {
    vec2 delta = abs(gl_FragCoord.xy - 0.5 - vCenter);
    vec2 axisWeight = max(vec2(0.0), vec2(1.0) - delta);
    float w = axisWeight.x * axisWeight.y;
    if (w <= 0.0) discard;
    acc0 = vec4(vA.rgb * w, w);
    acc1 = vec4(vA.a, vB.xyz) * w;
    acc2 = vec4(vB.a, vC.xyz) * w;
    acc3 = vec4(vC.w, vD.xyz) * w;
}
