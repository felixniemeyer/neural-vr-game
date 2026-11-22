#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"

// Resolve: normalise the splat accumulators once per pixel and composite the
// rendered scene (skyscraper G-buffer), so the raw render is never shown. Same math
// as train/sunk/model.py splat(): value = acc / w, holes (w < 0.01) all 0, certainty
// = clamp(sum certainty * w) (partial coverage lowers it).
//
// Scene blend where the scene has geometry and the state has data (the runtime
// analogue of training's "observe" blend): colour with u.mixing.x, depth, velocity and
// certainty with u.mixing.z, both fading to 0 at GROUND_REACH field radii. In holes
// nothing happens here: net.frag fills them by push-pull (colour too), then gives them
// the scene's depth and velocity. Where the scene has no geometry it is sky: depth at
// the `sky` distance, no colour, so the image's own guesses there don't come close.
//
// Output B.w is the coverage (inverse depth > 0), so that the mip chain of A, B, C
// is the hole-aware pyramid the net reads; certainty goes to its own image.

layout(push_constant) uniform Push { int target; int source; int mode; } pc;

layout(location = 0) out vec4 outA; // rgb, inverse depth (0 = no data)
layout(location = 1) out vec4 outB; // world velocity, coverage (0 or 1)
layout(location = 2) out vec4 outC; // h0..h3
layout(location = 3) out float outCertainty;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a0 = accFetch(TARGET, 0, p);
    vec4 a1 = accFetch(TARGET, 1, p);
    vec4 a2 = accFetch(TARGET, 2, p);
    float w = accFetch(TARGET, 3, p).x;
    bool hole = w < 0.01 || any(isnan(a0)) || any(isnan(a1)) || any(isnan(a2));
    vec4 a = hole ? vec4(0.0) : a0 / w;
    vec3 velocity = hole ? vec3(0.0) : a1.xyz / w;
    float certainty = hole ? 0.0 : clamp(a1.w, 0.0, 1.0);
    vec4 h = hole ? vec4(0.0) : a2 / w;

    // Holes stay holes here (push-pull fills them in net.frag, which then takes the
    // scene's depth): seeding them here would mark black pixels as covered.
    float fade;
    vec4 g = sceneAt(TARGET, p, fade);
    vec3 motion = sceneFetch(TARGET, 1, p).xyz;
    float sky = sceneFetch(TARGET, 0, p).w > 0.0 ? 0.0 : 1.0;
    float alpha = hole ? 0.0 : fade * u.mixing.x * (1.0 - sky);
    float depthAlpha = hole ? 0.0 : fade * u.mixing.z;
    a.rgb = mix(a.rgb, g.rgb, alpha);
    a.w = mix(a.w, g.w, depthAlpha);
    velocity = mix(velocity, motion, depthAlpha);
    certainty = mix(certainty, 1.0, depthAlpha);

    float covered = a.w > 0.0 ? 1.0 : 0.0;
    // Keep holes exactly 0 (the pyramid relies on it).
    outA = a * covered;
    outB = vec4(velocity * covered, covered);
    outC = h * covered;
    outCertainty = certainty * covered;
}
