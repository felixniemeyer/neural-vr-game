#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"
// Nearest-neighbour display of the new state.
// mode: 0 rgb, 1 certainty, 2 depth, 3 velocity, 4 hidden h0-h2,
// 6 scene G-buffer (shade, building id).
layout(push_constant) uniform Push { int target; int source; int mode; } pc;
layout(location = 0) out vec4 color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a = nextFetch(TARGET, 0, p); // rgb, inverse depth
    vec4 b = nextFetch(TARGET, 1, p); // velocity, certainty
    vec4 h = nextFetch(TARGET, 2, p); // hidden
    vec3 c = clamp(pow(max(a.rgb * u.tone[0].rgb + u.tone[1].rgb, 0.0), u.tone[2].rgb), 0.0, 1.0);
    if (pc.mode == 1) c = vec3(b.w);
    if (pc.mode == 2) c = vec3(clamp(a.w * u.net.z, 0.0, 1.0)); // near = white
    if (pc.mode == 3) c = b.xyz * 0.5 + 0.5;
    if (pc.mode == 4) c = h.xyz * 0.5 + 0.5;
    if (pc.mode == 6) c = sceneFetch(TARGET, 0, p).rgb;
    if (any(isnan(a)) || any(isnan(h)) || any(isinf(a)) || any(isinf(h))) c = vec3(1.0, 0.0, 1.0); // NaN: magenta
    color = vec4(c, 1.0);
}
