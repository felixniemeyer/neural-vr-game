#version 450
#extension GL_GOOGLE_include_directive : require
#include "state.glsl"
// Nearest-neighbour display of the new state.
// mode: 0 rgb, 1 certainty, 2 depth, 3 velocity, 4 hidden h0-h2, 5 hidden h3-h5.
layout(push_constant) uniform Push { int target; int source; int mode; } pc;
layout(location = 0) out vec4 color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a = nextFetch(pc.target, 0, p);
    vec4 b = nextFetch(pc.target, 1, p);
    vec3 c = a.rgb;
    if (pc.mode == 1) c = vec3(b.a);
    if (pc.mode == 2) c = vec3(clamp(a.a * 1.5, 0.0, 1.0));
    if (pc.mode == 3) c = b.xyz * 0.5 + 0.5;
    if (pc.mode == 4) c = nextFetch(pc.target, 2, p).xyz * 0.5 + 0.5;
    if (pc.mode == 5) c = vec3(nextFetch(pc.target, 2, p).w, nextFetch(pc.target, 3, p).xy) * 0.5 + 0.5;
    if (any(isnan(a)) || any(isnan(b)) || any(isinf(a)) || any(isinf(b))) c = vec3(1.0, 0.0, 1.0); // NaN: magenta
    color = vec4(c, 1.0);
}
