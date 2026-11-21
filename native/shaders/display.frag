#version 450
// Nearest-neighbour display of the new state. mode: 0 rgb, 1 certainty, 2 depth, 3 velocity.
layout(set = 0, binding = 11) uniform sampler2D stateA0;
layout(set = 0, binding = 12) uniform sampler2D stateB0;
layout(set = 0, binding = 13) uniform sampler2D stateA1;
layout(set = 0, binding = 14) uniform sampler2D stateB1;

layout(push_constant) uniform Push { int target; int source; int mode; } pc;
layout(location = 0) out vec4 color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a = pc.target == 0 ? texelFetch(stateA0, p, 0) : texelFetch(stateA1, p, 0);
    vec4 b = pc.target == 0 ? texelFetch(stateB0, p, 0) : texelFetch(stateB1, p, 0);
    vec3 c = a.rgb;
    if (pc.mode == 1) c = vec3(b.a);
    if (pc.mode == 2) c = vec3(clamp(a.a * 1.5, 0.0, 1.0));
    if (pc.mode == 3) c = b.xyz * 0.5 + 0.5;
    if (any(isnan(a)) || any(isnan(b)) || any(isinf(a)) || any(isinf(b))) c = vec3(1.0, 0.0, 1.0); // NaN: magenta
    color = vec4(c, 1.0);
}
