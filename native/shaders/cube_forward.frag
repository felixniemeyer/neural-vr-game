#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"
#include "cube.glsl"

// cube-v1 pull, part 1 (sphere.py pull: m and its pyramid): where each texel of the
// previous state goes this update, as the change of direction from the head:
// p = dir / inv + velocity * dt - head move, M = (normalize(p) - dir, 1) * alive with
// alive = its weight where it has depth and stays in front of its face. The mips of M are
// the premultiplied pyramid cube_pull.frag searches.

layout(location = 0) out vec4 outM;

void main() {
    int f;
    vec2 x;
    atlasTexel(gl_FragCoord.xy, f, x);
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a = texelFetch(cubePrevA, p, 0);
    vec4 b = texelFetch(cubePrevB, p, 0);
    vec3 d = faceDir(f, x);
    vec3 q = d / max(a.w, 1e-6) + b.xyz * u.cube.y - u.headMove.xyz;
    bool broken = any(isnan(a)) || any(isnan(b)) || any(isinf(a)) || any(isinf(b));
    float alive = (!broken && a.w > 1e-6 && dot(q, FACE_FORWARD[f]) > 1e-3) ? max(b.w, 0.0) : 0.0;
    outM = alive > 0.0 ? vec4((normalize(q) - d) * alive, alive) : vec4(0.0);
}
