#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Particle bodies (swarm.rs): one point per particle, sized in metres (push mode: the
// particle size in mm), read from the creatures buffer (binding 34: world.rs CREATURES_BINDING).
layout(push_constant) uniform Push { int target; int source; int mode; } pc;

struct Particle { vec4 position; vec4 color; vec4 velocity; };
layout(std430, set = 0, binding = 34) readonly buffer Particles { Particle particles[]; };

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec3 vVelocity;

void main() {
    Particle p = particles[gl_VertexIndex];
    vWorld = p.position.xyz;
    vColor = p.color.rgb;
    vVelocity = p.velocity.xyz;
    gl_Position = u.viewProj[pc.target] * vec4(vWorld, 1.0);
    // Pixels per metre at this depth: the projection's y scale times half the height.
    float focal = u.viewProj[pc.target][1][1] * u.info.y * 0.5;
    gl_PointSize = clamp(float(pc.mode) * 0.001 * abs(focal) / max(gl_Position.w, 0.05), 1.0, 32.0);
    gl_Position.y = -gl_Position.y; // state pixels are y down (see pixelFromClip)
}
