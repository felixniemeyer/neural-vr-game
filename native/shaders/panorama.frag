#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "state.glsl"
#include "cube.glsl"

// DESKTOP: the whole cube-v1 sphere store in one image, seen from its centre (no eye
// parallax), for a window: an equirectangular panorama or a domemaster (azimuthal
// equidistant fisheye). Both follow the camera (right, up, back: its world axes). The
// dome: x image right, y image up, z its zenith; the camera's forward sits `attention`
// from the zenith towards the bottom edge (fabric-artwork's camera.ts playerToDomemaster).
// Reads the latest state (prev: World::draw has flipped the parity after an update).

layout(push_constant) uniform Push {
    vec4 right;
    vec4 up;
    vec4 back;
    vec4 view; // attention, dome angle across (radians), kind (0 equirect, 1 dome), mode
    vec4 size; // pixels of the image (width, height); the square's offset, side (dome)
} pc;
layout(location = 0) out vec4 color;

const float PI = 3.14159265359;

void main() {
    vec2 p = gl_FragCoord.xy;
    vec3 local; // camera coordinates: x right, y up, -z forward
    if (pc.view.z > 0.5) {
        vec2 uv = (p - pc.size.z) / pc.size.w * 2.0 - 1.0;
        uv.y = -uv.y;
        float r = length(uv);
        if (r > 1.0) { color = vec4(0.0, 0.0, 0.0, 1.0); return; }
        float theta = r * pc.view.y * 0.5;
        vec3 dome = r > 0.0 ? vec3(sin(theta) * uv / r, cos(theta)) : vec3(0.0, 0.0, 1.0);
        float s = sin(pc.view.x), c = cos(pc.view.x);
        local = vec3(dome.x, dot(dome, vec3(0.0, c, s)), -dot(dome, vec3(0.0, -s, c)));
    } else {
        float lon = p.x / pc.size.x * 2.0 * PI - PI;
        float lat = 0.5 * PI - p.y / pc.size.y * PI;
        local = vec3(sin(lon) * cos(lat), sin(lat), -cos(lon) * cos(lat));
    }
    vec3 d = normalize(pc.right.xyz * local.x + pc.up.xyz * local.y + pc.back.xyz * local.z);
    vec4 a = cubeLatest(0, 0, d);
    vec3 c = clamp(pow(max(a.rgb * u.tone[0].rgb + u.tone[1].rgb, 0.0), u.tone[2].rgb), 0.0, 1.0);
    int mode = int(pc.view.w);
    if (mode == 1) c = vec3(cubeLatest(0, 1, d).w);
    if (mode == 2) c = vec3(clamp(a.w * u.net.z, 0.0, 1.0));
    if (any(isnan(a)) || any(isinf(a))) c = vec3(1.0, 0.0, 1.0);
    color = vec4(c, 1.0);
}
