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
// Reads the latest state (prev: the parity flips when an update finishes), blended from
// the one before (old) by right.w, so ~10 updates a second look smooth. up.w and back.w:
// the window's and the cube's frames per second, drawn top left (negative: hidden).

layout(push_constant) uniform Push {
    vec4 right; // equirect: the camera's right, up, back in the world; dome: its x, y, zenith
    vec4 up;    // (desktop.rs turns them by the attention)
    vec4 back;
    vec4 view; // half the dome angle across (radians), 0, kind (0 equirect, 1 dome), mode
    vec4 size; // pixels of the image (width, height); the square's offset, side (dome)
} pc;
layout(location = 0) out vec4 color;

const float PI = 3.14159265359;

// A 3 x 5 pixel font, rows from the top, bit 14 the top left: 0..9, then F P S C U B E.
const int GLYPHS[17] = int[](
    0x7B6F, 0x2C97, 0x73E7, 0x73CF, 0x5BC9, 0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BCF,
    0x79A4, 0x7BE4, 0x79CF, 0x7927, 0x5B6F, 0x6BAE, 0x79E7);
const float FONT = 3.0; // picture pixels per font pixel

// The glyph (-1: none) at column `col` of line `line`: "nnn FPS", "nnn CUBE".
int glyphAt(int line, int col) {
    int n = int(round(clamp(line == 0 ? pc.up.w : pc.back.w, 0.0, 999.0)));
    if (col < 3) {
        int place = col == 0 ? 100 : col == 1 ? 10 : 1;
        if (n < place && place > 1) return -1;
        return (n / place) % 10;
    }
    if (line == 0 && col >= 4 && col < 7) return 10 + col - 4;
    if (line == 1 && col >= 4 && col < 8) return 13 + col - 4;
    return -1;
}

// Text over the picture: 1 lit, 0 background box, -1 outside.
float text(vec2 p) {
    ivec2 c = ivec2(floor(p / FONT)) - ivec2(2);
    if (c.x < -1 || c.y < -1 || c.x > 4 * 8 || c.y > 7 * 2) return -1.0;
    if (c.x < 0 || c.y < 0) return 0.0;
    int line = c.y / 7, row = c.y % 7, col = c.x / 4, x = c.x % 4;
    if (row >= 5 || x >= 3 || line > 1) return 0.0;
    int glyph = glyphAt(line, col);
    if (glyph < 0) return 0.0;
    return float((GLYPHS[glyph] >> (14 - row * 3 - x)) & 1);
}

void main() {
    vec2 p = gl_FragCoord.xy;
    vec3 local; // equirect: camera coordinates (x right, y up, -z forward); dome: its own
    if (pc.view.z > 0.5) {
        vec2 uv = (p - pc.size.z) / pc.size.w * 2.0 - 1.0;
        uv.y = -uv.y;
        float r = length(uv);
        if (r > 1.0) { // outside the dome: black, the text over it
            float t = pc.up.w >= 0.0 ? text(p) : -1.0;
            color = vec4(vec3(max(t, 0.0)), 1.0);
            return;
        }
        float theta = r * pc.view.x;
        local = r > 0.0 ? vec3(sin(theta) * uv / r, cos(theta)) : vec3(0.0, 0.0, 1.0);
    } else {
        float lon = p.x / pc.size.x * 2.0 * PI - PI;
        float lat = 0.5 * PI - p.y / pc.size.y * PI;
        local = vec3(sin(lon) * cos(lat), sin(lat), -cos(lon) * cos(lat));
    }
    vec3 d = normalize(pc.right.xyz * local.x + pc.up.xyz * local.y + pc.back.xyz * local.z);
    float blend = clamp(pc.right.w, 0.0, 1.0);
    vec4 a = mix(cubeLatest(2, 0, d), cubeLatest(0, 0, d), blend);
    vec3 c = clamp(pow(max(a.rgb * u.tone[0].rgb + u.tone[1].rgb, 0.0), u.tone[2].rgb), 0.0, 1.0);
    int mode = int(pc.view.w);
    if (mode == 1) c = vec3(mix(cubeLatest(2, 1, d).w, cubeLatest(0, 1, d).w, blend));
    if (mode == 2) c = vec3(clamp(a.w * u.net.z, 0.0, 1.0));
    if (any(isnan(a)) || any(isinf(a))) c = vec3(1.0, 0.0, 1.0);
    if (pc.up.w >= 0.0) {
        float t = text(p);
        if (t >= 0.0) c = mix(c * 0.3, vec3(1.0), t);
    }
    color = vec4(c, 1.0);
}
