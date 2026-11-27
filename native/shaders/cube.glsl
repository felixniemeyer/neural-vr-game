// cube-v1 (LAYOUT 3): the state is one head-centred sphere, an equi-angular cubemap (EAC)
// whose faces are fixed to the world (train/sunk/sphere.py, train/train_cube.py).
//
// Faces: F x F texels each (u.cube.x), in a 3 x 2 atlas of cells F + 2 G square: a gutter of
// G texels (u.head.w, one texel of the coarsest mip level) around each face holds the face's
// extended coordinates (below), so bilinear reads and mips cross face edges with the right
// neighbours. Face f's texel 0 is at atlas texel (f % 3, f / 3) * (F + 2 G) + G.
// Face frame (right, up, forward) in world coordinates, x right, y up, z ahead as in
// training; texel rows go down. Texel coords x (continuous, texel centres at integers) map
// to the angles uv = (x + 0.5) * a - pi/4 (a = pi/2 / F, the texel angle) and to the
// direction normalize(right tan u - up tan v + forward). The formulas hold beyond the face
// (|uv| > pi/4), so taps and the pull's search work in the face's own extended coordinates
// and are mapped to whichever face holds that direction (cubeUV) once they leave the gutter.
// M and mid are computed in the gutters too (cube_forward, cube_pull: pure functions of the
// previous state, so they match the neighbour face); the state (prev / next) is not: its
// reads go to the owning face (cubeTexel), so every region has exactly one simulation.
//
// Images (atlas size, all RGBA16F), bindings after the per-eye ones (world.rs CUBE_BINDING):
//   prev / next  A = (rgb, inverse distance from the head), B = (world velocity, weight),
//                C = (h0..h3): the state before and after this update (ping-pong)
//   M            the forward motion of each prev texel: (moved direction - direction, 1) * alive,
//                with mips (the premultiplied pyramid the pull searches)
//   old A, B     split updates only: the state before the latest (World::advance)
//   acc A..D, W  splat reprojection only (net.json reproject "splat"): cube_splat.vert's
//                sums, folded into mid A..D by cube_fold.frag
//   mid A..D     the pulled state, premultiplied by its weight w like v2's resolved images:
//                A = (rgb, inv) w, B = (velocity w, w), C = h w, D = (motion, w): motion =
//                (texels moved, change of log distance) per tick; with mips (the net's pyramid)

layout(set = 0, binding = 35) uniform sampler2D cubePrevA;
layout(set = 0, binding = 36) uniform sampler2D cubePrevB;
layout(set = 0, binding = 37) uniform sampler2D cubePrevC;
layout(set = 0, binding = 38) uniform sampler2D cubeNextA;
layout(set = 0, binding = 39) uniform sampler2D cubeNextB;
layout(set = 0, binding = 40) uniform sampler2D cubeNextC;
layout(set = 0, binding = 41) uniform sampler2D cubeM;
layout(set = 0, binding = 42) uniform sampler2D cubeMidA;
layout(set = 0, binding = 43) uniform sampler2D cubeMidB;
layout(set = 0, binding = 44) uniform sampler2D cubeMidC;
layout(set = 0, binding = 45) uniform sampler2D cubeMidD;
// Splat reprojection (cube_splat, cube_fold): the accumulators (atlas size, float32), sums of
// w n c (value), w n c, w n (cube_splat.frag), and the coverage W = sum w.
layout(set = 0, binding = 46) uniform sampler2D cubeAccA; // (rgb, inv) w n c
layout(set = 0, binding = 47) uniform sampler2D cubeAccB; // (velocity w n c, w n c)
layout(set = 0, binding = 48) uniform sampler2D cubeAccC; // h w n c
layout(set = 0, binding = 49) uniform sampler2D cubeAccD; // (motion w n c, w n)
layout(set = 0, binding = 50) uniform sampler2D cubeAccW; // w
// Split updates (the desktop): the state before the latest (A, B), for the display's blend.
layout(set = 0, binding = 51) uniform sampler2D cubeOldA;
layout(set = 0, binding = 52) uniform sampler2D cubeOldB;

const float QUARTER = 0.78539816339;
const float CUBE_EPS = 1e-3; // model.py EPS: values = premultiplied sums / (weight + EPS)

const vec3 FACE_FORWARD[6] = vec3[](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
const vec3 FACE_UP[6] = vec3[](vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 0, -1), vec3(0, 1, 0), vec3(0, 1, 0));

float faceSize() { return u.cube.x; }
float texelAngle() { return u.cube.w; }
float gutter() { return u.head.w; }
float cellSize() { return faceSize() + 2.0 * gutter(); }
vec2 atlasSize() { return vec2(3.0, 2.0) * cellSize(); }
vec3 faceRight(int f) { return cross(FACE_FORWARD[f], FACE_UP[f]); }
vec2 faceOrigin(int f) { return vec2(f % 3, f / 3) * cellSize() + gutter(); }
bool inFace(vec2 x) { return all(greaterThanEqual(x, vec2(0.0))) && all(lessThanEqual(x, vec2(faceSize() - 1.0))); }

int faceOf(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x > 0.0 ? 0 : 1;
    if (a.y >= a.z) return d.y > 0.0 ? 2 : 3;
    return d.z > 0.0 ? 4 : 5;
}

// Unit direction of face f's texel coords x (extended beyond the face as in training).
vec3 faceDir(int f, vec2 x) {
    vec2 t = tan((x + 0.5) * texelAngle() - QUARTER);
    return normalize(faceRight(f) * t.x - FACE_UP[f] * t.y + FACE_FORWARD[f]);
}

// The gravity frame at the unit direction d: the image's right and up there (u.up the
// world's up; at the poles u.front stands in). The net's taps and motion live in it
// (sphere.py local_frame; its canvas coords are left-handed, so its crosses swap).
void localFrame(vec3 d, out vec3 right, out vec3 up) {
    vec3 r = cross(d, u.up.xyz);
    if (dot(r, r) <= 1e-8) r = cross(d, u.front.xyz);
    right = normalize(r);
    up = cross(right, d);
}

// Face f's texel coords of a direction / point p (in front of the face).
vec2 faceTexel(int f, vec3 p) {
    float z = dot(p, FACE_FORWARD[f]);
    vec2 uv = vec2(atan(dot(p, faceRight(f)), z), atan(-dot(p, FACE_UP[f]), z));
    return (uv + QUARTER) / texelAngle() - 0.5;
}

// The atlas texel (face, coords) of this fragment; x in -G .. F + G - 1 (the gutter outside 0 .. F - 1).
void atlasTexel(vec2 fragCoord, out int f, out vec2 x) {
    vec2 cell = floor((fragCoord - 0.5) / cellSize());
    f = int(cell.x) + 3 * int(cell.y);
    x = fragCoord - 0.5 - cell * cellSize() - gutter();
}

// Whether a bilinear read at mip `level` around face coords x stays within the face and its
// gutter (the level's texels: 2^level fine texels, centred).
bool inCell(vec2 x, float level) {
    float edge = 0.5 * (exp2(level) - 1.0);
    vec2 lo = vec2(edge - gutter()), hi = vec2(faceSize() - 1.0 + gutter() - edge);
    return all(greaterThanEqual(x, lo)) && all(lessThanEqual(x, hi));
}

// Atlas uv of the direction d (its own face; near an edge the bilinear footprint reaches
// into the gutter).
vec2 dirUV(vec3 d, float level) {
    int f = faceOf(d);
    float edge = 0.5 * (exp2(level) - 1.0);
    vec2 x = clamp(faceTexel(f, d), vec2(edge - gutter()), vec2(faceSize() - 1.0 + gutter() - edge));
    return (faceOrigin(f) + x + 0.5) / atlasSize();
}

// Atlas uv of face f's extended texel coords x: within its cell directly, else in the
// face that holds the direction.
vec2 cubeUV(int f, vec2 x, float level) {
    if (inCell(x, level)) return (faceOrigin(f) + x + 0.5) / atlasSize();
    return dirUV(faceDir(f, x), level);
}

// The atlas texel holding face f's integer extended coords c (nearest), always in the
// owning face's interior (the state has no gutter).
ivec2 cubeTexel(int f, ivec2 c) {
    int n = int(faceSize());
    if (all(greaterThanEqual(c, ivec2(0))) && all(lessThan(c, ivec2(n))))
        return ivec2(faceOrigin(f)) + c;
    vec3 d = faceDir(f, vec2(c));
    int g = faceOf(d);
    ivec2 y = clamp(ivec2(floor(faceTexel(g, d) + 0.5)), ivec2(0), ivec2(n - 1));
    return ivec2(faceOrigin(g)) + y;
}

// The state (image 0 A, 1 B, 2 C) at an atlas texel: source 1 = next (this update's
// output), 0 = prev, 2 = old (split updates: the state before prev; A and B only).
vec4 cubeFetch(int source, int image, ivec2 p) {
    if (source == 2) return image == 0 ? texelFetch(cubeOldA, p, 0) : texelFetch(cubeOldB, p, 0);
    if (source == 1) {
        if (image == 0) return texelFetch(cubeNextA, p, 0);
        if (image == 1) return texelFetch(cubeNextB, p, 0);
        return texelFetch(cubeNextC, p, 0);
    }
    if (image == 0) return texelFetch(cubePrevA, p, 0);
    if (image == 1) return texelFetch(cubePrevB, p, 0);
    return texelFetch(cubePrevC, p, 0);
}

// The state in direction d: bilinear in d's face, texels past its edge from the neighbour
// face (the state has no gutter).
vec4 cubeLatest(int source, int image, vec3 d) {
    int f = faceOf(d);
    vec2 x = faceTexel(f, d);
    vec2 base = floor(x), t = x - base;
    ivec2 c = ivec2(base);
    return mix(mix(cubeFetch(source, image, cubeTexel(f, c)), cubeFetch(source, image, cubeTexel(f, c + ivec2(1, 0))), t.x),
               mix(cubeFetch(source, image, cubeTexel(f, c + ivec2(0, 1))), cubeFetch(source, image, cubeTexel(f, c + ivec2(1, 1))), t.x), t.y);
}

// The rendered scene in direction d from the head: (colour, inverse distance from the
// head) and its fade (sceneAt), world velocity, sky (1: no geometry there), the focus
// position (resolve.frag); from the first eye that sees d. `guess`:
// a distance to place the point at for the projection (the state's own, else 20 m).
bool sceneDir(vec3 d, float guess, out vec4 g, out vec3 velocity, out float fade, out float sky, out vec2 centred) {
    vec3 head = u.head.xyz;
    for (int eye = 0; eye < 2; eye++) {
        vec4 clip = u.viewProj[eye] * vec4(head + d * guess, 1.0);
        if (clip.w <= 1e-4) continue;
        vec2 px = pixelFromClip(clip, u.info.xy);
        if (any(lessThan(px, vec2(-0.5))) || any(greaterThan(px, u.info.xy - 0.5))) continue;
        ivec2 p = clamp(ivec2(floor(px + 0.5)), ivec2(0), ivec2(u.info.xy) - 1);
        g = sceneAt(eye, p, fade);
        velocity = sceneFetch(eye, 1, p).xyz;
        sky = sceneFetch(eye, 0, p).w > 0.0 ? 0.0 : 1.0;
        centred = (vec2(p) + 0.5) / u.info.xy * 2.0 - 1.0 - (eye == 0 ? u.gaze.xy : u.gaze.zw);
        if (g.w > 0.0) { // re-measure from the head: the eye sits a few cm beside it
            vec4 q = u.invViewProj[eye] * vec4(ndcFromPixel(vec2(p), u.info.xy), 0.0, 1.0);
            vec3 from = u.eye[eye].xyz;
            g.w = 1.0 / max(length(from + normalize(q.xyz / q.w - from) / g.w - head), 0.05);
        }
        return true;
    }
    return false;
}
