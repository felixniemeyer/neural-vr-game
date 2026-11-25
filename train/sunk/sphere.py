"""Sphere math for cube-v1: equi-angular cubemap (EAC) faces and pull reprojection.

Frames: x right, y up, z ahead; texel rows go down. A face with texel angle `a` (rad)
maps texel (i, j) (continuous, centres at +0.5) to the angles
    u = (i + 0.5) * a - pi/4,  v = (j + 0.5) * a - pi/4
and to the direction normalize(tan u, -tan v, 1). Inverse: u = atan2(x, z),
v = atan2(-y, z). The formulas stay valid beyond +-pi/4, so a training canvas near a
face edge needs no seams.

A canvas is a rectangle of a virtual face: canvas texel (i, j) is face texel
(i + origin[0], j + origin[1]). Its frame is fixed to the world (the head's rotation
doesn't move it), so only translation (parallax) and object motion move content.

pull(): every target texel x_t looks for the source texel x with fwd(x) = x_t, where
fwd projects the source point after this step's motion. Fixed-point iteration
x <- x_t - m(x), m = fwd - id, read from a premultiplied pyramid of m, coarsest level
first (hole-free guesses), one level finer per iteration, plus one more at level 0
(after Bowles et al. 2012, "Iterative Image Warping").
"""
import math

import torch
import torch.nn.functional as F

from sunk.model import CERT, EPS, INV, VEL

QUARTER = math.pi / 4


def directions(angle, origin, h, w):
    """Unit directions [B, h, w, 3] of a canvas: angle [B] (rad per texel), origin [B, 2]
    (face texel of the canvas's texel (0, 0))."""
    j, i = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return direction_at(torch.stack([i, j], -1)[None], angle, origin)


def direction_at(x, angle, origin):
    """Unit directions [B, ..., 3] at canvas texel coords x [B, ..., 2] (continuous)."""
    shape = (-1,) + (1,) * (x.dim() - 2)
    a = angle.view(shape + (1,))
    face = x + origin.view(shape + (2,))
    uv = (face + 0.5) * a - QUARTER
    t = torch.tan(uv)
    d = torch.stack([t[..., 0], -t[..., 1], torch.ones_like(t[..., 0])], -1)
    return F.normalize(d, dim=-1)


def texel_of(p, angle, origin):
    """Canvas texel coords [B, ..., 2] of points p [B, ..., 3] (any length, z > 0)."""
    shape = (-1,) + (1,) * (p.dim() - 2)
    u = torch.atan2(p[..., 0], p[..., 2])
    v = torch.atan2(-p[..., 1], p[..., 2])
    face = (torch.stack([u, v], -1) + QUARTER) / angle.view(shape + (1,)) - 0.5
    return face - origin.view(shape + (2,))


def to_grid(x, h, w):
    """Canvas texel coords [B, H, W, 2] -> grid_sample coords (align_corners=False)."""
    return torch.stack([(x[..., 0] + 0.5) / w * 2 - 1, (x[..., 1] + 0.5) / h * 2 - 1], -1)


def sample(image, x, size=None, padding="border"):
    """Bilinear read of image [B, C, h, w] at canvas texel coords x [B, H, W, 2]; size:
    the canvas (h, w) when image is a mip level of it (default: the image's own)."""
    h, w = size or image.shape[-2:]
    return F.grid_sample(image, to_grid(x, h, w), mode="bilinear", padding_mode=padding, align_corners=False)


def pull(state, angle, origin, delta, dt, levels=5, sigma=0.5, inner=None):
    """One step of pull reprojection on a canvas.

    inner = (y0, y1, x0, x1): only these targets (the output covers just them); the
    sources are the whole canvas. Reads past the canvas clamp to its border, so every
    target finds some source (as on the runtime's sphere); a bad one shows as low weight.

    state [B, C, h, w] (normalised values, CERT = weight), angle [B], origin [B, 2],
    delta [B, 3]: the head's translation this step in canvas frame (a static point p
    becomes p - delta), dt [B] seconds (VEL is m/s, canvas frame).
    Returns (state with CERT = new weight, displacement [B, 2, h, w] in texels: where
    each target texel's content was before, as target - source, and the change of log
    distance [B, 1, h, w])."""
    b, c, h, w = state.shape
    weight = state[:, CERT:CERT + 1].clamp_min(0)
    inv = state[:, INV]
    step = torch.as_tensor(dt, dtype=state.dtype).reshape(-1, 1, 1, 1)
    j, i = torch.meshgrid(torch.arange(h, dtype=state.dtype), torch.arange(w, dtype=state.dtype), indexing="ij")
    grid = torch.stack([i, j], -1)[None].expand(b, -1, -1, -1)
    y0, y1, x0, x1 = inner or (0, h, 0, w)
    here = grid[:, y0:y1, x0:x1]
    # The search for each target's source needs no gradients: only the final read has them.
    with torch.no_grad():
        weight, inv, state_ = weight.detach(), inv.detach(), state.detach()
        dirs = directions(angle, origin, h, w).expand(b, -1, -1, -1)
        vel = state_[:, VEL].permute(0, 2, 3, 1)
        p = dirs / inv.clamp_min(1e-6)[..., None] + vel * step - delta[:, None, None]
        alive = ((inv > 1e-6) & (p[..., 2] > 1e-3)).float()[:, None] * weight
        m = (texel_of(p, angle, origin) - grid).permute(0, 3, 1, 2) * (alive > 0)
        mips = [(m * alive, alive)]
        for _ in range(levels - 1):
            mips.append((F.avg_pool2d(mips[-1][0], 2), F.avg_pool2d(mips[-1][1], 2)))

        def motion(level, x):
            v, cov = sample(mips[level][0], x, (h, w)), sample(mips[level][1], x, (h, w))
            return (v / (cov + EPS)).permute(0, 2, 3, 1)

        def refine(x, iters=3):
            for _ in range(iters):
                x = here - motion(0, x)
            return x

        def judge(x):
            """(error in texels, inverse distance) of a candidate source x."""
            error = (x + motion(0, x) - here).norm(dim=-1)[:, None]
            read = sample(torch.cat([state_[:, INV:INV + 1] * weight, weight], 1), x)
            return error, read[:, :1] / (read[:, 1:] + EPS)

        # Candidates: plain coarse-to-fine, and the motion of each coarse cell around the
        # target (3x3 cells at levels 2..), so a near object's source far away is found.
        # Each is refined at level 0; where several converge the nearest wins (it occludes).
        x = here
        for level in range(levels - 1, -1, -1):
            x = here - motion(level, x)
        candidates = [refine(x)]
        for level in range(2, levels):
            cell = 2.0 ** level
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    at = here + torch.tensor([ox * cell, oy * cell])
                    candidates.append(refine(here - motion(level, at)))
        best = candidates[0]
        best_error, best_inv = judge(best)
        for x in candidates[1:]:
            error, near = judge(x)
            ok, best_ok = error < sigma, best_error < sigma
            take = (ok & best_ok & (near > best_inv * 1.05)) | (ok & ~best_ok) | (~ok & ~best_ok & (error < best_error))
            t = take.permute(0, 2, 3, 1)
            best = torch.where(t, x, best)
            best_error, best_inv = torch.where(take, error, best_error), torch.where(take, near, best_inv)
    x = best
    weight = state[:, CERT:CERT + 1].clamp_min(0)
    read = sample(torch.cat([state * weight, weight], 1), x)
    got = read[:, -1:]
    out = read[:, :-1] / (got + EPS)
    new_weight = got * torch.exp(-(best_error / sigma) ** 2)

    # The moved point, from the source values read at x: distance and velocity.
    d = direction_at(x, angle, origin)
    q = d / out[:, INV].clamp_min(1e-6)[..., None] + out[:, VEL].permute(0, 2, 3, 1) * step - delta[:, None, None]
    distance = q.norm(dim=-1).clamp_min(0.05)
    seen = (out[:, INV] > 1e-6).float()
    log_change = (torch.log(distance * out[:, INV].clamp_min(1e-6)) * seen)[:, None]
    out = out.clone()
    out[:, INV] = seen / distance
    out[:, CERT:CERT + 1] = new_weight
    return out, (here - x).permute(0, 3, 1, 2), log_change


def look_at(d):
    """Rotation [3, 3] taking the unit direction d to +z with no roll (x stays horizontal)."""
    z = F.normalize(d, dim=0)
    x = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0]), z)
    x = F.normalize(x, dim=0) if x.norm() > 1e-6 else torch.tensor([1.0, 0.0, 0.0])
    y = torch.linalg.cross(z, x)
    return torch.stack([x, y, z])  # rows: the new axes in old coords -> maps d to +z


def _check():
    torch.manual_seed(0)
    b, h, w = 2, 24, 24
    angle = torch.tensor([0.01, 0.004])
    origin = torch.tensor([[30.0, 40.0], [100.0, 60.0]])
    d = directions(angle, origin, h, w)
    j, i = torch.meshgrid(torch.arange(h) + 0.0, torch.arange(w) + 0.0, indexing="ij")
    back = texel_of(d * 3.7, angle, origin)
    assert (back - torch.stack([i, j], -1)).abs().max() < 1e-3, "EAC round trip"

    state = torch.zeros(b, 12, h, w)
    state[:, :3] = torch.rand(b, 3, h, w)
    state[:, INV] = 0.2
    state[:, CERT] = 1
    out, disp, logc = pull(state, angle, origin, torch.zeros(b, 3), torch.zeros(b))
    assert (out[:, :3] - state[:, :3]).abs().max() < 2e-3 and (out[:, CERT] - 1).abs().max() < 2e-3, "identity pull"
    assert disp.abs().max() < 1e-4 and logc.abs().max() < 1e-4

    # A plane 5 m ahead, the head moves 0.05 m right: content moves left by about
    # 0.05 / 5 rad = 0.01 rad = 1 texel at angle 0.01 (sample 0) near the centre.
    plane = torch.zeros(b, 12, h, w)
    plane[:, :3] = torch.rand(b, 3, h, w)
    plane[:, INV] = d[..., 2] / 5.0 * torch.ones(b, 1, 1)  # radial 1/distance of the plane z = 5
    plane[:, CERT] = 1
    delta = torch.tensor([[0.05, 0, 0], [0.02, 0, 0]])
    out, disp, _ = pull(plane, angle, origin, delta, torch.zeros(b))
    centre = disp[:, 0, h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4].mean((1, 2))
    # x = 5 tan u, shifted by -dx: du = -dx / 5 * cos^2 u (u of the canvas centre)
    u = (origin[:, 0] + w / 2) * angle - QUARTER
    expect = -delta[:, 0] / 5.0 * torch.cos(u) ** 2 / angle
    print("plane shift texels", centre.tolist(), "expected", expect.tolist())
    assert (centre - expect).abs().max() < 0.02 * expect.abs().max(), "translation"
    # Convergence: the found source maps onto the target.
    inner = out[:, CERT, 4:-4, 4:-4]
    print("weight inside (consistency)", inner.min().item())
    assert inner.min() > 0.9, "fixed point converged"
    print("sphere checks ok")


if __name__ == "__main__":
    _check()
