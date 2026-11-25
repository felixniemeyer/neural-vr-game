"""Camera motion from depth + flow (prepare.py, fov.py), torch only.

fit_pair(): frame t's disparity and the flow t -> t+1 give R, t (p' = R p + t, camera t
coords -> camera t+1 coords, x right, y up, z ahead) and the motion the camera doesn't
explain (vel). Each frame has its own field of view, so a known zoom between t and t+1
is not mistaken for forward motion.
"""
import math

import torch
import torch.nn.functional as F

from sunk.video import rays


def sample(img, coords):
    """Bilinear sample img [C,H,W] at pixel coords [H,W,2]."""
    h, w = img.shape[-2:]
    grid = torch.stack([(coords[..., 0] + 0.5) / w * 2 - 1, (coords[..., 1] + 0.5) / h * 2 - 1], -1)
    return F.grid_sample(img[None], grid[None], mode="bilinear", padding_mode="border", align_corners=False)[0]


def skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def fit_camera(p, n, focal, still, iters=12):
    """Robust Gauss-Newton: R, t such that R p + t projects onto n (x/z, y/z).

    The robust (Cauchy) scale anneals from 4 to 0.3 px, so the dominant motion (usually
    the background) wins and the rest become outliers.
    `still` penalises translation (in near-units, per pixel weight): a near object moving
    sideways looks like camera translation with parallax; film cameras mostly pan.
    focal: of the target frame (residuals in its pixels).
    """
    r, t = torch.eye(3), torch.zeros(3)
    for k in range(iters):
        q = p @ r.T + t
        x, y, z = q.unbind(-1)
        res = (n - torch.stack([x / z, y / z], -1)) * focal  # pixels
        dn = torch.zeros(len(q), 2, 3)
        dn[:, 0, 0], dn[:, 0, 2] = 1 / z, -x / z ** 2
        dn[:, 1, 1], dn[:, 1, 2] = 1 / z, -y / z ** 2
        jac = torch.cat([dn @ -skew(q), dn], -1) * focal  # d proj / d(omega, t)
        error = res.norm(dim=1)
        sigma = max(4 * 0.75 ** k, 0.3)
        w = 1 / (1 + (error / sigma) ** 2)
        prior = torch.tensor([0, 0, 0, 1, 1, 1.]) * still * focal ** 2 * w.sum()
        a = torch.einsum("n,nij,nik->jk", w, jac, jac) + torch.diag(prior) + 1e-6 * torch.eye(6)
        delta = torch.linalg.solve(a, torch.einsum("n,nij,ni->j", w, jac, res) - prior[3:].sum() / 3 * torch.cat([torch.zeros(3), t]))
        r = torch.linalg.matrix_exp(skew(delta[:3])) @ r
        t = torch.linalg.matrix_exp(skew(delta[:3])) @ t + delta[3:]
    return r, t


def focal_of(size, fov):
    """Pinhole focal length (px) of a square frame of `size` px with field of view fov (deg)."""
    return size / 2 / math.tan(math.radians(fov) / 2)


def fit_pair(disp, flow, good, fov0, fov1, still, moving):
    """disp [h, w] (frame t, 1 = near), flow [2, h, w] (px, t -> t+1, y down), good [h, w]
    (pixels to fit), fov0 / fov1 (deg) of frames t and t+1. Returns R [3, 3], t [3],
    vel [h, w, 3] (unexplained motion, camera t+1 coords, lateral; soft above `moving` px)."""
    h, w = disp.shape
    pix = torch.stack(torch.meshgrid(torch.arange(w) + 0., torch.arange(h) + 0., indexing="xy"), -1)
    target = pix + flow.permute(1, 2, 0)
    ray0, ray1 = rays(h, w, fov0), rays(h, w, fov1)
    focal1 = focal_of(w, fov1)
    p = (ray0 / disp[..., None])[good]
    n = sample(ray1.permute(2, 0, 1), target).permute(1, 2, 0)[good][:, :2]
    r, t = fit_camera(p, n, focal1, still)
    q = p @ r.T + t
    res = n - q[:, :2] / q[:, 2:]
    # Soft: 0 below `moving` px of unexplained flow, full above twice that.
    weight = (res.norm(dim=1) * focal1 / moving - 1).clamp(0, 1)[:, None]
    v = torch.zeros(h, w, 3)
    v[good] = weight * torch.cat([res * q[:, 2:], torch.zeros_like(res[:, :1])], 1)
    return r, t, v
