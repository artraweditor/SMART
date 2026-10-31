"""The fill's middle against the surface round it (port of greycard's
`remove/middle.rs`, `evened`).

A fill matched at its edge can still drift a few percent from the
picture deeper in.  The picture round the hole (a band set back from its
edge) is read as a smooth surface: its smooth pixels are fitted per
channel with a quadratic in logs, robustly; what the band shows past the
quadratic is carried a short way in; where the surface fits closely the
fill's own low-pass is compared with it, the difference smoothed wide,
shrunk by its noise, held to LIMIT and divided out of the hole.
"""

import numpy as np
from scipy import ndimage

import colorspace

BAND = (4, 32)
PRE = 2
LOCAL = 6
SMOOTH = 0.02
FIT = (0.02, 0.03)
SHARE = (0.3, 0.5)
LEAST = 200
OCTANTS = 4
SAME = 0.08
SOFT = (0.04, 0.07)
CARRY = 8
RADIUS = 10
FEW = 0.02
SHRINK = 3.0
LIMIT = 0.05
PASSES = 4
TUKEY = 4.685
FLOOR = 0.004
GRAINS = 1.0
LAG = 6
PEDESTAL = 0.01


def _kernel():
    side = 2 * RADIUS + 1
    j = np.arange(2 * side - 1)
    tent = (side - np.abs(j - (side - 1))).astype(np.float64)
    return float((tent * tent).sum() / tent.sum() ** 2 * (2 * PRE + 1))


def _ramp(a, b, x):
    return float(np.clip((x - a) / (b - a), 0.0, 1.0))


def _median(v):
    """The element at len//2 of the sorted values (0 for none)."""
    v = np.asarray(v)
    if v.size == 0:
        return 0.0
    mid = v.size // 2
    return float(np.partition(v, mid)[mid])


def _box_sum(a, radius):
    """Sum over a (2r+1) box round each pixel, cut at the edges."""
    r = radius
    pad = [(r + 1, r)] + [(0, 0)] * (a.ndim - 1)
    c = np.cumsum(np.pad(a, pad), axis=0)
    a = c[2 * r + 1:] - c[:-(2 * r + 1)]
    pad = [(0, 0), (r + 1, r)] + [(0, 0)] * (a.ndim - 2)
    c = np.cumsum(np.pad(a, pad), axis=1)
    return c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]


def _ring(mask, r):
    """Pixels outside `mask` within a square's reach `r` of it."""
    near = ndimage.maximum_filter(mask, size=2 * r + 1, mode='constant', cval=False)
    return near & ~mask


def _solve(m, r):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(r)
    m = m.copy()
    r = r.copy()
    scale = np.abs(m).max()
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(m[col:, col])))
        if abs(m[pivot, col]) <= scale * 1e-12:
            return None
        if pivot != col:
            m[[col, pivot]] = m[[pivot, col]]
            r[[col, pivot]] = r[[pivot, col]]
        for row in range(col + 1, n):
            t = m[row, col] / m[col, col]
            if t == 0.0:
                continue
            m[row, col:] -= t * m[col, col:]
            r[row] -= t * r[col]
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        x[row] = (r[row] - m[row, row + 1:] @ x[row + 1:]) / m[row, row]
    return x


class _Counts:
    """Means over a mask's pixels alone, in boxes of PRE and LOCAL."""

    def __init__(self, weight):
        self.weight = weight
        self.count = {PRE: _box_sum(weight, PRE), LOCAL: _box_sum(weight, LOCAL)}

    def mean(self, a, radius):
        c = self.count[radius]
        s = _box_sum(a * (self.weight if a.ndim == 2 else self.weight[..., None]), radius)
        if a.ndim == 3:
            c = c[..., None]
        return np.where(c > 0.5, s / np.maximum(c, 0.5), 0.0)


def _texture_of(planes, counts):
    """Spread of the log luma of `planes` (H, W, 3) round its local mean."""
    luma = np.log(np.maximum(planes.sum(axis=2) / 3.0, 1e-6))
    mean = counts.mean(luma, LOCAL)
    sq = (luma - mean) ** 2
    return np.sqrt(np.maximum(counts.mean(sq, LOCAL), 0.0))


def _unchanged(fill):
    return np.array(fill, dtype=np.float32, copy=True), None


def evened(image, hole, fill):
    """`fill` (sRGB-encoded, 0..1) with its pixels under `hole` brought to
    the smooth surface the picture round the hole describes.  Returns
    (new float32 array, info or None); info is a dict with 'mean' (per
    channel gain put into the hole), 'largest' (largest change, in logs)
    and 'strength' (0..1)."""
    image = np.asarray(image)
    fill = np.asarray(fill)
    hole = np.asarray(hole)
    if (image.ndim != 3 or fill.shape != image.shape or hole.shape != image.shape[:2]):
        return _unchanged(fill)
    h, w = hole.shape
    hm = hole > 0
    if not hm.any():
        return _unchanged(fill)
    ys, xs = np.nonzero(hm)
    reach = BAND[1] + PRE + LOCAL
    x0, y0 = max(xs.min() - reach, 0), max(ys.min() - reach, 0)
    x1, y1 = min(xs.max() + reach, w - 1), min(ys.max() + reach, h - 1)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    n = cw * ch
    sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    mask = hm[sl]
    inside = mask.astype(np.float64)
    outside = 1.0 - inside

    def linear(img, weight):
        lin = colorspace.srgb_eotf(np.asarray(img[sl], dtype=np.float32))
        v = lin.astype(np.float64) + float(np.float32(PEDESTAL))
        return np.where(weight[..., None] > 0.0, v, 0.0)

    # Each side low-passed over its own pixels, and its texture.
    counts = _Counts(outside)
    picture = counts.mean(linear(image, outside), PRE)
    texture = _texture_of(picture, counts)
    near = _ring(mask, BAND[0] - 1)
    band = _ring(mask, BAND[1]) & ~near
    in_band = int(band.sum())

    filled = linear(fill, inside)
    fcounts = _Counts(inside)
    low = fcounts.mean(filled, PRE)
    fill_texture = _texture_of(low, fcounts)

    # The hole's middle and size, for the surface's coordinates.
    hy, hx = np.nonzero(mask)
    count = float(hx.size)
    mx, my = hx.mean(), hy.mean()
    size = max((max(hx.max() - hx.min(), hy.max() - hy.min())) / 2.0 + BAND[1], 1.0)
    gx, gy = np.meshgrid(np.arange(cw, dtype=np.float64), np.arange(ch, dtype=np.float64))
    U, V = (gx - mx) / size, (gy - my) / size
    T = np.stack([np.ones_like(U), U, V, U * U, U * V, V * V], axis=-1)  # (ch, cw, 6)

    # The surface: the band's smooth pixels, fitted robustly.
    pmask = band & (texture < SMOOTH)
    pidx = np.flatnonzero(pmask.ravel())
    if pidx.size < LEAST:
        return _unchanged(fill)
    Tp = T.reshape(n, 6)[pidx]
    logs = np.log(np.maximum(picture.reshape(n, 3)[pidx], 1e-6))
    center = np.array([_median(logs[:, k]) for k in range(3)])
    weights = np.all(np.abs(logs - center) < SAME, axis=1).astype(np.float64)
    at = np.full(n, -1, dtype=np.int64)
    at[pidx] = np.arange(pidx.size)
    grain = None
    surface = np.zeros((3, 6))
    residuals = np.zeros_like(logs)
    for _ in range(PASSES):
        wt = weights
        m = (Tp * wt[:, None]).T @ Tp
        for k in range(3):
            r = (Tp * wt[:, None]).T @ logs[:, k]
            c = _solve(m, r)
            if c is None:
                return _unchanged(fill)
            surface[k] = c
        residuals = logs - Tp @ surface.T
        if grain is None:
            pairs = []
            for off in (LAG, LAG * cw):
                o = pidx + off
                ok = o < n
                ok[ok] = at[o[ok]] >= 0
                pairs.append((np.flatnonzero(ok), at[o[ok]]))
            grain = np.zeros(3)
            for k in range(3):
                d = np.concatenate([np.abs(residuals[a, k] - residuals[b, k])
                                    for a, b in pairs])
                grain[k] = 1.4826 * _median(d) / np.sqrt(2.0)
        cutoff = np.array([
            TUKEY * max(min(1.4826 * _median(np.abs(residuals[:, k])), GRAINS * grain[k]), FLOOR)
            for k in range(3)])
        u = (np.abs(residuals) / cutoff).max(axis=1)
        weights = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
    total = weights.sum()
    spread = max(np.sqrt((weights * residuals[:, k] ** 2).sum() / max(total, 1e-12))
                 for k in range(3))
    inl = weights > 0.0
    inliers = int(inl.sum())
    share = inliers / max(in_band, 1)
    near_pts = np.all(np.abs(residuals) < SAME, axis=1)
    soft = float((~inl & near_pts).sum()) / pidx.size
    px, py = pidx % cw - mx, pidx // cw - my
    a = (np.arctan2(py[inl], px[inl]) + np.pi) / (2 * np.pi) * 8.0
    octants = np.bincount(np.minimum(a.astype(np.int64), 7), minlength=8)
    held = int((octants >= 0.02 * inliers).sum())
    strength = _ramp(FIT[1], FIT[0], spread) * _ramp(SHARE[0], SHARE[1], share)
    unshown = _ramp(SOFT[1], SOFT[0], soft)
    if strength <= 0.0 or inliers < LEAST or held < OCTANTS:
        return _unchanged(fill)

    # What the band shows past the quadratic, carried a short way in.
    carry = np.zeros((n, 4))
    carry[pidx[near_pts], :3] = residuals[near_pts]
    carry[pidx[near_pts], 3] = 1.0
    carry = carry.reshape(ch, cw, 4)
    carry = _box_sum(_box_sum(carry, CARRY), CARRY)
    carried, carried_n = carry[..., :3], carry[..., 3]
    enough = 0.25 * float(2 * CARRY + 1) ** 4
    reached = np.minimum(carried_n / enough, 1.0)
    shown = np.where((carried_n > 0.5)[..., None],
                     carried / np.maximum(carried_n, 0.5)[..., None] * reached[..., None],
                     0.0)
    trust = reached + (1.0 - reached) * unshown

    # The fill's side, read where smooth and near the reference.
    q = np.log(np.maximum(low, 1e-6)) - T @ surface.T - shown
    on_b = mask & (fill_texture < SMOOTH) & np.all(np.abs(q) < SAME, axis=2)
    if int(on_b.sum()) < LEAST:
        return _unchanged(fill)
    on = on_b.astype(np.float64)
    off = np.where(on_b[..., None], q * trust[..., None], 0.0)

    def smooth(a):
        return _box_sum(_box_sum(a, RADIUS), RADIUS)

    on_s = smooth(on)
    hole_s = smooth(inside)
    off_s = smooth(off)

    with np.errstate(divide='ignore', invalid='ignore'):
        field = off_s / on_s[..., None]
    sel = on_b & (on_s > 0.5)
    samples = float(sel.sum())
    scatter = ((off[sel] - field[sel]) ** 2).sum(axis=0)
    full = float(2 * RADIUS + 1) ** 4
    noise = np.sqrt(scatter / max(samples, 1.0)) * _kernel()

    elig = mask & (on_s >= FEW * full)
    with np.errstate(divide='ignore', invalid='ignore'):
        cover = np.minimum(2.0 * on_s / np.maximum(hole_s, 1.0), 1.0)
        thin = np.sqrt(full / on_s)
        kept = 1.0 - (SHRINK * noise * thin[..., None]
                      / np.maximum(np.abs(field), 1e-12)) ** 2
        c = np.clip(field * np.maximum(kept, 0.0) * cover[..., None], -LIMIT, LIMIT) * strength
    c = np.where(elig[..., None], c, 0.0)
    c = np.nan_to_num(c)
    moved_px = np.any(c != 0.0, axis=2)
    moved = int(moved_px.sum())
    if moved == 0:
        return _unchanged(fill)

    out = np.array(fill, dtype=np.float32, copy=True)
    v = filled * np.exp(-c) - float(np.float32(PEDESTAL))
    new = colorspace.srgb_oetf(v.astype(np.float32))
    region = out[sl]
    region[moved_px] = new[moved_px]
    out[sl] = region
    info = {
        'mean': [float(x) for x in np.exp(-c[moved_px].sum(axis=0) / count)],
        'largest': float(np.abs(c[moved_px]).max()),
        'strength': float(strength),
    }
    return out, info
