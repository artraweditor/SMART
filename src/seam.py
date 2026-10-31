"""The level under the hole: a numpy port of greycard's ``remove/seam.rs``.

The inpainted fill is compared with the picture across the hole's edge, in
linear light: a band of the picture set back from the edge (GAP) and a band of
the fill just inside it, in each of SECTORS sectors round the hole, are fitted
with a tilt shared by all of them, a level of the picture's in each sector and
a step on the fill's side of each.  Where the sectors agree on one step (and
it is clear of its own error, read also on the picture alone across
pseudo-edges), it is multiplied into the whole fill; otherwise the fill is
left as it is.

    fill_out, info = matched(image, hole, fill)

``image`` and ``fill`` are float32 (H, W, 3), sRGB-encoded, 0..1; ``hole`` is
(H, W), nonzero in the hole.  ``info`` is None when the fill is left as it is,
else a dict with ``gain`` (3 floats) and the reading's details.
"""

import numpy as np
from scipy import ndimage

import colorspace
import lama

BAND = 4
GAP = 8
OUTER = 8
SECTORS = 16
SHARE = 0.25 / SECTORS
COUNTED = 10
AGREE = (2.0, 0.75)
RESOLVED = 1.0
NULLS = [8, 12, 16]
TILT = 2
PASSES = 3
HUBER = 1.345
TEXTURE = 0.05
GAINS = (0.8, 1.25)
PEDESTAL = 0.01


def to_linear(v):
    return colorspace.srgb_eotf(v)


def to_srgb(v):
    return colorspace.srgb_oetf(v)


def ring(hole, r):
    """Pixels outside `hole` (bool) within a square's reach `r` of it."""
    hole = np.asarray(hole, dtype=bool)
    if r <= 0:
        return np.zeros_like(hole)
    d = hole.view(np.uint8)
    d = ndimage.maximum_filter1d(d, 2 * r + 1, axis=1, mode='constant', cval=0)
    d = ndimage.maximum_filter1d(d, 2 * r + 1, axis=0, mode='constant', cval=0)
    return d.astype(bool) & ~hole


def crop_round(hole, reach):
    """(y0, y1, x0, x1) (inclusive) of the hole's box grown by `reach`, held
    to the window; None for no hole."""
    h, w = hole.shape
    ys = np.flatnonzero(hole.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(hole.any(axis=0))
    return (max(int(ys[0]) - reach, 0), min(int(ys[-1]) + reach, h - 1),
            max(int(xs[0]) - reach, 0), min(int(xs[-1]) + reach, w - 1))


def _solve(m, r):
    """m x = r by Gaussian elimination with partial pivoting (r is n x c);
    None when m is singular or near it."""
    m = m.copy()
    r = r.copy()
    n = m.shape[0]
    scale = np.abs(m).max() if m.size else 0.0
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(m[col:, col])))
        if not abs(m[pivot, col]) > scale * 1e-12:
            return None
        if pivot != col:
            m[[col, pivot]] = m[[pivot, col]]
            r[[col, pivot]] = r[[pivot, col]]
        if col + 1 < n:
            t = m[col + 1:, col] / m[col, col]
            m[col + 1:, col:] -= t[:, None] * m[col, col:][None, :]
            r[col + 1:] -= t[:, None] * r[col][None, :]
    x = np.zeros_like(r)
    for row in range(n - 1, -1, -1):
        s = m[row, row + 1:] @ x[row + 1:] if row + 1 < n else 0.0
        x[row] = (r[row] - s) / m[row, row]
    return x


def _weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    v = values[order]
    cw = np.cumsum(weights[order])
    half = weights.sum() / 2.0
    hit = np.flatnonzero(cw >= half)
    return float(v[hit[0]]) if hit.size else 0.0


def _read(image, hole, fill, origin):
    """The edge of `hole` (bool) read across from `image` to `fill` (both
    sRGB, float32): a dict, or None where there is nothing to read.
    `origin` = (x, y) of the window in the one the hole was drawn in."""
    h, w = hole.shape
    if not hole.any() or lama.depth(hole) < BAND:
        return None
    known = ~hole
    near = ring(hole, GAP)
    outer = ring(hole, GAP + OUTER) & ~near
    inner = ring(known, BAND)

    ys, xs = np.nonzero(hole)
    gx = xs + origin[0]
    gy = ys + origin[1]
    mx, my = gx.mean(), gy.mean()
    size = max((max(gx.max() - gx.min(), gy.max() - gy.min()) + 1) / 2.0
               + (GAP + OUTER), 1.0)

    inside_px = inner & hole  # inner is outside `known`, i.e. in the hole
    band_mask = outer | inside_px
    by, bx = np.nonzero(band_mask)  # row-major order
    inside = inside_px[by, bx]
    u = (bx + origin[0] - mx) / size
    v = (by + origin[1] - my) / size
    sec = np.minimum(
        ((np.arctan2(v, u) + np.pi) / (2.0 * np.pi) * SECTORS).astype(np.int64),
        SECTORS - 1)

    counts = np.zeros((2, SECTORS))
    counts[0] = np.bincount(sec[~inside], minlength=SECTORS)
    counts[1] = np.bincount(sec[inside], minlength=SECTORS)
    mid_u = np.bincount(sec[inside], weights=u[inside], minlength=SECTORS)
    mid_v = np.bincount(sec[inside], weights=v[inside], minlength=SECTORS)
    totals = counts.sum(axis=1)
    ok = np.ones(SECTORS, dtype=bool)
    for s in range(2):
        ok &= (counts[s] > 0) & (counts[s] >= SHARE * totals[s])
    counted = np.flatnonzero(ok)
    nc = len(counted)
    if nc < COUNTED:
        return None

    place_of = np.full(SECTORS, -1, dtype=np.int64)
    place_of[counted] = np.arange(nc)

    # Band pixels in counted sectors only (the others are out of the fit).
    keep = ok[sec]
    inside, sec, u, v = inside[keep], sec[keep], u[keep], v[keep]
    by, bx = by[keep], bx[keep]
    src = np.where(inside[:, None], fill[by, bx], image[by, bx])
    z = to_linear(src).astype(np.float64)  # (n, 3)
    n = len(u)
    p = place_of[sec]
    params = TILT + 2 * nc
    X = np.zeros((n, params))
    X[:, 0] = u
    X[:, 1] = v
    rows = np.arange(n)
    X[rows, TILT + p] = 1.0
    X[rows[inside], TILT + nc + p[inside]] = 1.0

    def fit_with(weights):
        a = X.T @ (X * weights[:, None])
        rhs = X.T @ (z * weights[:, None])
        sol = _solve(a, rhs)  # (params, 3)
        return None if sol is None else sol.T  # (3, params)

    def model_at(x, inside_, p_, u_, v_):
        # x (3, params); arrays broadcast -> (3, ...)
        u_, v_, p_ = np.atleast_1d(u_), np.atleast_1d(v_), np.atleast_1d(p_)
        m = x[:, 0:1] * u_ + x[:, 1:2] * v_ + x[:, TILT + p_]
        if np.ndim(inside_) == 0:
            if inside_:
                m = m + x[:, TILT + nc + p_]
        else:
            m = m + np.where(inside_, x[:, TILT + nc + p_], 0.0)
        return m

    weights = np.ones(n)
    fit = None
    for pas in range(PASSES):
        fit = fit_with(weights)
        if fit is None:
            break
        if pas + 1 == PASSES:
            break
        res = np.abs((z.T - model_at(fit, inside, p, u, v)).sum(axis=0))
        pos = np.sort(res[res > 0])
        if pos.size == 0:
            break
        mid = pos[pos.size // 2]
        scale = HUBER * 1.4826 * mid
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.where(res > scale, scale / res, 1.0)
    x = fit
    if x is None:
        return None

    pedestal = float(PEDESTAL)
    r = (z.T - model_at(x, inside, p, u, v)).sum(axis=0) / 3.0
    r2 = r * r
    # spread[side][sector]: count and sum of squares
    cnt = np.zeros((2, SECTORS))
    sq = np.zeros((2, SECTORS))
    for side in (0, 1):
        sel = inside == bool(side)
        cnt[side] = np.bincount(sec[sel], minlength=SECTORS)
        sq[side] = np.bincount(sec[sel], weights=r2[sel], minlength=SECTORS)

    nfill = counts[1]
    sigmas = np.full(SECTORS, np.inf)
    mids = {}
    for s in counted:
        pp = int(place_of[s])
        at = (mid_u[s] / nfill[s], mid_v[s] / nfill[s])
        mids[int(s)] = at
        texture = 0.0
        for side in (False, True):
            lvl = model_at(x, side, pp, at[0], at[1]).sum() / 3.0 + pedestal
            o = int(side)
            texture += sq[o, s] / max(cnt[o, s] - 1.0, 1.0) / max(lvl, pedestal) ** 2
        sigmas[s] = np.sqrt(TEXTURE * TEXTURE + texture)

    weights = weights / sigmas[sec] ** 2
    x = fit_with(weights)
    if x is None:
        return None

    total = 0.0
    sectors = []  # (step[3], sigma, weight)
    for s in counted:
        pp = int(place_of[s])
        at = mids[int(s)]
        n_s = nfill[s]
        sigma = sigmas[s]
        weight = n_s / (sigma * sigma)
        total += weight
        picture = model_at(x, False, pp, at[0], at[1]).ravel() + pedestal
        filled = model_at(x, True, pp, at[0], at[1]).ravel() + pedestal
        if np.all(picture > 0) and np.all(filled > 0):
            step = np.log(picture / filled)
            if np.all(np.isfinite(step)):
                sectors.append((step, sigma, weight))
    if not sectors:
        return None
    steps = np.array([s[0] for s in sectors])
    sig = np.array([s[1] for s in sectors])
    wt = np.array([s[2] for s in sectors])
    weight_sum = wt.sum()
    if weight_sum <= 0.0:
        return None

    lumas_raw = steps.mean(axis=1)
    median = np.array([_weighted_median(steps[:, k], wt) for k in range(3)])
    middle = _weighted_median(lumas_raw, wt)
    agreeing = wt[np.abs(lumas_raw - middle) <= AGREE[0] * sig].sum()
    reach = AGREE[0] * sig
    held = np.clip(steps, median[None, :] - reach[:, None],
                   median[None, :] + reach[:, None])
    mean = (wt[:, None] * held).sum(axis=0) / weight_sum

    lumas = held.sum(axis=1) / 3.0
    level = float(mean.sum() / 3.0)
    scatter = (wt * (lumas - level) ** 2).sum() / weight_sum
    independent = weight_sum * weight_sum / (wt ** 2).sum()
    error = float(np.sqrt(scatter / max(independent - 1.0, 1.0)))
    return dict(mean=mean, level=level, error=error,
                agreeing=float(agreeing / total),
                asks=np.exp(steps), sigmas=sig)


def matched(image, hole, fill):
    """`fill` with its pixels under `hole` brought to `image`'s level across
    the hole's edge, and an info dict (with 'gain'); `fill` copied as it was
    and None where it is left alone."""
    image = np.asarray(image, dtype=np.float32)
    fill = np.asarray(fill, dtype=np.float32)
    hole = np.asarray(hole)
    left = fill.copy()
    if (image.ndim != 3 or image.shape[2] != 3 or fill.shape != image.shape
            or hole.shape != image.shape[:2]):
        return left, None
    hmask = hole > 0
    reach = max(NULLS) + GAP + OUTER
    box = crop_round(hmask, reach)
    if box is None:
        return left, None
    y0, y1, x0, x1 = box
    sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    image_c, fill_c, hole_c = image[sl], fill[sl], hmask[sl]
    origin = (x0, y0)

    rd = _read(image_c, hole_c, fill_c, origin)
    if rd is None:
        return left, None

    nulls = []
    for d in NULLS:
        grown = hole_c | ring(hole_c, d)
        nr = _read(image_c, grown, image_c, origin)
        if nr is not None:
            nulls.append(nr['level'])
    if not nulls:
        return left, None
    bias = sum(l * l for l in nulls) / len(nulls)
    error = float(np.sqrt(rd['error'] ** 2 + bias))
    mean, level = rd['mean'], rd['level']
    lo, hi = np.log(float(np.float32(GAINS[0]))), np.log(float(np.float32(GAINS[1])))
    if rd['agreeing'] < AGREE[1] or np.any(~((mean >= lo) & (mean <= hi))):
        return left, None
    if abs(level) > RESOLVED * error:
        kept = 1.0 - (RESOLVED * error / abs(level)) ** 2
    else:
        kept = 0.0
    if kept <= 0.0:
        return left, None
    gain = np.exp(mean * kept).astype(np.float32)
    if not np.all(np.isfinite(gain)):
        return left, None

    out = left
    lin = to_linear(fill[hmask])  # (n, 3) float32
    ped = np.float32(PEDESTAL)
    out[hmask] = to_srgb((lin + ped) * gain[None, :] - ped)
    info = dict(gain=[float(g) for g in gain], level=level, error=error,
                kept=kept, agreeing=rd['agreeing'])
    return out, info
