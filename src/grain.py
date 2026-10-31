"""Regrain: put the picture's own noise back under an inpainted hole.

A numpy port of greycard's ``develop/noise.rs`` picture-noise part
(``estimate_picture_noise``, ``add_picture_noise``) and of ``regrain`` in
``src_ai.rs``.  The noise of the picture is measured on the window's known
pixels on orthonormal luma/chroma axes (variance a line in the pixel's mean
level, per axis, plus the grain's coarseness), the same is measured on the
fill, and only the shortfall is drawn (white Gaussian noise smoothed by a
Gaussian whose width reproduces the measured shape) under the hole.
"""
import numpy as np
from scipy import ndimage

BLOCK = 16
BINS = 32
MIN_BLOCKS_PER_BIN = 8
RESIDUAL_GAIN = 1.25
MAX_GRAIN_WIDTH = 3.0
SHAPE_CEILING = 1.5
PEAK_FRACTION = 0.99
QUIET_FLOOR = 16.0
FLOOR_FRACTION = 0.25
STILL = 1e-14
MID_GREY = 0.18
SILENT_SIGMA = 1e-5

_S3, _S2, _S6 = np.sqrt(3.0), np.sqrt(2.0), np.sqrt(6.0)


def _to_yuv(r, g, b):
    return (r + g + b) / _S3, (r - b) / _S2, (r - 2 * g + b) / _S6


def _from_yuv(y, u, v):
    return (y / _S3 + u / _S2 + v / _S6,
            y / _S3 - 2 * v / _S6,
            y / _S3 - u / _S2 + v / _S6)


class PictureNoise:
    """Per axis (luma, chroma U, chroma V): variance = a*level + b, plus the
    per-level-bin shape sums."""

    def __init__(self):
        self.a = np.zeros(3)
        self.b = np.zeros(3)
        self.sums = np.zeros((BINS, 2, 2))
        self.blocks = np.zeros((BINS, 2), dtype=np.int64)

    def variance(self, k, level):
        return np.maximum(self.a[k] * level + self.b[k], 0.0)

    def is_silent(self):
        return all(np.sqrt(self.variance(k, MID_GREY)) < SILENT_SIGMA
                   for k in range(3))

    def ratio_at(self, level):
        """The grain's (luma, chroma) ratio of residual spreads at distance
        one over two, pooled from the level's bin and its neighbours."""
        b = _level_bin(level)
        out = []
        for g in range(2):
            lo, hi = max(b - 1, 0), min(b + 1, BINS - 1)
            n, f = self.sums[lo:hi + 1, g].sum(axis=0)
            c = self.blocks[lo:hi + 1, g].sum()
            if c < MIN_BLOCKS_PER_BIN:
                n, f = self.sums[:, g].sum(axis=0)
            out.append(float(np.sqrt(n / f)) if f > 0 else 1.0)
        return out


def _level_bin(level):
    return int(min(np.sqrt(max(float(level), 0.0)) * BINS, BINS - 1))


def _level_bins(level):
    return np.minimum((np.sqrt(np.maximum(level, 0.0)) * BINS).astype(np.int64),
                      BINS - 1)


def _fit(levels, var, keep, noise):
    """Weighted line through the quietest quarter of each level bin."""
    bins = _level_bins(levels)
    for k in range(3):
        sel = keep[k]
        lv, vr, bn = levels[sel], var[k][sel], bins[sel]
        sw = swx = swy = swxx = swxy = 0.0
        used = 0
        for b in range(BINS):
            m = bn == b
            cnt = int(m.sum())
            if cnt < MIN_BLOCKS_PER_BIN:
                continue
            order = np.argsort(vr[m], kind="stable")
            q = order[:max(cnt // 4, 2)]
            mean = float(lv[m][q].astype(np.float64).mean())
            v = float(vr[m][q].astype(np.float64).mean())
            n = float(BLOCK * BLOCK)
            bias = 1.0 - 1.271 * np.sqrt(2.0 / n)
            v /= max(bias, 0.5)
            w = float(cnt)
            sw += w
            swx += w * mean
            swy += w * v
            swxx += w * mean * mean
            swxy += w * mean * v
            used += 1
        if used == 0:
            noise.a[k] = noise.b[k] = 0.0
            continue
        det = sw * swxx - swx * swx
        if used >= 2 and abs(det) > 1e-18:
            a = (sw * swxy - swx * swy) / det
            b = (swxx * swy - swx * swxy) / det
        else:
            a, b = 0.0, swy / sw
        if a < 0:
            a, b = 0.0, swy / sw
        if b < 0:
            b = 0.0
            a = swxy / swxx if swxx > 0 else 0.0
        noise.a[k], noise.b[k] = a, b
    return noise


def _blocksum(arr, nby, nbx):
    sub = arr[2:2 + nby * BLOCK, 2:2 + nbx * BLOCK]
    return sub.reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3), dtype=np.float64)


def estimate_picture_noise(data, known, luma):
    """data: (H, W, 3) float; known: (H, W) bool.  Returns a PictureNoise
    (silent, all zeros, if the picture is too small)."""
    H, W = known.shape
    pn = PictureNoise()
    if W < BLOCK + 4 or H < BLOCK + 4:
        return pn
    data = np.asarray(data, dtype=np.float32)
    nby = len(range(2, H - 2 - BLOCK + 1, BLOCK))
    nbx = len(range(2, W - 2 - BLOCK + 1, BLOCK))
    if nby == 0 or nbx == 0:
        return pn
    # blocks whose pixels and the 2-pixel ring round them are all known
    unk = np.zeros((H + 1, W + 1), dtype=np.int64)
    unk[1:, 1:] = np.cumsum(np.cumsum(~known, axis=0), axis=1)
    y0 = 2 + BLOCK * np.arange(nby) - 2
    x0 = 2 + BLOCK * np.arange(nbx) - 2
    y1, x1 = y0 + BLOCK + 4, x0 + BLOCK + 4
    cnt = (unk[np.ix_(y1, x1)] + unk[np.ix_(y0, x0)]
           - unk[np.ix_(y0, x1)] - unk[np.ix_(y1, x0)])
    ok = cnt == 0
    if not ok.any():
        return pn
    peak = np.array([data[..., c][known].max() if known.any() else -np.inf
                     for c in range(3)], dtype=np.float32)
    near_peak = np.where(peak > 0, peak * np.float32(PEAK_FRACTION), np.float32(np.inf))
    toward = np.where(peak > 0, peak * np.float32(FLOOR_FRACTION), np.float32(np.inf))

    r, g, b = data[..., 0], data[..., 1], data[..., 2]
    ax = _to_yuv(r, g, b)
    lum = (np.float32(luma[0]) * r + np.float32(luma[1]) * g
           + np.float32(luma[2]) * b)

    def resid(p, d):
        # p minus the mean of its 4 neighbours d away, on the interior
        out = np.zeros_like(p)
        c = p[d:H - d, d:W - d]
        nb = (p[d:H - d, 0:W - 2 * d] + p[d:H - d, 2 * d:W]
              + p[0:H - 2 * d, d:W - d] + p[2 * d:H, d:W - d]) * np.float32(0.25)
        out[d:H - d, d:W - d] = c - nb
        return out

    far, near = [], []
    for k in range(3):
        for d, dst in ((2, far), (1, near)):
            rr = resid(ax[k], d)
            dst.append(_blocksum(rr * rr, nby, nbx) / (BLOCK * BLOCK) / RESIDUAL_GAIN)
    lum_n = []
    for d in (1, 2):
        rr = resid(lum, d)
        lum_n.append(_blocksum(rr * rr, nby, nbx) / (BLOCK * BLOCK))
    mean_level = _blocksum((r + g + b) / np.float32(3.0), nby, nbx) / (BLOCK * BLOCK)

    clipped = np.zeros((H, W), dtype=bool)
    highp = np.zeros((H, W), dtype=bool)
    for c in range(3):
        clipped |= data[..., c] >= near_peak[c]
        highp |= data[..., c] >= toward[c]
    clipped_b = _blocksum(clipped, nby, nbx) > 0
    high_b = _blocksum(highp, nby, nbx) > 0
    ok &= ~clipped_b
    if not ok.any():
        return pn

    levels = mean_level[ok].astype(np.float32)
    far = [f[ok] for f in far]
    near = [n[ok] for n in near]
    lum_n = [n[ok] for n in lum_n]
    high = high_b[ok]

    keep1 = [f > STILL for f in far]
    first = _fit(levels, far, keep1, PictureNoise())
    lv0 = np.maximum(levels, 0.0)
    kept = []
    for k in range(3):
        fv = np.maximum(first.a[k] * lv0 + first.b[k], 0.0)
        quiet_ok = ~high | (far[k] * QUIET_FLOOR >= fv)
        kept.append(quiet_ok)
    keep2 = [(far[k] > STILL) & kept[k] for k in range(3)]
    pn = _fit(levels, far, keep2, pn)

    bins = _level_bins(levels)
    for k in range(3):
        fit = np.maximum(pn.a[k] * lv0 + pn.b[k], 0.0)
        sel = (fit > STILL) & (far[k] > STILL) & kept[k] & (far[k] <= SHAPE_CEILING * fit)
        grp = 0 if k == 0 else 1
        n_, f_ = (lum_n[0], lum_n[1]) if k == 0 else (near[k], far[k])
        bb = bins[sel]
        np.add.at(pn.sums[:, grp, 0], bb, n_[sel].astype(np.float64))
        np.add.at(pn.sums[:, grp, 1], bb, f_[sel].astype(np.float64))
        np.add.at(pn.blocks[:, grp], bb, 1)
    return pn


def grain_kernel(width):
    """Sampled Gaussian out to 3 widths, L2-normalised."""
    if width <= 0:
        return np.ones(1, dtype=np.float32)
    r = int(np.ceil(3.0 * width))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / np.float32(2.0 * width * width)).astype(np.float32)
    return (k / np.sqrt((k * k).sum())).astype(np.float32)


def _residual_gain(kernel, d):
    def a(lag):
        return float(np.dot(kernel[:len(kernel) - lag], kernel[lag:])) if lag < len(kernel) else 0.0
    near, far = a(d), a(2 * d)
    return RESIDUAL_GAIN - 2 * near + far / 4 + near * near / 2


def _shape_ratio(width):
    k = grain_kernel(width)
    return np.sqrt(_residual_gain(k, 1) / _residual_gain(k, 2))


def _width_for(ratio):
    if ratio != ratio or ratio >= 1.0:
        return 0.0
    if ratio <= _shape_ratio(MAX_GRAIN_WIDTH):
        return MAX_GRAIN_WIDTH
    lo, hi = 0.0, MAX_GRAIN_WIDTH
    for _ in range(24):
        mid = (lo + hi) / 2
        if _shape_ratio(mid) > ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _draw_gain(kernel):
    return min(RESIDUAL_GAIN / _residual_gain(kernel, 2), 1.0)


def _shortfall_widths(out, mask, model, present):
    if mask.any():
        level = max(float(out[mask].astype(np.float64).sum(axis=1).mean() / 3.0), 0.0)
    else:
        level = 0.0
    fr, fl = model.ratio_at(level), present.ratio_at(level)
    widths = []
    for k in range(3):
        g = 0 if k == 0 else 1
        far_m = float(model.variance(k, level))
        far_p = float(present.variance(k, level))
        far = far_m - far_p
        if far <= 0:
            widths.append(0.0)
            continue
        near = far_m * fr[g] ** 2 - far_p * fl[g] ** 2
        widths.append(_width_for(float(np.sqrt(max(near, 0.0) / far))))
    return widths


def add_picture_noise(out, mask, model, present, seed):
    """Add the shortfall noise in place to float32 `out` (H, W, 3) where
    `mask` (bool) is set.  Returns the kernel widths."""
    H, W = mask.shape
    widths = _shortfall_widths(out, mask, model, present)
    kernels = [grain_kernel(w) for w in widths]
    gains = [_draw_gain(k) for k in kernels]
    pad = max(len(k) // 2 for k in kernels)
    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    white = rng.standard_normal((3, H + 2 * pad, W + 2 * pad), dtype=np.float32)
    fields = []
    for k in range(3):
        f = white[k]
        if len(kernels[k]) > 1:
            f = ndimage.convolve1d(f, kernels[k], axis=1, mode="constant")
            f = ndimage.convolve1d(f, kernels[k], axis=0, mode="constant")
        fields.append(f[pad:pad + H, pad:pad + W])
    level = np.maximum((out[..., 0] + out[..., 1] + out[..., 2]) / np.float32(3.0), 0.0)
    draw = []
    for k in range(3):
        var = model.variance(k, level) - present.variance(k, level)
        amp = np.sqrt(np.maximum(var, 0.0) * gains[k])
        draw.append(np.where(var > 0, amp * fields[k], 0.0).astype(np.float32))
    add = _from_yuv(*draw)
    for c in range(3):
        out[..., c] = np.where(mask, out[..., c] + add[c], out[..., c])
    return widths


def regrain(frame, out, hole, luma, seed):
    """Return a copy of `out` (H, W, 3 float32) with the frame's noise added
    under `hole` (nonzero = hole); `frame` holds the original picture."""
    frame = np.asarray(frame, dtype=np.float32)
    res = np.array(out, dtype=np.float32, copy=True)
    mask = np.asarray(hole) != 0
    luma = np.asarray(luma, dtype=np.float64)
    if res.ndim != 3 or res.shape[2] != 3 or not mask.any():
        return res
    known = ~mask
    fr = estimate_picture_noise(frame, known, luma)
    if fr.is_silent():
        return res
    present = estimate_picture_noise(res, mask, luma)  # the fill, over the hole
    add_picture_noise(res, mask, fr, present, seed)
    res[~mask] = np.asarray(out, dtype=np.float32)[~mask]
    return res
