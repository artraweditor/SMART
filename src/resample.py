"""Small numpy-only image resampling helpers (no OpenCV dependency).

Images are float32 arrays of shape (H, W) or (H, W, C).
"""

import numpy as np


def _cover(i, scale, length):
    """Source pixels covered by new pixel `i` (port of greycard's `cover`)."""
    a = min(int(i * scale), length - 1)
    b = int(np.ceil((i + 1) * scale))
    b = min(max(b, a + 1), length)
    return a, b


def _area_matrix(new, old):
    """(new x old) matrix of box means, as greycard's `shrink`."""
    scale = old / new
    m = np.zeros((new, old), dtype=np.float32)
    for i in range(new):
        a, b = _cover(i, scale, old)
        m[i, a:b] = 1.0 / (b - a)
    return m


def _linear_matrix(new, old):
    """(new x old) bilinear matrix, half-pixel centers, edge clamping."""
    m = np.zeros((new, old), dtype=np.float32)
    scale = old / new
    for i in range(new):
        p = (i + 0.5) * scale - 0.5
        p0 = int(np.floor(p))
        f = p - p0
        a = min(max(p0, 0), old - 1)
        b = min(max(p0 + 1, 0), old - 1)
        m[i, a] += 1.0 - f
        m[i, b] += f
    return m


def _apply(img, my, mx):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return my @ img @ mx.T
    out = np.empty((my.shape[0], mx.shape[0], img.shape[2]), np.float32)
    for c in range(img.shape[2]):
        out[:, :, c] = my @ img[:, :, c] @ mx.T
    return out


def area(img, width, height):
    """Downscale by the mean of the source pixels each new pixel covers."""
    h, w = img.shape[:2]
    return _apply(img, _area_matrix(height, h), _area_matrix(width, w))


def bilinear(img, width, height):
    """Resample bilinearly (half-pixel centers, clamped edges)."""
    h, w = img.shape[:2]
    return _apply(img, _linear_matrix(height, h), _linear_matrix(width, w))


def any_covered(mask, width, height):
    """Downscale a binary mask: a new pixel is set if any pixel it covers is."""
    h, w = mask.shape
    scale_y, scale_x = h / height, w / width
    m = (np.asarray(mask) > 0).astype(np.float32)
    # reduce with explicit boxes via cumulative sums
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    out = np.zeros((height, width), dtype=np.float32)
    xs =[_cover(x, scale_x, w) for x in range(width)]
    x0 = np.array([a for a, _ in xs])
    x1 = np.array([b for _, b in xs])
    for y in range(height):
        y0, y1 = _cover(y, scale_y, h)
        tot = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
        out[y] = tot > 0.5
    return out


def box(img, radius, weights=None):
    """Box filter of the given radius (window 2r+1), edge-normalized.

    With `weights` (HxW), averages only over weighted pixels: returns
    sum(img*w)/sum(w) over the window (0 where no weight).
    """
    img = np.asarray(img, dtype=np.float64)
    r = int(radius)
    if r <= 0:
        return img.astype(np.float32)

    def bsum(a):
        pad = [(r + 1, r)] + [(0, 0)] * (a.ndim - 1)
        c = np.cumsum(np.pad(a, pad), axis=0)
        a = c[2 * r + 1:] - c[:-(2 * r + 1)]
        pad = [(0, 0), (r + 1, r)] + [(0, 0)] * (a.ndim - 2)
        c = np.cumsum(np.pad(a, pad), axis=1)
        return c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]

    if weights is None:
        weights = np.ones(img.shape[:2], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ws = w if img.ndim == 2 else w[..., None]
    num = bsum(img * ws)
    den = bsum(w)
    if img.ndim == 3:
        den = den[..., None]
    out = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return out.astype(np.float32)
