"""AI healing: fill the painted coverage with LaMa.

A port of the fill path of greycard (https://github.com/jessolmstead/greycard,
GPL-3.0, commit bf63fa5), see lama.py.

`heal()` never touches its inputs: it returns the new pixel data for each
window it filled, and the caller applies it.

Interfaces of the post-fill stages (see seam.py, middle.py, grain.py), all on
float32 (H, W, 3) arrays and an (H, W) hole:
    seam.matched(image_enc, hole, fill_enc)  -> (fill_enc, info|None)
    middle.evened(image_enc, hole, fill_enc) -> (fill_enc, info|None)
    grain.regrain(frame_lin, out_lin, hole, luma, seed) -> out_lin
where *_enc are sRGB-encoded in 0..1 and *_lin are linear working values.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import colorspace
import lama


@dataclass
class Patch:
    window: lama.Window
    data: np.ndarray        # new float32 (h, w, 3) pixel data for the window
    changed: np.ndarray     # (h, w) bool: pixels that differ from the input
    scale: float
    tiles: int


def _groups(cover, frame_w, frame_h):
    """Windows to fill: one per connected painted component, merged when
    their windows overlap.  Returns [(Window, depth)]."""
    hole = cover > 0
    lab, n = ndimage.label(hole, structure=np.ones((3, 3), int))
    if n == 0:
        return []
    items = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = sl
        # depth is measured on the component's own box, grown by a pixel so
        # that the box's border counts as known unless it is the frame's
        y0, y1 = max(ys.start - 1, 0), min(ys.stop + 1, frame_h)
        x0, x1 = max(xs.start - 1, 0), min(xs.stop + 1, frame_w)
        d = lama.depth(lab[y0:y1, x0:x1] == i)
        box = lama.Window(xs.start, ys.start, xs.stop - xs.start,
                          ys.stop - ys.start)
        items.append((lama.window(box, d, frame_w, frame_h), d))
    # merge overlapping windows (the merged hole is the union; depth is the
    # largest)
    merged = True
    while merged:
        merged = False
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (wa, da), (wb, db) = items[a], items[b]
                if (wa.x < wb.x + wb.width and wb.x < wa.x + wa.width and
                        wa.y < wb.y + wb.height and wb.y < wa.y + wa.height):
                    x0, y0 = min(wa.x, wb.x), min(wa.y, wb.y)
                    x1 = max(wa.x + wa.width, wb.x + wb.width)
                    y1 = max(wa.y + wa.height, wb.y + wb.height)
                    # the union is the window itself (the model run mirrors
                    # out to a multiple of 8)
                    w2 = lama.Window(x0, y0, x1 - x0, y1 - y0)
                    d2 = lama.depth(hole[w2.y:w2.y + w2.height,
                                         w2.x:w2.x + w2.width])
                    items[a] = (w2, max(d2, da, db))
                    del items[b]
                    merged = True
                    break
            if merged:
                break
    return items


def luma_weights(to_srgb):
    """Luma weights of a working space, from its matrix to linear sRGB."""
    return (np.array([0.2126, 0.7152, 0.0722], np.float64) @
            np.asarray(to_srgb, np.float64)).astype(np.float32)


def heal_window(img, cover, win, d, filler, conf, seed, log=None):
    """Fill one window.  Returns a Patch."""
    log = log or (lambda *a: None)
    sl = (slice(win.y, win.y + win.height), slice(win.x, win.x + win.width))
    cov = cover[sl].astype(np.float32)
    hole = (cov > 0).astype(np.float32)
    known = hole <= 0
    orig = img.data[sl]
    work = np.nan_to_num(orig, nan=0.0, posinf=0.0, neginf=0.0)

    # --- to model space
    if img.linear:
        lin = np.maximum(colorspace.apply_matrix(img.to_srgb, work), 0.0)
        luma = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        mean = float(luma[known].mean()) if known.any() else 0.18
        gain = float(np.clip(0.18 / max(mean, 1e-4), 0.25, 16.0))
        enc = colorspace.srgb_oetf(lin * gain)
    else:
        gain = 1.0
        enc = np.clip(work, 0.0, 1.0).astype(np.float32)

    made = filler.fill_window(enc, hole, d)
    fill_enc = made.image
    if conf.heal_seam_match:
        import seam
        fill_enc, info = seam.matched(enc, hole, fill_enc)
        log('seam', info)
    if conf.heal_even:
        import middle
        fill_enc, info = middle.evened(enc, hole, fill_enc)
        log('even', info)

    # --- back to the working space (linear for float data)
    if img.linear:
        out = colorspace.apply_matrix(img.from_srgb,
                                      colorspace.srgb_eotf(fill_enc) / gain)
    else:
        out = fill_enc

    if conf.heal_regrain and made.scale < 1.0:
        import grain
        if img.linear:
            lw = luma_weights(img.to_srgb)
            out = grain.regrain(work, out, hole, lw, seed)
        else:
            lw = np.array([0.2126, 0.7152, 0.0722], np.float32)
            out = colorspace.srgb_oetf(grain.regrain(
                colorspace.srgb_eotf(work), colorspace.srgb_eotf(out),
                hole, lw, seed))

    # --- blend under the coverage; the rest stays bit-exact
    a = np.clip(cov, 0.0, 1.0)
    m = a > 0
    new = orig.copy()
    new[m] = work[m] + (out[m] - work[m]) * a[m][:, None]
    return Patch(win, new, m, made.scale, made.tiles)


def heal(img, cover, filler, conf, seed=0, log=None, progress=None):
    """Fill the pixels of `img` where `cover` (float HxW, 0..1, opacity
    already folded in) is positive.  Returns a list of Patch."""
    h, w = img.data.shape[:2]
    groups = _groups(cover, w, h)
    patches = []
    for k, (win, d) in enumerate(groups):
        if progress:
            progress(k, len(groups))
        s = (seed * 1000003 + win.y * 65537 + win.x) & 0xFFFFFFFF
        patches.append(heal_window(img, cover, win, d, filler, conf, s, log))
    if progress:
        progress(len(groups), len(groups))
    return patches
