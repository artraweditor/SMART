#!/usr/bin/env python3
"""Render icons/app-icon-1024.png and icons/app-icon-fullbleed.png from
icons/app-icon.svg (the ART logo with the cross replaced by AI sparkles), and
then regenerate icons/app-icon-macos.png via tools/make_macos_icon.py.

The SVG has two layers: the rainbow brush strokes (a bitmap) and the frame
with the sparkles (vectors).  The brush bitmap has see-through gaps; inside the
inner square these are intentional and are kept (transparent in
app-icon-1024.png, white in the opaque full-bleed variant).  For the full-bleed
variant, the transparent area around the frame is filled with the colour of
the nearest stroke and blurred, so that the strokes run out to the corners.  Needs rsvg-convert (brew install librsvg).

    python3 tools/make_app_icons.py
"""
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SIZE = 1024
here = os.path.dirname(os.path.abspath(__file__))
icons = os.path.join(here, '..', 'icons')
svg = open(os.path.join(icons, 'app-icon.svg')).read()


def render(hide):
    """render the SVG with the layer whose id is `hide` hidden"""
    s, n = re.subn(r'(<g\b[^>]*?)\bid="%s"' % hide,
                   r'\1style="display:none" id="%s"' % hide, svg, count=1)
    assert n == 1, hide
    s = re.sub(r'(<g\b[^>]*?)style="display:inline"([^>]*?'
               r'style="display:none")', r'\1\2', s)
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, 'in.svg'), os.path.join(d, 'out.png')
        with open(src, 'w') as f:
            f.write(s)
        subprocess.run(['rsvg-convert', '-w', str(SIZE), '-h', str(SIZE),
                        src, '-o', dst], check=True)
        return np.asarray(Image.open(dst).convert('RGBA')).astype(np.float64) \
            / 255.0


def over(top, bottom):
    """straight-alpha 'over' compositing"""
    ta, ba = top[..., 3:], bottom[..., 3:]
    a = ta + ba * (1 - ta)
    rgb = (top[..., :3] * ta + bottom[..., :3] * ba * (1 - ta)) \
        / np.maximum(a, 1e-8)
    return np.concatenate([rgb, a], axis=-1)


def nearest_fill(img, thr=0.6):
    """opaque image with every pixel set to the colour of the nearest pixel
    of `img` whose alpha is >= thr"""
    holes = img[..., 3] < thr
    _, (iy, ix) = ndimage.distance_transform_edt(holes, return_indices=True)
    out = img[iy, ix].copy()
    out[..., 3] = 1.0
    return out


def blur(img, sigma):
    out = img.copy()
    for c in range(3):
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma)
    return out


def save(img, name):
    dst = os.path.join(icons, name)
    Image.fromarray(np.round(np.clip(img, 0, 1) * 255).astype(np.uint8),
                    'RGBA').save(dst, optimize=True)
    print('wrote', os.path.normpath(dst), flush=True)


brush = render('layer1')         # brush strokes only
frame = render('g1180')          # frame and sparkles only

# the bitmap has near-black smudges along the frame's outlines: recolour them
# with the nearest stroke colour
dark = ndimage.binary_dilation(brush[..., :3].max(axis=-1) < 0.3,
                               iterations=2)
clean = brush.copy()
clean[dark, 3] = 0.0
fill = nearest_fill(clean)
brush[dark, :3] = fill[dark, :3]

# the inner square (in SVG units: x 17.037..62.037, y 16.5..61.5, out of 80),
# extended a bit under the frame's outline
k = SIZE / 80.0
yy, xx = np.mgrid[0:SIZE, 0:SIZE] + 0.5
inner = ((xx >= 15.5 * k) & (xx <= 63.5 * k) &
         (yy >= 15.0 * k) & (yy <= 63.0 * k))[..., None]

save(over(frame, brush), 'app-icon-1024.png')

outside = over(brush, blur(fill, 24.0))
white = np.ones_like(brush)
full = np.where(inner, over(brush, white), outside)
save(over(frame, full), 'app-icon-fullbleed.png')

subprocess.run([sys.executable, os.path.join(here, 'make_macos_icon.py')],
               check=True)
