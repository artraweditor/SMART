#!/usr/bin/env python3
"""Generate icons/app-icon-macos.png, the Dock icon used on macOS >= 26.

SMART runs as a script (no application bundle), so its Dock icon is set at
run time, and macOS draws such an icon as-is: no rounded-square plate and no
Liquid Glass.  This bakes a rounded square of the same style into the icon:
the full-bleed artwork icons/app-icon-fullbleed.png (brush strokes running out to
the corners) is cropped as in a Liquid Glass icon, masked with a squircle on
Apple's 824-of-1024 icon grid, and given a soft shadow and a thin light rim.

    python3 tools/make_macos_icon.py
"""
import os
import numpy as np
from PIL import Image, ImageFilter

icons = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'icons')
logo = Image.open(os.path.join(icons, 'app-icon-fullbleed.png')).convert('RGBA')

# --- macOS-style tile: same crop as the Liquid Glass icon (scale 1.1)
S = 1.1
tx, ty = 7.2, 14.9
x0 = 512 + (0 - 512 - tx) / S
y0 = 512 + (0 - 512 - ty) / S
x1 = 512 + (1024 - 512 - tx) / S
y1 = 512 + (1024 - 512 - ty) / S
tile = logo.crop((round(x0), round(y0), round(x1), round(y1)))
BODY = 824                                 # Apple's icon grid: 824 of 1024
SS = 4                                     # supersampling for the edge
tile = tile.resize((BODY, BODY), Image.LANCZOS)
yy, xx = np.mgrid[0:BODY * SS, 0:BODY * SS] / (BODY * SS / 2.0) - 1.0
mask = (np.abs(xx) ** 5 + np.abs(yy) ** 5 <= 1.0)       # squircle
mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(
    (BODY, BODY), Image.LANCZOS)

canvas = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
# soft shadow
sh = Image.new('L', (1024, 1024), 0)
sh.paste(mask.point(lambda v: int(v * 0.35)), (100, 112))
sh = sh.filter(ImageFilter.GaussianBlur(14))
canvas.paste(Image.new('RGBA', (1024, 1024), (0, 0, 0, 255)), (0, 0), sh)
# the tile, with a thin light rim like the glass edge
rim = mask.filter(ImageFilter.MinFilter(5))
rimmask = Image.fromarray(np.clip(np.asarray(mask).astype(int)
                                  - np.asarray(rim).astype(int), 0, 255
                                  ).astype(np.uint8))
tile.putalpha(mask)
canvas.alpha_composite(tile, (100, 100))
white = Image.new('RGBA', (BODY, BODY), (255, 255, 255, 255))
white.putalpha(rimmask.point(lambda v: int(v * 0.45)))
canvas.alpha_composite(white, (100, 100))
dst = os.path.join(icons, 'app-icon-macos.png')
canvas.save(dst)
print('wrote', os.path.normpath(dst))
