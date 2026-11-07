#!/usr/bin/env python3
"""Compile icons/app-icon-1024.png into a macOS 26 Liquid Glass app icon.

Writes <outdir>/Assets.car (the Liquid Glass icon, to be put in the bundle's
Contents/Resources together with CFBundleIconName=SMART in Info.plist) and
<outdir>/SMART.icns (the flattened icon, for older systems and for
PyInstaller's --icon).  Needs Xcode 26's actool.

The full-bleed artwork fills the whole rounded square of the system icon
(the layer is scaled up a little and re-centred so that the outline of the
white square touches the edges, and the brush strokes run out to the corners).

    python3 icons/make_liquid_glass_icon.py <outdir>
"""
import json
import os
import shutil
import subprocess
import sys

NAME = 'SMART'
ART = 'app-icon-1024.png'


def main(outdir):
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.abspath(outdir)
    icon = os.path.join(outdir, NAME + '.icon')
    build = os.path.join(outdir, 'actool')
    for d in (icon, build):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(icon, 'Assets'))
    os.makedirs(build)
    shutil.copy2(os.path.join(here, ART),
                 os.path.join(icon, 'Assets', ART))
    with open(os.path.join(icon, 'icon.json'), 'w') as f:
        json.dump({
            'fill': {'solid': 'srgb:1.00000,1.00000,1.00000,1.00000'},
            'groups': [{
                'layers': [{
                    'image-name': ART,
                    'name': 'logo',
                    # scale / offset (in points) chosen so that the square's
                    # outline just reaches the rounded edges
                    'position': {'scale': 1.1,
                                 'translation-in-points': [7.2, 14.9]},
                }],
                'shadow': {'kind': 'neutral', 'opacity': 0.5},
                'translucency': {'enabled': False, 'value': 0.5},
            }],
            'supported-platforms': {'squares': 'shared'},
        }, f, indent=2)
    subprocess.run(['xcrun', 'actool', icon, '--compile', build,
                    '--output-format', 'human-readable-text',
                    '--output-partial-info-plist',
                    os.path.join(build, 'partial.plist'),
                    '--app-icon', NAME, '--include-all-app-icons',
                    '--enable-on-demand-resources', 'NO',
                    '--development-region', 'en',
                    '--target-device', 'mac',
                    '--minimum-deployment-target', '26.0',
                    '--platform', 'macosx'], check=True)
    for fn in ('Assets.car', NAME + '.icns'):
        shutil.copy2(os.path.join(build, fn), os.path.join(outdir, fn))
    print('wrote', os.path.join(outdir, 'Assets.car'), 'and',
          os.path.join(outdir, NAME + '.icns'))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
