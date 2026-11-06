"""Command-line raw denoising: `SMART --mode denoise --cli INPUT.dng`."""

import os
import sys
import time
from pathlib import Path

import dng
import denoise


def options_from_args(opts, conf):
    """denoise.Options from the config, overridden by the command line."""
    o = denoise.Options.from_config(conf, verbose=bool(opts.verbose))
    for src, dst in (('model', 'model'), ('model_file', 'model_file'),
                     ('backend', 'backend'), ('provider', 'provider'),
                     ('tile', 'tile'), ('margin', 'margin'),
                     ('dng_bits', 'dng_bits'), ('wb', 'wb')):
        v = getattr(opts, src, None)
        if v is not None:
            setattr(o, dst, v)
    if opts.noise:
        o.noise = tuple(float(s) for s in opts.noise.split(','))
        if len(o.noise) != 6:
            raise ValueError('--noise wants six numbers')
    o.rotate = not opts.no_rotate
    o.tile, o.margin = denoise.tiling(o.tile, o.margin)
    return o


def run(opts, conf):
    """Returns the process exit code."""
    if not opts.input_file:
        sys.exit('error: an input DNG file is required')
    src = Path(opts.input_file)
    fmt = opts.format or ('dng' if opts.output
                          and opts.output.lower().endswith('.dng')
                          else 'tiff')
    out = Path(opts.output) if opts.output else \
        src.with_name(src.stem + '-denoised' + ('.dng' if fmt == 'dng'
                                                else '.tif'))
    if fmt == 'dng' and out.suffix.lower() != '.dng':
        out = out.with_name(out.name + '.dng')
    if out.exists() and out.resolve() == src.resolve():
        sys.exit('error: the output would overwrite the input')
    try:
        o = options_from_args(opts, conf)
    except ValueError as e:
        sys.exit(f'error: {e}')

    last = [-1]

    def report(pct):
        if opts.progress and int(pct) > last[0]:
            last[0] = int(pct)
            print(last[0], flush=True)

    t0 = time.time()
    try:
        raw = dng.read_mosaic_dng(src)
        report(3)
        run_ = denoise.make_run(o)
    except (dng.DNGError, denoise.ModelMissing, ValueError) as e:
        sys.exit(f'error: {e}')
    h, w = raw.levels.shape
    print(f'{raw.name} {w}x{h}  model {o.model_path().name}', file=sys.stderr)
    report(10)

    def progress(i, n):
        if opts.progress:
            report(10 + 88 * i / n)
        else:
            print(f'\r  tile {i}/{n}', end='', file=sys.stderr, flush=True)

    t1 = time.time()
    res = denoise.denoise_dng(raw, o, run_, progress)
    print(f'\n  {time.time() - t1:.1f} s on {res.provider}', file=sys.stderr)
    copied = denoise.save_result(out, res, raw, o, conf.exiftool)
    if out.suffix.lower() == '.dng' and not copied:
        print('note: exiftool not available or failed; exposure/lens '
              'metadata not copied', file=sys.stderr)
    print(f'wrote {out}  (total {time.time() - t0:.1f} s)', file=sys.stderr)
    report(100)
    return 0
