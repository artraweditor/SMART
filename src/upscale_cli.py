"""Command-line upscaling: `SMART --mode upscale --cli INPUT`."""

import sys
import time
from pathlib import Path

import dng
import upscale


def options_from_args(opts, conf):
    """upscale.Options from the config, overridden by the command line."""
    o = upscale.Options.from_config(conf, verbose=bool(opts.verbose))
    for src, dst in (('scale', 'scale'), ('model_file', 'model_file'),
                     ('provider', 'provider'), ('tile', 'tile'),
                     ('margin', 'margin'), ('dng_bits', 'dng_bits')):
        v = getattr(opts, src, None)
        if v is not None:
            setattr(o, dst, v)
    o.check()
    return o


def run(opts, conf):
    """Returns the process exit code."""
    if not opts.input_file:
        sys.exit('error: an input image is required')
    src_path = Path(opts.input_file)
    out = Path(opts.output) if opts.output \
        else upscale.default_output(src_path)
    if out.exists() and out.resolve() == src_path.resolve():
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
        src = upscale.load_source(src_path)
        report(3)
        run_ = upscale.make_run(o)
    except (upscale.UpscaleError, dng.DNGError, upscale.ModelMissing,
            ValueError, OSError, RuntimeError) as e:
        sys.exit(f'error: {e}')
    w, h = upscale.source_size(src)
    print(f'{src_path.name} {w}x{h} -> {w * o.scale}x{h * o.scale}  '
          f'model {o.model_path().name}', file=sys.stderr)
    report(10)

    def progress(i, n):
        if opts.progress:
            report(10 + 88 * i / n)
        else:
            print(f'\r  tile {i}/{n}', end='', file=sys.stderr, flush=True)

    t1 = time.time()
    try:
        res = upscale.upscale_source(src, o, run_, progress)
        print(f'\n  {time.time() - t1:.1f} s on {res.provider}',
              file=sys.stderr)
        copied = upscale.save(out, res, src, o, conf.exiftool)
    except (upscale.UpscaleError, RuntimeError, OSError, ValueError) as e:
        print(file=sys.stderr)
        sys.exit(f'error: {e}')
    if out.suffix.lower() == '.dng' and not copied:
        print('note: exiftool not available or failed; exposure/lens '
              'metadata not copied', file=sys.stderr)
    print(f'wrote {out}  (total {time.time() - t0:.1f} s)', file=sys.stderr)
    report(100)
    return 0
