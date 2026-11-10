import config
import argparse
import hashlib
import os
import sys
import urllib.request

def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-V', '--version', action='version',
                   version=f'SMART - an AI helper for ART: version ' +
                   config.version, help='print the version and exit')
    p.add_argument('--init-config', action='store_true',
                   help='(re)-create an initial configuration file and exit')
    p.add_argument('--mode', choices=['mask', 'heal', 'denoise', 'upscale'],
                   help='mask: AI mask builder; '
                   'heal: AI-assisted healing; '
                   'denoise: raw denoising of a mosaiced or linear DNG; '
                   'upscale: AI upscaling (2x or 4x) of an image or a '
                   'linear DNG, saved in the format of the input. '
                   'If not given, the GUI asks which mode to use')
    p.add_argument('--download-heal-model', action='store_true',
                   help='download the LaMa model used by the healing mode '
                   'and exit')
    p.add_argument('--download-mask-model', metavar='QUALITY',
                   choices=list(config.MASK_MODELS),
                   help='download a SAM2.1 model for the masking mode '
                   f'({", ".join(config.MASK_MODELS)}; larger is better '
                   'but slower) and exit')
    p.add_argument('--download-denoise-model', nargs='?', const='config',
                   choices=['config', 'all'] + list(config.DENOISE_MODELS),
                   metavar='MODEL',
                   help='download a raw denoising model (small, medium or '
                   'large; default: the configured one; all: every one) '
                   'and exit')
    p.add_argument('--download-upscale-model', nargs='?', const='config',
                   choices=['config', 'all'] +
                   [str(s) for s in config.UPSCALE_MODELS], metavar='SCALE',
                   help='download a Real-ESRGAN upscaling model (2 or 4; '
                   'default: the configured one; all: both) and exit')
    g = p.add_argument_group(
        'raw denoising (--mode denoise) and upscaling (--mode upscale)',
        'Upscaling: the input is an image (TIFF, PNG, JPEG) or a linear '
        'DNG, and the output has the same format (default: INPUT-upscaled.EXT); '
        'the options that apply are --cli, -o, --scale, --model-file, '
        '--provider, --dng-bits, --tile, --margin, -v and --progress.\n\n'
        'Denoising: the input is a mosaiced or a linear (demosaiced) DNG, made with '
        'ART-cli --raw-preprocess=mosaic or --raw-preprocess=demosaic. By '
        'default the GUI is started; with --cli '
        'the image is denoised and saved without it. Unset options default '
        'to the configuration file.')
    g.add_argument('--cli', action='store_true',
                   help='process INPUT without the GUI (denoise and upscale modes)')
    g.add_argument('-o', '--output', metavar='FILE',
                   help='output file (default: INPUT-denoised.tif)')
    g.add_argument('-m', '--model', choices=list(config.DENOISE_MODELS),
                   help='model (default: medium); mosaiced and linear DNGs '
                   'have their own networks for each choice')
    g.add_argument('--iso', type=float,
                   help='linear DNGs: the ISO the model is conditioned on '
                   '(default: from the file)')
    g.add_argument('--model-file', help='use this ONNX file instead of a '
                   'model')
    g.add_argument('--provider', help='onnxruntime provider: auto, cpu, '
                   'coreml, cuda, tensorrt, rocm, directml, openvino')
    g.add_argument('--noise', metavar='aR,bR,aG,bG,aB,bB',
                   help='noise model (levels 0..1) instead of estimating it')
    g.add_argument('-f', '--format', choices=['tiff', 'dng'],
                   help='output format; default from the -o extension, else '
                   'tiff. dng writes a LinearRaw DNG (camera matrix, as-shot '
                   'neutral, Make/Model), keeping the orientation flag')
    g.add_argument('--dng-bits', type=int, choices=[16, 32],
                   help='DNG sample type: 16 = unsigned integer (default; '
                   'clips negatives), 32 = IEEE float')
    g.add_argument('--wb', choices=['none', 'camera'],
                   help='TIFF output: multiply by the as-shot white balance '
                   'gains (G=1); default none')
    g.add_argument('--no-rotate', action='store_true',
                   help='TIFF output: keep the sensor orientation')
    g.add_argument('--scale', type=int, choices=list(config.UPSCALE_MODELS),
                   help='upscaling factor (default: from the configuration '
                   'file, 2)')
    g.add_argument('--tile', type=int, help='tile size (default 1024 for '
                   'denoising, 512 for upscaling)')
    g.add_argument('--margin', type=int, help='tile margin (default 96 for '
                   'denoising, 16 for upscaling)')
    g.add_argument('-v', '--verbose', action='store_true')
    g.add_argument('--progress', action='store_true',
                   help='print integer percentages, one per line, on stdout')
    p.add_argument('--outdir', metavar='DIR',
                   help='default directory for the save dialog (default: '
                   'the directory of the input image)')
    p.add_argument('--save-as', metavar='FILE',
                   help='default save name (for all modes): Save writes to '
                   'FILE at once, without the file dialog, and the Open '
                   'button is hidden. If FILE already exists when the '
                   'first save is made, the file dialog is shown instead '
                   '(at every save). The extension selects the output format in '
                   'denoise mode. Meant for integration with ART')
    p.add_argument('input_file', nargs='?',
                   help='input image')
    opts = p.parse_args()
    if opts.cli and opts.mode not in ('denoise', 'upscale'):
        p.error('--cli requires --mode denoise or --mode upscale')
    if opts.outdir is not None:
        opts.outdir = os.path.abspath(os.path.expanduser(opts.outdir))
        if not os.path.isdir(opts.outdir):
            p.error(f'--outdir: not a directory: {opts.outdir}')
    if opts.save_as is not None:
        opts.save_as = os.path.abspath(os.path.expanduser(opts.save_as))
        if os.path.isdir(opts.save_as):
            p.error(f'--save-as: is a directory: {opts.save_as}')
    return opts


def download(url, dest, size, sha):
    """Download `url` to `dest`, checking the size and, if known, the sha256."""
    import download as dl
    print(f'Downloading {url}\n  to {dest}')

    def progress(n, total):
        if total:
            print(f'\r  {n / 1e6:.0f}/{total / 1e6:.0f} MB', end='',
                  flush=True)
    try:
        digest = dl.download(url, dest, size, sha, progress)
    except dl.DownloadError as e:
        print()
        sys.exit(f'error: {e}')
    print()
    if sha is None:
        print(f'  sha256 (not verified): {digest}')
    print('Done.')


def download_heal_model(conf):
    name, url, size, sha = config.HEAL_MODEL
    download(url, conf.get_heal_model_file(), size, sha)


def download_mask_model(conf, quality):
    models = config.MASK_MODELS[quality]
    for (name, url, size, sha) in models:
        if not url:
            sys.exit(f'error: no download location for {name}; create it '
                     'with tools/export_sam2_onnx.py')
    for (name, url, size, sha) in models:
        download(url, config.Config.models_dir() / name, size, sha)
    print(f'To use it, set "model": "{quality}" in the configuration file '
          f'({config.Config.get_config_file()}).')


def download_denoise_model(conf, which):
    if which == 'config':
        which = conf.denoise_model
    for tier in (config.DENOISE_MODELS if which == 'all' else [which]):
        # the networks for mosaiced and for linear DNGs
        for kind in ('mosaic', 'linear'):
            name, url, size, sha = config.denoise_models(kind)[tier]
            download(url, conf.get_denoise_model_file(tier, kind), size, sha)


def download_upscale_model(conf, which):
    if which == 'config':
        which = str(conf.upscale_scale)
    for scale in (config.UPSCALE_MODELS if which == 'all' else [int(which)]):
        name, url, size, sha = config.UPSCALE_MODELS[scale]
        download(url, conf.get_upscale_model_file(scale), size, sha)


def main():
    import frozen_env
    frozen_env.setup()
    opts = getopts()
    conf = config.Config.load()
    if opts.download_denoise_model:
        download_denoise_model(conf, opts.download_denoise_model)
    elif opts.download_upscale_model:
        download_upscale_model(conf, opts.download_upscale_model)
    elif opts.mode == 'denoise' and opts.cli:
        import denoise_cli
        sys.exit(denoise_cli.run(opts, conf))
    elif opts.mode == 'upscale' and opts.cli:
        import upscale_cli
        sys.exit(upscale_cli.run(opts, conf))
    elif opts.download_heal_model:
        download_heal_model(conf)
    elif opts.download_mask_model:
        download_mask_model(conf, opts.download_mask_model)
    elif opts.init_config:
        fn = config.Config.get_config_file()
        if not os.path.exists(fn) or \
           input(f'Configuration file "{fn}" already exists; '
                 'overwrite (y/N)? ').strip() == 'y':
            conf.save()
            print(f'Configuration saved to: {fn}')
    else:
        legacy_fn = config.Config.get_config_file(True)
        if os.path.exists(legacy_fn) \
           and not os.path.exists(config.Config.get_config_file()):
            conf = config.Config.load(legacy_fn)
            conf.save()
            os.unlink(legacy_fn)
        import gui
        frozen_env.gui_started()
        gui.main(conf, opts.input_file, opts.mode, opts.outdir, opts)


if __name__ == '__main__':
    main()
