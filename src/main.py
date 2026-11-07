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
    p.add_argument('--mode', choices=['mask', 'heal', 'denoise'],
                   help='mask: AI mask builder; '
                   'heal: AI-assisted healing; '
                   'denoise: raw denoising of a mosaiced or linear DNG. '
                   'If not given, the GUI asks which mode to use')
    p.add_argument('--download-heal-model', nargs='?', const='auto',
                   choices=['auto', 'onnx', 'torch'], metavar='BACKEND',
                   help='download the LaMa model used by the healing mode '
                   '(onnx or torch; default: the one selected by the '
                   'configuration) and exit')
    p.add_argument('--download-mask-model', metavar='QUALITY',
                   choices=list(config.MASK_MODELS),
                   help='download a SAM2.1 checkpoint for the masking mode '
                   f'({", ".join(config.MASK_MODELS)}; larger is better '
                   'but slower) and exit')
    p.add_argument('--download-denoise-model', nargs='?', const='config',
                   choices=['config', 'all'] + list(config.DENOISE_MODELS),
                   metavar='MODEL',
                   help='download a raw denoising model (fast, balanced or '
                   'best; default: the configured one; all: every one) '
                   'and exit')
    g = p.add_argument_group(
        'raw denoising (--mode denoise)',
        'The input is a mosaiced or a linear (demosaiced) DNG, made with '
        'ART-cli --raw-preprocess=mosaic or --raw-preprocess=demosaic. By '
        'default the GUI is started; with --cli '
        'the image is denoised and saved without it. Unset options default '
        'to the configuration file.')
    g.add_argument('--cli', action='store_true',
                   help='denoise INPUT without the GUI')
    g.add_argument('-o', '--output', metavar='FILE',
                   help='output file (default: INPUT-denoised.tif)')
    g.add_argument('-m', '--model', choices=list(config.DENOISE_MODELS),
                   help='model (default: balanced); mosaiced and linear DNGs '
                   'have their own networks for each choice')
    g.add_argument('--iso', type=float,
                   help='linear DNGs: the ISO the model is conditioned on '
                   '(default: from the file)')
    g.add_argument('--model-file', help='use this ONNX file instead of a '
                   'model')
    g.add_argument('--backend', choices=['onnxruntime', 'torch'],
                   help='onnxruntime (CoreML/CUDA/DirectML/CPU) or torch '
                   '(cuda/mps/xpu/cpu)')
    g.add_argument('--provider', help='onnxruntime: auto, cpu, coreml, '
                   'cuda, tensorrt, rocm, directml, openvino; torch: auto, '
                   'cpu, cuda, mps, xpu')
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
    g.add_argument('--tile', type=int, help='tile size (default 1024)')
    g.add_argument('--margin', type=int, help='tile margin (default 96)')
    g.add_argument('-v', '--verbose', action='store_true')
    g.add_argument('--progress', action='store_true',
                   help='print integer percentages, one per line, on stdout')
    p.add_argument('--outdir', metavar='DIR',
                   help='default directory for the save dialog (default: '
                   'the directory of the input image)')
    p.add_argument('--save-as', metavar='FILE',
                   help='default save name (for all modes): Save writes to '
                   'FILE at once, without the file dialog (which is shown '
                   'only if FILE already exists), and the Open button is '
                   'hidden. The extension selects the output format in '
                   'denoise mode. Meant for integration with ART')
    p.add_argument('input_file', nargs='?',
                   help='input image')
    opts = p.parse_args()
    if opts.cli and opts.mode != 'denoise':
        p.error('--cli requires --mode denoise')
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


def download_heal_model(conf, backend):
    import lama
    if backend == 'auto':
        backend = lama.resolve_backend_name(conf)
    name, url, size, sha = config.HEAL_MODELS[backend]
    download(url, conf.get_heal_model_file(backend), size, sha)


def download_mask_model(conf, quality):
    name, url, size, sha = config.MASK_MODELS[quality]
    download(url, config.Config.models_dir() / name, size, sha)
    print(f'To use it, set "model": "{name}" in the configuration file '
          f'({config.Config.get_config_file()}).')


def download_denoise_model(conf, which):
    if which == 'config':
        which = conf.denoise_model
    for tier in (config.DENOISE_MODELS if which == 'all' else [which]):
        # the networks for mosaiced and for linear DNGs
        for kind in ('mosaic', 'linear'):
            name, url, size, sha = config.denoise_models(kind)[tier]
            download(url, conf.get_denoise_model_file(tier, kind), size, sha)


def main():
    import frozen_env
    frozen_env.setup()
    opts = getopts()
    conf = config.Config.load()
    if opts.download_denoise_model:
        download_denoise_model(conf, opts.download_denoise_model)
    elif opts.mode == 'denoise' and opts.cli:
        import denoise_cli
        sys.exit(denoise_cli.run(opts, conf))
    elif opts.download_heal_model:
        download_heal_model(conf, opts.download_heal_model)
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
