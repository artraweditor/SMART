import config
import argparse
import hashlib
import os
import sys
import urllib.request


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('--init-config', action='store_true',
                   help='(re)-create an initial configuration file and exit')
    p.add_argument('--mode', choices=['mask', 'heal'], default='mask',
                   help='mask: AI mask builder (default); '
                   'heal: AI-assisted healing')
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
    p.add_argument('input_file', nargs='?',
                   help='input image')
    return p.parse_args()


def download(url, dest, size, sha):
    """Download `url` to `dest`, checking the size and, if known, the sha256."""
    dest = str(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    print(f'Downloading {url}\n  to {dest}')
    h = hashlib.sha256()
    n = 0
    try:
        with urllib.request.urlopen(url) as r, open(part, 'wb') as out:
            total = size or int(r.headers.get('Content-Length') or 0)
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                n += len(chunk)
                if total:
                    print(f'\r  {n / 1e6:.0f}/{total / 1e6:.0f} MB', end='',
                          flush=True)
    except BaseException:
        if os.path.exists(part):
            os.unlink(part)
        raise
    print()
    if size is not None and n != size:
        os.unlink(part)
        sys.exit(f'error: expected {size} bytes, got {n}')
    if sha is not None and h.hexdigest() != sha:
        os.unlink(part)
        sys.exit('error: checksum mismatch')
    if sha is None:
        print(f'  sha256 (not verified): {h.hexdigest()}')
    os.replace(part, dest)
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


def main():
    opts = getopts()
    conf = config.Config.load()
    if opts.download_heal_model:
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
        gui.main(conf, opts.input_file, opts.mode)


if __name__ == '__main__':
    main()
