import config
import gui
import argparse
import os


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('--init-config', action='store_true',
                   help='(re)-create an initial configuration file and exit')
    p.add_argument('input_file', nargs='?',
                   help='input image')
    return p.parse_args()

def main():
    opts = getopts()
    conf = config.Config.load()
    if opts.init_config:
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
        gui.main(conf, opts.input_file)


if __name__ == '__main__':
    main()
