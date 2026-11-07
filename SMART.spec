# PyInstaller spec for the Windows build: two executables sharing one
# bundle (dist/SMART/_internal):
#   SMART.exe          windowed (no console), for double-clicking
#   SMART-console.exe  console subsystem, for use from cmd/PowerShell/scripts
# Paths are relative to this file; run from the repository root.
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = [('icons', 'icons'), ('data', 'data'), ('exiftool-bundle', 'exiftool')]
binaries = []
hiddenimports = []
# imagecodecs: only what tifffile and the 16-bit PNG code (imgio.py) use
hiddenimports += ['imagecodecs._imcd', 'imagecodecs._png', 'imagecodecs._shared',
                  'imagecodecs._shared_cython']
binaries += collect_dynamic_libs('onnxruntime')

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest',
              'torch', 'torchvision', 'sam2', 'hydra', 'omegaconf']
    # only scipy.ndimage (and what it imports) is used
    + ['scipy.' + m for m in (
        'fft', 'integrate', 'interpolate', 'optimize', 'sparse', 'spatial',
        'stats', 'signal', 'io', 'cluster', 'constants', 'datasets',
        'differentiate', 'odr')],
)
pyz = PYZ(a.pure)

common = dict(exclude_binaries=True, icon='icons/app-icon-1024.png')
exe_gui = EXE(pyz, a.scripts, [], name='SMART', console=False, **common)
exe_cli = EXE(pyz, a.scripts, [], name='SMART-console', console=True, **common)

coll = COLLECT(exe_gui, exe_cli, a.binaries, a.datas, name='SMART')
