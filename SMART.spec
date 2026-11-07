# PyInstaller spec for the Windows build: two executables sharing one
# bundle (dist/SMART/_internal):
#   SMART.exe          windowed (no console), for double-clicking
#   SMART-console.exe  console subsystem, for use from cmd/PowerShell/scripts
# Paths are relative to this file; run from the repository root.
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

datas = [('icons', 'icons'), ('data', 'data'), ('exiftool-bundle', 'exiftool')]
binaries = []
hiddenimports = []
hiddenimports += collect_submodules('imagecodecs')
binaries += collect_dynamic_libs('onnxruntime')

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest',
              'torch', 'torchvision', 'sam2', 'hydra', 'omegaconf'],
)
pyz = PYZ(a.pure)

common = dict(exclude_binaries=True, icon='icons/app-icon-1024.png')
exe_gui = EXE(pyz, a.scripts, [], name='SMART', console=False, **common)
exe_cli = EXE(pyz, a.scripts, [], name='SMART-console', console=True, **common)

coll = COLLECT(exe_gui, exe_cli, a.binaries, a.datas, name='SMART')
