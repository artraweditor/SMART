"""Runtime setup for the stand-alone (PyInstaller) builds."""
import atexit
import os
import subprocess
import sys
import tempfile

import config

_output_file = None
_show_output = True


def _windowed_output():
    """In a windowed Windows build there is no console: stdout/stderr are
    None. Write them to a temporary file instead and, unless a GUI was
    started, show it in notepad when the program exits.
    SMART_OUTPUT_FILE=<path> selects the file and disables notepad (testing).
    """
    global _output_file
    forced = os.environ.get('SMART_OUTPUT_FILE')
    if forced:
        _output_file = forced
    else:
        fd, _output_file = tempfile.mkstemp(prefix='SMART-output-',
                                            suffix='.txt')
        os.close(fd)
    f = open(_output_file, 'w', encoding='utf-8', errors='replace',
             buffering=1)
    sys.stdout = sys.stderr = f

    def show():
        f.flush()
        f.close()
        if (_show_output and not forced
                and os.path.getsize(_output_file) > 0):
            subprocess.Popen(['notepad.exe', _output_file])
    atexit.register(show)

    # do not flash a console window for every helper (exiftool...) we run
    orig = subprocess.Popen.__init__

    def init(self, *args, **kwargs):
        kwargs['creationflags'] = (kwargs.get('creationflags', 0)
                                   | subprocess.CREATE_NO_WINDOW)
        orig(self, *args, **kwargs)
    subprocess.Popen.__init__ = init


def gui_started():
    """Call before entering the GUI: its output is not shown at exit."""
    global _show_output
    _show_output = False


def setup():
    if not config.is_frozen():
        return
    # make the bundled exiftool the one found as "exiftool"
    d = config.resource_dir() / 'exiftool'
    if d.is_dir():
        os.environ['PATH'] = str(d) + os.pathsep + os.environ.get('PATH', '')
    if sys.platform == 'win32' and (sys.stdout is None or sys.stderr is None):
        _windowed_output()
