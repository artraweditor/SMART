"""The raw denoising engine: the loaded mosaic, the before/after previews,
background denoising and saving."""

import dataclasses
import os
import threading

import dng
import denoise
from engine_base import BaseEngine


class DenoiseEngine(BaseEngine):
    def __init__(self, conf, opts=None):
        super().__init__(conf)
        if opts is not None:
            import denoise_cli
            self.options = denoise_cli.options_from_args(opts, conf)
        else:
            self.options = denoise.Options.from_config(conf)
        self.default_format = (getattr(opts, 'format', None) or 'tiff') \
            if opts is not None else 'tiff'   # default of the save dialog
        self.raw = None
        self.result = None          # denoise.Result of the last run
        self.before_display = None  # uint8 HxWx3, upright
        self.after_display = None
        self.busy = False
        self._cancel = threading.Event()
        self._run = None            # cached network, with its options key
        self._run_key = None

    def busy_now(self):
        return self.busy

    # ------------------------------------------------------------ image
    def load_image(self, filename):
        self.reset(True)
        raw = dng.read_dng(filename)
        if raw.kind == 'linear':
            before = raw.rgb          # already demosaiced
        else:
            before = denoise.bilinear_demosaic(raw.levels, raw.cfa)
        self.before_display = denoise.render_display(
            before, raw, self.to_display)
        self.raw = raw
        h, w = self.before_display.shape[:2]
        self.size = w, h
        self.image_filename = os.path.abspath(filename)
        self.displayed_image = self.before_display
        self.saved = True

    def reset(self, clear_image=False):
        """Discard the denoised image (and, with `clear_image`, the loaded
        one)."""
        self.result = None
        self.after_display = None
        self.saved = True
        if clear_image:
            self.raw = None
            self.before_display = None
            self.displayed_image = None
            self.size = -1, -1

    def undo_last(self):
        self.reset()

    # ------------------------------------------------------------ denoising
    def denoise_async(self, model, on_done, on_error, progress=None):
        """Denoise in a background thread; the callbacks are called from that
        thread (wrap them with wx.CallAfter)."""
        if self.raw is None or self.busy:
            return False
        self.busy = True
        self._cancel.clear()
        raw = self.raw
        opts = dataclasses.replace(self.options, model=model)

        def check(k, n):
            if self._cancel.is_set():
                raise denoise.Cancelled()
            if progress:
                progress(k, n)

        def work():
            try:
                if self._cancel.is_set():
                    raise denoise.Cancelled()
                key = opts.runner_key(raw.kind)
                if self._run_key != key:
                    self._run = None    # free the old network first
                    self._run = denoise.make_run(opts, raw.kind)
                    self._run_key = key
                res = denoise.denoise_dng(raw, opts, self._run, check)
                after = denoise.render_display(res.rgb, raw, self.to_display)
            except Exception as e:
                self.busy = False
                on_error(e)
                return
            on_done(res, after)

        threading.Thread(target=work, daemon=True).start()
        return True

    def cancel(self):
        """Ask a running denoise to stop (after the current tile)."""
        self._cancel.set()

    def apply_result(self, res, after, model):
        self.result = res
        self.after_display = after
        self.options = dataclasses.replace(self.options, model=model)
        self.saved = False
        self.busy = False

    # ------------------------------------------------------------ saving
    def save_image(self, filename):
        if self.result is None:
            raise Exception("nothing to save: denoise the image first")
        denoise.save_result(filename, self.result, self.raw,
                            self.options, self.conf.exiftool)
        self.saved = True
