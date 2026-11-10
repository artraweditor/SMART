"""The upscaling engine: the loaded image, the before/after previews,
background upscaling and saving."""

import dataclasses
import os
import threading

import dng
import upscale
from engine_base import BaseEngine


class UpscaleEngine(BaseEngine):
    def __init__(self, conf, opts=None):
        super().__init__(conf)
        if opts is not None:
            import upscale_cli
            self.options = upscale_cli.options_from_args(opts, conf)
        else:
            self.options = upscale.Options.from_config(conf)
        self.src = None             # dng.LinearDNG or imgio.Img
        self.result = None          # upscale.Result of the last run
        self.before_display = None  # uint8 HxWx3, upright, input resolution
        self.after_display = None   # uint8, output resolution
        self.in_size = -1, -1
        self.exposure = 0.0         # preview exposure compensation, EV (DNG)
        self.busy = False
        self._cancel = threading.Event()
        self._run = None            # cached network, with its options key
        self._run_key = None

    def busy_now(self):
        return self.busy

    @property
    def raw(self):
        """The source if it is a linear DNG (the exposure preview applies to
        those only), else None."""
        return self.src if isinstance(self.src, dng.LinearDNG) else None

    @property
    def scale(self):
        return self.options.scale

    def _update_size(self):
        w, h = self.in_size
        self.size = (w * self.scale, h * self.scale) if w > 0 else (-1, -1)

    # ------------------------------------------------------------ image
    def load_image(self, filename):
        self.reset(True)
        src = upscale.load_source(filename)
        self.before_display = upscale.source_display(
            src, self.to_display, gain=self.gain)
        self.src = src
        self.in_size = upscale.source_size(src)
        h, w = self.before_display.shape[:2]
        self.in_size = w, h       # (upright: swapped for rotated DNGs)
        self._update_size()
        self.image_filename = os.path.abspath(filename)
        self.displayed_image = self.before_display
        self.saved = True

    def reset(self, clear_image=False):
        """Discard the upscaled image (and, with `clear_image`, the loaded
        one)."""
        self.result = None
        self.after_display = None
        self.saved = True
        if clear_image:
            self.src = None
            self.before_display = None
            self.displayed_image = None
            self.in_size = -1, -1
            self.size = -1, -1

    def set_scale(self, scale):
        """Change the factor; discards the result."""
        self.options = dataclasses.replace(self.options, scale=scale)
        self.reset()
        self._update_size()

    @property
    def gain(self):
        return 2.0 ** self.exposure

    def output_megapixels(self):
        w, h = self.size
        return max(w, 0) * max(h, 0) / 1e6

    def render_previews(self, exposure):
        """(before, after) display images for an exposure compensation in EV
        (after is None if there is no result); callable from any thread."""
        src, res = self.src, self.result
        if src is None:
            return None, None
        g = 2.0 ** exposure
        b = upscale.source_display(src, self.to_display, gain=g)
        a = None if res is None else upscale.render_display(
            res, src, self.to_display, gain=g)
        return b, a

    def set_previews(self, before, after, exposure):
        self.exposure = exposure
        self.before_display = before
        self.displayed_image = before
        if after is not None:
            self.after_display = after

    def undo_last(self):
        self.reset()

    # ------------------------------------------------------------ upscaling
    def upscale_async(self, on_done, on_error, progress=None):
        """Upscale in a background thread; the callbacks are called from that
        thread (wrap them with wx.CallAfter)."""
        if self.src is None or self.busy:
            return False
        self.busy = True
        self._cancel.clear()
        src = self.src
        opts = self.options

        def check(k, n):
            if self._cancel.is_set():
                raise upscale.Cancelled()
            if progress:
                progress(k, n)

        def work():
            try:
                if self._cancel.is_set():
                    raise upscale.Cancelled()
                key = opts.runner_key()
                if self._run_key != key:
                    self._run = None    # free the old network first
                    self._run = upscale.make_run(opts)
                    self._run_key = key
                res = upscale.upscale_source(src, opts, self._run, check)
                after = upscale.render_display(
                    res, src, self.to_display, gain=self.gain)
            except Exception as e:
                self.busy = False
                on_error(e)
                return
            on_done(res, after)

        threading.Thread(target=work, daemon=True).start()
        return True

    def cancel(self):
        """Ask a running upscale to stop (after the current tile)."""
        self._cancel.set()

    def apply_result(self, res, after):
        self.result = res
        self.after_display = after
        self.saved = False
        self.busy = False

    # ------------------------------------------------------------ saving
    def save_image(self, filename):
        if self.result is None:
            raise Exception("nothing to save: upscale the image first")
        upscale.save(filename, self.result, self.src, self.options,
                     self.conf.exiftool)
        self.saved = True
