"""The AI healing engine: image state, brush painting, undo, and healing."""

import os
import threading
from dataclasses import dataclass

import numpy as np

import heal as healmod
import imgio
import lama
from engine_base import BaseEngine

TINT = 0.5          # strength of the painted-coverage overlay
MIN_COVER = 1.0 / 255.0


@dataclass
class _Stroke:
    bbox: tuple             # (x0, y0, x1, y1)
    old_cover: np.ndarray


@dataclass
class _Heal:
    patches: list           # [(bbox, old data, old base display u8)]
    old_cover: np.ndarray   # whole-image copy of the cover before the heal


def brush_profile(dist, radius, feather):
    """greycard's brush: 1 inside the inner radius, smoothstep to 0 at the
    radius."""
    inner = radius * max(1.0 - feather, 0.02)
    if radius <= inner:
        return (dist <= radius).astype(np.float32)
    t = np.clip((dist - inner) / (radius - inner), 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)


class AIHealingEngine(BaseEngine):
    def __init__(self, conf):
        super().__init__(conf)
        self.img = None
        self.cover = None           # float32 HxW, opacity already folded in
        self.base_display = None    # uint8 HxWx3, image through the display xform
        self.radius = conf.brush_radius
        self.feather = conf.brush_feather
        self.opacity = conf.brush_opacity
        self.history = []
        self.filler = None
        self._stroke = None
        self._erase = False
        self._last = None
        self._stroke_bbox = None
        self._seed = 0
        self.busy = False

    def busy_now(self):
        return self.busy

    # ------------------------------------------------------------ image
    def load_image(self, filename):
        self.reset_state()
        img = imgio.load(filename)
        self.img = img
        self.size = img.size
        self.image_filename = os.path.abspath(filename)
        h, w = img.data.shape[:2]
        self.cover = np.zeros((h, w), np.float32)
        self.base_display = self.to_display(img.display_srgb())
        self.displayed_image = self.base_display.copy()
        self.saved = True

    def reset_state(self):
        self.history = []
        self._stroke = None
        self.saved = True

    def reset(self, clear_image=False):
        """Forget all strokes and heals; back to the image as loaded."""
        if clear_image:
            self.img = None
            self.cover = None
            self.base_display = None
            self.displayed_image = None
            self.history = []
            self.saved = True
            return
        if self.img is None:
            return
        # undo everything, newest first
        while self.history:
            self.undo_last()
        self.saved = True

    def has_cover(self):
        return self.cover is not None and bool((self.cover > 0).any())

    # ------------------------------------------------------------ display
    def _render(self, x0, y0, x1, y1, extra=None):
        """Recompute displayed_image in the given box from the base display
        and the coverage (and the live stroke, if any)."""
        c = self.cover[y0:y1, x0:x1]
        if extra is not None:
            c = np.maximum(c, extra[y0:y1, x0:x1])
        base = self.base_display[y0:y1, x0:x1].astype(np.float32)
        col = np.array(self.conf.brush_color, np.float32)
        a = (TINT * c)[..., None]
        out = base * (1.0 - a) + col * a
        self.displayed_image[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(
            np.uint8)

    def _clip_box(self, box):
        h, w = self.cover.shape
        x0, y0, x1, y1 = box
        return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)

    # ------------------------------------------------------------ painting
    def _dab(self, cx, cy):
        r = self.radius
        x0, y0, x1, y1 = self._clip_box((int(np.floor(cx - r)) - 1,
                                         int(np.floor(cy - r)) - 1,
                                         int(np.ceil(cx + r)) + 2,
                                         int(np.ceil(cy + r)) + 2))
        if x1 <= x0 or y1 <= y0:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xs - cx, ys - cy)
        p = brush_profile(dist, r, self.feather)
        np.maximum(self._stroke[y0:y1, x0:x1], p,
                   out=self._stroke[y0:y1, x0:x1])
        return x0, y0, x1, y1

    @staticmethod
    def _union(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])

    def begin_stroke(self, erase=False):
        if self.img is None or self.busy:
            return
        self._stroke = np.zeros_like(self.cover)
        self._erase = erase
        self._last = None
        self._stroke_bbox = None

    def stroke_to(self, x, y):
        """Extend the stroke to image point (x, y); returns the dirty box."""
        if self._stroke is None:
            return None
        pts = [(x, y)]
        if self._last is not None:
            lx, ly = self._last
            dist = np.hypot(x - lx, y - ly)
            step = max(1.0, self.radius / 4.0)
            n = int(dist // step)
            pts = [(lx + (x - lx) * k / (n + 1), ly + (y - ly) * k / (n + 1))
                   for k in range(1, n + 1)] + pts
        self._last = (x, y)
        dirty = None
        for px, py in pts:
            dirty = self._union(dirty, self._dab(px, py))
        if dirty is None:
            return None
        self._stroke_bbox = self._union(self._stroke_bbox, dirty)
        live = self._stroke if self._erase else self._stroke * self.opacity
        if self._erase:
            self._render_erase(dirty)
        else:
            self._render(*dirty, extra=live)
        return dirty

    def _render_erase(self, box):
        x0, y0, x1, y1 = box
        saved = self.cover[y0:y1, x0:x1].copy()
        self.cover[y0:y1, x0:x1] = saved * (1.0 - self._stroke[y0:y1, x0:x1])
        self._render(x0, y0, x1, y1)
        self.cover[y0:y1, x0:x1] = saved

    def end_stroke(self):
        """Commit the stroke; returns the dirty box (or None)."""
        s, box = self._stroke, self._stroke_bbox
        self._stroke = None
        self._last = None
        self._stroke_bbox = None
        if s is None or box is None:
            return None
        x0, y0, x1, y1 = box
        old = self.cover[y0:y1, x0:x1].copy()
        st = s[y0:y1, x0:x1]
        if self._erase:
            c = old * (1.0 - st)
        else:
            sc = st * self.opacity
            c = old + sc * (1.0 - old)
        c[c < MIN_COVER] = 0.0
        self.cover[y0:y1, x0:x1] = c
        self.history.append(_Stroke(box, old))
        self._render(x0, y0, x1, y1)
        return box

    def clear_strokes(self):
        """Remove all painted coverage (undoable); returns the dirty box."""
        if self.cover is None or not self.has_cover():
            return None
        h, w = self.cover.shape
        old = self.cover.copy()
        self.history.append(_Stroke((0, 0, w, h), old))
        self.cover[:] = 0
        self._render(0, 0, w, h)
        return 0, 0, w, h

    # ------------------------------------------------------------ undo
    def undo_last(self):
        """Undo the last stroke or heal; returns the dirty box (or None)."""
        if not self.history or self.busy:
            return None
        a = self.history.pop()
        if isinstance(a, _Stroke):
            x0, y0, x1, y1 = a.bbox
            self.cover[y0:y1, x0:x1] = a.old_cover
            self._render(x0, y0, x1, y1)
            box = a.bbox
        else:
            box = None
            for bb, old, old_disp in a.patches:
                x0, y0, x1, y1 = bb
                self.img.data[y0:y1, x0:x1] = old
                self.base_display[y0:y1, x0:x1] = old_disp
                box = self._union(box, bb)
            self.cover[:] = a.old_cover
            h, w = self.cover.shape
            box = (0, 0, w, h)
            self._render(*box)
        self.saved = not any(isinstance(x, _Heal) for x in self.history)
        return box

    # ------------------------------------------------------------ healing
    def get_filler(self):
        if self.filler is None:
            self.filler = lama.make_filler(self.conf)
        return self.filler

    def heal_async(self, on_done, on_error, progress=None):
        """Heal the painted coverage in a background thread.  The callbacks
        are called from that thread (wrap them with wx.CallAfter)."""
        if self.img is None or not self.has_cover() or self.busy:
            return False
        self.busy = True
        cover = self.cover.copy()
        self._seed += 1
        seed = self._seed

        def work():
            try:
                filler = self.get_filler()
                patches = healmod.heal(self.img, cover, filler, self.conf,
                                       seed, progress=progress)
            except Exception as e:
                self.busy = False
                on_error(e)
                return
            on_done(patches, cover)

        threading.Thread(target=work, daemon=True).start()
        return True

    def apply_patches(self, patches, cover_used):
        """Apply the result of a heal (on the main thread).  Returns the
        dirty box."""
        done = []
        for p in patches:
            w = p.window
            x0, y0, x1, y1 = w.x, w.y, w.x + w.width, w.y + w.height
            done.append(((x0, y0, x1, y1), self.img.data[y0:y1, x0:x1].copy(),
                         self.base_display[y0:y1, x0:x1].copy()))
        self.history.append(_Heal(done, self.cover.copy()))
        for p in patches:
            w = p.window
            sl = (slice(w.y, w.y + w.height), slice(w.x, w.x + w.width))
            self.img.data[sl] = p.data
            self.base_display[sl] = self.to_display(self.img.display_srgb(sl))
            # what was healed is no longer painted
            self.cover[sl][p.changed] = 0.0
        h, wd = self.cover.shape
        self._render(0, 0, wd, h)
        self.saved = False
        self.busy = False
        return 0, 0, wd, h

    # ------------------------------------------------------------ saving
    def save_image(self, filename):
        if self.img is None:
            raise Exception("no image loaded")
        imgio.save(self.img, filename)
        self._copy_metadata(filename)
        self.saved = True

    def _copy_metadata(self, filename):
        exiftool = self.conf.exiftool
        src = self.img.src_path
        if not exiftool or not src or not os.path.exists(src):
            return
        import subprocess
        try:
            subprocess.run([exiftool, '-tagsFromFile', src, '-all:all',
                            '-icc_profile', '-overwrite_original', filename],
                           capture_output=True, check=False)
        except Exception:
            pass
