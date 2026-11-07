"""LaMa inpainting ("generative fill"): a port of greycard's fill.rs
(https://github.com/jessolmstead/greycard, GPL-3.0, commit bf63fa5).

A hole is filled in a window cut around it, at the picture's own pixels as
long as no hole pixel lies deeper than DEPTH from a known one; a deeper hole
is filled at a smaller scale and scaled back up.  Windows larger than MAX
on a side are filled in overlapped tiles.

All images here are float32 arrays of shape (H, W, 3) with values in 0..1
(model space); holes are (H, W) arrays, nonzero where to be filled.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import config
import resample

MAX = 1024          # longest side given to the model at once
OUTSIDE = 1e-3      # tolerance for pixels outside the hole
OVERLAP = 64        # minimum overlap between tiles
MARGIN = 32         # minimum context around a hole
DEPTH = 64.0        # deepest hole LaMa fills well at full scale
STEP = 8            # model sizes are multiples of this
SMALLEST = 32       # smallest side the model takes
WARM = 512


class ImplausibleFill(Exception):
    pass


class ModelMissing(Exception):
    pass


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    width: int
    height: int


def _next_multiple(n, m):
    return -(-n // m) * m


def _span(start, length, margin, frame):
    start = min(start, frame)
    end = min(start + length, frame)
    a = max(start - margin, 0)
    b = min(end + margin, frame)
    want = min(_next_multiple(b - a, STEP), frame)
    grow = want - (b - a)
    right = min(grow, frame - b)
    b += right
    a -= grow - right
    return a, b - a


def window(hole: Window, depth: float, frame_width: int, frame_height: int) -> Window:
    """The window in which a hole of the given bounding box and depth is
    filled: its box plus half its longer side on every side (no more than
    twice the depth, no less than MARGIN), grown to multiples of 8."""
    margin = max(min(max(hole.width, hole.height) // 2,
                     int(math.ceil(2.0 * depth))), MARGIN)
    x, w = _span(hole.x, hole.width, margin, frame_width)
    y, h = _span(hole.y, hole.height, margin, frame_height)
    return Window(x, y, w, h)


def depth(hole):
    """The largest distance from a hole pixel to a known one.  The area
    outside the array counts as neither (so a hole at the frame's edge is as
    deep as its distance from the known side).  With no known pixel at all,
    the longer side."""
    h = np.asarray(hole) > 0
    if h.all():
        return float(max(h.shape))
    return float(ndimage.distance_transform_edt(h).max())


def scale_for(d):
    return 1.0 if d <= DEPTH else DEPTH / d


def mirror_index(n, length):
    """Indexes 0..n-1 folded into 0..length-1 as a mirror at the far edge,
    without repeating the edge pixel."""
    i = np.arange(n)
    if length < 2:
        return np.zeros(n, dtype=int)
    period = 2 * (length - 1)
    k = i % period
    return np.where(i < length, i, np.where(k < length, k, period - k))


def _spans(length, mx):
    if length <= mx:
        return [(0, length)]
    if mx <= OVERLAP:
        raise ValueError(f'tile size must exceed {OVERLAP}')
    n = -(-(length - OVERLAP) // (mx - OVERLAP))
    size = min(_next_multiple(-(-(length + (n - 1) * OVERLAP) // n), STEP), mx)
    return [((i * (length - size) + (n - 1) // 2) // (n - 1), size)
            for i in range(n)]


def tiles(width, height, mx=MAX):
    xs, ys = _spans(width, mx), _spans(height, mx)
    return [Window(x, y, w, h) for (y, h) in ys for (x, w) in xs]


def _ramp(spans, i):
    """Per-pixel weight along one axis for tile i (full axis length array
    over the tile's own extent)."""
    start, size = spans[i]
    end = start + size
    p = np.arange(start, end)
    center = p + 0.5
    w = np.ones(size, dtype=np.float32)
    if i > 0:
        prev_end = spans[i - 1][0] + spans[i - 1][1]
        m = p < prev_end
        w[m] = np.minimum(w[m], (center[m] - start) / (prev_end - start))
    if i + 1 < len(spans):
        nxt = spans[i + 1][0]
        m = p >= nxt
        w[m] = np.minimum(w[m], (end - center[m]) / (end - nxt))
    return np.clip(w, 0.0, 1.0)


def implausible(image, hole, out):
    """Why `out` cannot be right for `image`/`hole`, or None."""
    known = np.asarray(hole) <= 0
    d = np.abs(np.clip(image, 0.0, 1.0)[known] - out[known])
    if d.size and d.max() > OUTSIDE:
        return ('the picture outside the hole came back changed '
                f'(by {float(d.max()):.3f})')
    return None


def tiled(image, hole, mx, run):
    """Fill `image` where `hole` is set with `run(image, hole) -> image`,
    one tile at a time.  Returns (image, number of tiles run)."""
    h, w = image.shape[:2]
    xs, ys = _spans(w, mx), _spans(h, mx)
    if len(xs) == 1 and len(ys) == 1:
        return run(image, hole), 1
    acc = np.zeros((h, w, 3), np.float32)
    weight = np.zeros((h, w), np.float32)
    ran = 0
    for j, (ty, th) in enumerate(ys):
        wy = _ramp(ys, j)
        for i, (tx, tw) in enumerate(xs):
            wx = _ramp(xs, i)
            sl = (slice(ty, ty + th), slice(tx, tx + tw))
            wt = weight[sl]
            handed = wt >= 0.5
            data = np.where(handed[..., None],
                            acc[sl] / np.maximum(wt, 1e-12)[..., None],
                            image[sl]).astype(np.float32)
            part = np.where(handed, 0.0, hole[sl]).astype(np.float32)
            if (part > 0).any():
                ran += 1
                out = run(data, part)
            else:
                out = data
            tw_ = wy[:, None] * wx[None, :]
            acc[sl] += tw_[..., None] * out
            weight[sl] += tw_
    res = acc / np.maximum(weight, 1e-12)[..., None]
    res[weight <= 0] = 0
    return res.astype(np.float32), ran


@dataclass
class Filled:
    image: np.ndarray
    tiles: int
    scale: float
    size: tuple


def filled(image, hole, d, mx, run):
    """Fill `image` where `hole` is set; at its own scale when the depth `d`
    is within DEPTH, else shrunk, filled and brought back up under the hole
    alone."""
    h, w = image.shape[:2]
    scale = scale_for(d)
    if scale >= 1.0:
        out, n = tiled(image, hole, mx, run)
        return Filled(out, n, 1.0, (w, h))
    sw = max(int(round(w * scale)), 1)
    sh = max(int(round(h * scale)), 1)
    small = resample.area(image, sw, sh)
    small_hole = resample.any_covered(hole, sw, sh)
    answer, n = tiled(small, small_hole, mx, run)
    back = resample.bilinear(answer, w, h)
    out = image.copy()
    m = hole > 0
    out[m] = back[m]
    return Filled(out, n, sw / w, (sw, sh))


# --------------------------------------------------------------------------
# Backends: callables (image NCHW f32 0..1 blanked in hole, mask NCHW 0/1)
# -> output NCHW f32.

COREML = 'CoreMLExecutionProvider'
_ALIASES = {'cpu': 'CPUExecutionProvider', 'coreml': COREML,
            'cuda': 'CUDAExecutionProvider', 'rocm': 'ROCMExecutionProvider',
            'directml': 'DmlExecutionProvider',
            'openvino': 'OpenVINOExecutionProvider'}


class OnnxBackend:
    """LaMa on onnxruntime.  The CoreML provider only takes static shapes:
    with it (model made by tools/make_lama_coreml.py) there is one session
    for each window size, made when first needed (the compiled networks are
    cached on disk)."""
    name = 'onnx'
    # CoreML: windows are mirrored out to a square of one of these sizes (a
    # compiled network is cached on disk for each: 0.4 GB), and smaller ones
    # run on the CPU, which is as fast as starting a session
    LADDER = (256, 384, 512, 768, 1024)
    CPU_BELOW = 192
    MAX_SESSIONS = 3

    def __init__(self, path, providers=None, cache_dir=None):
        import onnxruntime as ort
        if not os.path.exists(path):
            raise ModelMissing(f'LaMa model not found: {path} '
                               '(run with --download-heal-model)')
        avail = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ('CUDAExecutionProvider',) if p in avail]
        providers = [_ALIASES.get(p, p) for p in providers
                     if p != 'CPUExecutionProvider']
        self._ort = ort
        self._path = path
        self._cache = cache_dir
        self._providers = providers
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._so = so
        self._sessions = {}     # (height, width) -> session (CoreML only)
        self.coreml = COREML in providers
        self.session = None     # the dynamic session (CPU, CUDA...)
        if self.coreml:
            self.provider = COREML
            self._session_for(WARM, WARM)
        else:
            self._cpu_session(providers)
        self(np.full((1, 3, WARM, WARM), 0.5, np.float32),
             np.zeros((1, 1, WARM, WARM), np.float32))

    def _cpu_session(self, providers=()):
        self.session = self._ort.InferenceSession(
            self._path, self._so,
            providers=list(providers) + ['CPUExecutionProvider'])
        if not self.coreml:
            self.provider = self.session.get_providers()[0]

    def _session_for(self, h, w):
        ort = self._ort
        key = (h, w)
        sess = self._sessions.pop(key, None)
        if sess is None:
            so = ort.SessionOptions()
            so.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            so.add_free_dimension_override_by_name('8*hk', h)
            so.add_free_dimension_override_by_name('8*wk', w)
            opts = {'ModelFormat': 'MLProgram',
                    'MLComputeUnits': 'CPUAndGPU'}
            if self._cache:
                opts['ModelCacheDirectory'] = os.path.join(
                    self._cache, f'lama-{h}x{w}')
            sess = ort.InferenceSession(
                self._path, so,
                providers=[(COREML, opts), 'CPUExecutionProvider'])
        self._sessions[key] = sess      # most recently used last
        while len(self._sessions) > self.MAX_SESSIONS:
            del self._sessions[next(iter(self._sessions))]
        return sess

    def _run(self, image, mask):
        if not self.coreml:
            return self.session.run(
                ['output'], {'image': image, 'mask': mask})[0]
        h, w = image.shape[2:]
        side = max(h, w)
        if side <= self.CPU_BELOW or side > self.LADDER[-1]:
            if self.session is None:
                self._cpu_session()
            return self.session.run(
                ['output'], {'image': image, 'mask': mask})[0]
        n = next(n for n in self.LADDER if n >= side)
        iy, ix = mirror_index(n, h), mirror_index(n, w)
        out = self._session_for(n, n).run(
            ['output'], {'image': np.ascontiguousarray(image[:, :, iy][..., ix]),
                         'mask': np.ascontiguousarray(mask[:, :, iy][..., ix])})[0]
        return out[:, :, :h, :w]

    def __call__(self, image, mask):
        try:
            return self._run(image, mask)
        except Exception:
            if self.provider == 'CPUExecutionProvider':
                raise
            # an accelerator that cannot run the model: use the CPU
            self.coreml = False
            self._sessions.clear()
            self._cpu_session()
            self.provider = 'CPUExecutionProvider'
            return self._run(image, mask)


class Filler:
    """Runs one backend on windows of any size (up to MAX per side)."""

    def __init__(self, backend, mx=MAX):
        self.backend = backend
        self.max = mx

    def fill(self, image, hole):
        """One model run: `image` in 0..1 with `hole` nonzero where made up.
        Sides not a multiple of 8 are mirrored out and cut back."""
        h, w = image.shape[:2]
        pw = max(_next_multiple(w, STEP), SMALLEST)
        ph = max(_next_multiple(h, STEP), SMALLEST)
        if w == 0 or h == 0 or pw > self.max or ph > self.max:
            raise ValueError(f'fill wants a side of 1 to {self.max}, '
                             f'got {w}x{h}')
        iy, ix = mirror_index(ph, h), mirror_index(pw, w)
        m = (hole[iy][:, ix] > 0).astype(np.float32)
        img = np.clip(image[iy][:, ix], 0.0, 1.0)
        planes = np.ascontiguousarray(
            (img * (1.0 - m[..., None])).transpose(2, 0, 1)[None])
        mask = np.ascontiguousarray(m[None, None])
        out = np.asarray(self.backend(planes, mask), dtype=np.float32)
        if out.shape != (1, 3, ph, pw):
            raise ImplausibleFill(f'unexpected output shape {out.shape}')
        if not np.isfinite(out).all():
            raise ImplausibleFill('the fill is not finite')
        if out.max() > 2.0:    # some exports answer in 0..255
            out = out / 255.0
        out = np.clip(out, 0.0, 1.0)
        # composite ourselves, so no backend can change the known part
        out = out[0].transpose(1, 2, 0)
        out = m[..., None] * out + (1.0 - m[..., None]) * img
        out = np.ascontiguousarray(out[:h, :w])
        why = implausible(image, hole, out)
        if why:
            raise ImplausibleFill(why)
        return out

    def fill_window(self, image, hole, d):
        return filled(image, hole, d, self.max, self.fill)


def make_filler(conf):
    """Filler on the configured providers (by default CoreML if available,
    else the CPU)."""
    provs = conf.heal_providers
    if provs is None:
        import onnxruntime as ort
        if COREML in ort.get_available_providers():
            provs = [COREML]
    be = OnnxBackend(str(conf.get_heal_model_file()), provs,
                     str(config.Config.models_dir() / 'coreml-cache'))
    return Filler(be, conf.heal_max_tile)
