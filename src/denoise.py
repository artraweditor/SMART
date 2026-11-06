"""Raw-domain denoising of a Bayer mosaic with greycard's learned denoiser.

A port of the raw denoiser of greycard
(https://github.com/jessolmstead/greycard, tools/denoise, GPL-3.0).  The
network demosaics *and* denoises a Bayer mosaic; this module holds the code
around it (noise estimation, variance-stabilising transform, tiling,
backends), plus the preview rendering and output writing used by SMART.

Results are linear, camera-native RGB (black removed, divided by white; no
white balance, no colour matrix) unless a white balance is requested.
"""

import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

import colorspace
import config
import dng

# ---------------------------------------------------------------------------
# Variance-stabilising transform (greycard-core `Vst`, tools/denoise/vst.py)

SILENT_B = 1e-8


class Vst:
    """var = a x + b -> unit variance, zero at x = 0 (generalised Anscombe)."""

    def __init__(self, a, b):
        if a <= 0 and b <= 0:  # a silent channel still needs a finite transform
            a, b = 0.0, SILENT_B
        self.a = np.float32(a)
        self.s0 = np.float32(np.sqrt(3.0 / 8.0 * a * a + b))

    def forward(self, x):
        a, s0 = self.a, self.s0
        return 2.0 * x / (np.sqrt(np.maximum(a * x + s0 * s0, 0.0)) + s0)

    def inverse_clean(self, d):
        return self.a * d * d / 4.0 + d * self.s0


# ---------------------------------------------------------------------------
# Noise model estimation (greycard-core `estimate_noise`, Bayer path)

BLOCK = 16
BINS = 32
MIN_BLOCKS_PER_BIN = 8
MIN_SAMPLES = 32
CLIP = 0.98
RESIDUAL_GAIN = 1.25


def estimate_noise(x, rgb_idx):
    """Noise model var = a*level + b per RGB channel.

    `x`: float32 HxW, levels normalised to 0..1 (black removed), before white
    balance. `rgb_idx`: 2x2 array of 0/1/2 (R/G/B) giving the CFA phase.
    Returns (a[3], b[3], info) -- both greens share one fit, as in greycard.
    """
    h, w = x.shape
    a = np.zeros(3)
    b = np.zeros(3)
    info = {"blocks": [0, 0, 0], "bins": [0, 0, 0]}
    if w < BLOCK + 4 or h < BLOCK + 4:
        return a, b, info

    # Squared residual against the mean of the four same-colour neighbours
    # two samples away; defined for the interior [2, n-2).
    r = x[2:-2, 2:-2] - 0.25 * (
        x[2:-2, :-4] + x[2:-2, 4:] + x[:-4, 2:-2] + x[4:, 2:-2]
    )
    r2 = (r * r).astype(np.float64)

    # Block grid: origins 2, 18, ... such that the block ends before n-2.
    nby = len(range(2, h - 2 - BLOCK + 1, BLOCK))
    nbx = len(range(2, w - 2 - BLOCK + 1, BLOCK))
    if nby == 0 or nbx == 0:
        return a, b, info
    v = x[2 : 2 + nby * BLOCK, 2 : 2 + nbx * BLOCK].astype(np.float64)
    # r2 is offset by 2 relative to x, so block origin 2 is r2 origin 0.
    q = r2[: nby * BLOCK, : nbx * BLOCK]

    def blocks(m):
        return m.reshape(nby, BLOCK, nbx, BLOCK)

    vb, qb = blocks(v), blocks(q)
    clipped = (vb >= CLIP).any(axis=(1, 3))  # (nby, nbx)
    # Block origins are even, so the colour of (y, x) in a block is
    # rgb_idx[y & 1, x & 1].
    cmap = np.tile(np.asarray(rgb_idx), (BLOCK // 2, BLOCK // 2))  # 16x16
    for c in range(3):
        m = (cmap == c)[None, :, None, :]
        n = int(m.sum())
        if n < MIN_SAMPLES:
            continue
        mean = (vb * m).sum(axis=(1, 3)) / n
        var = (qb * m).sum(axis=(1, 3)) / n / RESIDUAL_GAIN
        keep = ~clipped
        mean, var = mean[keep], var[keep]
        info["blocks"][c] = int(mean.size)
        bin_ = np.minimum(
            (np.sqrt(np.maximum(mean, 0.0)) * BINS).astype(int), BINS - 1
        )
        sw = swx = swy = swxx = swxy = 0.0
        used = 0
        for k in range(BINS):
            sel = bin_ == k
            cnt = int(sel.sum())
            if cnt < MIN_BLOCKS_PER_BIN:
                continue
            bm, bv = mean[sel], var[sel]
            order = np.argsort(bv, kind="stable")[: max(cnt // 4, 2)]
            qm, qv = bm[order].mean(), bv[order].mean()
            bias = 1.0 - 1.271 * np.sqrt(2.0 / n)
            qv /= max(bias, 0.5)
            wt = float(cnt)
            sw += wt
            swx += wt * qm
            swy += wt * qv
            swxx += wt * qm * qm
            swxy += wt * qm * qv
            used += 1
        info["bins"][c] = used
        if used == 0:
            continue
        det = sw * swxx - swx * swx
        if used >= 2 and abs(det) > 1e-18:
            ac = (sw * swxy - swx * swy) / det
            bc = (swxx * swy - swx * swxy) / det
        else:
            ac, bc = 0.0, swy / sw
        if ac < 0.0:
            ac, bc = 0.0, swy / sw
        if bc < 0.0:
            bc = 0.0
            ac = swxy / swxx if swxx > 0.0 else 0.0
        a[c], b[c] = ac, bc
    return a, b, info


# ---------------------------------------------------------------------------
# Tiled inference (greycard-ai `Denoiser::run`)

TREENET_TILE = 256         # TreeNet models (linear DNGs): tile and overlap
TREENET_OVERLAP = 64
DEFAULT_ISO = 800          # if a linear DNG does not say
DEFAULT_TILE = 1024
DEFAULT_MARGIN = 96
PLAUSIBLE = 3.0
PLAUSIBLE_RANGE = 0.05
RGGB = np.array([[0, 1], [1, 2]])


def rggb_offset(rgb_idx):
    """(dx, dy) at which the 2x2 pattern reads RGGB, or None."""
    p = np.asarray(rgb_idx)
    for dy in range(2):
        for dx in range(2):
            if all(
                p[(i + dy) % 2, (j + dx) % 2] == RGGB[i, j]
                for i in range(2)
                for j in range(2)
            ):
                return dx, dy
    return None


def mirror_index(i, n):
    """Reflection about the edge samples (keeps the Bayer phase).

    -1 -> 1, n -> n-2.
    """
    period = 2 * (n - 1)
    i = np.mod(i, period)
    return np.where(i >= n, period - i, i)


def tiling(tile, margin):
    """Normalise a tile/margin pair the way greycard does, and check that
    the network takes it."""
    tile += tile & 1
    margin += margin & 1
    tile, margin = max(tile, 32), max(margin, 2)
    side = tile + 2 * margin
    if (side // 2) % 8:
        # The U-Net halves its input three times; other sizes fail in a Concat.
        raise ValueError(f"tile + 2*margin = {side} must be a multiple of 16")
    return tile, margin


def denoise_mosaic(
    x,
    rgb_idx,
    a,
    b,
    run,
    tile=DEFAULT_TILE,
    margin=DEFAULT_MARGIN,
    progress=None,
):
    """Demosaic + denoise a Bayer mosaic.

    `x`: float32 HxW levels (>= 0, any common scale). `a`, `b`: noise model for
    those units, per R/G/B. `run(packed)`: float32 [1,4,s,s] -> [1,3,2s,2s].
    Returns float32 HxWx3 in the same units.
    """
    h, w = x.shape
    if h < 4 or w < 4:
        raise ValueError(f"{w}x{h} is too small")
    off = rggb_offset(rgb_idx)
    if off is None:
        raise ValueError(
            f"not a 2x2 Bayer pattern: {np.asarray(rgb_idx).tolist()}"
        )
    dx, dy = off
    tile, margin = tiling(tile, margin)
    side = tile + 2 * margin
    half = side // 2
    vst = [Vst(a[c], b[c]) for c in range(3)]
    plane_vst = [vst[0], vst[1], vst[1], vst[2]]  # R, G1, G2, B

    out = np.zeros((h, w, 3), np.float32)
    ys_tiles = list(range(-2 * dy, h, tile))
    xs_tiles = list(range(-2 * dx, w, tile))
    total, done = len(ys_tiles) * len(xs_tiles), 0
    for ty in ys_tiles:
        for tx in xs_tiles:
            y_in = ty + dy - margin
            x_in = tx + dx - margin
            ys = mirror_index(y_in + np.arange(side), h)
            xs = mirror_index(x_in + np.arange(side), w)
            win = np.maximum(x[np.ix_(ys, xs)], 0.0)
            packed = np.empty((1, 4, half, half), np.float32)
            for p, (py, px) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                packed[0, p] = plane_vst[p].forward(win[py::2, px::2])
            rgb = np.asarray(run(packed))
            if rgb.shape != (1, 3, side, side):
                raise RuntimeError(
                    f"network returned {rgb.shape}, "
                    f"wanted (1, 3, {side}, {side})"
                )
            rgb = rgb[0]
            _check_plausible(packed[0], rgb, side, margin)
            # Write the core: samples the margin does not cover.
            y0, y1 = max(y_in + margin, 0), min(y_in + side - margin, h)
            x0, x1 = max(x_in + margin, 0), min(x_in + side - margin, w)
            if y0 < y1 and x0 < x1:
                sl = (slice(y0 - y_in, y1 - y_in), slice(x0 - x_in, x1 - x_in))
                for c in range(3):
                    out[y0:y1, x0:x1, c] = vst[c].inverse_clean(rgb[c][sl])
            done += 1
            if progress:
                progress(done, total)
    return out


def _check_plausible(packed, rgb, side, margin):
    """greycard's per-tile sanity check: finite, and not far from the input."""
    core = slice(margin, side - margin)
    sub = rgb[:, core, core]
    if not np.isfinite(sub).all():
        raise RuntimeError("the network produced non-finite values")
    # Input value at each core sample, from its own phase's plane.
    s = side - 2 * margin
    inp = np.empty((s, s), np.float32)
    res = np.empty((s, s), np.float32)
    # margin is even, so core sample (yy, xx) has the phase of
    # (yy-margin, xx-margin).
    for p, (py, px) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        c = (0, 1, 1, 2)[p]
        plane = packed[p][
            margin // 2 : (side - margin) // 2,
            margin // 2 : (side - margin) // 2,
        ]
        inp[py::2, px::2] = plane
        res[py::2, px::2] = sub[c][py::2, px::2]
    mean = float(np.abs(res - inp).mean())
    bound = PLAUSIBLE + PLAUSIBLE_RANGE * float(inp.max() - inp.min())
    if mean > bound:
        raise RuntimeError(
            f"a tile whose answer sits {mean:.1f} stabilised units from its "
            "input"
        )


# ---------------------------------------------------------------------------
# ONNX Runtime backend


def _ort_sessions(model_path, provider, verbose, free_dims, tag):
    """An onnxruntime session on the best provider available (`state`), and
    `session_for(providers)` to rebuild it (e.g. on the CPU)."""
    import onnxruntime as ort

    avail = ort.get_available_providers()
    names = {
        "coreml": "CoreMLExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "directml": "DmlExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    if provider == "auto":
        order = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "OpenVINOExecutionProvider",
        ]
        chain = [p for p in order if p in avail]
    else:
        if names.get(provider, provider) not in avail:
            raise ValueError(f"provider {provider} not available; have {avail}")
        chain = [names.get(provider, provider)]
    chain = [p for p in chain if p != "CPUExecutionProvider"] + [
        "CPUExecutionProvider"
    ]

    def options(p):
        if p == "CoreMLExecutionProvider":
            return {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": str(
                    config.Config.models_dir()
                    / "coreml-cache"
                    / f"{Path(model_path).stem}-{tag}"
                ),
            }
        return {}

    def session_for(prov):
        so = ort.SessionOptions()
        so.log_severity_level = 0 if verbose else 3
        # Accelerators want static shapes (greycard's CoreML session is
        # fixed too).
        for dim, value in free_dims.items():
            so.add_free_dimension_override_by_name(dim, value)
        return ort.InferenceSession(
            str(model_path), so, providers=[(p, options(p)) for p in prov]
        )

    state = {"prov": chain}
    state["sess"] = session_for(chain)
    return state, session_for


def _guarded(state, session_for, call):
    """call(session), falling back to the CPU if an accelerator fails."""
    try:
        return call(state["sess"])
    except Exception as e:  # a GPU provider that cannot run the model
        if state["prov"] == ["CPUExecutionProvider"]:
            raise
        print(
            f"{state['prov'][0]} failed ({str(e).splitlines()[0]}); "
            "falling back to CPU",
            file=sys.stderr,
        )
        state["prov"] = ["CPUExecutionProvider"]
        state["sess"] = session_for(state["prov"])
        return call(state["sess"])


def make_runner(model_path, provider="auto", verbose=False, fixed_side=None):
    state, session_for = _ort_sessions(
        model_path, provider, verbose,
        {"h": fixed_side, "w": fixed_side} if fixed_side else {},
        fixed_side or "dyn")

    def run(packed):
        return _guarded(state, session_for,
                        lambda s: s.run(["rgb"], {"packed": packed})[0])

    run.state = state
    return run


def make_treenet_runner(model_path, provider="auto", verbose=False,
                        tile=TREENET_TILE):
    """A TreeNet model: run(float16 [1,3,T,T], float16 [1,1]) -> float16."""
    state, session_for = _ort_sessions(
        model_path, provider, verbose,
        {"batch_size": 1, "height": tile, "width": tile}, f"tn{tile}")

    def run(x, cond):
        return _guarded(
            state, session_for,
            lambda s: s.run(["output"], {"input": x, "cond": cond})[0])

    run.state = state
    return run


# ---------------------------------------------------------------------------
# PyTorch backend (optional: `pip install torch onnx onnx2torch`)


def make_torch_runner(model_path, device="auto"):
    """The same network through onnx2torch: CUDA / ROCm / MPS / XPU / CPU by
    one code path."""
    import onnx
    import torch
    from onnx2torch import convert

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    net = convert(onnx.load(str(model_path))).eval().to(device)

    def run(packed):
        with torch.inference_mode():
            return (
                net(torch.from_numpy(packed).to(device)).float().cpu().numpy()
            )

    run.state = {"prov": [f"torch:{device}"]}
    return run



# ---------------------------------------------------------------------------
# Options, models, and the whole pipeline


class ModelMissing(Exception):
    pass


class Cancelled(Exception):
    """Raised (e.g. from a progress callback) to abort denoising."""


@dataclasses.dataclass
class Options:
    model: str = "balanced"          # fast, balanced or best
    model_file: str | None = None    # ONNX file overriding the model
    backend: str = "onnxruntime"     # or "torch" (mosaic models only)
    provider: str = "auto"
    noise: tuple | None = None       # (aR, bR, aG, bG, aB, bB) or estimate
    iso: float | None = None         # linear DNGs: override the file's ISO
    tile: int = DEFAULT_TILE
    margin: int = DEFAULT_MARGIN
    wb: str = "none"                 # TIFF output: "none" or "camera"
    rotate: bool = True              # TIFF output: apply the orientation
    dng_bits: int = 16
    verbose: bool = False

    @staticmethod
    def from_config(conf, **kw):
        return Options(
            model=conf.denoise_model, backend=conf.denoise_backend,
            provider=conf.denoise_provider, tile=conf.denoise_tile,
            margin=conf.denoise_margin, dng_bits=conf.denoise_dng_bits, **kw)

    def model_path(self, kind="mosaic"):
        """The model file for this kind of DNG ('mosaic' or 'linear')."""
        if self.model_file:
            path = Path(self.model_file)
        else:
            path = config.Config().get_denoise_model_file(self.model, kind)
        if not path.is_file():
            hint = "" if self.model_file else \
                f" (run with --download-denoise-model {self.model})"
            raise ModelMissing(f"denoise model not found: {path}{hint}")
        return path

    def runner_key(self, kind="mosaic"):
        return (str(self.model_path(kind)), self.backend, self.provider,
                self.tile, self.margin)


def make_run(opts, kind="mosaic"):
    """The network as a callable (for `denoise_mosaic` or
    `denoise_linear`) for this kind of DNG."""
    path = opts.model_path(kind)
    if kind == "linear":
        if opts.backend == "torch":
            raise ValueError("the torch backend is not available for "
                             "linear DNG models")
        return make_treenet_runner(path, opts.provider, opts.verbose)
    if opts.backend == "torch":
        return make_torch_runner(path, opts.provider)
    tile, margin = tiling(opts.tile, opts.margin)
    return make_runner(path, opts.provider, opts.verbose,
                       (tile + 2 * margin) // 2)


@dataclasses.dataclass
class Result:
    rgb: np.ndarray          # float32 HxWx3, camera RGB, sensor orientation
    noise_a: list
    noise_b: list
    model: str
    provider: str
    iso: float | None = None         # linear DNGs: the ISO that was used
    iso_guessed: bool = False


def denoise_dng(raw, opts, run=None, progress=None):
    """Denoise a `dng.MosaicDNG` or `dng.LinearDNG`; `run` is a cached
    `make_run(opts, raw.kind)`."""
    if raw.kind == "linear":
        return denoise_linear_dng(raw, opts, run, progress)
    tile, margin = tiling(opts.tile, opts.margin)
    if opts.noise:
        v = [float(x) for x in opts.noise]
        if len(v) != 6:
            raise ValueError("the noise model wants six numbers")
        a, b = np.array(v[0::2]), np.array(v[1::2])
    else:
        a, b, ninfo = estimate_noise(raw.levels, raw.cfa)
        if opts.verbose:
            print(f"noise a={a.tolist()} b={b.tolist()} {ninfo}",
                  file=sys.stderr)
    if run is None:
        run = make_run(opts, raw.kind)
    rgb = denoise_mosaic(raw.levels, raw.cfa, a, b, run, tile, margin,
                         progress)
    return Result(rgb, a.tolist(), b.tolist(),
                  opts.model_path().name, run.state["prov"][0])


# ---------------------------------------------------------------------------
# Linear (demosaiced) DNGs: TreeNet models of RawForge, conditioned on the
# ISO (trained on linear Rec.2020 RGB; here fed the white balanced camera RGB).


def _ramp(n):
    return np.sin(0.5 * np.pi * (np.arange(n) + 0.5) / n) ** 2


def denoise_linear(x, run, iso, progress=None, tile=TREENET_TILE,
                   overlap=TREENET_OVERLAP):
    """Denoise a float32 HxWx3 linear image tile by tile, cross-fading the
    overlaps."""
    h, w, _ = x.shape
    step = tile - overlap
    ph = tile + -(-max(h - tile, 0) // step) * step
    pw = tile + -(-max(w - tile, 0) // step) * step
    xp = np.pad(x, ((0, ph - h), (0, pw - w), (0, 0)), mode="symmetric")
    acc = np.zeros((ph, pw, 3), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    cond = np.full((1, 1), min(iso, 65535.0) / 6400.0, np.float16)
    ys = range(0, ph - tile + 1, step)
    xs = range(0, pw - tile + 1, step)
    total = len(ys) * len(xs)
    ramp = _ramp(overlap).astype(np.float32)
    done = 0
    for y0 in ys:
        wy = np.ones(tile, np.float32)
        if y0 > 0:
            wy[:overlap] = ramp
        if y0 + tile < ph:
            wy[-overlap:] = ramp[::-1]
        for x0 in xs:
            wx = np.ones(tile, np.float32)
            if x0 > 0:
                wx[:overlap] = ramp
            if x0 + tile < pw:
                wx[-overlap:] = ramp[::-1]
            win = wy[:, None] * wx[None, :]
            inp = np.ascontiguousarray(
                xp[y0:y0 + tile, x0:x0 + tile].transpose(2, 0, 1)[None],
                dtype=np.float16)
            out = run(inp, cond)[0].astype(np.float32).transpose(1, 2, 0)
            acc[y0:y0 + tile, x0:x0 + tile] += out * win[..., None]
            wsum[y0:y0 + tile, x0:x0 + tile] += win
            done += 1
            if progress:
                progress(done, total)
    return (acc / wsum[..., None])[:h, :w]


def denoise_linear_dng(raw, opts, run=None, progress=None):
    if run is None:
        run = make_run(opts, "linear")
    if opts.iso:
        iso, guessed = float(opts.iso), False
    elif raw.iso:
        iso, guessed = float(raw.iso), False
    else:
        iso, guessed = float(DEFAULT_ISO), True
    # stay in camera space: only the as-shot white balance is applied (the
    # network sees roughly equal channels for neutrals) and undone after, with
    # no clipping, so that nothing is lost above white
    gains = raw.gains.astype(np.float32)
    saturated = raw.rgb.max(axis=2) >= 0.999
    out = denoise_linear(raw.rgb * gains, run, iso, progress)
    out = np.nan_to_num(np.maximum(out, 0.0)) / gains
    # the model never saw saturated pixels: keep them as they are
    out[saturated] = raw.rgb[saturated]
    return Result(out, [], [], opts.model_path("linear").name,
                  run.state["prov"][0], iso, guessed)


# ---------------------------------------------------------------------------
# Preview: linear rendering, bilinear demosaic

_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4


def bilinear_demosaic(x, cfa):
    """Plain bilinear demosaicing of a Bayer mosaic (any 2x2 phase):
    float32 HxW -> HxWx3 (camera RGB)."""
    h, w = x.shape
    cmap = np.tile(np.asarray(cfa), (h // 2 + 1, w // 2 + 1))[:h, :w]
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        k = _K_G if c == 1 else _K_RB
        out[..., c] = ndimage.convolve(np.where(cmap == c, x, 0.0), k,
                                       mode="mirror")
    return out


def display_matrix(raw):
    """Camera RGB (linear, as stored) -> linear sRGB: as-shot white balance,
    then the camera matrix, normalised so that the neutral stays neutral."""
    a = raw.xyz_cam @ colorspace._SRGB_TO_XYZ     # sRGB -> camera
    a = a / a.sum(axis=1, keepdims=True)
    return (np.linalg.inv(a) @ np.diag(raw.gains)).astype(np.float32)


def render_display(rgb, raw, to_display, band=256):
    """Linear camera RGB (HxWx3) -> uint8 display image, upright: only the
    display transform (colour space, sRGB encoding, display profile), no
    tone curve or exposure change."""
    m = display_matrix(raw)
    parts = []
    for y in range(0, rgb.shape[0], band):
        lin = colorspace.apply_matrix(m, rgb[y:y + band])
        parts.append(to_display(colorspace.srgb_oetf(np.nan_to_num(lin))))
    return np.ascontiguousarray(
        dng.apply_orientation(np.concatenate(parts), raw.orientation))


# ---------------------------------------------------------------------------
# Output


def copy_exif(exiftool, src, dst):
    """Copy the shooting metadata (exposure, ISO, lens, date...) from the
    source to the written file."""
    import shutil
    import subprocess

    if not exiftool or not shutil.which(exiftool):
        return False
    skip = ["ExifIFD:MakerNotes", "ExifIFD:MakerNote",
            "ExifIFD:ComponentsConfiguration", "ExifIFD:ExifImageWidth",
            "ExifIFD:ExifImageHeight", "ExifIFD:PixelXDimension",
            "ExifIFD:PixelYDimension"]
    cmd = [exiftool, "-q", "-overwrite_original", "-TagsFromFile", str(src),
           "-ExifIFD:all", "-GPS:all", "-IFD0:Artist", "-IFD0:Copyright",
           "-IFD0:DateTime"]
    for t in skip:
        cmd += ["-x", t]
    r = subprocess.run(cmd + [str(dst)], capture_output=True, text=True)
    return r.returncode == 0


def save_result(path, res, raw, opts, exiftool=None):
    """Write the result as a float32 TIFF or, for a .dng name, a LinearRaw
    DNG.  Returns True if the metadata was copied (DNG only)."""
    path = str(path)
    if path.lower().endswith(".dng"):
        dng.write_linear_dng(
            path, res.rgb, make=raw.make, model=raw.model,
            xyz_cam=raw.xyz_cam, neutral=raw.neutral,
            orientation=raw.orientation, bits=opts.dng_bits)
        return copy_exif(exiftool, raw.path, path)
    import tifffile
    rgb = res.rgb
    if opts.wb == "camera":
        rgb = rgb * raw.gains.astype(np.float32)
    if opts.rotate:
        rgb = dng.apply_orientation(rgb, raw.orientation)
    meta = {
        "source": raw.name,
        "model": res.model,
        "noise_a": res.noise_a,
        "noise_b": res.noise_b,
        "wb_applied": opts.wb == "camera",
        "levels": "linear, black removed, divided by white; camera RGB",
    }
    tifffile.imwrite(path, np.ascontiguousarray(rgb, dtype=np.float32),
                     photometric="rgb", description=json.dumps(meta),
                     software="SMART")
    return False
