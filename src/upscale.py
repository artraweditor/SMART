"""AI upscaling (2x or 4x) with Real-ESRGAN.

The networks are the RRDBNet models of Real-ESRGAN (BSD-3-Clause) exported to
ONNX by SceneWorks (https://huggingface.co/SceneWorks/real-esrgan-onnx).  They
take an sRGB-encoded RGB image in 0..1 and return it `scale` times larger.

Inputs are saved in their own format (see `save`): TIFF, PNG and JPEG images
through `imgio` (same bit depth, profile, alpha), and linear (demosaiced)
DNGs, which are written back as linear DNGs of the new size.  The network
works on white balanced, sRGB-encoded data in the latter case; the white
balance (and the encoding) is undone on the result, which stays in camera RGB.
"""

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import colorspace
import config
import denoise
import dng
import imgio

ModelMissing = denoise.ModelMissing
Cancelled = denoise.Cancelled

SATURATED = 0.999       # linear DNGs: samples at or above are kept as they are


class UpscaleError(Exception):
    pass


@dataclasses.dataclass
class Options:
    scale: int = 2
    model_file: str | None = None   # ONNX file overriding the model
    provider: str = "auto"
    tile: int = 512                 # input pixels per tile (without margin)
    margin: int = 16                # extra input pixels around each tile
    dng_bits: int = 16
    verbose: bool = False

    @staticmethod
    def from_config(conf, **kw):
        return Options(
            scale=conf.upscale_scale, provider=conf.upscale_provider,
            tile=conf.upscale_tile, margin=conf.upscale_margin,
            dng_bits=conf.upscale_dng_bits, **kw)

    def check(self):
        if self.scale not in config.UPSCALE_MODELS:
            raise ValueError(f"unsupported scale: {self.scale} (use one of "
                             f"{', '.join(map(str, config.UPSCALE_MODELS))})")
        if self.tile < 16:
            raise ValueError("the tile size must be at least 16")
        if self.margin < 0:
            raise ValueError("the margin cannot be negative")

    @property
    def side(self):
        """Side of the network input (tile and margins)."""
        return self.tile + 2 * self.margin

    def model_path(self):
        self.check()
        if self.model_file:
            path = Path(self.model_file)
        else:
            path = config.Config().get_upscale_model_file(self.scale)
        if not path.is_file():
            hint = "" if self.model_file else \
                f" (run with --download-upscale-model {self.scale})"
            raise ModelMissing(f"upscaling model not found: {path}{hint}")
        return path

    def runner_key(self):
        return (str(self.model_path()), self.provider, self.side)


def make_run(opts):
    """The network as a callable: float32 [1,3,s,s] -> [1,3,k*s,k*s]."""
    path = opts.model_path()
    # accelerators want static shapes: the tiles always have the same size
    state, session_for = denoise._ort_sessions(
        path, opts.provider, opts.verbose,
        {"h": opts.side, "w": opts.side}, f"up{opts.side}")

    def run(x):
        return denoise._guarded(
            state, session_for,
            lambda s: s.run(["output"], {"input": x})[0])

    run.state = state
    return run


# ---------------------------------------------------------------------------
# Tiled inference


def upscale_array(x, run, scale, tile, margin, progress=None):
    """Upscale a float32 HxWx3 array (sRGB-encoded, 0..1) `scale` times, tile
    by tile.  Every tile is fed to the network with a mirrored margin, of
    which only the core is kept.  `progress(i, n)` is called after each tile
    (it may raise `Cancelled`)."""
    h, w, _ = x.shape
    side = tile + 2 * margin
    out = np.empty((h * scale, w * scale, 3), np.float32)
    ys_tiles = range(0, h, tile)
    xs_tiles = range(0, w, tile)
    total, done = len(ys_tiles) * len(xs_tiles), 0
    for ty in ys_tiles:
        ys = denoise.mirror_index(ty - margin + np.arange(side), h) \
            if h > 1 else np.zeros(side, int)
        ch = min(tile, h - ty)
        for tx in xs_tiles:
            xs = denoise.mirror_index(tx - margin + np.arange(side), w) \
                if w > 1 else np.zeros(side, int)
            cw = min(tile, w - tx)
            win = x[np.ix_(ys, xs)]
            inp = np.ascontiguousarray(win.transpose(2, 0, 1)[None],
                                       dtype=np.float32)
            res = np.asarray(run(inp))
            if res.shape != (1, 3, side * scale, side * scale):
                raise RuntimeError(
                    f"network returned {res.shape}, wanted "
                    f"(1, 3, {side * scale}, {side * scale})")
            if not np.isfinite(res).all():
                raise RuntimeError("the network produced non-finite values")
            core = res[0, :, margin * scale:(margin + ch) * scale,
                       margin * scale:(margin + cw) * scale]
            out[ty * scale:(ty + ch) * scale,
                tx * scale:(tx + cw) * scale] = core.transpose(1, 2, 0)
            done += 1
            if progress:
                progress(done, total)
    return out


def nearest(arr, scale):
    """Nearest-neighbour upscaling of an HxW(x...) array."""
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _keep_source(out, src, mask, scale):
    """Copy the source samples (nearest neighbour) into `out` where `mask`
    (HxW, at the source resolution) is set."""
    if not mask.any():
        return
    up = nearest(mask, scale)
    ys, xs = np.nonzero(up)
    out[ys, xs] = src[ys // scale, xs // scale]


# ---------------------------------------------------------------------------
# Sources and results


@dataclasses.dataclass
class Result:
    scale: int
    model: str
    provider: str
    data: np.ndarray                 # float32 HxWx3: camera RGB (linear DNG)
    alpha: np.ndarray | None = None  # rasters: resized alpha channel
    # for the preview of a result: see `render_display`


def is_dng(path):
    return str(path).lower().endswith(".dng")


def load_source(path):
    """A `dng.LinearDNG` or an `imgio.Img`."""
    if is_dng(path):
        try:
            if dng.photometric(path) != dng.TAG_PHOTOMETRIC_LINEAR:
                raise UpscaleError(
                    "mosaiced DNGs cannot be upscaled: demosaic the image "
                    "first with `ART-cli --raw-preprocess=demosaic`")
        except dng.DNGError as e:
            raise UpscaleError(str(e)) from e
        return dng.read_linear_dng(path)
    return imgio.load(path)


def source_size(src):
    """(width, height) of a source."""
    arr = src.rgb if isinstance(src, dng.LinearDNG) else src.data
    return arr.shape[1], arr.shape[0]


def _resize_alpha(alpha, scale):
    h, w = alpha.shape
    big = Image.fromarray(alpha.astype(np.float32), "F").resize(
        (w * scale, h * scale), Image.LANCZOS)
    out = np.asarray(big)
    if alpha.dtype.kind == "f":
        return np.clip(out, 0.0, 1.0).astype(alpha.dtype)
    return np.clip(np.rint(out), 0, np.iinfo(alpha.dtype).max
                   ).astype(alpha.dtype)


def upscale_source(src, opts, run=None, progress=None):
    """Upscale a source from `load_source`; `run` is a cached `make_run`."""
    opts.check()
    if run is None:
        run = make_run(opts)
    s = opts.scale
    if isinstance(src, dng.LinearDNG):
        # work on the white balanced data (a neutral has equal channels), as
        # the denoiser does, encoded like the images the network was trained
        # on; both are undone on the result
        gains = src.gains.astype(np.float32)
        clip = float(gains.min())
        x = colorspace.srgb_oetf(np.minimum(src.rgb * gains, clip) / clip)
        out = upscale_array(x, run, s, opts.tile, opts.margin, progress)
        del x
        out = colorspace.srgb_eotf(out)
        out *= np.float32(clip) / gains
        # the network never saw clipped samples: keep them as they are
        _keep_source(out, src.rgb, src.rgb.max(axis=2) >= SATURATED, s)
        res = Result(s, opts.model_path().name, run.state["prov"][0], out)
        return res
    img = src
    data = img.data
    if img.linear:      # float TIFFs are linear
        x = colorspace.srgb_oetf(np.nan_to_num(data))
    else:
        x = np.clip(data, 0.0, 1.0)
    out = upscale_array(x, run, s, opts.tile, opts.margin, progress)
    del x
    if img.linear:
        out = colorspace.srgb_eotf(out)
        _keep_source(out, data, (np.nan_to_num(data) > 1.0).any(axis=2), s)
    else:
        np.clip(out, 0.0, 1.0, out=out)
    alpha = None if img.alpha is None else _resize_alpha(img.alpha, s)
    return Result(s, opts.model_path().name, run.state["prov"][0], out, alpha)


def result_image(res, src):
    """The `imgio.Img` of the result of a raster source."""
    img = src.copy()
    img.data = res.data
    img.alpha = res.alpha
    r = img.tiff_args.get("resolution")
    if r:
        def scaled(v):
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return (v[0] * res.scale, v[1])     # a rational
            return v * res.scale
        img.tiff_args = dict(img.tiff_args,
                             resolution=tuple(scaled(v) for v in r))
    return img


def default_output(src_path):
    """INPUT-upscaled.EXT, in the directory of the input."""
    p = Path(src_path)
    return p.with_name(p.stem + "-upscaled" + p.suffix)


def save(path, res, src, opts, exiftool=None):
    """Write the result in the format of its extension (the one of the input,
    by default).  Returns True if the metadata of a DNG was copied."""
    path = str(path)
    if isinstance(src, dng.LinearDNG):
        if not path.lower().endswith(".dng"):
            raise UpscaleError("a linear DNG is upscaled to a linear DNG: "
                               "the output needs a .dng name")
        dng.write_linear_dng(
            path, res.data, make=src.make, model=src.model,
            unique=getattr(src, "unique", None), xyz_cam=src.xyz_cam,
            neutral=src.neutral, orientation=src.orientation,
            bits=opts.dng_bits, opcodes=getattr(src, "opcodes", None))
        return denoise.copy_exif(exiftool, src.path, path)
    if path.lower().endswith(".dng"):
        raise UpscaleError("the output cannot be a DNG for this input")
    imgio.save(result_image(res, src), path)
    return False


# ---------------------------------------------------------------------------
# Preview


def render_display(res, src, to_display, gain=1.0):
    """uint8 display image (upright) of a result."""
    if isinstance(src, dng.LinearDNG):
        return denoise.render_display(res.data, src, to_display, gain=gain)
    return display_image(result_image(res, src), to_display)


def display_image(img, to_display):
    """uint8 display image of an `imgio.Img`, by bands (to save memory)."""
    h = img.data.shape[0]
    parts = []
    for y in range(0, h, 512):
        parts.append(to_display(img.display_srgb(
            (slice(y, y + 512), slice(None)))))
    return np.ascontiguousarray(np.concatenate(parts))


def source_display(src, to_display, gain=1.0):
    """uint8 display image of a source."""
    if isinstance(src, dng.LinearDNG):
        return denoise.render_display(src.rgb, src, to_display, gain=gain)
    return display_image(src, to_display)
