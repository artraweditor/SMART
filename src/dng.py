"""DNG input/output for raw denoising, on top of tifffile.

Input is deliberately limited to what `ART-cli --raw-preprocess=mosaic` or
`--raw-preprocess=demosaic` writes: a single-IFD, uncompressed DNG with
16-bit integer or 32-bit float samples, either one-sample CFA (Bayer) or
three-sample LinearRaw.  Other raw files (compressed, multi-IFD, ...) are not
supported: convert them first with ART-cli.

The levels are normalised as in the DNG specification: BlackLevel (with its
repeat pattern, and the per-row and per-column deltas) is subtracted and the
result is divided by WhiteLevel minus the black level (no BlackLevel means
black 0).  WhiteLevel is raised to the data maximum if the data exceed it.
For mosaics, the ActiveArea, if there is one, is cropped.  Values below black
are set to 0.

Output is a linear (LinearRaw) DNG 1.4, 16-bit integer or 32-bit float.
"""

import os
import struct
from dataclasses import dataclass

import numpy as np

import colorspace

TAG_PHOTOMETRIC_CFA = 32803
TAG_PHOTOMETRIC_LINEAR = 34892
_ILLUMINANT_PREFERENCE = (21, 20, 23)   # D65, D55, D50


class DNGError(Exception):
    pass


@dataclass
class MosaicDNG:
    levels: np.ndarray       # float32 HxW, 0 = black, 1 = white (may exceed 1)
    cfa: np.ndarray          # 2x2, 0/1/2 = R/G/B at each position
    white: float             # raw value that `levels` == 1 stands for
    make: str | None
    model: str | None
    orientation: int         # TIFF orientation tag (1, 3, 6 or 8)
    neutral: np.ndarray      # as-shot neutral, camera RGB (G = 1)
    xyz_cam: np.ndarray      # 3x3, XYZ (D65) -> camera RGB
    path: str
    iso: float | None = None
    unique: str | None = None   # UniqueCameraModel
    black: float = 0.0          # mean black level, in raw units

    kind = 'mosaic'

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def gains(self):
        """White-balance gains (G = 1) for the as-shot illuminant."""
        return _gains(self.neutral)


@dataclass
class LinearDNG:
    """A demosaiced linear (LinearRaw) DNG, as written by
    `ART-cli --raw-preprocess=demosaic` or by this program."""
    rgb: np.ndarray          # float32 HxWx3, 0 = black, 1 = white
    white: float             # raw value that `rgb` == 1 stands for
    make: str | None
    model: str | None
    orientation: int
    neutral: np.ndarray
    xyz_cam: np.ndarray
    path: str
    iso: float | None = None
    unique: str | None = None   # UniqueCameraModel
    black: float = 0.0          # mean black level, in raw units

    kind = 'linear'

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def gains(self):
        return _gains(self.neutral)


def _gains(neutral):
    g = 1.0 / np.maximum(neutral, 1e-6)
    return g / g[1]


def _floats(tag):
    v = np.asarray(tag.value, np.float64).ravel()
    if int(tag.dtype) in (5, 10):        # (numerator, denominator) pairs
        v = v[0::2] / v[1::2]
    return v


def _tag(page, name):
    t = page.tags.get(name)
    return None if t is None else t.value


def _color_matrix(page):
    """XYZ -> camera matrix, preferring a D65-ish calibration."""
    cands = []
    for k in (1, 2):
        t = page.tags.get(f'ColorMatrix{k}')
        if t is None:
            continue
        m = _floats(t)
        if m.size != 9:
            continue
        ill = _tag(page, f'CalibrationIlluminant{k}')
        cands.append((int(ill) if ill is not None else 0, m.reshape(3, 3)))
    for want in _ILLUMINANT_PREFERENCE:
        for ill, m in cands:
            if ill == want:
                return m
    if cands:
        return cands[0][1]
    # no matrix: pretend the camera is sRGB
    return np.linalg.inv(colorspace._SRGB_TO_XYZ)


def _iso(tf, page):
    """The ISO speed from the Exif IFD (or IFD0), if any."""
    for tags in (_tag(page, 'ExifTag'), None):
        if isinstance(tags, dict):
            v = tags.get('ISOSpeedRatings', tags.get('ISOSpeed'))
        elif tags is None:
            v = _tag(page, 'ISOSpeedRatings')
        else:
            continue
        if v is not None:
            v = np.atleast_1d(v)
            if v.size and float(v[0]) > 0:
                return float(v[0])
    return None


def _open(path):
    import tifffile
    path = str(path)
    try:
        return tifffile.TiffFile(path)
    except Exception as e:
        raise DNGError(f'cannot read {path}: {e}') from e


def _metadata(tf, page):
    nt = page.tags.get('AsShotNeutral')
    neutral = _floats(nt) if nt is not None else np.ones(3)
    if neutral.size != 3 or (neutral <= 0).any():
        neutral = np.ones(3)
    orientation = int(_tag(page, 'Orientation') or 1)
    return dict(
        orientation=orientation if orientation in (3, 6, 8) else 1,
        make=_tag(page, 'Make'), model=_tag(page, 'Model'),
        xyz_cam=np.asarray(_color_matrix(page), np.float64),
        neutral=neutral / neutral[1], iso=_iso(tf, page),
        unique=_tag(page, 'UniqueCameraModel'))


def _white(data, page):
    wl = _tag(page, 'WhiteLevel')
    if data.dtype.kind == 'f':
        nominal = 1.0
    else:
        nominal = float(np.atleast_1d(wl)[0]) if wl is not None \
            else float(np.iinfo(data.dtype).max)
    white = max(nominal, float(data.max()))
    if white <= 0:
        raise DNGError('empty image')
    return white


def _black_pattern(page, channels):
    """The BlackLevel of the page as a (rows, columns, channels) array of the
    repeat pattern, in raw units, or None if there is none."""
    t = page.tags.get('BlackLevel')
    if t is None:
        return None
    v = _floats(t)
    dim = _tag(page, 'BlackLevelRepeatDim')
    rr, rc = (int(dim[0]), int(dim[1])) \
        if dim is not None and len(dim) == 2 else (1, 1)
    if rr < 1 or rc < 1:
        raise DNGError('invalid BlackLevelRepeatDim')
    if v.size == rr * rc * channels:
        return v.reshape(rr, rc, channels)
    if v.size == rr * rc:       # the same for all the channels
        return np.repeat(v.reshape(rr, rc, 1), channels, axis=2)
    raise DNGError('invalid BlackLevel')


def _deltas(page, name, n):
    """BlackLevelDeltaH / BlackLevelDeltaV: n values (or None)."""
    t = page.tags.get(name)
    if t is None:
        return None
    v = _floats(t)
    if v.size < n or not v.any():
        return None
    return v[:n].astype(np.float32)


def _normalize(data, page, white):
    """`data` (HxW or HxWxC) as float32 levels, 0 = black and 1 = `white`
    (the raw value of white), and the mean black level (raw units)."""
    x = data.astype(np.float32)
    x3 = x[..., None] if x.ndim == 2 else x
    h, w, ch = x3.shape
    bp = _black_pattern(page, ch)
    if bp is None:
        x /= np.float32(white)
        black = 0.0
    else:
        if (white - bp <= 0).any():
            raise DNGError('the white level is not above the black level')
        dh = _deltas(page, 'BlackLevelDeltaH', w)
        dv = _deltas(page, 'BlackLevelDeltaV', h)
        if dh is not None:
            x3 -= dh[None, :, None]
        if dv is not None:
            x3 -= dv[:, None, None]
        rr, rc = bp.shape[:2]
        for i in range(rr):
            for j in range(rc):
                sub = x3[i::rr, j::rc]
                for c in range(ch):
                    sub[..., c] -= np.float32(bp[i, j, c])
                    sub[..., c] /= np.float32(white - bp[i, j, c])
        black = float(bp.mean())
    np.maximum(x, 0.0, out=x)
    return x, black


def _active_area(page, h, w):
    """(top, left, bottom, right) of the ActiveArea if it is a proper
    sub-rectangle of the image, else None."""
    aa = _tag(page, 'ActiveArea')
    if aa is None or len(aa) != 4:
        return None
    t, l, b, r = (int(v) for v in aa)
    if (t, l, b, r) == (0, 0, h, w) or not (0 <= t < b <= h and 0 <= l < r <= w):
        return None
    return t, l, b, r


def photometric(path):
    """The PhotometricInterpretation of a DNG (32803 CFA, 34892 LinearRaw)."""
    with _open(path) as tf:
        page = tf.pages[0]
        if 'DNGVersion' not in page.tags:
            raise DNGError('not a DNG file')
        return int(page.photometric)


def read_dng(path):
    """A `MosaicDNG` or a `LinearDNG`, according to the file."""
    if photometric(path) == TAG_PHOTOMETRIC_LINEAR:
        return read_linear_dng(path)
    return read_mosaic_dng(path)


def read_linear_dng(path):
    path = str(path)
    with _open(path) as tf:
        page = tf.pages[0]
        if 'DNGVersion' not in page.tags:
            raise DNGError('not a DNG file')
        if int(page.photometric) != TAG_PHOTOMETRIC_LINEAR:
            raise DNGError(
                'not a linear (demosaiced) DNG; create one with '
                '`ART-cli --raw-preprocess=demosaic`')
        if page.samplesperpixel != 3 or int(page.compression) != 1 \
           or page.dtype.kind not in 'uf' or page.dtype.itemsize < 2:
            raise DNGError(
                'unsupported DNG: only uncompressed 3-channel 16-bit '
                'integer or 32-bit float linear DNGs are supported; create '
                'one with `ART-cli --raw-preprocess=demosaic`')
        data = page.asarray()
        meta = _metadata(tf, page)
        white = _white(data, page)
        rgb, black = _normalize(data, page, white)
    if data.ndim != 3 or data.shape[2] != 3 or min(data.shape[:2]) < 4:
        raise DNGError('unsupported or too small linear DNG')
    return LinearDNG(rgb=rgb, white=white, path=path, black=black, **meta)


def read_mosaic_dng(path):
    path = str(path)
    with _open(path) as tf:
        page = tf.pages[0]
        if 'DNGVersion' not in page.tags:
            raise DNGError('not a DNG file')
        if int(page.photometric) != TAG_PHOTOMETRIC_CFA:
            raise DNGError(
                'not a mosaiced (CFA) DNG; create one with '
                '`ART-cli --raw-preprocess=mosaic`')
        if page.samplesperpixel != 1 or int(page.compression) != 1 \
           or page.dtype.kind not in 'uf' or page.dtype.itemsize < 2:
            raise DNGError(
                'unsupported DNG: only uncompressed 16-bit integer or '
                '32-bit float mosaics are supported; create one with '
                '`ART-cli --raw-preprocess=mosaic`')
        dim = _tag(page, 'CFARepeatPatternDim')
        pat = _tag(page, 'CFAPattern')
        if dim is None or pat is None or tuple(dim) != (2, 2) \
           or len(pat) != 4:
            raise DNGError('not a 2x2 Bayer mosaic (X-Trans is not '
                           'supported by the model)')
        planes = _tag(page, 'CFAPlaneColor')
        planes = list(planes) if planes is not None else [0, 1, 2]
        try:
            idx = [planes[c] for c in bytes(pat)]
        except IndexError:
            raise DNGError('invalid CFA pattern') from None
        if set(idx) - {0, 1, 2}:
            raise DNGError('unsupported colour filter: the model reads '
                           'RGB Bayer only')
        cfa = np.array(idx).reshape(2, 2)
        data = page.asarray()
        meta = _metadata(tf, page)
        white = _white(data, page)
        # the black level pattern is relative to the whole image
        levels, black = _normalize(data, page, white)
        area = _active_area(page, *data.shape)
    if area is not None:
        top, left, bottom, right = area
        levels = np.ascontiguousarray(levels[top:bottom, left:right])
        cfa = np.roll(cfa, (-top, -left), axis=(0, 1))   # the pattern at the new origin
    if levels.shape[0] < 4 or levels.shape[1] < 4:
        raise DNGError('raw image too small')
    return MosaicDNG(levels=levels, cfa=cfa, white=white, path=path,
                     black=black, **meta)


def apply_orientation(img, orientation):
    """Rotate an HxW(x...) array upright, given the TIFF orientation."""
    if orientation == 3:
        return np.rot90(img, 2)
    if orientation == 6:
        return np.rot90(img, -1)
    if orientation == 8:
        return np.rot90(img, 1)
    return img


# --------------------------------------------------------------------------
# Linear DNG output (DNG 1.4, LinearRaw)

_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL, _SRATIONAL = 1, 2, 3, 4, 5, 10
_FMT = {_BYTE: "B", _SHORT: "H", _LONG: "I"}


def _entry(tag, typ, values):
    """A TIFF IFD entry as (tag, type, count, payload bytes)."""
    if typ == _ASCII:
        data = values.encode("ascii", "replace") + b"\0"
        return tag, typ, len(data), data
    if typ in (_RATIONAL, _SRATIONAL):
        fmt = "<II" if typ == _RATIONAL else "<ii"
        return (tag, typ, len(values),
                b"".join(struct.pack(fmt, n, dn) for n, dn in values))
    return (tag, typ, len(values),
            struct.pack("<%d%s" % (len(values), _FMT[typ]), *values))


def _rational(x, den=10000):
    return int(round(x * den)), den


def write_linear_dng(path, rgb, *, make, model, xyz_cam, neutral,
                     orientation=1, software="SMART", bits=16, unique=None):
    """Write an HxWx3 float32 linear image as a LinearRaw DNG: 32-bit float
    (`bits=32`) or 16-bit unsigned integer (`bits=16`, white = 65535,
    negatives clipped).

    `xyz_cam`: 3x3 XYZ(D65) -> camera RGB matrix (DNG ColorMatrix1);
    `neutral`: the camera's response to the as-shot neutral (AsShotNeutral).
    The data are camera RGB with black at 0 and white at 1."""
    if bits == 16:
        rgb = np.ascontiguousarray(
            np.rint(np.clip(rgb, 0.0, 1.0) * 65535.0), dtype="<u2")
    else:
        rgb = np.ascontiguousarray(rgb, dtype="<f4")
    h, w, _ = rgb.shape
    if not unique:
        # UniqueCameraModel is what the camera profiles are matched on, so
        # it must not repeat the make when the model already includes it
        # ("Canon Canon EOS-1D Mark III")
        if not make:
            unique = model or "SMART"
        elif model and model.lower().startswith(make.lower()):
            unique = model
        else:
            unique = f"{make} {model}".strip()
    e = [
        _entry(254, _LONG, [0]),
        _entry(256, _LONG, [w]),
        _entry(257, _LONG, [h]),
        _entry(258, _SHORT, [bits] * 3),
        _entry(259, _SHORT, [1]),
        _entry(262, _SHORT, [34892]),  # LinearRaw
        _entry(271, _ASCII, make or ""),
        _entry(272, _ASCII, model or "SMART"),
        _entry(274, _SHORT, [orientation]),
        _entry(277, _SHORT, [3]),
        _entry(278, _LONG, [h]),
        _entry(284, _SHORT, [1]),
        _entry(305, _ASCII, software),
        _entry(339, _SHORT, [3 if bits == 32 else 1] * 3),
        _entry(50706, _BYTE, [1, 4, 0, 0]),
        _entry(50707, _BYTE, [1, 4, 0, 0]),
        _entry(50708, _ASCII, unique),
        _entry(50714, _RATIONAL, [(0, 1)] * 3),
        _entry(50717, _LONG, [1 if bits == 32 else 65535]),
        _entry(50721, _SRATIONAL,
               [_rational(v) for v in np.asarray(xyz_cam).reshape(9)]),
        _entry(50728, _RATIONAL,
               [_rational(v, 1000000) for v in neutral]),
        _entry(50778, _SHORT, [21]),  # CalibrationIlluminant1 = D65
    ]
    # StripOffsets / StripByteCounts need the final layout: IFD, then the
    # out-of-line values, then the pixel data (4-aligned).
    e += [(273, _LONG, 1, b"\0\0\0\0"),
          (279, _LONG, 1, struct.pack("<I", rgb.nbytes))]
    e.sort(key=lambda t: t[0])
    ifd_size = 2 + 12 * len(e) + 4
    extra = sum((len(d) + 1) & ~1 for _, _, _, d in e if len(d) > 4)
    data_off = (8 + ifd_size + extra + 3) & ~3
    out = bytearray(b"II*\0" + struct.pack("<I", 8)
                    + struct.pack("<H", len(e)))
    spill = bytearray()
    spill_base = 8 + ifd_size
    for tag, typ, count, data in e:
        if tag == 273:
            data = struct.pack("<I", data_off)
        if len(data) <= 4:
            out += struct.pack("<HHI", tag, typ, count) + data.ljust(4, b"\0")
        else:
            out += struct.pack("<HHII", tag, typ, count,
                               spill_base + len(spill))
            spill += data + (b"\0" if len(data) & 1 else b"")
    out += struct.pack("<I", 0)
    out += spill
    out += b"\0" * (data_off - len(out))
    with open(path, "wb") as f:
        f.write(out)
        rgb.tofile(f)
