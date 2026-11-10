"""Image loading/saving for AI healing mode.

Float TIFFs are read and written without any clipping; 8/16-bit images are
kept at their own bit depth.  Internally pixel data is float32 HxWx3.
"""

import io
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

import colorspace


@dataclass
class Img:
    data: np.ndarray                 # float32 HxWx3
    kind: str                        # 'float', 'u8' or 'u16'
    linear: bool                     # True: linear light (float data)
    icc: bytes | None = None
    exif: bytes | None = None
    alpha: np.ndarray | None = None  # untouched extra channel (HxW), if any
    gray: bool = False               # input was single channel
    src_path: str | None = None
    fmt: str = 'tiff'                # 'tiff', 'png' or 'jpeg'
    tiff_args: dict = field(default_factory=dict)
    to_srgb: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32))
    from_srgb: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32))

    @property
    def size(self):
        return self.data.shape[1], self.data.shape[0]

    def set_profile(self, icc):
        self.icc = icc
        self.to_srgb = colorspace.to_linear_srgb_matrix(icc)
        self.from_srgb = colorspace.from_linear_srgb_matrix(icc)

    def display_srgb(self, sl=None):
        """The image (or the region `sl`, a pair of slices) as sRGB-encoded
        float32 in 0..1 (clipped), for display.  Integer images are shown
        as they are, whatever their profile."""
        data = self.data if sl is None else self.data[sl]
        if self.linear:
            lin = colorspace.apply_matrix(self.to_srgb, data)
            return colorspace.srgb_oetf(np.nan_to_num(lin))
        return np.clip(data, 0.0, 1.0).astype(np.float32)

    def copy(self):
        c = Img(**{k: getattr(self, k) for k in (
            'kind', 'linear', 'icc', 'exif', 'gray', 'src_path', 'fmt',
            'tiff_args', 'to_srgb', 'from_srgb')},
            data=self.data.copy(),
            alpha=None if self.alpha is None else self.alpha.copy())
        return c


def _tifffile():
    import tifffile
    return tifffile


def _png16_decode(path):
    try:
        import imagecodecs
    except ImportError as e:
        raise RuntimeError('reading 16-bit PNG needs the "imagecodecs" '
                           'package') from e
    with open(path, 'rb') as f:
        return imagecodecs.png_decode(f.read())


def _to_rgb(arr):
    """Split an array into (HxWx3 data, alpha or None, gray flag)."""
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2), None, True
    c = arr.shape[2]
    if c == 1:
        return np.repeat(arr, 3, axis=2), None, True
    if c == 2:   # gray + alpha
        return np.repeat(arr[..., :1], 3, axis=2), arr[..., 1], True
    if c == 3:
        return arr, None, False
    return arr[..., :3], arr[..., 3], False


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        tf = _tifffile()
        with tf.TiffFile(path) as t:
            page = t.pages[0]
            arr = page.asarray()
            icc = page.tags['InterColorProfile'].value \
                if 'InterColorProfile' in page.tags else None
            compression = page.compression
            predictor = page.predictor
            planar = page.planarconfig
            res = None
            try:
                xr = page.tags['XResolution'].value
                yr = page.tags['YResolution'].value
                res = (xr, yr)
            except Exception:
                pass
        if arr.ndim == 3 and planar == 2:   # separate planes
            arr = np.moveaxis(arr, 0, -1)
        if isinstance(icc, (bytes, bytearray)):
            icc = bytes(icc)
        else:
            icc = None
        rgb, alpha, gray = _to_rgb(arr)
        if arr.dtype.kind == 'f':
            kind, linear = 'float', True
            data = rgb.astype(np.float32)
            alpha = None if alpha is None else alpha.astype(np.float32)
        elif arr.dtype == np.uint16:
            kind, linear = 'u16', False
            data = rgb.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            kind, linear = 'u8', False
            data = rgb.astype(np.float32) / 255.0
        else:
            raise ValueError(f'unsupported TIFF sample type: {arr.dtype}')
        img = Img(data=np.ascontiguousarray(data), kind=kind, linear=linear,
                  alpha=alpha, gray=gray, src_path=path, fmt='tiff')
        img.tiff_args = dict(compression=compression, predictor=predictor,
                             resolution=res)
        img.set_profile(icc)
        return img
    # PNG / JPEG / others via Pillow
    pil = Image.open(path)
    icc = pil.info.get('icc_profile')
    exif = pil.info.get('exif')
    fmt = 'jpeg' if pil.format == 'JPEG' else 'png'
    if pil.mode in ('I;16', 'I;16B', 'I;16L') or (
            fmt == 'png' and pil.mode in ('RGB', 'RGBA', 'L', 'LA')
            and pil.info.get('bitdepth') == 16):
        arr = _png16_decode(path)
        rgb, alpha, gray = _to_rgb(arr)
        img = Img(data=rgb.astype(np.float32) / 65535.0, kind='u16',
                  linear=False, alpha=alpha, gray=gray, src_path=path,
                  fmt=fmt, exif=exif)
    else:
        gray = pil.mode in ('L', 'LA')
        alpha = None
        if pil.mode in ('RGBA', 'LA', 'PA'):
            alpha = np.array(pil.getchannel('A'))
        pil = pil.convert('RGB')
        img = Img(data=np.array(pil).astype(np.float32) / 255.0,
                  kind='u8', linear=False, alpha=alpha, gray=gray,
                  src_path=path, fmt=fmt, exif=exif)
    img.set_profile(icc)
    return img


def save(img, path):
    """Write `img` in its own format/bit depth.  Float data is not clipped."""
    ext = os.path.splitext(path)[1].lower()
    data = img.data
    if img.gray:
        data = data[..., :1]
    if ext in ('.tif', '.tiff'):
        tf = _tifffile()
        args = img.tiff_args or {}
        if img.kind == 'float':
            arr = data.astype(np.float32)
            if img.alpha is not None:
                arr = np.concatenate([arr, img.alpha[..., None]], axis=2)
        elif img.kind == 'u16':
            arr = np.clip(np.rint(data * 65535.0), 0, 65535).astype(np.uint16)
            if img.alpha is not None:
                arr = np.concatenate([arr, img.alpha[..., None]], axis=2)
        else:
            arr = np.clip(np.rint(data * 255.0), 0, 255).astype(np.uint8)
            if img.alpha is not None:
                arr = np.concatenate([arr, img.alpha[..., None]], axis=2)
        if arr.shape[2] == 1:
            arr = arr[..., 0]
            photometric = 'minisblack'
        else:
            photometric = 'rgb'
        kw = dict(photometric=photometric)
        if img.alpha is not None and photometric == 'rgb':
            kw['extrasamples'] = ['unassalpha']
        comp = args.get('compression')
        name = getattr(comp, 'name', None)
        if name in (None, 'NONE'):
            comp_name = None
        elif name in ('ADOBE_DEFLATE', 'DEFLATE'):
            comp_name = 'zlib'
        elif name == 'LZW':
            comp_name = 'lzw'
        else:
            comp_name = 'zlib'
        pred = getattr(args.get('predictor'), 'name', None)
        want_pred = pred not in (None, 'NONE')
        if comp_name is not None:
            kw['compression'] = comp_name
            if want_pred:
                kw['predictor'] = True
        if args.get('resolution'):
            kw['resolution'] = args['resolution']
        if img.icc:
            kw['iccprofile'] = img.icc
        try:
            tf.imwrite(path, arr, **kw)
        except Exception:
            # e.g. imagecodecs missing for the predictor: write plain deflate
            kw.pop('predictor', None)
            tf.imwrite(path, arr, **kw)
        return
    if ext == '.png' and img.kind == 'u16':
        try:
            import imagecodecs
        except ImportError as e:
            raise RuntimeError('writing 16-bit PNG needs the "imagecodecs" '
                               'package') from e
        arr = np.clip(np.rint(data * 65535.0), 0, 65535).astype(np.uint16)
        if img.gray:
            arr = arr[..., 0]
        if img.alpha is not None:
            arr = np.concatenate(
                [arr if arr.ndim == 3 else arr[..., None],
                 img.alpha[..., None].astype(np.uint16)], axis=2)
        with open(path, 'wb') as f:
            f.write(imagecodecs.png_encode(arr))
        return
    arr = np.clip(np.rint(data * 255.0), 0, 255).astype(np.uint8)
    if img.gray:
        arr = arr[..., 0]
    if img.alpha is not None and ext == '.png':
        arr = np.concatenate(
            [arr if arr.ndim == 3 else arr[..., None],
             img.alpha[..., None].astype(np.uint8)], axis=2)
    pil = Image.fromarray(arr)
    kw = {}
    if img.icc:
        kw['icc_profile'] = img.icc
    if img.exif:
        kw['exif'] = img.exif
    if ext in ('.jpg', '.jpeg'):
        kw['quality'] = 95
    pil.save(path, **kw)
