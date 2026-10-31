"""Color space helpers: sRGB transfer functions and ICC matrix/TRC parsing."""

import struct
import numpy as np

# Bradford chromatic adaptation D50 -> D65
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])
_D50 = np.array([0.9642, 1.0, 0.8249])
_D65 = np.array([0.95047, 1.0, 1.08883])
# linear sRGB (D65) -> XYZ (D65)
_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)


def srgb_oetf(v):
    """Linear -> sRGB-encoded; input is clipped to 0..1."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v,
                    1.055 * np.power(v, 1.0 / 2.4) - 0.055).astype(np.float32)


def srgb_eotf(v):
    """sRGB-encoded -> linear; input is clipped to 0..1."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92,
                    np.power((v + 0.055) / 1.055, 2.4)).astype(np.float32)


def _adapt(src_white, dst_white):
    s = _BRADFORD @ src_white
    d = _BRADFORD @ dst_white
    return np.linalg.inv(_BRADFORD) @ np.diag(d / s) @ _BRADFORD


def icc_rgb_to_xyz(icc):
    """Parse a matrix/TRC ICC profile; return the 3x3 RGB->XYZ(D50) matrix
    (columns are rXYZ, gXYZ, bXYZ), or None if not a matrix profile."""
    try:
        if icc is None or len(icc) < 132:
            return None
        if icc[16:20] != b'RGB ':
            return None
        n = struct.unpack('>I', icc[128:132])[0]
        tags = {}
        for i in range(n):
            sig, off, size = struct.unpack('>4sII', icc[132 + 12 * i:144 + 12 * i])
            tags[sig] = (off, size)
        cols = []
        for sig in (b'rXYZ', b'gXYZ', b'bXYZ'):
            off, size = tags[sig]
            if icc[off:off + 4] != b'XYZ ':
                return None
            x, y, z = struct.unpack('>3i', icc[off + 8:off + 20])
            cols.append(np.array([x, y, z], dtype=np.float64) / 65536.0)
        return np.stack(cols, axis=1)
    except Exception:
        return None


def to_linear_srgb_matrix(icc):
    """3x3 matrix converting linear RGB in the profile's primaries to linear
    sRGB (D65). Identity if the profile is missing or not matrix-based."""
    m = icc_rgb_to_xyz(icc)
    if m is None:
        return np.eye(3, dtype=np.float32)
    xyz_d65 = _adapt(_D50, _D65) @ m
    return (_XYZ_TO_SRGB @ xyz_d65).astype(np.float32)


def from_linear_srgb_matrix(icc):
    return np.linalg.inv(to_linear_srgb_matrix(icc).astype(np.float64)).astype(np.float32)


def apply_matrix(m, img):
    """Apply a 3x3 matrix to an (H, W, 3) image."""
    return (img.reshape(-1, 3) @ np.asarray(m, np.float32).T).reshape(img.shape)
