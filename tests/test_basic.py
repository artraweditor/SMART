import numpy as np
from PIL import ImageCms

import colorspace
import resample


def test_srgb_roundtrip():
    v = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(colorspace.srgb_eotf(colorspace.srgb_oetf(v)), v, atol=1e-5)


def test_icc_srgb_matrix_is_identity():
    p = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    m = colorspace.to_linear_srgb_matrix(p)
    assert np.allclose(m, np.eye(3), atol=2e-3)


def test_area_block_mean():
    a = np.random.rand(16, 16, 3).astype(np.float32)
    got = resample.area(a, 4, 4)
    ref = a.reshape(4, 4, 4, 4, 3).mean(axis=(1, 3))
    assert np.allclose(got, ref, atol=1e-6)


def test_bilinear_constant():
    a = np.full((7, 9, 3), 0.3, np.float32)
    assert np.allclose(resample.bilinear(a, 20, 15), 0.3, atol=1e-6)


def test_any_covered():
    m = np.zeros((8, 8), np.float32)
    m[3, 3] = 1
    s = resample.any_covered(m, 4, 4)
    assert s.sum() == 1 and s[1, 1] == 1


def test_box_matches_bruteforce():
    a = np.random.rand(12, 10).astype(np.float32)
    got = resample.box(a, 2)
    ref = np.zeros_like(a)
    for y in range(12):
        for x in range(10):
            ref[y, x] = a[max(0, y-2):y+3, max(0, x-2):x+3].mean()
    assert np.allclose(got, ref, atol=1e-5)
