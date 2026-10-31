import numpy as np
from PIL import ImageCms, Image

import imgio


def _icc():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def test_float_tiff_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    d = (rng.random((20, 30, 3)) * 50.5 - 0.5).astype(np.float32)
    d[3, 4, 1] = np.nan
    d[5, 6, 0] = np.inf
    img = imgio.Img(data=d, kind='float', linear=True)
    img.set_profile(_icc())
    p = str(tmp_path / 'a.tif')
    imgio.save(img, p)
    back = imgio.load(p)
    assert back.kind == 'float' and back.linear
    assert np.array_equal(back.data, d, equal_nan=True)
    assert back.icc == img.icc


def test_u16_tiff_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    d = (rng.integers(0, 65536, (10, 12, 3)) / 65535.0).astype(np.float32)
    img = imgio.Img(data=d, kind='u16', linear=False)
    img.set_profile(_icc())
    p = str(tmp_path / 'a.tif')
    imgio.save(img, p)
    back = imgio.load(p)
    assert back.kind == 'u16'
    assert np.allclose(back.data, d, atol=1e-6)
    assert back.icc == img.icc


def test_u8_png_jpeg_keep_icc(tmp_path):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
    for name in ('a.png', 'a.jpg'):
        p = str(tmp_path / name)
        Image.fromarray(arr).save(p, icc_profile=_icc())
        im = imgio.load(p)
        assert im.kind == 'u8' and im.icc is not None
        q = str(tmp_path / ('o_' + name))
        imgio.save(im, q)
        assert imgio.load(q).icc is not None


def test_gray_float_tiff(tmp_path):
    import tifffile
    d = np.random.rand(8, 9).astype(np.float32) * 5
    p = str(tmp_path / 'g.tif')
    tifffile.imwrite(p, d)
    im = imgio.load(p)
    assert im.gray and im.data.shape == (8, 9, 3)
    q = str(tmp_path / 'g2.tif')
    imgio.save(im, q)
    assert np.array_equal(tifffile.imread(q), d)
