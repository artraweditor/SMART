import numpy as np
import pytest
import tifffile
from PIL import Image

import config
import dng
import imgio
import upscale


def fake_run(scale):
    """A network that enlarges by nearest neighbour."""
    def run(x):
        return np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)
    run.state = {'prov': ['CPUExecutionProvider']}
    return run


def opts(scale=2, **kw):
    o = upscale.Options(scale=scale, model_file=__file__, **kw)
    return o


def gradient(h=37, w=53):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx / w, yy / h, (xx + yy) / (w + h)], -1).astype(np.float32)


@pytest.mark.parametrize('scale', [2, 4])
@pytest.mark.parametrize('tile,margin', [(16, 4), (20, 0), (64, 8), (512, 16)])
def test_tiling_is_seamless(scale, tile, margin):
    x = gradient()
    out = upscale.upscale_array(x, fake_run(scale), scale, tile, margin)
    assert out.shape == (37 * scale, 53 * scale, 3)
    np.testing.assert_array_equal(out, upscale.nearest(x, scale))


def test_progress_and_cancel():
    calls = []
    upscale.upscale_array(gradient(), fake_run(2), 2, 16, 2,
                          lambda i, n: calls.append((i, n)))
    assert calls[-1] == (12, 12)

    def cancel(i, n):
        raise upscale.Cancelled()
    with pytest.raises(upscale.Cancelled):
        upscale.upscale_array(gradient(), fake_run(2), 2, 16, 2, cancel)


def test_bad_options():
    with pytest.raises(ValueError):
        upscale.Options(scale=3).check()
    with pytest.raises(ValueError):
        upscale.Options(tile=4).check()


def test_missing_model(tmp_path):
    o = upscale.Options(scale=2, model_file=str(tmp_path / 'nope.onnx'))
    with pytest.raises(upscale.ModelMissing):
        o.model_path()


def test_registry():
    assert set(config.UPSCALE_MODELS) == {2, 4}
    for scale, (name, url, size, sha) in config.UPSCALE_MODELS.items():
        assert url.endswith(name) and f'x{scale}' in name
        assert len(sha) == 64 and size > 0


def run_file(src_path, out_path, scale=2, **kw):
    o = opts(scale, tile=16, margin=4, **kw)
    src = upscale.load_source(src_path)
    res = upscale.upscale_source(src, o, fake_run(scale))
    upscale.save(out_path, res, src, o)
    return src, res


def test_u8_png_with_alpha_and_icc(tmp_path):
    x = (gradient() * 255).astype(np.uint8)
    a = np.arange(37 * 53, dtype=np.uint8).reshape(37, 53)
    p = tmp_path / 'a.png'
    Image.fromarray(np.dstack([x, a])).save(p)
    run_file(p, tmp_path / 'o.png')
    o = Image.open(tmp_path / 'o.png')
    assert o.mode == 'RGBA' and o.size == (106, 74)
    arr = np.asarray(o)
    np.testing.assert_array_equal(arr[::2, ::2, :3], x)


def test_jpeg_same_format(tmp_path):
    p = tmp_path / 'a.jpg'
    Image.fromarray((gradient() * 255).astype(np.uint8)).save(p)
    run_file(p, tmp_path / 'o.jpg')
    o = Image.open(tmp_path / 'o.jpg')
    assert o.format == 'JPEG' and o.size == (106, 74)


def test_u16_tiff_keeps_depth_and_scales_resolution(tmp_path):
    x = (gradient() * 65535).astype(np.uint16)
    p = tmp_path / 'a.tif'
    tifffile.imwrite(p, x, photometric='rgb', resolution=(300, 300))
    run_file(p, tmp_path / 'o.tif', scale=4)
    with tifffile.TiffFile(tmp_path / 'o.tif') as t:
        page = t.pages[0]
        assert page.dtype == np.uint16 and page.shape == (148, 212, 3)
        assert page.tags['XResolution'].value[0] == 1200
        np.testing.assert_array_equal(page.asarray()[::4, ::4], x)


def test_float_tiff_keeps_values_above_one(tmp_path):
    x = gradient() ** 2.2
    x[3, 5] = 4.0
    p = tmp_path / 'a.tif'
    tifffile.imwrite(p, x.astype(np.float32), photometric='rgb')
    run_file(p, tmp_path / 'o.tif')
    o = tifffile.imread(tmp_path / 'o.tif')
    assert o.dtype == np.float32 and o.shape == (74, 106, 3)
    np.testing.assert_allclose(o[::2, ::2], x, atol=2e-3)
    assert o[6:8, 10:12].min() == 4.0


def test_gray_stays_gray(tmp_path):
    p = tmp_path / 'g.png'
    Image.fromarray((gradient()[..., 0] * 255).astype(np.uint8)).save(p)
    run_file(p, tmp_path / 'o.png')
    assert Image.open(tmp_path / 'o.png').mode == 'L'


def write_linear(path, rgb, neutral=(0.5, 1.0, 0.7), orientation=6):
    dng.write_linear_dng(path, rgb, make='Foo', model='Bar',
                         xyz_cam=np.eye(3), neutral=np.array(neutral),
                         orientation=orientation, bits=32)


def test_linear_dng(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.uniform(0.02, 0.4, (30, 40, 1)).astype(np.float32)
    rgb = base * np.array([0.5, 1.0, 0.7], np.float32)   # a grey, in camera RGB
    rgb[:4, :4] = 1.0                                   # clipped corner
    p = tmp_path / 'l.dng'
    write_linear(p, rgb)
    src, res = run_file(p, tmp_path / 'o.dng', dng_bits=32)
    o = dng.read_linear_dng(tmp_path / 'o.dng')
    assert o.rgb.shape == (60, 80, 3)
    assert o.orientation == 6 and (o.make, o.model) == ('Foo', 'Bar')
    np.testing.assert_allclose(o.neutral, [0.5, 1, 0.7], atol=1e-5)
    # the white balance is undone: the grey stays proportional to the neutral
    ref = upscale.nearest(rgb, 2)
    np.testing.assert_allclose(o.rgb, ref, atol=2e-3)
    assert (o.rgb[:8, :8] == 1.0).all()      # saturated samples untouched


def test_dng_output_needs_dng_name_and_mosaic_rejected(tmp_path):
    p = tmp_path / 'l.dng'
    write_linear(p, gradient())
    src = upscale.load_source(p)
    res = upscale.upscale_source(src, opts(tile=16, margin=4), fake_run(2))
    with pytest.raises(upscale.UpscaleError):
        upscale.save(tmp_path / 'o.tif', res, src, opts())
    m = tmp_path / 'm.dng'
    tifffile.imwrite(m, np.zeros((8, 8), np.uint16), photometric=32803,
                     extratags=[(33421, 'H', 2, (2, 2)),
                                (33422, 'B', 4, bytes((0, 1, 1, 2))),
                                (50706, 'B', 4, (1, 4, 0, 0))])
    with pytest.raises(upscale.UpscaleError):
        upscale.load_source(m)


def test_default_output_name():
    assert upscale.default_output('/x/y/photo.JPG').name == 'photo-upscaled.JPG'
