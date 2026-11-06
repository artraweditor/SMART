import numpy as np
import pytest
import tifffile

import denoise
import dng

RGGB = np.array([[0, 1], [1, 2]])


def write_cfa(path, data, pattern=(0, 1, 1, 2), white=None, orientation=1,
              neutral=(0.5, 1.0, 0.7), extra=()):
    h, w = data.shape
    tags = [
        (33421, 'H', 2, (2, 2)),
        (33422, 'B', 4, bytes(pattern)),
        (50706, 'B', 4, (1, 4, 0, 0)),
        (50721, 10, 9, [v for x in np.eye(3).ravel()
                          for v in (int(x * 10000), 10000)]),
        (50728, 5, 3, [v for x in neutral
                         for v in (int(x * 1000000), 1000000)]),
        (50778, 'H', 1, 21),
    ]
    tags.append((274, 'H', 1, orientation))
    if white is not None:
        tags.append((50717, 'I', 1, white))
    tifffile.imwrite(path, data, photometric=32803, extratags=tags + list(extra),
                     metadata=None, software=False,
                     compression=None)
    return path


def test_read_u16(tmp_path):
    d = (np.arange(8 * 10, dtype=np.uint16).reshape(8, 10) * 100)
    p = write_cfa(tmp_path / 'a.dng', d, white=int(d.max()) + 1000)
    r = dng.read_mosaic_dng(p)
    assert r.white == d.max() + 1000
    np.testing.assert_allclose(r.levels, d / r.white, rtol=1e-6)
    assert r.cfa.tolist() == RGGB.tolist()
    np.testing.assert_allclose(r.neutral, [0.5, 1, 0.7], atol=1e-6)
    np.testing.assert_allclose(r.gains, [2, 1, 1 / 0.7], atol=1e-5)


def test_stale_white_level_is_raised_to_the_data(tmp_path):
    d = np.full((8, 8), 5000, np.uint16)
    r = dng.read_mosaic_dng(write_cfa(tmp_path / 'a.dng', d, white=1023))
    assert r.white == 5000 and r.levels.max() == pytest.approx(1.0)


def test_read_float_and_pattern(tmp_path):
    d = np.random.default_rng(0).random((8, 8), dtype=np.float32)
    p = write_cfa(tmp_path / 'a.dng', d, pattern=(2, 1, 1, 0), white=4095)
    r = dng.read_mosaic_dng(p)
    assert r.cfa.tolist() == [[2, 1], [1, 0]]
    assert r.white == pytest.approx(max(1.0, float(d.max())))


def test_rejects_linear_and_non_dng(tmp_path):
    rgb = np.zeros((8, 8, 3), np.float32)
    dng.write_linear_dng(tmp_path / 'l.dng', rgb, make='a', model='b',
                         xyz_cam=np.eye(3), neutral=[1, 1, 1])
    with pytest.raises(dng.DNGError, match='mosaic'):
        dng.read_mosaic_dng(tmp_path / 'l.dng')
    tifffile.imwrite(tmp_path / 'x.tif', np.zeros((8, 8), np.uint16))
    with pytest.raises(dng.DNGError):
        dng.read_mosaic_dng(tmp_path / 'x.tif')


def test_linear_dng_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    rgb = rng.random((6, 7, 3), dtype=np.float32)
    for bits, dt, tol in ((16, np.uint16, 1 / 65535), (32, np.float32, 0)):
        p = tmp_path / f'l{bits}.dng'
        dng.write_linear_dng(p, rgb, make='Mk', model='Md', xyz_cam=np.eye(3),
                             neutral=[0.5, 1, 0.6], orientation=6, bits=bits)
        with tifffile.TiffFile(p) as t:
            pg = t.pages[0]
            a = pg.asarray()
            assert a.dtype == dt and int(pg.photometric) == 34892
            assert pg.tags['Orientation'].value == 6
        back = a / 65535.0 if bits == 16 else a
        np.testing.assert_allclose(back, rgb, atol=tol + 1e-7)


def test_bilinear_demosaic_exact_on_flat_and_ramp():
    for pat in (RGGB, np.array([[2, 1], [1, 0]]), np.array([[1, 0], [2, 1]])):
        flat = np.full((16, 18), 0.25, np.float32)
        out = denoise.bilinear_demosaic(flat, pat)
        np.testing.assert_allclose(out, 0.25, atol=1e-6)
    # sampled values are kept as they are
    x = np.random.default_rng(2).random((16, 18), dtype=np.float32)
    out = denoise.bilinear_demosaic(x, RGGB)
    assert np.allclose(out[0::2, 0::2, 0], x[0::2, 0::2])
    assert np.allclose(out[0::2, 1::2, 1], x[0::2, 1::2])
    assert np.allclose(out[1::2, 1::2, 2], x[1::2, 1::2])


def test_display_matrix_keeps_neutral_neutral(tmp_path):
    d = np.zeros((8, 8), np.uint16)
    r = dng.read_mosaic_dng(write_cfa(tmp_path / 'a.dng', d, white=100))
    m = denoise.display_matrix(r)
    white = m @ (r.neutral)          # as-shot neutral -> grey
    np.testing.assert_allclose(white, white[0], rtol=1e-5)


def test_orientation():
    a = np.arange(6).reshape(2, 3)
    assert dng.apply_orientation(a, 6).shape == (3, 2)
    assert dng.apply_orientation(a, 6)[0].tolist() == [3, 0]
    assert dng.apply_orientation(a, 8)[0].tolist() == [2, 5]
    assert dng.apply_orientation(a, 3).tolist() == [[5, 4, 3], [2, 1, 0]]
    assert dng.apply_orientation(a, 1) is a


def test_denoise_pipeline_with_fake_network(tmp_path):
    rng = np.random.default_rng(3)
    d = (rng.random((100, 120)) * 3000 + 500).astype(np.uint16)
    r = dng.read_mosaic_dng(write_cfa(tmp_path / 'a.dng', d, white=4095,
                                      orientation=6))
    opts = denoise.Options(tile=32, margin=16, noise=(1e-3, 1e-5) * 3)

    def run(packed):
        # "network": every plane is copied to R, G and B
        n, _, s, _ = packed.shape
        out = np.zeros((1, 3, 2 * s, 2 * s), np.float32)
        for c, p in enumerate((0, 1, 3)):
            out[0, c, 0::2, 0::2] = packed[0, p]
            out[0, c, 0::2, 1::2] = packed[0, p]
            out[0, c, 1::2, 0::2] = packed[0, p]
            out[0, c, 1::2, 1::2] = packed[0, p]
        return out

    run.state = {'prov': ['fake']}
    opts.model_file = str(tmp_path / 'm.onnx')
    open(opts.model_file, 'wb').close()
    res = denoise.denoise_dng(r, opts, run)
    assert res.rgb.shape == (100, 120, 3) and np.isfinite(res.rgb).all()
    # the sampled value of each pixel's own colour is reproduced
    lv = r.levels
    np.testing.assert_allclose(res.rgb[0::2, 0::2, 0], lv[0::2, 0::2],
                               atol=1e-4)
    # outputs
    t = tmp_path / 'o.tif'
    denoise.save_result(t, res, r, opts)
    assert tifffile.imread(t).shape == (120, 100, 3)   # rotated
    opts.rotate = False
    denoise.save_result(t, res, r, opts)
    assert tifffile.imread(t).shape == (100, 120, 3)
    denoise.save_result(tmp_path / 'o.dng', res, r, opts)
    assert tifffile.imread(tmp_path / 'o.dng').shape == (100, 120, 3)


def test_missing_model_message(tmp_path):
    with pytest.raises(denoise.ModelMissing, match='not found'):
        denoise.Options(model_file=str(tmp_path / 'nope.onnx')).model_path()
