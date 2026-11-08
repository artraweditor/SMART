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


def test_black_and_white_levels(tmp_path):
    rng = np.random.default_rng(0)
    d = rng.integers(500, 4000, (8, 10)).astype(np.uint16)
    black = (512, 520, 530, 540)          # RGGB pattern: R, G1, G2, B
    p = write_cfa(tmp_path / 'a.dng', d, white=4095, extra=[
        (50713, 'H', 2, (2, 2)), (50714, 'H', 4, black)])
    r = dng.read_mosaic_dng(p)
    exp = np.zeros(d.shape, np.float32)
    for i in range(2):
        for j in range(2):
            b = black[2 * i + j]
            exp[i::2, j::2] = np.maximum(d[i::2, j::2].astype(np.float32) - b,
                                         0) / (max(4095, d.max()) - b)
    np.testing.assert_allclose(r.levels, exp, rtol=1e-5, atol=1e-7)
    assert r.black == pytest.approx(np.mean(black))
    assert (r.levels >= 0).all()


def test_black_level_single_value_and_below_black(tmp_path):
    d = np.full((8, 8), 600, np.uint16)
    d[0, 0] = 100                           # below black: clipped to 0
    p = write_cfa(tmp_path / 'a.dng', d, white=4095, extra=[
        (50714, 'H', 1, 512)])
    r = dng.read_mosaic_dng(p)
    assert r.levels[0, 0] == 0
    assert r.levels[1, 1] == pytest.approx((600 - 512) / (4095 - 512))


def test_black_level_deltas(tmp_path):
    d = np.full((8, 8), 1000, np.uint16)
    p = write_cfa(tmp_path / 'a.dng', d, white=4095, extra=[
        (50714, 'H', 1, 500),
        (50715, 10, 8, [v for k in range(8) for v in (10 * k, 1)])])
    r = dng.read_mosaic_dng(p)
    for k in range(8):
        assert r.levels[3, k] == pytest.approx((1000 - 500 - 10 * k)
                                               / (4095 - 500))


def test_active_area_is_cropped_and_shifts_the_pattern(tmp_path):
    d = np.arange(12 * 14, dtype=np.uint16).reshape(12, 14) + 100
    p = write_cfa(tmp_path / 'a.dng', d, white=4095, extra=[
        (50829, 'I', 4, (1, 3, 11, 13))])     # top, left, bottom, right
    r = dng.read_mosaic_dng(p)
    assert r.levels.shape == (10, 10)
    np.testing.assert_allclose(r.levels, d[1:11, 3:13] / max(4095, d.max()),
                               rtol=1e-6)
    # RGGB read from (1, 3) is GBRG-shifted: (y odd, x odd) -> B at (0, 0)
    assert r.cfa.tolist() == [[2, 1], [1, 0]]


def test_original_levels_trust_the_white_level(tmp_path):
    # ART's chromatic aberration correction can overshoot above white at a
    # few samples: they must not rescale the image
    d = np.full((8, 10), 2000, np.uint16)
    d[3, 4] = 6000                          # above white (4095)
    p = write_cfa(tmp_path / 'a.dng', d, white=4095, extra=[
        (50713, 'H', 2, (2, 2)), (50714, 'H', 4, (500, 500, 500, 500))])
    r = dng.read_mosaic_dng(p)
    assert r.white == 4095
    assert r.levels[0, 0] == pytest.approx((2000 - 500) / (4095 - 500))
    assert r.levels[3, 4] == pytest.approx((6000 - 500) / (4095 - 500))
    assert r.levels.max() > 1


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


# ---------------------------------------------------------------------------
# linear (demosaiced) DNGs and the TreeNet models


def make_linear(rgb, iso=None, neutral=(0.5, 1.0, 0.7)):
    return dng.LinearDNG(rgb=rgb.astype(np.float32), white=1.0, make='Mk',
                         model='Md', orientation=1,
                         neutral=np.asarray(neutral), xyz_cam=np.eye(3),
                         path='/x/l.dng', iso=iso)


def fake_treenet(calls=None):
    def run(x, cond):
        assert x.dtype == np.float16 and x.shape[0] == 1 and x.shape[1] == 3
        assert x.shape[2] == x.shape[3] and cond.dtype == np.float16
        if calls is not None:
            calls.append((x.shape[2], float(cond[0, 0])))
        return x
    run.state = {'prov': ['fake']}
    return run


def test_read_linear_dng(tmp_path):
    rng = np.random.default_rng(5)
    d = (rng.random((9, 11, 3)) * 20000).astype(np.uint16)
    tags = [(50706, 'B', 4, (1, 4, 0, 0)), (50717, 'I', 1, 25000),
            (50728, 5, 3, [v for x in (0.5, 1.0, 0.7)
                           for v in (int(x * 1000000), 1000000)]),
            (34855, 'H', 1, 1600), (274, 'H', 1, 8)]
    p = tmp_path / 'l.dng'
    tifffile.imwrite(p, d, photometric=34892, planarconfig='contig',
                     extratags=tags, metadata=None, software=False,
                     compression=None)
    assert dng.photometric(p) == 34892
    r = dng.read_dng(p)
    assert r.kind == 'linear' and r.rgb.shape == (9, 11, 3)
    assert r.white == 25000 and r.iso == 1600 and r.orientation == 8
    np.testing.assert_allclose(r.rgb, d / 25000.0, rtol=1e-6)
    np.testing.assert_allclose(r.gains, [2, 1, 1 / 0.7], atol=1e-5)
    with pytest.raises(dng.DNGError, match='linear'):
        dng.read_linear_dng(write_cfa(tmp_path / 'm.dng',
                                      np.zeros((8, 8), np.uint16)))
    assert dng.read_dng(tmp_path / 'm.dng').kind == 'mosaic'


def test_read_dng_roundtrip_of_our_writer(tmp_path):
    rgb = np.random.default_rng(6).random((8, 9, 3), dtype=np.float32)
    for bits in (16, 32):
        p = tmp_path / f'w{bits}.dng'
        dng.write_linear_dng(p, rgb, make='Mk', model='Md',
                             xyz_cam=np.eye(3), neutral=[0.8, 1, 0.9],
                             bits=bits)
        r = dng.read_dng(p)
        assert r.kind == 'linear' and r.iso is None
        np.testing.assert_allclose(r.rgb * r.white / (65535.0 if bits == 16
                                                      else 1.0),
                                   rgb, atol=2e-5)


@pytest.mark.parametrize('shape,tile,ov', [((100, 130), 32, 8),
                                           ((20, 17), 32, 8),
                                           ((64, 64), 32, 8)])
def test_denoise_linear_tiling_is_seamless(shape, tile, ov):
    x = np.random.default_rng(7).random(shape + (3,), dtype=np.float32)
    seen = []

    def run(t, cond):
        assert t.shape == (1, 3, tile, tile)
        return t

    n = []
    out = denoise.denoise_linear(x, run, 400, lambda k, m: n.append((k, m)),
                                 tile=tile, overlap=ov)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, x, atol=2e-3)       # float16 round trip
    assert n[-1][0] == n[-1][1]


def test_denoise_linear_dng_stays_in_camera_space(tmp_path):
    rng = np.random.default_rng(8)
    rgb = 0.2 + 0.5 * rng.random((70, 90, 3), dtype=np.float32)
    rgb[10:14, 10:14] = 1.0                      # saturated sensor values
    raw = make_linear(rgb, iso=3200, neutral=(0.5, 1, 0.7))  # gains 2, 1, 1.4
    raw.xyz_cam = np.array([[0.8, 0.1, 0.0], [0.1, 0.9, 0.0],
                            [0.0, 0.1, 0.7]])     # must not matter
    seen = []

    def run(x, cond):
        seen.append((float(x.max()), float(cond[0, 0])))
        return x

    run.state = {'prov': ['fake']}
    (tmp_path / 'm.onnx').write_bytes(b'')
    opts = denoise.Options(model_file=str(tmp_path / 'm.onnx'))
    res = denoise.denoise_linear_dng(raw, opts, run)
    assert res.iso == 3200 and not res.iso_guessed
    assert seen[0][1] == pytest.approx(3200 / 6400, abs=1e-3)
    # the network gets the white balanced data without any clipping...
    assert max(m for m, _ in seen) > 1.5
    # ...and an identity network gives the input back (float16 round trip),
    # whatever the colour matrix; saturated pixels are exactly untouched
    np.testing.assert_allclose(res.rgb, rgb, atol=2e-3)
    np.testing.assert_array_equal(res.rgb[10:14, 10:14], rgb[10:14, 10:14])
    # ISO: override, then file, then the default
    opts.iso = 100
    assert denoise.denoise_linear_dng(raw, opts, run).iso == 100
    opts.iso = None
    raw.iso = None
    r = denoise.denoise_linear_dng(raw, opts, run)
    assert r.iso == denoise.DEFAULT_ISO and r.iso_guessed


def test_models_for_both_kinds(tmp_path):
    import config
    assert set(config.LINEAR_DENOISE_MODELS) == set(config.DENOISE_MODELS) \
        == {'small', 'medium', 'large'}
    for tier in config.DENOISE_MODELS:
        a = config.Config().get_denoise_model_file(tier)
        b = config.Config().get_denoise_model_file(tier, 'linear')
        assert a != b and b.name.startswith('treenet')
        assert config.denoise_models('linear')[tier][0] == b.name
    o = denoise.Options(model='large')
    assert 'denoise-v20' in str(config.Config().get_denoise_model_file(
        o.model))
    with pytest.raises(denoise.ModelMissing):
        denoise.Options(model_file=str(tmp_path / 'none.onnx')).model_path(
            'linear')
    lin = make_linear(np.zeros((8, 8, 3)))
    with pytest.raises(denoise.ModelMissing):
        denoise.denoise_dng(lin, denoise.Options(
            model_file=str(tmp_path / 'none.onnx')))
