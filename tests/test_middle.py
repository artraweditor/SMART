import numpy as np

import colorspace
import middle
from middle import evened

W, H = 256, 192


def grain(i, amount):
    i = np.asarray(i, dtype=np.uint64)
    with np.errstate(over='ignore'):
        z = i + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(40)).astype(np.float64) / float(1 << 24) - 0.5
    return 1.0 + u * amount * np.sqrt(12.0)


IDX = np.arange(W * H).reshape(H, W)
X, Y = IDX % W, IDX // W
TINT = np.array([0.7, 0.9, 1.2])


def enc(v):
    return colorspace.srgb_oetf(np.asarray(v, dtype=np.float32))


def lin(v):
    return colorspace.srgb_eotf(v).astype(np.float64)


def sky(seed=0, shade=1.0):
    v = (0.2 + 0.05 * X / W + 0.08 * Y / H) * grain(IDX * 31 + seed, 0.02) * shade
    return enc(v[..., None] * TINT)


def round_hole(r=40.0):
    return (((X - 128.0) ** 2 + (Y - 96.0) ** 2) < r * r).astype(np.float32)


def stroke():
    return (((X >= 125) & (X < 132) & (Y >= 30) & (Y < 170))).astype(np.float32)


def fill_of(truth, hole, f):
    out = truth.copy()
    f = np.asarray(f, dtype=np.float64)
    v = lin(truth) * (f[..., None] if f.ndim == 2 else f)
    new = enc(v)
    m = hole > 0
    out[m] = new[m]
    return out


def ratio(a, b, keep):
    return lin(a)[keep].mean(axis=0) / lin(b)[keep].mean(axis=0)


def largest(a, b):
    return np.abs(lin(a) / lin(b) - 1.0).max()


def test_bulge_taken_out():
    truth = sky()
    hole = round_hole(40)
    r2 = (X - 128.0) ** 2 + (Y - 96.0) ** 2
    fill = fill_of(truth, hole, 1.0 + 0.02 * (1.0 - r2 / 1600.0))
    out, info = evened(truth, hole, fill)
    assert info is not None
    deep = r2 < 400
    before, after = ratio(fill, truth, deep), ratio(out, truth, deep)
    assert (before > 1.012).all()
    assert (np.abs(after - 1) < 0.003).all(), after
    assert info['largest'] <= middle.LIMIT + 1e-9
    assert len(info['mean']) == 3


def test_ramp_drift_reduced_and_clamped():
    truth = sky()
    hole = round_hole(40)
    ramp = 1.0 + 0.05 * (X - 88.0) / 80.0   # 0.. 1.05 left to right
    fill = fill_of(truth, hole, ramp)
    out, info = evened(truth, hole, fill)
    assert info is not None
    m = hole > 0
    err_before = np.abs(np.log(lin(fill)[m] / lin(truth)[m])).mean()
    err_after = np.abs(np.log(lin(out)[m] / lin(truth)[m])).mean()
    assert err_after < err_before
    assert info['largest'] <= 0.05 + 1e-9
    delta = np.abs(np.log((lin(out)[m] + 0.01) / (lin(fill)[m] + 0.01)))
    assert delta.max() <= 0.05 + 1e-3


def test_fill_equal_truth_unchanged():
    truth = sky()
    hole = round_hole(40)
    fill = truth.copy()
    out, info = evened(truth, hole, fill)
    assert np.abs(out - fill).max() < 0.005


def test_level_under_grain_left_within_a_hair():
    truth = sky()
    hole = round_hole(40)
    for seed in range(5):
        fill = fill_of(truth, hole, grain(IDX * 31 + seed * 977 + 7, 0.02) / grain(IDX, 0.02))
        out, _ = evened(truth, hole, fill)
        assert largest(out, fill) < 0.005, seed


def test_thin_stroke_light_comes_back():
    truth = sky()
    hole = stroke()
    f = grain(IDX * 7 + 3, 0.02) / grain(IDX, 0.02)
    fill = fill_of(truth, hole, f[..., None] * np.array([1.02, 1.025, 1.035]))
    out, info = evened(truth, hole, fill)
    assert info is not None
    r = ratio(out, truth, hole > 0)
    assert (np.abs(r - 1) < 0.003).all(), r


def test_textured_wall_left_alone():
    stone = grain((X // 16) * 1000 + Y // 12 + 17, 0.1)
    v = 0.25 * stone * grain(IDX, 0.03)
    truth = enc(np.repeat(v[..., None], 3, axis=2))
    hole = round_hole(40)
    fill = fill_of(truth, hole, np.where(X < 128, 0.85, 1.0))
    out, info = evened(truth, hole, fill)
    assert info is None
    assert np.array_equal(out, fill)


def test_too_few_known_pixels():
    truth = sky()
    hole = np.ones((H, W), np.float32)
    hole[:2] = 0
    fill = truth * 0.9
    out, info = evened(truth, hole, fill.astype(np.float32))
    assert info is None
    assert np.array_equal(out, fill.astype(np.float32))


def test_no_hole():
    truth = sky()
    out, info = evened(truth, np.zeros((H, W), np.float32), truth)
    assert info is None


def test_outside_hole_bit_identical_and_inputs_not_mutated():
    truth = sky()
    hole = round_hole(40)
    fill = fill_of(truth, hole, 1.0 + 0.04 * (1.0 - ((X - 128.0) ** 2 + (Y - 96.0) ** 2) / 1600.0))
    f0, t0, h0 = fill.copy(), truth.copy(), hole.copy()
    out, info = evened(truth, hole, fill)
    assert info is not None
    outside = hole == 0
    assert np.array_equal(out[outside].view(np.uint32), fill[outside].view(np.uint32))
    assert not np.array_equal(out, fill)
    assert np.array_equal(fill, f0) and np.array_equal(truth, t0) and np.array_equal(hole, h0)
    assert out is not fill


def test_shape_dtype():
    truth = sky()
    hole = round_hole(40)
    fill = fill_of(truth, hole, 1.03)
    out, _ = evened(truth, hole, fill)
    assert out.shape == (H, W, 3) and out.dtype == np.float32
    out2, info2 = evened(truth, round_hole(40)[:, :-1], fill)
    assert info2 is None and out2.shape == fill.shape and out2.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_hole_over_sky_and_hill_corrects_sky_only():
    ridge = 85.0 + 0.2 * X
    base = sky()
    hv = 0.04 * grain((X // 6) * 977 + Y // 5, 0.25) * grain(IDX, 0.05)
    hill = enc(hv[..., None] * np.array([1.1, 1.0, 0.8]))
    truth = np.where((Y < ridge)[..., None], base, hill).astype(np.float32)
    hole = round_hole(44)
    in_sky = Y < ridge - 3.0
    fill = fill_of(truth, hole, np.where(in_sky, 1.02, 1.0))
    out, info = evened(truth, hole, fill)
    assert info is not None
    sky_deep = (hole > 0) & (Y < ridge - 12.0)
    assert (np.abs(ratio(out, truth, sky_deep) - 1) < 0.005).all()
    hill_deep = (hole > 0) & (Y > ridge + 10.0)
    assert (np.abs(ratio(out, fill, hill_deep) - 1) < 0.01).all()


def test_runtime_1024():
    import time
    n = 1024
    yy, xx = np.mgrid[:n, :n]
    v = (0.2 + 0.05 * xx / n + 0.08 * yy / n)
    truth = enc(v[..., None] * TINT)
    hole = (((xx - 512.0) ** 2 + (yy - 512.0) ** 2) < 200.0 ** 2).astype(np.float32)
    fill = truth.copy()
    m = hole > 0
    fill[m] = enc(lin(truth)[m] * 1.04)
    t = time.time()
    out, info = evened(truth, hole, fill)
    assert time.time() - t < 15
    assert info is not None
