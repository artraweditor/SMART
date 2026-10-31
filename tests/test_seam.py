import numpy as np

import colorspace
import seam

W, H = 256, 192


def lin(a):
    return colorspace.srgb_eotf(a).astype(np.float64)


def picture(f):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    v = f(x, y).astype(np.float32)
    return np.repeat(colorspace.srgb_oetf(v)[:, :, None], 3, axis=2)


def round_hole(r, cx=120.0, cy=90.0):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return ((x - cx) ** 2 + (y - cy) ** 2 < r * r).astype(np.float32)


def fill_of(truth, hole, f):
    out = truth.copy()
    m = hole > 0
    out[m] = colorspace.srgb_oetf(f(lin(truth[m])))
    return out


def worst(a, b, keep):
    return float(np.abs(lin(a)[keep] / lin(b)[keep] - 1.0).max())


def within(r):
    y, x = np.mgrid[0:H, 0:W]
    return (x - 120.0) ** 2 + (y - 90.0) ** 2 < r * r


def gradient():
    return picture(lambda x, y: 0.1 + 0.3 * x / W + 0.05 * y / H)


def test_light_fill_on_gradient_comes_back():
    truth = gradient()
    hole = round_hole(40)
    light = fill_of(truth, hole, lambda v: v * 1.14)
    out, info = seam.matched(truth, hole, light)
    assert info is not None
    for g in info['gain']:
        assert 0.85 < g < 0.91
    assert worst(out, truth, within(20)) < 0.01
    assert worst(out, truth, within(40)) < 0.03


def test_rectangle_hole_scaled_gradient():
    truth = picture(lambda x, y: 0.1 + 0.4 * x / W + 0.1 * y / H)
    hole = np.zeros((H, W), np.float32)
    hole[60:130, 70:180] = 1
    fill = fill_of(truth, hole, lambda v: v * 1.12)
    out, info = seam.matched(truth, hole, fill)
    assert info is not None
    for g in info['gain']:
        assert 0.8 <= g <= 1.25
        assert abs(g * 1.12 - 1) < 0.02
    m = hole > 0
    ratio = lin(out)[m].mean() / lin(truth)[m].mean()
    assert abs(ratio - 1) < 0.01


def test_identical_fill_left_alone():
    truth = gradient()
    hole = round_hole(40)
    out, info = seam.matched(truth, hole, truth)
    assert info is None
    assert np.array_equal(out, truth)


def test_thin_stroke_and_no_hole_left_alone():
    truth = picture(lambda x, y: 0.3 + 0 * x)
    light = picture(lambda x, y: 0.4 + 0 * x)
    out, info = seam.matched(truth, np.zeros((H, W), np.float32), light)
    assert info is None
    out, info = seam.matched(truth, np.ones((H, W), np.float32), light)
    assert info is None
    stroke = np.zeros((H, W), np.float32)
    stroke[90:92, 40:200] = 1
    out, info = seam.matched(truth, stroke, light)
    assert info is None
    assert np.array_equal(out, light)
    stroke = np.zeros((H, W), np.float32)
    stroke[90:93, 40:200] = 1  # 3 px stroke
    out, info = seam.matched(truth, stroke, light)
    assert info is None and np.array_equal(out, light)


def test_outside_hole_bit_identical_and_inputs_not_mutated():
    truth = gradient()
    hole = round_hole(40)
    light = fill_of(truth, hole, lambda v: v * 1.14)
    light0, truth0, hole0 = light.copy(), truth.copy(), hole.copy()
    out, info = seam.matched(truth, hole, light)
    assert info is not None
    outside = hole <= 0
    assert np.array_equal(out[outside].view(np.uint32), light[outside].view(np.uint32))
    assert not np.array_equal(out, light)
    assert np.array_equal(light, light0) and np.array_equal(truth, truth0)
    assert np.array_equal(hole, hole0)
    assert out is not light


def test_shape_dtype():
    truth = gradient()
    hole = round_hole(40)
    out, info = seam.matched(truth, hole.astype(np.float64), truth.astype(np.float64))
    assert out.shape == (H, W, 3) and out.dtype == np.float32
    out, info = seam.matched(truth, hole, truth[:, :-1])
    assert info is None and out.dtype == np.float32
    light = fill_of(truth, hole, lambda v: v * 1.1)
    out, info = seam.matched(truth, hole, light)
    assert out.shape == (H, W, 3) and out.dtype == np.float32
    assert len(info['gain']) == 3


def test_sectors_disagree_left_alone():
    truth = gradient()
    hole = round_hole(40)
    x = np.mgrid[0:H, 0:W][1]
    m = hole > 0
    fill = truth.copy()
    f = np.where(x[m] < 120, 0.8, 1.2)[:, None]
    fill[m] = colorspace.srgb_oetf(lin(truth[m]) * f)
    out, info = seam.matched(truth, hole, fill)
    assert info is None
    assert np.array_equal(out, fill)


def test_two_levels_near_and_far():
    truth = gradient()
    hole = round_hole(40)
    x = np.mgrid[0:H, 0:W][1]
    m = hole > 0
    for light in (1.03, 1.05, 1.08):
        f = np.where(x[m] < 120, light, 1.0)[:, None]
        fill = truth.copy()
        fill[m] = colorspace.srgb_oetf(lin(truth[m]) * f)
        _, info = seam.matched(truth, hole, fill)
        assert info is not None
        for g in info['gain']:
            assert 1.0 / light < g < 1.0
    f = np.where(x[m] < 120, 1.14, 1.0)[:, None]
    fill = truth.copy()
    fill[m] = colorspace.srgb_oetf(lin(truth[m]) * f)
    out, info = seam.matched(truth, hole, fill)
    assert info is None and np.array_equal(out, fill)


def test_near_black_matched_by_offset():
    truth = picture(lambda x, y: 0.002 + 0 * x)
    hole = round_hole(40)
    for f in (2.0, 0.0):
        fill = fill_of(truth, hole, lambda v: v * f)
        out, info = seam.matched(truth, hole, fill)
        assert info is not None
        for g in info['gain']:
            assert 0.8 <= g <= 1.25
        v = lin(out)[90, 120, 0]
        assert abs(v - 0.002) < 2e-4


def test_edge_of_picture_along_hole_left_to_fill():
    def boxed(x0, x1, y0, y1):
        b = np.zeros((H, W), np.float32)
        b[y0:y1, x0:x1] = 1
        return b
    wall = picture(lambda x, y: np.where(x < 120, 0.6, 0.1))
    horizon = picture(lambda x, y: np.where(y < 80, 0.5, 0.08))
    for pic, hole in ((wall, boxed(120, 200, 60, 130)),
                      (horizon, boxed(30, 210, 80, 110))):
        out, info = seam.matched(pic, hole, pic)
        if info is None:
            assert np.array_equal(out, pic)
        else:
            assert worst(out, pic, hole > 0) < 0.01


def test_sky_over_dark_hill():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    hill = y > 105.0 + 0.3 * (x - 120.0)

    def rough(dx):
        return 0.04 * (1 + 0.6 * np.sin((x + dx) / 3.0) * np.sin(y / 4.0))
    truth = picture(lambda x_, y_: np.where(hill, rough(0.0), 0.3 + 0.02 * x_ / W))
    hole = round_hole(40)
    m = hole > 0
    fill = truth.copy()
    v = np.where(hill, 1.4 * rough(5.0), lin(truth)[:, :, 0] * 1.05)
    fill[m] = colorspace.srgb_oetf(np.repeat(v[:, :, None], 3, 2)[m])
    out, info = seam.matched(truth, hole, fill)
    assert info is not None
    for g in info['gain']:
        assert abs(g - 1 / 1.05) < 0.01
    sky = within(36) & ~hill
    assert worst(out, truth, sky) < 0.015


def test_window_past_reach_does_not_change_match():
    scene = lambda x, y: 0.2 * (1 + 0.25 * np.sin(x / 10.0) * np.sin(y / 12.0))
    truth = picture(scene)
    hole = round_hole(40)
    light = fill_of(truth, hole, lambda v: v * 1.08)
    small, info = seam.matched(truth, hole, light)

    def grow(a):
        shape = (H + 200, W + 300) + a.shape[2:]
        g = np.zeros(shape, a.dtype)
        if a.ndim == 3:
            g[:] = np.array([0.9, 0.1, 0.5], a.dtype)
        g[:H, :W] = a
        return g
    big, big_info = seam.matched(grow(truth), grow(hole), grow(light))
    assert info is not None
    assert info['gain'] == big_info['gain']
    assert np.array_equal(grow(small), big)


def test_runtime_1024():
    import time
    y, x = np.mgrid[0:1024, 0:1024].astype(np.float32)
    truth = np.repeat(colorspace.srgb_oetf(0.1 + 0.3 * x / 1024 + 0.05 * y / 1024)[:, :, None], 3, 2)
    hole = (((x - 512) ** 2 + (y - 512) ** 2) < 300 ** 2).astype(np.float32)
    fill = truth.copy()
    m = hole > 0
    fill[m] = colorspace.srgb_oetf(lin(truth[m]) * 1.1)
    t = time.time()
    out, info = seam.matched(truth, hole, fill)
    dt = time.time() - t
    assert info is not None
    assert dt < 5.0


def test_patches_every_phase_left_alone():
    hole = round_hole(40)
    base = picture(lambda x, y: 0.2 + 0.15 * np.sin(x / 10.0) * np.sin(y / 12.0))
    for dx in range(0, 63, 3):
        shifted = picture(lambda x, y: 0.2 + 0.15 * np.sin((x + dx) / 10.0) * np.sin(y / 12.0))
        out, _ = seam.matched(shifted, hole, shifted)
        assert worst(out, shifted, within(40)) < 0.002
        fill = base.copy()
        fill[hole > 0] = shifted[hole > 0]
        out, _ = seam.matched(base, hole, fill)
        assert worst(out, fill, within(40)) < 0.01
