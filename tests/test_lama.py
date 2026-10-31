import numpy as np
import pytest

import lama


def test_mirror_index():
    assert list(lama.mirror_index(9, 5)) == [0, 1, 2, 3, 4, 3, 2, 1, 0]


def test_window_alignment_and_clip():
    w = lama.window(lama.Window(100, 100, 10, 10), 5.0, 1000, 800)
    assert w.width % 8 == 0 and w.height % 8 == 0
    assert w.x <= 100 and w.x + w.width >= 110
    w = lama.window(lama.Window(0, 0, 10, 10), 5.0, 50, 50)
    assert w.x == 0 and w.y == 0 and w.width <= 50


def test_window_margin_limited_by_depth():
    w = lama.window(lama.Window(500, 500, 400, 5), 3.0, 4000, 4000)
    assert w.height < 100  # thin stroke: no frame's worth around


def test_depth():
    h = np.zeros((50, 50), np.float32)
    h[20:31, 20:31] = 1
    assert 5.0 <= lama.depth(h) <= 6.5
    assert lama.depth(np.ones((10, 20))) == 20


def test_scale_for():
    assert lama.scale_for(10) == 1.0
    assert lama.scale_for(128) == 0.5


def test_spans_formulas():
    assert lama._spans(1000, 1024) == [(0, 1000)]
    s = lama._spans(2500, 1024)
    n = -(-(2500 - 64) // (1024 - 64))
    assert len(s) == n == 3
    assert all(size % 8 == 0 and size <= 1024 for _, size in s)
    assert s[0][0] == 0 and s[-1][0] + s[-1][1] == 2500
    for (a, sa), (b, _) in zip(s, s[1:]):
        assert a + sa - b >= 64


class Dummy:
    """A fake model: fills the hole with the mean colour of known pixels."""
    def __call__(self, image, mask):
        known = 1 - mask
        mean = (image * known).sum((2, 3), keepdims=True) / max(known.sum(), 1)
        return mask * mean + known * image


def test_fill_exact_outside_and_odd_size():
    f = lama.Filler(Dummy())
    rng = np.random.default_rng(0)
    img = rng.random((37, 45, 3)).astype(np.float32)
    hole = np.zeros((37, 45), np.float32)
    hole[10:20, 10:20] = 1
    out = f.fill(img, hole)
    assert out.shape == img.shape
    assert np.array_equal(out[hole == 0], img[hole == 0])
    assert np.isfinite(out).all()


def test_implausible_helper():
    img = np.full((8, 8, 3), 0.5, np.float32)
    hole = np.zeros((8, 8), np.float32)
    out = img.copy()
    out[0, 0] = 0.9
    assert lama.implausible(img, hole, out) is not None
    assert lama.implausible(img, hole, img) is None


def test_backend_garbage_cannot_leak_outside_hole():
    class Bad:
        def __call__(self, image, mask):
            return image * 0 + 0.123
    img = np.full((32, 32, 3), 0.5, np.float32)
    hole = np.zeros((32, 32), np.float32)
    assert np.array_equal(lama.Filler(Bad()).fill(img, hole), img)


def test_not_finite():
    class Nan:
        def __call__(self, image, mask):
            return image * np.nan
    with pytest.raises(lama.ImplausibleFill):
        lama.Filler(Nan()).fill(np.zeros((32, 32, 3), np.float32),
                                np.zeros((32, 32), np.float32))


def test_tiled_constant_image():
    f = lama.Filler(Dummy(), mx=96)
    img = np.full((100, 150, 3), 0.4, np.float32)
    hole = np.zeros((100, 150), np.float32)
    hole[40:60, 10:140] = 1
    res = f.fill_window(img, hole, 10.0)
    assert res.tiles > 1
    assert np.allclose(res.image, 0.4, atol=1e-5)


def test_deep_hole_scaled():
    f = lama.Filler(Dummy())
    img = np.full((400, 400, 3), 0.25, np.float32)
    hole = np.zeros((400, 400), np.float32)
    hole[50:350, 50:350] = 1
    d = lama.depth(hole)
    assert d > 64
    res = f.fill_window(img, hole, d)
    assert res.scale < 1
    assert np.allclose(res.image, 0.25, atol=1e-4)
