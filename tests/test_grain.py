import numpy as np
import pytest

import grain

LUMA = np.array([0.2627, 0.6780, 0.0593])
A, B = 4e-3, 1e-5


def ramp(h, w):
    x = np.linspace(0.08, 0.8, w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 0.1, h, dtype=np.float32)[:, None]
    return np.repeat((x + y)[..., None], 3, axis=2).astype(np.float32)


def noisy(h, w, seed=1, a=A, b=B):
    rng = np.random.default_rng(seed)
    base = ramp(h, w)
    sig = np.sqrt(a * base + b)
    return base, (base + sig * rng.standard_normal(base.shape)).astype(np.float32)


def disc(h, w, r=60):
    yy, xx = np.mgrid[:h, :w]
    return ((xx - w / 2) ** 2 + (yy - h / 2) ** 2 < r * r).astype(np.float32)


def local_std(img, base, m):
    d = (img - base)[m]
    return d.std(), d.mean()


def test_noise_restored_in_hole():
    h, w = 256, 384
    base, frame = noisy(h, w)
    hole = disc(h, w)
    m = hole > 0
    out = base.copy()
    new = grain.regrain(frame, out, hole, LUMA, 5)
    s_frame = (frame - base)[~(disc(h, w, 80) > 0)].std()
    s_new, mean_new = local_std(new, base, m)
    # level inside the hole differs from the frame average; compare to truth
    lvl = base[m].mean()
    # per-channel noise of the model on the mean level, as luma-ish std
    true_s = np.sqrt(A * base[m] + B).mean()
    assert abs(s_new - true_s) / true_s < 0.25
    assert abs(mean_new) < 0.05 * true_s
    assert s_frame > 0


def test_noiseless_unchanged():
    h, w = 128, 128
    base = ramp(h, w)
    hole = disc(h, w, 20)
    new = grain.regrain(base, base.copy(), hole, LUMA, 1)
    np.testing.assert_array_equal(new, base)


def test_outside_bit_identical_and_shapes():
    h, w = 256, 384
    base, frame = noisy(h, w)
    hole = disc(h, w, 25)
    out = base.copy()
    out0 = out.copy()
    new = grain.regrain(frame, out, hole, LUMA, 3)
    assert new.shape == (h, w, 3) and new.dtype == np.float32
    np.testing.assert_array_equal(out, out0)  # input untouched
    assert np.array_equal(new[hole == 0], out0[hole == 0])
    assert not np.array_equal(new[hole > 0], out0[hole > 0])


def test_determinism():
    h, w = 256, 384
    base, frame = noisy(h, w)
    hole = disc(h, w, 25)
    a = grain.regrain(frame, base, hole, LUMA, 7)
    b = grain.regrain(frame, base, hole, LUMA, 7)
    c = grain.regrain(frame, base, hole, LUMA, 8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_too_small_or_no_hole():
    base = ramp(10, 10)
    assert np.array_equal(grain.regrain(base + 0.01, base, np.ones((10, 10)), LUMA, 1), base)
    base, frame = noisy(64, 64)
    assert np.array_equal(grain.regrain(frame, base, np.zeros((64, 64)), LUMA, 1), base)


def test_clipped_highlights():
    h, w = 256, 384
    base, frame = noisy(h, w)
    frame[:100, :150] = 3.0  # blown-out region, perfectly smooth
    frame[:100, :150] += 0  # (values > peak of the rest)
    hole = disc(h, w, 30)
    hole = np.roll(hole, 60, axis=1)
    new = grain.regrain(frame, base.copy(), hole, LUMA, 2)
    m = hole > 0
    assert np.isfinite(new).all()
    s = (new - base)[m].std()
    true_s = np.sqrt(A * base[m] + B).mean()
    assert 0.5 * true_s < s < 1.5 * true_s


def test_recovers_model_around_hole():
    h, w = 256, 384
    _, frame = noisy(h, w, seed=3)
    known = disc(h, w, 60) == 0
    pn = grain.estimate_picture_noise(frame, known, LUMA)
    # luma axis of equal-channel white noise: variance 3*(a*level+b)
    # at the mean level; compare at level 0.3
    est = pn.variance(0, 0.3)
    true = A * 0.3 + B  # orthonormal axes keep the channel variance
    assert abs(est - true) / true < 0.3


def test_grain_width_gives_ratio_back():
    for ratio in [0.95, 0.8, 0.6, 0.45, 0.35]:
        wd = grain._width_for(ratio)
        assert abs(grain._shape_ratio(wd) - ratio) < 1e-3
    assert grain._width_for(1.0) == 0.0
    assert grain._width_for(1.3) == 0.0
    assert grain._width_for(float("nan")) == 0.0
    k = grain.grain_kernel(1.2)
    assert abs((k ** 2).sum() - 1) < 1e-5
    assert grain._residual_gain(grain.grain_kernel(0.0), 2) == grain.RESIDUAL_GAIN


def test_correlated_grain_drawn_coarse():
    h, w = 256, 384
    rng = np.random.default_rng(4)
    base = ramp(h, w)
    n = rng.standard_normal(base.shape).astype(np.float32)
    from scipy import ndimage
    n = ndimage.gaussian_filter(n, (1.5, 1.5, 0)) * 6
    frame = base + 0.02 * n
    hole = disc(h, w, 50)
    new = grain.regrain(frame, base.copy(), hole, LUMA, 9)
    m = hole > 0
    d = (new - base)[..., 1]
    # added grain has positive neighbour correlation (coarse), like the frame
    c = np.corrcoef(d[:, :-1][m[:, :-1] & m[:, 1:]], d[:, 1:][m[:, :-1] & m[:, 1:]])[0, 1]
    assert c > 0.3


def test_speed():
    import time
    h = w = 1024
    base, frame = noisy(h, w)
    hole = disc(h, w, 150)
    t = time.time()
    grain.regrain(frame, base, hole, LUMA, 1)
    assert time.time() - t < 6
