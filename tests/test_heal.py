import dataclasses

import numpy as np

import config
import heal
import heal_engine
import imgio
import lama
from test_lama import Dummy


def _conf(**kw):
    c = config.Config(heal_seam_match=False, heal_even=False,
                      heal_regrain=False)
    return dataclasses.replace(c, **kw)


def _img(h=120, w=160, linear=True):
    rng = np.random.default_rng(5)
    d = (rng.random((h, w, 3)) * 8).astype(np.float32)   # values above 1
    d[0, 0, 0] = np.nan
    return imgio.Img(data=d, kind='float' if linear else 'u8', linear=linear)


def test_heal_outside_bit_exact_and_hdr():
    img = _img()
    cover = np.zeros(img.data.shape[:2], np.float32)
    cover[40:60, 50:90] = 1.0
    cover[40:42, 50:90] = 0.5
    orig = img.data.copy()
    patches = heal.heal(img, cover, lama.Filler(Dummy()), _conf())
    assert len(patches) == 1
    assert np.array_equal(img.data, orig, equal_nan=True)   # inputs untouched
    p = patches[0]
    out = orig.copy()
    w = p.window
    out[w.y:w.y + w.height, w.x:w.x + w.width] = p.data
    assert np.array_equal(out[cover == 0], orig[cover == 0], equal_nan=True)
    assert np.isfinite(out[cover > 0]).all()
    # fully covered pixels are the fill, which is bright (HDR) here: > 1
    assert out[cover == 1].mean() > 1.0


def test_heal_two_far_components_two_windows():
    img = _img(300, 600)
    cover = np.zeros(img.data.shape[:2], np.float32)
    cover[100:110, 20:30] = 1
    cover[100:110, 560:570] = 1
    patches = heal.heal(img, cover, lama.Filler(Dummy()), _conf())
    assert len(patches) == 2


def test_heal_integer_image():
    img = _img(linear=False)
    img.data = np.clip(img.data / 8, 0, 1)
    img.data[0, 0, 0] = 0
    cover = np.zeros(img.data.shape[:2], np.float32)
    cover[40:60, 50:90] = 1.0
    patches = heal.heal(img, cover, lama.Filler(Dummy()), _conf())
    assert np.isfinite(patches[0].data).all()


def test_brush_profile():
    d = np.array([0.0, 6.9, 7.0, 8.5, 10.0, 11.0])
    p = heal_engine.brush_profile(d, 10.0, 0.3)
    assert p[0] == 1 and p[2] == 1 and p[-2] == 0 and p[-1] == 0
    assert 0 < p[3] < 1


def _engine(tmp_path):
    import tifffile
    path = str(tmp_path / 'in.tif')
    tifffile.imwrite(path, np.random.rand(100, 120, 3).astype(np.float32) * 4)
    e = heal_engine.AIHealingEngine(
        dataclasses.replace(_conf(), exiftool='', brush_radius=10.0))
    e.display_xform = None
    e.load_image(path)
    return e


def test_strokes_opacity_undo(tmp_path):
    e = _engine(tmp_path)
    e.opacity = 0.5
    e.begin_stroke()
    e.stroke_to(30, 30)
    e.stroke_to(60, 30)
    e.end_stroke()
    assert abs(e.cover[30, 45] - 0.5) < 1e-6      # max within a stroke
    e.begin_stroke()
    e.stroke_to(45, 30)
    e.end_stroke()
    assert abs(e.cover[30, 45] - 0.75) < 1e-6     # opacities compose
    e.begin_stroke(erase=True)
    e.stroke_to(45, 30)
    e.end_stroke()
    assert e.cover[30, 45] == 0
    for _ in range(3):
        e.undo_last()
    assert not e.has_cover()
    assert not e.history


def test_engine_heal_and_undo(tmp_path):
    e = _engine(tmp_path)
    e.filler = lama.Filler(Dummy())
    orig = e.img.data.copy()
    disp = e.base_display.copy()
    e.opacity = 1.0
    e.begin_stroke()
    e.stroke_to(60, 50)
    e.end_stroke()
    res = {}
    import threading
    ev = threading.Event()
    assert e.heal_async(lambda p, c: (res.update(p=p, c=c), ev.set()),
                        lambda ex: (res.update(err=ex), ev.set()))
    ev.wait(30)
    assert 'err' not in res, res.get('err')
    e.apply_patches(res['p'], res['c'])
    assert not e.has_cover() and not e.saved
    assert not np.array_equal(e.img.data, orig)
    e.undo_last()
    assert np.array_equal(e.img.data, orig)
    assert np.array_equal(e.base_display, disp)
    assert e.has_cover()    # the stroke is back
    out = str(tmp_path / 'out.tif')
    e.save_image(out)
    assert np.array_equal(imgio.load(out).data, orig)


def test_heal_all_stages_enabled():
    rng = np.random.default_rng(9)
    h, w = 400, 500
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.2 + 0.5 * (xx / w)[..., None] * np.ones(3)
    d = (base * (1 + 0.05 * rng.standard_normal((h, w, 3)))).astype(np.float32)
    img = imgio.Img(data=d, kind='float', linear=True)
    cover = np.zeros((h, w), np.float32)
    cover[100:300, 120:380] = 1.0           # deep: scale < 1, regrain runs
    c = config.Config()                      # all stages on
    patches = heal.heal(img, cover, lama.Filler(Dummy()), c)
    p = patches[0]
    assert p.scale < 1
    out = d.copy()
    wn = p.window
    out[wn.y:wn.y + wn.height, wn.x:wn.x + wn.width] = p.data
    assert np.array_equal(out[cover == 0], d[cover == 0])
    assert np.isfinite(out).all()
    # grain was put back: the hole is not smooth
    assert out[150:250, 200:300, 1].std() > 0.005
