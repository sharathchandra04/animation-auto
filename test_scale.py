"""Tests for scale.py — run: python test_scale.py"""

import numpy as np

from remove_background import remove_background
from scale import scale_image
from test_remove_background import make_insect


def test_factor_doubles_and_halves():
    img = make_insect()
    big = scale_image(img, factor=2.0)
    assert big.shape[1] == 800 and big.shape[0] == 800, "2x must double"
    small = scale_image(img, factor=0.5)
    assert small.shape[1] == 200 and small.shape[0] == 200, "0.5x must halve"
    print("factor doubles and halves: ok")


def test_factor_one_is_identity():
    img = make_insect()
    same = scale_image(img, factor=1.0)
    assert np.array_equal(same, img), "factor 1 must change nothing"
    print("factor one is identity: ok")


def test_round_trip_with_nearest():
    # 2x then 0.5x with nearest interpolation must give back the
    # original image exactly (whole-number zooms are lossless).
    img = make_insect()
    back = scale_image(scale_image(img, factor=2.0, interpolation="nearest"),
                       factor=0.5, interpolation="nearest")
    assert np.array_equal(back, img), "2x then 0.5x must be exact"
    print("round trip with nearest: ok")


def test_exact_size():
    img = make_insect()  # 400x400
    w800 = scale_image(img, width=800)
    assert w800.shape[1] == 800 and w800.shape[0] == 800, \
        "width alone keeps the aspect ratio"
    wide = scale_image(img, width=800, height=200)
    assert wide.shape[1] == 800 and wide.shape[0] == 200, \
        "width+height together set the exact size"
    tall = scale_image(img, height=100)
    assert tall.shape[0] == 100 and tall.shape[1] == 100
    print("exact size: ok")


def test_non_uniform_fx_fy():
    img = make_insect()
    out = scale_image(img, fx=2.0, fy=0.5)
    assert out.shape[1] == 800 and out.shape[0] == 200, \
        "fx=2, fy=0.5 must stretch x and squash y"
    print("non-uniform fx/fy: ok")


def test_transparency_preserved():
    cut = remove_background(make_insect(), key_color="magenta")
    assert cut.shape[2] == 4
    big = scale_image(cut, factor=2.0)
    assert big.shape[2] == 4, "alpha channel must survive scaling"
    assert big[0, 0, 3] == 0, "transparent corners stay transparent"
    assert big[400, 400, 3] == 255, "subject stays opaque"
    small = scale_image(cut, factor=0.5)
    assert small[0, 0, 3] == 0 and small[100, 100, 3] == 255
    print("transparency preserved: ok")


def test_bad_inputs_rejected():
    img = make_insect()
    for kwargs in ({"factor": 0}, {"factor": -1}, {"fx": 0}, {}):
        try:
            scale_image(img, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} must be rejected")
    print("bad inputs rejected: ok")


if __name__ == "__main__":
    test_factor_doubles_and_halves()
    test_factor_one_is_identity()
    test_round_trip_with_nearest()
    test_exact_size()
    test_non_uniform_fx_fy()
    test_transparency_preserved()
    test_bad_inputs_rejected()
    print("ALL SCALE TESTS PASSED")
