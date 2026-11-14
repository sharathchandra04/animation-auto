"""Tests for tilt.py — run: python test_tilt.py"""

import numpy as np

from test_remove_background import make_insect
from tilt import parse_step, tilt_corners, tilt_image, tilt_sequence


def test_zero_is_identity():
    img = make_insect()
    out = tilt_image(img, "left", 0)
    assert out.shape[:2] == img.shape[:2]
    assert np.array_equal(out[:, :, :3], img)
    print("zero is identity: ok")


def test_hinge_stays_put():
    # Left hinge: both hinge corners sit on x=0, and the hinge keeps
    # its full length h. Top hinge: both hinge corners share one y,
    # and the hinge keeps its full length w.
    w, h = 400, 300
    dst, _, _ = tilt_corners(w, h, "left", 30, focal=800)
    assert abs(dst[0][0]) < 1.0 and abs(dst[3][0]) < 1.0, \
        "left hinge corners must sit on x=0"
    assert abs((dst[3][1] - dst[0][1]) - h) < 1.0, \
        "left hinge must keep its full height"
    dst, _, _ = tilt_corners(w, h, "top", 30, focal=800)
    assert abs(dst[0][1] - dst[1][1]) < 1.0, \
        "top hinge corners must share one y"
    assert abs((dst[1][0] - dst[0][0]) - w) < 1.0, \
        "top hinge must keep its full width"
    print("hinge stays put: ok")


def test_free_edge_grows():
    # Left tilt, positive angle: the free (right) edge is magnified,
    # so the canvas gets taller than the original.
    w, h = 400, 300
    _, _, ch = tilt_corners(w, h, "left", 30, focal=800)
    assert ch > h, f"canvas height {ch} must exceed {h}"
    _, _, ch_neg = tilt_corners(w, h, "left", -30, focal=800)
    assert ch_neg == h, "pushed-away tilt keeps the hinge height"
    _, cw, _ = tilt_corners(w, h, "left", -30, focal=800)
    assert cw < w, "pushed-away tilt narrows the image"
    print("free edge grows: ok")


def test_all_four_axes():
    img = make_insect()
    shapes = {}
    for axis in ("left", "right", "top", "bottom"):
        out = tilt_image(img, axis, 30)
        assert out.shape[2] == 4
        assert (out[:, :, 3] == 0).any(), "corners must be transparent"
        shapes[axis] = out.shape[:2]
    # Left and right give the same canvas size (mirrored), as do top
    # and bottom; left differs from top.
    assert shapes["left"] == shapes["right"]
    assert shapes["top"] == shapes["bottom"]
    assert shapes["left"] != shapes["top"]
    print(f"all four axes: ok {shapes}")


def test_transparency_preserved():
    from remove_background import remove_background

    cut = remove_background(make_insect(), key_color="magenta")
    out = tilt_image(cut, "left", 30)
    assert out.shape[2] == 4
    before = int((cut[:, :, 3] > 127).sum())
    after = int((out[:, :, 3] > 127).sum())
    # Magnification grows the subject a little; it must not shrink.
    assert after > before, "tilted subject must not lose pixels"
    assert out[0, 0, 3] == 0, "corners transparent"
    print(f"transparency preserved: ok ({before} -> {after} px)")


def test_angle_limits():
    for bad in (90, -90, 120):
        try:
            tilt_image(make_insect(), "left", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"angle {bad} must be rejected")
    print("angle limits: ok")


def test_sequence_single_step_matches_single_tilt():
    img = make_insect()
    one = tilt_sequence(img, [("left", 45)])
    single = tilt_image(img, "left", 45)
    assert np.array_equal(one, single), \
        "a one-step sequence must equal the single tilt"
    none = tilt_sequence(img, [])
    assert np.array_equal(none[:, :, :3], img), \
        "an empty sequence must be the identity"
    print("sequence single step matches single tilt: ok")


def test_sequence_order_matters():
    # 3D rotations do not commute: left then top must differ from
    # top then left.
    img = make_insect()
    a = tilt_sequence(img, [("left", 45), ("top", 45)])
    b = tilt_sequence(img, [("top", 45), ("left", 45)])
    assert not np.array_equal(a, b), "order must change the result"
    assert a.shape[2] == 4 and b.shape[2] == 4
    print(f"sequence order matters: ok "
          f"(left->top {a.shape[1]}x{a.shape[0]}, "
          f"top->left {b.shape[1]}x{b.shape[0]})")


def test_sequence_left_then_top():
    # The requested chain: left 45, then top 45 on the result.
    # The canvas must grow in both directions versus the original,
    # and the subject must survive both warps.
    img = make_insect()
    out = tilt_sequence(img, [("left", 45), ("top", 45)], focal=800)
    assert out.shape[1] > img.shape[1] and out.shape[0] > img.shape[0], \
        f"chained canvas {out.shape[1]}x{out.shape[0]} must exceed 400x400"
    subject = int((out[:, :, 3] > 127).sum())
    assert subject > 10000, f"subject must survive, got {subject} px"
    print(f"sequence left then top: ok "
          f"({out.shape[1]}x{out.shape[0]}, subject {subject} px)")


def test_parse_step():
    assert parse_step("left:45") == ("left", 45.0)
    assert parse_step("top:-30") == ("top", -30.0)
    for bad in ("left", "side:45", "left:abc", "left:45:60"):
        try:
            parse_step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must be rejected")
    print("parse step: ok")


if __name__ == "__main__":
    test_zero_is_identity()
    test_hinge_stays_put()
    test_free_edge_grows()
    test_all_four_axes()
    test_transparency_preserved()
    test_angle_limits()
    test_sequence_single_step_matches_single_tilt()
    test_sequence_order_matters()
    test_sequence_left_then_top()
    test_parse_step()
    print("ALL TILT TESTS PASSED")
