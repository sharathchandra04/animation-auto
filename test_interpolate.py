"""Unit tests for interpolate.py — run: python test_interpolate.py"""

import numpy as np

import lip_outline as lo
from interpolate import (interpolate_at, interpolate_points,
                         interpolate_shapes)

LIB = lo.load_mouth_library("mouth_library_v2/manifest.json")
A = "CLOSED"
B = "AH_I_2"


def test_endpoints_exact():
    pts0, _ = interpolate_at(LIB, A, B, 0.0)
    pts1, _ = interpolate_at(LIB, A, B, 1.0)
    assert np.allclose(pts0, LIB[A][0]), "t=0 must be shape A exactly"
    assert np.allclose(pts1, LIB[B][0]), "t=1 must be shape B exactly"
    print("endpoints exact: ok")


def test_midpoint_linear():
    # With linear easing, t=0.5 must be the exact average of A and B.
    mid, _ = interpolate_at(LIB, A, B, 0.5, ease="linear")
    expect = 0.5 * (LIB[A][0] + LIB[B][0])
    assert np.allclose(mid, expect), "linear midpoint must be the average"
    # The midpoint's height must sit strictly between the two ends.
    h_a = LIB[A][0][:, 1].max() - LIB[A][0][:, 1].min()
    h_b = LIB[B][0][:, 1].max() - LIB[B][0][:, 1].min()
    h_m = mid[:, 1].max() - mid[:, 1].min()
    assert h_a < h_m < h_b, f"mid height {h_m:.3f} not between {h_a:.3f} and {h_b:.3f}"
    print(f"midpoint linear: ok (height {h_a:.3f} -> {h_m:.3f} -> {h_b:.3f})")


def test_smoothstep_eases():
    # Smoothstep moves slower near the ends: at t=0.25 it must be
    # closer to A than the linear point is.
    mid_lin, _ = interpolate_at(LIB, A, B, 0.25, ease="linear")
    mid_sm, _ = interpolate_at(LIB, A, B, 0.25, ease="smoothstep")
    d_lin = np.linalg.norm(mid_lin - LIB[A][0])
    d_sm = np.linalg.norm(mid_sm - LIB[A][0])
    assert d_sm < d_lin, "smoothstep must lag linear near the start"
    print("smoothstep eases: ok")


def test_step_count():
    for steps in (1, 2, 4, 6):
        frames = interpolate_shapes(LIB, A, B, steps)
        assert len(frames) == steps + 1, f"steps={steps} must give {steps+1} shapes"
        assert np.allclose(frames[0][0], LIB[A][0])
        assert np.allclose(frames[-1][0], LIB[B][0])
    print("step count: ok (1->2, 2->3, 4->5, 6->7 shapes)")


def test_raw_points():
    # Two raw point lists of different lengths still interpolate.
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    tri = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.9]], dtype=float)
    mid = interpolate_points(sq, tri, 0.5, ease="linear")
    assert mid.shape == (64, 2), f"expected (64, 2), got {mid.shape}"
    assert np.allclose(interpolate_points(sq, tri, 0.0), interpolate_points(sq, tri, 0.0))
    print("raw points: ok")


def test_unknown_shape():
    try:
        interpolate_at(LIB, "NOPE", B, 0.5)
    except ValueError as e:
        print(f"unknown shape: ok ({e})")
    else:
        raise AssertionError("unknown shape name must raise ValueError")


def test_all_pairs():
    # Every pair of the 12 library shapes interpolates without error.
    names = sorted(LIB)
    for a in names:
        for b in names:
            frames = interpolate_shapes(LIB, a, b, 2)
            assert len(frames) == 3
            assert all(f[0].shape == (64, 2) for f in frames)
    print(f"all pairs: ok ({len(names)} shapes, "
          f"{len(names) * len(names)} pairs)")


if __name__ == "__main__":
    test_endpoints_exact()
    test_midpoint_linear()
    test_smoothstep_eases()
    test_step_count()
    test_raw_points()
    test_unknown_shape()
    test_all_pairs()
    print("ALL INTERPOLATE TESTS PASSED")
