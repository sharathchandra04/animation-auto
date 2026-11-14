"""Tests for scissor.py — run: python test_scissor.py"""

import re

import numpy as np

from scissor import (asset_name, cut_image, draw_grid, line_from_equation,
                     line_from_points, scissor_pipeline)
from test_remove_background import make_insect


def _gradient(w=400, h=300):
    """An image whose colour encodes position, so a cut is checkable."""
    xs = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.uint8)[:, None], (1, w))
    return np.dstack([xs, ys, np.full((h, w), 128, np.uint8)])


def test_vertical_cut():
    img = _gradient()
    a, b, length = cut_image(img, (200, 0), (200, 300))
    assert length == 300.0
    # Piece A keeps the left half including the line itself (x <= 200),
    # piece B keeps the right half (x > 200).
    assert (a[:, :201, 3] == 255).all() and (a[:, 201:, 3] == 0).all()
    assert (b[:, :201, 3] == 0).all() and (b[:, 201:, 3] == 255).all()
    # Colours inside each piece are the original colours.
    assert np.array_equal(a[10, 10, :3], img[10, 10])
    assert np.array_equal(b[10, 390, :3], img[10, 390])
    print("vertical cut: ok")


def test_horizontal_and_diagonal_rebuild():
    img = _gradient()
    for p1, p2 in (((0, 150), (400, 150)), ((0, 0), (400, 300)),
                   ((50, 250), (350, 20))):
        a, b, _ = cut_image(img, p1, p2)
        # The two pieces rebuild the original exactly.
        assert ((a[:, :, 3] > 0) | (b[:, :, 3] > 0)).all(), "no pixel lost"
        assert not ((a[:, :, 3] > 0) & (b[:, :, 3] > 0)).any(), \
            "no pixel in both pieces"
        merged = np.where(a[:, :, 3:4] > 0, a, b)
        assert np.array_equal(merged, np.dstack(
            [img, np.full(img.shape[:2], 255, np.uint8)])), \
            "pieces must rebuild the original"
    print("horizontal and diagonal rebuild: ok")


def test_equation_matches_points():
    img = _gradient()
    a1, b1, _ = cut_image(img, (0, 10), (400, 210))  # y = 0.5x + 10
    p1, p2 = line_from_equation("y = 0.5x + 10")
    a2, b2, _ = cut_image(img, p1, p2)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    p1, p2 = line_from_equation("x = 200")
    a3, _, _ = cut_image(img, p1, p2)
    a4, _, _ = cut_image(img, (200, 0), (200, 300))
    assert np.array_equal(a3, a4)
    print("equation matches points: ok")


def test_same_point_rejected():
    try:
        line_from_points((10, 10), (10, 10))
    except ValueError:
        print("same point rejected: ok")
    else:
        raise AssertionError("identical points must raise ValueError")


def test_asset_naming():
    name_a = asset_name("insect", "A")
    name_b = asset_name("insect", "B")
    pat = re.compile(r"^insect_[AB]_[0-9a-f]{6}_\d{8}_\d{6}\.png$")
    assert pat.match(name_a), f"bad name {name_a}"
    assert pat.match(name_b), f"bad name {name_b}"
    assert name_a != name_b, "the two pieces must differ (A vs B)"
    # Two cuts at different times get different names.
    assert asset_name("insect", "A") != asset_name("insect", "A")
    print(f"asset naming: ok (e.g. {name_a})")


def test_pipeline_saves_two_assets():
    import os

    import cv2

    img = make_insect()
    cv2.imwrite("test_scissor_insect.png", img)
    path_a, path_b, length = scissor_pipeline(
        "test_scissor_insect.png", p1=(100, 50), p2=(300, 350),
        out_dir="assets/cuts")
    assert os.path.isfile(path_a) and os.path.isfile(path_b)
    assert str(path_a).startswith("assets/cuts/")
    # The name is <base>_<side>_<rand>_<date>_<time>.png; the base
    # itself can contain underscores, so parse from the right.
    parts_a, parts_b = path_a.name.split("_"), path_b.name.split("_")
    assert parts_a[-3] == parts_b[-3], "both pieces share the random ID"
    assert parts_a[-2:] == parts_b[-2:], "both pieces share the timestamp"
    a = cv2.imread(str(path_a), cv2.IMREAD_UNCHANGED)
    b = cv2.imread(str(path_b), cv2.IMREAD_UNCHANGED)
    assert a.shape[2] == 4 and b.shape[2] == 4
    assert ((a[:, :, 3] > 0) | (b[:, :, 3] > 0)).all()
    print(f"pipeline: ok ({path_a.name}, {path_b.name}, "
          f"line {length:.1f}px)")


def test_grid_drawing():
    img = _gradient()
    grid = draw_grid(img, step=50)
    # A grid line pixel differs from the original image.
    assert not np.array_equal(grid[0, 50], img[0, 50]), "vertical line drawn"
    assert not np.array_equal(grid[50, 0], img[50, 0]), "horizontal line drawn"
    # Away from lines, the image is unchanged.
    assert np.array_equal(grid[25, 25], img[25, 25])
    print("grid drawing: ok")


if __name__ == "__main__":
    test_vertical_cut()
    test_horizontal_and_diagonal_rebuild()
    test_equation_matches_points()
    test_same_point_rejected()
    test_asset_naming()
    test_pipeline_saves_two_assets()
    test_grid_drawing()
    print("ALL SCISSOR TESTS PASSED")
