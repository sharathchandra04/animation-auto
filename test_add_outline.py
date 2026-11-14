"""Tests for add_outline.py — run: python test_add_outline.py"""

import numpy as np

import cv2

from add_outline import add_outline, flatten, outline_pipeline
from remove_background import remove_background
from test_remove_background import make_insect


def _cut_insect():
    img = make_insect()
    return remove_background(img, key_color="magenta")


def test_outline_ring_is_black_and_outside():
    cut = _cut_insect()
    out = add_outline(cut, outline_color="black", thickness=6)
    assert out.shape == cut.shape and out.shape[2] == 4
    # Body centre untouched: same colour, still opaque.
    assert np.array_equal(out[200, 200], cut[200, 200])
    assert out[200, 200, 3] == 255
    # Just above the head (the head top is ~y=67 at x=200): the ring
    # must be black and opaque there, and the subject must not be.
    ring_y = 67 - 3  # 3 px above the subject, inside the 6 px ring
    assert out[ring_y, 200, 3] == 255, "ring must be opaque"
    assert tuple(out[ring_y, 200, :3]) == (0, 0, 0), "ring must be black"
    # Far from the insect: still transparent.
    assert out[0, 0, 3] == 0
    print("ring black and outside: ok")


def test_thickness():
    cut = _cut_insect()
    # Topmost opaque pixel of the subject vs of the outlined image.
    subj_top = np.argmax((cut[:, :, 3] > 127).any(axis=1))
    for thickness in (3, 6, 10):
        out = add_outline(cut, thickness=thickness)
        out_top = np.argmax((out[:, :, 3] > 127).any(axis=1))
        assert subj_top - out_top == thickness, (
            f"thickness {thickness}: outline reaches "
            f"{subj_top - out_top}px above the subject")
    print("thickness: ok (3, 6, 10 px reach exactly that far)")


def test_outline_colour():
    cut = _cut_insect()
    out = add_outline(cut, outline_color="#FF0000", thickness=6)
    ring_y = 67 - 3
    # #FF0000 is red: BGR (0, 0, 255).
    assert tuple(out[ring_y, 200, :3]) == (0, 0, 255), "ring must be red"
    print("outline colour: ok")


def test_pipeline_from_magenta_image():
    cut, outlined, cut_path, out_path = outline_pipeline(
        "test_insect_magenta.png", key_color="magenta",
        outline_color="black", thickness=6)
    assert cut.shape[2] == 4 and outlined.shape[2] == 4
    assert (outlined[:, :, 3] > 127).sum() > (cut[:, :, 3] > 127).sum(), \
        "outlined image must cover more than the cut-out"
    assert cut_path.endswith("_cut.png") and out_path.endswith("_outlined.png")
    print(f"pipeline: ok ({cut_path}, {out_path})")


def test_pipeline_from_cut_png():
    # Already-cut input skips removal and still gets the outline.
    _, outlined, _, _ = outline_pipeline(
        "test_insect_cut.png", outline_color="black", thickness=6)
    assert outlined[200, 200, 3] == 255
    assert outlined[67 - 3, 200, 3] == 255
    print("pipeline from cut PNG: ok")


def test_flatten():
    cut = _cut_insect()
    out = add_outline(cut, thickness=6)
    flat = flatten(out, background="white")
    assert flat.shape[2] == 3, "flattened image has no alpha"
    assert tuple(flat[0, 0]) == (255, 255, 255), "background is white"
    assert tuple(flat[200, 200]) == tuple(cut[200, 200, :3])
    print("flatten: ok")


if __name__ == "__main__":
    test_outline_ring_is_black_and_outside()
    test_thickness()
    test_outline_colour()
    test_pipeline_from_magenta_image()
    test_pipeline_from_cut_png()
    test_flatten()
    print("ALL ADD-OUTLINE TESTS PASSED")
