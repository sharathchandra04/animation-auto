"""Tests for rotate.py — run: python test_rotate.py"""

import numpy as np

from rotate import rotate_image
from test_remove_background import make_insect


def _arrow(w=200, h=100):
    """A white right-pointing arrow on black, for direction checks."""
    import cv2

    img = np.zeros((h, w, 3), np.uint8)
    cv2.arrowedLine(img, (20, 50), (180, 50), (255, 255, 255), 8,
                    tipLength=0.25)
    return img


def test_zero_and_full_turn():
    img = make_insect()
    assert np.array_equal(rotate_image(img, 0), rotate_image(img, 0))
    # 0 degrees changes nothing (exact, nearest interpolation).
    r0 = rotate_image(img, 0, interpolation="nearest")
    assert r0.shape == (400, 400, 4)
    assert np.array_equal(r0[:, :, :3], img)
    # 360 degrees with nearest interpolation is the identity.
    r360 = rotate_image(img, 360, interpolation="nearest")
    assert np.array_equal(r360[:, :, :3], img), "360 must be the identity"
    print("zero and full turn: ok")


def test_ninety_swaps_dimensions():
    img = _arrow()  # 200 wide, 100 tall
    r = rotate_image(img, 90, interpolation="nearest")
    assert r.shape[1] == 100 and r.shape[0] == 200, \
        f"90 deg must swap dimensions, got {r.shape[1]}x{r.shape[0]}"
    # Four quarter turns (nearest) bring it back exactly.
    back = img
    for _ in range(4):
        back = rotate_image(back, 90, interpolation="nearest")
    assert np.array_equal(back[:, :, :3], img), "4 x 90 must be identity"
    print("ninety swaps dimensions: ok")


def test_direction_counterclockwise():
    # A marker dot at the top centre of a square image: +90 degrees
    # (counter-clockwise) must move it to the centre-left. (100x100
    # so the centre is exactly (50, 50) and the math lands on a pixel.)
    img = np.zeros((100, 100, 3), np.uint8)
    img[10, 50] = (255, 255, 255)
    r = rotate_image(img, 90, expand=False, interpolation="nearest")
    assert tuple(r[50, 10][:3]) == (255, 255, 255), \
        "+90 must move the top dot to the left side"
    r2 = rotate_image(img, -90, expand=False, interpolation="nearest")
    assert tuple(r2[50, 90][:3]) == (255, 255, 255), \
        "-90 must move the top dot to the right side"
    print("direction counter-clockwise: ok")


def test_expand_vs_keep_size():
    img = make_insect()  # 400x400
    big = rotate_image(img, 45)  # expand (default)
    assert big.shape[0] > 400 and big.shape[1] > 400, \
        "expanded canvas must grow for 45 degrees"
    small = rotate_image(img, 45, expand=False)
    assert small.shape[:2] == (400, 400), "keep-size must keep 400x400"
    print(f"expand vs keep-size: ok (45 deg -> "
          f"{big.shape[1]}x{big.shape[0]} vs 400x400)")


def test_transparency():
    from remove_background import remove_background

    cut = remove_background(make_insect(), key_color="magenta")
    assert cut.shape[2] == 4
    out = rotate_image(cut, 45)
    assert out.shape[2] == 4, "output keeps the alpha channel"
    # Corners of the expanded canvas are transparent.
    assert out[0, 0, 3] == 0 and out[0, -1, 3] == 0
    # The subject survives: its opaque pixel count is roughly preserved
    # (linear interpolation softens edges, so allow 5%). Compare
    # absolute counts, not fractions — the canvas grows when expanding.
    before = int((cut[:, :, 3] > 127).sum())
    after = int((out[:, :, 3] > 127).sum())
    assert abs(before - after) < 0.05 * before, \
        f"subject pixels changed too much: {before} -> {after}"
    print(f"transparency: ok (subject {before} px -> {after} px)")


def test_background_fill():
    img = make_insect()
    out = rotate_image(img, 45, background="white")
    assert out.shape[2] == 3, "flattened output has no alpha"
    assert tuple(out[0, 0]) == (255, 255, 255), "corner is white"
    print("background fill: ok")


if __name__ == "__main__":
    test_zero_and_full_turn()
    test_ninety_swaps_dimensions()
    test_direction_counterclockwise()
    test_expand_vs_keep_size()
    test_transparency()
    test_background_fill()
    print("ALL ROTATE TESTS PASSED")
