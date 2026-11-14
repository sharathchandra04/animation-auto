"""Tests for remove_background.py — run: python test_remove_background.py"""

import numpy as np

import cv2

from remove_background import parse_color, remove_background


def make_insect(size=(400, 400), bg=(255, 0, 255), noise=8, seed=7):
    """Synthetic insect on a solid background (BGR), like an AI image.

    Magenta background with a little noise, a dark-green beetle-ish
    body, head, six legs and two antennae — all colours far from
    magenta, the way a real subject would be.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
    if noise:
        n = rng.integers(-noise, noise + 1, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    cx, cy = size[0] // 2, size[1] // 2
    # Body and head (dark green), wings (pale yellow).
    cv2.ellipse(img, (cx, cy + 20), (70, 100), 0, 0, 360, (30, 90, 30), -1)
    cv2.ellipse(img, (cx - 35, cy - 10), (45, 70), -25, 0, 360, (200, 220, 120), -1)
    cv2.ellipse(img, (cx + 35, cy - 10), (45, 70), 25, 0, 360, (200, 220, 120), -1)
    cv2.circle(img, (cx, cy - 95), 38, (20, 70, 20), -1)
    # Six legs and two antennae (dark lines, 6 px wide).
    for dy in (-40, 0, 40):
        for s in (-1, 1):
            cv2.line(img, (cx + s * 55, cy + dy),
                     (cx + s * 150, cy + dy + 55), (15, 50, 15), 6)
    for s in (-1, 1):
        cv2.line(img, (cx + s * 20, cy - 125),
                 (cx + s * 70, cy - 185), (15, 50, 15), 5)
    return img


def test_magenta_removed():
    img = make_insect()
    cut = remove_background(img, key_color="magenta")
    assert cut.shape[2] == 4, "output must have an alpha channel"
    alpha = cut[:, :, 3]
    # Corners are background: fully transparent.
    assert alpha[0, 0] == 0 and alpha[0, -1] == 0 and alpha[-1, 0] == 0
    # Body centre is subject: fully opaque.
    assert alpha[200, 200] == 255
    # Most of the image is background, a minority is insect.
    frac = float((alpha > 127).mean())
    assert 0.05 < frac < 0.5, f"subject fraction {frac:.2f} looks wrong"
    print(f"magenta removed: ok (subject {100 * frac:.1f}% of image)")


def test_other_colours_and_hex():
    img = make_insect(bg=(0, 255, 0))  # green background
    cut = remove_background(img, key_color="green")
    assert cut[0, 0, 3] == 0 and cut[200, 200, 3] == 255
    cut2 = remove_background(img, key_color="#00FF00")
    assert np.array_equal(cut, cut2), "hex #00FF00 must equal 'green'"
    assert parse_color("#FF00FF") == (255, 0, 255)
    print("other colours and hex: ok")


def test_tolerance():
    # A background that drifts from pure magenta needs tolerance;
    # with tolerance 0 only exact magenta goes transparent.
    img = make_insect(noise=8)
    strict = remove_background(img, key_color="magenta", tolerance=0.0,
                               feather=0.0, keep_largest=False)
    loose = remove_background(img, key_color="magenta", tolerance=40.0,
                              feather=0.0, keep_largest=False)
    bg_strict = float((strict[:, :, 3] == 0).mean())
    bg_loose = float((loose[:, :, 3] == 0).mean())
    assert bg_loose > bg_strict, "higher tolerance must remove more"
    print(f"tolerance: ok (transparent {100 * bg_strict:.1f}% -> "
          f"{100 * bg_loose:.1f}%)")


def test_keep_largest_removes_specks():
    # A stray subject-coloured dot far from the insect must go.
    img = make_insect()
    cv2.circle(img, (30, 30), 8, (30, 90, 30), -1)  # body-coloured speck
    cut = remove_background(img, key_color="magenta")
    assert cut[30, 30, 3] == 0, "isolated speck must be removed"
    assert cut[200, 200, 3] == 255, "the insect itself must stay"
    print("keep largest: ok")


if __name__ == "__main__":
    test_magenta_removed()
    test_other_colours_and_hex()
    test_tolerance()
    test_keep_largest_removes_specks()
    print("ALL REMOVE-BACKGROUND TESTS PASSED")
