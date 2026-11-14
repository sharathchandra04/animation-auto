"""Tests for render_anchor.py — run: python test_render_anchor.py"""

import numpy as np

from render import render_assets
from render_anchor import (AnchoredAsset, anchor_offset,
                           render_anchored_assets)


def _square(color, size=100):
    return np.full((size, size, 3), color, np.uint8)


# A 100x100 square placed at grid point (200, 150) on a 400x300
# screen. Each anchor puts it in a different place:
#   bottom-left: x 200..299, y 150..249
#   bottom-right: x 100..199, y 150..249
#   top-left: x 200..299, y 50..149
#   top-right: x 100..199, y 50..149
#   center: x 150..249, y 100..199
#   custom (0.25, 0.75): x 175..274, y 75..174
def _screen_for(anchor, **kwargs):
    return render_anchored_assets(
        [AnchoredAsset(_square((0, 0, 255)), x=200, y=150, z=0,
                       anchor=anchor, **kwargs)],
        width=400, height=300)


def _grid_pixel(screen, gx, gy):
    """The pixel of a screen image at grid point (gx, gy)."""
    return tuple(screen[300 - 1 - gy, gx, :3])


def test_each_anchor_lands_in_its_place():
    cases = {
        "bottom-left": ((250, 200), (150, 100)),
        "bottom-right": ((150, 200), (250, 100)),
        "top-left": ((250, 100), (150, 200)),
        "top-right": ((150, 100), (250, 200)),
        "center": ((200, 150), (100, 100)),
    }
    for anchor, (inside, outside) in cases.items():
        screen = _screen_for(anchor)
        assert _grid_pixel(screen, *inside) == (0, 0, 255), \
            f"{anchor}: point {inside} must be red"
        assert _grid_pixel(screen, *outside) == (0, 0, 0), \
            f"{anchor}: point {outside} must be background"
    print("each anchor lands in its place: ok")


def test_custom_anchor():
    screen = _screen_for("custom", anchor_fx=0.25, anchor_fy=0.75)
    assert _grid_pixel(screen, 225, 125) == (0, 0, 255), \
        "custom (0.25, 0.75): centre of the square must be red"
    assert _grid_pixel(screen, 150, 200) == (0, 0, 0), \
        "custom: a point outside the square must be background"
    # Custom corners equal the named anchors.
    assert np.array_equal(
        _screen_for("custom", anchor_fx=0, anchor_fy=0),
        _screen_for("bottom-left")), "custom (0, 0) is bottom-left"
    assert np.array_equal(
        _screen_for("custom", anchor_fx=1, anchor_fy=1),
        _screen_for("top-right")), "custom (1, 1) is top-right"
    assert np.array_equal(
        _screen_for("custom", anchor_fx=0.5, anchor_fy=0.5),
        _screen_for("center")), "custom (0.5, 0.5) is center"
    print("custom anchor: ok")


def test_default_anchor_matches_old_render():
    # With no anchor given (bottom-left), this render must produce
    # exactly what the first render produces.
    from render import Asset

    img = _square((0, 0, 255))
    old = render_assets([Asset(img, x=50, y=60, z=0)], width=400, height=300)
    new = render_anchored_assets(
        [AnchoredAsset(img, x=50, y=60, z=0)], width=400, height=300)
    assert np.array_equal(old, new), "default anchor must match render.py"
    print("default anchor matches old render: ok")


def test_z_order_still_works():
    red = AnchoredAsset(_square((0, 0, 255)), x=200, y=150, z=1,
                        anchor="center")
    blue = AnchoredAsset(_square((255, 0, 0)), x=200, y=150, z=2,
                         anchor="center")
    screen = render_anchored_assets([red, blue], width=400, height=300)
    assert _grid_pixel(screen, 200, 150) == (255, 0, 0), \
        "higher z must be on top"
    print("z order still works: ok")


def test_bad_anchors_rejected():
    for anchor, fx, fy in (("middle", None, None),
                            ("custom", None, None),
                            ("custom", 0.5, None),
                            ("custom", 1.5, 0.5),
                            ("custom", 0.5, -0.1)):
        try:
            anchor_offset(anchor, 100, 100, fx, fy)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{anchor} {fx} {fy} must be rejected")
    print("bad anchors rejected: ok")


if __name__ == "__main__":
    test_each_anchor_lands_in_its_place()
    test_custom_anchor()
    test_default_anchor_matches_old_render()
    test_z_order_still_works()
    test_bad_anchors_rejected()
    print("ALL ANCHORED RENDER TESTS PASSED")
