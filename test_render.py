"""Tests for render.py — run: python test_render.py"""

import numpy as np

from render import Asset, render_assets


def _square(color, size=100):
    """A solid opaque square of a BGR colour."""
    return np.full((size, size, 3), color, np.uint8)


def test_origin_bottom_left():
    # A red square hooked at (0, 0) must sit at the bottom-left of
    # the screen: the bottom-left pixel is red, the top-left is not.
    screen = render_assets([Asset(_square((0, 0, 255)), x=0, y=0, z=0)],
                           width=400, height=300)
    assert tuple(screen[-1, 0, :3]) == (0, 0, 255), "bottom-left is red"
    assert tuple(screen[0, 0, :3]) == (0, 0, 0), "top-left is background"
    assert screen[-1, 0, 3] == 255
    print("origin bottom-left: ok")


def test_y_grows_up():
    # The same square at y=200 must appear higher on screen (a
    # smaller row index) than at y=0.
    low = render_assets([Asset(_square((0, 0, 255)), x=0, y=0, z=0)],
                        width=400, height=300)
    high = render_assets([Asset(_square((0, 0, 255)), x=0, y=200, z=0)],
                         width=400, height=300)
    rows_low = np.where(low[:, 0, 3] > 0)[0]
    rows_high = np.where(high[:, 0, 3] > 0)[0]
    assert rows_high.max() < rows_low.min(), "higher y must be higher up"
    assert rows_low.min() == 200 and rows_low.max() == 299
    assert rows_high.min() == 0 and rows_high.max() == 99
    print("y grows up: ok")


def test_hook_is_bottom_left():
    # The hook is the bottom-left corner: a 100x100 asset at (50, 60)
    # covers grid x 50..149 and y 60..159 — check its bottom row sits
    # at grid y=60 (pixel row 300-60-1=239) and its top row at y=159.
    screen = render_assets([Asset(_square((0, 255, 0)), x=50, y=60, z=0)],
                           width=400, height=300)
    assert tuple(screen[239, 50, :3]) == (0, 255, 0), "bottom row at y=60"
    assert tuple(screen[140, 50, :3]) == (0, 255, 0), "top row at y=159"
    assert screen[240, 50, 3] == 0, "one row below the hook is empty"
    assert screen[239, 49, 3] == 0, "one column left of the hook is empty"
    print("hook is bottom-left: ok")


def test_z_order_highest_on_top():
    # A red square (z=1) and a blue square (z=2) on top of each
    # other: the overlap must be blue.
    red = Asset(_square((0, 0, 255)), x=0, y=0, z=1)
    blue = Asset(_square((255, 0, 0)), x=50, y=0, z=2)
    screen = render_assets([red, blue], width=400, height=300)
    # Overlap is grid x 50..99, y 0..99 -> pixel rows 200..299.
    assert tuple(screen[250, 75, :3]) == (255, 0, 0), "overlap is blue"
    # The part of the red square that sticks out stays red.
    assert tuple(screen[250, 25, :3]) == (0, 0, 255)
    # And reversed input order must not matter: z decides, not list order.
    screen2 = render_assets([blue, red], width=400, height=300)
    assert tuple(screen2[250, 75, :3]) == (255, 0, 0)
    print("z order highest on top: ok")


def test_equal_z_keeps_list_order():
    red = Asset(_square((0, 0, 255)), x=0, y=0, z=1)
    blue = Asset(_square((255, 0, 0)), x=50, y=0, z=1)
    screen = render_assets([red, blue], width=400, height=300)
    assert tuple(screen[250, 75, :3]) == (255, 0, 0), \
        "equal z: later in the list is on top"
    print("equal z keeps list order: ok")


def test_clipping():
    # Half off the left and bottom edges: only the visible part is
    # drawn, and nothing crashes.
    screen = render_assets(
        [Asset(_square((0, 0, 255)), x=-50, y=-50, z=0)],
        width=400, height=300)
    assert tuple(screen[-1, 0, :3]) == (0, 0, 255), "visible part drawn"
    assert screen[0, 0, 3] == 0, "top of screen untouched"
    # Entirely off-screen: an empty screen, no crash.
    empty = render_assets(
        [Asset(_square((0, 0, 255)), x=500, y=500, z=0)],
        width=400, height=300)
    assert (empty[:, :, 3] == 0).all()
    print("clipping: ok")


def test_transparent_assets_blend():
    # A half-transparent blue square over a red one: the overlap is
    # a blend, not a replacement.
    red = Asset(_square((0, 0, 255)), x=0, y=0, z=0)
    blue = np.full((100, 100, 4), (255, 0, 0, 128), np.uint8)
    top = Asset(blue, x=0, y=0, z=1)
    screen = render_assets([red, top], width=400, height=300)
    px = tuple(screen[250, 50, :3])
    assert px[0] > 100 and px[2] > 100, f"overlap must blend, got {px}"
    print(f"transparent assets blend: ok (overlap pixel {px})")


if __name__ == "__main__":
    test_origin_bottom_left()
    test_y_grows_up()
    test_hook_is_bottom_left()
    test_z_order_highest_on_top()
    test_equal_z_keeps_list_order()
    test_clipping()
    test_transparent_assets_blend()
    print("ALL RENDER TESTS PASSED")
