"""Tests for group.py — run: python test_group.py"""

import math

import numpy as np

from group import asset_centre, resolve_pivot, rotate_point, transform_group
from render import Asset, render_assets


def _square(color, size=100):
    return np.full((size, size, 3), color, np.uint8)


def test_rotate_point():
    # (700, 400) is 200 px right of the pivot (500, 400). 90 deg CCW
    # puts it 200 px above the pivot; -90 puts it below; 180 opposite.
    assert rotate_point(700, 400, 500, 400, 90) == (500.0, 600.0)
    assert rotate_point(700, 400, 500, 400, -90) == (500.0, 200.0)
    assert rotate_point(700, 400, 500, 400, 180) == (300.0, 400.0)
    assert rotate_point(700, 400, 500, 400, 0) == (700.0, 400.0)
    x, y = rotate_point(700, 400, 500, 400, 30)
    assert abs(x - 673.2) < 0.1 and abs(y - 500.0) < 0.1
    print("rotate_point: ok")


def test_rotation_keeps_distance_from_pivot():
    # After any rotation, every asset must be the same distance
    # from the pivot as before — it orbits, it does not drift.
    pivot = (500.0, 400.0)
    assets = [Asset(_square((0, 0, 255)), x=50, y=50, z=0),
              Asset(_square((255, 0, 0)), x=350, y=100, z=1)]
    before = [asset_centre(a) for a in assets]
    for angle in (30, 90, -45, 180):
        moved = transform_group(assets, pivot, angle=angle)
        for (bx, by), m in zip(before, moved):
            ax, ay = asset_centre(m)
            d0 = math.hypot(bx - pivot[0], by - pivot[1])
            d1 = math.hypot(ax - pivot[0], ay - pivot[1])
            assert abs(d0 - d1) < 2.0, f"distance changed at {angle} deg"
    print("rotation keeps distance from pivot: ok")


def test_spin_rotates_the_image():
    # A square with a dot at its top-centre, rotated 90 deg CCW
    # around its own centre: the dot must end up at the left-centre
    # of the rotated image. (The image spins with the orbit.)
    img = np.full((100, 100, 3), 255, np.uint8)
    img[5:15, 45:55] = (0, 0, 255)  # red dot at top-centre
    asset = Asset(img, x=0, y=0, z=0)
    moved = transform_group([asset], (50.0, 50.0), angle=90)
    rot = moved[0].load()
    assert rot.shape[1] == 100 and rot.shape[0] == 100
    # Left-centre of the rotated image must be red, top-centre not.
    assert tuple(rot[50, 10, :3]) == (0, 0, 255), "dot moved to the left"
    assert tuple(rot[10, 50, :3]) == (255, 255, 255), "top is now empty"
    print("spin rotates the image: ok")


def test_orbit_and_spin_together_on_screen():
    # A red square at (50, 50) on a 400x300 screen, rotated 90 deg
    # CCW around the screen centre (200, 150): its centre (100, 100)
    # is (-100, -50) from the pivot, which rotates to (50, -100),
    # so the centre goes to (250, 50) — the square ends up on the
    # right, near the bottom, at grid x 200..299, y 0..99.
    asset = Asset(_square((0, 0, 255)), x=50, y=50, z=0)
    moved = transform_group([asset], (200.0, 150.0), angle=90)
    assert (moved[0].x, moved[0].y) == (200, 0)
    screen = render_assets(moved, width=400, height=300)
    # Grid y 0..99 -> pixel rows 200..299; check well inside.
    assert tuple(screen[250, 250, :3]) == (0, 0, 255), "square on the right"
    assert tuple(screen[200, 100, :3]) == (0, 0, 0), "old spot is empty"
    print("orbit and spin together on screen: ok")


def test_translate_and_scale():
    assets = [Asset(_square((0, 0, 255)), x=100, y=100, z=0)]
    pivot = (200.0, 200.0)
    moved = transform_group(assets, pivot, dx=10, dy=-20)
    assert (moved[0].x, moved[0].y) == (110, 80), "translate shifts the hook"
    # Scale 2 about the pivot: the centre (150, 150) is 50 left and
    # 50 below the pivot, so it goes to 100 left and 100 below;
    # the image doubles to 200x200, so the hook lands at (0, 0).
    moved = transform_group(assets, pivot, scale=2.0)
    assert (moved[0].x, moved[0].y) == (0, 0), "scale doubles the offset"
    # Scale 0.5: the centre goes to (175, 175) and the image
    # shrinks to 50x50, so the hook lands at (150, 150).
    moved = transform_group(assets, pivot, scale=0.5)
    assert (moved[0].x, moved[0].y) == (150, 150), "scale 0.5 halves it"
    print("translate and scale: ok")


def test_resolve_pivot():
    assets = [Asset(_square((0, 0, 255)), x=0, y=0, z=0),
              Asset(_square((255, 0, 0)), x=200, y=200, z=0)]
    assert resolve_pivot("100,200", assets) == (100.0, 200.0)
    assert resolve_pivot("center", assets, 400, 300) == (200.0, 150.0)
    # Group pivot: average of the centres (50, 50) and (250, 250).
    assert resolve_pivot("group", assets) == (150.0, 150.0)
    print("resolve pivot: ok")


def test_identity_changes_nothing():
    assets = [Asset(_square((0, 0, 255)), x=50, y=50, z=0)]
    moved = transform_group(assets, (200.0, 200.0))
    assert (moved[0].x, moved[0].y) == (50, 50)
    assert moved[0].image is assets[0].image, "no new file for identity"
    print("identity changes nothing: ok")


if __name__ == "__main__":
    test_rotate_point()
    test_rotation_keeps_distance_from_pivot()
    test_spin_rotates_the_image()
    test_orbit_and_spin_together_on_screen()
    test_translate_and_scale()
    test_resolve_pivot()
    test_identity_changes_nothing()
    print("ALL GROUP TESTS PASSED")
