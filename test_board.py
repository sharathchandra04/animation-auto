"""Tests for the Auto Animate board (PySide6 GUI).

Run headless:

    QT_QPA_PLATFORM=offscreen python test_board.py

Data-layer tests need no Qt; Qt tests use an offscreen
QApplication. Prints PASS/FAIL per test and exits non-zero if any
test fails. No test framework dependency.
"""

import copy
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import board_data as bd  # noqa: E402

LIB = bd.load_mouth_library(HERE / "mouth_library_v2" / "manifest.json")
MOCK = HERE / "board_mock"

RESULTS = []


def check(name, fn):
    try:
        fn()
    except Exception:
        RESULTS.append((name, False, traceback.format_exc()))
        print(f"FAIL {name}")
    else:
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")


def expect_error(fn, fragment=""):
    try:
        fn()
    except ValueError as exc:
        assert fragment in str(exc), f"error {exc!r} lacks {fragment!r}"
        return
    raise AssertionError("expected ValueError, got none")


def good_scene():
    return {
        "width": 1000, "height": 800, "t": 0.0,
        "assets": [
            {"image": "test_insect_magenta_outlined.png",
             "x": 50, "y": 50, "z": 0},
            {"label": "mouth", "shape": "CLOSED", "x": 350, "y": 100,
             "z": 2, "width": 200, "height": 29},
        ],
    }


# ----------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------

def test_load_good_manifest():
    scene = bd.load_manifest(MOCK / "frame_0001.json", library=LIB)
    assert scene["width"] == 1000 and scene["height"] == 800
    assert scene["t"] == 0.0
    assert len(scene["assets"]) == 3


def test_load_keyframes_sorted():
    kfs = bd.load_keyframes(MOCK, library=LIB)
    assert [k["t"] for k in kfs] == [0.0, 1.0]
    assert all(len(k["assets"]) == 3 for k in kfs)


def test_validate_bad_scale():
    scene = good_scene()
    scene["assets"][0]["scale"] = 0
    expect_error(lambda: bd.validate_scene(scene, base_dir=HERE),
                 "scale")


def test_validate_bad_anchor():
    scene = good_scene()
    scene["assets"][0]["anchor"] = "middle"
    expect_error(lambda: bd.validate_scene(scene, base_dir=HERE),
                 "anchor")


def test_validate_bad_anchor_fx():
    scene = good_scene()
    asset = scene["assets"][0]
    asset["anchor"] = "custom"
    asset["anchor_fx"] = 1.5
    asset["anchor_fy"] = 0.5
    expect_error(lambda: bd.validate_scene(scene, base_dir=HERE),
                 "anchor_fx")


def test_validate_missing_image():
    scene = good_scene()
    scene["assets"][0]["image"] = "no_such_file.png"
    expect_error(lambda: bd.validate_scene(scene, base_dir=HERE),
                 "not found")


def test_validate_points_out_of_range():
    scene = good_scene()
    scene["assets"].append({
        "label": "bad", "points_01": [[0.0, 0.0], [1.2, 0.5]],
        "x": 0, "y": 0, "z": 0, "width": 100, "height": 100})
    expect_error(lambda: bd.validate_scene(scene, base_dir=HERE),
                 "[0, 1]")


def test_validate_z_float():
    scene = good_scene()
    scene["assets"][0]["z"] = 1.5
    expect_error(lambda: bd.validate_scene(scene, base_dir=HERE),
                 "'z'")


def test_validate_unknown_shape():
    scene = good_scene()
    scene["assets"][1]["shape"] = "NOPE"
    expect_error(lambda: bd.validate_scene(scene, base_dir=HERE, library=LIB),
                 "unknown shape")


def test_anchor_math():
    assert bd.anchor_offset("bottom-left", 100, 50) == (0.0, 0.0)
    assert bd.anchor_offset("bottom-right", 100, 50) == (100.0, 0.0)
    assert bd.anchor_offset("top-left", 100, 50) == (0.0, 50.0)
    assert bd.anchor_offset("top-right", 100, 50) == (100.0, 50.0)
    assert bd.anchor_offset("center", 100, 50) == (50.0, 25.0)
    assert bd.anchor_offset("custom", 100, 50, 0.25, 0.75) == (25.0, 37.5)
    rect = bd.asset_rect({"x": 500, "y": 400, "anchor": "center",
                          "scale": 1.0}, 100, 50)
    assert rect == (450.0, 375.0, 100.0, 50.0)


def test_interpolation_midpoint():
    kfs = bd.load_keyframes(MOCK, library=LIB)
    mid = bd.frame_at_time(kfs, 0.5, LIB)
    insect_a = mid["assets"][0]
    assert abs(insect_a["x"] - 175.0) < 1e-6, insect_a["x"]
    insect_b = mid["assets"][1]
    assert abs(insect_b.get("scale", 1.0) - 0.75) < 1e-6
    mouth = mid["assets"][2]
    assert abs(mouth["height"] - 70.0) < 1e-6, mouth["height"]
    assert mouth["label"] == "mouth"


def test_shape_morph_midpoint():
    kfs = bd.load_keyframes(MOCK, library=LIB)
    a = kfs[0]["assets"][2]
    b = kfs[1]["assets"][2]
    pa, _ = bd._vector_interp_points(a, LIB)
    pb, _ = bd._vector_interp_points(b, LIB)
    span_a = float(pa[:, 1].max() - pa[:, 1].min())
    span_b = float(pb[:, 1].max() - pb[:, 1].min())
    mid = bd.interpolate_vector(a, b, 0.5, LIB)
    pts = mid["points_01"]
    span_m = max(p[1] for p in pts) - min(p[1] for p in pts)
    assert span_a < span_m < span_b, (span_a, span_m, span_b)
    assert len(pts) == 64


def test_frame_at_time_ends():
    kfs = bd.load_keyframes(MOCK, library=LIB)
    first = bd.frame_at_time(kfs, -1.0, LIB)
    assert first["assets"][0]["x"] == 50
    last = bd.frame_at_time(kfs, 5.0, LIB)
    assert last["assets"][0]["x"] == 300
    assert last["assets"][1].get("scale", 1.0) == 0.5


def test_generate_in_betweens():
    kfs = bd.load_keyframes(MOCK, library=LIB)
    new = bd.generate_in_betweens(kfs, 0.0, 1.0, fps=12, library=LIB)
    assert len(new) == 11, len(new)
    times = [t for t, _ in new]
    assert abs(times[0] - 1 / 12) < 1e-9
    assert abs(times[-1] - 11 / 12) < 1e-9
    # Never lands on an existing keyframe.
    assert all(abs(t - 0.0) > 1e-6 and abs(t - 1.0) > 1e-6 for t in times)
    # An existing middle keyframe is skipped, not overwritten.
    mid = copy.deepcopy(kfs[0])
    mid["t"] = 0.5
    kfs2 = kfs + [mid]
    new2 = bd.generate_in_betweens(kfs2, 0.0, 1.0, fps=12, library=LIB)
    assert len(new2) == 10, len(new2)
    assert all(abs(t - 0.5) > 1e-6 for t, _ in new2)


def test_image_data_base64():
    scene = bd.load_manifest(MOCK / "frame_0001.json", library=LIB)
    asset = scene["assets"][0]
    data = bd.load_image_bytes(asset)
    assert bd.png_size(data) == (400, 400)
    asset2 = scene["assets"][1]
    assert bd.png_size(bd.load_image_bytes(asset2)) == (566, 566)


def test_frame_filename():
    assert bd.frame_filename(0.0) == "frame_0001.json"
    assert bd.frame_filename(1.0) == "frame_0013.json"
    assert bd.frame_filename(0.5) == "frame_0007.json"


# ----------------------------------------------------------------------
# Qt
# ----------------------------------------------------------------------

def test_scene_item_count():
    from PySide6 import QtWidgets
    from board_view import BoardScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = BoardScene()
    data = bd.load_manifest(MOCK / "frame_0001.json", library=LIB)
    scene.set_scene(data, base_dir=MOCK, library=LIB)
    asset_items = [i for i in scene.items()
                   if i is not scene.grid_group
                   and i.parentItem() is None
                   and i is not scene.onion_group]
    # 3 asset items, plus the grid group as a 4th top-level item.
    assert len(scene.items_by_asset) == 3, len(scene.items_by_asset)
    assert scene.grid_group is not None
    assert len(asset_items) == 3, len(asset_items)
    top_level = [i for i in scene.items() if i.parentItem() is None]
    assert len(top_level) == 4, len(top_level)  # 3 assets + grid group


def test_grid_toggle():
    from PySide6 import QtWidgets
    from board_view import BoardScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = BoardScene()
    data = bd.load_manifest(MOCK / "frame_0001.json", library=LIB)
    scene.set_scene(data, base_dir=MOCK, library=LIB)
    assert scene.grid_group.isVisible()
    scene.set_grid_visible(False)
    assert not scene.grid_group.isVisible()
    scene.set_grid_visible(True)
    assert scene.grid_group.isVisible()


def test_zoom_to_asset():
    from PySide6 import QtWidgets
    from board_view import BoardScene, BoardView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = BoardScene()
    data = bd.load_manifest(MOCK / "frame_0001.json", library=LIB)
    scene.set_scene(data, base_dir=MOCK, library=LIB)
    view = BoardView(scene)
    view.resize(800, 600)
    view.show()
    app.processEvents()
    before = view.transform().m11()
    asset, item = scene.items_by_asset[0]
    view.zoom_to_asset(item)
    app.processEvents()
    after = view.transform().m11()
    assert after != before, (before, after)
    assert 0.1 <= after <= 8.0
    view.close()


def test_brush_stroke_finalisation():
    from board_brush import build_stroke_asset

    scene = {"width": 1000, "height": 800, "t": 0.0, "assets": []}
    asset = build_stroke_asset(
        [(100.0, 100.0), (150.0, 120.0), (220.0, 100.0)], scene)
    assert asset is not None
    assert asset["label"] == "stroke_1"
    assert asset["x"] == 100.0 and asset["y"] == 100.0
    assert asset["width"] == 120.0 and asset["height"] == 20.0
    assert asset["color"] == "black" and asset["stroke_width"] == 3
    assert asset["closed"] is False
    xs = [p[0] for p in asset["points_01"]]
    ys = [p[1] for p in asset["points_01"]]
    assert min(xs) == 0.0 and max(xs) == 1.0
    assert min(ys) == 0.0 and max(ys) == 1.0
    # Second stroke in the same scene auto-numbers to stroke_2.
    scene["assets"].append(asset)
    asset2 = build_stroke_asset([(0.0, 0.0), (10.0, 10.0)], scene)
    assert asset2["label"] == "stroke_2"
    # A dot with no extent is not a stroke.
    assert build_stroke_asset([(5.0, 5.0), (5.0, 5.0)], scene) is None


def test_board_window_loads_mock():
    from PySide6 import QtWidgets
    from board_app import BoardWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = BoardWindow(MOCK)
    window.resize(1200, 900)
    window.show()
    app.processEvents()
    assert len(window.keyframes) == 2
    assert window.timeline.key_list.count() == 2
    assert len(window.scene.items_by_asset) == 3
    assert window.asset_list.count() == 3
    window.view.zoom_fit()
    app.processEvents()
    assert window.view.transform().m11() > 0
    window.close()


def test_brush_draws_stroke_in_window():
    from PySide6 import QtCore, QtTest, QtWidgets
    from board_app import BoardWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = BoardWindow(MOCK)
    window.resize(1200, 900)
    window.show()
    app.processEvents()
    window.view.zoom_fit()
    app.processEvents()
    window._set_mode("brush")
    app.processEvents()
    viewport = window.view.viewport()
    before = len(window.keyframes[0]["assets"])
    QtTest.QTest.mousePress(viewport, QtCore.Qt.LeftButton,
                            QtCore.Qt.NoModifier, QtCore.QPoint(300, 300))
    QtTest.QTest.mouseMove(viewport, QtCore.QPoint(350, 320))
    QtTest.QTest.mouseMove(viewport, QtCore.QPoint(400, 300))
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.LeftButton,
                              QtCore.Qt.NoModifier, QtCore.QPoint(400, 300))
    app.processEvents()
    strokes = [a for a in window.keyframes[0]["assets"]
               if a.get("label", "").startswith("stroke_")]
    assert len(strokes) == 1, window.keyframes[0]["assets"]
    assert len(window.keyframes[0]["assets"]) == before + 1
    assert strokes[0]["color"] == "black"
    assert len(strokes[0]["points_01"]) >= 2
    window.close()


def test_onion_skins():
    from PySide6 import QtWidgets
    from board_view import BoardScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = BoardScene()
    data = bd.load_manifest(MOCK / "frame_0001.json", library=LIB)
    nxt = bd.load_manifest(MOCK / "frame_0013.json", library=LIB)
    scene.set_scene(data, base_dir=MOCK, library=LIB)
    assert scene.onion_group is None
    scene.show_onion_skins(None, nxt)
    assert scene.onion_group is not None
    ghosts = [i for i in scene.onion_group.childItems()]
    assert len(ghosts) == 3, len(ghosts)
    assert all(abs(i.opacity() - 0.3) < 1e-6 for i in ghosts)
    assert all(not i.flags() & QtWidgets.QGraphicsItem.ItemIsSelectable
               for i in ghosts)
    scene.hide_onion_skins()
    assert scene.onion_group is None


def test_generate_in_betweens_writes_files():
    import shutil
    import tempfile

    from PySide6 import QtWidgets
    from board_app import BoardWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tmp = Path(tempfile.mkdtemp(prefix="board_mock_copy_"))
    try:
        shutil.copytree(MOCK, tmp / "board_mock")
        folder = tmp / "board_mock"
        window = BoardWindow(folder)
        window.resize(1200, 900)
        window.show()
        app.processEvents()
        assert len(window.keyframes) == 2
        window._generate_in_betweens(0.0, 1.0)
        app.processEvents()
        assert len(window.keyframes) == 13, len(window.keyframes)
        files = sorted(folder.glob("frame_*.json"))
        assert len(files) == 13, len(files)
        mid = bd.load_manifest(folder / "frame_0007.json", library=LIB)
        assert abs(mid["t"] - 0.5) < 1e-6
        assert abs(mid["assets"][0]["x"] - 175.0) < 1e-6
        window.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_frame_to_png(tmp_path=None):
    from PySide6 import QtWidgets
    from board_view import BoardScene

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = BoardScene()
    data = bd.load_manifest(MOCK / "frame_0001.json", library=LIB)
    scene.set_scene(data, base_dir=MOCK, library=LIB)
    out = HERE / "test_board_render.png"
    try:
        scene.render_to_image(out)
        assert out.is_file()
        data_bytes = out.read_bytes()
        assert bd.png_size(data_bytes) == (1000, 800)
    finally:
        if out.exists():
            out.unlink()


def main():
    tests = [
        ("load_good_manifest", test_load_good_manifest),
        ("load_keyframes_sorted", test_load_keyframes_sorted),
        ("validate_bad_scale", test_validate_bad_scale),
        ("validate_bad_anchor", test_validate_bad_anchor),
        ("validate_bad_anchor_fx", test_validate_bad_anchor_fx),
        ("validate_missing_image", test_validate_missing_image),
        ("validate_points_out_of_range", test_validate_points_out_of_range),
        ("validate_z_float", test_validate_z_float),
        ("validate_unknown_shape", test_validate_unknown_shape),
        ("anchor_math", test_anchor_math),
        ("interpolation_midpoint", test_interpolation_midpoint),
        ("shape_morph_midpoint", test_shape_morph_midpoint),
        ("frame_at_time_ends", test_frame_at_time_ends),
        ("generate_in_betweens", test_generate_in_betweens),
        ("image_data_base64", test_image_data_base64),
        ("frame_filename", test_frame_filename),
        ("scene_item_count", test_scene_item_count),
        ("grid_toggle", test_grid_toggle),
        ("zoom_to_asset", test_zoom_to_asset),
        ("brush_stroke_finalisation", test_brush_stroke_finalisation),
        ("brush_draws_stroke_in_window",
         test_brush_draws_stroke_in_window),
        ("onion_skins", test_onion_skins),
        ("generate_in_betweens_writes_files",
         test_generate_in_betweens_writes_files),
        ("board_window_loads_mock", test_board_window_loads_mock),
        ("render_frame_to_png", test_render_frame_to_png),
    ]
    for name, fn in tests:
        check(name, fn)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} tests passed")
    for name, ok, tb in RESULTS:
        if not ok:
            print(f"\n--- {name} FAILED ---\n{tb}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
