"""Tests for project / sub-project / scene management (board_project).

Run headless:

    QT_QPA_PLATFORM=offscreen python test_board_project.py

Data-layer tests need no Qt; Qt tests use an offscreen
QApplication. Prints PASS/FAIL per test and exits non-zero if any
test fails. No test framework dependency.
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import board_data as bd  # noqa: E402
import board_project as bp  # noqa: E402

DEMO_ROOT = HERE / "projects"
DEMO = DEMO_ROOT / "DemoProject"
OPENING = DEMO / "scenes" / "01_opening"
PART2 = DEMO / "subprojects" / "Part2"
CHASE = PART2 / "scenes" / "02_chase"
LIB = bd.load_mouth_library(HERE / "mouth_library_v2" / "manifest.json")

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


# ----------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------

def test_scan_demo_tree():
    nodes = bp.scan_projects(DEMO_ROOT)
    assert len(nodes) == 1, nodes
    demo = nodes[0]
    assert demo["kind"] == "project" and demo["name"] == "DemoProject"
    assert demo["fps"] == 12 and demo["width"] == 1000
    assert demo["error"] is None, demo["error"]
    scenes = [c for c in demo["children"] if c["kind"] == "scene"]
    subs = [c for c in demo["children"] if c["kind"] == "subproject"]
    assert len(scenes) == 1 and scenes[0]["name"] == "01_opening"
    assert scenes[0]["keyframe_count"] == 2
    assert abs(scenes[0]["duration"] - 1.0) < 1e-6
    assert len(subs) == 1 and subs[0]["name"] == "Part2"
    chase = [c for c in subs[0]["children"] if c["kind"] == "scene"]
    assert len(chase) == 1 and chase[0]["name"] == "02_chase"
    assert chase[0]["keyframe_count"] == 2
    assert abs(chase[0]["duration"] - 1.0) < 1e-6


def test_nesting_to_any_depth():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        top = bp.create_project(root, "Deep")
        mid = bp.create_subproject(top, "Mid")
        deep = bp.create_subproject(mid, "Leaf")
        bp.create_scene(deep, "end")
        nodes = bp.scan_projects(root)
        assert len(nodes) == 1
        mid_node = [c for c in nodes[0]["children"]
                    if c["kind"] == "subproject"][0]
        assert mid_node["name"] == "Mid"
        leaf = [c for c in mid_node["children"]
                if c["kind"] == "subproject"][0]
        assert leaf["name"] == "Leaf"
        assert leaf["kind"] == "subproject"
        scenes = [c for c in leaf["children"] if c["kind"] == "scene"]
        assert len(scenes) == 1 and scenes[0]["name"] == "end"
        assert scenes[0]["keyframe_count"] == 1


def test_create_project_subproject_scene():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proj = bp.create_project(root, "My Film")
        assert (proj / "project.json").is_file()
        info = json.loads((proj / "project.json").read_text())
        assert info == {"name": "My Film", "width": 1000,
                        "height": 800, "fps": 12}
        assert (proj / "scenes").is_dir()
        assert (proj / "subprojects").is_dir()
        sub = bp.create_subproject(proj, "Act 2")
        assert (sub / "project.json").is_file()
        assert json.loads((sub / "project.json").read_text())["name"] \
            == "Act 2"
        scene = bp.create_scene(sub, "finale")
        assert (scene / "scene.json").is_file()
        assert json.loads((scene / "scene.json").read_text()) == \
            {"name": "finale"}
        assert (scene / "frame_0001.json").is_file()
        # The starter keyframe loads through board_data.
        starter = bd.load_manifest(scene / "frame_0001.json", library=LIB)
        assert len(starter["assets"]) == 3
        assert starter["t"] == 0.0


def test_scene_keyframes_load_via_board_data():
    # scene.json is metadata, not a keyframe: keyframe_files never
    # lists it, so the frame files load cleanly through board_data.
    files = bp.keyframe_files(OPENING)
    assert [p.name for p in files] == ["frame_0001.json",
                                       "frame_0013.json"]
    scenes = [bd.load_manifest(p, library=LIB) for p in files]
    assert [s["t"] for s in scenes] == [0.0, 1.0]
    assert all(len(s["assets"]) == 3 for s in scenes)
    # The chase scene's second keyframe is the edited one.
    chase_files = bp.keyframe_files(CHASE)
    chase2 = bd.load_manifest(chase_files[1], library=LIB)
    assert chase2["t"] == 1.0
    assert chase2["assets"][0]["x"] == 600
    assert chase2["assets"][0]["y"] == 200
    assert chase2["assets"][2]["shape"] == "AH_I_2"


def test_missing_json_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        bare = Path(tmp) / "Bare"
        (bare / "scenes").mkdir(parents=True)
        info, err = bp.load_project_info(bare)
        assert info["name"] == "Bare"
        assert info["fps"] == 12 and info["width"] == 1000
        assert err is not None, "missing project.json must be surfaced"
        node = bp.scan_project(bare)
        assert node["error"] is not None
        assert node["name"] == "Bare"
        info, err = bp.load_scene_info(bare / "scenes")
        assert info["name"] == "scenes" and err is not None


def test_invalid_json_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "Broken"
        bad.mkdir()
        (bad / "project.json").write_text("{not json")
        info, err = bp.load_project_info(bad)
        assert info["fps"] == 12 and info["name"] == "Broken"
        assert err is not None and "not valid JSON" in err
        node = bp.scan_project(bad)  # never raises
        assert node["error"] is not None
        # A bad fps value falls back to the default, with a note.
        (bad / "project.json").write_text(
            json.dumps({"name": "Broken", "fps": -3}))
        info, err = bp.load_project_info(bad)
        assert info["fps"] == 12
        assert err is not None and "fps" in err


def test_owning_project():
    assert bp.owning_project(OPENING).name == "DemoProject"
    assert bp.owning_project(CHASE).name == "Part2"


def test_first_scene():
    first = bp.first_scene(bp.scan_projects(DEMO_ROOT))
    assert first is not None and first["name"] == "01_opening"
    assert first["keyframe_count"] == 2
    assert first["project_name"] == "DemoProject"
    assert bp.first_scene([]) is None
    assert bp.list_projects(DEMO_ROOT) == ["DemoProject"]


def test_duplicate_names_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proj = bp.create_project(root, "Dup")
        expect_error(lambda: bp.create_project(root, "Dup"), "already")
        bp.create_subproject(proj, "Act")
        expect_error(lambda: bp.create_subproject(proj, "Act"), "already")
        bp.create_scene(proj, "one")
        expect_error(lambda: bp.create_scene(proj, "one"), "already")


def test_sanitize_name():
    assert bp.sanitize_name("My Film") == "My Film"
    assert bp.sanitize_name("  spaced  ") == "spaced"
    expect_error(lambda: bp.sanitize_name(""), "name")
    expect_error(lambda: bp.sanitize_name("a/b"), "Windows")
    expect_error(lambda: bp.sanitize_name("a:b"), "Windows")


def test_keyframe_files_exclude_scene_json():
    files = bp.keyframe_files(CHASE)
    assert all(p.name.startswith("frame_") for p in files)
    assert not any(p.name == "scene.json" for p in files)
    assert bp.time_from_frame_file(CHASE / "frame_0013.json") == 1.0
    assert bp.time_from_frame_file(CHASE / "frame_0001.json") == 0.0


# ----------------------------------------------------------------------
# Qt
# ----------------------------------------------------------------------

def _app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() \
        or QtWidgets.QApplication([])


def _scene_names(panel):
    from PySide6 import QtCore
    names = []

    def walk(item):
        node = item.data(0, QtCore.Qt.UserRole)
        if node is not None and node["kind"] == "scene":
            names.append(node["name"])
        for i in range(item.childCount()):
            walk(item.child(i))

    for i in range(panel.tree.topLevelItemCount()):
        walk(panel.tree.topLevelItem(i))
    return names


def test_panel_lists_demo_scenes():
    from board_project_panel import ProjectPanel

    _app()
    panel = ProjectPanel()
    panel.set_root(DEMO_ROOT)
    assert panel.tree.topLevelItemCount() == 1
    assert _scene_names(panel) == ["01_opening", "02_chase"]
    assert panel.root_label.text().endswith("projects")


def test_panel_warning_marker():
    from PySide6 import QtCore
    from board_project_panel import ProjectPanel

    _app()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bad = root / "Broken"
        bad.mkdir()
        (bad / "project.json").write_text("{not json")
        panel = ProjectPanel()
        panel.set_root(root)
        assert panel.tree.topLevelItemCount() == 1
        item = panel.tree.topLevelItem(0)
        assert item.text(0).startswith("⚠"), item.text(0)
        assert item.toolTip(0), "warning must explain itself"
        assert "not valid JSON" in item.toolTip(0)


def test_window_loads_first_scene_from_projects():
    from PySide6 import QtWidgets
    from board_app import BoardWindow

    _app()
    window = BoardWindow(projects_root=DEMO_ROOT)
    window.resize(1200, 900)
    window.show()
    QtWidgets.QApplication.instance().processEvents()
    assert window.folder is not None
    assert window.folder.name == "01_opening", window.folder
    assert len(window.keyframes) == 2
    assert window.timeline.key_list.count() == 2
    assert len(window.scene.items_by_asset) == 3
    window.close()


def test_window_scene_switch_via_panel():
    from PySide6 import QtWidgets
    from board_app import BoardWindow

    _app()
    window = BoardWindow(projects_root=DEMO_ROOT)
    window.resize(1200, 900)
    window.show()
    QtWidgets.QApplication.instance().processEvents()
    assert window.folder.name == "01_opening"
    window._on_project_scene_selected(
        CHASE, {"name": "Part2", "path": PART2, "fps": 12})
    QtWidgets.QApplication.instance().processEvents()
    assert window.folder.name == "02_chase", window.folder
    assert len(window.keyframes) == 2
    assert window.keyframes[1]["assets"][0]["x"] == 600
    assert window.keyframes[1]["assets"][2]["shape"] == "AH_I_2"
    window.close()


def test_window_positional_path_still_works():
    from PySide6 import QtWidgets
    from board_app import BoardWindow

    _app()
    window = BoardWindow(HERE / "board_mock")
    window.resize(1200, 900)
    window.show()
    QtWidgets.QApplication.instance().processEvents()
    assert window.folder is not None
    assert window.folder.name == "board_mock", window.folder
    assert len(window.keyframes) == 2
    window.close()


def test_window_load_path_scene_folder():
    from PySide6 import QtWidgets
    from board_app import BoardWindow

    _app()
    window = BoardWindow(HERE / "board_mock")
    window.resize(1200, 900)
    window.show()
    QtWidgets.QApplication.instance().processEvents()
    # A scene folder (scene.json + frames) loads via load_path too —
    # scene.json is ignored, not mistaken for a keyframe.
    window.load_path(OPENING)
    QtWidgets.QApplication.instance().processEvents()
    assert window.folder.name == "01_opening"
    assert len(window.keyframes) == 2
    window.close()


def test_window_new_scene_via_panel():
    import shutil
    from PySide6 import QtWidgets
    from board_app import BoardWindow

    _app()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proj = bp.create_project(root, "Work")
        window = BoardWindow(projects_root=root)
        window.resize(1200, 900)
        window.show()
        QtWidgets.QApplication.instance().processEvents()
        # No scenes yet: the window fell back to the mock data.
        assert window.folder.name == "board_mock"
        # Create a scene through the data layer and load it the way
        # the panel's New scene button does.
        scene = bp.create_scene(proj, "first")
        window.project_panel.refresh()
        window._on_project_scene_selected(
            scene, {"name": "Work", "path": proj, "fps": 12})
        QtWidgets.QApplication.instance().processEvents()
        assert window.folder.name == "first"
        assert len(window.keyframes) == 1
        assert len(window.keyframes[0]["assets"]) == 3
        window.close()


def main():
    tests = [
        ("scan_demo_tree", test_scan_demo_tree),
        ("nesting_to_any_depth", test_nesting_to_any_depth),
        ("create_project_subproject_scene",
         test_create_project_subproject_scene),
        ("scene_keyframes_load_via_board_data",
         test_scene_keyframes_load_via_board_data),
        ("missing_json_defaults", test_missing_json_defaults),
        ("invalid_json_defaults", test_invalid_json_defaults),
        ("owning_project", test_owning_project),
        ("first_scene", test_first_scene),
        ("duplicate_names_rejected", test_duplicate_names_rejected),
        ("sanitize_name", test_sanitize_name),
        ("keyframe_files_exclude_scene_json",
         test_keyframe_files_exclude_scene_json),
        ("panel_lists_demo_scenes", test_panel_lists_demo_scenes),
        ("panel_warning_marker", test_panel_warning_marker),
        ("window_loads_first_scene_from_projects",
         test_window_loads_first_scene_from_projects),
        ("window_scene_switch_via_panel",
         test_window_scene_switch_via_panel),
        ("window_positional_path_still_works",
         test_window_positional_path_still_works),
        ("window_load_path_scene_folder",
         test_window_load_path_scene_folder),
        ("window_new_scene_via_panel", test_window_new_scene_via_panel),
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
