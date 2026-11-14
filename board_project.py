"""Project / sub-project / scene management for the Auto Animate board.

Data layer only — no Qt imports, pure stdlib — so the hierarchy can
be scanned, created and checked without a GUI.

Hierarchy
---------
projects/                      <- projects root (a folder the user picks)
  DemoProject/                 <- a project: project.json + scenes/ + subprojects/
    project.json               {"name", "width", "height", "fps"}
    scenes/
      01_opening/              <- a scene: scene.json + frame_*.json
        scene.json             {"name"}
        frame_0001.json        <- a keyframe (the scene JSON format)
        frame_0013.json
    subprojects/
      Part2/                   <- a sub-project: same layout as a project
        project.json
        scenes/
          02_chase/
            scene.json
            frame_0001.json
            ...

A sub-project is a project, so the layout recurses to any depth.
A project is one self-contained folder: copy the folder and you copy
the project.

Missing or invalid project.json / scene.json never raises: the folder
still appears, with defaults (name = folder name, fps = 12, canvas
1000x800) and the problem recorded on the node as "error", so the
panel can show a warning marker instead of crashing.
"""

import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTER_KEYFRAME = HERE / "board_mock" / "frame_0001.json"

DEFAULT_FPS = 12
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 800

PROJECT_FILENAME = "project.json"
SCENE_FILENAME = "scene.json"
SCENES_DIR = "scenes"
SUBPROJECTS_DIR = "subprojects"

_BAD_NAME_CHARS = set('<>:"/\\|?*')
_FRAME_NUMBER_RE = re.compile(r"(\d+)")


def sanitize_name(name):
    """A folder name safe on Windows and Linux; raises ValueError."""
    name = str(name).strip().strip(" .")
    if not name or name in (".", ".."):
        raise ValueError("give the project/scene a name")
    bad = sorted(set(name) & _BAD_NAME_CHARS)
    if bad:
        raise ValueError(
            f"name {name!r} contains characters a folder name cannot "
            f"have on Windows: {' '.join(bad)}")
    return name


def _read_json(path):
    """(data dict or None, error string or None) for a JSON file."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc})"
    except OSError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "must be a JSON object"
    return data, None


def _is_good_number(value):
    return isinstance(value, (int, float)) \
        and not isinstance(value, bool) and value > 0


def load_project_info(path):
    """(info, error) for a project folder.

    info has name/width/height/fps with defaults filled in; error is
    None when project.json was fine, else what was wrong.
    """
    path = Path(path)
    info = {"name": path.name, "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT, "fps": DEFAULT_FPS}
    data, error = _read_json(path / PROJECT_FILENAME)
    if data is None:
        return info, error
    problems = []
    for key in ("width", "height", "fps"):
        if key in data:
            if _is_good_number(data[key]):
                info[key] = int(data[key]) if key == "fps" else data[key]
            else:
                problems.append(f"'{key}' must be a number above 0 "
                                f"(using {info[key]})")
    if isinstance(data.get("name"), str) and data["name"].strip():
        info["name"] = data["name"].strip()
    if problems:
        error = "project.json: " + "; ".join(problems)
    return info, error


def load_scene_info(path):
    """(info, error) for a scene folder (info has "name")."""
    path = Path(path)
    info = {"name": path.name}
    data, error = _read_json(path / SCENE_FILENAME)
    if data is None:
        return info, error
    if isinstance(data.get("name"), str) and data["name"].strip():
        info["name"] = data["name"].strip()
    return info, None


# ----------------------------------------------------------------------
# Keyframe files inside a scene folder
# ----------------------------------------------------------------------

def _frame_number(path):
    m = _FRAME_NUMBER_RE.search(Path(path).stem)
    return int(m.group(1)) if m else 0


def time_from_frame_file(path, fps=DEFAULT_FPS):
    """Time in seconds from a frame file name (frame_0001 -> t=0)."""
    return (_frame_number(path) - 1) / fps


def keyframe_files(scene_path):
    """Frame files in a scene folder, in time order.

    Only frame_*.json — scene.json is metadata, not a keyframe, and
    the board's plain folder loader would reject it (it has no
    width/height/assets), so it is never listed here.
    """
    scene_path = Path(scene_path)
    files = [p for p in scene_path.glob("frame_*.json") if p.is_file()]
    return sorted(files, key=lambda p: (_frame_number(p), p.name))


def scene_stats(scene_path, fps=DEFAULT_FPS):
    """(keyframe_count, duration_seconds, error) for a scene folder.

    Reads each frame file's "t" when it parses, else derives it from
    the file name. A file that does not parse is skipped and reported.
    """
    scene_path = Path(scene_path)
    count = 0
    duration = 0.0
    problems = []
    for path in keyframe_files(scene_path):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        t_value = data.get("t") if isinstance(data, dict) else None
        if isinstance(t_value, (int, float)) \
                and not isinstance(t_value, bool) and t_value >= 0:
            t = float(t_value)
        else:
            t = time_from_frame_file(path, fps)
        count += 1
        duration = max(duration, t)
    error = "; ".join(problems) if problems else None
    return count, duration, error


# ----------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------

def scan_scene(path, fps=DEFAULT_FPS):
    """One scene node for a scene folder."""
    path = Path(path)
    info, info_error = load_scene_info(path)
    count, duration, stats_error = scene_stats(path, fps)
    error = "; ".join(x for x in (info_error, stats_error) if x) or None
    if count == 0:
        error = "; ".join(x for x in (error, "no keyframe files") if x)
    return {"kind": "scene", "name": info["name"], "path": path,
            "children": [], "keyframe_count": count,
            "duration": duration, "error": error,
            "project_path": None, "project_fps": fps,
            "project_name": None}


def _is_project_folder(path):
    path = Path(path)
    return (path / PROJECT_FILENAME).is_file() \
        or (path / SCENES_DIR).is_dir() \
        or (path / SUBPROJECTS_DIR).is_dir()


def scan_project(path):
    """One project/sub-project node, scanned recursively to any depth.

    Children are the project's scenes first, then its sub-projects,
    each in folder-name order.
    """
    path = Path(path)
    info, error = load_project_info(path)
    children = []
    scenes_dir = path / SCENES_DIR
    if scenes_dir.is_dir():
        for child in sorted(scenes_dir.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                node = scan_scene(child, info["fps"])
                node["project_path"] = path
                node["project_fps"] = info["fps"]
                node["project_name"] = info["name"]
                children.append(node)
    sub_dir = path / SUBPROJECTS_DIR
    if sub_dir.is_dir():
        for child in sorted(sub_dir.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                node = scan_project(child)
                node["kind"] = "subproject"
                children.append(node)
    return {"kind": "project", "name": info["name"], "path": path,
            "children": children, "width": info["width"],
            "height": info["height"], "fps": info["fps"], "error": error}


def scan_projects(root):
    """Top-level project nodes under a projects root ([] if none)."""
    root = Path(root) if root is not None else None
    if root is None or not root.is_dir():
        return []
    nodes = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        node = scan_project(child)
        if not _is_project_folder(child):
            node["error"] = "; ".join(
                x for x in (node["error"],
                            "no project.json, scenes/ or subprojects/ "
                            "found — treating as an empty project") if x)
        nodes.append(node)
    return nodes


def iter_scenes(node):
    """Every scene node under `node`, depth-first."""
    if node["kind"] == "scene":
        yield node
        return
    for child in node.get("children", []):
        yield from iter_scenes(child)


def count_scenes(node):
    """Number of scenes under a project/sub-project node."""
    return sum(1 for _ in iter_scenes(node))


def first_scene(projects):
    """First scene with keyframes in a scanned tree, or None."""
    for node in projects:
        for scene in iter_scenes(node):
            if scene.get("keyframe_count", 0) > 0:
                return scene
    return None


def find_scene(projects, scene_path):
    """The scene node for `scene_path` in a scanned tree, or None."""
    scene_path = Path(scene_path)
    for node in projects:
        for scene in iter_scenes(node):
            if scene["path"] == scene_path:
                return scene
    return None


def list_projects(root):
    """Names of the top-level projects under a root."""
    return [node["name"] for node in scan_projects(root)]


def owning_project(scene_path):
    """The project (or sub-project) folder a scene belongs to.

    Walks up from the scene folder until a folder with a project.json
    is found; falls back to two levels up (<project>/scenes/<name>).
    """
    scene_path = Path(scene_path)
    for parent in scene_path.parents:
        if (parent / PROJECT_FILENAME).is_file():
            return parent
    return scene_path.parent.parent


# ----------------------------------------------------------------------
# Creating
# ----------------------------------------------------------------------

def _write_project_json(path, name, width=DEFAULT_WIDTH,
                       height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
    (Path(path) / PROJECT_FILENAME).write_text(json.dumps(
        {"name": name, "width": width, "height": height, "fps": fps},
        indent=2) + "\n")


def _make_project_folder(path, name):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / SCENES_DIR).mkdir()
    (path / SUBPROJECTS_DIR).mkdir()
    _write_project_json(path, name)
    return path


def create_project(root, name):
    """Create a project under `root`; returns its path."""
    name = sanitize_name(name)
    root = Path(root)
    path = root / name
    if path.exists():
        raise ValueError(f"a project called {name!r} already exists "
                         f"in {root}")
    root.mkdir(parents=True, exist_ok=True)
    return _make_project_folder(path, name)


def create_subproject(project_path, name):
    """Create a sub-project inside a project; returns its path."""
    name = sanitize_name(name)
    project_path = Path(project_path)
    path = project_path / SUBPROJECTS_DIR / name
    if path.exists():
        raise ValueError(f"a sub-project called {name!r} already exists")
    (project_path / SUBPROJECTS_DIR).mkdir(parents=True, exist_ok=True)
    return _make_project_folder(path, name)


def create_scene(project_path, name):
    """Create a scene with a starter keyframe; returns its path.

    The starter is a copy of board_mock/frame_0001.json, so the board
    shows something immediately — it has embedded images and renders
    anywhere.
    """
    name = sanitize_name(name)
    project_path = Path(project_path)
    path = project_path / SCENES_DIR / name
    if path.exists():
        raise ValueError(f"a scene called {name!r} already exists")
    if not STARTER_KEYFRAME.is_file():
        raise ValueError(f"starter keyframe missing: {STARTER_KEYFRAME}")
    (project_path / SCENES_DIR).mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True)
    (path / SCENE_FILENAME).write_text(
        json.dumps({"name": name}, indent=2) + "\n")
    shutil.copyfile(STARTER_KEYFRAME, path / "frame_0001.json")
    return path
