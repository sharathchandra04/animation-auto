"""Auto Animate board — the PySide6 GUI editor (v1).

QMainWindow wiring the pieces together:

- centre: BoardView / BoardScene (board_view.py)
- right dock: asset panel — list + property editor
- bottom dock: TimelinePanel (board_timeline.py)
- toolbar: Load / Save, grid toggle, zoom, Select/Brush, render, export
- brush tool (board_brush.py) for freehand vector diagrams

Data flow
--------
The board edits one keyframe file at a time; the player/exporter
consumes the whole sequence. `self.keyframes` is the list of scene
dicts (sorted by "t"); `self.displayed_scene` is the scene shown
right now — a copy from board_data.frame_at_time. Any edit writes
it back into the keyframe list (replacing the keyframe at the
playhead time, or inserting a new one if the playhead is between
keyframes), so the board is always editing a real keyframe.

Mock data lives in board_mock/ next to this file (two keyframes at
t=0 and t=1.0, 12fps) and is the default when no path is given.

Usage
-----
    python board_app.py                # mock data
    python board_app.py scene.json     # one frame file
    python board_app.py keyframes/     # folder of frame_*.json

    python board_app.py --screenshot board.png   # offscreen render

On Windows: Python 3.11+, `pip install PySide6`, clone the repo,
`python board_app.py board_mock`. Video export needs ffmpeg on
PATH; without it the app offers the PNG frames instead.
"""

import copy
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import board_data as bd
from board_brush import BrushTool
from board_timeline import TimelinePanel
from board_view import BoardScene, BoardView

HERE = Path(__file__).resolve().parent
MOCK_DIR = HERE / "board_mock"
LIBRARY_PATH = HERE / "mouth_library_v2" / "manifest.json"


class BoardWindow(QtWidgets.QMainWindow):
    """The editor window."""

    def __init__(self, start_path=None):
        super().__init__()
        self.setWindowTitle("Auto Animate Board")
        self.resize(1200, 900)

        self.library = None
        if LIBRARY_PATH.is_file():
            try:
                self.library = bd.load_mouth_library(LIBRARY_PATH)
            except Exception as exc:  # library is nice-to-have, not fatal
                print(f"warning: could not load mouth library: {exc}")

        self.keyframes = []
        self.current_t = 0.0
        self.displayed_scene = None
        self.folder = None      # keyframe folder (dir mode)
        self.file_path = None   # single manifest file (file mode)
        self.fps = 12
        self.dirty = False
        self.mode = "select"    # "select" or "brush"
        self._filling_panel = False
        self._preview_item = None

        self.scene = BoardScene(self)
        self.scene.on_item_moved = self._on_item_moved
        self.view = BoardView(self.scene, self)
        self.view.zoom_changed = self._on_zoom_changed
        self.setCentralWidget(self.view)

        self._build_toolbar()
        self._build_asset_panel()
        self._build_timeline()
        self._build_statusbar()

        self.brush = BrushTool(self.view, self._on_brush_stroke,
                               self._on_brush_preview)
        self.view.viewport().installEventFilter(self)

        self.load_path(start_path or MOCK_DIR)

    # -- building ---------------------------------------------------

    def _build_toolbar(self):
        bar = self.addToolBar("Board")
        bar.setMovable(False)

        act = QtGui.QAction("Load file…", self)
        act.setToolTip("Load one frame manifest file")
        act.triggered.connect(self._load_file_dialog)
        bar.addAction(act)
        act = QtGui.QAction("Load folder…", self)
        act.setToolTip("Load a folder of keyframe files")
        act.triggered.connect(self._load_folder_dialog)
        bar.addAction(act)
        self.save_act = QtGui.QAction("Save", self)
        self.save_act.setShortcut("Ctrl+S")
        self.save_act.setToolTip("Write the manifest file(s) back")
        self.save_act.triggered.connect(self.save)
        bar.addAction(self.save_act)
        bar.addSeparator()

        self.grid_act = QtGui.QAction("Grid", self)
        self.grid_act.setCheckable(True)
        self.grid_act.setChecked(True)
        self.grid_act.setShortcut("G")
        self.grid_act.setToolTip("Show/hide the grid")
        self.grid_act.triggered.connect(self._toggle_grid)
        bar.addAction(self.grid_act)
        bar.addSeparator()

        for label, slot in (("Zoom +", self.view.zoom_in),
                            ("Zoom −", self.view.zoom_out),
                            ("Fit", self.view.zoom_fit)):
            act = QtGui.QAction(label, self)
            act.triggered.connect(slot)
            bar.addAction(act)
        act = QtGui.QAction("Zoom to asset", self)
        act.setToolTip("Zoom so the selected asset fills the view")
        act.triggered.connect(self._zoom_to_selected)
        bar.addAction(act)
        bar.addSeparator()

        mode_group = QtGui.QActionGroup(self)
        mode_group.setExclusive(True)
        self.select_act = QtGui.QAction("Select", self)
        self.select_act.setCheckable(True)
        self.select_act.setChecked(True)
        self.select_act.setShortcut("V")
        self.select_act.triggered.connect(lambda: self._set_mode("select"))
        mode_group.addAction(self.select_act)
        bar.addAction(self.select_act)
        self.brush_act = QtGui.QAction("Brush", self)
        self.brush_act.setCheckable(True)
        self.brush_act.setShortcut("B")
        self.brush_act.setToolTip(
            "Draw freehand vector diagrams (SVG paths) on the board")
        self.brush_act.triggered.connect(lambda: self._set_mode("brush"))
        mode_group.addAction(self.brush_act)
        bar.addAction(self.brush_act)
        bar.addSeparator()

        act = QtGui.QAction("Render frame…", self)
        act.setToolTip("Render the current frame to a PNG")
        act.triggered.connect(self._render_frame_dialog)
        bar.addAction(act)
        act = QtGui.QAction("Export video…", self)
        act.setToolTip("Render every frame and stitch them to an MP4")
        act.triggered.connect(self._export_video_dialog)
        bar.addAction(act)

    def _build_asset_panel(self):
        dock = QtWidgets.QDockWidget("Assets", self)
        dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.asset_list = QtWidgets.QListWidget()
        self.asset_list.itemClicked.connect(self._asset_list_clicked)
        layout.addWidget(self.asset_list)

        form = QtWidgets.QFormLayout()
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-10000, 10000)
        self.x_spin.setDecimals(1)
        form.addRow("x", self.x_spin)
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-10000, 10000)
        self.y_spin.setDecimals(1)
        form.addRow("y", self.y_spin)
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 10.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(2)
        form.addRow("scale", self.scale_spin)
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.setRange(-100, 100)
        form.addRow("z", self.z_spin)
        self.anchor_combo = QtWidgets.QComboBox()
        self.anchor_combo.addItems(list(bd.ANCHORS))
        form.addRow("anchor", self.anchor_combo)
        self.fx_spin = QtWidgets.QDoubleSpinBox()
        self.fx_spin.setRange(0.0, 1.0)
        self.fx_spin.setSingleStep(0.1)
        self.fx_spin.setDecimals(2)
        form.addRow("anchor_fx", self.fx_spin)
        self.fy_spin = QtWidgets.QDoubleSpinBox()
        self.fy_spin.setRange(0.0, 1.0)
        self.fy_spin.setSingleStep(0.1)
        self.fy_spin.setDecimals(2)
        form.addRow("anchor_fy", self.fy_spin)
        layout.addLayout(form)
        for spin in (self.x_spin, self.y_spin, self.scale_spin,
                     self.z_spin, self.fx_spin, self.fy_spin):
            spin.valueChanged.connect(self._property_changed)
        self.anchor_combo.currentTextChanged.connect(self._property_changed)

        row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add image…")
        add_btn.clicked.connect(self._add_image_asset)
        row.addWidget(add_btn)
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected_asset)
        row.addWidget(remove_btn)
        layout.addLayout(row)
        layout.addStretch(1)

        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_timeline(self):
        dock = QtWidgets.QDockWidget("Timeline", self)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.timeline = TimelinePanel(self)
        self.timeline.on_seek = self.show_time
        self.timeline.on_fps_changed = self._on_fps_changed
        self.timeline.on_add_keyframe = self._add_keyframe
        self.timeline.on_onion_skin = self._on_onion_skin
        self.timeline.on_generate = self._generate_in_betweens
        dock.setWidget(self.timeline)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _build_statusbar(self):
        self.zoom_label = QtWidgets.QLabel("100%")
        self.time_label = QtWidgets.QLabel("t = 0.00s")
        self.dirty_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.zoom_label)
        self.statusBar().addPermanentWidget(self.time_label)
        self.statusBar().addPermanentWidget(self.dirty_label)

    # -- loading ----------------------------------------------------

    def load_path(self, path):
        """Load a manifest file, a keyframe folder, or the mock data."""
        path = Path(path)
        try:
            if path.is_dir():
                self.keyframes = bd.load_keyframes(path, self.fps,
                                                   self.library)
                self.folder = path
                self.file_path = None
            elif path.is_file():
                scene = bd.load_manifest(path, library=self.library)
                scene.setdefault("t", 0.0)
                self.keyframes = [scene]
                self.folder = None
                self.file_path = path
            else:
                raise ValueError(f"not found: {path}")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not load", str(exc))
            self.keyframes = [{"width": 1000, "height": 800, "t": 0.0,
                               "assets": []}]
            self.folder = None
            self.file_path = None
        self.current_t = self.keyframes[0]["t"]
        self.dirty = False
        self.timeline.set_keyframes(self.keyframes, self.current_t)
        self.show_time(self.current_t)
        self._update_dirty_label()
        self.statusBar().showMessage(f"loaded {len(self.keyframes)} "
                                     f"keyframe(s) from {path}")

    def _load_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load frame manifest", str(HERE),
            "Scene JSON (*.json)")
        if path:
            self.load_path(path)

    def _load_folder_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Load keyframe folder", str(HERE))
        if path:
            self.load_path(path)

    # -- showing ----------------------------------------------------

    def show_time(self, t):
        """Show the scene at time `t` (interpolated between keyframes)."""
        self.current_t = float(t)
        self.displayed_scene = bd.frame_at_time(self.keyframes, t,
                                                self.library)
        self._refresh_board()
        self._refresh_asset_list()
        self.timeline.set_time(t)
        self.time_label.setText(f"t = {t:.2f}s")
        self._refresh_onion_skins()

    def _refresh_board(self):
        selected = self.scene.selected_assets()
        self.scene.set_scene(self.displayed_scene,
                             base_dir=self.folder or
                             (self.file_path.parent
                              if self.file_path else HERE),
                             library=self.library)
        for asset in selected:
            item = self.scene.item_for_asset(asset)
            if item is not None:
                item.setSelected(True)

    def _refresh_asset_list(self):
        self.asset_list.clear()
        for asset in self.displayed_scene.get("assets", []):
            if bd.is_raster(asset):
                name = asset.get("image") or "embedded image"
            else:
                name = f"{asset.get('label', '?')} (vector)"
            self.asset_list.addItem(
                f"{name} — z={asset.get('z', 0)} "
                f"x={asset.get('x', 0):.0f} y={asset.get('y', 0):.0f}")
        self._fill_property_form()

    # -- editing ----------------------------------------------------

    def _commit_displayed(self):
        """Write the displayed scene back into the keyframe list.

        Replaces the keyframe at the playhead time, or inserts a new
        one if the playhead is between keyframes — so the board is
        always editing a real keyframe.
        """
        if self.displayed_scene is None:
            return
        t = self.current_t
        for i, kf in enumerate(self.keyframes):
            if abs(kf["t"] - t) < 1e-6:
                self.keyframes[i] = copy.deepcopy(self.displayed_scene)
                self.keyframes[i]["t"] = t
                return
        new_kf = copy.deepcopy(self.displayed_scene)
        new_kf["t"] = t
        self.keyframes.append(new_kf)
        self.keyframes.sort(key=lambda s: s["t"])
        self.timeline.set_keyframes(self.keyframes, t)

    def _mark_dirty(self):
        if not self.dirty:
            self.dirty = True
            self._update_dirty_label()

    def _update_dirty_label(self):
        self.dirty_label.setText("● unsaved changes" if self.dirty else "")

    def _on_item_moved(self, asset):
        self._commit_displayed()
        self._mark_dirty()
        self._refresh_asset_list()

    def _selected_asset(self):
        selected = self.scene.selected_assets()
        return selected[0] if selected else None

    def _fill_property_form(self):
        asset = self._selected_asset()
        self._filling_panel = True
        try:
            for widget in (self.x_spin, self.y_spin, self.scale_spin,
                           self.z_spin, self.fx_spin, self.fy_spin):
                widget.setEnabled(asset is not None)
            self.anchor_combo.setEnabled(asset is not None)
            if asset is None:
                return
            self.x_spin.setValue(float(asset.get("x", 0)))
            self.y_spin.setValue(float(asset.get("y", 0)))
            self.scale_spin.setValue(float(asset.get("scale", 1.0)))
            self.z_spin.setValue(int(asset.get("z", 0)))
            anchor = asset.get("anchor", "bottom-left")
            self.anchor_combo.setCurrentText(anchor)
            self.fx_spin.setValue(float(asset.get("anchor_fx", 0.0)))
            self.fy_spin.setValue(float(asset.get("anchor_fy", 0.0)))
            custom = anchor == "custom"
            self.fx_spin.setVisible(custom)
            self.fy_spin.setVisible(custom)
        finally:
            self._filling_panel = False

    def _property_changed(self, *args):
        if self._filling_panel:
            return
        asset = self._selected_asset()
        if asset is None:
            return
        asset["x"] = self.x_spin.value()
        asset["y"] = self.y_spin.value()
        asset["scale"] = self.scale_spin.value()
        asset["z"] = self.z_spin.value()
        asset["anchor"] = self.anchor_combo.currentText()
        if asset["anchor"] == "custom":
            asset["anchor_fx"] = self.fx_spin.value()
            asset["anchor_fy"] = self.fy_spin.value()
        self._commit_displayed()
        self._mark_dirty()
        self._refresh_board()
        self._refresh_asset_list()

    def _asset_list_clicked(self, item):
        row = self.asset_list.row(item)
        asset = self.displayed_scene["assets"][row]
        scene_item = self.scene.item_for_asset(asset)
        if scene_item is not None:
            self.scene.clearSelection()
            scene_item.setSelected(True)
        self._fill_property_form()

    def _add_image_asset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Add image asset", str(HERE),
            "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            data = Path(path).read_bytes()
            w, h = bd.png_size(data)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not add image",
                                          str(exc))
            return
        rel = os.path.relpath(path, self.folder or HERE)
        scene = self.displayed_scene
        z = max([a.get("z", 0) for a in scene.get("assets", [])],
                default=-1) + 1
        asset = {"image": rel,
                 "x": round((scene["width"] - w) / 2, 1),
                 "y": round((scene["height"] - h) / 2, 1),
                 "z": z, "scale": 1.0, "anchor": "bottom-left"}
        scene.setdefault("assets", []).append(asset)
        self._commit_displayed()
        self._mark_dirty()
        self._refresh_board()
        self._refresh_asset_list()

    def _remove_selected_asset(self):
        asset = self._selected_asset()
        if asset is None:
            return
        self.displayed_scene["assets"] = [
            a for a in self.displayed_scene.get("assets", [])
            if a is not asset]
        self._commit_displayed()
        self._mark_dirty()
        self._refresh_board()
        self._refresh_asset_list()

    def save(self):
        """Write the manifest file(s) back."""
        try:
            if self.folder is not None:
                for kf in self.keyframes:
                    bd.save_keyframe(kf, self.folder, self.fps)
                where = str(self.folder)
            elif self.file_path is not None:
                bd.save_manifest(self.keyframes[0], self.file_path)
                where = str(self.file_path)
            else:
                QtWidgets.QMessageBox.information(
                    self, "Save", "Nothing to save to — load a file or "
                    "folder first.")
                return
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.dirty = False
        self._update_dirty_label()
        self.statusBar().showMessage(f"saved to {where}")

    # -- toolbar actions --------------------------------------------

    def _toggle_grid(self, checked):
        self.scene.set_grid_visible(checked)

    def _zoom_to_selected(self):
        asset = self._selected_asset()
        if asset is None:
            self.statusBar().showMessage("select an asset first")
            return
        item = self.scene.item_for_asset(asset)
        self.view.zoom_to_asset(item)

    def _set_mode(self, mode):
        self.mode = mode
        self.brush.cancel()
        self._clear_preview()
        self.statusBar().showMessage(
            "Brush: drag on the board to draw a vector diagram"
            if mode == "brush" else "Select: click an asset, drag to move")

    def _on_zoom_changed(self, scale):
        self.zoom_label.setText(f"{scale * 100:.0f}%")

    # -- brush ------------------------------------------------------

    def eventFilter(self, watched, event):
        if watched is self.view.viewport() and self.mode == "brush":
            etype = event.type()
            if etype in (QtCore.QEvent.MouseButtonPress,
                         QtCore.QEvent.MouseMove,
                         QtCore.QEvent.MouseButtonRelease):
                # mapToScene takes ints (QPoint), not the QPointF
                # from event.position().
                pos = event.position()
                scene_pos = self.view.mapToScene(int(pos.x()),
                                                 int(pos.y()))
                if etype == QtCore.QEvent.MouseButtonPress \
                        and event.button() == QtCore.Qt.LeftButton:
                    self.brush.start(scene_pos)
                    return True
                if etype == QtCore.QEvent.MouseMove and self.brush.drawing:
                    self.brush.add_point(scene_pos)
                    return True
                if etype == QtCore.QEvent.MouseButtonRelease \
                        and event.button() == QtCore.Qt.LeftButton:
                    self.brush.finish()
                    return True
        return super().eventFilter(watched, event)

    def _on_brush_preview(self, path):
        self._clear_preview()
        if path is None:
            return
        item = QtWidgets.QGraphicsPathItem(path)
        item.setPen(QtGui.QPen(QtGui.QColor("black"), 3,
                               QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                               QtCore.Qt.RoundJoin))
        item.setZValue(10000)
        self.scene.addItem(item)
        self._preview_item = item

    def _clear_preview(self):
        if self._preview_item is not None:
            self.scene.removeItem(self._preview_item)
            self._preview_item = None

    def _on_brush_stroke(self, asset):
        # The stroke goes into a real keyframe at the playhead time.
        self._commit_displayed()
        self.displayed_scene = self.keyframes[
            [i for i, kf in enumerate(self.keyframes)
             if abs(kf["t"] - self.current_t) < 1e-6][0]]
        z = max([a.get("z", 0)
                 for a in self.displayed_scene.get("assets", [])],
                default=-1) + 1
        asset["z"] = z
        self.displayed_scene.setdefault("assets", []).append(asset)
        self._commit_displayed()
        self._mark_dirty()
        self._refresh_board()
        self._refresh_asset_list()
        self.statusBar().showMessage(f"drew {asset['label']}")

    # -- timeline ---------------------------------------------------

    def _on_fps_changed(self, fps):
        self.fps = fps

    def _add_keyframe(self):
        self._commit_displayed()
        self._mark_dirty()
        self.statusBar().showMessage(
            f"keyframe saved at t={self.current_t:.2f}s")

    def _on_onion_skin(self, on):
        self._refresh_onion_skins()

    def _refresh_onion_skins(self):
        if not self.timeline.onion_check.isChecked():
            self.scene.hide_onion_skins()
            return
        t = self.current_t
        prev = next((kf for kf in reversed(self.keyframes)
                     if kf["t"] < t - 1e-6), None)
        nxt = next((kf for kf in self.keyframes
                    if kf["t"] > t + 1e-6), None)
        self.scene.show_onion_skins(prev, nxt)

    def _generate_in_betweens(self, src_t, tgt_t):
        try:
            new_frames = bd.generate_in_betweens(
                self.keyframes, src_t, tgt_t, self.fps, self.library)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot generate", str(exc))
            return
        if not new_frames:
            QtWidgets.QMessageBox.information(
                self, "In-betweens",
                "No new frames to generate — every step between the two "
                "keyframes already has a keyframe.")
            return
        for t, scene in new_frames:
            self.keyframes.append(scene)
        self.keyframes.sort(key=lambda s: s["t"])
        if self.folder is not None:
            for t, scene in new_frames:
                bd.save_keyframe(scene, self.folder, self.fps)
            self.statusBar().showMessage(
                f"generated {len(new_frames)} in-between frame file(s) in "
                f"{self.folder}")
        else:
            self.statusBar().showMessage(
                f"generated {len(new_frames)} in-between frame(s) "
                "(in memory — save to write them out)")
        self._mark_dirty()
        self.timeline.set_keyframes(self.keyframes, self.current_t)

    # -- export -----------------------------------------------------

    def _render_frame_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Render frame", str(HERE / "frame.png"),
            "PNG image (*.png)")
        if path:
            self.scene.render_to_image(path)
            self.statusBar().showMessage(f"rendered frame -> {path}")

    def _render_frames_to(self, folder):
        """Render every frame of the sequence to PNGs in `folder`.

        Returns the number of frames written.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        start = self.keyframes[0]["t"]
        end = self.keyframes[-1]["t"]
        n = int(round((end - start) * self.fps)) + 1
        tmp_scene = BoardScene()
        for i in range(n):
            t = start + i / self.fps
            scene_dict = bd.frame_at_time(self.keyframes, t, self.library)
            tmp_scene.set_scene(
                scene_dict,
                base_dir=self.folder or (self.file_path.parent
                                         if self.file_path else HERE),
                library=self.library)
            tmp_scene.render_to_image(folder / ("frame_%04d.png" % (i + 1)))
        return n

    def _export_video_dialog(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export video", str(HERE / "board.mp4"),
            "MP4 video (*.mp4)")
        if not out:
            return
        if shutil.which("ffmpeg") is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "ffmpeg not found — save the PNG frames to")
            if folder:
                n = self._render_frames_to(folder)
                QtWidgets.QMessageBox.information(
                    self, "ffmpeg not found",
                    f"ffmpeg is not on PATH, so no MP4 was made.\n\n"
                    f"{n} PNG frames were saved to {folder} instead — "
                    "stitch them with ffmpeg on a machine that has it:\n"
                    f"ffmpeg -framerate {self.fps} -i frame_%04d.png "
                    "-c:v libx264 -pix_fmt yuv420p board.mp4")
            return
        tmp = tempfile.mkdtemp(prefix="board_frames_")
        try:
            n = self._render_frames_to(tmp)
            cmd = ["ffmpeg", "-y", "-framerate", str(self.fps),
                   "-i", os.path.join(tmp, "frame_%04d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", out]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                QtWidgets.QMessageBox.warning(
                    self, "ffmpeg failed",
                    f"ffmpeg exited with code {result.returncode}:\n"
                    f"{result.stderr[-500:]}")
                return
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.statusBar().showMessage(
            f"exported {n} frames at {self.fps}fps -> {out}")


def _grab_view(window):
    """Screenshot of the board view; falls back to a scene render.

    Returns the path written.
    """
    pixmap = window.view.grab()
    if not pixmap.isNull():
        image = pixmap.toImage()
        # A blank grab (all one colour) means the offscreen paint did
        # not happen — fall back to rendering the scene directly.
        def rgb(x, y):
            c = image.pixelColor(x, y)
            return (c.red(), c.green(), c.blue())
        colors = {rgb(0, 0),
                  rgb(image.width() // 2, image.height() // 2),
                  rgb(image.width() - 1, image.height() - 1)}
        if len(colors) > 1:
            return pixmap
    return None


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Auto Animate board (PySide6).")
    ap.add_argument("path", nargs="?", default=None,
                    help="frame manifest file or keyframe folder "
                         "(default: board_mock)")
    ap.add_argument("--screenshot", default=None, metavar="PNG",
                    help="render the board offscreen to a PNG and exit")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    window = BoardWindow(args.path)
    window.resize(1200, 900)
    window.show()
    app.processEvents()
    if args.screenshot:
        window.view.zoom_fit()
        app.processEvents()
        pixmap = _grab_view(window)
        if pixmap is not None:
            pixmap.save(args.screenshot)
        else:
            window.scene.render_to_image(args.screenshot)
        print(f"screenshot -> {args.screenshot}")
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
