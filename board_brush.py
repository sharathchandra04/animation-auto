"""Brush tool for the Auto Animate board (PySide6).

Freehand drawing on the board: mouse drag captures board points,
and on release the stroke is finalised as a vector asset — a single
black line, the same form as the mouth library shapes:

    {"label": "stroke_1", "points_01": [[x, y], ...],
     "x": <bbox bottom-left>, "y": <bbox bottom-left>,
     "width": <bbox width>, "height": <bbox height>,
     "color": "black", "stroke_width": 3}

points_01 is normalised to the stroke's bounding box (0..1 in both
axes, y = 0 at the TOP of the box, chart convention), so the stroke
is a shape like any other — it can be moved, scaled, and morphed
by the interpolation code.

The tool is deliberately small: it only captures points and builds
the asset dict. The main window owns the scene, the keyframe the
stroke goes into, and redrawing the board.
"""

import math

from PySide6 import QtCore, QtGui, QtWidgets

import board_data as bd

MIN_POINT_GAP = 2.0  # board units between captured points


class BrushTool:
    """Freehand brush over a BoardView.

    `on_stroke` is called with the finished vector asset dict.
    `on_preview` is called with the live QPainterPath (scene coords)
    while drawing, or None when the stroke ends.
    """

    def __init__(self, view, on_stroke, on_preview=None):
        self.view = view
        self.on_stroke = on_stroke
        self.on_preview = on_preview
        self.drawing = False
        self.points = []  # board coords (x, y), y up
        self.stroke_width = 3
        self.color = "black"

    def scene_to_board(self, scene_pos):
        scene = self.view.scene()
        return scene.scene_to_board(scene_pos.x(), scene_pos.y())

    def start(self, scene_pos):
        """Begin a stroke at a scene position."""
        self.drawing = True
        self.points = [self.scene_to_board(scene_pos)]
        self._update_preview()

    def add_point(self, scene_pos):
        """Extend the stroke; points closer than MIN_POINT_GAP are skipped."""
        if not self.drawing:
            return
        bx, by = self.scene_to_board(scene_pos)
        if self.points:
            px, py = self.points[-1]
            if math.hypot(bx - px, by - py) < MIN_POINT_GAP:
                return
        self.points.append((bx, by))
        self._update_preview()

    def finish(self):
        """End the stroke and emit the vector asset (or None if too short)."""
        if not self.drawing:
            return None
        self.drawing = False
        self._update_preview()
        if len(self.points) < 2:
            self.points = []
            return None
        asset = build_stroke_asset(
            self.points, self.view.scene().scene_dict,
            color=self.color, stroke_width=self.stroke_width)
        self.points = []
        if asset is not None and self.on_stroke is not None:
            self.on_stroke(asset)
        return asset

    def cancel(self):
        self.drawing = False
        self.points = []
        self._update_preview()

    def _update_preview(self):
        if self.on_preview is None:
            return
        if not self.drawing or len(self.points) < 2:
            self.on_preview(None)
            return
        scene = self.view.scene()
        path = QtGui.QPainterPath()
        for i, (bx, by) in enumerate(self.points):
            p = scene.board_to_scene(bx, by)
            if i == 0:
                path.moveTo(p)
            else:
                path.lineTo(p)
        self.on_preview(path)


def build_stroke_asset(points, scene_dict, color="black", stroke_width=3):
    """Build a vector asset dict from board points (x, y, y up).

    Returns None if the points have no extent (a single dot).
    The label is auto-numbered from the scene's existing strokes.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    w, h = right - left, top - bottom
    if w < 1.0 and h < 1.0:
        return None
    w = max(w, 1.0)
    h = max(h, 1.0)
    points_01 = [[round((x - left) / w, 4), round((top - y) / h, 4)]
                 for x, y in points]
    label = "stroke_1"
    if scene_dict is not None:
        label = bd.next_stroke_label(scene_dict)
    return {
        "label": label,
        "points_01": points_01,
        "x": round(left, 2),
        "y": round(bottom, 2),
        "z": 0,
        "scale": 1.0,
        "anchor": "bottom-left",
        "width": round(w, 2),
        "height": round(h, 2),
        "color": color,
        "stroke_width": stroke_width,
        "closed": False,
    }
