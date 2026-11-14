"""Qt board view for the Auto Animate editor (PySide6).

BoardScene builds QGraphicsItems from a scene dict (see board_data):
raster assets are QGraphicsPixmapItem, vector shapes are
QGraphicsPathItem. BoardView adds zoom (wheel around the cursor,
buttons, shortcuts), zoom-to-asset, zoom-to-fit and pan.

Coordinates
-----------
The board grid has its origin at the BOTTOM-LEFT corner, y up —
the opposite of Qt's scene coordinates (y down). The conversion is
the same flip render.py does in pixels:

    scene_x = board_x
    scene_y = height - board_y

An asset whose board rect is (left, bottom, w, h) sits in the scene
at (left, height - bottom - h, w, h). Zoom never changes board
coordinates — it is a transform on the view only.
"""

import base64

from PySide6 import QtCore, QtGui, QtWidgets

import board_data as bd

GRID_STEP = 50
GRID_COLOR = QtGui.QColor(200, 200, 200)
AXIS_COLOR = QtGui.QColor(150, 150, 150)

ZOOM_MIN = 0.1
ZOOM_MAX = 8.0
ZOOM_STEP = 1.15


def _pen(color="black", width=3):
    return QtGui.QPen(QtGui.QColor(color), width, QtCore.Qt.SolidLine,
                      QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)


class _AssetPixmapItem(QtWidgets.QGraphicsPixmapItem):
    """A movable/selectable image item that reports its moves."""

    def __init__(self, pixmap, asset, on_moved):
        super().__init__(pixmap)
        self.asset = asset
        self._on_moved = on_moved
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable
                      | QtWidgets.QGraphicsItem.ItemIsSelectable)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange \
                and self.scene() is not None:
            # value is the new top-left in scene coords.
            scene = self.scene()
            left, top = value.x(), value.y()
            w, h = self.pixmap().width(), self.pixmap().height()
            bottom = scene.height - top - h
            ax, ay = bd.anchor_offset(
                self.asset.get("anchor", "bottom-left"), w, h,
                self.asset.get("anchor_fx"), self.asset.get("anchor_fy"))
            self.asset["x"] = round(left + ax, 2)
            self.asset["y"] = round(bottom + ay, 2)
            if self._on_moved is not None:
                self._on_moved(self.asset)
        return super().itemChange(change, value)


class _AssetPathItem(QtWidgets.QGraphicsPathItem):
    """A movable/selectable vector item that reports its moves."""

    def __init__(self, path, asset, on_moved):
        super().__init__(path)
        self.asset = asset
        self._on_moved = on_moved
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable
                      | QtWidgets.QGraphicsItem.ItemIsSelectable)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange \
                and self.scene() is not None:
            scene = self.scene()
            rect = self.path().boundingRect()
            left, top = value.x(), value.y()
            w, h = rect.width(), rect.height()
            bottom = scene.height - top - h
            ax, ay = bd.anchor_offset(
                self.asset.get("anchor", "bottom-left"), w, h,
                self.asset.get("anchor_fx"), self.asset.get("anchor_fy"))
            self.asset["x"] = round(left + ax, 2)
            self.asset["y"] = round(bottom + ay, 2)
            if self._on_moved is not None:
                self._on_moved(self.asset)
        return super().itemChange(change, value)


class BoardScene(QtWidgets.QGraphicsScene):
    """A scene built from one frame manifest."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_dict = None
        self.base_dir = None
        self.library = None
        self.width = 1000
        self.height = 800
        self.grid_group = None
        self.grid_visible = True
        self.items_by_asset = []  # [(asset dict, QGraphicsItem)]
        self.onion_group = None
        self.on_item_moved = None  # callback(asset) — scene is dirty

    # -- coordinates ------------------------------------------------

    def board_to_scene(self, x, y):
        """Board point (origin bottom-left, y up) -> scene point."""
        return QtCore.QPointF(float(x), self.height - float(y))

    def scene_to_board(self, px, py):
        """Scene point -> board point (origin bottom-left, y up)."""
        return (float(px), self.height - float(py))

    def asset_qt_rect(self, asset, w, h):
        """QRectF of an asset in scene coords.

        `w`, `h` are the asset's DISPLAY size (after `scale` — the
        pixmap is already scaled, the path already built scaled).
        The anchor offset is taken from the display size, so no
        scale is applied here.
        """
        ax, ay = bd.anchor_offset(asset.get("anchor", "bottom-left"),
                                  w, h, asset.get("anchor_fx"),
                                  asset.get("anchor_fy"))
        left = float(asset["x"]) - ax
        bottom = float(asset["y"]) - ay
        return QtCore.QRectF(left, self.height - bottom - h, w, h)

    # -- building ---------------------------------------------------

    def set_scene(self, scene_dict, base_dir=None, library=None):
        """Rebuild all items from a scene dict."""
        self.clear()
        self.items_by_asset = []
        self.onion_group = None
        self.scene_dict = scene_dict
        self.base_dir = base_dir
        self.library = library
        self.width = scene_dict["width"]
        self.height = scene_dict["height"]
        self.setSceneRect(0, 0, self.width, self.height)
        self._build_grid()
        for asset in sorted(scene_dict.get("assets", []),
                            key=lambda a: a.get("z", 0)):
            item = self._make_item(asset)
            if item is None:
                continue
            self.addItem(item)
            self.items_by_asset.append((asset, item))
        self._apply_grid_visibility()

    def _build_grid(self):
        group = QtWidgets.QGraphicsItemGroup()
        pen = QtGui.QPen(GRID_COLOR, 1)
        axis_pen = QtGui.QPen(AXIS_COLOR, 1.5)
        x = 0
        while x <= self.width:
            line = QtWidgets.QGraphicsLineItem(x, 0, x, self.height)
            line.setPen(axis_pen if x == 0 else pen)
            group.addToGroup(line)
            x += GRID_STEP
        y = 0
        while y <= self.height:
            line = QtWidgets.QGraphicsLineItem(0, y, self.width, y)
            line.setPen(axis_pen if y == 0 else pen)
            group.addToGroup(line)
            y += GRID_STEP
        group.setZValue(-1000)
        self.addItem(group)
        self.grid_group = group

    def set_grid_visible(self, visible):
        self.grid_visible = bool(visible)
        self._apply_grid_visibility()

    def _apply_grid_visibility(self):
        if self.grid_group is not None:
            self.grid_group.setVisible(self.grid_visible)

    def _make_item(self, asset):
        if bd.is_raster(asset):
            return self._make_pixmap_item(asset)
        return self._make_path_item(asset)

    def _make_pixmap_item(self, asset):
        data = bd.load_image_bytes(asset, self.base_dir)
        pixmap = QtGui.QPixmap()
        if not pixmap.loadFromData(data):
            raise ValueError(f"could not decode image for asset "
                             f"{asset.get('label', asset.get('image', '?'))!r}")
        scale = asset.get("scale", 1.0)
        if scale != 1.0:
            pixmap = pixmap.scaled(
                max(1, round(pixmap.width() * scale)),
                max(1, round(pixmap.height() * scale)),
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation)
        item = _AssetPixmapItem(pixmap, asset, self.on_item_moved)
        rect = self.asset_qt_rect(asset, pixmap.width(), pixmap.height())
        item.setPos(rect.x(), rect.y())
        item.setZValue(asset.get("z", 0))
        return item

    def _vector_path(self, asset):
        """QPainterPath of a vector asset in its own local coords.

        Local origin is the asset's top-left; x runs 0..width,
        y runs 0..height (y down, chart convention). The item is
        then placed with setPos, so the board flip happens once,
        in asset_qt_rect.
        """
        if "shape" in asset:
            if self.library is None:
                raise ValueError(
                    f"vector asset {asset['label']!r}: no mouth library "
                    "loaded, cannot resolve its shape")
            entry = self.library[asset["shape"]]
            pts = entry["points"]
            closed = entry["drawn_as"] != "single_line"
        else:
            pts = asset["points_01"]
            closed = bool(asset.get("closed", False))
        w, h = bd.vector_size(asset, self.library)
        scale = asset.get("scale", 1.0)
        w, h = w * scale, h * scale
        path = QtGui.QPainterPath()
        for i, (px, py) in enumerate(pts):
            point = QtCore.QPointF(float(px) * w, float(py) * h)
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        if closed:
            path.closeSubpath()
        return path, w, h

    def _make_path_item(self, asset):
        path, w, h = self._vector_path(asset)
        item = _AssetPathItem(path, asset, self.on_item_moved)
        item.setPen(_pen(asset.get("color", "black"),
                         asset.get("stroke_width", 3)))
        item.setBrush(QtCore.Qt.NoBrush)
        rect = self.asset_qt_rect(asset, w, h)
        item.setPos(rect.x(), rect.y())
        item.setZValue(asset.get("z", 0))
        return item

    def item_for_asset(self, asset):
        for a, item in self.items_by_asset:
            if a is asset:
                return item
        return None

    def selected_assets(self):
        out = []
        for asset, item in self.items_by_asset:
            if item.isSelected():
                out.append(asset)
        return out

    # -- onion skin -------------------------------------------------

    def show_onion_skins(self, prev_scene, next_scene):
        """Draw neighbouring keyframes faded underneath, non-selectable.

        Previous in red, next in green — the standard onion-skin
        colours. Ghost items are plain QGraphicsItems (not the
        movable asset items), at opacity 0.3.
        """
        self.hide_onion_skins()
        group = QtWidgets.QGraphicsItemGroup()
        group.setZValue(-500)
        for scene_dict, color in ((prev_scene, "red"),
                                  (next_scene, "green")):
            if not scene_dict:
                continue
            for asset in sorted(scene_dict.get("assets", []),
                                key=lambda a: a.get("z", 0)):
                ghost = self._make_ghost_item(asset, color)
                if ghost is not None:
                    group.addToGroup(ghost)
        self.addItem(group)
        self.onion_group = group

    def _make_ghost_item(self, asset, color):
        try:
            if bd.is_raster(asset):
                data = bd.load_image_bytes(asset, self.base_dir)
                pixmap = QtGui.QPixmap()
                if not pixmap.loadFromData(data):
                    return None
                scale = asset.get("scale", 1.0)
                if scale != 1.0:
                    pixmap = pixmap.scaled(
                        max(1, round(pixmap.width() * scale)),
                        max(1, round(pixmap.height() * scale)),
                        QtCore.Qt.IgnoreAspectRatio,
                        QtCore.Qt.SmoothTransformation)
                item = QtWidgets.QGraphicsPixmapItem(pixmap)
                rect = self.asset_qt_rect(asset, pixmap.width(),
                                          pixmap.height())
            else:
                path, w, h = self._vector_path(asset)
                item = QtWidgets.QGraphicsPathItem(path)
                item.setPen(_pen(asset.get("color", "black"),
                                 asset.get("stroke_width", 3)))
                item.setBrush(QtCore.Qt.NoBrush)
                rect = self.asset_qt_rect(asset, w, h)
            item.setPos(rect.x(), rect.y())
            item.setZValue(asset.get("z", 0))
            effect = QtWidgets.QGraphicsColorizeEffect()
            effect.setColor(QtGui.QColor(color))
            effect.setStrength(1.0)
            item.setGraphicsEffect(effect)
            item.setOpacity(0.3)
            item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
            item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            return item
        except Exception:
            return None

    def hide_onion_skins(self):
        if self.onion_group is not None:
            self.removeItem(self.onion_group)
            self.onion_group = None

    # -- export -----------------------------------------------------

    def render_to_image(self, path, background="white"):
        """Render the current scene to a PNG (white background)."""
        image = QtGui.QImage(self.width, self.height,
                             QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(background))
        painter = QtGui.QPainter(image)
        self.render(painter, QtCore.QRectF(0, 0, self.width, self.height),
                    self.sceneRect())
        painter.end()
        image.save(str(path))
        return str(path)


class BoardView(QtWidgets.QGraphicsView):
    """The board: a QGraphicsView with zoom and pan.

    Wheel zooms around the cursor; scrollbars pan. Zoom is a
    transform on the view only — board coordinates never change.
    """

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.zoom_changed = None  # callback(scale) — status bar

    def _apply_zoom(self, factor):
        new_scale = self.transform().m11() * factor
        if new_scale < ZOOM_MIN or new_scale > ZOOM_MAX:
            factor = max(ZOOM_MIN / self.transform().m11(),
                         min(ZOOM_MAX / self.transform().m11(), factor))
            if factor <= 0:
                return
        self.scale(factor, factor)
        if self.zoom_changed is not None:
            self.zoom_changed(self.transform().m11())

    def zoom_in(self):
        self._apply_zoom(ZOOM_STEP)

    def zoom_out(self):
        self._apply_zoom(1.0 / ZOOM_STEP)

    def zoom_fit(self):
        if self.scene() is None:
            return
        self.fitInView(self.scene().sceneRect(),
                       QtCore.Qt.KeepAspectRatio)
        if self.zoom_changed is not None:
            self.zoom_changed(self.transform().m11())

    def zoom_to_asset(self, item):
        """Zoom so the given item fills the view."""
        if item is None:
            return
        rect = item.sceneBoundingRect()
        if rect.isEmpty():
            return
        margin = 40
        rect = rect.adjusted(-margin, -margin, margin, margin)
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        if self.zoom_changed is not None:
            self.zoom_changed(self.transform().m11())

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.ControlModifier:
            # Ctrl+wheel scrolls vertically instead of zooming.
            super().wheelEvent(event)
            return
        factor = ZOOM_STEP if event.angleDelta().y() > 0 \
            else 1.0 / ZOOM_STEP
        self._apply_zoom(factor)
        event.accept()
