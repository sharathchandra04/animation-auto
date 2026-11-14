"""Projects panel for the Auto Animate board (PySide6).

Left dock of the editor: a tree of projects -> sub-projects ->
scenes. Clicking a scene loads it on the board; clicking a project
shows its summary. Buttons create projects, sub-projects and scenes
through board_project.py; the tree never crashes on a bad folder —
those nodes carry a warning marker instead.

Callbacks (set by the main window):

- on_scene_selected(scene_path, project) — a scene was clicked;
  project is {"name", "path", "fps"} for the owning project.
- on_project_selected(node) — a project/sub-project was clicked.
"""

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import board_project as bp

_ROLE_NODE = QtCore.Qt.UserRole


def _iter_scene_nodes(node):
    if node["kind"] == "scene":
        yield node
        return
    for child in node.get("children", []):
        yield from _iter_scene_nodes(child)


class ProjectPanel(QtWidgets.QWidget):
    """Tree of projects/sub-projects/scenes plus create buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = None
        self.on_scene_selected = None
        self.on_project_selected = None
        self._build_ui()

    # -- UI ---------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.root_label = QtWidgets.QLabel("No projects folder")
        self.root_label.setWordWrap(True)
        layout.addWidget(self.root_label)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Project", "Info"])
        self.tree.setColumnWidth(0, 140)
        self.tree.setMinimumWidth(260)
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.tree, 1)

        row = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open…")
        self.open_btn.setToolTip("Choose a projects folder")
        self.open_btn.clicked.connect(self._open_root)
        row.addWidget(self.open_btn)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-scan the projects folder")
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.refresh_btn)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.new_project_btn = QtWidgets.QPushButton("New project")
        self.new_project_btn.setToolTip("Create a project in this folder")
        self.new_project_btn.clicked.connect(self._new_project)
        row.addWidget(self.new_project_btn)
        self.new_sub_btn = QtWidgets.QPushButton("New sub-project")
        self.new_sub_btn.setToolTip(
            "Create a sub-project inside the selected project")
        self.new_sub_btn.clicked.connect(self._new_subproject)
        row.addWidget(self.new_sub_btn)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.new_scene_btn = QtWidgets.QPushButton("New scene")
        self.new_scene_btn.setToolTip(
            "Create a scene inside the selected project, with a "
            "starter keyframe")
        self.new_scene_btn.clicked.connect(self._new_scene)
        row.addWidget(self.new_scene_btn)
        layout.addLayout(row)

    # -- model ------------------------------------------------------

    def set_root(self, root):
        """Point the panel at a projects folder (None clears it)."""
        self.root = Path(root) if root else None
        self.refresh()

    def refresh(self):
        """Rebuild the tree from the projects root."""
        self.tree.clear()
        if self.root is None or not self.root.is_dir():
            item = QtWidgets.QTreeWidgetItem(
                ["No projects folder", "use Open… to choose one"])
            self.tree.addTopLevelItem(item)
            self.root_label.setText("No projects folder")
            self._update_buttons()
            return
        self.root_label.setText(f"Projects: {self.root}")
        for node in bp.scan_projects(self.root):
            self._add_node(None, node)
        self.tree.expandAll()
        self._update_buttons()

    def _add_node(self, parent_item, node):
        label = node["name"]
        if node.get("error"):
            label = "⚠ " + label
        if node["kind"] == "scene":
            info = (f"{node['keyframe_count']} keyframe(s), "
                    f"{node['duration']:.1f}s")
        else:
            n_scenes = bp.count_scenes(node)
            n_subs = sum(1 for c in node["children"]
                         if c["kind"] == "subproject")
            info = f"{n_scenes} scene(s)"
            if n_subs:
                info += f", {n_subs} sub-project(s)"
            info += f", {node['fps']}fps"
        item = QtWidgets.QTreeWidgetItem([label, info])
        item.setData(0, _ROLE_NODE, node)
        if node.get("error"):
            item.setToolTip(0, node["error"])
            item.setForeground(0, QtGui.QBrush(QtGui.QColor("darkred")))
        if parent_item is None:
            self.tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)
        for child in node.get("children", []):
            self._add_node(item, child)
        return item

    # -- selection --------------------------------------------------

    def _selected_node(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        return item.data(0, _ROLE_NODE)

    def _selection_changed(self, current, previous):
        self._update_buttons()

    def _update_buttons(self):
        node = self._selected_node()
        is_project = node is not None \
            and node["kind"] in ("project", "subproject")
        self.new_sub_btn.setEnabled(is_project)
        self.new_scene_btn.setEnabled(is_project)

    def _item_clicked(self, item, column):
        node = item.data(0, _ROLE_NODE)
        if node is None:
            return
        if node["kind"] == "scene":
            self._emit_scene(node)
        elif self.on_project_selected is not None:
            self.on_project_selected(node)

    def _emit_scene(self, node):
        if self.on_scene_selected is None:
            return
        project = {"name": node.get("project_name", "?"),
                   "path": node.get("project_path"),
                   "fps": node.get("project_fps", bp.DEFAULT_FPS)}
        self.on_scene_selected(node["path"], project)

    def select_scene(self, scene_path):
        """Highlight the tree item for a scene folder (if present)."""
        item = self._find_item(Path(scene_path))
        if item is not None:
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)

    def _find_item(self, path, parent=None):
        if parent is None:
            items = [self.tree.topLevelItem(i)
                     for i in range(self.tree.topLevelItemCount())]
        else:
            items = [parent.child(i) for i in range(parent.childCount())]
        for item in items:
            node = item.data(0, _ROLE_NODE)
            if node is not None and node.get("path") == Path(path):
                return item
            found = self._find_item(path, item)
            if found is not None:
                return found
        return None

    def _select_path(self, path):
        """Highlight the tree item for a folder; returns the item."""
        item = self._find_item(Path(path))
        if item is not None:
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)
        return item

    # -- actions ----------------------------------------------------

    def _open_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose projects folder",
            str(self.root) if self.root else str(Path.home()))
        if path:
            self.set_root(path)

    def _ask_name(self, title, label):
        name, ok = QtWidgets.QInputDialog.getText(self, title, label)
        if not ok or not name.strip():
            return None
        return name

    def _new_project(self):
        if self.root is None:
            QtWidgets.QMessageBox.information(
                self, "New project",
                "Choose a projects folder first (Open…).")
            return
        name = self._ask_name("New project", "Project name:")
        if name is None:
            return
        try:
            path = bp.create_project(self.root, name)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not create",
                                          str(exc))
            return
        self.refresh()
        self._select_path(path)

    def _new_subproject(self):
        node = self._selected_node()
        if node is None or node["kind"] not in ("project", "subproject"):
            return
        name = self._ask_name("New sub-project", "Sub-project name:")
        if name is None:
            return
        try:
            path = bp.create_subproject(node["path"], name)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not create",
                                          str(exc))
            return
        self.refresh()
        self._select_path(path)

    def _new_scene(self):
        node = self._selected_node()
        if node is None or node["kind"] not in ("project", "subproject"):
            return
        name = self._ask_name("New scene", "Scene name:")
        if name is None:
            return
        try:
            path = bp.create_scene(node["path"], name)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not create",
                                          str(exc))
            return
        self.refresh()
        item = self._select_path(path)
        if item is not None:
            scene_node = item.data(0, _ROLE_NODE)
            if scene_node is not None:
                # Load the new scene straight away so the board shows
                # the starter keyframe.
                self._emit_scene(scene_node)
