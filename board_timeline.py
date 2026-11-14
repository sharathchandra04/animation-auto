"""Timeline panel for the Auto Animate board (PySide6).

Bottom dock of the editor. It is a dumb view: the main window owns
the keyframes and the playhead, and this panel only shows them and
emits callbacks:

- on_seek(t)            — playhead moved (list click or slider)
- on_fps_changed(fps)   — fps spinbox changed
- on_add_keyframe()     — Add keyframe button
- on_onion_skin(on)     — onion-skin checkbox
- on_generate(src_t, tgt_t) — Generate in-betweens button

Playback is a QTimer ticking every 1000/fps ms; each tick advances
the playhead by 1/fps and stops at the last keyframe time.
"""

from PySide6 import QtCore, QtWidgets

import board_data as bd


class TimelinePanel(QtWidgets.QWidget):
    """Keyframe strip, transport controls and in-between generation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyframes = []       # scene dicts, sorted by "t"
        self.current_t = 0.0
        self.fps = 12
        self.playing = False

        self.on_seek = None
        self.on_fps_changed = None
        self.on_add_keyframe = None
        self.on_onion_skin = None
        self.on_generate = None

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build_ui()

    # -- UI ---------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # Keyframe list.
        self.key_list = QtWidgets.QListWidget()
        self.key_list.setMaximumWidth(220)
        self.key_list.itemClicked.connect(self._key_clicked)
        layout.addWidget(self.key_list)

        controls = QtWidgets.QVBoxLayout()

        # Transport row: step back, play/pause, step forward, fps.
        transport = QtWidgets.QHBoxLayout()
        self.step_back_btn = QtWidgets.QPushButton("◀")
        self.step_back_btn.setToolTip("Step one frame back")
        self.step_back_btn.clicked.connect(self._step_back)
        transport.addWidget(self.step_back_btn)
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self.play_btn)
        self.step_fwd_btn = QtWidgets.QPushButton("▶|")
        self.step_fwd_btn.setToolTip("Step one frame forward")
        self.step_fwd_btn.clicked.connect(self._step_fwd)
        transport.addWidget(self.step_fwd_btn)
        transport.addWidget(QtWidgets.QLabel("fps:"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(12)
        self.fps_spin.valueChanged.connect(self._fps_changed)
        transport.addWidget(self.fps_spin)
        self.time_label = QtWidgets.QLabel("t = 0.00s")
        transport.addWidget(self.time_label)
        transport.addStretch(1)
        controls.addLayout(transport)

        # Time slider.
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._slider_changed)
        controls.addWidget(self.slider)

        # Keyframe + onion skin + in-between row.
        row = QtWidgets.QHBoxLayout()
        self.add_key_btn = QtWidgets.QPushButton("Add keyframe")
        self.add_key_btn.setToolTip(
            "Save the current scene as a new keyframe at the playhead")
        self.add_key_btn.clicked.connect(self._add_keyframe)
        row.addWidget(self.add_key_btn)
        self.onion_check = QtWidgets.QCheckBox("Onion skin")
        self.onion_check.setToolTip(
            "Show the previous (red) and next (green) keyframes faded")
        self.onion_check.toggled.connect(self._onion_toggled)
        row.addWidget(self.onion_check)
        row.addWidget(QtWidgets.QLabel("Source:"))
        self.src_combo = QtWidgets.QComboBox()
        row.addWidget(self.src_combo)
        row.addWidget(QtWidgets.QLabel("Target:"))
        self.tgt_combo = QtWidgets.QComboBox()
        row.addWidget(self.tgt_combo)
        self.gen_btn = QtWidgets.QPushButton("Generate in-betweens")
        self.gen_btn.setToolTip(
            "Interpolate between the source and target keyframes and "
            "write the intermediate frame files")
        self.gen_btn.clicked.connect(self._generate)
        row.addWidget(self.gen_btn)
        row.addStretch(1)
        controls.addLayout(row)

        layout.addLayout(controls, 1)

    # -- model ------------------------------------------------------

    def set_keyframes(self, keyframes, current_t=None):
        """Refresh the list/slider from the keyframe scenes."""
        self.keyframes = sorted(keyframes, key=lambda s: s["t"])
        self.key_list.clear()
        for scene in self.keyframes:
            t = scene["t"]
            name = bd.frame_filename(t, self.fps)
            self.key_list.addItem(f"t={t:.2f}s — {name}")
        for combo in (self.src_combo, self.tgt_combo):
            combo.blockSignals(True)
            combo.clear()
            for scene in self.keyframes:
                combo.addItem(f"t={scene['t']:.2f}s",
                              userData=scene["t"])
            combo.blockSignals(False)
        if self.keyframes:
            self.src_combo.setCurrentIndex(0)
            self.tgt_combo.setCurrentIndex(len(self.keyframes) - 1)
        duration = self.keyframes[-1]["t"] if self.keyframes else 0.0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, int(round(duration * self.fps))))
        self.slider.blockSignals(False)
        if current_t is not None:
            self.set_time(current_t)

    def set_time(self, t):
        """Move the playhead display (does not emit on_seek)."""
        self.current_t = float(t)
        self.time_label.setText(f"t = {t:.2f}s")
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(t * self.fps)))
        self.slider.blockSignals(False)
        for i, scene in enumerate(self.keyframes):
            if abs(scene["t"] - t) < 1e-6:
                self.key_list.setCurrentRow(i)
                break

    # -- transport --------------------------------------------------

    def _tick(self):
        duration = self.keyframes[-1]["t"] if self.keyframes else 0.0
        nxt = self.current_t + 1.0 / self.fps
        if nxt >= duration - 1e-9:
            self.set_time(duration)
            self.stop()
            if self.on_seek is not None:
                self.on_seek(duration)
            return
        self.set_time(nxt)
        if self.on_seek is not None:
            self.on_seek(nxt)

    def _toggle_play(self, checked):
        if checked:
            if not self.keyframes:
                self.play_btn.setChecked(False)
                return
            duration = self.keyframes[-1]["t"]
            if self.current_t >= duration - 1e-9:
                self.set_time(self.keyframes[0]["t"])
                if self.on_seek is not None:
                    self.on_seek(self.current_t)
            self.playing = True
            self.play_btn.setText("⏸ Pause")
            self._timer.start(int(1000 / self.fps))
        else:
            self.stop()

    def stop(self):
        self.playing = False
        self._timer.stop()
        self.play_btn.setChecked(False)
        self.play_btn.setText("▶ Play")

    def _step_back(self):
        self.stop()
        t = max(0.0, self.current_t - 1.0 / self.fps)
        self.set_time(t)
        if self.on_seek is not None:
            self.on_seek(t)

    def _step_fwd(self):
        self.stop()
        duration = self.keyframes[-1]["t"] if self.keyframes else 0.0
        t = min(duration, self.current_t + 1.0 / self.fps)
        self.set_time(t)
        if self.on_seek is not None:
            self.on_seek(t)

    def _fps_changed(self, fps):
        self.fps = fps
        if self.playing:
            self._timer.setInterval(int(1000 / fps))
        self.set_keyframes(self.keyframes, self.current_t)
        if self.on_fps_changed is not None:
            self.on_fps_changed(fps)

    def _slider_changed(self, value):
        t = value / self.fps
        self.set_time(t)
        if self.on_seek is not None:
            self.on_seek(t)

    def _key_clicked(self, item):
        row = self.key_list.row(item)
        t = self.keyframes[row]["t"]
        self.stop()
        self.set_time(t)
        if self.on_seek is not None:
            self.on_seek(t)

    def _add_keyframe(self):
        if self.on_add_keyframe is not None:
            self.on_add_keyframe()

    def _onion_toggled(self, checked):
        if self.on_onion_skin is not None:
            self.on_onion_skin(checked)

    def _generate(self):
        if self.on_generate is None:
            return
        if self.src_combo.count() < 2:
            return
        src_t = self.src_combo.currentData()
        tgt_t = self.tgt_combo.currentData()
        if src_t is None or tgt_t is None or tgt_t <= src_t:
            return
        self.on_generate(src_t, tgt_t)
