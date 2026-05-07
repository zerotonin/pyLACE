# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.main_window                                        ║
# ║  « pylace-roi standalone window — thin shell around RoiEditPanel ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The editor itself lives in pylace.roi.panel as a reusable       ║
# ║  QWidget so the same widget can be embedded in pylace-tune's     ║
# ║  Edit ROIs… dialog. This module just wraps the panel in a        ║
# ║  QMainWindow with a status bar and Ctrl+Q.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Thin QMainWindow shell around RoiEditPanel for pylace-roi."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6 import QtGui, QtWidgets

from pylace.annotator.sidecar import (
    Sidecar,
    default_sidecar_path,
    read_sidecar,
)
from pylace.roi.panel import RoiEditPanel
from pylace.roi.sidecar import default_rois_path


class RoiBuilderWindow(QtWidgets.QMainWindow):
    """Top-level window for ``pylace-roi``.

    Hosts a :class:`RoiEditPanel` as the central widget; everything
    interesting lives there. The window itself just owns the title,
    the status bar, and the window-level Ctrl+Q.
    """

    def __init__(
        self,
        video: Path,
        rois_path: Path,
        arena_sidecar: Sidecar | None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"pylace-roi — {video.name}")
        self._panel = RoiEditPanel(
            video=video, rois_path=rois_path,
            arena_sidecar=arena_sidecar, parent=self,
        )
        self._panel.statusMessage.connect(self._on_panel_status)
        self.setCentralWidget(self._panel)
        self.statusBar().showMessage("Ready.")

        quit_act = QtGui.QAction("Quit", self)
        quit_act.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)

    def _on_panel_status(self, text: str, ms: int) -> None:
        self.statusBar().showMessage(text, ms)


def run(video: Path, rois_path: Path | None = None) -> int:
    """Launch the GUI event loop. Returns a process exit code."""
    out_path = rois_path if rois_path is not None else default_rois_path(video)
    arena_path = default_sidecar_path(video)
    arena_sidecar: Sidecar | None = None
    if arena_path.exists():
        try:
            arena_sidecar = read_sidecar(arena_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: cannot read arena sidecar: {exc}", file=sys.stderr)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = RoiBuilderWindow(
        video=video, rois_path=out_path, arena_sidecar=arena_sidecar,
    )
    window.resize(1100, 720)
    window.show()
    return app.exec()


__all__ = ["RoiBuilderWindow", "run"]
