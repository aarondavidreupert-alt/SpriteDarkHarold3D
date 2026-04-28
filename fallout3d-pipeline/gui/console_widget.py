"""
ConsoleWidget — scrollable log panel wired to Python's logging module.
"""

import logging

from PyQt6.QtCore import pyqtSignal, QObject, Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
)

_LEVEL_COLOR = {
    logging.DEBUG:    "#888888",
    logging.INFO:     "#dddddd",
    logging.WARNING:  "#ffcc00",
    logging.ERROR:    "#ff4444",
    logging.CRITICAL: "#ff0000",
}


class _Bridge(QObject):
    """Carries log records across threads via a queued Qt signal."""
    record = pyqtSignal(object)


class QTextEditHandler(logging.Handler):
    """
    Thread-safe logging.Handler that appends colour-coded HTML lines
    to a QTextEdit on the main thread via a queued signal.
    """

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self._te = text_edit
        self._bridge = _Bridge()
        self._bridge.record.connect(
            self._append, Qt.ConnectionType.QueuedConnection
        )

    def emit(self, record: logging.LogRecord):
        try:
            self._bridge.record.emit(record)
        except Exception:
            self.handleError(record)

    def _append(self, record: logging.LogRecord):
        color = _LEVEL_COLOR.get(record.levelno, "#dddddd")
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        escaped = (
            msg.replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
               .replace("\n", "<br/>")
        )
        self._te.append(f'<span style="color:{color};">{escaped}</span>')
        self._te.moveCursor(QTextCursor.MoveOperation.End)


class ConsoleWidget(QWidget):
    """Compact log console intended to live at the bottom of the main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        lbl = QLabel("Console Log")
        lbl.setStyleSheet("font-weight: bold; color: #aaaaaa;")
        header.addWidget(lbl)
        header.addStretch()
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(60)
        btn_clear.setFixedHeight(20)
        btn_clear.clicked.connect(lambda: self._te.clear())
        header.addWidget(btn_clear)
        layout.addLayout(header)

        self._te = QTextEdit()
        self._te.setReadOnly(True)
        mono = QFont("Courier New", 9)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._te.setFont(mono)
        self._te.setStyleSheet("background:#0a0a0f; color:#dddddd; border:none;")
        layout.addWidget(self._te)

        self.handler = QTextEditHandler(self._te)
        self.handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
