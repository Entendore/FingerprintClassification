#!/usr/bin/env python3
"""
Fingerprint Classification System
==================================
Entry point for the application.

Classifies fingerprints into Arch, Loop, and Whorl categories using
orientation field analysis and Poincaré index singular point detection.

Dependencies: PySide6, opencv-python, numpy
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from mainwindow import MainWindow
from theme import apply_dark_theme


def main():
    # Enable high DPI scaling for crisp rendering on HiDPI displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Fingerprint Classification System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("FPClassifier")

    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()