#!/usr/bin/env python3
"""
Fingerprint Classification System
==================================
Entry point for the application.

Classifies fingerprints into Arch, Loop, and Whorl categories using
orientation field analysis and Poincare index singular point detection.

Dependencies: PySide6, opencv-python, numpy
"""

import sys
from PySide6.QtWidgets import QApplication

from mainwindow import MainWindow
from theme import apply_dark_theme


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()