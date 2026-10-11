"""
Dark Theme
===========
Professional dark theme for the application.
"""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt


def apply_dark_theme(app):
    """Apply a professional dark theme to the application."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 10, 26))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 240))
    palette.setColor(QPalette.Base, QColor(26, 26, 46))
    palette.setColor(QPalette.AlternateBase, QColor(22, 33, 62))
    palette.setColor(QPalette.ToolTipBase, QColor(22, 33, 62))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 240))
    palette.setColor(QPalette.Text, QColor(220, 220, 240))
    palette.setColor(QPalette.Button, QColor(22, 33, 62))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 240))
    palette.setColor(QPalette.BrightText, QColor(233, 69, 96))
    palette.setColor(QPalette.Link, QColor(83, 52, 131))
    palette.setColor(QPalette.Highlight, QColor(83, 52, 131))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(
        QPalette.Disabled, QPalette.Text, QColor(100, 100, 120)
    )
    palette.setColor(
        QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 120)
    )
    app.setPalette(palette)

    app.setStyleSheet(
        "* { font-family: 'Segoe UI', 'Arial', sans-serif; }"
        "QGroupBox { color: #93c5fd; font-size: 13px; font-weight: bold; }"
        "QLabel { color: #ddd; }"
        "QSpinBox {"
        "  background-color: #1a1a2e; color: white;"
        "  border: 1px solid #0f3460; border-radius: 4px; padding: 4px;"
        "}"
        "QSpinBox::up-button, QSpinBox::down-button {"
        "  background-color: #0f3460; border: none; width: 20px;"
        "}"
        "QSpinBox::up-button:hover, QSpinBox::down-button:hover {"
        "  background-color: #533483;"
        "}"
        "QMenuBar { background-color: #0f3460; color: white; }"
        "QMenuBar::item:selected { background-color: #533483; }"
        "QMenu {"
        "  background-color: #16213e; color: white;"
        "  border: 1px solid #0f3460;"
        "}"
        "QMenu::item:selected { background-color: #533483; }"
    )