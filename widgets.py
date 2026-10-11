"""
Custom Widgets
===============
Reusable UI widgets for the fingerprint classification application.

Includes:
- ImageDisplayLabel  : image viewer with mouse-wheel zoom
- HistogramWidget    : live grayscale histogram with QPainter
- ClassificationResultWidget : result panel with bars & details
"""

import numpy as np
import cv2
from PySide6.QtWidgets import (
    QLabel, QFrame, QProgressBar, QVBoxLayout, QHBoxLayout, QWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QFont, QPainter, QPen, QLinearGradient, QColor,
)

from processor import FingerprintProcessor


# ── Image Display ────────────────────────────────────────────────────────

class ImageDisplayLabel(QLabel):
    """Label that displays OpenCV images with fit-to-view scaling and zoom."""

    def __init__(self, placeholder_text="No Image", parent=None):
        super().__init__(parent)
        self.placeholder_text = placeholder_text
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel {"
            "  background-color: #1a1a2e;"
            "  border: 2px solid #16213e;"
            "  border-radius: 8px;"
            "  color: #555;"
            "  font-size: 13px;"
            "}"
        )
        self._pixmap = None
        self._img_data = None
        self._zoom = 1.0

    # ── public ──

    def set_image(self, cv_img):
        """Set image from an OpenCV numpy array (grayscale or BGR)."""
        if cv_img is None:
            self.clear()
            self.setText(self.placeholder_text)
            self._pixmap = None
            self._img_data = None
            self._zoom = 1.0
            return

        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            self._img_data = cv_img.copy()
            qimg = QImage(
                self._img_data.data, w, h, w, QImage.Format_Grayscale8
            )
        else:
            h, w, ch = cv_img.shape
            self._img_data = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            bpl = ch * w
            qimg = QImage(
                self._img_data.data, w, h, bpl, QImage.Format_RGB888
            )

        self._pixmap = QPixmap.fromImage(qimg)
        self._zoom = 1.0
        self._update_display()

    # ── zoom ──

    def wheelEvent(self, event):
        """Mouse-wheel zoom in / out."""
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * 1.2, 8.0)
        else:
            self._zoom = max(self._zoom / 1.2, 0.25)
        self._update_display()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        """Double-click resets zoom to fit-to-view."""
        self._zoom = 1.0
        self._update_display()

    # ── internal ──

    def _update_display(self):
        if self._pixmap is None:
            return
        target = self.size()
        if self._zoom != 1.0:
            target = target.__class__(
                int(target.width() * self._zoom),
                int(target.height() * self._zoom),
            )
        scaled = self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


# ── Histogram ────────────────────────────────────────────────────────────

class HistogramWidget(QWidget):
    """Custom-painted grayscale histogram display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self.setMinimumSize(250, 160)

    def set_image(self, gray_image):
        """Compute and display histogram from a grayscale image."""
        if gray_image is None:
            self._data = None
            self.update()
            return
        self._data = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        m = 12
        pw, ph = w - 2 * m, h - 2 * m

        # Background
        p.fillRect(self.rect(), QColor(26, 26, 46))
        p.setPen(QPen(QColor(15, 52, 96), 1))
        p.drawRect(m, m, pw, ph)

        if self._data is None or self._data.size == 0:
            p.setPen(QColor(100, 100, 120))
            p.drawText(self.rect(), Qt.AlignCenter, "No histogram data")
            p.end()
            return

        mx = float(np.max(self._data))
        if mx <= 0:
            p.end()
            return

        bw = max(1.0, pw / 256.0)
        grad = QLinearGradient(m, m, m, m + ph)
        grad.setColorAt(0.0, QColor(233, 69, 96))
        grad.setColorAt(1.0, QColor(83, 52, 131))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)

        for i in range(256):
            bh = int(self._data[i][0] / mx * ph)
            x = m + int(i * pw / 256)
            p.drawRect(x, m + ph - bh, max(1, int(bw) + 1), bh)

        # Axis labels
        p.setPen(QColor(150, 150, 170))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(m, h - 1, "0")
        p.drawText(m + pw - 20, h - 1, "255")
        p.end()


# ── Classification Result ───────────────────────────────────────────────

class ClassificationResultWidget(QFrame):
    """Panel that shows classification, confidence, quality and details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(
            "QFrame {"
            "  background-color: #16213e;"
            "  border: 2px solid #0f3460;"
            "  border-radius: 12px;"
            "  padding: 10px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        title = QLabel("Classification Result")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #e94560; border: none;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.class_label = QLabel("\u2014")
        self.class_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.class_label.setStyleSheet("color: #ffffff; border: none;")
        self.class_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.class_label)

        # Confidence bar
        cl = QHBoxLayout()
        cl.addWidget(self._small_label("Confidence:"))
        self.confidence_bar = self._make_bar(
            "stop:0 #e94560, stop:1 #533483"
        )
        cl.addWidget(self.confidence_bar)
        layout.addLayout(cl)

        # Quality bar
        ql = QHBoxLayout()
        ql.addWidget(self._small_label("Quality:"))
        self.quality_bar = self._make_bar(
            "stop:0 #0f3460, stop:1 #00b4d8"
        )
        ql.addWidget(self.quality_bar)
        layout.addLayout(ql)

        self.details_label = QLabel("")
        self.details_label.setStyleSheet(
            "color: #bbb; border: none; font-size: 11px;"
        )
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

    # ── helpers ──

    @staticmethod
    def _small_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aaa; border: none; font-size: 12px;")
        return lbl

    @staticmethod
    def _make_bar(gradient_stops):
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFormat("%v%")
        bar.setFixedHeight(22)
        bar.setStyleSheet(
            "QProgressBar {"
            "  background-color: #1a1a2e;"
            "  border: 1px solid #0f3460;"
            "  border-radius: 6px;"
            "  text-align: center;"
            "  color: white; font-size: 11px;"
            "}"
            f"QProgressBar::chunk {{"
            f"  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"    {gradient_stops});"
            f"  border-radius: 5px;"
            f"}}"
        )
        return bar

    # ── public ──

    def set_result(self, processor: FingerprintProcessor):
        """Update display with classification results."""
        self.class_label.setText(processor.classification)

        class_colors = {
            FingerprintProcessor.ARCH: "#4ade80",
            FingerprintProcessor.TENTED_ARCH: "#86efac",
            FingerprintProcessor.LEFT_LOOP: "#60a5fa",
            FingerprintProcessor.RIGHT_LOOP: "#93c5fd",
            FingerprintProcessor.PLAIN_WHORL: "#f472b6",
            FingerprintProcessor.DOUBLE_LOOP_WHORL: "#e879f9",
            FingerprintProcessor.CENTRAL_POCKET: "#c084fc",
            FingerprintProcessor.UNKNOWN: "#f87171",
        }
        color = class_colors.get(processor.classification, "#ffffff")
        self.class_label.setStyleSheet(
            f"color: {color}; border: none; font-size: 22px;"
        )

        self.confidence_bar.setValue(int(processor.confidence * 100))
        self.quality_bar.setValue(int(processor.quality_score * 100))

        lines = [
            f"Cores: {len(processor.cores)}  |  "
            f"Deltas: {len(processor.deltas)}"
        ]
        if "avg_coherence" in processor.details:
            lines.append(
                f"Orientation Coherence: {processor.details['avg_coherence']:.2f}"
            )
        if "contrast" in processor.details:
            lines.append(f"Contrast: {processor.details['contrast']:.2f}")
        if "coverage" in processor.details:
            lines.append(
                f"Foreground Coverage: {processor.details['coverage']:.1%}"
            )
        if "processing_time" in processor.details:
            lines.append(
                f"Processing Time: {processor.details['processing_time']:.3f}s"
            )
        self.details_label.setText("\n".join(lines))

    def clear(self):
        self.class_label.setText("\u2014")
        self.class_label.setStyleSheet(
            "color: #ffffff; border: none; font-size: 22px;"
        )
        self.confidence_bar.setValue(0)
        self.quality_bar.setValue(0)
        self.details_label.setText("")