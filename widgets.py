"""
Custom Widgets
===============
Reusable UI widgets for the fingerprint classification application.
"""

import numpy as np
import cv2
from PySide6.QtWidgets import QLabel, QFrame, QProgressBar, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont

from processor import FingerprintProcessor


class ImageDisplayLabel(QLabel):
    """Label widget that displays images with proper scaling."""

    def __init__(self, placeholder_text="No Image", parent=None):
        super().__init__(parent)
        self.placeholder_text = placeholder_text
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 8px;
                color: #555;
                font-size: 13px;
            }
        """)
        self._pixmap = None
        self._img_data = None

    def set_image(self, cv_img):
        """Set image from OpenCV numpy array."""
        if cv_img is None:
            self.clear()
            self.setText(self.placeholder_text)
            self._pixmap = None
            self._img_data = None
            return

        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            self._img_data = cv_img.copy()
            qimg = QImage(
                self._img_data.data, w, h, w,
                QImage.Format_Grayscale8
            )
        else:
            h, w, ch = cv_img.shape
            self._img_data = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            qimg = QImage(
                self._img_data.data, w, h, bytes_per_line,
                QImage.Format_RGB888
            )

        self._pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self):
        """Update the displayed pixmap, scaled to fit."""
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class ClassificationResultWidget(QFrame):
    """Widget that displays classification results with visual flair."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border: 2px solid #0f3460;
                border-radius: 12px;
                padding: 10px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

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
        conf_layout = QHBoxLayout()
        conf_text = QLabel("Confidence:")
        conf_text.setStyleSheet(
            "color: #aaa; border: none; font-size: 12px;"
        )
        conf_layout.addWidget(conf_text)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%v%")
        self.confidence_bar.setFixedHeight(22)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a2e;
                border: 1px solid #0f3460;
                border-radius: 6px;
                text-align: center;
                color: white;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #533483);
                border-radius: 5px;
            }
        """)
        conf_layout.addWidget(self.confidence_bar)
        layout.addLayout(conf_layout)

        # Quality bar
        qual_layout = QHBoxLayout()
        qual_text = QLabel("Quality:")
        qual_text.setStyleSheet(
            "color: #aaa; border: none; font-size: 12px;"
        )
        qual_layout.addWidget(qual_text)
        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setTextVisible(True)
        self.quality_bar.setFormat("%v%")
        self.quality_bar.setFixedHeight(22)
        self.quality_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a2e;
                border: 1px solid #0f3460;
                border-radius: 6px;
                text-align: center;
                color: white;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f3460, stop:1 #00b4d8);
                border-radius: 5px;
            }
        """)
        qual_layout.addWidget(self.quality_bar)
        layout.addLayout(qual_layout)

        self.details_label = QLabel("")
        self.details_label.setStyleSheet(
            "color: #bbb; border: none; font-size: 11px;"
        )
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

    def set_result(self, processor):
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

        conf = int(processor.confidence * 100)
        self.confidence_bar.setValue(conf)

        qual = int(processor.quality_score * 100)
        self.quality_bar.setValue(qual)

        details = []
        details.append(
            f"Cores: {len(processor.cores)}  |  "
            f"Deltas: {len(processor.deltas)}"
        )
        if "avg_coherence" in processor.details:
            coh = processor.details["avg_coherence"]
            details.append(f"Orientation Coherence: {coh:.2f}")
        if "contrast" in processor.details:
            con = processor.details["contrast"]
            details.append(f"Contrast: {con:.2f}")
        if "coverage" in processor.details:
            cov = processor.details["coverage"]
            details.append(f"Foreground Coverage: {cov:.1%}")
        self.details_label.setText("\n".join(details))

    def clear(self):
        self.class_label.setText("\u2014")
        self.class_label.setStyleSheet(
            "color: #ffffff; border: none; font-size: 22px;"
        )
        self.confidence_bar.setValue(0)
        self.quality_bar.setValue(0)
        self.details_label.setText("")