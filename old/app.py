#!/usr/bin/env python3
"""
Fingerprint Classification System
==================================
Classifies fingerprints into Arch, Loop, and Whorl categories using
orientation field analysis and Poincare index singular point detection.

Dependencies: PySide6, opencv-python, numpy
"""

import sys
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTabWidget, QGroupBox,
    QProgressBar, QStatusBar, QSpinBox, QGridLayout,
    QMessageBox, QFrame, QToolBar
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import (
    QPixmap, QImage, QFont, QColor, QPalette, QAction
)


# ==============================================================================
#  FINGERPRINT PROCESSING ENGINE
# ==============================================================================

class FingerprintProcessor:
    """Core fingerprint image processing and classification engine."""

    ARCH = "Arch"
    TENTED_ARCH = "Tented Arch"
    LEFT_LOOP = "Left Loop"
    RIGHT_LOOP = "Right Loop"
    PLAIN_WHORL = "Plain Whorl"
    DOUBLE_LOOP_WHORL = "Double Loop Whorl"
    CENTRAL_POCKET = "Central Pocket Whorl"
    UNKNOWN = "Unknown"

    def __init__(self):
        self.block_size = 16
        self.original = None
        self.gray = None
        self.enhanced = None
        self.orientation_map = None
        self.orientation_strength = None
        self.binary_img = None
        self.thinned = None
        self.segmentation_mask = None
        self.cores = []
        self.deltas = []
        self.classification = self.UNKNOWN
        self.confidence = 0.0
        self.details = {}
        self.quality_score = 0.0

    def load_image(self, path):
        """Load a fingerprint image from file."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        self.original = img.copy()
        if len(img.shape) == 3:
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = img.copy()
        self._reset_results()
        return True

    def load_from_array(self, gray_img):
        """Load a grayscale numpy array."""
        self.gray = gray_img.copy()
        self.original = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        self._reset_results()
        return True

    def _reset_results(self):
        """Reset all processing results."""
        self.enhanced = None
        self.orientation_map = None
        self.orientation_strength = None
        self.binary_img = None
        self.thinned = None
        self.segmentation_mask = None
        self.cores = []
        self.deltas = []
        self.classification = self.UNKNOWN
        self.confidence = 0.0
        self.details = {}
        self.quality_score = 0.0

    def process(self, block_size=16):
        """Run the full processing pipeline."""
        if self.gray is None:
            raise ValueError("No image loaded")
        self.block_size = max(8, block_size)
        self._reset_results()
        self._preprocess()
        self._segment()
        self._compute_orientation()
        self._enhance_ridges()
        self._binarize()
        self._thin()
        self._detect_singular_points()
        self._classify()
        self._assess_quality()

    def _preprocess(self):
        """Enhance contrast and normalize the image."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.enhanced = clahe.apply(self.gray)
        self.enhanced = cv2.normalize(
            self.enhanced, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    def _segment(self):
        """Segment foreground from background using block variance."""
        h, w = self.gray.shape
        bh, bw = self.block_size, self.block_size
        mask = np.zeros_like(self.gray, dtype=np.float32)
        for y in range(0, h - bh + 1, bh):
            for x in range(0, w - bw + 1, bw):
                block = self.gray[y:y + bh, x:x + bw]
                variance = np.var(block)
                mean_val = np.mean(block)
                score = min(variance / 500.0, 1.0)
                if 30 < mean_val < 220:
                    mask[y:y + bh, x:x + bw] = score
                else:
                    mask[y:y + bh, x:x + bw] = 0.0
        mask = cv2.GaussianBlur(mask, (self.block_size * 2 + 1,) * 2, 0)
        self.segmentation_mask = (mask > 0.2).astype(np.uint8) * 255

    def _compute_orientation(self):
        """Estimate the ridge orientation field using the gradient method."""
        h, w = self.enhanced.shape
        bh, bw = self.block_size, self.block_size

        gx = cv2.Sobel(self.enhanced, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.enhanced, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.GaussianBlur(gx, (5, 5), 0)
        gy = cv2.GaussianBlur(gy, (5, 5), 0)

        rows = h // bh
        cols = w // bw
        self.orientation_map = np.zeros((rows, cols), dtype=np.float32)
        self.orientation_strength = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                y1 = i * bh
                y2 = (i + 1) * bh
                x1 = j * bw
                x2 = (j + 1) * bw

                block_gx = gx[y1:y2, x1:x2]
                block_gy = gy[y1:y2, x1:x2]

                gxx = np.sum(block_gx ** 2)
                gyy = np.sum(block_gy ** 2)
                gxy = np.sum(block_gx * block_gy)

                theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                coherence = np.sqrt(
                    (gxx - gyy) ** 2 + 4 * gxy ** 2
                ) / (gxx + gyy + 1e-10)

                self.orientation_map[i, j] = theta
                self.orientation_strength[i, j] = coherence

        self._smooth_orientation()

    def _smooth_orientation(self):
        """Low-pass filter the orientation field using vector averaging."""
        h, w = self.orientation_map.shape
        smoothed = np.zeros_like(self.orientation_map)
        sin2 = np.sin(2 * self.orientation_map)
        cos2 = np.cos(2 * self.orientation_map)
        kernel_size = 3
        pad = kernel_size // 2
        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                avg_sin = np.mean(
                    sin2[i - pad:i + pad + 1, j - pad:j + pad + 1]
                )
                avg_cos = np.mean(
                    cos2[i - pad:i + pad + 1, j - pad:j + pad + 1]
                )
                smoothed[i, j] = 0.5 * np.arctan2(avg_sin, avg_cos)
        self.orientation_map = smoothed

    def _enhance_ridges(self):
        """Enhance ridges using oriented Gabor filters."""
        if self.orientation_map is None:
            return

        freq = self._estimate_ridge_frequency()
        sigma_x = 4.0
        ksize = max(21, self.block_size * 2 + 1)

        # Compute dominant orientation via weighted vector averaging
        valid_mask = self.orientation_strength > 0.1
        if np.any(valid_mask):
            sin_vals = np.sin(2 * self.orientation_map[valid_mask])
            cos_vals = np.cos(2 * self.orientation_map[valid_mask])
            dominant_theta = 0.5 * np.arctan2(
                np.mean(sin_vals), np.mean(cos_vals)
            )
        else:
            dominant_theta = 0.0

        # Apply Gabor filter with dominant orientation
        kernel1 = cv2.getGaborKernel(
            (ksize, ksize), sigma_x, dominant_theta,
            freq, 0.5, 0, ktype=cv2.CV_32F
        )
        filtered1 = cv2.filter2D(self.enhanced, cv2.CV_32F, kernel1)

        # Apply perpendicular orientation
        kernel2 = cv2.getGaborKernel(
            (ksize, ksize), sigma_x, dominant_theta + np.pi / 2,
            freq, 0.5, 0, ktype=cv2.CV_32F
        )
        filtered2 = cv2.filter2D(self.enhanced, cv2.CV_32F, kernel2)

        # Take the stronger response at each pixel
        result = np.maximum(np.abs(filtered1), np.abs(filtered2))

        # Apply segmentation mask
        if self.segmentation_mask is not None:
            mask_float = self.segmentation_mask.astype(np.float32) / 255.0
            result = result * mask_float

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        self.enhanced = np.clip(result, 0, 255).astype(np.uint8)

    def _estimate_ridge_frequency(self):
        """Estimate average ridge frequency in the image."""
        h, w = self.enhanced.shape
        cy, cx = h // 2, w // 2
        region_size = min(h, w) // 3
        if region_size < 2:
            return 1.0 / 10.0
        y1 = max(0, cy - region_size // 2)
        y2 = min(h, cy + region_size // 2)
        x1 = max(0, cx - region_size // 2)
        x2 = min(w, cx + region_size // 2)
        region = self.enhanced[y1:y2, x1:x2]
        if region.size == 0:
            return 1.0 / 10.0

        col_mean = np.mean(region, axis=0).astype(np.float64)
        n = len(col_mean)
        if n < 4:
            return 1.0 / 10.0

        autocorr = np.correlate(
            col_mean - np.mean(col_mean),
            col_mean - np.mean(col_mean),
            mode='full'
        )
        autocorr = autocorr[n - 1:]

        for k in range(3, len(autocorr) // 2):
            if (autocorr[k] > autocorr[k - 1] and
                    autocorr[k] > autocorr[k + 1] and
                    autocorr[k] > 0.1 * autocorr[0]):
                return 1.0 / max(k, 3)

        return 1.0 / 10.0

    def _binarize(self):
        """Binarize the enhanced image."""
        self.binary_img = cv2.adaptiveThreshold(
            self.enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )
        if self.segmentation_mask is not None:
            self.binary_img = cv2.bitwise_and(
                self.binary_img, self.segmentation_mask
            )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.binary_img = cv2.morphologyEx(
            self.binary_img, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        self.binary_img = cv2.morphologyEx(
            self.binary_img, cv2.MORPH_OPEN, kernel, iterations=1
        )

    def _thin(self):
        """Thin the binary image to get the ridge skeleton."""
        if self.binary_img is None:
            return
        binary = self.binary_img.copy() // 255
        skeleton = np.zeros_like(binary, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = binary.copy()
        iterations = 0
        max_iter = 100
        while np.count_nonzero(img) > 0 and iterations < max_iter:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            iterations += 1
        self.thinned = skeleton * 255

    def _detect_singular_points(self):
        """Detect core and delta points using the Poincare index method."""
        if self.orientation_map is None:
            return
        rows, cols = self.orientation_map.shape
        self.cores = []
        self.deltas = []

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if self.orientation_strength[i, j] < 0.15:
                    continue
                by = i * self.block_size
                bx = j * self.block_size
                if (self.segmentation_mask is not None and
                        by < self.segmentation_mask.shape[0] and
                        bx < self.segmentation_mask.shape[1]):
                    if self.segmentation_mask[by, bx] == 0:
                        continue

                pi_val = self._compute_poincare_index(i, j)

                if abs(pi_val - np.pi) < np.pi / 3:
                    self.cores.append((j, i))
                elif abs(pi_val + np.pi) < np.pi / 3:
                    self.deltas.append((j, i))

        self.cores = self._merge_nearby_points(self.cores, min_dist=3)
        self.deltas = self._merge_nearby_points(self.deltas, min_dist=3)

    def _compute_poincare_index(self, i, j):
        """Compute the Poincare index at block position (i, j)."""
        neighbors = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
            (i, j + 1),
            (i + 1, j + 1), (i + 1, j), (i + 1, j - 1),
            (i, j - 1)
        ]
        total = 0.0
        n = len(neighbors)
        for k in range(n):
            i1, j1 = neighbors[k]
            i2, j2 = neighbors[(k + 1) % n]
            o1 = self.orientation_map[i1, j1]
            o2 = self.orientation_map[i2, j2]
            diff = o2 - o1
            while diff > np.pi / 2:
                diff -= np.pi
            while diff <= -np.pi / 2:
                diff += np.pi
            total += diff
        return total

    def _merge_nearby_points(self, points, min_dist=3):
        """Merge nearby singular points via non-maximum suppression."""
        if not points:
            return []
        merged = []
        used = [False] * len(points)
        for i, p1 in enumerate(points):
            if used[i]:
                continue
            cluster = [p1]
            used[i] = True
            for j, p2 in enumerate(points):
                if used[j]:
                    continue
                dist = np.sqrt(
                    (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                )
                if dist < min_dist:
                    cluster.append(p2)
                    used[j] = True
            avg_x = int(np.mean([p[0] for p in cluster]))
            avg_y = int(np.mean([p[1] for p in cluster]))
            merged.append((avg_x, avg_y))
        return merged

    def _classify(self):
        """Classify fingerprint based on detected singular points."""
        n_cores = len(self.cores)
        n_deltas = len(self.deltas)
        self.details = {
            "cores": n_cores,
            "deltas": n_deltas,
            "core_positions": self.cores,
            "delta_positions": self.deltas,
        }

        if n_cores == 0 and n_deltas == 0:
            self.classification = self.ARCH
            self.confidence = 0.85
        elif n_cores == 1 and n_deltas == 0:
            self.classification = self.TENTED_ARCH
            self.confidence = 0.75
        elif n_cores == 1 and n_deltas == 1:
            cx, _ = self.cores[0]
            dx, _ = self.deltas[0]
            if dx > cx:
                self.classification = self.LEFT_LOOP
            else:
                self.classification = self.RIGHT_LOOP
            self.confidence = 0.80
        elif n_cores >= 2 or n_deltas >= 2:
            if n_cores >= 2 and n_deltas >= 2:
                self.classification = self.DOUBLE_LOOP_WHORL
                self.confidence = 0.75
            elif n_cores == 1 and n_deltas >= 2:
                self.classification = self.CENTRAL_POCKET
                self.confidence = 0.70
            else:
                self.classification = self.PLAIN_WHORL
                self.confidence = 0.70
        else:
            self.classification = self.UNKNOWN
            self.confidence = 0.3

        # Adjust confidence by orientation coherence
        if self.orientation_strength is not None:
            valid = self.orientation_strength > 0.1
            avg_coherence = (
                float(np.mean(self.orientation_strength[valid]))
                if np.any(valid) else 0.0
            )
            self.confidence *= (0.5 + 0.5 * min(avg_coherence, 1.0))
            self.details["avg_coherence"] = avg_coherence

        self.confidence = min(self.confidence, 0.99)

    def _assess_quality(self):
        """Assess the quality of the fingerprint image."""
        if self.gray is None:
            self.quality_score = 0.0
            return

        contrast = float(np.std(self.gray)) / 128.0

        if self.segmentation_mask is not None:
            coverage = float(np.count_nonzero(self.segmentation_mask)) / \
                       float(self.segmentation_mask.size)
        else:
            coverage = 0.5

        if self.orientation_strength is not None:
            valid = self.orientation_strength > 0.1
            coherence = (
                float(np.mean(self.orientation_strength[valid]))
                if np.any(valid) else 0.0
            )
        else:
            coherence = 0.5

        self.quality_score = float(np.clip(
            0.3 * contrast + 0.3 * coverage + 0.4 * min(coherence, 1.0),
            0.0, 1.0
        ))
        self.details["quality"] = self.quality_score
        self.details["contrast"] = contrast
        self.details["coverage"] = coverage

    def get_orientation_overlay(self):
        """Create an image with orientation field overlay."""
        if self.orientation_map is None or self.original is None:
            return None

        overlay = self.original.copy()
        rows, cols = self.orientation_map.shape
        bh, bw = self.block_size, self.block_size
        half_len = self.block_size // 3

        for i in range(rows):
            for j in range(cols):
                if self.orientation_strength[i, j] < 0.1:
                    continue
                cx = j * bw + bw // 2
                cy = i * bh + bh // 2
                if (self.segmentation_mask is not None and
                        cy < self.segmentation_mask.shape[0] and
                        cx < self.segmentation_mask.shape[1]):
                    if self.segmentation_mask[cy, cx] == 0:
                        continue

                theta = self.orientation_map[i, j]
                dx = int(half_len * np.cos(theta))
                dy = int(half_len * np.sin(theta))

                strength = min(self.orientation_strength[i, j], 1.0)
                color = (
                    int(255 * (1 - strength)),
                    int(200 * strength),
                    int(255 * strength)
                )
                cv2.line(
                    overlay,
                    (cx - dx, cy - dy),
                    (cx + dx, cy + dy),
                    color, 1, cv2.LINE_AA
                )
        return overlay

    def get_singular_points_overlay(self):
        """Create an image with singular points marked."""
        if self.original is None:
            return None

        overlay = self.original.copy()
        bh, bw = self.block_size, self.block_size
        radius = max(bh, bw) // 2 + 5

        for cx, cy in self.cores:
            px = cx * bw + bw // 2
            py = cy * bh + bh // 2
            cv2.circle(overlay, (px, py), radius, (0, 0, 255), 2)
            cv2.putText(
                overlay, "Core", (px - 15, py - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
            )

        for cx, cy in self.deltas:
            px = cx * bw + bw // 2
            py = cy * bh + bh // 2
            pts = np.array([
                [px, py - radius],
                [px - radius, py + radius],
                [px + radius, py + radius]
            ], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, (255, 150, 0), 2)
            cv2.putText(
                overlay, "Delta", (px - 15, py + radius + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1
            )
        return overlay


# ==============================================================================
#  SYNTHETIC FINGERPRINT GENERATOR
# ==============================================================================

class FingerprintGenerator:
    """Generates synthetic fingerprint-like patterns for testing."""

    @staticmethod
    def generate_arch(size=300, noise_level=0.15):
        """Generate a synthetic Arch pattern."""
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 4 * np.pi, size)
        X, Y = np.meshgrid(x, y)
        pattern = np.sin(X * 3 + 0.4 * np.sin(Y * 1.5))
        noise = np.random.randn(size, size) * noise_level
        pattern = pattern + noise
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        pattern = cv2.GaussianBlur(pattern, (3, 3), 0.5)
        return pattern

    @staticmethod
    def generate_loop(size=300, direction='left', noise_level=0.15):
        """Generate a synthetic Loop pattern."""
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        cx, cy = (-0.5 if direction == 'left' else 0.5, -0.5)
        theta = np.arctan2(Y - cy, X - cx)
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        pattern = np.sin(theta * 3 + r * 5)
        parallel = np.sin(X * 5)
        blend = np.clip(r / 2.0, 0, 1)
        pattern = pattern * (1 - blend) + parallel * blend
        noise = np.random.randn(size, size) * noise_level
        pattern = pattern + noise
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        pattern = cv2.GaussianBlur(pattern, (3, 3), 0.5)
        return pattern

    @staticmethod
    def generate_whorl(size=300, noise_level=0.15):
        """Generate a synthetic Whorl pattern."""
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        pattern = np.sin(r * 6 + theta * 0.5)
        noise = np.random.randn(size, size) * noise_level
        pattern = pattern + noise
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        pattern = cv2.GaussianBlur(pattern, (3, 3), 0.5)
        return pattern

    @staticmethod
    def _fingerprint_mask(size):
        """Create an elliptical fingerprint-shaped mask."""
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        axes = (int(size * 0.42), int(size * 0.46))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        mask[mask > 128] = 255
        mask[mask <= 128] = 0
        return mask


# ==============================================================================
#  CUSTOM WIDGETS
# ==============================================================================

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


# ==============================================================================
#  MAIN WINDOW
# ==============================================================================

class MainWindow(QMainWindow):
    """Main application window for fingerprint classification."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fingerprint Classification System")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 800)

        self.processor = FingerprintProcessor()
        self.processing = False

        self._setup_menu()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()

    # ── Menu ─────────────────────────────────────────────────────────────

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Image...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        demo_menu = file_menu.addMenu("Generate &Demo")
        demo_items = [
            ("Arch Pattern", "arch"),
            ("Left Loop Pattern", "left_loop"),
            ("Right Loop Pattern", "right_loop"),
            ("Whorl Pattern", "whorl"),
        ]
        for name, key in demo_items:
            action = QAction(name, self)
            action.triggered.connect(
                lambda checked, k=key: self.generate_demo(k)
            )
            demo_menu.addAction(action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # ── Toolbar ──────────────────────────────────────────────────────────

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #0f3460;
                border: none;
                padding: 4px;
                spacing: 6px;
            }
        """)
        self.addToolBar(toolbar)

        btn_open = QPushButton("  Open Image")
        btn_open.setStyleSheet(
            self._toolbar_btn_style("#16213e", "#533483", "white", "#533483")
        )
        btn_open.clicked.connect(self.open_image)
        toolbar.addWidget(btn_open)

        btn_classify = QPushButton("  Classify")
        btn_classify.setStyleSheet(
            self._toolbar_btn_style("#e94560", "#ff6b81")
        )
        btn_classify.clicked.connect(self.classify)
        toolbar.addWidget(btn_classify)

        toolbar.addSeparator()

        demo_buttons = [
            ("Demo: Arch", "arch"),
            ("Demo: Loop", "left_loop"),
            ("Demo: Whorl", "whorl"),
        ]
        for label, key in demo_buttons:
            btn = QPushButton(label)
            btn.setStyleSheet(
                self._toolbar_btn_style(
                    "#1a1a3e", "#1e3a5f", "#93c5fd", "#2563eb"
                )
            )
            btn.clicked.connect(
                lambda checked, k=key: self.generate_demo(k)
            )
            toolbar.addWidget(btn)

        toolbar.addSeparator()

        btn_clear = QPushButton("  Clear")
        btn_clear.setStyleSheet(
            self._toolbar_btn_style("#2d2d44", "#3d3d5c", "#aaa", "#444")
        )
        btn_clear.clicked.connect(self.clear_all)
        toolbar.addWidget(btn_clear)

    @staticmethod
    def _toolbar_btn_style(bg, hover_bg, text_color="white",
                           border_color=None):
        bc = border_color or bg
        return (
            f"QPushButton {{"
            f"  background-color: {bg};"
            f"  color: {text_color};"
            f"  border: 1px solid {bc};"
            f"  border-radius: 6px;"
            f"  padding: 6px 14px;"
            f"  font-size: 12px;"
            f"  font-weight: bold;"
            f"}}"
            f"QPushButton:hover {{ background-color: {hover_bg}; }}"
            f"QPushButton:pressed {{ background-color: #e94560; color: white; }}"
        )

    # ── Main UI ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ── Left Panel ──
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        # Original image group
        orig_group = QGroupBox("Original Fingerprint")
        orig_group.setStyleSheet(self._group_style())
        orig_layout = QVBoxLayout(orig_group)
        self.original_display = ImageDisplayLabel(
            "Load or generate a fingerprint"
        )
        orig_layout.addWidget(self.original_display)
        left_panel.addWidget(orig_group, stretch=3)

        # Settings group
        settings_group = QGroupBox("Processing Settings")
        settings_group.setStyleSheet(self._group_style())
        settings_layout = QGridLayout(settings_group)

        lbl_bs = QLabel("Block Size:")
        lbl_bs.setStyleSheet("color: #ddd;")
        settings_layout.addWidget(lbl_bs, 0, 0)
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(8, 32)
        self.block_size_spin.setValue(16)
        self.block_size_spin.setSingleStep(4)
        self.block_size_spin.setStyleSheet(self._spin_style())
        settings_layout.addWidget(self.block_size_spin, 0, 1)

        lbl_ds = QLabel("Demo Size:")
        lbl_ds.setStyleSheet("color: #ddd;")
        settings_layout.addWidget(lbl_ds, 1, 0)
        self.demo_size_spin = QSpinBox()
        self.demo_size_spin.setRange(150, 500)
        self.demo_size_spin.setValue(300)
        self.demo_size_spin.setSingleStep(50)
        self.demo_size_spin.setStyleSheet(self._spin_style())
        settings_layout.addWidget(self.demo_size_spin, 1, 1)

        left_panel.addWidget(settings_group, stretch=1)

        # Classification result
        self.result_widget = ClassificationResultWidget()
        left_panel.addWidget(self.result_widget, stretch=2)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(380)
        main_layout.addWidget(left_widget)

        # ── Right Panel (Tabbed) ──
        right_panel = QVBoxLayout()
        tabs = QTabWidget()
        tabs.setStyleSheet(self._tab_style())

        self.displays = {}
        tab_configs = [
            ("Enhanced", "Enhanced image will appear here"),
            ("Orientation", "Orientation field will appear here"),
            ("Binary", "Binary image will appear here"),
            ("Skeleton", "Thinned skeleton will appear here"),
            ("Singular Points", "Singular points will be marked here"),
            ("Segmentation", "Segmentation mask will appear here"),
        ]
        for name, placeholder in tab_configs:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            display = ImageDisplayLabel(placeholder)
            tab_layout.addWidget(display)
            tabs.addTab(tab, name)
            self.displays[name] = display

        right_panel.addWidget(tabs)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget, stretch=1)

    @staticmethod
    def _group_style():
        return (
            "QGroupBox {"
            "  color: #93c5fd;"
            "  font-size: 13px;"
            "  font-weight: bold;"
            "  border: 1px solid #0f3460;"
            "  border-radius: 8px;"
            "  margin-top: 12px;"
            "  padding-top: 8px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 12px;"
            "  padding: 0 6px;"
            "}"
        )

    @staticmethod
    def _spin_style():
        return (
            "background-color: #1a1a2e; color: white; "
            "border: 1px solid #0f3460; border-radius: 4px; padding: 4px;"
        )

    @staticmethod
    def _tab_style():
        return (
            "QTabWidget::pane {"
            "  border: 1px solid #0f3460;"
            "  border-radius: 8px;"
            "  background-color: #0a0a1a;"
            "}"
            "QTabBar::tab {"
            "  background-color: #16213e;"
            "  color: #aaa;"
            "  border: 1px solid #0f3460;"
            "  border-bottom: none;"
            "  border-top-left-radius: 6px;"
            "  border-top-right-radius: 6px;"
            "  padding: 8px 16px;"
            "  margin-right: 2px;"
            "  font-size: 12px;"
            "}"
            "QTabBar::tab:selected {"
            "  background-color: #0a0a1a;"
            "  color: #e94560;"
            "  font-weight: bold;"
            "}"
            "QTabBar::tab:hover {"
            "  background-color: #1a1a3e;"
            "  color: white;"
            "}"
        )

    # ── Status Bar ───────────────────────────────────────────────────────

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.setStyleSheet(
            "QStatusBar {"
            "  background-color: #0f3460;"
            "  color: #93c5fd;"
            "  font-size: 11px;"
            "  padding: 4px;"
            "}"
        )
        self.statusbar.showMessage(
            "Ready. Open an image or generate a demo."
        )

    # ── Actions ──────────────────────────────────────────────────────────

    def open_image(self):
        """Open a fingerprint image file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Fingerprint Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
            "All Files (*)"
        )
        if not path:
            return
        try:
            self.processor.load_image(path)
            self.original_display.set_image(self.processor.original)
            self.statusbar.showMessage(f"Loaded: {path}")
            self._clear_results()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def generate_demo(self, pattern_type):
        """Generate a synthetic fingerprint pattern."""
        size = self.demo_size_spin.value()
        self.statusbar.showMessage(
            f"Generating {pattern_type} pattern..."
        )
        try:
            if pattern_type == "arch":
                img = FingerprintGenerator.generate_arch(size)
            elif pattern_type == "left_loop":
                img = FingerprintGenerator.generate_loop(size, "left")
            elif pattern_type == "right_loop":
                img = FingerprintGenerator.generate_loop(size, "right")
            elif pattern_type == "whorl":
                img = FingerprintGenerator.generate_whorl(size)
            else:
                img = FingerprintGenerator.generate_arch(size)

            self.processor.load_from_array(img)
            self.original_display.set_image(self.processor.original)
            self.statusbar.showMessage(
                f"Generated {pattern_type} demo ({size}x{size})"
            )
            self._clear_results()
        except Exception as e:
            QMessageBox.warning(self, "Generation Error", str(e))

    def classify(self):
        """Run classification on the loaded image."""
        if self.processor.gray is None:
            QMessageBox.information(
                self, "No Image",
                "Please load an image or generate a demo first."
            )
            return
        if self.processing:
            return
        self.processing = True
        self.statusbar.showMessage("Processing... Please wait.")
        QTimer.singleShot(50, self._do_classify)

    def _do_classify(self):
        """Perform the actual classification."""
        try:
            block_size = self.block_size_spin.value()
            self.processor.process(block_size=block_size)

            self.displays["Enhanced"].set_image(self.processor.enhanced)
            self.displays["Binary"].set_image(self.processor.binary_img)
            self.displays["Skeleton"].set_image(self.processor.thinned)
            self.displays["Segmentation"].set_image(
                self.processor.segmentation_mask
            )
            self.displays["Orientation"].set_image(
                self.processor.get_orientation_overlay()
            )
            self.displays["Singular Points"].set_image(
                self.processor.get_singular_points_overlay()
            )

            self.result_widget.set_result(self.processor)
            self.statusbar.showMessage(
                f"Classification: {self.processor.classification} | "
                f"Confidence: {self.processor.confidence:.0%} | "
                f"Quality: {self.processor.quality_score:.0%}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Processing Error", str(e))
            self.statusbar.showMessage(f"Error: {e}")
        finally:
            self.processing = False

    def clear_all(self):
        """Clear all images and results."""
        self.processor = FingerprintProcessor()
        self.original_display.set_image(None)
        self._clear_results()
        self.result_widget.clear()
        self.statusbar.showMessage("Cleared. Ready for new image.")

    def _clear_results(self):
        """Clear all result displays."""
        for display in self.displays.values():
            display.set_image(None)
        self.result_widget.clear()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Fingerprint Classification",
            "<h2>Fingerprint Classification System</h2>"
            "<p>Version 1.0</p>"
            "<p>Classifies fingerprints into Arch, Loop, and Whorl "
            "categories using orientation field analysis and "
            "Poincar&eacute; index singular point detection.</p>"
            "<h3>Classification Categories</h3>"
            "<ul>"
            "<li><b>Arch</b> \u2014 Ridges enter from one side and "
            "exit the other</li>"
            "<li><b>Tented Arch</b> \u2014 Arch with a sharp upthrust</li>"
            "<li><b>Left Loop</b> \u2014 Loop opening to the left</li>"
            "<li><b>Right Loop</b> \u2014 Loop opening to the right</li>"
            "<li><b>Whorl</b> \u2014 Circular or spiral patterns</li>"
            "</ul>"
            "<p>Built with PySide6, OpenCV, and NumPy.</p>"
        )


# ==============================================================================
#  DARK THEME
# ==============================================================================

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


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()