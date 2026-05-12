"""
Fingerprint Processing Engine
==============================
Core fingerprint image processing and classification using
orientation field analysis and Poincare index singular point detection.
"""

import numpy as np
import cv2


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