"""
Fingerprint Processing Engine
==============================
Core fingerprint image processing and classification using
orientation field analysis and Poincaré index singular point detection.

Improvements over v1:
- Orientation-selective Gabor filter bank (per-block) replaces single-dominant
- Padded orientation smoothing avoids zero-border artefacts
- Tighter Poincaré-index thresholds reduce false detections
- Bounds-checking in all overlay renderers
- Processing-time instrumentation
- Image metadata tracking
- Morphological cleanup on segmentation mask
- Robust ridge-frequency estimator (row + column median)
- CLAHE stage preserved separately for display
- Ridge-density quality metric
"""

import time
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
        self.clahe_enhanced = None
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
        self.processing_time = 0.0
        self.image_info = {}
        self.file_path = None

    # ── Loading ──────────────────────────────────────────────────────────

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
        self.file_path = path
        self._update_image_info()
        self._reset_results()
        return True

    def load_from_array(self, gray_img, label="synthetic"):
        """Load a grayscale numpy array."""
        self.gray = gray_img.copy()
        self.original = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        self.file_path = label
        self._update_image_info()
        self._reset_results()
        return True

    def _update_image_info(self):
        """Record basic image metadata."""
        if self.gray is not None:
            h, w = self.gray.shape
            self.image_info = {
                "width": w,
                "height": h,
                "size_str": f"{w} × {h}",
                "pixels": w * h,
                "mean_intensity": float(np.mean(self.gray)),
                "std_intensity": float(np.std(self.gray)),
                "min_intensity": int(np.min(self.gray)),
                "max_intensity": int(np.max(self.gray)),
                "source": self.file_path or "unknown",
            }
        else:
            self.image_info = {}

    def _reset_results(self):
        """Reset all processing results."""
        self.enhanced = None
        self.clahe_enhanced = None
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
        self.processing_time = 0.0

    # ── Pipeline ─────────────────────────────────────────────────────────

    def process(self, block_size=16):
        """Run the full processing pipeline."""
        if self.gray is None:
            raise ValueError("No image loaded")
        self.block_size = max(8, block_size)
        self._reset_results()

        t0 = time.perf_counter()
        self._preprocess()
        self._segment()
        self._compute_orientation()
        self._enhance_ridges()
        self._binarize()
        self._thin()
        self._detect_singular_points()
        self._classify()
        self._assess_quality()
        self.processing_time = time.perf_counter() - t0
        self.details["processing_time"] = self.processing_time

    # ── Steps ────────────────────────────────────────────────────────────

    def _preprocess(self):
        """CLAHE contrast enhancement + normalisation."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_enhanced = clahe.apply(self.gray)
        self.clahe_enhanced = cv2.normalize(
            self.clahe_enhanced, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        # Working copy – will be overwritten after Gabor step
        self.enhanced = self.clahe_enhanced.copy()

    def _segment(self):
        """Block-variance foreground segmentation with morphological cleanup."""
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

        # Morphological cleanup – remove small holes / specks
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.block_size, self.block_size)
        )
        self.segmentation_mask = cv2.morphologyEx(
            self.segmentation_mask, cv2.MORPH_CLOSE, kern, iterations=1
        )
        self.segmentation_mask = cv2.morphologyEx(
            self.segmentation_mask, cv2.MORPH_OPEN, kern, iterations=1
        )

    def _compute_orientation(self):
        """Gradient-based ridge orientation estimation (block-wise)."""
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
                y1, y2 = i * bh, (i + 1) * bh
                x1, x2 = j * bw, (j + 1) * bw
                block_gx = gx[y1:y2, x1:x2]
                block_gy = gy[y1:y2, x1:x2]

                gxx = np.sum(block_gx ** 2)
                gyy = np.sum(block_gy ** 2)
                gxy = np.sum(block_gx * block_gy)
                denom = gxx + gyy

                if denom < 1e-10:
                    self.orientation_map[i, j] = 0.0
                    self.orientation_strength[i, j] = 0.0
                    continue

                theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / denom
                self.orientation_map[i, j] = theta
                self.orientation_strength[i, j] = coherence

        self._smooth_orientation()

    def _smooth_orientation(self):
        """Vector-averaged low-pass filter with edge padding (no zero borders)."""
        h, w = self.orientation_map.shape
        sin2 = np.sin(2 * self.orientation_map)
        cos2 = np.cos(2 * self.orientation_map)
        ksz = 5  # wider kernel → smoother field
        pad = ksz // 2

        # Edge-pad so border blocks get a valid estimate
        sin2_pad = np.pad(sin2, pad, mode="edge")
        cos2_pad = np.pad(cos2, pad, mode="edge")

        smoothed = np.zeros_like(self.orientation_map)
        for i in range(h):
            for j in range(w):
                ip, jp = i + pad, j + pad
                avg_sin = np.mean(
                    sin2_pad[ip - pad:ip + pad + 1, jp - pad:jp + pad + 1]
                )
                avg_cos = np.mean(
                    cos2_pad[ip - pad:ip + pad + 1, jp - pad:jp + pad + 1]
                )
                smoothed[i, j] = 0.5 * np.arctan2(avg_sin, avg_cos)

        self.orientation_map = smoothed

    def _enhance_ridges(self):
        """Orientation-selective Gabor filter bank (per-block selection)."""
        if self.orientation_map is None:
            return

        freq = self._estimate_ridge_frequency()
        ksize = max(21, self.block_size * 2 + 1)
        sigma = 4.0
        n_ori = 12
        orientations = np.arange(n_ori) * np.pi / n_ori

        # --- filter bank ---
        filtered = []
        for theta in orientations:
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, freq, 0.5, 0, ktype=cv2.CV_32F
            )
            resp = cv2.filter2D(self.clahe_enhanced, cv2.CV_32F, kern)
            filtered.append(np.abs(resp))

        # --- per-block best-response selection ---
        h, w = self.clahe_enhanced.shape
        rows, cols = self.orientation_map.shape
        result = np.zeros((h, w), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                if self.orientation_strength[i, j] < 0.05:
                    continue

                y1 = i * self.block_size
                y2 = min((i + 1) * self.block_size, h)
                x1 = j * self.block_size
                x2 = min((j + 1) * self.block_size, w)

                theta = self.orientation_map[i, j] % np.pi
                if theta < 0:
                    theta += np.pi
                idx = int(round(theta / np.pi * n_ori)) % n_ori

                # Pick strongest among current + adjacent orientations
                candidates = [(idx - 1) % n_ori, idx, (idx + 1) % n_ori]
                best_idx = max(
                    candidates,
                    key=lambda k: float(np.mean(filtered[k][y1:y2, x1:x2])),
                )
                result[y1:y2, x1:x2] = filtered[best_idx][y1:y2, x1:x2]

        if self.segmentation_mask is not None:
            result *= self.segmentation_mask.astype(np.float32) / 255.0

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        self.enhanced = np.clip(result, 0, 255).astype(np.uint8)

    def _estimate_ridge_frequency(self):
        """Robust average ridge frequency using row + column autocorrelation."""
        h, w = self.enhanced.shape
        cy, cx = h // 2, w // 2
        region_size = min(h, w) // 3
        if region_size < 4:
            return 1.0 / 10.0

        y1 = max(0, cy - region_size // 2)
        y2 = min(h, cy + region_size // 2)
        x1 = max(0, cx - region_size // 2)
        x2 = min(w, cx + region_size // 2)
        region = self.enhanced[y1:y2, x1:x2]
        if region.size == 0:
            return 1.0 / 10.0

        estimates = []
        for signal in [
            np.mean(region, axis=0).astype(np.float64),
            np.mean(region, axis=1).astype(np.float64),
        ]:
            n = len(signal)
            if n < 4:
                continue
            sig = signal - np.mean(signal)
            ac = np.correlate(sig, sig, mode="full")[n - 1:]
            for k in range(3, len(ac) // 2):
                if ac[k] > ac[k - 1] and ac[k] > ac[k + 1] and ac[k] > 0.1 * ac[0]:
                    estimates.append(1.0 / max(k, 3))
                    break

        return float(np.median(estimates)) if estimates else 1.0 / 10.0

    def _binarize(self):
        """Adaptive threshold + morphological cleanup."""
        self.binary_img = cv2.adaptiveThreshold(
            self.enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5,
        )
        if self.segmentation_mask is not None:
            self.binary_img = cv2.bitwise_and(
                self.binary_img, self.segmentation_mask
            )
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.binary_img = cv2.morphologyEx(
            self.binary_img, cv2.MORPH_CLOSE, kern, iterations=1
        )
        self.binary_img = cv2.morphologyEx(
            self.binary_img, cv2.MORPH_OPEN, kern, iterations=1
        )

    def _thin(self):
        """Morphological skeleton extraction."""
        if self.binary_img is None:
            return
        binary = (self.binary_img // 255).astype(np.uint8)
        skeleton = np.zeros_like(binary)
        kern = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = binary.copy()
        for _ in range(200):
            if np.count_nonzero(img) == 0:
                break
            eroded = cv2.erode(img, kern)
            opened = cv2.dilate(eroded, kern)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(img, opened))
            img = eroded
        self.thinned = skeleton * 255

    # ── Singular point detection ─────────────────────────────────────────

    def _detect_singular_points(self):
        """Poincaré-index based core / delta detection."""
        if self.orientation_map is None:
            return
        rows, cols = self.orientation_map.shape
        self.cores = []
        self.deltas = []

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if self.orientation_strength[i, j] < 0.20:
                    continue
                by = i * self.block_size + self.block_size // 2
                bx = j * self.block_size + self.block_size // 2
                if self.segmentation_mask is not None:
                    if (
                        0 <= by < self.segmentation_mask.shape[0]
                        and 0 <= bx < self.segmentation_mask.shape[1]
                    ):
                        if self.segmentation_mask[by, bx] == 0:
                            continue

                pi_val = self._compute_poincare_index(i, j)
                # Tighter thresholds: π/4 instead of π/3
                if abs(pi_val - np.pi) < np.pi / 4:
                    self.cores.append((j, i))
                elif abs(pi_val + np.pi) < np.pi / 4:
                    self.deltas.append((j, i))

        self.cores = self._merge_nearby_points(self.cores, min_dist=3)
        self.deltas = self._merge_nearby_points(self.deltas, min_dist=3)

    def _compute_poincare_index(self, i, j):
        """Sum angle differences around 8-connected neighbourhood."""
        nbrs = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
            (i, j + 1),
            (i + 1, j + 1), (i + 1, j), (i + 1, j - 1),
            (i, j - 1),
        ]
        total = 0.0
        n = len(nbrs)
        for k in range(n):
            o1 = self.orientation_map[nbrs[k]]
            o2 = self.orientation_map[nbrs[(k + 1) % n]]
            diff = o2 - o1
            while diff > np.pi / 2:
                diff -= np.pi
            while diff <= -np.pi / 2:
                diff += np.pi
            total += diff
        return total

    @staticmethod
    def _merge_nearby_points(points, min_dist=3):
        """Non-maximum suppression: cluster and average nearby points."""
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
                if np.hypot(p1[0] - p2[0], p1[1] - p2[1]) < min_dist:
                    cluster.append(p2)
                    used[j] = True
            merged.append(
                (int(np.mean([p[0] for p in cluster])),
                 int(np.mean([p[1] for p in cluster])))
            )
        return merged

    # ── Classification ───────────────────────────────────────────────────

    def _classify(self):
        """Rule-based classification from singular-point counts."""
        n_cores = len(self.cores)
        n_deltas = len(self.deltas)
        self.details = {
            "cores": n_cores,
            "deltas": n_deltas,
            "core_positions": list(self.cores),
            "delta_positions": list(self.deltas),
        }

        if n_cores == 0 and n_deltas == 0:
            self.classification = self.ARCH
            self.confidence = 0.85
        elif n_cores == 1 and n_deltas == 0:
            self.classification = self.TENTED_ARCH
            self.confidence = 0.70
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

        # Weight confidence by orientation coherence
        if self.orientation_strength is not None:
            valid = self.orientation_strength > 0.1
            avg_coh = (
                float(np.mean(self.orientation_strength[valid]))
                if np.any(valid) else 0.0
            )
            self.confidence *= 0.5 + 0.5 * min(avg_coh, 1.0)
            self.details["avg_coherence"] = avg_coh

        # Penalise very low or very high foreground coverage
        if self.segmentation_mask is not None:
            cov = float(np.count_nonzero(self.segmentation_mask)) / float(
                self.segmentation_mask.size
            )
            if cov < 0.15:
                self.confidence *= 0.7
            elif cov < 0.30:
                self.confidence *= 0.85
            self.details["coverage"] = cov

        self.confidence = min(self.confidence, 0.99)

    # ── Quality ──────────────────────────────────────────────────────────

    def _assess_quality(self):
        """Multi-factor image quality score."""
        if self.gray is None:
            self.quality_score = 0.0
            return

        contrast = float(np.std(self.gray)) / 128.0

        coverage = 0.5
        if self.segmentation_mask is not None:
            coverage = float(np.count_nonzero(self.segmentation_mask)) / float(
                self.segmentation_mask.size
            )

        coherence = 0.5
        if self.orientation_strength is not None:
            valid = self.orientation_strength > 0.1
            if np.any(valid):
                coherence = float(np.mean(self.orientation_strength[valid]))

        ridge_density = 0.5
        if self.thinned is not None and self.segmentation_mask is not None:
            fg = float(np.count_nonzero(self.segmentation_mask))
            if fg > 0:
                ridge_density = min(float(np.count_nonzero(self.thinned)) / fg, 1.0)

        self.quality_score = float(np.clip(
            0.25 * contrast
            + 0.20 * coverage
            + 0.35 * min(coherence, 1.0)
            + 0.20 * ridge_density,
            0.0, 1.0,
        ))
        self.details["quality"] = self.quality_score
        self.details["contrast"] = contrast
        self.details["coverage"] = coverage
        self.details["ridge_density"] = ridge_density

    # ── Overlays ─────────────────────────────────────────────────────────

    def get_orientation_overlay(self):
        """Render orientation field lines on the original image."""
        if self.orientation_map is None or self.original is None:
            return None

        overlay = self.original.copy()
        rows, cols = self.orientation_map.shape
        bh, bw = self.block_size, self.block_size
        half = self.block_size // 3
        h, w = overlay.shape[:2]

        for i in range(rows):
            for j in range(cols):
                if self.orientation_strength[i, j] < 0.1:
                    continue
                cx = j * bw + bw // 2
                cy = i * bh + bh // 2
                if cx >= w or cy >= h:
                    continue
                if self.segmentation_mask is not None:
                    if (
                        0 <= cy < self.segmentation_mask.shape[0]
                        and 0 <= cx < self.segmentation_mask.shape[1]
                        and self.segmentation_mask[cy, cx] == 0
                    ):
                        continue

                theta = self.orientation_map[i, j]
                dx = int(half * np.cos(theta))
                dy = int(half * np.sin(theta))
                s = min(self.orientation_strength[i, j], 1.0)
                color = (int(255 * (1 - s)), int(200 * s), int(255 * s))
                p1 = (max(0, cx - dx), max(0, cy - dy))
                p2 = (min(w - 1, cx + dx), min(h - 1, cy + dy))
                cv2.line(overlay, p1, p2, color, 1, cv2.LINE_AA)

        return overlay

    def get_singular_points_overlay(self):
        """Mark detected core and delta points on the original image."""
        if self.original is None:
            return None

        overlay = self.original.copy()
        bh, bw = self.block_size, self.block_size
        radius = max(bh, bw) // 2 + 5
        h, w = overlay.shape[:2]

        for cx, cy in self.cores:
            px, py = cx * bw + bw // 2, cy * bh + bh // 2
            if not (0 <= px < w and 0 <= py < h):
                continue
            cv2.circle(overlay, (px, py), radius, (0, 0, 255), 2)
            cv2.putText(
                overlay, "Core",
                (max(0, px - 15), max(12, py - radius - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )

        for cx, cy in self.deltas:
            px, py = cx * bw + bw // 2, cy * bh + bh // 2
            if not (0 <= px < w and 0 <= py < h):
                continue
            pts = np.array(
                [[px, py - radius],
                 [px - radius, py + radius],
                 [px + radius, py + radius]],
                dtype=np.int32,
            )
            cv2.polylines(overlay, [pts], True, (255, 150, 0), 2)
            cv2.putText(
                overlay, "Delta",
                (max(0, px - 15), min(h - 4, py + radius + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1,
            )
        return overlay