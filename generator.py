"""
Synthetic Fingerprint Generator
=================================
Generates synthetic fingerprint-like patterns for testing.
"""

import numpy as np
import cv2


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