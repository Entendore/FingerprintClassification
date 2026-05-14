"""
Synthetic Fingerprint Generator
=================================
Generates synthetic fingerprint-like patterns for testing the
Fingerprint Classification System.

Improvements over v1:
- Added Tented Arch pattern generator
- Improved ridge continuity and pattern realism
- Better blending of singular points with surrounding parallel ridges
- Consistent parameter handling and noise application
"""

import numpy as np
import cv2


class FingerprintGenerator:
    """Generates synthetic fingerprint-like patterns for testing."""

    @staticmethod
    def generate_arch(size=300, noise_level=0.15):
        """Generate a synthetic Arch pattern.
        
        Ridges enter from one side, rise slightly in the middle,
        and exit the other side without a sharp upthrust.
        """
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # Gentle rise in the center
        center_rise = 0.8 * np.exp(-Y**2 / 2.0)
        pattern = np.sin(X * 3 + center_rise * np.sin(X * 0.5))
        
        noise = np.random.randn(size, size) * noise_level
        pattern = pattern + noise
        
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        pattern = cv2.GaussianBlur(pattern, (3, 3), 0.5)
        return pattern

    @staticmethod
    def generate_tented_arch(size=300, noise_level=0.15):
        """Generate a synthetic Tented Arch pattern.
        
        Similar to a plain arch but with a sharp upthrust in the center,
        creating a point that resembles a tent.
        """
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        
        # Sharp upthrust in the center
        upthrust = 3.0 * np.exp(-(X**2) / 0.5)
        pattern = np.sin(Y * 5 + upthrust)
        
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
        """Generate a synthetic Loop pattern.
        
        Creates a pattern with a single core and delta. 
        Direction determines whether the loop opens to the left or right.
        """
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        
        # Offset center based on direction
        cx = -0.5 if direction == 'left' else 0.5
        cy = -0.5
        
        # Polar coordinates relative to the core
        theta = np.arctan2(Y - cy, X - cx)
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Circular pattern near core, blending to parallel ridges away
        loop_pattern = np.sin(theta * 3 + r * 5)
        parallel_pattern = np.sin(Y * 5)  # Horizontal parallel ridges
        
        # Blend based on distance from core
        blend = np.clip(r / 2.0, 0, 1)
        pattern = loop_pattern * (1 - blend) + parallel_pattern * blend
        
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
        """Generate a synthetic Whorl pattern.
        
        Creates concentric circular/spiral ridges, yielding
        two cores and two deltas.
        """
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Spiral pattern
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
        """Create an elliptical fingerprint-shaped mask with smooth edges."""
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        axes = (int(size * 0.42), int(size * 0.46))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Smooth the mask edges for a more realistic transition
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        mask[mask > 128] = 255
        mask[mask <= 128] = 0
        
        return mask