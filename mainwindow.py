"""
Main Window
============
Main application window for fingerprint classification.

New features over v1:
- Drag & drop image loading
- Save current tab / save all processed images
- Recent-files menu (persisted via QSettings)
- Histogram tab
- Processing-time display in status bar
- Image info in status bar
- Window title tracks loaded file
- Export classification report as HTML
- Keyboard shortcuts for all major actions
- Zoom reset action
"""

import os
import base64
import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTabWidget, QGroupBox,
    QStatusBar, QSpinBox, QGridLayout,
    QMessageBox, QToolBar,
)
from PySide6.QtCore import Qt, QSize, QTimer, QSettings, QMimeData
from PySide6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent

import cv2
import numpy as np

from processor import FingerprintProcessor
from generator import FingerprintGenerator
from widgets import ImageDisplayLabel, ClassificationResultWidget, HistogramWidget


class MainWindow(QMainWindow):
    """Main application window for fingerprint classification."""

    MAX_RECENT = 5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fingerprint Classification System")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self.processor = FingerprintProcessor()
        self.processing = False
        self.settings = QSettings("FPClassifier", "FingerprintApp")

        self._setup_menu()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()

    # ── Drag & Drop ──────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
                ):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
            ):
                self._load_image_file(path)
                break

    # ── Menu ─────────────────────────────────────────────────────────────

    def _setup_menu(self):
        menubar = self.menuBar()

        # --- File ---
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._refresh_recent_menu()

        file_menu.addSeparator()

        save_action = QAction("&Save Current Tab...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_current_tab)
        file_menu.addAction(save_action)

        save_all_action = QAction("Save &All Results...", self)
        save_all_action.triggered.connect(self.save_all_results)
        file_menu.addAction(save_all_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Report (HTML)...", self)
        export_action.triggered.connect(self.export_report)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        demo_menu = file_menu.addMenu("Generate &Demo")
        for name, key in [
            ("Arch Pattern", "arch"),
            ("Tented Arch Pattern", "tented_arch"),
            ("Left Loop Pattern", "left_loop"),
            ("Right Loop Pattern", "right_loop"),
            ("Whorl Pattern", "whorl"),
        ]:
            act = QAction(name, self)
            act.triggered.connect(lambda checked, k=key: self.generate_demo(k))
            demo_menu.addAction(act)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- View ---
        view_menu = menubar.addMenu("&View")
        fit_action = QAction("Reset &Zoom", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._reset_zoom)
        view_menu.addAction(fit_action)

        # --- Help ---
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # ── Recent files ─────────────────────────────────────────────────────

    def _recent_files(self):
        return self.settings.value("recent_files", []) or []

    def _add_recent(self, path):
        recent = [p for p in self._recent_files() if p != path]
        recent.insert(0, path)
        self.settings.setValue("recent_files", recent[: self.MAX_RECENT])
        self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        self._recent_menu.clear()
        for path in self._recent_files():
            act = QAction(os.path.basename(path), self)
            act.setToolTip(path)
            act.triggered.connect(lambda checked, p=path: self._load_image_file(p))
            self._recent_menu.addAction(act)
        if not self._recent_files():
            act = QAction("(empty)", self)
            act.setEnabled(False)
            self._recent_menu.addAction(act)

    # ── Toolbar ──────────────────────────────────────────────────────────

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet(
            "QToolBar {"
            "  background-color: #0f3460; border: none;"
            "  padding: 4px; spacing: 6px;"
            "}"
        )
        self.addToolBar(toolbar)

        for label, slot, bg, hover in [
            ("  Open Image", self.open_image, "#16213e", "#533483"),
            ("  Classify", self.classify, "#e94560", "#ff6b81"),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(self._toolbar_btn_style(bg, hover))
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)

        toolbar.addSeparator()

        for label, key in [
            ("Demo: Arch", "arch"),
            ("Demo: Loop", "left_loop"),
            ("Demo: Whorl", "whorl"),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(
                self._toolbar_btn_style("#1a1a3e", "#1e3a5f", "#93c5fd", "#2563eb")
            )
            btn.clicked.connect(lambda checked, k=key: self.generate_demo(k))
            toolbar.addWidget(btn)

        toolbar.addSeparator()

        btn_clear = QPushButton("  Clear")
        btn_clear.setStyleSheet(
            self._toolbar_btn_style("#2d2d44", "#3d3d5c", "#aaa", "#444")
        )
        btn_clear.clicked.connect(self.clear_all)
        toolbar.addWidget(btn_clear)

    @staticmethod
    def _toolbar_btn_style(bg, hover_bg, text_color="white", border_color=None):
        bc = border_color or bg
        return (
            f"QPushButton {{"
            f"  background-color: {bg}; color: {text_color};"
            f"  border: 1px solid {bc}; border-radius: 6px;"
            f"  padding: 6px 14px; font-size: 12px; font-weight: bold;"
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

        orig_group = QGroupBox("Original Fingerprint")
        orig_group.setStyleSheet(self._group_style())
        orig_layout = QVBoxLayout(orig_group)
        self.original_display = ImageDisplayLabel(
            "Load or generate a fingerprint"
        )
        orig_layout.addWidget(self.original_display)
        left_panel.addWidget(orig_group, stretch=3)

        settings_group = QGroupBox("Processing Settings")
        settings_group.setStyleSheet(self._group_style())
        sl = QGridLayout(settings_group)

        for row, (label, lo, hi, val, step) in enumerate([
            ("Block Size:", 8, 32, 16, 4),
            ("Demo Size:", 150, 500, 300, 50),
        ]):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #ddd;")
            sl.addWidget(lbl, row, 0)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(val)
            spin.setSingleStep(step)
            spin.setStyleSheet(self._spin_style())
            sl.addWidget(spin, row, 1)
            if row == 0:
                self.block_size_spin = spin
            else:
                self.demo_size_spin = spin

        left_panel.addWidget(settings_group, stretch=1)

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
            ("CLAHE", "CLAHE pre-enhanced image"),
            ("Orientation", "Orientation field will appear here"),
            ("Binary", "Binary image will appear here"),
            ("Skeleton", "Thinned skeleton will appear here"),
            ("Singular Points", "Singular points will be marked here"),
            ("Segmentation", "Segmentation mask will appear here"),
            ("Histogram", ""),
        ]
        for name, placeholder in tab_configs:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            if name == "Histogram":
                self.histogram_widget = HistogramWidget()
                tab_layout.addWidget(self.histogram_widget)
            else:
                display = ImageDisplayLabel(placeholder)
                tab_layout.addWidget(display)
                self.displays[name] = display
            tabs.addTab(tab, name)

        self._tabs_widget = tabs
        right_panel.addWidget(tabs)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget, stretch=1)

    @staticmethod
    def _group_style():
        return (
            "QGroupBox {"
            "  color: #93c5fd; font-size: 13px; font-weight: bold;"
            "  border: 1px solid #0f3460; border-radius: 8px;"
            "  margin-top: 12px; padding-top: 8px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin; left: 12px; padding: 0 6px;"
            "}"
        )

    @staticmethod
    def _spin_style():
        return (
            "background-color: #1a1a2e; color: white;"
            "border: 1px solid #0f3460; border-radius: 4px; padding: 4px;"
        )

    @staticmethod
    def _tab_style():
        return (
            "QTabWidget::pane {"
            "  border: 1px solid #0f3460; border-radius: 8px;"
            "  background-color: #0a0a1a;"
            "}"
            "QTabBar::tab {"
            "  background-color: #16213e; color: #aaa;"
            "  border: 1px solid #0f3460; border-bottom: none;"
            "  border-top-left-radius: 6px; border-top-right-radius: 6px;"
            "  padding: 8px 12px; margin-right: 2px; font-size: 11px;"
            "}"
            "QTabBar::tab:selected {"
            "  background-color: #0a0a1a; color: #e94560; font-weight: bold;"
            "}"
            "QTabBar::tab:hover { background-color: #1a1a3e; color: white; }"
        )

    # ── Status Bar ───────────────────────────────────────────────────────

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.setStyleSheet(
            "QStatusBar {"
            "  background-color: #0f3460; color: #93c5fd;"
            "  font-size: 11px; padding: 4px;"
            "}"
        )
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #bbb; padding: 0 8px;")
        self.statusbar.addPermanentWidget(self._info_label)
        self.statusbar.showMessage("Ready. Open an image or generate a demo.")

    def _update_info_label(self):
        info = self.processor.image_info
        if not info:
            self._info_label.setText("")
            return
        self._info_label.setText(
            f"{info.get('size_str', '')} | "
            f"Mean: {info.get('mean_intensity', 0):.1f} | "
            f"Std: {info.get('std_intensity', 0):.1f}"
        )

    # ── Actions ──────────────────────────────────────────────────────────

    def open_image(self):
        """Open a fingerprint image file via dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Fingerprint Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
            "All Files (*)",
        )
        if path:
            self._load_image_file(path)

    def _load_image_file(self, path):
        """Internal: load an image from *path*."""
        try:
            self.processor.load_image(path)
            self.original_display.set_image(self.processor.original)
            self._add_recent(path)
            fname = os.path.basename(path)
            self.setWindowTitle(f"Fingerprint Classification — {fname}")
            self.statusbar.showMessage(f"Loaded: {path}")
            self._update_info_label()
            self._clear_results()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def generate_demo(self, pattern_type):
        """Generate a synthetic fingerprint pattern."""
        size = self.demo_size_spin.value()
        self.statusbar.showMessage(f"Generating {pattern_type} pattern...")
        try:
            gen_map = {
                "arch": lambda: FingerprintGenerator.generate_arch(size),
                "tented_arch": lambda: FingerprintGenerator.generate_tented_arch(size),
                "left_loop": lambda: FingerprintGenerator.generate_loop(size, "left"),
                "right_loop": lambda: FingerprintGenerator.generate_loop(size, "right"),
                "whorl": lambda: FingerprintGenerator.generate_whorl(size),
            }
            img = gen_map.get(pattern_type, gen_map["arch"])()

            self.processor.load_from_array(img, label=f"demo_{pattern_type}")
            self.original_display.set_image(self.processor.original)
            self.setWindowTitle(
                f"Fingerprint Classification — Demo: {pattern_type}"
            )
            self.statusbar.showMessage(
                f"Generated {pattern_type} demo ({size}×{size})"
            )
            self._update_info_label()
            self._clear_results()
        except Exception as e:
            QMessageBox.warning(self, "Generation Error", str(e))
            self.statusbar.showMessage(f"Error generating demo: {e}")

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
            self.displays["CLAHE"].set_image(self.processor.clahe_enhanced)
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

            self.histogram_widget.set_image(self.processor.gray)

            self.result_widget.set_result(self.processor)
            
            pt = self.processor.processing_time
            self.statusbar.showMessage(
                f"Classification: {self.processor.classification} | "
                f"Confidence: {self.processor.confidence:.0%} | "
                f"Quality: {self.processor.quality_score:.0%} | "
                f"Time: {pt:.3f}s"
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
        self.histogram_widget.set_image(None)
        self.setWindowTitle("Fingerprint Classification System")
        self._info_label.setText("")
        self.statusbar.showMessage("Cleared. Ready for new image.")

    def _clear_results(self):
        """Clear all result displays."""
        for display in self.displays.values():
            display.set_image(None)
        self.result_widget.clear()

    def _reset_zoom(self):
        """Reset zoom on all image displays."""
        self.original_display._zoom = 1.0
        self.original_display._update_display()
        for display in self.displays.values():
            display._zoom = 1.0
            display._update_display()

    # ── Save / Export ────────────────────────────────────────────────────

    def _get_tab_image(self, tab_name):
        """Retrieve the cv2 image associated with a tab name."""
        p = self.processor
        if tab_name == "Enhanced":
            return p.enhanced
        if tab_name == "CLAHE":
            return p.clahe_enhanced
        if tab_name == "Orientation":
            return p.get_orientation_overlay()
        if tab_name == "Binary":
            return p.binary_img
        if tab_name == "Skeleton":
            return p.thinned
        if tab_name == "Singular Points":
            return p.get_singular_points_overlay()
        if tab_name == "Segmentation":
            return p.segmentation_mask
        return None

    def save_current_tab(self):
        """Save the image from the currently selected tab to a file."""
        idx = self._tabs_widget.currentIndex()
        tab_name = self._tabs_widget.tabText(idx)

        if tab_name == "Histogram":
            pixmap = self.histogram_widget.grab()
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Histogram", "histogram.png",
                "PNG Image (*.png);;All Files (*)"
            )
            if path:
                pixmap.save(path)
                self.statusbar.showMessage(f"Saved histogram to {path}")
            return

        img = self._get_tab_image(tab_name)
        if img is None:
            QMessageBox.information(
                self, "No Data",
                f"No processed image available for '{tab_name}'. "
                "Please classify first."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {tab_name}", f"{tab_name.lower().replace(' ', '_')}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp);;All Files (*)"
        )
        if path:
            cv2.imwrite(path, img)
            self.statusbar.showMessage(f"Saved {tab_name} to {path}")

    def save_all_results(self):
        """Save all processed images into a selected directory."""
        if self.processor.gray is None:
            QMessageBox.information(self, "No Data", "Please classify first.")
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results"
        )
        if not dir_path:
            return

        saved = []
        for name in [
            "Enhanced", "CLAHE", "Orientation", "Binary", 
            "Skeleton", "Singular Points", "Segmentation"
        ]:
            img = self._get_tab_image(name)
            if img is not None:
                fname = f"{name.lower().replace(' ', '_')}.png"
                fpath = os.path.join(dir_path, fname)
                cv2.imwrite(fpath, img)
                saved.append(name)

        # Save histogram via grab
        pixmap = self.histogram_widget.grab()
        pixmap.save(os.path.join(dir_path, "histogram.png"))
        saved.append("Histogram")

        self.statusbar.showMessage(
            f"Saved {len(saved)} images to {dir_path}"
        )

    def export_report(self):
        """Export an HTML classification report with embedded images."""
        if self.processor.gray is None:
            QMessageBox.information(self, "No Data", "Please classify first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "report.html",
            "HTML Files (*.html);;All Files (*)"
        )
        if not path:
            return

        def img_to_base64(cv_img):
            if cv_img is None:
                return ""
            _, buffer = cv2.imencode('.png', cv_img)
            return base64.b64encode(buffer).decode('utf-8')

        p = self.processor
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def row(label, val):
            return f"<tr><td style='padding:4px 12px; font-weight:bold;'>{label}</td><td style='padding:4px 12px;'>{val}</td></tr>"

        images_html = ""
        for name in ["Original", "Enhanced", "Binary", "Singular Points"]:
            if name == "Original":
                img = p.original
            else:
                img = self._get_tab_image(name)
            b64 = img_to_base64(img)
            if b64:
                images_html += (
                    f"<div style='display:inline-block; margin:10px; text-align:center;'>"
                    f"<h4>{name}</h4>"
                    f"<img src='data:image/png;base64,{b64}' width='300' />"
                    f"</div>"
                )

        html = f"""<!DOCTYPE html>
<html>
<head><title>Fingerprint Classification Report</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
    h1, h2 {{ color: #e94560; }}
    table {{ border-collapse: collapse; margin: 10px 0; }}
    td {{ border: 1px solid #333; }}
</style>
</head>
<body>
    <h1>Fingerprint Classification Report</h1>
    <p>Generated: {now}</p>
    
    <h2>Results</h2>
    <table>
        {row("Classification", f"<span style='color:#4ade80; font-size:1.2em; font-weight:bold;'>{p.classification}</span>")}
        {row("Confidence", f"{p.confidence:.1%}")}
        {row("Quality", f"{p.quality_score:.1%}")}
        {row("Cores Detected", len(p.cores))}
        {row("Deltas Detected", len(p.deltas))}
        {row("Processing Time", f"{p.processing_time:.3f}s")}
    </table>

    <h2>Details</h2>
    <table>
        {row("Source", p.image_info.get('source', 'N/A'))}
        {row("Dimensions", p.image_info.get('size_str', 'N/A'))}
        {row("Mean Intensity", f"{p.image_info.get('mean_intensity', 0):.1f}")}
        {row("Std Deviation", f"{p.image_info.get('std_intensity', 0):.1f}")}
        {row("Avg Orientation Coherence", f"{p.details.get('avg_coherence', 0):.3f}")}
        {row("Foreground Coverage", f"{p.details.get('coverage', 0):.1%}")}
        {row("Ridge Density", f"{p.details.get('ridge_density', 0):.3f}")}
    </table>

    <h2>Images</h2>
    <div>{images_html}</div>
</body>
</html>"""

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            self.statusbar.showMessage(f"Report exported to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Fingerprint Classification",
            "<h2>Fingerprint Classification System</h2>"
            "<p>Version 2.0</p>"
            "<p>Classifies fingerprints into Arch, Loop, and Whorl "
            "categories using orientation field analysis and "
            "Poincar&eacute; index singular point detection.</p>"
            "<h3>Classification Categories</h3>"
            "<ul>"
            "<li><b>Arch</b> &mdash; Ridges enter from one side and "
            "exit the other</li>"
            "<li><b>Tented Arch</b> &mdash; Arch with a sharp upthrust</li>"
            "<li><b>Left Loop</b> &mdash; Loop opening to the left</li>"
            "<li><b>Right Loop</b> &mdash; Loop opening to the right</li>"
            "<li><b>Whorl</b> &mdash; Circular or spiral patterns</li>"
            "</ul>"
            "<h3>Features</h3>"
            "<ul>"
            "<li>Drag &amp; Drop support</li>"
            "<li>Zoomable image views (Mouse wheel / Double click)</li>"
            "<li>HTML Report export with embedded images</li>"
            "<li>Recent files menu</li>"
            "<li>Live histogram &amp; quality metrics</li>"
            "</ul>"
            "<p>Built with PySide6, OpenCV, and NumPy.</p>"
        )