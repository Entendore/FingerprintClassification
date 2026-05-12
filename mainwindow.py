"""
Main Window
============
Main application window for fingerprint classification.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTabWidget, QGroupBox,
    QStatusBar, QSpinBox, QGridLayout,
    QMessageBox, QFrame, QToolBar
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QAction

from processor import FingerprintProcessor
from generator import FingerprintGenerator
from widgets import ImageDisplayLabel, ClassificationResultWidget


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