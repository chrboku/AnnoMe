import sys
import os
from collections import defaultdict, OrderedDict
import io
import json
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QScrollArea,
    QCheckBox,
    QTextEdit,
    QSplitter,
    QMessageBox,
    QProgressDialog,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QSlider,
    QMenu,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import math

import rdkit.Chem
from rdkit.Chem import Descriptors
import natsort
import tempfile
from pprint import pprint

from . import Filters


class CollapsibleSection(QWidget):
    """A collapsible section widget with a toggle button."""

    toggled = pyqtSignal(object)  # Signal emitted when section is toggled

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(f"▼ {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.toggle)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_area.setLayout(self.content_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def toggle(self):
        checked = self.toggle_button.isChecked()
        self.content_area.setVisible(checked)
        text = self.toggle_button.text()
        self.toggle_button.setText(text.replace("▼" if checked else "▶", "▶" if not checked else "▼"))
        self.toggled.emit(self)

    def collapse(self):
        """Collapse this section without emitting signal."""
        if self.toggle_button.isChecked():
            self.toggle_button.setChecked(False)
            self.content_area.setVisible(False)
            text = self.toggle_button.text()
            self.toggle_button.setText(text.replace("▼", "▶"))

    def expand(self):
        """Expand this section without emitting signal."""
        if not self.toggle_button.isChecked():
            self.toggle_button.setChecked(True)
            self.content_area.setVisible(True)
            text = self.toggle_button.text()
            self.toggle_button.setText(text.replace("▶", "▼"))

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class StructureViewerWindow(QMainWindow):
    """A separate window to display matched and non-matched structures."""

    def __init__(self, filter_name, matched_smiles, non_matched_smiles, parent=None):
        super().__init__(parent)
        self.filter_name = filter_name
        self.matched_smiles = matched_smiles
        self.non_matched_smiles = non_matched_smiles
        self.temp_dir = tempfile.TemporaryDirectory()
        self.zoom_level = 100  # Default zoom level

        self.setWindowTitle(f"Structure Viewer - {filter_name}")
        self.setGeometry(150, 150, 1200, 800)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)  # Connect to slot
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)

        zoom_reset_btn = QPushButton("Reset Zoom")
        zoom_reset_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(zoom_reset_btn)

        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

        # Create splitter for matched and non-matched
        splitter = QSplitter(Qt.Horizontal)

        # Matched structures
        matched_widget = QWidget()
        matched_layout = QVBoxLayout()

        matched_header = QLabel(f"Matched Structures ({len(self.matched_smiles)} unique SMILES)")
        matched_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        matched_layout.addWidget(matched_header)

        self.matched_scroll = QScrollArea()
        self.matched_scroll.setWidgetResizable(True)
        self.matched_container = QLabel()
        self.matched_container.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.matched_scroll.setWidget(self.matched_container)
        matched_layout.addWidget(self.matched_scroll)

        matched_widget.setLayout(matched_layout)
        splitter.addWidget(matched_widget)

        # Non-matched structures
        non_matched_widget = QWidget()
        non_matched_layout = QVBoxLayout()

        non_matched_header = QLabel(f"Non-Matched Structures ({len(self.non_matched_smiles)} unique SMILES)")
        non_matched_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        non_matched_layout.addWidget(non_matched_header)

        self.non_matched_scroll = QScrollArea()
        self.non_matched_scroll.setWidgetResizable(True)
        self.non_matched_container = QLabel()
        self.non_matched_container.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.non_matched_scroll.setWidget(self.non_matched_container)
        non_matched_layout.addWidget(self.non_matched_scroll)

        non_matched_widget.setLayout(non_matched_layout)
        splitter.addWidget(non_matched_widget)

        layout.addWidget(splitter)
        central_widget.setLayout(layout)

        # Store original pixmaps
        self.matched_pixmap = None
        self.non_matched_pixmap = None

        # Render structures
        self.render_structures()

    def render_structures(self):
        """Render the structures as images."""
        # Render matched structures
        if self.matched_smiles:
            try:
                print(f"Rendering {len(self.matched_smiles)} matched structures...")

                # Generate image using custom rendering to avoid RDKit PNG issues
                matched_path = os.path.join(self.temp_dir.name, f"{self.filter_name}_matched.png")
                self.render_molecules_to_file(self.matched_smiles[:500], matched_path, 5)

                # Load with QPixmap
                self.matched_pixmap = QPixmap(matched_path)

                if self.matched_pixmap.isNull():
                    raise ValueError(f"Failed to load pixmap from {matched_path}")

                print(f"Pixmap loaded: {self.matched_pixmap.width()}x{self.matched_pixmap.height()}")
                self.update_matched_display()

            except Exception as e:
                error_msg = f"Error displaying matched structures:\n{str(e)}\n\nType: {type(e).__name__}"
                print(error_msg)
                import traceback

                traceback.print_exc()
                self.matched_container.setText(error_msg)
        else:
            self.matched_container.setText("No matched structures")

        # Render non-matched structures
        if self.non_matched_smiles:
            try:
                print(f"Rendering {len(self.non_matched_smiles)} non-matched structures...")

                # Generate image using custom rendering to avoid RDKit PNG issues
                non_matched_path = os.path.join(self.temp_dir.name, f"{self.filter_name}_non_matched.png")
                self.render_molecules_to_file(self.non_matched_smiles[:500], non_matched_path, 5)

                # Load with QPixmap
                self.non_matched_pixmap = QPixmap(non_matched_path)

                if self.non_matched_pixmap.isNull():
                    raise ValueError(f"Failed to load pixmap from {non_matched_path}")

                print(f"Pixmap loaded: {self.non_matched_pixmap.width()}x{self.non_matched_pixmap.height()}")
                self.update_non_matched_display()

            except Exception as e:
                error_msg = f"Error displaying non-matched structures:\n{str(e)}\n\nType: {type(e).__name__}"
                print(error_msg)
                import traceback

                traceback.print_exc()
                self.non_matched_container.setText(error_msg)
        else:
            self.non_matched_container.setText("No non-matched structures")

    def render_molecules_to_file(self, smiles_list, output_path, mols_per_row=5):
        """
        Render molecules to a PNG file using SVG intermediate to avoid PNG header issues.

        Args:
            smiles_list: List of SMILES strings
            output_path: Path to save the PNG file
            mols_per_row: Number of molecules per row
        """
        from rdkit.Chem import Draw
        from rdkit import Chem

        try:
            from cairosvg import svg2png

            has_cairosvg = True
        except ImportError:
            has_cairosvg = False
            print("Warning: cairosvg not available, will use alternative method")

        # Convert SMILES to molecules
        mols = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
            except:
                pass

        if not mols:
            # Create empty image with message
            img = PILImage.new("RGB", (800, 200), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 90), "No valid molecules to display", fill="black")
            img.save(output_path, format="PNG")
            return

        print(f"Rendering {len(mols)} valid molecules...")

        # Calculate grid dimensions
        n_rows = math.ceil(len(mols) / mols_per_row)

        # Try SVG route if cairosvg is available
        if has_cairosvg:
            try:
                print("mols to grid", mols)
                svg_data = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), maxMols=len(mols), useSVG=True)

                # Convert SVG to PNG using cairosvg
                png_data = svg2png(bytestring=svg_data)

                # Save PNG data
                with open(output_path, "wb") as f:
                    f.write(png_data)

                print(f"Successfully rendered using SVG->PNG conversion")
                return

            except Exception as e:
                print(f"SVG conversion failed: {e}, trying alternative method")

        # Fallback: Render individual molecules and combine manually
        print("Using fallback rendering method...")
        mol_size = (400, 400)

        # Create individual molecule images
        mol_images = []
        for mol in mols:
            try:
                # Generate SVG for single molecule
                svg = Draw.MolToImage(mol, size=mol_size, kekulize=True, wedgeBonds=True, fitImage=True)
                mol_images.append(svg)
            except Exception as e:
                print(f"Error rendering molecule: {e}")
                # Create blank image
                blank = PILImage.new("RGB", mol_size, color="white")
                mol_images.append(blank)

        # Create grid image
        grid_width = mols_per_row * mol_size[0]
        grid_height = n_rows * mol_size[1]
        grid_img = PILImage.new("RGB", (grid_width, grid_height), color="white")

        # Paste molecules into grid
        for idx, mol_img in enumerate(mol_images):
            row = idx // mols_per_row
            col = idx % mols_per_row
            x = col * mol_size[0]
            y = row * mol_size[1]
            grid_img.paste(mol_img, (x, y))

        # Save combined image
        grid_img.save(output_path, format="PNG")
        print(f"Successfully rendered using fallback method")

    def on_zoom_changed(self, value):
        """Handle zoom slider changes."""
        self.zoom_level = value
        self.zoom_label.setText(f"{value}%")
        self.update_matched_display()
        self.update_non_matched_display()

    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_slider.setValue(100)

    def update_matched_display(self):
        """Update the matched structures display with current zoom."""
        if self.matched_pixmap and not self.matched_pixmap.isNull():
            try:
                scaled_width = int(self.matched_pixmap.width() * self.zoom_level / 100)
                scaled_height = int(self.matched_pixmap.height() * self.zoom_level / 100)
                scaled_pixmap = self.matched_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.matched_container.setPixmap(scaled_pixmap)
                self.matched_container.adjustSize()
            except Exception as e:
                print(f"Error updating matched display: {e}")
                import traceback

                traceback.print_exc()

    def update_non_matched_display(self):
        """Update the non-matched structures display with current zoom."""
        if self.non_matched_pixmap and not self.non_matched_pixmap.isNull():
            try:
                scaled_width = int(self.non_matched_pixmap.width() * self.zoom_level / 100)
                scaled_height = int(self.non_matched_pixmap.height() * self.zoom_level / 100)
                scaled_pixmap = self.non_matched_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.non_matched_container.setPixmap(scaled_pixmap)
                self.non_matched_container.adjustSize()
            except Exception as e:
                print(f"Error updating non-matched display: {e}")
                import traceback

                traceback.print_exc()

    def closeEvent(self, event):
        """Clean up temporary directory on close."""
        self.temp_dir.cleanup()
        event.accept()


class MGFFilterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mgf_files = {}  # {file_path: {blocks, smiles_field, filters}}
        self.smarts_filters = {}  # {filter_name: smarts_string}
        self.filtered_results = {}  # {filter_name: {matched, non_matched}}
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sections = []  # List to keep track of all sections
        self.structure_windows = []  # Keep track of open structure viewer windows

        # Define predefined SMARTS filters
        self.predefined_filters = self.get_predefined_filters()

        self.init_ui()

    def get_predefined_filters(self):
        """Get dictionary of predefined SMARTS filters."""
        return OrderedDict(
            [
                (
                    "Prenyl Flavonoid",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3>> AND CC=C(C)CCC=C(C)C",
                ),
                (
                    "Flavone",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3>>",
                ),
                (
                    "Isoflavone",
                    "<<[O,o]~[C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3 OR [C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3>>",
                ),
                (
                    "Flavone or Isoflavone",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [O,o]~[C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3 OR [C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3>>",
                ),
                (
                    "Prenyl Flavone or Isoflavone",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [O,o]~[C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3 OR [C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3>> AND CC=C(C)CCC=C(C)C",
                ),
                ("Chalcone (saturated)", "[O,o]=[C,c](-[CH2]-[CH2]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2"),
                ("Chalcone (unsaturated)", "[O,o]=[C,c](-[CH]=[CH]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2"),
                ("Chromone", "O=C@1@C@C@O@C@2@C@C@C@C@C@12"),
                ("Geranyl", "CC=C(C)CCC=C(C)C"),
                ("Farnesyl", "[CH2][CH]=C([CH3])[CH2]([CH2][CH]=C([CH3])[CH2]([CH2][CH]=C([CH3])[CH3]))"),
            ]
        )

    def init_ui(self):
        self.setWindowTitle("MGF SMARTS Filter Wizard")
        self.setGeometry(100, 100, 1400, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)

        layout = QVBoxLayout()
        scroll_content.setLayout(layout)

        # Section 1: Load MGF Files
        self.section1 = CollapsibleSection("1. Load MGF Files")
        self.section1.toggled.connect(self.on_section_toggled)
        self.init_section1()
        layout.addWidget(self.section1)
        self.sections.append(self.section1)

        # Section 2: Canonicalization
        self.section2 = CollapsibleSection("2. SMILES Canonicalization")
        self.section2.toggled.connect(self.on_section_toggled)
        self.section2.collapse()  # Start collapsed
        self.init_section2()
        layout.addWidget(self.section2)
        self.sections.append(self.section2)

        # Section 3: SMARTS Filters
        self.section3 = CollapsibleSection("3. Define SMARTS Filters")
        self.section3.toggled.connect(self.on_section_toggled)
        self.section3.collapse()  # Start collapsed
        self.init_section3()
        layout.addWidget(self.section3)
        self.sections.append(self.section3)

        # Section 4: Export Results
        self.section4 = CollapsibleSection("4. Export Filtered Results")
        self.section4.toggled.connect(self.on_section_toggled)
        self.section4.collapse()  # Start collapsed
        self.init_section4()
        layout.addWidget(self.section4)
        self.sections.append(self.section4)

        layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        main_widget.setLayout(main_layout)

    def on_section_toggled(self, toggled_section):
        """Handle section toggle to ensure only one section is open at a time."""
        if toggled_section.toggle_button.isChecked():
            # Collapse all other sections
            for section in self.sections:
                if section != toggled_section:
                    section.collapse()

    def init_section1(self):
        """Initialize the MGF file loading section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Load MGF Files</h3>
            <p>This section allows you to load one or more MGF files for filtering.</p>
            <ol>
                <li><b>Load MGF File:</b> Click to browse and load MGF files (can select multiple)</li>
                <li><b>SMILES Required:</b> Toggle to filter out spectra without SMILES information</li>
                <li><b>Select SMILES Field:</b> Choose which field contains the SMILES strings</li>
                <li><b>Filter by Meta-Field:</b> Optionally filter spectra by specific field values</li>
                <li><b>Statistics:</b> View how many spectra are loaded, filtered, and removed</li>
            </ol>
            <p><b>Tip:</b> You can select multiple files in the table (Ctrl+Click or Shift+Click) to apply settings to all at once.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        # Load button and SMILES required checkbox
        load_layout = QHBoxLayout()
        load_btn = QPushButton("Load MGF File(s)")
        load_btn.clicked.connect(self.load_mgf_file)
        load_layout.addWidget(load_btn)

        self.smiles_required_checkbox = QCheckBox("SMILES Required")
        self.smiles_required_checkbox.setChecked(False)
        self.smiles_required_checkbox.stateChanged.connect(self.on_smiles_required_changed)
        load_layout.addWidget(self.smiles_required_checkbox)
        load_layout.addStretch()
        layout.addLayout(load_layout)

        # File table
        layout.addWidget(QLabel("Loaded MGF Files (select one or more):"))
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(7)
        self.file_table.setHorizontalHeaderLabels(["File Name", "Total Entries", "No SMILES", "SMILES Field", "Filtered", "Removed", "Meta-Filters"])
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_table.itemSelectionChanged.connect(self.on_file_selection_changed)

        # Adjust column widths
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # File Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Total
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # No SMILES
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # SMILES Field
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Filtered
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Removed
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # Meta-Filters

        layout.addWidget(self.file_table)

        # File details
        details_group = QGroupBox("File Details (applies to selected files)")
        details_layout = QVBoxLayout()

        # SMILES field selection
        smiles_layout = QHBoxLayout()
        smiles_layout.addWidget(QLabel("SMILES Field:"))
        self.smiles_field_combo = QComboBox()
        self.smiles_field_combo.currentTextChanged.connect(self.on_smiles_field_changed)
        smiles_layout.addWidget(self.smiles_field_combo)
        apply_smiles_btn = QPushButton("Apply to Selected")
        apply_smiles_btn.clicked.connect(self.apply_smiles_field_to_selected)
        smiles_layout.addWidget(apply_smiles_btn)
        details_layout.addLayout(smiles_layout)

        # Meta-field filtering
        details_layout.addWidget(QLabel("Filter by Meta-Field:"))
        self.meta_field_combo = QComboBox()
        self.meta_field_combo.currentTextChanged.connect(self.on_meta_field_changed)
        details_layout.addWidget(self.meta_field_combo)

        self.meta_values_list = QListWidget()
        self.meta_values_list.setSelectionMode(QListWidget.MultiSelection)
        self.meta_values_list.itemSelectionChanged.connect(self.on_meta_values_changed)
        details_layout.addWidget(self.meta_values_list)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section1.add_widget(content)

    def init_section2(self):
        """Initialize the canonicalization section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>SMILES Canonicalization</h3>
            <p>Canonicalization converts all SMILES strings to a standard format.</p>
            <p><b>Why canonicalize?</b></p>
            <ul>
                <li>Different SMILES can represent the same molecule</li>
                <li>Canonical form ensures consistent comparisons</li>
                <li>Helps identify duplicates</li>
            </ul>
            <p><b>Note:</b> This process may take some time for large datasets. A progress bar will show the status.</p>
            <p><b>Tip:</b> This step is optional but recommended for accurate filtering.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        # Canonicalization
        self.canon_checkbox = QCheckBox("Calculate Canonical SMILES")
        self.canon_checkbox.setChecked(True)
        layout.addWidget(self.canon_checkbox)

        canon_btn = QPushButton("Apply Canonicalization")
        canon_btn.clicked.connect(self.apply_canonicalization)
        layout.addWidget(canon_btn)

        self.canon_status_label = QLabel("Status: Not applied")
        layout.addWidget(self.canon_status_label)

        layout.addStretch()

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section2.add_widget(content)

    def init_section3(self):
        """Initialize the SMARTS filter section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Define SMARTS Filters</h3>
            <p>SMARTS (SMiles ARbitrary Target Specification) is a pattern language for chemical structures.</p>
            <p><b>How to use:</b></p>
            <ol>
                <li>Enter a filter name</li>
                <li>Enter SMARTS pattern(s) using AND/OR logic</li>
                <li>Use <b>&lt;&lt; ... &gt;&gt;</b> for OR groups, <b>AND</b> to separate groups</li>
                <li>Click "Add Filter" or press Enter</li>
                <li>View statistics in the table</li>
                <li>Click "View Structures" to see matched/non-matched compounds</li>
                <li>Edit existing filters by double-clicking them</li>
                <li>Use "Predefined Filters" button for common patterns</li>
                <li>Use "Load Filters from JSON" to import multiple filters at once</li>
            </ol>
            <p><b>Example 1:</b> Simple AND:<br>
            <code>c1ccccc1 AND [OH]</code></p>
            <p><b>Example 2:</b> OR group (flavone OR isoflavone):<br>
            <code>&lt;&lt;flavone_smarts OR isoflavone_smarts&gt;&gt;</code></p>
            <p><b>Example 3:</b> Combined (flavone OR isoflavone) AND prenyl:<br>
            <code>&lt;&lt;flavone_smarts OR isoflavone_smarts&gt;&gt; AND CC=C(C)CCC=C(C)C</code></p>
            <p><b>JSON Format:</b> {"filter_name": "smarts_pattern", ...}</p>
            <p><b>Note:</b> Use AND (case-insensitive) to separate patterns that must all match, and &lt;&lt; pattern1 OR pattern2 &gt;&gt; for alternatives.</p>
            <p><b>Tip:</b> Use tools like <a href="https://smarts.plus">smarts.plus</a> to build and test SMARTS patterns.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        # Add/Edit filter
        add_layout = QHBoxLayout()
        add_layout.addWidget(QLabel("Filter Name:"))
        self.filter_name_input = QLineEdit()
        add_layout.addWidget(self.filter_name_input)

        add_layout.addWidget(QLabel("SMARTS (AND/OR logic):"))
        self.smarts_input = QLineEdit()
        self.smarts_input.returnPressed.connect(self.add_smarts_filter)
        add_layout.addWidget(self.smarts_input)

        add_btn = QPushButton("Add/Update Filter")
        add_btn.clicked.connect(self.add_smarts_filter)
        add_layout.addWidget(add_btn)

        # Predefined filters button
        predefined_btn = QPushButton("Predefined Filters")
        predefined_btn.clicked.connect(self.show_predefined_filters_menu)
        add_layout.addWidget(predefined_btn)

        # Load from JSON button
        load_json_btn = QPushButton("Load Filters from JSON")
        load_json_btn.clicked.connect(self.load_filters_from_json)
        add_layout.addWidget(load_json_btn)

        # Save to JSON button
        save_json_btn = QPushButton("Save Defined Filters")
        save_json_btn.clicked.connect(self.save_filters_to_json)
        add_layout.addWidget(save_json_btn)

        layout.addLayout(add_layout)

        # Filter table with control buttons
        filter_control_layout = QHBoxLayout()
        filter_control_layout.addWidget(QLabel("Defined Filters:"))
        edit_filter_btn = QPushButton("Edit Selected")
        edit_filter_btn.clicked.connect(self.edit_selected_filter)
        filter_control_layout.addWidget(edit_filter_btn)
        delete_filter_btn = QPushButton("Delete Selected")
        delete_filter_btn.clicked.connect(self.delete_selected_filter)
        filter_control_layout.addWidget(delete_filter_btn)
        view_structures_btn = QPushButton("View Structures")
        view_structures_btn.clicked.connect(self.view_structures)
        filter_control_layout.addWidget(view_structures_btn)
        filter_control_layout.addStretch()
        layout.addLayout(filter_control_layout)

        # Filter table
        self.filter_table = QTableWidget()
        self.filter_table.setColumnCount(4)
        self.filter_table.setHorizontalHeaderLabels(["Filter Name", "Total SMILES", "Matched", "Non-Matched"])
        self.filter_table.horizontalHeader().setStretchLastSection(False)
        self.filter_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.filter_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.filter_table.itemDoubleClicked.connect(self.edit_selected_filter)
        self.filter_table.itemSelectionChanged.connect(self.on_filter_selection_changed)

        # Adjust column widths
        header = self.filter_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Filter Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Total
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Matched
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Non-Matched

        layout.addWidget(self.filter_table)

        # SMARTS Visualization Section
        layout.addWidget(QLabel("SMARTS Pattern Visualization:"))
        self.smarts_viz_scroll = QScrollArea()
        self.smarts_viz_scroll.setWidgetResizable(True)
        self.smarts_viz_scroll.setMinimumHeight(300)
        self.smarts_viz_container = QLabel()
        self.smarts_viz_container.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.smarts_viz_container.setText("Select a filter to view its SMARTS patterns")
        self.smarts_viz_scroll.setWidget(self.smarts_viz_container)
        layout.addWidget(self.smarts_viz_scroll)

        # Store pixmap for visualization
        self.smarts_viz_pixmap = None

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section3.add_widget(content)

    def show_predefined_filters_menu(self):
        """Show a context menu with predefined SMARTS filters."""
        menu = QMenu(self)

        for filter_name, smarts_pattern in self.predefined_filters.items():
            action = menu.addAction(filter_name)
            action.triggered.connect(lambda checked, fn=filter_name, sp=smarts_pattern: self.load_predefined_filter(fn, sp))

        # Show menu at button position
        sender = self.sender()
        menu.exec_(sender.mapToGlobal(sender.rect().bottomLeft()))

    def load_predefined_filter(self, filter_name, smarts_pattern):
        """Load a predefined filter into the input fields."""
        self.filter_name_input.setText(filter_name)
        self.smarts_input.setText(smarts_pattern)
        QMessageBox.information(self, "Filter Loaded", f"Predefined filter '{filter_name}' has been loaded.\n\nYou can modify the name or SMARTS pattern before adding it.")

    def init_section4(self):
        """Initialize the export section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Export Filtered Results</h3>
            <p>Export your filtered spectra to MGF files.</p>
            <p><b>Output files:</b></p>
            <ul>
                <li><code>&lt;base_name&gt;_&lt;filter_name&gt;_matched.mgf</code> - Spectra matching the filter</li>
                <li><code>&lt;base_name&gt;_&lt;filter_name&gt;_noMatch.mgf</code> - Spectra not matching the filter</li>
            </ul>
            <p><b>Steps:</b></p>
            <ol>
                <li>Select which filters to export (or export all)</li>
                <li>Choose base output file name</li>
                <li>Click "Export Filtered MGF Files"</li>
            </ol>
            <p><b>Note:</b> All loaded MGF files will be combined in the export. Each filter will create two files with the filter name as a suffix.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select filters to export:"))

        # Checkbox for "Export All"
        export_all_layout = QHBoxLayout()
        self.export_all_checkbox = QCheckBox("Export All Filters")
        self.export_all_checkbox.setChecked(True)
        self.export_all_checkbox.stateChanged.connect(self.on_export_all_changed)
        export_all_layout.addWidget(self.export_all_checkbox)
        export_all_layout.addStretch()
        layout.addLayout(export_all_layout)

        # List of filters with checkboxes
        self.export_filter_list = QListWidget()
        self.export_filter_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.export_filter_list)

        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("Base Output File:"))
        self.output_file_input = QLineEdit()
        export_layout.addWidget(self.output_file_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_output_file)
        export_layout.addWidget(browse_btn)
        layout.addLayout(export_layout)

        export_btn = QPushButton("Export Filtered MGF Files")
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)

        layout.addStretch()

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section4.add_widget(content)

    def load_mgf_file(self):
        """Load one or more MGF files."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select MGF File(s)", "", "MGF Files (*.mgf);;All Files (*)")

        if not file_paths:
            return

        progress = QProgressDialog("Loading MGF files...", "Cancel", 0, len(file_paths), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        loaded_count = 0
        error_count = 0

        for idx, file_path in enumerate(file_paths):
            if progress.wasCanceled():
                break

            progress.setValue(idx)
            progress.setLabelText(f"Loading {os.path.basename(file_path)}...")
            QApplication.processEvents()

            try:
                blocks = Filters.parse_mgf_file(file_path, check_required_keys=False)

                fields = Filters.get_fields(blocks)

                # Auto-detect SMILES field (case-insensitive)
                smiles_field = None
                for field in fields:
                    if field.lower() == "smiles":
                        smiles_field = field
                        print(f"Auto-detected SMILES field: {field}")
                        break

                self.mgf_files[file_path] = {
                    "blocks": blocks,
                    "original_blocks": blocks.copy(),
                    "fields": fields,
                    "smiles_field": smiles_field,  # Auto-set if 'smiles' field exists
                    "meta_filters": {},
                    "filtered_blocks": blocks,
                    "canonicalized": False,
                    "smiles_required": self.smiles_required_checkbox.isChecked(),
                }

                # Apply SMILES filter if required
                if self.smiles_required_checkbox.isChecked():
                    self.apply_smiles_filter_to_file(file_path)

                self.add_file_to_table(file_path)
                loaded_count += 1

            except Exception as e:
                error_count += 1
                print(f"Error loading {file_path}: {str(e)}")

        progress.setValue(len(file_paths))

        if loaded_count > 0:
            # Select the first loaded file
            self.file_table.selectRow(self.file_table.rowCount() - loaded_count)

        msg = f"Successfully loaded {loaded_count} file(s)"
        if error_count > 0:
            msg += f"\n{error_count} file(s) failed to load"

        QMessageBox.information(self, "Load Complete", msg)

    def on_smiles_required_changed(self, state):
        """Handle SMILES required checkbox state change."""
        smiles_required = self.smiles_required_checkbox.isChecked()

        # Apply to all loaded files
        for file_path, file_data in self.mgf_files.items():
            file_data["smiles_required"] = smiles_required

            if smiles_required:
                self.apply_smiles_filter_to_file(file_path)
            else:
                # Reset to use all blocks from original
                file_data["filtered_blocks"] = file_data["original_blocks"].copy()

            self.update_file_in_table(file_path)

    def apply_smiles_filter_to_file(self, file_path):
        """Filter blocks to only include those with SMILES information."""
        file_data = self.mgf_files[file_path]
        smiles_field = file_data.get("smiles_field")

        if not smiles_field:
            # No SMILES field selected yet, filter out any block that doesn't have any potential SMILES field
            filtered = []
            for block in file_data["original_blocks"]:
                # Check if any field might contain SMILES
                has_smiles = False
                for key, value in block.items():
                    if key.lower() in ["smiles", "inchi", "inchikey", "canonical_smiles"]:
                        if value and str(value).strip() and str(value).strip().lower() not in ["", "n/a", "na", "none", "null"]:
                            has_smiles = True
                            break
                if has_smiles:
                    filtered.append(block)
            file_data["filtered_blocks"] = filtered
        else:
            # Filter based on the selected SMILES field
            filtered = []
            for block in file_data["original_blocks"]:
                if smiles_field in block:
                    value = block[smiles_field]
                    if value and str(value).strip() and str(value).strip().lower() not in ["", "n/a", "na", "none", "null"]:
                        filtered.append(block)
            file_data["filtered_blocks"] = filtered

    def add_file_to_table(self, file_path):
        """Add a file to the file table."""
        file_data = self.mgf_files[file_path]

        row_position = self.file_table.rowCount()
        self.file_table.insertRow(row_position)

        # File Name
        self.file_table.setItem(row_position, 0, QTableWidgetItem(os.path.basename(file_path)))

        # Total Entries
        total = len(file_data["original_blocks"])
        self.file_table.setItem(row_position, 1, QTableWidgetItem(str(total)))

        # Count blocks without SMILES
        no_smiles_count = self.count_blocks_without_smiles(file_path)
        self.file_table.setItem(row_position, 2, QTableWidgetItem(str(no_smiles_count)))

        # SMILES Field
        smiles_field = file_data.get("smiles_field", "")
        self.file_table.setItem(row_position, 3, QTableWidgetItem(smiles_field if smiles_field else "-"))

        # Filtered
        filtered = len(file_data["filtered_blocks"])
        self.file_table.setItem(row_position, 4, QTableWidgetItem(str(filtered)))

        # Removed
        removed = total - filtered
        self.file_table.setItem(row_position, 5, QTableWidgetItem(str(removed)))

        # Meta-Filters
        meta_filters = file_data.get("meta_filters", {})
        filter_text = ", ".join([f"{k}({len(v)})" for k, v in meta_filters.items()]) if meta_filters else "-"
        self.file_table.setItem(row_position, 6, QTableWidgetItem(filter_text))

    def count_blocks_without_smiles(self, file_path):
        """Count how many blocks don't have SMILES information."""
        file_data = self.mgf_files[file_path]
        smiles_field = file_data.get("smiles_field")

        count = 0
        if not smiles_field:
            # Check for any SMILES-like field
            for block in file_data["original_blocks"]:
                has_smiles = False
                for key, value in block.items():
                    if key.lower() in ["smiles", "inchi", "inchikey", "canonical_smiles"]:
                        if value and str(value).strip() and str(value).strip().lower() not in ["", "n/a", "na", "none", "null"]:
                            has_smiles = True
                            break
                if not has_smiles:
                    count += 1
        else:
            # Count based on selected SMILES field
            for block in file_data["original_blocks"]:
                if smiles_field not in block:
                    count += 1
                else:
                    value = block[smiles_field]
                    if not value or not str(value).strip() or str(value).strip().lower() in ["", "n/a", "na", "none", "null"]:
                        count += 1

        return count

    def update_file_in_table(self, file_path):
        """Update a file's information in the table."""
        # Find the row for this file
        file_name = os.path.basename(file_path)
        for row in range(self.file_table.rowCount()):
            if self.file_table.item(row, 0).text() == file_name:
                file_data = self.mgf_files[file_path]

                # Update No SMILES count
                no_smiles_count = self.count_blocks_without_smiles(file_path)
                self.file_table.item(row, 2).setText(str(no_smiles_count))

                # Update SMILES Field
                smiles_field = file_data.get("smiles_field", "")
                self.file_table.item(row, 3).setText(smiles_field if smiles_field else "-")

                # Update Filtered
                filtered = len(file_data["filtered_blocks"])
                self.file_table.item(row, 4).setText(str(filtered))

                # Update Removed
                total = len(file_data["original_blocks"])
                removed = total - filtered
                self.file_table.item(row, 5).setText(str(removed))

                # Update Meta-Filters
                meta_filters = file_data.get("meta_filters", {})
                filter_text = ", ".join([f"{k}({len(v)})" for k, v in meta_filters.items()]) if meta_filters else "-"
                self.file_table.item(row, 6).setText(filter_text)

                break

    def get_file_path_by_name(self, file_name):
        """Get full file path from file name."""
        for path in self.mgf_files.keys():
            if os.path.basename(path) == file_name:
                return path
        return None

    def on_file_selection_changed(self):
        """Handle file selection changes in the table."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())

        if not selected_rows:
            return

        # Collect all fields from selected files
        all_fields = set()
        for row in selected_rows:
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if file_path:
                all_fields.update(self.mgf_files[file_path]["fields"])

        # Update SMILES field combo with common fields
        self.smiles_field_combo.clear()
        self.smiles_field_combo.addItems(sorted(all_fields))

        # If single file selected, set its SMILES field
        if len(selected_rows) == 1:
            row = list(selected_rows)[0]
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if file_path and self.mgf_files[file_path]["smiles_field"]:
                self.smiles_field_combo.setCurrentText(self.mgf_files[file_path]["smiles_field"])

        # Update meta-field combo
        self.meta_field_combo.clear()
        self.meta_field_combo.addItem("")
        self.meta_field_combo.addItems(sorted(all_fields))

    def apply_smiles_field_to_selected(self):
        """Apply the selected SMILES field to all selected files."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())
        field_name = self.smiles_field_combo.currentText()

        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select one or more files first.")
            return

        if not field_name:
            QMessageBox.warning(self, "Warning", "Please select a SMILES field.")
            return

        applied_count = 0
        for row in selected_rows:
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if file_path:
                if field_name in self.mgf_files[file_path]["fields"]:
                    self.mgf_files[file_path]["smiles_field"] = field_name

                    # Reapply SMILES filter if required
                    if self.mgf_files[file_path].get("smiles_required", False):
                        self.apply_smiles_filter_to_file(file_path)

                    self.update_file_in_table(file_path)
                    applied_count += 1

        QMessageBox.information(self, "Success", f"Applied SMILES field '{field_name}' to {applied_count} file(s)")

    def on_smiles_field_changed(self, field_name):
        """Handle SMILES field selection (only updates combo, doesn't apply)."""
        # This method is kept for compatibility but actual application
        # is now done via the "Apply to Selected" button
        pass

    def on_meta_field_changed(self, field_name):
        """Handle meta-field selection for filtering."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())
        if not selected_rows:
            return

        self.meta_values_list.clear()

        if not field_name:
            return

        # Collect all unique values from all selected files
        all_values = set()
        for row in selected_rows:
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if not file_path:
                continue

            file_data = self.mgf_files[file_path]

            for block in file_data["original_blocks"]:
                if field_name in block:
                    all_values.add(str(block[field_name]))

        # Add to list
        for value in sorted(all_values):
            self.meta_values_list.addItem(value)

        # Select all by default
        for i in range(self.meta_values_list.count()):
            self.meta_values_list.item(i).setSelected(True)

    def on_meta_values_changed(self):
        """Handle meta-value selection changes - applies to all selected files."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())
        if not selected_rows:
            return

        field_name = self.meta_field_combo.currentText()
        if not field_name:
            return

        selected_values = {item.text() for item in self.meta_values_list.selectedItems()}

        # Apply filter to all selected files
        for row in selected_rows:
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if not file_path:
                continue

            file_data = self.mgf_files[file_path]

            # Start with original blocks or SMILES-filtered blocks
            base_blocks = file_data["original_blocks"]
            if file_data.get("smiles_required", False):
                # Apply SMILES filter first
                self.apply_smiles_filter_to_file(file_path)
                base_blocks = file_data["filtered_blocks"]

            # Filter blocks by meta-field
            filtered_blocks = []
            for block in base_blocks:
                if field_name not in block:
                    continue
                if str(block[field_name]) in selected_values:
                    filtered_blocks.append(block)

            file_data["filtered_blocks"] = filtered_blocks
            file_data["meta_filters"][field_name] = selected_values

            self.update_file_in_table(file_path)

    def apply_canonicalization(self):
        """Apply SMILES canonicalization to all loaded files."""
        if not self.canon_checkbox.isChecked():
            QMessageBox.warning(self, "Warning", "Please check the canonicalization checkbox first.")
            return

        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "Please load at least one MGF file first.")
            return

        # Count total blocks to process
        total_blocks = sum(len(file_data["filtered_blocks"]) for file_data in self.mgf_files.values())

        progress = QProgressDialog("Canonicalizing SMILES...", "Cancel", 0, total_blocks, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        processed = 0
        success_count = 0
        error_count = 0

        for file_path, file_data in self.mgf_files.items():
            if progress.wasCanceled():
                break

            smiles_field = file_data["smiles_field"]
            if not smiles_field:
                processed += len(file_data["filtered_blocks"])
                continue

            for block in file_data["filtered_blocks"]:
                if progress.wasCanceled():
                    break

                if smiles_field in block and block[smiles_field]:
                    try:
                        mol = rdkit.Chem.MolFromSmiles(block[smiles_field])
                        if mol:
                            block[smiles_field] = rdkit.Chem.MolToSmiles(mol, canonical=True)
                            success_count += 1
                        else:
                            error_count += 1
                    except:
                        error_count += 1

                processed += 1
                progress.setValue(processed)
                QApplication.processEvents()

            file_data["canonicalized"] = True
            self.update_file_in_table(file_path)

        progress.setValue(total_blocks)

        status_msg = f"Canonicalized {success_count} SMILES"
        if error_count > 0:
            status_msg += f" ({error_count} errors)"

        self.canon_status_label.setText(f"Status: {status_msg}")

        QMessageBox.information(self, "Success", status_msg)

    def add_smarts_filter(self):
        """Add or update a SMARTS filter."""
        filter_name = self.filter_name_input.text().strip()
        smarts_string = self.smarts_input.text().strip()

        if not filter_name or not smarts_string:
            QMessageBox.warning(self, "Warning", "Please provide both filter name and SMARTS string.")
            return

        # Parse AND-separated groups (case-insensitive)
        import re

        and_groups = re.split(r"\s+AND\s+", smarts_string, flags=re.IGNORECASE)
        and_groups = [s.strip() for s in and_groups if s.strip()]

        # Parse each AND group for OR patterns (marked with << >>)
        parsed_smarts = []
        for and_group in and_groups:
            # Check if this group has << >> markers
            if "<<" in and_group and ">>" in and_group:
                # Extract content within << >>
                or_match = re.search(r"<<(.+?)>>", and_group, re.DOTALL)
                if or_match:
                    or_content = or_match.group(1)
                    # Split by OR (case-insensitive)
                    or_patterns = re.split(r"\s+OR\s+", or_content, flags=re.IGNORECASE)
                    or_patterns = [p.strip() for p in or_patterns if p.strip()]
                else:
                    # If markers exist but regex doesn't match, treat as single pattern
                    or_patterns = [and_group.strip()]
            else:
                # No OR patterns, treat as single pattern
                or_patterns = [and_group.strip()]

            parsed_smarts.append(or_patterns)

        print(f"Parsed SMARTS for filter '{filter_name}':")
        pprint(parsed_smarts)

        # Validate all SMARTS patterns
        try:
            for or_group in parsed_smarts:
                for smarts in or_group:
                    mol = rdkit.Chem.MolFromSmarts(smarts)
                    if not mol:
                        raise ValueError(f"Invalid SMARTS: {smarts}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid SMARTS string:\n{str(e)}")
            return

        # Check if updating existing filter
        is_update = filter_name in self.smarts_filters

        self.smarts_filters[filter_name] = parsed_smarts

        # Update or add to export list
        if not is_update:
            item = QListWidgetItem(filter_name)
            self.export_filter_list.addItem(item)
            if self.export_all_checkbox.isChecked():
                item.setSelected(True)

        self.filter_name_input.clear()
        self.smarts_input.clear()

        # Trigger filtering
        self.apply_filter(filter_name)

        action = "Updated" if is_update else "Added"
        QMessageBox.information(self, "Success", f"{action} filter: {filter_name}")

    def add_filter_to_table(self, filter_name):
        """Add a filter to the filter table."""
        row_position = self.filter_table.rowCount()
        self.filter_table.insertRow(row_position)

        # Filter Name
        self.filter_table.setItem(row_position, 0, QTableWidgetItem(filter_name))

        # Statistics will be updated by update_filter_in_table
        self.filter_table.setItem(row_position, 1, QTableWidgetItem("0"))
        self.filter_table.setItem(row_position, 2, QTableWidgetItem("0"))
        self.filter_table.setItem(row_position, 3, QTableWidgetItem("0"))

    def update_filter_in_table(self, filter_name):
        """Update a filter's statistics in the table."""
        # Find the row for this filter
        for row in range(self.filter_table.rowCount()):
            if self.filter_table.item(row, 0).text() == filter_name:
                if filter_name in self.filtered_results:
                    results = self.filtered_results[filter_name]
                    total = len(results["matched_smiles"]) + len(results["non_matched_smiles"])
                    matched = len(results["matched_smiles"])
                    non_matched = len(results["non_matched_smiles"])

                    self.filter_table.item(row, 1).setText(str(total))
                    self.filter_table.item(row, 2).setText(str(matched))
                    self.filter_table.item(row, 3).setText(str(non_matched))
                return

        # If not found, add new row
        self.add_filter_to_table(filter_name)
        self.update_filter_in_table(filter_name)

    def edit_selected_filter(self):
        """Edit the selected filter."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a filter to edit.")
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        if filter_name in self.smarts_filters:
            self.filter_name_input.setText(filter_name)

            # Reconstruct the SMARTS string from parsed format
            parsed_smarts = self.smarts_filters[filter_name]
            smarts_parts = []

            for or_group in parsed_smarts:
                if len(or_group) > 1:
                    # Has OR patterns
                    or_string = "<<" + " OR ".join(or_group) + ">>"
                    smarts_parts.append(or_string)
                else:
                    # Single pattern
                    smarts_parts.append(or_group[0])

            self.smarts_input.setText(" AND ".join(smarts_parts))

    def delete_selected_filter(self):
        """Delete the selected filter."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a filter to delete.")
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete filter '{filter_name}'?", QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Remove from data structures
            if filter_name in self.smarts_filters:
                del self.smarts_filters[filter_name]
            if filter_name in self.filtered_results:
                del self.filtered_results[filter_name]

            # Remove from UI
            self.filter_table.removeRow(row)

            # Remove from export list
            for i in range(self.export_filter_list.count()):
                if self.export_filter_list.item(i).text() == filter_name:
                    self.export_filter_list.takeItem(i)
                    break

    def on_filter_selection_changed(self):
        """Handle filter selection changes in the table and update SMARTS visualization."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            self.smarts_viz_container.setText("Select a filter to view its SMARTS patterns")
            self.smarts_viz_pixmap = None
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        if filter_name in self.smarts_filters:
            self.render_smarts_structures(filter_name)
        else:
            self.smarts_viz_container.setText("No SMARTS patterns available for this filter")
            self.smarts_viz_pixmap = None

    def view_structures(self):
        """Open a new window to view structures for the selected filter."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a filter to view structures.")
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        if filter_name not in self.filtered_results:
            QMessageBox.warning(self, "Warning", "Filter results not available. Please apply the filter first.")
            return

        results = self.filtered_results[filter_name]

        # Create and show structure viewer window
        viewer = StructureViewerWindow(filter_name, results["matched_smiles"], results["non_matched_smiles"], self)
        viewer.show()

        # Keep reference to prevent garbage collection
        self.structure_windows.append(viewer)

    def render_smarts_structures(self, filter_name):
        """Render SMARTS patterns as molecular structure images."""
        if filter_name not in self.smarts_filters:
            return

        parsed_smarts = self.smarts_filters[filter_name]

        try:
            from rdkit.Chem import Draw
            from rdkit import Chem

            # Collect all SMARTS patterns from the filter
            smarts_list = []
            labels = []

            for and_idx, or_group in enumerate(parsed_smarts):
                for or_idx, smarts_pattern in enumerate(or_group):
                    smarts_list.append(smarts_pattern)
                    if len(or_group) > 1:
                        # Multiple patterns in OR group
                        labels.append(f"AND Group {and_idx + 1}, OR {or_idx + 1}")
                    else:
                        # Single pattern in AND group
                        labels.append(f"AND Group {and_idx + 1}")

            if not smarts_list:
                self.smarts_viz_container.setText("No SMARTS patterns found")
                return

            # Convert SMARTS to molecules
            mols = []
            valid_labels = []
            for smarts, label in zip(smarts_list, labels):
                try:
                    mol = Chem.MolFromSmarts(smarts)
                    if mol:
                        mols.append(mol)
                        valid_labels.append(label)
                except Exception as e:
                    print(f"Failed to parse SMARTS '{smarts}': {e}")
                    continue

            if not mols:
                self.smarts_viz_container.setText("Could not parse any SMARTS patterns")
                return

            # Render molecules to image
            mols_per_row = min(3, len(mols))
            mol_size = (400, 400)
            n_rows = math.ceil(len(mols) / mols_per_row)

            img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=mol_size, legends=valid_labels, returnPNG=False)

            # Convert PIL image to QPixmap
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            qimage = QImage()
            qimage.loadFromData(img_byte_arr.read())
            pixmap = QPixmap.fromImage(qimage)

            if not pixmap.isNull():
                self.smarts_viz_pixmap = pixmap
                # Scale to fit container width while maintaining aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(self.smarts_viz_scroll.viewport().width() - 20, Qt.SmoothTransformation)
                self.smarts_viz_container.setPixmap(scaled_pixmap)
            else:
                self.smarts_viz_container.setText("Failed to create visualization")

        except Exception as e:
            self.smarts_viz_container.setText(f"Error rendering SMARTS: {str(e)}")
            print(f"Error rendering SMARTS structures: {e}")

    def apply_filter(self, filter_name):
        """Apply the SMARTS filter and display results."""
        if filter_name not in self.smarts_filters:
            return

        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "Please load at least one MGF file first.")
            return

        parsed_smarts = self.smarts_filters[filter_name]

        # Collect all unique SMILES from all files
        all_smiles = set()
        smiles_to_blocks = defaultdict(list)

        for file_path, file_data in self.mgf_files.items():
            smiles_field = file_data["smiles_field"]
            if not smiles_field:
                continue

            for block in file_data["filtered_blocks"]:
                if smiles_field in block and block[smiles_field]:
                    smiles = block[smiles_field]
                    all_smiles.add(smiles)
                    smiles_to_blocks[smiles].append(block)

        if not all_smiles:
            QMessageBox.warning(self, "Warning", "No SMILES found in loaded files. Please select a SMILES field.")
            return

        # Filter using substructure matching with AND/OR logic
        matched_smiles = []
        non_matched_smiles = []

        progress = QProgressDialog("Filtering SMILES...", "Cancel", 0, len(all_smiles), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        for idx, smiles in enumerate(all_smiles):
            if progress.wasCanceled():
                break

            progress.setValue(idx)
            QApplication.processEvents()

            try:
                if self.check_smarts_match(smiles, parsed_smarts):
                    matched_smiles.append(smiles)
                else:
                    non_matched_smiles.append(smiles)
            except:
                non_matched_smiles.append(smiles)

        progress.setValue(len(all_smiles))

        # Store results
        self.filtered_results[filter_name] = {"matched_smiles": matched_smiles, "non_matched_smiles": non_matched_smiles, "smiles_to_blocks": smiles_to_blocks}

        # Update table
        self.update_filter_in_table(filter_name)

    def check_smarts_match(self, smiles, parsed_smarts):
        """
        Check if a SMILES string matches the parsed SMARTS pattern.

        Args:
            smiles: SMILES string to check
            parsed_smarts: List of OR groups, where each OR group is a list of SMARTS patterns
                          Format: [[pattern1_or1, pattern1_or2], [pattern2], ...]
                          Top level is AND, inner lists are OR

        Returns:
            bool: True if matches, False otherwise
        """
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if not mol:
            return False

        # All AND groups must match (top level)
        for or_group in parsed_smarts:
            # At least one pattern in the OR group must match
            or_match = False
            for smarts_pattern in or_group:
                try:
                    pattern_mol = rdkit.Chem.MolFromSmarts(smarts_pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        or_match = True
                        break
                except:
                    continue

            # If no pattern in this OR group matched, the whole filter fails
            if not or_match:
                return False

        # All AND groups matched
        return True

    def on_export_all_changed(self, state):
        """Handle export all checkbox state change."""
        if self.export_all_checkbox.isChecked():
            # Select all filters
            for i in range(self.export_filter_list.count()):
                self.export_filter_list.item(i).setSelected(True)
        else:
            # Deselect all filters
            self.export_filter_list.clearSelection()

    def browse_output_folder(self):
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_input.setText(folder)

    def browse_output_file(self):
        """Browse for output file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Base Output File", "", "MGF Files (*.mgf);;All Files (*)")
        if file_path:
            self.output_file_input.setText(file_path)

    def export_results(self):
        """Export filtered results to MGF files."""
        base_output_path = self.output_file_input.text()

        if not base_output_path:
            QMessageBox.warning(self, "Warning", "Please select a base output file.")
            return

        # Get selected filters
        selected_filters = [item.text() for item in self.export_filter_list.selectedItems()]

        if not selected_filters:
            QMessageBox.warning(self, "Warning", "Please select at least one filter to export.")
            return

        # Parse base path
        base_dir = os.path.dirname(base_output_path)
        base_name = os.path.splitext(os.path.basename(base_output_path))[0]

        exported_files = []
        skipped_filters = []

        progress = QProgressDialog("Exporting filtered results...", "Cancel", 0, len(selected_filters), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        for idx, filter_name in enumerate(selected_filters):
            if progress.wasCanceled():
                break

            progress.setValue(idx)
            progress.setLabelText(f"Exporting filter: {filter_name}")
            QApplication.processEvents()

            if filter_name not in self.filtered_results:
                skipped_filters.append(filter_name)
                continue

            results = self.filtered_results[filter_name]

            # Prepare matched and non-matched blocks
            matched_blocks = []
            non_matched_blocks = []

            for smiles, blocks in results["smiles_to_blocks"].items():
                if smiles in results["matched_smiles"]:
                    matched_blocks.extend(blocks)
                else:
                    non_matched_blocks.extend(blocks)

            # Export files with filter name as suffix
            try:
                # Sanitize filter name for filename
                safe_filter_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in filter_name)

                matched_path = os.path.join(base_dir, f"{base_name}_{safe_filter_name}_matched.mgf")
                non_matched_path = os.path.join(base_dir, f"{base_name}_{safe_filter_name}_noMatch.mgf")

                Filters.export_mgf_file(matched_blocks, matched_path)
                Filters.export_mgf_file(non_matched_blocks, non_matched_path)

                exported_files.append(f"{filter_name}: {len(matched_blocks)} matched, {len(non_matched_blocks)} non-matched")

            except Exception as e:
                skipped_filters.append(f"{filter_name} (Error: {str(e)})")

        progress.setValue(len(selected_filters))

        # Show summary
        msg = f"Successfully exported {len(exported_files)} filter(s):\n\n"
        for export_info in exported_files:
            msg += f"  • {export_info}\n"

        if skipped_filters:
            msg += f"\n\nSkipped {len(skipped_filters)} filter(s):\n"
            for skipped in skipped_filters:
                msg += f"  • {skipped}\n"

        msg += f"\n\nFiles saved to: {base_dir}"

        if skipped_filters:
            QMessageBox.warning(self, "Export Complete with Warnings", msg)
        else:
            QMessageBox.information(self, "Export Complete", msg)

    def closeEvent(self, event):
        """Clean up temporary directory on close."""
        self.temp_dir.cleanup()

        # Close all structure viewer windows
        for window in self.structure_windows:
            window.close()

        event.accept()

    def load_filters_from_json(self):
        """Load multiple filters from a JSON file and apply them automatically."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON Filter File", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                filters_dict = json.load(f)

            if not isinstance(filters_dict, dict):
                QMessageBox.critical(self, "Error", "Invalid JSON format. Expected a dictionary with filter names as keys and SMARTS patterns as values.")
                return

            # Load and apply each filter
            loaded_count = 0
            error_count = 0
            errors = []

            progress = QProgressDialog("Loading filters from JSON...", "Cancel", 0, len(filters_dict), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            for idx, (filter_name, smarts_string) in enumerate(filters_dict.items()):
                if progress.wasCanceled():
                    break

                progress.setValue(idx)
                progress.setLabelText(f"Loading filter: {filter_name}")
                QApplication.processEvents()

                try:
                    # Validate filter name and SMARTS string
                    if not filter_name or not isinstance(filter_name, str):
                        raise ValueError("Invalid filter name")

                    if not smarts_string or not isinstance(smarts_string, str):
                        raise ValueError("Invalid SMARTS pattern")

                    # Parse AND-separated groups (case-insensitive)
                    import re

                    and_groups = re.split(r"\s+AND\s+", smarts_string, flags=re.IGNORECASE)
                    and_groups = [s.strip() for s in and_groups if s.strip()]

                    # Parse each AND group for OR patterns (marked with << >>)
                    parsed_smarts = []
                    for and_group in and_groups:
                        # Check if this group has << >> markers
                        if "<<" in and_group and ">>" in and_group:
                            # Extract content within << >>
                            or_match = re.search(r"<<(.+?)>>", and_group, re.DOTALL)
                            if or_match:
                                or_content = or_match.group(1)
                                # Split by OR (case-insensitive)
                                or_patterns = re.split(r"\s+OR\s+", or_content, flags=re.IGNORECASE)
                                or_patterns = [p.strip() for p in or_patterns if p.strip()]
                            else:
                                # If markers exist but regex doesn't match, treat as single pattern
                                or_patterns = [and_group.strip()]
                        else:
                            # No OR patterns, treat as single pattern
                            or_patterns = [and_group.strip()]

                        parsed_smarts.append(or_patterns)

                    # Validate all SMARTS patterns
                    for or_group in parsed_smarts:
                        for smarts in or_group:
                            mol = rdkit.Chem.MolFromSmarts(smarts)
                            if not mol:
                                raise ValueError(f"Invalid SMARTS: {smarts}")

                    # Check if updating existing filter
                    is_update = filter_name in self.smarts_filters

                    # Store the filter
                    self.smarts_filters[filter_name] = parsed_smarts

                    # Update or add to export list
                    if not is_update:
                        item = QListWidgetItem(filter_name)
                        self.export_filter_list.addItem(item)
                        if self.export_all_checkbox.isChecked():
                            item.setSelected(True)

                    # Apply the filter
                    self.apply_filter(filter_name)

                    loaded_count += 1

                except Exception as e:
                    error_count += 1
                    errors.append(f"{filter_name}: {str(e)}")
                    print(f"Error loading filter '{filter_name}': {e}")

            progress.setValue(len(filters_dict))

            # Show results
            msg = f"Successfully loaded and applied {loaded_count} filter(s)"
            if error_count > 0:
                msg += f"\n\n{error_count} filter(s) failed to load:"
                for error in errors[:10]:  # Show first 10 errors
                    msg += f"\n  - {error}"
                if len(errors) > 10:
                    msg += f"\n  ... and {len(errors) - 10} more"

            if error_count > 0:
                QMessageBox.warning(self, "Load Complete with Errors", msg)
            else:
                QMessageBox.information(self, "Load Complete", msg)

        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Error", f"Failed to parse JSON file:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load filters from JSON:\n{str(e)}")

    def save_filters_to_json(self):
        """Save all defined filters to a JSON file."""
        if not self.smarts_filters:
            QMessageBox.warning(self, "Warning", "No filters defined to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Filters to JSON", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            # Reconstruct SMARTS strings from parsed format
            filters_dict = {}

            for filter_name, parsed_smarts in self.smarts_filters.items():
                smarts_parts = []

                for or_group in parsed_smarts:
                    if len(or_group) > 1:
                        # Has OR patterns
                        or_string = "<<" + " OR ".join(or_group) + ">>"
                        smarts_parts.append(or_string)
                    else:
                        # Single pattern
                        smarts_parts.append(or_group[0])

                filters_dict[filter_name] = " AND ".join(smarts_parts)

            # Save to JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(filters_dict, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "Success", f"Successfully saved {len(filters_dict)} filter(s) to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save filters:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = MGFFilterGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
